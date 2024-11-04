"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger
from PIL import Image
from torchvision import transforms as T
from preprocess_infer import img_face_crop
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def grad_cam(model, x, target_class):
    model.eval()
    feature_extractor = model.features
    classifier = model.classifier
    
    # Forward
    features = feature_extractor(x)
    forgery_features, content_features = features['forgery'], features['content']
    output = classifier(forgery_features)
    class_score = output[target_class].sum() 
    
    # Gradient
    model.zero_grad()
    class_score.backward()  # backpropagate for the specific class
    gradients = model.gradients['forgery']
    activations = output

    # calculate Gradient channel-wise mean
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # calculate gradients for each activation map
    for i in range(len(activations)):
        activations[i][:, :, :, :] *= pooled_gradients[i]

    # class 0 activation + class 1 activation
    heatmap1 = torch.mean(activations[0], dim=1).squeeze()
    heatmap1 = F.relu(heatmap1)

    heatmap2 = torch.mean(activations[1], dim=1).squeeze()
    heatmap2 = F.relu(heatmap2)

    # avg
    heatmap = (heatmap1 + heatmap2) / 2

    # scaling to the image size
    heatmap_resized = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0),
                                              x['image'].shape[2:],
                                              mode='bilinear',
                                              align_corners=False).squeeze()
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())

    return heatmap_normalized.cpu().detach().numpy()


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

def load_rgb(img, config):

    size = config['resolution']
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(np.array(img, dtype=np.uint8))

def normalize(img, config):
    mean = config['mean']
    std = config['std']
    normalize = T.Normalize(mean=mean, std=std)

    return normalize(img)

def to_tensor(img):
    return T.ToTensor()(img)

def load_img(img, config):
    # Load the crop image
    image = load_rgb(img, config)
    image = np.array(image)

    # To tensor and normalize
    image_tensors = normalize(to_tensor(image), config)
    image_tensors = image_tensors.unsqueeze(0).to(device)

    return image_tensors


@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

def grad_cam_img(image, grad_cam):

    fig, ax = plt.subplots()
    ax.imshow(image.squeeze(0).permute(1,2,0).cpu().detach(), alpha=0.7)
    ax.imshow(grad_cam, cmap='jet', alpha=0.3) 
    ax.axis('off') 

    # plt.savefig('overlayed_image.png', bbox_inches='tight', pad_inches=0)
    # plt.close()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # 리소스 해제
    return data



def main(args):
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.test_config_path, 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # modifying detector config
    config['test_batchSize']= 1

    if config['backbone_name'] =='xception':
        config['pretrained'] = args.xception
    elif config['backbone_name'] =='efficientnetb4':
        config['pretrained'] = args.efficientnet

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    file_name = args.image_path.split('/')[-1][:-4]

    # crop the input image
    img = img_face_crop(args.image_path, args.face_crop_predictor)
    # Save cropped face
    crop_save_path = file_name + '_crop.png'
    cv2.imwrite(str(args.save_path + crop_save_path), img)

    
    # input
    data_dict={}
    data_dict['image']= load_img(img, config)
    # prevent error
    data_dict['label'] = torch.tensor([0]).to(device)

    model.eval()
    predictions, out_spe = inference(model, data_dict)
    print('Model Name:', config['model_name'])
    _, class_result = torch.max(predictions['cls'], 1)
    _, spec_num = torch.max(out_spe, 1)
    cls_prob = torch.softmax(predictions['cls'],1)
    spec_prob = torch.softmax(out_spe, dim=1)
    
    classes = ["REAL", "FAKE"]
    spec_classes = ["REAL", "FF-DeepFakes","FF-Face2Face", "FF-FaceSwap", "FF-NeuralTextures"]

   
    print("-----------Fake or Real-----------")
    for i in range(0,2):
        print("Probability of being "+classes[i]+" : ", cls_prob[0, i].item())
    print('Detection Result:', classes[class_result])

    print("-----------Manipulation Method-----------")
    if class_result==1:
        for i in range(len(spec_classes)):
            print("Probability of being created with "+spec_classes[i]+" : ", spec_prob[0,i].item())
        print("Manipulation method : ", spec_classes[spec_num])

    print('===> Test Done!')

    # Apply Grad-CAM after inference
    gradcam_map = grad_cam(model, data_dict, class_result)
    # save the gradcam image
    plt.imsave(args.save_path + file_name + '_grad_cam.png', gradcam_map, cmap='viridis')

    # Grad-CAM and image overlay
    overlayed_image = grad_cam_img(data_dict['image'], gradcam_map)
    # save the overlayed image
    img = Image.fromarray(overlayed_image)
    img.save(args.save_path + file_name + '_grad_cam_overlay.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--detector_path', type=str, 
                        default='/workspace/DeepfakeBench/training/config/detector/ucf.yaml',
                        help='path to detector YAML file')
    parser.add_argument("--test_config_path", type=str, 
                        default='/workspace/DeepfakeBench/training/config/test_config.yaml',
                        help='path to detector YAML file')
    parser.add_argument('--weights_path', type=str, 
                        default='/workspace/DeepfakeBench/training/weights/ucf_best.pth')
    parser.add_argument('--xception', type=str, 
                        default='/workspace/DeepfakeBench/training/weights/xception-b5690688.pth')
    parser.add_argument('--efficientnet', type=str, 
                        default='/workspace/DeepfakeBench/training/weights/efficientnet-b4-6ed6700e.pth')
    parser.add_argument('--face_crop_predictor', type=str, 
                        default='/workspace/DeepfakeBench/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat')
    parser.add_argument('--image_path', type=str, 
                        default='/workspace/DeepfakeBench/training/images/354_fake.png')
    parser.add_argument('--save_path', type=str,
                        default='/workspace/DeepfakeBench/training/images/')
    args = parser.parse_args()

    main(args)
