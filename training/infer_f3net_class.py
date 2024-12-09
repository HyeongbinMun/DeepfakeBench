import os
import numpy as np
from os.path import join
import cv2
import random
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision import transforms
from preprocess_infer import img_face_crop
import matplotlib.pyplot as plt
from detectors import DETECTOR
from PIL import Image


class F3NetInference:
    def __init__(self,
                 detector_path='/user/training/config/detector/f3net.yaml',
                 test_config_path='/user/training/config/test_config.yaml',
                 weights_path='/nfs_shared/deepfake/pretrained_model/f3net_best.pth',
                 xception_path='/nfs_shared/deepfake/pretrained_model/xception-b5690688.pth',
                 efficientnet_path='/nfs_shared/deepfake/pretrained_model/efficientnet-b4-6ed6700e.pth',
                 face_crop_predictor='/nfs_shared/deepfake/pretrained_model/shape_predictor_81_face_landmarks.dat',
                 save_path='/user/'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_path = detector_path
        self.test_config_path = test_config_path
        self.weights_path = weights_path
        self.xception_path = xception_path
        self.efficientnet_path = efficientnet_path
        self.face_crop_predictor = face_crop_predictor
        self.save_path = save_path

        # Load configuration
        with open(detector_path, 'r') as f:
            config = yaml.safe_load(f)
        with open(test_config_path, 'r') as f:
            config2 = yaml.safe_load(f)
        config.update(config2)
        self.config = config

        # Initialize random seed
        self.init_seed()

        # Set up model
        self.model = self.prepare_model()

    def init_seed(self):
        if self.config['manualSeed'] is None:
            self.config['manualSeed'] = random.randint(1, 10000)
        random.seed(self.config['manualSeed'])
        torch.manual_seed(self.config['manualSeed'])
        if self.config['cuda']:
            torch.cuda.manual_seed_all(self.config['manualSeed'])

    def prepare_model(self):
        model_class = DETECTOR[self.config['model_name']]
        if self.config['backbone_name'] =='xception':
            self.config['pretrained'] = self.xception_path
        elif self.config['backbone_name'] =='efficientnetb4':
            self.config['pretrained'] = self.efficientnet_path
        model = model_class(self.config).to(self.device)

        if self.weights_path:
            ckpt = torch.load(self.weights_path, map_location=self.device)
            model.load_state_dict(ckpt, strict=True)
            print('===> Load checkpoint done!')
        else:
            print('Fail to load the pre-trained weights')
        return model

    def load_rgb(self, img):
        size = self.config['resolution']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def normalize(self, img):
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def to_tensor(self, img):
        return T.ToTensor()(img)

    def load_img(self, img):
        image = self.load_rgb(img)
        image = np.array(image)
        image_tensors = self.normalize(self.to_tensor(image))
        return image_tensors.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def inference(self, model, data_dict):
        predictions = model(data_dict, inference=True)
        return predictions

    def run_inference(self, image_path):
           
        img = img_face_crop(image_path, self.face_crop_predictor)
        cropped_img_np = np.array(img)
        cropped_img_pil = Image.fromarray(cropped_img_np)
        out_images = [cropped_img_pil]


        data_dict = {'image': self.load_img(img), 'label': torch.tensor([0]).to(self.device)}

        self.model.eval()
        predictions, fea_FAD = self.inference(self.model, data_dict)
        _, class_result = torch.max(predictions['cls'], 1)

        cls_prob = F.softmax(predictions['cls'], dim=1)

        to_pil = transforms.ToPILImage()
        filters = ['low_filter', 'middle_filter', 'high_filter', 'all_filter']
        for i in range(4):
            img = to_pil(fea_FAD[:,i*3:(i+1)*3,:,:].cpu().squeeze(0))
            out_images.append(img)
            # img.save(f'/user/{filters[i]}.png')

        classes = ["REAL", "FAKE"]

        results = {
            "FakeOrReal": {},
            "DetectionResult": classes[class_result.item()],
        }

        for i in range(2):
            results["FakeOrReal"][classes[i]] = cls_prob[0, i].item()

        return results, out_images

# 클래스 사용 예제
# inference_runner = F3NetInference()
# img_path = '/user/visualization_images/107_fake_crop.png'
# results, out_images = inference_runner.run_inference(img_path)