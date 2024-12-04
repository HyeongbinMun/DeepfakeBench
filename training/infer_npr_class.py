import os
import numpy as np
from os.path import join
import cv2
import random
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
from npr_networks.resnet import resnet50
from torchvision import transforms
from preprocess_infer import img_face_crop
from PIL import Image
import torch.nn as nn


class NPRInference:
    def __init__(self,
                 face_crop_predictor='/nfs_shared/deepfake/pretrained_model/shape_predictor_81_face_landmarks.dat',
                 npr_model_path = '/nfs_shared/deepfake/pretrained_model/npr_final.pth',
                 save_path='user/'):
        
        self.face_crop_predictor = face_crop_predictor
        self.model_path = npr_model_path

        # Initialize random seed
        self.seed_torch(100)
        
    def seed_torch(self,seed=1029):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
    
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)


    def predict_image(self, image_path, model):
        trans = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
        img = trans(Image.open(image_path).convert('RGB')).unsqueeze(0)
        _, c, w, h = img.shape
        if w%2 == 1: img = img[:, :, :-1,:  ]
        if h%2 == 1: img = img[:, :, :  ,:-1]
        NPR = img - self.interpolate(img, 0.5)
        with torch.no_grad():
            x   = model.conv1(NPR*2.0/3.0)
            x   = model.bn1(x)
            x   = model.relu(x)
            x   = model.maxpool(x)
            x   = model.layer1(x)
            x   = model.layer2(x).mean(dim=(2,3), keepdim=False)
            x   = model.fc1(x)
        
        pred = x.sigmoid().cpu().numpy()
        return pred, NPR

    def run_inference(self, image_path):

        NPRmodel = resnet50()
        NPRmodel.fc1 = nn.Linear(512, 1)
        NPRmodel.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        NPRmodel.load_state_dict(torch.load(self.model_path, map_location='cpu'), strict=True)
        

        pred, NPR = self.predict_image(image_path, NPRmodel)

        img = img_face_crop(image_path, self.face_crop_predictor)
        cropped_img_np = np.array(img)

        class_result = "FAKE" if pred>0.5 else "REAL"
        
        results = {
            "FakeOrReal" : {'REAL': 1.0 - float(pred), 'FAKE': float(pred), },
            "DetectionResult": class_result
        } 
        
        #tensor to numpy
        npr_np = NPR.squeeze(0).numpy().transpose(1,2,0)
        npr_np = (npr_np * 255).astype(np.uint8)

         # save image as tensor (more accurate image)
        # save_image(NPR, '/user/npr_image.png')

        # save image as numpy
        # cv2.imwrite("/user/npr_image.png", npr_np)

        out_images = [cropped_img_np, npr_np]

        return results, out_images

# if __name__ == '__main__':
#     inference_runner = NPRInference()
#     img_path = '/user/visualization_images/ddpm_fake.png'
#     results, out_images = inference_runner.run_inference(img_path)