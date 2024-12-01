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
import npr_networks.resnet as resnet
from torchvision import transforms
from preprocess_infer import img_face_crop
from PIL import Image
from collections import OrderedDict
from copy import deepcopy


class NPRInference:
    def __init__(self,
                 face_crop_predictor='/nfs_shared/deepfake/pretrained_model/shape_predictor_81_face_landmarks.dat',
                 npr_model_path = '/nfs_shared/deepfake/pretrained_model/NPR.pth',
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
    
    def load_single_image(self, image_path, opt):
        transform = transforms.Compose([
            transforms.Resize((opt['loadSize'], opt['loadSize'])),  # 이미지 리사이즈
            transforms.CenterCrop(opt['cropSize']),  # 이미지 크롭
            transforms.ToTensor(),  # PIL 이미지를 PyTorch 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
        ])
        
        image = Image.open(image_path).convert('RGB')  # 이미지를 RGB로 로드
        image = transform(image)  # 변환 적용
        image = image.unsqueeze(0)  # 배치 차원 추가 (batch_size, C, H, W)
        
        return image
    
    def npr_image(self, image_path):
        image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        image = torch.Tensor(image)
        image = image.permute(2,0,1).contiguous()
        gray_img = torch.mean(image, dim=0, keepdim=True)

        reconstructed = F.interpolate(F.interpolate(gray_img, scale_factor=0.5, mode='nearest', recompute_scale_factor=True), scale_factor=2, mode='nearest', recompute_scale_factor=True)

        npr  = image - reconstructed
        return npr

    
    def run_inference(self, image_path):

        img = img_face_crop(image_path, self.face_crop_predictor)
        cropped_img_np = np.array(img)

        model = resnet50(num_classes=1)
        state_dict = torch.load(self.model_path, map_location='cpu')['model']
        pretrained_dict = OrderedDict()
        for ki in state_dict.keys():
            pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])
        model.load_state_dict(pretrained_dict, strict=True)
        model.cuda()
        model.eval()
        opt = {
        'loadSize': 256,  # 이미지 로드 크기
        'cropSize': 224,  # 크롭 크기
        }

        image_tensor = self.load_single_image(image_path, opt)
        image_tensor = image_tensor.cuda()


        y_pred = model(image_tensor)
        prob = y_pred.sigmoid().flatten().item()

        results = {
            "FakeOrReal": {},
        }

        results["FakeOrReal"]["REAL"] = 1-prob
        results["FakeOrReal"]["FAKE"] = prob

        npr_image = self.npr_image(img)
        npr_image_np= npr_image.numpy().transpose(1, 2, 0)

        # save image as tensor (more accurate image)
        # save_image(npr_image, '/user/npr_image.png')

        # save image as numpy
        # cv2.imwrite("/user/npr_image_fake.png", npr_image_np)

        out_images = [cropped_img_np, npr_image_np]
        print(results)

        return results, out_images

# if __name__ == '__main__':
#     inference_runner = NPRInference()
#     img_path = '/user/visualization_images/ddpm_fake.png'
#     results, out_images = inference_runner.run_inference(img_path)