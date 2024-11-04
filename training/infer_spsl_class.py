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
from preprocess_infer import img_face_crop
import matplotlib.pyplot as plt
from detectors import DETECTOR


class DeepfakeInference:
    def __init__(self,
                 detector_path='/workspace/DeepfakeBench/training/config/detector/spsl.yaml',
                 test_config_path='/workspace/DeepfakeBench/training/config/test_config.yaml',
                 weights_path='/workspace/DeepfakeBench/training/weights/spsl_best.pth',
                 xception_path='/workspace/DeepfakeBench/training/weights/xception-b5690688.pth',
                 efficientnet_path='/workspace/DeepfakeBench/training/weights/efficientnet-b4-6ed6700e.pth',
                 face_crop_predictor='/workspace/DeepfakeBench/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat',
                 save_path='/workspace/DeepfakeBench/training/images/'):

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
        file_name = image_path.split('/')[-1][:-4]

        img = img_face_crop(image_path, self.face_crop_predictor)
        cropped_img_np = np.array(img)

        data_dict = {'image': self.load_img(img), 'label': torch.tensor([0]).to(self.device)}

        self.model.eval()
        predictions, phase_feature = inference(model, data_dict)
        _, class_result = torch.max(predictions['cls'], 1)

        cls_prob = F.softmax(predictions['cls'], dim=1)

        classes = ["REAL", "FAKE"]

        results = {
            "FakeOrReal": {},
            "DetectionResult": classes[class_result.item()],
        }

        for i in range(2):
            results["FakeOrReal"][classes[i]] = cls_prob[0, i].item()

        ##요것 gray_scale image 임 !
        phase_img_np = phase_feature.detach().squeeze().cpu().numpy() if phase_feature.is_cuda else phase_feature.squeeze().numpy()
        # plt.imsave('/user/array_feature.png', phase_img_np, cmap='gray')

        out_images = [cropped_img_np, phase_img_np]

        return results, out_images

# 클래스 사용 예제
# inference_runner = DeepfakeInference()
# img_path = '/workspace/DeepfakeBench/training/images/354_fake.png'
# results, out_images = inference_runner.run_inference(img_path)