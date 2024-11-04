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
                 detector_path='/workspace/DeepfakeBench/training/config/detector/ucf.yaml',
                 test_config_path='/workspace/DeepfakeBench/training/config/test_config.yaml',
                 weights_path='/workspace/DeepfakeBench/training/weights/ucf_best.pth',
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

    def grad_cam(self, model, x, target_class):
        model.eval()
        feature_extractor = model.features
        classifier = model.classifier

        features = feature_extractor(x)
        forgery_features, content_features = features['forgery'], features['content']
        output = classifier(forgery_features)
        class_score = output[target_class].sum()

        model.zero_grad()
        class_score.backward()
        gradients = model.gradients['forgery']
        activations = output
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(len(activations)):
            activations[i][:, :, :, :] *= pooled_gradients[i]

        heatmap1 = torch.mean(activations[0], dim=1).squeeze()
        heatmap1 = F.relu(heatmap1)

        heatmap2 = torch.mean(activations[1], dim=1).squeeze()
        heatmap2 = F.relu(heatmap2)

        heatmap = (heatmap1 + heatmap2) / 2
        heatmap_resized = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0),
                                                          x['image'].shape[2:],
                                                          mode='bilinear',
                                                          align_corners=False).squeeze()
        heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
        return heatmap_normalized.cpu().detach().numpy()

    def grad_cam_img(self, image, grad_cam):
        fig, ax = plt.subplots()
        ax.imshow(image.squeeze(0).permute(1, 2, 0).cpu().detach(), alpha=0.7)
        ax.imshow(grad_cam, cmap='jet', alpha=0.3)
        ax.axis('off')

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return data

    import torch.nn.functional as F

    def run_inference(self, image_path):
        file_name = image_path.split('/')[-1][:-4]

        img = img_face_crop(image_path, self.face_crop_predictor)
        cropped_img_np = np.array(img)

        data_dict = {'image': self.load_img(img), 'label': torch.tensor([0]).to(self.device)}

        self.model.eval()
        predictions, out_spe = self.inference(self.model, data_dict)
        _, class_result = torch.max(predictions['cls'], 1)
        _, spec_num = torch.max(out_spe, 1)

        cls_prob = F.softmax(predictions['cls'], dim=1)
        spec_prob = F.softmax(out_spe, dim=1)

        classes = ["REAL", "FAKE"]
        spec_classes = ["REAL", "FF-DeepFakes", "FF-Face2Face", "FF-FaceSwap", "FF-NeuralTextures"]

        results = {
            "FakeOrReal": {},
            "DetectionResult": classes[class_result.item()],
            "ManipulationMethod": {}
        }

        for i in range(2):
            results["FakeOrReal"][classes[i]] = cls_prob[0, i].item()

        if class_result == 1:
            for i in range(len(spec_classes)):
                results["ManipulationMethod"][spec_classes[i]] = spec_prob[0, i].item()
            results["DetectedManipulationMethod"] = spec_classes[spec_num.item()]

        # Generate Grad-CAM image
        gradcam_map = self.grad_cam(self.model, data_dict, class_result)

        gradcam_map_uint8 = (gradcam_map * 255).astype(np.uint8)
        colormap = plt.get_cmap("viridis")
        gradcam_map_colored = colormap(gradcam_map_uint8 / 255.0)[:, :, :3]
        gradcam_map_rgb = (gradcam_map_colored * 255).astype(np.uint8)

        overlayed_image = self.grad_cam_img(data_dict['image'], gradcam_map)
        overlay_img_np = np.array(overlayed_image)

        out_images = [cropped_img_np, gradcam_map_rgb, overlay_img_np]

        return results, out_images

# 클래스 사용 예제
# inference_runner = DeepfakeInference()
# img_path = '/workspace/DeepfakeBench/training/images/354_fake.png'
# results, out_images = inference_runner.run_inference(img_path)
