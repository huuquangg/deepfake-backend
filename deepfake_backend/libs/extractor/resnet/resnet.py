import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ResNetExtractor:
    def __init__(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image: Image.Image):
        input_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(input_tensor).numpy().tolist()[0]
        return features
