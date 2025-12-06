import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

class UncertaintyPainter2D:
    def __init__(self, model_name='deeplabv3_resnet101', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the 2D Segmentation model for uncertainty estimation.
        """
        self.device = device
        print(f"Initializing 2D Painter with {model_name} on {device}...")
        
        if model_name == 'deeplabv3_resnet101':
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        elif model_name == 'deeplabv3_resnet50':
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image_path_or_pil):
        """
        Loads and preprocesses an image.
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil
            
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return input_tensor, image

    def compute_entropy(self, probabilities):
        """
        Computes Shannon Entropy from probabilities: H(p) = -sum(p * log(p))
        """
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1)
        return entropy

    def predict(self, image_path_or_pil):
        """
        Runs inference and returns the Entropy Map and Class Probabilities.
        
        Returns:
            entropy_map (numpy array): (H, W)
            class_probs (numpy array): (K, H, W)
            original_image (PIL Image)
        """
        input_tensor, original_image = self.preprocess(image_path_or_pil)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out']
            probabilities = F.softmax(output, dim=1)
            
            entropy = self.compute_entropy(probabilities)
            
        return (
            entropy.squeeze(0).cpu().numpy(),
            probabilities.squeeze(0).cpu().numpy(),
            original_image
        )
