import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
from . import edl  # Import the new EDL module

class UncertaintyPainter2D:
    def __init__(self, model_name='deeplabv3_resnet101', device='cuda' if torch.cuda.is_available() else 'cpu', uncertainty_method='entropy', model_path=None):
        """
        Initializes the 2D Segmentation model for uncertainty estimation.
        
        Args:
            model_name (str): Name of the segmentation model.
            device (str): Device to run the model on.
            uncertainty_method (str): Method to calculate uncertainty. Options: 'entropy', 'edl' (Evidential Deep Learning).
            model_path (str): Optional path to a fine-tuned checkpoint.
        """
        self.device = device
        self.uncertainty_method = uncertainty_method
        print(f"Initializing 2D Painter with {model_name} on {device} using {uncertainty_method}...")
        
        if model_name == 'deeplabv3_resnet101':
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        elif model_name == 'deeplabv3_resnet50':
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        if model_path:
            print(f"Loading custom weights from {model_path}...")
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict, strict=False)
            
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

    def extract_features(self, image_path_or_pil):
        """
        Extracts backbone features (before classifier).
        """
        input_tensor, _ = self.preprocess(image_path_or_pil)
        with torch.no_grad():
            # Access the backbone directly
            # For DeepLabv3 in torchvision, 'backbone' returns 'out' (2048 ch) and 'aux'
            features = self.model.backbone(input_tensor)['out']
        return features


    def predict(self, image_path_or_pil):
        """
        Runs inference and returns the Uncertainty Map and Class Probabilities.
        
        Returns:
            uncertainty_map (numpy array): (H, W) - either Entropy or EDL Uncertainty
            class_probs (numpy array): (K, H, W)
            original_image (PIL Image)
        """
        input_tensor, original_image = self.preprocess(image_path_or_pil)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'] # Logits: (1, K, H, W)
            
            if self.uncertainty_method == 'edl':
                # Use Evidential Deep Learning
                uncertainty, probabilities = edl.calculate_uncertainty(output, evidence_func=edl.relu_evidence)
                # uncertainty: (1, H, W), probabilities: (1, K, H, W)
                uncertainty_map = uncertainty
            else:
                # Use Shannon Entropy
                probabilities = F.softmax(output, dim=1)
                uncertainty_map = self.compute_entropy(probabilities)
            
        return (
            uncertainty_map.squeeze(0).cpu().numpy(),
            probabilities.squeeze(0).cpu().numpy(),
            original_image
        )
