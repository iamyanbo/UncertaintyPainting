import numpy as np
import torch
from .segmentation import UncertaintyPainter2D
from .projection import Calibration, filter_points_in_image

class FeaturePainter:
    def __init__(self, segmentation_model_name='deeplabv3_resnet50'):
        self.segmentor = UncertaintyPainter2D(model_name=segmentation_model_name)

    def paint(self, image_path, lidar_points, calib):
        """
        Paints the LiDAR points with 2D uncertainty and class features.
        
        Args:
            image_path (str): Path to the image.
            lidar_points (np.array): (N, 4) [x, y, z, intensity]
            calib (Calibration): Calibration object.
            
        Returns:
            painted_points (np.array): (N, 4 + K + 1) [x, y, z, r, C_1..C_K, Uncertainty]
        """
        # 1. Run 2D Segmentation & Uncertainty
        entropy_map, class_probs, _ = self.segmentor.predict(image_path)
        # entropy_map: (H, W)
        # class_probs: (K, H, W)
        
        H, W = entropy_map.shape
        K = class_probs.shape[0]
        
        # 2. Project LiDAR to Image
        pts_3d = lidar_points[:, :3]
        pts_2d, depth = calib.project_velo_to_image(pts_3d)
        
        # 3. Filter points inside image
        mask = filter_points_in_image(pts_2d, (H, W))
        mask &= (depth > 0)
        
        # Initialize features with zeros (or -1) for points outside FOV
        # Dimensions: K class probs + 1 entropy
        num_points = lidar_points.shape[0]
        features = np.zeros((num_points, K + 1), dtype=np.float32)
        
        # 4. Sample Features
        valid_indices = np.where(mask)[0]
        valid_pts_2d = pts_2d[valid_indices]
        
        # Round to nearest integer pixel for sampling (could use bilinear interpolation for better precision)
        u = np.clip(np.round(valid_pts_2d[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(valid_pts_2d[:, 1]).astype(int), 0, H - 1)
        
        # Sample Class Probabilities
        # class_probs is (K, H, W) -> we want (N_valid, K)
        # We can transpose class_probs to (H, W, K) for easier indexing
        class_probs_t = class_probs.transpose(1, 2, 0)
        sampled_probs = class_probs_t[v, u, :] # (N_valid, K)
        
        # Sample Entropy
        sampled_entropy = entropy_map[v, u] # (N_valid,)
        
        # Assign to features array
        features[valid_indices, :K] = sampled_probs
        features[valid_indices, K] = sampled_entropy
        
        # 5. Concatenate
        painted_points = np.hstack((lidar_points, features))
        
        return painted_points
