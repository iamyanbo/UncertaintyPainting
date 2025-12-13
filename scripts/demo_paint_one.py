import sys
import os
import numpy as np
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.painting import FeaturePainter
from src.projection import Calibration

def main():
    idx = '000015'
    split = 'training'
    model_path = 'checkpoints/edl_deeplabv3_overnight.pth'
    
    base_dir = os.path.join('data', split)
    image_dir = os.path.join(base_dir, 'image_2')
    velo_dir = os.path.join(base_dir, 'velodyne')
    calib_dir = os.path.join(base_dir, 'calib')
    output_dir = os.path.join(base_dir, 'velodyne_painted')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Painting sample {idx}...")
    
    filename = f"{idx}.png"
    image_path = os.path.join(image_dir, filename)
    velo_path = os.path.join(velo_dir, f"{idx}.bin")
    calib_path = os.path.join(calib_dir, f"{idx}.txt")
    output_path = os.path.join(output_dir, f"{idx}.bin")
    
    # Initialize Painter
    painter = FeaturePainter(
        segmentation_model_name='deeplabv3_resnet101', 
        uncertainty_method='edl', 
        model_path=model_path
    )
    
    # Load Data
    calib = Calibration(calib_path)
    lidar_points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    
    # Paint
    painted_points = painter.paint(image_path, lidar_points, calib)
    
    # Save
    painted_points.astype(np.float32).tofile(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
