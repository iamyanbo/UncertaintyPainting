import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.painting import FeaturePainter
from src.projection import Calibration, filter_points_in_image

def main():
    # 1. Setup - Use the test data files we already have
    image_path = "data/test_image.png"
    calib_path = "data/test_calib.txt"
    lidar_path = "data/test_lidar.bin"
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found. Please run scripts 01 and 02 first.")
        return
    if not os.path.exists(calib_path):
        print(f"Error: {calib_path} not found. Please run script 02 first.")
        return
    if not os.path.exists(lidar_path):
        print(f"Error: {lidar_path} not found. Please run script 02 first.")
        return
    
    # 2. Initialize Painter
    # Use resnet50 for faster testing
    painter = FeaturePainter(segmentation_model_name='deeplabv3_resnet50')
    
    # 3. Load Data
    calib = Calibration(calib_path)
    lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    
    # 4. Paint!
    print(f"Painting {lidar_points.shape[0]} points...")
    painted_points = painter.paint(image_path, lidar_points, calib)
    
    print(f"Original Shape: {lidar_points.shape}")
    print(f"Painted Shape: {painted_points.shape}")
    
    # The last column is Entropy (Uncertainty)
    entropy_vals = painted_points[:, -1]
    print(f"Max Entropy on points: {np.max(entropy_vals)}")
    print(f"Mean Entropy on points: {np.mean(entropy_vals)}")
    
    # 5. Visualize (Project back to 2D to verify)
    # We will color the points by their assigned Entropy
    pts_3d = painted_points[:, :3]
    pts_2d, depth = calib.project_velo_to_image(pts_3d)
    
    image_shape = (375, 1242) # Dummy size
    mask = filter_points_in_image(pts_2d, image_shape)
    mask &= (depth > 0)
    
    pts_2d_valid = pts_2d[mask]
    entropy_valid = entropy_vals[mask]
    
    plt.figure(figsize=(12, 4))
    plt.title("Painted Points (Colored by Uncertainty)")
    plt.xlim(0, image_shape[1])
    plt.ylim(image_shape[0], 0)
    
    # Plot points
    sc = plt.scatter(pts_2d_valid[:, 0], pts_2d_valid[:, 1], c=entropy_valid, cmap='inferno', s=3)
    plt.colorbar(sc, label='Entropy')
    
    output_path = "data/test_painting_result.png"
    plt.savefig(output_path)
    print(f"Painting visualization saved to {output_path}")
    
    # Save the painted point cloud for inspection
    painted_points.astype(np.float32).tofile("data/test_painted_cloud.bin")
    print("Saved painted point cloud to data/test_painted_cloud.bin")

if __name__ == "__main__":
    main()
