import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.projection import Calibration, filter_points_in_image

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Painted Point Clouds')
    parser.add_argument('--idx', type=str, default='000015', help='Index of the sample to visualize (e.g., 000015)')
    parser.add_argument('--split', type=str, default='training', help='Data split')
    return parser.parse_args()

def main():
    args = parse_args()
    
    base_dir = os.path.join('data', args.split)
    painted_dir = os.path.join(base_dir, 'velodyne_painted')
    calib_dir = os.path.join(base_dir, 'calib')
    
    painted_path = os.path.join(painted_dir, f"{args.idx}.bin")
    calib_path = os.path.join(calib_dir, f"{args.idx}.txt")
    
    if not os.path.exists(painted_path):
        print(f"Error: Painted file not found at {painted_path}")
        print("Did you run scripts/04_process_dataset.py for this sample?")
        return
        
    if not os.path.exists(calib_path):
        print(f"Error: Calibration file not found at {calib_path}")
        return

    # 1. Load Data
    # Painted points are (N, 26) -> x, y, z, i, C1..C21, Entropy
    points = np.fromfile(painted_path, dtype=np.float32).reshape(-1, 26)
    
    # Extract channels
    xyz = points[:, :3]
    entropy = points[:, -1]
    
    print(f"Loaded sample {args.idx}")
    print(f"Points: {points.shape[0]}")
    print(f"Entropy Range: [{np.min(entropy):.4f}, {np.max(entropy):.4f}]")
    
    # 2. Project to Image for Visualization
    calib = Calibration(calib_path)
    pts_2d, depth = calib.project_velo_to_image(xyz)
    
    # Filter for FOV (assuming standard KITTI image size approx 375x1242)
    H, W = 375, 1242 
    mask = filter_points_in_image(pts_2d, (H, W))
    mask &= (depth > 0)
    
    pts_valid = pts_2d[mask]
    entropy_valid = entropy[mask]
    
    # 3. Plot
    plt.figure(figsize=(12, 4))
    plt.title(f"Sample {args.idx}: Painted Uncertainty (Entropy)")
    plt.xlim(0, W)
    plt.ylim(H, 0)
    
    # Scatter plot colored by entropy
    sc = plt.scatter(pts_valid[:, 0], pts_valid[:, 1], c=entropy_valid, cmap='inferno', s=2)
    plt.colorbar(sc, label='Entropy')
    
    output_path = f"data/view_result_{args.idx}.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
