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
        return
        
    if not os.path.exists(calib_path):
        print(f"Error: Calibration file not found at {calib_path}")
        return

    # 1. Load Data
    # Painted points are (N, 26) -> x, y, z, i, C1..C21, Entropy
    points = np.fromfile(painted_path, dtype=np.float32).reshape(-1, 26)
    
    # Extract channels
    xyz = points[:, :3]
    
    # Class Scores (Indices 4 to 24 covers 21 classes)
    class_scores = points[:, 4:25] 
    pred_labels = np.argmax(class_scores, axis=1) # (N,)
    
    # Uncertainty (Index 25)
    entropy = points[:, -1]
    
    print(f"Loaded sample {args.idx}")
    print(f"Points: {points.shape[0]}")
    print(f"Entropy Range: [{np.min(entropy):.4f}, {np.max(entropy):.4f}]")
    print(f"Classes Found: {np.unique(pred_labels)}")
    
    # 2. Project to Image for Visualization
    calib = Calibration(calib_path)
    pts_2d, depth = calib.project_velo_to_image(xyz)
    
    # Filter for FOV (assuming standard KITTI image size approx 375x1242)
    H, W = 375, 1242 
    mask = filter_points_in_image(pts_2d, (H, W))
    mask &= (depth > 0)
    
    pts_valid = pts_2d[mask]
    entropy_valid = entropy[mask]
    labels_valid = pred_labels[mask]
    
    # 3. Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Uncertainty
    ax1.set_title(f"Sample {args.idx}: Uncertainty (Entropy)")
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    sc1 = ax1.scatter(pts_valid[:, 0], pts_valid[:, 1], c=entropy_valid, cmap='inferno', s=2)
    plt.colorbar(sc1, ax=ax1, label='Entropy')
    
    # Plot 2: Classification
    # Use a qualitative colormap (tab20) since classes are discrete
    ax2.set_title(f"Sample {args.idx}: Predicted Semantic Class")
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    sc2 = ax2.scatter(pts_valid[:, 0], pts_valid[:, 1], c=labels_valid, cmap='tab20', s=2, vmin=0, vmax=20)
    plt.colorbar(sc2, ax=ax2, label='Class ID (VOC)')
    
    plt.tight_layout()
    output_path = f"data/view_result_{args.idx}_dual.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
