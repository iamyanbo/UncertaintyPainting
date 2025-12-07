import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.segmentation import UncertaintyPainter2D
from src.painting import FeaturePainter
from src.projection import Calibration, filter_points_in_image

def main():
    idx = '005985'
    base_dir = r'OpenPCDet/data/kitti/training'
    image_path = os.path.join(base_dir, 'image_2', f'{idx}.png')
    velo_path = os.path.join(base_dir, 'velodyne', f'{idx}.bin')
    calib_path = os.path.join(base_dir, 'calib', f'{idx}.txt')
    
    output_dir = 'vis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing sample {idx}...")

    # 1. 2D Visualization (Image | Segmentation | Uncertainty)
    print("Generating 2D Visualization...")
    painter_2d = UncertaintyPainter2D(model_name='deeplabv3_resnet50')
    entropy_map, class_probs, original_img = painter_2d.predict(image_path)
    
    # Get predicted classes
    pred_mask = np.argmax(class_probs, axis=0)

    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image (Cyclist)")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Predicted Classes")
    plt.imshow(pred_mask, cmap='tab20')
    plt.colorbar(label='Class ID')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Predictive Uncertainty (Entropy)")
    plt.imshow(entropy_map, cmap='inferno')
    plt.colorbar(label='Entropy')
    plt.axis('off')
    
    path_2d = os.path.join(output_dir, f'cyclist_2d_analysis_{idx}.png')
    plt.savefig(path_2d)
    print(f"Saved 2D visualization to {path_2d}")
    plt.close()

    # 2. 3D Project Visualization (Points painted with Uncertainty)
    print("Generating 3D Painted Visualization...")
    calib = Calibration(calib_path)
    lidar_points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    
    # We can assume the painting logic is consistent, or we can just sample the entropy we already computed
    # Let's manually project and sample to be sure we visualize exactly what's painted
    
    pts_3d = lidar_points[:, :3]
    pts_2d, depth = calib.project_velo_to_image(pts_3d)
    
    H, W = entropy_map.shape
    mask = filter_points_in_image(pts_2d, (H, W))
    mask &= (depth > 0)
    
    pts_valid = pts_2d[mask]
    
    # Sample entropy for these valid points
    u = np.clip(np.round(pts_valid[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(pts_valid[:, 1]).astype(int), 0, H - 1)
    entropy_vals = entropy_map[v, u]
    
    # Load original image for background
    img_np = np.array(original_img)
    
    plt.figure(figsize=(12, 4))
    plt.title(f"Painted Point Cloud (Uncertainty) - Sample {idx}")
    plt.imshow(img_np) # Show image as background
    plt.axis('off')
    
    # Scatter points
    # Use small marker size and alpha for better look
    sc = plt.scatter(pts_valid[:, 0], pts_valid[:, 1], c=entropy_vals, cmap='inferno', s=1, alpha=0.8)
    plt.colorbar(sc, label='Point Entropy')
    plt.xlim(0, W)
    plt.ylim(H, 0) # Flip Y to match image coords
    
    path_3d = os.path.join(output_dir, f'cyclist_3d_painted_{idx}.png')
    plt.savefig(path_3d)
    print(f"Saved 3D visualization to {path_3d}")
    plt.close()

    print("Done.")

if __name__ == "__main__":
    main()
