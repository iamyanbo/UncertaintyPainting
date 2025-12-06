import sys
import os
import numpy as np
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.painting import FeaturePainter
from src.projection import Calibration

def parse_args():
    parser = argparse.ArgumentParser(description='Paint KITTI dataset with Uncertainty Features')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory of KITTI data')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'testing'], help='Data split')
    parser.add_argument('--model', type=str, default='deeplabv3_resnet50', help='Segmentation model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    base_dir = os.path.join(args.data_root, args.split)
    image_dir = os.path.join(base_dir, 'image_2')
    velo_dir = os.path.join(base_dir, 'velodyne')
    calib_dir = os.path.join(base_dir, 'calib')
    
    # Output directory
    output_dir = os.path.join(base_dir, 'velodyne_painted')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Processing {args.split} set...")
    print(f"Input: {velo_dir}")
    print(f"Output: {output_dir}")
    
    # Get list of files
    # We assume file names match: 000000.png, 000000.bin, 000000.txt
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return

    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    # Initialize Painter
    painter = FeaturePainter(segmentation_model_name=args.model)
    
    for filename in tqdm(files):
        idx = filename.split('.')[0]
        
        image_path = os.path.join(image_dir, filename)
        velo_path = os.path.join(velo_dir, f"{idx}.bin")
        calib_path = os.path.join(calib_dir, f"{idx}.txt")
        output_path = os.path.join(output_dir, f"{idx}.bin")
        
        if not os.path.exists(velo_path):
            print(f"Warning: LiDAR file missing for {idx}")
            continue
        if not os.path.exists(calib_path):
            print(f"Warning: Calib file missing for {idx}")
            continue
            
        # Load Data
        calib = Calibration(calib_path)
        lidar_points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
        
        # Paint
        try:
            painted_points = painter.paint(image_path, lidar_points, calib)
            
            # Save
            painted_points.astype(np.float32).tofile(output_path)
        except Exception as e:
            print(f"Error processing {idx}: {e}")

    print("Done!")

if __name__ == "__main__":
    main()
