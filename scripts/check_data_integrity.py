"""
Check data integrity for KITTI dataset.
Verifies that for every sample in train.txt/val.txt, all required files exist.
"""
import os
from pathlib import Path

def check_integrity(data_root='data'):
    """Check if all files exist for samples in ImageSets."""
    
    imageset_dir = os.path.join(data_root, 'ImageSets')
    base_dir = os.path.join(data_root, 'training')
    
    # Check train.txt
    with open(os.path.join(imageset_dir, 'train.txt'), 'r') as f:
        train_indices = [x.strip() for x in f.readlines()]
        
    # Check val.txt
    with open(os.path.join(imageset_dir, 'val.txt'), 'r') as f:
        val_indices = [x.strip() for x in f.readlines()]
        
    all_indices = train_indices + val_indices
    print(f"Checking {len(all_indices)} samples...")
    
    missing = []
    
    for idx in all_indices:
        # Check Image
        img_path = os.path.join(base_dir, 'image_2', f'{idx}.png')
        if not os.path.exists(img_path):
            missing.append((idx, 'image'))
            continue
            
        # Check Calib
        calib_path = os.path.join(base_dir, 'calib', f'{idx}.txt')
        if not os.path.exists(calib_path):
            missing.append((idx, 'calib'))
            continue
            
        # Check Label (only for training split usually, but let's check if they exist)
        label_path = os.path.join(base_dir, 'label_2', f'{idx}.txt')
        if not os.path.exists(label_path):
            # Labels might not exist for testing data, but we are looking at training folder
            missing.append((idx, 'label'))
            continue
            
        # Check Painted LiDAR
        lidar_path = os.path.join(base_dir, 'velodyne_painted', f'{idx}.bin')
        if not os.path.exists(lidar_path):
            missing.append((idx, 'velodyne_painted'))
            continue

    if missing:
        print(f"Found {len(missing)} samples with missing files:")
        for idx, missing_type in missing[:10]:
            print(f"  {idx}: missing {missing_type}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")
            
        # Suggest fixing
        print("\nSuggest regenerating ImageSets with only valid samples.")
        return False
    else:
        print("✓ All samples have complete data (Image, Calib, Label, Painted LiDAR)")
        return True

if __name__ == '__main__':
    check_integrity()
