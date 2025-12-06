"""
Create train/val splits for KITTI dataset.
This just creates index files - NO data processing!
"""
import os

def create_imagesets(data_root='data/training', train_ratio=0.8):
    """Generate train.txt and val.txt from existing painted point clouds."""
    
    velodyne_painted = os.path.join(data_root, 'velodyne_painted')
    
    if not os.path.exists(velodyne_painted):
        print(f"Error: {velodyne_painted} not found!")
        return
    
    # Get all indices from painted files
    files = sorted([f for f in os.listdir(velodyne_painted) if f.endswith('.bin')])
    indices = [f.split('.')[0] for f in files]
    
    print(f"Found {len(indices)} painted point clouds")
    
    # Split into train/val
    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create ImageSets directory
    imageset_dir = 'data/ImageSets'
    os.makedirs(imageset_dir, exist_ok=True)
    
    # Write train.txt
    with open(os.path.join(imageset_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_indices))
    
    # Write val.txt
    with open(os.path.join(imageset_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_indices))
    
    print(f"Created train.txt: {len(train_indices)} samples")
    print(f"Created val.txt: {len(val_indices)} samples")

if __name__ == '__main__':
    create_imagesets()
