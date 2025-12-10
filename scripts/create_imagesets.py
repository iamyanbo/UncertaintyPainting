"""
Create train/val splits for KITTI dataset.
This just creates index files - NO data processing!
"""
import os

def create_imagesets(data_root='data', train_ratio=0.8):
    """Generate train.txt, val.txt, and test.txt."""
    
    # 1. Training/Validation
    train_dir = os.path.join(data_root, 'training', 'velodyne_painted')
    imageset_dir = os.path.join(data_root, 'ImageSets')
    os.makedirs(imageset_dir, exist_ok=True)

    if os.path.exists(train_dir):
        files = sorted([f for f in os.listdir(train_dir) if f.endswith('.bin')])
        indices = [f.split('.')[0] for f in files]
        print(f"Found {len(indices)} training samples")
        
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        with open(os.path.join(imageset_dir, 'train.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_indices))
        with open(os.path.join(imageset_dir, 'val.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_indices))
            
        print(f"Created train.txt: {len(train_indices)} samples")
        print(f"Created val.txt: {len(val_indices)} samples")
    
    # 2. Testing
    test_dir = os.path.join(data_root, 'testing', 'velodyne_painted')
    if os.path.exists(test_dir):
        files = sorted([f for f in os.listdir(test_dir) if f.endswith('.bin')])
        indices = [f.split('.')[0] for f in files]
        print(f"Found {len(indices)} testing samples")
        
        with open(os.path.join(imageset_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(indices))
        print(f"Created test.txt: {len(indices)} samples")
    else:
        # Fallback: check standard velodyne if painted doesn't exist yet (so we can run painting)
        test_dir_raw = os.path.join(data_root, 'testing', 'velodyne')
        if os.path.exists(test_dir_raw):
             files = sorted([f for f in os.listdir(test_dir_raw) if f.endswith('.bin')])
             indices = [f.split('.')[0] for f in files]
             with open(os.path.join(imageset_dir, 'test.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(indices))
             print(f"Created test.txt (from raw velodyne): {len(indices)} samples")

if __name__ == '__main__':
    create_imagesets()
