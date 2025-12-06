import torch
import os

ckpt_path = 'OpenPCDet/output/kitti_models/pointpillar_painted_full/default/ckpt/checkpoint_epoch_80.pth'
clean_ckpt_path = 'OpenPCDet/output/kitti_models/pointpillar_painted_full/default/ckpt/checkpoint_epoch_80_clean.pth'

import numpy
print(f"NumPy Version: {numpy.__version__}")
print(f"Loading checkpoint from {ckpt_path}...")
try:
    # Load with explicit weights_only=False to allow numpy globals
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print("Checkpoint loaded successfully.")
    
    # Create a new dictionary with ONLY the essential keys
    # We specifically exclude 'optimizer_state' and 'version' if they contain numpy objects
    clean_checkpoint = {
        'model_state': checkpoint['model_state'],
        'epoch': checkpoint['epoch'],
        'it': checkpoint.get('it', 0)
    }
    
    print("Saving clean checkpoint...")
    torch.save(clean_checkpoint, clean_ckpt_path)
    print(f"Clean checkpoint saved to {clean_ckpt_path}")
    
except Exception as e:
    print(f"FAILED to process checkpoint: {e}")
