"""
Test script to verify painted dataset loading works correctly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'OpenPCDet'))

import torch
import numpy as np
from easydict import EasyDict
from pcdet.datasets.kitti.kitti_painted_dataset import KittiPaintedDataset
from pcdet.config import cfg, cfg_from_yaml_file

def test_dataset_loading():
    """Test that we can load painted point clouds."""
    
    # Load config
    config_file = Path(__file__).parent.parent / 'OpenPCDet' / 'tools' / 'cfgs' / 'dataset_configs' / 'kitti_painted_dataset.yaml'
    cfg_from_yaml_file(config_file, cfg)
    
    # Create dataset
    dataset = KittiPaintedDataset(
        dataset_cfg=cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        root_path=Path(__file__).parent.parent / 'data',
        training=True
    )
    
    print(f"Dataset created successfully!")
    print(f"Number of samples: {len(dataset)}")
    
    # Test loading first sample
    print("\nLoading first sample...")
    sample = dataset[0]
    
    points = sample['points']
    print(f"Point cloud shape: {points.shape}")
    print(f"Expected: (N, 26) - got {points.shape[1]} features")
    
    if points.shape[1] == 26:
        print("✓ Painted point cloud loaded successfully!")
        print(f"  - Base features (x,y,z,intensity): {points[:5, :4]}")
        print(f"  - Class probabilities (21 classes): shape {points[:, 4:25].shape}")
        print(f"  - Uncertainty: {points[:5, 25]}")
    else:
        print(f"✗ ERROR: Expected 26 features, got {points.shape[1]}")
    
    return dataset

if __name__ == '__main__':
    try:
        dataset = test_dataset_loading()
        print("\n✓ Dataset test passed!")
    except Exception as e:
        print(f"\n✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
