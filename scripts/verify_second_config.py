
import sys
import yaml
from pathlib import Path
import torch
import numpy as np

# Add OpenPCDet to path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'OpenPCDet'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network

class MockPointFeatureEncoder:
    def __init__(self, num_point_features):
        self.num_point_features = num_point_features

class MockDataset:
    def __init__(self, class_names, num_point_features):
        self.class_names = class_names
        # Random grid size
        self.grid_size = np.array([512, 512, 1]) 
        self.voxel_size = [0.05, 0.05, 0.1]
        self.point_cloud_range = [0, -40, -3, 70.4, 40, 1]
        self.depth_downsample_factor = None
        self.point_feature_encoder = MockPointFeatureEncoder(num_point_features)

def verify_config():
    cfg_file = 'second_painted_no_uncertainty.yaml'
    print(f"Loading config from {cfg_file}...")
    cfg_from_yaml_file(cfg_file, cfg)
    
    # Check key configurations
    print(f"Dataset: {cfg.DATA_CONFIG.DATASET}")
    
    expected_used = 25
    if hasattr(cfg.DATA_CONFIG, 'POINT_FEATURE_ENCODING'):
        actual_used = len(cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list)
        print(f"Used Feature List Length: {actual_used}")
        if actual_used != expected_used:
            print(f"ERROR: Expected {expected_used} used features, got {actual_used}")
            return
    
    num_point_features = cfg.DATA_CONFIG.NUM_POINT_FEATURES
    print(f"NUM_POINT_FEATURES: {num_point_features}")
    if num_point_features != 25:
         print(f"WARNING: NUM_POINT_FEATURES is {num_point_features}, expected 25")

    # Try to build model (dry run)
    print("Building model...")
    try:
        mock_dataset = MockDataset(
            class_names=cfg.CLASS_NAMES,
            num_point_features=num_point_features
        )
        
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=mock_dataset)
        print("Model built successfully.")
        print("Model Architecture:")
        print(model)
        print("SUCCESS: Configured for 25 features (No Uncertainty).")
        
    except Exception as e:
        print(f"ERROR building model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    verify_config()
