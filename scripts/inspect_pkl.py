import pickle
import sys

file_path = r'c:/Users/yanbo/Downloads/Papers/uncertainty_painting/OpenPCDet/output/kitti_models/pointpillar_painted_full/default/eval/epoch_80/val/default/result.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} samples")
if len(data) > 0:
    print("Keys in first sample:")
    print(data[0].keys())
    print("Sample[0]:")
    for k, v in data[0].items():
        print(f"{k}: {type(v)}")
        if hasattr(v, 'shape'):
            print(f"  shape: {v.shape}")
