"""
Check samples for crashes in roiaware_pool3d_utils.
"""
import sys
import torch
import numpy as np
from pathlib import Path
import yaml
from easydict import EasyDict

# Add OpenPCDet to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'OpenPCDet'))

from pcdet.datasets.kitti.kitti_painted_dataset import KittiPaintedDataset
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

def check_samples(indices, dataset):
    for idx in indices:
        print(f"Checking {idx}...", flush=True)
        try:
            info = dataset.get_infos(sample_id_list=[idx], num_workers=1, has_label=True, count_inside_pts=False)[0]
            
            sample_idx = info['point_cloud']['lidar_idx']
            points = dataset.get_lidar(sample_idx)
            annos = info['annos']
            gt_boxes = annos['gt_boxes_lidar']
            
            # This is the line that crashes
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()
            
            num_obj = gt_boxes.shape[0]
            for i in range(num_obj):
                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]
                
                # Try writing to a dummy file to ensure no I/O crash
                # We don't need to actually write to disk to test if it crashes, 
                # but let's do a dummy operation that touches the data
                _ = gt_points.tobytes()

            
        except Exception as e:
            print(f"Exception on {idx}: {e}", flush=True)
            # If it's a python exception, we catch it. If segfault, we die.

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_samples_batch.py <start_index> <end_index>")
        sys.exit(1)
        
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    
    # Config path
    cfg_path = Path(__file__).parent.parent / 'OpenPCDet/tools/cfgs/dataset_configs/kitti_painted_dataset.yaml'
    dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
    
    # Data path
    ROOT_DIR = Path(__file__).parent.parent / 'OpenPCDet'
    data_path = ROOT_DIR / 'data' / 'kitti'
    
    dataset = KittiPaintedDataset(dataset_cfg=dataset_cfg, class_names=['Car', 'Pedestrian', 'Cyclist'], root_path=data_path, training=False)
    dataset.set_split('train')
    
    # Get all train indices
    all_indices = dataset.sample_id_list
    
    # Slice
    indices_to_check = all_indices[start_idx:end_idx]
    
    check_samples(indices_to_check, dataset)
