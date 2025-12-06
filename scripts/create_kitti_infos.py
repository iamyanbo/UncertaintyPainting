"""
Create KITTI dataset info files for painted point clouds.
This script generates the .pkl files needed by OpenPCDet.
"""
import pickle
import os
import sys
from pathlib import Path

# Add OpenPCDet to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'OpenPCDet'))

from pcdet.datasets.kitti import kitti_dataset
import pcdet.datasets.kitti.kitti_utils as kitti_utils

def create_kitti_infos(data_path, save_path):
    """
    Create info files for KITTI dataset.
    This reads from the existing painted point clouds.
    """
    dataset = kitti_dataset.KittiDataset(
        dataset_cfg=None,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        root_path=Path(data_path),
        training=True
    )
    
    # Generate info files
    kitti_dataset.create_kitti_infos(
        dataset_cfg=None,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=Path(data_path),
        save_path=Path(save_path)
    )
    
    print("Dataset info files created successfully!")

if __name__ == '__main__':
    data_path = 'data'
    save_path = 'OpenPCDet/data/kitti'
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    create_kitti_infos(data_path, save_path)
