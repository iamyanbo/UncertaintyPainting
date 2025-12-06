"""
Custom KITTI dataset that loads painted point clouds (26 features) instead of raw LiDAR (4 features).
This DOES NOT re-paint - it loads from pre-computed velodyne_painted/ folder.
"""
import copy
import pickle
import numpy as np
from pathlib import Path

from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from .kitti_dataset import KittiDataset


class KittiPaintedDataset(KittiDataset):
    """
    Extends KittiDataset to load painted point clouds with 26 features:
    - 4 base: (x, y, z, intensity)
    - 21 class probabilities from DeepLabV3
    - 1 uncertainty (entropy)
    """
    
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            dataset_cfg: EasyDict
            class_names: list
            training: bool
            root_path: Path
            logger: Logger
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training,
            root_path=root_path, logger=logger
        )
        
    def get_lidar(self, idx):
        """
        Load painted point cloud from velodyne_painted/ folder.
        Returns: (N, 26) array instead of (N, 4)
        """
        # Change folder from 'velodyne' to 'velodyne_painted'
        lidar_file = self.root_split_path / 'velodyne_painted' / ('%s.bin' % idx)
        assert lidar_file.exists(), f"Painted point cloud not found: {lidar_file}"
        
        # Load with 26 features instead of 4
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 26)
        
        return points
