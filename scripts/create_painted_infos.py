"""
Generate KITTI infos and GT database for Painted Point Clouds (26 features).
"""
import sys
import pickle
from pathlib import Path
import yaml
from easydict import EasyDict

# Add OpenPCDet to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'OpenPCDet'))

from pcdet.datasets.kitti.kitti_painted_dataset import KittiPaintedDataset

def create_painted_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    print(f"Generating infos for painted dataset at {data_path}")
    
    dataset = KittiPaintedDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    
    train_split, val_split = 'train', 'val'
    
    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'
    
    print('---------------Start to generate data infos---------------')
    
    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)
    
    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)
    
    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)
    
    # Skip test split for now as we don't have labels/painted data for testing usually? 
    # User said "i currently have all the data in lidar painted".
    # But usually testing doesn't have labels, so we can't generate GT database anyway.
    # We'll skip test infos for now to save time/errors.
    
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    # This will use KittiPaintedDataset.get_lidar -> 26 features
    dataset.create_groundtruth_database(train_filename, split=train_split)
    
    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    # Config path
    cfg_path = Path(__file__).parent.parent / 'OpenPCDet/tools/cfgs/dataset_configs/kitti_painted_dataset.yaml'
    dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
    
    # Data path (use the junction in OpenPCDet/data/kitti)
    ROOT_DIR = Path(__file__).parent.parent / 'OpenPCDet'
    data_path = ROOT_DIR / 'data' / 'kitti'
    
    create_painted_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=data_path,
        save_path=data_path
    )
