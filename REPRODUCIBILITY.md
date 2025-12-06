# Uncertainty-Painted 3D Object Detection - Reproducibility Guide

This guide provides step-by-step instructions to reproduce the uncertainty-painted PointPillars, SECOND, and PointRCNN results on the KITTI dataset.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| **OS** | Windows 10/11 or Linux |
| **Python** | 3.10.x |
| **CUDA** | 12.4 |
| **GPU** | NVIDIA GPU with 8+ GB VRAM |
| **RAM** | 32+ GB recommended |

---

## 1. Environment Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/uncertainty_painting.git
cd uncertainty_painting
```

### 1.2 Install Python Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.26.4
pip install opencv-python pillow matplotlib tqdm easydict pyyaml scikit-image
pip install spconv-cu124
```

### 1.3 Install OpenPCDet

```bash
cd OpenPCDet
pip install -r requirements.txt
python setup.py develop
cd ..
```

> **Note:** On Windows, use the fork: `Uzukidd/OpenPCDet-Win11-Compatible`

---

## 2. Dataset Preparation

### 2.1 Download KITTI Dataset

Download from [KITTI 3D Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):
- `data_object_velodyne.zip` (LiDAR point clouds)
- `data_object_image_2.zip` (Left color images)
- `data_object_calib.zip` (Calibration files)
- `data_object_label_2.zip` (Training labels)

### 2.2 Organize Dataset Structure

```text
uncertainty_painting/
├── OpenPCDet/
│   └── data/
│       └── kitti/
│           ├── ImageSets/
│           │   ├── train.txt
│           │   └── val.txt
│           └── training/
│               ├── velodyne/          # Original LiDAR (.bin)
│               ├── velodyne_painted/  # Painted LiDAR (.bin) - generated
│               ├── image_2/           # RGB images (.png)
│               ├── calib/             # Calibration files (.txt)
│               └── label_2/           # Labels (.txt)
```

### 2.3 Generate Train/Val Splits

```bash
python scripts/create_imagesets.py
```

This creates `train.txt` and `val.txt` in `OpenPCDet/data/kitti/ImageSets/`.

---

## 3. Uncertainty Painting Pipeline

### 3.1 Run the Painting Script

```bash
python scripts/04_process_dataset.py
```

This script:
1. Loads each LiDAR point cloud
2. Projects points onto the camera image
3. Runs semantic segmentation (DeepLabV3+)
4. Paints each point with 21 class probabilities + Shannon entropy
5. Saves painted point clouds to `velodyne_painted/`

**Output:** Each `.bin` file in `velodyne_painted/` has 26 features per point:
- `[x, y, z, intensity, class_0, class_1, ..., class_20, entropy]`

### 3.2 Generate Dataset Info Files

```bash
python scripts/create_painted_infos.py --workers 1
```

This generates:
- `kitti_infos_train.pkl`
- `kitti_infos_val.pkl`
- `kitti_dbinfos_train.pkl` (GT database for augmentation)

---

## 4. Training

### 4.1 Train PointPillars

```bash
cd OpenPCDet/tools
python train.py --cfg_file cfgs/kitti_models/pointpillar_painted_full.yaml --batch_size 4 --epochs 80
```

**Expected time:** ~11 hours on RTX 3060 Ti

### 4.2 Train SECOND (VoxelNet)

```bash
python train.py --cfg_file cfgs/kitti_models/second_painted_full.yaml --batch_size 4 --epochs 80
```

**Expected time:** ~12-15 hours on RTX 3060 Ti



---

## 5. Evaluation

### 5.1 Evaluate PointPillars

```bash
python test.py --cfg_file cfgs/kitti_models/pointpillar_painted_full.yaml \
    --batch_size 4 \
    --ckpt ../output/kitti_models/pointpillar_painted_full/default/ckpt/checkpoint_epoch_80.pth
```

### 5.2 Evaluate SECOND

```bash
python test.py --cfg_file cfgs/kitti_models/second_painted_full.yaml \
    --batch_size 4 \
    --ckpt ../output/kitti_models/second_painted_full/default/ckpt/checkpoint_epoch_80.pth
```



**Results** are saved to:
```
output/<model>/default/eval/epoch_80/val/default/
├── log_eval_*.txt   # Human-readable metrics
└── result.pkl       # Serialized results
```

---

## 6. Troubleshooting

### NumPy Version Error
If you see `WeightsUnpickler error: numpy._core.multiarray`:
```bash
pip install numpy==1.26.4
python scripts/create_painted_infos.py --workers 1  # Regenerate infos
```

### Road Plane Crash
If training crashes with `KeyError: 'road_plane'`:
- Ensure config has `USE_ROAD_PLANE: False` in the `gt_sampling` section

### CUDA Out of Memory
- Reduce `--batch_size` to 2
- Reduce `MAX_NUMBER_OF_VOXELS` in config

---

## 7. Expected Results

| Model | Car (Mod.) | Pedestrian (Mod.) | Cyclist (Mod.) |
|-------|------------|-------------------|----------------|
| **Uncertainty-Painted PointPillars** | 87.81 | 59.56 | 82.20 |
| **Uncertainty-Painted SECOND** | TBD | TBD | TBD |


---

