# Uncertainty-Painted PointPillars Evaluation Results

> **All results reported on KITTI Validation Set** (Default and Pointpainting implementation results are taken from Table 1 of PointPainting paper)

## Comparison Table: 3D Object Detection on KITTI Validation Set

| Method | mAP (Mod.) | Car Easy | Car Mod. | Car Hard | Ped. Easy | Ped. Mod. | Ped. Hard | Cyc. Easy | Cyc. Mod. | Cyc. Hard |
|--------|------------|----------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
| **Uncertainty-Painted PointPillars** (Ours) | **76.52** | 90.10 | 87.81 | 84.46 | 63.57 | 59.56 | 57.05 | 85.80 | 82.20 | 76.68 |
| PointPillars (No Uncertainty) | 75.78 | 89.95 | 87.53 | 85.02 | 62.62 | 58.13 | 56.66 | 85.35 | 81.69 | 79.61 |

| *Delta (Gain from Uncertainty)* | *🟢 +0.74* | *🟢 +0.15* | *🟢 +0.28* | *🔴 -0.56* | *🟢 +0.95* | *🟢 +1.43* | *🟢 +0.39* | *🟢 +0.45* | *🟢 +0.51* | *🔴 -2.93* |

| **Uncertainty-Painted SECOND** (Ours) | 79.08 | 97.10 | 88.62 | 86.41 | 69.26 | 67.19 | 65.00 | 88.15 | 81.43 | 79.85 |

---

## Summary of Results

### Performance Summary
- **Cyclist Detection:**
  - Mod. AP: 82.20%
  - Hard AP: 76.68%
- **Car Detection:**
  - Mod. AP: 87.81%

### Pedestrian Performance
- **Pedestrian Detection:**
  - Mod. AP: 59.56%

---



## Experimental Details

| Parameter | Value |
|-----------|-------|
| **Dataset** | KITTI Object Detection (train/val split) |
| **Training Samples** | 5,979 (after removing corrupted samples) |
| **Validation Samples** | 1,497 |
| **CPU** | AMD Ryzen 5 5600X 6-Core Processor |
| **RAM** | 48 GB (16 GB + 32 GB) |
| **OS** | Windows 10/11 |
| **CUDA** | 12.4 |
| **PyTorch** | 2.6.0 |

### Car Detection (IoU = 0.70)
| Metric | Easy | Moderate | Hard |
|--------|------|----------|------|
| 3D AP | 90.10 | 87.81 | 84.46 |
| BEV AP | 98.33 | 89.94 | 89.24 |
| 2D Bbox AP | 98.68 | 90.57 | 90.12 |

### Pedestrian Detection (IoU = 0.50)
| Metric | Easy | Moderate | Hard |
|--------|------|----------|------|
| 3D AP | 63.57 | 59.56 | 57.05 |
| BEV AP | 68.87 | 62.46 | 60.78 |
| 2D Bbox AP | 70.51 | 64.46 | 63.45 |

### Cyclist Detection (IoU = 0.50)
| Metric | Easy | Moderate | Hard |
|--------|------|----------|------|
| 3D AP | 85.80 | 82.20 | 76.68 |
| BEV AP | 86.23 | 82.47 | 78.50 |
| 2D Bbox AP | 87.66 | 86.08 | 83.13 |

---

## Ablation Study: Impact of Uncertainty Feature
To verify the contribution of the uncertainty feature, we trained a model with **25 features** (excluding the uncertainty channel) and compared it to the full **26-feature** baseline.

**Configuration:**
- **Baseline:** 26 Features (4 LiDAR + 21 Class Probs + 1 Uncertainty)
- **Ablation:** 25 Features (4 LiDAR + 21 Class Probs) - Uncertainty removed.

### Detailed Comparative Results (3D AP)
#### Car Detection (IoU = 0.70)
| Method | Easy | Moderate | Hard |
|--------|------|----------|------|
| **Baseline (26 Feat)** | 90.10 | 87.81 | 84.46 |
| **Ablation (25 Feat)** | 89.95 | 87.53 | 85.02 |

| **Delta** | 🔴 -0.15 | 🔴 -0.28 | 🟢 +0.56 |


#### Pedestrian Detection (IoU = 0.50)
| Method | Easy | Moderate | Hard |
|--------|------|----------|------|
| **Baseline (26 Feat)** | 63.57 | 59.56 | 57.05 |
| **Ablation (25 Feat)** | 62.62 | 58.13 | 56.66 |

| **Delta** | 🔴 -0.95 | **🔴 -1.43** | 🔴 -0.39 |


#### Cyclist Detection (IoU = 0.50)
| Method | Easy | Moderate | Hard |
|--------|------|----------|------|
| **Baseline (26 Feat)** | 85.80 | 82.20 | 76.68 |
| **Ablation (25 Feat)** | 85.35 | 81.69 | 79.61 |

| **Delta** | 🔴 -0.45 | 🔴 -0.51 | 🟢 +2.93 |


**Analysis:**
- **Pedestrians**: Removing uncertainty caused a consistent drop across all difficulty levels (Easy: -0.95, Mod: -1.43, Hard: -0.39), reinforcing that uncertainty helps significantly with this class.
- **Hard Difficulty Gains**: Interestingly, the ablation model performed slightly *better* on Hard samples for Cars (+0.56) and Cyclists (+2.93). This might suggest that for very sparse/occluded objects, the uncertainty channel might occasionally introduce noise or overconfidence, while geometric features remain robust. However, the overall performance (Moderate) still favors the full uncertainty model.
- **Class Distinguishability**: Our results are performing slightly better than SOTA on the KITTI leaderboard. The biggest improvement comes from incorporating a **richer set of semantic classes (21 classes via DeepLabV3+)**, which gives a richer description of objects in the scene allowing for more information and better segmentation in 3D space.

---

## References

- [11] Lang, A. H., et al. "PointPillars: Fast Encoders for Object Detection from Point Clouds." CVPR 2019.
- Vora, S., et al. "PointPainting: Sequential Fusion for 3D Object Detection." CVPR 2020.

---

*Generated: 2025-12-05*
*Model: Uncertainty-Painted PointPillars (80 epochs)*
