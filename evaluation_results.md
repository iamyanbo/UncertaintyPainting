# Uncertainty-Painted PointPillars Evaluation Results

> **All results reported on KITTI Validation Set** (Default and Pointpainting implementation results are taken from Table 1 of PointPainting paper)

## Comparison Table: 3D Object Detection on KITTI Validation Set

| Method | mAP (Mod.) | Car Easy | Car Mod. | Car Hard | Ped. Easy | Ped. Mod. | Ped. Hard | Cyc. Easy | Cyc. Mod. | Cyc. Hard |
|--------|------------|----------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
| **PointPillars** [11] | 73.78 | 90.09 | 87.57 | 86.03 | 71.97 | 67.84 | 62.41 | 85.74 | 65.92 | 62.40 |
| **PaintedPointPillars** (PointPainting) | 76.27 | 90.01 | 87.65 | 85.56 | <u>**77.25**</u> | <u>**72.41**</u> | <u>**67.53**</u> | 81.72 | 68.76 | 63.99 |
| **Delta (Painted vs Base)** | $\color{green}{+2.50}$ | $\color{red}{-0.08}$ | $\color{green}{+0.08}$ | $\color{red}{-0.47}$ | $\color{green}{+5.28}$ | $\color{green}{+4.57}$ | $\color{green}{+5.12}$ | $\color{red}{-4.02}$ | $\color{green}{+2.84}$ | $\color{green}{+1.59}$ |
| **Uncertainty-Painted PointPillars** (Ours) | 76.52 | 90.10 | 87.81 | 84.46 | 63.57 | 59.56 | 57.05 | 85.80 | <u>**82.20**</u> | 76.68 |
| **Delta (Ours vs Base)** | $\color{green}{+2.74}$ | $\color{green}{+0.01}$ | $\color{green}{+0.24}$ | $\color{red}{-1.57}$ | $\color{red}{-8.40}$ | $\color{red}{-8.28}$ | $\color{red}{-5.36}$ | $\color{green}{+0.06}$ | $\color{green}{+16.28}$ | $\color{green}{+14.28}$ |
| **Delta (Ours vs Painted)** | $\color{green}{+0.25}$ | $\color{green}{+0.09}$ | $\color{green}{+0.16}$ | $\color{red}{-1.10}$ | $\color{red}{-13.68}$ | $\color{red}{-12.85}$ | $\color{red}{-10.48}$ | $\color{green}{+4.08}$ | $\color{green}{+13.44}$ | $\color{green}{+12.69}$ |
| **VoxelNet (SECOND)** [34] | 71.83 | 89.87 | 87.29 | 86.30 | 70.08 | 62.44 | 55.02 | 85.48 | 65.77 | 58.97 |
| **PaintedVoxelNet** (PointPainting) | 73.55 | 90.05 | 87.51 | <u>**86.66**</u> | 73.16 | 65.05 | 57.33 | 87.46 | 68.08 | 65.59 |
| **Delta (Painted vs Base)** | $\color{green}{+1.71}$ | $\color{green}{+0.18}$ | $\color{green}{+0.22}$ | $\color{green}{+0.36}$ | $\color{green}{+3.08}$ | $\color{green}{+2.61}$ | $\color{green}{+2.31}$ | $\color{green}{+1.98}$ | $\color{green}{+2.31}$ | $\color{green}{+6.62}$ |
| **Uncertainty-Painted SECOND** (Ours) | <u>**79.08**</u> | <u>**97.10**</u> | <u>**88.63**</u> | 86.41 | 69.26 | 67.19 | 65.00 | <u>**88.15**</u> | 81.43 | <u>**79.85**</u> |
| **Delta (Ours vs Base)** | $\color{green}{+7.25}$ | $\color{green}{+7.23}$ | $\color{green}{+1.34}$ | $\color{green}{+0.11}$ | $\color{red}{-0.82}$ | $\color{green}{+4.75}$ | $\color{green}{+9.98}$ | $\color{green}{+2.67}$ | $\color{green}{+15.66}$ | $\color{green}{+20.88}$ |
| **Delta (Ours vs Painted)** | $\color{green}{+5.53}$ | $\color{green}{+7.05}$ | $\color{green}{+1.12}$ | $\color{red}{-0.25}$ | $\color{red}{-3.90}$ | $\color{green}{+2.14}$ | $\color{green}{+7.67}$ | $\color{green}{+0.69}$ | $\color{green}{+13.35}$ | $\color{green}{+14.26}$ |

---

## Summary of Our Results

### Strong Performance
- **Cyclist Detection:** Significant improvement over both baseline and PaintedPointPillars
  - Mod. AP: 82.20% (vs 65.92% baseline, +16.28%)
  - Hard AP: 76.68% (vs 62.40% baseline, +14.28%)
- **Car Detection:** Comparable to state-of-the-art
  - Mod. AP: 87.81% (on par with painted methods)

### Weaker Performance
- **Pedestrian Detection:** Lower than expected
  - Mod. AP: 59.56% (vs 67.84% baseline, -8.28%)

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

## References

- [11] Lang, A. H., et al. "PointPillars: Fast Encoders for Object Detection from Point Clouds." CVPR 2019.
- Vora, S., et al. "PointPainting: Sequential Fusion for 3D Object Detection." CVPR 2020.

---

*Generated: 2025-12-05*
*Model: Uncertainty-Painted PointPillars (80 epochs)*
