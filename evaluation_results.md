# Uncertainty-Painted PointPillars Evaluation Results

> **All results reported on KITTI Validation Set** (Default and Pointpainting implementation results are taken from Table 1 of PointPainting paper)

## Comparison Table: 3D Object Detection on KITTI Validation Set

| Method | mAP (Mod.) | Car Easy | Car Mod. | Car Hard | Ped. Easy | Ped. Mod. | Ped. Hard | Cyc. Easy | Cyc. Mod. | Cyc. Hard |
|--------|------------|----------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
| **Uncertainty-Painted PointPillars** (Ours) | 76.52 | 90.10 | 87.81 | 84.46 | 63.57 | 59.56 | 57.05 | 85.80 | **82.20** | 76.68 |
| **Uncertainty-Painted SECOND** (Ours) | **79.08** | **97.10** | **88.62** | 86.41 | 69.26 | 67.19 | 65.00 | **88.15** | 81.43 | **79.85** |

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
