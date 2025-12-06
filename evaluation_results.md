# Uncertainty-Painted PointPillars Evaluation Results

> **All results reported on KITTI Validation Set** (consistent with PointPainting Table 1 for fair comparison)

## Comparison Table: 3D Object Detection on KITTI Validation Set

| Method | mAP (Mod.) | Car Easy | Car Mod. | Car Hard | Ped. Easy | Ped. Mod. | Ped. Hard | Cyc. Easy | Cyc. Mod. | Cyc. Hard |
|--------|------------|----------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
| **PointPillars** [11] | 73.78 | 90.09 | 87.57 | 86.03 | 71.97 | 67.84 | 62.41 | 85.74 | 65.92 | 62.40 |
| **PaintedPointPillars** (PointPainting) | 76.27 | 90.01 | 87.65 | 85.56 | 77.25 | 72.41 | 67.53 | 81.72 | 68.76 | 63.99 |
| **Delta (Painted vs Base)** | <span style="color:green">+2.50</span> | <span style="color:red">-0.08</span> | <span style="color:green">+0.08</span> | <span style="color:red">-0.47</span> | <span style="color:green">+5.28</span> | <span style="color:green">+4.57</span> | <span style="color:green">+5.12</span> | <span style="color:red">-4.02</span> | <span style="color:green">+2.84</span> | <span style="color:green">+1.59</span> |
| **Uncertainty-Painted PointPillars** (Ours) | **76.52** | **90.10** | **87.81** | **84.46** | **63.57** | **59.56** | **57.05** | **85.80** | **82.20** | **76.68** |
| **Delta (Ours vs Base)** | <span style="color:green">+2.74</span> | <span style="color:green">+0.01</span> | <span style="color:green">+0.24</span> | <span style="color:red">-1.57</span> | <span style="color:red">-8.40</span> | <span style="color:red">-8.28</span> | <span style="color:red">-5.36</span> | <span style="color:green">+0.06</span> | <span style="color:green">+16.28</span> | <span style="color:green">+14.28</span> |
| **Delta (Ours vs Painted)** | <span style="color:green">+0.25</span> | <span style="color:green">+0.09</span> | <span style="color:green">+0.16</span> | <span style="color:red">-1.10</span> | <span style="color:red">-13.68</span> | <span style="color:red">-12.85</span> | <span style="color:red">-10.48</span> | <span style="color:green">+4.08</span> | <span style="color:green">+13.44</span> | <span style="color:green">+12.69</span> |

> **Note:** All values are 3D AP (%) at standard KITTI IoU thresholds (Car: 0.70, Pedestrian: 0.50, Cyclist: 0.50)

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

## Analysis: Why Pedestrian Performance is Lower

### Shannon Entropy at Object Edges

The uncertainty-painted features use **Shannon entropy** to quantify prediction confidence:

$$H = -\sum_{i=1}^{K} p_i \log(p_i)$$

**Observation:** Shannon entropy produces **high uncertainty values at pedestrian edges**.

#### Root Cause:
1. **Pedestrians have complex, irregular boundaries** (arms, legs, clothing edges)
2. The semantic segmentation network produces **ambiguous class probabilities** at these boundaries
3. High entropy at edges means the painted features contain **noisy uncertainty signals** for pedestrian points
4. The 3D detector may learn to **down-weight** pedestrian detections due to this noise

#### Why Cyclists Perform Better:
1. **Cyclists have more regular silhouettes** (bicycle frame provides consistent structure)
2. The bike + person combination creates a **larger, more stable region** in the image
3. Lower edge-to-area ratio means fewer boundary pixels with high entropy
4. Uncertainty values are more **consistent and informative** for cyclist detection

---

## Experimental Details

| Parameter | Value |
|-----------|-------|
| **Dataset** | KITTI Object Detection (train/val split) |
| **Training Samples** | ~5,979 (after removing corrupted samples) |
| **Validation Samples** | ~1,500 |
| **Features per Point** | 26 (x, y, z, intensity + 21 class probs + 1 entropy) |
| **Epochs** | 80 |
| **Batch Size** | 4 |
| **Optimizer** | Adam OneCycle (LR: 0.003) |
| **Checkpoint** | `checkpoint_epoch_80.pth` |

### Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 3060 Ti (8 GB VRAM) |
| **CPU** | AMD Ryzen 5 5600X 6-Core Processor |
| **RAM** | 48 GB (16 GB + 32 GB) |
| **OS** | Windows 10/11 |
| **CUDA** | 12.4 |
| **PyTorch** | 2.6.0 |

### Training Time

| Model | Training Time | Time per Epoch |
|-------|---------------|----------------|
| **PointPillars** | ~11 hours | ~8.5 min |
| **SECOND** | (TODO: Update after training) | ~TBD |
| **PointRCNN** | (TODO: Update after training) | ~TBD |

---

## Raw Evaluation Metrics (Our Model)

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
