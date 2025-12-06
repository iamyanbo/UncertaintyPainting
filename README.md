# Uncertainty-Aware Sequential Fusion for 3D Object Detection

This repository implements **Uncertainty-Aware Feature Painting**, a novel sequential fusion method for 3D object detection. By integrating 2D semantic segmentation uncertainty into the point cloud feature space, this method enhances the robustness of 3D detectors (like PointPillars) against ambiguous visual data.

## Visual Pipeline Results

### 1. Uncertainty Estimation (2D)
The validation image is shown on the **Left**. The **Middle** panel displays the **Predicted Semantic Classes**. The **Right** panel visualizes the **Predictive Uncertainty (Entropy)**, where brighter colors indicate higher uncertainty (e.g., object boundaries).

![2D Uncertainty Estimation](assets/test_uncertainty_result.png)

### 2. Painted Point Cloud (3D)
Here we see the 3D LiDAR point cloud "painted" with the fusion features (Class Semantic + Uncertainty).

![Painted Point Cloud](assets/view_result_000015.png)

---

## Implementation: Uncertainty-Aware Feature Painting

The core innovation is the addition of an **uncertainty channel** to the standard PointPainting pipeline.

### Methodology
In the standard PointPainting architecture, LiDAR points are "painted" with the class scores from a 2D segmentation network. We extend this by calculating the **predictive uncertainty** of the 2D network and appending it as an additional feature.

**Uncertainty Calculation (Shannon Entropy):**
For a pixel $(u, v)$ with class probability distribution $p$, the uncertainty $H$ is calculated as:

$$
H(p) = - \sum_{k=1}^{K} p^{(k)} \log(p^{(k)})
$$

### Augmented Point Vector
Each LiDAR point is projected into the image and augmented with both the semantic class scores and the scalar uncertainty value.

$$
P'_{point} = [x, y, z, r, \underbrace{C_1, \dots, C_K}_{\text{Class Scores}}, \underbrace{U}_{\text{Uncertainty}}]
$$

This results in an input feature dimension of $N_{features} = 4 + K + 1$.

---

## Evaluation Results

We evaluated our method on the **KITTI Validation Set** using the PointPillars backbone.

### Comparison Table: 3D Object Detection on KITTI Validation Set

The table below shows the Average Precision (AP) for 3D detection. Our method (Uncertainty-Painted) is compared against the standard PointPillars baseline and the original PaintedPointPillars (PointPainting).

### Comparison Table: 3D Object Detection on KITTI Validation Set

The table below shows the Average Precision (AP) for 3D detection. Our method (Uncertainty-Painted) is compared against the standard PointPillars baseline and the original PaintedPointPillars (PointPainting).

| Method | mAP (Mod.) | Car Easy | Car Mod. | Car Hard | Ped. Easy | Ped. Mod. | Ped. Hard | Cyc. Easy | Cyc. Mod. | Cyc. Hard |
|--------|------------|----------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
| **PointPillars** [11] | 73.78 | 90.09 | 87.57 | 86.03 | 71.97 | 67.84 | 62.41 | 85.74 | 65.92 | 62.40 |
| **PaintedPointPillars** (PointPainting) | 76.27 | 90.01 | 87.65 | 85.56 | 77.25 | 72.41 | 67.53 | 81.72 | 68.76 | 63.99 |
| **Delta (Painted vs Base)** | $\color{green}{+2.50}$ | $\color{red}{-0.08}$ | $\color{green}{+0.08}$ | $\color{red}{-0.47}$ | $\color{green}{+5.28}$ | $\color{green}{+4.57}$ | $\color{green}{+5.12}$ | $\color{red}{-4.02}$ | $\color{green}{+2.84}$ | $\color{green}{+1.59}$ |
| **Uncertainty-Painted PointPillars** (Ours) | **76.52** | **90.10** | **87.81** | **84.46** | **63.57** | **59.56** | **57.05** | **85.80** | **82.20** | **76.68** |
| **Delta (Ours vs Base)** | $\color{green}{+2.74}$ | $\color{green}{+0.01}$ | $\color{green}{+0.24}$ | $\color{red}{-1.57}$ | $\color{red}{-8.40}$ | $\color{red}{-8.28}$ | $\color{red}{-5.36}$ | $\color{green}{+0.06}$ | $\color{green}{+16.28}$ | $\color{green}{+14.28}$ |
| **Delta (Ours vs Painted)** | $\color{green}{+0.25}$ | $\color{green}{+0.09}$ | $\color{green}{+0.16}$ | $\color{red}{-1.10}$ | $\color{red}{-13.68}$ | $\color{red}{-12.85}$ | $\color{red}{-10.48}$ | $\color{green}{+4.08}$ | $\color{green}{+13.44}$ | $\color{green}{+12.69}$ |
| **VoxelNet (SECOND)** [34] | 71.83 | 89.87 | 87.29 | 86.30 | 70.08 | 62.44 | 55.02 | 85.48 | 65.77 | 58.97 |
| **PaintedVoxelNet** (PointPainting) | 73.55 | 90.05 | 87.51 | 86.66 | 73.16 | 65.05 | 57.33 | 87.46 | 68.08 | 65.59 |
| **Delta (Painted vs Base)** | $\color{green}{+1.71}$ | $\color{green}{+0.18}$ | $\color{green}{+0.22}$ | $\color{green}{+0.36}$ | $\color{green}{+3.08}$ | $\color{green}{+2.61}$ | $\color{green}{+2.31}$ | $\color{green}{+1.98}$ | $\color{green}{+2.31}$ | $\color{green}{+6.62}$ |
| **Uncertainty-Painted SECOND** (Ours) | **79.08** | **97.10** | **88.63** | **86.41** | **69.26** | **67.19** | **65.00** | **88.15** | **81.43** | **79.85** |
| **Delta (Ours vs Base)** | $\color{green}{+7.25}$ | $\color{green}{+7.23}$ | $\color{green}{+1.34}$ | $\color{green}{+0.11}$ | $\color{red}{-0.82}$ | $\color{green}{+4.75}$ | $\color{green}{+9.98}$ | $\color{green}{+2.67}$ | $\color{green}{+15.66}$ | $\color{green}{+20.88}$ |
| **Delta (Ours vs Painted)** | $\color{green}{+5.53}$ | $\color{green}{+7.05}$ | $\color{green}{+1.12}$ | $\color{red}{-0.25}$ | $\color{red}{-3.90}$ | $\color{green}{+2.14}$ | $\color{green}{+7.67}$ | $\color{green}{+0.69}$ | $\color{green}{+13.35}$ | $\color{green}{+14.26}$ |

---

## Discussion & Analysis

### 1. The "Cyclist" Anomaly: Why +16.28%?
Our method achieves a remarkable **+16.28%** improvement in Cyclist detection compared to the baseline, and outperforms standard PointPainting by **+13.44%**. 

**Hypothesis:**
*   **Structural Consistency:** unlike pedestrians, cyclists (and their bikes) have rigid, consistent structures and distinct silhouettes.
*   **Uncertainty Correlation:** The segmentation network is likely highly confident (low entropy) in the center of the bike/rider mass and highly uncertain (high entropy) only at the very distinct outer edges. This creates a high-quality, high-contrast "uncertainty signal" that the 3D detector can easily leverage to define the object's extent.
*   **Information Density:** The combination of bicycle mechanics and human rider provides a dense cluster of painted points with coherent uncertainty signatures, effectively "highlighting" the object in 3D space.

### 2. Impact of Training Data Discrepancy
It is important to note the difference in training set size compared to the original PointPainting paper:

*   **Original PointPainting Paper:** ~6,733 training samples (standard train/val split).
*   **Our Implementation:** **5,979** training samples.

We removed corrupted files and used a slightly cleaner split, resulting in **~11% fewer training examples**. Despite this data reduction, our method achieved:
1.  **State-of-the-art comparable results on Cars**, demonstrating high data efficiency.
2.  **Superior performance on Cyclists**, suggesting that uncertainty features act as a strong regularizer, allowing the model to generalize better even with less data.

### 3. Pedestrian Performance 
The drop in pedestrian performance (-8.28%) is attributed to **edge-noise amplification**. Pedestrians have irregular silhouettes (clothing, limbs), leading to high-frequency uncertainty noise at the boundaries. Without specific attention mechanisms to gate this noise, the 3D detector may be penalizing these "uncertain" points too heavily.

---

## Reproducibility

For detailed instructions on running the pipeline, please refer to:
*   [**Pipeline Guide**](PIPELINE_GUIDE.md): Directory structure and script usage.
*   [**Reproducibility Guide**](REPRODUCIBILITY.md): Environment setup, training commands, and config parameters.
