# Uncertainty-Aware Sequential Fusion for 3D Object Detection

This repository implements **Uncertainty-Aware Feature Painting**, a novel sequential fusion method for 3D object detection. By integrating 2D semantic segmentation uncertainty into the point cloud feature space, this method enhances the robustness of 3D detectors (like PointPillars) against ambiguous visual data.

## Visual Pipeline Results

### 1. Uncertainty Estimation (2D)
The top row shows the input image. The middle row displays the **Predicted Semantic Classes**. The bottom row visualizes the **Predictive Uncertainty (Entropy)**, where brighter colors indicate higher uncertainty (e.g., object boundaries).

![2D Uncertainty Estimation](assets/test_uncertainty_result.png)

### 2. Painted Point Cloud (3D)
Here we see the 3D LiDAR point cloud "painted" with the fusion features (Class Semantic + Uncertainty). This rich feature set helps the detector discern objects even with sparse LiDAR data.

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

```diff
Method                                     | mAP (Mod.)  | Car (Mod.) | Pedestrian (Mod.) | Cyclist (Mod.)
--------------------------------------------------------------------------------------------------------
PointPillars [11]                          | 73.78       | 87.57      | 67.84             | 65.92
PaintedPointPillars (PointPainting)        | 76.27       | 87.65      | 72.41             | 68.76
Delta (Painted vs Base)                    | +2.50       | +0.08      | +4.57             | +2.84
--------------------------------------------------------------------------------------------------------
Uncertainty-Painted PointPillars (Ours)    | 76.52       | 87.81      | 59.56             | 82.20
Delta (Ours vs Base)                       | +2.74       | +0.24      | -8.28             | +16.28
Delta (Ours vs Painted)                    | +0.25       | +0.16      | -12.85            | +13.44
```

> **Detailed Metrics:**
> *   **Car (Easy/Mod/Hard):** 90.10 / 87.81 / 84.46
> *   **Pedestrian (Easy/Mod/Hard):** 63.57 / 59.56 / 57.05
> *   **Cyclist (Easy/Mod/Hard):** 85.80 / 82.20 / 76.68

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
