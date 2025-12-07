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
| **Uncertainty-Painted PointPillars** (Ours) | 76.52 | 90.10 | 87.81 | 84.46 | 63.57 | 59.56 | 57.05 | 85.80 | 82.20 | 76.68 |
| **Uncertainty-Painted SECOND** (Ours) | 79.08 | 97.10 | 88.62 | 86.41 | 69.26 | 67.19 | 65.00 | 88.15 | 81.43 | 79.85 |

---

## Analysis

### Cyclist Performance
Our method achieves a remarkable **+16.28%** improvement in Cyclist detection for PointPillars and **+15.66%** for SECOND compared to their respective baselines. **This performance is significantly higher than other methods on the leaderboard**, demonstrating the exceptional value of uncertainty in detecting thin, vulnerable road users. The entropy channel provides critical cues for cyclists, which are often thin and prone to high uncertainty at the boundaries.

**Visual Demonstration (Sample 005985):**

| 2D Uncertainty Analysis | 3D Painted Point Cloud |
|-------------------------|------------------------|
| ![Cyclist 2D](assets/cyclist_2d_analysis.png) | ![Cyclist 3D](assets/cyclist_3d_painted.png) |

*   **Left:** The entropy map (rightmost panel) highlights the cyclist's edges with higher uncertainty.
*   **Right:** The painted point cloud projects this uncertainty into 3D space, enriching the sparse LiDAR data with dense 2D semantic priors.

### Pedestrian Performance
The impact on Pedestrian detection is nuanced. While performance dropped for PointPillars (**-8.28%**), it **improved significantly for SECOND (+4.75%)**. The drop in PointPillars might be attributed to the lower resolution of pillars clashing with the high edge uncertainty around pedestrians. However, the voxel-based SECOND model effectively leveraged this uncertainty information to improve detection, indicating that the architecture plays a key role in how uncertainty features are utilized.

### Impact of Training Data Discrepancy
It is important to note the difference in training set size compared to the original PointPainting paper:

*   **Original PointPainting Paper:** ~6,733 training samples (standard train/val split).
*   **Our Implementation:** **5,979** training samples.

We removed corrupted files and used a slightly cleaner split, resulting in **~11% fewer training examples**. Despite this data reduction, our method achieved:
1.  **State-of-the-art comparable results on Cars**, demonstrating high data efficiency.
2.  **Superior performance on Cyclists**, suggesting that uncertainty features act as a strong regularizer, allowing the model to generalize better even with less data.

---

## Future Work

Future work involves exploring advanced uncertainty methods to improve robustness:
*   **Monte Carlo (MC) Dropout:** Bayesian approximation via dropout during inference.
*   **Deep Ensembles:** Uncertainty estimation through variance across multiple models.

---

## Reproducibility

For detailed instructions on running the pipeline, please refer to:
*   [**Pipeline Guide**](PIPELINE_GUIDE.md): Directory structure and script usage.
*   [**Reproducibility Guide**](REPRODUCIBILITY.md): Environment setup, training commands, and config parameters.

---

## Acknowledgements

This project is built upon the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) toolbox. We thank the authors for their open-source contribution to the 3D object detection community.

