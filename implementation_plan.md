# Uncertainty-Aware Feature Painting: Implementation Plan

## 1. Project Overview
**Goal:** Enhance 3D object detection by fusing 2D camera features with 3D LiDAR points, weighted by pixel-level uncertainty.
**Hypothesis:** Explicitly modeling uncertainty allows the 3D network to trust camera information only when it is reliable, improving robustness in challenging conditions (glare, darkness, occlusion).

## 2. Implementation Pipeline

### Phase 1: Data Preparation & Environment
- **Dataset:** KITTI Object Detection Benchmark (easier to start) or NuScenes (more diverse conditions).
- **Goal:** Set up dataloaders that provide synchronized Image and LiDAR frames.

### Phase 2: 2D Uncertainty Estimation
- **Model:** Use a standard semantic segmentation network (e.g., DeepLabV3+, UNet, or a pre-trained detector).
- **Uncertainty Method:**
    - **Entropy:** Calculate Shannon entropy of the softmax class probabilities per pixel. High entropy = high uncertainty.
    - **MC Dropout:** Run the network multiple times with dropout enabled during inference and calculate the variance of the predictions. (More accurate but slower).
- **Output:** An "Uncertainty Map" (H x W) aligned with the RGB image.

### Phase 3: Projection & Painting
- **Projection:** Use calibration matrices (Camera-to-LiDAR) to project 3D LiDAR points onto the 2D image plane.
- **Painting:** Append the sampled 2D features to the 3D points.
    - *Standard PointPainting:* Appends Class Scores (C dimensions).
    - *Uncertainty Painting:* Appends Class Scores + Uncertainty Score (C + 1 dimensions).
- **Result:** Decorated Point Cloud (N x (3 + C + 1)).

### Phase 4: 3D Detection Network
- **Model:** Use a LiDAR-based detector compatible with extra features (e.g., PointPillars, PointRCNN, or VoxelNet).
- **Modification:** Adjust the input layer to accept the extra feature dimensions.
- **Training:** Train the 3D detector on the "Uncertainty Painted" point clouds.

### Phase 5: Evaluation & Robustness Testing
- **Baseline:** Compare against (1) LiDAR-only, (2) Standard PointPainting.
- **Robustness:** Test on "corrupted" validation sets (e.g., add Gaussian noise to images, simulate rain/fog, or darken images) to see if the Uncertainty-Aware model degrades less than the Standard PointPainting model.

## 3. Technology Stack (What to Use)

- **Language:** Python 3.8+
- **Deep Learning Framework:** PyTorch
- **3D Detection Library:** **OpenPCDet** (Highly recommended for KITTI/NuScenes) or **MMDetection3D**.
- **2D Segmentation:** `torchvision.models.segmentation` (e.g., DeepLabV3) or `segmentation_models_pytorch`.
- **Data Processing:** `numpy`, `open3d` (for visualization), `kitti-object-eval-python`.

## 4. Next Steps

1.  **Environment Setup:** Install PyTorch and OpenPCDet.
2.  **Data Acquisition:** Download the KITTI mini-dataset (or full if space allows).
3.  **2D Prototype:** Write a script to load an image, run a pre-trained segmentation model, and visualize the Entropy map.
4.  **Projection Test:** Write a script to project LiDAR points to the image and color them by their uncertainty value to verify alignment.
