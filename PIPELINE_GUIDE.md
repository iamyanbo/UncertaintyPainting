# Modular Pipeline & User Guide

## 1. Directory Structure
- `src/`: Core implementation modules (Segmentation, Projection, Painting).
- `scripts/`: Executable scripts to run each step of the pipeline.
- `tests/`: Unit tests and sanity checks.
- `data/`: Place your KITTI dataset here.

## 2. User Action Required: Data Setup
To run the full pipeline, you need the **KITTI Object Detection Benchmark** dataset.
1.  Go to [KITTI Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
2.  Download:
    - **left color images of object data set (12 GB)** -> Unzip to `data/training/image_2/`
    - **Velodyne point clouds (29 GB)** -> Unzip to `data/training/velodyne/`
    - **camera calibration matrices (16 MB)** -> Unzip to `data/training/calib/`
    - **training labels of object data set (5 MB)** -> Unzip to `data/training/label_2/`

**Directory Layout:**
```text
uncertainty_painting/
  data/
    training/
      image_2/
      velodyne/
      calib/
      label_2/
    testing/
      image_2/
      velodyne/
      calib/
```
*Note: The testing set does not contain labels (`label_2`).*

## 3. Pipeline Steps (How to Test)

### Step 1: 2D Uncertainty Estimation
*   **Goal:** Verify the 2D network produces Entropy maps from images.
*   **Run:** `python scripts/01_test_2d_uncertainty.py`
*   **Output:** Visualizes Original Image vs. Entropy Map.

### Step 2: LiDAR Projection
*   **Goal:** Verify 3D points project correctly onto the image.
*   **Run:** `python scripts/02_test_projection.py`
*   **Output:** Image with LiDAR points overlaid (colored by depth).

### Step 3: Feature Painting
*   **Goal:** Verify points are decorated with Uncertainty scores.
*   **Run:** `python scripts/03_test_painting.py`
*   **Output:** 3D Viewer (Open3D) showing points colored by Uncertainty.

### Step 4: Full Dataset Conversion
*   **Goal:** Process the entire dataset offline.
*   **Run:** `python scripts/04_process_dataset.py`
*   **Note:** Processing the full training set takes approximately **4 hours 43 minutes**.
