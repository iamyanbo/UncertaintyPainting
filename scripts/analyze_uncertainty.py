
import pickle
import numpy as np
import torch
import math
from pcdet.ops.iou3d_nms import iou3d_nms_utils

# Paths
RESULT_PATH = r"OpenPCDet/output/kitti_models/pointpillar_painted/edl_overnight/eval/eval_with_train/epoch_80/val/result.pkl"
GT_INFO_PATH = r"OpenPCDet/data/kitti/kitti_infos_val.pkl"

def sigmoid_inverse(x):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))

def get_edl_uncertainty(score):
    # derived from binary EDL:
    # score (p) = alpha / S = (relu(logit)+1) / (relu(logit)+1 + relu(-logit)+1)
    # This relationship is monotonic. 
    # We recover logit from sigmoid score.
    logit = sigmoid_inverse(score)
    
    # Calculate Evidence
    alpha = np.maximum(logit, 0) + 1
    beta = np.maximum(-logit, 0) + 1
    S = alpha + beta
    
    uncertainty = 2.0 / S
    return uncertainty

def analyze():
    print("Loading results...")
    with open(RESULT_PATH, 'rb') as f:
        detections = pickle.load(f)
        
    print("Loading GT infos...")
    with open(GT_INFO_PATH, 'rb') as f:
        gt_infos = pickle.load(f)

    print(f"Loaded {len(detections)} detections and {len(gt_infos)} GT frames.")
    
    # Check alignment
    assert len(detections) == len(gt_infos), "Mismatch in dataset length!"

    # Settings
    IOU_THRESH = {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
    
    # Bins: 0.70 to 1.00 with 0.02 step (approx 15 bins)
    # Map u to bin index: (u - 0.70) / 0.02
    NUM_BINS = 20
    MIN_U = 0.70
    STEP = (1.0 - MIN_U) / NUM_BINS
    
    bin_correct = {cls: np.zeros(NUM_BINS) for cls in IOU_THRESH}
    bin_total = {cls: np.zeros(NUM_BINS) for cls in IOU_THRESH}
    
    print("Processing frames (this may take a moment)...")
    for i in range(len(detections)):
        det = detections[i]
        gt = gt_infos[i]
        
        # Verify frame ID match (heuristic)
        # det['frame_id'] vs gt['image']['image_idx']
        
        # Get Predictions
        pred_boxes = det['boxes_lidar'] # (N, 7)
        pred_scores = det['score']      # (N,)
        pred_names = det['name'] # (N,) e.g. ['Car', 'Car', ...]
        # However, check 'name' key in det? usually pred_labels is int. 
        # Map: 1:Car, 2:Pedestrian, 3:Cyclist
        class_map = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
        
        # Get GT
        gt_annos = gt['annos']
        gt_boxes = gt_annos['gt_boxes_lidar'] # (M, 7)
        gt_names = gt_annos['name']            # (M,)
        
        if len(pred_boxes) == 0:
            continue
            
        # Convert to torch for IoU
        pred_boxes_t = torch.tensor(pred_boxes, dtype=torch.float32).cuda()
        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32).cuda() if len(gt_boxes) > 0 else None
        
        # Determine correctness for each prediction
        # We match each Pred to the best GT.
        if gt_boxes_t is not None and gt_boxes_t.shape[0] > 0:
            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes_t, gt_boxes_t) # (N, M)
            max_ious, _ = torch.max(iou_matrix, dim=1)
            max_ious = max_ious.cpu().numpy()
        else:
            max_ious = np.zeros(len(pred_boxes))
            
        # Analysis Loop
        for j in range(len(pred_boxes)):
            cls_name = pred_names[j]
            if cls_name not in IOU_THRESH: continue
            
            score = pred_scores[j]
            uncertainty = get_edl_uncertainty(score)
            
            is_correct = max_ious[j] >= IOU_THRESH[cls_name]
            
            
            # Binning
            if uncertainty < MIN_U: bin_idx = 0
            else:
                bin_idx = int((uncertainty - MIN_U) / STEP)
            
            if bin_idx >= NUM_BINS: bin_idx = NUM_BINS - 1
            
            bin_total[cls_name][bin_idx] += 1
            if is_correct:
                bin_correct[cls_name][bin_idx] += 1

    # Print Report
    print("\n=== Uncertainty vs Correctness Analysis ===")
    print("Metric: Percentage of Predictions that are True Positives (Correct) per Uncertainty Level")
    print("Uncertainty calculated via Binary EDL transformation of Sigmoid Scores.\n")
    
    for cls in ['Car', 'Pedestrian', 'Cyclist']:
        print(f"--- Class: {cls} (IoU > {IOU_THRESH[cls]}) ---")
        print(f"{'Uncertainty':<15} | {'Accuracy (%)':<15} | {'Count':<10}")
        print("-" * 45)
        for b in range(NUM_BINS):
            u_min = MIN_U + b * STEP
            u_max = MIN_U + (b + 1) * STEP
            total = bin_total[cls][b]
            correct = bin_correct[cls][b]
            
            if total > 0:
                acc = (correct / total) * 100
                print(f"{u_min:.3f} - {u_max:.3f}   | {acc:5.1f}           | {int(total)}")
        print("")

if __name__ == "__main__":
    analyze()
