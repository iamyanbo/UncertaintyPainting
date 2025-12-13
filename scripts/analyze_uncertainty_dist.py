
import pickle
import numpy as np
import torch
from pcdet.ops.iou3d_nms import iou3d_nms_utils

# Paths
RESULT_PATH = r"OpenPCDet/output/kitti_models/pointpillar_painted/edl_overnight/eval/eval_with_train/epoch_80/val/result.pkl"
GT_INFO_PATH = r"OpenPCDet/data/kitti/kitti_infos_val.pkl"

def sigmoid_inverse(x):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))

def get_edl_uncertainty(score):
    logit = sigmoid_inverse(score)
    alpha = np.maximum(logit, 0) + 1
    beta = np.maximum(-logit, 0) + 1
    S = alpha + beta
    uncertainty = 2.0 / S
    # Also return alpha/beta for debugging if needed, but just U for now
    return uncertainty

def analyze_recall():
    print("Loading datasets...")
    with open(RESULT_PATH, 'rb') as f:
        detections = pickle.load(f)
    with open(GT_INFO_PATH, 'rb') as f:
        gt_infos = pickle.load(f)

    # Settings
    IOU_THRESH = {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
    
    # Store uncertainties for two populations
    # 1. TP_Uncertainties: The uncertainty of predictions that successfully found a GT object.
    # 2. FP_Uncertainties: The uncertainty of predictions that found nothing (Background).
    
    tp_uncertainties = {cls: [] for cls in IOU_THRESH}
    fp_uncertainties = {cls: [] for cls in IOU_THRESH}
    
    total_gt_count = {cls: 0 for cls in IOU_THRESH}
    detected_gt_count = {cls: 0 for cls in IOU_THRESH}

    print("Analyzing frames...")
    for i in range(len(detections)):
        det = detections[i]
        gt = gt_infos[i]
        
        # Predictions
        pred_boxes = det['boxes_lidar']
        pred_scores = det['score']
        pred_names = det['name']
        
        # Ground Truth
        gt_annos = gt['annos']
        gt_boxes = gt_annos['gt_boxes_lidar']
        gt_names = gt_annos['name']
        
        # Count total GT
        for name in gt_names:
            if name in IOU_THRESH:
                total_gt_count[name] += 1
        
        if len(pred_boxes) == 0:
            continue
            
        # Tensors
        pred_boxes_t = torch.tensor(pred_boxes, dtype=torch.float32).cuda()
        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32).cuda() if len(gt_boxes) > 0 else None
        
        # 1. Compute IoU
        if gt_boxes_t is not None and gt_boxes_t.shape[0] > 0:
            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes_t, gt_boxes_t) # (NumPreds, NumGT)
            # Find best GT for each Pred (for FP classification)
            max_iou_per_pred, _ = torch.max(iou_matrix, dim=1)
            max_iou_per_pred = max_iou_per_pred.cpu().numpy()
            
            # Find best Pred for each GT (for TP classification / Recall)
            # A GT is "Detected" if ANY pred overlaps it > threshold
            # But here we want the uncertainty of that matching pred.
            # If multiple preds match one GT, we usually take the highest score one (standard eval).
            # But for simplicity, let's collect ALL Valid True Positives.
            pass
        else:
            max_iou_per_pred = np.zeros(len(pred_boxes))
            
        # Classify Predictions into TP / FP populations
        for j in range(len(pred_boxes)):
            cls_name = pred_names[j]
            if cls_name not in IOU_THRESH: continue
            
            score = pred_scores[j]
            unc = get_edl_uncertainty(score)
            
            if max_iou_per_pred[j] >= IOU_THRESH[cls_name]:
                # Matches a GT -> It is a True Positive (valid detection)
                tp_uncertainties[cls_name].append(unc)
            else:
                # Matches nothing -> It is a False Positive (background/noise)
                fp_uncertainties[cls_name].append(unc)

    # Report Distributions
    print("\n=== Uncertainty Distribution Analysis ===")
    print("Separating True Positives (Real Objects) vs False Positives (Background).\n")
    
    # Bins for histogram
    bins = np.linspace(0.70, 1.00, 16) # 0.70, 0.72, ... 1.00
    
    for cls in ['Car', 'Pedestrian', 'Cyclist']:
        print(f"--- Class: {cls} ---")
        tps = np.array(tp_uncertainties[cls])
        fps = np.array(fp_uncertainties[cls])
        
        if len(tps) == 0:
            print("No TPs found.")
            continue
            
        print(f"Total True Positives (Hits): {len(tps)}")
        print(f"Total False Positives (Noise): {len(fps)}")
        
        # Histogram
        tp_hist, _ = np.histogram(tps, bins=bins)
        fp_hist, _ = np.histogram(fps, bins=bins)
        
        print(f"{'Uncertainty':<15} | {'TP Count (Real)':<15} | {'FP Count (Noise)':<15} | {'TP Ratio (%)':<10}")
        print("-" * 65)
        for b in range(len(bins)-1):
            low = bins[b]
            high = bins[b+1]
            tp_c = tp_hist[b]
            fp_c = fp_hist[b]
            total = tp_c + fp_c
            ratio = (tp_c / total * 100) if total > 0 else 0.0
            
            # Simple bar for visual
            # bar = '#' * int(ratio / 10)
            print(f"{low:.3f} - {high:.3f}   | {tp_c:<15} | {fp_c:<15} | {ratio:5.1f}%")
        print("\n")

if __name__ == "__main__":
    analyze_recall()
