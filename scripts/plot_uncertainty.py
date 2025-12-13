
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import os

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
    return uncertainty

def analyze_and_plot():
    print("Loading data...")
    with open(RESULT_PATH, 'rb') as f:
        detections = pickle.load(f)
    with open(GT_INFO_PATH, 'rb') as f:
        gt_infos = pickle.load(f)

    # Settings
    IOU_THRESH = {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
    CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    
    tp_uncertainties = {cls: [] for cls in CLASSES}
    fp_uncertainties = {cls: [] for cls in CLASSES}

    print("Computing stats...")
    for i in range(len(detections)):
        det = detections[i]
        gt = gt_infos[i]
        
        pred_boxes = det['boxes_lidar']
        pred_scores = det['score']
        pred_names = det['name']
        
        gt_annos = gt['annos']
        gt_boxes = gt_annos['gt_boxes_lidar']
        
        if len(pred_boxes) == 0:
            continue
            
        pred_boxes_t = torch.tensor(pred_boxes, dtype=torch.float32).cuda()
        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32).cuda() if len(gt_boxes) > 0 else None
        
        if gt_boxes_t is not None and gt_boxes_t.shape[0] > 0:
            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes_t, gt_boxes_t)
            max_iou_per_pred, _ = torch.max(iou_matrix, dim=1)
            max_iou_per_pred = max_iou_per_pred.cpu().numpy()
        else:
            max_iou_per_pred = np.zeros(len(pred_boxes))
            
        for j in range(len(pred_boxes)):
            cls_name = pred_names[j]
            if cls_name not in CLASSES: continue
            
            score = pred_scores[j]
            unc = get_edl_uncertainty(score)
            
            if max_iou_per_pred[j] >= IOU_THRESH[cls_name]:
                tp_uncertainties[cls_name].append(unc)
            else:
                fp_uncertainties[cls_name].append(unc)

    # Plotting
    print("Generating plot...")
    bins = np.linspace(0.70, 1.00, 20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.3)
    
    for ax, cls in zip(axes, CLASSES):
        tps = tp_uncertainties[cls]
        fps = fp_uncertainties[cls]
        
        tp_hist, _ = np.histogram(tps, bins=bins)
        fp_hist, _ = np.histogram(fps, bins=bins)
        
        # Calculate Accuracy per bin
        total = tp_hist + fp_hist
        accuracy = np.zeros_like(total, dtype=float)
        valid_mask = total > 0
        accuracy[valid_mask] = (tp_hist[valid_mask] / total[valid_mask]) * 100
        
        # Plot Counts (Bar Chart) - Log Scale due to huge FP spike
        ax2 = ax.twinx()  # Secondary axis for Accuracy
        
        width = (bins[1] - bins[0])
        
        # We plot Counts on Logs scale
        # Handle 0s for log scale
        tp_hist_plot = np.where(tp_hist > 0, tp_hist, 0.1)
        fp_hist_plot = np.where(fp_hist > 0, fp_hist, 0.1)
        
        # Stacked? Or Side by side? Side by side might be clearer with log.
        # Let's just plot lines/steps for counts
        # ax.step(bin_centers, tp_hist, where='mid', label='True Positives', color='green', linewidth=2)
        # ax.step(bin_centers, fp_hist, where='mid', label='False Positives', color='red', linewidth=2)
        
        # Bars
        ax.bar(bin_centers, tp_hist, width=width*0.4, color='green', align='center', label='True Positives', alpha=0.7)
        ax.bar(bin_centers, fp_hist, width=width*0.4, color='red', align='edge', label='False Positives', alpha=0.5)
        
        ax.set_yscale('log')
        ax.set_ylim(bottom=1) # Don't show < 1
        ax.set_xlabel('Uncertainty (EDL Vacuum)')
        ax.set_ylabel('Count (Log Scale)')
        ax.set_title(f'Class: {cls} (IoU > {IOU_THRESH[cls]})')
        ax.legend(loc='upper left')
        
        # Plot Accuracy (Line)
        ax2.plot(bin_centers[valid_mask], accuracy[valid_mask], color='blue', marker='o', linewidth=2, label='Accuracy (%)')
        ax2.set_ylabel('Accuracy %', color='blue')
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.legend(loc='upper right')
        
        # Mark the "Cutoff" visually
        ax.axvline(x=0.86, color='black', linestyle='--', label='Suggested Cutoff (0.86)')

    plt.suptitle('Uncertainty Distribution Analysis: Correctness vs Uncertainty', fontsize=16)
    save_path = 'uncertainty_analysis_plot.png'
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    analyze_and_plot()
