import pickle
import numpy as np
import os
import sys

def iou_bev(box_a, box_b):
    # Simplified BEV IoU (Axis Aligned) for quick check
    xa_min = box_a[0] - box_a[3]/2
    xa_max = box_a[0] + box_a[3]/2
    ya_min = box_a[1] - box_a[4]/2
    ya_max = box_a[1] + box_a[4]/2
    
    xb_min = box_b[0] - box_b[3]/2
    xb_max = box_b[0] + box_b[3]/2
    yb_min = box_b[1] - box_b[4]/2
    yb_max = box_b[1] + box_b[4]/2
    
    inter_x_min = max(xa_min, xb_min)
    inter_x_max = min(xa_max, xb_max)
    inter_y_min = max(ya_min, yb_min)
    inter_y_max = min(ya_max, yb_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    area_a = box_a[3] * box_a[4]
    area_b = box_b[3] * box_b[4]
    
    return inter_area / (area_a + area_b - inter_area + 1e-6)

def load_gt_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    gt_boxes = []
    gt_names = []
    gt_occluded = []
    gt_truncated = []
    
    for line in lines:
        parts = line.strip().split(' ')
        cls_name = parts[0]
        if cls_name == 'DontCare':
            continue
            
        bbox_2d = [float(x) for x in parts[4:8]]
        gt_boxes.append(bbox_2d)
        gt_names.append(cls_name)
        gt_truncated.append(float(parts[1]))
        gt_occluded.append(int(float(parts[2])))
        
    return gt_boxes, gt_names, gt_occluded, gt_truncated

def main():
    pkl_path = r'c:/Users/yanbo/Downloads/Papers/uncertainty_painting/OpenPCDet/output/kitti_models/pointpillar_painted_full/default/eval/epoch_80/val/default/result.pkl'
    label_dir = r'OpenPCDet/data/kitti/training/label_2'
    
    print(f"Loading results from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        predictions = pickle.load(f)
        
    target_class = 'Pedestrian'
    print(f"Scanning samples for Missed (Fully Visible) {target_class}...")
    
    total_gt_checked = 0
    
    for i, pred in enumerate(predictions):
        frame_id = str(pred['frame_id'])
        
        label_file = os.path.join(label_dir, f"{frame_id}.txt")
        if not os.path.exists(label_file):
            continue
            
        gt_boxes, gt_names, gt_occ, gt_trunc = load_gt_labels(label_file)
        
        # Only check fully visible (0)
        gt_indices = []
        for idx, name in enumerate(gt_names):
            if name == target_class and gt_occ[idx] == 0:
                gt_indices.append(idx)
                
        total_gt_checked += len(gt_indices)
        
        # Check False Negatives
        pred_boxes_2d = pred['bbox']
        pred_scores = pred['score']
        pred_names = pred['name']
        
        for gt_idx in gt_indices:
            gt_box = gt_boxes[gt_idx]
            is_detected = False
            for p_idx, p_box in enumerate(pred_boxes_2d):
                if pred_names[p_idx] != target_class or pred_scores[p_idx] < 0.3:
                    continue
                
                # Check 2D IoU
                ixmin = max(gt_box[0], p_box[0])
                iymin = max(gt_box[1], p_box[1])
                ixmax = min(gt_box[2], p_box[2])
                iymax = min(gt_box[3], p_box[3])
                
                iw = max(ixmax - ixmin, 0)
                ih = max(iymax - iymin, 0)
                inter = iw * ih
                
                area_gt = (gt_box[2]-gt_box[0]) * (gt_box[3]-gt_box[1])
                area_p = (p_box[2]-p_box[0]) * (p_box[3]-p_box[1])
                
                iou = inter / (area_gt + area_p - inter + 1e-6)
                
                if iou > 0.3:
                    is_detected = True
                    break
            
            if not is_detected:
                print(f"Found MISSING {target_class} in Frame {frame_id} (Occlusion=0)")
                print(f"GT Box: {gt_box}")
                with open('failure_case.txt', 'w') as f:
                    f.write(frame_id)
                return

    print(f"Checked {total_gt_checked} GT {target_class} objects. None missed.")

if __name__ == "__main__":
    main()
