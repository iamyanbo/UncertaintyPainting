import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.segmentation import UncertaintyPainter2D
from src.projection import Calibration

def compute_box_3d(obj, P2):
    # obj: list of floats [h, w, l, x, y, z, ry] (from label file)
    # KITTI Label: type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
    # We parse manually
    
    h, w, l = obj[0], obj[1], obj[2]
    x, y, z = obj[3], obj[4], obj[5]
    ry = obj[6]
    
    # Compute rotation matrix around Y axis
    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    
    # 3D bounding box corners
    # KITTI coords: x right, y down, z forward
    # l implies x size? No, l is length (along heading).
    # w is width. h is height.
    # Usually in KITTI devkit:
    # x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    # y_corners = [0,0,0,0, -h, -h, -h, -h]
    # z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    
    # Translate
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    # Project to image
    corners_3d_hom = np.vstack((corners_3d, np.ones((1, 8))))
    corners_2d = np.dot(P2, corners_3d_hom)
    corners_2d[0, :] /= corners_2d[2, :]
    corners_2d[1, :] /= corners_2d[2, :]
    
    return corners_2d[:2, :].T

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/example_kitti.html
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i,0], qs[i,1]), (qs[j,0], qs[j,1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i,0], qs[i,1]), (qs[j,0], qs[j,1]), color, thickness)
        i, j = k, k + 4
        cv2.line(image, (qs[i,0], qs[i,1]), (qs[j,0], qs[j,1]), color, thickness)
    return image

def main():
    with open('failure_case.txt', 'r') as f:
        idx = f.read().strip()
        
    print(f"Visualizing failure case: {idx}")
    
    base_dir = r'OpenPCDet/data/kitti/training'
    image_path = os.path.join(base_dir, 'image_2', f'{idx}.png')
    calib_path = os.path.join(base_dir, 'calib', f'{idx}.txt')
    label_path = os.path.join(base_dir, 'label_2', f'{idx}.txt')
    
    output_dir = 'vis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Load Image and Run 2D Segmentation
    print("Running 2D Segmentation...")
    painter_2d = UncertaintyPainter2D(model_name='deeplabv3_resnet50')
    entropy_map, class_probs, original_img = painter_2d.predict(image_path)
    
    img_np = np.array(original_img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 2. Load GT and Draw (Missed Pedestrians)
    calib = Calibration(calib_path)
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split(' ')
        cls_name = parts[0]
        if cls_name == 'DontCare': continue
        
        # Parse dimensions and location
        # h, w, l, x, y, z, ry
        # parts[8], [9], [10], [11], [12], [13], [14]
        h = float(parts[8])
        w = float(parts[9])
        l = float(parts[10])
        x = float(parts[11])
        y = float(parts[12])
        z = float(parts[13])
        ry = float(parts[14])
        
        # Color: Green for Cyclist, Red for Pedestrian (Failed Case)
        color = (0, 255, 0)
        if cls_name == 'Pedestrian':
            color = (0, 0, 255) # Red for missed pedestrian
        elif cls_name == 'Cyclist':
            color = (255, 0, 0) # Blue
            
        corners_2d = compute_box_3d([h, w, l, x, y, z, ry], calib.P2)
        img_cv2 = draw_projected_box3d(img_cv2, corners_2d, color=color, thickness=2)
        
        # Add label
        cv2.putText(img_cv2, f"GT_{cls_name}", (int(corners_2d[0][0]), int(corners_2d[0][1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 3. Save
    out_path = os.path.join(output_dir, f'failure_case_{idx}.png')
    cv2.imwrite(out_path, img_cv2)
    print(f"Saved visualization to {out_path}")
    
    # Also save entropy side-by-side
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Image with GT (Red=Missed Pedestrian)")
    plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Classification Mask
    pred_mask = np.argmax(class_probs, axis=0)
    
    plt.subplot(1, 3, 2)
    plt.title("2D Classification (DeepLabV3+)")
    plt.imshow(pred_mask, cmap='tab20')
    plt.colorbar(label='Class ID')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Entropy / Uncertainty")
    plt.imshow(entropy_map, cmap='inferno')
    plt.axis('off')
    
    combined_path = os.path.join(output_dir, f'failure_analysis_{idx}.png')
    plt.savefig(combined_path)
    print(f"Saved combined analysis to {combined_path}")

if __name__ == "__main__":
    main()
