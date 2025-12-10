
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ----------------- UTILS -----------------

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    p2_line = [x for x in lines if x.startswith('P2:')][0]
    P2 = np.array(p2_line.strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
    r0_line = [x for x in lines if x.startswith('R0_rect:')][0]
    R0 = np.array(r0_line.strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
    tr_line = [x for x in lines if x.startswith('Tr_velo_to_cam:')][0]
    Tr = np.array(tr_line.strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
    return P2, R0, Tr

def project_to_image(pts_3d, P2, R0, Tr):
    """
    pts_3d: (N, 3)
    """
    N = len(pts_3d)
    ones = np.ones((N, 1))
    pts_hom = np.hstack([pts_3d, ones]) # (N, 4)
    
    Tr_4x4 = np.eye(4)
    Tr_4x4[:3, :4] = Tr
    R0_4x4 = np.eye(4)
    R0_4x4[:3, :3] = R0
    
    # LiDAR -> CamRef
    pts_cam_ref = pts_hom @ Tr_4x4.T @ R0_4x4.T # (N, 4)
    
    # CamRef -> Image
    # Check z > 0
    mask = pts_cam_ref[:, 2] > 0
    pts_cam_ref_valid = pts_cam_ref[mask]
    
    pts_img_hom = pts_cam_ref_valid @ P2.T # (M, 3)
    pts_img = pts_img_hom[:, :2] / pts_img_hom[:, 2:3]
    
    return pts_img, mask

def get_box_corners(box):
    # box: x, y, z, dx, dy, dz, heading
    x, y, z, dx, dy, dz, yaw = box
    
    # Corners relative to center (dx=l, dy=w, dz=h)
    x_corners = [dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2]
    y_corners = [dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2]
    z_corners = [dz/2, dz/2, dz/2, dz/2, -dz/2, -dz/2, -dz/2, -dz/2]
    
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    corners = R @ corners
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    
    return corners.T

def draw_3d_box(img, corners, color, thickness=2, label=None):
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for s, e in lines:
        p1 = tuple(corners[s].astype(int))
        p2 = tuple(corners[e].astype(int))
        cv2.line(img, p1, p2, color, thickness)
    if label:
        cv2.putText(img, label[:3], tuple(corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def render_bev_matplotlib(points, base_boxes, ablation_boxes, frame_id, out_name):
    # Re-using the BEV logic but ensuring consistency
    # (Simplified for script length)
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.set_facecolor('black')
    
    mask = (points[:, 0] > 0) & (points[:, 0] < 50) & (points[:, 1] > -25) & (points[:, 1] < 25)
    pts = points[mask]
    unc = pts[:, -1]
    
    ax.scatter(pts[:, 0], pts[:, 1], c=unc, s=1, cmap='turbo', vmin=0, vmax=2.0)
    
    # Draw Boxes Logic (Simplified)
    def draw_box_bev(box, color, style='-'):
        x, y, z, dx, dy, dz, ang = box # dx=l, dy=w
        corners = np.array([[dx/2, dy/2], [dx/2, -dy/2], [-dx/2, -dy/2], [-dx/2, dy/2]])
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]])
        rot_corners = corners @ R.T + np.array([x, y])
        full = np.vstack([rot_corners, rot_corners[0]])
        ax.plot(full[:, 0], full[:, 1], color=color, linestyle=style, linewidth=2)
        
    for b in base_boxes: draw_box_bev(b, 'lime')
    for b in ablation_boxes: draw_box_bev(b, 'red', '--')
        
    ax.set_xlim(0, 50); ax.set_ylim(-25, 25); ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_name, dpi=100, facecolor='black')
    plt.close()

# ----------------- MAIN -----------------

def main():
    frame_id = '006679'
    
    calib_path = f'../data/kitti/training/calib/{frame_id}.txt'
    image_path = f'../data/kitti/training/image_2/{frame_id}.png'
    pc_path = f'../data/kitti/training/velodyne_painted/{frame_id}.bin'
    
    base_pkl = '../output/kitti_models/pointpillar_painted_full/default/eval/epoch_80/val/default/result.pkl'
    abl_pkl = '../output/kitti_models/pointpillar_painted_no_uncert/default/eval/epoch_80/val/default/result.pkl'
    
    # 1. Load Data
    img_raw = cv2.imread(image_path)
    P2, R0, Tr = load_calibration(calib_path)
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 26)
    
    base_data = load_pickle(base_pkl)
    abl_data = load_pickle(abl_pkl)
    
    base_res = next(d for d in base_data if d['frame_id'] == frame_id)
    abl_res = next(d for d in abl_data if d['frame_id'] == frame_id)
    
    mask_b = base_res['score'] > 0.5
    base_boxes = base_res['boxes_lidar'][mask_b]
    base_names = base_res['name'][mask_b]
    
    mask_a = abl_res['score'] > 0.3
    abl_boxes = abl_res['boxes_lidar'][mask_a]
    abl_names = abl_res['name'][mask_a]
    
    # ----------------- 1. Normal Image -----------------
    cv2.imwrite(f'view_1_normal_{frame_id}.png', img_raw)
    
    # ----------------- 2. 2D Boxes -----------------
    img_boxes = img_raw.copy()
    
    # Project Baseline
    for i, box in enumerate(base_boxes):
        corners_3d = get_box_corners(box)
        pts_2d, _ = project_to_image(corners_3d, P2, R0, Tr) 
        # project_to_image filters z>0, but a box might be partially behind. 
        # Simplified: if we get 8 points, draw.
        if len(pts_2d) == 8:
            img_boxes = draw_3d_box(img_boxes, pts_2d, (0, 255, 0), 2, base_names[i])
            
    # Project Ablation
    for i, box in enumerate(abl_boxes):
        corners_3d = get_box_corners(box)
        pts_2d, _ = project_to_image(corners_3d, P2, R0, Tr)
        if len(pts_2d) == 8:
            img_boxes = draw_3d_box(img_boxes, pts_2d, (0, 0, 255), 2, abl_names[i])
            
    cv2.imwrite(f'view_2_boxes_{frame_id}.png', img_boxes)
    
    # ----------------- 3. Entropy Image (Projected) -----------------
    img_entropy = img_raw.copy()
    # Darken original image to make points pop
    img_entropy = (img_entropy * 0.3).astype(np.uint8)
    
    xyz = points[:, :3]
    unc = points[:, -1]
    
    pts_2d, mask = project_to_image(xyz, P2, R0, Tr)
    unc_valid = unc[mask]
    
    # Filter to image bounds
    H, W, _ = img_entropy.shape
    valid_indices = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
    
    pts_2d = pts_2d[valid_indices]
    unc_valid = unc_valid[valid_indices]
    
    # Colormap
    # Normalize uncert 0-2 (entropy)
    norm_unc = np.clip(unc_valid / 2.0, 0, 1) * 255
    norm_unc = norm_unc.astype(np.uint8)
    
    # Apply colormap
    # cv2.applyColorMap expects (N, 1) image-like
    # We'll just manually color: Red is high entropy.
    # actually let's use matplotlib colormap 'turbo' mapping
    cmap = plt.get_cmap('turbo')
    colors = cmap(unc_valid / 2.0)[:, :3] * 255 # RGB
    colors = colors[:, ::-1] # BGR for opencv
    
    for pt, color in zip(pts_2d, colors):
        cv2.circle(img_entropy, tuple(pt.astype(int)), 1, color.tolist(), -1)
        
    cv2.imwrite(f'view_3_entropy_{frame_id}.png', img_entropy)
    
    # ----------------- 4. 3D BEV -----------------
    render_bev_matplotlib(points, base_boxes, abl_boxes, frame_id, f'view_4_bev_{frame_id}.png')
    
    # ----------------- Combine -----------------
    # Grid 2x2
    # Load all
    v1 = cv2.imread(f'view_1_normal_{frame_id}.png')
    v2 = cv2.imread(f'view_2_boxes_{frame_id}.png')
    v3 = cv2.imread(f'view_3_entropy_{frame_id}.png')
    v4 = cv2.imread(f'view_4_bev_{frame_id}.png')
    
    # Resize to match v1 size
    target_h, target_w = v1.shape[:2]
    
    def resize_crop(img, h, w):
        # simple resize
        return cv2.resize(img, (w, h))

    v2 = resize_crop(v2, target_h, target_w)
    v3 = resize_crop(v3, target_h, target_w)
    v4 = resize_crop(v4, target_h, target_w) # BEV square -> rectangular distorts.
    # Better to fit BEV into square on right, but for 2x2 grid, let's keep aspect ratio reasonable
    # Actually, BEV is square. Camera is wide.
    # Let's resize BEV to have same Height, preserve aspect?
    # User just wants 4 images.
    # Let's stack them:
    # Top: Normal | 2D Boxes
    # Bot: Entropy | 3D BEV
    
    top = np.hstack([v1, v2])
    bot = np.hstack([v3, v4])
    final = np.vstack([top, bot])
    
    cv2.imwrite(f'failure_case_quad_{frame_id}.png', final)
    print("Saved failure_case_quad.")

if __name__ == '__main__':
    main()
