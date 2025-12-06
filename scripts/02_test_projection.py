import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.projection import Calibration, filter_points_in_image

def create_dummy_calib(path):
    print("Creating dummy calibration file...")
    content = """
P0: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
P1: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_to_velo: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
    """
    with open(path, 'w') as f:
        f.write(content.strip())

def create_dummy_lidar(path):
    print("Creating dummy LiDAR points...")
    # Generate points in a grid in front of the car
    # x: forward, y: left, z: up
    points = []
    for x in np.linspace(5, 50, 50):
        for y in np.linspace(-10, 10, 20):
            for z in np.linspace(-2, 2, 5):
                points.append([x, y, z])
    
    points = np.array(points, dtype=np.float32)
    # Save as binary (KITTI format)
    # KITTI points are (x, y, z, r)
    points_with_intensity = np.hstack((points, np.zeros((points.shape[0], 1))))
    points_with_intensity.astype(np.float32).tofile(path)
    return points

def main():
    # 1. Setup Paths - Try real data first
    calib_path = "data/test_calib.txt"
    lidar_path = "data/test_lidar.bin"
    
    # Fallback to dummy data if real data doesn't exist
    if not os.path.exists(calib_path):
        calib_path = "data/dummy_calib.txt"
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(calib_path):
            create_dummy_calib(calib_path)
            
    if not os.path.exists(lidar_path):
        lidar_path = "data/dummy_lidar.bin"
        if not os.path.exists(lidar_path):
            create_dummy_lidar(lidar_path)
    
    print(f"Using calibration: {calib_path}")
    print(f"Using LiDAR: {lidar_path}")

    # 2. Load Calibration
    calib = Calibration(calib_path)
    
    # 3. Load LiDAR
    # KITTI .bin files are float32 Nx4
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    points_3d = points[:, :3]
    
    # 4. Project
    print(f"Projecting {points_3d.shape[0]} points...")
    pts_2d, depth = calib.project_velo_to_image(points_3d)
    
    # 5. Filter
    image_shape = (375, 1242) # Typical KITTI size
    mask = filter_points_in_image(pts_2d, image_shape)
    mask &= (depth > 0) # Only points in front
    
    pts_2d_valid = pts_2d[mask]
    depth_valid = depth[mask]
    
    print(f"Points in image FOV: {pts_2d_valid.shape[0]}")

    # 6. Visualize
    plt.figure(figsize=(12, 4))
    plt.title("LiDAR Projection Test (Dummy Data)")
    plt.xlim(0, image_shape[1])
    plt.ylim(image_shape[0], 0) # Invert Y for image coords
    
    plt.scatter(pts_2d_valid[:, 0], pts_2d_valid[:, 1], c=depth_valid, cmap='viridis', s=2)
    plt.colorbar(label='Depth (m)')
    
    output_path = "data/test_projection_result.png"
    plt.savefig(output_path)
    print(f"Projection visualization saved to {output_path}")

if __name__ == "__main__":
    main()
