"""
Check for NaNs or Infs in painted point cloud 002042.
"""
import numpy as np
import sys

def check_nan_inf(idx='001273'):
    file_path = f'data/training/velodyne_painted/{idx}.bin'
    print(f"Checking {file_path}...")
    
    try:
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 26)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if np.isnan(points).any():
        print("Found NaNs in points!")
        print(f"NaN count: {np.isnan(points).sum()}")
    else:
        print("No NaNs found.")
        
    if np.isinf(points).any():
        print("Found Infs in points!")
        print(f"Inf count: {np.isinf(points).sum()}")
    else:
        print("No Infs found.")
        
    print(f"Min: {points.min()}")
    print(f"Max: {points.max()}")

if __name__ == '__main__':
    check_nan_inf()
