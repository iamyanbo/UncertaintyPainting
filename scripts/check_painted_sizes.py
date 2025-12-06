"""
Check if painted point cloud files have valid sizes.
Each point has 26 features (float32), so 104 bytes per point.
File size must be divisible by 104.
"""
import os
import glob

def check_file_sizes(data_root='data/training/velodyne_painted'):
    files = glob.glob(os.path.join(data_root, '*.bin'))
    print(f"Checking {len(files)} files in {data_root}...")
    
    corrupted = []
    
    for f in files:
        size = os.path.getsize(f)
        if size % 104 != 0:
            print(f"Corrupted file: {os.path.basename(f)} (Size: {size})")
            corrupted.append(f)
            
    if corrupted:
        print(f"\nFound {len(corrupted)} corrupted files.")
    else:
        print("\nAll files have valid sizes (divisible by 104 bytes).")

if __name__ == '__main__':
    check_file_sizes()
