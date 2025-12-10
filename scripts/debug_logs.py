
import os

def check_log_tail(log_path, label):
    if not os.path.exists(log_path):
        print(f"[{label}] File not found: {log_path}")
        return

    print(f"--- TAIL of {label} ---")
    try:
        with open(log_path, 'rb') as f:
            f.seek(-4000, 2)
            print(f.read().decode('utf-8', errors='ignore'))
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    print("\n")

def main():
    logs = [
        # (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\pointpillar_painted\default\train_20251205-100116.log', 'PointPillars (With Uncertainty)'), # This failed with File Not Found, check path first
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\pointpillar_painted_no_uncert\default\train_20251209-102335.log', 'PointPillars (No Uncertainty)'),
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\second_painted_full\default\train_20251205-224847.log', 'SECOND (With Uncertainty)'),
    ]
    
    for path, label in logs:
        check_log_tail(path, label)

if __name__ == '__main__':
    main()
