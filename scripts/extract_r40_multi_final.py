
import os

def extract_r40(log_path, label):
    if not os.path.exists(log_path):
        print(f"[{label}] File not found: {log_path}")
        return

    print(f"--- {label} ---")
    try:
        with open(log_path, 'rb') as f:
            # Read last 8KB to be safe
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16000))
            lines = f.read().decode('utf-8', errors='ignore').splitlines()
            
        # Scan backward for the last "Performance of EPOCH"
        start_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if "Performance of EPOCH" in lines[i]:
                start_idx = i
                break
        
        if start_idx != -1:
            epoch_line = lines[start_idx].strip()
            print(f"Found results for: {epoch_line}")
            
            class_order = ["Car", "Pedestrian", "Cyclist"]
            metrics = {c: None for c in class_order}
            
            for j in range(start_idx, len(lines)):
                line = lines[j].strip()
                for cls_name in class_order:
                    if f"{cls_name} AP_R40" in line:
                        # Extract 3d AP from 3 lines down
                        if j + 3 < len(lines):
                            ap_3d_line = lines[j+3].strip() # "3d   AP:98.8879, 96.3008, 90.2465"
                            # Parse numbers
                            if "AP:" in ap_3d_line:
                                parts = ap_3d_line.split("AP:")[-1].split(",")
                                if len(parts) == 3:
                                    metrics[cls_name] = [float(p.strip()) for p in parts]
            
            # Print in formatted way for easy copy-paste
            # Format: Easy, Mod, Hard
            for cls_name in class_order:
                vals = metrics[cls_name]
                if vals:
                    print(f"{cls_name}: {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f}")
                else:
                    print(f"{cls_name}: Not Found")
        else:
            print("No 'Performance of EPOCH' found in last 16KB.")

    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    print("\n")

def main():
    logs = [
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\pointpillar_painted_full\default\train_20251205-100116.log', 'PointPillars (With Uncertainty)'),
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\pointpillar_painted_no_uncert\default\train_20251209-102335.log', 'PointPillars (No Uncertainty) - Attempt 1'),
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\second_painted_full\default\train_20251205-224847.log', 'SECOND (With Uncertainty)'),
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\second_painted_no_uncertainty\default\train_20251210-004850.log', 'SECOND (No Uncertainty)'),
    ]
    
    for path, label in logs:
        extract_r40(path, label)

if __name__ == '__main__':
    main()
