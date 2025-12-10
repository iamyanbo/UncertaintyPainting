
import os

def extract_r40(log_path, label):
    if not os.path.exists(log_path):
        print(f"[{label}] File not found: {log_path}")
        return

    print(f"--- {label} ---")
    try:
        with open(log_path, 'rb') as f:
            # Read last 200KB to be safe (logs can rely on verbose output)
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200000))
            lines = f.read().decode('utf-8', errors='ignore').splitlines()
            
        # Scan backward for the last "Performance of EPOCH"
        start_idx = -1
        last_epoch_str = ""
        for i in range(len(lines) - 1, -1, -1):
            if "Performance of EPOCH" in lines[i]:
                start_idx = i
                last_epoch_str = lines[i].strip()
                break
        
        if start_idx != -1:
            print(f"Found results for: {last_epoch_str}")
            
            class_order = ["Car", "Pedestrian", "Cyclist"]
            metrics = {c: None for c in class_order}
            
            # Forward scan from start_idx
            for j in range(start_idx, len(lines)):
                line = lines[j].strip()
                for cls_name in class_order:
                    # Look for "Car AP_R40"
                    if f"{cls_name} AP_R40" in line:
                        # usually 3 lines down is 3d ap
                        # line j: Car AP_R40...
                        # line j+1: bbox...
                        # line j+2: bev...
                        # line j+3: 3d...
                        found_3d = False
                        # Search next 5 lines for "3d   AP:"
                        for k in range(1, 6):
                            if j+k < len(lines):
                                subline = lines[j+k].strip()
                                if subline.startswith("3d   AP:"):
                                    parts = subline.split("AP:")[-1].split(",")
                                    if len(parts) == 3:
                                        metrics[cls_name] = [float(p.strip()) for p in parts]
                                        found_3d = True
                                    break
            
            for cls_name in class_order:
                vals = metrics[cls_name]
                if vals:
                    print(f"{cls_name}: {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f}")
                else:
                    print(f"{cls_name}: Not Found")
        else:
            print("No 'Performance of EPOCH' found in last 200KB.")

    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    print("\n")

def main():
    logs = [
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\pointpillar_painted_full\default\train_20251205-100116.log', 'PointPillars (With Uncertainty)'),
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\pointpillar_painted_no_uncert\default\train_20251209-102335.log', 'PointPillars (No Uncertainty)'),
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\second_painted_full\default\train_20251205-224847.log', 'SECOND (With Uncertainty)'),
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\second_painted_no_uncertainty\default\train_20251210-004850.log', 'SECOND (No Uncertainty)'),
    ]
    
    for path, label in logs:
        extract_r40(path, label)

if __name__ == '__main__':
    main()
