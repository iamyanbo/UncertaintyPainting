
import os

def simple_grep_r40(log_path, label):
    if not os.path.exists(log_path):
        print(f"[{label}] File not found")
        return

    print(f"--- {label} ---")
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Find the LAST instance of "AP_R40" for each class
        results = {}
        for cls in ["Car", "Pedestrian", "Cyclist"]:
            # Search backwards
            for i in range(len(lines)-1, -1, -1):
                if f"{cls} AP_R40" in lines[i]:
                    # Grab this line and the next few
                    # We look for "3d   AP:" in the next 10 lines
                    for k in range(1, 10):
                        if i+k < len(lines) and "3d   AP:" in lines[i+k]:
                            parts = lines[i+k].split("AP:")[-1].split(",")
                            if len(parts) == 3:
                                results[cls] = [float(p.strip()) for p in parts]
                            break
                    if cls in results: break # Found latest
        
        # Print
        for cls in ["Car", "Pedestrian", "Cyclist"]:
            if cls in results:
                vals = results[cls]
                print(f"{cls}: {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f}")
            else:
                print(f"{cls}: Not Found")
                
    except Exception as e:
        print(f"Error: {e}")
    print("\n")

def main():
    logs = [
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\pointpillar_painted_full\default\train_20251205-100116.log', 'PointPillars (Uncertainty)'),
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\pointpillar_painted_no_uncert\default\train_20251209-102335.log', 'PointPillars (No Uncertainty)'),
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\second_painted_full\default\train_20251205-224847.log', 'SECOND (Uncertainty)'),
        (r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\second_painted_no_uncertainty\default\train_20251210-004850.log', 'SECOND (No Uncertainty)'),
    ]
    
    for path, label in logs:
        simple_grep_r40(path, label)

if __name__ == '__main__':
    main()
