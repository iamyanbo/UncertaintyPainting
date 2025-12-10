
import os
import subprocess

def run_evaluation(ckpt_path, cfg_file, tag):
    cmd = [
        "python", "OpenPCDet/tools/test.py",
        "--cfg_file", cfg_file,
        "--ckpt", ckpt_path,
        "--batch_size", "4",
        "--extra_tag", tag
    ]
    print(f"Running evaluation for {tag}...")
    try:
        # Run and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"--- Evaluation Results for {tag} ---")
        
        # Parse output for R40
        lines = result.stdout.splitlines()
        found_r40 = False
        for i, line in enumerate(lines):
            if "Performance of EPOCH" in line:
                 print(f"Header: {line}")
            if "AP_R40" in line:
                print(line)
                # Print next 3 lines (bbox, bev, 3d)
                for k in range(1, 4):
                    if i + k < len(lines):
                        print(lines[i+k])
                found_r40 = True
        
        if not found_r40:
             print("No R40 metrics found in output.")
             # print(result.stdout[-2000:]) # Debug print if needed

    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {tag}: {e}")
        # print(e.stdout)
        print(e.stderr)
    except Exception as e:
         print(f"An unexpected error occurred for {tag}: {e}")
    print("\n" + "="*50 + "\n")

def main():
    root = r"c:\Users\yanbo\Downloads\Papers\uncertainty_painting"
    
    # Correct config paths assumed to be in root based on list_dir
    models = [
        (
            r"c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\pointpillar_painted_full\default\ckpt\checkpoint_epoch_80.pth",
            os.path.join(root, "pointpillar_painted.yaml"), 
            "pointpillar_painted_uncertainty"
        ),
        (
            r"c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\pointpillar_painted_no_uncert\default\ckpt\checkpoint_epoch_80.pth",
            os.path.join(root, "pointpillar_painted_no_uncertainty.yaml"),
            "pointpillar_no_uncertainty"
        ),
        (
            r"c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\kitti_models\second_painted_full\default\ckpt\checkpoint_epoch_80.pth",
            os.path.join(root, "second_painted.yaml"), 
            "second_painted_uncertainty"
        )
    ]

    for ckpt, cfg, tag in models:
         if os.path.exists(ckpt) and os.path.exists(cfg):
             run_evaluation(ckpt, cfg, tag)
         else:
             print(f"Missing file: CKPT={os.path.exists(ckpt)}, CFG={os.path.exists(cfg)} \nCKPT: {ckpt}\nCFG: {cfg}")

if __name__ == "__main__":
    main()
