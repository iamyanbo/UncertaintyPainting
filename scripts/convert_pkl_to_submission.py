import pickle
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Convert result.pkl to KITTI submission txt files')
    parser.add_argument('--pkl_file', type=str, required=True, help='Path to result.pkl')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save txt files')
    args = parser.parse_args()

    pkl_path = Path(args.pkl_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        detections = pickle.load(f)

    print(f"Found {len(detections)} detection (frames)")
    
    count = 0
    for det in detections:
        frame_id = det['frame_id']
        txt_file = output_dir / f"{frame_id}.txt"
        
        # Keys in det: name, alpha, bbox, dimensions, location, rotation_y, score
        names = det['name']
        alphas = det['alpha']
        bboxes = det['bbox']
        dims = det['dimensions'] # l, h, w (camera coords?) -> check logic
        # In generate_prediction_dicts: 
        # dims = single_pred_dict['dimensions'] # lhw -> hwl (OpenPCDet comment says lhw->hwl?)
        # Let's check kitti_dataset.py lines 346-352:
        # dims = single_pred_dict['dimensions']
        # print format: ... dims[idx][1], dims[idx][2], dims[idx][0] ...
        # This implies stored dimensions are (l, h, w) and printed as (h, w, l).
        
        locs = det['location']
        rys = det['rotation_y']
        scores = det['score']

        with open(txt_file, 'w') as f:
            for i in range(len(names)):
                # Filter low scores if needed? Usually not for submission, but Kitti eval handles it.
                # Format: type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry score
                # truncated, occluded are -1 -1 explicitly in kitti_dataset.py
                name = names[i]
                alpha = alphas[i]
                bbox = bboxes[i] # x1 y1 x2 y2
                l, h, w = dims[i] # Stored as l, h, w in dictionary from generate_prediction_dicts
                x, y, z = locs[i]
                ry = rys[i]
                score = scores[i]
                
                # Height, Width, Length order for KITTI txt
                # Using logic from kitti_dataset.py: dims[idx][1], dims[idx][2], dims[idx][0] -> h, w, l
                
                line = f"{name} -1 -1 {alpha:.4f} {bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f} " \
                       f"{h:.4f} {w:.4f} {l:.4f} {x:.4f} {y:.4f} {z:.4f} {ry:.4f} {score:.4f}"
                f.write(line + "\n")
        count += 1

    print(f"Converted {count} files to {output_dir}")

if __name__ == '__main__':
    main()
