
import os

def extract_results():
    log_path = r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\second_painted_no_uncertainty\default\train_20251210-004850.log'
    out_path = r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\results_summary.txt'
    
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Find the last occurrence of "Performance of EPOCH"
    start_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Performance of EPOCH" in lines[i]:
            start_idx = i
            break
    
    with open(out_path, 'w', encoding='utf-8') as f_out:
        if start_idx != -1:
            f_out.write("Found results section:\n")
            for j in range(start_idx, len(lines)):
                # Filter out progress bar updates if they contain carriage returns or weird stuff
                # But usually reading lines separates them.
                line = lines[j].strip()
                if not line: continue
                # Simple heuristic: if it looks like a log line or result
                f_out.write(line + "\n")
                if "Evaluation done" in line:
                    break
        else:
            f_out.write("Could not find 'Performance of EPOCH' in log.\n")
    
    print(f"Results written to {out_path}")

if __name__ == '__main__':
    extract_results()
