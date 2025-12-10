
import os

def parse_log_results():
    log_path = r'c:\Users\yanbo\Downloads\Papers\uncertainty_painting\OpenPCDet\output\second_painted_no_uncertainty\default\train_20251210-004850.log'
    
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Find the last occurrence of "Performance of EPOCH"
    start_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Performance of EPOCH" in lines[i]:
            start_idx = i
            break
    
    if start_idx != -1:
        print("Found results section:")
        # Print subsequent lines until we see something unrelated or end of file
        for j in range(start_idx, len(lines)):
            print(lines[j].strip())
            # Stop if we hit "Evaluation done" or similar if needed, 
            # but usually the table is short.
            if "Evaluation done" in lines[j]:
                break
    else:
        print("Could not find 'Performance of EPOCH' in log.")

if __name__ == '__main__':
    parse_log_results()
