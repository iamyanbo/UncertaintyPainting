"""
Robustly filter crashing samples.
Runs check_samples_batch.py in subprocesses.
"""
import subprocess
import sys
import os

def robust_filter():
    # Get total count
    with open('data/ImageSets/train.txt', 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    total_samples = len(lines)
    print(f"Total samples to check: {total_samples}")
    
    current_idx = 0
    bad_samples = []
    
    while current_idx < total_samples:
        print(f"Starting batch from index {current_idx}...")
        
        # Run batch check
        cmd = [sys.executable, 'scripts/check_samples_batch.py', str(current_idx), str(total_samples)]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        last_checked = -1
        
        try:
            for line in process.stdout:
                print(line, end='')
                if line.startswith("Checking "):
                    # Extract index
                    idx_str = line.strip().split(' ')[1].replace('...', '')
                    # Find index in lines list
                    try:
                        last_checked = lines.index(idx_str)
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Error reading stdout: {e}")
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\nCRASH DETECTED! Return code: {process.returncode}")
            if last_checked != -1:
                # The crash happened AFTER checking 'last_checked' started.
                # So 'last_checked' IS the bad sample (it started checking but didn't finish or crashed during it).
                # Wait, my script prints "Checking X..." BEFORE the crash.
                # So if the last line is "Checking 002042...", then 002042 is the bad one.
                bad_sample = lines[last_checked]
                print(f"Identified bad sample: {bad_sample}")
                bad_samples.append(bad_sample)
                current_idx = last_checked + 1
            else:
                print("Could not identify bad sample (crash at start?). Skipping 1.")
                current_idx += 1
        else:
            print("\nBatch finished successfully.")
            break
            
    print(f"\nFound {len(bad_samples)} bad samples: {bad_samples}")
    
    # Write new train.txt
    if bad_samples:
        new_lines = [l for l in lines if l not in bad_samples]
        with open('data/ImageSets/train.txt', 'w') as f:
            f.write('\n'.join(new_lines))
        print(f"Updated train.txt. Removed {len(bad_samples)} samples. New count: {len(new_lines)}")
    else:
        print("No bad samples found.")

if __name__ == '__main__':
    robust_filter()
