"""
Automatically clean dataset by running info generation and removing crashing samples.
"""
import subprocess
import sys
import os
import time

def get_last_sample(output_lines):
    """Parse the last 'train sample_idx: XXXXXX' line."""
    for line in reversed(output_lines):
        if "train sample_idx:" in line:
            return line.strip().split(': ')[1]
    return None

def remove_sample(sample_idx):
    print(f"Removing bad sample: {sample_idx}")
    train_txt = 'data/ImageSets/train.txt'
    with open(train_txt, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    
    if sample_idx in lines:
        lines.remove(sample_idx)
        with open(train_txt, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Removed {sample_idx}. Remaining: {len(lines)}")
        return True
    else:
        print(f"Sample {sample_idx} not found in train.txt")
        return False

def auto_clean():
    max_retries = 100
    cmd = [sys.executable, 'scripts/create_painted_infos.py', '--workers', '1']
    
    for attempt in range(max_retries):
        print(f"\n=== Attempt {attempt+1}/{max_retries} ===")
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
        output_lines = []
        last_sample = None
        
        try:
            for line in process.stdout:
                # Print only every 100 lines to reduce noise, or if it's not a sample line
                if "train sample_idx:" in line:
                    last_sample = line.strip().split(': ')[1]
                    if int(last_sample) % 100 == 0:
                        print(line, end='')
                else:
                    print(line, end='')
                output_lines.append(line)
        except Exception as e:
            print(f"Error reading output: {e}")
            
        process.wait()
        
        if process.returncode == 0:
            print("\nSUCCESS! Info generation completed.")
            return
        
        print(f"\nCRASH DETECTED! Return code: {process.returncode}")
        
        # Identify bad sample
        # If it crashed, the last printed sample is likely the one that STARTED processing
        # and caused the crash. Or maybe the one AFTER it?
        # Usually "train sample_idx: X" is printed BEFORE processing.
        # So if the last line is "train sample_idx: 005104", then 005104 is the bad one.
        
        if last_sample:
            print(f"Last processed sample: {last_sample}")
            if not remove_sample(last_sample):
                print("Could not remove sample. Aborting to prevent infinite loop.")
                break
        else:
            print("Could not determine last sample. Aborting.")
            break
            
        # Wait a bit before restarting
        time.sleep(1)

if __name__ == '__main__':
    auto_clean()
