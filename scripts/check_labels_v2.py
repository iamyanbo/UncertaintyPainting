"""
Check labels v2. Debug 002087.
"""
import os
import glob

def check_labels(label_dir='data/training/label_2'):
    files = glob.glob(os.path.join(label_dir, '*.txt'))
    print(f"Checking {len(files)} label files...")
    
    bad_files = []
    
    for f in files:
        is_debug = '002087' in f
        
        with open(f, 'r') as file:
            lines = file.readlines()
            
        for i, line in enumerate(lines):
            line_strip = line.strip()
            if not line_strip: 
                if is_debug: print(f"Line {i} is empty/whitespace")
                continue
            
            parts = line_strip.split(' ')
            parts = [p for p in parts if p]
            
            if is_debug: print(f"Line {i}: {parts} (len={len(parts)})")
            
            if len(parts) != 15:
                print(f"Corruption in {os.path.basename(f)} line {i+1}: {len(parts)} fields")
                bad_files.append(os.path.basename(f).replace('.txt', ''))
                break
                
            try:
                for p in parts[1:]:
                    float(p)
            except ValueError:
                 print(f"Parsing error in {os.path.basename(f)} line {i+1}")
                 bad_files.append(os.path.basename(f).replace('.txt', ''))
                 break

    print(f"\nFound {len(bad_files)} bad label files.")
    return bad_files

if __name__ == '__main__':
    check_labels()
