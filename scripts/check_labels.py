"""
Check all label files for corruption.
Standard KITTI label has 15 fields per line.
"""
import os
import glob

def check_labels(label_dir='data/training/label_2'):
    files = glob.glob(os.path.join(label_dir, '*.txt'))
    print(f"Checking {len(files)} label files in {label_dir}...")
    
    bad_files = []
    
    for f in files:
        try:
            with open(f, 'rb') as file:
                content = file.read()
                
            if b'\x00' in content:
                print(f"Found null bytes in {os.path.basename(f)}")
                bad_files.append(os.path.basename(f).replace('.txt', ''))
                continue
                
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                print(f"Unicode decode error in {os.path.basename(f)}")
                bad_files.append(os.path.basename(f).replace('.txt', ''))
                continue
                
            lines = text_content.splitlines()
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                
                parts = line.split(' ')
                # Filter out empty strings from multiple spaces
                parts = [p for p in parts if p]
                
                if len(parts) != 15:
                    print(f"Potential corruption in {os.path.basename(f)} line {i+1}: {len(parts)} fields (expected 15)")
                    # print(f"Content: {line[:50]}...")
                    bad_files.append(os.path.basename(f).replace('.txt', ''))
                    break
                    
                try:
                    # Check if fields 1-14 are numbers
                    for p in parts[1:]:
                        float(p)
                except ValueError:
                     print(f"Parsing error in {os.path.basename(f)} line {i+1}")
                     bad_files.append(os.path.basename(f).replace('.txt', ''))
                     break
        except Exception as e:
            print(f"Error reading {f}: {e}")
            bad_files.append(os.path.basename(f).replace('.txt', ''))

    print(f"\nFound {len(bad_files)} bad label files.")
    return bad_files

if __name__ == '__main__':
    bad_samples = check_labels()
    
    if bad_samples:
        print("Removing bad samples from train.txt...")
        with open('data/ImageSets/train.txt', 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        
        original_count = len(lines)
        lines = [l for l in lines if l not in bad_samples]
        new_count = len(lines)
        
        with open('data/ImageSets/train.txt', 'w') as f:
            f.write('\n'.join(lines))
            
        print(f"Removed {original_count - new_count} samples from train.txt")
