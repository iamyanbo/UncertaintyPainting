"""
Debug 002087.txt
"""
import os

def debug():
    fpath = 'data/training/label_2/002087.txt'
    size = os.path.getsize(fpath)
    print(f"Size: {size}")
    
    with open(fpath, 'rb') as f:
        content = f.read()
        print(f"Content (bytes): {content}")
        print(f"Hex: {content.hex()}")
        
    try:
        text = content.decode('utf-8')
        print(f"Decoded text repr: {repr(text)}")
        lines = text.splitlines()
        print(f"Line count: {len(lines)}")
        for i, line in enumerate(lines):
            print(f"Line {i}: {repr(line)}")
    except Exception as e:
        print(f"Decode error: {e}")

if __name__ == '__main__':
    debug()
