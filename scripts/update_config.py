import re

config_path = 'OpenPCDet/tools/cfgs/kitti_models/pointpillar_painted_full.yaml'

with open(config_path, 'r') as f:
    content = f.read()

# Update Batch Size
if 'BATCH_SIZE_PER_GPU: 2' in content:
    content = content.replace('BATCH_SIZE_PER_GPU: 2', 'BATCH_SIZE_PER_GPU: 4')
    print("Updated BATCH_SIZE_PER_GPU to 4")
else:
    print("BATCH_SIZE_PER_GPU not found or already updated")

# Update Num Workers
# Check if NUM_WORKERS exists
if 'NUM_WORKERS:' in content:
    content = re.sub(r'NUM_WORKERS: \d+', 'NUM_WORKERS: 4', content)
    print("Updated existing NUM_WORKERS to 4")
else:
    # Add it under OPTIMIZATION if not present
    # Look for BATCH_SIZE_PER_GPU line and add it after
    if 'BATCH_SIZE_PER_GPU: 4' in content:
        content = content.replace('BATCH_SIZE_PER_GPU: 4', 'BATCH_SIZE_PER_GPU: 4\n    NUM_WORKERS: 4')
        print("Added NUM_WORKERS: 4")
    else:
        print("Could not find insertion point for NUM_WORKERS")

with open(config_path, 'w') as f:
    f.write(content)
