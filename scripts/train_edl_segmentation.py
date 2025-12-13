import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.segmentation import UncertaintyPainter2D
from src import edl

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune DeepLabV3 with EDL Loss')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default 8 for speed)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_root', type=str, default='./data/voc', help='Path to VOC dataset')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints/edl_deeplabv3.pth', help='Path to save model')
    return parser.parse_args()

def get_dataloader(data_root, batch_size):
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((513, 513)), # Optimal DeepLabV3 size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((513, 513), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    print(f"Loading VOC Segmentation dataset from {data_root}...")
    
    # helper to manually download if needed
    def download_voc_manual(root):
        import urllib.request
        import tarfile
        os.makedirs(root, exist_ok=True)
        
        filename = "VOCtrainval_11-May-2012.tar"
        fpath = os.path.join(root, filename)
        
        # Check if file exists and is valid
        if os.path.exists(fpath) and os.path.getsize(fpath) > 100*1024*1024:
            print("Dataset file exists and appears valid.")
            return

        urls = [
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
            "https://web.archive.org/web/20140815141459/http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        ]

        for url in urls:
            print(f"Attempting download from {url}...")
            try:
                # Add headers
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                
                urllib.request.urlretrieve(url, fpath)
                
                if os.path.exists(fpath) and os.path.getsize(fpath) > 10*1024*1024:
                    print("Download Success!")
                    return
                else:
                    print("Downloaded file too small, trying next mirror...")
                    if os.path.exists(fpath): os.remove(fpath)
            except Exception as e:
                print(f"Failed: {e}")
        
        print("All mirrors failed. Please download VOCtrainval_11-May-2012.tar manually.")

    # Loop to clean potentially corrupted existing file before start
    tar_path = os.path.join(data_root, "VOCtrainval_11-May-2012.tar")
    if os.path.exists(tar_path) and os.path.getsize(tar_path) < 100*1024*1024: # Less than 100MB is definitely wrong
        print("Removing corrupted file...")
        try: os.remove(tar_path)
        except: pass

    download_voc_manual(data_root)

    try:
        train_dataset = datasets.VOCSegmentation(
            root=data_root,
            year='2012',
            image_set='train',
            download=True,
            transform=transform,
            target_transform=target_transform
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have internet access to download PASCAL VOC 2012.")
        sys.exit(1)

    # Aggressive Optimization for Speed
    # num_workers=4: Parallel loading
    # persistent_workers=True: Keep workers alive (avoids respawn overhead on Windows)
    # pin_memory=True: Fast transfer
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f"Using Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch Size: {args.batch_size}")
    print(f"==================================================")
    
    # 1. Initialize Model
    # We use the existing class but access the internal model
    print("Initializing model...")
    painter = UncertaintyPainter2D(model_name='deeplabv3_resnet101', device='cpu') # Load on CPU first
    model = painter.model
    model.to(device)
    model.train()
    
    # 2. Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Dataloader
    dataloader = get_dataloader(args.data_root, args.batch_size)
    
    # 4. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    
    # Create checkpoint dir
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    annealing_step = 10 * len(dataloader) # Anneal over 10 epochs roughly
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device) # (N, 1, H, W)
            
            # VOC targets are 0-20, with 255 as ignore/boundary
            # We need to one-hot encode targets for EDL Mean Square Error
            # Ignore 255
            
            # Simple Mask for valid pixels
            valid_mask = (targets != 255).squeeze(1) # (N, H, W)
            
            if valid_mask.sum() == 0:
                continue
                
            # Forward pass
            outputs = model(images)['out'] # (N, 21, H, W)
            
            # We need to flatten and only keep valid pixels to save memory and handle ignore_index
            # Permute to (N, H, W, C)
            outputs = outputs.permute(0, 2, 3, 1)
            
            # Flatten to (N*H*W, C)
            outputs_flat = outputs[valid_mask] # (Num_Valid, 21)
            targets_flat = targets.squeeze(1)[valid_mask].long() # (Num_Valid,)
            
            # Convert targets to One-Hot
            # Num classes = 21
            y_onehot = torch.zeros_like(outputs_flat).scatter_(1, targets_flat.unsqueeze(1), 1)
            
            # Calculate Evidence via ReLU (or whatever activation we chose)
            # IMPORTANT: We use relu_evidence inside calculate_uncertainty, so here we must match
            # Actually, edl_loss functions usually expect *alpha* or *evidence*?
            # src/edl.py loss functions take 'alpha' as input usually, or we wrap it.
            # Let's inspect edl.edl_loss wrapper.
            # It calls func(y, alpha, ...)
            
            # So we need to calculate alpha first
            evidence = edl.relu_evidence(outputs_flat)
            alpha = evidence + 1
            
            # Calculate Loss
            loss = edl.edl_loss(
                edl.mse_loss,
                y_onehot,
                alpha,
                epoch_num=epoch,
                num_classes=21,
                annealing_step=10, # Passed as epochs usually, or we can use global step logic
                device=device
            )
            
            # Optimization
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            
            epoch_loss += loss.mean().item()
            pbar.set_postfix({'loss': loss.mean().item()})
            global_step += 1
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save checkpoint every epoch
        torch.save(model.state_dict(), args.model_save_path)
        print(f"Model saved to {args.model_save_path}")

if __name__ == "__main__":
    main()
