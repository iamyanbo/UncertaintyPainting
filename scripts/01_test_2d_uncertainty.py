import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add the project root to the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.segmentation import UncertaintyPainter2D

def create_dummy_image(path):
    print("Creating dummy image for testing...")
    # Create a simple image with a circle and a square
    img = Image.new('RGB', (500, 375), color = (73, 109, 137))
    
    # Just save some random noise or simple shapes if PIL draw is too complex for now
    # Actually, let's just make a gradient
    arr = np.zeros((375, 500, 3), dtype=np.uint8)
    for i in range(375):
        for j in range(500):
            arr[i, j] = [i % 255, j % 255, (i+j) % 255]
    
    img = Image.fromarray(arr)
    img.save(path)

def main():
    # 1. Setup
    image_path = "data/test_image.png"
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists(image_path):
        create_dummy_image(image_path)
        
    # 2. Initialize Model
    try:
        painter = UncertaintyPainter2D(model_name='deeplabv3_resnet50') # Use resnet50 for speed
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # 3. Run Prediction
    print(f"Running inference on {image_path}...")
    entropy_map, class_probs, original_img = painter.predict(image_path)
    
    print(f"Entropy Map Shape: {entropy_map.shape}")
    print(f"Max Entropy: {np.max(entropy_map)}")
    print(f"Min Entropy: {np.min(entropy_map)}")

    # 4. Visualize
    # Get predicted classes (argmax of probabilities)
    # class_probs is (K, H, W)
    pred_mask = np.argmax(class_probs, axis=0)
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Predicted Classes")
    plt.imshow(pred_mask, cmap='tab20') # Use a categorical colormap
    plt.colorbar(label='Class ID')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Uncertainty (Entropy) Map")
    plt.imshow(entropy_map, cmap='inferno')
    plt.colorbar(label='Entropy')
    plt.axis('off')
    
    output_path = "data/test_uncertainty_result.png"
    plt.savefig(output_path)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()
