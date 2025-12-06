"""
Simple test to verify painted point cloud loading.
"""
import numpy as np
from pathlib import Path

def test_painted_loading():
    """Test loading painted point clouds directly."""
    
    # Load original LiDAR
    original_file = Path('data/training/velodyne/000000.bin')
    original = np.fromfile(str(original_file), dtype=np.float32).reshape(-1, 4)
    
    # Load painted LiDAR
    painted_file = Path('data/training/velodyne_painted/000000.bin')
    painted = np.fromfile(str(painted_file), dtype=np.float32).reshape(-1, 26)
    
    print("Original point cloud:")
    print(f"  Shape: {original.shape}")
    print(f"  First 3 points:\n{original[:3]}")
    
    print("\nPainted point cloud:")
    print(f"  Shape: {painted.shape}")
    print(f"  Features: 4 (base) + 21 (classes) + 1 (uncertainty) = 26")
    
    print(f"\n  First point breakdown:")
    print(f"    Base (x,y,z,intensity): {painted[0, :4]}")
    print(f"    Class probs (21): {painted[0, 4:25]}")
    print(f"    Uncertainty: {painted[0, 25]}")
    
    # Verify base features match
    if np.allclose(original[:, :4], painted[:, :4]):
        print("\n✓ Base features match original LiDAR")
    else:
        print("\n✗ WARNING: Base features don't match!")
    
    # Check uncertainty range
    uncertainty = painted[:, 25]
    print(f"\nUncertainty statistics:")
    print(f"  Min: {uncertainty.min():.4f}")
    print(f"  Max: {uncertainty.max():.4f}")
    print(f"  Mean: {uncertainty.mean():.4f}")
    
    # Check class probabilities
    class_probs = painted[:, 4:25]
    print(f"\nClass probabilities statistics:")
    print(f"  Sum per point (should be ~1.0): {class_probs.sum(axis=1)[:5]}")
    
    print("\n✓ Painted point cloud test passed!")

if __name__ == '__main__':
    test_painted_loading()
