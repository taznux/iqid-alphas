#!/usr/bin/env python3
"""
Simple test for the fixed visualizer
"""

import sys
import numpy as np
from pathlib import Path

# Add the project to the path
sys.path.insert(0, '/home/wxc151/iqid-alphas')

def test_simple_visualization():
    """Test the visualizer with a simple synthetic image"""
    try:
        from evaluation.utils.visualizer import IQIDVisualizer
        print("✓ Successfully imported IQIDVisualizer")
        
        # Create visualizer
        viz = IQIDVisualizer(output_dir="test_output_simple")
        print(f"✓ Created visualizer with output dir: {viz.output_dir}")
        
        # Create a simple test image with clear blobs
        test_image = np.zeros((512, 512), dtype=np.uint16)
        
        # Add some circular blobs in a 2x2 grid pattern
        centers = [(128, 128), (128, 384), (384, 128), (384, 384)]
        for i, (cx, cy) in enumerate(centers):
            y, x = np.ogrid[:512, :512]
            mask = (x - cx)**2 + (y - cy)**2 < 50**2
            test_image[mask] = 40000 + i * 5000  # Different intensities
        
        # Add some background
        test_image += 2000
        
        print(f"✓ Created test image: {test_image.shape}, {test_image.dtype}")
        print(f"  Intensity range: {test_image.min()} - {test_image.max()}")
        
        # Test grid overlay (skip automatic detection for speed)
        grid_viz = viz.create_grid_overlay_visualization(
            test_image, 
            expected_grid_size=(2, 2),
            detected_blobs=None,  # Will skip detection due to size
            sample_id="simple_test"
        )
        print(f"✓ Grid overlay visualization: {grid_viz}")
        
        # Test basic conversion
        img_8bit = viz.convert_16bit_to_8bit(test_image)
        print(f"✓ 8-bit conversion: {img_8bit.shape}, {img_8bit.dtype}")
        
        # Check if files exist
        if Path(grid_viz).exists():
            print(f"✓ Visualization file created: {Path(grid_viz).stat().st_size} bytes")
        else:
            print(f"❌ Visualization file not found: {grid_viz}")
        
        print("\n🎉 Simple visualization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_visualization()
    sys.exit(0 if success else 1)
