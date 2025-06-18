#!/usr/bin/env python3
"""
Simple test for the enhanced visualizer
"""

import sys
import numpy as np
from pathlib import Path

# Add the project to the path
sys.path.insert(0, '/home/wxc151/iqid-alphas')

def test_enhanced_visualizer():
    """Test the enhanced visualizer capabilities"""
    try:
        from evaluation.utils.visualizer import IQIDVisualizer
        print("✓ Successfully imported IQIDVisualizer")
        
        # Create visualizer
        viz = IQIDVisualizer(output_dir="test_output")
        print(f"✓ Created visualizer with output dir: {viz.output_dir}")
        
        # Create a simple test image
        test_image = np.random.randint(1000, 50000, (256, 256), dtype=np.uint16)
        print(f"✓ Created test image: {test_image.shape}, {test_image.dtype}")
        
        # Test 16-bit to 8-bit conversion
        img_8bit = viz.convert_16bit_to_8bit(test_image)
        print(f"✓ Converted to 8-bit: {img_8bit.shape}, {img_8bit.dtype}")
        
        # Test colormap application
        img_colored = viz.apply_colormap(img_8bit, 'viridis')
        print(f"✓ Applied colormap: {img_colored.shape}, {img_colored.dtype}")
        
        # Test blob segmentation
        blobs, mask = viz.robust_blob_segmentation(test_image, min_blob_size=100)
        print(f"✓ Blob segmentation: detected {len(blobs)} blobs")
        
        # Test grid overlay
        grid_viz = viz.create_grid_overlay_visualization(
            test_image, 
            expected_grid_size=(2, 2),
            sample_id="test"
        )
        print(f"✓ Grid overlay visualization: {grid_viz}")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_visualizer()
    sys.exit(0 if success else 1)
