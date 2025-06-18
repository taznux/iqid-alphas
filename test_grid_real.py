#!/usr/bin/env python3
"""
Test grid visualization with a real iQID image
"""

import sys
import numpy as np
from pathlib import Path
import skimage.io

# Add the project to the path
sys.path.insert(0, '/home/wxc151/iqid-alphas')

def test_grid_visualization():
    """Test grid visualization with a real image"""
    try:
        from evaluation.utils.visualizer import IQIDVisualizer
        
        # Find a real iQID image
        data_path = Path('/home/wxc151/iqid-alphas/data/ReUpload/iQID_reupload/iQID')
        
        # Look for raw images
        raw_images = list(data_path.rglob("*iqid_event_image.tif"))
        if not raw_images:
            print("❌ No raw iQID images found")
            return False
        
        raw_image_path = raw_images[0]
        print(f"✓ Found test image: {raw_image_path}")
        
        # Load image
        image = skimage.io.imread(str(raw_image_path))
        print(f"✓ Loaded image: shape={image.shape}, dtype={image.dtype}")
        print(f"   Value range: {image.min()} - {image.max()}")
        
        # Create visualizer
        viz = IQIDVisualizer(output_dir="test_grid_output")
        print("✓ Created visualizer")
        
        # Test data type conversion
        image_8bit = viz.convert_16bit_to_8bit(image)
        print(f"✓ Converted to 8-bit: shape={image_8bit.shape}, dtype={image_8bit.dtype}")
        print(f"   Value range: {image_8bit.min()} - {image_8bit.max()}")
        
        # Test grid visualization
        grid_viz = viz.create_grid_overlay_visualization(
            image,
            expected_grid_size=(3, 4),
            sample_id="real_test"
        )
        print(f"✓ Grid visualization saved to: {grid_viz}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_grid_visualization()
    sys.exit(0 if success else 1)
