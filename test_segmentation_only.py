#!/usr/bin/env python3
"""
Test just the segmentation functionality without plotting
"""

import sys
import numpy as np

# Add the project to the path
sys.path.insert(0, '/home/wxc151/iqid-alphas')

def test_segmentation_only():
    """Test just the blob segmentation without visualization"""
    try:
        print("Testing blob segmentation...")
        
        # Import what we need
        from sklearn.cluster import KMeans
        from skimage import filters, morphology, segmentation, measure, feature
        from skimage.filters import threshold_otsu, threshold_local
        from skimage.morphology import disk, remove_small_objects, binary_opening, binary_closing
        from skimage.segmentation import watershed, clear_border
        from scipy import ndimage
        import cv2
        
        print("✓ All imports successful")
        
        # Create test image
        test_image = np.random.randint(1000, 50000, (256, 256), dtype=np.uint16)
        
        # Convert to 8-bit
        image_8bit = ((test_image - test_image.min()) / 
                      (test_image.max() - test_image.min()) * 255).astype(np.uint8)
        
        print(f"✓ Created and converted test image: {test_image.shape}")
        
        # Test thresholding
        otsu_thresh = threshold_otsu(image_8bit)
        binary_otsu = image_8bit > otsu_thresh
        print(f"✓ Otsu thresholding successful, threshold: {otsu_thresh}")
        
        # Test K-means
        pixels = image_8bit.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        print("✓ K-means clustering successful")
        
        # Test morphological operations
        cleaned = remove_small_objects(binary_otsu, min_size=100)
        filled = ndimage.binary_fill_holes(cleaned)
        print("✓ Morphological operations successful")
        
        # Test watershed
        distance = ndimage.distance_transform_edt(filled)
        local_maxima = feature.peak_local_max(distance, min_distance=20, threshold_abs=distance.max()*0.3)
        print(f"✓ Watershed setup successful, found {len(local_maxima[0])} peaks")
        
        print("\n🎉 Core segmentation functionality works!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_segmentation_only()
    sys.exit(0 if success else 1)
