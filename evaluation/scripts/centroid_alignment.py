#!/usr/bin/env python3
"""
Enhanced Alignment with Centroid Centering and Proper Padding

This module provides improved alignment that:
1. Centers images based on tissue centroid
2. Pads to twice the largest slice size
3. Allows full movement range during alignment
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from skimage import measure, transform, filters
from scipy import ndimage
import logging


class CentroidAlignedPadding:
    """
    Handles centroid-based centering and consistent padding for alignment
    """
    
    def __init__(self):
        self.logger = logging.getLogger("CentroidAlignedPadding")
    
    def calculate_tissue_centroid(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the centroid of tissue regions in the image
        """
        try:
            # Create binary mask of tissue regions
            if image.max() > 1:
                # Apply threshold to find tissue
                threshold = filters.threshold_otsu(image)
                binary_mask = image > threshold
            else:
                # Already binary
                binary_mask = image > 0.5
            
            # Remove small objects
            from skimage.morphology import remove_small_objects
            binary_mask = remove_small_objects(binary_mask, min_size=100)
            
            if not np.any(binary_mask):
                # If no tissue found, use image center
                return image.shape[0] / 2, image.shape[1] / 2
            
            # Calculate centroid
            props = measure.regionprops(binary_mask.astype(int))
            if props:
                # Use the largest region's centroid
                largest_region = max(props, key=lambda p: p.area)
                centroid_y, centroid_x = largest_region.centroid
                return centroid_y, centroid_x
            else:
                # Fallback to center of mass
                cy, cx = ndimage.center_of_mass(binary_mask)
                return cy, cx
                
        except Exception as e:
            self.logger.warning(f"Centroid calculation failed: {e}, using image center")
            return image.shape[0] / 2, image.shape[1] / 2
    
    def find_target_size(self, images: List[np.ndarray]) -> Tuple[int, int]:
        """
        Find target size (twice the largest image dimensions)
        """
        max_height = 0
        max_width = 0
        
        for img in images:
            max_height = max(max_height, img.shape[0])
            max_width = max(max_width, img.shape[1])
        
        # Target size is twice the largest
        target_height = max_height * 2
        target_width = max_width * 2
        
        self.logger.debug(f"Target size: {target_height} x {target_width}")
        return target_height, target_width
    
    def center_and_pad_image(self, image: np.ndarray, 
                           target_size: Tuple[int, int],
                           centroid: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Center image based on centroid and pad to target size
        """
        target_height, target_width = target_size
        
        # Calculate centroid if not provided
        if centroid is None:
            centroid_y, centroid_x = self.calculate_tissue_centroid(image)
        else:
            centroid_y, centroid_x = centroid
        
        # Create padded image
        padded_image = np.zeros((target_height, target_width), dtype=image.dtype)
        
        # Calculate where to place the image so its centroid is at the center of padded image
        target_center_y = target_height // 2
        target_center_x = target_width // 2
        
        # Calculate offset to center the tissue centroid
        offset_y = target_center_y - centroid_y
        offset_x = target_center_x - centroid_x
        
        # Calculate placement coordinates
        start_y = int(offset_y)
        start_x = int(offset_x)
        end_y = start_y + image.shape[0]
        end_x = start_x + image.shape[1]
        
        # Handle boundary cases
        img_start_y = max(0, -start_y)
        img_start_x = max(0, -start_x)
        img_end_y = min(image.shape[0], image.shape[0] - (end_y - target_height))
        img_end_x = min(image.shape[1], image.shape[1] - (end_x - target_width))
        
        pad_start_y = max(0, start_y)
        pad_start_x = max(0, start_x)
        pad_end_y = min(target_height, end_y)
        pad_end_x = min(target_width, end_x)
        
        # Place the image
        try:
            padded_image[pad_start_y:pad_end_y, pad_start_x:pad_end_x] = \
                image[img_start_y:img_end_y, img_start_x:img_end_x]
        except Exception as e:
            self.logger.warning(f"Padding failed: {e}, using center placement")
            # Fallback: center placement
            start_y = (target_height - image.shape[0]) // 2
            start_x = (target_width - image.shape[1]) // 2
            end_y = start_y + image.shape[0]
            end_x = start_x + image.shape[1]
            padded_image[start_y:end_y, start_x:end_x] = image
        
        return padded_image
    
    def prepare_images_for_alignment(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        """
        Prepare all images for alignment with centroid centering and consistent padding
        """
        if not images:
            return [], {}
        
        # Find target size (twice the largest)
        target_size = self.find_target_size(images)
        
        # Calculate centroids for all images
        centroids = []
        for i, img in enumerate(images):
            centroid = self.calculate_tissue_centroid(img)
            centroids.append(centroid)
            self.logger.debug(f"Image {i} centroid: {centroid}")
        
        # Center and pad all images
        padded_images = []
        for i, img in enumerate(images):
            padded_img = self.center_and_pad_image(img, target_size, centroids[i])
            padded_images.append(padded_img)
        
        preparation_info = {
            'target_size': target_size,
            'original_sizes': [img.shape for img in images],
            'centroids': centroids,
            'method': 'centroid_centered_padding'
        }
        
        return padded_images, preparation_info


class EnhancedImageAligner:
    """
    Enhanced image aligner with centroid centering and proper padding
    """
    
    def __init__(self, method: str = 'phase_correlation'):
        self.method = method
        self.padding_handler = CentroidAlignedPadding()
        self.logger = logging.getLogger("EnhancedImageAligner")
    
    def align_image_stack(self, images: List[np.ndarray], 
                         reference_idx: Optional[int] = None) -> Tuple[List[np.ndarray], Dict]:
        """
        Align a stack of images with centroid centering and proper padding
        """
        if not images:
            return [], {}
        
        if len(images) == 1:
            padded_images, prep_info = self.padding_handler.prepare_images_for_alignment(images)
            return padded_images, {'preparation': prep_info, 'alignment': 'single_image'}
        
        # Prepare images with centroid centering and padding
        padded_images, preparation_info = self.padding_handler.prepare_images_for_alignment(images)
        
        # Determine reference image
        if reference_idx is None:
            reference_idx = self._find_best_reference(padded_images)
        
        reference_img = padded_images[reference_idx]
        aligned_images = [None] * len(padded_images)
        aligned_images[reference_idx] = reference_img.copy()
        
        alignment_info = {
            'reference_idx': reference_idx,
            'transforms': {},
            'quality_scores': {},
            'preparation': preparation_info
        }
        
        # Align all other images to reference
        for i, img in enumerate(padded_images):
            if i == reference_idx:
                alignment_info['quality_scores'][i] = 1.0
                continue
            
            try:
                aligned_img, transform_info = self._align_single_pair(reference_img, img)
                aligned_images[i] = aligned_img
                alignment_info['transforms'][i] = transform_info
                
                # Calculate alignment quality
                quality = self._calculate_alignment_quality(reference_img, aligned_img)
                alignment_info['quality_scores'][i] = quality
                
            except Exception as e:
                self.logger.warning(f"Failed to align image {i}: {e}")
                aligned_images[i] = img.copy()  # Use original padded image
                alignment_info['quality_scores'][i] = 0.0
                alignment_info['transforms'][i] = {'error': str(e)}
        
        return aligned_images, alignment_info
    
    def _find_best_reference(self, images: List[np.ndarray]) -> int:
        """
        Find the best reference image (highest tissue content and quality)
        """
        scores = []
        
        for i, img in enumerate(images):
            # Calculate tissue area
            if img.max() > 1:
                threshold = filters.threshold_otsu(img)
                tissue_mask = img > threshold
            else:
                tissue_mask = img > 0.5
            
            tissue_area = np.sum(tissue_mask)
            
            # Calculate contrast
            contrast = img.std()
            
            # Combined score
            score = tissue_area * contrast
            scores.append(score)
        
        return int(np.argmax(scores))
    
    def _align_single_pair(self, reference: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Align a single pair of images using phase correlation
        """
        try:
            from skimage.registration import phase_cross_correlation
            
            # Normalize images for better correlation
            ref_norm = (reference - reference.mean()) / (reference.std() + 1e-8)
            mov_norm = (moving - moving.mean()) / (moving.std() + 1e-8)
            
            # Compute phase correlation
            result = phase_cross_correlation(ref_norm, mov_norm, upsample_factor=10)
            if isinstance(result, tuple) and len(result) >= 2:
                shift, error = result[0], result[1] if len(result) > 1 else 0
            else:
                shift = result
                error = 0
            
            # Apply shift to moving image
            shift_y, shift_x = shift
            
            # Use scipy's shift function for sub-pixel accuracy
            aligned_img = ndimage.shift(moving, shift, order=1, prefilter=False)
            
            transform_info = {
                'method': 'phase_correlation',
                'shift': shift.tolist() if hasattr(shift, 'tolist') else shift,
                'error': float(error) if hasattr(error, 'item') else error,
                'success': True
            }
            
            return aligned_img, transform_info
            
        except Exception as e:
            self.logger.warning(f"Phase correlation failed: {e}, using identity transform")
            return moving.copy(), {'method': 'identity', 'error': str(e), 'success': False}
    
    def _calculate_alignment_quality(self, reference: np.ndarray, aligned: np.ndarray) -> float:
        """
        Calculate alignment quality using SSIM
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Ensure same data range
            data_range = max(reference.max() - reference.min(),
                           aligned.max() - aligned.min())
            
            if data_range > 0:
                quality = ssim(reference, aligned, data_range=data_range)
                return max(0.0, quality)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Quality calculation failed: {e}")
            return 0.0
    
    def crop_to_original_content(self, aligned_images: List[np.ndarray], 
                               preparation_info: Dict) -> List[np.ndarray]:
        """
        Crop aligned images back to remove excess padding while keeping alignment
        """
        if not aligned_images or 'target_size' not in preparation_info:
            return aligned_images
        
        target_height, target_width = preparation_info['target_size']
        
        # Find the bounding box of actual content across all aligned images
        min_y, max_y = target_height, 0
        min_x, max_x = target_width, 0
        
        for img in aligned_images:
            # Find non-zero regions
            nonzero_y, nonzero_x = np.nonzero(img > 0)
            if len(nonzero_y) > 0:
                min_y = min(min_y, nonzero_y.min())
                max_y = max(max_y, nonzero_y.max())
                min_x = min(min_x, nonzero_x.min())
                max_x = max(max_x, nonzero_x.max())
        
        # Add some padding around content
        padding = 50
        min_y = max(0, min_y - padding)
        max_y = min(target_height, max_y + padding)
        min_x = max(0, min_x - padding)
        max_x = min(target_width, max_x + padding)
        
        # Crop all images to this bounding box
        cropped_images = []
        for img in aligned_images:
            cropped = img[min_y:max_y, min_x:max_x]
            cropped_images.append(cropped)
        
        return cropped_images


def test_centroid_alignment():
    """
    Test the centroid alignment functionality
    """
    import matplotlib.pyplot as plt
    from skimage import data, transform as tf
    
    # Create test images with different sizes and positions
    test_images = []
    
    # Base image
    base = data.camera()[100:300, 150:350]  # 200x200
    test_images.append(base.astype(np.float32))
    
    # Shifted and rotated versions
    for i in range(1, 4):
        # Apply random transform
        shift_y = np.random.randint(-30, 30)
        shift_x = np.random.randint(-30, 30)
        angle = np.random.uniform(-10, 10)
        
        # Transform
        transform_matrix = tf.AffineTransform(
            translation=(shift_x, shift_y),
            rotation=np.radians(angle)
        )
        
        transformed = tf.warp(base, transform_matrix.inverse, 
                            output_shape=(250, 280),  # Different size
                            preserve_range=True)
        test_images.append(transformed.astype(np.float32))
    
    print("Testing Centroid Alignment...")
    print(f"Original image sizes: {[img.shape for img in test_images]}")
    
    # Test alignment
    aligner = EnhancedImageAligner()
    aligned_images, alignment_info = aligner.align_image_stack(test_images)
    
    print(f"Aligned image sizes: {[img.shape for img in aligned_images]}")
    print(f"Target size: {alignment_info['preparation']['target_size']}")
    print(f"Reference index: {alignment_info['reference_idx']}")
    
    # Test cropping
    cropped_images = aligner.crop_to_original_content(aligned_images, alignment_info['preparation'])
    print(f"Cropped image sizes: {[img.shape for img in cropped_images]}")
    
    # Visualize results
    fig, axes = plt.subplots(3, len(test_images), figsize=(15, 10))
    
    for i in range(len(test_images)):
        # Original
        axes[0, i].imshow(test_images[i], cmap='gray')
        axes[0, i].set_title(f'Original {i}\n{test_images[i].shape}')
        axes[0, i].axis('off')
        
        # Aligned
        axes[1, i].imshow(aligned_images[i], cmap='gray')
        quality = alignment_info['quality_scores'].get(i, 0)
        axes[1, i].set_title(f'Aligned {i}\nQ: {quality:.3f}')
        axes[1, i].axis('off')
        
        # Cropped
        axes[2, i].imshow(cropped_images[i], cmap='gray')
        axes[2, i].set_title(f'Cropped {i}\n{cropped_images[i].shape}')
        axes[2, i].axis('off')
    
    plt.suptitle('Centroid-Based Alignment Test')
    plt.tight_layout()
    plt.savefig('centroid_alignment_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return aligned_images, alignment_info


if __name__ == "__main__":
    # Run test
    test_centroid_alignment()
