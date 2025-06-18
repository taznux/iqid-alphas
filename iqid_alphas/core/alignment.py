"""
Image Alignment Module

Simplified image alignment and registration functionality.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import warnings

try:
    from skimage import transform, measure # feature might not be needed if using phase_cross_correlation
    from skimage.registration import phase_cross_correlation # More modern way
    from scipy import ndimage
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False


class ImageAligner:
    """
    Simplified image alignment class.
    
    Provides basic image registration and alignment capabilities
    for iQID and H&E image pairs.
    """
    
    def __init__(self, method: str = 'phase_correlation'):
        """
        Initialize the image aligner.
        
        Parameters
        ----------
        method : str, optional
            Alignment method ('phase_correlation', 'feature_matching')
        """
        self.method = method
        self.transformation = None
        
    def align_images(self, fixed_image: np.ndarray, moving_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Align two images.
        
        Parameters
        ----------
        fixed_image : np.ndarray
            Reference image (fixed)
        moving_image : np.ndarray
            Image to be aligned (moving)
            
        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Aligned image and transformation parameters
        """
        if not HAS_IMAGING:
            raise ImportError("Imaging libraries not available")
            
        if self.method == 'phase_correlation':
            return self._phase_correlation_alignment(fixed_image, moving_image)
        else:
            return self._simple_translation_alignment(fixed_image, moving_image)
    
    def _phase_correlation_alignment(self, fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Phase correlation based alignment with improved shape handling."""
        try:
            # Convert to grayscale if needed
            if len(fixed.shape) > 2:
                fixed = np.mean(fixed, axis=2)
            if len(moving.shape) > 2:
                moving = np.mean(moving, axis=2)
            
            # Handle different image shapes by cropping to common size
            min_h = min(fixed.shape[0], moving.shape[0])
            min_w = min(fixed.shape[1], moving.shape[1])
            
            # Crop both images to same size from center
            fixed_h, fixed_w = fixed.shape
            moving_h, moving_w = moving.shape
            
            # Calculate crop regions
            fixed_start_h = (fixed_h - min_h) // 2
            fixed_start_w = (fixed_w - min_w) // 2
            moving_start_h = (moving_h - min_h) // 2
            moving_start_w = (moving_w - min_w) // 2
            
            fixed_cropped = fixed[fixed_start_h:fixed_start_h+min_h, fixed_start_w:fixed_start_w+min_w]
            moving_cropped = moving[moving_start_h:moving_start_h+min_h, moving_start_w:moving_start_w+min_w]
            
            # Normalize images for better correlation
            fixed_norm = (fixed_cropped - fixed_cropped.mean()) / (fixed_cropped.std() + 1e-8)
            moving_norm = (moving_cropped - moving_cropped.mean()) / (moving_cropped.std() + 1e-8)
            
            # Compute phase correlation
            result = phase_cross_correlation(fixed_norm, moving_norm, upsample_factor=10)
            if isinstance(result, tuple) and len(result) == 3:
                shift, error, diffphase = result
            else:
                shift = result
                error, diffphase = None, None
            
            # Adjust shift for original image size and crop offset
            shift_adjusted = [
                shift[0] + fixed_start_h - moving_start_h,
                shift[1] + fixed_start_w - moving_start_w
            ]
            
            # Apply transformation to original moving image
            aligned = ndimage.shift(moving, shift_adjusted)
            
            transformation = {
                'method': 'phase_correlation',
                'shift': shift.tolist(),
                'shift_adjusted': shift_adjusted,
                'error': float(error) if error is not None else None,
                'phasediff': float(diffphase) if diffphase is not None else None,
                'translation_x': float(shift_adjusted[1]),
                'translation_y': float(shift_adjusted[0]),
                'crop_info': {
                    'original_size': [fixed.shape, moving.shape],
                    'cropped_size': [min_h, min_w],
                    'crop_offsets': [[fixed_start_h, fixed_start_w], [moving_start_h, moving_start_w]]
                }
            }
            
            return aligned, transformation
            
        except Exception as e:
            print(f"Phase correlation failed: {e}, using simple alignment")
            return self._simple_translation_alignment(fixed, moving)
    
    def _simple_translation_alignment(self, fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simple center-based alignment."""
        # Calculate center-of-mass based alignment
        fixed_gray = np.mean(fixed, axis=2) if len(fixed.shape) > 2 else fixed
        moving_gray = np.mean(moving, axis=2) if len(moving.shape) > 2 else moving
        
        # Find centers of mass
        fixed_com = ndimage.center_of_mass(fixed_gray)
        moving_com = ndimage.center_of_mass(moving_gray)
        
        # Calculate shift
        shift = [fixed_com[0] - moving_com[0], fixed_com[1] - moving_com[1]]
        
        # Apply shift
        aligned = ndimage.shift(moving, shift)
        
        transformation = {
            'method': 'simple_translation',
            'shift': shift,
            'translation_x': float(shift[1]),
            'translation_y': float(shift[0])
        }
        
        return aligned, transformation
    
    def calculate_alignment_quality(self, fixed: np.ndarray, aligned: np.ndarray) -> Dict[str, float]:
        """
        Calculate alignment quality metrics.
        
        Parameters
        ----------
        fixed : np.ndarray
            Reference image
        aligned : np.ndarray
            Aligned image
            
        Returns
        -------
        Dict[str, float]
            Quality metrics
        """
        if not HAS_IMAGING:
            return {'correlation': 0.0, 'mse': float('inf')}
            
        # Convert to same format
        if len(fixed.shape) > 2:
            fixed = np.mean(fixed, axis=2)
        if len(aligned.shape) > 2:
            aligned = np.mean(aligned, axis=2)
        
        # Ensure same shape
        min_shape = [min(fixed.shape[i], aligned.shape[i]) for i in range(2)]
        fixed = fixed[:min_shape[0], :min_shape[1]]
        aligned = aligned[:min_shape[0], :min_shape[1]]
        
        # Calculate correlation
        correlation = np.corrcoef(fixed.flatten(), aligned.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Calculate MSE
        mse = np.mean((fixed - aligned) ** 2)
        
        return {
            'correlation': float(correlation),
            'mse': float(mse),
            'alignment_score': float(max(0, correlation))  # Normalized score
        }
    
    def align(self, fixed_image: np.ndarray, moving_image: np.ndarray) -> np.ndarray:
        """
        Simple wrapper for image alignment.
        
        Parameters
        ----------
        fixed_image : np.ndarray
            Reference image (fixed)
        moving_image : np.ndarray
            Image to be aligned (moving)
            
        Returns
        -------
        np.ndarray
            Aligned moving image
        """
        aligned_image, _ = self.align_images(fixed_image, moving_image)
        return aligned_image

    def align_image_stack(self, image_paths: list, output_dir: str, 
                         reference_index: int = 0) -> list:
        """
        Align a stack of images to a reference image.
        
        Parameters
        ----------
        image_paths : list
            List of paths to images to align
        output_dir : str
            Output directory for aligned images
        reference_index : int, optional
            Index of the reference image (default: 0 = first image)
            
        Returns
        -------
        list
            List of output file paths
        """
        import skimage.io
        from pathlib import Path
        
        if not HAS_IMAGING:
            raise ImportError("Imaging libraries not available")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if len(image_paths) == 0:
            return []
            
        # Load all images
        images = []
        image_names = []
        for path in image_paths:
            try:
                img = skimage.io.imread(path)
                images.append(img)
                image_names.append(Path(path).name)
            except Exception as e:
                print(f"Failed to load image {path}: {e}")
                continue
        
        if len(images) == 0:
            return []
            
        # Use specified image as reference
        if reference_index >= len(images):
            reference_index = 0
            
        reference_image = images[reference_index]
        reference_name = image_names[reference_index]
        
        output_files = []
        
        # Save reference image (copy as-is)
        ref_output_path = output_dir / f"aligned_{reference_name}"
        skimage.io.imsave(str(ref_output_path), reference_image)
        output_files.append(str(ref_output_path))
        
        # Align other images to reference
        for i, (img, name) in enumerate(zip(images, image_names)):
            if i == reference_index:
                continue  # Skip reference image
                
            try:
                # Align to reference
                aligned_img, transform_info = self.align_images(reference_image, img)
                
                # Generate aligned filename
                aligned_name = f"aligned_{name}"
                output_path = output_dir / aligned_name
                
                # Save aligned image
                skimage.io.imsave(str(output_path), aligned_img.astype(img.dtype))
                output_files.append(str(output_path))
                
                print(f"Aligned {name}: translation=({transform_info.get('translation_x', 0):.1f}, {transform_info.get('translation_y', 0):.1f})")
                
            except Exception as e:
                print(f"Failed to align {name}: {e}")
                # Save original image as fallback
                fallback_path = output_dir / f"fallback_{name}"
                skimage.io.imsave(str(fallback_path), img)
                output_files.append(str(fallback_path))
        
        return output_files

    def align_images_files(self, image_paths: list, output_dir: str) -> list:
        """
        Convenience method that matches the interface expected by evaluation script.
        
        Parameters
        ----------
        image_paths : list
            List of paths to images to align
        output_dir : str
            Output directory for aligned images
            
        Returns
        -------
        list
            List of output file paths
        """
        return self.align_image_stack(image_paths, output_dir)
    
    def _pad_images_for_alignment(self, images: list) -> Tuple[list, Dict[str, Any]]:
        """
        Pad all images to twice the area of the largest slice to ensure full coverage during alignment.
        
        Parameters
        ----------
        images : list
            List of numpy arrays (images)
            
        Returns
        -------
        Tuple[list, Dict[str, Any]]
            List of padded images and padding information
        """
        if not images:
            return [], {}
        
        # Find the largest dimensions
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)
        
        # Calculate target size (twice the area of largest slice)
        # Area = height * width, so twice the area means sqrt(2) times each dimension
        scale_factor = np.sqrt(2.0)
        target_height = int(max_height * scale_factor)
        target_width = int(max_width * scale_factor)
        
        # Make sure dimensions are even for easier processing
        target_height = target_height + (target_height % 2)
        target_width = target_width + (target_width % 2)
        
        padded_images = []
        padding_info = []
        
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            
            # Calculate padding needed
            pad_h = target_height - h
            pad_w = target_width - w
            
            # Distribute padding evenly on both sides
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # Apply padding with edge values to avoid artifacts
            if len(img.shape) == 2:
                padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                               mode='edge')
            else:
                padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                               mode='edge')
            
            padded_images.append(padded)
            padding_info.append({
                'original_shape': img.shape,
                'padded_shape': padded.shape,
                'padding': ((pad_top, pad_bottom), (pad_left, pad_right)),
                'crop_coords': (pad_top, pad_top + h, pad_left, pad_left + w)
            })
        
        return padded_images, {
            'target_size': (target_height, target_width),
            'scale_factor': scale_factor,
            'max_original_size': (max_height, max_width),
            'padding_info': padding_info
        }
    
    def _unpad_image(self, padded_image: np.ndarray, padding_info: Dict) -> np.ndarray:
        """
        Remove padding from an aligned image to restore original size.
        
        Parameters
        ----------
        padded_image : np.ndarray
            Padded and aligned image
        padding_info : Dict
            Padding information from _pad_images_for_alignment
            
        Returns
        -------
        np.ndarray
            Unpadded image with original dimensions
        """
        top, bottom, left, right = padding_info['crop_coords']
        return padded_image[top:bottom, left:right]
    
    def align_image_pair(image1: np.ndarray, image2: np.ndarray, method: str = 'phase_correlation') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Quick function to align two images.
        
        Parameters
        ----------
        image1 : np.ndarray
            Reference image
        image2 : np.ndarray
            Image to align
        method : str, optional
            Alignment method
            
        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Aligned image and transformation info
        """
        aligner = ImageAligner(method=method)
        return aligner.align_images(image1, image2)
    
    def align_image_stack_with_padding(self, image_paths: list, output_dir: str = None, 
                                      reference_index: int = None) -> Dict[str, Any]:
        """
        Align a stack of images with proper padding to ensure full coverage.
        
        Parameters
        ----------
        image_paths : list
            List of image file paths
        output_dir : str, optional
            Directory to save aligned images
        reference_index : int, optional
            Index of reference image (middle by default)
            
        Returns
        -------
        Dict[str, Any]
            Alignment results with padding information
        """
        if not HAS_IMAGING:
            raise ImportError("Imaging libraries not available")
            
        import skimage.io
        from pathlib import Path
        
        # Load images
        images = []
        for path in image_paths:
            try:
                img = skimage.io.imread(path)
                if img.ndim > 2:
                    img = img.squeeze()
                if img.ndim > 2:
                    img = np.mean(img, axis=2)
                images.append(img.astype(np.float32))
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                continue
        
        if len(images) < 2:
            return {'success': False, 'error': 'Not enough valid images to align'}
        
        # Pad all images for alignment
        padded_images, padding_metadata = self._pad_images_for_alignment(images)
        
        # Choose reference image (middle slice by default)
        if reference_index is None:
            reference_index = len(padded_images) // 2
        reference_image = padded_images[reference_index]
        
        # Align all images to reference
        aligned_padded = []
        transformations = []
        alignment_scores = []
        
        for i, img in enumerate(padded_images):
            if i == reference_index:
                aligned_padded.append(img)
                transformations.append({'method': 'reference', 'is_reference': True})
                alignment_scores.append(1.0)  # Perfect score for reference
            else:
                try:
                    aligned_img, transform_info = self.align_images(reference_image, img)
                    aligned_padded.append(aligned_img)
                    transformations.append(transform_info)
                    
                    # Calculate alignment quality (SSIM between aligned and reference)
                    from skimage.metrics import structural_similarity as ssim
                    score = ssim(reference_image, aligned_img, data_range=reference_image.max() - reference_image.min())
                    alignment_scores.append(float(score))
                    
                except Exception as e:
                    print(f"Warning: Alignment failed for image {i}: {e}")
                    aligned_padded.append(img)  # Use original if alignment fails
                    transformations.append({'method': 'failed', 'error': str(e)})
                    alignment_scores.append(0.0)
        
        # Unpad aligned images to original size
        aligned_images = []
        for i, aligned_img in enumerate(aligned_padded):
            unpadded = self._unpad_image(aligned_img, padding_metadata['padding_info'][i])
            aligned_images.append(unpadded)
        
        # Save aligned images if output directory specified
        output_paths = []
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, aligned_img in enumerate(aligned_images):
                original_path = Path(image_paths[i])
                output_path = output_dir / f"aligned_{original_path.name}"
                
                # Normalize to 0-255 for saving
                img_norm = ((aligned_img - aligned_img.min()) / 
                           (aligned_img.max() - aligned_img.min() + 1e-8) * 255).astype(np.uint8)
                skimage.io.imsave(str(output_path), img_norm)
                output_paths.append(str(output_path))
        
        return {
            'success': True,
            'aligned_images': aligned_images,
            'transformations': transformations,
            'alignment_scores': alignment_scores,
            'reference_index': reference_index,
            'padding_metadata': padding_metadata,
            'output_paths': output_paths,
            'mean_alignment_score': np.mean([s for s in alignment_scores if s > 0]),
            'num_successful_alignments': sum(1 for t in transformations if t.get('method') != 'failed')
        }
    
    def _pad_image_to_size(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Pad image to target size, centering the original image.
        
        Args:
            image: Input image array
            target_size: (height, width) target size
            
        Returns:
            Padded image centered in target size
        """
        target_h, target_w = target_size
        img_h, img_w = image.shape[:2]
        
        # Calculate padding amounts
        pad_h = max(0, target_h - img_h)
        pad_w = max(0, target_w - img_w)
        
        # Split padding evenly on both sides (center the image)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Handle multi-channel images
        if image.ndim == 2:
            padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        elif image.ndim == 3:
            padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")
        
        return padded

    def align_stack_with_padding(self, images: List[np.ndarray], padding_factor: float = 2.0) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Align a stack of images with padding to cover all areas.
        Uses the largest slice as reference and centers all slices to minimize movement.
        
        Args:
            images: List of image arrays to align
            padding_factor: Factor to multiply the largest dimension (default 2.0 for twice the area)
            
        Returns:
            Tuple of (aligned_images, transformations)
        """
        if not images:
            return [], []
        
        # Find the largest image by area to use as reference
        largest_idx = 0
        largest_area = 0
        for i, img in enumerate(images):
            area = img.shape[0] * img.shape[1]
            if area > largest_area:
                largest_area = area
                largest_idx = i
        
        # Find the maximum dimensions
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)
        
        # Calculate padded size (twice the area means sqrt(2) times each dimension)
        pad_height = int(max_height * np.sqrt(padding_factor))
        pad_width = int(max_width * np.sqrt(padding_factor))
        
        print(f"Using slice {largest_idx} as reference (largest area: {largest_area} pixels)")
        print(f"Padding all slices to {pad_height}x{pad_width} (factor: {padding_factor})")
        
        # Pad all images to the same size and center them
        padded_images = []
        for i, img in enumerate(images):
            padded_img = self._pad_image_to_size(img, (pad_height, pad_width))
            padded_images.append(padded_img)
        
        # Use the largest image as reference
        reference_img = padded_images[largest_idx]
        aligned_images = [None] * len(images)  # Preserve original order
        transformations = [None] * len(images)
        
        # Set reference image
        aligned_images[largest_idx] = reference_img
        transformations[largest_idx] = {
            'method': 'reference', 
            'shift': [0, 0], 
            'is_reference': True,
            'original_size': images[largest_idx].shape,
            'reference_index': largest_idx,
            'padded_size': (pad_height, pad_width)
        }
        
        # Align all other images to the reference
        for i, img in enumerate(padded_images):
            if i == largest_idx:
                continue  # Skip reference
                
            aligned_img, transform = self.align_images(reference_img, img)
            aligned_images[i] = aligned_img
            transform['is_reference'] = False
            transform['original_size'] = images[i].shape
            transform['reference_index'] = largest_idx
            transform['padded_size'] = (pad_height, pad_width)
            transformations[i] = transform
        
        return aligned_images, transformations

    def align_stack_simple(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Simple stack alignment using the largest slice as reference.
        
        Args:
            images: List of image arrays to align
            
        Returns:
            Tuple of (aligned_images, transformations)
        """
        if not images:
            return [], []
        
        # Find the largest image by area to use as reference
        largest_idx = 0
        largest_area = 0
        for i, img in enumerate(images):
            area = img.shape[0] * img.shape[1]
            if area > largest_area:
                largest_area = area
                largest_idx = i
        
        print(f"Using slice {largest_idx} as reference (largest area: {largest_area} pixels)")
        
        # Use the largest image as reference
        reference_img = images[largest_idx]
        aligned_images = [None] * len(images)  # Preserve original order
        transformations = [None] * len(images)
        
        # Set reference image
        aligned_images[largest_idx] = reference_img
        transformations[largest_idx] = {
            'method': 'reference', 
            'shift': [0, 0], 
            'is_reference': True,
            'original_size': images[largest_idx].shape,
            'reference_index': largest_idx
        }
        
        # Align all other images to the reference
        for i, img in enumerate(images):
            if i == largest_idx:
                continue  # Skip reference
                
            aligned_img, transform = self.align_images(reference_img, img)
            aligned_images[i] = aligned_img
            transform['is_reference'] = False
            transform['original_size'] = images[i].shape
            transform['reference_index'] = largest_idx
            transformations[i] = transform
        
        return aligned_images, transformations
