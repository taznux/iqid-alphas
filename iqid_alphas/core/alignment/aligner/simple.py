"""
Simple Image Aligner

Provides basic image alignment functionality with simplified methods.
Extracted from the original ImageAligner class.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

try:
    from skimage import transform, measure
    from skimage.registration import phase_cross_correlation
    from scipy import ndimage
    import cv2
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False

from ..types import AlignmentResult, AlignmentMethod


class SimpleAligner:
    """
    Simple image alignment class.
    
    Provides basic image registration and alignment capabilities
    for image pairs using simple methods like phase correlation
    and center-of-mass alignment.
    """
    
    def __init__(self, method: str = 'phase_correlation'):
        """
        Initialize the simple aligner.
        
        Parameters
        ----------
        method : str, optional
            Alignment method ('phase_correlation', 'center_of_mass')
        """
        self.method = method
        self.transformation = None
        
    def align_images(self, 
                    fixed_image: np.ndarray, 
                    moving_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            raise ImportError("Required imaging libraries (skimage, scipy) not available")
            
        if self.method == 'phase_correlation':
            return self._phase_correlation_alignment(fixed_image, moving_image)
        else:
            return self._center_of_mass_alignment(fixed_image, moving_image)
    
    def _phase_correlation_alignment(self, 
                                   fixed: np.ndarray, 
                                   moving: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Phase correlation based alignment with improved shape handling."""
        try:
            # Convert to grayscale if needed
            fixed_gray = self._ensure_grayscale(fixed)
            moving_gray = self._ensure_grayscale(moving)
            
            # Ensure same shape
            min_shape = (min(fixed_gray.shape[0], moving_gray.shape[0]),
                        min(fixed_gray.shape[1], moving_gray.shape[1]))
            
            fixed_cropped = fixed_gray[:min_shape[0], :min_shape[1]]
            moving_cropped = moving_gray[:min_shape[0], :min_shape[1]]
            
            # Calculate phase correlation
            shift, error, _ = phase_cross_correlation(fixed_cropped, moving_cropped)
            
            # Apply shift to original moving image
            aligned = ndimage.shift(moving, shift)
            
            transformation = {
                'method': 'phase_correlation',
                'shift': shift.tolist(),
                'translation_x': float(shift[1]),
                'translation_y': float(shift[0]),
                'error': float(error)
            }
            
            return aligned, transformation
            
        except Exception as e:
            # Fallback to center of mass alignment
            return self._center_of_mass_alignment(fixed, moving)
    
    def _center_of_mass_alignment(self, 
                                 fixed: np.ndarray, 
                                 moving: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Center-of-mass based alignment."""
        # Calculate center-of-mass based alignment
        fixed_gray = self._ensure_grayscale(fixed)
        moving_gray = self._ensure_grayscale(moving)
        
        # Find centers of mass
        fixed_com = ndimage.center_of_mass(fixed_gray)
        moving_com = ndimage.center_of_mass(moving_gray)
        
        # Calculate shift
        shift = [fixed_com[0] - moving_com[0], fixed_com[1] - moving_com[1]]
        
        # Apply shift
        aligned = ndimage.shift(moving, shift)
        
        transformation = {
            'method': 'center_of_mass',
            'shift': shift,
            'translation_x': float(shift[1]),
            'translation_y': float(shift[0])
        }
        
        return aligned, transformation
    
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image
    
    def calculate_alignment_quality(self, 
                                  fixed: np.ndarray, 
                                  aligned: np.ndarray) -> Dict[str, float]:
        """Calculate alignment quality metrics."""
        try:
            # Ensure same shape
            min_shape = (min(fixed.shape[0], aligned.shape[0]),
                        min(fixed.shape[1], aligned.shape[1]))
            
            fixed_crop = fixed[:min_shape[0], :min_shape[1]]
            aligned_crop = aligned[:min_shape[0], :min_shape[1]]
            
            # Convert to grayscale
            fixed_gray = self._ensure_grayscale(fixed_crop)
            aligned_gray = self._ensure_grayscale(aligned_crop)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(fixed_gray.flatten(), aligned_gray.flatten())[0, 1]
            
            # Calculate normalized cross-correlation
            ncc = cv2.matchTemplate(
                fixed_gray.astype(np.float32), 
                aligned_gray.astype(np.float32), 
                cv2.TM_CCOEFF_NORMED
            )
            max_ncc = np.max(ncc)
            
            # Calculate mean squared error
            mse = np.mean((fixed_gray - aligned_gray) ** 2)
            
            return {
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'normalized_cross_correlation': float(max_ncc),
                'mean_squared_error': float(mse)
            }
            
        except Exception as e:
            return {
                'correlation': 0.0,
                'normalized_cross_correlation': 0.0,
                'mean_squared_error': float('inf')
            }
    
    def align_slice(self, 
                   moving_slice: np.ndarray, 
                   reference_slice: np.ndarray,
                   method: Optional[AlignmentMethod] = None) -> AlignmentResult:
        """
        Align a single slice to a reference slice.
        
        Parameters
        ----------
        moving_slice : np.ndarray
            Slice to be aligned
        reference_slice : np.ndarray
            Reference slice
        method : AlignmentMethod, optional
            Alignment method to use
            
        Returns
        -------
        AlignmentResult
            Result containing aligned slice and transformation
        """
        try:
            aligned_image, transform_info = self.align_images(reference_slice, moving_slice)
            
            # Create transformation matrix
            if 'shift' in transform_info:
                shift = transform_info['shift']
                transform_matrix = np.array([
                    [1, 0, shift[1]],
                    [0, 1, shift[0]],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                transform_matrix = np.eye(3, dtype=np.float32)
            
            # Calculate quality metrics
            quality_metrics = self.calculate_alignment_quality(reference_slice, aligned_image)
            confidence = max(0.0, min(1.0, quality_metrics.get('correlation', 0.0)))
            
            return AlignmentResult(
                aligned_slice=aligned_image,
                transform_matrix=transform_matrix,
                confidence=confidence,
                method=AlignmentMethod.PHASE_CORRELATION if self.method == 'phase_correlation' else AlignmentMethod.FEATURE_BASED,
                processing_time=0.0,  # Not tracking time in simple aligner
                metadata=transform_info
            )
            
        except Exception as e:
            # Return identity transformation on failure
            return AlignmentResult(
                aligned_slice=moving_slice.copy(),
                transform_matrix=np.eye(3, dtype=np.float32),
                confidence=0.0,
                method=AlignmentMethod.PHASE_CORRELATION,
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
    def align_stack(self, 
                   stack: List[np.ndarray], 
                   reference_index: int = 0) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Align a stack of images to a reference image.
        
        Parameters
        ----------
        stack : List[np.ndarray]
            List of images to align
        reference_index : int
            Index of the reference image in the stack
            
        Returns
        -------
        Tuple[List[np.ndarray], List[Dict]]
            Aligned images and transformation info for each
        """
        if not stack:
            return [], []
            
        reference = stack[reference_index]
        aligned_stack = []
        transform_info = []
        
        for i, image in enumerate(stack):
            if i == reference_index:
                aligned_stack.append(image.copy())
                transform_info.append({'method': 'reference', 'shift': [0, 0]})
            else:
                aligned_image, transform = self.align_images(reference, image)
                aligned_stack.append(aligned_image)
                transform_info.append(transform)
        
        return aligned_stack, transform_info
