"""
Stack Alignment Processing

This module handles alignment of image stacks (multiple slices) using
different reference strategies and quality assessment.
"""

import logging
from typing import List, Optional
import numpy as np
import cv2

from .types import ReferenceStrategy, StackAlignmentResult
from .algorithms import FeatureBasedAligner, IntensityBasedAligner, PhaseCorrelationAligner


class StackAligner:
    """Handles alignment of image stacks with configurable reference strategies."""
    
    def __init__(self, 
                 reference_strategy: ReferenceStrategy = ReferenceStrategy.PREVIOUS_IMAGE,
                 rolling_window: int = 3):
        """
        Initialize stack aligner.
        
        Parameters
        ----------
        reference_strategy : ReferenceStrategy
            Strategy for selecting reference images
        rolling_window : int
            Window size for rolling average strategy
        """
        self.reference_strategy = reference_strategy
        self.rolling_window = rolling_window
        self.logger = logging.getLogger(__name__)
        
        # Initialize alignment algorithms
        self.feature_aligner = FeatureBasedAligner()
        self.intensity_aligner = IntensityBasedAligner()
        self.phase_aligner = PhaseCorrelationAligner()
    
    def align_stack(self, 
                   image_stack: np.ndarray,
                   reference_strategy: Optional[ReferenceStrategy] = None) -> StackAlignmentResult:
        """
        Align a complete image stack.
        
        Parameters
        ----------
        image_stack : np.ndarray
            Stack of images to align (shape: [n_images, height, width])
        reference_strategy : ReferenceStrategy, optional
            Strategy for selecting reference images
            
        Returns
        -------
        StackAlignmentResult
            Complete stack alignment result
        """
        if reference_strategy is None:
            reference_strategy = self.reference_strategy
        
        n_images = image_stack.shape[0]
        self.logger.info(f"Aligning stack of {n_images} images using {reference_strategy.value}")
        
        # Initialize results
        aligned_images = []
        transforms = []
        quality_metrics = {}
        
        # First image is typically unchanged
        aligned_images.append(image_stack[0].copy())
        transforms.append(np.eye(3))
        
        # Align each subsequent image
        for i in range(1, n_images):
            try:
                # Select reference image based on strategy
                reference = self._select_reference_image(
                    aligned_images, i, reference_strategy, image_stack
                )
                
                # Try different alignment methods and pick the best
                results = []
                
                # Feature-based alignment
                try:
                    result = self.feature_aligner.align(image_stack[i], reference)
                    results.append(result)
                except Exception as e:
                    self.logger.debug(f"Feature-based alignment failed for slice {i}: {e}")
                
                # Intensity-based alignment
                try:
                    result = self.intensity_aligner.align(image_stack[i], reference)
                    results.append(result)
                except Exception as e:
                    self.logger.debug(f"Intensity-based alignment failed for slice {i}: {e}")
                
                # Phase correlation alignment
                try:
                    result = self.phase_aligner.align(image_stack[i], reference)
                    results.append(result)
                except Exception as e:
                    self.logger.debug(f"Phase correlation alignment failed for slice {i}: {e}")
                
                # Select best result based on confidence
                if results:
                    best_result = max(results, key=lambda r: r.confidence)
                    
                    # Apply transformation to get aligned image
                    aligned_image = self._apply_transform(image_stack[i], best_result.transform_matrix)
                    
                    aligned_images.append(aligned_image)
                    transforms.append(best_result.transform_matrix)
                    
                    self.logger.debug(f"Aligned slice {i}: confidence={best_result.confidence:.3f}, method={best_result.method.value}")
                else:
                    # No alignment worked, use original
                    self.logger.warning(f"All alignment methods failed for slice {i}, using original")
                    aligned_images.append(image_stack[i].copy())
                    transforms.append(np.eye(3))
                
            except Exception as e:
                self.logger.error(f"Failed to align slice {i}: {e}")
                # Use identity transformation as fallback
                aligned_images.append(image_stack[i].copy())
                transforms.append(np.eye(3))
        
        # Calculate quality metrics
        quality_metrics = self._calculate_stack_quality_metrics(
            image_stack, aligned_images
        )
        
        return StackAlignmentResult(
            aligned_images=aligned_images,
            transforms=transforms,
            quality_metrics=quality_metrics,
            reference_strategy=reference_strategy,
            metadata={
                'n_images': n_images,
                'methods_used': ['feature_based', 'intensity_based', 'phase_correlation']
            }
        )
    
    def _select_reference_image(self,
                               aligned_images: List[np.ndarray],
                               current_index: int,
                               strategy: ReferenceStrategy,
                               original_stack: np.ndarray) -> np.ndarray:
        """Select reference image based on strategy."""
        if strategy == ReferenceStrategy.FIRST_IMAGE:
            return aligned_images[0]
        
        elif strategy == ReferenceStrategy.PREVIOUS_IMAGE:
            return aligned_images[current_index - 1]
        
        elif strategy == ReferenceStrategy.ROLLING_AVERAGE:
            start_idx = max(0, current_index - self.rolling_window)
            end_idx = current_index
            
            if end_idx > start_idx:
                # Calculate rolling average
                images_to_average = aligned_images[start_idx:end_idx]
                return np.mean(images_to_average, axis=0).astype(aligned_images[0].dtype)
            else:
                return aligned_images[0]
        
        elif strategy == ReferenceStrategy.TEMPLATE_MATCHING:
            # Find the image with highest average intensity as template
            if len(aligned_images) >= 3:
                intensities = [np.mean(img) for img in aligned_images]
                template_idx = np.argmax(intensities)
                return aligned_images[template_idx]
            else:
                return aligned_images[0]
        
        else:
            # Default to previous image
            return aligned_images[current_index - 1]
    
    def _apply_transform(self, image: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Apply transformation matrix to image."""
        try:
            # Convert 3x3 to 2x3 for cv2.warpAffine
            if transform_matrix.shape == (3, 3):
                transform_2x3 = transform_matrix[:2, :]
            else:
                transform_2x3 = transform_matrix
            
            # Apply transformation
            aligned = cv2.warpAffine(
                image, 
                transform_2x3, 
                (image.shape[1], image.shape[0])
            )
            
            return aligned
            
        except Exception as e:
            self.logger.warning(f"Failed to apply transform: {e}")
            return image.copy()
    
    def _calculate_stack_quality_metrics(self,
                                        original_stack: np.ndarray,
                                        aligned_stack: List[np.ndarray]) -> dict:
        """Calculate quality metrics for the aligned stack."""
        try:
            n_images = len(aligned_stack)
            
            # Calculate image-to-image similarities
            similarities = []
            for i in range(1, n_images):
                # Normalized cross-correlation between consecutive images
                corr = cv2.matchTemplate(
                    aligned_stack[i-1].astype(np.float32),
                    aligned_stack[i].astype(np.float32),
                    cv2.TM_CCOEFF_NORMED
                )
                similarities.append(np.max(corr))
            
            # Calculate overall metrics
            metrics = {
                'mean_similarity': np.mean(similarities) if similarities else 0.0,
                'std_similarity': np.std(similarities) if similarities else 0.0,
                'min_similarity': np.min(similarities) if similarities else 0.0,
                'max_similarity': np.max(similarities) if similarities else 0.0,
                'n_images': n_images
            }
            
            # Calculate stack sharpness (variance of Laplacian)
            sharpness_scores = []
            for img in aligned_stack:
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
                sharpness_scores.append(sharpness)
            
            metrics['mean_sharpness'] = np.mean(sharpness_scores)
            metrics['sharpness_consistency'] = 1.0 - (np.std(sharpness_scores) / np.mean(sharpness_scores)) if np.mean(sharpness_scores) > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality metrics: {e}")
            return {
                'mean_similarity': 0.0,
                'std_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
                'mean_sharpness': 0.0,
                'sharpness_consistency': 0.0,
                'n_images': len(aligned_stack)
            }
