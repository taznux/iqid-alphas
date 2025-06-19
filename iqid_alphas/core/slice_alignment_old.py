"""
Enhanced Slice Alignment Module

This module provides robust, multi-method slice alignment functionality for iQID image stacks.
Migrated and modernized from legacy align.py with improved architecture, error handling, and validation.

Key Features:
- Multi-method alignment: coarse rotation, feature-based registration, intensity-based alignment
- Stack processing with configurable reference strategies
- Quality metrics and validation
- Robust error handling and fallback mechanisms
- Integration with reverse mapping evaluation framework
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import cv2
from skimage import transform, filters, exposure, measure
from scipy import optimize
import matplotlib.pyplot as plt

# Internal imports
from ..utils.io_utils import natural_sort, load_image
from ..utils.math_utils import decompose_affine_matrix
from .mapping import ReverseMapper, AlignmentTransform


class AlignmentMethod(Enum):
    """Available alignment methods."""
    COARSE_ROTATION = "coarse_rotation"
    FEATURE_BASED = "feature_based" 
    INTENSITY_BASED = "intensity_based"
    PHASE_CORRELATION = "phase_correlation"
    HYBRID = "hybrid"


class ReferenceStrategy(Enum):
    """Reference image selection strategies."""
    FIRST_IMAGE = "first_image"           # Align all to first image
    PREVIOUS_IMAGE = "previous_image"     # Align each to previous
    ROLLING_AVERAGE = "rolling_average"   # Align to average of previous N images
    TEMPLATE_MATCHING = "template_matching"  # Find best template and align to it


@dataclass
class AlignmentResult:
    """Result of slice alignment operation."""
    aligned_image: np.ndarray
    transform_matrix: np.ndarray
    confidence_score: float
    method_used: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class StackAlignmentResult:
    """Result of stack alignment operation."""
    aligned_stack: np.ndarray
    transform_matrices: np.ndarray
    confidence_scores: np.ndarray
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedAligner:
    """
    Enhanced slice alignment with multiple methods and robust error handling.
    
    This class provides comprehensive slice alignment functionality including:
    - Multiple alignment algorithms (rotation, feature-based, intensity-based)
    - Configurable reference strategies  
    - Quality assessment and validation
    - Integration with reverse mapping evaluation
    """
    
    def __init__(self,
                 default_method: AlignmentMethod = AlignmentMethod.HYBRID,
                 reference_strategy: ReferenceStrategy = ReferenceStrategy.ROLLING_AVERAGE,
                 rotation_range: Tuple[float, float] = (-180, 180),
                 rotation_step: float = 2.0,
                 feature_detector: str = 'ORB',
                 max_features: int = 5000,
                 match_threshold: float = 0.75,
                 gaussian_sigma: float = 1.0,
                 rolling_window: int = 3,
                 quality_threshold: float = 0.5,
                 enable_validation: bool = True):
        """
        Initialize the enhanced aligner.
        
        Parameters
        ----------
        default_method : AlignmentMethod
            Default alignment method to use
        reference_strategy : ReferenceStrategy  
            Strategy for selecting reference images
        rotation_range : Tuple[float, float]
            Range of rotation angles to search (degrees)
        rotation_step : float
            Step size for rotation search (degrees)
        feature_detector : str
            Feature detector type ('ORB', 'SIFT', 'SURF')
        max_features : int
            Maximum number of features to detect
        match_threshold : float
            Threshold for feature matching quality
        gaussian_sigma : float
            Sigma for Gaussian smoothing in intensity-based alignment
        rolling_window : int
            Window size for rolling average reference strategy
        quality_threshold : float
            Minimum quality threshold for accepting alignment
        enable_validation : bool
            Whether to enable alignment validation
        """
        self.default_method = default_method
        self.reference_strategy = reference_strategy
        self.rotation_range = rotation_range
        self.rotation_step = rotation_step
        self.feature_detector = feature_detector
        self.max_features = max_features
        self.match_threshold = match_threshold
        self.gaussian_sigma = gaussian_sigma
        self.rolling_window = rolling_window
        self.quality_threshold = quality_threshold
        self.enable_validation = enable_validation
        
        # Initialize components
        self.logger = logging.getLogger(__name__)
        self._init_feature_detector()
        self.reverse_mapper = ReverseMapper() if enable_validation else None
        
        # Statistics tracking
        self.stats = {
            'total_alignments': 0,
            'successful_alignments': 0,
            'method_usage': {method.value: 0 for method in AlignmentMethod},
            'average_confidence': 0.0,
            'processing_times': []
        }
    
    def _init_feature_detector(self):
        """Initialize the feature detector based on the specified type."""
        if self.feature_detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.feature_detector == 'SIFT':
            try:
                self.detector = cv2.SIFT_create(nfeatures=self.max_features)
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            except AttributeError:
                self.logger.warning("SIFT not available, falling back to ORB")
                self.detector = cv2.ORB_create(nfeatures=self.max_features)
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.feature_detector == 'SURF':
            try:
                self.detector = cv2.xfeatures2d.SURF_create()
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            except AttributeError:
                self.logger.warning("SURF not available, falling back to ORB")
                self.detector = cv2.ORB_create(nfeatures=self.max_features)
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.logger.warning(f"Unknown detector {self.feature_detector}, using ORB")
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def align_slice(self, 
                   moving_image: np.ndarray,
                   reference_image: np.ndarray,
                   method: Optional[AlignmentMethod] = None) -> AlignmentResult:
        """
        Align a single slice to a reference image.
        
        Parameters
        ----------
        moving_image : np.ndarray
            Image to be aligned
        reference_image : np.ndarray
            Reference image to align to
        method : AlignmentMethod, optional
            Alignment method to use (defaults to configured method)
            
        Returns
        -------
        AlignmentResult
            Alignment result with aligned image and metadata
        """
        import time
        start_time = time.time()
        
        if method is None:
            method = self.default_method
        
        self.logger.debug(f"Aligning slice using method: {method.value}")
        
        try:
            # Convert to grayscale if needed
            moving_gray = self._ensure_grayscale(moving_image)
            reference_gray = self._ensure_grayscale(reference_image)
            
            # Perform alignment based on method
            if method == AlignmentMethod.COARSE_ROTATION:
                result = self._align_coarse_rotation(moving_gray, reference_gray)
            elif method == AlignmentMethod.FEATURE_BASED:
                result = self._align_feature_based(moving_gray, reference_gray)
            elif method == AlignmentMethod.INTENSITY_BASED:
                result = self._align_intensity_based(moving_gray, reference_gray)
            elif method == AlignmentMethod.PHASE_CORRELATION:
                result = self._align_phase_correlation(moving_gray, reference_gray)
            elif method == AlignmentMethod.HYBRID:
                result = self._align_hybrid(moving_gray, reference_gray)
            else:
                raise ValueError(f"Unknown alignment method: {method}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(method, result.confidence_score, processing_time)
            
            # Add processing time to result
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Alignment failed: {e}")
            # Return identity alignment as fallback
            return AlignmentResult(
                aligned_image=moving_image.copy(),
                transform_matrix=np.eye(3),
                confidence_score=0.0,
                method_used=f"{method.value}_failed",
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()
    
    def _align_coarse_rotation(self, moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        """Align using coarse rotation method (migrated from legacy coarse_rotation)."""
        # Smooth images for better rotation matching
        if self.gaussian_sigma > 0:
            moving_smooth = cv2.GaussianBlur(moving, (5, 5), self.gaussian_sigma)
            reference_smooth = cv2.GaussianBlur(reference, (5, 5), self.gaussian_sigma)
        else:
            moving_smooth = moving
            reference_smooth = reference
        
        # Generate rotation angles to test
        angles = np.arange(self.rotation_range[0], self.rotation_range[1], self.rotation_step)
        best_angle = 0.0
        best_score = float('inf')
        
        # Test each rotation angle
        for angle in angles:
            # Rotate moving image
            rotated = transform.rotate(moving_smooth, angle, preserve_range=True, order=1)
            
            # Calculate similarity (using SSD as in legacy code)
            ssd = np.sum((rotated - reference_smooth) ** 2) / reference_smooth.size
            
            if ssd < best_score:
                best_score = ssd
                best_angle = angle
        
        # Apply best rotation to original image
        aligned_image = transform.rotate(moving, best_angle, preserve_range=True, order=0)
        
        # Create transformation matrix
        transform_matrix = self._create_rotation_matrix(best_angle, moving.shape)
        
        # Calculate confidence score (inverse of normalized SSD)
        max_possible_ssd = np.var(reference) * 2  # Rough estimate
        confidence_score = max(0.0, 1.0 - (best_score / max_possible_ssd))
        
        return AlignmentResult(
            aligned_image=aligned_image,
            transform_matrix=transform_matrix,
            confidence_score=confidence_score,
            method_used=AlignmentMethod.COARSE_ROTATION.value,
            processing_time=0.0,  # Will be set by caller
            metadata={'rotation_angle': best_angle, 'ssd_score': best_score}
        )
    
    def _align_feature_based(self, moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        """Align using feature-based matching."""
        # Detect features
        kp1, des1 = self.detector.detectAndCompute(reference, None)
        kp2, des2 = self.detector.detectAndCompute(moving, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            # Insufficient features, return identity
            return AlignmentResult(
                aligned_image=moving.copy(),
                transform_matrix=np.eye(3),
                confidence_score=0.0,
                method_used=AlignmentMethod.FEATURE_BASED.value,
                processing_time=0.0,
                metadata={'error': 'insufficient_features', 'features_ref': len(kp1), 'features_mov': len(kp2)}
            )
        
        # Match features
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            # Insufficient matches
            return AlignmentResult(
                aligned_image=moving.copy(),
                transform_matrix=np.eye(3),
                confidence_score=0.0,
                method_used=AlignmentMethod.FEATURE_BASED.value,
                processing_time=0.0,
                metadata={'error': 'insufficient_matches', 'matches': len(matches)}
            )
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate transformation
        try:
            transform_matrix_2x3, inliers = cv2.estimateAffine2D(
                dst_pts, src_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99
            )
            
            if transform_matrix_2x3 is None:
                # Transformation estimation failed
                return AlignmentResult(
                    aligned_image=moving.copy(),
                    transform_matrix=np.eye(3),
                    confidence_score=0.0,
                    method_used=AlignmentMethod.FEATURE_BASED.value,
                    processing_time=0.0,
                    metadata={'error': 'transform_estimation_failed'}
                )
            
            # Convert to 3x3 matrix
            transform_matrix = np.vstack([transform_matrix_2x3, [0, 0, 1]])
            
            # Apply transformation
            aligned_image = cv2.warpAffine(moving, transform_matrix_2x3, moving.shape[::-1])
            
            # Calculate confidence
            num_inliers = np.sum(inliers) if inliers is not None else 0
            inlier_ratio = num_inliers / len(matches)
            confidence_score = inlier_ratio * min(1.0, len(matches) / 50.0)
            
            return AlignmentResult(
                aligned_image=aligned_image,
                transform_matrix=transform_matrix,
                confidence_score=confidence_score,
                method_used=AlignmentMethod.FEATURE_BASED.value,
                processing_time=0.0,
                metadata={
                    'total_matches': len(matches),
                    'inliers': num_inliers,
                    'inlier_ratio': inlier_ratio
                }
            )
            
        except Exception as e:
            return AlignmentResult(
                aligned_image=moving.copy(),
                transform_matrix=np.eye(3),
                confidence_score=0.0,
                method_used=AlignmentMethod.FEATURE_BASED.value,
                processing_time=0.0,
                metadata={'error': f'transformation_error: {str(e)}'}
            )
    
    def _align_intensity_based(self, moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        """Align using intensity-based optimization."""
        try:
            # Simple translation-only alignment using normalized cross-correlation
            result = cv2.matchTemplate(reference, moving, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # Calculate translation
            h, w = moving.shape
            dx = max_loc[0] - (reference.shape[1] - w) // 2
            dy = max_loc[1] - (reference.shape[0] - h) // 2
            
            # Create transformation matrix
            transform_matrix = np.array([
                [1, 0, dx],
                [0, 1, dy], 
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Apply transformation
            aligned_image = cv2.warpAffine(moving, transform_matrix[:2], moving.shape[::-1])
            
            return AlignmentResult(
                aligned_image=aligned_image,
                transform_matrix=transform_matrix,
                confidence_score=max_val,
                method_used=AlignmentMethod.INTENSITY_BASED.value,
                processing_time=0.0,
                metadata={'translation': (dx, dy), 'correlation_score': max_val}
            )
            
        except Exception as e:
            return AlignmentResult(
                aligned_image=moving.copy(),
                transform_matrix=np.eye(3),
                confidence_score=0.0,
                method_used=AlignmentMethod.INTENSITY_BASED.value,
                processing_time=0.0,
                metadata={'error': f'intensity_alignment_error: {str(e)}'}
            )
    
    def _align_phase_correlation(self, moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        """Align using phase correlation (if available)."""
        try:
            from skimage.registration import phase_cross_correlation
            
            # Calculate phase correlation
            shift, error, diffphase = phase_cross_correlation(reference, moving)
            
            # Create transformation matrix (translation only)
            transform_matrix = np.array([
                [1, 0, shift[1]],  # x shift
                [0, 1, shift[0]],  # y shift
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Apply transformation
            aligned_image = cv2.warpAffine(moving, transform_matrix[:2], moving.shape[::-1])
            
            # Confidence is inverse of error
            confidence_score = max(0.0, 1.0 - error)
            
            return AlignmentResult(
                aligned_image=aligned_image,
                transform_matrix=transform_matrix,
                confidence_score=confidence_score,
                method_used=AlignmentMethod.PHASE_CORRELATION.value,
                processing_time=0.0,
                metadata={'shift': shift.tolist(), 'error': error}
            )
            
        except ImportError:
            self.logger.warning("scikit-image phase correlation not available, falling back to intensity-based")
            return self._align_intensity_based(moving, reference)
        except Exception as e:
            return AlignmentResult(
                aligned_image=moving.copy(),
                transform_matrix=np.eye(3),
                confidence_score=0.0,
                method_used=AlignmentMethod.PHASE_CORRELATION.value,
                processing_time=0.0,
                metadata={'error': f'phase_correlation_error: {str(e)}'}
            )
    
    def _align_hybrid(self, moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        """Hybrid alignment using multiple methods and selecting the best result."""
        methods_to_try = [
            AlignmentMethod.FEATURE_BASED,
            AlignmentMethod.INTENSITY_BASED,
            AlignmentMethod.COARSE_ROTATION
        ]
        
        best_result = None
        best_confidence = -1.0
        
        for method in methods_to_try:
            try:
                if method == AlignmentMethod.FEATURE_BASED:
                    result = self._align_feature_based(moving, reference)
                elif method == AlignmentMethod.INTENSITY_BASED:
                    result = self._align_intensity_based(moving, reference)
                elif method == AlignmentMethod.COARSE_ROTATION:
                    result = self._align_coarse_rotation(moving, reference)
                else:
                    continue
                
                # Select result with highest confidence
                if result.confidence_score > best_confidence:
                    best_confidence = result.confidence_score
                    best_result = result
                    
            except Exception as e:
                self.logger.warning(f"Method {method.value} failed in hybrid alignment: {e}")
                continue
        
        if best_result is None:
            # All methods failed
            return AlignmentResult(
                aligned_image=moving.copy(),
                transform_matrix=np.eye(3),
                confidence_score=0.0,
                method_used=AlignmentMethod.HYBRID.value,
                processing_time=0.0,
                metadata={'error': 'all_methods_failed'}
            )
        
        # Update method used to indicate hybrid selection
        best_result.method_used = f"{AlignmentMethod.HYBRID.value}_{best_result.method_used}"
        return best_result
    
    def _create_rotation_matrix(self, angle: float, image_shape: Tuple[int, int]) -> np.ndarray:
        """Create a 3x3 rotation transformation matrix."""
        center_x = image_shape[1] / 2
        center_y = image_shape[0] / 2
        
        # Convert to radians
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation around center
        transform_matrix = np.array([
            [cos_a, -sin_a, center_x - center_x*cos_a + center_y*sin_a],
            [sin_a, cos_a, center_y - center_x*sin_a - center_y*cos_a],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return transform_matrix
    
    def _update_stats(self, method: AlignmentMethod, confidence: float, processing_time: float):
        """Update internal statistics."""
        self.stats['total_alignments'] += 1
        if confidence > self.quality_threshold:
            self.stats['successful_alignments'] += 1
        
        self.stats['method_usage'][method.value] += 1
        
        # Update running average of confidence
        n = self.stats['total_alignments']
        self.stats['average_confidence'] = (
            (self.stats['average_confidence'] * (n-1) + confidence) / n
        )
        
        self.stats['processing_times'].append(processing_time)
    
    def get_alignment_stats(self) -> Dict[str, Any]:
        """Get alignment statistics."""
        stats = self.stats.copy()
        if self.stats['processing_times']:
            stats['average_processing_time'] = np.mean(self.stats['processing_times'])
            stats['median_processing_time'] = np.median(self.stats['processing_times'])
        return stats
    
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
        aligned_stack = np.zeros_like(image_stack)
        transform_matrices = np.zeros((n_images, 3, 3))
        confidence_scores = np.zeros(n_images)
        
        # First image is typically unchanged
        aligned_stack[0] = image_stack[0]
        transform_matrices[0] = np.eye(3)
        confidence_scores[0] = 1.0
        
        # Align each subsequent image
        for i in range(1, n_images):
            try:
                # Select reference image based on strategy
                reference = self._select_reference_image(
                    aligned_stack, i, reference_strategy
                )
                
                # Align current image
                result = self.align_slice(image_stack[i], reference)
                
                # Store results
                aligned_stack[i] = result.aligned_image
                transform_matrices[i] = result.transform_matrix
                confidence_scores[i] = result.confidence_score
                
                self.logger.debug(f"Aligned slice {i}: confidence={result.confidence_score:.3f}, method={result.method_used}")
                
            except Exception as e:
                self.logger.error(f"Failed to align slice {i}: {e}")
                # Use identity transformation as fallback
                aligned_stack[i] = image_stack[i]
                transform_matrices[i] = np.eye(3)
                confidence_scores[i] = 0.0
        
        # Calculate quality metrics
        quality_metrics = self._calculate_stack_quality_metrics(
            image_stack, aligned_stack, confidence_scores
        )
        
        return StackAlignmentResult(
            aligned_stack=aligned_stack,
            transform_matrices=transform_matrices,
            confidence_scores=confidence_scores,
            quality_metrics=quality_metrics,
            metadata={
                'reference_strategy': reference_strategy.value,
                'alignment_method': self.default_method.value,
                'n_images': n_images
            }
        )
    
    def _select_reference_image(self,
                               aligned_stack: np.ndarray,
                               current_index: int,
                               strategy: ReferenceStrategy) -> np.ndarray:
        """Select reference image based on strategy."""
        if strategy == ReferenceStrategy.FIRST_IMAGE:
            return aligned_stack[0]
        
        elif strategy == ReferenceStrategy.PREVIOUS_IMAGE:
            return aligned_stack[current_index - 1]
        
        elif strategy == ReferenceStrategy.ROLLING_AVERAGE:
            start_idx = max(0, current_index - self.rolling_window)
            end_idx = current_index
            return np.mean(aligned_stack[start_idx:end_idx], axis=0)
        
        elif strategy == ReferenceStrategy.TEMPLATE_MATCHING:
            # Find the image with highest average intensity (likely best quality)
            # This is a simple heuristic - could be made more sophisticated
            if current_index < 3:
                return aligned_stack[0]  # Use first image for early slices
            
            window_start = max(0, current_index - 5)
            window_images = aligned_stack[window_start:current_index]
            mean_intensities = np.mean(window_images, axis=(1, 2))
            best_idx = window_start + np.argmax(mean_intensities)
            return aligned_stack[best_idx]
        
        else:
            raise ValueError(f"Unknown reference strategy: {strategy}")
    
    def _calculate_stack_quality_metrics(self,
                                        original_stack: np.ndarray,
                                        aligned_stack: np.ndarray,
                                        confidence_scores: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for the aligned stack."""
        metrics = {}
        
        # Basic confidence statistics
        metrics['mean_confidence'] = np.mean(confidence_scores)
        metrics['median_confidence'] = np.median(confidence_scores)
        metrics['min_confidence'] = np.min(confidence_scores)
        metrics['std_confidence'] = np.std(confidence_scores)
        
        # Success rate (above threshold)
        success_rate = np.sum(confidence_scores >= self.quality_threshold) / len(confidence_scores)
        metrics['success_rate'] = success_rate
        
        # Stack consistency metrics
        try:
            # Calculate inter-slice variation in aligned stack
            slice_means = np.mean(aligned_stack, axis=(1, 2))
            metrics['intensity_stability'] = 1.0 - (np.std(slice_means) / np.mean(slice_means))
            
            # Calculate alignment consistency (variation in transformations)
            # Look at translation components
            translations = np.array([[mat[0, 2], mat[1, 2]] for mat in self.stats.get('transform_matrices', [])])
            if len(translations) > 1:
                translation_distances = np.sqrt(np.sum(translations**2, axis=1))
                metrics['translation_consistency'] = 1.0 / (1.0 + np.std(translation_distances))
            else:
                metrics['translation_consistency'] = 1.0
                
        except Exception as e:
            self.logger.warning(f"Could not calculate some quality metrics: {e}")
            metrics['intensity_stability'] = 0.0
            metrics['translation_consistency'] = 0.0
        
        # Overall quality score
        weights = {
            'mean_confidence': 0.4,
            'success_rate': 0.3,
            'intensity_stability': 0.2,
            'translation_consistency': 0.1
        }
        
        overall_quality = sum(
            weights.get(key, 0) * value 
            for key, value in metrics.items() 
            if key in weights
        )
        metrics['overall_quality'] = overall_quality
        
        return metrics
    
    def align_directory(self,
                       input_dir: Union[str, Path],
                       output_dir: Union[str, Path],
                       file_pattern: str = "*.tif",
                       save_transformations: bool = True) -> Dict[str, Any]:
        """
        Align all images in a directory and save results.
        
        Parameters
        ----------
        input_dir : str or Path
            Directory containing images to align
        output_dir : str or Path  
            Directory to save aligned images
        file_pattern : str
            Pattern to match input files
        save_transformations : bool
            Whether to save transformation matrices
            
        Returns
        -------
        Dict with alignment results and statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        image_files = list(input_dir.glob(file_pattern))
        image_files = sorted(image_files, key=lambda x: natural_sort([str(x)])[0])
        
        if not image_files:
            raise ValueError(f"No images found matching pattern {file_pattern} in {input_dir}")
        
        self.logger.info(f"Found {len(image_files)} images to align")
        
        # Load images into stack
        first_image = load_image(image_files[0])
        image_stack = np.zeros((len(image_files), *first_image.shape), dtype=first_image.dtype)
        
        for i, file_path in enumerate(image_files):
            image_stack[i] = load_image(file_path)
        
        # Perform alignment
        result = self.align_stack(image_stack)
        
        # Save aligned images
        for i, file_path in enumerate(image_files):
            output_filename = output_dir / f"aligned_{file_path.stem}.tif"
            # Use appropriate save function based on image type
            if result.aligned_stack[i].dtype in [np.float32, np.float64]:
                # For float images, save as TIFF
                from skimage import io
                io.imsave(str(output_filename), result.aligned_stack[i].astype(np.float32))
            else:
                # For integer images
                cv2.imwrite(str(output_filename), result.aligned_stack[i])
        
        # Save transformation matrices if requested
        if save_transformations:
            transforms_file = output_dir / "transformation_matrices.npy"
            np.save(transforms_file, result.transform_matrices)
            
            # Also save as text for human readability
            transforms_txt = output_dir / "transformation_matrices.txt"
            with open(transforms_txt, 'w') as f:
                f.write("# Transformation matrices for aligned images\n")
                f.write("# Format: one 3x3 matrix per image, space-separated\n")
                for i, matrix in enumerate(result.transform_matrices):
                    f.write(f"# Image {i}: {image_files[i].name}\n")
                    for row in matrix:
                        f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
                    f.write("\n")
        
        # Return summary
        return {
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'n_images_processed': len(image_files),
            'quality_metrics': result.quality_metrics,
            'alignment_stats': self.get_alignment_stats(),
            'files_processed': [f.name for f in image_files]
        }
    
    def validate_alignment(self,
                          original_image: np.ndarray,
                          aligned_image: np.ndarray) -> Dict[str, float]:
        """
        Validate alignment quality using various metrics.
        
        Parameters
        ----------
        original_image : np.ndarray
            Original image before alignment
        aligned_image : np.ndarray
            Image after alignment
            
        Returns
        -------
        Dict with validation metrics
        """
        if not self.enable_validation:
            return {'validation_enabled': False}
        
        metrics = {}
        
        try:
            # Structural similarity
            from skimage.metrics import structural_similarity as ssim
            metrics['ssim'] = ssim(original_image, aligned_image)
        except ImportError:
            self.logger.warning("scikit-image not available for SSIM calculation")
            metrics['ssim'] = None
        
        # Mean squared error
        mse = np.mean((original_image.astype(np.float32) - aligned_image.astype(np.float32)) ** 2)
        metrics['mse'] = mse
        
        # Peak signal-to-noise ratio
        if mse > 0:
            max_pixel = np.max(original_image)
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            metrics['psnr'] = psnr
        else:
            metrics['psnr'] = float('inf')
        
        # Normalized cross-correlation
        original_norm = original_image - np.mean(original_image)
        aligned_norm = aligned_image - np.mean(aligned_image)
        
        if np.std(original_norm) > 0 and np.std(aligned_norm) > 0:
            ncc = np.sum(original_norm * aligned_norm) / (
                np.sqrt(np.sum(original_norm**2)) * np.sqrt(np.sum(aligned_norm**2))
            )
            metrics['normalized_cross_correlation'] = ncc
        else:
            metrics['normalized_cross_correlation'] = 0.0
        
        # Use reverse mapper for additional validation if available
        if self.reverse_mapper is not None:
            try:
                alignment_result = self.reverse_mapper.calculate_alignment_transform(
                    original_image, aligned_image
                )
                metrics['reverse_mapping_quality'] = alignment_result.match_quality
                metrics['reverse_mapping_matches'] = alignment_result.num_matches
            except Exception as e:
                self.logger.warning(f"Reverse mapping validation failed: {e}")
        
        return metrics


# Utility functions migrated from legacy code

def assemble_image_stack(image_dir: Union[str, Path],
                        file_pattern: str = "*.tif",
                        pad_to_max: bool = True) -> np.ndarray:
    """
    Assemble a stack of images from a directory.
    
    Migrated from legacy assemble_stack function with improvements.
    
    Parameters
    ----------
    image_dir : str or Path
        Directory containing images
    file_pattern : str
        Pattern to match image files
    pad_to_max : bool
        Whether to pad images to maximum dimensions
        
    Returns
    -------
    np.ndarray
        Assembled image stack
    """
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob(file_pattern))
    image_files = sorted(image_files, key=lambda x: natural_sort([str(x)])[0])
    
    if not image_files:
        raise ValueError(f"No images found matching {file_pattern} in {image_dir}")
    
    # Load first image to determine dimensions and dtype
    first_image = load_image(image_files[0])
    
    if pad_to_max:
        # Find maximum dimensions
        max_height, max_width = first_image.shape[:2]
        
        for file_path in image_files[1:]:
            img = load_image(file_path)
            max_height = max(max_height, img.shape[0])
            max_width = max(max_width, img.shape[1])
        
        # Create padded stack
        if len(first_image.shape) == 3:
            image_stack = np.zeros((len(image_files), max_height, max_width, first_image.shape[2]), 
                                 dtype=first_image.dtype)
        else:
            image_stack = np.zeros((len(image_files), max_height, max_width), dtype=first_image.dtype)
        
        # Load and pad each image
        for i, file_path in enumerate(image_files):
            img = load_image(file_path)
            
            # Calculate padding
            pad_height = max_height - img.shape[0]
            pad_width = max_width - img.shape[1]
            
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            if len(img.shape) == 3:
                padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                              mode='constant', constant_values=0)
            else:
                padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                              mode='constant', constant_values=0)
            
            image_stack[i] = padded
    else:
        # Simple stack without padding (all images must be same size)
        image_stack = np.array([load_image(f) for f in image_files])
    
    return image_stack


def calculate_ssd(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Sum of Squared Differences between two images.
    
    Migrated from legacy get_SSD function.
    """
    return np.sum((image1.astype(np.float32) - image2.astype(np.float32)) ** 2) / image1.size


def create_affine_transform(rotation: float = 0.0,
                           translation: Tuple[float, float] = (0.0, 0.0),
                           scale: Tuple[float, float] = (1.0, 1.0),
                           shear: float = 0.0,
                           center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Create an affine transformation matrix from parameters.
    
    Parameters
    ----------
    rotation : float
        Rotation angle in degrees
    translation : Tuple[float, float]
        Translation in (x, y)
    scale : Tuple[float, float]
        Scale factors in (x, y)
    shear : float
        Shear angle in degrees
    center : Tuple[float, float], optional
        Center of rotation/scaling
        
    Returns
    -------
    np.ndarray
        3x3 affine transformation matrix
    """
    if center is None:
        center = (0.0, 0.0)
    
    # Convert angles to radians
    rotation_rad = np.radians(rotation)
    shear_rad = np.radians(shear)
    
    # Create individual transformation matrices
    # Translation to origin
    T1 = np.array([
        [1, 0, -center[0]],
        [0, 1, -center[1]],
        [0, 0, 1]
    ])
    
    # Rotation
    cos_r = np.cos(rotation_rad)
    sin_r = np.sin(rotation_rad)
    R = np.array([
        [cos_r, -sin_r, 0],
        [sin_r, cos_r, 0],
        [0, 0, 1]
    ])
    
    # Scale
    S = np.array([
        [scale[0], 0, 0],
        [0, scale[1], 0],
        [0, 0, 1]
    ])
    
    # Shear
    Sh = np.array([
        [1, np.tan(shear_rad), 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Translation back and final translation
    T2 = np.array([
        [1, 0, center[0] + translation[0]],
        [0, 1, center[1] + translation[1]],
        [0, 0, 1]
    ])
    
    # Combine transformations: T2 * Sh * S * R * T1
    transform = T2 @ Sh @ S @ R @ T1
    
    return transform.astype(np.float32)
