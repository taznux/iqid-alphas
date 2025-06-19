"""
Multi-Modal Coregistration Module

This module provides comprehensive multi-modal image registration functionality
for aligning different imaging modalities (e.g., iQID and H&E images).

Key Features:
- Mutual information-based registration
- Feature-based multi-modal alignment  
- Intensity-based registration with preprocessing
- Quality assessment metrics
- Robust error handling and validation
- Integration with single-modal alignment framework
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import cv2
from skimage import filters, exposure, feature, measure, transform
from scipy import optimize, ndimage
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Internal imports
from ..utils.io_utils import natural_sort, load_image
from ..utils.math_utils import decompose_affine
from .alignment import EnhancedAligner, AlignmentMethod, AlignmentResult

Key Features:
- Mutual information-based registration
- Feature-based multi-modal alignment  
- Intensity-based registration with preprocessing
- Quality assessment metrics
- Robust error handling and validation
- Integration with single-modal alignment framework
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import cv2
from skimage import filters, exposure, feature, measure, transform
from scipy import optimize, ndimage
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Internal imports
from ..utils.io_utils import natural_sort
from ..utils.math_utils import decompose_affine
from .alignment import UnifiedAligner, AlignerType, AlignmentMethod, AlignmentResult

# Image loading function (temporary until we add it to io_utils)
def load_image(file_path: Union[str, Path]) -> np.ndarray:
    """Load an image from file."""
    import cv2
    image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image from {file_path}")
    return image


class RegistrationMethod(Enum):
    """Available multi-modal registration methods."""
    MUTUAL_INFORMATION = "mutual_information"
    FEATURE_BASED = "feature_based"
    INTENSITY_BASED = "intensity_based"
    NORMALIZED_CROSS_CORRELATION = "normalized_cross_correlation"
    PHASE_CORRELATION = "phase_correlation"
    HYBRID = "hybrid"


class QualityMetric(Enum):
    """Available quality assessment metrics."""
    MUTUAL_INFORMATION = "mutual_information"
    NORMALIZED_MUTUAL_INFORMATION = "normalized_mutual_information"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    CROSS_CORRELATION = "cross_correlation"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    PEAK_SIGNAL_NOISE_RATIO = "peak_signal_noise_ratio"


@dataclass
class RegistrationResult:
    """Result from multi-modal registration."""
    transform_matrix: np.ndarray
    registered_image: np.ndarray
    confidence: float
    method: RegistrationMethod
    processing_time: float
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityPair:
    """Container for a pair of images from different modalities."""
    modality1_image: np.ndarray
    modality2_image: np.ndarray
    modality1_name: str = "modality1"
    modality2_name: str = "modality2"
    preprocessing_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiModalAligner:
    """
    Comprehensive multi-modal image registration system.
    
    Provides robust registration between different imaging modalities
    using various algorithms optimized for different image characteristics.
    """
    
    def __init__(self,
                 registration_method: RegistrationMethod = RegistrationMethod.MUTUAL_INFORMATION,
                 quality_metrics: List[QualityMetric] = None,
                 preprocessing_enabled: bool = True,
                 optimization_method: str = "powell",
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6):
        """
        Initialize the multi-modal aligner.
        
        Parameters
        ----------
        registration_method : RegistrationMethod
            Primary registration method to use
        quality_metrics : List[QualityMetric], optional
            Metrics to calculate for quality assessment
        preprocessing_enabled : bool
            Whether to apply preprocessing to improve registration
        optimization_method : str
            Optimization method for parameter search ('powell', 'bfgs', 'nelder-mead')
        max_iterations : int
            Maximum iterations for optimization
        convergence_threshold : float
            Convergence threshold for optimization
        """
        self.registration_method = registration_method
        self.quality_metrics = quality_metrics or [
            QualityMetric.MUTUAL_INFORMATION,
            QualityMetric.STRUCTURAL_SIMILARITY
        ]
        self.preprocessing_enabled = preprocessing_enabled
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize single-modal aligner for fallback
        self.single_modal_aligner = UnifiedAligner(aligner_type=AlignerType.ENHANCED)
        
        # Statistics tracking
        self.registration_stats = {
            'total_registrations': 0,
            'method_usage': {method: 0 for method in RegistrationMethod},
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'successful_registrations': 0
        }
    
    def register_images(self,
                       fixed_image: np.ndarray,
                       moving_image: np.ndarray,
                       method: Optional[RegistrationMethod] = None,
                       fixed_modality: str = "fixed",
                       moving_modality: str = "moving") -> RegistrationResult:
        """
        Register two images from different modalities.
        
        Parameters
        ----------
        fixed_image : np.ndarray
            Reference image (stays fixed)
        moving_image : np.ndarray
            Image to be transformed to align with fixed image
        method : RegistrationMethod, optional
            Registration method to use
        fixed_modality : str
            Name/type of the fixed image modality
        moving_modality : str
            Name/type of the moving image modality
            
        Returns
        -------
        RegistrationResult
            Registration result with transform and quality metrics
        """
        if method is None:
            method = self.registration_method
        
        start_time = time.time()
        
        try:
            # Create modality pair
            pair = ModalityPair(
                modality1_image=fixed_image,
                modality2_image=moving_image,
                modality1_name=fixed_modality,
                modality2_name=moving_modality
            )
            
            # Apply preprocessing if enabled
            if self.preprocessing_enabled:
                pair = self._preprocess_modality_pair(pair)
            
            # Perform registration based on method
            if method == RegistrationMethod.MUTUAL_INFORMATION:
                result = self._register_mutual_information(pair)
            elif method == RegistrationMethod.FEATURE_BASED:
                result = self._register_feature_based(pair)
            elif method == RegistrationMethod.INTENSITY_BASED:
                result = self._register_intensity_based(pair)
            elif method == RegistrationMethod.NORMALIZED_CROSS_CORRELATION:
                result = self._register_cross_correlation(pair)
            elif method == RegistrationMethod.PHASE_CORRELATION:
                result = self._register_phase_correlation(pair)
            elif method == RegistrationMethod.HYBRID:
                result = self._register_hybrid(pair)
            else:
                raise ValueError(f"Unknown registration method: {method}")
            
            # Calculate quality metrics
            result.quality_metrics = self._calculate_quality_metrics(
                pair.modality1_image, result.registered_image
            )
            
            # Update processing time
            result.processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(method, result.confidence, result.processing_time, success=True)
            
            self.logger.debug(f"Registration completed: method={method.value}, confidence={result.confidence:.3f}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Registration failed: {e}")
            
            # Update statistics
            self._update_stats(method, 0.0, processing_time, success=False)
            
            # Return identity transformation as fallback
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=moving_image.copy(),
                confidence=0.0,
                method=method,
                processing_time=processing_time,
                quality_metrics={},
                metadata={'error': str(e)}
            )
    
    def register_stacks(self,
                       fixed_stack: np.ndarray,
                       moving_stack: np.ndarray,
                       method: Optional[RegistrationMethod] = None,
                       slice_correspondence: Optional[List[Tuple[int, int]]] = None) -> List[RegistrationResult]:
        """
        Register corresponding slices between two image stacks.
        
        Parameters
        ----------
        fixed_stack : np.ndarray
            Fixed image stack (shape: [n_slices, height, width])
        moving_stack : np.ndarray
            Moving image stack to be registered
        method : RegistrationMethod, optional
            Registration method to use
        slice_correspondence : List[Tuple[int, int]], optional
            List of (fixed_idx, moving_idx) pairs to register
            
        Returns
        -------
        List[RegistrationResult]
            Registration results for each slice pair
        """
        if slice_correspondence is None:
            # Default: register corresponding slice indices
            n_pairs = min(fixed_stack.shape[0], moving_stack.shape[0])
            slice_correspondence = [(i, i) for i in range(n_pairs)]
        
        self.logger.info(f"Registering {len(slice_correspondence)} slice pairs")
        
        results = []
        for fixed_idx, moving_idx in slice_correspondence:
            try:
                result = self.register_images(
                    fixed_stack[fixed_idx],
                    moving_stack[moving_idx],
                    method=method,
                    fixed_modality=f"fixed_slice_{fixed_idx}",
                    moving_modality=f"moving_slice_{moving_idx}"
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to register slice pair ({fixed_idx}, {moving_idx}): {e}")
                # Add failed result
                results.append(RegistrationResult(
                    transform_matrix=np.eye(3),
                    registered_image=moving_stack[moving_idx].copy(),
                    confidence=0.0,
                    method=method or self.registration_method,
                    processing_time=0.0,
                    metadata={'error': str(e), 'slice_pair': (fixed_idx, moving_idx)}
                ))
        
        return results
    
    def _preprocess_modality_pair(self, pair: ModalityPair) -> ModalityPair:
        """Apply preprocessing to improve multi-modal registration."""
        try:
            # Convert to grayscale if needed
            img1 = self._ensure_grayscale(pair.modality1_image)
            img2 = self._ensure_grayscale(pair.modality2_image)
            
            # Histogram equalization for better contrast
            img1 = exposure.equalize_adapthist(img1, clip_limit=0.03)
            img2 = exposure.equalize_adapthist(img2, clip_limit=0.03)
            
            # Gaussian smoothing to reduce noise
            img1 = filters.gaussian(img1, sigma=1.0)
            img2 = filters.gaussian(img2, sigma=1.0)
            
            # Edge enhancement for feature-based methods
            if self.registration_method in [RegistrationMethod.FEATURE_BASED, RegistrationMethod.HYBRID]:
                img1 = filters.unsharp_mask(img1, radius=1, amount=1)
                img2 = filters.unsharp_mask(img2, radius=1, amount=1)
            
            # Convert back to appropriate dtype
            img1 = (img1 * 255).astype(np.uint8)
            img2 = (img2 * 255).astype(np.uint8)
            
            pair.modality1_image = img1
            pair.modality2_image = img2
            pair.preprocessing_applied = True
            
            return pair
            
        except Exception as e:
            self.logger.warning(f"Preprocessing failed: {e}, using original images")
            return pair
    
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _register_mutual_information(self, pair: ModalityPair) -> RegistrationResult:
        """Register using mutual information optimization."""
        fixed = pair.modality1_image.astype(np.float32)
        moving = pair.modality2_image.astype(np.float32)
        
        # Define optimization objective
        def objective(params):
            # params: [tx, ty, rotation, scale_x, scale_y]
            if len(params) == 2:
                # Translation only
                tx, ty = params
                transform_matrix = np.array([
                    [1, 0, tx],
                    [0, 1, ty],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                # Full affine transform
                tx, ty, rotation, scale_x, scale_y = params
                cos_r = np.cos(rotation)
                sin_r = np.sin(rotation)
                transform_matrix = np.array([
                    [scale_x * cos_r, -scale_x * sin_r, tx],
                    [scale_y * sin_r, scale_y * cos_r, ty],
                    [0, 0, 1]
                ], dtype=np.float32)
            
            # Apply transformation
            transformed = cv2.warpAffine(moving, transform_matrix[:2, :], fixed.shape[::-1])
            
            # Calculate mutual information (negative for minimization)
            mi = self._calculate_mutual_information(fixed, transformed)
            return -mi
        
        try:
            # Start with translation-only optimization
            initial_params = [0.0, 0.0]  # tx, ty
            
            result = optimize.minimize(
                objective,
                initial_params,
                method=self.optimization_method,
                options={'maxiter': self.max_iterations}
            )
            
            # Build final transformation matrix
            if result.success:
                tx, ty = result.x
                transform_matrix = np.array([
                    [1, 0, tx],
                    [0, 1, ty],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Apply final transformation
                registered_image = cv2.warpAffine(
                    moving, transform_matrix[:2, :], fixed.shape[::-1]
                )
                
                # Calculate confidence based on mutual information improvement
                original_mi = self._calculate_mutual_information(fixed, moving)
                final_mi = self._calculate_mutual_information(fixed, registered_image)
                confidence = min(1.0, max(0.0, (final_mi - original_mi) / (original_mi + 1e-8)))
                
                return RegistrationResult(
                    transform_matrix=transform_matrix,
                    registered_image=registered_image,
                    confidence=confidence,
                    method=RegistrationMethod.MUTUAL_INFORMATION,
                    processing_time=0.0,  # Will be set by caller
                    metadata={
                        'optimization_success': True,
                        'optimization_iterations': result.nit,
                        'final_mutual_information': final_mi,
                        'original_mutual_information': original_mi
                    }
                )
            else:
                # Optimization failed, return identity
                return RegistrationResult(
                    transform_matrix=np.eye(3),
                    registered_image=moving.copy(),
                    confidence=0.0,
                    method=RegistrationMethod.MUTUAL_INFORMATION,
                    processing_time=0.0,
                    metadata={'optimization_success': False, 'error': 'optimization_failed'}
                )
                
        except Exception as e:
            self.logger.error(f"Mutual information registration failed: {e}")
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=moving.copy(),
                confidence=0.0,
                method=RegistrationMethod.MUTUAL_INFORMATION,
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
    def _register_feature_based(self, pair: ModalityPair) -> RegistrationResult:
        """Register using feature-based matching."""
        # Use the unified aligner
        try:
            single_modal_result = self.single_modal_aligner.align_pair(
                pair.modality1_image, 
                pair.modality2_image,
                requirements={'method': 'feature_based'}
            )
            
            return RegistrationResult(
                transform_matrix=single_modal_result.transform_matrix,
                registered_image=single_modal_result.aligned_image,
                confidence=single_modal_result.confidence,
                method=RegistrationMethod.FEATURE_BASED,
                processing_time=0.0,
                metadata={'single_modal_result': single_modal_result.metadata}
            )
            
        except Exception as e:
            self.logger.error(f"Feature-based registration failed: {e}")
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=pair.modality2_image.copy(),
                confidence=0.0,
                method=RegistrationMethod.FEATURE_BASED,
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
    def _register_intensity_based(self, pair: ModalityPair) -> RegistrationResult:
        """Register using intensity-based template matching."""
        # Use the unified aligner
        try:
            single_modal_result = self.single_modal_aligner.align_pair(
                pair.modality1_image,
                pair.modality2_image,
                requirements={'method': 'intensity_based'}
            )
            
            return RegistrationResult(
                transform_matrix=single_modal_result.transform_matrix,
                registered_image=single_modal_result.aligned_image,
                confidence=single_modal_result.confidence,
                method=RegistrationMethod.INTENSITY_BASED,
                processing_time=0.0,
                metadata={'single_modal_result': single_modal_result.metadata}
            )
            
        except Exception as e:
            self.logger.error(f"Intensity-based registration failed: {e}")
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=pair.modality2_image.copy(),
                confidence=0.0,
                method=RegistrationMethod.INTENSITY_BASED,
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
    def _register_cross_correlation(self, pair: ModalityPair) -> RegistrationResult:
        """Register using normalized cross-correlation."""
        try:
            # Normalize images
            fixed = pair.modality1_image.astype(np.float32)
            moving = pair.modality2_image.astype(np.float32)
            
            fixed = (fixed - np.mean(fixed)) / (np.std(fixed) + 1e-8)
            moving = (moving - np.mean(moving)) / (np.std(moving) + 1e-8)
            
            # Perform cross-correlation
            correlation = cv2.matchTemplate(fixed, moving, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(correlation)
            
            # Calculate translation
            h, w = moving.shape
            dx = max_loc[0] - (fixed.shape[1] - w) // 2
            dy = max_loc[1] - (fixed.shape[0] - h) // 2
            
            # Create transformation matrix
            transform_matrix = np.array([
                [1, 0, dx],
                [0, 1, dy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Apply transformation
            registered_image = cv2.warpAffine(
                pair.modality2_image, transform_matrix[:2, :], pair.modality1_image.shape[::-1]
            )
            
            # Use correlation value as confidence
            confidence = max(0.0, min(1.0, max_val))
            
            return RegistrationResult(
                transform_matrix=transform_matrix,
                registered_image=registered_image,
                confidence=confidence,
                method=RegistrationMethod.NORMALIZED_CROSS_CORRELATION,
                processing_time=0.0,
                metadata={
                    'correlation_value': max_val,
                    'translation': (dx, dy)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Cross-correlation registration failed: {e}")
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=pair.modality2_image.copy(),
                confidence=0.0,
                method=RegistrationMethod.NORMALIZED_CROSS_CORRELATION,
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
    def _register_phase_correlation(self, pair: ModalityPair) -> RegistrationResult:
        """Register using phase correlation."""
        # Use the unified aligner
        try:
            single_modal_result = self.single_modal_aligner.align_pair(
                pair.modality1_image,
                pair.modality2_image,
                requirements={'method': 'phase_correlation'}
            )
            
            return RegistrationResult(
                transform_matrix=single_modal_result.transform_matrix,
                registered_image=single_modal_result.aligned_image,
                confidence=single_modal_result.confidence,
                method=RegistrationMethod.PHASE_CORRELATION,
                processing_time=0.0,
                metadata={'single_modal_result': single_modal_result.metadata}
            )
            
        except Exception as e:
            self.logger.error(f"Phase correlation registration failed: {e}")
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=pair.modality2_image.copy(),
                confidence=0.0,
                method=RegistrationMethod.PHASE_CORRELATION,
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
    def _register_hybrid(self, pair: ModalityPair) -> RegistrationResult:
        """Register using hybrid approach - try multiple methods and pick the best."""
        methods = [
            RegistrationMethod.MUTUAL_INFORMATION,
            RegistrationMethod.FEATURE_BASED,
            RegistrationMethod.NORMALIZED_CROSS_CORRELATION
        ]
        
        results = []
        for method in methods:
            try:
                if method == RegistrationMethod.MUTUAL_INFORMATION:
                    result = self._register_mutual_information(pair)
                elif method == RegistrationMethod.FEATURE_BASED:
                    result = self._register_feature_based(pair)
                elif method == RegistrationMethod.NORMALIZED_CROSS_CORRELATION:
                    result = self._register_cross_correlation(pair)
                else:
                    continue
                
                results.append(result)
                
            except Exception as e:
                self.logger.debug(f"Hybrid method {method.value} failed: {e}")
                continue
        
        if results:
            # Select best result based on confidence
            best_result = max(results, key=lambda r: r.confidence)
            best_result.method = RegistrationMethod.HYBRID
            best_result.metadata['hybrid_methods_tried'] = [r.method.value for r in results]
            best_result.metadata['hybrid_confidences'] = [r.confidence for r in results]
            return best_result
        else:
            # All methods failed
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=pair.modality2_image.copy(),
                confidence=0.0,
                method=RegistrationMethod.HYBRID,
                processing_time=0.0,
                metadata={'error': 'all_hybrid_methods_failed'}
            )
    
    def _calculate_mutual_information(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate mutual information between two images."""
        try:
            # Convert to uint8 and flatten
            img1_flat = (img1.flatten()).astype(np.uint8)
            img2_flat = (img2.flatten()).astype(np.uint8)
            
            # Calculate joint histogram
            hist_2d, _, _ = np.histogram2d(img1_flat, img2_flat, bins=256)
            
            # Convert to probabilities
            joint_prob = hist_2d / np.sum(hist_2d)
            
            # Calculate marginal probabilities
            prob1 = np.sum(joint_prob, axis=1)
            prob2 = np.sum(joint_prob, axis=0)
            
            # Calculate mutual information
            mi = 0.0
            for i in range(len(prob1)):
                for j in range(len(prob2)):
                    if joint_prob[i, j] > 0 and prob1[i] > 0 and prob2[j] > 0:
                        mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (prob1[i] * prob2[j]))
            
            return mi
            
        except Exception as e:
            self.logger.warning(f"Mutual information calculation failed: {e}")
            return 0.0
    
    def _calculate_quality_metrics(self, fixed: np.ndarray, registered: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for registration assessment."""
        metrics = {}
        
        try:
            # Ensure same size
            if fixed.shape != registered.shape:
                registered = cv2.resize(registered, fixed.shape[::-1])
            
            # Convert to same dtype
            fixed = fixed.astype(np.float32)
            registered = registered.astype(np.float32)
            
            # Mutual Information
            if QualityMetric.MUTUAL_INFORMATION in self.quality_metrics:
                mi = self._calculate_mutual_information(fixed, registered)
                metrics['mutual_information'] = mi
            
            # Normalized Mutual Information
            if QualityMetric.NORMALIZED_MUTUAL_INFORMATION in self.quality_metrics:
                mi = self._calculate_mutual_information(fixed, registered)
                h1 = entropy(fixed.flatten())
                h2 = entropy(registered.flatten())
                nmi = 2 * mi / (h1 + h2) if (h1 + h2) > 0 else 0.0
                metrics['normalized_mutual_information'] = nmi
            
            # Structural Similarity (using simple correlation as approximation)
            if QualityMetric.STRUCTURAL_SIMILARITY in self.quality_metrics:
                # Normalize images
                fixed_norm = (fixed - np.mean(fixed)) / (np.std(fixed) + 1e-8)
                registered_norm = (registered - np.mean(registered)) / (np.std(registered) + 1e-8)
                
                # Calculate correlation
                correlation = np.corrcoef(fixed_norm.flatten(), registered_norm.flatten())[0, 1]
                metrics['structural_similarity'] = max(0.0, correlation)
            
            # Cross Correlation
            if QualityMetric.CROSS_CORRELATION in self.quality_metrics:
                correlation = cv2.matchTemplate(fixed, registered, cv2.TM_CCOEFF_NORMED)
                metrics['cross_correlation'] = np.max(correlation)
            
            # Mean Squared Error
            if QualityMetric.MEAN_SQUARED_ERROR in self.quality_metrics:
                mse = np.mean((fixed - registered) ** 2)
                metrics['mean_squared_error'] = mse
            
            # Peak Signal-to-Noise Ratio
            if QualityMetric.PEAK_SIGNAL_NOISE_RATIO in self.quality_metrics:
                mse = np.mean((fixed - registered) ** 2)
                if mse > 0:
                    max_val = max(np.max(fixed), np.max(registered))
                    psnr = 20 * np.log10(max_val / np.sqrt(mse))
                    metrics['peak_signal_noise_ratio'] = psnr
                else:
                    metrics['peak_signal_noise_ratio'] = float('inf')
            
        except Exception as e:
            self.logger.warning(f"Quality metric calculation failed: {e}")
        
        return metrics
    
    def get_registration_stats(self) -> Dict[str, Any]:
        """Get registration statistics."""
        return self.registration_stats.copy()
    
    def reset_stats(self):
        """Reset registration statistics."""
        self.registration_stats = {
            'total_registrations': 0,
            'method_usage': {method: 0 for method in RegistrationMethod},
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'successful_registrations': 0
        }
    
    def _update_stats(self, method: RegistrationMethod, confidence: float, processing_time: float, success: bool):
        """Update registration statistics."""
        self.registration_stats['total_registrations'] += 1
        self.registration_stats['method_usage'][method] += 1
        
        if success:
            self.registration_stats['successful_registrations'] += 1
        
        # Update running averages
        n = self.registration_stats['total_registrations']
        self.registration_stats['average_confidence'] = (
            (self.registration_stats['average_confidence'] * (n - 1) + confidence) / n
        )
        self.registration_stats['average_processing_time'] = (
            (self.registration_stats['average_processing_time'] * (n - 1) + processing_time) / n
        )
