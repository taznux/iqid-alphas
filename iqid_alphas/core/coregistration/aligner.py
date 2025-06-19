"""
Main Multi-Modal Aligner

Orchestrates different registration methods and provides a unified interface.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any

from .types import (
    RegistrationMethod, 
    QualityMetric, 
    RegistrationResult, 
    ModalityPair
)
from .mutual_info import MutualInfoAligner
from .correlation import CrossCorrelationAligner
from .preprocessing import ImagePreprocessor
from .quality import QualityAssessor
from ..alignment.aligner import EnhancedAligner
from ..alignment import AlignmentMethod


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
            Optimization method for parameter search
        max_iterations : int
            Maximum iterations for optimization
        convergence_threshold : float
            Convergence threshold for optimization
        """
        self.registration_method = registration_method
        self.preprocessing_enabled = preprocessing_enabled
        self.quality_metrics = quality_metrics or [
            QualityMetric.MUTUAL_INFORMATION,
            QualityMetric.STRUCTURAL_SIMILARITY
        ]
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize component classes
        self.mutual_info_aligner = MutualInfoAligner(
            optimization_method=optimization_method,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold
        )
        self.correlation_aligner = CrossCorrelationAligner()
        self.preprocessor = ImagePreprocessor()
        self.quality_assessor = QualityAssessor(self.quality_metrics)
        
        # Initialize single-modal aligner for fallback
        self.single_modal_aligner = EnhancedAligner()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
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
        import numpy as np
        
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
                pair = self.preprocessor.preprocess_pair(pair)
            
            # Perform registration based on method
            if method == RegistrationMethod.MUTUAL_INFORMATION:
                result = self.mutual_info_aligner.register(pair)
            elif method == RegistrationMethod.NORMALIZED_CROSS_CORRELATION:
                result = self.correlation_aligner.register_normalized(pair)
            elif method == RegistrationMethod.PHASE_CORRELATION:
                result = self.correlation_aligner.register_phase(pair)
            elif method == RegistrationMethod.FEATURE_BASED:
                result = self._register_feature_based(pair)
            elif method == RegistrationMethod.INTENSITY_BASED:
                result = self._register_intensity_based(pair)
            elif method == RegistrationMethod.HYBRID:
                result = self._register_hybrid(pair)
            else:
                raise ValueError(f"Unknown registration method: {method}")
            
            # Calculate quality metrics
            result.quality_metrics = self.quality_assessor.calculate_metrics(
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
    
    def _register_feature_based(self, pair: ModalityPair) -> RegistrationResult:
        """Register using feature-based matching."""
        import numpy as np
        import cv2
        
        try:
            single_modal_result = self.single_modal_aligner.align_slice(
                pair.modality2_image, 
                pair.modality1_image,
                method=AlignmentMethod.FEATURE_BASED
            )
            
            # Apply the transformation
            registered_image = cv2.warpAffine(
                pair.modality2_image,
                single_modal_result.transform_matrix[:2, :],
                pair.modality1_image.shape[::-1]
            )
            
            return RegistrationResult(
                transform_matrix=single_modal_result.transform_matrix,
                registered_image=registered_image,
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
        import numpy as np
        import cv2
        
        try:
            single_modal_result = self.single_modal_aligner.align_slice(
                pair.modality2_image,
                pair.modality1_image,
                method=AlignmentMethod.INTENSITY_BASED
            )
            
            # Apply the transformation
            registered_image = cv2.warpAffine(
                pair.modality2_image,
                single_modal_result.transform_matrix[:2, :],
                pair.modality1_image.shape[::-1]
            )
            
            return RegistrationResult(
                transform_matrix=single_modal_result.transform_matrix,
                registered_image=registered_image,
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
                    result = self.mutual_info_aligner.register(pair)
                elif method == RegistrationMethod.FEATURE_BASED:
                    result = self._register_feature_based(pair)
                elif method == RegistrationMethod.NORMALIZED_CROSS_CORRELATION:
                    result = self.correlation_aligner.register_normalized(pair)
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
            import numpy as np
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=pair.modality2_image.copy(),
                confidence=0.0,
                method=RegistrationMethod.HYBRID,
                processing_time=0.0,
                metadata={'error': 'all_hybrid_methods_failed'}
            )
    
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
