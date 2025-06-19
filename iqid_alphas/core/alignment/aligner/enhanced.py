"""
Enhanced Aligner - Main Orchestrator

This module provides the main EnhancedAligner class that orchestrates
all alignment functionality and provides a clean API.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import cv2

from ..types import AlignmentMethod, ReferenceStrategy, AlignmentResult, StackAlignmentResult
from ..algorithms import (
    FeatureBasedAligner, 
    IntensityBasedAligner, 
    PhaseCorrelationAligner, 
    CoarseRotationAligner
)
from .stack import StackAligner
from ..utils import assemble_image_stack
from ...mapping import ReverseMapper


class EnhancedAligner:
    """
    Enhanced slice alignment class providing comprehensive alignment functionality.
    
    This class provides comprehensive slice alignment functionality including:
    - Multiple alignment methods (feature-based, intensity-based, phase correlation, coarse rotation)
    - Stack processing with configurable reference strategies
    - Quality metrics and validation
    - Integration with reverse mapping evaluation
    """
    
    def __init__(self,
                 default_method: AlignmentMethod = AlignmentMethod.FEATURE_BASED,
                 reference_strategy: ReferenceStrategy = ReferenceStrategy.PREVIOUS_IMAGE,
                 feature_detector: str = 'ORB',
                 max_features: int = 5000,
                 rolling_window: int = 3,
                 enable_validation: bool = True):
        """
        Initialize the Enhanced Aligner.
        
        Parameters
        ----------
        default_method : AlignmentMethod
            Default alignment method to use
        reference_strategy : ReferenceStrategy
            Default reference selection strategy for stack alignment
        feature_detector : str
            Feature detector type ('ORB', 'SIFT')
        max_features : int
            Maximum number of features to detect
        rolling_window : int
            Window size for rolling average reference strategy
        enable_validation : bool
            Whether to enable alignment validation
        """
        self.default_method = default_method
        self.reference_strategy = reference_strategy
        self.feature_detector = feature_detector
        self.max_features = max_features
        self.rolling_window = rolling_window
        self.enable_validation = enable_validation
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize alignment algorithms
        self.aligners = {
            AlignmentMethod.FEATURE_BASED: FeatureBasedAligner(
                detector_type=feature_detector,
                max_features=max_features
            ),
            AlignmentMethod.INTENSITY_BASED: IntensityBasedAligner(),
            AlignmentMethod.PHASE_CORRELATION: PhaseCorrelationAligner(),
            AlignmentMethod.COARSE_ROTATION: CoarseRotationAligner()
        }
        
        # Initialize stack aligner
        self.stack_aligner = StackAligner(
            reference_strategy=reference_strategy,
            rolling_window=rolling_window
        )
        
        # Initialize reverse mapper for validation
        if enable_validation:
            self.reverse_mapper = ReverseMapper()
        
        # Statistics tracking
        self.alignment_stats = {
            'total_alignments': 0,
            'method_usage': {method: 0 for method in AlignmentMethod},
            'average_confidence': 0.0,
            'average_processing_time': 0.0
        }
    
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
            Alignment method to use (defaults to default_method)
            
        Returns
        -------
        AlignmentResult
            Alignment result with transform matrix and confidence
        """
        if method is None:
            method = self.default_method
        
        start_time = time.time()
        
        try:
            # Get the appropriate aligner
            aligner = self.aligners.get(method)
            if aligner is None:
                raise ValueError(f"Unknown alignment method: {method}")
            
            # Perform alignment
            result = aligner.align(moving_image, reference_image)
            
            # Update statistics
            self._update_stats(method, result.confidence, result.processing_time)
            
            self.logger.debug(f"Alignment completed: method={method.value}, confidence={result.confidence:.3f}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Alignment failed: {e}")
            
            # Return identity transformation as fallback
            return AlignmentResult(
                transform_matrix=np.eye(3),
                confidence=0.0,
                method=method,
                processing_time=processing_time,
                metadata={'error': str(e)}
            )
    
    def align_stack(self,
                   image_stack: np.ndarray,
                   reference_strategy: Optional[ReferenceStrategy] = None,
                   save_intermediate: bool = False,
                   output_dir: Optional[Path] = None) -> StackAlignmentResult:
        """
        Align a complete image stack.
        
        Parameters
        ----------
        image_stack : np.ndarray
            Stack of images to align
        reference_strategy : ReferenceStrategy, optional
            Reference selection strategy
        save_intermediate : bool
            Whether to save intermediate aligned images
        output_dir : Path, optional
            Directory to save intermediate results
            
        Returns
        -------
        StackAlignmentResult
            Complete stack alignment result
        """
        self.logger.info(f"Starting stack alignment of {image_stack.shape[0]} images")
        
        # Use stack aligner
        result = self.stack_aligner.align_stack(image_stack, reference_strategy)
        
        # Save intermediate results if requested
        if save_intermediate and output_dir:
            self._save_intermediate_results(result, output_dir)
        
        # Validate alignment if enabled
        if self.enable_validation:
            validation_metrics = self._validate_stack_alignment(image_stack, result)
            result.metadata.update({'validation_metrics': validation_metrics})
        
        self.logger.info(f"Stack alignment completed: mean_similarity={result.quality_metrics.get('mean_similarity', 0):.3f}")
        
        return result
    
    def align_directory(self,
                       input_dir: Union[str, Path],
                       output_dir: Union[str, Path],
                       file_pattern: str = "*.tif",
                       reference_strategy: Optional[ReferenceStrategy] = None) -> StackAlignmentResult:
        """
        Align all images in a directory.
        
        Parameters
        ----------
        input_dir : str or Path
            Directory containing images to align
        output_dir : str or Path
            Directory to save aligned images
        file_pattern : str
            Pattern to match image files
        reference_strategy : ReferenceStrategy, optional
            Reference selection strategy
            
        Returns
        -------
        StackAlignmentResult
            Alignment results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Aligning directory: {input_dir} -> {output_dir}")
        
        # Assemble image stack
        image_stack = assemble_image_stack(input_dir, file_pattern)
        
        # Align stack
        result = self.align_stack(image_stack, reference_strategy)
        
        # Save aligned images
        self._save_aligned_images(result, output_dir, input_dir, file_pattern)
        
        return result
    
    def validate_alignment(self,
                          original_images: List[np.ndarray],
                          aligned_images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Validate alignment results using reverse mapping.
        
        Parameters
        ----------
        original_images : List[np.ndarray]
            Original unaligned images
        aligned_images : List[np.ndarray]
            Aligned images
            
        Returns
        -------
        Dict[str, Any]
            Validation metrics
        """
        if not self.enable_validation:
            return {'validation_enabled': False}
        
        validation_results = []
        
        for i, (orig, aligned) in enumerate(zip(original_images, aligned_images)):
            try:
                # Use reverse mapper to calculate transformation
                reverse_result = self.reverse_mapper.calculate_alignment_transform(orig, aligned)
                
                validation_results.append({
                    'slice_index': i,
                    'reverse_confidence': reverse_result.match_quality,
                    'reverse_matches': reverse_result.num_matches,
                    'reverse_inlier_ratio': reverse_result.inlier_ratio
                })
                
            except Exception as e:
                self.logger.warning(f"Validation failed for slice {i}: {e}")
                validation_results.append({
                    'slice_index': i,
                    'reverse_confidence': 0.0,
                    'reverse_matches': 0,
                    'reverse_inlier_ratio': 0.0,
                    'error': str(e)
                })
        
        # Calculate aggregate metrics
        confidences = [r['reverse_confidence'] for r in validation_results]
        matches = [r['reverse_matches'] for r in validation_results]
        
        return {
            'validation_enabled': True,
            'per_slice_results': validation_results,
            'mean_reverse_confidence': np.mean(confidences),
            'mean_reverse_matches': np.mean(matches),
            'successful_validations': sum(1 for c in confidences if c > 0.1),
            'total_slices': len(validation_results)
        }
    
    def get_alignment_stats(self) -> Dict[str, Any]:
        """Get alignment statistics."""
        return self.alignment_stats.copy()
    
    def reset_stats(self):
        """Reset alignment statistics."""
        self.alignment_stats = {
            'total_alignments': 0,
            'method_usage': {method: 0 for method in AlignmentMethod},
            'average_confidence': 0.0,
            'average_processing_time': 0.0
        }
    
    def _update_stats(self, method: AlignmentMethod, confidence: float, processing_time: float):
        """Update alignment statistics."""
        self.alignment_stats['total_alignments'] += 1
        self.alignment_stats['method_usage'][method] += 1
        
        # Update running averages
        n = self.alignment_stats['total_alignments']
        self.alignment_stats['average_confidence'] = (
            (self.alignment_stats['average_confidence'] * (n - 1) + confidence) / n
        )
        self.alignment_stats['average_processing_time'] = (
            (self.alignment_stats['average_processing_time'] * (n - 1) + processing_time) / n
        )
    
    def _validate_stack_alignment(self, 
                                 original_stack: np.ndarray,
                                 alignment_result: StackAlignmentResult) -> Dict[str, Any]:
        """Validate stack alignment using reverse mapping."""
        return self.validate_alignment(
            list(original_stack),
            alignment_result.aligned_images
        )
    
    def _save_intermediate_results(self, 
                                  result: StackAlignmentResult,
                                  output_dir: Path):
        """Save intermediate alignment results."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save transform matrices
            transforms_file = output_dir / "transform_matrices.npy"
            np.save(transforms_file, np.array(result.transforms))
            
            # Save quality metrics
            import json
            metrics_file = output_dir / "quality_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(result.quality_metrics, f, indent=2)
            
            self.logger.info(f"Intermediate results saved to {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results: {e}")
    
    def _save_aligned_images(self,
                            result: StackAlignmentResult,
                            output_dir: Path,
                            input_dir: Path,
                            file_pattern: str):
        """Save aligned images to directory."""
        try:
            # Get original filenames
            image_files = sorted(list(input_dir.glob(file_pattern)))
            
            for i, (aligned_img, filename) in enumerate(zip(result.aligned_images, image_files)):
                output_path = output_dir / filename.name
                cv2.imwrite(str(output_path), aligned_img)
            
            self.logger.info(f"Aligned images saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save aligned images: {e}")
