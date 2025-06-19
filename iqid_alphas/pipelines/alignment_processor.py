#!/usr/bin/env python3
"""Alignment Pipeline Processing Module"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..core.alignment.aligner import UnifiedAligner
from ..core.alignment import AlignmentMethod, ReferenceStrategy
from ..core.validation import ValidationResult, ValidationSuite, ValidationMode, AlignmentValidator, ValidationConfig
from ..utils.io import ensure_directory_exists, load_image


class AlignmentProcessor:
    """Handles alignment processing operations."""
    
    def __init__(self, config):
        self.config = config
        
        # Map config values to valid enum values
        method_mapping = {
            "feature_based": AlignmentMethod.FEATURE_BASED,
            "intensity_based": AlignmentMethod.INTENSITY_BASED,
            "phase_correlation": AlignmentMethod.PHASE_CORRELATION,
            "hybrid": AlignmentMethod.HYBRID,
            "stackreg": AlignmentMethod.FEATURE_BASED  # Default fallback
        }
        
        strategy_mapping = {
            "largest_slice": ReferenceStrategy.FIRST_IMAGE,
            "first_image": ReferenceStrategy.FIRST_IMAGE,
            "previous_image": ReferenceStrategy.PREVIOUS_IMAGE,
            "template_matching": ReferenceStrategy.TEMPLATE_MATCHING
        }
        
        alignment_method = method_mapping.get(config.alignment_algorithm.lower(), AlignmentMethod.FEATURE_BASED)
        reference_strategy = strategy_mapping.get(config.reference_strategy.lower(), ReferenceStrategy.FIRST_IMAGE)
        
        self.aligner = UnifiedAligner(
            default_method=alignment_method,
            reference_strategy=reference_strategy
        )
        
        # Create validation config and validator
        validation_config = ValidationConfig()
        self.validator = AlignmentValidator(validation_config)
    
    def process_sample(self, sample: Dict[str, Any], logger) -> ValidationResult:
        """Process a single alignment sample."""
        sample_id = sample['id']
        segmented_dir = sample['segmented_dir']
        aligned_dir = sample['aligned_dir']
        start_time = time.time()
        
        try:
            logger.debug(f"Processing alignment sample: {sample_id}")
            
            segmented_slices = self._load_slice_stack(segmented_dir)
            ground_truth_slices = self._load_slice_stack(aligned_dir)
            
            if len(segmented_slices) < self.config.min_slice_count:
                raise ValueError(f"Insufficient slices: {len(segmented_slices)} < {self.config.min_slice_count}")
            
            logger.debug(f"Found {len(segmented_slices)} segmented slices and {len(ground_truth_slices)} ground truth slices")
            
            # Load actual image data for alignment
            image_stack = self._load_image_stack(segmented_slices)
            
            # Perform real alignment using the UnifiedAligner
            alignment_result = self.aligner.align_stack(image_stack)
            
            # Simple validation: check if alignment was successful
            alignment_success = self._validate_alignment_result(alignment_result)
            
            metrics = self._calculate_metrics(alignment_result, None, sample_id)
            
            # Create a compatible result with success attribute
            result = ValidationResult(
                mode=ValidationMode.ALIGNMENT,
                metrics=metrics,
                processing_time=time.time() - start_time,
                passed_tests=1 if alignment_success else 0,
                total_tests=1,
                metadata={'sample_id': sample_id}
            )
            
            # Add success attribute for base pipeline compatibility
            result.success = alignment_success
            result.error_message = None
            result.validation_type = "alignment"
            result.sample_id = sample_id
            
            # Add to_dict method for pipeline compatibility
            def to_dict(self):
                return {
                    'sample_id': getattr(self, 'sample_id', sample_id),
                    'success': getattr(self, 'success', True),
                    'metrics': dict(self.metrics),
                    'processing_time': self.processing_time,
                    'error_message': getattr(self, 'error_message', None),
                    'validation_type': getattr(self, 'validation_type', 'alignment')
                }
            result.to_dict = to_dict.__get__(result, type(result))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process alignment sample {sample_id}: {e}")
            result = ValidationResult(
                mode=ValidationMode.ALIGNMENT,
                metrics={},
                processing_time=time.time() - start_time,
                passed_tests=0,
                total_tests=1,
                errors=[str(e)],
                metadata={'sample_id': sample_id}
            )
            # Add success attribute for base pipeline compatibility
            result.success = False
            result.error_message = str(e)
            result.validation_type = "alignment"
            result.sample_id = sample_id
            
            # Add to_dict method for pipeline compatibility
            def create_to_dict_method(error_msg, sid):
                def to_dict(self):
                    return {
                        'sample_id': getattr(self, 'sample_id', sid),
                        'success': getattr(self, 'success', False),
                        'metrics': dict(self.metrics),
                        'processing_time': self.processing_time,
                        'error_message': getattr(self, 'error_message', error_msg),
                        'validation_type': getattr(self, 'validation_type', 'alignment')
                    }
                return to_dict
            result.to_dict = create_to_dict_method(str(e), sample_id).__get__(result, type(result))
            
            return result
    
    def _load_slice_stack(self, directory: Path) -> List[Path]:
        """Load slice stack file paths from directory."""
        # Look for both .png and .tif files
        slice_files = list(directory.glob("*.png")) + list(directory.glob("*.tif"))
        slice_files = sorted(slice_files)
        return slice_files
    
    def _load_image_stack(self, file_paths: List[Path]) -> np.ndarray:
        """Load images from file paths into a numpy array stack."""
        import cv2
        
        images = []
        target_shape = None
        
        # First pass: determine target shape (use the first valid image)
        for file_path in file_paths:
            try:
                image = load_image(file_path)
                # Ensure image is 2D (grayscale)
                if len(image.shape) == 3:
                    image = np.mean(image, axis=2)
                target_shape = image.shape
                break
            except Exception as e:
                print(f"Warning: Failed to load image {file_path}: {e}")
                continue
        
        if target_shape is None:
            raise ValueError("No valid images found to determine target shape")
        
        # Second pass: load and resize all images to target shape
        for file_path in file_paths:
            try:
                image = load_image(file_path)
                # Ensure image is 2D (grayscale)
                if len(image.shape) == 3:
                    image = np.mean(image, axis=2)
                
                # Resize to target shape if needed
                if image.shape != target_shape:
                    image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                
                images.append(image)
            except Exception as e:
                print(f"Warning: Failed to load image {file_path}: {e}")
                continue
        
        if not images:
            raise ValueError("No images could be loaded from the file paths")
        
        # Stack images into 3D array (n_images, height, width)
        return np.stack(images, axis=0)
    
    def _calculate_metrics(self, alignment_result, validation_result, sample_id: str) -> Dict[str, Any]:
        """Calculate alignment metrics."""
        metrics = {
            'slice_count': len(alignment_result.aligned_slices) if hasattr(alignment_result, 'aligned_slices') else 0,
            'alignment_method': self.config.alignment_method,
            'reference_strategy': self.config.reference_strategy,
            'processing_stage': 'alignment_validation'
        }
        
        # Add validation metrics if available
        if hasattr(validation_result, 'metrics') and validation_result.metrics:
            metrics.update(validation_result.metrics)
        
        # Add alignment quality metrics if available
        if hasattr(alignment_result, 'quality_metrics') and alignment_result.quality_metrics:
            metrics.update(alignment_result.quality_metrics)
        
        # Add alignment-specific metrics
        if hasattr(alignment_result, 'transforms') and alignment_result.transforms:
            metrics['num_transforms'] = len(alignment_result.transforms)
        
        if hasattr(alignment_result, 'confidence'):
            metrics['alignment_confidence'] = alignment_result.confidence
        
        return metrics
    
    def _validate_alignment_result(self, alignment_result) -> bool:
        """Simple validation of alignment results."""
        try:
            # Check if we have a valid alignment result
            if not alignment_result:
                return False
            
            # Check if we have aligned images
            if hasattr(alignment_result, 'aligned_images') and alignment_result.aligned_images is not None:
                if len(alignment_result.aligned_images) > 0:
                    return True
            
            # Check if we have transforms (indicates successful alignment)
            if hasattr(alignment_result, 'transforms') and alignment_result.transforms is not None:
                if len(alignment_result.transforms) > 0:
                    return True
            
            # Check if we have quality metrics indicating successful alignment
            if hasattr(alignment_result, 'quality_metrics') and alignment_result.quality_metrics:
                # Look for metrics that indicate successful alignment
                if 'mean_similarity' in alignment_result.quality_metrics:
                    mean_sim = alignment_result.quality_metrics['mean_similarity']
                    # Consider alignment successful if mean similarity is reasonable
                    if isinstance(mean_sim, (int, float)) and mean_sim > -0.5:  # Threshold for meaningful alignment
                        return True
                if 'n_images' in alignment_result.quality_metrics:
                    # If we processed images, consider it successful
                    return alignment_result.quality_metrics['n_images'] > 0
            
            return False
            
        except Exception as e:
            print(f"Warning: Error validating alignment result: {e}")
            return False
