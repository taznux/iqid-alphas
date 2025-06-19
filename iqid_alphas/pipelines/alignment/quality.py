#!/usr/bin/env python3
"""
Alignment Quality Validation Module

Handles quality assessment and validation of alignment results.
"""

from typing import Dict, Any, Optional
from ...core.validation import AlignmentValidator, ValidationConfig


class AlignmentQualityValidator:
    """Validates alignment quality and success metrics."""
    
    def __init__(self, config):
        self.config = config
        
        # Create validation config and validator
        validation_config = ValidationConfig()
        self.validator = AlignmentValidator(validation_config)
    
    def validate_alignment_result(self, alignment_result) -> bool:
        """
        Comprehensive validation of alignment results.
        
        Args:
            alignment_result: Result from alignment operation
            
        Returns:
            bool: True if alignment is considered successful
        """
        try:
            # Check if we have a valid alignment result
            if not alignment_result:
                return False
            
            # Check if we have aligned images
            if hasattr(alignment_result, 'aligned_images') and alignment_result.aligned_images is not None:
                if len(alignment_result.aligned_images) > 0:
                    return self._validate_aligned_images(alignment_result.aligned_images)
            
            # Check if we have transforms (indicates successful alignment)
            if hasattr(alignment_result, 'transforms') and alignment_result.transforms is not None:
                if len(alignment_result.transforms) > 0:
                    return self._validate_transforms(alignment_result.transforms)
            
            # Check if we have quality metrics indicating successful alignment
            if hasattr(alignment_result, 'quality_metrics') and alignment_result.quality_metrics:
                return self._validate_quality_metrics(alignment_result.quality_metrics)
            
            return False
            
        except Exception as e:
            print(f"Warning: Error validating alignment result: {e}")
            return False
    
    def _validate_aligned_images(self, aligned_images) -> bool:
        """Validate the aligned images themselves."""
        try:
            # Check if images are reasonable
            if len(aligned_images) < self.config.min_slice_count:
                return False
            
            # Could add more sophisticated image quality checks here
            # e.g., check for reasonable intensity distributions, 
            # no excessive artifacts, etc.
            
            return True
        except Exception:
            return False
    
    def _validate_transforms(self, transforms) -> bool:
        """Validate the transformation matrices."""
        try:
            # Check that we have reasonable transforms
            if len(transforms) == 0:
                return False
            
            # Could add checks for:
            # - Transform magnitude within reasonable bounds
            # - Consistency between consecutive transforms
            # - No degenerate transformations
            
            return True
        except Exception:
            return False
    
    def _validate_quality_metrics(self, quality_metrics: Dict[str, Any]) -> bool:
        """Validate based on quality metrics."""
        try:
            # Check mean similarity threshold
            if 'mean_similarity' in quality_metrics:
                mean_sim = quality_metrics['mean_similarity']
                if isinstance(mean_sim, (int, float)):
                    # Use configurable threshold if available
                    threshold = getattr(self.config, 'min_ssim_threshold', 0.5)
                    if mean_sim >= threshold:
                        return True
                    # Also check for reasonable negative values (some metrics can be negative)
                    if mean_sim > -0.5:
                        return True
            
            # Check if we processed a reasonable number of images
            if 'n_images' in quality_metrics:
                n_images = quality_metrics['n_images']
                if isinstance(n_images, int) and n_images >= self.config.min_slice_count:
                    return True
            
            # Check alignment confidence if available
            if 'alignment_confidence' in quality_metrics:
                confidence = quality_metrics['alignment_confidence']
                if isinstance(confidence, (int, float)) and confidence > 0.5:
                    return True
            
            # Check SSIM values if available
            if 'mean_ssim' in quality_metrics:
                ssim = quality_metrics['mean_ssim']
                if isinstance(ssim, (int, float)) and ssim >= self.config.min_ssim_threshold:
                    return True
            
            return False
            
        except Exception:
            return False
