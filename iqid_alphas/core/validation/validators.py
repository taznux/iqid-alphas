"""
Individual Validation Modules

Contains specialized validators for different pipeline components:
segmentation, alignment, and coregistration.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .types import ValidationMode, ValidationResult, ValidationConfig
from .metrics import (
    calculate_segmentation_metrics,
    calculate_alignment_metrics, 
    calculate_coregistration_metrics
)
from .utils import (
    find_images,
    find_corresponding_raw_image,
    find_corresponding_segmented_image
)
from ..mapping import ReverseMapper
from ..tissue_segmentation import TissueSeparator
from ..alignment.aligner import UnifiedAligner
from ..coregistration import MultiModalAligner, RegistrationMethod
from ...utils.io import load_image

logger = logging.getLogger(__name__)


class SegmentationValidator:
    """Validates segmentation accuracy using reverse mapping."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.reverse_mapper = ReverseMapper()
        self.logger = logging.getLogger(__name__)
    
    def validate(self, data_path: Path) -> ValidationResult:
        """Validate segmentation accuracy."""
        result = ValidationResult(mode=ValidationMode.SEGMENTATION)
        start_time = time.time()
        
        try:
            # Find raw and segmented images
            raw_images = find_images(data_path / "raw", self.config.raw_image_patterns)
            segmented_images = find_images(data_path / "1_segmented", self.config.supported_extensions)
            
            if not raw_images or not segmented_images:
                result.errors.append("Could not find raw or segmented images")
                return result
            
            result.total_tests = len(segmented_images)
            
            for seg_image_path in segmented_images:
                try:
                    # Load segmented image
                    seg_image = load_image(seg_image_path)
                    
                    # Find corresponding raw image
                    raw_image_path = find_corresponding_raw_image(seg_image_path, raw_images)
                    if not raw_image_path:
                        result.warnings.append(f"No corresponding raw image for {seg_image_path.name}")
                        continue
                    
                    raw_image = load_image(raw_image_path)
                    
                    # Use reverse mapping to find crop location
                    crop_result = self.reverse_mapper.find_crop_location(raw_image, seg_image)
                    
                    # Validate crop location
                    if crop_result.confidence > self.config.segmentation_confidence_threshold:
                        result.passed_tests += 1
                        
                        # Calculate segmentation quality metrics
                        quality_metrics = calculate_segmentation_metrics(
                            raw_image, seg_image, crop_result
                        )
                        
                        # Add to overall metrics
                        for metric_name, value in quality_metrics.items():
                            if metric_name not in result.metrics:
                                result.metrics[metric_name] = []
                            result.metrics[metric_name].append(value)
                    else:
                        result.warnings.append(f"Low confidence crop location for {seg_image_path.name}")
                
                except Exception as e:
                    result.errors.append(f"Error processing {seg_image_path.name}: {e}")
            
            # Calculate average metrics
            for metric_name, values in result.metrics.items():
                if values:
                    result.metrics[metric_name] = np.mean(values)
            
        except Exception as e:
            result.errors.append(f"Segmentation validation failed: {e}")
        
        finally:
            result.processing_time = time.time() - start_time
        
        return result


class AlignmentValidator:
    """Validates alignment accuracy using reverse mapping."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.reverse_mapper = ReverseMapper()
        self.logger = logging.getLogger(__name__)
    
    def validate(self, data_path: Path) -> ValidationResult:
        """Validate alignment accuracy."""
        result = ValidationResult(mode=ValidationMode.ALIGNMENT)
        start_time = time.time()
        
        try:
            # Find segmented and aligned images
            segmented_images = find_images(data_path / "1_segmented", self.config.supported_extensions)
            aligned_images = find_images(data_path / "2_aligned", self.config.supported_extensions)
            
            if not segmented_images or not aligned_images:
                result.errors.append("Could not find segmented or aligned images")
                return result
            
            result.total_tests = len(aligned_images)
            
            for aligned_image_path in aligned_images:
                try:
                    # Load aligned image
                    aligned_image = load_image(aligned_image_path)
                    
                    # Find corresponding segmented image
                    seg_image_path = find_corresponding_segmented_image(aligned_image_path, segmented_images)
                    if not seg_image_path:
                        result.warnings.append(f"No corresponding segmented image for {aligned_image_path.name}")
                        continue
                    
                    seg_image = load_image(seg_image_path)
                    
                    # Calculate alignment transform
                    transform_result = self.reverse_mapper.calculate_alignment_transform(
                        seg_image, aligned_image
                    )
                    
                    # Validate alignment quality
                    if transform_result.match_quality > self.config.alignment_quality_threshold:
                        result.passed_tests += 1
                        
                        # Calculate alignment quality metrics
                        quality_metrics = calculate_alignment_metrics(
                            seg_image, aligned_image, transform_result
                        )
                        
                        # Add to overall metrics
                        for metric_name, value in quality_metrics.items():
                            if metric_name not in result.metrics:
                                result.metrics[metric_name] = []
                            result.metrics[metric_name].append(value)
                    else:
                        result.warnings.append(f"Poor alignment quality for {aligned_image_path.name}")
                
                except Exception as e:
                    result.errors.append(f"Error processing {aligned_image_path.name}: {e}")
            
            # Calculate average metrics
            for metric_name, values in result.metrics.items():
                if values:
                    result.metrics[metric_name] = np.mean(values)
            
        except Exception as e:
            result.errors.append(f"Alignment validation failed: {e}")
        
        finally:
            result.processing_time = time.time() - start_time
        
        return result


class CoregistrationValidator:
    """Validates coregistration accuracy between modalities."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.multimodal_aligner = MultiModalAligner()
        self.unified_aligner = UnifiedAligner()
        self.logger = logging.getLogger(__name__)
    
    def validate(self, data_path: Path) -> ValidationResult:
        """Validate coregistration accuracy."""
        result = ValidationResult(mode=ValidationMode.COREGISTRATION)
        start_time = time.time()
        
        try:
            # Find different modality images for coregistration testing
            iqid_images = find_images(data_path / "iqid", self.config.supported_extensions)
            he_images = find_images(data_path / "he", self.config.supported_extensions)
            
            if not iqid_images or not he_images:
                result.warnings.append("Could not find multi-modal images for coregistration validation")
                # Try with segmented images as different modalities
                iqid_images = find_images(data_path / "1_segmented", self.config.supported_extensions)
                he_images = find_images(data_path / "2_aligned", self.config.supported_extensions)
            
            if not iqid_images or not he_images:
                result.errors.append("Could not find images for coregistration validation")
                return result
            
            result.total_tests = min(len(iqid_images), len(he_images))
            
            for i in range(result.total_tests):
                try:
                    iqid_image = load_image(iqid_images[i])
                    he_image = load_image(he_images[i])
                    
                    # Test different registration methods
                    methods = [
                        RegistrationMethod.MUTUAL_INFORMATION,
                        RegistrationMethod.NORMALIZED_CROSS_CORRELATION,
                        RegistrationMethod.FEATURE_BASED
                    ]
                    
                    best_confidence = 0.0
                    best_result = None
                    
                    for method in methods:
                        try:
                            reg_result = self.multimodal_aligner.register_images(
                                iqid_image, he_image, method=method
                            )
                            
                            if reg_result.confidence > best_confidence:
                                best_confidence = reg_result.confidence
                                best_result = reg_result
                        
                        except Exception as e:
                            result.warnings.append(f"Method {method.value} failed: {e}")
                    
                    # Check if best result is acceptable
                    if best_result and best_result.confidence > self.config.coregistration_confidence_threshold:
                        result.passed_tests += 1
                        
                        # Add quality metrics
                        for metric_name, value in best_result.quality_metrics.items():
                            if metric_name not in result.metrics:
                                result.metrics[metric_name] = []
                            result.metrics[metric_name].append(value)
                
                except Exception as e:
                    result.errors.append(f"Error processing image pair {i}: {e}")
            
            # Calculate average metrics
            for metric_name, values in result.metrics.items():
                if values:
                    result.metrics[metric_name] = np.mean(values)
            
        except Exception as e:
            result.errors.append(f"Coregistration validation failed: {e}")
        
        finally:
            result.processing_time = time.time() - start_time
        
        return result
