#!/usr/bin/env python3
"""
Alignment Processing Core Module

Handles the main alignment processing operations.
Split from alignment_processor.py to maintain <500 line guideline.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from ...core.alignment.aligner import UnifiedAligner
from ...core.alignment import AlignmentMethod, ReferenceStrategy
from ...core.validation import ValidationResult, ValidationMode
from .quality import AlignmentQualityValidator
from .utils import AlignmentDataLoader, AlignmentMetricsCalculator


class AlignmentProcessor:
    """Core alignment processing operations."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize sub-components
        self.data_loader = AlignmentDataLoader(config)
        self.quality_validator = AlignmentQualityValidator(config)
        self.metrics_calculator = AlignmentMetricsCalculator(config)
        
        # Setup aligner with proper enum mappings
        self.aligner = self._create_aligner()
    
    def _create_aligner(self) -> UnifiedAligner:
        """Create and configure the UnifiedAligner."""
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
        
        alignment_method = method_mapping.get(
            self.config.alignment_algorithm.lower(), 
            AlignmentMethod.FEATURE_BASED
        )
        reference_strategy = strategy_mapping.get(
            self.config.reference_strategy.lower(), 
            ReferenceStrategy.FIRST_IMAGE
        )
        
        return UnifiedAligner(
            default_method=alignment_method,
            reference_strategy=reference_strategy
        )
    
    def process_sample(self, sample: Dict[str, Any], logger) -> ValidationResult:
        """Process a single alignment sample."""
        sample_id = sample['id']
        segmented_dir = sample['segmented_dir']
        aligned_dir = sample['aligned_dir']
        start_time = time.time()
        
        try:
            logger.debug(f"Processing alignment sample: {sample_id}")
            
            # Load and validate data
            segmented_slices = self.data_loader.load_slice_stack(segmented_dir)
            ground_truth_slices = self.data_loader.load_slice_stack(aligned_dir)
            
            if len(segmented_slices) < self.config.min_slice_count:
                raise ValueError(
                    f"Insufficient slices: {len(segmented_slices)} < {self.config.min_slice_count}"
                )
            
            logger.debug(
                f"Found {len(segmented_slices)} segmented slices and "
                f"{len(ground_truth_slices)} ground truth slices"
            )
            
            # Load actual image data for alignment
            image_stack = self.data_loader.load_image_stack(segmented_slices)
            
            # Perform alignment using the UnifiedAligner
            alignment_result = self.aligner.align_stack(image_stack)
            
            # Validate alignment quality
            alignment_success = self.quality_validator.validate_alignment_result(alignment_result)
            
            # Calculate comprehensive metrics
            metrics = self.metrics_calculator.calculate_metrics(
                alignment_result, None, sample_id
            )
            
            # Create validation result
            result = self._create_validation_result(
                sample_id, alignment_success, metrics, start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process alignment sample {sample_id}: {e}")
            return self._create_error_result(sample_id, str(e), start_time)
    
    def _create_validation_result(self, sample_id: str, success: bool, 
                                  metrics: Dict[str, Any], start_time: float) -> ValidationResult:
        """Create a successful validation result."""
        result = ValidationResult(
            mode=ValidationMode.ALIGNMENT,
            metrics=metrics,
            processing_time=time.time() - start_time,
            passed_tests=1 if success else 0,
            total_tests=1,
            metadata={'sample_id': sample_id, 'validation_type': 'alignment'}
        )
        return result

    def _create_error_result(self, sample_id: str, error_message: str, 
                             start_time: float) -> ValidationResult:
        """Create an error validation result."""
        result = ValidationResult(
            mode=ValidationMode.ALIGNMENT,
            metrics={},
            processing_time=time.time() - start_time,
            passed_tests=0,
            total_tests=1,
            errors=[error_message],
            metadata={'sample_id': sample_id, 'validation_type': 'alignment'}
        )
        return result
