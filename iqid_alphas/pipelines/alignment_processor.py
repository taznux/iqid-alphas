#!/usr/bin/env python3
"""Alignment Pipeline Processing Module"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..core.alignment.aligner import UnifiedAligner
from ..core.alignment import AlignmentMethod, ReferenceStrategy
from ..core.validation import ValidationResult, ValidationSuite
from ..utils.io import ensure_directory_exists


class AlignmentProcessor:
    """Handles alignment processing operations."""
    
    def __init__(self, config):
        self.config = config
        self.aligner = UnifiedAligner(
            method=getattr(AlignmentMethod, config.alignment_algorithm.upper()),
            reference_strategy=getattr(ReferenceStrategy, config.reference_strategy.upper())
        )
        self.validator = ValidationSuite()
    
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
            
            alignment_result = self.aligner.align_slice_stack(segmented_slices)
            
            validation_result = self.validator.validate_alignment(
                predicted_slices=alignment_result.aligned_slices,
                ground_truth_slices=ground_truth_slices,
                sample_id=sample_id
            )
            
            metrics = self._calculate_metrics(alignment_result, validation_result, sample_id)
            
            return ValidationResult(
                sample_id=sample_id,
                success=True,
                metrics=metrics,
                processing_time=time.time() - start_time,
                validation_type="alignment"
            )
            
        except Exception as e:
            logger.error(f"Failed to process alignment sample {sample_id}: {e}")
            return ValidationResult(
                sample_id=sample_id,
                success=False,
                error_message=str(e),
                metrics={},
                processing_time=time.time() - start_time,
                validation_type="alignment"
            )
    
    def _load_slice_stack(self, directory: Path) -> List[Any]:
        """Load slice stack from directory."""
        slice_files = sorted(directory.glob("*.png"))
        return slice_files
    
    def _calculate_metrics(self, alignment_result, validation_result, sample_id: str) -> Dict[str, Any]:
        """Calculate alignment metrics."""
        metrics = {
            'slice_count': len(alignment_result.aligned_slices) if hasattr(alignment_result, 'aligned_slices') else 0,
            'alignment_method': self.config.alignment_method,
            'reference_strategy': self.config.reference_strategy,
            'processing_stage': 'alignment_validation'
        }
        
        if hasattr(validation_result, 'metrics'):
            metrics.update(validation_result.metrics)
        if hasattr(alignment_result, 'quality_metrics'):
            metrics.update(alignment_result.quality_metrics)
        
        return metrics
