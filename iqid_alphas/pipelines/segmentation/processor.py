#!/usr/bin/env python3
"""
Segmentation Processing Core Module

Handles the main segmentation processing operations.
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from ...core.tissue_segmentation import TissueSeparator
from ...core.validation import ValidationResult
from ...core.mapping import ReverseMapper
from ...utils.io import ensure_directory_exists
from .quality import SegmentationQualityValidator
from .utils import SegmentationDataLoader, SegmentationMetricsCalculator


class SegmentationProcessor:
    """Core segmentation processing operations."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize Phase 1 components
        self.tissue_separator = TissueSeparator(method=config.segmentation_method)
        self.reverse_mapper = ReverseMapper()
        
        # Initialize sub-components
        self.data_loader = SegmentationDataLoader(config)
        self.quality_validator = SegmentationQualityValidator(config)
        self.metrics_calculator = SegmentationMetricsCalculator(config)
    
    def process_sample(self, sample: Dict[str, Any], logger) -> ValidationResult:
        """
        Process a single raw image sample for tissue segmentation.
        
        Args:
            sample: Sample dictionary with metadata
            logger: Logger instance
            
        Returns:
            ValidationResult with segmentation outcomes
        """
        sample_id = sample["id"]
        raw_image_path = sample["raw_image_path"]
        ground_truth_dir = sample["ground_truth_dir"]
        
        start_time = time.time()
        
        try:
            logger.debug(f"Processing segmentation for sample: {sample_id}")
            
            # Perform tissue separation using Phase 1 component
            segmentation_results = self.tissue_separator.separate_tissues(str(raw_image_path))
            
            # Calculate basic metrics
            metrics = {
                "total_tissues_found": len(segmentation_results),
                "processing_time": time.time() - start_time
            }
            
            # Validate against ground truth if available
            if ground_truth_dir and sample["has_ground_truth"]:
                validation_metrics = self._validate_against_ground_truth(
                    segmentation_results, ground_truth_dir, sample_id, logger
                )
                metrics.update(validation_metrics)
            else:
                logger.warning(f"No ground truth available for sample {sample_id}")
                metrics.update({
                    "avg_iou": 0.0,
                    "avg_dice": 0.0,
                    "validation_status": "no_ground_truth"
                })
            
            # Save intermediate results if configured
            if self.config.save_intermediate:
                self._save_intermediate_results(sample_id, segmentation_results, logger)
            
            # Create validation result
            return ValidationResult(
                sample_id=sample_id,
                success=True,
                metrics=metrics,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process sample {sample_id}: {str(e)}")
            return ValidationResult(
                sample_id=sample_id,
                success=False,
                error_message=str(e),
                metrics={},
                processing_time=time.time() - start_time
            )
    
    def _validate_against_ground_truth(self, segmentation_results: Dict[str, List[Path]], 
                                     ground_truth_dir: Path, sample_id: str, logger) -> Dict[str, float]:
        """
        Validate segmentation results against ground truth using reverse mapping.
        
        Args:
            segmentation_results: Results from tissue separator
            ground_truth_dir: Directory containing ground truth data
            sample_id: Sample identifier
            logger: Logger instance
            
        Returns:
            Dictionary containing validation metrics
        """
        logger.debug(f"Validating {sample_id} against ground truth")
        
        try:
            # Use Phase 1 reverse mapper for validation
            validation_result = self.reverse_mapper.validate_segmentation_against_ground_truth(
                automated_results=segmentation_results,
                ground_truth_dir=ground_truth_dir
            )
            
            return {
                "avg_iou": validation_result.get("average_iou", 0.0),
                "avg_dice": validation_result.get("average_dice", 0.0),
                "avg_precision": validation_result.get("average_precision", 0.0),
                "avg_recall": validation_result.get("average_recall", 0.0),
                "total_tissues_expected": validation_result.get("total_tissues", 0),
                "validation_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Ground truth validation failed for {sample_id}: {str(e)}")
            return {
                "avg_iou": 0.0,
                "avg_dice": 0.0,
                "validation_status": "failed",
                "validation_error": str(e)
            }
    
    def _save_intermediate_results(self, sample_id: str, 
                                 segmentation_results: Dict[str, List[Path]], logger) -> None:
        """
        Save intermediate segmentation results.
        
        Args:
            sample_id: Sample identifier
            segmentation_results: Segmentation results to save
            logger: Logger instance
        """
        try:
            intermediate_dir = Path(self.config.output_dir) / "intermediate" / sample_id
            ensure_directory_exists(intermediate_dir)
            
            # Save segmentation result metadata
            metadata = {
                "sample_id": sample_id,
                "total_tissues": len(segmentation_results),
                "tissue_types": list(segmentation_results.keys()),
                "tissue_counts": {tissue: len(paths) for tissue, paths in segmentation_results.items()}
            }
            
            import json
            with open(intermediate_dir / "segmentation_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Saved intermediate results for {sample_id}")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate results for {sample_id}: {str(e)}")
    
    def validate_results(self, results: List[ValidationResult]) -> ValidationResult:
        """
        Validate overall pipeline results and calculate aggregate metrics.
        
        Args:
            results: List of individual sample results
            
        Returns:
            ValidationResult with aggregate metrics
        """
        if not results:
            return ValidationResult(
                sample_id="pipeline_aggregate",
                success=False,
                error_message="No results to validate",
                metrics={},
                processing_time=0.0
            )
        
        # Calculate aggregate metrics using metrics calculator
        aggregate_metrics = self.metrics_calculator.calculate_aggregate_metrics(results)
        
        successful_results = [r for r in results if r.success]
        
        return ValidationResult(
            sample_id="pipeline_aggregate",
            success=len(successful_results) > 0,
            metrics=aggregate_metrics,
            processing_time=sum(r.processing_time for r in results)
        )
