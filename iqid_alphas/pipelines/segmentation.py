#!/usr/bin/env python3
"""
Segmentation Pipeline for IQID-Alphas

Implements Raw → Segmented validation pipeline using Phase 1 tissue segmentation
components. Validates automated tissue separation against ground truth data.

Architecture: Inherits from BasePipeline, integrates Phase 1 components.
Module size: <500 lines following architectural guidelines.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .base import BasePipeline, PipelineConfig
from ..core.tissue_segmentation import TissueSeparator
from ..core.validation import ValidationResult, ValidationSuite
from ..core.batch_processing import BatchProcessor
from ..core.mapping import ReverseMapper
from ..configs import load_pipeline_config
from ..utils.io import ensure_directory_exists


@dataclass
class SegmentationPipelineConfig(PipelineConfig):
    """Segmentation-specific pipeline configuration."""
    
    # Segmentation parameters
    segmentation_method: str = "adaptive_clustering"
    min_blob_area: int = 50
    max_blob_area: int = 50000
    preserve_blobs: bool = True
    use_clustering: bool = True
    
    # Quality control
    min_tissue_area: int = 100
    max_tissue_area: int = 1000000
    aspect_ratio_threshold: float = 10.0
    
    @classmethod
    def from_config_file(cls, config_path: Optional[Path] = None) -> 'SegmentationPipelineConfig':
        """Load configuration from file."""
        config_dict = load_pipeline_config("segmentation", config_path)
        
        # Extract pipeline-specific parameters
        seg_config = config_dict.get("segmentation", {})
        pipeline_config = config_dict.get("pipeline", {})
        processing_config = config_dict.get("processing", {})
        
        return cls(
            data_path="",  # Will be set during initialization
            output_dir="",  # Will be set during initialization
            max_samples=processing_config.get("batch_size"),
            parallel_workers=pipeline_config.get("max_workers", 4),
            memory_limit_gb=pipeline_config.get("memory_limit_gb"),
            validation_enabled=config_dict.get("validation", {}).get("generate_visualizations", True),
            generate_visualizations=config_dict.get("validation", {}).get("generate_visualizations", True),
            log_level=pipeline_config.get("log_level", "INFO"),
            segmentation_method=seg_config.get("method", "adaptive_clustering"),
            min_blob_area=seg_config.get("min_blob_area", 50),
            max_blob_area=seg_config.get("max_blob_area", 50000),
            preserve_blobs=seg_config.get("preserve_blobs", True),
            use_clustering=seg_config.get("use_clustering", True),
            min_tissue_area=config_dict.get("quality_control", {}).get("min_tissue_area", 100),
            max_tissue_area=config_dict.get("quality_control", {}).get("max_tissue_area", 1000000),
            aspect_ratio_threshold=config_dict.get("quality_control", {}).get("aspect_ratio_threshold", 10.0)
        )


class SegmentationPipeline(BasePipeline):
    """
    Raw → Segmented validation pipeline.
    
    Processes multi-tissue raw iQID images and validates automated tissue
    separation against ground truth data using Phase 1 components.
    """
    
    def __init__(self, data_path: Union[str, Path], output_dir: Union[str, Path],
                 config: Optional[SegmentationPipelineConfig] = None,
                 config_path: Optional[Path] = None):
        """
        Initialize segmentation pipeline.
        
        Args:
            data_path: Path to ReUpload dataset
            output_dir: Output directory for results  
            config: Pipeline configuration (optional)
            config_path: Path to configuration file (optional)
        """
        # Load configuration if not provided
        if config is None:
            config = SegmentationPipelineConfig.from_config_file(config_path)
        
        # Set data and output paths
        config.data_path = str(data_path)
        config.output_dir = str(output_dir)
        
        # Initialize base pipeline
        super().__init__(config)
        
        # Initialize Phase 1 components
        self.tissue_separator = TissueSeparator(method=config.segmentation_method)
        self.validator = ValidationSuite()
        self.batch_processor = BatchProcessor(str(data_path), str(output_dir))
        self.reverse_mapper = ReverseMapper()
        
        self.logger.info("SegmentationPipeline initialized with Phase 1 components")
    
    def discover_samples(self) -> List[Dict[str, Any]]:
        """
        Discover raw image samples for segmentation processing.
        
        Returns:
            List of sample dictionaries with metadata
        """
        self.logger.info("Discovering raw image samples...")
        
        samples = []
        data_path = Path(self.config.data_path)
        
        # Look for raw image files (typically in raw/ or root directory)
        raw_patterns = ["raw/*.tiff", "raw/*.tif", "*.tiff", "*.tif"]
        
        for pattern in raw_patterns:
            raw_files = list(data_path.glob(pattern))
            if raw_files:
                break
        
        if not raw_files:
            raise FileNotFoundError(f"No raw image files found in {data_path}")
        
        for raw_file in raw_files:
            # Extract sample ID (e.g., D1M1, D2M2, etc.)
            sample_id = raw_file.stem
            
            # Look for corresponding ground truth directory
            gt_dir = data_path / "1_segmented" / sample_id
            
            sample_info = {
                "id": sample_id,
                "raw_image_path": raw_file,
                "ground_truth_dir": gt_dir if gt_dir.exists() else None,
                "has_ground_truth": gt_dir.exists()
            }
            
            samples.append(sample_info)
        
        self.logger.info(f"Discovered {len(samples)} raw image samples")
        return samples
    
    def process_sample(self, sample: Dict[str, Any]) -> ValidationResult:
        """
        Process a single raw image sample for tissue segmentation.
        
        Args:
            sample: Sample dictionary with metadata
            
        Returns:
            ValidationResult with segmentation outcomes
        """
        sample_id = sample["id"]
        raw_image_path = sample["raw_image_path"]
        ground_truth_dir = sample["ground_truth_dir"]
        
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing segmentation for sample: {sample_id}")
            
            # Perform tissue separation using Phase 1 component
            segmentation_results = self.tissue_separator.separate_tissues(str(raw_image_path))
            
            # Initialize metrics
            metrics = {
                "total_tissues_found": len(segmentation_results),
                "processing_time": time.time() - start_time
            }
            
            # Validate against ground truth if available
            if ground_truth_dir and sample["has_ground_truth"]:
                validation_metrics = self._validate_against_ground_truth(
                    segmentation_results, ground_truth_dir, sample_id
                )
                metrics.update(validation_metrics)
            else:
                self.logger.warning(f"No ground truth available for sample {sample_id}")
                metrics.update({
                    "avg_iou": 0.0,
                    "avg_dice": 0.0,
                    "validation_status": "no_ground_truth"
                })
            
            # Save intermediate results if configured
            if self.config.extra_params.get("save_intermediate", True):
                self._save_intermediate_results(sample_id, segmentation_results)
            
            return ValidationResult(
                sample_id=sample_id,
                success=True,
                metrics=metrics,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process sample {sample_id}: {str(e)}")
            return ValidationResult(
                sample_id=sample_id,
                success=False,
                error_message=str(e),
                metrics={},
                processing_time=time.time() - start_time
            )
    
    def validate_results(self, results: List[Dict[str, Any]]) -> ValidationResult:
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
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return ValidationResult(
                sample_id="pipeline_aggregate",
                success=False,
                error_message="No successful results",
                metrics={"total_samples": len(results), "successful_samples": 0},
                processing_time=0.0
            )
        
        # Extract metrics from successful results
        iou_scores = [r.metrics.get("avg_iou", 0.0) for r in successful_results 
                     if "avg_iou" in r.metrics]
        dice_scores = [r.metrics.get("avg_dice", 0.0) for r in successful_results 
                      if "avg_dice" in r.metrics]
        processing_times = [r.processing_time for r in successful_results]
        
        aggregate_metrics = {
            "total_samples": len(results),
            "successful_samples": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0.0,
        }
        
        if iou_scores:
            aggregate_metrics.update({
                "avg_iou": sum(iou_scores) / len(iou_scores),
                "min_iou": min(iou_scores),
                "max_iou": max(iou_scores)
            })
        
        if dice_scores:
            aggregate_metrics.update({
                "avg_dice": sum(dice_scores) / len(dice_scores),
                "min_dice": min(dice_scores),
                "max_dice": max(dice_scores)
            })
        
        return ValidationResult(
            sample_id="pipeline_aggregate",
            success=True,
            metrics=aggregate_metrics,
            processing_time=sum(processing_times)
        )
    
    def _validate_against_ground_truth(self, segmentation_results: Dict[str, List[Path]], 
                                     ground_truth_dir: Path, sample_id: str) -> Dict[str, float]:
        """
        Validate segmentation results against ground truth using reverse mapping.
        
        Args:
            segmentation_results: Results from tissue separator
            ground_truth_dir: Directory containing ground truth data
            sample_id: Sample identifier
            
        Returns:
            Dictionary containing validation metrics
        """
        self.logger.debug(f"Validating {sample_id} against ground truth")
        
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
            self.logger.error(f"Ground truth validation failed for {sample_id}: {str(e)}")
            return {
                "avg_iou": 0.0,
                "avg_dice": 0.0,
                "validation_status": "failed",
                "validation_error": str(e)
            }
    
    def _save_intermediate_results(self, sample_id: str, 
                                 segmentation_results: Dict[str, List[Path]]) -> None:
        """
        Save intermediate segmentation results.
        
        Args:
            sample_id: Sample identifier
            segmentation_results: Segmentation results to save
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
            
            self.logger.debug(f"Saved intermediate results for {sample_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results for {sample_id}: {str(e)}")
    
    def generate_detailed_report(self, results: List[ValidationResult]) -> None:
        """
        Generate detailed segmentation pipeline report.
        
        Args:
            results: List of validation results
        """
        report_path = Path(self.config.output_dir) / "segmentation_detailed_report.md"
        
        try:
            with open(report_path, 'w') as f:
                f.write("# Segmentation Pipeline Detailed Report\n\n")
                f.write(f"**Pipeline**: Raw → Segmented Validation\n")
                f.write(f"**Dataset**: {self.config.data_path}\n")
                f.write(f"**Total Samples**: {len(results)}\n")
                
                successful_results = [r for r in results if r.success]
                f.write(f"**Successful**: {len(successful_results)}\n")
                f.write(f"**Failed**: {len(results) - len(successful_results)}\n\n")
                
                # Summary statistics
                if successful_results:
                    iou_scores = [r.metrics.get("avg_iou", 0.0) for r in successful_results 
                                 if "avg_iou" in r.metrics]
                    if iou_scores:
                        f.write("## Quality Metrics Summary\n\n")
                        f.write(f"- **Average IoU**: {sum(iou_scores)/len(iou_scores):.3f}\n")
                        f.write(f"- **Min IoU**: {min(iou_scores):.3f}\n")
                        f.write(f"- **Max IoU**: {max(iou_scores):.3f}\n\n")
                
                # Individual sample results
                f.write("## Individual Sample Results\n\n")
                for result in results:
                    status = "✅ SUCCESS" if result.success else "❌ FAILED"
                    f.write(f"### {result.sample_id} - {status}\n\n")
                    
                    if result.success and result.metrics:
                        f.write("**Metrics:**\n")
                        for metric, value in result.metrics.items():
                            if isinstance(value, float):
                                f.write(f"- {metric}: {value:.4f}\n")
                            else:
                                f.write(f"- {metric}: {value}\n")
                    
                    if result.error_message:
                        f.write(f"**Error:** {result.error_message}\n")
                    
                    f.write(f"**Processing Time:** {result.processing_time:.2f}s\n\n")
            
            self.logger.info(f"Detailed report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate detailed report: {str(e)}")


# Convenience function for CLI usage
def run_segmentation_pipeline(data_path: str, output_dir: str, 
                            max_samples: Optional[int] = None,
                            config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run segmentation pipeline.
    
    Args:
        data_path: Path to input data
        output_dir: Output directory
        max_samples: Maximum samples to process
        config_path: Optional configuration file
        
    Returns:
        Pipeline execution summary
    """
    # Load configuration
    config = SegmentationPipelineConfig.from_config_file(
        Path(config_path) if config_path else None
    )
    
    if max_samples:
        config.max_samples = max_samples
    
    # Initialize and run pipeline
    pipeline = SegmentationPipeline(data_path, output_dir, config)
    result = pipeline.run()
    
    # Generate detailed report
    pipeline.generate_detailed_report(result.sample_results)
    
    return {
        "pipeline_name": "SegmentationPipeline",
        "total_samples": result.total_samples,
        "successful_samples": result.successful_samples,
        "failed_samples": result.failed_samples,
        "success_rate": result.success_rate,
        "duration": result.duration,
        "metrics": result.metrics,
        "output_dir": str(output_dir)
    }


# Main CLI entry point
def main():
    """Main entry point for segmentation pipeline CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IQID-Alphas Segmentation Pipeline")
    parser.add_argument("--data", required=True, help="Path to ReUpload dataset")
    parser.add_argument("--output", default="outputs/segmentation", help="Output directory")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to process")
    
    args = parser.parse_args()
    
    # Run pipeline
    result = run_segmentation_pipeline(
        data_path=args.data,
        output_dir=args.output,
        max_samples=args.max_samples,
        config_path=args.config
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SEGMENTATION PIPELINE RESULTS")
    print("="*60)
    print(f"📊 Pipeline: {result['pipeline_name']}")
    print(f"📊 Total samples: {result['total_samples']}")
    print(f"✅ Successful: {result['successful_samples']}")
    print(f"❌ Failed: {result['failed_samples']}")
    print(f"📈 Success rate: {result['success_rate']:.1%}")
    print(f"⏱️  Duration: {result['duration']:.2f}s")
    print(f"📁 Output: {result['output_dir']}")


if __name__ == "__main__":
    main()
