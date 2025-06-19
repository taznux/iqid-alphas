#!/usr/bin/env python3
"""
Segmentation Pipeline Module

Main pipeline class for Raw → Segmented validation.
Modularized for maintainability and follows <500 line guideline.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from ..base import BasePipeline
from .config import SegmentationPipelineConfig
from .processor import SegmentationProcessor
from .utils import SegmentationDataLoader, SegmentationReporter
from ...core.batch_processing import BatchProcessor
from ...core.validation import ValidationSuite


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
        super().__init__(data_path, output_dir, config)
        
        # Initialize components
        self.processor = SegmentationProcessor(config)
        self.data_loader = SegmentationDataLoader(config)
        self.reporter = SegmentationReporter(config)
        self.validator = ValidationSuite()
        self.batch_processor = BatchProcessor(str(data_path), str(output_dir))
        
        self.logger.info("SegmentationPipeline initialized with Phase 1 components")
        self.logger.info(f"Segmentation method: {config.segmentation_method}")
    
    def discover_samples(self) -> List[Dict[str, Any]]:
        """
        Discover raw image samples for segmentation processing.
        
        Returns:
            List of sample dictionaries with metadata
        """
        self.logger.info("Discovering raw image samples...")
        
        data_path = Path(self.config.data_path)
        samples = self.data_loader.discover_samples(data_path)
        
        self.logger.info(f"Discovered {len(samples)} raw image samples")
        return samples
    
    def process_sample(self, sample: Dict[str, Any]):
        """
        Process a single raw image sample for tissue segmentation.
        
        Args:
            sample: Sample dictionary with metadata
            
        Returns:
            ValidationResult with segmentation outcomes
        """
        return self.processor.process_sample(sample, self.logger)
    
    def generate_detailed_report(self, results: List[Dict[str, Any]]) -> None:
        """
        Generate detailed segmentation pipeline report.
        
        Args:
            results: List of validation results
        """
        report_path = Path(self.config.output_dir) / "segmentation_detailed_report.md"
        self.reporter.generate_detailed_report(results, report_path)
        self.logger.info(f"Detailed report saved to: {report_path}")
    
    def run(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the segmentation pipeline.
        
        Args:
            max_samples: Maximum number of samples to process (optional)
        
        Returns:
            Dictionary with pipeline run results, including:
                - total_samples
                - successful_samples
                - failed_samples
                - success_rate
                - duration
                - output_dir
                - metrics (optional)
        """
        self.logger.info("Running segmentation pipeline...")
        
        # Discover samples
        samples = self.discover_samples()
        if max_samples is not None:
            samples = samples[:max_samples]
        
        # Process samples in batch
        results = self.batch_processor.process(samples, self.process_sample)
        
        # Generate reports
        self.generate_report(results)
        if hasattr(results, 'sample_results'):
            self.generate_detailed_report(results.sample_results)
        
        self.logger.info("Segmentation pipeline run completed")
        return {
            "total_samples": results.total_samples,
            "successful_samples": results.successful_samples,
            "failed_samples": results.failed_samples,
            "success_rate": results.success_rate,
            "duration": results.duration,
            "output_dir": str(self.config.output_dir),
            "metrics": results.performance_metrics if hasattr(results, 'performance_metrics') else {}
        }


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
    
    # Initialize and run pipeline
    pipeline = SegmentationPipeline(data_path, output_dir, config)
    result = pipeline.run(max_samples)
    
    return {
        "pipeline_name": "SegmentationPipeline",
        "total_samples": result["total_samples"],
        "successful_samples": result["successful_samples"],
        "failed_samples": result["failed_samples"],
        "success_rate": result["success_rate"],
        "duration": result["duration"],
        "output_dir": result["output_dir"]
    }
