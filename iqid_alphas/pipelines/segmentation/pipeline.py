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
    result = pipeline.run_pipeline(max_samples)
    
    # Generate reports
    pipeline.generate_report(result)
    if hasattr(result, 'sample_results'):
        pipeline.generate_detailed_report(result.sample_results)
    
    return {
        "pipeline_name": "SegmentationPipeline",
        "total_samples": result.total_samples,
        "successful_samples": result.successful_samples,
        "failed_samples": result.failed_samples,
        "success_rate": result.success_rate,
        "duration": result.duration,
        "performance_metrics": result.performance_metrics,
        "output_dir": str(output_dir)
    }
