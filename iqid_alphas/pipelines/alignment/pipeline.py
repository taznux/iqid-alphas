#!/usr/bin/env python3
"""
Alignment Pipeline Module

Main pipeline class for Segmented → Aligned validation.
Modularized for maintainability and follows <500 line guideline.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from ..base import BasePipeline
from .config import AlignmentPipelineConfig
from .processor import AlignmentProcessor
from ...core.batch_processing import BatchProcessor
from ...core.mapping import ReverseMapper


class AlignmentPipeline(BasePipeline):
    """Alignment Pipeline for Segmented → Aligned validation."""
    
    def __init__(self, data_path: Union[str, Path], output_dir: Union[str, Path],
                 config: Optional[AlignmentPipelineConfig] = None,
                 config_path: Optional[Path] = None):
        """
        Initialize alignment pipeline.
        
        Args:
            data_path: Path to ReUpload dataset directory
            output_dir: Path to output directory
            config: Pipeline configuration (optional)
            config_path: Path to configuration file (optional)
        """
        if config is None:
            config = AlignmentPipelineConfig.from_config_file(config_path)
        
        super().__init__(data_path, output_dir, config)
        
        # Initialize components
        self.processor = AlignmentProcessor(config)
        self.batch_processor = BatchProcessor()
        self.reverse_mapper = ReverseMapper()
        
        self.logger.info("AlignmentPipeline initialized successfully")
        self.logger.info(f"Alignment method: {config.alignment_method}")
        self.logger.info(f"Reference strategy: {config.reference_strategy}")
    
    def discover_samples(self) -> List[Dict[str, Any]]:
        """
        Discover alignment samples from ReUpload dataset.
        
        Looks for samples with both 1_segmented and 2_aligned directories.
        
        Returns:
            List of sample dictionaries with metadata
        """
        self.logger.info("Discovering alignment samples...")
        samples = []
        
        for sample_dir in self.data_path.iterdir():
            if not sample_dir.is_dir():
                continue
                
            segmented_dir = sample_dir / "1_segmented"
            aligned_dir = sample_dir / "2_aligned"
            
            if segmented_dir.exists() and aligned_dir.exists():
                # Check if directories contain images
                segmented_files = list(segmented_dir.glob("*.png")) + list(segmented_dir.glob("*.tif"))
                aligned_files = list(aligned_dir.glob("*.png")) + list(aligned_dir.glob("*.tif"))
                
                if segmented_files and aligned_files:
                    samples.append({
                        'id': sample_dir.name,
                        'sample_dir': sample_dir,
                        'segmented_dir': segmented_dir,
                        'aligned_dir': aligned_dir,
                        'processing_stage': 'alignment_validation',
                        'segmented_count': len(segmented_files),
                        'aligned_count': len(aligned_files)
                    })
        
        self.logger.info(f"Found {len(samples)} alignment samples")
        return samples
    
    def process_sample(self, sample: Dict[str, Any]):
        """
        Process single alignment sample.
        
        Args:
            sample: Sample dictionary with metadata
            
        Returns:
            ValidationResult with processing outcomes
        """
        return self.processor.process_sample(sample, self.logger)


def run_alignment_pipeline(data_path: str, output_dir: str, 
                          max_samples: Optional[int] = None,
                          config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the complete alignment pipeline.
    
    Args:
        data_path: Path to ReUpload dataset directory
        output_dir: Path to output directory  
        max_samples: Maximum number of samples to process (None = all)
        config_path: Path to configuration file (optional)
        
    Returns:
        Dictionary with pipeline execution results
    """
    # Load configuration
    config = AlignmentPipelineConfig.from_config_file(
        Path(config_path) if config_path else None
    )
    
    # Create and run pipeline
    pipeline = AlignmentPipeline(data_path, output_dir, config)
    result = pipeline.run_pipeline(max_samples)
    
    # Generate reports
    pipeline.generate_report(result)
    
    # Return summary
    return {
        "pipeline_name": "AlignmentPipeline",
        "total_samples": result.total_samples,
        "successful_samples": result.successful_samples,
        "failed_samples": result.failed_samples,
        "success_rate": result.success_rate,
        "duration": result.duration,
        "performance_metrics": result.performance_metrics,
        "output_dir": str(output_dir)
    }
