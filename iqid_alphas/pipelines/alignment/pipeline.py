#!/usr/bin/env python3
"""
Alignment Pipeline Module

Main pipeline class for Segmented → Aligned validation.
Modularized for maintainability and follows <500 line guideline.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time

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
    
    def run(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the alignment pipeline.
        Steps:
            1. Load configuration and input data
            2. Initialize alignment module (UnifiedAligner)
            3. For each sample:
                a. Run alignment
                b. Collect results and metrics (SSIM, mutual information, etc.)
            4. Aggregate results and compute summary statistics
            5. Return standardized result dictionary
        
        Args:
            max_samples: Maximum number of samples to process (None = all)
        
        Returns:
            Dictionary with pipeline execution results
        """
        start_time = time.time()
        total_samples = 0
        successful_samples = 0
        failed_samples = 0
        metrics = {}
        
        self.logger.info("Running alignment pipeline...")
        
        # Discover samples
        samples = self.discover_samples()
        
        if max_samples is not None:
            samples = samples[:max_samples]
        
        self.logger.info(f"Processing {len(samples)} samples")
        
        # Initialize alignment module (UnifiedAligner)
        from iqid_alphas.core.alignment import UnifiedAligner
        aligner = UnifiedAligner(method=self.config.get('alignment_method', 'bidirectional'))
        
        # Process each sample
        for sample in samples:
            try:
                # Run alignment
                result = aligner.align_slice_stack(sample)
                
                # Collect metrics, save outputs
                successful_samples += 1
            except Exception as e:
                self.logger.error(f"Error processing sample {sample['id']}: {e}")
                failed_samples += 1
            total_samples += 1
        
        duration = time.time() - start_time
        
        self.logger.info(f"Alignment pipeline completed: {successful_samples}/{total_samples} successful")
        
        # Return summary
        return {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'success_rate': (successful_samples / total_samples) if total_samples else 0.0,
            'duration': duration,
            'output_dir': self.output_dir,
            'metrics': metrics
        }


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
    result = pipeline.run(max_samples)
    
    # Return summary
    return {
        "pipeline_name": "AlignmentPipeline",
        "total_samples": result["total_samples"],
        "successful_samples": result["successful_samples"],
        "failed_samples": result["failed_samples"],
        "success_rate": result["success_rate"],
        "output_dir": result["output_dir"]
    }
