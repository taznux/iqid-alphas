#!/usr/bin/env python3
"""Alignment Pipeline for IQID-Alphas"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .base import BasePipeline
from .alignment_config import AlignmentPipelineConfig
from .alignment_processor import AlignmentProcessor
from ..core.batch_processing import BatchProcessor
from ..core.mapping import ReverseMapper


class AlignmentPipeline(BasePipeline):
    """Alignment Pipeline for Segmented → Aligned validation."""
    
    def __init__(self, data_path: Union[str, Path], output_dir: Union[str, Path],
                 config: Optional[AlignmentPipelineConfig] = None,
                 config_path: Optional[Path] = None):
        if config is None:
            config = AlignmentPipelineConfig.from_config_file(config_path)
        
        super().__init__(data_path, output_dir, config)
        
        self.processor = AlignmentProcessor(config)
        self.batch_processor = BatchProcessor()
        self.reverse_mapper = ReverseMapper()
        
        self.logger.info("AlignmentPipeline initialized")
    
    def discover_samples(self) -> List[Dict[str, Any]]:
        """Discover alignment samples from dataset."""
        self.logger.info("Discovering alignment samples...")
        samples = []
        
        for sample_dir in self.data_path.iterdir():
            if not sample_dir.is_dir():
                continue
                
            segmented_dir = sample_dir / "1_segmented"
            aligned_dir = sample_dir / "2_aligned"
            
            if segmented_dir.exists() and aligned_dir.exists():
                samples.append({
                    'id': sample_dir.name,
                    'sample_dir': sample_dir,
                    'segmented_dir': segmented_dir,
                    'aligned_dir': aligned_dir,
                    'processing_stage': 'alignment_validation'
                })
        
        self.logger.info(f"Found {len(samples)} alignment samples")
        return samples
    
    def process_sample(self, sample: Dict[str, Any]):
        """Process single alignment sample."""
        return self.processor.process_sample(sample, self.logger)


def run_alignment_pipeline(data_path: str, output_dir: str, 
                          max_samples: Optional[int] = None,
                          config_path: Optional[str] = None) -> Dict[str, Any]:
    """Run alignment pipeline."""
    config = AlignmentPipelineConfig.from_config_file(
        Path(config_path) if config_path else None
    )
    
    pipeline = AlignmentPipeline(data_path, output_dir, config)
    result = pipeline.run_pipeline(max_samples)
    pipeline.generate_report(result)
    
    return {
        "pipeline_name": "AlignmentPipeline",
        "total_samples": result.total_samples,
        "successful_samples": result.successful_samples,
        "success_rate": result.success_rate,
        "duration": result.duration,
        "output_dir": str(output_dir)
    }
