"""
Segmentation Pipeline Package

Modular segmentation pipeline for Raw → Segmented validation.

Architecture:
- pipeline.py: Main SegmentationPipeline class
- config.py: Configuration management
- processor.py: Core segmentation processing logic
- quality.py: Quality assessment and validation
- utils.py: Segmentation-specific utilities
"""

from .pipeline import SegmentationPipeline, run_segmentation_pipeline
from .config import SegmentationPipelineConfig

__all__ = [
    'SegmentationPipeline',
    'run_segmentation_pipeline',
    'SegmentationPipelineConfig'
]
