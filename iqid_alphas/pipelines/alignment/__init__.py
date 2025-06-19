"""
Alignment Pipeline Package

Modular alignment pipeline for Segmented → Aligned validation.

Architecture:
- pipeline.py: Main AlignmentPipeline class
- config.py: Configuration management
- processor.py: Core processing logic
- quality.py: Quality metrics and validation
- utils.py: Alignment-specific utilities
"""

from .pipeline import AlignmentPipeline, run_alignment_pipeline
from .config import AlignmentPipelineConfig

__all__ = [
    'AlignmentPipeline',
    'run_alignment_pipeline', 
    'AlignmentPipelineConfig'
]
