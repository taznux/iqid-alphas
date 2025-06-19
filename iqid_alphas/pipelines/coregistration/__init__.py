"""
Coregistration Pipeline Package

Modular coregistration pipeline for iQID + H&E multi-modal registration.

Architecture:
- pipeline.py: Main CoregistrationPipeline class
- config.py: Configuration management
- processor.py: Core coregistration processing logic
- quality.py: Quality assessment without ground truth
- utils.py: Coregistration-specific utilities
"""

from .pipeline import CoregistrationPipeline
from .config import CoregistrationPipelineConfig

__all__ = [
    'CoregistrationPipeline',
    'CoregistrationPipelineConfig'
]
