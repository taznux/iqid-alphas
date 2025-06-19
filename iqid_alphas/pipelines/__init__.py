"""
IQID-Alphas Pipeline Package

Production-ready pipelines that leverage the modular core components.
Organized into focused packages following the <500 line architectural guideline.
"""

# Import legacy pipelines for backward compatibility
from .simple import SimplePipeline
from .advanced import AdvancedPipeline
from .combined import CombinedPipeline

# Import from modular packages
try:
    from .alignment import AlignmentPipeline, run_alignment_pipeline, AlignmentPipelineConfig
    from .segmentation import SegmentationPipeline, run_segmentation_pipeline, SegmentationPipelineConfig
    from .coregistration import CoregistrationPipeline, CoregistrationPipelineConfig
    
    __all__ = [
        # Legacy pipelines
        'SimplePipeline', 'AdvancedPipeline', 'CombinedPipeline',
        # Modular pipelines
        'AlignmentPipeline', 'run_alignment_pipeline', 'AlignmentPipelineConfig',
        'SegmentationPipeline', 'run_segmentation_pipeline', 'SegmentationPipelineConfig',
        'CoregistrationPipeline', 'CoregistrationPipelineConfig'
    ]
except ImportError as e:
    # Fallback to legacy pipelines if modular imports fail
    print(f"Warning: Failed to import modular pipelines: {e}")
    __all__ = ['SimplePipeline', 'AdvancedPipeline', 'CombinedPipeline']
