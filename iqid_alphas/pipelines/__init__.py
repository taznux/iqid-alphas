"""
Processing pipelines for IQID-Alphas package.
"""

from .simple import SimplePipeline
from .advanced import AdvancedPipeline
from .combined import CombinedPipeline

# TODO: Import new validation pipelines when core modules are implemented
try:
    from .segmentation import SegmentationPipeline
    from .alignment import AlignmentPipeline
    from .coregistration import CoregistrationPipeline
    
    __all__ = [
        'SimplePipeline', 'AdvancedPipeline', 'CombinedPipeline',
        'SegmentationPipeline', 'AlignmentPipeline', 'CoregistrationPipeline'
    ]
except ImportError:
    # Core modules not yet implemented - only include existing pipelines
    __all__ = ['SimplePipeline', 'AdvancedPipeline', 'CombinedPipeline']
