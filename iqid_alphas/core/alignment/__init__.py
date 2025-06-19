"""
Core Alignment Package

This package provides comprehensive slice alignment functionality organized into logical modules:

- types: Data structures, enums, and result classes
- algorithms: Core alignment algorithms (feature-based, intensity, phase correlation)
- utils: Utility functions and transformations
- aligner/: Package containing all aligner implementations
  - simple: Basic alignment functionality (SimpleAligner)
  - enhanced: Advanced alignment with multiple methods (EnhancedAligner) 
  - stack: Stack processing functionality (StackAligner)
  - unified: Common interface for all aligners (UnifiedAligner)

Usage Examples:

    # Simple alignment (lightweight, fast)
    from iqid_alphas.core.alignment.aligner import SimpleAligner
    aligner = SimpleAligner()
    aligned_image, metadata = aligner.align_images(fixed_image, moving_image)

    # Enhanced alignment (advanced features)
    from iqid_alphas.core.alignment.aligner import EnhancedAligner
    from iqid_alphas.core.alignment import AlignmentMethod
    aligner = EnhancedAligner()
    result = aligner.align_slice(moving_image, reference_image, method=AlignmentMethod.FEATURE_BASED)
    
    # Stack alignment
    from iqid_alphas.core.alignment.aligner import StackAligner
    from iqid_alphas.core.alignment import ReferenceStrategy
    aligner = StackAligner()
    result = aligner.align_stack(image_stack, reference_strategy=ReferenceStrategy.PREVIOUS_IMAGE)
    
    # Unified interface (recommended)
    from iqid_alphas.core.alignment.aligner import UnifiedAligner
    aligner = UnifiedAligner()
    result = aligner.align_images(moving_image, fixed_image)
    stack_result = aligner.align_stack(image_stack)
"""

from .types import AlignmentMethod, ReferenceStrategy, AlignmentResult, StackAlignmentResult
from .algorithms import (
    FeatureBasedAligner, 
    IntensityBasedAligner, 
    PhaseCorrelationAligner,
    CoarseRotationAligner
)
from .utils import calculate_ssd, create_affine_transform, assemble_image_stack

# Import aligners from the aligner package
from .aligner import (
    SimpleAligner,
    EnhancedAligner, 
    StackAligner,
    UnifiedAligner,
    align_images,
    align_stack
)

# For backwards compatibility, export the main aligner as ImageAligner too
ImageAligner = UnifiedAligner

__all__ = [
    # Types and enums
    'AlignmentMethod',
    'ReferenceStrategy', 
    'AlignmentResult',
    'StackAlignmentResult',
    
    # Algorithm classes
    'FeatureBasedAligner',
    'IntensityBasedAligner', 
    'PhaseCorrelationAligner',
    'CoarseRotationAligner',
    
    # Utility functions
    'calculate_ssd',
    'create_affine_transform',
    'assemble_image_stack',
    
    # Aligner classes
    'SimpleAligner',
    'EnhancedAligner',
    'StackAligner', 
    'UnifiedAligner',
    
    # Convenience functions
    'align_images',
    'align_stack',
    
    # Backwards compatibility
    'ImageAligner'
]
