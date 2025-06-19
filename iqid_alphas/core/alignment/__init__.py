"""
Core Alignment Package

This package provides comprehensive slice alignment functionality split into logical modules:

- types: Data structures, enums, and result classes
- algorithms: Core alignment algorithms (feature-based, intensity, phase correlation)
- stack: Stack processing and multi-image alignment
- utils: Utility functions and transformations
- aligner: Main EnhancedAligner orchestrator class

Usage:
    from iqid_alphas.core.alignment import EnhancedAligner, AlignmentMethod
    
    aligner = EnhancedAligner()
    result = aligner.align_slice(moving_image, reference_image, method=AlignmentMethod.FEATURE_BASED)
"""

from .types import AlignmentMethod, ReferenceStrategy, AlignmentResult, StackAlignmentResult
from .types import AlignmentMethod, ReferenceStrategy, AlignmentResult, StackAlignmentResult
from .algorithms import (
    FeatureBasedAligner, 
    IntensityBasedAligner, 
    PhaseCorrelationAligner,
    CoarseRotationAligner
)
from .stack import StackAligner
from .utils import calculate_ssd, create_affine_transform, assemble_image_stack
from .aligner import EnhancedAligner

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
    
    # Stack processing
    'StackAligner',
    
    # Utility functions
    'calculate_ssd',
    'create_affine_transform',
    'assemble_image_stack',
    
    # Main aligner
    'EnhancedAligner'
]
