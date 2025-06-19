"""
Alignment Types and Data Structures

This module contains the core data structures, enums, and result classes
used throughout the alignment package.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class AlignmentMethod(Enum):
    """Available alignment methods."""
    COARSE_ROTATION = "coarse_rotation"
    FEATURE_BASED = "feature_based" 
    INTENSITY_BASED = "intensity_based"
    PHASE_CORRELATION = "phase_correlation"
    HYBRID = "hybrid"


class ReferenceStrategy(Enum):
    """Reference image selection strategies."""
    FIRST_IMAGE = "first_image"           # Align all to first image
    PREVIOUS_IMAGE = "previous_image"     # Align each to previous
    ROLLING_AVERAGE = "rolling_average"   # Align to average of previous N images
    TEMPLATE_MATCHING = "template_matching"  # Find best template and align to it


@dataclass
class AlignmentResult:
    """Result from a single slice alignment operation."""
    transform_matrix: np.ndarray
    confidence: float
    method: AlignmentMethod
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class StackAlignmentResult:
    """Result from aligning a stack of images."""
    aligned_images: List[np.ndarray]
    transforms: List[np.ndarray]
    quality_metrics: Dict[str, float]
    reference_strategy: ReferenceStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
