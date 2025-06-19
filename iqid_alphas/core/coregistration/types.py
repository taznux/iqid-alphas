"""
Data Types and Enums for Coregistration

Contains all the data structures, enums, and result classes
used throughout the coregistration package.
"""

from typing import Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class RegistrationMethod(Enum):
    """Available multi-modal registration methods."""
    MUTUAL_INFORMATION = "mutual_information"
    FEATURE_BASED = "feature_based"
    INTENSITY_BASED = "intensity_based"
    NORMALIZED_CROSS_CORRELATION = "normalized_cross_correlation"
    PHASE_CORRELATION = "phase_correlation"
    HYBRID = "hybrid"


class QualityMetric(Enum):
    """Available quality assessment metrics."""
    MUTUAL_INFORMATION = "mutual_information"
    NORMALIZED_MUTUAL_INFORMATION = "normalized_mutual_information"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    CROSS_CORRELATION = "cross_correlation"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    PEAK_SIGNAL_NOISE_RATIO = "peak_signal_noise_ratio"


@dataclass
class RegistrationResult:
    """Result from multi-modal registration."""
    transform_matrix: np.ndarray
    registered_image: np.ndarray
    confidence: float
    method: RegistrationMethod
    processing_time: float
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityPair:
    """Container for a pair of images from different modalities."""
    modality1_image: np.ndarray
    modality2_image: np.ndarray
    modality1_name: str = "modality1"
    modality2_name: str = "modality2"
    preprocessing_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
