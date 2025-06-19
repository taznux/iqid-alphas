"""
Validation Types and Enums

Contains all data structures, enums, and configuration classes
used throughout the validation package.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ValidationMode(Enum):
    """Available validation modes."""
    SEGMENTATION = "segmentation"
    ALIGNMENT = "alignment"
    COREGISTRATION = "coregistration"
    COMPREHENSIVE = "comprehensive"


class MetricType(Enum):
    """Types of validation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    MUTUAL_INFORMATION = "mutual_information"
    CROSS_CORRELATION = "cross_correlation"
    TRANSFORMATION_ERROR = "transformation_error"
    DICE_COEFFICIENT = "dice_coefficient"


@dataclass
class ValidationResult:
    """Result from a validation run."""
    mode: ValidationMode
    metrics: Dict[str, float] = field(default_factory=dict)
    passed_tests: int = 0
    total_tests: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100.0
    
    @property
    def is_passing(self) -> bool:
        """Check if validation is passing (>= 90% success rate)."""
        return self.success_rate >= 90.0


@dataclass
class GroundTruth:
    """Container for ground truth data."""
    segmentation_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    alignment_transforms: Dict[str, np.ndarray] = field(default_factory=dict)
    registration_transforms: Dict[str, np.ndarray] = field(default_factory=dict)
    expected_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Configuration for validation operations."""
    tolerance: float = 0.1
    min_success_rate: float = 90.0
    segmentation_confidence_threshold: float = 0.8
    alignment_quality_threshold: float = 0.7
    coregistration_confidence_threshold: float = 0.5
    
    # File patterns for discovery
    raw_image_patterns: List[str] = field(default_factory=lambda: ["*.tif", "*.tiff", "*.png", "*.jpg"])
    supported_extensions: List[str] = field(default_factory=lambda: [".tif", ".tiff", ".png", ".jpg", ".jpeg"])
    
    # Validation modes to run
    enable_segmentation_validation: bool = True
    enable_alignment_validation: bool = True
    enable_coregistration_validation: bool = True
    
    # Output settings
    save_intermediate_results: bool = False
    generate_visualizations: bool = False
    output_format: str = "json"  # json, csv, both
