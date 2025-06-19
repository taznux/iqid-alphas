"""
Validation Package

Comprehensive validation system for evaluating pipeline outputs,
comparing results against ground truth, and generating quality metrics.

The validation package is organized into several modules:
- types: Data structures and enums for validation
- metrics: Functions for calculating various validation metrics
- utils: Helper functions for file discovery and result management
- validators: Specialized validators for different pipeline components
- suite: Main ValidationSuite orchestrator

Usage:
    from iqid_alphas.core.validation import ValidationSuite, ValidationMode
    
    # Create validation suite
    validator = ValidationSuite(ground_truth_path="path/to/ground_truth")
    
    # Run comprehensive validation
    result = validator.validate("path/to/data", mode=ValidationMode.COMPREHENSIVE)
    
    # Generate report
    report = validator.generate_report("validation_report.txt")
"""

from .types import (
    ValidationMode,
    MetricType,
    ValidationResult,
    GroundTruth,
    ValidationConfig
)
from .metrics import (
    calculate_segmentation_metrics,
    calculate_alignment_metrics,
    calculate_coregistration_metrics,
    calculate_comprehensive_metrics
)
from .utils import (
    find_images,
    find_corresponding_raw_image,
    find_corresponding_segmented_image,
    load_ground_truth,
    save_validation_result,
    generate_validation_report
)
from .validators import (
    SegmentationValidator,
    AlignmentValidator,
    CoregistrationValidator
)
from .suite import ValidationSuite

__all__ = [
    # Types and enums
    'ValidationMode',
    'MetricType',
    'ValidationResult',
    'GroundTruth',
    'ValidationConfig',
    
    # Metrics functions
    'calculate_segmentation_metrics',
    'calculate_alignment_metrics',
    'calculate_coregistration_metrics',
    'calculate_comprehensive_metrics',
    
    # Utility functions
    'find_images',
    'find_corresponding_raw_image',
    'find_corresponding_segmented_image',
    'load_ground_truth',
    'save_validation_result',
    'generate_validation_report',
    
    # Validators
    'SegmentationValidator',
    'AlignmentValidator',
    'CoregistrationValidator',
    
    # Main suite
    'ValidationSuite'
]
