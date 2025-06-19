"""
Coregistration Package

Multi-modal image registration package with modular components:

- types: Data structures and enums for registration
- mutual_info: Mutual information-based registration
- feature_based: Feature-based registration methods
- intensity_based: Intensity and template matching methods
- quality: Quality assessment metrics
- aligner: Main multi-modal aligner class

Usage:
    from iqid_alphas.core.coregistration import MultiModalAligner, RegistrationMethod
    
    aligner = MultiModalAligner()
    result = aligner.register_images(fixed_img, moving_img)
"""

from .types import (
    RegistrationMethod,
    QualityMetric,
    RegistrationResult,
    ModalityPair
)
from .aligner import MultiModalAligner

__all__ = [
    'RegistrationMethod',
    'QualityMetric', 
    'RegistrationResult',
    'ModalityPair',
    'MultiModalAligner'
]
