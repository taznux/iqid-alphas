"""
IQID-Alphas Utilities Package

This package contains utility modules migrated and refactored from the legacy codebase.
All functions have been modernized with proper typing, documentation, and error handling.

Modules:
- io_utils: File I/O and directory management utilities
- math: Mathematical operations and array manipulations  
- transform: Geometric transformations and alignment utilities
- data_discovery: Data discovery utilities (existing)
"""

# Existing utilities
from .data_discovery import UCSFDataDiscoverer, DataDiscoveryFormatter

# Import key functions from migrated utilities for convenient access
from .io import (
    list_directories,
    list_subdirectories,
    natural_sort,
    find_files_with_pattern,
    ensure_directory_exists,
    copy_file_safe
)

from .math import (
    bin_ndarray,
    decompose_affine,
    calculate_statistics,
    percent_error,
    check_monotonic,
    normalize_array
)

from .transform import (
    create_affine_matrix,
    apply_affine_transform,
    estimate_rigid_transform,
    compute_transform_error,
    validate_user_input
)

__all__ = [
    # Existing utilities
    'UCSFDataDiscoverer',
    'DataDiscoveryFormatter',
    
    # io_utils
    'list_directories',
    'list_subdirectories', 
    'natural_sort',
    'find_files_with_pattern',
    'ensure_directory_exists',
    'copy_file_safe',
    
    # math
    'bin_ndarray',
    'decompose_affine',
    'calculate_statistics',
    'percent_error',
    'check_monotonic',
    'normalize_array',
    
    # transform
    'create_affine_matrix',
    'apply_affine_transform',
    'estimate_rigid_transform', 
    'compute_transform_error',
    'validate_user_input'
]
