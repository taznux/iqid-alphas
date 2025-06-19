"""
Alignment Utility Functions

This module contains utility functions for image alignment operations,
transformations, and stack assembly.
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np

from ...utils.io_utils import natural_sort, load_image


def calculate_ssd(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Sum of Squared Differences between two images.
    
    Migrated from legacy get_SSD function.
    
    Parameters
    ----------
    image1, image2 : np.ndarray
        Images to compare
        
    Returns
    -------
    float
        Normalized sum of squared differences
    """
    return np.sum((image1.astype(np.float32) - image2.astype(np.float32)) ** 2) / image1.size


def create_affine_transform(rotation: float = 0.0,
                           translation: Tuple[float, float] = (0.0, 0.0),
                           scale: Tuple[float, float] = (1.0, 1.0),
                           shear: float = 0.0,
                           center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Create an affine transformation matrix from parameters.
    
    Parameters
    ----------
    rotation : float
        Rotation angle in degrees
    translation : Tuple[float, float]
        Translation in (x, y)
    scale : Tuple[float, float]
        Scale factors in (x, y)
    shear : float
        Shear angle in degrees
    center : Tuple[float, float], optional
        Center of rotation/scaling
        
    Returns
    -------
    np.ndarray
        3x3 affine transformation matrix
    """
    if center is None:
        center = (0.0, 0.0)
    
    # Convert angles to radians
    rotation_rad = np.radians(rotation)
    shear_rad = np.radians(shear)
    
    # Create individual transformation matrices
    # Translation to origin
    T1 = np.array([
        [1, 0, -center[0]],
        [0, 1, -center[1]],
        [0, 0, 1]
    ])
    
    # Rotation
    cos_r = np.cos(rotation_rad)
    sin_r = np.sin(rotation_rad)
    R = np.array([
        [cos_r, -sin_r, 0],
        [sin_r, cos_r, 0],
        [0, 0, 1]
    ])
    
    # Scale
    S = np.array([
        [scale[0], 0, 0],
        [0, scale[1], 0],
        [0, 0, 1]
    ])
    
    # Shear
    Sh = np.array([
        [1, np.tan(shear_rad), 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Translation back and final translation
    T2 = np.array([
        [1, 0, center[0] + translation[0]],
        [0, 1, center[1] + translation[1]],
        [0, 0, 1]
    ])
    
    # Combine transformations: T2 * Sh * S * R * T1
    transform = T2 @ Sh @ S @ R @ T1
    
    return transform.astype(np.float32)


def assemble_image_stack(image_dir: Union[str, Path],
                        file_pattern: str = "*.tif",
                        pad_to_max: bool = True) -> np.ndarray:
    """
    Assemble a stack of images from a directory.
    
    Migrated from legacy assemble_stack function with improvements.
    
    Parameters
    ----------
    image_dir : str or Path
        Directory containing images
    file_pattern : str
        Pattern to match image files
    pad_to_max : bool
        Whether to pad images to maximum dimensions
        
    Returns
    -------
    np.ndarray
        Assembled image stack
    """
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob(file_pattern))
    image_files = sorted(image_files, key=lambda x: natural_sort([str(x)])[0])
    
    if not image_files:
        raise ValueError(f"No images found matching {file_pattern} in {image_dir}")
    
    # Load first image to determine dimensions and dtype
    first_image = load_image(image_files[0])
    
    if pad_to_max:
        # Find maximum dimensions
        max_height, max_width = first_image.shape[:2]
        
        for file_path in image_files[1:]:
            img = load_image(file_path)
            max_height = max(max_height, img.shape[0])
            max_width = max(max_width, img.shape[1])
        
        # Create padded stack
        if len(first_image.shape) == 3:
            image_stack = np.zeros((len(image_files), max_height, max_width, first_image.shape[2]), 
                                 dtype=first_image.dtype)
        else:
            image_stack = np.zeros((len(image_files), max_height, max_width), dtype=first_image.dtype)
        
        # Load and pad each image
        for i, file_path in enumerate(image_files):
            img = load_image(file_path)
            
            # Calculate padding
            pad_height = max_height - img.shape[0]
            pad_width = max_width - img.shape[1]
            
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            if len(img.shape) == 3:
                padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                              mode='constant', constant_values=0)
            else:
                padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                              mode='constant', constant_values=0)
            
            image_stack[i] = padded
    else:
        # Simple stack without padding (all images must be same size)
        image_stack = np.array([load_image(f) for f in image_files])
    
    return image_stack
