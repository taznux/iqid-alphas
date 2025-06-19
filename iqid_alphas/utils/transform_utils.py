"""
Transformation and Alignment Utilities

Contains geometric transformations, coordinate conversions, and alignment utilities
that will be used by the core alignment and coregistration modules.
"""

import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from datetime import datetime, timedelta


def get_time_difference(time1: datetime, 
                       time2: datetime, 
                       grain: str = 'us', 
                       verbose: bool = False) -> float:
    """Find the time difference between two datetime objects.
    
    Migrated from helper.get_dt()

    Parameters
    ----------
    time1, time2 : datetime
        The two times to be assessed
    grain : {'us', 's'}, default 'us'
        Temporal resolution of the result
        Note that dt.seconds != dt.total_seconds()
    verbose : bool, default False
        Whether to print verbose output
        
    Returns
    -------
    float
        Time difference in specified units
    """
    # Find dt between two datetime objects
    dt = min(abs(time1 - time2), abs(time2 - time1))
    
    if grain == 'us':
        result = dt.seconds + dt.microseconds * 1e-6
    elif grain == 's':
        result = dt.total_seconds()
    else:
        raise ValueError("grain must be 's' or 'us'")
    
    if verbose:
        print(f"Time difference: {result} {grain}")
    
    return result


def create_affine_matrix(translation: Tuple[float, float] = (0, 0),
                        rotation: float = 0,
                        scale: Tuple[float, float] = (1, 1),
                        shear: float = 0) -> np.ndarray:
    """Create a 2D affine transformation matrix.
    
    Parameters
    ----------
    translation : Tuple[float, float], default (0, 0)
        Translation in x and y directions
    rotation : float, default 0
        Rotation angle in radians
    scale : Tuple[float, float], default (1, 1)
        Scaling factors in x and y directions
    shear : float, default 0
        Shear angle in radians
        
    Returns
    -------
    np.ndarray
        3x3 affine transformation matrix
    """
    # Translation matrix
    T = np.array([[1, 0, translation[0]],
                  [0, 1, translation[1]],
                  [0, 0, 1]])
    
    # Rotation matrix
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    R = np.array([[cos_r, -sin_r, 0],
                  [sin_r, cos_r, 0],
                  [0, 0, 1]])
    
    # Scale matrix
    S = np.array([[scale[0], 0, 0],
                  [0, scale[1], 0],
                  [0, 0, 1]])
    
    # Shear matrix
    Sh = np.array([[1, np.tan(shear), 0],
                   [0, 1, 0],
                   [0, 0, 1]])
    
    # Combine transformations: T * R * S * Sh
    return T @ R @ S @ Sh


def apply_affine_transform(points: np.ndarray, 
                          transform_matrix: np.ndarray) -> np.ndarray:
    """Apply affine transformation to a set of points.
    
    Parameters
    ----------
    points : np.ndarray
        Array of points with shape (N, 2) for 2D points
    transform_matrix : np.ndarray
        3x3 affine transformation matrix
        
    Returns
    -------
    np.ndarray
        Transformed points with same shape as input
    """
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack([points, ones])
    
    # Apply transformation
    transformed = homogeneous_points @ transform_matrix.T
    
    # Convert back to Cartesian coordinates
    return transformed[:, :2]


def estimate_rigid_transform(source_points: np.ndarray, 
                           target_points: np.ndarray) -> np.ndarray:
    """Estimate rigid transformation (rotation + translation) between point sets.
    
    Uses Procrustes analysis to find optimal rigid transformation.
    
    Parameters
    ----------
    source_points : np.ndarray
        Source points with shape (N, 2)
    target_points : np.ndarray
        Target points with shape (N, 2)
        
    Returns
    -------
    np.ndarray
        3x3 affine transformation matrix for rigid transform
    """
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target points must have same shape")
    
    # Center the points
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # Compute rotation using SVD
    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = target_centroid - R @ source_centroid
    
    # Create affine matrix
    transform_matrix = np.eye(3)
    transform_matrix[:2, :2] = R
    transform_matrix[:2, 2] = t
    
    return transform_matrix


def compute_transform_error(source_points: np.ndarray,
                          target_points: np.ndarray,
                          transform_matrix: np.ndarray) -> Dict[str, float]:
    """Compute error metrics for a transformation.
    
    Parameters
    ----------
    source_points : np.ndarray
        Source points with shape (N, 2)
    target_points : np.ndarray
        Target points with shape (N, 2)
    transform_matrix : np.ndarray
        3x3 transformation matrix
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing error metrics:
        - 'rmse': Root mean square error
        - 'mae': Mean absolute error
        - 'max_error': Maximum error
        - 'std_error': Standard deviation of errors
    """
    # Apply transformation to source points
    transformed_points = apply_affine_transform(source_points, transform_matrix)
    
    # Compute point-wise distances
    distances = np.linalg.norm(transformed_points - target_points, axis=1)
    
    return {
        'rmse': np.sqrt(np.mean(distances**2)),
        'mae': np.mean(distances),
        'max_error': np.max(distances),
        'std_error': np.std(distances)
    }


def interpolate_transform(transform1: np.ndarray,
                         transform2: np.ndarray,
                         alpha: float) -> np.ndarray:
    """Interpolate between two transformation matrices.
    
    Parameters
    ----------
    transform1 : np.ndarray
        First transformation matrix (3x3)
    transform2 : np.ndarray
        Second transformation matrix (3x3)
    alpha : float
        Interpolation parameter (0 = transform1, 1 = transform2)
        
    Returns
    -------
    np.ndarray
        Interpolated transformation matrix
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")
    
    # Simple linear interpolation (note: this is not optimal for rotations)
    # For production, consider using proper rotation interpolation (SLERP)
    return (1 - alpha) * transform1 + alpha * transform2


def invert_affine_transform(transform_matrix: np.ndarray) -> np.ndarray:
    """Compute the inverse of an affine transformation matrix.
    
    Parameters
    ----------
    transform_matrix : np.ndarray
        3x3 affine transformation matrix
        
    Returns
    -------
    np.ndarray
        Inverse transformation matrix
    """
    return np.linalg.inv(transform_matrix)


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose multiple transformation matrices.
    
    Parameters
    ----------
    *transforms : np.ndarray
        Variable number of 3x3 transformation matrices
        
    Returns
    -------
    np.ndarray
        Composed transformation matrix
    """
    if not transforms:
        return np.eye(3)
    
    result = transforms[0]
    for transform in transforms[1:]:
        result = result @ transform
    
    return result


def transform_bounding_box(bbox: Tuple[float, float, float, float],
                          transform_matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """Transform a bounding box and compute the new axis-aligned bounding box.
    
    Parameters
    ----------
    bbox : Tuple[float, float, float, float]
        Bounding box as (x_min, y_min, x_max, y_max)
    transform_matrix : np.ndarray
        3x3 transformation matrix
        
    Returns
    -------
    Tuple[float, float, float, float]
        Transformed bounding box as (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Create corner points
    corners = np.array([
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_min],
        [x_max, y_max]
    ])
    
    # Transform corners
    transformed_corners = apply_affine_transform(corners, transform_matrix)
    
    # Compute new bounding box
    new_x_min = np.min(transformed_corners[:, 0])
    new_y_min = np.min(transformed_corners[:, 1])
    new_x_max = np.max(transformed_corners[:, 0])
    new_y_max = np.max(transformed_corners[:, 1])
    
    return new_x_min, new_y_min, new_x_max, new_y_max


def validate_user_input(prompt: str, 
                       valid_options: Optional[Dict[str, Any]] = None) -> Any:
    """Validate user input with predefined options.
    
    Migrated and enhanced from helper.get_yn()
    
    Parameters
    ----------
    prompt : str
        Prompt to display to user
    valid_options : Optional[Dict[str, Any]]
        Dictionary mapping valid input strings to return values
        If None, defaults to yes/no validation
        
    Returns
    -------
    Any
        Validated user input value
    """
    if valid_options is None:
        valid_options = {
            "yes": True, "no": False, "y": True, "n": False,
            "true": True, "false": False, "1": True, "0": False
        }
    
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in valid_options:
            return valid_options[user_input]
        else:
            print(f"Invalid input. Valid options: {list(valid_options.keys())}")
