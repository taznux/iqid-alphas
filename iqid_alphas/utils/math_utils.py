"""
Mathematical and Array Utilities

Migrated and refactored from old/iqid-alphas/iqid/helper.py
Contains mathematical operations, array manipulations, and statistics utilities.
"""

import numpy as np
from typing import Tuple, Union, Optional, Literal


def bin_ndarray(ndarray: np.ndarray, 
                new_shape: Tuple[int, ...], 
                operation: Literal['sum', 'mean'] = 'sum') -> np.ndarray:
    """
    Bin an ndarray in all axes based on the target shape, by summing or averaging.
    
    Migrated from helper.bin_ndarray()

    Number of output dimensions must match number of input dimensions and 
    new axes must divide old ones.

    Parameters
    ----------
    ndarray : np.ndarray
        Input array to bin
    new_shape : Tuple[int, ...]
        Target shape for binning
    operation : {'sum', 'mean'}, default 'sum'
        Operation to apply when binning
        
    Returns
    -------
    np.ndarray
        Binned array
        
    Examples
    --------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    operation = operation.lower()
    if operation not in ['sum', 'mean']:
        raise ValueError("Operation not supported. Use 'sum' or 'mean'.")
    if ndarray.ndim != len(new_shape):
        raise ValueError(
            f"Shape mismatch: {ndarray.shape} -> {new_shape}")
    
    compression_pairs = [(d, c//d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [x for p in compression_pairs for x in p]
    ndarray = ndarray.reshape(flattened)
    
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    
    return ndarray


def decompose_affine(affine_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 3x3 transformation matrix into its affine constituents.
    
    Migrated from helper.decompose_affine()

    Decomposes into translation (T), rotation (R), scaling (Z), and shear (S).
    From https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/affines.py

    Parameters
    ----------
    affine_matrix : np.ndarray
        3x3 or 4x4 affine transformation matrix
        
    Returns
    -------
    T : np.ndarray
        Translation vector
    R : np.ndarray  
        Rotation matrix
    Z : np.ndarray
        Scaling factors
    S : np.ndarray
        Shear parameters
    """
    A = np.asarray(affine_matrix)
    T = A[:-1, -1]
    RZS = A[:-1, :-1]
    ZS = np.linalg.cholesky(np.dot(RZS.T, RZS)).T
    Z = np.diag(ZS).copy()
    shears = ZS / Z[:, np.newaxis]
    n = len(Z)
    S = shears[np.triu(np.ones((n, n)), 1).astype(bool)]
    R = np.dot(RZS, np.linalg.inv(ZS))
    
    if np.linalg.det(R) < 0:
        Z[0] *= -1
        ZS[0] *= -1
        R = np.dot(RZS, np.linalg.inv(ZS))
    
    return T, R, Z, S


def calculate_statistics(array: np.ndarray, 
                        return_values: bool = False) -> Optional[Tuple[float, float]]:
    """Calculate and optionally print mean and standard deviation.
    
    Migrated from helper.mean_stdev()
    
    Parameters
    ----------
    array : np.ndarray
        Input array
    return_values : bool, default False
        Whether to return the calculated values
        
    Returns
    -------
    Optional[Tuple[float, float]]
        Mean and standard deviation if return_values=True, else None
    """
    mean_val = np.mean(array)
    std_val = np.std(array)
    
    print(f'{mean_val:.2f} +/- {std_val:.2f}')
    
    if return_values:
        return mean_val, std_val
    return None


def percent_error(reference: Union[float, np.ndarray], 
                  experimental: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate percent error between reference and experimental values.
    
    Migrated from helper.pct_err()
    
    Parameters
    ----------
    reference : float or np.ndarray
        Reference (true) value(s)
    experimental : float or np.ndarray
        Experimental (measured) value(s)
        
    Returns
    -------
    float or np.ndarray
        Percent error: 100 * (reference - experimental) / reference
    """
    return 100 * (reference - experimental) / reference


def check_monotonic(array: np.ndarray, strict: bool = False) -> bool:
    """Check if an array is monotonically increasing.
    
    Migrated from helper.check_mono()

    Parameters
    ----------
    array : np.ndarray
        Input array to check
    strict : bool, default False
        If True, check for strictly increasing (no equal consecutive values)
        
    Returns
    -------
    bool
        True if monotonically increasing, False otherwise
    """
    if strict:
        return np.all(array[:-1] < array[1:])
    else:
        return np.all(array[:-1] <= array[1:])


def robust_mean(array: np.ndarray, 
                exclude_outliers: bool = True,
                outlier_std_threshold: float = 2.0) -> float:
    """Calculate robust mean, optionally excluding outliers.
    
    Parameters
    ----------
    array : np.ndarray
        Input array
    exclude_outliers : bool, default True
        Whether to exclude outliers before calculating mean
    outlier_std_threshold : float, default 2.0
        Number of standard deviations to use as outlier threshold
        
    Returns
    -------
    float
        Robust mean value
    """
    if not exclude_outliers:
        return np.mean(array)
    
    # Remove outliers using z-score method
    z_scores = np.abs((array - np.mean(array)) / np.std(array))
    filtered_array = array[z_scores < outlier_std_threshold]
    
    # Fallback to original array if too many values were filtered
    if len(filtered_array) < len(array) * 0.5:
        return np.mean(array)
    
    return np.mean(filtered_array)


def normalize_array(array: np.ndarray, 
                   method: Literal['minmax', 'zscore', 'robust'] = 'minmax') -> np.ndarray:
    """Normalize array using different methods.
    
    Parameters
    ----------
    array : np.ndarray
        Input array to normalize
    method : {'minmax', 'zscore', 'robust'}
        Normalization method:
        - 'minmax': Scale to [0, 1] range
        - 'zscore': Zero mean, unit variance
        - 'robust': Robust scaling using median and IQR
        
    Returns
    -------
    np.ndarray
        Normalized array
    """
    if method == 'minmax':
        array_min = np.min(array)
        array_max = np.max(array)
        if array_max == array_min:
            return np.zeros_like(array)
        return (array - array_min) / (array_max - array_min)
    
    elif method == 'zscore':
        mean_val = np.mean(array)
        std_val = np.std(array)
        if std_val == 0:
            return np.zeros_like(array)
        return (array - mean_val) / std_val
    
    elif method == 'robust':
        median_val = np.median(array)
        q75, q25 = np.percentile(array, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return np.zeros_like(array)
        return (array - median_val) / iqr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def safe_divide(numerator: np.ndarray, 
                denominator: np.ndarray, 
                fill_value: float = 0.0) -> np.ndarray:
    """Safely divide arrays, handling division by zero.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array
    denominator : np.ndarray
        Denominator array
    fill_value : float, default 0.0
        Value to use where denominator is zero
        
    Returns
    -------
    np.ndarray
        Result of division with safe handling of zeros
    """
    result = np.full_like(numerator, fill_value, dtype=float)
    nonzero_mask = denominator != 0
    result[nonzero_mask] = numerator[nonzero_mask] / denominator[nonzero_mask]
    return result


def moving_average(array: np.ndarray, 
                   window_size: int,
                   mode: Literal['valid', 'same', 'full'] = 'same') -> np.ndarray:
    """Calculate moving average of an array.
    
    Parameters
    ----------
    array : np.ndarray
        Input array
    window_size : int
        Size of the moving window
    mode : {'valid', 'same', 'full'}, default 'same'
        Convolution mode
        
    Returns
    -------
    np.ndarray
        Moving average array
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if window_size > len(array):
        raise ValueError("Window size cannot be larger than array length")
    
    kernel = np.ones(window_size) / window_size
    return np.convolve(array, kernel, mode=mode)
