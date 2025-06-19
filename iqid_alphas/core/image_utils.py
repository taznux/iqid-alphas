"""
Image Loading and Preprocessing Utilities for Tissue Segmentation
"""
import numpy as np
from pathlib import Path
import os
from skimage import io, filters
from ..utils.math import normalize_array


def load_raw_image(image_path: Path, logger=None) -> np.ndarray:
    """
    Load raw iQID image data.
    Handles both standard image formats and binary iQID formats.
    """
    try:
        if image_path.suffix.lower() in ['.dat', '.raw']:
            return load_iqid_binary_data(image_path)
        else:
            return io.imread(str(image_path))
    except Exception as e:
        if logger:
            logger.error(f"Failed to load image {image_path}: {e}")
        raise

def load_iqid_binary_data(file_path: Path) -> np.ndarray:
    """
    Load binary iQID data format.
    """
    header = np.fromfile(str(file_path), dtype=np.int32, count=100)
    header_size = header[0]
    xdim = header[1]
    ydim = header[2]
    file_size_bytes = os.path.getsize(str(file_path))
    num_data_elements = 14  # Default for processed_lm format
    byte_size = 8  # Using float64
    byte_fac = 2
    num_clusters = np.floor((file_size_bytes - 4 * header_size) / (byte_size * num_data_elements))
    unshaped_data = np.fromfile(
        str(file_path), 
        dtype=np.float64, 
        count=header_size // byte_fac + int(num_clusters * num_data_elements)
    )
    data = unshaped_data[header_size // byte_fac:].reshape(
        int(num_data_elements), int(num_clusters), order='F'
    )
    yC_global = data[4, :]
    xC_global = data[5, :]
    cluster_area = data[3, :]
    sum_cluster_signal = data[2, :]
    image = np.zeros((ydim, xdim), dtype=np.float64)
    for i in range(len(xC_global)):
        x_coord = int(np.clip(xC_global[i], 0, xdim - 1))
        y_coord = int(np.clip(yC_global[i], 0, ydim - 1))
        intensity = sum_cluster_signal[i] if len(sum_cluster_signal) > i else cluster_area[i]
        image[y_coord, x_coord] = intensity
    return image

def preprocess_raw_image(raw_image: np.ndarray) -> np.ndarray:
    """
    Preprocess raw image for segmentation: normalization and smoothing.
    """
    preprocessed = normalize_array(raw_image, method='minmax')
    preprocessed = filters.gaussian(preprocessed, sigma=1.0, preserve_range=True)
    return preprocessed
