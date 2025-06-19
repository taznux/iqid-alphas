"""
Validation Metrics Calculation

Contains functions for calculating various validation metrics
for segmentation, alignment, and coregistration.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity

from ..mapping import CropLocation, AlignmentTransform
from ...utils.math_utils import calculate_statistics

logger = logging.getLogger(__name__)


def calculate_segmentation_metrics(raw_image: np.ndarray, 
                                 seg_image: np.ndarray, 
                                 crop_result: CropLocation) -> Dict[str, float]:
    """Calculate segmentation quality metrics."""
    metrics = {}
    
    try:
        # Extract crop from raw image
        bbox = crop_result.bbox
        cropped = raw_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        
        # Resize to match segmented image if needed
        if cropped.shape != seg_image.shape:
            cropped = cv2.resize(cropped, seg_image.shape[:2][::-1])
        
        # Calculate metrics
        metrics['crop_confidence'] = crop_result.confidence
        metrics['size_ratio'] = (seg_image.shape[0] * seg_image.shape[1]) / (raw_image.shape[0] * raw_image.shape[1])
        
        # Convert to grayscale for comparison
        if len(cropped.shape) == 3:
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            cropped_gray = cropped
            
        if len(seg_image.shape) == 3:
            seg_gray = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
        else:
            seg_gray = seg_image
        
        # Structural similarity
        if cropped_gray.shape == seg_gray.shape:
            ssim_score = structural_similarity(cropped_gray, seg_gray)
            metrics['structural_similarity'] = ssim_score
        
        # Cross correlation
        correlation = cv2.matchTemplate(cropped_gray.astype(np.float32), 
                                      seg_gray.astype(np.float32), 
                                      cv2.TM_CCOEFF_NORMED)
        metrics['cross_correlation'] = np.max(correlation)
        
    except Exception as e:
        logger.warning(f"Error calculating segmentation metrics: {e}")
    
    return metrics


def calculate_alignment_metrics(original_image: np.ndarray, 
                              aligned_image: np.ndarray, 
                              transform_result: AlignmentTransform) -> Dict[str, float]:
    """Calculate alignment quality metrics."""
    metrics = {}
    
    try:
        # Basic metrics from transform result
        metrics['match_quality'] = transform_result.match_quality
        metrics['feature_matches'] = transform_result.num_matches
        
        # Calculate transformation accuracy
        if transform_result.transform_matrix is not None:
            # Extract translation and rotation
            tx, ty = transform_result.transform_matrix[0, 2], transform_result.transform_matrix[1, 2]
            metrics['translation_magnitude'] = np.sqrt(tx**2 + ty**2)
            
            # Calculate rotation angle
            angle = np.arctan2(transform_result.transform_matrix[1, 0], 
                             transform_result.transform_matrix[0, 0])
            metrics['rotation_angle_degrees'] = np.degrees(angle)
        
        # Image similarity metrics
        if original_image.shape == aligned_image.shape:
            # Convert to grayscale if needed
            if len(original_image.shape) == 3:
                orig_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original_image
                
            if len(aligned_image.shape) == 3:
                aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
            else:
                aligned_gray = aligned_image
            
            # Structural similarity
            ssim_score = structural_similarity(orig_gray, aligned_gray)
            metrics['ssim_score'] = ssim_score
            
            # Mean squared error
            mse = mean_squared_error(orig_gray.flatten(), aligned_gray.flatten())
            metrics['mean_squared_error'] = mse
            
            # Peak signal-to-noise ratio
            if mse > 0:
                max_pixel = 255.0 if orig_gray.dtype == np.uint8 else np.max(orig_gray)
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
                metrics['peak_snr'] = psnr
        
    except Exception as e:
        logger.warning(f"Error calculating alignment metrics: {e}")
    
    return metrics


def calculate_coregistration_metrics(fixed_image: np.ndarray,
                                   moving_image: np.ndarray,
                                   registered_image: np.ndarray,
                                   transform_matrix: np.ndarray) -> Dict[str, float]:
    """Calculate coregistration quality metrics."""
    metrics = {}
    
    try:
        # Image similarity metrics
        if fixed_image.shape == registered_image.shape:
            # Convert to grayscale if needed
            if len(fixed_image.shape) == 3:
                fixed_gray = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
            else:
                fixed_gray = fixed_image
                
            if len(registered_image.shape) == 3:
                reg_gray = cv2.cvtColor(registered_image, cv2.COLOR_BGR2GRAY)
            else:
                reg_gray = registered_image
            
            # Structural similarity
            ssim_score = structural_similarity(fixed_gray, reg_gray)
            metrics['ssim_score'] = ssim_score
            
            # Normalized cross-correlation
            correlation = cv2.matchTemplate(fixed_gray.astype(np.float32),
                                          reg_gray.astype(np.float32),
                                          cv2.TM_CCOEFF_NORMED)
            metrics['normalized_cross_correlation'] = np.max(correlation)
            
            # Mutual information (simplified)
            hist_2d, _, _ = np.histogram2d(fixed_gray.flatten(), reg_gray.flatten(), bins=256)
            hist_2d = hist_2d + 1e-10  # Avoid log(0)
            
            # Marginal histograms
            p_x = np.sum(hist_2d, axis=1)
            p_y = np.sum(hist_2d, axis=0)
            
            # Normalize
            p_xy = hist_2d / np.sum(hist_2d)
            p_x = p_x / np.sum(p_x)
            p_y = p_y / np.sum(p_y)
            
            # Calculate mutual information
            mi = 0
            for i in range(p_xy.shape[0]):
                for j in range(p_xy.shape[1]):
                    if p_xy[i, j] > 0:
                        mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            metrics['mutual_information'] = mi
        
        # Transformation metrics
        if transform_matrix is not None:
            # Translation magnitude
            tx, ty = transform_matrix[0, 2], transform_matrix[1, 2]
            metrics['translation_magnitude'] = np.sqrt(tx**2 + ty**2)
            
            # Rotation angle
            angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
            metrics['rotation_angle_degrees'] = np.degrees(angle)
            
            # Scale factors
            scale_x = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[1, 0]**2)
            scale_y = np.sqrt(transform_matrix[0, 1]**2 + transform_matrix[1, 1]**2)
            metrics['scale_x'] = scale_x
            metrics['scale_y'] = scale_y
            metrics['scale_uniformity'] = abs(scale_x - scale_y)
        
    except Exception as e:
        logger.warning(f"Error calculating coregistration metrics: {e}")
    
    return metrics


def calculate_comprehensive_metrics(all_results: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate comprehensive metrics across all validation results."""
    if not all_results:
        return {}
    
    comprehensive = {}
    
    # Collect all metric names
    all_metric_names = set()
    for result in all_results:
        all_metric_names.update(result.keys())        # Calculate statistics for each metric
        for metric_name in all_metric_names:
            values = [result.get(metric_name) for result in all_results if metric_name in result]
            values = [v for v in values if v is not None]
            
            if values:
                values_array = np.array(values)
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                min_val = np.min(values_array)
                max_val = np.max(values_array)
                median_val = np.median(values_array)
                
                comprehensive[f"{metric_name}_mean"] = mean_val
                comprehensive[f"{metric_name}_std"] = std_val
                comprehensive[f"{metric_name}_min"] = min_val
                comprehensive[f"{metric_name}_max"] = max_val
                comprehensive[f"{metric_name}_median"] = median_val
    
    return comprehensive
