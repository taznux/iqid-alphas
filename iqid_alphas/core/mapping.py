"""
Core Mapping Module

This module provides reverse mapping functionality for validation and evaluation:
1. Segmentation Evaluation: Locating cropped images from `1_segmented` within the `raw` image
2. Alignment Evaluation: Calculating geometric transformations between `2_aligned` and `1_segmented` slices

The reverse mapping enables automated evaluation and improvement of pipeline components.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass

# Image processing imports
try:
    from skimage import feature, transform, measure
    from skimage.registration import phase_cross_correlation
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Internal imports
from ..utils.io import natural_sort


@dataclass
class CropLocation:
    """Result of crop location finding."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    method: str
    center: Tuple[int, int]
    
    
@dataclass
class AlignmentTransform:
    """Result of alignment transformation calculation."""
    transform_matrix: np.ndarray  # 2x3 or 3x3 affine transformation matrix
    match_quality: float
    method: str
    num_matches: int
    inlier_ratio: float


class ReverseMapper:
    """
    Unified reverse mapping functionality for pipeline evaluation.
    
    Supports both segmentation evaluation (finding crops in raw images) 
    and alignment evaluation (calculating transformations between aligned images).
    """
    
    def __init__(self, 
                 template_matching_method: int = cv2.TM_CCOEFF_NORMED,
                 feature_detector: str = 'ORB',
                 max_features: int = 5000,
                 match_threshold: float = 0.75):
        """
        Initialize the reverse mapper.
        
        Parameters
        ----------
        template_matching_method : int
            OpenCV template matching method for crop location finding
        feature_detector : str
            Feature detector type ('ORB', 'SIFT', 'SURF') for alignment evaluation
        max_features : int
            Maximum number of features to detect
        match_threshold : float
            Threshold for feature matching quality
        """
        self.template_matching_method = template_matching_method
        self.feature_detector = feature_detector
        self.max_features = max_features
        self.match_threshold = match_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature detector
        self._init_feature_detector()
    
    def _init_feature_detector(self):
        """Initialize the feature detector based on the specified type."""
        if self.feature_detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
        elif self.feature_detector == 'SIFT':
            try:
                self.detector = cv2.SIFT_create(nfeatures=self.max_features)
            except AttributeError:
                self.logger.warning("SIFT not available, falling back to ORB")
                self.detector = cv2.ORB_create(nfeatures=self.max_features)
        elif self.feature_detector == 'SURF':
            try:
                self.detector = cv2.xfeatures2d.SURF_create()
            except AttributeError:
                self.logger.warning("SURF not available, falling back to ORB")
                self.detector = cv2.ORB_create(nfeatures=self.max_features)
        else:
            self.logger.warning(f"Unknown detector {self.feature_detector}, using ORB")
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
    
    def find_crop_location(self, 
                          source_image: np.ndarray, 
                          template_image: np.ndarray,
                          method: Optional[str] = None) -> CropLocation:
        """
        Find the location of a cropped template image within a larger source image.
        
        This function implements template matching to locate where a segmented slice
        was cropped from the original raw image.
        
        Parameters
        ----------
        source_image : np.ndarray
            Large source image (e.g., raw iQID image)
        template_image : np.ndarray
            Small template image to find (e.g., segmented slice)
        method : str, optional
            Method to use ('template_matching', 'phase_correlation')
            
        Returns
        -------
        CropLocation
            Location information including bounding box and confidence
        """
        if method is None:
            method = 'template_matching'
        
        if method == 'template_matching':
            return self._find_crop_template_matching(source_image, template_image)
        elif method == 'phase_correlation':
            return self._find_crop_phase_correlation(source_image, template_image)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _find_crop_template_matching(self, 
                                   source_image: np.ndarray, 
                                   template_image: np.ndarray) -> CropLocation:
        """Find crop location using OpenCV template matching."""
        # Convert to grayscale if needed
        if len(source_image.shape) == 3:
            source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source_image.copy()
            
        if len(template_image.shape) == 3:
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_image.copy()
        
        # Ensure both images are the same data type
        source_gray = source_gray.astype(np.float32)
        template_gray = template_gray.astype(np.float32)
        
        # Perform template matching
        result = cv2.matchTemplate(source_gray, template_gray, self.template_matching_method)
        
        # Find the best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # For correlation methods, use max_val and max_loc
        if self.template_matching_method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
            confidence = max_val
            top_left = max_loc
        else:
            confidence = 1.0 - min_val  # Convert to higher-is-better
            top_left = min_loc
        
        # Calculate bounding box
        h, w = template_gray.shape
        bbox = (top_left[0], top_left[1], w, h)
        center = (top_left[0] + w//2, top_left[1] + h//2)
        
        return CropLocation(
            bbox=bbox,
            confidence=confidence,
            method='template_matching',
            center=center
        )
    
    def _find_crop_phase_correlation(self, 
                                   source_image: np.ndarray, 
                                   template_image: np.ndarray) -> CropLocation:
        """Find crop location using phase correlation (requires scikit-image)."""
        if not HAS_SKIMAGE:
            self.logger.warning("scikit-image not available, falling back to template matching")
            return self._find_crop_template_matching(source_image, template_image)
        
        # Convert to grayscale if needed
        if len(source_image.shape) == 3:
            source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source_image.copy()
            
        if len(template_image.shape) == 3:
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_image.copy()
        
        # Use phase correlation to find shift
        try:
            shift, error, diffphase = phase_cross_correlation(source_gray, template_gray)
            
            # Calculate position from shift
            h, w = template_gray.shape
            y_pos = max(0, int(-shift[0]))
            x_pos = max(0, int(-shift[1]))
            
            bbox = (x_pos, y_pos, w, h)
            center = (x_pos + w//2, y_pos + h//2)
            confidence = 1.0 - error  # Lower error = higher confidence
            
            return CropLocation(
                bbox=bbox,
                confidence=confidence,
                method='phase_correlation',
                center=center
            )
        except Exception as e:
            self.logger.warning(f"Phase correlation failed: {e}, falling back to template matching")
            return self._find_crop_template_matching(source_image, template_image)
    
    def calculate_alignment_transform(self, 
                                    original_image: np.ndarray, 
                                    aligned_image: np.ndarray,
                                    method: Optional[str] = None) -> AlignmentTransform:
        """
        Calculate the transformation matrix between original and aligned images.
        
        This function uses feature-based registration to determine the geometric
        transformation applied during alignment.
        
        Parameters
        ----------
        original_image : np.ndarray
            Original image (e.g., from 1_segmented)
        aligned_image : np.ndarray
            Aligned/transformed image (e.g., from 2_aligned)
        method : str, optional
            Method to use ('feature_matching', 'phase_correlation')
            
        Returns
        -------
        AlignmentTransform
            Transformation information including matrix and quality metrics
        """
        if method is None:
            method = 'feature_matching'
        
        if method == 'feature_matching':
            return self._calculate_transform_feature_matching(original_image, aligned_image)
        elif method == 'phase_correlation':
            return self._calculate_transform_phase_correlation(original_image, aligned_image)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _calculate_transform_feature_matching(self, 
                                            original_image: np.ndarray, 
                                            aligned_image: np.ndarray) -> AlignmentTransform:
        """Calculate transformation using feature-based matching."""
        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            orig_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original_image.copy()
            
        if len(aligned_image.shape) == 3:
            aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        else:
            aligned_gray = aligned_image.copy()
        
        # Detect keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(orig_gray, None)
        kp2, des2 = self.detector.detectAndCompute(aligned_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            # Insufficient features, return identity transform
            return AlignmentTransform(
                transform_matrix=np.eye(3),
                match_quality=0.0,
                method='feature_matching',
                num_matches=0,
                inlier_ratio=0.0
            )
        
        # Match features
        if self.feature_detector == 'ORB':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            # Insufficient matches
            return AlignmentTransform(
                transform_matrix=np.eye(3),
                match_quality=0.0,
                method='feature_matching',
                num_matches=len(matches),
                inlier_ratio=0.0
            )
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Calculate affine transformation
        try:
            transform_matrix, inliers = cv2.estimateAffine2D(
                src_pts, dst_pts, 
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99
            )
            
            if transform_matrix is None:
                # Estimation failed
                return AlignmentTransform(
                    transform_matrix=np.eye(3),
                    match_quality=0.0,
                    method='feature_matching',
                    num_matches=len(matches),
                    inlier_ratio=0.0
                )
            
            # Convert 2x3 to 3x3 matrix
            full_transform = np.vstack([transform_matrix, [0, 0, 1]])
            
            # Calculate quality metrics
            num_inliers = np.sum(inliers) if inliers is not None else 0
            inlier_ratio = num_inliers / len(matches) if len(matches) > 0 else 0.0
            match_quality = inlier_ratio * min(1.0, len(matches) / 50.0)  # Quality based on inliers and total matches
            
            return AlignmentTransform(
                transform_matrix=full_transform,
                match_quality=match_quality,
                method='feature_matching',
                num_matches=len(matches),
                inlier_ratio=inlier_ratio
            )
            
        except Exception as e:
            self.logger.warning(f"Transform estimation failed: {e}")
            return AlignmentTransform(
                transform_matrix=np.eye(3),
                match_quality=0.0,
                method='feature_matching',
                num_matches=len(matches),
                inlier_ratio=0.0
            )
    
    def _calculate_transform_phase_correlation(self, 
                                             original_image: np.ndarray, 
                                             aligned_image: np.ndarray) -> AlignmentTransform:
        """Calculate transformation using phase correlation (translation only)."""
        if not HAS_SKIMAGE:
            self.logger.warning("scikit-image not available, falling back to feature matching")
            return self._calculate_transform_feature_matching(original_image, aligned_image)
        
        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            orig_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original_image.copy()
            
        if len(aligned_image.shape) == 3:
            aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        else:
            aligned_gray = aligned_image.copy()
        
        try:
            # Calculate phase correlation
            shift, error, diffphase = phase_cross_correlation(orig_gray, aligned_gray)
            
            # Create translation matrix
            transform_matrix = np.array([
                [1, 0, shift[1]],  # x translation
                [0, 1, shift[0]],  # y translation
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Quality is inverse of error
            match_quality = 1.0 - error
            
            return AlignmentTransform(
                transform_matrix=transform_matrix,
                match_quality=match_quality,
                method='phase_correlation',
                num_matches=1,  # Phase correlation gives one "match"
                inlier_ratio=1.0
            )
            
        except Exception as e:
            self.logger.warning(f"Phase correlation failed: {e}")
            return AlignmentTransform(
                transform_matrix=np.eye(3),
                match_quality=0.0,
                method='phase_correlation',
                num_matches=0,
                inlier_ratio=0.0
            )


# Convenience functions for backward compatibility
def find_crop_location(source_image: np.ndarray, 
                      template_image: np.ndarray,
                      method: str = 'template_matching') -> Dict[str, Any]:
    """
    Convenience function to find crop location.
    
    Returns
    -------
    Dict with keys: 'bbox', 'confidence', 'method', 'center'
    """
    mapper = ReverseMapper()
    result = mapper.find_crop_location(source_image, template_image, method)
    
    return {
        'bbox': result.bbox,
        'confidence': result.confidence,
        'method': result.method,
        'center': result.center
    }


def calculate_alignment_transform(original_image: np.ndarray, 
                                aligned_image: np.ndarray,
                                method: str = 'feature_matching') -> Dict[str, Any]:
    """
    Convenience function to calculate alignment transformation.
    
    Returns
    -------
    Dict with keys: 'transform_matrix', 'match_quality', 'method', 'num_matches', 'inlier_ratio'
    """
    mapper = ReverseMapper()
    result = mapper.calculate_alignment_transform(original_image, aligned_image, method)
    
    return {
        'transform_matrix': result.transform_matrix.tolist(),  # Convert to list for JSON serialization
        'match_quality': result.match_quality,
        'method': result.method,
        'num_matches': result.num_matches,
        'inlier_ratio': result.inlier_ratio
    }
