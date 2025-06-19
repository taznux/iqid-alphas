"""
Core Alignment Algorithms

This module contains the individual alignment algorithm implementations,
separated from the main aligner orchestrator for better modularity.
"""

import time
import logging
from typing import Tuple, Optional
import numpy as np
import cv2
from skimage.registration import phase_cross_correlation

from .types import AlignmentMethod, AlignmentResult


class BaseAligner:
    """Base class for alignment algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def align(self, moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        """Align moving image to reference image."""
        raise NotImplementedError("Subclasses must implement align method")
    
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _create_result(self, 
                      transform_matrix: np.ndarray,
                      confidence: float,
                      method: AlignmentMethod,
                      processing_time: float,
                      metadata: dict = None) -> AlignmentResult:
        """Create a standardized AlignmentResult."""
        return AlignmentResult(
            transform_matrix=transform_matrix,
            confidence=confidence,
            method=method,
            processing_time=processing_time,
            metadata=metadata or {}
        )


class FeatureBasedAligner(BaseAligner):
    """Feature-based alignment using ORB/SIFT keypoints."""
    
    def __init__(self, 
                 detector_type: str = 'ORB',
                 max_features: int = 5000,
                 match_threshold: float = 0.75):
        super().__init__()
        self.detector_type = detector_type
        self.max_features = max_features
        self.match_threshold = match_threshold
        
        # Initialize feature detector
        self._init_detector()
    
    def _init_detector(self):
        """Initialize the feature detector and matcher."""
        if self.detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.detector_type == 'SIFT':
            try:
                self.detector = cv2.SIFT_create(nfeatures=self.max_features)
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            except AttributeError:
                self.logger.warning("SIFT not available, falling back to ORB")
                self.detector = cv2.ORB_create(nfeatures=self.max_features)
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")
    
    def align(self, moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        """Align using feature-based matching."""
        start_time = time.time()
        
        # Ensure grayscale
        moving_gray = self._ensure_grayscale(moving)
        reference_gray = self._ensure_grayscale(reference)
        
        # Detect features
        kp1, des1 = self.detector.detectAndCompute(reference_gray, None)
        kp2, des2 = self.detector.detectAndCompute(moving_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            processing_time = time.time() - start_time
            return self._create_result(
                transform_matrix=np.eye(3),
                confidence=0.0,
                method=AlignmentMethod.FEATURE_BASED,
                processing_time=processing_time,
                metadata={'error': 'insufficient_features', 'features_ref': len(kp1), 'features_mov': len(kp2)}
            )
        
        # Match features
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            processing_time = time.time() - start_time
            return self._create_result(
                transform_matrix=np.eye(3),
                confidence=0.0,
                method=AlignmentMethod.FEATURE_BASED,
                processing_time=processing_time,
                metadata={'error': 'insufficient_matches', 'matches': len(matches)}
            )
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate transformation
        try:
            transform_matrix_2x3, inliers = cv2.estimateAffine2D(
                dst_pts, src_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99
            )
            
            if transform_matrix_2x3 is None:
                processing_time = time.time() - start_time
                return self._create_result(
                    transform_matrix=np.eye(3),
                    confidence=0.0,
                    method=AlignmentMethod.FEATURE_BASED,
                    processing_time=processing_time,
                    metadata={'error': 'transform_estimation_failed'}
                )
            
            # Convert to 3x3 matrix
            transform_matrix = np.vstack([transform_matrix_2x3, [0, 0, 1]])
            
            # Calculate confidence
            num_inliers = np.sum(inliers) if inliers is not None else 0
            inlier_ratio = num_inliers / len(matches)
            confidence = inlier_ratio * min(1.0, len(matches) / 50.0)
            
            processing_time = time.time() - start_time
            return self._create_result(
                transform_matrix=transform_matrix,
                confidence=confidence,
                method=AlignmentMethod.FEATURE_BASED,
                processing_time=processing_time,
                metadata={
                    'total_matches': len(matches),
                    'inliers': num_inliers,
                    'inlier_ratio': inlier_ratio
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return self._create_result(
                transform_matrix=np.eye(3),
                confidence=0.0,
                method=AlignmentMethod.FEATURE_BASED,
                processing_time=processing_time,
                metadata={'error': f'transformation_error: {str(e)}'}
            )


class IntensityBasedAligner(BaseAligner):
    """Intensity-based alignment using template matching."""
    
    def __init__(self, method: int = cv2.TM_CCOEFF_NORMED):
        super().__init__()
        self.method = method
    
    def align(self, moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        """Align using intensity-based template matching."""
        start_time = time.time()
        
        # Ensure grayscale
        moving_gray = self._ensure_grayscale(moving)
        reference_gray = self._ensure_grayscale(reference)
        
        try:
            # Perform template matching
            result = cv2.matchTemplate(reference_gray, moving_gray, self.method)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # Calculate translation
            h, w = moving_gray.shape
            dx = max_loc[0] - (reference_gray.shape[1] - w) // 2
            dy = max_loc[1] - (reference_gray.shape[0] - h) // 2
            
            # Create transformation matrix (translation only)
            transform_matrix = np.array([
                [1, 0, dx],
                [0, 1, dy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Use template matching confidence as alignment confidence
            confidence = max_val if max_val > 0 else 0.0
            
            processing_time = time.time() - start_time
            return self._create_result(
                transform_matrix=transform_matrix,
                confidence=confidence,
                method=AlignmentMethod.INTENSITY_BASED,
                processing_time=processing_time,
                metadata={
                    'translation': (dx, dy),
                    'template_match_value': max_val
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return self._create_result(
                transform_matrix=np.eye(3),
                confidence=0.0,
                method=AlignmentMethod.INTENSITY_BASED,
                processing_time=processing_time,
                metadata={'error': f'intensity_alignment_error: {str(e)}'}
            )


class PhaseCorrelationAligner(BaseAligner):
    """Phase correlation alignment using FFT."""
    
    def __init__(self, upsample_factor: int = 1):
        super().__init__()
        self.upsample_factor = upsample_factor
    
    def align(self, moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        """Align using phase correlation."""
        start_time = time.time()
        
        # Ensure grayscale and same size
        moving_gray = self._ensure_grayscale(moving)
        reference_gray = self._ensure_grayscale(reference)
        
        # Resize to same dimensions if needed
        if moving_gray.shape != reference_gray.shape:
            moving_gray = cv2.resize(moving_gray, reference_gray.shape[::-1])
        
        try:
            # Perform phase correlation
            shift, error, diffphase = phase_cross_correlation(
                reference_gray, moving_gray, 
                upsample_factor=self.upsample_factor
            )
            
            # Create transformation matrix (translation only)
            transform_matrix = np.array([
                [1, 0, shift[1]],  # x translation
                [0, 1, shift[0]],  # y translation
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Convert error to confidence (lower error = higher confidence)
            confidence = max(0.0, 1.0 - error)
            
            processing_time = time.time() - start_time
            return self._create_result(
                transform_matrix=transform_matrix,
                confidence=confidence,
                method=AlignmentMethod.PHASE_CORRELATION,
                processing_time=processing_time,
                metadata={
                    'shift': shift.tolist(),
                    'error': error,
                    'diffphase': diffphase
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return self._create_result(
                transform_matrix=np.eye(3),
                confidence=0.0,
                method=AlignmentMethod.PHASE_CORRELATION,
                processing_time=processing_time,
                metadata={'error': f'phase_correlation_error: {str(e)}'}
            )


class CoarseRotationAligner(BaseAligner):
    """Coarse rotation alignment using template matching at different angles."""
    
    def __init__(self, 
                 angle_range: Tuple[float, float] = (-10.0, 10.0),
                 angle_step: float = 1.0):
        super().__init__()
        self.angle_range = angle_range
        self.angle_step = angle_step
    
    def align(self, moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        """Align using coarse rotation search."""
        start_time = time.time()
        
        # Ensure grayscale
        moving_gray = self._ensure_grayscale(moving)
        reference_gray = self._ensure_grayscale(reference)
        
        best_angle = 0.0
        best_score = -np.inf
        
        # Search through rotation angles
        angles = np.arange(self.angle_range[0], self.angle_range[1] + self.angle_step, self.angle_step)
        
        try:
            center = (moving_gray.shape[1] // 2, moving_gray.shape[0] // 2)
            
            for angle in angles:
                # Rotate image
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(moving_gray, M, moving_gray.shape[::-1])
                
                # Calculate correlation
                result = cv2.matchTemplate(reference_gray, rotated, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_angle = angle
            
            # Create transformation matrix with best rotation
            M_best = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            transform_matrix = np.vstack([M_best, [0, 0, 1]])
            
            # Normalize confidence score
            confidence = max(0.0, min(1.0, best_score))
            
            processing_time = time.time() - start_time
            return self._create_result(
                transform_matrix=transform_matrix,
                confidence=confidence,
                method=AlignmentMethod.COARSE_ROTATION,
                processing_time=processing_time,
                metadata={
                    'best_angle': best_angle,
                    'best_score': best_score,
                    'angles_tested': len(angles)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return self._create_result(
                transform_matrix=np.eye(3),
                confidence=0.0,
                method=AlignmentMethod.COARSE_ROTATION,
                processing_time=processing_time,
                metadata={'error': f'coarse_rotation_error: {str(e)}'}
            )
