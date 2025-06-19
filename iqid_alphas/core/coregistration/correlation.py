"""
Cross-Correlation Registration

Implements normalized cross-correlation and phase correlation registration.
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any

from .types import RegistrationResult, RegistrationMethod, ModalityPair


class CrossCorrelationAligner:
    """Cross-correlation based registration."""
    
    def __init__(self):
        """Initialize cross-correlation aligner."""
        self.logger = logging.getLogger(__name__)
    
    def register_normalized(self, pair: ModalityPair) -> RegistrationResult:
        """Register using normalized cross-correlation."""
        try:
            # Normalize images
            fixed = pair.modality1_image.astype(np.float32)
            moving = pair.modality2_image.astype(np.float32)
            
            fixed = (fixed - np.mean(fixed)) / (np.std(fixed) + 1e-8)
            moving = (moving - np.mean(moving)) / (np.std(moving) + 1e-8)
            
            # Perform cross-correlation
            correlation = cv2.matchTemplate(fixed, moving, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(correlation)
            
            # Calculate translation
            h, w = moving.shape
            dx = max_loc[0] - (fixed.shape[1] - w) // 2
            dy = max_loc[1] - (fixed.shape[0] - h) // 2
            
            # Create transformation matrix
            transform_matrix = np.array([
                [1, 0, dx],
                [0, 1, dy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Apply transformation
            registered_image = cv2.warpAffine(
                pair.modality2_image, transform_matrix[:2, :], pair.modality1_image.shape[::-1]
            )
            
            # Use correlation value as confidence
            confidence = max(0.0, min(1.0, max_val))
            
            return RegistrationResult(
                transform_matrix=transform_matrix,
                registered_image=registered_image,
                confidence=confidence,
                method=RegistrationMethod.NORMALIZED_CROSS_CORRELATION,
                processing_time=0.0,
                metadata={
                    'correlation_value': max_val,
                    'translation': (dx, dy)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Cross-correlation registration failed: {e}")
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=pair.modality2_image.copy(),
                confidence=0.0,
                method=RegistrationMethod.NORMALIZED_CROSS_CORRELATION,
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
    def register_phase(self, pair: ModalityPair) -> RegistrationResult:
        """Register using phase correlation."""
        try:
            # Convert to grayscale if needed
            img1 = self._ensure_grayscale(pair.modality1_image)
            img2 = self._ensure_grayscale(pair.modality2_image)
            
            # Ensure same size
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, img1.shape[::-1])
            
            # Apply FFT
            f_img1 = np.fft.fft2(img1)
            f_img2 = np.fft.fft2(img2)
            
            # Calculate cross power spectrum
            cross_power = (f_img1 * np.conj(f_img2)) / (np.abs(f_img1 * np.conj(f_img2)) + 1e-8)
            
            # Inverse FFT to get correlation
            correlation = np.fft.ifft2(cross_power)
            correlation = np.abs(correlation)
            
            # Find peak
            peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            # Calculate shift (handle wrap-around)
            h, w = img1.shape
            shift_y = peak_y if peak_y < h // 2 else peak_y - h
            shift_x = peak_x if peak_x < w // 2 else peak_x - w
            
            # Create transformation matrix
            transform_matrix = np.array([
                [1, 0, shift_x],
                [0, 1, shift_y],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Apply transformation
            registered_image = cv2.warpAffine(
                pair.modality2_image, transform_matrix[:2, :], pair.modality1_image.shape[::-1]
            )
            
            # Use normalized peak value as confidence
            confidence = float(np.max(correlation) / np.mean(correlation))
            confidence = min(1.0, max(0.0, (confidence - 1.0) / 10.0))  # Normalize to 0-1
            
            return RegistrationResult(
                transform_matrix=transform_matrix,
                registered_image=registered_image,
                confidence=confidence,
                method=RegistrationMethod.PHASE_CORRELATION,
                processing_time=0.0,
                metadata={
                    'shift': (shift_x, shift_y),
                    'peak_value': float(np.max(correlation))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Phase correlation registration failed: {e}")
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=pair.modality2_image.copy(),
                confidence=0.0,
                method=RegistrationMethod.PHASE_CORRELATION,
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
