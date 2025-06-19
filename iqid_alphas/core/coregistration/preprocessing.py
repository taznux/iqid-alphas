"""
Preprocessing Utilities

Image preprocessing functions for improving multi-modal registration.
"""

import logging
import numpy as np
import cv2
from skimage import filters, exposure

from .types import ModalityPair


class ImagePreprocessor:
    """Image preprocessing for multi-modal registration."""
    
    def __init__(self):
        """Initialize image preprocessor."""
        self.logger = logging.getLogger(__name__)
    
    def preprocess_pair(self, pair: ModalityPair, method: str = "adaptive") -> ModalityPair:
        """Apply preprocessing to improve multi-modal registration."""
        try:
            # Convert to grayscale if needed
            img1 = self._ensure_grayscale(pair.modality1_image)
            img2 = self._ensure_grayscale(pair.modality2_image)
            
            if method == "adaptive":
                # Histogram equalization for better contrast
                img1 = exposure.equalize_adapthist(img1, clip_limit=0.03)
                img2 = exposure.equalize_adapthist(img2, clip_limit=0.03)
                
                # Gaussian smoothing to reduce noise
                img1 = filters.gaussian(img1, sigma=1.0)
                img2 = filters.gaussian(img2, sigma=1.0)
                
            elif method == "edge_enhance":
                # Edge enhancement for feature-based methods
                img1 = filters.unsharp_mask(img1, radius=1, amount=1)
                img2 = filters.unsharp_mask(img2, radius=1, amount=1)
                
            elif method == "normalize":
                # Simple normalization
                img1 = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
                img2 = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
                img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1) + 1e-8)
                img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2) + 1e-8)
            
            # Convert back to appropriate dtype
            img1 = (img1 * 255).astype(np.uint8)
            img2 = (img2 * 255).astype(np.uint8)
            
            pair.modality1_image = img1
            pair.modality2_image = img2
            pair.preprocessing_applied = True
            pair.metadata['preprocessing_method'] = method
            
            return pair
            
        except Exception as e:
            self.logger.warning(f"Preprocessing failed: {e}, using original images")
            return pair
    
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
