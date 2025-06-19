"""
Quality Assessment Metrics

Functions for calculating registration quality metrics.
"""

import logging
import numpy as np
import cv2
from scipy.stats import entropy
from typing import Dict, List

from .types import QualityMetric


class QualityAssessor:
    """Registration quality assessment."""
    
    def __init__(self, metrics: List[QualityMetric] = None):
        """
        Initialize quality assessor.
        
        Parameters
        ----------
        metrics : List[QualityMetric], optional
            List of metrics to calculate
        """
        self.metrics = metrics or [
            QualityMetric.MUTUAL_INFORMATION,
            QualityMetric.STRUCTURAL_SIMILARITY
        ]
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, fixed: np.ndarray, registered: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for registration assessment."""
        metrics = {}
        
        try:
            # Ensure same size
            if fixed.shape != registered.shape:
                registered = cv2.resize(registered, fixed.shape[::-1])
            
            # Convert to same dtype
            fixed = fixed.astype(np.float32)
            registered = registered.astype(np.float32)
            
            # Mutual Information
            if QualityMetric.MUTUAL_INFORMATION in self.metrics:
                mi = self._calculate_mutual_information(fixed, registered)
                metrics['mutual_information'] = mi
            
            # Normalized Mutual Information
            if QualityMetric.NORMALIZED_MUTUAL_INFORMATION in self.metrics:
                mi = self._calculate_mutual_information(fixed, registered)
                h1 = entropy(fixed.flatten())
                h2 = entropy(registered.flatten())
                nmi = 2 * mi / (h1 + h2) if (h1 + h2) > 0 else 0.0
                metrics['normalized_mutual_information'] = nmi
            
            # Structural Similarity (using simple correlation as approximation)
            if QualityMetric.STRUCTURAL_SIMILARITY in self.metrics:
                # Normalize images
                fixed_norm = (fixed - np.mean(fixed)) / (np.std(fixed) + 1e-8)
                registered_norm = (registered - np.mean(registered)) / (np.std(registered) + 1e-8)
                
                # Calculate correlation
                correlation = np.corrcoef(fixed_norm.flatten(), registered_norm.flatten())[0, 1]
                metrics['structural_similarity'] = max(0.0, correlation)
            
            # Cross Correlation
            if QualityMetric.CROSS_CORRELATION in self.metrics:
                correlation = cv2.matchTemplate(fixed, registered, cv2.TM_CCOEFF_NORMED)
                metrics['cross_correlation'] = np.max(correlation)
            
            # Mean Squared Error
            if QualityMetric.MEAN_SQUARED_ERROR in self.metrics:
                mse = np.mean((fixed - registered) ** 2)
                metrics['mean_squared_error'] = mse
            
            # Peak Signal-to-Noise Ratio
            if QualityMetric.PEAK_SIGNAL_NOISE_RATIO in self.metrics:
                mse = np.mean((fixed - registered) ** 2)
                if mse > 0:
                    max_val = max(np.max(fixed), np.max(registered))
                    psnr = 20 * np.log10(max_val / np.sqrt(mse))
                    metrics['peak_signal_noise_ratio'] = psnr
                else:
                    metrics['peak_signal_noise_ratio'] = float('inf')
            
        except Exception as e:
            self.logger.warning(f"Quality metric calculation failed: {e}")
        
        return metrics
    
    def _calculate_mutual_information(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate mutual information between two images."""
        try:
            # Convert to uint8 and flatten
            img1_flat = (img1.flatten()).astype(np.uint8)
            img2_flat = (img2.flatten()).astype(np.uint8)
            
            # Calculate joint histogram
            hist_2d, _, _ = np.histogram2d(img1_flat, img2_flat, bins=256)
            
            # Convert to probabilities
            joint_prob = hist_2d / np.sum(hist_2d)
            
            # Calculate marginal probabilities
            prob1 = np.sum(joint_prob, axis=1)
            prob2 = np.sum(joint_prob, axis=0)
            
            # Calculate mutual information
            mi = 0.0
            for i in range(len(prob1)):
                for j in range(len(prob2)):
                    if joint_prob[i, j] > 0 and prob1[i] > 0 and prob2[j] > 0:
                        mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (prob1[i] * prob2[j]))
            
            return mi
            
        except Exception as e:
            self.logger.warning(f"Mutual information calculation failed: {e}")
            return 0.0
