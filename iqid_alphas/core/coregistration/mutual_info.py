"""
Mutual Information Registration

Implements mutual information-based registration for multi-modal images.
"""

import logging
import numpy as np
import cv2
from scipy import optimize
from typing import Dict, Any

from .types import RegistrationResult, RegistrationMethod, ModalityPair


class MutualInfoAligner:
    """Mutual information-based image registration."""
    
    def __init__(self, 
                 optimization_method: str = "powell",
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6):
        """
        Initialize mutual information aligner.
        
        Parameters
        ----------
        optimization_method : str
            Optimization method ('powell', 'bfgs', 'nelder-mead')
        max_iterations : int
            Maximum optimization iterations
        convergence_threshold : float
            Convergence threshold
        """
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.logger = logging.getLogger(__name__)
    
    def register(self, pair: ModalityPair) -> RegistrationResult:
        """Register using mutual information optimization."""
        fixed = pair.modality1_image.astype(np.float32)
        moving = pair.modality2_image.astype(np.float32)
        
        # Define optimization objective
        def objective(params):
            # params: [tx, ty, rotation, scale_x, scale_y]
            if len(params) == 2:
                # Translation only
                tx, ty = params
                transform_matrix = np.array([
                    [1, 0, tx],
                    [0, 1, ty],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                # Full affine transform
                tx, ty, rotation, scale_x, scale_y = params
                cos_r = np.cos(rotation)
                sin_r = np.sin(rotation)
                transform_matrix = np.array([
                    [scale_x * cos_r, -scale_x * sin_r, tx],
                    [scale_y * sin_r, scale_y * cos_r, ty],
                    [0, 0, 1]
                ], dtype=np.float32)
            
            # Apply transformation
            transformed = cv2.warpAffine(moving, transform_matrix[:2, :], fixed.shape[::-1])
            
            # Calculate mutual information (negative for minimization)
            mi = self._calculate_mutual_information(fixed, transformed)
            return -mi
        
        try:
            # Start with translation-only optimization
            initial_params = [0.0, 0.0]  # tx, ty
            
            result = optimize.minimize(
                objective,
                initial_params,
                method=self.optimization_method,
                options={'maxiter': self.max_iterations}
            )
            
            # Build final transformation matrix
            if result.success:
                tx, ty = result.x
                transform_matrix = np.array([
                    [1, 0, tx],
                    [0, 1, ty],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Apply final transformation
                registered_image = cv2.warpAffine(
                    moving, transform_matrix[:2, :], fixed.shape[::-1]
                )
                
                # Calculate confidence based on mutual information improvement
                original_mi = self._calculate_mutual_information(fixed, moving)
                final_mi = self._calculate_mutual_information(fixed, registered_image)
                confidence = min(1.0, max(0.0, (final_mi - original_mi) / (original_mi + 1e-8)))
                
                return RegistrationResult(
                    transform_matrix=transform_matrix,
                    registered_image=registered_image,
                    confidence=confidence,
                    method=RegistrationMethod.MUTUAL_INFORMATION,
                    processing_time=0.0,  # Will be set by caller
                    metadata={
                        'optimization_success': True,
                        'optimization_iterations': result.nit,
                        'final_mutual_information': final_mi,
                        'original_mutual_information': original_mi
                    }
                )
            else:
                # Optimization failed, return identity
                return RegistrationResult(
                    transform_matrix=np.eye(3),
                    registered_image=moving.copy(),
                    confidence=0.0,
                    method=RegistrationMethod.MUTUAL_INFORMATION,
                    processing_time=0.0,
                    metadata={'optimization_success': False, 'error': 'optimization_failed'}
                )
                
        except Exception as e:
            self.logger.error(f"Mutual information registration failed: {e}")
            return RegistrationResult(
                transform_matrix=np.eye(3),
                registered_image=moving.copy(),
                confidence=0.0,
                method=RegistrationMethod.MUTUAL_INFORMATION,
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
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
