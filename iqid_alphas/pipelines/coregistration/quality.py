#!/usr/bin/env python3
"""
Coregistration Quality Assessment Module

Handles quality assessment without ground truth using proxy metrics.
"""

import numpy as np
from typing import Dict, Any, Optional
from skimage.metrics import structural_similarity
from scipy.stats import entropy


class CoregistrationQualityValidator:
    """Validates coregistration quality using proxy metrics."""
    
    def __init__(self, config):
        self.config = config
    
    def assess_registration_quality(self, iqid_image: np.ndarray, he_image: np.ndarray, 
                                   registration_result) -> Dict[str, Any]:
        """
        Assess registration quality using proxy metrics.
        
        Args:
            iqid_image: Original iQID image
            he_image: Original H&E image
            registration_result: Result from registration process
            
        Returns:
            Dictionary with quality assessment metrics
        """
        quality_metrics = {}
        
        try:
            # Calculate mutual information
            mutual_info = self._calculate_mutual_information(iqid_image, he_image)
            quality_metrics['mutual_information'] = mutual_info
            
            # Calculate structural similarity
            # Note: SSIM works best on similar modalities, so this is a rough proxy
            if iqid_image.shape == he_image.shape:
                ssim_score = structural_similarity(iqid_image, he_image)
                quality_metrics['structural_similarity'] = ssim_score
            else:
                quality_metrics['structural_similarity'] = 0.0
            
            # Extract registration-specific metrics
            if hasattr(registration_result, 'confidence'):
                quality_metrics['registration_confidence'] = registration_result.confidence
            else:
                quality_metrics['registration_confidence'] = 0.5  # Default
            
            if hasattr(registration_result, 'feature_matches'):
                quality_metrics['feature_matches'] = registration_result.feature_matches
            else:
                quality_metrics['feature_matches'] = 0
            
            if hasattr(registration_result, 'transform'):
                quality_metrics['transform_parameters'] = self._extract_transform_params(
                    registration_result.transform
                )
            else:
                quality_metrics['transform_parameters'] = {}
            
            # Assess overall registration success
            quality_metrics['registration_success'] = self._assess_registration_success(quality_metrics)
            
        except Exception as e:
            # If quality assessment fails, return minimal metrics
            quality_metrics = {
                'mutual_information': 0.0,
                'structural_similarity': 0.0,
                'registration_confidence': 0.0,
                'feature_matches': 0,
                'transform_parameters': {},
                'registration_success': False,
                'quality_assessment_error': str(e)
            }
        
        return quality_metrics
    
    def _calculate_mutual_information(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate mutual information between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Mutual information score
        """
        try:
            # Ensure images are the same size
            if img1.shape != img2.shape:
                # Resize the second image to match the first
                import cv2
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Calculate histogram
            hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=50)
            
            # Normalize
            hist_2d = hist_2d / np.sum(hist_2d)
            
            # Calculate marginal distributions
            px = np.sum(hist_2d, axis=1)
            py = np.sum(hist_2d, axis=0)
            
            # Calculate mutual information
            # MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
            mi = 0.0
            for i in range(len(px)):
                for j in range(len(py)):
                    if hist_2d[i, j] > 0 and px[i] > 0 and py[j] > 0:
                        mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (px[i] * py[j]))
            
            return mi
            
        except Exception:
            return 0.0
    
    def _extract_transform_params(self, transform) -> Dict[str, Any]:
        """Extract parameters from transformation matrix/object."""
        try:
            if hasattr(transform, 'params') and hasattr(transform.params, 'shape'):
                # Likely a scikit-image transform
                if transform.params.shape == (3, 3):
                    # 2D affine transform
                    return {
                        'translation_x': float(transform.params[0, 2]),
                        'translation_y': float(transform.params[1, 2]),
                        'scale_x': float(np.sqrt(transform.params[0, 0]**2 + transform.params[0, 1]**2)),
                        'scale_y': float(np.sqrt(transform.params[1, 0]**2 + transform.params[1, 1]**2)),
                        'rotation': float(np.arctan2(transform.params[1, 0], transform.params[0, 0]))
                    }
            elif isinstance(transform, np.ndarray):
                # Direct transformation matrix
                if transform.shape == (3, 3):
                    return {
                        'translation_x': float(transform[0, 2]),
                        'translation_y': float(transform[1, 2]),
                        'scale_x': float(np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)),
                        'scale_y': float(np.sqrt(transform[1, 0]**2 + transform[1, 1]**2))
                    }
            
            return {'transform_type': str(type(transform))}
            
        except Exception:
            return {}
    
    def _assess_registration_success(self, quality_metrics: Dict[str, Any]) -> bool:
        """
        Assess if registration was successful based on quality metrics.
        
        Args:
            quality_metrics: Dictionary of calculated quality metrics
            
        Returns:
            bool: True if registration is considered successful
        """
        # Check mutual information threshold
        mi = quality_metrics.get('mutual_information', 0.0)
        if mi < self.config.mutual_info_threshold:
            return False
        
        # Check structural similarity threshold (if available)
        ssim = quality_metrics.get('structural_similarity', 0.0)
        if ssim > 0 and ssim < self.config.ssim_threshold:
            return False
        
        # Check feature matches (if available)
        feature_matches = quality_metrics.get('feature_matches', 0)
        if feature_matches > 0 and feature_matches < self.config.feature_match_threshold:
            return False
        
        # Check registration confidence
        confidence = quality_metrics.get('registration_confidence', 0.0)
        if confidence < 0.3:  # Low confidence threshold
            return False
        
        return True
