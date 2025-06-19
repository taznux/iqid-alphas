#!/usr/bin/env python3
"""
Coregistration Processing Core Module

Handles the main coregistration processing operations.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from ...core.coregistration import MultiModalAligner, RegistrationMethod
from ...core.validation import ValidationResult, ValidationMode
from .utils import CoregistrationResult, CoregistrationDataLoader
from .quality import CoregistrationQualityValidator


class CoregistrationProcessor:
    """Core coregistration processing operations."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        self.data_loader = CoregistrationDataLoader(config)
        self.quality_validator = CoregistrationQualityValidator(config)
        
        # Setup aligner with proper method mapping
        self.aligner = self._create_aligner()
    
    def _create_aligner(self) -> MultiModalAligner:
        """Create and configure the MultiModalAligner."""
        method_mapping = {
            "mutual_information": RegistrationMethod.MUTUAL_INFORMATION,
            "feature_based": RegistrationMethod.FEATURE_BASED,
            "intensity_based": RegistrationMethod.INTENSITY_BASED,
            "hybrid": RegistrationMethod.HYBRID
        }
        
        reg_method = method_mapping.get(
            self.config.registration_method.lower(), 
            RegistrationMethod.MUTUAL_INFORMATION
        )
        
        return MultiModalAligner(method=reg_method)
    
    def process_sample(self, sample: Dict[str, Any], logger) -> ValidationResult:
        """
        Process a single coregistration sample.
        
        Args:
            sample: Sample dictionary with metadata
            logger: Logger instance
            
        Returns:
            ValidationResult with coregistration outcomes
        """
        sample_id = sample['id']
        iqid_dir = sample['iqid_dir']
        he_dir = sample['he_dir']
        start_time = time.time()
        
        try:
            logger.debug(f"Processing coregistration sample: {sample_id}")
            
            # Load image stacks
            iqid_files = self.data_loader.load_slice_stack(iqid_dir)
            he_files = self.data_loader.load_slice_stack(he_dir)
            
            if not iqid_files or not he_files:
                raise ValueError(f"No images found in {iqid_dir} or {he_dir}")
            
            # Load first representative images for registration
            iqid_image = self.data_loader.load_image_safe(iqid_files[0])
            he_image = self.data_loader.load_image_safe(he_files[0])
            
            if iqid_image is None or he_image is None:
                raise ValueError("Failed to load images for registration")
            
            logger.debug(f"Loaded images: iQID {iqid_image.shape}, H&E {he_image.shape}")
            
            # Perform coregistration
            registration_result = self.aligner.register_images(iqid_image, he_image)
            
            # Validate registration quality
            quality_metrics = self.quality_validator.assess_registration_quality(
                iqid_image, he_image, registration_result
            )
            
            # Create detailed result
            coreg_result = CoregistrationResult(
                pair_id=sample_id,
                iqid_dir=str(iqid_dir),
                he_dir=str(he_dir),
                success=quality_metrics['registration_success'],
                registration_method=self.config.registration_method,
                pairing_confidence=sample.get('pairing_confidence', 0.0),
                processing_time=time.time() - start_time,
                iqid_slice_count=len(iqid_files),
                he_slice_count=len(he_files),
                mutual_information=quality_metrics.get('mutual_information', 0.0),
                structural_similarity=quality_metrics.get('structural_similarity', 0.0),
                registration_confidence=quality_metrics.get('registration_confidence', 0.0),
                feature_matches=quality_metrics.get('feature_matches', 0),
                transform_parameters=quality_metrics.get('transform_parameters', {})
            )
            
            # Convert to ValidationResult for pipeline compatibility
            validation_result = self._create_validation_result(coreg_result, sample_id, start_time)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to process coregistration sample {sample_id}: {e}")
            return self._create_error_result(sample_id, str(e), start_time)
    
    def _create_validation_result(self, coreg_result: CoregistrationResult, 
                                  sample_id: str, start_time: float) -> ValidationResult:
        """Create a ValidationResult from CoregistrationResult."""
        metrics = {
            'mutual_information': coreg_result.mutual_information,
            'structural_similarity': coreg_result.structural_similarity,
            'registration_confidence': coreg_result.registration_confidence,
            'feature_matches': coreg_result.feature_matches,
            'pairing_confidence': coreg_result.pairing_confidence,
            'iqid_slice_count': coreg_result.iqid_slice_count,
            'he_slice_count': coreg_result.he_slice_count,
            'registration_method': coreg_result.registration_method,
            'processing_stage': 'coregistration'
        }
        
        result = ValidationResult(
            mode=ValidationMode.COREGISTRATION,
            metrics=metrics,
            processing_time=coreg_result.processing_time,
            passed_tests=1 if coreg_result.success else 0,
            total_tests=1,
            metadata={'sample_id': sample_id}
        )
        
        # Add compatibility attributes for base pipeline
        result.success = coreg_result.success
        result.error_message = coreg_result.error_message
        result.validation_type = "coregistration"
        result.sample_id = sample_id
        
        # Add to_dict method for pipeline compatibility
        def to_dict(self):
            return {
                'sample_id': getattr(self, 'sample_id', sample_id),
                'success': getattr(self, 'success', True),
                'metrics': dict(self.metrics),
                'processing_time': self.processing_time,
                'error_message': getattr(self, 'error_message', None),
                'validation_type': getattr(self, 'validation_type', 'coregistration')
            }
        result.to_dict = to_dict.__get__(result, type(result))
        
        return result
    
    def _create_error_result(self, sample_id: str, error_message: str, 
                             start_time: float) -> ValidationResult:
        """Create an error validation result."""
        result = ValidationResult(
            mode=ValidationMode.COREGISTRATION,
            metrics={},
            processing_time=time.time() - start_time,
            passed_tests=0,
            total_tests=1,
            errors=[error_message],
            metadata={'sample_id': sample_id}
        )
        
        # Add compatibility attributes for base pipeline
        result.success = False
        result.error_message = error_message
        result.validation_type = "coregistration"
        result.sample_id = sample_id
        
        # Add to_dict method for pipeline compatibility
        def create_to_dict_method(error_msg, sid):
            def to_dict(self):
                return {
                    'sample_id': getattr(self, 'sample_id', sid),
                    'success': getattr(self, 'success', False),
                    'metrics': dict(self.metrics),
                    'processing_time': self.processing_time,
                    'error_message': getattr(self, 'error_message', error_msg),
                    'validation_type': getattr(self, 'validation_type', 'coregistration')
                }
            return to_dict
        
        result.to_dict = create_to_dict_method(error_message, sample_id).__get__(result, type(result))
        
        return result
