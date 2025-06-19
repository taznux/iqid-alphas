"""
Tests for the Multi-Modal Coregistration Module

Tests all registration methods, quality metrics, and edge cases
for the MultiModalAligner class.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from iqid_alphas.core.coregistration import (
    MultiModalAligner,
    RegistrationMethod,
    QualityMetric,
    RegistrationResult,
    ModalityPair
)


class TestMultiModalAligner:
    """Test the MultiModalAligner class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.aligner = MultiModalAligner()
        
        # Create synthetic test images
        self.create_test_images()
    
    def create_test_images(self):
        """Create synthetic test images for different modalities."""
        # Create a base synthetic image
        self.base_image = np.zeros((256, 256), dtype=np.uint8)
        
        # Add some geometric features
        cv2.rectangle(self.base_image, (50, 50), (150, 150), 128, -1)
        cv2.circle(self.base_image, (200, 200), 30, 255, -1)
        cv2.line(self.base_image, (10, 10), (240, 240), 64, 3)
        
        # Create modality 1 (reference)
        self.modality1 = self.base_image.copy()
        
        # Create modality 2 (with transformation)
        self.translation = (15, -10)
        transform_matrix = np.array([
            [1, 0, self.translation[0]],
            [0, 1, self.translation[1]]
        ], dtype=np.float32)
        
        self.modality2 = cv2.warpAffine(
            self.base_image, 
            transform_matrix, 
            (256, 256)
        )
        
        # Add some noise to simulate different modalities
        noise = np.random.normal(0, 10, self.modality2.shape).astype(np.int16)
        self.modality2 = np.clip(self.modality2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Create modality 3 (with rotation and scaling)
        angle = 5  # degrees
        scale = 1.1
        center = (128, 128)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        self.modality3 = cv2.warpAffine(
            self.base_image,
            rotation_matrix,
            (256, 256)
        )
    
    def test_initialization(self):
        """Test aligner initialization with different parameters."""
        # Test default initialization
        aligner = MultiModalAligner()
        assert aligner.registration_method == RegistrationMethod.MUTUAL_INFORMATION
        assert len(aligner.quality_metrics) == 2
        assert aligner.preprocessing_enabled is True
        
        # Test custom initialization
        custom_metrics = [QualityMetric.CROSS_CORRELATION, QualityMetric.MEAN_SQUARED_ERROR]
        aligner = MultiModalAligner(
            registration_method=RegistrationMethod.FEATURE_BASED,
            quality_metrics=custom_metrics,
            preprocessing_enabled=False,
            max_iterations=500
        )
        
        assert aligner.registration_method == RegistrationMethod.FEATURE_BASED
        assert aligner.quality_metrics == custom_metrics
        assert aligner.preprocessing_enabled is False
        assert aligner.max_iterations == 500
    
    def test_mutual_information_registration(self):
        """Test mutual information-based registration."""
        result = self.aligner.register_images(
            self.modality1,
            self.modality2,
            method=RegistrationMethod.MUTUAL_INFORMATION
        )
        
        assert isinstance(result, RegistrationResult)
        assert result.method == RegistrationMethod.MUTUAL_INFORMATION
        assert result.transform_matrix.shape == (3, 3)
        assert result.registered_image.shape == self.modality1.shape
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time >= 0.0
        assert 'mutual_information' in result.quality_metrics
    
    def test_feature_based_registration(self):
        """Test feature-based registration."""
        result = self.aligner.register_images(
            self.modality1,
            self.modality2,
            method=RegistrationMethod.FEATURE_BASED
        )
        
        assert isinstance(result, RegistrationResult)
        assert result.method == RegistrationMethod.FEATURE_BASED
        assert result.transform_matrix.shape == (3, 3)
        assert result.registered_image.shape == self.modality1.shape
        assert 0.0 <= result.confidence <= 1.0
    
    def test_intensity_based_registration(self):
        """Test intensity-based registration."""
        result = self.aligner.register_images(
            self.modality1,
            self.modality2,
            method=RegistrationMethod.INTENSITY_BASED
        )
        
        assert isinstance(result, RegistrationResult)
        assert result.method == RegistrationMethod.INTENSITY_BASED
        assert result.transform_matrix.shape == (3, 3)
        assert result.registered_image.shape == self.modality1.shape
        assert 0.0 <= result.confidence <= 1.0
    
    def test_cross_correlation_registration(self):
        """Test normalized cross-correlation registration."""
        result = self.aligner.register_images(
            self.modality1,
            self.modality2,
            method=RegistrationMethod.NORMALIZED_CROSS_CORRELATION
        )
        
        assert isinstance(result, RegistrationResult)
        assert result.method == RegistrationMethod.NORMALIZED_CROSS_CORRELATION
        assert result.transform_matrix.shape == (3, 3)
        assert result.registered_image.shape == self.modality1.shape
        assert 0.0 <= result.confidence <= 1.0
        assert 'correlation_value' in result.metadata
        assert 'translation' in result.metadata
    
    def test_phase_correlation_registration(self):
        """Test phase correlation registration."""
        result = self.aligner.register_images(
            self.modality1,
            self.modality2,
            method=RegistrationMethod.PHASE_CORRELATION
        )
        
        assert isinstance(result, RegistrationResult)
        assert result.method == RegistrationMethod.PHASE_CORRELATION
        assert result.transform_matrix.shape == (3, 3)
        assert result.registered_image.shape == self.modality1.shape
        assert 0.0 <= result.confidence <= 1.0
    
    def test_hybrid_registration(self):
        """Test hybrid registration approach."""
        result = self.aligner.register_images(
            self.modality1,
            self.modality2,
            method=RegistrationMethod.HYBRID
        )
        
        assert isinstance(result, RegistrationResult)
        assert result.method == RegistrationMethod.HYBRID
        assert result.transform_matrix.shape == (3, 3)
        assert result.registered_image.shape == self.modality1.shape
        assert 0.0 <= result.confidence <= 1.0
        assert 'hybrid_methods_tried' in result.metadata
        assert 'hybrid_confidences' in result.metadata
    
    def test_preprocessing(self):
        """Test preprocessing functionality."""
        # Test with preprocessing enabled
        aligner = MultiModalAligner(preprocessing_enabled=True)
        pair = ModalityPair(self.modality1, self.modality2)
        processed_pair = aligner._preprocess_modality_pair(pair)
        
        assert processed_pair.preprocessing_applied is True
        assert processed_pair.modality1_image.shape == self.modality1.shape
        assert processed_pair.modality2_image.shape == self.modality2.shape
        
        # Test with preprocessing disabled
        aligner = MultiModalAligner(preprocessing_enabled=False)
        result = aligner.register_images(self.modality1, self.modality2)
        assert isinstance(result, RegistrationResult)
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        # Set specific quality metrics
        metrics = [
            QualityMetric.MUTUAL_INFORMATION,
            QualityMetric.NORMALIZED_MUTUAL_INFORMATION,
            QualityMetric.STRUCTURAL_SIMILARITY,
            QualityMetric.CROSS_CORRELATION,
            QualityMetric.MEAN_SQUARED_ERROR,
            QualityMetric.PEAK_SIGNAL_NOISE_RATIO
        ]
        
        aligner = MultiModalAligner(quality_metrics=metrics)
        result = aligner.register_images(self.modality1, self.modality2)
        
        # Check that all requested metrics are calculated
        for metric in metrics:
            metric_name = metric.value
            assert metric_name in result.quality_metrics
            assert isinstance(result.quality_metrics[metric_name], (int, float))
    
    def test_stack_registration(self):
        """Test registration of image stacks."""
        # Create synthetic stacks
        stack1 = np.array([self.modality1, self.modality1, self.modality1])
        stack2 = np.array([self.modality2, self.modality2, self.modality2])
        
        results = self.aligner.register_stacks(stack1, stack2)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, RegistrationResult)
            assert result.transform_matrix.shape == (3, 3)
            assert result.registered_image.shape == self.modality1.shape
    
    def test_custom_slice_correspondence(self):
        """Test stack registration with custom slice correspondence."""
        stack1 = np.array([self.modality1, self.modality1, self.modality1])
        stack2 = np.array([self.modality2, self.modality2, self.modality2])
        
        # Custom correspondence: register slice 0 to slice 2, etc.
        correspondence = [(0, 2), (1, 1), (2, 0)]
        
        results = self.aligner.register_stacks(
            stack1, stack2, slice_correspondence=correspondence
        )
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, RegistrationResult)
            expected_modalities = f"fixed_slice_{correspondence[i][0]}"
            assert expected_modalities in str(result.metadata)
    
    def test_statistics_tracking(self):
        """Test registration statistics tracking."""
        # Reset stats
        self.aligner.reset_stats()
        
        # Perform several registrations
        methods = [
            RegistrationMethod.MUTUAL_INFORMATION,
            RegistrationMethod.FEATURE_BASED,
            RegistrationMethod.NORMALIZED_CROSS_CORRELATION
        ]
        
        for method in methods:
            self.aligner.register_images(self.modality1, self.modality2, method=method)
        
        stats = self.aligner.get_registration_stats()
        
        assert stats['total_registrations'] == 3
        assert stats['successful_registrations'] <= 3
        assert all(count >= 0 for count in stats['method_usage'].values())
        assert stats['average_confidence'] >= 0.0
        assert stats['average_processing_time'] >= 0.0
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with invalid method
        with pytest.raises(ValueError):
            self.aligner.register_images(
                self.modality1, 
                self.modality2, 
                method="invalid_method"
            )
        
        # Test with empty images
        empty_image = np.array([])
        result = self.aligner.register_images(self.modality1, empty_image)
        assert result.confidence == 0.0
        assert 'error' in result.metadata
        
        # Test with mismatched dimensions
        small_image = np.zeros((50, 50), dtype=np.uint8)
        result = self.aligner.register_images(self.modality1, small_image)
        # Should still work due to robust implementation
        assert isinstance(result, RegistrationResult)
    
    def test_mutual_information_calculation(self):
        """Test mutual information calculation."""
        mi = self.aligner._calculate_mutual_information(self.modality1, self.modality2)
        assert isinstance(mi, float)
        assert mi >= 0.0
        
        # Test with identical images (should have high MI)
        mi_identical = self.aligner._calculate_mutual_information(
            self.modality1, self.modality1
        )
        assert mi_identical >= mi  # Identical images should have higher MI
    
    def test_grayscale_conversion(self):
        """Test grayscale conversion."""
        # Create color image
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        gray_image = self.aligner._ensure_grayscale(color_image)
        assert len(gray_image.shape) == 2
        assert gray_image.shape[:2] == color_image.shape[:2]
        
        # Test with already grayscale image
        gray_input = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        gray_output = self.aligner._ensure_grayscale(gray_input)
        np.testing.assert_array_equal(gray_input, gray_output)
    
    def test_modality_pair_dataclass(self):
        """Test ModalityPair dataclass functionality."""
        pair = ModalityPair(
            modality1_image=self.modality1,
            modality2_image=self.modality2,
            modality1_name="iQID",
            modality2_name="H&E"
        )
        
        assert np.array_equal(pair.modality1_image, self.modality1)
        assert np.array_equal(pair.modality2_image, self.modality2)
        assert pair.modality1_name == "iQID"
        assert pair.modality2_name == "H&E"
        assert pair.preprocessing_applied is False
        assert isinstance(pair.metadata, dict)
    
    def test_registration_result_dataclass(self):
        """Test RegistrationResult dataclass functionality."""
        transform = np.eye(3)
        image = np.zeros((100, 100), dtype=np.uint8)
        
        result = RegistrationResult(
            transform_matrix=transform,
            registered_image=image,
            confidence=0.85,
            method=RegistrationMethod.MUTUAL_INFORMATION,
            processing_time=1.5
        )
        
        assert np.array_equal(result.transform_matrix, transform)
        assert np.array_equal(result.registered_image, image)
        assert result.confidence == 0.85
        assert result.method == RegistrationMethod.MUTUAL_INFORMATION
        assert result.processing_time == 1.5
        assert isinstance(result.quality_metrics, dict)
        assert isinstance(result.metadata, dict)
    
    def test_translation_accuracy(self):
        """Test if detected translation is reasonably accurate."""
        # Use cross-correlation which should be most accurate for translation
        result = self.aligner.register_images(
            self.modality1,
            self.modality2,
            method=RegistrationMethod.NORMALIZED_CROSS_CORRELATION
        )
        
        # Extract translation from transformation matrix
        detected_translation = (result.transform_matrix[0, 2], result.transform_matrix[1, 2])
        
        # Should be reasonably close to the original translation
        # (allowing for some error due to noise and discrete pixel grid)
        tolerance = 5  # pixels
        assert abs(detected_translation[0] - self.translation[0]) <= tolerance
        assert abs(detected_translation[1] - self.translation[1]) <= tolerance
    
    def test_robustness_with_noise(self):
        """Test registration robustness with noisy images."""
        # Add significant noise
        noise_level = 50
        noisy_modality2 = self.modality2.astype(np.int16)
        noise = np.random.normal(0, noise_level, noisy_modality2.shape)
        noisy_modality2 = np.clip(noisy_modality2 + noise, 0, 255).astype(np.uint8)
        
        # Test with different methods
        methods = [
            RegistrationMethod.MUTUAL_INFORMATION,
            RegistrationMethod.NORMALIZED_CROSS_CORRELATION,
            RegistrationMethod.HYBRID
        ]
        
        for method in methods:
            result = self.aligner.register_images(
                self.modality1, noisy_modality2, method=method
            )
            
            # Should still produce reasonable results
            assert isinstance(result, RegistrationResult)
            assert result.confidence >= 0.0
            assert result.registered_image.shape == self.modality1.shape
