"""
Tests for the core mapping module.

Tests both segmentation evaluation (crop location finding) and alignment evaluation
(transformation calculation) functionality.
"""

import numpy as np
import pytest
import cv2
from unittest.mock import patch, MagicMock

from iqid_alphas.core.mapping import (
    ReverseMapper, 
    CropLocation, 
    AlignmentTransform,
    find_crop_location,
    calculate_alignment_transform
)


class TestReverseMapper:
    """Test the ReverseMapper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = ReverseMapper()
        
        # Create test images with distinctive patterns
        self.source_image = np.zeros((200, 200), dtype=np.uint8)
        # Add distinctive pattern in known location
        cv2.rectangle(self.source_image, (50, 50), (100, 100), 255, -1)  # White square
        cv2.circle(self.source_image, (75, 75), 10, 128, -1)  # Gray circle in center
        
        # Template extracted from the known location  
        self.template_image = self.source_image[50:100, 50:100].copy()
        
        # Create a known transformed image for alignment testing
        self.original_alignment = np.zeros((100, 100), dtype=np.uint8)
        self.original_alignment[25:75, 25:75] = 255
        
        # Apply known transformation (translation)
        self.aligned_image = np.zeros((100, 100), dtype=np.uint8)
        self.aligned_image[30:80, 30:80] = 255  # Shifted by (5, 5)
    
    def test_initialization_default(self):
        """Test default initialization."""
        mapper = ReverseMapper()
        assert mapper.template_matching_method == cv2.TM_CCOEFF_NORMED
        assert mapper.feature_detector == 'ORB'
        assert mapper.max_features == 5000
        assert mapper.match_threshold == 0.75
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        mapper = ReverseMapper(
            template_matching_method=cv2.TM_CCORR_NORMED,
            feature_detector='SIFT',
            max_features=1000,
            match_threshold=0.8
        )
        assert mapper.template_matching_method == cv2.TM_CCORR_NORMED
        assert mapper.feature_detector == 'SIFT'
        assert mapper.max_features == 1000
        assert mapper.match_threshold == 0.8
    
    def test_find_crop_location_template_matching(self):
        """Test crop location finding using template matching."""
        result = self.mapper.find_crop_location(self.source_image, self.template_image)
        
        assert isinstance(result, CropLocation)
        assert result.method == 'template_matching'
        assert result.confidence > 0.8  # Should have high confidence for exact match
        
        # Check that bbox is correct (should find the exact location)
        x, y, w, h = result.bbox
        assert w == 50  # Template width
        assert h == 50  # Template height
        assert x == 50  # Should be exact match
        assert y == 50  # Should be exact match
    
    def test_find_crop_location_exact_match(self):
        """Test crop location finding with an exact template match."""
        # Extract exact template from source
        exact_template = self.source_image[50:100, 50:100].copy()
        
        result = self.mapper.find_crop_location(self.source_image, exact_template)
        
        assert isinstance(result, CropLocation)
        assert result.confidence > 0.9  # Should have very high confidence
        
        x, y, w, h = result.bbox
        assert x == 50
        assert y == 50
        assert w == 50
        assert h == 50
    
    def test_find_crop_location_no_match(self):
        """Test crop location finding when template doesn't match."""
        # Create a template that doesn't exist in source
        no_match_template = np.ones((20, 20), dtype=np.uint8) * 128  # Gray square
        
        result = self.mapper.find_crop_location(self.source_image, no_match_template)
        
        assert isinstance(result, CropLocation)
        assert result.method == 'template_matching'
        # Should still return a result, but with lower confidence
        assert 0.0 <= result.confidence <= 1.0
    
    @patch('iqid_alphas.core.mapping.HAS_SKIMAGE', True)
    @patch('iqid_alphas.core.mapping.phase_cross_correlation')
    def test_find_crop_location_phase_correlation(self, mock_phase_corr):
        """Test crop location finding using phase correlation."""
        # Mock the phase correlation result
        mock_phase_corr.return_value = (np.array([5, 5]), 0.1, None)  # shift, error, diffphase
        
        result = self.mapper.find_crop_location(
            self.source_image, self.template_image, method='phase_correlation'
        )
        
        assert isinstance(result, CropLocation)
        assert result.method == 'phase_correlation'
        mock_phase_corr.assert_called_once()
    
    def test_calculate_alignment_transform_feature_matching(self):
        """Test alignment transform calculation using feature matching."""
        result = self.mapper.calculate_alignment_transform(
            self.original_alignment, self.aligned_image
        )
        
        assert isinstance(result, AlignmentTransform)
        assert result.method == 'feature_matching'
        assert result.transform_matrix.shape == (3, 3)
        assert 0.0 <= result.match_quality <= 1.0
        assert result.num_matches >= 0
        assert 0.0 <= result.inlier_ratio <= 1.0
    
    def test_calculate_alignment_transform_known_translation(self):
        """Test alignment transform with a known translation."""
        # Create images with sufficient features for matching
        original = np.zeros((100, 100), dtype=np.uint8)
        # Add some distinctive features
        cv2.circle(original, (30, 30), 10, 255, -1)
        cv2.rectangle(original, (60, 20), (80, 40), 255, -1)
        cv2.circle(original, (20, 70), 8, 255, -1)
        
        # Apply known translation
        aligned = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(aligned, (35, 35), 10, 255, -1)  # Shifted by (5, 5)
        cv2.rectangle(aligned, (65, 25), (85, 45), 255, -1)
        cv2.circle(aligned, (25, 75), 8, 255, -1)
        
        result = self.mapper.calculate_alignment_transform(original, aligned)
        
        # Check that some transformation was detected
        assert isinstance(result, AlignmentTransform)
        assert result.method == 'feature_matching'
        
        # If we have good matches, the translation should be approximately (5, 5)
        if result.match_quality > 0.5:
            transform = result.transform_matrix
            tx = transform[0, 2]  # x translation
            ty = transform[1, 2]  # y translation
            
            # Should be close to the known translation
            assert abs(tx - 5) < 10  # Allow some tolerance
            assert abs(ty - 5) < 10
    
    def test_calculate_alignment_transform_insufficient_features(self):
        """Test alignment transform with insufficient features."""
        # Create images with very few/no features
        blank1 = np.ones((50, 50), dtype=np.uint8) * 100
        blank2 = np.ones((50, 50), dtype=np.uint8) * 110
        
        result = self.mapper.calculate_alignment_transform(blank1, blank2)
        
        assert isinstance(result, AlignmentTransform)
        assert result.match_quality == 0.0
        assert result.num_matches == 0
        # Should return identity matrix
        np.testing.assert_array_almost_equal(result.transform_matrix, np.eye(3))
    
    @patch('iqid_alphas.core.mapping.HAS_SKIMAGE', True)
    @patch('iqid_alphas.core.mapping.phase_cross_correlation')
    def test_calculate_alignment_transform_phase_correlation(self, mock_phase_corr):
        """Test alignment transform using phase correlation."""
        # Mock the phase correlation result
        mock_phase_corr.return_value = (np.array([3, 4]), 0.1, None)  # shift, error, diffphase
        
        result = self.mapper.calculate_alignment_transform(
            self.original_alignment, self.aligned_image, method='phase_correlation'
        )
        
        assert isinstance(result, AlignmentTransform)
        assert result.method == 'phase_correlation'
        
        # Check that translation matrix is correct
        expected_matrix = np.array([
            [1, 0, 4],  # x translation
            [0, 1, 3],  # y translation
            [0, 0, 1]
        ], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result.transform_matrix, expected_matrix)
        mock_phase_corr.assert_called_once()
    
    def test_invalid_method_crop_location(self):
        """Test that invalid method raises ValueError for crop location."""
        with pytest.raises(ValueError, match="Unknown method"):
            self.mapper.find_crop_location(
                self.source_image, self.template_image, method='invalid_method'
            )
    
    def test_invalid_method_alignment_transform(self):
        """Test that invalid method raises ValueError for alignment transform."""
        with pytest.raises(ValueError, match="Unknown method"):
            self.mapper.calculate_alignment_transform(
                self.original_alignment, self.aligned_image, method='invalid_method'
            )


class TestConvenienceFunctions:
    """Test the convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.source_image = np.zeros((100, 100), dtype=np.uint8)
        self.source_image[25:75, 25:75] = 255
        
        self.template_image = np.ones((25, 25), dtype=np.uint8) * 255
        
        self.original_image = np.zeros((50, 50), dtype=np.uint8)
        self.original_image[10:40, 10:40] = 255
        
        self.aligned_image = np.zeros((50, 50), dtype=np.uint8)
        self.aligned_image[12:42, 12:42] = 255  # Shifted by (2, 2)
    
    def test_find_crop_location_convenience(self):
        """Test the convenience function for crop location finding."""
        result = find_crop_location(self.source_image, self.template_image)
        
        assert isinstance(result, dict)
        assert 'bbox' in result
        assert 'confidence' in result
        assert 'method' in result
        assert 'center' in result
        
        assert result['method'] == 'template_matching'
        assert isinstance(result['bbox'], tuple)
        assert len(result['bbox']) == 4
        assert isinstance(result['confidence'], float)
        assert isinstance(result['center'], tuple)
    
    def test_calculate_alignment_transform_convenience(self):
        """Test the convenience function for alignment transform calculation."""
        result = calculate_alignment_transform(self.original_image, self.aligned_image)
        
        assert isinstance(result, dict)
        assert 'transform_matrix' in result
        assert 'match_quality' in result
        assert 'method' in result
        assert 'num_matches' in result
        assert 'inlier_ratio' in result
        
        assert result['method'] == 'feature_matching'
        assert isinstance(result['transform_matrix'], list)  # Should be serializable
        assert len(result['transform_matrix']) == 3
        assert len(result['transform_matrix'][0]) == 3


class TestDataclasses:
    """Test the dataclass structures."""
    
    def test_crop_location_dataclass(self):
        """Test CropLocation dataclass."""
        crop = CropLocation(
            bbox=(10, 20, 30, 40),
            confidence=0.95,
            method='template_matching',
            center=(25, 40)
        )
        
        assert crop.bbox == (10, 20, 30, 40)
        assert crop.confidence == 0.95
        assert crop.method == 'template_matching'
        assert crop.center == (25, 40)
    
    def test_alignment_transform_dataclass(self):
        """Test AlignmentTransform dataclass."""
        transform_matrix = np.eye(3)
        alignment = AlignmentTransform(
            transform_matrix=transform_matrix,
            match_quality=0.85,
            method='feature_matching',
            num_matches=50,
            inlier_ratio=0.8
        )
        
        np.testing.assert_array_equal(alignment.transform_matrix, transform_matrix)
        assert alignment.match_quality == 0.85
        assert alignment.method == 'feature_matching'
        assert alignment.num_matches == 50
        assert alignment.inlier_ratio == 0.8


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = ReverseMapper()
    
    @patch('iqid_alphas.core.mapping.HAS_SKIMAGE', False)
    def test_fallback_when_skimage_unavailable(self):
        """Test fallback to template matching when scikit-image is unavailable."""
        source = np.zeros((50, 50), dtype=np.uint8)
        template = np.ones((10, 10), dtype=np.uint8) * 255
        
        # Should fallback to template matching
        result = self.mapper.find_crop_location(source, template, method='phase_correlation')
        assert result.method == 'template_matching'
        
        # Should fallback for alignment too
        original = np.zeros((30, 30), dtype=np.uint8)
        aligned = np.ones((30, 30), dtype=np.uint8) * 128
        
        result = self.mapper.calculate_alignment_transform(
            original, aligned, method='phase_correlation'
        )
        assert result.method == 'feature_matching'
    
    def test_color_image_handling(self):
        """Test that color images are properly converted to grayscale."""
        # Create color images
        source_color = np.zeros((50, 50, 3), dtype=np.uint8)
        source_color[10:40, 10:40] = [255, 255, 255]  # White square
        
        template_color = np.ones((15, 15, 3), dtype=np.uint8) * 255
        
        # Should handle color images without error
        result = self.mapper.find_crop_location(source_color, template_color)
        assert isinstance(result, CropLocation)
        
        # Should handle color images for alignment too
        original_color = np.zeros((30, 30, 3), dtype=np.uint8)
        aligned_color = np.ones((30, 30, 3), dtype=np.uint8) * 128
        
        result = self.mapper.calculate_alignment_transform(original_color, aligned_color)
        assert isinstance(result, AlignmentTransform)
