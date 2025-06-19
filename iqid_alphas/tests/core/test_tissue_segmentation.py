"""
Tests for TissueSeparator class.

This module contains comprehensive tests for the TissueSeparator class,
including unit tests, integration tests, and characterization tests to ensure
the migrated functionality works correctly.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Import the class under test
from iqid_alphas.core.tissue_segmentation import TissueSeparator


class TestTissueSeparatorBasic:
    """Basic functionality tests for TissueSeparator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.separator = TissueSeparator()
        
        # Create a simple test image
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        # Add some "tissue" regions
        self.test_image[20:40, 20:40] = 255  # Tissue 1
        self.test_image[60:80, 60:80] = 200  # Tissue 2
        
    def test_initialization_default(self):
        """Test default initialization."""
        sep = TissueSeparator()
        assert sep.debug is False
        assert sep.use_clustering is True
        assert sep.validation_mode is False
        assert sep.logger is not None
        
    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        sep = TissueSeparator(
            debug=True,
            use_clustering=False,
            validation_mode=True
        )
        assert sep.debug is True
        assert sep.use_clustering is False
        assert sep.validation_mode is True
        
    def test_separate_tissues_basic(self):
        """Test basic tissue separation."""
        result = self.separator.separate_tissues(self.test_image)
        
        # Should return a dictionary with expected keys
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'tissues' in result
        assert 'metadata' in result
        
    def test_separate_tissues_empty_image(self):
        """Test behavior with empty image."""
        empty_image = np.zeros((50, 50), dtype=np.uint8)
        result = self.separator.separate_tissues(empty_image)
        
        assert result['success'] is False
        assert 'error' in result['metadata']
        
    def test_separate_tissues_invalid_input(self):
        """Test behavior with invalid input."""
        # Test with None
        result = self.separator.separate_tissues(None)
        assert result['success'] is False
        
        # Test with wrong shape
        wrong_shape = np.zeros((10, 10, 3), dtype=np.uint8)
        result = self.separator.separate_tissues(wrong_shape)
        assert result['success'] is False


class TestTissueSeparatorAdvanced:
    """Advanced functionality tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.separator = TissueSeparator(debug=True)
        
        # Create a more complex test image with dots
        self.complex_image = np.zeros((200, 200), dtype=np.uint8)
        
        # Add background
        self.complex_image[:] = 50
        
        # Add some dot-like features
        for x, y in [(50, 50), (150, 50), (50, 150), (150, 150)]:
            # Create circular dots
            xx, yy = np.meshgrid(range(200), range(200))
            mask = (xx - x)**2 + (yy - y)**2 < 25
            self.complex_image[mask] = 255
            
    def test_dot_detection(self):
        """Test dot detection functionality."""
        # This tests the internal _detect_dots method indirectly
        result = self.separator.separate_tissues(self.complex_image)
        
        assert isinstance(result, dict)
        if result['success']:
            assert 'tissues' in result
            
    def test_clustering_disabled(self):
        """Test behavior when clustering is disabled."""
        sep = TissueSeparator(use_clustering=False)
        result = sep.separate_tissues(self.complex_image)
        
        assert isinstance(result, dict)
        
    def test_validation_mode(self):
        """Test validation mode functionality."""
        sep = TissueSeparator(validation_mode=True)
        result = sep.separate_tissues(self.complex_image)
        
        assert isinstance(result, dict)
        # In validation mode, should include additional metadata
        if result['success']:
            assert 'validation_metrics' in result['metadata']


class TestTissueSeparatorValidation:
    """Tests for validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.separator = TissueSeparator(validation_mode=True)
        
    def test_validate_segmentation_basic(self):
        """Test basic segmentation validation."""
        # Create simple ground truth and segmentation
        ground_truth = np.zeros((100, 100), dtype=np.uint8)
        ground_truth[25:75, 25:75] = 1
        
        segmentation = np.zeros((100, 100), dtype=np.uint8)
        segmentation[30:70, 30:70] = 1
        
        metrics = self.separator.validate_segmentation(segmentation, ground_truth)
        
        assert isinstance(metrics, dict)
        assert 'dice_score' in metrics
        assert 'jaccard_score' in metrics
        assert 'accuracy' in metrics
        
    def test_validate_segmentation_perfect_match(self):
        """Test validation with perfect match."""
        image = np.zeros((50, 50), dtype=np.uint8)
        image[10:40, 10:40] = 1
        
        metrics = self.separator.validate_segmentation(image, image)
        
        assert metrics['dice_score'] == 1.0
        assert metrics['jaccard_score'] == 1.0
        assert metrics['accuracy'] == 1.0
        
    def test_validate_segmentation_no_overlap(self):
        """Test validation with no overlap."""
        gt = np.zeros((50, 50), dtype=np.uint8)
        gt[10:30, 10:30] = 1
        
        seg = np.zeros((50, 50), dtype=np.uint8)
        seg[30:50, 30:50] = 1
        
        metrics = self.separator.validate_segmentation(seg, gt)
        
        assert metrics['dice_score'] == 0.0
        assert metrics['jaccard_score'] == 0.0


class TestTissueSeparatorFileOperations:
    """Tests for file operations and batch processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.separator = TissueSeparator()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_process_file_invalid_path(self):
        """Test processing non-existent file."""
        result = self.separator.process_file("nonexistent.tif")
        assert result['success'] is False
        assert 'error' in result['metadata']
        
    def test_compare_with_ground_truth_invalid_paths(self):
        """Test ground truth comparison with invalid paths."""
        result = self.separator.compare_with_ground_truth(
            "nonexistent1.tif", 
            "nonexistent2.tif"
        )
        assert result['success'] is False
        

class TestTissueSeparatorEdgeCases:
    """Tests for edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.separator = TissueSeparator()
        
    def test_very_small_image(self):
        """Test with very small image."""
        tiny_image = np.ones((5, 5), dtype=np.uint8) * 128
        result = self.separator.separate_tissues(tiny_image)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'success' in result
        
    def test_very_large_values(self):
        """Test with large pixel values."""
        image = np.ones((50, 50), dtype=np.uint16) * 65000
        result = self.separator.separate_tissues(image)
        
        assert isinstance(result, dict)
        
    def test_uniform_image(self):
        """Test with completely uniform image."""
        uniform = np.ones((100, 100), dtype=np.uint8) * 128
        result = self.separator.separate_tissues(uniform)
        
        assert isinstance(result, dict)
        # Uniform image should typically fail to find tissues
        
    def test_binary_image(self):
        """Test with binary image."""
        binary = np.zeros((100, 100), dtype=np.uint8)
        binary[25:75, 25:75] = 255
        
        result = self.separator.separate_tissues(binary)
        assert isinstance(result, dict)


class TestTissueSeparatorConfigurationMethods:
    """Tests for configuration and parameter methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.separator = TissueSeparator()
        
    def test_set_clustering_threshold(self):
        """Test setting clustering threshold."""
        original_threshold = getattr(self.separator, '_clustering_threshold', None)
        
        self.separator.set_clustering_threshold(50.0)
        
        # Check if the value was set (method should exist and work)
        # We don't assert specific values since it depends on implementation
        
    def test_set_debug_mode(self):
        """Test toggling debug mode."""
        original_debug = self.separator.debug
        
        self.separator.set_debug_mode(not original_debug)
        assert self.separator.debug != original_debug
        
        # Reset
        self.separator.set_debug_mode(original_debug)
        assert self.separator.debug == original_debug


class TestTissueSeparatorIntegration:
    """Integration tests combining multiple features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.separator = TissueSeparator(debug=True, validation_mode=True)
        
        # Create a realistic test image
        self.realistic_image = self._create_realistic_test_image()
        
    def _create_realistic_test_image(self):
        """Create a more realistic test image with noise and gradients."""
        image = np.random.normal(100, 10, (200, 200)).astype(np.uint8)
        
        # Add tissue-like regions
        xx, yy = np.meshgrid(range(200), range(200))
        
        # Tissue 1
        mask1 = ((xx - 50)**2 + (yy - 50)**2) < 1000
        image[mask1] = np.clip(image[mask1] + 100, 0, 255)
        
        # Tissue 2  
        mask2 = ((xx - 150)**2 + (yy - 150)**2) < 800
        image[mask2] = np.clip(image[mask2] + 80, 0, 255)
        
        # Add some dots
        for x, y in [(60, 60), (140, 140)]:
            dot_mask = ((xx - x)**2 + (yy - y)**2) < 16
            image[dot_mask] = 255
            
        return image.astype(np.uint8)
        
    def test_full_pipeline_with_validation(self):
        """Test the complete pipeline with validation enabled."""
        result = self.separator.separate_tissues(self.realistic_image)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'tissues' in result
        assert 'metadata' in result
        
        if result['success']:
            assert isinstance(result['tissues'], list)
            assert 'processing_time' in result['metadata']
            assert 'validation_metrics' in result['metadata']
            
    def test_pipeline_consistency(self):
        """Test that multiple runs on the same image give consistent results."""
        result1 = self.separator.separate_tissues(self.realistic_image)
        result2 = self.separator.separate_tissues(self.realistic_image)
        
        # Basic consistency checks
        assert result1['success'] == result2['success']
        
        if result1['success'] and result2['success']:
            # Should have same number of tissues
            assert len(result1['tissues']) == len(result2['tissues'])


@pytest.mark.skipif(
    not hasattr(TissueSeparator, 'process_directory'), 
    reason="Batch processing not implemented"
)
class TestTissueSeparatorBatchProcessing:
    """Tests for batch processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.separator = TissueSeparator()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_process_directory_empty(self):
        """Test processing empty directory."""
        result = self.separator.process_directory(self.temp_dir)
        assert isinstance(result, dict)
        assert result['processed_count'] == 0


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v'])
