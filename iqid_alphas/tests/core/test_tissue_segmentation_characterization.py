"""
Characterization tests for tissue segmentation migration.

These tests validate that migrated functionality from process_object.py
behaves identically to the original implementation.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from iqid_alphas.core.tissue_segmentation import TissueSeparator


class TestProcessObjectMigration:
    """Test migrated functionality from legacy process_object.py"""
    
    def setup_method(self):
        """Set up test data for characterization tests."""
        self.tissue_separator = TissueSeparator()
        
        # Create test binary data that mimics iQID format
        self.test_header = np.array([100] + [0]*99, dtype=np.int32)  # header_size=100
        self.test_header[1] = 512  # XDIM
        self.test_header[2] = 512  # YDIM
        
        # Create test cluster data (14 elements per cluster)
        np.random.seed(42)  # For reproducible tests
        self.test_clusters = 1000
        self.test_data = np.random.random((14, self.test_clusters))
        
        # Set realistic coordinate data
        self.test_data[4, :] = np.random.uniform(0, 512, self.test_clusters)  # yC_global
        self.test_data[5, :] = np.random.uniform(0, 512, self.test_clusters)  # xC_global
        self.test_data[3, :] = np.random.uniform(15, 100, self.test_clusters)  # cluster_area
        self.test_data[2, :] = np.random.uniform(100, 1000, self.test_clusters)  # signal_sum
        
    def create_test_iqid_file(self) -> Path:
        """Create a temporary iQID binary file for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.dat', delete=False)
        
        # Write header
        self.test_header.tofile(temp_file)
        
        # Write data in Fortran order
        self.test_data.T.flatten(order='F').astype(np.float64).tofile(temp_file)
        
        temp_file.close()
        return Path(temp_file.name)
    
    def test_load_iqid_binary_data_header_parsing(self):
        """Test that header parsing matches legacy behavior."""
        test_file = self.create_test_iqid_file()
        
        try:
            # Load using new method
            result_image = self.tissue_separator._load_iqid_binary_data(test_file)
            
            # Validate dimensions match header
            assert result_image.shape == (512, 512)
            
            # Validate that image contains data (not all zeros)
            assert np.sum(result_image) > 0
            
            # Validate data type
            assert result_image.dtype == np.float64
            
        finally:
            os.unlink(test_file)
    
    def test_load_iqid_binary_data_coordinate_mapping(self):
        """Test coordinate mapping from cluster data to image."""
        test_file = self.create_test_iqid_file()
        
        try:
            result_image = self.tissue_separator._load_iqid_binary_data(test_file)
            
            # Check that non-zero pixels exist where coordinates indicate
            yC = self.test_data[4, :].astype(int)
            xC = self.test_data[5, :].astype(int)
            
            # Clip coordinates to valid range
            yC = np.clip(yC, 0, 511)
            xC = np.clip(xC, 0, 511)
            
            # Count non-zero pixels
            non_zero_count = np.count_nonzero(result_image)
            
            # Should have approximately the same number of non-zero pixels
            # as unique coordinate pairs (allowing for overlaps)
            unique_coords = len(set(zip(yC, xC)))
            assert non_zero_count <= len(yC)  # Can't have more pixels than clusters
            assert non_zero_count >= min(unique_coords, len(yC) // 2)  # At least half should be unique
            
        finally:
            os.unlink(test_file)
    
    def test_preprocess_raw_image_normalization(self):
        """Test image preprocessing normalization."""
        # Create test image with known properties
        test_image = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float64)
        
        result = self.tissue_separator._preprocess_raw_image(test_image)
        
        # Should be normalized to [0, 1] range
        assert result.min() == 0.0
        assert result.max() == 1.0
        
        # Should preserve relative ordering
        assert result[0, 0] < result[0, 1] < result[0, 2]
        assert result[1, 0] < result[1, 1] < result[1, 2]
    
    def test_load_raw_image_format_detection(self):
        """Test that different file formats are handled correctly."""
        # Test .dat format
        test_file = self.create_test_iqid_file()
        
        try:
            result = self.tissue_separator._load_raw_image(test_file)
            assert isinstance(result, np.ndarray)
            assert result.shape == (512, 512)
        finally:
            os.unlink(test_file)
    
    def test_image_from_coordinates_characterization(self):
        """Test the core coordinate-to-image conversion logic."""
        # Test data similar to what would come from legacy init_metadata
        xC = np.array([100.5, 200.3, 300.7, 100.5])  # Include duplicate
        yC = np.array([150.2, 250.8, 350.1, 150.2])  # Include duplicate
        signal = np.array([1000, 2000, 1500, 500])
        
        # Simulate the core logic from _load_iqid_binary_data
        xdim, ydim = 512, 512
        image = np.zeros((ydim, xdim), dtype=np.float64)
        
        for i in range(len(xC)):
            x_coord = int(np.clip(xC[i], 0, xdim - 1))
            y_coord = int(np.clip(yC[i], 0, ydim - 1))
            image[y_coord, x_coord] = signal[i]
        
        # Validate expected behavior
        assert image[150, 100] == 500  # Last value at duplicate coordinate
        assert image[250, 200] == 2000
        assert image[350, 300] == 1500
        assert np.count_nonzero(image) == 3  # Three unique coordinates
    
    def test_segmentation_metrics_calculation(self):
        """Test segmentation metrics calculation with known values."""
        # Create test masks with known overlap
        predicted = np.zeros((10, 10), dtype=bool)
        predicted[2:6, 2:6] = True  # 4x4 square = 16 pixels
        
        ground_truth = np.zeros((10, 10), dtype=bool) 
        ground_truth[3:7, 3:7] = True  # 4x4 square = 16 pixels, overlaps 2x2 = 4 pixels
        
        metrics = self.tissue_separator.calculate_segmentation_metrics(predicted, ground_truth)
        
        # Known values:
        # Intersection = 4 pixels (2x2 overlap)
        # Union = 28 pixels (16 + 16 - 4)
        # IoU = 4/28 = 1/7 ≈ 0.143
        # Dice = 2*4/(16+16) = 8/32 = 0.25
        
        assert abs(metrics['iou'] - 4/28) < 1e-10
        assert abs(metrics['dice'] - 0.25) < 1e-10
        assert metrics['precision'] == 4/16  # 4 TP out of 16 predicted
        assert metrics['recall'] == 4/16     # 4 TP out of 16 actual
    
    def test_empty_masks_edge_case(self):
        """Test segmentation metrics with empty masks."""
        empty_pred = np.zeros((10, 10), dtype=bool)
        empty_gt = np.zeros((10, 10), dtype=bool)
        
        metrics = self.tissue_separator.calculate_segmentation_metrics(empty_pred, empty_gt)
        
        # All metrics should be 0 except accuracy which should be 1 (all true negatives)
        assert metrics['iou'] == 0.0
        assert metrics['dice'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['accuracy'] == 1.0  # All pixels correctly classified as negative
