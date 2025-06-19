"""
Characterization Tests for Migrated Utility Functions

These tests verify that migrated functions produce the same outputs as the original
legacy functions, ensuring behavioral parity during the migration process.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from iqid_alphas.utils.io_utils import (
    list_directories, 
    list_subdirectories,
    natural_sort_key,
    natural_sort
)
from iqid_alphas.utils.math_utils import (
    bin_ndarray,
    decompose_affine,
    percent_error,
    check_monotonic
)
from iqid_alphas.utils.transform_utils import (
    get_time_difference,
    create_affine_matrix,
    apply_affine_transform
)


class TestIOUtilities:
    """Test migrated I/O utilities."""
    
    def setup_method(self):
        """Create temporary directory structure for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test directory structure
        (self.temp_dir / "study1").mkdir()
        (self.temp_dir / "study2").mkdir()
        (self.temp_dir / "study1" / "substudy1").mkdir()
        (self.temp_dir / "study1" / "substudy2").mkdir()
        (self.temp_dir / "study2" / "substudy3").mkdir()
        
        # Create some test files
        (self.temp_dir / "study1" / "file1.txt").touch()
        (self.temp_dir / "study2" / "file2.txt").touch()
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_list_directories(self):
        """Test directory listing functionality."""
        dirs = list_directories(self.temp_dir)
        dir_names = [Path(d).name for d in dirs]
        assert "study1" in dir_names
        assert "study2" in dir_names
        assert len(dir_names) == 2
    
    def test_list_subdirectories(self):
        """Test subdirectory listing functionality."""
        subdirs = list_subdirectories(self.temp_dir)
        subdir_names = [Path(d).name for d in subdirs]
        assert "substudy1" in subdir_names
        assert "substudy2" in subdir_names
        assert "substudy3" in subdir_names
        assert len(subdir_names) == 3
    
    def test_natural_sort(self):
        """Test natural sorting functionality."""
        test_strings = ["file10.txt", "file2.txt", "file1.txt", "file20.txt"]
        sorted_strings = natural_sort(test_strings)
        expected = ["file1.txt", "file2.txt", "file10.txt", "file20.txt"]
        assert sorted_strings == expected
    
    def test_natural_sort_key(self):
        """Test natural sort key generation."""
        key1 = natural_sort_key("file1.txt")
        key2 = natural_sort_key("file10.txt")
        key3 = natural_sort_key("file2.txt")
        
        # Verify that sorting by keys produces natural order
        items = [("file10.txt", key2), ("file2.txt", key3), ("file1.txt", key1)]
        sorted_items = sorted(items, key=lambda x: x[1])
        sorted_names = [item[0] for item in sorted_items]
        
        assert sorted_names == ["file1.txt", "file2.txt", "file10.txt"]


class TestMathUtilities:
    """Test migrated mathematical utilities."""
    
    def test_bin_ndarray_sum(self):
        """Test array binning with sum operation."""
        # Create test array matching the example in the original function
        m = np.arange(0, 100, 1).reshape((10, 10))
        n = bin_ndarray(m, new_shape=(5, 5), operation='sum')
        
        # Expected result from original function documentation
        expected = np.array([
            [22, 30, 38, 46, 54],
            [102, 110, 118, 126, 134],
            [182, 190, 198, 206, 214],
            [262, 270, 278, 286, 294],
            [342, 350, 358, 366, 374]
        ])
        
        np.testing.assert_array_equal(n, expected)
    
    def test_bin_ndarray_mean(self):
        """Test array binning with mean operation."""
        m = np.arange(0, 100, 1).reshape((10, 10))
        n = bin_ndarray(m, new_shape=(5, 5), operation='mean')
        
        # For mean operation, divide the sum result by 4 (2x2 bins)
        expected_sum = np.array([
            [22, 30, 38, 46, 54],
            [102, 110, 118, 126, 134],
            [182, 190, 198, 206, 214],
            [262, 270, 278, 286, 294],
            [342, 350, 358, 366, 374]
        ])
        expected_mean = expected_sum / 4.0
        
        np.testing.assert_array_almost_equal(n, expected_mean)
    
    def test_decompose_affine(self):
        """Test affine matrix decomposition."""
        # Create a simple affine transformation
        original_matrix = np.array([
            [2, 0, 5],
            [0, 3, 10],
            [0, 0, 1]
        ])
        
        T, R, Z, S = decompose_affine(original_matrix)
        
        # For this simple case: pure scaling + translation
        expected_T = np.array([5, 10])  # Translation
        expected_Z = np.array([2, 3])   # Scaling
        
        np.testing.assert_array_almost_equal(T, expected_T)
        np.testing.assert_array_almost_equal(Z, expected_Z)
        
        # Rotation should be identity for this simple case
        np.testing.assert_array_almost_equal(R, np.eye(2))
    
    def test_percent_error(self):
        """Test percent error calculation."""
        ref = 100.0
        exp = 95.0
        error = percent_error(ref, exp)
        expected = 5.0  # (100 - 95) / 100 * 100
        assert error == expected
        
        # Test with arrays
        ref_array = np.array([100, 200, 300])
        exp_array = np.array([95, 190, 285])
        error_array = percent_error(ref_array, exp_array)
        expected_array = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(error_array, expected_array)
    
    def test_check_monotonic(self):
        """Test monotonic array checking."""
        # Monotonically increasing
        mono_array = np.array([1, 2, 3, 4, 5])
        assert check_monotonic(mono_array) == True
        
        # Non-monotonic
        non_mono_array = np.array([1, 3, 2, 4, 5])
        assert check_monotonic(non_mono_array) == False
        
        # Equal consecutive values (should pass non-strict test)
        equal_array = np.array([1, 2, 2, 3, 4])
        assert check_monotonic(equal_array, strict=False) == True
        assert check_monotonic(equal_array, strict=True) == False


class TestTransformUtilities:
    """Test migrated transformation utilities."""
    
    def test_get_time_difference(self):
        """Test time difference calculation."""
        time1 = datetime(2025, 1, 1, 12, 0, 0)
        time2 = datetime(2025, 1, 1, 12, 0, 1)  # 1 second later
        
        # Test seconds
        diff_s = get_time_difference(time1, time2, grain='s')
        assert diff_s == 1.0
        
        # Test microseconds (1 second = 1,000,000 microseconds)
        diff_us = get_time_difference(time1, time2, grain='us')
        assert diff_us == 1.0  # The function returns seconds with microsecond precision
    
    def test_create_affine_matrix(self):
        """Test affine matrix creation."""
        # Identity transformation
        identity = create_affine_matrix()
        expected_identity = np.eye(3)
        np.testing.assert_array_almost_equal(identity, expected_identity)
        
        # Pure translation
        translation = create_affine_matrix(translation=(5, 10))
        expected_translation = np.array([
            [1, 0, 5],
            [0, 1, 10],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(translation, expected_translation)
        
        # Pure scaling
        scaling = create_affine_matrix(scale=(2, 3))
        expected_scaling = np.array([
            [2, 0, 0],
            [0, 3, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(scaling, expected_scaling)
    
    def test_apply_affine_transform(self):
        """Test affine transformation application."""
        # Test points
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        
        # Pure translation
        translation_matrix = create_affine_matrix(translation=(5, 10))
        transformed_points = apply_affine_transform(points, translation_matrix)
        expected = points + np.array([5, 10])
        np.testing.assert_array_almost_equal(transformed_points, expected)
        
        # Pure scaling
        scaling_matrix = create_affine_matrix(scale=(2, 3))
        transformed_points = apply_affine_transform(points, scaling_matrix)
        expected = points * np.array([2, 3])
        np.testing.assert_array_almost_equal(transformed_points, expected)


class TestCharacterizationComparison:
    """Compare migrated functions with expected legacy behavior."""
    
    def test_legacy_helper_functions_behavior(self):
        """Test that migrated functions match expected legacy behavior."""
        
        # Test natural sorting matches expected behavior
        test_files = ["file1.txt", "file10.txt", "file2.txt", "file20.txt"]
        sorted_files = natural_sort(test_files)
        
        # This should match the behavior of the original natural_keys function
        assert sorted_files == ["file1.txt", "file2.txt", "file10.txt", "file20.txt"]
        
        # Test binning behavior matches original
        test_array = np.arange(16).reshape(4, 4)
        binned = bin_ndarray(test_array, (2, 2), 'sum')
        
        # Verify the binning produces expected sums
        expected_sums = np.array([[28, 44], [92, 108]])  # Hand-calculated
        np.testing.assert_array_equal(binned, expected_sums)


if __name__ == "__main__":
    pytest.main([__file__])
