"""Unit tests for the automate_processing.py script."""
import os
import numpy as np
import unittest
import shutil # For cleaning up test directories
from unittest.mock import patch, MagicMock
# Import ClusterData for type hinting, but it will be mocked in tests
from iqid.process_object import ClusterData
from automate_processing import (
    load_and_process_listmode_data,
    filter_correct_analyze_data,
    generate_spatial_images,
    generate_temporal_information,
    detect_contours_extract_ROIs,
    save_processed_data
)

class TestAutomateProcessing(unittest.TestCase):
    """Tests for the main workflow and helper functions in automate_processing.py."""

    @patch('automate_processing.ClusterData') # Mock ClusterData to avoid real file I/O
    def setUp(self, MockClusterData: MagicMock):
        """Sets up mock objects and configuration for each test.

        Mocks `iqid.process_object.ClusterData` to prevent actual file operations
        and isolate testing to the logic within `automate_processing.py` functions.
        """
        self.mock_cluster_data_instance = MockClusterData.return_value
        # Configure mock instance to return expected values from its methods
        self.mock_cluster_data_instance.init_header.return_value = (1024, 256, 256)
        self.mock_cluster_data_instance.load_cluster_data.return_value = np.random.rand(14, 1000)
        self.mock_cluster_data_instance.init_metadata.return_value = (
            np.random.rand(1000), np.random.rand(1000),
            np.random.rand(1000), np.random.rand(1000), np.arange(1000)
        )
        # Set up attributes on the mock object that would be set by ClusterData methods
        self.mock_cluster_data_instance.ftype = 'processed_lm'
        self.mock_cluster_data_instance.xC = np.random.rand(100)
        self.mock_cluster_data_instance.yC = np.random.rand(100)
        self.mock_cluster_data_instance.f = np.arange(100)
        self.mock_cluster_data_instance.time_ms = np.arange(100) * 10.0
        # Mock methods that return images or data structures
        self.mock_cluster_data_instance.image_from_listmode.return_value = np.random.rand(256, 256)
        self.mock_cluster_data_instance.image_from_big_listmode.return_value = np.random.rand(256, 256)
        self.mock_cluster_data_instance.get_contours.return_value = [np.array([[10,10],[10,20],[20,20],[20,10]])]
        self.mock_cluster_data_instance.get_ROIs.return_value = np.array([[10,10,10,10]])

        self.file_name = "dummy_listmode_data.dat" # Mock file path
        self.output_dir = "dummy_output_processed_data_test" # Use a distinct test output dir

        # Create dummy output directory for tests that might save files
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Note: The original setUp called load_and_process_listmode_data directly.
        # This is generally not ideal for setUp if that function itself is being tested.
        # For tests *other* than test_load_and_process_listmode_data, they will use
        # self.mock_cluster_data_instance which is already configured.
        # test_load_and_process_listmode_data will need its own @patch if it calls
        # the module-level ClusterData.

    @patch('automate_processing.ClusterData') # Mock ClusterData for this specific test
    def test_load_and_process_listmode_data(self, MockClusterDataInTest: MagicMock):
        """Tests the `load_and_process_listmode_data` function for successful execution and return types."""
        # Configure the mock specifically for this test's call
        mock_instance_for_test = MockClusterDataInTest.return_value
        mock_instance_for_test.init_header.return_value = (1024, 256, 256)
        mock_instance_for_test.load_cluster_data.return_value = np.random.rand(14, 1000)
        mock_instance_for_test.init_metadata.return_value = (
            np.random.rand(1000), np.random.rand(1000),
            np.random.rand(1000), np.random.rand(1000), np.arange(1000)
        )

        # Now call the function being tested
        cluster_data_obj, time_ms, cluster_area, xC_global, yC_global, frame_num = \
            load_and_process_listmode_data(self.file_name, c_area_thresh=10, makedir=True, ftype='offset_lm')

        # Assertions
        self.assertIsNotNone(cluster_data_obj)
        # Since ClusterData is mocked, cluster_data_obj is a MagicMock instance
        self.assertIsInstance(cluster_data_obj, MagicMock)
        MockClusterDataInTest.assert_called_once_with(self.file_name, 10, True, 'offset_lm')
        cluster_data_obj.init_header.assert_called_once()
        cluster_data_obj.load_cluster_data.assert_called_once()
        cluster_data_obj.init_metadata.assert_called_once()

        self.assertIsInstance(time_ms, np.ndarray)
        self.assertIsInstance(cluster_area, np.ndarray)
        self.assertIsNotNone(self.cluster_data)
        self.assertIsNotNone(self.time_ms)
        self.assertIsNotNone(self.cluster_area)
        self.assertIsNotNone(xC_global)
        self.assertIsNotNone(yC_global)
        self.assertIsNotNone(frame_num)

    def test_filter_correct_analyze_data(self):
        """Tests `filter_correct_analyze_data` for successful execution and method calls."""
        binfac = 1
        ROI_area_thresh = 100
        t_binsize = 1000.0 # Make float for consistency
        t_half = 3600.0 # Make float
        # self.mock_cluster_data_instance is used here as it's configured in setUp
        processed_cluster_data = filter_correct_analyze_data(self.mock_cluster_data_instance, binfac, ROI_area_thresh, t_binsize, t_half)
        self.assertIsNotNone(processed_cluster_data)
        # Check if methods on mock were called
        self.mock_cluster_data_instance.set_process_params.assert_called_once_with(binfac, ROI_area_thresh, t_binsize, t_half)
        self.mock_cluster_data_instance.get_mean_n.assert_called_once()
        self.mock_cluster_data_instance.estimate_missed_timestamps.assert_called_once()


    def test_generate_spatial_images(self):
        """Tests `generate_spatial_images` for successful execution and image return."""
        subpx = 1
        # self.mock_cluster_data_instance is used here
        cluster_image = generate_spatial_images(self.mock_cluster_data_instance, subpx)
        self.assertIsNotNone(cluster_image)
        self.mock_cluster_data_instance.image_from_listmode.assert_called_once_with(subpx)

    def test_generate_temporal_information(self):
        """Tests `generate_temporal_information` for successful execution and image return."""
        event_fx = 0.1
        xlim = (0, None)
        ylim = (0, None)
        # self.mock_cluster_data_instance is used here
        temporal_image = generate_temporal_information(self.mock_cluster_data_instance, event_fx, xlim, ylim)
        self.assertIsNotNone(temporal_image)
        self.mock_cluster_data_instance.image_from_big_listmode.assert_called_once_with(event_fx, xlim, ylim)

    def test_detect_contours_extract_ROIs(self):
        """Tests `detect_contours_extract_ROIs` for successful execution and return of contours/ROIs."""
        im = np.zeros((100, 100)) # Dummy image for testing
        gauss = 15
        thresh = 0
        # self.mock_cluster_data_instance is used here
        contours, ROIs = detect_contours_extract_ROIs(self.mock_cluster_data_instance, im, gauss, thresh)
        self.assertIsNotNone(contours)
        self.assertIsNotNone(ROIs)
        self.mock_cluster_data_instance.set_contour_params.assert_called_once_with(gauss, thresh)
        self.mock_cluster_data_instance.get_contours.assert_called_once_with(im)
        self.mock_cluster_data_instance.get_ROIs.assert_called_once()

    @patch('numpy.save') # Mock np.save to prevent actual file writing during this test
    def test_save_processed_data(self, mock_np_save: MagicMock):
        """Tests `save_processed_data` for attempting to save all relevant .npy files."""
        # self.mock_cluster_data_instance is used here
        save_processed_data(self.mock_cluster_data_instance, self.output_dir)

        # Check that np.save was called for expected files
        # These attributes are set on the mock_cluster_data_instance in setUp
        expected_calls = [
            unittest.mock.call(os.path.join(self.output_dir, 'xC.npy'), self.mock_cluster_data_instance.xC),
            unittest.mock.call(os.path.join(self.output_dir, 'yC.npy'), self.mock_cluster_data_instance.yC),
            unittest.mock.call(os.path.join(self.output_dir, 'f.npy'), self.mock_cluster_data_instance.f),
            unittest.mock.call(os.path.join(self.output_dir, 'time_ms.npy'), self.mock_cluster_data_instance.time_ms)
        ]
        # If ftype was 'clusters', add those calls too
        # To test this branch, we could have another test case or reconfigure self.mock_cluster_data_instance.ftype
        if self.mock_cluster_data_instance.ftype == 'clusters':
            # Add expected calls for 'clusters' ftype specific attributes
            # Ensure these attributes are also set on the mock object in setUp if testing this path
            self.mock_cluster_data_instance.raws = np.random.rand(10,10) # example
            self.mock_cluster_data_instance.cim_sum = np.random.rand(10) # example
            self.mock_cluster_data_instance.cim_px = np.random.rand(10) # example
            expected_calls.extend([
                unittest.mock.call(os.path.join(self.output_dir, 'raws.npy'), self.mock_cluster_data_instance.raws),
                unittest.mock.call(os.path.join(self.output_dir, 'cim_sum.npy'), self.mock_cluster_data_instance.cim_sum),
                unittest.mock.call(os.path.join(self.output_dir, 'cim_px.npy'), self.mock_cluster_data_instance.cim_px)
            ])

        mock_np_save.assert_has_calls(expected_calls, any_order=False)
        # Verify that os.makedirs was called if output_dir didn't exist.
        # This is harder to test without further mocking os.path.exists and os.makedirs.
        # For now, we assume it's called if needed.

    def tearDown(self):
        """Cleans up any artifacts created during tests, like dummy directories."""
        # Remove the dummy output directory created in setUp
        if os.path.exists(self.output_dir) and self.output_dir == "dummy_output_processed_data_test":
            shutil.rmtree(self.output_dir)


if __name__ == '__main__':
    unittest.main()
