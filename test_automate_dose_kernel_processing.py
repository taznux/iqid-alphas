"""Unit tests for the automate_dose_kernel_processing.py script."""
import os
import numpy as np
import unittest
import shutil # For cleaning up test directories
from unittest.mock import patch, MagicMock
# Import functions to be mocked from iqid.dpk
from iqid.dpk import load_txt_kernel, mev_to_mgy, radial_avg_kernel, pad_kernel_to_vsize
from automate_dose_kernel_processing import process_dose_kernels, save_processed_kernels

class TestAutomateDoseKernelProcessing(unittest.TestCase):
    """Tests for the dose kernel processing automation script."""

    def setUp(self):
        """Sets up mock paths and parameters for dose kernel processing tests."""
        self.kernel_file = "dummy_kernel.txt" # Mock file path
        self.dim = 129 # Odd dimension for easier center finding in mocks if needed
        self.num_alpha_decays = 1e6
        self.vox_vol_m = 1e-9
        self.dens_kgm = 1000.0 # Use float
        self.vox_xy = 1
        self.slice_z = 12
        self.output_dir = "dummy_output_processed_kernels"

        # Create dummy output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Create a dummy kernel file for load_txt_kernel tests if not fully mocking it
        # For process_dose_kernels, load_txt_kernel will be mocked.
        # For test_load_txt_kernel, we might want a real dummy file or mock genfromtxt.
        # For now, assume process_dose_kernels will have all its dependencies mocked.
        # For the standalone dpk function tests, they will call the actual iqid.dpk functions.
        if not os.path.exists(self.kernel_file) and self.kernel_file == "dummy_kernel.txt":
            # Create a minimal dummy kernel file that load_txt_kernel can parse
            header = "Header Line 1\nHeader Line 2\nHeader Line 3\nHeader Line 4\nHeader Line 5\n"
            # Create dim*dim*dim data points (x, y, z, E)
            # For simplicity, let's make a small dummy file if dim is large
            # This dummy file is only for the direct test of load_txt_kernel.
            # For process_dose_kernels, load_txt_kernel itself will be mocked.
            data_lines = []
            actual_dim_for_file = 3 # Use a small dim for the dummy file to keep it small
            if self.dim == actual_dim_for_file: # Only create if dim matches this test file's intent
                for i in range(actual_dim_for_file):
                    for j in range(actual_dim_for_file):
                        for k in range(actual_dim_for_file):
                            data_lines.append(f"{i} {j} {k} {np.random.rand()}")
            with open(self.kernel_file, 'w') as f:
                f.write(header)
                f.write("\n".join(data_lines))


    @patch('automate_dose_kernel_processing.save_processed_kernels')
    @patch('iqid.dpk.pad_kernel_to_vsize')
    @patch('iqid.dpk.radial_avg_kernel')
    @patch('iqid.dpk.mev_to_mgy')
    @patch('iqid.dpk.load_txt_kernel')
    def test_process_dose_kernels(self, mock_load_txt: MagicMock, mock_mev_to_mgy: MagicMock,
                                  mock_radial_avg: MagicMock, mock_pad_kernel: MagicMock,
                                  mock_save_kernels: MagicMock):
        """Tests the main `process_dose_kernels` workflow, ensuring all sub-functions are called."""
        mock_load_txt.return_value = np.random.rand(self.dim, self.dim, self.dim)
        mock_mev_to_mgy.return_value = np.random.rand(self.dim, self.dim, self.dim)
        mock_radial_avg.return_value = np.random.rand(self.dim, self.dim, self.dim)
        mock_pad_kernel.return_value = np.random.rand(132, 132, 132) # Example padded shape

        process_dose_kernels(
            self.kernel_file, self.dim, self.num_alpha_decays, self.vox_vol_m,
            self.dens_kgm, self.vox_xy, self.slice_z, self.output_dir
        )

        mock_load_txt.assert_called_once_with(self.kernel_file, self.dim, self.num_alpha_decays)
        mock_mev_to_mgy.assert_called_once_with(mock_load_txt.return_value, self.vox_vol_m, self.dens_kgm)
        mock_radial_avg.assert_called_once_with(mock_mev_to_mgy.return_value, mode="whole", bin_size=0.5)
        mock_pad_kernel.assert_called_once_with(mock_radial_avg.return_value, self.vox_xy, self.slice_z)
        mock_save_kernels.assert_called_once_with(mock_pad_kernel.return_value, self.output_dir)

    @patch('numpy.save')
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_save_processed_kernels(self, mock_path_exists: MagicMock, mock_makedirs: MagicMock, mock_np_save: MagicMock):
        """Tests the `save_processed_kernels` function for correct file saving attempt."""
        mock_path_exists.return_value = False # Simulate directory doesn't exist
        kernel_data = np.zeros((128, 128, 128))

        save_processed_kernels(kernel_data, self.output_dir)

        mock_path_exists.assert_called_once_with(self.output_dir)
        mock_makedirs.assert_called_once_with(self.output_dir)
        mock_np_save.assert_called_once_with(os.path.join(self.output_dir, 'processed_kernel.npy'), kernel_data)

    # The following tests are for iqid.dpk functions, not directly for automate_dose_kernel_processing.
    # They are kept here as they were in the original file, but ideally would be in a separate test_dpk.py
    def test_load_txt_kernel(self):
        """Tests the `load_txt_kernel` function from iqid.dpk."""
        # This test requires a real dummy file or mocking np.genfromtxt
        # Using the dummy file created in setUp if dim matches
        if self.dim == 3 and os.path.exists(self.kernel_file): # Ensure it's the small dummy file test
            kernel = load_txt_kernel(self.kernel_file, 3, self.num_alpha_decays)
            self.assertIsNotNone(kernel)
            self.assertEqual(kernel.shape, (3, 3, 3))
        elif self.dim !=3: # Skip if dim is not for the dummy file to avoid large file creation
            self.skipTest(f"Skipping test_load_txt_kernel for dim={self.dim} as dummy file is for dim=3.")
        else:
            self.skipTest("Dummy kernel file not found for test_load_txt_kernel.")


    def test_mev_to_mgy(self):
        """Tests the `mev_to_mgy` function from iqid.dpk."""
        kernel_mev = np.ones((10, 10, 10)) # 1 MeV per voxel
        vox_vol_m = 1e-9 # 1 mm^3 voxel (1e-3 m)^3
        dens_kgm = 1000 # kg/m^3 (water)
        # Expected: 1 MeV * 1.602e-13 J/MeV / (1000 kg/m^3 * 1e-9 m^3/voxel) = 1.602e-13 J/kg / 1e-6 kg = 1.602e-7 Gy/voxel
        # mGy = Gy * 1000 = 1.602e-4 mGy/voxel
        mgy_kernel = mev_to_mgy(kernel_mev, vox_vol_m, dens_kgm)
        self.assertIsNotNone(mgy_kernel)
        self.assertAlmostEqual(mgy_kernel[0,0,0], 1.602e-13 / (1e-6) * 1e3, places=5)


    def test_radial_avg_kernel(self):
        """Tests the `radial_avg_kernel` function from iqid.dpk."""
        # Simple kernel: center voxel high, others low
        kernel = np.zeros((5, 5, 5))
        kernel[2, 2, 2] = 100
        avg_kernel_whole = radial_avg_kernel(kernel, mode="whole", bin_size=0.5)
        self.assertIsNotNone(avg_kernel_whole)
        self.assertEqual(avg_kernel_whole.shape, kernel.shape)
        avg_kernel_segment = radial_avg_kernel(kernel, mode="segment", bin_size=0.5)
        self.assertIsNotNone(avg_kernel_segment)
        self.assertTrue(len(avg_kernel_segment) > 0)


    def test_pad_kernel_to_vsize(self):
        """Tests the `pad_kernel_to_vsize` function from iqid.dpk."""
        kernel = np.zeros((128, 128, 128)) # Example current size
        vox_xy_target = 10
        slice_z_target = 10
        padded_kernel = pad_kernel_to_vsize(kernel, vox_xy_target, slice_z_target)
        self.assertIsNotNone(padded_kernel)
        self.assertEqual(padded_kernel.shape[0] % slice_z_target, 0)
        self.assertEqual(padded_kernel.shape[1] % vox_xy_target, 0)
        self.assertEqual(padded_kernel.shape[2] % vox_xy_target, 0)

    def tearDown(self):
        """Cleans up any created dummy files or directories."""
        if os.path.exists(self.output_dir) and self.output_dir == "dummy_output_processed_kernels":
            shutil.rmtree(self.output_dir)
        if os.path.exists(self.kernel_file) and self.kernel_file == "dummy_kernel.txt":
            os.remove(self.kernel_file)

if __name__ == '__main__':
    unittest.main()
