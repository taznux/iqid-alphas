"""Unit tests for the automate_image_alignment.py script."""
import os
import numpy as np
import unittest
import shutil # For cleaning up test directories
from unittest.mock import patch, MagicMock
# Import functions to be mocked from iqid.align
from iqid.align import assemble_stack_hne, coarse_stack
from automate_image_alignment import align_and_register_images, save_registered_images

class TestAutomateImageAlignment(unittest.TestCase):
    """Tests for the image alignment automation script, focusing on workflow and file operations."""

    def setUp(self):
        """Sets up mock paths and parameters for image alignment tests."""
        self.image_dir = "dummy_image_dir_for_alignment"
        self.output_dir = "dummy_output_aligned_images"
        self.fformat = 'png' # Using png for easier mocking if needed, tif is also fine
        self.deg = 3.0
        self.avg_over = 2
        self.subpx = 1 # Parameter kept, though its direct use in mocked functions might not be asserted
        self.color = (0, 0, 0)
        self.convert_to_grayscale_for_ssd = True

        # Create dummy input/output directories for tests that might interact with filesystem
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @patch('automate_image_alignment.save_registered_images') # Mock the saving function
    @patch('iqid.align.coarse_stack') # Mock the core processing function
    @patch('iqid.align.assemble_stack_hne') # Mock the image loading function
    def test_align_and_register_images(self, mock_assemble_stack: MagicMock, mock_coarse_stack: MagicMock, mock_save_registered: MagicMock):
        """Tests the `align_and_register_images` main workflow function.

        Ensures that image assembly, coarse stacking, and saving functions are called.
        """
        # Configure mocks to return plausible values
        mock_assemble_stack.return_value = np.random.rand(5, 100, 100, 3) # Mock a stack of 5 color images
        mock_coarse_stack.return_value = np.random.rand(5, 100, 100, 3) # Mock an aligned stack

        align_and_register_images(
            self.image_dir, self.output_dir, self.fformat,
            self.deg, self.avg_over, self.subpx, self.color,
            self.convert_to_grayscale_for_ssd
        )

        mock_assemble_stack.assert_called_once_with(imdir=self.image_dir, fformat=self.fformat, color=self.color, pad=True)
        mock_coarse_stack.assert_called_once_with(
            mock_assemble_stack.return_value,
            deg=self.deg,
            avg_over=self.avg_over,
            convert_to_grayscale_for_ssd=self.convert_to_grayscale_for_ssd
        )
        mock_save_registered.assert_called_once_with(mock_coarse_stack.return_value, self.output_dir, self.fformat)

    @patch('skimage.io.imsave') # Mock skimage.io.imsave to prevent actual file writing
    @patch('os.makedirs') # Mock os.makedirs
    @patch('os.path.exists') # Mock os.path.exists
    def test_save_registered_images(self, mock_path_exists: MagicMock, mock_makedirs: MagicMock, mock_imsave: MagicMock):
        """Tests the `save_registered_images` function for correct file saving attempts."""
        mock_path_exists.return_value = False # Simulate directory does not exist initially
        image_stack = np.zeros((3, 100, 100, 3), dtype=np.uint8) # Mock 3 color images

        save_registered_images(image_stack, self.output_dir, self.fformat)

        mock_path_exists.assert_called_once_with(self.output_dir)
        mock_makedirs.assert_called_once_with(self.output_dir)

        expected_calls = [
            unittest.mock.call(os.path.join(self.output_dir, f'registered_image_0.{self.fformat}'), image_stack[0], check_contrast=False),
            unittest.mock.call(os.path.join(self.output_dir, f'registered_image_1.{self.fformat}'), image_stack[1], check_contrast=False),
            unittest.mock.call(os.path.join(self.output_dir, f'registered_image_2.{self.fformat}'), image_stack[2], check_contrast=False)
        ]
        mock_imsave.assert_has_calls(expected_calls)

    def tearDown(self):
        """Cleans up any created dummy directories after tests."""
        if os.path.exists(self.image_dir) and self.image_dir == "dummy_image_dir_for_alignment":
            shutil.rmtree(self.image_dir)
        if os.path.exists(self.output_dir) and self.output_dir == "dummy_output_aligned_images":
            shutil.rmtree(self.output_dir)

if __name__ == '__main__':
    unittest.main()
