import os
import json
import logging
import argparse
import numpy as np
from skimage import io
from iqid.align import assemble_stack, assemble_stack_hne, coarse_stack, pad_stack_he, crop_down

"""
Automates the alignment and registration of image stacks, particularly for
H&E stained histology images or similar series of 2D images.

This script is configured via `config.json` for most parameters, but key
input/output directories (`image_dir`, `output_dir`) are passed as
command-line arguments.

Workflow:
1.  Assembles a stack of color images (e.g., H&E) from the specified `image_dir`
    using `iqid.align.assemble_stack_hne`. This function also handles padding
    of images to consistent dimensions.
2.  Performs coarse alignment of the image stack using `iqid.align.coarse_stack`.
    This typically involves sequential rotation alignment minimizing SSD.
    Grayscale conversion can be used for SSD calculation if specified.
3.  Saves the aligned/registered image stack to the `output_dir`.

Key inputs:
- `image_dir` (command-line argument): Directory containing raw image slices.
- `output_dir` (command-line argument): Directory to save aligned images.
- `config.json` (for parameters like `fformat`, `deg`, `avg_over`, `color`).

Key outputs:
- A series of registered image files (e.g., `registered_image_0.tif`, ...)
  saved in the `output_dir`.

Logging of operations is performed to `automate_image_alignment.log`.
"""
import os
import json
import logging
import argparse
import numpy as np
from skimage import io
# Assuming crop_down and pad_stack_he are not used based on comments.
# If they were to be used, they would be imported from iqid.align.
from iqid.align import assemble_stack_hne, coarse_stack # Removed pad_stack_he, crop_down

# Configure logging
logging.basicConfig(filename='automate_image_alignment.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def align_and_register_images(image_dir: str, output_dir: str, fformat: str = 'tif', deg: float = 2, avg_over: int = 1, subpx: int = 1, color: tuple[int,int,int] = (0, 0, 0), convert_to_grayscale_for_ssd: bool = True) -> None:
    """
    Assembles, aligns, and saves a stack of images.

    Parameters
    ----------
    image_dir : str
        Directory containing the raw image files.
    output_dir : str
        Directory where the aligned images will be saved.
    fformat : str, optional
        File format (extension) of the images. Defaults to 'tif'.
    deg : float, optional
        Angular increment for coarse rotation in `coarse_stack`. Defaults to 2.
    avg_over : int, optional
        Number of previous images to average for reference in `coarse_stack`.
        Defaults to 1.
    subpx : int, optional
        Subpixel factor, currently noted as kept for signature but not explicitly
        used by `assemble_stack_hne` or `coarse_stack` in this context.
        Defaults to 1.
    color : tuple[int, int, int], optional
        RGB color tuple for padding in `assemble_stack_hne`. Defaults to (0,0,0).
    convert_to_grayscale_for_ssd : bool, optional
        If True, images are converted to grayscale before SSD calculation in
        `coarse_stack`. Defaults to True.

    Returns
    -------
    None
        Aligned images are saved to disk.

    Raises
    ------
    Exception
        Propagates exceptions from underlying image processing or file operations.
    """
    try:
        logging.info(f"Assembling H&E stack from: {image_dir}")
        # `assemble_stack_hne` handles padding internally if images are of different sizes.
        image_stack = assemble_stack_hne(imdir=image_dir, fformat=fformat, color=color, pad=True)

        if image_stack is None or len(image_stack) == 0:
            logging.warning(f"No images found or assembled from {image_dir}. Skipping alignment.")
            return
        
        logging.info(f"Coarsely aligning stack with {len(image_stack)} images.")
        # coarse_stack performs the alignment.
        aligned_stack = coarse_stack(image_stack, deg=deg, avg_over=avg_over, convert_to_grayscale_for_ssd=convert_to_grayscale_for_ssd)
        
        # Removed commented-out lines for pad_stack_he and crop_down as per instructions
        
        logging.info(f"Saving registered stack to: {output_dir}")
        save_registered_images(aligned_stack, output_dir, fformat)
    except Exception as e:
        logging.error(f"Failed to align and register images for {image_dir}: {str(e)}", exc_info=True)
        raise

def save_registered_images(image_stack: np.ndarray, output_dir: str, fformat: str = 'tif') -> None:
    """Saves each image in the provided stack to the output directory.

    Filenames are `registered_image_0.ext`, `registered_image_1.ext`, etc.

    Parameters
    ----------
    image_stack : np.ndarray
        The stack of images (num_images, height, width, [channels]) to save.
    output_dir : str
        Directory to save the registered images. Will be created if it doesn't exist.
    fformat : str, optional
        File format (extension) for the saved images. Defaults to 'tif'.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Propagates exceptions from file I/O operations.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logging.info(f"Saving {len(image_stack)} registered images to {output_dir}.")
        for i, image in enumerate(image_stack):
            # skimage.io.imsave handles various dtypes appropriately.
            # For TIF, it can save float data; for PNG, it often converts to uint8.
            save_path = os.path.join(output_dir, f'registered_image_{i}.{fformat}')
            io.imsave(save_path, image, check_contrast=False) # Add check_contrast=False for wider data ranges
        logging.info("Successfully saved registered images.")
    except Exception as e:
        logging.error(f"Failed to save registered images: {e}", exc_info=True)
        raise

def main(image_dir: str, output_dir: str) -> None:
    """
    Main function to automate image alignment.

    Workflow:
    1. Loads configuration parameters from `config.json` ("automate_image_alignment" section).
    2. Calls `align_and_register_images` with `image_dir` and `output_dir` from
       command-line arguments, and other parameters from the config file.

    Parameters
    ----------
    image_dir : str
        Directory containing images to align. Passed from command-line argument.
    output_dir : str
        Directory to save registered images. Passed from command-line argument.

    Returns
    -------
    None

    Side Effects
    ------------
    Creates output directory and saves aligned images. Logs operations.
    """
    logging.info(f"Starting image alignment for image_dir: {image_dir}, output_dir: {output_dir}")
    try:
        with open('config.json', 'r') as f:
            config_params = json.load(f)['automate_image_alignment']

        # image_dir and output_dir are from command-line args
        fformat = config_params.get('fformat', 'tif')
        # 'pad' parameter for assemble_stack_hne is True by default in align_and_register_images.
        # The config 'pad' might be for a different context if it was intended for the removed pad_stack_he.
        deg = config_params.get('deg', 2.0) # Ensure float if degrees can be non-integer
        avg_over = config_params.get('avg_over', 1)
        subpx = config_params.get('subpx', 1) # Noted as potentially unused in called functions
        color_list = config_params.get('color', [0, 0, 0])
        color = tuple(color_list) if isinstance(color_list, list) else (0,0,0) # Ensure tuple

        # convert_to_grayscale_for_ssd is hardcoded as True in the call below,
        # but could be made configurable.
        convert_to_grayscale_for_ssd = config_params.get('convert_to_grayscale_for_ssd', True)


        align_and_register_images(image_dir, output_dir, fformat, deg, avg_over, subpx, color, convert_to_grayscale_for_ssd=convert_to_grayscale_for_ssd)
        logging.info(f"Successfully completed image alignment for {image_dir}.")
    except Exception as e:
        logging.error(f"Failed to complete main image alignment for {image_dir}: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and register image stacks.")
    parser.add_argument("image_dir", help="Directory containing images to align.")
    parser.add_argument("output_dir", help="Directory to save registered images.")
    # You can add other arguments here later if needed, e.g., for config file path
    # parser.add_argument("--config", default="config.json", help="Path to the configuration file.")
    args = parser.parse_args()

    # Call main with the parsed command-line arguments for image_dir and output_dir
    # Other parameters will still be loaded from config.json within main() for now
    main(args.image_dir, args.output_dir)
