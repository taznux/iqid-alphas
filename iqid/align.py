import os
import glob
import numpy as np
from skimage import transform, io, exposure, filters, draw, util
import cv2
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import warnings
import imagesize


from iqid import helper


"""
This module provides functions for alignment and registration of iQID activity
image stacks with corresponding histology (e.g., H&E) images.

It includes utilities for:
- Assembling image stacks from sequences of image files.
- Padding images to consistent dimensions.
- Performing coarse alignment using rotation and Sum of Squared Differences (SSD) minimization.
- Cropping and centering images.
- Various image transformation, type conversion, and visualization helper functions.
- Re-binning images and centroid data between different resolutions (e.g., H&E to iQID).

The module relies on libraries such as NumPy, scikit-image, OpenCV, and Matplotlib.
"""


def get_maxdim(fileList: list[str]) -> tuple[int, int]:
    """Gets the maximum image height and width from a list of image files
    without reading the entire image data into memory.

    Uses the `imagesize` package. (https://pypi.org/project/imagesize/)

    Parameters
    ----------
    fileList : list[str]
        List of file paths for the images.

    Returns
    -------
    tuple[int, int]
        (maxh, maxw) - Maximum height and maximum width found among the images.

    Notes
    -----
    A TODO in the original code suggests incorporating this into `assemble_stack` functions.
    """
    temp = np.zeros((len(fileList), 2))
    for i in range(len(fileList)):
        w, h = imagesize.get(fileList[i])
        temp[i, :] = h, w

    maxh = int(np.max(temp[:, 0]))
    maxw = int(np.max(temp[:, 1]))
    return (maxh, maxw)


def assemble_stack(imdir: str | None = None, fformat: str = 'tif', pad: bool = False) -> np.ndarray:
    """Loads a sequence of images (grayscale or single-channel) from a directory,
    sorts them using natural sort order, and assembles them into a NumPy stack.

    For proper alphabetization (natural sort), files should be named with
    numerical suffixes, e.g.:
    img_01.tif, img_02.tif, ..., img_14.tif.

    Parameters
    ----------
    imdir : str | None, optional
        The directory containing the image files. If None, uses the current
        directory. Defaults to None.
    fformat : str, optional
        File extension of the images (e.g., 'tif', 'png'). Defaults to 'tif'.
    pad : bool, optional
        If True, images are padded with zeros to match the dimensions of the
        largest image in the stack. Padding is applied symmetrically.
        Defaults to False. If False, and images are not the same size,
        `skimage.io.ImageCollection` might behave unexpectedly or raise an error
        when converted to a NumPy array if images are not of identical dimensions.

    Returns
    -------
    np.ndarray
        A 3D NumPy array representing the image stack (num_images, height, width).

    Dependencies
    ------------
    - `glob`
    - `os`
    - `numpy`
    - `skimage.io`
    - `cv2` (if `pad` is True)
    - `iqid.helper.natural_keys`
    """
    if imdir is None:
        imdir = '.'
    data_path = os.path.join('.', imdir) # Note: os.path.join behaviour with '.' if imdir is absolute.
    fileList = glob.glob(os.path.join(data_path, '*.' + fformat))
    fileList.sort(key=helper.natural_keys)

    if not fileList:
        warnings.warn(f"No files found in directory {data_path} with format {fformat}. Returning empty array.")
        return np.array([])

    if pad:
        # Determine maximum dimensions for padding
        max_h, max_w = 0, 0
        for f_path in fileList:
            try:
                h_f, w_f = imagesize.get(f_path) # Use imagesize to avoid loading full images
                max_h = max(max_h, h_f)
                max_w = max(max_w, w_f)
            except Exception as e:
                warnings.warn(f"Could not get size of {f_path} using imagesize: {e}. Reading image.")
                try:
                    img_temp = io.imread(f_path)
                    max_h = max(max_h, img_temp.shape[0])
                    max_w = max(max_w, img_temp.shape[1])
                except Exception as e_read:
                     warnings.warn(f"Could not read {f_path} to determine size: {e_read}. Skipping this file for max dim calc.")
                     continue

        # Pre-allocate stack
        # Infer dtype from first image, assuming homogeneity for grayscale
        try:
            first_image_dtype = io.imread(fileList[0]).dtype
        except Exception as e:
            warnings.warn(f"Could not read first image {fileList[0]} to determine dtype: {e}. Defaulting to float32.")
            first_image_dtype = np.float32

        imcollection = np.zeros((len(fileList), max_h, max_w), dtype=first_image_dtype)

        for i, f_path in enumerate(fileList):
            try:
                im = io.imread(f_path)
                h, w = im.shape[:2] # Works for 2D and 3D (if color, but this fn is for grayscale)

                pad_x_total = max_w - w
                pad_y_total = max_h - h

                pad_top = pad_y_total // 2
                pad_bottom = pad_y_total - pad_top
                pad_left = pad_x_total // 2
                pad_right = pad_x_total - pad_left

                # cv2.copyMakeBorder expects specific padding order: top, bottom, left, right
                im_padded = cv2.copyMakeBorder(im, pad_top, pad_bottom, pad_left, pad_right,
                                               cv2.BORDER_CONSTANT, value=0) # value=(0,0,0) for color
                imcollection[i, :, :] = im_padded
            except Exception as e:
                warnings.warn(f"Could not process or pad image {f_path}: {e}. Leaving as zeros in stack.")
                # Optionally, fill with np.nan or handle differently
    else:
        try:
            imcollection = io.ImageCollection(fileList)
            imcollection = np.array(imcollection) # This may fail if images are not same size
        except Exception as e:
            warnings.warn(f"Failed to create stack with skimage.io.ImageCollection (images might differ in size and pad=False): {e}")
            # Attempt to load images individually and stack if they happen to be same size, or raise error
            try:
                imgs = [io.imread(f) for f in fileList]
                imcollection = np.stack(imgs)
            except ValueError as ve:
                 raise ValueError(f"Images in {imdir} are not the same size and pad=False. Original error: {ve}") from ve


    return imcollection


def assemble_stack_hne(imdir: str | None = None, fformat: str = 'tif', color: tuple[int, int, int] = (0, 0, 0), pad: bool = True) -> np.ndarray:
    """Assembles a stack of H&E (color) images from a directory, sorting them
    naturally. Optionally pads images to match the largest dimensions using a
    specified color.

    Parameters
    ----------
    imdir : str | None, optional
        Directory containing the image files. If None, uses current directory.
        Defaults to None.
    fformat : str, optional
        File extension of the images (e.g., 'tif', 'png'). Defaults to 'tif'.
    color : tuple[int, int, int], optional
        RGB tuple representing the color to use for padding if `pad` is True.
        Defaults to (0, 0, 0) (black).
    pad : bool, optional
        If True (default), images are padded to the maximum dimensions found in
        the stack. Padding is applied by centering the image within the new
        dimensions and filling the rest with `color`.

    Returns
    -------
    np.ndarray
        A 4D NumPy array (num_images, height, width, channels) of type `int`
        (values typically 0-255 for uint8 images).

    Dependencies
    ------------
    - `glob`
    - `os`
    - `numpy`
    - `skimage.io`
    - `imagesize` (for efficient dimension checking if `pad` is True)
    - `iqid.helper.natural_keys`
    """
    if imdir is None:
        imdir = '.'
    data_path = os.path.join('.', imdir)
    fileList = glob.glob(os.path.join(data_path, '*.' + fformat))
    fileList.sort(key=helper.natural_keys)

    if not fileList:
        warnings.warn(f"No files found in directory {data_path} with format {fformat}. Returning empty array.")
        return np.array([])

    if pad:
        max_h, max_w = 0, 0
        # Determine maximum dimensions and number of channels from first image
        try:
            first_im_props = io.imread(fileList[0])
            CHA = first_im_props.shape[2] if len(first_im_props.shape) == 3 else 1
            if CHA == 1: # If accidentally a grayscale image
                warnings.warn(f"First image {fileList[0]} is grayscale, but H&E stack expects color. Will proceed assuming 3 channels for padding.")
                CHA = 3 # Default to 3 channels for color padding
        except Exception as e:
            warnings.warn(f"Could not read first image {fileList[0]} to determine channels: {e}. Assuming 3 channels.")
            CHA = 3

        for f_path in fileList:
            try:
                h_f, w_f = imagesize.get(f_path)
                max_h = max(max_h, h_f)
                max_w = max(max_w, w_f)
            except Exception as e:
                warnings.warn(f"Could not get size of {f_path} using imagesize: {e}. Reading image.")
                try:
                    img_temp = io.imread(f_path)
                    max_h = max(max_h, img_temp.shape[0])
                    max_w = max(max_w, img_temp.shape[1])
                    # Update CHA if subsequent images have different channel numbers (should be consistent for H&E)
                    if len(img_temp.shape) == 3 and img_temp.shape[2] != CHA and CHA ==3: # Only warn if we assumed 3 and find different
                         warnings.warn(f"Image {f_path} has {img_temp.shape[2]} channels, inconsistent with first image's {CHA}. Using {CHA}.")
                except Exception as e_read:
                    warnings.warn(f"Could not read {f_path} to determine size: {e_read}. Skipping for max dim calc.")
                    continue


        imcollection = np.zeros((len(fileList), max_h, max_w, CHA), dtype=np.uint8) # Usually H&E are uint8

        for i, f_path in enumerate(fileList):
            try:
                im = io.imread(f_path)
                if len(im.shape) == 2: # Handle grayscale image found in stack
                    warnings.warn(f"Image {f_path} is grayscale. Converting to color by repeating channels.")
                    im = np.stack((im,)*CHA, axis=-1)
                elif im.shape[2] != CHA: # Handle inconsistent channels
                     warnings.warn(f"Image {f_path} has {im.shape[2]} channels, attempting to conform to {CHA} channels for stack.")
                     if CHA == 3 and im.shape[2] == 4: # RGBA to RGB
                         im = im[:,:,:3]
                     elif CHA == 1 and im.shape[2] >=3: # Color to Grayscale (take first channel or average)
                         im = im[:,:,0] # Example: take first channel
                     # Add more sophisticated channel conversion if needed, or make it an error
                     # For now, attempt to make it fit or it will error in assignment

                h, w = im.shape[:2]
                result = np.full((max_h, max_w, CHA), color, dtype=np.uint8)

                x_center = (max_w - w) // 2
                y_center = (max_h - h) // 2
                result[y_center:y_center+h, x_center:x_center+w, :] = im
                imcollection[i, :, :, :] = result # Corrected slicing
            except Exception as e:
                warnings.warn(f"Could not process or pad image {f_path}: {e}. Leaving as zeros in stack.")

    else:
        try:
            imcollection = io.ImageCollection(fileList)
            imcollection = np.array(imcollection) # May fail if images not same size/channels
        except Exception as e:
            warnings.warn(f"Failed to create stack with skimage.io.ImageCollection (images might differ in size/channels and pad=False): {e}")
            try:
                imgs = [io.imread(f) for f in fileList]
                # Basic check for consistent shapes before stacking
                first_shape = imgs[0].shape if imgs else None
                if first_shape and all(img.shape == first_shape for img in imgs):
                    imcollection = np.stack(imgs)
                else:
                    raise ValueError(f"Images in {imdir} are not the same size/channels and pad=False.")
            except ValueError as ve:
                raise ValueError(f"Images in {imdir} are not the same size/channels and pad=False. Original error: {ve}") from ve


    return imcollection.astype(np.uint8) # Ensure uint8 for typical image data


def pad_stack_he(data_path: str, fformat: str = 'png', color: tuple[int, int, int] | int = (0, 0, 0), savedir: str | None = None, verbose: bool = False) -> None:
    """Pads a stack of images (grayscale or color) from a directory to the
    maximum dimensions found within the set and saves them to a new directory.

    Each image is centered within the new dimensions and padded with the
    specified `color`.

    Parameters
    ----------
    data_path : str
        Directory containing the image files to be padded.
    fformat : str, optional
        File extension of the images (e.g., 'png', 'tif'). Defaults to 'png'.
    color : tuple[int, int, int] | int, optional
        Color to use for padding.
        If images are RGB (3 channels), this should be an RGB tuple like (R, G, B).
        If images are grayscale, the first element of the tuple `color[0]` or an
        integer value will be used. Defaults to (0, 0, 0) (black).
    savedir : str | None, optional
        The base directory where the padded images will be saved. A subdirectory
        named 'padded' will be created within `savedir` to store the images.
        If None, it might behave unexpectedly or use current directory depending
        on `os.path.join` behavior with `None`. It's recommended to specify this.
        Defaults to None.
    verbose : bool, optional
        If True, prints the list of files being processed. Defaults to False.

    Returns
    -------
    None
        This function saves the padded images to disk and does not return any value.

    Raises
    ------
    FileNotFoundError
        If `data_path` does not exist or no files matching `fformat` are found.
    TypeError
        If an image has an unexpected number of dimensions (not 2 or 3).

    Dependencies
    ------------
    - `glob`
    - `os`
    - `numpy`
    - `skimage.io`
    - `pathlib.Path`
    - `tqdm.trange`
    - `iqid.align.get_maxdim`
    - `iqid.helper.natural_keys`
    """
    fileList = glob.glob(os.path.join(data_path, '*.' + fformat))
    if not fileList:
        raise FileNotFoundError(f"No files found in {data_path} with format {fformat}")
    fileList.sort(key=helper.natural_keys)

    if verbose:
        print("Files to be padded:")
        print(*fileList, sep='\n')

    if savedir is None:
        # Default to creating 'padded' in the parent of data_path or current dir
        # This behavior should be well-defined. Let's make it parent of data_path.
        parent_dir = Path(data_path).parent
        savedir = parent_dir
        warnings.warn(f"No savedir provided. Padded images will be saved in a 'padded' subdirectory under {parent_dir}.")

    newdir = os.path.join(savedir, 'padded')
    Path(newdir).mkdir(parents=True, exist_ok=True)

    # print('Detecting maximum dimensions...') # Original print statement
    maxh, maxw = get_maxdim(fileList)

    for i in trange(len(fileList), desc='Padding images'):
        try:
            im = io.imread(fileList[i])
            im_name_base = Path(fileList[i]).stem # More robust way to get filename without ext
            # imname = os.path.basename(os.path.splitext(fileList[i])[0]) # Original

            s = im.shape
            if len(s) == 3: # Color image (H, W, C)
                h, w, CHA = im.shape
                # Ensure color is a tuple of 3 for RGB, even if CHA is 4 (ignore alpha for padding color)
                pad_color_value = color if isinstance(color, tuple) and len(color) == CHA else (color[0],color[0],color[0]) if isinstance(color,tuple) else (color,color,color)
                if CHA == 4 and isinstance(color, tuple) and len(color)==3: # if image is RGBA and color is RGB, add full opacity
                    pad_color_value = (*pad_color_value, 255)

                result = np.full((maxh, maxw, CHA), pad_color_value[:CHA], dtype=im.dtype) # Use image's dtype
            elif len(s) == 2: # Grayscale image (H, W)
                h, w = im.shape
                pad_color_value = color[0] if isinstance(color, tuple) else color
                result = np.full((maxh, maxw), pad_color_value, dtype=im.dtype) # Use image's dtype
            else:
                warnings.warn(f"Image {fileList[i]} has unsupported shape {s}. Skipping.")
                continue # Skip this image

            pad_w_total = maxw - w
            pad_h_total = maxh - h

            y_center_offset = pad_h_total // 2
            x_center_offset = pad_w_total // 2

            if len(s) == 3:
                result[y_center_offset:y_center_offset+h, x_center_offset:x_center_offset+w, :] = im
            elif len(s) == 2:
                result[y_center_offset:y_center_offset+h, x_center_offset:x_center_offset+w] = im

            save_path = os.path.join(newdir, f'{im_name_base}_pad.{fformat}')
            if fformat.lower() in ['tif', 'tiff']:
                # Ensure float32 for tif if original was, or if it's standard for Slicer.
                # However, preserving original dtype is often better unless conversion is required.
                # Let's assume Slicer might prefer float32 for certain analytical TIFs.
                # The original code cast to float32 unconditionally for tif.
                io.imsave(save_path, result.astype(np.float32) if im.dtype != np.float32 else result, plugin='tifffile', check_contrast=False)
            else:
                # For PNGs etc., save in original or uint8 if appropriate
                io.imsave(save_path, result if result.dtype == np.uint8 else util.img_as_ubyte(result), check_contrast=False)
        except Exception as e:
            warnings.warn(f"Failed to process or save image {fileList[i]}: {e}")


def organize_onedir(imdir: str | None = None, include_idx: list[bool] | np.ndarray = [], order_idx: list[int] | np.ndarray = [], fformat: str = 'png') -> None:
    """Organizes images within a single directory by optionally deleting some
    images and renaming others based on provided indices and naming conventions.

    Prompts the user for confirmation before performing deletions or renames.
    Uses a predefined dictionary (`nameDict`) to determine the base name for
    renamed files based on the parent directory's name.

    Parameters
    ----------
    imdir : str | None, optional
        The directory containing images to organize. If None, behavior might be
        unpredictable; it's recommended to provide a path. Defaults to None.
    include_idx : list[bool] | np.ndarray, optional
        A boolean list or array indicating which files to keep (True) or
        discard (False). Length must match the number of image files found.
        If empty, all files are initially included. Defaults to an empty list.
    order_idx : list[int] | np.ndarray, optional
        An array specifying the new numerical suffix for renaming the included files.
        Length must match the number of files to be included (sum of `include_idx`).
        If empty, files are ordered naturally. Defaults to an empty list.
    fformat : str, optional
        File extension of the images to process (e.g., 'png', 'tif').
        Defaults to 'png'.

    Returns
    -------
    None
        This function modifies files in place within `imdir` and does not return
        any value.

    Raises
    ------
    AssertionError
        If `include_idx` length doesn't match the number of found files, or if
        `order_idx` length doesn't match the number of included files.
    FileNotFoundError
        If `imdir` is None or invalid, `glob` might return an empty list,
        leading to issues if not handled or if assertions fail.

    Notes
    -----
    - **Caution:** This function can permanently delete or rename files.
    - The renaming logic uses a `nameDict` which maps parent directory names
      (e.g., 'full_masks') to new file prefixes (e.g., 'mask').
    - The line `fileList = glob.glob(onepath + '\*.' + fformat)` appears to
      contain a typo (`onepath`) and non-standard glob pattern; it should likely
      use `imdir` and `os.path.join`. This has been corrected in the assumed fixed code.
    """
    if imdir is None:
        warnings.warn("imdir is None, file operations may fail or use current directory.")
        # Or raise ValueError("imdir cannot be None")
        return

    nameDict = {'full_masks': 'mask',
                'mBq_images': 'mBq',
                'ROI_masks': 'mask'}

    fileList = glob.glob(os.path.join(imdir, '*.' + fformat))
    fileList.sort(key=helper.natural_keys)

    if not fileList and not (len(include_idx)==0 and len(order_idx)==0): # Allow if no files and no ops expected
        warnings.warn(f"No files found in {imdir} with format {fformat}. Cannot organize.")
        return

    # error checking for input indices
    if len(include_idx) == 0 and fileList: # if empty, default to include all
        include_idx = np.ones(len(fileList), dtype=bool)
    if len(order_idx) == 0 and np.sum(include_idx) > 0: # if empty, default to current order for included
        order_idx = np.arange(np.sum(include_idx).astype(int))


    err_1 = f"Inclusion index length ({len(include_idx)}) doesn't match found files ({len(fileList)})."
    assert len(include_idx) == len(fileList), err_1

    num_included_files = np.sum(include_idx)
    err_2 = f"Order index length ({len(order_idx)}) doesn't match number of included files ({num_included_files})."
    assert len(order_idx) == num_included_files, err_2


    if helper.get_yn(input(f'Delete images in-place from {imdir}? (This is permanent)')):
        # Ensure include_idx is boolean for indexing
        del_list = np.array(fileList)[~np.array(include_idx, dtype=bool)]
        for f_path in del_list:
            try:
                os.remove(f_path)
                print(f"Deleted: {f_path}")
            except OSError as e:
                print(f"Error deleting {f_path}: {e}")
        # Update fileList to only contain files that were kept
        fileList = [f for i, f in enumerate(fileList) if include_idx[i]]


    if helper.get_yn(input(f'Rename images in-place in {imdir}? (This is permanent)')):
        # fileList should now only contain files to be renamed if deletion happened
        # If deletion didn't happen, we need to select based on include_idx again
        # However, the original code re-globs. For safety, let's use the potentially filtered fileList.
        # If deletion was skipped, fileList still contains all files.
        # We only want to rename files that are marked for inclusion.

        files_to_rename = [f for i, f in enumerate(fileList) if include_idx[i]] if not usr_yn else fileList
        if len(files_to_rename) != len(order_idx):
            # This case should be caught by assertions if include_idx was used correctly for deletion.
            # If deletion was skipped, and include_idx was not all True, this check is critical.
            print("Mismatch between files to rename and order_idx length. Skipping rename.")
            return

        # Corrected globbing path for renaming, assuming 'imdir' is the correct path.
        # The original 'onepath' was likely a typo for 'imdir'.
        # Renaming needs to be careful about overwriting if names collide.
        # A common strategy is to rename to temporary names first, then to final names.

        temp_names = []
        try:
            # Stage 1: Rename to temporary names
            for i, old_fpath in enumerate(files_to_rename):
                temp_suffix = f"_temp_{i}.{fformat}"
                temp_fpath = os.path.join(imdir, Path(old_fpath).stem + temp_suffix)
                os.rename(old_fpath, temp_fpath)
                temp_names.append(temp_fpath)

            # Stage 2: Rename from temporary to final names
            # Ensure order_idx is applied to the correctly sorted list of temp files
            temp_names.sort(key=helper.natural_keys) # Sort temp names if original order was important beyond include_idx

            dir_basename = os.path.basename(imdir)
            file_prefix = nameDict.get(dir_basename, dir_basename) # Use dir name if not in dict

            for i, temp_fpath in enumerate(temp_names):
                final_name = f"{file_prefix}_{order_idx[i]}.{fformat}"
                final_fpath = os.path.join(imdir, final_name)
                os.rename(temp_fpath, final_fpath)
                print(f"Renamed: {Path(temp_fpath).name} -> {final_name}")

        except OSError as e:
            print(f"Error during renaming: {e}")
            # Potentially try to revert renames from temp if error occurs mid-way (complex)
            print("Renaming process might be incomplete. Please check the directory.")


def preprocess_topdir(topdir: str, include_idx: list[bool] | np.ndarray = [], order_idx: list[int] | np.ndarray = []) -> None:
    """Preprocesses images across multiple subdirectories within a top directory.

    Allows for selective deletion and renaming of images based on predefined
    dictionaries for naming prefixes (`nameDict`) and file formats (`fformatDict`)
    associated with specific subdirectory names.

    Prompts the user for confirmation before performing deletions or renames.
    **Caution:** This function can modify files in place.

    Parameters
    ----------
    topdir : str
        The top-level directory containing subdirectories of images to be processed.
    include_idx : list[bool] | np.ndarray, optional
        A boolean list or array indicating which files to keep. If empty,
        defaults to keeping all files in each processed subdirectory. Length
        must match the number of files in *each* targeted subdirectory.
    order_idx : list[int] | np.ndarray, optional
        An array specifying the new numerical suffix for renaming included files.
        If empty, files are ordered naturally. Length must match the sum of
        True values in `include_idx` for *each* subdirectory.

    Returns
    -------
    None
        Modifies files within subdirectories of `topdir` and may rename `topdir` itself.

    Raises
    ------
    AssertionError
        If `include_idx` or `order_idx` lengths do not match file counts
        in a subdirectory being processed.
    """
    # rewrite function to just do one prompt for all three folderrs
    # then prompt to move to subdirectory under Mouse 1 kidney 1st
    nameDict = {'full_masks': 'mask',
                'mBq_images': 'mBq',
                'mBq_image_previews': 'mBq_preview',
                'ROI_masks': 'mask'}

    fformatDict = {'full_masks': 'png',
                   'mBq_images': 'tif',
                   'mBq_image_previews': 'png',
                   'ROI_masks': 'png'}

    subdirs = helper.list_studies(topdir)
    if not subdirs:
        print(f"No subdirectories found in {topdir}.")
        return

    # Consolidate user prompts for deletion and renaming to once per topdir run
    confirm_delete = helper.get_yn(input(f'Delete images in-place across subdirectories of {topdir}? (This is permanent)'))
    confirm_rename = helper.get_yn(input(f'Rename images in-place across subdirectories of {topdir}? (This is permanent)'))

    for subdir_path_str in subdirs: # Assuming list_studies returns list of paths
        subdir_basename = os.path.basename(subdir_path_str)

        fformat = fformatDict.get(subdir_basename)
        if not fformat:
            print(f"Skipping subdirectory '{subdir_basename}': No format defined in fformatDict.")
            continue

        fileList = glob.glob(os.path.join(subdir_path_str, '*.' + fformat))
        fileList.sort(key=helper.natural_keys)

        if not fileList:
            print(f"No files with format '.{fformat}' found in {subdir_path_str}.")
            continue

        current_include_idx = np.array(include_idx) if len(include_idx) > 0 else np.ones(len(fileList), dtype=bool)

        err_msg_include = (f"Inclusion index length ({len(current_include_idx)}) doesn't match "
                           f"found files ({len(fileList)}) in subdir {subdir_basename}.")
        assert len(current_include_idx) == len(fileList), err_msg_include

        if confirm_delete:
            del_list = np.array(fileList)[~current_include_idx]
            print(f"In {subdir_basename}, attempting to delete {len(del_list)} files...")
            for f_path in del_list:
                try:
                    os.remove(f_path)
                    # print(f"Deleted: {f_path}") # Can be too verbose
                except OSError as e:
                    print(f"Error deleting {f_path}: {e}")
            fileList = [f for i, f in enumerate(fileList) if current_include_idx[i]] # Update fileList

        if confirm_rename:
            num_included_files = len(fileList) # After potential deletion
            current_order_idx = np.array(order_idx) if len(order_idx) > 0 else np.arange(num_included_files)

            err_msg_order = (f"Order index length ({len(current_order_idx)}) doesn't match "
                             f"number of included files ({num_included_files}) in subdir {subdir_basename}.")
            assert len(current_order_idx) == num_included_files, err_msg_order

            # files_to_rename = fileList # Already filtered if deletion occurred
            print(f"In {subdir_basename}, attempting to rename {len(fileList)} files...")
            temp_names = []
            try:
                # Stage 1: Rename to temporary names
                for i, old_fpath in enumerate(fileList): # fileList is already sorted naturally
                    temp_suffix = f"_temp_{i}.{fformat}"
                    # Corrected: use subdir_path_str for joining temp name
                    temp_fpath = os.path.join(subdir_path_str, Path(old_fpath).stem + temp_suffix)
                    os.rename(old_fpath, temp_fpath)
                    temp_names.append(temp_fpath)

                # Stage 2: Rename from temporary to final names
                # temp_names list is already in the correct order corresponding to original fileList
                # (which was sorted and potentially filtered)
                file_prefix = nameDict.get(subdir_basename, subdir_basename)

                for i, temp_fpath in enumerate(temp_names):
                    final_name = f"{file_prefix}_{current_order_idx[i]}.{fformat}"
                    final_fpath = os.path.join(subdir_path_str, final_name) # Use subdir_path_str
                    os.rename(temp_fpath, final_fpath)
                    # print(f"Renamed: {Path(temp_fpath).name} -> {final_name}") # Can be too verbose
            except OSError as e:
                print(f"Error during renaming in {subdir_basename}: {e}")
                print("Renaming process for this subdirectory might be incomplete.")


    if helper.get_yn(input(f'Rename the top directory {topdir}?')):
        new_topdir_name_base = input('Enter new name for the top directory: ')
        # Construct new path at the same level as the original topdir
        new_topdir_path = os.path.join(os.path.dirname(topdir), new_topdir_name_base)
        try:
            os.rename(topdir, new_topdir_path)
            print(f"Renamed top directory '{topdir}' to '{new_topdir_path}'")
        except OSError as e:
            print(f"Error renaming top directory {topdir}: {e}")


def ignore_images(data: np.ndarray, exclude_list: list[int] | np.ndarray, pad: str | bool = 'backwards') -> np.ndarray:
    """Generates a new image stack by excluding or replacing images specified
    in `exclude_list`.

    Parameters
    ----------
    data : np.ndarray
        The input 3D image stack (num_images, height, width).
    exclude_list : list[int] | np.ndarray[int]
        A list or array of integer indices corresponding to the images in `data`
        that should be excluded or replaced.
    pad : str | bool, optional
        Determines how excluded images are handled:
        - If `'backwards'` (default if `pad` is True): Replaces the image at `idx`
          from `exclude_list` with the image at `idx-1`. If `idx` is 0, it's
          replaced by the image at `idx+1` (i.e., index 1).
        - If `'forwards'`: Replaces the image at `idx` with the image at `idx+1`.
          Care should be taken if the last image is in `exclude_list` as this
          can cause an IndexError.
        - If `False` (boolean type) or any string other than 'backwards'/'forwards':
          Images in `exclude_list` are removed from the stack, and the stack
          size decreases.

    Returns
    -------
    np.ndarray
        The modified image stack. Its size along the first axis may be smaller
        if images are removed (i.e., `pad` is False or an unrecognized string).

    Notes
    -----
    - The original note "This function can only handle 2 'bad' images in a row"
      likely refers to the padding logic. If multiple consecutive images are
      flagged for padding, the replacement source image might itself be a
      previously padded image, leading to duplication. This implementation iterates
      through `exclude_list` so the order of exclusions matters.
    - If `pad='forwards'` and the last image index is in `exclude_list`, an
      IndexError will occur if not handled carefully (current code attempts to use idx-1).
    - If `pad` is a boolean `True`, it's treated as `'backwards'`.
    """
    idx_list_arr = np.array(exclude_list, dtype=int)

    if len(idx_list_arr) == 0:
        return data.copy() # Return a copy to avoid modifying original if no changes

    # Determine effective padding mode
    padding_active = False
    padding_mode = ''
    if isinstance(pad, str) and pad.lower() in ['backwards', 'forwards']:
        padding_active = True
        padding_mode = pad.lower()
    elif isinstance(pad, bool) and pad: # if pad is True, default to 'backwards'
        padding_active = True
        padding_mode = 'backwards'

    if padding_active:
        data_clean = np.copy(data)
        # Iterate through sorted unique indices to handle replacements consistently
        # especially if an index is listed multiple times or order is random.
        for idx in sorted(np.unique(idx_list_arr)):
            if idx < 0 or idx >= len(data):
                warnings.warn(f"Index {idx} in exclude_list is out of bounds for data of length {len(data)}. Skipping.")
                continue

            if padding_mode == 'backwards':
                if idx == 0: # First image
                    replacement_idx = idx + 1 if len(data) > 1 else idx # Use next if exists, else self (no change)
                else: # Not the first image
                    replacement_idx = idx - 1
            elif padding_mode == 'forwards':
                if idx == len(data) - 1: # Last image
                    replacement_idx = idx - 1 if len(data) > 1 else idx # Use previous if exists, else self
                else: # Not the last image
                    replacement_idx = idx + 1

            if replacement_idx < 0 or replacement_idx >= len(data) or replacement_idx == idx : # safety check
                 warnings.warn(f"Cannot find suitable replacement for image at index {idx}. Original image kept at this position.")
                 continue
            data_clean[idx] = data[replacement_idx] # Use original data for replacement source
        return data_clean
    else: # pad is False or an unrecognized string, so remove images
        # Ensure idx_list_arr is sorted and unique to prevent issues with boolean mask creation
        # if original exclude_list had duplicates or was unsorted, though np.array(unique()) handles this.
        valid_indices_mask = np.ones(len(data), dtype=bool)
        if len(idx_list_arr) > 0: # Ensure idx_list_arr is not empty before trying to index
            valid_indices_mask[idx_list_arr[idx_list_arr < len(data)]] = False # Only use valid indices
        data_clean = data[valid_indices_mask]
        return data_clean


def get_SSD(im1: np.ndarray, im2: np.ndarray) -> float:
    """Computes the Sum of Squared Differences (SSD) between two images.

    SSD is calculated as `sum((im1 - im2)^2) / N`, where N is the total
    number of pixels in `im1`. This is equivalent to Mean Squared Error (MSE).

    Parameters
    ----------
    im1 : np.ndarray
        The first image. Must have the same shape as `im2`.
    im2 : np.ndarray
        The second image. Must have the same shape as `im1`.

    Returns
    -------
    float
        The Sum of Squared Differences (SSD) or Mean Squared Error (MSE) value.
        Returns NaN if images are empty or shapes mismatch (though shape mismatch
        would raise error earlier).
    """
    if im1.shape != im2.shape:
        raise ValueError("Input images must have the same shape.")
    if im1.size == 0:
        return np.nan # Or raise error

    N = np.size(im1)
    SSD = np.sum((im1.astype(np.float64)-im2.astype(np.float64))**2)/N # Use float64 for precision
    return SSD


def coarse_rotation(mov: np.ndarray, ref: np.ndarray, deg: float = 2, interpolation: int = 0, gauss: int | None = 5, preserve_range: bool = True, recenter: bool = False, convert_to_grayscale_for_ssd: bool = False) -> tuple[np.ndarray, float]:
    """Performs coarse rotation of a "moving" image to align it with a
    "reference" image by minimizing Sum of Squared Differences (SSD/MSE).

    This method tests rotations in angular increments of `deg` over a full
    360-degree range and applies the rotation that yields the minimum SSD.
    Images can be optionally blurred before SSD calculation.

    Parameters
    ----------
    mov : np.ndarray
        The 2D image to be rotated (the "moving" image).
    ref : np.ndarray
        The 2D reference image to align to. Must have the same shape as `mov`
        after potential recentering and before rotation.
    deg : float, optional
        The angular increment in degrees for testing rotations. Defaults to 2.0.
    interpolation : int, optional
        Order of interpolation for `skimage.transform.rotate`.
        0: Nearest-neighbor (fastest, good for masks).
        1: Bi-linear (default for `skimage.transform.rotate`).
        Higher orders (up to 5) are B-splines. Defaults to 0.
    gauss : int | None, optional
        Sigma for Gaussian blur applied to both images before SSD calculation.
        If None or 0, no blurring is applied. Defaults to 5.
    preserve_range : bool, optional
        Passed to `skimage.transform.rotate`. If True, output image has the
        same intensity range as the input. Defaults to True.
    recenter : bool, optional
        If True, both `mov` and `ref` images are recentered using `recenter_im`
        before alignment. Defaults to False.
    convert_to_grayscale_for_ssd : bool, optional
        If True and input images are RGB(A), they will be converted to grayscale
        before Gaussian blur (if any) and SSD calculation. The final rotation
        is still applied to the original `mov` image. Defaults to False.

    Returns
    -------
    tuple[np.ndarray, float]
        - rot_image : np.ndarray
            The rotated version of the input `mov` image that best aligns with `ref`.
        - optimal_deg : float
            The rotation angle (in degrees) that resulted in the minimum SSD.

    Dependencies
    ------------
    - `numpy`
    - `cv2.GaussianBlur` (if `gauss` is not None)
    - `skimage.transform.rotate`
    - `iqid.align.get_SSD`
    - `iqid.align.recenter_im` (if `recenter` is True)
    - `cv2.cvtColor` (if `convert_to_grayscale_for_ssd` is True)
    """

    if recenter:
        mov = recenter_im(mov)
        ref = recenter_im(ref)

    # Work with float copies for processing, original 'mov' is rotated at the end.
    mov_proc = mov.astype(np.float32)
    ref_proc = ref.astype(np.float32)

    if convert_to_grayscale_for_ssd:
        if mov_proc.ndim == 3 and mov_proc.shape[-1] in [3, 4]: # Check if color
            # Normalize if it's float in 0-255 range, then convert
            max_val_mov = np.max(mov_proc)
            if max_val_mov > 1.5 and max_val_mov <=255: # Heuristic for 0-255 range float
                 mov_proc_u8 = mov_proc.astype(np.uint8)
            elif max_val_mov <=1.0 and max_val_mov > 0: # Float 0-1 range
                 mov_proc_u8 = (mov_proc * 255).astype(np.uint8)
            else: # Already uint8 or other cases
                 mov_proc_u8 = mov_proc.astype(np.uint8) if np.issubdtype(mov_proc.dtype, np.integer) else mov_proc
            mov_proc = cv2.cvtColor(mov_proc_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if max_val_mov <=1.0: mov_proc /= 255.0 # Renormalize to 0-1 if original was

        if ref_proc.ndim == 3 and ref_proc.shape[-1] in [3, 4]: # Check if color
            max_val_ref = np.max(ref_proc)
            if max_val_ref > 1.5 and max_val_ref <=255:
                 ref_proc_u8 = ref_proc.astype(np.uint8)
            elif max_val_ref <=1.0 and max_val_ref > 0:
                 ref_proc_u8 = (ref_proc * 255).astype(np.uint8)
            else:
                 ref_proc_u8 = ref_proc.astype(np.uint8) if np.issubdtype(ref_proc.dtype, np.integer) else ref_proc
            ref_proc = cv2.cvtColor(ref_proc_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if max_val_ref <=1.0: ref_proc /= 255.0


    if gauss is not None and gauss > 0:
        # Ensure kernel size is odd for GaussianBlur
        gauss_ksize = gauss if gauss % 2 == 1 else gauss + 1
        gmov = cv2.GaussianBlur(mov_proc, (gauss_ksize, gauss_ksize), 0)
        gref = cv2.GaussianBlur(ref_proc, (gauss_ksize, gauss_ksize), 0)
    else:
        gmov = mov_proc
        gref = ref_proc

    num_measurements = int(np.floor(360.0/deg))
    ssd_values = np.zeros(num_measurements)
    for i in range(num_measurements):
        current_angle = deg * i
        # Rotate a copy of gmov for SSD calculation
        rotated_gmov = transform.rotate(gmov, current_angle, resize=False, center=None, order=interpolation, mode='constant', cval=0, clip=True, preserve_range=True)
        # Ensure shapes match after rotation if resize=False (it should, but padding might differ)
        # If shapes are problematic, one might need to crop/pad `rotated_gmov` to `gref.shape`
        if rotated_gmov.shape != gref.shape:
             rotated_gmov = transform.resize(rotated_gmov, gref.shape, order=interpolation, mode='constant', cval=0, clip=True, preserve_range=True)

        ssd_values[i] = get_SSD(rotated_gmov, gref)

    optimal_deg_idx = np.argmin(ssd_values)
    optimal_deg = deg * optimal_deg_idx

    # Rotate the original 'mov' image (not the potentially grayscaled/blurred one)
    rot_image = transform.rotate(mov, optimal_deg, resize=False, center=None,
                                 order=interpolation, mode='constant', cval=0,
                                 clip=True, preserve_range=preserve_range)

    # If original mov was uint8 and preserve_range is True, output might be float.
    # Consider converting back if needed, though often float is fine for further processing.
    # if mov.dtype == np.uint8 and preserve_range:
    #    rot_image = util.img_as_ubyte(rot_image)

    return rot_image, optimal_deg


def coarse_stack(unreg: np.ndarray, deg: float = 2, avg_over: int = 1, preserve_range: bool = True, return_deg: bool = False, convert_to_grayscale_for_ssd: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Aligns a stack of images using SSD-minimizing coarse rotation.

    Each image `n` (from 1 to end) is coarsely aligned to a reference.
    The reference is either the immediately preceding aligned image (`avg_over=1`)
    or the mean of `avg_over` previously aligned images.
    The 0-th image in the stack is typically not changed unless `avg_over` logic
    implies it (e.g., if it were part of an averaging window for the first few images,
    though current logic starts alignment from index 1).

    Parameters
    ----------
    unreg : np.ndarray
        A 3D NumPy array representing the stack of unregistered images
        (num_images, height, width) or (num_images, height, width, channels).
    deg : float, optional
        The angular increment in degrees for `coarse_rotation`. Defaults to 2.0.
    avg_over : int, optional
        The number of previously aligned images to average to create the
        reference image for the current image's alignment. If 1, aligns to
        the immediate previous successfully aligned image. Defaults to 1.
    preserve_range : bool, optional
        Passed to `coarse_rotation` (and then to `skimage.transform.rotate`).
        If True, the output of rotation has the same intensity range as input.
        Defaults to True.
    return_deg : bool, optional
        If True, returns a tuple containing the registered stack and an array
        of the rotation degrees applied to each image. Defaults to False.
    convert_to_grayscale_for_ssd : bool, optional
        Passed to `coarse_rotation`. If True and images are color, SSD calculation
        is done on grayscale versions. Defaults to False.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        - reg_stack : np.ndarray
            The (coarsely) registered stack of images, same shape as `unreg`.
        - rotation_degrees : np.ndarray (optional)
            If `return_deg` is True, a 1D array of rotation degrees applied to
            each image. `rotation_degrees[0]` will be 0.

    Notes
    -----
    - The implementation detail: "If avg_over = m, then the first m images are
      aligned to the average of as many aligned images as possible." This means
      for image `i < avg_over`, the reference is `mean(reg[0:i])`. For
      `i >= avg_over`, reference is `mean(reg[i-avg_over:i])`.
    - The 0-th image is copied directly if `avg_over` is 1 or more, as alignment loop starts from 1.
    """

    reg = np.zeros_like(unreg)
    reg[0:avg_over, :, :] = unreg[0:avg_over, :, :]

    degs = np.zeros(len(unreg))

    for i in np.arange(1, avg_over):
        ref = np.mean(reg[:i, :, :], axis=0)
        mov = unreg[i, :, :]
        reg[i, :, :], degs[i] = coarse_rotation(
            mov, ref, deg, preserve_range=preserve_range, convert_to_grayscale_for_ssd=convert_to_grayscale_for_ssd)

    for i in np.arange(avg_over, len(unreg)):
        ref = np.mean(reg[(i-avg_over):i, :, :], axis=0)
        mov = unreg[i, :, :]
        reg[i, :, :], degs[i] = coarse_rotation(
            mov, ref, deg, preserve_range=preserve_range, convert_to_grayscale_for_ssd=convert_to_grayscale_for_ssd)

    if return_deg:
        return (reg, degs)
    else:
        return (reg)


def format_circle(h: float, k: float, r: float, numpoints: int = 400) -> np.ndarray:
    """Generates coordinates for a circle, often used as an initial contour
    for active contour algorithms (snakes).

    The circle is defined by its center (h, k) and radius r. Coordinates are
    returned in (row, column) format suitable for scikit-image functions.

    Parameters
    ----------
    h : float
        Row-coordinate (vertical) of the circle's center.
    k : float
        Column-coordinate (horizontal) of the circle's center.
    r : float
        Radius of the circle.
    numpoints : int, optional
        Number of points to generate for defining the circle's perimeter.
        Defaults to 400.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (`numpoints`, 2) where each row is `[row, col]`,
        representing the coordinates of points on the circle.

    Notes
    -----
    The original docstring mentioned: "Future work could incorporate oblate-ness."
    """
    s = np.linspace(0, 2*np.pi, numpoints)
    row = h + r*np.sin(s)
    col = k + r*np.cos(s)
    init_circle = np.array([row, col]).T
    return init_circle # Removed extra parentheses


def binary_mask(img: np.ndarray, finagle: float = 1) -> np.ndarray:
    """Creates a binary mask from a grayscale image using Otsu's thresholding
    method, with an optional adjustment factor.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (2D NumPy array).
    finagle : float, optional
        A "finagle-factor" that multiplies the Otsu threshold. Values > 1
        will make the thresholding stricter (less foreground), while values < 1
        will make it more lenient (more foreground). Defaults to 1 (standard Otsu).

    Returns
    -------
    np.ndarray
        A boolean NumPy array representing the binary mask (True for foreground,
        False for background).
    """
    # simple Otsu method to generate a mask
    thresh = filters.threshold_otsu(img)
    mask = img > (finagle * thresh)
    return mask # Removed extra parentheses


def mask_from_contour(img: np.ndarray, snake: np.ndarray) -> np.ndarray:
    """Creates a binary mask from a contour (snake).

    The area enclosed by the `snake` coordinates is filled to create the mask.

    Parameters
    ----------
    img : np.ndarray
        The reference image, used to determine the shape of the output mask.
        Can be 2D or 3D (e.g. color image, mask will be 2D).
    snake : np.ndarray
        A NumPy array of shape (N, 2) where N is the number of points in the
        contour, and each row is `[row_coordinate, column_coordinate]`.

    Returns
    -------
    np.ndarray
        A 2D binary mask of the same height and width as `img`, with pixels
        inside the contour set to 1 and others to 0. The dtype will be the
        same as `np.zeros_like(img)`.
    """
    rr, cc = draw.polygon(snake[:, 0], snake[:, 1], img.shape[:2]) # Use img.shape[:2] for H,W
    newmask = np.zeros_like(img, dtype=np.uint8) # Ensure uint8 mask
    if img.ndim == 3: # If original image is color, make mask 2D
        newmask = np.zeros(img.shape[:2], dtype=np.uint8)
    newmask[rr, cc] = 1
    return newmask # Removed extra parentheses


def to_shape(mov: np.ndarray, target_w: int, target_h: int, pad_val: int | float | tuple = 0) -> np.ndarray:
    """Pads a 2D image (`mov`) with a constant value to reach a target shape
    (`target_w` width, `target_h` height).

    Padding is distributed as symmetrically as possible.

    Parameters
    ----------
    mov : np.ndarray
        The 2D image (grayscale or single channel) to be padded.
    target_w : int
        The target width for the padded image.
    target_h : int
        The target height for the padded image.
    pad_val : int | float | tuple, optional
        Value used for padding. If `mov` is multi-channel and `pad_val` is
        a single value, it's used for all channels. If a tuple, its length
        should match the number of channels. Defaults to 0.
        The original `vals=(0,0)` seemed to imply separate x/y padding values,
        but `np.pad` with `constant_values` takes a single value or pair for before/after.
        This is simplified to a single `pad_val` for the constant mode.

    Returns
    -------
    np.ndarray
        The padded 2D image of shape (`target_h`, `target_w`).

    Raises
    ------
    ValueError
        If `target_w` or `target_h` are smaller than the dimensions of `mov`.
    """
    current_h, current_w = mov.shape[:2] # Works for 2D, for 3D takes first two

    if target_h < current_h or target_w < current_w:
        raise ValueError("Target dimensions must be greater than or equal to current dimensions.")

    pad_h_total = target_h - current_h
    pad_w_total = target_w - current_w

    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    padding_config = ((pad_top, pad_bottom), (pad_left, pad_right))

    # If image has channels, add (0,0) padding for channel axis
    if mov.ndim == 3:
        padding_config += ((0,0),)

    return np.pad(mov, padding_config, mode='constant', constant_values=pad_val)


def pad_2d_masks(movmask: np.ndarray, refmask: np.ndarray, pad_func: callable = to_shape) -> tuple[np.ndarray, np.ndarray]:
    """Pads two 2D masks to the maximum dimensions found between them.

    This function determines the largest height and width from `movmask` and
    `refmask`, then uses `pad_func` (defaulting to `iqid.align.to_shape`)
    to pad both masks to these maximum dimensions. This is useful for
    ensuring two masks have the same shape before comparison or combination.

    Parameters
    ----------
    movmask : np.ndarray
        The first 2D binary mask.
    refmask : np.ndarray
        The second 2D binary mask.
    pad_func : callable, optional
        The function used for padding. It should accept an image, target width,
        and target height as arguments. Defaults to `iqid.align.to_shape`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - movpad : np.ndarray
            The `movmask` padded to the maximum dimensions.
        - refpad : np.ndarray
            The `refmask` padded to the maximum dimensions.
    """
    h_mov, w_mov = movmask.shape
    h_ref, w_ref = refmask.shape

    max_h = max(h_mov, h_ref)
    max_w = max(w_mov, w_ref)

    # Assuming to_shape uses default padding value 0, suitable for binary masks
    movpad = pad_func(movmask, max_w, max_h)
    refpad = pad_func(refmask, max_w, max_h)

    return movpad, refpad # Removed extra parentheses


def to_shape_rgb(mov: np.ndarray, target_w: int, target_h: int, pad_color_rgb: tuple[int,int,int] = (255, 255, 255)) -> np.ndarray:
    """Pads a 3-channel (RGB) image to a target shape (`target_w`, `target_h`)
    using a specified RGB color for padding.

    This is a convenience wrapper around `np.pad` for RGB images.
    Padding is distributed as symmetrically as possible.

    Parameters
    ----------
    mov : np.ndarray
        The 3D RGB image (height, width, 3 channels) to be padded.
    target_w : int
        The target width for the padded image.
    target_h : int
        The target height for the padded image.
    pad_color_rgb : tuple[int, int, int], optional
        RGB tuple (e.g., (R,G,B)) to use for padding. Defaults to (255, 255, 255) (white).

    Returns
    -------
    np.ndarray
        The padded RGB image of shape (`target_h`, `target_w`, 3).

    Raises
    ------
    ValueError
        If `target_w` or `target_h` are smaller than current image dimensions,
        or if `mov` is not a 3-channel image.
    """
    if mov.ndim != 3 or mov.shape[2] != 3:
        raise ValueError("Input image `mov` must be a 3-channel RGB image.")

    current_h, current_w, _ = mov.shape

    if target_h < current_h or target_w < current_w:
        raise ValueError("Target dimensions must be greater than or equal to current dimensions.")

    pad_h_total = target_h - current_h
    pad_w_total = target_w - current_w

    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    # np.pad expects pad_width to be ((top, bottom), (left, right), (before_c, after_c))
    # For constant_values, if it's a sequence, it should match the number of axes.
    # However, for RGB, it's often easier to pad each channel if values differ,
    # or create a target full array and place the image.
    # Given 'constant_values=(255,255)' in original, it's ambiguous.
    # Assuming a single pad_color_rgb for all padded areas.

    # Create a new array of the target size filled with the pad color
    padded_image = np.full((target_h, target_w, 3), pad_color_rgb, dtype=mov.dtype)
    # Place the original image in the center
    padded_image[pad_top:pad_top+current_h, pad_left:pad_left+current_w, :] = mov

    return padded_image


def pad_rgb_im(im_2d: np.ndarray, im_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pads a 2D grayscale/binary image and an RGB image to the maximum
    dimensions found between them.

    The 2D image is padded with zeros. The RGB image is padded with white.

    Parameters
    ----------
    im_2d : np.ndarray
        The 2D grayscale or binary image.
    im_rgb : np.ndarray
        The 3D RGB image.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - movpad : np.ndarray
            The padded 2D image.
        - refpad : np.ndarray
            The padded RGB image.
    """
    h_2d, w_2d = im_2d.shape
    h_rgb, w_rgb, _ = im_rgb.shape # Assumes im_rgb is 3D

    max_h = max(h_2d, h_rgb)
    max_w = max(w_2d, w_rgb)

    # Pad 2D image (e.g., mask) using to_shape (default pad_val=0)
    movpad = to_shape(im_2d, max_w, max_h)
    # Pad RGB image using to_shape_rgb (default pad_color_rgb=(255,255,255) white)
    refpad = to_shape_rgb(im_rgb, max_w, max_h)

    return movpad, refpad # Removed extra parentheses


def to_shape_center(im: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Pads a 2D image to a target shape by centering the original image
    within a new array filled with zeros.

    Parameters
    ----------
    im : np.ndarray
        The 2D input image.
    target_w : int
        Target width of the output image.
    target_h : int
        Target height of the output image.

    Returns
    -------
    np.ndarray
        The padded and centered image of shape (`target_h`, `target_w`).
        The dtype of the output array matches the dtype of the input `im`.
        If target dimensions are smaller than image dimensions, the image is
        placed at (0,0) and cropped if this function were to allow it, but
        np.full would require target to be larger or equal.
        This implementation assumes target dimensions are >= image dimensions.
    """
    h, w = im.shape[:2] # Handles 2D, for 3D would take spatial dims

    if target_h < h or target_w < w:
        # This case should ideally be handled by cropping or raising an error.
        # For now, let's adapt to behavior similar to other padding funcs if target is smaller.
        # However, the logic `np.full` then slice assignment implies target must be larger.
        # Sticking to the original logic: create a full-sized canvas.
        warnings.warn("Target dimensions are smaller than image dimensions. Image will be cropped if placed at center.")
        # To strictly match original logic: result = np.full((target_h, target_w), 0, dtype=im.dtype)
        # Then placement. If target_h < h, this placement is an error.
        # For now, assume target_h >=h and target_w >=w as per typical padding.
        # If not, the slicing result[y_center:y_center+h, x_center:x_center+w] will be problematic.
        # Let's proceed assuming target dimensions are for padding, not cropping.
        pass


    result = np.full((target_h, target_w), 0, dtype=im.dtype)

    x_center_offset = (target_w - w) // 2
    y_center_offset = (target_h - h) // 2

    # Define the slices for placing the original image
    # Ensure these slices are within the bounds of the 'result' array
    # And also that they correspond to the full original image 'im'

    # Start and end for y dimension in result array
    y_start_res = max(0, y_center_offset)
    y_end_res = min(target_h, y_center_offset + h)
    # Start and end for x dimension in result array
    x_start_res = max(0, x_center_offset)
    x_end_res = min(target_w, x_center_offset + w)

    # Start and end for y dimension in original image 'im'
    y_start_im = max(0, -y_center_offset)
    y_end_im = min(h, target_h - y_center_offset)
    # Start and end for x dimension in original image 'im'
    x_start_im = max(0, -x_center_offset)
    x_end_im = min(w, target_w - x_center_offset)

    result[y_start_res:y_end_res, x_start_res:x_end_res] = im[y_start_im:y_end_im, x_start_im:x_end_im]
    return result


def crop_down(rgb_overlay: np.ndarray, rgb_ref: np.ndarray, axis: str = 'both') -> np.ndarray:
    """Crops an overlay image (`rgb_overlay`) to match the spatial dimensions
    (height and width) of a reference image (`rgb_ref`).

    Cropping is done by removing pixels symmetrically from the edges of
    `rgb_overlay` until its H and W match `rgb_ref`. This function assumes
    `rgb_overlay` is larger than or equal to `rgb_ref` in the dimensions
    being cropped.

    Parameters
    ----------
    rgb_overlay : np.ndarray
        The image to be cropped. Expected to be 3D (H, W, C) for RGB images,
        but could also work for 2D if `axis` logic is adapted or channel
        padding `(0,0)` in `util.crop` is handled.
    rgb_ref : np.ndarray
        The reference image whose spatial dimensions (height, width) will be
        used as the target for cropping `rgb_overlay`.
    axis : str, optional
        Specifies which axes to crop:
        - 'both' (default): Crop along height (axis 0) and width (axis 1).
        - 'x': Crop along width (axis 1) only.
        - 'y': Crop along height (axis 0) only.
        An invalid choice prints a message but currently doesn't raise an error
        and might lead to unexpected behavior due to `xbool`/`ybool` not being set.

    Returns
    -------
    np.ndarray
        The cropped version of `rgb_overlay`.

    Raises
    ------
    ValueError
        If `rgb_overlay` is smaller than `rgb_ref` in a dimension being cropped.
    """
    h_overlay, w_overlay = rgb_overlay.shape[:2]
    h_ref, w_ref = rgb_ref.shape[:2]

    dheight = h_overlay - h_ref
    dwidth = w_overlay - w_ref

    if dheight < 0 or dwidth < 0:
        raise ValueError("Overlay image must be larger than or equal to reference image in dimensions being cropped.")

    xbool, ybool = 0, 0
    if axis == 'both':
        xbool, ybool = 1, 1
    elif axis == 'x': # Crop width only
        xbool = 1
    elif axis == 'y': # Crop height only
        ybool = 1
    else:
        # Consider raising ValueError for invalid axis choice
        print(f"Invalid axis choice: '{axis}'. No cropping will be performed along specified axis logic.")
        # Defaulting to no crop for the problematic axis logic if axis is invalid
        # However, the original tuple creation might still fail or misbehave.
        # For safety, if axis is invalid, perhaps return original image or raise error.
        # For now, let it proceed but it might not do what's expected.

    # crop_width_h / crop_width_w are tuples for ((before_0, after_0), (before_1, after_1), ...)
    crop_height_tuple = (ybool * (dheight // 2 + dheight % 2), ybool * (dheight // 2))
    crop_width_tuple = (xbool * (dwidth // 2 + dwidth % 2), xbool * (dwidth // 2))

    # Channel dimension cropping (3rd dim for color, not cropped here)
    crop_channel_tuple = (0,0) if rgb_overlay.ndim == 3 else () # No cropping for channels

    crop_config = [crop_height_tuple, crop_width_tuple]
    if rgb_overlay.ndim == 3:
        crop_config.append(crop_channel_tuple)

    try:
        cropped_ol = util.crop(rgb_overlay, tuple(crop_config))
    except Exception as e:
        print(f"Error during cropping with config {crop_config} for shapes {rgb_overlay.shape}, {rgb_ref.shape}: {e}")
        return rgb_overlay # Return original on error
    return cropped_ol


def crop_to(im: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Crops a 2D or 3D image to a target width (`target_w`) and height
    (`target_h`) by removing pixels symmetrically from its borders.

    If the image dimensions are smaller than the target dimensions, a warning
    is issued, and the original image is returned.

    Parameters
    ----------
    im : np.ndarray
        The input image to be cropped (2D grayscale or 3D RGB/RGBA).
    target_w : int
        The target width for the cropped image.
    target_h : int
        The target height for the cropped image.

    Returns
    -------
    np.ndarray
        The cropped image. If original dimensions are smaller than target,
        the original image is returned with a warning.
    """
    current_h, current_w = im.shape[:2]

    if current_w < target_w or current_h < target_h:
        warnings.warn(
            f"Image dimensions ({current_h}H x {current_w}W) are smaller than "
            f"target crop dimensions ({target_h}H x {target_w}W). "
            "Returning original image."
        )
        return im

    delta_h = current_h - target_h
    delta_w = current_w - target_w

    crop_top = delta_h // 2
    crop_bottom = delta_h - crop_top
    crop_left = delta_w // 2
    crop_right = delta_w - crop_left

    if im.ndim == 3:  # RGB or RGBA image
        return im[crop_top:current_h-crop_bottom, crop_left:current_w-crop_right, :]
    elif im.ndim == 2:  # Grayscale image
        return im[crop_top:current_h-crop_bottom, crop_left:current_w-crop_right]
    else:
        warnings.warn(f"Unsupported image ndim: {im.ndim}. Returning original image.")
        return im


# Several helpful visualization functions from PyStackReg documentation
# https://pystackreg.readthedocs.io/en/latest/

def overlay_images(imgs: list[np.ndarray], equalize: bool = False, aggregator: callable = np.mean) -> np.ndarray:
    """Overlays a list of images into a single image, typically by averaging.
    Optionally equalizes histograms of input images first.

    (Adapted from PyStackReg documentation)

    Parameters
    ----------
    imgs : list[np.ndarray]
        A list of 2D NumPy arrays (images) to be overlaid.
        Images should ideally be of the same shape.
    equalize : bool, optional
        If True, applies histogram equalization (`skimage.exposure.equalize_hist`)
        to each image before overlaying. Defaults to False.
    aggregator : callable, optional
        A function that takes a NumPy array and an axis argument, used to
        combine the images along the newly stacked axis (axis=0).
        Defaults to `np.mean`. Other options could be `np.sum`, `np.median`, etc.

    Returns
    -------
    np.ndarray
        A 2D NumPy array representing the combined (e.g., averaged) overlay image.
    """
    if equalize:
        processed_imgs = [exposure.equalize_hist(img) for img in imgs]
    else:
        processed_imgs = imgs

    # Ensure all images are NumPy arrays before stacking
    processed_imgs = [np.asanyarray(img) for img in processed_imgs]

    if not processed_imgs:
        raise ValueError("Input image list is empty.")

    # Check if all images have the same shape, warn if not
    first_shape = processed_imgs[0].shape
    if not all(img.shape == first_shape for img in processed_imgs):
        warnings.warn("Input images for overlay do not all have the same shape. Aggregation might fail or produce unexpected results.")

    stacked_imgs = np.stack(processed_imgs, axis=0)
    return aggregator(stacked_imgs, axis=0)


def composite_images(imgs: list[np.ndarray], equalize: bool = False, aggregator: callable = np.mean) -> np.ndarray:
    """Creates a color composite image from a list of up to 3 grayscale images.

    Each image is normalized to its max, assigned to a color channel (R, G, B),
    and then stacked. If fewer than 3 images are provided, empty channels are
    added (filled with zeros).
    (Adapted from PyStackReg documentation)

    Parameters
    ----------
    imgs : list[np.ndarray]
        A list of 2D NumPy arrays (grayscale images). Expects 1 to 3 images.
    equalize : bool, optional
        If True, applies histogram equalization (`skimage.exposure.equalize_hist`)
        to each image before normalization and compositing. Defaults to False.
    aggregator : callable, optional
        This parameter is present in the original function signature from
        PyStackReg documentation but is not used in this implementation for the
        final dstacking operation. It's kept for signature consistency if adapting
        other parts but has no effect here. Defaults to `np.mean`.

    Returns
    -------
    np.ndarray
        An RGB composite image (H, W, 3).
    """
    if not imgs:
        raise ValueError("Input image list is empty.")

    if equalize:
        processed_imgs = [exposure.equalize_hist(img) for img in imgs]
    else:
        processed_imgs = [np.asanyarray(img) for img in imgs]

    # Normalize each image to its max
    normalized_imgs = []
    for img_arr in processed_imgs:
        max_val = img_arr.max()
        if max_val == 0: # Avoid division by zero for empty/black images
            normalized_imgs.append(img_arr.astype(np.float32)) # Keep as zeros
        else:
            normalized_imgs.append(img_arr.astype(np.float32) / max_val)

    # Ensure we have 3 images for RGB channels, pad with zeros if fewer
    if len(normalized_imgs) < 3:
        zeros_shape = normalized_imgs[0].shape # Get shape from first image
        for _ in range(3 - len(normalized_imgs)):
            normalized_imgs.append(np.zeros(zeros_shape, dtype=np.float32))

    # Use only the first 3 images if more are provided
    rgb_channels = normalized_imgs[:3]

    return np.dstack(rgb_channels)


def save_imbatch(imstack: np.ndarray, newdir: str, prefix: str, fformat: str = 'tif') -> None:
    """Saves each image in a 3D stack to a specified directory.

    Filenames are constructed using the given `prefix` and an index.
    Example: `prefix_0.tif`, `prefix_1.tif`, ...

    Parameters
    ----------
    imstack : np.ndarray
        A 3D NumPy array representing the image stack (num_images, height, width)
        or a 4D array for color images (num_images, height, width, channels).
    newdir : str
        The directory where the images will be saved. It will be created if
        it doesn't exist.
    prefix : str
        A prefix string for the output filenames.
    fformat : str, optional
        The file format (extension) for the saved images (e.g., 'tif', 'png').
        Defaults to 'tif'.

    Returns
    -------
    None
        Images are saved to disk.

    Notes
    -----
    When saving as 'tif', images are saved with `photometric='minisblack'`
    and `check_contrast=False` using the 'tifffile' plugin. This is often
    suitable for scientific image data. Other formats are saved with default
    `skimage.io.imsave` behavior.
    """
    Path(newdir).mkdir(parents=True, exist_ok=True)
    for i in trange(len(imstack), desc=f"Saving batch {prefix}"):
        save_path = os.path.join(newdir, f"{prefix}_{i}.{fformat}")
        current_image = imstack[i]
        try:
            if fformat.lower() in ['tif', 'tiff']:
                # Ensure image is not boolean for tifffile, convert if necessary
                if current_image.dtype == bool:
                    current_image = current_image.astype(np.uint8) * 255
                io.imsave(save_path, current_image, plugin='tifffile', photometric='minisblack', check_contrast=False)
            else:
                io.imsave(save_path, current_image, check_contrast=False) # check_contrast can be good for general images
        except Exception as e:
            warnings.warn(f"Could not save image {save_path}: {e}")


def concatenate_dsets(astack_1: np.ndarray, astack_2: np.ndarray) -> np.ndarray:
    """Concatenates two 3D image stacks along the first axis (number of images).

    Before concatenation, it pads all images in both stacks to the maximum
    height and width found across both stacks. Padding uses `pad_2d_masks`
    which in turn uses `to_shape` with default zero padding.

    Parameters
    ----------
    astack_1 : np.ndarray
        First 3D image stack (num_images1, height1, width1).
    astack_2 : np.ndarray
        Second 3D image stack (num_images2, height2, width2).

    Returns
    -------
    np.ndarray
        The concatenated image stack of shape
        (num_images1 + num_images2, max_height, max_width).

    Notes
    -----
    - Assumes input stacks are 3D (grayscale/single-channel images).
    - The padding logic within the loop `pad_2d_masks(astack_1[i], astack_2[0], ...)`
      uses `astack_2[0]` and `astack_1[0]` as shape references for determining
      max dimensions for padding each slice. This seems unusual as the max dimensions
      `xx` and `yy` are already computed from the entire stacks. A more direct
      padding of each slice to `(yy, xx)` would be expected.
      The current implementation might not correctly pad all slices to the
      overall `xx, yy` if `pad_2d_masks` doesn't use them directly.
      However, `pad_2d_masks` itself calculates max dims from its two inputs.
      This should be revised for clarity and correctness if `xx,yy` are the intended target.
      Assuming the goal is to make all images conform to `xx,yy`:
    """
    if astack_1.ndim != 3 or astack_2.ndim != 3:
        raise ValueError("Input stacks must be 3-dimensional.")

    # Determine overall max height (yy) and max width (xx)
    max_h_1 = astack_1.shape[1] if astack_1.size > 0 else 0
    max_w_1 = astack_1.shape[2] if astack_1.size > 0 else 0
    max_h_2 = astack_2.shape[1] if astack_2.size > 0 else 0
    max_w_2 = astack_2.shape[2] if astack_2.size > 0 else 0

    target_h = max(max_h_1, max_h_2)
    target_w = max(max_w_1, max_w_2)

    # Create new stacks with target dimensions
    # Infer dtype from first stack, or use a default like float if empty
    dtype_1 = astack_1.dtype if astack_1.size > 0 else np.float32
    dtype_2 = astack_2.dtype if astack_2.size > 0 else np.float32
    # Choose a compatible dtype, e.g. by promoting or assuming they should match
    final_dtype = np.promote_types(dtype_1, dtype_2)

    new_a1 = np.zeros((len(astack_1), target_h, target_w), dtype=final_dtype)
    new_a2 = np.zeros((len(astack_2), target_h, target_w), dtype=final_dtype)

    for i in range(len(astack_1)):
        # Pad each slice of astack_1 to target_h, target_w
        new_a1[i] = to_shape(astack_1[i], target_w, target_h) # Using to_shape directly

    for i in range(len(astack_2)):
        # Pad each slice of astack_2 to target_h, target_w
        new_a2[i] = to_shape(astack_2[i], target_w, target_h) # Using to_shape directly

    concatenated_stack = np.concatenate((new_a1, new_a2), axis=0)
    return concatenated_stack


def quantify_err(imstack: np.ndarray, reg: np.ndarray, tmat: np.ndarray, vis: bool = True) -> float:
    """Quantifies errors introduced by a registration process.

    Calculates maximum shear and zoom from transformation matrices `tmat`,
    and the percentage difference in summed intensity between the original
    `imstack` and the registered stack `reg`. Optionally visualizes overlays
    of the mean unregistered and registered images.

    Parameters
    ----------
    imstack : np.ndarray
        The original (unregistered) 3D image stack (num_images, H, W).
    reg : np.ndarray
        The registered 3D image stack (num_images, H, W).
    tmat : np.ndarray
        An array of transformation matrices, typically of shape
        (num_images, D, D+1) or (num_images, D+1, D+1), where D is غالباً 2.
        Used by `iqid.helper.decompose_affine`.
    vis : bool, optional
        If True, displays a side-by-side comparison of the mean unregistered
        image and the mean registered image. Defaults to True.

    Returns
    -------
    float
        The percentage difference in summed intensity (activity) between the
        registered stack and the original stack, calculated as
        ` (sum(reg) - sum(imstack)) / sum(imstack) * 100 `.

    Dependencies
    ------------
    - `numpy`
    - `matplotlib.pyplot` (if `vis` is True)
    - `iqid.helper.decompose_affine`
    - `iqid.align.overlay_images`
    """
    Ss = np.zeros(len(tmat)) # Shear values
    Zs = np.zeros((len(tmat), 2)) # Zoom values for X and Y

    for i in range(len(tmat)):
        try:
            # Decompose affine should handle various matrix sizes if general enough
            T_translate, R_rotate, Z_zoom, S_shear = helper.decompose_affine(tmat[i])
            Ss[i] = S_shear if np.isscalar(S_shear) else np.mean(S_shear) # Assuming S_shear might not be scalar
            Zs[i] = Z_zoom[:2] if hasattr(Z_zoom, '__len__') and len(Z_zoom) >=2 else [np.nan, np.nan] # Take X,Y zoom
        except Exception as e:
            warnings.warn(f"Could not decompose tmat index {i}: {e}. Skipping.")
            Ss[i] = np.nan
            Zs[i,:] = np.nan


    print(f"The largest amount of shear in this stack is {np.nanmax(Ss):.2f}.")
    # Assuming Zs contains scale factors, so (value - 1)*100 is percentage change
    max_zoom_pct = (np.nanmax(Zs, axis=0) - 1) * 100
    print(f"The largest zoom components in this stack are X: {max_zoom_pct[0]:.1f}%, Y: {max_zoom_pct[1]:.1f}%.")


    # Quantification of small-value errors introduced by rotations/transformations.
    sum_unreg = np.sum(imstack)
    sum_reg = np.sum(reg)
    if sum_unreg == 0: # Avoid division by zero
        pct_diff = np.inf if sum_reg != 0 else 0.0
    else:
        pct_diff = (sum_reg - sum_unreg) / sum_unreg * 100

    print(f'Small-value errors result in summed activity difference of {pct_diff:.2f}%.')

    if vis:
        # Ensure images are suitable for overlay (e.g., float, normalized if needed by overlay_images)
        agg_unreg = overlay_images(list(imstack.astype(np.float32)), aggregator=np.mean)
        agg_aff = overlay_images(list(reg.astype(np.float32)), aggregator=np.mean)

        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(agg_unreg, cmap='inferno')
        ax[0].set_title('Unregistered Stack (Mean)')
        ax[0].axis('off')

        ax[1].imshow(agg_aff, cmap='inferno')
        ax[1].set_title('Registered Stack (Mean)')
        ax[1].axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

    return pct_diff


def simple_slice(arr: np.ndarray, inds: int | slice | list[int], axis: int) -> np.ndarray:
    """Extracts a slice from a NumPy array along a specified axis using given indices.

    This is a utility function, noted as being from PyStackReg.
    (https://github.com/glichtner/pystackreg/blob/b5d9c032f7d0ba48d8472f8c8e6b4589a52bcdab/pystackreg/util/__init__.py#L57)

    Parameters
    ----------
    arr : np.ndarray
        The input NumPy array from which to extract the slice.
    inds : int | slice | list[int]
        The indices or slice object to use for slicing along the specified `axis`.
    axis : int
        The axis along which the slicing operation should be performed.

    Returns
    -------
    np.ndarray
        A view of the input array representing the extracted slice.
        Modifications to the returned array will affect the original array.
    """
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[tuple(sl)]


def transform_stack(img: np.ndarray, tmat: np.ndarray, axis: int = 0, order: int = 0) -> np.ndarray:
    """Applies a series of affine transformations to each slice of an image stack.

    This function iterates through slices of the input `img` stack along the
    specified `axis`, applying a corresponding transformation matrix from `tmat`
    to each slice using `skimage.transform.warp`. This version is modified
    from a PyStackReg function to allow explicit selection of interpolation order.

    Parameters
    ----------
    img : np.ndarray
        The input image stack (e.g., 3D for a series of 2D images, or 4D for
        a series of 3D volumes, though typically used for 3D stacks of 2D images).
    tmat : np.ndarray
        An array of transformation matrices. If `img` is a stack of N images,
        `tmat` should be of shape (N, D, D+1) or (N, D+1, D+1) for D-dimensional
        transformations (typically D=2 for 2D images, so matrices are (N,2,3) or (N,3,3)).
        `tmat[i]` is the transformation matrix for `img` slice `i`.
    axis : int, optional
        The axis along which slices are taken from `img` and iterated over.
        Defaults to 0 (e.g., for a stack (num_images, height, width)).
    order : int, optional
        The order of interpolation for `skimage.transform.warp`.
        0: Nearest-neighbor (preserves quantitative accuracy, good for masks).
        1: Bi-linear.
        Higher orders (up to 5) are B-splines.
        Defaults to 0 (Nearest-neighbor).

    Returns
    -------
    np.ndarray
        The transformed image stack, with the same shape as `img`, but of float type.
        Values outside boundaries are filled with 0 (cval=0).
    """
    out = img.copy().astype(float) # Output array, ensure float for warp

    for i in range(img.shape[axis]):
        sl_in = [slice(None)] * img.ndim
        sl_in[axis] = i
        current_slice = img[tuple(sl_in)]

        sl_out = [slice(None)] * out.ndim
        sl_out[axis] = i

        # Ensure tmat[i] is a valid transformation matrix (e.g., 2x3 or 3x3 for 2D image)
        # skimage.transform.warp expects a ProjectiveTransform or AffineTransform object,
        # or a (D+1, D+1) matrix.
        # If tmat[i] is (2,3), it needs to be converted to (3,3) affine.
        current_tmat = tmat[i]
        if current_tmat.shape == (2, 3): # Convert 2x3 to 3x3 if necessary
            affine_mat_3x3 = np.identity(3)
            affine_mat_3x3[:2, :3] = current_tmat
            tf = transform.AffineTransform(matrix=affine_mat_3x3)
        elif current_tmat.shape == (3, 3):
            tf = transform.AffineTransform(matrix=current_tmat)
        else:
            raise ValueError(f"Transformation matrix at index {i} has unsupported shape {current_tmat.shape}. Expected (2,3) or (3,3).")

        out[tuple(sl_out)] = transform.warp(current_slice,
                                            tf.inverse, # Warp takes inverse transform
                                            order=order,
                                            mode='constant',
                                            cval=0.,
                                            preserve_range=True, # Important for quantitative data
                                            output_shape=current_slice.shape)

    return out


def recenter_im(im: np.ndarray) -> np.ndarray:
    """Recenters the content of a 2D image.

    This is achieved by finding the bounding box of non-zero pixels,
    cropping the image to this bounding box, and then padding the cropped
    image with zeros back to its original dimensions, effectively centering
    the content.

    Based on: https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil

    Parameters
    ----------
    im : np.ndarray
        The 2D input image (grayscale or single channel).

    Returns
    -------
    np.ndarray
        The recentered image, with the same dimensions and dtype as the input.
        Returns the original image if it contains no non-zero pixels.
    """
    if im.ndim != 2:
        warnings.warn("recenter_im expects a 2D image. Returning original for other dimensions.")
        return im

    # Find non-empty rows and columns
    # Check if any non-zero elements exist to avoid errors with empty images
    if not np.any(im): # If image is all zeros
        return im.copy()

    non_empty_rows = np.where(np.any(im, axis=1))[0]
    non_empty_columns = np.where(np.any(im, axis=0))[0]

    if len(non_empty_rows) == 0 or len(non_empty_columns) == 0 : # Should be caught by np.any(im)
        return im.copy() # Or handle as an empty content image

    y_min, y_max = non_empty_rows[0], non_empty_rows[-1]
    x_min, x_max = non_empty_columns[0], non_empty_columns[-1]

    cropped_im = im[y_min:y_max+1, x_min:x_max+1]

    original_h, original_w = im.shape

    # Use to_shape_center for robust centering padding
    # to_shape_center expects target_w, target_h
    im_padded = to_shape_center(cropped_im, original_w, original_h)

    return im_padded.astype(im.dtype) # Ensure original dtype is preserved


def downsamp(ref: np.ndarray, factor: int) -> np.ndarray:
    """Downsamples a 3-channel (RGB) image by an integer factor using block reduction
    (local averaging).

    The input image `ref` is assumed to have pixel values in the range [0, 255]
    and is normalized to [0, 1] before block reduction, then implicitly
    returned in the [0,1] float range by `block_reduce`.

    Parameters
    ----------
    ref : np.ndarray
        The input RGB image (H, W, C) to be downsampled. C should be 3.
    factor : int
        The integer downsampling factor. Both height and width will be
        reduced by this factor.

    Returns
    -------
    np.ndarray
        The downsampled RGB image as a float array, with values in [0,1] range.
        Shape will be (H/factor, W/factor, C).

    Raises
    ------
    ValueError
        If `ref` is not a 3-channel image.
    """
    if ref.ndim != 3 or ref.shape[2] != 3:
        raise ValueError("Input image `ref` must be a 3-channel (RGB) image.")

    # Normalize image from [0, 255] to [0, 1] for averaging
    # Assuming input is uint8 or float that needs normalization
    if ref.dtype == np.uint8:
        ds_array_normalized = ref.astype(np.float32) / 255.0
    elif np.max(ref) > 1.0: # If float but seems to be in 0-255 range
        ds_array_normalized = ref.astype(np.float32) / 255.0
    else: # Assume it's already float in 0-1 range
        ds_array_normalized = ref.astype(np.float32)

    r = block_reduce(ds_array_normalized[:, :, 0], (factor, factor), np.mean)
    g = block_reduce(ds_array_normalized[:, :, 1], (factor, factor), np.mean)
    b = block_reduce(ds_array_normalized[:, :, 2], (factor, factor), np.mean)
    ds_array_out = np.stack((r, g, b), axis=-1)
    return ds_array_out


def shape_colorise(dr: np.ndarray, ref: np.ndarray, cmap: plt.cm.colors.Colormap = plt.cm.inferno) -> np.ndarray:
    """Colorizes a 2D grayscale image, crops it to match a reference RGB image's
    spatial dimensions, pads it to those dimensions, and converts to uint8.

    The process is:
    1. Normalize `dr` and apply colormap `cmap` (resulting in RGBA).
    2. Take only RGB channels from the colormapped image.
    3. Crop this RGB image to the spatial dimensions of `ref` using `crop_down`.
    4. Pad the cropped image to ensure it exactly matches `ref` dimensions using
       `to_shape_rgb` (padding with white).
    5. Convert the final image to `np.uint8` scale [0, 255].

    Parameters
    ----------
    dr : np.ndarray
        The 2D grayscale input image to be colorized.
    ref : np.ndarray
        A 3D RGB reference image (H, W, 3) used for target shape.
    cmap : matplotlib.colors.Colormap, optional
        The Matplotlib colormap to apply. Defaults to `plt.cm.inferno`.

    Returns
    -------
    np.ndarray
        The processed image: colorized, shaped like `ref`, and of type `np.uint8`.
    """
    if dr.ndim != 2:
        raise ValueError("Input image `dr` must be 2D (grayscale).")
    if ref.ndim != 3 or ref.shape[2] != 3:
        raise ValueError("Reference image `ref` must be 3D RGB.")

    # Normalize dr for colormapping (vmin=0, vmax=max(dr) or specified range)
    norm = plt.Normalize(vmin=np.min(dr), vmax=np.max(dr)) # Use min/max of dr for full range
    # Apply colormap, get RGB channels (ignore alpha if cmap returns RGBA)
    colorized_dr = cmap(norm(dr))[:, :, :3]

    target_h, target_w, _ = ref.shape

    # Crop the colorized_dr to be no larger than ref dimensions initially
    # This step assumes colorized_dr might be larger than ref.
    # If it's smaller, crop_down would raise error. We need to handle both cases.
    # A better approach might be to pad first if smaller, then crop if larger.
    # Or, directly create target canvas and place/crop.

    # For robust shaping:
    # 1. Create a canvas of ref's shape (filled with white, as to_shape_rgb would do for padding)
    # 2. Place the (potentially smaller) colorized_dr onto this canvas, then crop if it was larger.

    # Let's assume colorized_dr needs to be resized/rescaled to ref's dimensions first.
    # The original sequence was: crop_down, then to_shape_rgb.
    # crop_down assumes overlay is larger. to_shape_rgb assumes input is smaller or equal.
    # This implies a specific workflow.

    # If colorized_dr is larger than ref, crop it down.
    if colorized_dr.shape[0] > target_h or colorized_dr.shape[1] > target_w:
         # Calculate how much to crop from each side
        h_diff = colorized_dr.shape[0] - target_h
        w_diff = colorized_dr.shape[1] - target_w

        crop_top = h_diff // 2
        crop_bottom = h_diff - crop_top
        crop_left = w_diff // 2
        crop_right = w_diff - crop_left

        # Ensure no negative crop if one dim is smaller but other is larger
        crop_top = max(0, crop_top)
        crop_bottom = max(0, crop_bottom)
        crop_left = max(0, crop_left)
        crop_right = max(0, crop_right)

        colorized_dr_cropped = colorized_dr[crop_top : colorized_dr.shape[0]-crop_bottom,
                                            crop_left : colorized_dr.shape[1]-crop_right, :]
    else:
        colorized_dr_cropped = colorized_dr

    # Now, ensure it's padded up to the exact ref dimensions if it was smaller or became smaller after crop.
    # to_shape_rgb will handle this.
    final_dr_shaped = to_shape_rgb(colorized_dr_cropped, target_w, target_h, pad_color_rgb=(1,1,1)) # Pad with white in [0,1] scale

    # Convert to uint8: cmap output is [0,1] float, to_shape_rgb preserves dtype or pads with [0,1] float white.
    final_dr_uint8 = (final_dr_shaped * 255).astype(np.uint8)

    return final_dr_uint8


def do_transform(mov: np.ndarray, fac: float, deg: float, tf: transform.ProjectiveTransform | transform.AffineTransform) -> np.ndarray:
    """Applies a sequence of transformations: rescale, rotate, and then a
    warp based on a given transformation object.

    The output is rounded and clipped to ensure non-negative integer values,
    suitable for image data.

    Parameters
    ----------
    mov : np.ndarray
        The input image (2D NumPy array) to be transformed.
    fac : float
        Rescaling factor applied by `skimage.transform.rescale`.
    deg : float
        Rotation angle in degrees, applied by `skimage.transform.rotate`.
    tf : skimage.transform.ProjectiveTransform | skimage.transform.AffineTransform
        A scikit-image transformation object (e.g., `AffineTransform`,
        `ProjectiveTransform`) used by `skimage.transform.warp`.

    Returns
    -------
    np.ndarray
        The transformed image, with values rounded and clipped at 0.
        The dtype may change due to operations (e.g. to float then rounded).
    """
    out = transform.rescale(mov, fac, anti_aliasing=True, preserve_range=True) # Added options
    out = transform.rotate(out, deg, resize=False, preserve_range=True) # Added options
    out = transform.warp(out, tf, preserve_range=True) # Added option
    # Rounding and clipping can result in information loss.
    # Consider if returning float and letting user handle is better.
    out = np.round(out.clip(min=0))
    return out.astype(mov.dtype if np.issubdtype(mov.dtype, np.integer) else out.dtype) # Preserve int if original


def rescale_tmat(tmat: np.ndarray, s: float) -> np.ndarray:
    """Rescales the translation components (last column) of an affine
    transformation matrix.

    This is often used when the image associated with the matrix has been
    rescaled, and the translation needs to be adjusted accordingly.
    Assumes `tmat` is a 2x3 or 3x3 matrix where the translation components
    are `tmat[0,2]` and `tmat[1,2]`.

    Parameters
    ----------
    tmat : np.ndarray
        A 2x3 or 3x3 affine transformation matrix.
        Example: [[R_xx, R_xy, T_x], [R_yx, R_yy, T_y], [0, 0, 1 (optional)]]
    s : float
        The scaling factor to apply to the translation components (T_x, T_y).

    Returns
    -------
    np.ndarray
        The transformation matrix with its translation components scaled by `s`.
        The input `tmat` is modified in-place and also returned.
    """
    if tmat.shape not in [(2,3), (3,3)]:
        raise ValueError("Transformation matrix must be 2x3 or 3x3.")
    tmat[0, 2] = tmat[0, 2] * s
    tmat[1, 2] = tmat[1, 2] * s
    return tmat


def tmat_3to2(tmat: np.ndarray) -> np.ndarray:
    """Converts a 3D affine transformation matrix (e.g., from BigWarp in Fiji,
    often 4x4 or using 3D coordinates) into a 2D affine matrix (3x3)
    suitable for 2D image transformations in scikit-image.

    The conversion extracts the top-left 2x2 rotation/scaling/shear part,
    and the translation components for x and y from the last column (or row,
    depending on convention). The specific indexing `tmat[:2, -1]` for translation
    and `tmat[2, :3]` for the last row of the 2D matrix implies a certain
    structure for the input `tmat` (likely a 3xN or similar where relevant parts
    are in these slices).

    Parameters
    ----------
    tmat : np.ndarray
        The input 3D transformation matrix. Common shapes from tools like Fiji
        might be 3x4 or 4x4. The indexing suggests it expects at least 3 rows
        and columns. For a typical 3x4 matrix:
        [[Rxx, Rxy, Rxz, Tx],
         [Ryx, Ryy, Ryz, Ty],
         [Rzx, Rzy, Rzz, Tz]]
        This function would take the 2D part as:
        [[Rxx, Rxy, Tx],
         [Ryx, Ryy, Ty],
         [Rzx, Rzy, Rzz]] (if tmat[2,:3] is [P_x, P_y, P_w] for perspective)
        Or more standardly for affine 2D from a 3D affine (ignoring Z):
        [[Rxx, Rxy, Tx],
         [Ryx, Ryy, Ty],
         [  0,   0,  1]]

    Returns
    -------
    np.ndarray
        A 3x3 2D affine transformation matrix.
    """
    tmat_2d = np.zeros((3, 3))
    # Rotation/Scaling/Shear part
    tmat_2d[:2, :2] = tmat[:2, :2]
    # Translation part
    tmat_2d[:2, 2] = tmat[:2, -1] # Assumes translation is in the last column
    # Perspective/Bottom row (original was tmat[2, :3])
    # For a standard 2D affine matrix, this should be [0, 0, 1]
    # If tmat is from a 3D affine (e.g. 4x4 or 3x4), tmat[2,:3] might be [P_x, P_y, P_w] or similar.
    # For pure 2D affine, we set it to [0,0,1].
    # The original tmat[2, :3] was likely specific to a particular 3D->2D projection.
    # For general 2D work from a 3D matrix where Z is ignored:
    tmat_2d[2, :] = [0, 0, 1] # Standard 2D affine last row
    # If the input tmat was indeed a specific 3x3 projection matrix already,
    # then the original tmat_2d[2,:] = tmat[2,:3] might be correct for that context.
    # Given it's "from BigWarp", this might be specific.
    # Let's stick to creating a standard 2D affine from the 2D components.
    return tmat_2d


def do_transform_noscale(mov: np.ndarray, ref: np.ndarray, deg: float, tf: transform.ProjectiveTransform | transform.AffineTransform) -> np.ndarray:
    """Applies rotation and a warp transformation (e.g., affine) to an image,
    then recenters and crops it to match a reference image's shape.

    This version uses nearest-neighbor interpolation (`order=0`) and preserves
    the intensity range for transformations, making it suitable for quantitative
    images or masks where intensity values should not be altered by interpolation.
    No rescaling is performed.

    Parameters
    ----------
    mov : np.ndarray
        The 2D input image to be transformed.
    ref : np.ndarray
        The 2D reference image used for final cropping to shape.
    deg : float
        Rotation angle in degrees.
    tf : skimage.transform.ProjectiveTransform | skimage.transform.AffineTransform
        A scikit-image transformation object for the warp operation.

    Returns
    -------
    np.ndarray
        The transformed, recentered, and cropped image.
    """
    out = transform.rotate(mov, deg, order=0, preserve_range=True, resize=False)
    out = transform.warp(out, tf, order=0, preserve_range=True, output_shape=out.shape) # Ensure output shape is maintained
    out = out.clip(min=0) # Ensure non-negative values
    out = recenter_im(out)
    out = crop_to(out, ref.shape[1], ref.shape[0]) # target_w, target_h
    return out


def do_transform_noPSR(mov: np.ndarray, ref: np.ndarray, deg: float) -> np.ndarray:
    """Applies only coarse rotation to an image, then recenters and crops it
    to match a reference image's shape.

    "PSR" likely refers to PyStackReg, implying this function avoids more
    complex registration methods from that library, using only rotation.
    Uses nearest-neighbor interpolation and preserves intensity range.

    Parameters
    ----------
    mov : np.ndarray
        The 2D input image to be transformed.
    ref : np.ndarray
        The 2D reference image used for final cropping.
    deg : float
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray
        The rotated, recentered, and cropped image.
    """
    # just coarse rotation and housekeeping
    out = transform.rotate(mov, deg, order=0, preserve_range=True, resize=False)
    out = out.clip(min=0) # Ensure non-negative values
    out = recenter_im(out)
    out = crop_to(out, ref.shape[1], ref.shape[0]) # target_w, target_h
    return out


def plot_compare(mov: np.ndarray, ref: np.ndarray, lab1: str | None = None, lab2: str | None = None, cmap1: str = 'inferno', cmap2: str | bool = False, axis: str = 'off') -> None:
    """Plots two images side-by-side for visual comparison.

    Parameters
    ----------
    mov : np.ndarray
        The first image (e.g., moving or processed image).
    ref : np.ndarray
        The second image (e.g., reference or original image).
    lab1 : str | None, optional
        Title for the subplot displaying `mov`. Defaults to None.
    lab2 : str | None, optional
        Title for the subplot displaying `ref`. Defaults to None.
    cmap1 : str, optional
        Colormap for `mov`. Defaults to 'inferno'.
    cmap2 : str | bool, optional
        Colormap for `ref`. If False (default), uses Matplotlib's default
        colormap (usually 'viridis' or as configured). If a string, uses that colormap.
    axis : str, optional
        Argument for `plt.axis()` for both subplots (e.g., 'on', 'off', 'equal').
        Defaults to 'off'.

    Returns
    -------
    None
        Displays a Matplotlib figure.
    """
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(mov, cmap=cmap1)
    if isinstance(cmap2, str): # Check if cmap2 is a string (colormap name)
        ax[1].imshow(ref, cmap=cmap2)
    else: # cmap2 is False or other, use default
        ax[1].imshow(ref)
    ax[0].axis(axis)
    ax[1].axis(axis)
    ax[0].set_title(lab1)
    ax[1].set_title(lab2)
    plt.tight_layout()
    plt.show()
    plt.close(f) # Close the figure object


def norm_im(im: np.ndarray) -> np.ndarray:
    """Normalizes an image to the 0-255 range and converts it to `np.uint8`.

    Normalization is performed by scaling the image based on its max value:
    `norm_im = (255 * im) / max(im)`. If `max(im)` is 0, it returns a zero image
    of the same shape and `np.uint8` type.

    Parameters
    ----------
    im : np.ndarray
        Input image (grayscale or color). If color, normalization is global.

    Returns
    -------
    np.ndarray
        The normalized image as `np.uint8`.
    """
    max_val = np.max(im)
    if max_val == 0:
        return np.zeros_like(im, dtype=np.uint8)
    return (255 * im / max_val).astype(np.uint8)


def colorise_out(im: np.ndarray, cmap: str = 'inferno') -> np.ndarray:
    """Applies a colormap to a normalized version of a grayscale image and
    converts it to an RGB uint8 image.

    The process involves:
    1. Normalizing the input image `im` to the [0, 255] uint8 range using `norm_im`.
    2. Applying the specified colormap `cmap` to this normalized image.
       The colormap application typically results in an RGBA image (float, 0-1 range).
    3. Taking only the RGB channels ([: , :, :-1]).
    4. Normalizing this RGB image again to [0, 255] uint8 using `norm_im`.

    Parameters
    ----------
    im : np.ndarray
        Input 2D grayscale image.
    cmap : str, optional
        Name of the Matplotlib colormap to apply. Defaults to 'inferno'.

    Returns
    -------
    np.ndarray
        The colorized RGB image as `np.uint8` (H, W, 3).
    """
    if im.ndim != 2:
        warnings.warn("Input `im` for colorise_out is expected to be 2D grayscale. Result may be unexpected.")

    # First normalization to uint8 [0,255] as input for colormap application
    normalized_im_uint8 = norm_im(im)

    # Apply colormap
    colormap_func = plt.get_cmap(cmap)
    # Colormap typically returns RGBA float [0,1]
    colored_im_float_rgba = colormap_func(normalized_im_uint8)

    # Take RGB, discard Alpha
    colored_im_float_rgb = colored_im_float_rgba[:, :, :3]

    # Second normalization to convert float [0,1] RGB to uint8 [0,255] RGB
    # norm_im expects input that might not be in [0,1] range, so this is effectively scaling by max.
    # If colored_im_float_rgb is already [0,1], then norm_im scales it to [0,255].
    final_rgb_uint8 = norm_im(colored_im_float_rgb)

    return final_rgb_uint8


def myround(x: float | np.ndarray, base: int = 5) -> float | np.ndarray:
    """Rounds a number `x` (or array of numbers) down to the nearest multiple of `base`.

    Example: `myround(12, 5)` returns `10.0`.
    `myround(np.array([10, 12, 18]), 5)` returns `array([10., 10., 15.])`.

    Parameters
    ----------
    x : float | np.ndarray
        Number or NumPy array of numbers to round.
    base : int, optional
        The base to round down to. Defaults to 5.

    Returns
    -------
    float | np.ndarray
        The rounded number(s) as float or array of floats.
    """
    return base * np.floor(np.asanyarray(x)/base)


def bin_bin(binary: np.ndarray, he_um_px: float, iq_um_px: float, method: str = 'ndarray', op: str = 'sum') -> np.ndarray:
    """Re-bins a binary image (presumably from H&E resolution) to iQID resolution.

    This function calculates target dimensions based on the H&E and iQID pixel
    sizes, crops the input binary image to be an integer multiple of the
    iQID pixel dimensions (in terms of H&E pixels), and then re-bins it.

    Parameters
    ----------
    binary : np.ndarray
        The input 2D binary image (e.g., from H&E scale). Expected to be 0s and 1s,
        or 0s and 255s which will be scaled if `method='ndarray'`.
    he_um_px : float
        Pixel size of the H&E image in microns per pixel.
    iq_um_px : float
        Pixel size of the target iQID image in microns per pixel.
    method : str, optional
        Re-binning method:
        - 'ndarray': Uses `iqid.helper.bin_ndarray` with the specified `op`.
                     Input `binary` image is divided by 255 before binning if max > 1.
        - 'cv2_nn': Uses `cv2.resize` with nearest-neighbor interpolation.
        Defaults to 'ndarray'.
    op : str, optional
        Operation for `iqid.helper.bin_ndarray` if `method='ndarray'`
        (e.g., 'sum', 'mean'). Defaults to 'sum'.

    Returns
    -------
    np.ndarray
        The re-binned image at the target iQID resolution.
    """
    # operation: sum or mean
    # how many bins are in the whole image?
    y_he, x_he = binary.shape  # HE pixels

    # Total physical dimensions of the HE image in microns
    x_um_total = he_um_px * x_he
    y_um_total = he_um_px * y_he

    # Target dimensions in terms of iQID pixels
    x_iq_dim = int(x_um_total // iq_um_px)
    y_iq_dim = int(y_um_total // iq_um_px)

    if x_iq_dim == 0 or y_iq_dim == 0:
        warnings.warn("Target iQID dimensions are zero. Check pixel sizes and input image size.")
        return np.array([])


    # Number of HE pixels that correspond to an integer number of iQID pixels
    # This is the size the HE image needs to be cropped to before binning.
    x_he_for_cropping = int(np.floor(x_iq_dim * iq_um_px / he_um_px))
    y_he_for_cropping = int(np.floor(y_iq_dim * iq_um_px / he_um_px))

    # The myround function here seems to be used to make xtt/ytt multiples of xdim/ydim.
    # This might be redundant if x_he_for_cropping/y_he_for_cropping are calculated correctly
    # to be divisible by the binning factor (iq_um_px / he_um_px).
    # However, the original intention might be to ensure the cropped HE image size (xtt, ytt)
    # is such that when binned by a factor (iq_um_px / he_um_px), it results in exactly (x_iq_dim, y_iq_dim).
    # Let's simplify crop target to x_he_for_cropping, y_he_for_cropping.
    # The crop_to function expects target width then height.
    cropped = crop_to(binary, x_he_for_cropping, y_he_for_cropping)

    if method == 'ndarray':
        # If binary image is 0s and 255s, scale to 0s and 1s for sum/mean
        img_to_bin = cropped
        if np.max(cropped) > 1 and (op == 'mean' or op == 'sum'): # Assume it's a 0-255 mask
            img_to_bin = cropped / 255.0
        h = helper.bin_ndarray(img_to_bin, (y_iq_dim, x_iq_dim), operation=op)
    elif method == 'cv2_nn':
        # cv2.resize expects (width, height) for dsize
        h = cv2.resize(cropped, (x_iq_dim, y_iq_dim), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'ndarray' or 'cv2_nn'.")
    return h


def bin_centroids(fileName: str, imsize: tuple[int, int], he_um_px: float, iq_um_px: float, minA: float = 1) -> np.ndarray:
    """Reads centroid coordinates and areas from a CSV file, filters them by
    minimum area, scales coordinates from H&E to iQID resolution, and
    generates a 2D histogram (binned image) of these centroids.

    Parameters
    ----------
    fileName : str
        Path to the CSV file. Expected columns are Area, X, Y (in that order,
        after skipping one header row). Column indices used are 1 (Area),
        2 (X), and 3 (Y).
    imsize : tuple[int, int]
        Tuple `(height, width)` of the original H&E image space in pixels.
        Used to calculate total physical dimensions.
    he_um_px : float
        Pixel size of the H&E image in microns per pixel.
    iq_um_px : float
        Pixel size of the target iQID image in microns per pixel.
    minA : float, optional
        Minimum area threshold for filtering centroids. Centroids with area
        less than or equal to this value (if A is column 1) are excluded.
        Defaults to 1.

    Returns
    -------
    np.ndarray
        A 2D histogram representing the binned/counted centroids at iQID resolution.
        The values in the histogram are counts of centroids per iQID pixel.
    """
    # Columns are: 0=?, 1=Area, 2=X_centroid, 3=Y_centroid
    data = np.genfromtxt(fileName, delimiter=',', skip_header=1, usecols=(1, 2, 3))
    area = data[:, 0]
    X_he = data[:, 1] # X coordinates in HE pixel units
    Y_he = data[:, 2] # Y coordinates in HE pixel units

    # Filter by minimum area
    valid_indices = area > minA
    X_he_filt = X_he[valid_indices]
    Y_he_filt = Y_he[valid_indices]

    # Conversion factor from HE pixel units to iQID pixel units
    # If 1 iQID pixel = 50um, and 1 HE pixel = 0.5um, then fac = 50/0.5 = 100 HE px per iQID px.
    # So, X_iq = X_he / fac.
    scale_factor_he_to_iqid = iq_um_px / he_um_px

    # Original image total dimensions in microns
    h_he_px, w_he_px = imsize
    x_um_total = he_um_px * w_he_px
    y_um_total = he_um_px * h_he_px

    # Target dimensions in iQID pixels
    x_iq_dim_total = int(x_um_total // iq_um_px)
    y_iq_dim_total = int(y_um_total // iq_um_px)

    # Scale HE centroid coordinates to iQID pixel coordinates
    xc_iq = X_he_filt / scale_factor_he_to_iqid
    yc_iq = Y_he_filt / scale_factor_he_to_iqid

    # Define edges for 2D histogram in iQID pixel units
    # Bins should cover the range [0, x_iq_dim_total] and [0, y_iq_dim_total]
    xedges = np.arange(x_iq_dim_total + 1)
    yedges = np.arange(y_iq_dim_total + 1)

    # Create 2D histogram
    # hist2d expects x first, then y.
    h, _, _ = np.histogram2d(xc_iq, yc_iq, bins=[xedges, yedges])

    # np.histogram2d returns (counts, xedges, yedges). Transpose because hist2d(x,y)
    # means x are columns and y are rows, but imshow(H) expects H[row,col].
    return h.T
