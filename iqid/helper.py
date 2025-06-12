import os
import re
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
# Import datetime explicitly for type hinting if not already present globally
import datetime


from tqdm import trange


"""
This module provides various helper functions for the iQID processing pipeline.

It includes utilities for:
- File system operations: Listing studies and substudies, organizing directories.
- String manipulation: Natural sorting of file names.
- Numerical calculations: Mean/standard deviation, percentage error.
- User interaction: Yes/no prompts.
- Date/time utilities: Calculating time differences.
- Array manipulations: Checking for monotonicity, binning/resampling arrays,
  decomposing affine matrices.
- Plotting utilities: Setting plot parameters, drawing lines and scalebars,
  creating colorbars, and various comparison plots for images and histograms.
"""


def list_studies(rootdir: str) -> list[str]:
    """Gets a list of directory paths for all folders directly within the
    specified `rootdir`.

    Parameters
    ----------
    rootdir : str
        The path to the root directory to scan.

    Returns
    -------
    list[str]
        A list of full directory paths for folders in `rootdir`.
    """
    study_list = [f.path for f in os.scandir(rootdir) if f.is_dir()]
    return study_list # Removed extra parentheses


def list_substudies(rootdir: str) -> list[str]:
    """Gets a list of directory paths for all subfolders one level down
    from the `rootdir` (i.e., subfolders of folders within `rootdir`).

    Parameters
    ----------
    rootdir : str
        The path to the root directory.

    Returns
    -------
    list[str]
        A list of full directory paths for subfolders.
    """
    study_list = list_studies(rootdir)
    substudy_list = []
    for study in study_list:
        substudies = list_studies(study)
        substudy_list = substudy_list + substudies # Can use extend for efficiency
    return substudy_list # Removed extra parentheses


def organize_dirs(rootdir: str, study_list: list[str], copy_files: bool = True, copy_meta: bool = True) -> None:
    """Organizes a set of subdirectories by creating a new directory structure
    under `../analysis/` relative to `rootdir`.

    It then copies files (e.g., `.dat` listmode files) and metadata files
    (e.g., `Acquisition_Info*.txt`) into appropriately named new subdirectories.
    The new directory names are constructed from parts of the original directory paths.

    Parameters
    ----------
    rootdir : str
        The root directory from which the analysis structure will be created
        (one level up, in a sibling 'analysis' directory).
    study_list : list[str]
        List of identified directory paths containing the original data.
        Each path is expected to have a structure like
        `.../dayname/seqname/name/Listmode/*.dat`.
    copy_files : bool, optional
        If True (default), copies the primary data files (expects one `.dat`
        file in a 'Listmode' subdirectory).
    copy_meta : bool, optional
        If True (default), copies metadata files (expects one `Acquisition_Info*.txt`
        file in the study directory).

    Returns
    -------
    None
        This function modifies the file system by creating directories and
        copying files.

    Notes
    -----
    - This function uses `shutil.copy2` for copying files.
    - It relies on `string_underscores` (a helper function, potentially deprecated
      or local, not shown in this snippet) to create new directory names.
    - Assumes a specific input directory structure to correctly parse `dayname`,
      `seqname`, and `name`.
    - Error handling for `glob.glob(...)[0]` (if no files are found) is not present.
    """
    import shutil # Moved import inside function as it's specific here

    analysis_base_dir = Path(rootdir).parent / 'analysis'
    analysis_base_dir.mkdir(parents=True, exist_ok=True)

    for i in trange(len(study_list), desc="Organizing directories"):
        study_path = Path(study_list[i])
        try:
            # Assumes 'Listmode/*.dat' exists and is unique
            filename = glob.glob(os.path.join(study_path, 'Listmode', '*.dat'))[0]
            name = study_path.name
            seqname = study_path.parent.name.split()[0] # Takes first part if space in seqname
            dayname = study_path.parent.parent.name

            # Assuming string_underscores is available or to be replaced
            # For now, let's define a simple version if it's missing for robustness
            try:
                newname_parts = string_underscores([dayname, seqname, name])
            except NameError: # string_underscores not defined
                newname_parts = f"{dayname}_{seqname}_{name}"


            newdir = analysis_base_dir / newname_parts
            newdir.mkdir(parents=True, exist_ok=True)

            if copy_files:
                shutil.copy2(filename, newdir)
            if copy_meta:
                # Assumes 'Acquisition_Info*.txt' exists and is unique
                metaname = glob.glob(os.path.join(study_path, 'Acquisition_Info*.txt'))[0]
                shutil.copy2(metaname, newdir)
        except IndexError:
            print(f"Warning: Could not find expected files in {study_path}. Skipping this entry.")
        except Exception as e:
            print(f"Warning: An error occurred while processing {study_path}: {e}. Skipping.")

# two helper functions for natural sorting of file names
# from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

def atoi(text: str) -> int | str:
    """Helper function for natural sort order."""
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> list[int | str]:
    """Generates a list of string and number parts from a given text string
    for use as a key in natural sorting.

    E.g., "image10.tif" becomes `['image', 10, '.tif']`.
    From: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

    Parameters
    ----------
    text : str
        The string to be split into parts for natural sorting.

    Returns
    -------
    list[int | str]
        A list where numbers are converted to integers and text parts remain strings.
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def natural_sort(l: list[str]) -> list[str]:
    """Sorts a list of strings in natural order (e.g., "item2" comes before "item10").
    Modifies the list in-place and also returns it.

    Parameters
    ----------
    l : list[str]
        The list of strings to sort.

    Returns
    -------
    list[str]
        The naturally sorted list (same object as input `l`).
    """
    l.sort(key=natural_keys)
    return l # Removed extra parentheses


def mean_stdev(array: np.ndarray, return_vals: bool = False) -> tuple[float, float] | str:
    """Prints the mean and standard deviation of a NumPy array, formatted to
    two significant figures (actually two decimal places in current format).
    Optionally returns these values.

    Parameters
    ----------
    array : np.ndarray
        The input NumPy array of numerical values.
    return_vals : bool, optional
        If True, returns the mean and standard deviation as a tuple.
        If False (default), prints them and returns an empty string.

    Returns
    -------
    tuple[float, float] | str
        If `return_vals` is True, returns `(mean, std_dev)`.
        Otherwise, returns an empty string after printing.
    """
    mean_val = np.mean(array)
    std_val = np.std(array)
    print(f'{mean_val:.2f} +/- {std_val:.2f}') # Using f-string for clarity
    if return_vals:
        return mean_val, std_val
    else:
        return ''


def pct_err(ref: float | np.ndarray, exp: float | np.ndarray) -> float | np.ndarray:
    """Calculates the percentage error between a reference value and an
    experimental value.

    Formula: `100 * (reference - experimental) / reference`.

    Parameters
    ----------
    ref : float | np.ndarray
        The reference value(s).
    exp : float | np.ndarray
        The experimental value(s).

    Returns
    -------
    float | np.ndarray
        The percentage error. Returns `np.inf` or `np.nan` if `ref` is zero.
    """
    # Ensure safe division, handle ref == 0 if necessary
    with np.errstate(divide='ignore', invalid='ignore'):
        error = 100 * (np.asanyarray(ref) - np.asanyarray(exp)) / np.asanyarray(ref)
    return error


def get_yn(prompt_string: str) -> bool | None:
    """Prompts the user with `prompt_string` and converts a yes/no, y/n
    answer to a boolean value.

    Parameters
    ----------
    prompt_string : str
        The question or prompt to display to the user. This function will
        append " (yes/no): " to it.

    Returns
    -------
    bool | None
        True for "yes" or "y" (case-insensitive).
        False for "no" or "n" (case-insensitive).
        Prints an error message and returns None for invalid input.
    """
    try:
        user_input = input(f"{prompt_string} (yes/no): ")
        return {"yes": True, "no": False, "y": True, "n": False}[user_input.lower()]
    except KeyError:
        print("Invalid input, please enter yes or no!")
        return None # Explicitly return None for invalid input


def get_dt(t1: datetime.datetime, t2: datetime.datetime, grain: str = 'us', verbose: bool = False) -> float | None:
    """Calculates the absolute time difference between two `datetime.datetime` objects.

    Parameters
    ----------
    t1 : datetime.datetime
        The first datetime object.
    t2 : datetime.datetime
        The second datetime object.
    grain : str, optional
        The temporal resolution of the result. Must be 'us' (microseconds)
        or 's' (total seconds). Defaults to 'us'.
    verbose : bool, optional
        This parameter is present but not used in the function. Defaults to False.

    Returns
    -------
    float | None
        The absolute time difference in the specified `grain`.
        Returns None if `grain` is invalid (after printing an error message).
    """
    # Calculate absolute difference robustly
    if t1 > t2:
        dt_timedelta = t1 - t2
    else:
        dt_timedelta = t2 - t1

    if grain == 'us':
        # total_seconds() provides full resolution, then convert to microseconds
        dt_val = dt_timedelta.total_seconds() * 1e6
    elif grain == 's':
        dt_val = dt_timedelta.total_seconds()
    else:
        print("Invalid 'grain' specified. Choose 's' or 'us'.")
        return None
    return dt_val


def check_mono(arr: np.ndarray) -> bool:
    """Checks if a NumPy array is monotonically increasing.

    A monotonically increasing array is one where for all i <= j, arr[i] <= arr[j].

    Parameters
    ----------
    arr : np.ndarray
        The input NumPy array. Should be 1D.

    Returns
    -------
    bool
        True if the array is monotonically increasing, False otherwise.
    """
    if arr.ndim != 1:
        warnings.warn("check_mono expects a 1D array. Result may be unexpected for >1D.")
    return np.all(arr[:-1] <= arr[1:])

# ---------------------------- ARRAYS AND LINALG ---------------------------- #


def bin_ndarray(ndarray: np.ndarray, new_shape: tuple[int, ...], operation: str = 'sum') -> np.ndarray:
    """Bins/resamples a NumPy ndarray to a `new_shape` by summing or averaging
    values in the bins.

    The dimensions of `new_shape` must be factors of the original dimensions
    (i.e., `ndarray.shape[i]` must be divisible by `new_shape[i]`).
    Adapted from J.F. Sebastian:
    https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array

    Parameters
    ----------
    ndarray : np.ndarray
        The input NumPy array to be binned.
    new_shape : tuple[int, ...]
        The target shape for the binned array. The length of this tuple must
        match `ndarray.ndim`. Each element `new_shape[i]` must be a divisor
        of `ndarray.shape[i]`.
    operation : str, optional
        The operation to perform when combining elements in a bin:
        - 'sum': Sum the elements in each bin.
        - 'mean': Calculate the mean of elements in each bin.
        Defaults to 'sum'.

    Returns
    -------
    np.ndarray
        The binned/resampled array with shape `new_shape`.

    Raises
    ------
    ValueError
        If `operation` is not 'sum' or 'mean'.
        If `ndarray.ndim` does not match `len(new_shape)`.
        If any dimension in `new_shape` is not a divisor of the corresponding
        dimension in `ndarray.shape` (implicitly through reshape error).

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    operation = operation.lower()
    if operation not in ['sum', 'mean']:
        raise ValueError("Operation not supported. Choose 'sum' or 'mean'.")
    if ndarray.ndim != len(new_shape):
        raise ValueError(
            f"Shape mismatch: input ndim {ndarray.ndim} -> new_shape ndim {len(new_shape)}")

    # Check if new_shape dimensions are factors of ndarray.shape dimensions
    for old_dim, new_dim in zip(ndarray.shape, new_shape):
        if old_dim % new_dim != 0:
            raise ValueError(
                f"New shape dimension {new_dim} is not a factor of old shape dimension {old_dim}.")

    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                 ndarray.shape)]
    flattened = [x for p in compression_pairs for x in p]
    binned_array = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation == 'sum':
            binned_array = np.sum(binned_array, axis=-1*(i+1))
        elif operation == 'mean':
            binned_array = np.mean(binned_array, axis=-1*(i+1))
    return binned_array # Renamed to avoid confusion with input `ndarray`


def decompose_affine(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decomposes a 2D affine transformation matrix (3x3) into its
    constituent parts: translation, rotation, scaling/zoom, and shear.

    This implementation is based on the method described in:
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/affines.py
    Specifically, it handles a 3x3 matrix where the affine part is 2x2.

    Parameters
    ----------
    A : np.ndarray
        A 3x3 affine transformation matrix. The last row is expected to be [0, 0, 1].
        The top-left 2x2 submatrix represents rotation, scaling, and shear.
        The first two elements of the last column represent translation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple `(T, R, Z, S)` containing:
        - T : np.ndarray
            Translation vector `[tx, ty]` (2 elements).
        - R : np.ndarray
            Rotation matrix (2x2).
        - Z : np.ndarray
            Scaling/zoom factors `[sx, sy]` for x and y axes (2 elements).
        - S : np.ndarray
            Shear factor(s). For a 2D transformation from a 3x3 matrix, this is
            typically a single shear parameter (1 element array).

    Raises
    ------
    ValueError
        If the input matrix `A` is not 3x3.
    LinAlgError
        If the decomposition fails (e.g., due to a singular matrix).
    """
    A_arr = np.asarray(A)
    if A_arr.shape != (3,3):
        raise ValueError("Input matrix A must be a 3x3 array.")

    T = A_arr[:-1, -1]     # Translation components (tx, ty)
    RZS = A_arr[:-1, :-1]  # Rotation, Zoom, and Shear combined matrix (2x2 part)
    ZS = np.linalg.cholesky(np.dot(RZS.T, RZS)).T
    Z = np.diag(ZS).copy()
    shears = ZS / Z[:, np.newaxis]
    n = len(Z)
    S = shears[np.triu(np.ones((n, n)), 1).astype(bool)]
    R = np.dot(RZS, np.linalg.inv(ZS))
    if np.linalg.det(R) < 0:
        Z[0] *= -1
        ZS[0] *= -1
        R = np.dot(RZS, np.linalg.inv(ZS))
    return T, R, Z, S


# ---------------------------- PLOTTING FUNCTIONS ---------------------------- #


def set_plot_parms() -> plt.colors.Colormap: # Corrected Colormap import path from matplotlib
    """Sets preferred Matplotlib parameters for a consistent plotting style
    and returns the 'tab10' colormap.

    Modifies `plt.rcParams` with predefined settings for font, font size,
    line width, axes, grid, and legend.

    Parameters
    ----------
    None

    Returns
    -------
    matplotlib.colors.Colormap
        The 'tab10' colormap object.
    """
    pltmap = plt.get_cmap("tab10")

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 1

    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5

    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'k'

    return pltmap


def draw_lines(es: list[float], label: str | None = None, ls: str = '--', **kwargs) -> None:
    """Draws vertical dashed lines on the current Matplotlib plot axis at
    specified x-values (e.g., energies).

    If `label` is provided, an invisible line is plotted first to create a
    legend entry for the set of lines.

    Parameters
    ----------
    es : list[float]
        A list of x-values (e.g., energies in keV) at which to draw vertical lines.
    label : str | None, optional
        A label for the set of lines, used for creating a legend entry.
        If None, no legend entry is created for the group. Defaults to None.
    ls : str, optional
        Linestyle for the vertical lines. Defaults to '--' (dashed).
    **kwargs
        Additional keyword arguments passed to `plt.axvline` (for the lines)
        and `plt.plot` (for the legend entry). Examples: `color`, `linewidth`.

    Returns
    -------
    None
        Modifies the current Matplotlib plot axes.
    """
    # Plot an invisible line for the legend entry if a label is provided
    if label is not None:
        plt.plot([], [], label=label, ls=ls, **kwargs) # Empty x,y creates invisible line for legend

    for x_val in es: # Iterate directly over values
        plt.axvline(x_val, label=None, ls=ls, **kwargs) # Individual lines should not have labels


def add_scalebar(scalebarlen: float, ax: plt.Axes, px_mm: float = (2048.0/80.0), loc: str = 'lower right', **kwargs) -> None:
    """Adds a formatted and labeled scale bar to a Matplotlib axes object.

    Parameters
    ----------
    scalebarlen : float
        The physical length the scale bar should represent, in millimeters (mm).
    ax : matplotlib.axes.Axes
        The Matplotlib axes object to which the scale bar will be added.
    px_mm : float, optional
        The conversion factor: pixels per millimeter for the image associated
        with the `ax`. The default `2048.0/80.0 = 25.6` px/mm might correspond
        to a specific iQID setup (e.g., 80mm field of view over 2048 pixels).
    loc : str, optional
        The location of the scale bar on the axes. Valid Matplotlib location
        strings (e.g., 'upper right', 'lower left', 'center').
        Defaults to 'lower right'.
    **kwargs
        Additional keyword arguments passed to
        `mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`.
        Common arguments include `pad`, `borderpad`, `sep`, `frameon`, `color`.

    Returns
    -------
    None
        Modifies the provided `ax` object by adding a scale bar.
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    # The size of the scale bar in data coordinates (pixels)
    scalebar_px_len = scalebarlen * px_mm
    label_text = f'{scalebarlen} mm'

    scalebar = AnchoredSizeBar(ax.transData,
                               scalebar_px_len,
                               label_text,
                               loc=loc,
                               **kwargs)
    ax.add_artist(scalebar)


def nice_colorbar(mappable: plt.cm.ScalarMappable, cbar_location: str = "right", orientation: str = "vertical", pad: float = 0.05, **cbar_kwargs) -> plt.colorbar.Colorbar:
    """Produces a nicely formatted colorbar associated with a given `mappable`
    (e.g., an image plotted with `imshow`).

    This function creates a new axes for the colorbar, which can be useful for
    controlling its position and size relative to the main plot axes.

    Parameters
    ----------
    mappable : matplotlib.cm.ScalarMappable
        The Matplotlib ScalarMappable object (e.g., the object returned by
        `imshow` or `scatter`) to which the colorbar is linked.
    cbar_location : str, optional
        Position of the colorbar relative to the axes of the `mappable`.
        Valid options: "left", "right", "top", "bottom". Defaults to "right".
    orientation : str, optional
        Orientation of the colorbar, "vertical" or "horizontal".
        Defaults to "vertical".
    pad : float, optional
        Padding between the main axes and the new colorbar axes, as a fraction
        of the main axes size. Defaults to 0.05.
    **cbar_kwargs
        Additional keyword arguments passed to `fig.colorbar()`.
        Examples: `label`, `ticks`, `format`.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created Matplotlib Colorbar object.

    Notes
    -----
    The original docstring mentioned: "Only really works on single image,
    don't try with subplots." This is likely because `plt.gca()` might not
    refer to the intended axes if multiple subplots are active without explicit
    axes management. The function now uses `mappable.axes` directly.
    """
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(cbar_location, size="5%", pad=pad)
    # Pass additional cbar_kwargs to the colorbar call
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation, **cbar_kwargs)
    # plt.sca(last_axes) # This line is removed as it might cause side effects if not managed carefully.
                       # The caller should manage current axes if needed after calling this.
    return cbar


def compare_metrics(ref: np.ndarray, mov: np.ndarray) -> None:
    """Compares two images (`ref` and `mov`) using several metrics.

    Prints:
    - Mean percentage difference between `mov` and `ref`, relative to `ref`.
    - Maximum percentage difference between `mov` and `ref`, relative to `ref`.

    Displays:
    - A visual side-by-side comparison using `plot_compare(ref, mov)`.
    - A comparison of their pixel intensity histograms using `plot_hists(ref, mov)`.

    Parameters
    ----------
    ref : np.ndarray
        The reference image.
    mov : np.ndarray
        The "moving" or comparison image.

    Returns
    -------
    None
        Prints metrics and displays plots.
    """
    # Ensure safe division for percentage error calculation
    mean_ref = np.mean(ref)
    max_ref = np.max(ref)

    if mean_ref == 0:
        mean_diff_pct = np.inf if np.mean(mov) != 0 else 0.0
    else:
        mean_diff_pct = (np.mean(mov) - mean_ref) / mean_ref * 100

    if max_ref == 0:
        max_diff_pct = np.inf if np.max(mov) != 0 else 0.0
    else:
        max_diff_pct = (np.max(mov) - max_ref) / max_ref * 100

    print(f'Mean % err (compared to reference image): {mean_diff_pct:.2f} %')
    print(f'Max % err (compared to reference image): {max_diff_pct:.2f} %')
    
    # Call the second, more detailed plot_compare function
    plot_compare_detailed(ref, mov, lab1='Reference Image', lab2='Comparison Image')
    plot_hists(ref, mov, hist_label1='Reference Image', hist_label2='Comparison Image')


def plot_hists(slice1: np.ndarray, slice2: np.ndarray, hist_label1: str = 'Image 1', hist_label2: str = 'Image 2', xlabel: str = 'Pixel Intensity', ylabel: str = 'Frequency', bins: int = 100) -> None:
    """Compares the pixel intensity histograms of two image slices.

    Parameters
    ----------
    slice1 : np.ndarray
        The first image slice (2D NumPy array).
    slice2 : np.ndarray
        The second image slice (2D NumPy array).
    hist_label1 : str, optional
        Label for the histogram of `slice1`. Defaults to 'Image 1'.
    hist_label2 : str, optional
        Label for the histogram of `slice2`. Defaults to 'Image 2'.
    xlabel : str, optional
        Label for the x-axis. Defaults to 'Pixel Intensity'.
    ylabel : str, optional
        Label for the y-axis. Defaults to 'Frequency'.
    bins : int, optional
        Number of bins for the histograms. Defaults to 100.

    Returns
    -------
    None
        Displays a Matplotlib plot.
    """
    fig = plt.figure(figsize=(8, 4)) # Renamed f to fig for clarity
    # Use the min/max of both slices for a shared bin range for better comparison
    combined_min = min(np.min(slice1), np.min(slice2))
    combined_max = max(np.max(slice1), np.max(slice2))
    bin_edges = np.linspace(combined_min, combined_max, bins + 1)

    n1, bins1_ret = np.histogram(slice1.ravel(), density=True, bins=bin_edges)
    n2, bins2_ret = np.histogram(slice2.ravel(), density=True, bins=bin_edges)

    # plot steps at bin centers
    bin_centers1 = (bins1_ret[:-1] + bins1_ret[1:]) / 2
    bin_centers2 = (bins2_ret[:-1] + bins2_ret[1:]) / 2

    plt.step(bin_centers1, n1, alpha=0.75, label=hist_label1) # Adjusted alpha for visibility
    plt.step(bin_centers2, n2, alpha=0.75, label=hist_label2) # Adjusted alpha
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Pixel Intensity Histograms")
    plt.grid(True, linestyle=':', alpha=0.7) # Ensure grid is visible
    plt.show()
    plt.close(fig) # Close the figure object


# Assuming this is the plot_compare to keep, renamed for clarity during refactoring
def plot_compare_detailed(slice1: np.ndarray, slice2: np.ndarray, lab1: str | None = 'Image 1', lab2: str | None = 'Image 2', cmap: str = 'inferno', cbar_label: str = 'Intensity') -> None:
    """Displays two image slices side-by-side for visual comparison,
    with a shared colorbar. (This is the more detailed version).

    Parameters
    ----------
    slice1 : np.ndarray
        The first image slice (e.g., reference image).
    slice2 : np.ndarray
        The second image slice (e.g., comparison image).
    lab1 : str | None, optional
        Title for the subplot of `slice1`. Defaults to 'Image 1'.
    lab2 : str | None, optional
        Title for the subplot of `slice2`. Defaults to 'Image 2'.
    cmap : str, optional
        Colormap to use for displaying the images. Defaults to 'inferno'.
    cbar_label : str, optional
        Label for the colorbar. Defaults to 'Intensity'.

    Returns
    -------
    None
        Displays a Matplotlib plot.
    """
    from mpl_toolkits.axes_grid1 import ImageGrid # Keep import local if only used here
    from matplotlib.colorbar import Colorbar # Keep import local

    fig = plt.figure(figsize=(8, 4)) # Increased figure size slightly for better layout
    # Setup ImageGrid
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 2),
                     axes_pad=0.1, # Increased padding slightly
                     share_all=True,
                     cbar_mode="single", # Single colorbar for both images
                     cbar_location="right",
                     cbar_size="5%", # Adjusted for aesthetics
                     cbar_pad=0.1) # Adjusted for aesthetics

    # Determine global min/max for consistent color scaling
    vmin = min(np.min(slice1), np.min(slice2))
    vmax = max(np.max(slice1), np.max(slice2))

    # Display images
    im1 = grid[0].imshow(slice1, cmap=cmap, vmin=vmin, vmax=vmax)
    im2 = grid[1].imshow(slice2, cmap=cmap, vmin=vmin, vmax=vmax)

    # Configure axes
    grid[0].set_title(lab1)
    grid[0].axis('off')
    grid[1].set_title(lab2)
    grid[1].axis('off')

    # Configure colorbar
    # grid[1].cax.cla() # Not needed if cbar_mode="single" handled by ImageGrid properly
    cb = grid.cbar_axes[0].colorbar(im1) # Link colorbar to one of the images
    cb.set_label(cbar_label)

    plt.show()
    plt.close(fig) # Close the figure object


def plot_sequence(imstack: np.ndarray, nrows: int = 2, figsize: tuple[int, int] = (18, 10), balance_clim: bool = False) -> None:
    """Plots a sequence of images from an image stack in a grid layout.

    Each image in the stack is displayed in a subplot. Image indices are
    overlaid on each subplot.

    Parameters
    ----------
    imstack : np.ndarray
        A 3D NumPy array representing the image stack, where the first
        dimension is the number of images (num_images, height, width).
    nrows : int, optional
        Number of rows in the subplot grid. The number of columns will be
        calculated based on `len(imstack)` and `nrows`. Defaults to 2.
    figsize : tuple[int, int], optional
        Size of the Matplotlib figure. Defaults to (18, 10).
    balance_clim : bool, optional
        If True, all images are displayed with the same color limits (`vmax`
        set to the global maximum of `imstack`). This helps in visually
        comparing intensities across images. If False (default), each image
        is scaled independently. Original parameter name was `balance`.

    Returns
    -------
    None
        Displays a Matplotlib plot.
    """
    if imstack.ndim != 3:
        raise ValueError("imstack must be a 3D array (num_images, height, width).")
    if len(imstack) == 0:
        print("Input imstack is empty. Nothing to plot.")
        return

    ncols = int(np.ceil(len(imstack) / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize) # Renamed f, ax to fig, axes
    axes = axes.ravel() # Flatten in case of single row/col or multi-row/col

    vmax_global = np.max(imstack) if balance_clim else None

    for i in range(len(axes)): # Iterate up to number of axes created
        if i < len(imstack): # Check if there's an image for this subplot
            current_image = imstack[i]
            if balance_clim:
                axes[i].imshow(current_image, cmap='gray', vmax=vmax_global)
            else:
                axes[i].imshow(current_image, cmap='gray')

            axes[i].text(0.95, 0.05, str(i), fontsize=12, color='white', # Adjusted text properties
                       transform=axes[i].transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       bbox=dict(facecolor='black', alpha=0.5, pad=0.1)) # Added background for visibility
        axes[i].axis('off') # Turn off axis for all subplots, even empty ones

    plt.tight_layout()
    plt.show()
    plt.close(fig) # Close the figure object

# ---------------------------- DEPRECATED FUNCTIONS ---


# def plot_all(drstack, nrows=4):
#     f, ax = plt.subplots(nrows=nrows, ncols=int(
#         np.ceil(len(drstack)/nrows)), figsize=(12, 9))
#     ax = ax.ravel()

#     vma = np.max(drstack)

#     for i in range(len(ax)):
#         try:
#             ax[i].imshow(drstack[i], cmap='inferno', vmax=vma)
#             ax[i].axis('off')
#             ax[i].text(1, 0, i, fontsize=18, color='white', transform=ax[i].transAxes,
#                        horizontalalignment='right', verticalalignment='bottom')
#         except IndexError:
#             ax[i].axis('off')

#     # plt.tight_layout()
#     plt.subplots_adjust(wspace=-0.3, hspace=0.05)
#     plt.show()
#     plt.close()


# def subplot_label(text, index):
#     ax[index].text(0.5, 0.5, text, fontsize=16,
#                    color='white', horizontalalignment='center', verticalalignment='center', transform=ax[index].transAxes)
#     return (1)


# def string_underscores(string_list):
#     newstring = ''
#     for i in range(len(string_list)):
#         newstring = newstring + string_list[i] + '_'
#     newstring = newstring[:-1]  # remove last _
#     return (newstring)
