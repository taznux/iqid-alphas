"""
Automates the initial processing of iQID listmode data.

This script is configured primarily via `config.json`, which specifies input file paths,
output directories, and various processing parameters. The workflow involves:
1.  Loading raw listmode data using `iqid.process_object.ClusterData`.
2.  Extracting initial metadata (headers, dimensions).
3.  Applying filtering, corrections (e.g., for missed timestamps), and basic analysis.
4.  Generating spatial images from the processed listmode data.
5.  Optionally, performing temporal analysis and ROI extraction based on contours.
6.  Saving key processed data arrays (coordinates, frame numbers, timestamps, etc.)
    to a specified output directory.

Key inputs (from config.json):
- `file_name`: Path to the raw iQID listmode data file.
- Various processing parameters for `ClusterData` methods.

Key outputs:
- Processed NumPy arrays (e.g., `xC.npy`, `yC.npy`, `time_ms.npy`) saved in `output_dir`.

Logging of operations is performed to `automate_processing.log`.
"""
import os
import json
import logging
import numpy as np
from iqid.process_object import ClusterData

# Configure logging
# Consider making log filename and level configurable via config.json if not already handled by a central setup
logging.basicConfig(filename='automate_processing.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_and_process_listmode_data(file_name: str, c_area_thresh: int = 15, makedir: bool = False, ftype: str = 'processed_lm') -> tuple[ClusterData, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads iQID listmode data and extracts initial metadata.

    Uses `iqid.process_object.ClusterData` to initialize, load data, and
    parse metadata from the listmode file.

    Parameters
    ----------
    file_name : str
        Path to the iQID listmode data file.
    c_area_thresh : int, optional
        Cluster area threshold, passed to `ClusterData`. Defaults to 15.
    makedir : bool, optional
        Flag to create analysis subdirectory, passed to `ClusterData`.
        Defaults to False.
    ftype : str, optional
        File type of the listmode data, passed to `ClusterData`.
        Defaults to 'processed_lm'.

    Returns
    -------
    tuple[ClusterData, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - cluster_data_obj : ClusterData
            The initialized and partially processed ClusterData object.
        - time_ms : np.ndarray
            Timestamps in milliseconds.
        - cluster_area : np.ndarray
            Area of each cluster.
        - xC_global : np.ndarray
            Global X-coordinates of cluster centroids.
        - yC_global : np.ndarray
            Global Y-coordinates of cluster centroids.
        - frame_num : np.ndarray
            Frame numbers for each event.

    Raises
    ------
    Exception
        Propagates exceptions from `ClusterData` methods or file operations.
    """
    try:
        logging.info(f"Loading listmode data from: {file_name} with ftype: {ftype}")
        cluster_data_obj = ClusterData(file_name, c_area_thresh, makedir, ftype)
        cluster_data_obj.init_header() # Loads header info into the object
        data = cluster_data_obj.load_cluster_data() # Loads the actual listmode data
        # Parses the loaded data into specific attributes of cluster_data_obj
        time_ms, cluster_area, xC_global, yC_global, frame_num = cluster_data_obj.init_metadata(data)
        logging.info(f"Successfully loaded and initialized metadata for {file_name}.")
        return cluster_data_obj, time_ms, cluster_area, xC_global, yC_global, frame_num
    except Exception as e:
        logging.error(f"Failed to load and process listmode data for {file_name}: {e}", exc_info=True)
        raise

def filter_correct_analyze_data(cluster_data_obj: ClusterData, binfac: int, ROI_area_thresh: int, t_binsize: float, t_half: float) -> ClusterData:
    """Applies processing parameters, estimates missed timestamps, and calculates
    mean number of events per frame.

    Parameters
    ----------
    cluster_data_obj : ClusterData
        The ClusterData object to process.
    binfac : int
        Binning factor for image processing.
    ROI_area_thresh : int
        Area threshold for ROI detection.
    t_binsize : float
        Time bin size for temporal analysis (in seconds).
    t_half : float
        Half-life for decay correction (in seconds).

    Returns
    -------
    ClusterData
        The processed ClusterData object.

    Raises
    ------
    Exception
        Propagates exceptions from `ClusterData` methods.
    """
    try:
        logging.info("Setting processing parameters and performing initial analysis.")
        cluster_data_obj.set_process_params(binfac, ROI_area_thresh, t_binsize, t_half)
        cluster_data_obj.get_mean_n() # Calculates and stores mean events per frame
        cluster_data_obj.estimate_missed_timestamps() # Estimates timestamps for missed events
        logging.info("Successfully applied processing parameters and initial analysis.")
        return cluster_data_obj
    except Exception as e:
        logging.error(f"Failed to filter, correct, and analyze data: {e}", exc_info=True)
        raise

def generate_spatial_images(cluster_data_obj: ClusterData, subpx: int = 1) -> np.ndarray:
    """Generates a spatial image from the listmode data.

    Parameters
    ----------
    cluster_data_obj : ClusterData
        The ClusterData object containing processed listmode data.
    subpx : int, optional
        Subpixel factor for image reconstruction. Defaults to 1.

    Returns
    -------
    np.ndarray
        The reconstructed cluster image.

    Raises
    ------
    Exception
        Propagates exceptions from `image_from_listmode`.
    """
    try:
        logging.info(f"Generating spatial image with subpixel factor: {subpx}.")
        cluster_image = cluster_data_obj.image_from_listmode(subpx)
        logging.info("Successfully generated spatial image.")
        return cluster_image
    except Exception as e:
        logging.error(f"Failed to generate spatial images: {e}", exc_info=True)
        raise

def generate_temporal_information(cluster_data_obj: ClusterData, event_fx: float = 0.1, xlim: tuple = (0, None), ylim: tuple = (0, None)) -> np.ndarray:
    """Generates an image from listmode data, typically for temporal analysis,
    allowing for event fraction and spatial limits.

    This method calls `image_from_big_listmode` which also updates `cluster_data_obj.t_s`.

    Parameters
    ----------
    cluster_data_obj : ClusterData
        The ClusterData object.
    event_fx : float, optional
        Fraction of events to load for generating the image. Defaults to 0.1.
    xlim : tuple, optional
        X-axis limits (min_x, max_x) for event filtering. Defaults to (0, None).
    ylim : tuple, optional
        Y-axis limits (min_y, max_y) for event filtering. Defaults to (0, None).

    Returns
    -------
    np.ndarray
        The reconstructed image based on the specified parameters.
        The `cluster_data_obj` is updated with `xC`, `yC`, and `t_s` attributes.

    Raises
    ------
    Exception
        Propagates exceptions from `image_from_big_listmode`.
    """
    try:
        logging.info(f"Generating image for temporal analysis with event_fx: {event_fx}, xlim: {xlim}, ylim: {ylim}.")
        # This method updates self.xC, self.yC, self.t_s in cluster_data_obj
        temporal_image = cluster_data_obj.image_from_big_listmode(event_fx, xlim, ylim)
        logging.info("Successfully generated image and updated temporal information (t_s).")
        return temporal_image
    except Exception as e:
        logging.error(f"Failed to generate temporal information: {e}", exc_info=True)
        raise

def detect_contours_extract_ROIs(cluster_data_obj: ClusterData, im: np.ndarray, gauss: int = 15, thresh: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    """Detects contours in an image and extracts ROI bounding boxes.

    Parameters
    ----------
    cluster_data_obj : ClusterData
        The ClusterData object.
    im : np.ndarray
        The image (2D NumPy array) on which to perform contour detection.
    gauss : int, optional
        Gaussian blur kernel size for contour preparation. Defaults to 15.
    thresh : int, optional
        Threshold value for binarizing image before contour detection. Defaults to 0.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray]
        - contours : list[np.ndarray]
            A list of detected contours. Each contour is an array of points.
        - ROIs : np.ndarray
            A 2D array where each row is `[x, y, w, h]` for a bounding box.

    Raises
    ------
    Exception
        Propagates exceptions from contour detection methods.
    """
    try:
        logging.info(f"Detecting contours with gauss: {gauss}, thresh: {thresh}.")
        cluster_data_obj.set_contour_params(gauss, thresh)
        contours = cluster_data_obj.get_contours(im)
        # Note: get_ROIs typically uses self.contours set by setup_ROIs or get_contours.
        # If get_contours doesn't set self.contours, this might need adjustment.
        # Assuming get_contours populates self.contours or get_ROIs can use the returned contours.
        # For now, assuming get_contours implicitly sets self.contours for get_ROIs.
        # If not, one might need: cluster_data_obj.contours = contours
        ROIs = cluster_data_obj.get_ROIs() # Gets bounding boxes for self.contours
        logging.info(f"Successfully detected {len(contours)} contours and extracted ROIs.")
        return contours, ROIs
    except Exception as e:
        logging.error(f"Failed to detect contours and extract ROIs: {e}", exc_info=True)
        raise

def save_processed_data(cluster_data_obj: ClusterData, output_dir: str) -> None:
    """Saves key processed data arrays from the ClusterData object to .npy files.

    Saved arrays include: xC, yC, f (frame numbers), time_ms.
    If `ftype` is 'clusters', also saves raws, cim_sum, cim_px.

    Parameters
    ----------
    cluster_data_obj : ClusterData
        The ClusterData object containing data to be saved.
    output_dir : str
        The directory where data files will be saved. Created if it doesn't exist.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Propagates exceptions from file I/O operations.
    """
    try:
        logging.info(f"Saving processed data to directory: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save common attributes
        if hasattr(cluster_data_obj, 'xC') and cluster_data_obj.xC is not None:
            np.save(os.path.join(output_dir, 'xC.npy'), cluster_data_obj.xC)
        if hasattr(cluster_data_obj, 'yC') and cluster_data_obj.yC is not None:
            np.save(os.path.join(output_dir, 'yC.npy'), cluster_data_obj.yC)
        if hasattr(cluster_data_obj, 'f') and cluster_data_obj.f is not None:
            np.save(os.path.join(output_dir, 'f.npy'), cluster_data_obj.f)
        if hasattr(cluster_data_obj, 'time_ms') and cluster_data_obj.time_ms is not None:
            np.save(os.path.join(output_dir, 'time_ms.npy'), cluster_data_obj.time_ms)

        # Save attributes specific to 'clusters' ftype
        if cluster_data_obj.ftype == 'clusters':
            if hasattr(cluster_data_obj, 'raws') and cluster_data_obj.raws is not None:
                np.save(os.path.join(output_dir, 'raws.npy'), cluster_data_obj.raws)
            if hasattr(cluster_data_obj, 'cim_sum') and cluster_data_obj.cim_sum is not None:
                np.save(os.path.join(output_dir, 'cim_sum.npy'), cluster_data_obj.cim_sum)
            if hasattr(cluster_data_obj, 'cim_px') and cluster_data_obj.cim_px is not None:
                np.save(os.path.join(output_dir, 'cim_px.npy'), cluster_data_obj.cim_px)
        logging.info("Successfully saved processed data.")
    except Exception as e:
        logging.error(f"Failed to save processed data: {e}", exc_info=True)
        raise

def main():
    """
    Main function to automate the processing of iQID listmode data.

    Workflow:
    1. Loads configuration from `config.json`.
    2. Calls `load_and_process_listmode_data` to load data and get initial metadata.
    3. Calls `filter_correct_analyze_data` to apply processing settings and perform
       initial analysis steps (mean events, missed timestamp estimation).
    4. Calls `generate_spatial_images` to create a representative image from listmode.
    5. Calls `generate_temporal_information` to potentially create another image and
       importantly to populate `cluster_data.t_s` (event timestamps in seconds).
    6. Calls `detect_contours_extract_ROIs` to find ROIs in one of the generated images.
    7. Calls `save_processed_data` to save key arrays for further analysis.

    All parameters for these steps are sourced from the `config.json` file under
    the "automate_processing" section.
    Logs errors and progress to `automate_processing.log`.
    """
    logging.info("Starting main processing workflow.")
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)['automate_processing'] # Get specific section

        file_name = config['file_name']
        output_dir = config['automate_processing']['output_dir']
        c_area_thresh = config['automate_processing']['c_area_thresh']
        makedir = config['automate_processing']['makedir']
        ftype = config['automate_processing']['ftype']
        binfac = config['automate_processing']['binfac']
        ROI_area_thresh = config['automate_processing']['ROI_area_thresh']
        t_binsize = config['automate_processing']['t_binsize']
        t_half = config['automate_processing']['t_half']
        subpx = config['automate_processing']['subpx']
        event_fx = config['automate_processing']['event_fx']
        xlim = config['automate_processing']['xlim']
        ylim = config['automate_processing']['ylim']
        gauss = config['automate_processing']['gauss']
        thresh = config['automate_processing']['thresh']

        cluster_data, time_ms, cluster_area, xC_global, yC_global, frame_num = load_and_process_listmode_data(file_name, c_area_thresh, makedir, ftype)
        cluster_data = filter_correct_analyze_data(cluster_data, binfac, ROI_area_thresh, t_binsize, t_half)
        cluster_image = generate_spatial_images(cluster_data, subpx)
        temporal_image = generate_temporal_information(cluster_data, event_fx, xlim, ylim)
        contours, ROIs = detect_contours_extract_ROIs(cluster_data, cluster_image, gauss, thresh)
        save_processed_data(cluster_data, output_dir)
    except Exception as e:
        logging.error("Failed to complete main processing: %s", str(e))
        raise

if __name__ == "__main__":
    main()
