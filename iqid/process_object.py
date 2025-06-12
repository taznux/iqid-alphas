"""
This module defines classes and functions for importing, processing, and
analyzing listmode data from the iQID camera. It focuses on the `ClusterData`
class, which provides a comprehensive suite of tools for handling various data
formats, performing image reconstruction, ROI analysis, and temporal fitting.
"""
# Code for importing and processing the various forms of listmode data from the iQID camera
# It's not the cleanest code, but please contact Robin Peter if you need any assistance or find errors

import os
import numpy as np
import cv2
import glob
from scipy.optimize import curve_fit
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import Normalize

from tqdm import tqdm, trange

import ipywidgets as widgets
import functools

from pathlib import Path

from iqid import helper


def exponential(x, a, thalf):
    """Defines an exponential decay function.

    Parameters
    ----------
    x : float | np.ndarray
        The independent variable (e.g., time).
    a : float
        The initial amplitude or activity.
    thalf : float
        The half-life of the decay.

    Returns
    -------
    float | np.ndarray
        The value of the exponential function at x.
    """
    return a*np.exp(-np.log(2)/thalf*x)


class ClusterData:
    """
    This class is designed to load, process, and analyze iQID camera data.
    It handles different listmode file formats (`processed_lm`, `offset_lm`,
    `clusters`), extracts metadata, reconstructs images, allows for ROI
    definition and analysis, and performs temporal fitting of activity data.
    """

    def __init__(self, file_name, c_area_thresh=15, makedir=False, ftype='processed_lm'):
        """Initializes a ClusterData object.

        Parameters
        ----------
        file_name : str
            Path to the iQID data file.
        c_area_thresh : int, optional
            Cluster area threshold. Defaults to 15.
        makedir : bool, optional
            If True, creates an analysis subdirectory. Defaults to False.
        ftype : str, optional
            File type of the input data. Must be one of "processed_lm",
            "offset_lm", or "clusters". Defaults to 'processed_lm'.

        Instance Attributes Initialized
        -------------------------------
        file_name : str
            Stores `file_name`.
        ftype : str
            Stores `ftype`.
        savedir : str
            Path to a directory for saving analysis results, derived from `file_name`.
        c_area_thresh : int
            Stores `c_area_thresh`.

        Raises
        ------
        TypeError
            If `ftype` is not one of the recognized values.
        """
        self.file_name = file_name
        self.ftype = ftype
        base = os.path.basename(os.path.normpath(self.file_name))
        self.savedir = os.path.join(
            os.path.dirname(self.file_name), base[:-4] + '_Analysis')
        if ftype not in ["processed_lm", "offset_lm", "clusters"]:
            raise (
                TypeError, "File format not specified: processed_lm, offset_lm, clusters")
        # self.NUM_NAN_DATA = 50
        self.c_area_thresh = c_area_thresh

        if makedir:
            Path(self.savedir).mkdir(parents=True, exist_ok=True)

    def init_header(self):
        """Reads and parses the header of the iQID data file to extract metadata.

        Parameters
        ----------
        None.

        Returns
        -------
        tuple[int, int, int]
            A tuple containing (HEADER_SIZE, XDIM, YDIM).

        Side Effects
        ------------
        Populates the following instance attributes:
        self.HEADER_SIZE : int
            Size of the header in bytes (from file, value read is number of int32 words, so multiplied by 4 for bytes implicitly by how it's used later).
        self.XDIM : int
            X dimension of the detector/image.
        self.YDIM : int
            Y dimension of the detector/image.
        self.NUM_DATA_ELEMENTS : int
            Number of data elements per cluster/event, depends on self.ftype.
        If self.ftype == 'clusters':
            self.cluster_radius : int
                Radius of the cluster images.
            self.cluster_imsize : int
                Dimension of the square cluster images (2 * radius + 1).

        Raises
        ------
        TypeError
            If self.ftype is not recognized.
        """
        HEADER = np.fromfile(self.file_name, dtype=np.int32, count=100)
        self.HEADER_SIZE = HEADER[0] # This is number of int32 words
        self.XDIM = HEADER[1]
        self.YDIM = HEADER[2]

        if self.ftype == 'processed_lm':
            self.NUM_DATA_ELEMENTS = 14
        elif self.ftype == 'offset_lm':
            self.NUM_DATA_ELEMENTS = 6
        elif self.ftype == 'clusters':
            # "Cropped Listmode File" is the other name for cluster image file type.
            self.cluster_radius = HEADER[20]
            self.cluster_imsize = 2 * self.cluster_radius + 1  # e.g. 10-px radius = 21x21 clusters
            self.NUM_DATA_ELEMENTS = 8 + 2 * (self.cluster_imsize**2)
        else:
            # This was raising a tuple, which is not standard. Changed to raise TypeError directly.
            raise TypeError("File format not specified: processed_lm, offset_lm, clusters")

        return (self.HEADER_SIZE, self.XDIM, self.YDIM)

    def set_process_params(self, binfac, ROI_area_thresh, t_binsize, t_half):
        """Sets parameters for subsequent processing steps.

        Parameters
        ----------
        binfac : int
            Binning factor for image processing.
        ROI_area_thresh : int
            Area threshold for ROI detection.
        t_binsize : float | int
            Time bin size for temporal analysis (in seconds).
        t_half : float | int
            Half-life for decay correction (in seconds).

        Returns
        -------
        tuple
            An empty tuple.

        Side Effects
        ------------
        Populates instance attributes:
        self.binfac
        self.ROI_area_thresh
        self.t_binsize
        self.t_half
        """
        self.binfac = binfac
        self.ROI_area_thresh = ROI_area_thresh
        self.t_binsize = t_binsize
        self.t_half = t_half
        return ()

    def load_cluster_data(self, event_fx=1, dtype=np.float64):
        """Loads cluster data from the file specified in self.file_name.

        Parameters
        ----------
        event_fx : float, optional
            Fraction of total events/clusters to load. Defaults to 1 (load all).
        dtype : np.dtype, optional
            NumPy data type to load data as. Defaults to np.float64.

        Returns
        -------
        np.ndarray
            A 2D NumPy array where columns are individual events/clusters and
            rows are data elements. Shape: (self.NUM_DATA_ELEMENTS, NUM_LOAD).

        Side Effects
        ------------
        Calls self.init_header() implicitly if not already called by a previous
        method, to ensure self.HEADER_SIZE and self.NUM_DATA_ELEMENTS are set.
        """
        self.init_header()
        file_size_bytes = os.path.getsize(self.file_name)

        if dtype == np.float32 or dtype == np.int32:
            byteSize = 4
            byteFac = 1
        else:
            byteSize = 8
            # if loading whole thing as f64 (8), header will only take up 50=100//2 values instead of 100
            byteFac = 2
        NUM_CLUSTERS = np.floor(
            (file_size_bytes - 4*self.HEADER_SIZE) / (byteSize*self.NUM_DATA_ELEMENTS))

        # for very large data, you may only want to load the first 10% , e.g.
        NUM_LOAD = int(event_fx * NUM_CLUSTERS)

        unshaped_data = np.fromfile(
            self.file_name, dtype=dtype, count=self.HEADER_SIZE // byteFac + int(NUM_LOAD*self.NUM_DATA_ELEMENTS))
        data = unshaped_data[self.HEADER_SIZE // byteFac:].reshape(
            int(self.NUM_DATA_ELEMENTS), int(NUM_LOAD), order='F')
        return (data)

    def load_raws(self, cluster_size=10):
        """Loads raw listmode data from an associated "*Cropped_Raw_Listmode.dat"
        file, which is expected to be in the same directory as self.file_name.

        Parameters
        ----------
        cluster_size : int, optional
            This parameter seems unused in the current implementation of the
            method but might be intended for future use or is a remnant.
            Defaults to 10.

        Returns
        -------
        np.ndarray
            A 2D NumPy array containing the raw listmode data, similar in
            structure to load_cluster_data output but loaded as np.int32.
            Shape: (self.NUM_DATA_ELEMENTS, NUM_CLUSTERS).

        Raises
        ------
        Exception
            If the associated raw listmode file is not found.

        Side Effects
        ------------
        Uses self.HEADER_SIZE and self.NUM_DATA_ELEMENTS (which should be
        initialized by init_header() typically called via load_cluster_data()
        or directly).
        """

        # find associated raw listmode file in folder if there is one
        # ensure that there is only one in the directory, or it could grab the wrong one
        rlistmode = glob.glob(os.path.join(
            self.file_name, '..', '*Cropped_Raw_Listmode.dat'))

        if not rlistmode:
            raise Exception(
                "No cropped raw listmode file found in directory. " +
                "Check that it is located in the same location: \n{}".format(self.file_name))
        else:
            rlistmode = rlistmode[0]

        byteSize = 4  # INTs rather than DOUBLEs
        file_size_bytes = os.path.getsize(rlistmode)

        # in all cases, HEADER SIZE *should* be the same between cprlm and crlm
        NUM_CLUSTERS = np.floor(
            (file_size_bytes - 4*self.HEADER_SIZE) / (byteSize*self.NUM_DATA_ELEMENTS))

        unshaped_data = np.fromfile(
            rlistmode, dtype=np.int32, count=self.HEADER_SIZE + int(NUM_CLUSTERS*self.NUM_DATA_ELEMENTS))

        data = unshaped_data[self.HEADER_SIZE:].reshape(
            int(self.NUM_DATA_ELEMENTS), int(NUM_CLUSTERS), order='F')

        return data

    def init_metadata(self, data):
        """Parses the loaded raw data array into meaningful instance attributes
        based on self.ftype.

        Parameters
        ----------
        data : np.ndarray
            The raw data array loaded by `load_cluster_data` or `load_raws`.
            This is a 2D array with shape (NUM_DATA_ELEMENTS, NUM_EVENTS).

        Returns
        -------
        tuple
            A tuple of NumPy arrays containing key metadata. The content of the
            tuple varies based on self.ftype:
            - 'processed_lm': (time_ms, cluster_area, xC_global, yC_global, frame_num)
            - 'offset_lm': (frame_num, time_ms, n_miss, n_cluster, pix, cam_temp_10K)
            - 'clusters': (frame_num, time_ms, xC, yC, raw_imgs, cim_sum, cim_px)

        Side Effects
        ------------
        Populates many instance attributes depending on self.ftype.
        Common attributes:
            self.f : np.ndarray
                Frame numbers.
            self.time_ms : np.ndarray
                Timestamps in milliseconds.
        For 'processed_lm':
            self.cluster_area : np.ndarray
            self.xC : np.ndarray
                Global x-centroids.
            self.yC : np.ndarray
                Global y-centroids.
            self.miss : None
            self.nevents_per_frame : None
            self.offset_frame_time : None
        For 'offset_lm':
            self.miss : np.ndarray
                Number of missed events.
            self.nevents_per_frame : np.ndarray
                Number of clusters per frame.
            self.offset_frame_time : np.ndarray
                Timestamp of each frame from offset file.
        For 'clusters':
            self.xC : np.ndarray
                Local x-centroids within cropped cluster image.
            self.yC : np.ndarray
                Local y-centroids within cropped cluster image.
            self.cim_sum : np.ndarray
                Sum of pixel values in the filtered cluster image.
            self.cim_px : np.ndarray
                Number of pixels (area) in the cluster.
            self.raws : np.ndarray
                Raw cluster images (reshaped to N x imsize*imsize).
        """
        # parses the loaded data into relevant arrays
        # data formats for each file type are provided in the iQID header info.
        # offset_lm: previously "listmode_frames"

        if self.ftype == 'processed_lm':
            frame_num = data[0, :]
            time_ms = data[1, :]
            sum_cluster_signal = data[2, :]
            cluster_area = data[3, :]
            yC_global = data[4, :]
            xC_global = data[5, :]
            var_y = data[6, :]
            var_x = data[7, :]
            covar_xy = data[8, :]
            eccentricity = data[9, :]
            skew_y = data[10, :]
            skew_x = data[11, :]
            kurt_y = data[12, :]
            kurt_x = data[13, :]

            self.cluster_area = cluster_area
            self.xC = xC_global
            self.yC = yC_global
            self.f = frame_num
            self.time_ms = time_ms

            # initialize the offset variables with None
            self.miss = None
            self.nevents_per_frame = None
            self.offset_frame_time = None

            return time_ms, cluster_area, xC_global, yC_global, frame_num
            # in the future, update this to be the same order as offset lm

        elif self.ftype == 'offset_lm':
            frame_num = data[0, :]
            time_ms = data[1, :]
            n_miss = data[2, :]
            n_cluster = data[3, :]
            pix = data[4, :]
            cam_temp_10K = data[5, :]

            self.f = frame_num
            self.time_ms = time_ms

            self.miss = n_miss
            self.nevents_per_frame = n_cluster
            self.offset_frame_time = time_ms
            return frame_num, time_ms, n_miss, n_cluster, pix, cam_temp_10K

        elif self.ftype == 'clusters':
            a = self.cluster_imsize

            raw_imgs = data[:a**2, :].T
            # fil_imgs = data[a**2: 2*a**2, :] # filtered/binarized version of image

            frame_num = data[2*a**2, :]  # Frame number (size INT)
            # yC (row) centroid coordinate (size INT) of the cropped cluster
            yC = data[2*a**2 + 1, :]
            # xC (column) centroid coordinate (size INT) of the cropped cluster
            xC = data[2*a**2 + 2, :]
            # Sum of the filtered cluster signal within the cropped sub-image (size INT)
            cim_sum = data[2*a**2 + 5, :]
            # Time elapsed since start of acquisition (size Unsigned INT)
            time_ms = data[2*a**2 + 6, :]
            # Number of pixels (area) in the cluster (size INT)
            cim_px = data[2*a**2 + 7, :]

            self.xC = xC
            self.yC = yC
            self.f = frame_num
            self.cim_sum = cim_sum
            self.cim_px = cim_px
            self.raws = raw_imgs
            self.time_ms = time_ms

            return frame_num, time_ms, xC, yC, raw_imgs, cim_sum, cim_px

        else:
            print('Accepted types: process_lm, offset_lm, clusters')

    def image_from_xy(self, x, y):
        """Creates a 2D histogram (image) from x and y coordinates.

        Parameters
        ----------
        x : np.ndarray
            Array of x-coordinates.
        y : np.ndarray
            Array of y-coordinates.

        Returns
        -------
        np.ndarray
            A 2D NumPy array representing the image, with dimensions
            (self.YDIM, self.XDIM). Pixel values are counts of events.

        Side Effects
        ------------
        Requires self.XDIM and self.YDIM to be initialized (e.g., by init_header).
        Rounds the input x and y coordinates.
        """
        # original spatial image
        x = np.round(x)
        y = np.round(y)

        cim = np.zeros((int(self.YDIM), int(self.XDIM)))
        for i in trange(len(x), desc='Building image...'):
            cim[y[i].astype(int), x[i].astype(int)] += 1
        return cim

    def image_from_listmode(self, subpx=1):
        """Reconstructs a spatial image from listmode data, applying an area
        threshold and optional subpixel positioning.

        This method loads the full dataset (unless previously filtered) and
        processes it to generate an image.

        Parameters
        ----------
        subpx : int, optional
            Subpixel factor. If > 1, coordinates are scaled by this factor
            before flooring to effectively increase image resolution.
            Defaults to 1 (no subpixel positioning).

        Returns
        -------
        np.ndarray
            The reconstructed cluster image. Dimensions will be
            (subpx * self.YDIM, subpx * self.XDIM).

        Side Effects
        ------------
        - Calls self.init_header(), self.load_cluster_data(), and
          self.init_metadata(). This means existing data in self.xC, self.yC,
          etc., might be overwritten.
        - Updates self.xC, self.yC, and self.t_s with the filtered,
          (optionally) subpixelated coordinates and corresponding timestamps
          (in seconds) of events that passed the cluster area threshold and
          are included in the generated image.
        - Filters events based on self.c_area_thresh.
        """
        self.init_header()
        data = self.load_cluster_data()
        time_ms, _, xC, yC, _ = self.init_metadata(data)

        xC_filtered = xC[np.logical_and(
            xC > 0, self.cluster_area > self.c_area_thresh)]
        yC_filtered = yC[np.logical_and(
            yC > 0, self.cluster_area > self.c_area_thresh)]

        # apply subpixelization if desired
        if subpx == 1:
            xC_px_rounded = np.round(xC_filtered, 0)
            yC_px_rounded = np.round(yC_filtered, 0)
        else:
            xC_px_rounded = np.floor(xC_filtered * subpx)
            yC_px_rounded = np.floor(yC_filtered * subpx)

        # logical "and" with four statements to get positive coordinates only
        cluster_bool = ((xC_px_rounded > 0) * (yC_px_rounded > 0)
                        * np.isfinite(xC_px_rounded) * np.isfinite(yC_px_rounded))
        xC_good = xC_px_rounded[cluster_bool].astype(int)
        yC_good = yC_px_rounded[cluster_bool].astype(int)

        # build spatial image (no temporal information)
        cluster_image = np.zeros((int(subpx*self.YDIM), int(subpx*self.XDIM)))
        for i in range(len(yC_good)):
            cluster_image[yC_good[i], xC_good[i]] += 1

        # sort into arrays and get associated temporal information
        time_s = time_ms[self.cluster_area > self.c_area_thresh]/1e3
        time_s = time_s[cluster_bool]

        # the listmode coordinates that are actually used in the cluster image
        self.xC = xC_good
        self.yC = yC_good
        self.t_s = time_s

        return (cluster_image)

    def image_from_big_listmode(self, event_fx=0.1, xlim=(0, None), ylim=(0, None)):
        """Efficiently reconstructs an image from a potentially large listmode
        dataset, allowing for loading a fraction of events and spatial cropping.
        This method only supports subpx=1 (no subpixel positioning).

        Parameters
        ----------
        event_fx : float, optional
            Fraction of total events to load. Defaults to 0.1.
        xlim : tuple[int | float, int | float | None], optional
            (min_x, max_x) crop limits for the x-coordinates. `None` for max_x
            means self.XDIM. Defaults to (0, None). Coordinates are rounded.
        ylim : tuple[int | float, int | float | None], optional
            (min_y, max_y) crop limits for the y-coordinates. `None` for max_y
            means self.YDIM. Defaults to (0, None). Coordinates are rounded.

        Returns
        -------
        np.ndarray
            The reconstructed cluster image of dimensions (self.YDIM, self.XDIM).

        Side Effects
        ------------
        - Calls self.init_header(), self.load_cluster_data(event_fx=event_fx),
          and self.init_metadata().
        - Updates self.xC, self.yC, and self.t_s with the filtered and
          cropped coordinates (rounded) and corresponding timestamps (in seconds)
          of events used in the image.
        - Filters events based on self.c_area_thresh and the provided xlim/ylim.

        Raises
        ------
        ValueError
            If xlim or ylim exceed acquisition dimensions (self.XDIM, self.YDIM).
        """
        # only subpx=1
        self.init_header()

        xlim = np.array(xlim)
        ylim = np.array(ylim)

        if xlim[1] is None:
            xlim[1] = self.XDIM
        if ylim[1] is None:
            ylim[1] = self.YDIM

        if xlim[1] > self.XDIM:
            raise ValueError(
                'X-limit ({}, {}) exceeds dimensions of acquisition ({})'.format(xlim[0], xlim[1], self.XDIM))
        if ylim[1] > self.YDIM:
            raise ValueError(
                'Y-limit ({}, {}) exceeds dimensions of acquisition ({})'.format(ylim[0], ylim[1], self.YDIM))

        data = self.load_cluster_data(event_fx=event_fx)
        time_ms, _, xC, yC, _ = self.init_metadata(data)

        # more efficient version of image_from_listmode for only subpx=1
        cluster_bool = ((xC > xlim[0]) * (yC > ylim[0])
                        * (xC < xlim[1]) * (yC < ylim[1])
                        * np.isfinite(xC) * np.isfinite(yC)
                        * (self.cluster_area > self.c_area_thresh))
        xC = np.round(xC[cluster_bool]).astype(int)
        yC = np.round(yC[cluster_bool]).astype(int)
        t_s = time_ms[cluster_bool] / 1e3

        # build spatial image (no temporal information)
        cluster_image = np.zeros((self.YDIM, self.XDIM))
        for i in range(len(yC)):
            cluster_image[yC[i], xC[i]] += 1

        self.xC = xC
        self.yC = yC
        self.t_s = t_s

        return (cluster_image)

    def apply_selection(self, selection_bool):
        """Filters instance attributes self.xC, self.yC, and self.f in place
        based on a boolean array.

        Parameters
        ----------
        selection_bool : np.ndarray[bool]
            Boolean array of the same length as self.xC, self.yC, self.f.
            True values indicate elements to keep.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The filtered self.xC, self.yC, self.f.

        Side Effects
        ------------
        Modifies self.xC, self.yC, self.f in place.
        Prints an error message if any of the attributes (xC, yC, f) do not
        exist or if the selection causes an error, but attempts to continue.
        """
        try:
            self.xC = self.xC[selection_bool]
            self.yC = self.yC[selection_bool]
            self.f = self.f[selection_bool]
        except Exception as e:
            print(e)  # do this catch so that the in-place doesn't happen if any fail
        return (self.xC, self.yC, self.f)

    def get_subset(self, selection_bool):
        """Creates a new Subset object containing a subset of the data based
        on a boolean selection array.

        This method does not modify the current ClusterData object in place.

        Parameters
        ----------
        selection_bool : np.ndarray[bool]
            Boolean array used to filter the data. True values indicate
            elements to include in the subset. This array should be applicable
            to self.xC, self.yC, self.f, self.time_ms, and potentially
            clusters-specific attributes.

        Returns
        -------
        Subset
            A new Subset object initialized with the filtered data. The Subset
            object will have its own copies of the selected data arrays.

        Side Effects
        ------------
        - Initializes a new Subset object.
        - Populates attributes of the Subset object (e.g., HEADER_SIZE, XDIM,
          YDIM, xC, yC, f, time_ms).
        - If self.ftype is 'clusters', it also copies subsets of self.raws,
          self.cim_sum, and self.cim_px to the new Subset object.
        - Requires relevant attributes (e.g., self.xC, self.time_ms) to be
          populated in the parent ClusterData object.
        """
        # basically, not doing it in place.
        # makes a subset class with each partial dataset.
        # might need caution for big data...
        subset_data = Subset(self.file_name,
                             c_area_thresh=self.c_area_thresh,
                             makedir=False,
                             ftype=self.ftype)
        subset_data.HEADER_SIZE = self.HEADER_SIZE
        subset_data.XDIM = self.XDIM
        subset_data.YDIM = self.YDIM

        subset_data.xC = self.xC[selection_bool]
        subset_data.yC = self.yC[selection_bool]
        subset_data.f = self.f[selection_bool]
        subset_data.time_ms = self.time_ms[selection_bool]
        if self.ftype == 'clusters':
            subset_data.raws = self.raws[selection_bool]
            subset_data.cim_sum = self.cim_sum[selection_bool]
            subset_data.cim_px = self.cim_px[selection_bool]

        return subset_data

    def get_mean_n(self, vis=False):
        """Calculates and optionally visualizes the mean number of events per frame
        using data from an 'offset_lm' file.

        This method relies on `self.nevents_per_frame` which is populated
        by `init_metadata` when `ftype` is 'offset_lm'.

        Parameters
        ----------
        vis : bool, optional
            If True, displays a histogram of events per frame along with the mean.
            Defaults to False.

        Returns
        -------
        float | None
            The mean number of events per frame. Returns None and prints a
            message if offset data (self.nevents_per_frame) is not loaded.

        Side Effects
        ------------
        Sets `self.mean_n` to the calculated mean number of events per frame.
        If `vis` is True, a matplotlib plot is shown.
        """
        # finds and sets mean number of events per frame
        # requires load of the offset file
        if self.nevents_per_frame is not None:
            n = self.nevents_per_frame
            self.mean_n = np.mean(n)
            if vis:
                hist_N, _, _ = plt.hist(
                    n, bins=np.arange(12), edgecolor='white')
                plt.xlabel('number of events per frame')
                plt.ylabel('number of frames')
                plt.axvline(np.mean(n), color='gray', ls='--')
                plt.title('mean events/frame = {:.2f}'.format(np.mean(n)))
                plt.show()

            return self.mean_n
        else:
            print('Offset file not loaded. TODO implement fallback using LM file.')
            return None

    def estimate_missed_timestamps(self):
        """Estimates timestamps for missed events using data from an 'offset_lm' file.

        This method is intended to provide an approximation for missed event times,
        useful for correcting time histograms for activity quantification, but
        should not be used for rigorous timing coincidence analysis. It relies on
        `self.miss` (number of missed events) and `self.offset_frame_time`
        (timestamps from offset file), populated by `init_metadata` when
        `ftype` is 'offset_lm'.

        Parameters
        ----------
        None.

        Returns
        -------
        np.ndarray | None
            An array of estimated timestamps (in milliseconds) for missed events.
            Each missed event is assigned the timestamp of the frame *before*
            it was noted as missed. Returns None and prints a message if
            offset data (self.miss or self.offset_frame_time) is not loaded.
        """
        # estimate the approximate timestamp of missed frames using offset file
        # this allows for correction in the time histogram for activity quantification
        # do not use this for rigorous timing coincidence
        if self.miss is not None and self.offset_frame_time is not None:
            m = self.miss
            t = self.offset_frame_time

            num_new_missed = np.diff(m)
            missed_events_time = np.array([])
            for i in range(len(num_new_missed)):
                if num_new_missed[i] > 0:
                    missed_events_time = np.append(missed_events_time,
                                                   np.repeat(t[i], num_new_missed[i]))
            return missed_events_time
        else:
            print('Offset file not loaded. TODO implement fallback using LM file.')
            return None

    def filter_singles(self, fmax, vis=False):
        """Filters the event data (self.xC, self.yC, self.f) in place to keep
        only events from frames that originally contained a single event.

        This method identifies frames with only one event based on self.f,
        then filters self.xC, self.yC, and self.f to include only those events.
        It also performs a basic validity check on coordinates (positive and finite).
        Note: This filter does NOT apply the self.c_area_thresh.

        Parameters
        ----------
        fmax : int
            The maximum frame number in the acquisition. This is used for
            calculating percentages if visualization is enabled.
        vis : bool, optional
            If True, displays a histogram of the number of events per frame
            before filtering, along with statistics about single-event frames
            and empty frames. Defaults to False.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The filtered self.xC, self.yC, self.f arrays.

        Side Effects
        ------------
        Modifies self.xC, self.yC, and self.f in place.
        If `vis` is True, a matplotlib plot is shown.
        Requires self.f, self.xC, self.yC to be populated.
        """
        vals, N = np.unique(self.f, return_counts=True)
        single_fnums = vals[N == 1]

        focc = len(vals)  # occupied frames
        fempty = fmax - focc

        if vis:
            pct_singles = len(single_fnums) / fmax * 100
            hist_N, _, _ = plt.hist(N, bins=np.arange(12), edgecolor='white')
            plt.xlabel('number of events per frame')
            plt.ylabel('number of frames')
            plt.title('{:.1f} % of frames are single-event,\n{} ({:.1f}%) are empty'.format(pct_singles,
                                                                                            fempty, fempty/fmax * 100))  # much improved singles rate
            plt.show()

        # NOTE:: DOES NOT INCLUDE CLUSTER AREA THRESHOLD FILTRATION
        cluster_bool = ((self.xC > 0) * (self.yC > 0) * np.isfinite(self.xC) * np.isfinite(self.yC)  # finite value check
                        * np.isin(self.f, single_fnums))   # singles only

        self.xC = self.xC[cluster_bool].astype(int)
        self.yC = self.yC[cluster_bool].astype(int)
        self.f = self.f[cluster_bool]
        return (self.xC, self.yC, self.f)

    def set_coin_params(self, fps, t0_dt, TS_ROI, binfac=1, verbose=True):
        """Sets parameters for coincidence analysis between iQID events and
        external timestamps (e.g., from an IDM - Ionization Detection Module).

        This method prepares timestamp arrays and time bins for use in
        `find_coin`. It assumes iQID event times are derived from frame numbers
        and the camera's frames per second (fps).

        Parameters
        ----------
        fps : float
            Frames per second of the iQID camera.
        t0_dt : float
            Time offset in seconds to be added to `TS_ROI`. This accounts for
            differences in start times between iQID and the external device.
            Positive if external device started after iQID, negative if before.
        TS_ROI : np.ndarray
            Timestamps from the external device (e.g., IDM) in seconds.
            These should be monotonically increasing.
        binfac : int, optional
            Binning factor for creating time bins (`s_bins`). The effective
            time bin size will be `(1 / fps) * binfac`. Defaults to 1.
        verbose : bool, optional
            If True, prints information about frame duration, first few
            timestamps from each source, and the generated time bins.
            Defaults to True.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - a : np.ndarray
                iQID event times in seconds, derived from `self.f` and `fps`.
            - b : np.ndarray
                Adjusted external event times in seconds (`TS_ROI + t0_dt`).
            - s_bins : np.ndarray
                Time bins (in seconds) for histogramming, covering the range of
                iQID event times.

        Side Effects
        ------------
        Sets the following instance attributes:
        - self.fps : float (stores `fps`)
        - self.exp : float (exposure time per frame, `1 / fps`)
        - self.a : np.ndarray (iQID event times)
        - self.b : np.ndarray (adjusted external times)
        - self.s_bins : np.ndarray (time bins)
        Requires `self.f` (iQID frame numbers) to be populated.

        Raises
        ------
        AssertionError
            If the calculated iQID times (`a`) or the adjusted external times (`b`)
            are not monotonically increasing, which can indicate issues with
            data acquisition or input parameters.
        """
        # TS_ROI: external from IDM data, code this in later
        # t0_dt: amount of time to delay IDM timestamp, i.e. IDM STARTED AFTER IQID
        # must adjust to be NEGATIVE if IDM STARTED BEFORE IQID
        self.fps = fps
        self.exp = 1 / fps

        a = self.f * self.exp
        b = TS_ROI + t0_dt
        s_bins = np.arange(np.max(self.f) + 2, step=binfac) * self.exp
        # reason for +2: 1 so that it goes up to max f, +1 so that it includes the time "during" that frame

        # check that we aren't accidentally stacking two acquisitions
        # not sure how to replicate this , but it happens possibly when reset is not properly done between IDM LM acquisitions
        assert np.all(
            a[:-1] <= a[1:]), "iQID array not monotonically increasing"
        assert np.all(
            b[:-1] <= b[1:]), "IDM array not monotonically increasing"

        if verbose:
            print('{:.2f} ms per frame'.format(self.exp * 1e3))
            print('iQID first 10:', a[:10])
            print('IDM first 10:', b[:10])
            print('Bins', s_bins[:10])

        self.a = a
        self.b = b
        self.s_bins = s_bins

        return a, b, s_bins

    def find_coin(self, singles=False, return_hist=False, verbose=True):
        """Finds coincident events by histogramming iQID and external (IDM)
        event times into shared time bins and identifying bins with events
        from both sources.

        This method uses `self.a` (iQID times), `self.b` (external times), and
        `self.s_bins` (time bins) set by `set_coin_params`.

        Parameters
        ----------
        singles : str | bool, optional
            Controls filtering for bins with single events.
            - False (default): No single-event filtering applied. A bin is
              coincident if it has one or more events from both iQID and external.
            - 'iqid': Coincident bins must have exactly one iQID event and one
              or more external events.
            - 'idm': Coincident bins must have one or more iQID events and
              exactly one external event.
            - 'both': Coincident bins must have exactly one iQID event and
              exactly one external event.
            Defaults to False.
        return_hist : bool, optional
            If True, also returns the raw iQID and IDM histograms (counts per bin).
            Defaults to False.
        verbose : bool, optional
            If True, prints progress messages and the number of multi-events
            found before and after `singles` filtering. Defaults to True.

        Returns
        -------
        np.ndarray[bool] | tuple[np.ndarray[bool], np.ndarray, np.ndarray]
            - coin : np.ndarray[bool]
                A boolean array where True indicates a coincident time bin
                according to the specified criteria. Length is `len(self.s_bins) - 1`.
            - iq_n : np.ndarray (optional, if `return_hist` is True)
                Histogram of iQID event counts per bin.
            - idm_n : np.ndarray (optional, if `return_hist` is True)
                Histogram of external event counts per bin.

        Side Effects
        ------------
        Relies on `self.a`, `self.b`, and `self.s_bins` having been set by
        `set_coin_params`.
        """
        if verbose:
            print('Generating iQID hist...')
        iq_n, _ = np.histogram(self.a, bins=self.s_bins)
        if verbose:
            print('Generating IDM hist...')
        idm_n, _ = np.histogram(self.b, bins=self.s_bins)

        coin = np.logical_and(iq_n, idm_n)  # if both are positive (non-zero)
        multi = np.sum(coin)
        if verbose:
            print('{} multi events found'.format(multi))

        if singles == 'iqid':
            # discard bins for which multiple iqid events id'd
            iq_n = (iq_n == 1)
        elif singles == 'idm':
            idm_n = (idm_n == 1)  # discard bins for which multiple gammas id'd
        elif singles == 'both':
            iq_n = (iq_n == 1)
            idm_n = (idm_n == 1)
        else:
            pass

        coin = np.logical_and(iq_n, idm_n)  # if both are positive (non-zero)
        if verbose:
            print('Selected {} coincident bins'.format(np.sum(coin)))

        if return_hist:
            return coin, iq_n, idm_n
        else:
            return coin

    def image_from_coin(self, coin=None, verbose=True, binfac=1, **kwargs):
        """Reconstructs an image using only the iQID events that are determined
        to be coincident with external events.

        This method uses a boolean array (`coin`) indicating coincident time bins,
        typically generated by `find_coin`. It then identifies the iQID events
        falling into these bins and constructs an image from their coordinates.

        Parameters
        ----------
        coin : np.ndarray[bool], optional
            A boolean array where True indicates a coincident time bin. This array
            should align with the time bins used in `find_coin` (derived from
            `self.s_bins`). If None, `find_coin(**kwargs)` is called internally
            to generate it. Defaults to None.
        verbose : bool, optional
            If True, prints progress messages, including the number of "good"
            (coincident) events found and a progress bar for image construction.
            Defaults to True.
        binfac : int, optional
            Binning factor used for frame numbers to align with time bins. This
            should typically match the `binfac` used in `set_coin_params` if
            `coin` is generated internally or based on similar binning.
            Defaults to 1.
        **kwargs : dict, optional
            Additional keyword arguments passed to `self.find_coin` if `coin`
            is None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - x_good : np.ndarray
                X-coordinates of the iQID events identified as coincident.
            - y_good : np.ndarray
                Y-coordinates of the iQID events identified as coincident.
            - nim : np.ndarray
                The reconstructed 2D image from the coincident events.
                Dimensions are (self.YDIM, self.XDIM).

        Side Effects
        ------------
        - Relies on `self.f` (iQID frame numbers), `self.xC` (x-coordinates),
          `self.yC` (y-coordinates), `self.XDIM`, and `self.YDIM`.
        - If `coin` is None, `find_coin` is called, which in turn relies on
          attributes set by `set_coin_params`.
        """
        if coin is None:
            coin = self.find_coin(**kwargs)

        # recover the frame number from the histogram
        f_bins = np.arange(np.max(self.f) + 2, step=binfac)
        good_f = f_bins[:-1][coin]
        good_events = np.isin(self.f, good_f)

        if verbose:
            print('Found {:.2e} "good" events.'.format(np.sum(good_events)))

        x_good = self.xC[good_events]
        y_good = self.yC[good_events]

        # manually rebuild spatial image
        nim = np.zeros((int(self.YDIM), int(self.XDIM)))

        if verbose:
            for i in trange(len(y_good), desc='Building selected image...'):
                nim[y_good[i], x_good[i]] += 1
        else:
            for i in range(len(y_good)):
                nim[y_good[i], x_good[i]] += 1

        return x_good, y_good, nim

    def check_elements(self, a, idx, x):
        """Recursive helper function to find the index of element `x` in array `a`,
        starting the search from index `idx`.

        This function is likely used to align frame numbers between different
        data arrays where one might be a subset or have gaps relative to the other.

        Parameters
        ----------
        a : np.ndarray
            The array to search within.
        idx : int
            The starting index in array `a` for the current search iteration.
        x : any
            The value to find in array `a`.

        Returns
        -------
        int
            The index of the first occurrence of `x` in `a` at or after the
            initial `idx`.

        Raises
        ------
        IndexError
            If `x` is not found in `a` starting from `idx` before the end of
            the array is reached (this will manifest as a Python recursion
            depth limit error if `a` is large and `x` is not present, or an
            actual IndexError if `idx` goes out of bounds).
        """
        if a[idx] == x:
            return idx
        else:
            idx += 1
            return self.check_elements(a, idx, x)

    def correct_frames(self, a1, a2, m1):
        """Aligns missed frame counts from an offset file (m1, corresponding to
        offset frames a1) to the frames in a listmode file (a2).

        This function iterates through listmode frames (a2) and finds the
        corresponding entry in offset frames (a1) to assign the correct
        missed frame count (m1). It handles cases where listmode frames might
        not have a direct match in offset frames, typically by carrying over
        the last known missed count.

        Parameters
        ----------
        a1 : np.ndarray
            Frame numbers from the offset file (N elements). Expected to be sorted.
        a2 : np.ndarray
            Frame numbers from the listmode file (M elements, M >= N). Expected to be sorted.
        m1 : np.ndarray
            Missed frame counts corresponding to each frame in `a1` (N elements).

        Returns
        -------
        np.ndarray
            m2 : An array of missed frame counts corresponding to each frame
            in `a2` (M elements).

        Side Effects
        ------------
        Prints a message and fills remaining `m2` values with the last valid
        missed count if listmode data contains frame numbers higher than
        those present in the offset file. Uses `self.check_elements`.
        """
        running_idx = 0
        m2 = np.zeros_like(a2)

        for i in trange(len(a2), desc='Assigning missed frames'):
            # check that current offset element is equal to current full element
            # if not, increment offset element
            try:
                running_idx = self.check_elements(a1, running_idx, a2[i])
                m2[i] = m1[running_idx]
            except IndexError:  # sometimes listmode data has values that offset doesn't have
                # I am not certain why this is the case, maybe old-version iqid bug
                print('aborting : list-mode data contains higher frames than offset file. Remainder will be filled with previous m-values')
                # fill remaining values with same number of missed as previous
                m2[i:] = m2[i-1]
                break

        return m2

    def correct_listmode(self, offset_frames, missed_frames, vis=True):
        """Corrects the frame numbers in `self.f` (listmode frames) by adding
        the number of missed frames. The missed frame information is derived
        from an offset file (containing `offset_frames` and `missed_frames`).

        This method uses `correct_frames` to map missed frame counts to the
        listmode frame sequence and then updates `self.f` by adding these
        counts. This process effectively converts frame numbers into a sequence
        that accounts for data acquisition gaps.

        Parameters
        ----------
        offset_frames : np.ndarray
            Frame numbers from the offset file.
        missed_frames : np.ndarray
            Missed frame counts corresponding to each frame in `offset_frames`.
        vis : bool, optional
            If True, plots a comparison of missed frames (from `missed_frames`
            aligned to unique listmode frames) against the newly calculated
            missed frames for the full listmode sequence. Defaults to True.

        Returns
        -------
        np.ndarray
            The corrected listmode frame numbers (`self.f` after modification).

        Side Effects
        ------------
        - Modifies `self.f` in place by adding the determined missed frame counts.
          Be cautious as this overwrites the original frame numbers.
        - Calls `self.correct_frames`.
        - If `vis` is True, a matplotlib plot is shown.
        - Requires `self.f` to be populated.
        """
        # use offset file to correct missed frames
        a1 = offset_frames
        a2 = self.f
        m1 = missed_frames
        m2 = self.correct_frames(a1, a2, m1)

        u2, idx2 = np.unique(a2, return_index=True)
        u1, idx1, _ = np.intersect1d(a1, u2, return_indices=True)

        if vis:
            plt.plot(m1[idx1])
            plt.plot(m2[idx2])
            plt.xlabel('Index')
            plt.ylabel('Number of missed frames')
            plt.show()
            plt.close()

        corr_lm = m2 + self.f

        self.f = corr_lm  # overwrites listmode frames in place, be careful to save separate variable if needed for some reason
        return corr_lm

    def set_contour_params(self, gauss=15, thresh=0):
        """Sets parameters for contour detection used in `get_contours` and
        `prep_contour`.

        Parameters
        ----------
        gauss : int, optional
            Kernel size for Gaussian blurring applied during contour preparation.
            Should be an odd integer. Defaults to 15.
        thresh : float | int, optional
            Threshold value used to binarize the image before contour detection.
            Pixels with values greater than this threshold will be set to 1 (or True),
            others to 0 (or False). Defaults to 0.

        Returns
        -------
        tuple
            An empty tuple.

        Side Effects
        ------------
        Sets instance attributes:
        - self.gauss : int
        - self.thresh : float | int
        """
        self.gauss = gauss
        self.thresh = thresh
        return ()

    def prep_contour(self, im, gauss=15, thresh=0):
        """Prepares an image for contour detection by binning (downsampling),
        thresholding, and Gaussian blurring.

        Parameters
        ----------
        im : np.ndarray
            The input image (2D NumPy array).
        gauss : int, optional
            Kernel size for Gaussian blurring. If not provided, uses `self.gauss`.
            Defaults to 15. (Note: current implementation shadows `self.gauss`
            if `gauss` is passed as an argument).
        thresh : float | int, optional
            Threshold value for binarization. If not provided, uses `self.thresh`.
            Defaults to 0. (Note: current implementation shadows `self.thresh`
            if `thresh` is passed as an argument).

        Returns
        -------
        np.ndarray
            The prepared image: a blurred binary mask (uint8 type) ready for
            contour finding.

        Side Effects
        ------------
        Requires `self.binfac` to be set for binning the image.
        Uses `self.gauss` and `self.thresh` if the corresponding parameters
        are not explicitly provided to the method.
        """
        binned_image = helper.bin_ndarray(
            im, (np.array(np.shape(im))/self.binfac).astype(int), operation='sum')
        mask = binned_image > thresh
        bin_im = mask.astype('uint8')
        prep_im = cv2.GaussianBlur(bin_im, (gauss, gauss), 0)
        return (prep_im)

    def get_contours(self, im):
        """Finds contours in an image after preparing it using `prep_contour`.

        The preparation involves binning, thresholding, and blurring. Contours
        are then filtered by area and scaled up by the binning factor.

        Parameters
        ----------
        im : np.ndarray
            The input image (2D NumPy array) from which to find contours.

        Returns
        -------
        list[np.ndarray]
            A list of "good" contours. Each contour is a NumPy array of
            (x, y) coordinates defining the boundary. "Good" contours are those
            whose area (after preparation but before scaling) is greater than
            `self.ROI_area_thresh`. The contour coordinates are scaled by
            `self.binfac` to match the original image dimensions.

        Side Effects
        ------------
        - Calls `self.prep_contour` using `self.gauss` and `self.thresh` as set
          by `set_contour_params` (or their defaults).
        - Uses `self.binfac` for scaling contours and `self.ROI_area_thresh` for
          filtering them.
        """
        prep_im = self.prep_contour(
            im=im, gauss=self.gauss, thresh=self.thresh)
        contours, _ = cv2.findContours(
            prep_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        good_contours = [
            self.binfac*c for c in contours if cv2.contourArea(c) > self.ROI_area_thresh]
        return (good_contours)

    def get_contours_from_dir(self, mask_dir, fformat='png'):
        """Loads contours from pre-existing manual mask image files stored in a
        specified directory.

        Each image file in the directory is treated as a binary mask. Contours
        are found in each mask, filtered by area, and scaled. It expects one
        dominant contour per mask file after filtering.

        Parameters
        ----------
        mask_dir : str
            Path to the directory containing the mask image files.
        fformat : str, optional
            File format (extension) of the mask images (e.g., 'png', 'tif').
            Defaults to 'png'.

        Returns
        -------
        list[np.ndarray]
            A list of contours. Each contour is a NumPy array of (x,y) coordinates.
            Assumes the first "good" contour found in each file is the desired one.
            Contours are scaled by `self.binfac` and filtered by `self.ROI_area_thresh`.

        Side Effects
        ------------
        - Reads image files from the specified directory.
        - Uses `cv2.findContours` to find contours in each image.
        - Uses `self.binfac` for scaling and `self.ROI_area_thresh` for filtering.
        - Sorts file names naturally.
        """
        # generate contours from manual masks
        c = []
        file_names = glob.glob(os.path.join(mask_dir, "*."+fformat))
        helper.natural_sort(file_names)
        # print('Loading {} masks...'.format(len(file_names)))
        for i in range(len(file_names)):
            im = io.imread(file_names[i])
            contours, _ = cv2.findContours(
                im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            good_contours = [
                self.binfac*c for c in contours if cv2.contourArea(c) > self.ROI_area_thresh]
            c.append(good_contours[0])
        # print('Returning {} contours'.format(len(c)))
        return c

    def get_maskstack(self, im):
        """Creates a stack of binary masks from the instance's `self.contours`.

        Each contour in `self.contours` is drawn filled onto a separate
        binary mask image. These masks are then stacked into a 3D array.

        Parameters
        ----------
        im : np.ndarray
            An image whose dimensions (shape) are used to create the
            individual masks in the stack. Typically, this is the original
            image from which contours were derived or an image of the same size.

        Returns
        -------
        np.ndarray
            A 3D NumPy array where each slice `[i, :, :]` is a binary mask
            (uint8, values 0 or 1) corresponding to `self.contours[i]`.
            The dimensions are (num_contours, height, width).

        Side Effects
        ------------
        Requires `self.contours` to be previously set (e.g., by `get_contours`
        or `get_contours_from_dir`).
        """
        xdim, ydim = np.shape(im)
        maskstack = np.zeros((len(self.contours), xdim, ydim))
        for i in range(len(self.contours)):
            mask = np.zeros_like(im)
            cv2.drawContours(mask, self.contours, i, (255, 255, 255), -1)
            mask_norm = (mask/255).astype(np.uint8)
            maskstack[i, :, :] = mask_norm
        return (maskstack)

    def events_in_ROI(self, maskstack):
        """Determines which events (from `self.xC`, `self.yC`, `self.t_s`) fall
        within each ROI defined by a mask stack.

        For each mask in the `maskstack`, this method checks each event's
        coordinates (xC, yC) to see if it lies within the mask. It also
        filters for valid events (t_s > 0, finite coordinates).

        Parameters
        ----------
        maskstack : np.ndarray
            A 3D array of ROI masks, typically generated by `get_maskstack`.
            Shape should be (num_masks, height, width).

        Returns
        -------
        np.ndarray[bool]
            A 2D boolean array of shape (num_masks, num_events).
            `ROI_array_bool[i, j]` is True if event `j` is within mask `i`
            and is a valid event.

        Side Effects
        ------------
        Requires `self.xC`, `self.yC` (event coordinates), and `self.t_s`
        (event times in seconds) to be populated. Event coordinates are
        cast to int for indexing into the mask.
        """
        ROI_array_bool = np.zeros((len(maskstack), len(self.xC)))
        for i in range(len(maskstack)):
            mask = maskstack[i, :, :]
            inROI_bool = mask[self.yC.astype(int), self.xC.astype(int)] * \
                (self.t_s > 0) * np.isfinite(self.xC) * np.isfinite(self.yC)
            ROI_array_bool[i, :] = inROI_bool
        return (ROI_array_bool.astype(bool))

    def get_ROIs(self, pad=10):
        """Calculates bounding boxes for each contour in `self.contours`,
        with optional padding.

        Parameters
        ----------
        pad : int, optional
            Padding to add around each bounding box. Half of this value is
            subtracted from x and y, and the full value is added to width
            and height, effectively centering the padding. Defaults to 10 pixels.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (`num_contours`, 4), where each row
            represents a bounding box as `[x, y, width, height]`.
            These values are integers.

        Side Effects
        ------------
        Requires `self.contours` to be previously set.
        """
        ROI_array = np.zeros((len(self.contours), 4))
        for i in range(len(self.contours)):
            cnt = self.contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            ROI_array[i, :] = x-pad//2, y-pad//2, w+pad, h+pad
        return (ROI_array.astype(int))

    def setup_ROIs(self, im, mode='auto', **kwargs):
        """A convenience function to orchestrate the setup of ROIs.

        This involves:
        1. Getting contours, either automatically from an image (`mode='auto'`)
           or by loading them from a directory of manual masks (`mode='manual'`).
        2. Creating a stack of binary masks from these contours (`self.maskstack`).
        3. Determining which events fall within each ROI (`self.ROIbool`).
        4. Calculating bounding boxes for each ROI (`self.ROIlists`).

        Parameters
        ----------
        im : np.ndarray
            The image to use for automatic ROI detection (if `mode='auto'`)
            and as a reference for mask dimensions.
        mode : str, optional
            Specifies the method for obtaining contours:
            - 'auto': Use `self.get_contours(im)` to find contours automatically.
            - 'manual': Use `self.get_contours_from_dir(**kwargs)` to load
              manually created contours.
            Defaults to 'auto'.
        **kwargs : dict, optional
            Additional keyword arguments passed directly to
            `self.get_contours_from_dir` if `mode='manual'`. Common arguments
            might include `mask_dir` and `fformat`.

        Returns
        -------
        None
            This method does not return a value but sets several instance attributes.

        Side Effects
        ------------
        Populates the following instance attributes:
        - `self.contours` : list[np.ndarray]
        - `self.maskstack` : np.ndarray (3D stack of ROI masks)
        - `self.ROIbool` : np.ndarray (2D boolean array of events in ROIs)
        - `self.ROIlists` : np.ndarray (2D array of ROI bounding boxes [x,y,w,h])
        Calls `get_contours` or `get_contours_from_dir`, `get_maskstack`,
        `events_in_ROI`, and `get_ROIs`.

        Raises
        ------
        TypeError
            If `mode` is not 'auto' or 'manual'.
        """
        if mode == 'auto':
            self.contours = self.get_contours(im)
        elif mode == 'manual':
            self.contours = self.get_contours_from_dir(**kwargs)
        else:
            raise TypeError('Mode must be "auto" or "manual".')
        self.maskstack = self.get_maskstack(im)
        self.ROIbool = self.events_in_ROI(self.maskstack)
        self.ROIlists = self.get_ROIs()

    def fitHist(self, t, n, func=exponential, p0=[1, 9.92*24*3600], tol=0.05):
        """Fits a given function (default: exponential decay) to histogram data
        (counts `n` at time points `t`).

        This method uses `scipy.optimize.curve_fit` for the fitting procedure.
        It calculates optimal parameters, their covariance and standard deviation,
        residuals, and chi-squared statistics.

        Parameters
        ----------
        t : np.ndarray
            An array of time points, typically bin centers of a histogram.
        n : np.ndarray
            An array of counts corresponding to the time points in `t`.
        func : callable, optional
            The function to fit to the data. It should take `x` (time) as its
            first argument, followed by its parameters. Defaults to `exponential`
            (defined in this module).
        p0 : list[float], optional
            Initial guess for the parameters of `func`. The second element `p0[1]`
            is treated as a reference half-life if `func` is `exponential`,
            used for setting parameter bounds. Defaults to `[1, 9.92*24*3600]`
            (amplitude 1, half-life approx 9.92 days in seconds).
        tol : float, optional
            Tolerance used to set bounds for the second parameter of `func`
            (typically half-life). Bounds are set as `[p0[1]*(1-tol), p0[1]*(1+tol)]`.
            The first parameter (amplitude) is bounded at `[0, np.inf]`.
            Defaults to 0.05 (i.e., +/- 5% for half-life).

        Returns
        -------
        tuple
            Contains the following elements:
            - popt : np.ndarray
                Optimal values for the parameters of `func`.
            - pcov : np.ndarray
                The estimated covariance matrix of `popt`. The diagonals provide
                the variance of the parameter estimates.
            - param_std : np.ndarray
                Standard deviation of the parameters (sqrt of variances).
            - res : np.ndarray
                Residuals (n - func(t, *popt)).
            - chisq : float
                Chi-squared statistic: sum((residuals^2) / func(t, *popt)).
            - chisqn : float
                Reduced chi-squared statistic (chisq / degrees_of_freedom), where
                degrees_of_freedom is `len(n)`.

        Notes
        -----
        - The sigma for `curve_fit` is set to `np.maximum(np.ones_like(n), np.sqrt(n))`,
          meaning a minimum uncertainty of 1 is assumed for low counts.
        - The bounds for the second parameter are `[thalf * (1-tol), thalf * (1+tol)]`,
          and for the first parameter `[0, np.inf]`.
        """
        thalf = p0[1]
        popt, pcov = curve_fit(f=func, xdata=t, ydata=n,
                               p0=p0, sigma=np.maximum(
                                   np.ones_like(n), np.sqrt(n)),
                               bounds=([0, thalf * (1-tol)], [np.inf, thalf * (1 + tol)]))
        param_std = np.sqrt(np.diag(pcov))
        res = n - func(t, *popt)
        chisq = np.sum(res**2/func(t, *popt))
        chisqn = chisq/len(n)
        return (popt, pcov, param_std, res, chisq, chisqn)

    def fitROI(self, temporal_array, func=exponential, p0=None, binsize=1000, tol=0.05):
        """Performs a temporal fit for events within a single ROI.

        This method first histograms the `temporal_array` (timestamps of events
        within an ROI) using the specified `binsize`. Then, it calls `fitHist`
        to fit the chosen function (e.g., exponential decay) to this histogram.

        Parameters
        ----------
        temporal_array : np.ndarray
            An array of timestamps (in seconds) for events belonging to a specific ROI.
            Assumed to be sorted for histogramming if `nbins` calculation is critical,
            though `np.histogram` handles unsorted data.
        func : callable, optional
            The function to fit to the histogram data. Passed to `fitHist`.
            Defaults to `exponential`.
        p0 : list[float] | None, optional
            Initial guess for the parameters of `func`. If None, it defaults to
            `[1, self.t_half]`, where `self.t_half` is an instance attribute
            (expected to be in seconds). Passed to `fitHist`. Defaults to None.
        binsize : int | float, optional
            The size of time bins (in seconds) used to histogram `temporal_array`.
            Defaults to 1000 seconds.
        tol : float, optional
            Tolerance for parameter bounds in `fitHist`. Defaults to 0.05.

        Returns
        -------
        tuple
            Contains the following elements:
            - count : np.ndarray
                The counts per bin from histogramming `temporal_array`.
            - timepoints : np.ndarray
                The center time points of the histogram bins.
            - popt : np.ndarray
                Optimal parameters from `fitHist`.
            - pcov : np.ndarray
                Covariance matrix from `fitHist`.
            - param_std : np.ndarray
                Standard deviation of parameters from `fitHist`.
            - res : np.ndarray
                Residuals from `fitHist`.
            - chisq : float
                Chi-squared value from `fitHist`.
            - chisqn : float
                Reduced chi-squared value from `fitHist`.

        Side Effects
        ------------
        Requires `self.t_half` to be set if `p0` is None.
        Calls `self.fitHist`.
        """
        if p0 is None:
            p0 = [1, self.t_half]

        nbins = np.round(temporal_array[-1]/binsize)
        count, bins = np.histogram(temporal_array, np.arange(0, nbins)*binsize)
        timepoints = bins[:-1] + binsize/2
        popt, pcov, param_std, res, chisq, chisqn = self.fitHist(
            timepoints, count, func=func, p0=p0, tol=tol)

        return (count, timepoints, popt, pcov, param_std, res, chisq, chisqn)

    def get_imslice(self, im, idx):
        """Extracts a masked image slice corresponding to a specific ROI.

        The slice is defined by the bounding box of the ROI (`self.ROIlists[idx]`)
        and is masked by the corresponding ROI mask (`self.maskstack[idx]`).

        Parameters
        ----------
        im : np.ndarray
            The source image (2D NumPy array) from which to extract the slice.
        idx : int
            The index of the ROI in `self.ROIlists` and `self.maskstack`.

        Returns
        -------
        np.ndarray
            A 2D NumPy array representing the masked image slice. The values
            outside the ROI mask within the bounding box will be zero.

        Side Effects
        ------------
        Requires `self.ROIlists` (list of ROI bounding boxes [x,y,w,h]) and
        `self.maskstack` (stack of ROI masks) to be populated.
        """
        x, y, w, h = self.ROIlists[idx, :]
        imslice = im[y:y+h, x:x+w] * self.maskstack[idx, y:y+h, x:x+w]
        return imslice

    # def test_countours(self):
    #     good_contours = []
    #     for i in range(len(man_maskstack)):
    #         contours, _ = cv2.findContours(man_maskstack[i, :, :],
    #                                        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         gcontours = [c for c in contours if cv2.contourArea(
    #             c) > self.ROI_area_thresh]
    #         good_contours.append(gcontours)

    def get_manual_maskstack(self):
        """Loads manual masks from TIFF files located in a 'manual_masks'
        subdirectory within `self.savedir`.

        This method processes these loaded masks to:
        1. Find contours within each mask image.
        2. Filter these contours by area (`self.ROI_area_thresh`).
        3. Recreate a binary mask stack (`maskstack`) from these filtered contours,
           ensuring they match `self.XDIM`, `self.YDIM`.
        4. Determine which events fall within these manual ROIs (`ROI_array_bool`).
        5. Calculate bounding boxes for these ROIs (`ROI_array`).

        It then updates several instance attributes (`self.ROIlists`,
        `self.ROIbool`, `self.contours`, `self.maskstack`) with this information
        derived from the manually loaded masks.

        Parameters
        ----------
        None.

        Returns
        -------
        np.ndarray
            The stack of binary masks (uint8) created from the processed
            manual TIFF files. Shape: (num_masks, self.YDIM, self.XDIM).

        Side Effects
        ------------
        - Reads TIFF files from `os.path.join(self.savedir, 'manual_masks', '*.tif')`.
        - Populates/updates the following instance attributes:
            - `self.ROIlists` : np.ndarray (bounding boxes)
            - `self.ROIbool` : np.ndarray[bool] (events in ROIs)
            - `self.contours` : list (list of contour arrays; each element can be a list itself)
            - `self.maskstack` : np.ndarray (3D stack of masks)
        - Requires `self.savedir`, `self.XDIM`, `self.YDIM`, `self.ROI_area_thresh`,
          `self.xC`, `self.yC`, and `self.t_s` to be set.
        - Assumes each good contour list `gcontours` will have at least one element
          when calculating bounding boxes (`good_contours[i][0]`).
        """
        base = os.path.basename(os.path.normpath(self.file_name))
        maskdir = os.path.join(self.savedir, 'manual_masks')
        fileList = glob.glob(os.path.join(maskdir, '*.tif'))
        man_maskstack = np.array(io.ImageCollection(fileList))

        good_contours = []

        for i in range(len(man_maskstack)):
            contours, _ = cv2.findContours(man_maskstack[i, :, :],
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            gcontours = [c for c in contours if cv2.contourArea(
                c) > self.ROI_area_thresh]
            good_contours.append(gcontours)

        maskstack = np.zeros((len(good_contours), self.YDIM, self.XDIM))
        for i in range(len(good_contours)):
            mask = np.zeros((self.YDIM, self.XDIM))
            cv2.drawContours(mask, good_contours[i], -1, (255, 255, 255), -1)
            mask_norm = (mask/255).astype(np.uint8)
            maskstack[i, :, :] = mask_norm

        ROI_array_bool = np.zeros((len(maskstack), len(self.xC)))
        for i in range(len(maskstack)):
            mask = maskstack[i, :, :]
            inROI_bool = mask[self.yC.astype(int), self.xC.astype(int)] * \
                (self.t_s > 0) * np.isfinite(self.xC) * np.isfinite(self.yC)
            ROI_array_bool[i, :] = inROI_bool

        pad = 10
        ROI_array = np.zeros((len(good_contours), 4))
        for i in range(len(good_contours)):
            x, y, w, h = cv2.boundingRect(good_contours[i][0]) # Assumes gcontours[i] is not empty
            ROI_array[i, :] = x-pad//2, y-pad//2, w+pad, h+pad

        self.ROIlists = ROI_array.astype(int)
        self.ROIbool = ROI_array_bool.astype(bool)
        self.contours = good_contours
        self.maskstack = maskstack
        return (maskstack)

    def save_manual_mask(self, mask):
        """Saves a given mask array as both a PNG preview and a TIFF file
        in the instance's `self.savedir` directory.

        The filenames used are 'manual_mask_preview.png' and 'manual_mask.tif'.

        Parameters
        ----------
        mask : np.ndarray
            The 2D mask image (NumPy array) to be saved.

        Returns
        -------
        None

        Side Effects
        ------------
        - Creates `self.savedir` directory if it does not already exist.
        - Saves two files in `self.savedir`:
            - 'manual_mask_preview.png' (via `matplotlib.pyplot.imsave`)
            - 'manual_mask.tif' (via `skimage.io.imsave` with 'tifffile' plugin)
        - Prints the path to `self.savedir`.
        """
        # base = os.path.basename(os.path.normpath(self.file_name))
        # newdir = os.path.join(self.file_name, '..', base[:-4] + '_Analysis')
        Path(self.savedir).mkdir(parents=True, exist_ok=True)
        plt.imsave(os.path.join(self.savedir, 'manual_mask_preview.png'), mask)
        io.imsave(os.path.join(self.savedir, 'manual_mask.tif'),
                  mask, plugin='tifffile')
        print('Manual mask saved to:')
        print(self.savedir)

    def fitROIs(self, im, vis=True, corr=0,
                tol=0.05, tstart=0, tcond=None,
                idxs='all',
                save=False, savemasks=False, save_ts=False):
        """Performs activity fitting for multiple ROIs, calculates activity
        values (A0, Ai), and optionally visualizes and saves results.

        This method iterates through specified ROIs (or all ROIs), extracts
        event data for each, performs temporal fitting using `fitROI`,
        calculates initial activity (Ai) and decay-corrected activity (A0),
        and can generate plots and save various outputs.

        Parameters
        ----------
        im : np.ndarray
            The base image (2D) from which ROI image slices (`imslice`) are taken
            for visualization or saving. Activity values are calculated per pixel
            within these slices.
        vis : bool, optional
            If True, generates and returns a matplotlib figure and axes array
            showing the image slice and the temporal fit for each processed ROI.
            Defaults to True.
        corr : float, optional
            Time correction value in seconds (e.g., time from sacrifice to imaging).
            This value is used to decay-correct Ai to A0 (activity at t=0,
            presumably sacrifice time). `A0 = Ai * exp(ln(2) * corr / t_half)`.
            Defaults to 0.
        tol : float, optional
            Tolerance for parameter bounds in the fitting process, passed to
            `fitROI` and subsequently to `fitHist`. Defaults to 0.05.
        tstart : float, optional
            Start time (in seconds) for the fitting. Data points before this time
            will be excluded from the fit. Applied after initial `fitROI`.
            Defaults to 0 (fit all data).
        tcond : tuple[float, float] | tuple[float, float, float, float] | None, optional
            Time condition(s) for selecting data for fitting, applied after
            initial `fitROI`.
            - If 2 floats `(start1, stop1)`: fits data between start1 and stop1.
            - If 4 floats `(start1, stop1, start2, stop2)`: fits data in two
              disjoint intervals.
            This overrides `tstart` if not None. Defaults to None.
        idxs : str | list[int], optional
            Specifies which ROIs to process.
            - 'all': Process all ROIs found in `self.ROIlists`.
            - list[int]: A list of integer indices specifying the ROIs to process.
            Defaults to 'all'.
        save : bool | str, optional
            If True or 'slicer3d', saves the calculated activity image slices
            (`aslice`) as TIFF files. If 'slicer3d', data type is float32.
            Also saves PNG previews if `vis` is True. Files are saved in
            `self.savedir/mBq_images/` and `self.savedir/mBq_image_previews/`.
            Defaults to False.
        savemasks : bool, optional
            If True, saves full-size ROI masks and cropped ROI masks as PNG files
            in `self.savedir/full_masks/` and `self.savedir/ROI_masks/` respectively.
            Defaults to False.
        save_ts : bool, optional
            If True, saves the timestamps (`t_data`) for each processed ROI as a
            text file in `self.savedir/ROI_tstamps/`. Defaults to False.

        Returns
        -------
        tuple
            - all_A0 : np.ndarray
                An array containing the calculated A0 (initial, decay-corrected
                activity in Curies) for each processed ROI.
            - all_dA0 : np.ndarray
                An array containing the uncertainties (d(A0) in Curies) for each
                processed ROI.
            - f : matplotlib.figure.Figure (optional, if `vis` is True)
                The matplotlib Figure object containing the plots.
            - ax : np.ndarray[matplotlib.axes.Axes] (optional, if `vis` is True)
                An array of matplotlib Axes objects, one row per ROI, with two
                columns (image slice, temporal fit).

        Side Effects
        ------------
        - Iterates through ROIs, calling `self.get_imslice` and `self.fitROI`.
        - May call `self.fitHist` if `tstart` or `tcond` are used.
        - Creates directories and saves image, mask, or timestamp files if
          `save`, `savemasks`, or `save_ts` options are enabled.
        - Uses many instance attributes: `self.ROIlists`, `self.t_s`,
          `self.ROIbool`, `self.t_binsize`, `self.t_half`, `self.savedir`,
          `self.maskstack`.

        Raises
        ------
        ValueError
            If `tcond` is provided but does not have 2 or 4 elements.
        """
        if idxs == 'all':
            iterlist = range(len(self.ROIlists))
            # FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison if idxs == 'all':
        else:
            iterlist = idxs

        base = os.path.basename(os.path.normpath(self.file_name))

        if vis:
            f, ax = plt.subplots(nrows=len(iterlist), ncols=2,
                                 figsize=(8, 4*len(iterlist)))
            if save:
                Path(self.savedir).mkdir(parents=True, exist_ok=True)
                pvdir = os.path.join(self.savedir, 'mBq_image_previews')
                Path(pvdir).mkdir(parents=True, exist_ok=True)

        if save:
            Path(self.savedir).mkdir(parents=True, exist_ok=True)
            imdir = os.path.join(self.savedir, 'mBq_images')
            Path(imdir).mkdir(parents=True, exist_ok=True)

        all_A0 = np.zeros(len(iterlist))
        all_dA0 = np.zeros(len(iterlist))
        for i in tqdm(range(len(iterlist)), desc='Getting ROIs'):
            # cut out the ROI from the full image
            imslice = self.get_imslice(im, iterlist[i])  # cts

            t_data = self.t_s[self.ROIbool[iterlist[i]]]

            # use the events (with temporal data) inside the ROI to fit whole slice activity
            count, ts, popt, _, pstd, _, _, chisqn = self.fitROI(t_data,
                                                                 binsize=self.t_binsize,
                                                                 tol=tol)

            if save_ts:
                Path(self.savedir).mkdir(parents=True, exist_ok=True)
                newdir = os.path.join(self.savedir, 'ROI_tstamps')
                Path(newdir).mkdir(parents=True, exist_ok=True)
                np.savetxt(os.path.join(
                    newdir, 'tstamps_{}.txt'.format(i)), t_data)

            if tstart > 0:
                if vis:
                    ax[i, 1].axvspan(tstart, ts[-1], color='gray', alpha=0.3)
                gidx = ts > tstart
                popt, _, pstd, _, _, chisqn = self.fitHist(
                    ts[gidx], count[gidx])
            elif tcond is not None:
                # this is extremely hacky. beware
                # input should be (start1, stop1, start2, stop2) in s
                if len(tcond) != 4 and len(tcond) != 2:
                    raise ValueError(
                        'tcond input should be (start1, stop1) OR (start1, stop1, start2, stop2)')

                if len(tcond) == 2:
                    if vis:
                        ax[i, 1].axvspan(tcond[0], tcond[1],
                                         color='gray', alpha=0.3)

                    gidx = (ts > tcond[0])*(ts < tcond[1])
                    popt, _, pstd, _, _, chisqn = self.fitHist(
                        ts[gidx], count[gidx])

                elif len(tcond) == 4:
                    if vis:
                        ax[i, 1].axvspan(tcond[0], tcond[1],
                                         color='gray', alpha=0.3)
                        ax[i, 1].axvspan(tcond[2], tcond[3],
                                         color='gray', alpha=0.3)

                    gidx = (ts > tcond[0])*(ts < tcond[1]) + \
                        (ts > tcond[2])*(ts < tcond[3])
                    popt, _, pstd, _, _, chisqn = self.fitHist(
                        ts[gidx], count[gidx])

            # activity (mBq) at the start of imaging
            Ai = popt[0] * 1000/self.t_binsize
            dAi = pstd[0] * 1000/self.t_binsize

            # activity correction to the time of sacrifice
            A0 = Ai * np.exp(np.log(2) * corr / self.t_half)  # mBq
            A0_Bq = A0 / 1e3
            A0_Ci = A0_Bq / 3.7e10
            all_dA0[i] = dAi * \
                np.exp(np.log(2) * corr / self.t_half) / 1e3 / 3.7e10
            all_A0[i] = A0_Ci

            # total A over whole image * pxcts / totalcts = A in each px
            aslice = A0 * imslice / np.sum(imslice)

            if vis:
                ax[i, 0].imshow(aslice, cmap='gray',
                                vmax=0.15 * np.max(aslice))
                ax[i, 0].axis('off')
                xdummy = np.linspace(0, max(ts))
                ax[i, 1].scatter(ts, count)
                ax[i, 1].errorbar(ts, count, np.sqrt(
                    count), ls='none', capsize=3)
                ax[i, 1].plot(
                    xdummy, exponential(xdummy, *popt))
                ax[i, 1].set_xlabel('time (s)')
                ax[i, 1].set_ylabel(
                    'Counts in {} s bins'.format(self.t_binsize))
                ax[i, 1].set_title('Ai={:.0f} pm {:.0f} mBq\n'.format(Ai, dAi) +
                                   '$\\chi^2/N$: {:.3f}\nA0: {:.0f} mBq ({:.3f} nCi)'.format(
                                       chisqn, A0, A0_Ci*1e9),
                                   y=-0.5, fontsize=12)

            if save:
                if save == 'slicer3d':
                    aslice = aslice.astype(np.float32)
                if vis:
                    plt.imsave(os.path.join(pvdir, 'mBq_preview_{}.png'.format(i)),
                               aslice, cmap='gray', vmax=0.15 * np.max(aslice))
                io.imsave(os.path.join(imdir, 'mBq_{}.tif'.format(i)),
                          aslice, plugin='tifffile', photometric='minisblack', check_contrast=False)

        if savemasks:
            base = os.path.basename(os.path.normpath(self.file_name))
            maskdir = os.path.join(self.savedir, 'full_masks')
            Path(maskdir).mkdir(parents=True, exist_ok=True)
            for i in range(len(iterlist)):
                io.imsave(os.path.join(maskdir, 'mask_{}.png'.format(i)),
                          (255*self.maskstack[iterlist[i], :, :]).astype(np.uint8), check_contrast=False)

            maskdir = os.path.join(self.savedir, 'ROI_masks')
            Path(maskdir).mkdir(parents=True, exist_ok=True)
            for i in range(len(iterlist)):
                x, y, w, h = self.ROIlists[iterlist[i], :]
                mask = self.maskstack[iterlist[i], y:y+h, x:x+w]
                io.imsave(os.path.join(maskdir, 'mask_{}.png'.format(i)),
                          (255*mask).astype(np.uint8))
            print('masks saved to:', maskdir)

        if vis:
            return (all_A0, all_dA0, f, ax)
        else:
            return (all_A0, all_dA0)

    def plot_vis_masks(self, im):
        """Plots each mask from `self.maskstack` overlaid on a given background
        image `im`. Each mask is displayed in a separate matplotlib figure.

        Note: The current implementation attempts to encode a masked NumPy array
        to PNG bytes and then display it with `plt.imshow`. This is unconventional
        and may have issues. A more standard approach would be to use
        `plt.imshow` with alpha blending or to overlay contours.

        Parameters
        ----------
        im : np.ndarray
            The background image (2D NumPy array) upon which masks will be visualized.

        Returns
        -------
        None
            This method shows matplotlib plots directly and does not return any value.

        Side Effects
        ------------
        - Iterates through `self.maskstack`.
        - For each mask, creates a new matplotlib figure and displays the mask
          (apparently attempting to overlay it on `im` by masking `im` and then
          encoding/decoding, which is indirect).
        - Uses `matplotlib.pyplot` for plotting and `cv2.imencode` (OpenCV)
          for image encoding.
        - Requires `self.maskstack` to be populated.
        """
        for i in range(len(self.maskstack)):
            plt.figure(figsize=(8, 6))
            # plt.imshow((1-weight) * im + weight*self.maskstack[i,:,:], cmap='gray')
            # masked_array = np.ma.array(im, mask=self.maskstack[i,:,:])

            helper_im = np.ma.array(im, mask=self.maskstack[i, :, :])
            _, helper_im = cv2.imencode('.png', helper_im.astype(np.uint8))
            helper_im = helper_im.tobytes() # This might be problematic for plt.imshow
            cmap = matplotlib.cm.inferno
            plt.imshow(helper_im, interpolation='nearest', cmap=cmap) # plt.imshow usually expects an array
            plt.axis('off')
            plt.title(i)
            plt.show()
            plt.close()

    def widget_labelling(self, im, vmax=1, deg=0, IMG_WIDTH=200, IMG_HEIGHT=200, COLS=4):
        """Creates an IPython widget interface for interactively labeling and
        selecting/discarding ROIs based on their masks.

        This widget displays images of ROIs (derived from `im` and `self.maskstack`)
        in a grid. Each ROI image is accompanied by a text input for a label
        (defaulting to its index + 1) and a toggle button to "discard" it.
        A "Generate Indices Array" button processes the labels and discard states
        to produce output arrays representing the desired order and inclusion of ROIs.

        Parameters
        ----------
        im : np.ndarray
            The base image (2D NumPy array) used to display ROIs. Masks from
            `self.maskstack` are applied to this image.
        vmax : float, optional
            Multiplier for image maximum value used for display scaling of ROIs.
            `display_roi_im = roi_im * 255.0 / (vmax * roi_im.max())`.
            Defaults to 1.
        deg : int, optional
            Angle in degrees for rotating the displayed ROI images. Defaults to 0.
        IMG_WIDTH : int, optional
            Width (in pixels) of each individual ROI image displayed in the widget.
            Defaults to 200.
        IMG_HEIGHT : int, optional
            Height (in pixels) of each individual ROI image displayed in the widget.
            Defaults to 200.
        COLS : int, optional
            Number of columns in the grid layout for displaying ROIs.
            Defaults to 4.

        Returns
        -------
        None
            This method displays an IPython widget directly and does not return
            any value. The results (inclusion and order arrays) are printed to
            an `ipywidgets.Output` area within the widget upon button click.

        Side Effects
        ------------
        - Displays an interactive IPython widget if run in a compatible environment
          (e.g., Jupyter Notebook).
        - Requires `self.maskstack` to be populated.
        - Uses `ipywidgets`, `functools`, `matplotlib.cm`, `cv2`, `skimage.transform`.
        - Defines and uses several nested helper functions for widget callbacks.
        """
        ROWS = int(np.ceil(len(self.maskstack)/COLS))

        rows = []
        for row in tqdm(range(ROWS), desc="Building widget"):
            cols = []
            for col in range(COLS):
                try:
                    idx = row * COLS + col

                    helper_im = np.copy(im)
                    midx = (self.maskstack[idx, :, :] == 1)
                    helper_im[midx] = np.max(helper_im)
                    helper_im = transform.rotate(helper_im, deg)
                    helper_im *= 255.0/(vmax * helper_im.max())
                    helper_im = cv2.imencode('.png', helper_im)[1].tobytes()

                    image = widgets.Image(
                        value=helper_im, width=IMG_WIDTH, height=IMG_HEIGHT
                    )

                    labelbox = widgets.BoundedIntText(
                        value=idx+1,
                        min=-1,
                        max=len(self.maskstack)+1,
                        step=1,
                        disabled=False,
                        layout=widgets.Layout(flex='1 1 0%', width='auto')
                    )

                    nobutton = widgets.ToggleButton(
                        value=False,
                        description='discard',
                        button_style='',
                        layout=widgets.Layout(flex='1 1 0%', width='auto'))

                    # Create a vertical layout box, image above the button
                    buttonbox = widgets.HBox([labelbox, nobutton],
                                             layout=widgets.Layout(display='flex',
                                                                   align_items='stretch',
                                                                   width='100%'))
                    box = widgets.VBox([image, buttonbox])
                    cols.append(box)
                except IndexError:
                    break  # for when # of images is not divisible by 4

            # Create a horizontal layout box, grouping all the columns together
            rows.append(widgets.HBox(cols))

        # Create a vertical layout box, grouping all the rows together
        result = widgets.VBox(rows)
        submit_button = widgets.Button(
            description="Generate Indices Array",
            layout=widgets.Layout(width='100%'))

        output = widgets.Output()

        def button_click(rs_, b):
            generate_idxs(rs_)

        def text_yn(arr_, darr_, wdgt):
            yn = wdgt.value
            with output:
                yn = helper.get_yn(yn)
                if yn:
                    arr_ = arr_[np.logical_not(darr_)]
                    new_arr = np.arange(1, len(arr_) + 1)[np.argsort(arr_)]
                    print('Inclusion:', repr(np.logical_not(darr_).astype(int)))
                    print('Order:', repr(new_arr.astype(int)))
                else:
                    print(repr(arr_.astype(int)))

        def generate_idxs(row_widgt):
            # obtain value of all of the manual buttons
            manual_idxs = np.zeros(len(self.maskstack))
            manual_discard = np.zeros(len(self.maskstack))
            counter = 0  # could be more elegant, possibly fix in future with grid and flatten
            for i in range(ROWS):
                single_row = row_widgt[i]
                for j in range(COLS):
                    try:
                        single_box = single_row.children[j]
                        manual_idxs[counter] = single_box.children[1].children[0].value
                        manual_discard[counter] = single_box.children[1].children[1].value
                        counter += 1
                    except IndexError:
                        break

            idx_arr = manual_idxs.flatten()
            dis_arr = manual_discard.flatten()
            out_arr = idx_arr[np.logical_not(dis_arr)]

            with output:
                err_mesg = "Duplicate indices detected. Please ensure each positive index appears exactly once."
                assert len(out_arr) == len(np.unique(out_arr)), err_mesg

                err_mesg = "Skipped indices detected:\n{}".format(
                    np.sort(out_arr))
                if np.array_equal(np.sort(out_arr), np.arange(1, len(out_arr) + 1)):
                    print('Inclusion:', repr(np.logical_not(dis_arr).astype(int)))
                    print('Order:', repr(out_arr.astype(int) - 1))
                else:
                    print(err_mesg)
                    prompt = widgets.Text(
                        value='yes',
                        placeholder='yes/no',
                        description="Fix?",
                        style={'description_width': 'initial'}
                    )
                    with output:
                        display(prompt)
                    prompt.on_submit(functools.partial(
                        text_yn, idx_arr, dis_arr))

        submit_button.on_click(functools.partial(button_click, rows))

        display(result)
        display(submit_button, output)


class Subset(ClusterData):
    """
    Represents a subset of data derived from a `ClusterData` object.

    This class inherits all methods from `ClusterData`. It is typically
    created by the `ClusterData.get_subset()` method, which populates its
    attributes with a filtered portion of the data from the parent
    `ClusterData` instance.
    """
    def __init__(self, file_name, c_area_thresh, makedir, ftype):
        """Initializes a Subset object.

        This constructor primarily calls the superclass (`ClusterData`)
        initializer. The actual data subset is typically populated by the
        `ClusterData.get_subset()` method after this initialization.

        Parameters
        ----------
        file_name : str
            Path to the original iQID data file (passed to superclass).
        c_area_thresh : int
            Cluster area threshold (passed to superclass).
        makedir : bool
            If True, creates an analysis subdirectory (passed to superclass).
            Typically False for Subsets as they share the parent's directory.
        ftype : str
            File type of the input data (passed to superclass).

        Side Effects
        ------------
        Calls `super().__init__(file_name, c_area_thresh, makedir, ftype)`.
        Instance attributes specific to the data subset (e.g., self.xC, self.yC)
        are intended to be populated externally, usually by the method creating
        this Subset instance.
        """
        super().__init__(file_name, c_area_thresh, makedir, ftype)
