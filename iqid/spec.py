"""
This module provides functions and classes for gamma spectroscopy analysis,
particularly tailored for data from systems like Hidex detectors and for
analyzing isotopes such as Ac-225 and Ce-134.

It includes functionalities for:
- Loading and processing spectral data, including from Hidex Excel files.
- Peak finding and fitting (e.g., Gaussian fits).
- Energy calibration of spectra.
- Summing counts within specified Regions of Interest (ROIs).
- Calculating activities and performing decay corrections, especially for Ac-225
  and its progeny (Fr-221, Bi-213).
- Estimating dose rates.
- Handling and analyzing biodistribution data.

The module defines classes like `energy_spectrum` for general spectral
analysis, `BioD` for detailed biodistribution calculations of single or
grouped samples, and `MultiBioD` for aggregating results from multiple
BioD instances, often used for time-series analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import becquerel as bq

from tqdm import trange

from iqid import helper as iq

from scipy.stats import linregress
pltmap = plt.get_cmap("tab10")


####################################################################################
#### Functions made for Hidex / BioD analysis in general (Ac225/Ce134)##############
####################################################################################

def sumROI_arr(arr: np.ndarray, x: np.ndarray, ROI: tuple[float, float]) -> float:
    """Sums counts within a specified Region of Interest (ROI) from a NumPy array.

    This function is typically used for spectra where counts (`arr`) and
    corresponding energy values (`x`) are provided as separate NumPy arrays.
    It excludes negative count values from the sum.

    Parameters
    ----------
    arr : np.ndarray
        Array of counts (spectrum).
    x : np.ndarray
        Array of energy values (keV) corresponding to the counts in `arr`.
    ROI : tuple[float, float]
        A tuple defining the energy range (min_kev, max_kev) of the ROI.

    Returns
    -------
    float
        The sum of counts within the specified ROI.
    """
    inROI = (x > ROI[0]) * (x < ROI[1])
    Nvals = arr[inROI]
    N = np.sum(Nvals[Nvals > 0])  # exclude negatives
    # todo: dN (uncertainty calculation)
    return N


def sumROI_spec(spec: bq.Spectrum, ROI: tuple[float, float]) -> float:
    """Sums counts within a specified Region of Interest (ROI) from a
    `becquerel.Spectrum` object.

    This function utilizes the `bin_centers_kev` and `counts_vals` attributes
    of a `bq.Spectrum` object. It excludes negative count values from the sum.

    Parameters
    ----------
    spec : bq.Spectrum
        A becquerel Spectrum object.
    ROI : tuple[float, float]
        A tuple defining the energy range (min_kev, max_kev) of the ROI.

    Returns
    -------
    float
        The sum of counts within the specified ROI.
    """
    inROI = (spec.bin_centers_kev > ROI[0]) * (spec.bin_centers_kev < ROI[1])
    Nvals = spec.counts_vals[inROI]
    N = np.sum(Nvals[Nvals > 0])  # exclude negatives
    # todo: dN (uncertainty calculation)
    return N


def th227_fit_hidex(s: bq.Spectrum, ROI: tuple[float, float], model: bq.fitting.base.Model, vis: bool = False) -> tuple[float, bq.Fitter, np.ndarray]:
    """Fits a given model to a specified ROI within a `becquerel.Spectrum`.

    This function is tailored for fitting peaks in Hidex spectra, potentially
    for Th-227 analysis or similar. It uses `becquerel.Fitter`.

    Parameters
    ----------
    s : bq.Spectrum
        The becquerel Spectrum object to be fitted.
    ROI : tuple[float, float]
        The Region of Interest (min_kev, max_kev) to perform the fit on.
    model : bq.fitting.base.Model
        A becquerel Model object (e.g., Gaussian, Polynomial) to fit to the data.
    vis : bool, optional
        If True, displays the fit plot using `fitter.custom_plot()`.
        Defaults to False.

    Returns
    -------
    tuple[float, bq.Fitter, np.ndarray]
        - counts : float
            The net counts in the fitted peak (sum of `model(x) - background(x)`).
        - fitter : bq.Fitter
            The becquerel Fitter object containing the fit result and parameters.
        - bg : np.ndarray
            The calculated background component of the fit within the ROI.
    """
    fitter = bq.Fitter(
        model,
        x=s.bin_centers_kev,
        y=s.counts_vals,
        y_unc=s.counts_uncs,
        roi=ROI,
    )

    fitter.fit()

    if vis:
        # this flag doesnt work for some reason >:(
        fitter.custom_plot(enable_fit_panel=False)
        plt.show()

    xdata = fitter.x_roi
    ydata = fitter.y_roi

    bg = fitter.x_roi * \
        fitter.result.params['linear_m'] + fitter.result.params['linear_b']
    gaussnet = fitter.eval(xdata, **fitter.best_values) - bg
    counts = np.sum(gaussnet)
    # chisq = fitter.result.chisqr

    # plt.plot(xdata, ydata)
    # plt.plot(xdata, bg)
    # plt.plot(xdata, gaussnet)
    return counts, fitter, bg


def load_hidex_xls(fpath: str, headstring: str = "Results", headskip: int = 19, colskip: int = 2, other_idx: int | None = None, peak_flag: bool = False):
    """Loads spectral data and timestamps from a Hidex Excel file.

    This function parses a specific Excel file format typically generated by
    Hidex gamma counters. It extracts spectral data, measurement timestamps,
    and optionally other metadata columns. It can also identify expected
    peak energies if `peak_flag` is True and an isotope column is specified.

    Parameters
    ----------
    fpath : str
        Path to the Hidex Excel file.
    headstring : str, optional
        A string to identify the start of the data section in the metadata sheet.
        Defaults to "Results".
    headskip : int, optional
        Number of rows to skip at the beginning of the spectrum sheet.
        Defaults to 19.
    colskip : int, optional
        Number of columns to skip at the beginning of the spectrum sheet.
        Defaults to 2.
    other_idx : int | None, optional
        Column index (0-based) in the metadata sheet from which to extract
        additional data. If None, no additional data is returned.
        Defaults to None.
    peak_flag : bool, optional
        If True and `other_idx` points to an isotope information column,
        this function will attempt to determine characteristic gamma peak
        energies (e.g., for Ce-134 or Ac-225) based on keywords in that column.
        Defaults to False.

    Returns
    -------
    tuple
        The content of the tuple depends on `other_idx` and `peak_flag`:
        - If `other_idx` is None:
            `(spec_df, ts)`
            - `spec_df` (np.ndarray): 2D array of spectra, (num_spectra, num_channels).
            - `ts` (np.ndarray): 1D array of timestamps for each spectrum.
        - If `other_idx` is an int and `peak_flag` is False:
            `(spec_df, ts, other_data)`
            - `other_data` (np.ndarray): 1D array of data from the `other_idx` column.
        - If `other_idx` is an int and `peak_flag` is True:
            `(spec_df, ts, other_data, e_peaks)`
            - `e_peaks` (list[list[float]]): List where each sublist contains
              expected peak energies (keV) for the corresponding spectrum based
              on isotope info. Possible peaks: [511, 1022] for Ce, [218, 440] for Ac.

    Raises
    ------
    ValueError
        If pandas encounters an error reading the file (e.g., file is open).
    IndexError
        If `headstring` is not found in the metadata sheet.
    """
    try:
        xls = pd.ExcelFile(fpath)
    except ValueError as e:
        print(
            'Pandas ValueError: make sure to close the data files before trying to import.')
        raise ValueError(e)

    # Spectrum sheet has consistent 19-row header. Rack/vial are not saved.
    spec_df = pd.read_excel(
        xls, sheet_name=1, skiprows=headskip, header=None).to_numpy()[:, colskip:]

    # Metadata sheet has variable rack-description header
    # Look for "Results" row header to separate metadata vs measured data
    meta_df = pd.read_excel(xls, sheet_name=0, header=None)
    results_row = meta_df.loc[meta_df.iloc[:, 0] == headstring].index[0]

    time_df = meta_df.iloc[(results_row + 2):, 2]
    ts = time_df[~time_df.isna()].to_numpy()

    if type(other_idx) is int:
        other_data = meta_df.iloc[(results_row + 2):, other_idx]

        if peak_flag:  # special flag to do analysis with Ac and Ce
            iso = other_data.to_numpy()
            e_peaks = []

            for i in range(len(iso)):
                peaks = []
                if pd.isna(iso[i]):
                    e_peaks.append(peaks)
                    continue
                if 'Ce' in iso[i]:
                    peaks.append(511)
                    peaks.append(1022)
                if 'Ac' in iso[i]:
                    peaks.append(218)
                    peaks.append(440)
                e_peaks.append(peaks)
            return spec_df, ts, other_data, e_peaks

        else:
            return spec_df, ts, other_data.to_numpy()
    else:
        return spec_df, ts


def pick_peaks(found_peak_array: np.ndarray, es: np.ndarray, de: float = 50) -> tuple[list[float], list[float]]:
    """Matches found peak centroids to a list of known emission energies.

    Given an array of peak centroids detected in a spectrum and an array of
    known (expected) emission energies, this function identifies which found
    peaks correspond to the known energies within a given tolerance `de`.

    Parameters
    ----------
    found_peak_array : np.ndarray
        An array of peak centroids (e.g., in channels or rough keV) found by a
        peak-finding algorithm.
    es : np.ndarray
        An array of known emission energies (in keV) to match against.
    de : float, optional
        The maximum allowed difference (delta energy) between a found peak
        centroid and a known emission energy for them to be considered a match.
        Defaults to 50 (units should be consistent with `found_peak_array`
        if it's not already in keV, or `es` should be converted).

    Returns
    -------
    tuple[list[float], list[float]]
        - good_centroid_array : list[float]
            A list of the found peak centroids that were successfully matched to
            known emission energies.
        - good_es : list[float]
            A list of the known emission energies that were matched. The order
            corresponds to `good_centroid_array`.
    """
    good_centroid_array = []
    good_es = []
    for i in range(len(found_peak_array)):
        centroid = found_peak_array[i]
        differ = np.abs(es - centroid)
        if min(differ) < de:
            best_match = es[np.argmin(differ)]
            good_centroid_array.append(centroid)
            good_es.append(best_match)
    return good_centroid_array, good_es


def calibrate_spec(spectrum: np.ndarray | bq.Spectrum, es: np.ndarray, kernel, min_snr: float = 2, xmin: float = 200, livetime: float = 60., de: float = 50, import_new: bool = True, verbose: bool = False) -> tuple[bq.Calibration, bq.Spectrum]:
    """Performs energy calibration for a gamma spectrum.

    This function takes a spectrum (either raw counts or a `bq.Spectrum` object),
    finds peaks using `bq.PeakFinder`, matches them to known energies `es`
    using `pick_peaks`, and then creates and applies a linear energy calibration
    (`p[0] + p[1] * x`) using `bq.Calibration`.

    Parameters
    ----------
    spectrum : np.ndarray | bq.Spectrum
        The spectrum to calibrate. If np.ndarray, it's treated as counts and
        a new `bq.Spectrum` object is created if `import_new` is True.
    es : np.ndarray
        An array of known emission energies (in keV) for calibration.
    kernel :
        The kernel to use for `bq.PeakFinder` (e.g., `bq.GaussianPeakFilter`).
    min_snr : float, optional
        Minimum signal-to-noise ratio for peak detection in `bq.PeakFinder`.
        Defaults to 2.
    xmin : float, optional
        Minimum x-value (channel or rough keV) for peak finding. Defaults to 200.
    livetime : float, optional
        Livetime of the spectrum in seconds. Used if `import_new` is True and
        `spectrum` is an np.ndarray. Defaults to 60.0.
    de : float, optional
        Maximum energy difference for matching found peaks to known energies in
        `pick_peaks`. Defaults to 50.
    import_new : bool, optional
        If True and `spectrum` is an np.ndarray, a new `bq.Spectrum` object
        is created. If False and `spectrum` is already a `bq.Spectrum` object,
        it's used directly. Defaults to True.
    verbose : bool, optional
        If True, prints found centroids and matched peaks. Defaults to False.

    Returns
    -------
    tuple[bq.Calibration, bq.Spectrum]
        - cal : bq.Calibration
            The `becquerel.Calibration` object representing the fitted energy
            calibration.
        - spec : bq.Spectrum
            The (potentially newly created) `becquerel.Spectrum` object with the
            calibration applied.
    """
    if import_new:
        spec = bq.Spectrum(counts=spectrum, livetime=livetime)
    else:
        spec = spectrum # Assumes spectrum is already a bq.Spectrum object

    finder = bq.PeakFinder(spec, kernel)
    finder.find_peaks(min_snr=min_snr, xmin=xmin)
    good_peaks, good_es = pick_peaks(finder.centroids, es, de=de)

    if verbose:
        print('Centroids:', finder.centroids)
        print('Peaks:', good_peaks, good_es)

    cal = bq.Calibration.from_points(
        "p[0] + p[1] * x", good_peaks, good_es, params0=[5.0, 0.15])
    spec.apply_calibration(cal)
    return cal, spec


def get_cali_fn(a: np.ndarray, n: np.ndarray, nsamples: int = 3, return_res: bool = True):
    """Generates a calibration function from activity vs. count measurements.

    Given known activities (`a`) and corresponding measured counts (`n`), this
    function performs a linear regression and returns a callable function that
    can convert new count measurements back to activities. It assumes that
    `n` contains multiple samples per activity level, specified by `nsamples`.

    Parameters
    ----------
    a : np.ndarray
        1D array of known activities (e.g., nCi). Length should be `len(n) / nsamples`.
    n : np.ndarray
        1D array of measured counts (e.g., CPM). Its length must be a multiple
        of `nsamples`. Data is reshaped to (`len(a)`, `nsamples`).
    nsamples : int, optional
        Number of replicate samples measured at each activity level in `a`.
        Defaults to 3.
    return_res : bool, optional
        If True, also returns the `scipy.stats.linregress` result object.
        Defaults to True.

    Returns
    -------
    callable | tuple[callable, scipy.stats._stats_mstats_common.LinregressResult]
        - cali_fn : callable
            A function that takes a new count measurement (float) and returns
            the corresponding calibrated activity (float).
            `activity = intercept + slope * new_count_measurement`
        - res : scipy.stats._stats_mstats_common.LinregressResult (optional)
            The full result object from `scipy.stats.linregress(mean_counts, activities)`
            if `return_res` is True.
    """
    n_reshaped = n.reshape(len(n)//nsamples, nsamples)
    mean_counts = np.mean(n_reshaped, axis=1)
    # dx = np.std(n_reshaped, axis=1)  # Standard deviation of counts, not currently used
    res = linregress(mean_counts, a)

    def cali_fn(new_count_measurement: float) -> float:
        return res.intercept + res.slope * new_count_measurement

    if return_res:
        return cali_fn, res
    else:
        return cali_fn


def get_ac225_counts(spec: np.ndarray | list[np.ndarray], e_peaks: list[list[float]] | np.ndarray, ROIFr: list[float] = [180, 260], ROIBi: list[float] = [400, 480], kernel = bq.GaussianPeakFilter(500, 50, fwhm_at_0=10), vis: bool = False, xlim: tuple[float, float] = (-10, 700)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Processes a list of Ac-225 spectra to get counts in Fr-221 and Bi-213 ROIs.

    For each spectrum in the input list `spec`:
    1. It attempts to perform energy calibration using `calibrate_spec` with
       the corresponding expected peak energies from `e_peaks`.
    2. If calibration fails (e.g., due to too few peaks), it applies the
       previously successful calibration (`cal` from the last good spectrum).
       Note: `cal` might be undefined if the very first spectrum fails calibration.
    3. It sums counts in the specified Fr-221 (`ROIFr`) and Bi-213 (`ROIBi`)
       regions of interest using `sumROI_spec`.
    4. Optionally, it visualizes the calibrated spectrum with ROIs highlighted.

    Parameters
    ----------
    spec : np.ndarray | list[np.ndarray]
        A list or 2D NumPy array of spectra. Each element/row is a spectrum
        (array of counts).
    e_peaks : list[list[float]] | np.ndarray
        A list where each sublist contains the expected peak energies (keV)
        for the corresponding spectrum in `spec`. Used for calibration.
    ROIFr : list[float], optional
        Energy ROI (min_kev, max_kev) for Fr-221 (typically around 218 keV).
        Defaults to [180, 260].
    ROIBi : list[float], optional
        Energy ROI (min_kev, max_kev) for Bi-213 (typically around 440 keV).
        Defaults to [400, 480].
    kernel : becquerel filter object, optional
        Kernel used for peak finding in `calibrate_spec`.
        Defaults to `bq.GaussianPeakFilter(500, 50, fwhm_at_0=10)`.
    vis : bool, optional
        If True, plots each calibrated spectrum with ROIs and expected peaks.
        Defaults to False.
    xlim : tuple[float, float], optional
        X-axis limits (keV) for visualization if `vis` is True.
        Defaults to (-10, 700).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - frs : np.ndarray
            Array of summed counts in `ROIFr` for each spectrum.
        - bis : np.ndarray
            Array of summed counts in `ROIBi` for each spectrum.
        - totals : np.ndarray
            Array of total summed counts (all bins) for each spectrum.

    Notes
    -----
    - The fallback calibration logic (`except: s.apply_calibration(cal)`)
      can be problematic if the first spectrum fails to calibrate, as `cal`
      would be undefined. This should be handled more robustly.
    - Assumes a fixed livetime of 60s when creating `bq.Spectrum` objects
      during fallback calibration.
    """
    frs = np.zeros(len(spec))
    bis = np.zeros(len(spec))
    totals = np.zeros(len(spec))

    # To store the last successful calibration
    last_successful_cal = None

    for i in trange(len(spec)):
        try:
            cal, s = calibrate_spec(
                spec[i], e_peaks[i], kernel, import_new=True, livetime=60.) # Assuming livetime from context
            last_successful_cal = cal
        except Exception as e:  # Catching a more general exception for robustness
            # print(f"Warning: Calibration failed for spectrum {i}. Error: {e}. Using last successful calibration.")
            if last_successful_cal is not None:
                s = bq.Spectrum(counts=spec[i], livetime=60.)
                s.apply_calibration(last_successful_cal)
            else:
                # print(f"Error: Calibration failed for spectrum {i} and no prior successful calibration exists. Skipping.")
                frs[i], bis[i], totals[i] = np.nan, np.nan, np.nan # Or handle as appropriate
                continue


        frs[i] = sumROI_spec(s, ROIFr)
        bis[i] = sumROI_spec(s, ROIBi)
        totals[i] = np.sum(s.counts_vals)

        if vis:
            f, ax = plt.subplots(1, 1, figsize=(8, 4))

            s.plot(xlim=xlim, ax=ax)
            # Assuming calibrated spectrum, bin width might not be simply 1 keV.
            # For uncalibrated, it's channels. For calibrated, it's energy.
            # This part of the label might need adjustment based on actual bin width post-calibration.
            ax.set_ylabel('counts/bin') # Simplified label

            if e_peaks[i]: # Check if e_peaks[i] is not empty
                 iq.draw_lines(e_peaks[i])
            plt.axvspan(ROIBi[0], ROIBi[1], color='gray', alpha=0.3)
            plt.axvspan(ROIFr[0], ROIFr[1], color='gray', alpha=0.3)

            plt.suptitle('Measurement {:.0f}'.format(i))
            # plt.yscale('log')
            plt.tight_layout()
            plt.show()
            plt.close()

    return frs, bis, totals


#######################################################################################

def gaussian(x: np.ndarray, a: float, mu: float, width: float, b: float, c: float) -> np.ndarray:
    """Defines a Gaussian function with a linear background.

    f(x) = a * exp(-(x - mu)^2 / (2 * width^2)) + b*x + c

    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g., energy channels or keV).
    a : float
        Amplitude of the Gaussian peak.
    mu : float
        Mean (center) of the Gaussian peak.
    width : float
        Standard deviation (sigma) of the Gaussian peak.
    b : float
        Slope of the linear background.
    c : float
        Intercept of the linear background.

    Returns
    -------
    np.ndarray
        The values of the Gaussian function with linear background at each point in `x`.
    """
    return (a*np.exp(-(x-mu)**2 / (2*width**2)) + b*x + c)


def gaussian_nobg(x: np.ndarray, a: float, mu: float, width: float) -> np.ndarray:
    """Defines a Gaussian function without any background component.

    f(x) = a * exp(-(x - mu)^2 / (2 * width^2))

    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g., energy channels or keV).
    a : float
        Amplitude of the Gaussian peak.
    mu : float
        Mean (center) of the Gaussian peak.
    width : float
        Standard deviation (sigma) of the Gaussian peak.

    Returns
    -------
    np.ndarray
        The values of the Gaussian function at each point in `x`.
    """
    return (a*np.exp(-(x-mu)**2 / (2*width**2)))


def gaussian_cbg(x: np.ndarray, a: float, mu: float, width: float, c: float) -> np.ndarray:
    """Defines a Gaussian function with a constant background.

    f(x) = a * exp(-(x - mu)^2 / (2 * width^2)) + c

    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g., energy channels or keV).
    a : float
        Amplitude of the Gaussian peak.
    mu : float
        Mean (center) of the Gaussian peak.
    width : float
        Standard deviation (sigma) of the Gaussian peak.
    c : float
        Constant background value.

    Returns
    -------
    np.ndarray
        The values of the Gaussian function with constant background at each point in `x`.
    """
    return (a*np.exp(-(x-mu)**2 / (2*width**2)) + c)


def load_masses_old(xls: pd.ExcelFile, sheet_name: str = "Meta", headskip: int = 23) -> tuple[np.ndarray, np.ndarray]:
    """Loads tumor and kidney masses from a specific Excel sheet format.

    .. deprecated:: Unknown
       This function appears to be an older version for loading mass data.
       Consider using newer data loading functions if available.

    Parameters
    ----------
    xls : pd.ExcelFile
        Pandas ExcelFile object of the input spreadsheet.
    sheet_name : str, optional
        Name of the sheet containing mass data. Defaults to "Meta".
    headskip : int, optional
        Number of header rows to skip in the sheet. Defaults to 23.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - tumor_g : np.ndarray
            Array of tumor masses in grams.
        - kidney_g : np.ndarray
            Array of kidney masses in grams.
    """
    meta_df = pd.read_excel(xls, sheet_name=sheet_name,
                            skiprows=headskip, header=None).to_numpy()[:, 1:]
    tumor_g = meta_df[0, :]
    kidney_g = meta_df[1, :]
    return (tumor_g, kidney_g)


def load_time_pi_old(xls: pd.ExcelFile, unit: str = 'h', sheet_name: str = "Meta", headskip: int = 17) -> tuple[float, float]:
    """Loads time post-injection (spi) and time post-sacrifice (sps) from an Excel sheet.

    .. deprecated:: Unknown
       This function appears to be an older version for loading time data.
       Consider using newer data loading functions if available.

    Parameters
    ----------
    xls : pd.ExcelFile
        Pandas ExcelFile object of the input spreadsheet.
    unit : str, optional
        The unit for the returned time values ('h', 'm', 's', 'ms').
        Defaults to 'h' (hours).
    sheet_name : str, optional
        Name of the sheet containing time data. Defaults to "Meta".
    headskip : int, optional
        Number of header rows to skip. Defaults to 17.

    Returns
    -------
    tuple[float, float]
        - spi : float
            Time from injection to sacrifice, in the specified `unit`.
        - sps : float
            Time from sacrifice to start of counting, in the specified `unit`.

    Raises
    ------
    ValueError
        If `unit` is not one of 'h', 'm', 's', 'ms'.
    """
    if not unit in ['h', 'm', 's', 'ms']:
        raise ValueError(
            "Invalid unit '{}'. Accepted units: 'h', 'm', 's', 'ms'".format(unit))

    meta_df = pd.read_excel(xls, sheet_name=sheet_name,
                            skiprows=headskip, nrows=3, header=None)
    dt = pd.to_datetime(meta_df[1])
    spi = pd.Timedelta(dt[1]-dt[0]).total_seconds()
    sps = pd.Timedelta(dt[2]-dt[1]).total_seconds()

    base_unit = ['h', 'm', 's', 'ms'].index(unit)
    fac = [3600, 60, 1, 1e-6]

    return spi/fac[base_unit], sps/fac[base_unit]


def load_macpeg_xls_old(fpath: str) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads various data from an old MaCPET-PEG Excel file format.

    .. deprecated:: Unknown
       This function aggregates data using other 'old' loading functions.
       Consider using newer, more modular data loading functions if available.

    This function serves as a wrapper to load time post-injection,
    time post-sacrifice, tumor and kidney masses, and tumor and kidney spectra
    from a specified Excel file, presumably from a MaCPET-PEG study.

    Parameters
    ----------
    fpath : str
        Path to the Excel file.

    Returns
    -------
    tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - time_pi : float
            Time from injection to sacrifice (unit depends on `load_time_pi_old` default).
        - time_ps : float
            Time from sacrifice to counting start (unit depends on `load_time_pi_old` default).
        - tumor_g : np.ndarray
            Array of tumor masses (grams).
        - kidney_g : np.ndarray
            Array of kidney masses (grams).
        - tumor_spec : np.ndarray
            2D array of tumor spectra (spectra as rows or columns, check implementation).
        - kidney_spec : np.ndarray
            2D array of kidney spectra.
    """
    xls = pd.ExcelFile(fpath)
    time_pi, time_ps = load_time_pi_old(xls)
    tumor_g, kidney_g = load_masses_old(xls)
    tumor_spec = pd.read_excel(
        xls, sheet_name='PEG4-Tumor', header=0).to_numpy()[:, 1:].T
    kidney_spec = pd.read_excel(
        xls, sheet_name='PEG4-kidney', header=0).to_numpy()[:, 1:].T
    return (time_pi, time_ps, tumor_g, kidney_g, tumor_spec, kidney_spec)


def load_t0s(xls: pd.ExcelFile, sheet_name: str = "Results", t0_rowskip: int = 17, hpi_unit: str = 'h') -> tuple[float, pd.Timestamp]:
    """Loads initial time points from a specified Excel sheet.

    This function extracts the time from injection to sacrifice (spi) and the
    absolute timestamp of sacrifice (t0_sac_s).

    Parameters
    ----------
    xls : pd.ExcelFile
        Pandas ExcelFile object of the input spreadsheet.
    sheet_name : str, optional
        Name of the sheet containing the time data. Defaults to "Results".
    t0_rowskip : int, optional
        Number of rows to skip to reach the time data block. Defaults to 17.
    hpi_unit : str, optional
        Unit for the returned `spi` value ('h', 'm', 's', 'ms').
        Defaults to 'h' (hours).

    Returns
    -------
    tuple[float, pd.Timestamp]
        - spi : float
            Time from injection to sacrifice, in the specified `hpi_unit`.
        - t0_sac_s : pd.Timestamp
            The timestamp corresponding to the time of sacrifice.

    Raises
    ------
    ValueError
        If `hpi_unit` is not one of 'h', 'm', 's', 'ms'.
    """
    if not hpi_unit in ['h', 'm', 's', 'ms']:
        raise ValueError(
            "Invalid unit '{}'. Accepted units: 'h', 'm', 's', 'ms'".format(hpi_unit))
    base_unit = ['h', 'm', 's', 'ms'].index(hpi_unit)
    fac = [3600, 60, 1, 1e-6]

    meta_df = pd.read_excel(xls, sheet_name=sheet_name,
                            skiprows=t0_rowskip, nrows=3, header=None)
    dt = pd.to_datetime(meta_df[meta_df.columns[1]])
    # injection time to sac time
    spi = pd.Timedelta(dt[1]-dt[0]).total_seconds()
    # dt[2] is the counting start time, but each rack/vial has a more precise measurement time
    t0_sac_s = dt[1]
    return spi/fac[base_unit], t0_sac_s


def load_meta(xls: pd.ExcelFile, sheet_name: str = "Results", header_rowskip: int = 41, res_rowskip: int = 102) -> tuple[pd.Series, np.ndarray]:
    """Loads metadata (measurement times and sample masses) from an Excel sheet.

    Parameters
    ----------
    xls : pd.ExcelFile
        Pandas ExcelFile object of the input spreadsheet.
    sheet_name : str, optional
        Name of the sheet containing metadata. Defaults to "Results".
    header_rowskip : int, optional
        Number of rows to skip to reach the header row for column names.
        Defaults to 41.
    res_rowskip : int, optional
        Number of rows to skip to reach the actual data rows.
        Defaults to 102.

    Returns
    -------
    tuple[pd.Series, np.ndarray]
        - dt : pd.Series
            Pandas Series of datetime objects for each measurement.
        - mass_g : np.ndarray
            NumPy array of sample masses in grams.
    """
    df_header = pd.read_excel(xls, sheet_name=sheet_name,
                              skiprows=header_rowskip, header=0, nrows=0)
    df = pd.read_excel(xls, sheet_name=sheet_name,
                       skiprows=res_rowskip, header=None, names=list(df_header))
    dt = pd.to_datetime(df['Time'])
    mass_g = df['Sample mass (g)'].to_numpy()
    return dt, mass_g


def load_macpeg_xls(fpath: str, skiprows: int = 79) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and processes data from a MaCPET-PEG Excel file.

    This function acts as a higher-level loader, utilizing `load_t0s` and
    `load_meta` to extract time post-injection, time post-sacrifice for each
    sample, sample masses, and the spectral data itself.

    Parameters
    ----------
    fpath : str
        Path to the MaCPET-PEG Excel file.
    skiprows : int, optional
        Number of rows to skip in the "Spectra" sheet before reading data.
        Defaults to 79.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray]
        - time_pi : float
            Time from injection to sacrifice (unit from `load_t0s` default, typically hours).
        - time_ps : np.ndarray
            1D array of times from sacrifice to counting for each spectrum (seconds).
        - mass_g : np.ndarray
            1D array of sample masses (grams).
        - spectra : np.ndarray
            2D array of spectra (num_spectra, num_channels).
    """
    xls = pd.ExcelFile(fpath)
    time_pi, t0_sac = load_t0s(xls) # time_pi is spi from load_t0s
    dt, mass_g = load_meta(xls)

    time_ps = np.zeros(len(dt))
    for i in range(len(dt)):
        time_ps[i] = pd.Timedelta(dt[i]-t0_sac).total_seconds()

    spectra = pd.read_excel(xls, sheet_name="Spectra",
                            skiprows=skiprows, header=None).to_numpy()[:, 2:]
    return (time_pi, time_ps, mass_g, spectra)


def ac225_spectra_stats(kev: np.ndarray, spectra: np.ndarray, roi1: tuple[float, float], roi2: tuple[float, float], min_peak_signal: float, vis: bool = False, unpack: bool = False, fp0: list | str | None = None, bp0: list | str | None = None):
    """Calculates statistics (net counts, uncertainty, gross counts) for Ac-225
    related peaks (typically Fr-221 and Bi-213) in one or more spectra.

    It uses the `energy_spectrum` class to perform Gaussian fitting on ROIs.

    Parameters
    ----------
    kev : np.ndarray
        Array of energy values (keV) corresponding to the channels in the spectra.
    spectra : np.ndarray
        A 1D array (single spectrum) or 2D array (multiple spectra, one per row)
        of counts.
    roi1 : tuple[float, float]
        Region of Interest (min_kev, max_kev) for the first peak (e.g., Fr-221).
    roi2 : tuple[float, float]
        Region of Interest (min_kev, max_kev) for the second peak (e.g., Bi-213).
    min_peak_signal : float
        Minimum peak signal (max counts in ROI) required to attempt a fit.
        Passed to `energy_spectrum`.
    vis : bool, optional
        If True, visualizes the fits using `energy_spectrum.plot_fit()`.
        Defaults to False.
    unpack : bool, optional
        If True, returns individual arrays for Fr and Bi stats. Otherwise,
        returns a single data matrix. Defaults to False.
    fp0 : list | str | None, optional
        Initial guess parameters for the Gaussian fit of the first peak (roi1).
        Can be a list of parameters `[amplitude, mean, width, bg_slope, bg_intercept]`,
        a string ('fr' for default Fr-221 guess), or None to use default 'fr'.
        Passed to `energy_spectrum.fit_gaussian`. Defaults to None.
    bp0 : list | str | None, optional
        Initial guess parameters for the Gaussian fit of the second peak (roi2).
        Can be a list of parameters, a string ('bi' for default Bi-213 guess),
        or None to use default 'bi'. Passed to `energy_spectrum.fit_gaussian`.
        Defaults to None.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, ...]
        If `unpack` is False:
            `data_matrix` (np.ndarray): A 2D array of shape (num_spectra, 6)
            containing [Fr_net, Fr_dnet, Fr_gross, Bi_net, Bi_dnet, Bi_gross].
        If `unpack` is True:
            A tuple of six 1D arrays:
            (fr_net, dfr_net, fr_gross, bi_net, dbi_net, bi_gross).
            Each array has length num_spectra. Squeezed if num_spectra is 1.
    """
    # hacky way to handle single array vs multiple array
    # there's probably a better way to do this?
    if len(np.shape(spectra)) == 1:
        spectra = spectra[np.newaxis, :]

    data_matrix = np.zeros((len(spectra), 6))
    # fr_nets, dfr_nets, fr_gross, bi_nets, dbi_nets, bi_gross

    for i in range(len(spectra)):
        espec = energy_spectrum(
            kev, spectra[i], min_peak_signal=min_peak_signal)

        if vis:
            espec.init_plot()

        # Fr
        current_fp0 = fp0 if fp0 is not None else 'fr'
        espec.fit_gaussian(roi1, p0=current_fp0, func=gaussian)
        net, dnet, gross = espec.get_stats(roi1)
        data_matrix[i, 0] = net
        data_matrix[i, 1] = dnet
        data_matrix[i, 2] = gross

        if vis:
            espec.plot_fit(roi1, 1)

        # Bi
        current_bp0 = bp0 if bp0 is not None else 'bi'
        # Bug: roi1 was passed to fit_gaussian for Bi peak instead of roi2. Corrected.
        espec.fit_gaussian(roi2, p0=current_bp0, func=gaussian)
        net, dnet, gross = espec.get_stats(roi2)
        data_matrix[i, 3] = net
        data_matrix[i, 4] = dnet
        data_matrix[i, 5] = gross

        if vis:
            espec.plot_fit(roi2, 2)
            plt.tight_layout()
            plt.show()

    # this is gross (ha, not net) but I'm too lazy to fix it right now

    if unpack:
        fr, dfr, frg, bi, dbi, big = np.split(data_matrix, 6, axis=1)
        return np.squeeze(fr), np.squeeze(dfr), np.squeeze(frg), np.squeeze(bi), np.squeeze(dbi), np.squeeze(big)
    else:
        return data_matrix


def calibrate_ac225(kev: np.ndarray, spectra: np.ndarray, roi1: tuple[float, float], roi2: tuple[float, float], uCi: float, min_peak_signal: float = 5, t: float = 60, vis: bool = False) -> tuple[float, float]:
    """Calculates detection efficiencies for Ac-225 peaks (Fr-221 and Bi-213)
    using calibration spectra with known activity.

    Parameters
    ----------
    kev : np.ndarray
        Array of energy values (keV) for the spectra.
    spectra : np.ndarray
        A 1D array (single spectrum) or 2D array (multiple spectra) of counts
        from calibration source(s) with known activity.
    roi1 : tuple[float, float]
        ROI for the Fr-221 peak (e.g., [180, 260] keV).
    roi2 : tuple[float, float]
        ROI for the Bi-213 peak (e.g., [400, 480] keV).
    uCi : float
        Known activity of the Ac-225 calibration source in microCuries (µCi).
    min_peak_signal : float, optional
        Minimum peak signal for `ac225_spectra_stats`. Defaults to 5.
    t : float, optional
        Counting time (livetime) in seconds for the calibration spectra.
        Defaults to 60.
    vis : bool, optional
        If True, visualizes fits during `ac225_spectra_stats` call.
        Defaults to False.

    Returns
    -------
    tuple[float, float]
        - fr_eff_mean : float
            Mean detection efficiency for the Fr-221 peak.
        - bi_eff_mean : float
            Mean detection efficiency for the Bi-213 peak.
    """
    data_matrix = ac225_spectra_stats(
        kev, spectra, roi1, roi2, min_peak_signal=min_peak_signal, vis=vis)
    # Theoretical counts = Activity (Bq) * livetime (s)
    cts_theory_total_ac225 = uCi * 1e-6 * 3.7e10 * t
    # Expected counts in Fr peak = total Ac-225 decays * Fr branching ratio
    fr_theory_counts = 0.116 * cts_theory_total_ac225 # Fr-221 (218 keV) BR from Ac-225 decay chain
    # Expected counts in Bi peak = total Ac-225 decays * Bi branching ratio
    bi_theory_counts = 0.261 * cts_theory_total_ac225 # Bi-213 (440 keV) BR from Ac-225 decay chain

    # data_matrix columns: 0:Fr_net, 3:Bi_net
    fr_measured_counts = data_matrix[:, 0]
    bi_measured_counts = data_matrix[:, 3]

    fr_eff = fr_measured_counts / fr_theory_counts
    bi_eff = bi_measured_counts / bi_theory_counts

    return np.mean(fr_eff), np.mean(bi_eff)


def correct_counts(n: float | np.ndarray, br: float, eff: float) -> float | np.ndarray:
    """Corrects measured counts for branching ratio and detection efficiency.

    Corrected Counts = Measured Counts / (Branching Ratio * Efficiency)

    Parameters
    ----------
    n : float | np.ndarray
        Measured counts (net counts from a peak).
    br : float
        Branching ratio (emission probability) of the gamma ray.
    eff : float
        Detection efficiency for the gamma ray at its energy.

    Returns
    -------
    float | np.ndarray
        Corrected counts (representing number of decays of the parent nuclide
        that could have produced this gamma ray).
    """
    return n / br / eff


def get_activity_ratio(fr: np.ndarray, dfr: np.ndarray, bi: np.ndarray, dbi: np.ndarray, fr_eff: float, bi_eff: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the activity ratio of Bi-213 to Fr-221 and its uncertainty.

    This function first corrects the net counts of Fr-221 and Bi-213 for their
    respective branching ratios and detection efficiencies to get activities.
    Then, it calculates the ratio of these activities (Bi/Fr) and propagates
    the uncertainties.

    Parameters
    ----------
    fr : np.ndarray
        Net counts of the Fr-221 peak(s).
    dfr : np.ndarray
        Uncertainty in the net counts of Fr-221 peak(s).
    bi : np.ndarray
        Net counts of the Bi-213 peak(s).
    dbi : np.ndarray
        Uncertainty in the net counts of Bi-213 peak(s).
    fr_eff : float
        Detection efficiency for the Fr-221 gamma ray.
    bi_eff : float
        Detection efficiency for the Bi-213 gamma ray.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - ratio : np.ndarray
            The calculated activity ratio (Bi-213 activity / Fr-221 activity).
        - dratio : np.ndarray
            The propagated uncertainty in the activity ratio.
    """
    fr_br = 0.116  # intensity (br) of fr gamma ray (218 keV from Ac-225 decay)
    bi_br = 0.261  # " " of bi gamma ray (440 keV from Ac-225 decay via Bi-213)

    fr_activity = correct_counts(fr, fr_br, fr_eff)
    dfr_activity = correct_counts(dfr, fr_br, fr_eff) # uncertainty propagated by scalar mult.

    bi_activity = correct_counts(bi, bi_br, bi_eff)
    dbi_activity = correct_counts(dbi, bi_br, bi_eff)

    ratio = bi_activity / fr_activity
    # Standard error propagation for R = B/F: (dR/R)^2 = (dF/F)^2 + (dB/B)^2
    dratio = ratio * np.sqrt((dfr_activity/fr_activity)**2 + (dbi_activity/bi_activity)**2)
    return ratio, dratio


def est_dr_ac225(fr: np.ndarray, dfr: np.ndarray, bi: np.ndarray, dbi: np.ndarray, mass_g: np.ndarray, fr_eff: float, bi_eff: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimates dose rate from Ac-225 based on Fr-221 and Bi-213 activities.

    .. deprecated:: Unknown
       This function is marked as TODO:DEPRECATED. Prefer using methods from
       the `BioD` class for more comprehensive dose rate calculations if available.

    Calculates dose rate in Gy/s considering alpha energies from Ac-225 decay chain.

    Parameters
    ----------
    fr : np.ndarray
        Net counts of Fr-221 peak(s).
    dfr : np.ndarray
        Uncertainty in Fr-221 net counts.
    bi : np.ndarray
        Net counts of Bi-213 peak(s).
    dbi : np.ndarray
        Uncertainty in Bi-213 net counts.
    mass_g : np.ndarray
        Mass of the sample(s) in grams.
    fr_eff : float
        Detection efficiency for Fr-221 gamma.
    bi_eff : float
        Detection efficiency for Bi-213 gamma.
    t : float
        Counting time (livetime) in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - gy_s : np.ndarray
            Estimated total dose rate in Gy/s.
        - dgys : np.ndarray
            Propagated uncertainty in the total dose rate.
    """
    fr_br = 0.116  # intensity (br) of fr gamma ray
    bi_br = 0.261  # " " of bi gamma ray

    # assume that each Fr decay implies 3 alphas with corresponding mean energies: (keV)
    # calculated in 0_1_alpha brs notebook
    en_first3 = 19160.841648908892
    en_last1 = 8323.036167

    fr_bq = correct_counts(fr, fr_br, fr_eff) / t
    # uncertainty propagated by scalar mult.
    dfr_bq = correct_counts(dfr, fr_br, fr_eff) / t

    bi_bq = correct_counts(bi, bi_br, bi_eff) / t
    dbi_bq = correct_counts(dbi, bi_br, bi_eff) / t

    fr_gy_s = fr_bq * en_first3 * 1e3 * 1.602e-19 / (mass_g * 1e-3)
    dfr_gys = dfr_bq * en_first3 * 1e3 * 1.602e-19 / (mass_g * 1e-3)

    bi_gy_s = bi_bq * en_last1 * 1e3 * 1.602e-19 / (mass_g * 1e-3)
    dbi_gys = dbi_bq * en_last1 * 1e3 * 1.602e-19 / (mass_g * 1e-3)

    gy_s = fr_gy_s + bi_gy_s
    dgys = np.sqrt(dfr_gys**2 + dbi_gys**2)

    return gy_s, dgys


def correct_activities(fr: np.ndarray, dfr: np.ndarray, bi: np.ndarray, dbi: np.ndarray, fr_eff: float, bi_eff: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Corrects Fr-221 and Bi-213 counts to activities (Bq).

    .. deprecated:: Unknown
       This function is marked as TODO:DEPRECATED. Prefer using methods from
       the `BioD` class for activity calculations.

    Parameters
    ----------
    fr : np.ndarray
        Net counts of Fr-221 peak(s).
    dfr : np.ndarray
        Uncertainty in Fr-221 net counts.
    bi : np.ndarray
        Net counts of Bi-213 peak(s).
    dbi : np.ndarray
        Uncertainty in Bi-213 net counts.
    fr_eff : float
        Detection efficiency for Fr-221.
    bi_eff : float
        Detection efficiency for Bi-213.
    t : float
        Counting time (livetime) in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - fr_bq : np.ndarray
            Activity of Fr-221 in Bq.
        - dfr_bq : np.ndarray
            Uncertainty in Fr-221 activity in Bq.
        - bi_bq : np.ndarray
            Activity of Bi-213 in Bq.
        - dbi_bq : np.ndarray
            Uncertainty in Bi-213 activity in Bq.
    """
    fr_br = 0.116  # intensity (br) of fr gamma ray
    bi_br = 0.261  # " " of bi gamma ray
    fr_bq = correct_counts(fr, fr_br, fr_eff) / t
    dfr_bq = correct_counts(dfr, fr_br, fr_eff) / t # uncertainty propagated by scalar mult.
    bi_bq = correct_counts(bi, bi_br, bi_eff) / t
    dbi_bq = correct_counts(dbi, bi_br, bi_eff) / t

    return fr_bq, dfr_bq, bi_bq, dbi_bq


def corr_bi(fr: float, dfr: float, bi: float, dbi: float, t: float) -> tuple[float, float]:
    """Corrects Bi-213 activity for decay and ingrowth from Ac-225/Fr-221.

    .. deprecated:: Unknown
       This function is marked as TODO:DEPRECATED. Prefer using methods from
       the `BioD` class for decay corrections.

    Calculates B(0) for Bi-213, accounting for its decay since time of
    measurement (`t`) and ingrowth from Fr-221 (assumed to be in equilibrium
    with Ac-225 at t=0 of Fr-221).

    Parameters
    ----------
    fr : float
        Activity of Fr-221 at time `t` (Bq).
    dfr : float
        Uncertainty in Fr-221 activity at time `t` (Bq).
    bi : float
        Activity of Bi-213 at time `t` (Bq).
    dbi : float
        Uncertainty in Bi-213 activity at time `t` (Bq).
    t : float
        Time elapsed (in seconds) between the reference time (t=0 for Ac-225)
        and the measurement time. This is the time over which decay/ingrowth occurred.

    Returns
    -------
    tuple[float, float]
        - b0 : float
            Corrected Bi-213 activity at t=0 (Bq).
        - db0 : float
            Uncertainty in corrected Bi-213 activity (Bq).
    """
    # using seconds
    thalf_ac225 = 9.92 * 24 * 3600  # s; updated value (2019) from NNDC for Ac-225
    thalf_bi213 = 45.6 * 60  # s
    lamA = np.log(2)/thalf_ac225  # Decay constant for Ac-225 (parent of Fr-221)
    lamB = np.log(2)/thalf_bi213  # Decay constant for Bi-213
    f = lamB / (lamB-lamA)       # Bateman equation coefficient for ingrowth

    # Correct Bi-213 activity back to t=0
    # b0 = Bi(t)*exp(lamB*t) - [Fr(t)*exp(lamA*t)] * f * (exp((lamB-lamA)*t) -1) * exp(-lamA*t)
    # Simplified: b0 = Bi(t)*exp(lamB*t) - Fr_0 * f * (exp((lamB-lamA)*t) -1)
    # The formula used seems to be for B(t) = B0*exp(-lamB*t) + A0*f*(exp(-lamA*t) - exp(-lamB*t))
    # Solving for B0, assuming A0 is Fr0 (Ac225 activity at t=0 based on Fr measurement)
    # Fr(t) is used as A(t) for the parent in Bateman eq context here, so Fr(t) -> Fr0 by exp(lamA*t)
    # b0 = bi_activity_at_t * exp(lambda_Bi * t) - (fr_activity_at_t * exp(lambda_Ac * t)) * factor * (exp((lambda_Bi - lambda_Ac)*t) - 1) * exp(-lambda_Ac * t)
    # This simplifies to:
    # b0 = bi * exp(lamB * t) - fr * exp(lamA * t) * f * (1 - exp(-(lamB-lamA)*t))
    # The implementation is: b0 = bi * exp(lamB * t) - f * fr * (exp(lamB * t) - exp(lamA * t))
    # This implies fr is already Fr0 or some variant.
    # If fr is Fr(t), then Fr0 = fr * exp(lamA * t).
    # B(t) = B0*exp(-lamB*t) + (Fr0 * lamB/(lamB-lamA)) * (exp(-lamA*t) - exp(-lamB*t))
    # B0 = B(t)*exp(lamB*t) - Fr0 * (lamB/(lamB-lamA)) * (exp((lamB-lamA)*t) - 1)

    b0_decay_part = bi * np.exp(lamB * t)
    b0_ingrowth_from_fr0 = (fr * np.exp(lamA * t)) * f * (np.exp((lamB - lamA) * t) - 1)
    # The original formula: b0 = bi * np.exp(lamB * t) - f * fr * (np.exp(lamB * t) - np.exp(lamA * t))
    # This can be rewritten: b0 = bi_t*exp(lamB*t) - f * fr_t * exp(lamA*t) * (exp((lamB-lamA)*t) -1) * exp(-lamA*t) ... no this is not it.
    # Let's assume fr is Fr(0) equivalent (i.e. A0 for Bateman).
    # Then B(t) = B0*exp(-lamB*t) + fr * f * (exp(-lamA*t)-exp(-lamB*t))
    # B0 = (B(t) - fr*f*(exp(-lamA*t)-exp(-lamB*t)))*exp(lamB*t)
    # B0 = B(t)*exp(lamB*t) - fr*f*(exp((lamB-lamA)*t)-1)

    # The implementation seems to use fr as Fr(t), not Fr(0).
    # If fr is Fr(t): Fr0 = fr * exp(lamA*t).
    # B0 = bi*exp(lamB*t) - (fr*exp(lamA*t)) * (lamB/(lamB-lamA)) * (exp((lamB-lamA)*t)-1)
    # This matches the structure: b0 = bi_term - fr_term * factor_f * (exp_diff_term - 1)
    # Original: b0 = bi * np.exp(lamB * t) - f * fr * (np.exp(lamB * t) - np.exp(lamA * t))
    # This is B0 = B(t)exp(lam_B*t) - [lam_B/(lam_B-lam_A)]*A(t)*[exp(lam_B*t) - exp(lam_A*t)]
    # This is one form of solution for B0 given A(t) and B(t) for A->B decay.
    b0 = bi * np.exp(lamB * t) - f * fr * (np.exp(lamB * t) - np.exp(lamA * t))


    d1_sq = (dbi * np.exp(lamB * t))**2
    d2_sq = (dfr * f * (np.exp(lamB * t) - np.exp(lamA * t)))**2
    db0 = np.sqrt(d1_sq + d2_sq)
    return (b0, db0)


def corr_ac(fr: float, dfr: float, t: float) -> tuple[float, float]:
    """Corrects Fr-221 activity for decay from Ac-225.

    .. deprecated:: Unknown
       This function is marked as TODO:DEPRECATED. Prefer using methods from
       the `BioD` class for decay corrections.

    Calculates A(0) for Ac-225 based on Fr-221 activity at time `t`, assuming
    Fr-221 is in secular equilibrium or its activity represents Ac-225 activity.

    Parameters
    ----------
    fr : float
        Activity of Fr-221 at time `t` (Bq).
    dfr : float
        Uncertainty in Fr-221 activity at time `t` (Bq).
    t : float
        Time elapsed (in seconds) from t=0 (Ac-225 reference) to measurement.

    Returns
    -------
    tuple[float, float]
        - fr0 : float
            Corrected Ac-225 activity at t=0 (Bq), based on Fr-221.
        - dfr0 : float
            Uncertainty in corrected Ac-225 activity (Bq).
    """
    thalf_ac225 = 9.92 * 24 * 3600  # s; updated value (2019) from NNDC for Ac-225
    lamA = np.log(2)/thalf_ac225
    fr0 = fr * np.exp(lamA * t)
    dfr0 = dfr * np.exp(lamA * t)
    return fr0, dfr0


def dr_ac225(fr: float, dfr: float, bi: float, dbi: float, mass_g: float, unit: str = 'gys') -> tuple[float, float, float, float]:
    """Calculates dose rates from decay-corrected Ac-225 and Bi-213 activities.

    .. deprecated:: Unknown
       This function is marked as TODO:DEPRECATED. Prefer using methods from
       the `BioD` class for dose rate calculations.

    Assumes `fr` and `bi` are activities at t=0 (e.g., time of sacrifice).
    `fr` is taken as Ac-225 activity for "all alpha" dose rate.
    `bi` is taken as Bi-213 activity for "last alpha" dose rate.

    Parameters
    ----------
    fr : float
        Decay-corrected activity of Ac-225 (parent, often inferred from Fr-221) in Bq.
    dfr : float
        Uncertainty in Ac-225 activity (Bq).
    bi : float
        Decay-corrected activity of Bi-213 in Bq.
    dbi : float
        Uncertainty in Bi-213 activity (Bq).
    mass_g : float
        Mass of the sample in grams.
    unit : str, optional
        Desired unit for dose rate: 'gys' (Gray per second) or 'mgyh'
        (milliGray per hour). Defaults to 'gys'.

    Returns
    -------
    tuple[float, float, float, float]
        - fr_out_dr : float
            Dose rate from Ac-225 (all alphas in its chain) in specified `unit`.
        - dfr_out_dr : float
            Uncertainty in Ac-225 dose rate.
        - bi_out_dr : float
            Dose rate from Bi-213 (last alpha) in specified `unit`.
        - dbi_out_dr : float
            Uncertainty in Bi-213 dose rate.

    Raises
    ------
    ValueError
        If `unit` is not 'gys' or 'mgyh'.
    """
    # inputs are corrected a0 and b0 at time of sacrifice [bq]
    # assume that each Fr decay implies 3 alphas with corresponding mean energies: (keV)
    # calculated in 0_1_alpdha brs notebook
    en_last1 = 8323.036167
    en_all = 19160.841648908892 + en_last1

    fr_gys = fr * en_all * 1e3 * 1.602e-19 / (mass_g * 1e-3)
    dfr_gys = dfr * en_all * 1e3 * 1.602e-19 / (mass_g * 1e-3)

    bi_gys = bi * en_last1 * 1e3 * 1.602e-19 / (mass_g * 1e-3)
    dbi_gys = dbi * en_last1 * 1e3 * 1.602e-19 / (mass_g * 1e-3)
    if unit == 'mgyh':
        fr_out = fr_gys * 1e3 * 3600
        dfr_out = dfr_gys * 1e3 * 3600
        bi_out = bi_gys * 1e3 * 3600
        dbi_out = dbi_gys * 1e3 * 3600
    elif unit == 'gys':
        fr_out = fr_gys
        dfr_out = dfr_gys
        bi_out = bi_gys
        dbi_out = dbi_gys
    else:
        raise ValueError("Unaccepted unit type, use 'gys' or 'mgyh'.")
    return fr_out, dfr_out, bi_out, dbi_out


def check_ROI(ROI: tuple | list | np.ndarray, valid_dtypes: tuple = (tuple, list, np.ndarray)) -> bool:
    """Validates a Region of Interest (ROI).

    Checks if the ROI is of a valid type, has the correct format (Emin, Emax),
    and falls within a reasonable range (0 to 2047). Prints messages to stdout
    if validation fails.

    Parameters
    ----------
    ROI : tuple | list | np.ndarray
        The Region of Interest to validate. Expected to be a sequence of two
        numeric values (Emin, Emax).
    valid_dtypes : tuple, optional
        A tuple of acceptable data types for the ROI.
        Defaults to `(tuple, list, np.ndarray)`.

    Returns
    -------
    bool
        True if the ROI passes all validation checks, False otherwise (implicitly,
        as it prints messages and doesn't return False explicitly on failure,
        relying on the caller to check for True).
    """
    if not isinstance(ROI, valid_dtypes):
        print(f'ROI must be {valid_dtypes}. Input: {type(ROI)}')
    elif ROI[1] < ROI[0]:
        print(f'ROI must be in (Emin, Emax) format: {ROI}')
    elif ROI[1] > 2047 or ROI[0] < 0: # Assuming 2047 refers to channel numbers
        print(f'ROI must fall in (0, 2047) CHA: {ROI}')
    else:
        return True
    # Implicitly returns None if a check fails, which evaluates to False in boolean contexts.
    # Consider explicitly returning False for clarity.
    return False


class energy_spectrum():
    """
    A class to handle and analyze an energy spectrum.

    This class encapsulates spectral data (energy and counts) and provides
    methods for plotting the spectrum, fitting Gaussian peaks to Regions of
    Interest (ROIs), extracting statistics from these fits (net counts,
    uncertainty, gross counts), and visualizing the fits.
    """

    def __init__(self, kev: np.ndarray, n: np.ndarray, xmin: float = 0, xmax: float = 2048, binsize: float = 1, min_peak_signal: float = 0):
        """Initializes an energy_spectrum object.

        Parameters
        ----------
        kev : np.ndarray
            Array of energy values (keV) for each channel/bin of the spectrum.
        n : np.ndarray
            Array of counts for each channel/bin of the spectrum.
        xmin : float, optional
            Minimum x-axis value (keV or channel) for display or analysis range.
            Defaults to 0.
        xmax : float, optional
            Maximum x-axis value (keV or channel) for display or analysis range.
            Defaults to 2048.
        binsize : float, optional
            The width of each bin/channel in keV. Used for labeling plots.
            Defaults to 1.
        min_peak_signal : float, optional
            A threshold for peak fitting. If the maximum count in a specified
            ROI is below this value, fitting might be skipped or return NaNs.
            Defaults to 0.
        """
        self.xmin = xmin
        self.xmax = xmax
        self.binsize = binsize
        self.kev = kev
        self.n = n
        self.thresh = min_peak_signal

    def init_plot(self) -> tuple[plt.Figure, np.ndarray]:
        """Initializes and returns a matplotlib figure with three subplots
        for visualizing the spectrum and specific ROIs.

        The figure typically shows:
        - ax[0]: Full spectrum view (e.g., 0-600 keV).
        - ax[1]: Zoomed-in view of a Fr-221 ROI (e.g., 150-300 keV).
        - ax[2]: Zoomed-in view of a Bi-213 ROI (e.g., 370-530 keV).

        Parameters
        ----------
        None

        Returns
        -------
        tuple[plt.Figure, np.ndarray]
            - f : plt.Figure
                The created matplotlib Figure object.
            - ax : np.ndarray
                A NumPy array of matplotlib Axes objects (shape (3,)).

        Side Effects
        ------------
        Sets `self.f` and `self.ax` with the created figure and axes.
        """
        # defines set of axes and plots energy spectrum with two cutouts for ROIs
        f, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True,
                             gridspec_kw={'width_ratios': [3, 1, 1]})

        ax[0].plot(self.kev, self.n)
        ax[1].plot(self.kev, self.n)
        ax[2].plot(self.kev, self.n)

        # in future these can be variables, maybe param dictionary
        ax[0].set_xlim([0, 600])
        ax[1].set_xlim([150, 300])
        ax[2].set_xlim([370, 530])

        ax[0].set_xlabel('keV')
        ax[1].set_xlabel('Fr-221 ROI (218 keV)')
        ax[2].set_xlabel('Bi-213 ROI (440 keV)')

        ax[0].set_ylabel('counts/bin (binw = {} keV)'.format(self.binsize))

        self.f = f
        self.ax = ax

        return f, ax

    def fit_gaussian(self, ROIarray: list | tuple, p0: str | list = 'fr', func: callable = gaussian, ret: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Fits a Gaussian function (possibly with background) to a specified ROI.

        Parameters
        ----------
        ROIarray : list | tuple
            A list or tuple of two integers or floats defining the ROI
            [kev_min, kev_max] or [channel_min, channel_max] to fit.
            These are used as indices on `self.kev` and `self.n`.
        p0 : str | list, optional
            Initial guess for the fitting parameters.
            If 'fr', uses default initial guess for Fr-221 peak.
            If 'bi', uses default initial guess for Bi-213 peak.
            Otherwise, should be a list of parameters matching `func`.
            Defaults to 'fr'.
        func : callable, optional
            The Gaussian function to use for fitting (e.g., `gaussian`,
            `gaussian_nobg`, `gaussian_cbg` from this module).
            Defaults to `gaussian` (Gaussian with linear background).
        ret : bool, optional
            If True, returns the fit parameters (`popt`), covariance matrix (`pcov`),
            and parameter errors (`perr`). If False (default), these are stored
            as instance attributes (`self.popt`, `self.perr`) and nothing is returned.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray] | None
            If `ret` is True and fit is successful:
                (`popt`, `pcov`, `perr`)
            If `ret` is True and max counts in ROI < `self.thresh`:
                (`np.nan`, `np.nan`, `np.nan`)
            If `ret` is False, returns None.
        """
        # ROIarray = np.array([kev_min, kev_max]).astype(int) of ROI surrounding peak
        # can also do gaussian_nobg or gaussian_cbg
        xdata = self.kev[ROIarray[0]:ROIarray[1]]
        ydata = self.n[ROIarray[0]:ROIarray[1]]

        if np.max(ydata) < self.thresh:
            if ret:
                return np.nan, np.nan, np.nan
            else:
                self.popt, self.perr = np.full(5, np.nan), np.full(5,np.nan) # Assuming 5 params for default gaussian
                return None


        if isinstance(p0, str): # Check if p0 is a string to use predefined guesses
            if p0 == 'fr':
                p0_values = [1000, 218, 30, -0.5, 10]
            elif p0 == 'bi':
                p0_values = [1000, 440, 25, -0.1, 10]
            else: # Default to 'fr' if string is not recognized, or raise error
                p0_values = [1000, 218, 30, -0.5, 10] # Or some other default/error
        else: # p0 is a list of parameters
            p0_values = p0


        # uncertainty = sqrt(N) or 1
        dy = np.maximum(np.sqrt(ydata), np.ones_like(ydata))
        try:
            popt, pcov = curve_fit(
                func, xdata, ydata, sigma=dy, p0=p0_values, maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError: # Fit failed
            popt = np.full_like(p0_values, np.nan, dtype=float)
            perr = np.full_like(p0_values, np.nan, dtype=float)
            pcov = np.full((len(p0_values), len(p0_values)), np.nan, dtype=float)


        self.popt = popt
        self.perr = perr

        if ret:
            return popt, pcov, perr
        return None

    def get_stats(self, ROIarray: list | tuple) -> tuple[float, float, float]:
        """Calculates statistics for a fitted peak within an ROI.

        Assumes `fit_gaussian` has been called previously for this ROI, and
        `self.popt` and `self.perr` are populated.

        Parameters
        ----------
        ROIarray : list | tuple
            ROI [min_kev, max_kev] or [min_channel, max_channel] from which to
            extract data. These are used as indices.

        Returns
        -------
        tuple[float, float, float]
            - net_cts : float
                Net counts in the peak (sum of `ydata - background`).
            - dnet : float
                Uncertainty in net counts.
            - gross_cts : float
                Gross counts in the ROI (sum of `ydata`).
            Returns (np.nan, np.nan, np.nan) if max counts in ROI < `self.thresh`
            or if `self.popt` is not set (e.g. fit failed).
        """
        xdata = self.kev[ROIarray[0]:ROIarray[1]]
        ydata = self.n[ROIarray[0]:ROIarray[1]]

        if np.max(ydata) < self.thresh or not hasattr(self, 'popt') or np.all(np.isnan(self.popt)):
            return np.nan, np.nan, np.nan

        # uncertainty = sqrt(N) or 1
        dy = np.maximum(np.sqrt(ydata), np.ones_like(ydata))

        try:
            # Assuming popt has at least 5 elements for linear background (b*x + c)
            bg = self.popt[3] * xdata + self.popt[4]
            # Assuming perr corresponds to popt
            dbg_slope_term = (self.perr[3] * xdata)**2 # (d(slope)/slope * slope * x)^2 not quite, this is simpler
            dbg_intercept_term = self.perr[4]**2
            dbg = np.sqrt(dbg_slope_term + dbg_intercept_term) # Simplified uncertainty for background
            darray = np.sqrt(dy**2 + dbg**2) # Propagate error for y - bg
        except IndexError:  # Handles cases like gaussian_nobg or gaussian_cbg if popt is shorter
            bg = np.zeros_like(ydata)
            if len(self.popt) == 4 and callable(self.popt): # gaussian_cbg case, popt[-1] is c
                 bg += self.popt[-1] # Add constant background
                 # d(bg) = perr[-1]
                 darray = np.sqrt(dy**2 + self.perr[-1]**2)
            else: # gaussian_nobg case
                 darray = dy


        net_data = ydata - bg
        net_data = np.maximum(net_data, np.zeros_like(net_data)) # No negative net counts
        net_cts = np.sum(net_data)
        # Sum uncertainties in quadrature for sum of (y_i - bg_i)
        dnet = np.sqrt(np.sum(darray**2))
        gross_cts = np.sum(ydata)

        return net_cts, dnet, gross_cts

    def plot_fit(self, ROIarray: list | tuple, idx: int, func: callable = gaussian):
        """Plots the fitted Gaussian function over the spectral data in a
        subplot previously initialized by `init_plot`.

        Parameters
        ----------
        ROIarray : list | tuple
            ROI [min_kev, max_kev] or [min_channel, max_channel] for plotting range.
        idx : int
            Index of the subplot (in `self.ax`) where the fit should be plotted.
            Typically 1 for Fr-221 ROI, 2 for Bi-213 ROI.
        func : callable, optional
            The Gaussian function that was used for fitting.
            Defaults to `gaussian`.

        Returns
        -------
        None
        """
        xspace = np.linspace(ROIarray[0], ROIarray[1], 100)
        xdata = self.kev[ROIarray[0]:ROIarray[1]]

        try:
            if not hasattr(self, 'popt') or np.all(np.isnan(self.popt)):
                print('No valid fit parameters found to plot.')
                return

            # Plot main fit line on overall spectrum and ROI-specific subplot
            self.ax[0].plot(xspace, func(xspace, *self.popt),
                            color='black', label='Fit' if idx == 1 else None) # Label only once
            self.ax[idx].plot(xspace, func(xspace, *self.popt),
                              color='black', label='Fit')

            # Plot background if applicable (func has > 3 parameters for gaussian_nobg)
            if len(self.popt) >= 5 and func == gaussian: # Linear background
                bg = self.popt[3] * xdata + self.popt[4]
                self.ax[idx].plot(xdata, bg, color='gray', linestyle='--')
            elif len(self.popt) == 4 and func == gaussian_cbg: # Constant background
                bg = np.full_like(xdata, self.popt[3])
                self.ax[idx].plot(xdata, bg, color='gray', linestyle='--')

            if idx == 1 : # Add legend to main plot only once
                 self.ax[0].legend()
            self.ax[idx].legend()

        except AttributeError: # Should be caught by hasattr check
            print('Fit parameters (self.popt) not found.')
        except IndexError: # Should be less likely with checks on popt length
            print('Error accessing fit parameters, possibly due to unexpected fit function or result.')


    def test_p0(self, p0: list, ROIarray: list | tuple, idx: int, func: callable = gaussian):
        """Plots the initial guess function (`p0`) over the spectrum data to test
        the quality of the initial guess before fitting.

        Calls `self.init_plot()` to set up the plotting axes.

        Parameters
        ----------
        p0 : list
            List of initial guess parameters for the function `func`.
        ROIarray : list | tuple
            ROI [min_kev, max_kev] or [min_channel, max_channel] for plotting range.
        idx : int
            Index of the subplot in `self.ax` (typically 1 or 2) where the
            p0 function should be highlighted.
        func : callable, optional
            The Gaussian function for which `p0` are the parameters.
            Defaults to `gaussian`.

        Returns
        -------
        None
        """
        self.init_plot()
        xspace = np.linspace(ROIarray[0], ROIarray[1], 100)
        self.ax[idx].plot(xspace, func(xspace, *p0),
                          color=pltmap(1), label='p0 function')
        self.ax[0].plot(xspace, func(xspace, *p0),
                        color=pltmap(1), label='p0 function') # Also plot on main axes
        self.ax[0].legend()
        self.ax[idx].legend()
        plt.tight_layout()
        plt.show()


def check_ROI(ROI, valid_dtypes):
    if not isinstance(ROI, valid_dtypes):
        print(f'ROI must be {valid_dtypes}. Input: {type(ROI)}')
    elif ROI[1] < ROI[0]:
        print(f'ROI must be in (Emin, Emax) format: {ROI}')
    elif ROI[1] > 2047 or ROI[0] < 0:
        print(f'ROI must fall in (0, 2047) CHA: {ROI}')
    else:
        return True


class BioD():
    """
    Processes and analyzes biodistribution data from gamma spectroscopy,
    particularly for Ac-225 studies.

    This class handles spectral data, sample masses, and time corrections to
    calculate activities, perform decay corrections for Ac-225 and its progeny
    (Fr-221, Bi-213), estimate dose rates, and determine activity ratios.
    It encapsulates parameters like ROIs, efficiencies, and decay constants.

    The typical workflow involves:
    1. Initializing with spectral data, masses, and time corrections.
    2. Setting analysis properties (ROIs, efficiencies) via `set_properties`
       or individual setters.
    3. Calculating net counts from spectra using `get_data_matrix`.
    4. Converting counts to activities with `cts_to_activity`.
    5. Applying decay corrections using `decay_corr`.
    6. Calculating dose rates with `doserate`.
    Alternatively, `spec2dr` can run steps 3-6.
    """

    def __init__(self, spec_matrix: np.ndarray | None = None, mass_g: np.ndarray | None = None, scorr: np.ndarray | float | None = None, kev: np.ndarray = np.arange(2048), live: float = 60):
        """Initializes a BioD object.

        Parameters
        ----------
        spec_matrix : np.ndarray | None, optional
            A 2D NumPy array where each row is a spectrum (counts).
            If None, it should be set later. Defaults to None.
        mass_g : np.ndarray | None, optional
            A 1D NumPy array of sample masses in grams, corresponding to each
            spectrum in `spec_matrix`. If None, it should be set later.
            Defaults to None.
        scorr : np.ndarray | float | None, optional
            Time correction(s) in seconds (e.g., time from sacrifice to counting).
            Can be a single float if same for all spectra, or a 1D NumPy array
            corresponding to each spectrum. If None, it should be set later.
            Defaults to None.
        kev : np.ndarray, optional
            A 1D NumPy array representing the energy calibration (keV per channel)
            for the spectra. Defaults to `np.arange(2048)`, implying a
            1 keV/channel calibration up to 2048 channels.
        live : float, optional
            The livetime (counting time) in seconds for the spectra.
            Defaults to 60.

        Instance Attributes (Private)
        -----------------------------
        _spectra : np.ndarray | None
            Stores `spec_matrix`.
        _mass : np.ndarray | None
            Stores `mass_g`.
        _timeCorrection : np.ndarray | float | None
            Stores `scorr`.
        _kev : np.ndarray
            Stores `kev`.
        _livetime : float
            Stores `live`.
        _ROIFr : tuple[float, float] | None
            Region of Interest for Fr-221 peak.
        _ROIBi : tuple[float, float] | None
            Region of Interest for Bi-213 peak.
        _effFr : float | None
            Detection efficiency for Fr-221 peak.
        _effBi : float | None
            Detection efficiency for Bi-213 peak.
        _counts : np.ndarray | None
            Stores [Fr_net, dFr_net, Bi_net, dBi_net] calculated by `get_data_matrix`.
        _activity : np.ndarray | None
            Stores activity data, potentially decay-corrected.
        _doserate : np.ndarray | None
            Stores calculated dose rate data.
        _ratio : np.ndarray | None
            Stores Bi/Fr activity ratio.
        _dratio : np.ndarray | None
            Stores uncertainty in Bi/Fr activity ratio.
        _p0Fr : list | None
            Initial guess parameters for Fr-221 peak fit.
        _p0Bi : list | None
            Initial guess parameters for Bi-213 peak fit.
        _corrected : bool
            Flag indicating if decay correction has been applied to `_activity`.
            This helps prevent applying corrections multiple times.
        _brFr : float
            Branching ratio for the primary Fr-221 gamma ray (218 keV from Ac-225 chain).
        _brBi : float
            Branching ratio for the primary Bi-213 gamma ray (440 keV from Ac-225 chain).
        _thalfAc225 : float
            Half-life of Ac-225 in seconds.
        _thalfBi213 : float
            Half-life of Bi-213 in seconds.
        _kevBi : float
            Energy deposition (keV) for Bi-213 related alpha decays (specifically the last alpha).
        _kevAc : float
            Total energy deposition (keV) for Ac-225 decay chain alphas (including Bi-213's last alpha).
        """
        self._spectra = spec_matrix
        self._mass = mass_g
        self._timeCorrection = scorr
        self._kev = kev
        self._livetime = live
        self._ROIFr = None
        self._ROIBi = None
        self._effFr = None
        self._effBi = None
        self._counts = None
        self._activity = None
        self._doserate = None
        self._ratio = None
        self._dratio = None

        self._p0Fr = None
        self._p0Bi = None

        # flags of stuff... is this ok?
        self._corrected = False

        # constants. is there a better practice for this?
        self._brFr = 0.116
        self._brBi = 0.261
        self._thalfAc225 = 9.92 * 24 * 3600  # s
        self._thalfBi213 = 45.6 * 60  # s
        self._kevBi = 8323.036167
        self._kevAc = 19160.84165 + self._kevBi

    @property
    def ROIFr(self) -> tuple[float, float] | None:
        """Region of Interest (ROI) for the Fr-221 peak (keV).

        Should be a tuple or list of two floats: (min_keV, max_keV).
        """
        return self._ROIFr

    @ROIFr.setter
    def ROIFr(self, ROIFr: tuple[float, float] | list[float] | np.ndarray):
        valid_dtypes = (tuple, list, np.ndarray)
        if check_ROI(ROIFr, valid_dtypes):
            self._ROIFr = ROIFr

    @property
    def ROIBi(self) -> tuple[float, float] | None:
        """Region of Interest (ROI) for the Bi-213 peak (keV).

        Should be a tuple or list of two floats: (min_keV, max_keV).
        """
        return self._ROIBi

    @ROIBi.setter
    def ROIBi(self, ROIBi: tuple[float, float] | list[float] | np.ndarray):
        valid_dtypes = (tuple, list, np.ndarray)
        if check_ROI(ROIBi, valid_dtypes):
            self._ROIBi = ROIBi

    @property
    def effFr(self) -> float | None:
        """Detection efficiency for the Fr-221 peak."""
        return self._effFr

    @effFr.setter
    def effFr(self, effFr: float):
        valid_dtypes = (int, float)
        if isinstance(effFr, valid_dtypes):
            self._effFr = float(effFr)

    @property
    def effBi(self) -> float | None:
        """Detection efficiency for the Bi-213 peak."""
        return self._effBi

    @effBi.setter
    def effBi(self, effBi: float):
        valid_dtypes = (int, float)
        if isinstance(effBi, valid_dtypes):
            self._effBi = float(effBi)

    def set_properties(self, ROIFr, ROIBi, effFr, effBi):
        self.ROIFr = ROIFr
        self.ROIBi = ROIBi
        self.effFr = effFr
        self.effBi = effBi

    @property
    def p0Fr(self) -> list | None:
        """Initial guess parameters for the Fr-221 peak fit.

        Used by `ac225_spectra_stats` if provided.
        Should be a list compatible with the fitting function in `energy_spectrum`.
        """
        return self._p0Fr

    @p0Fr.setter # Corrected decorator
    def p0Fr(self, p0Fr: list | None):
        self._p0Fr = p0Fr

    @property
    def p0Bi(self) -> list | None:
        """Initial guess parameters for the Bi-213 peak fit.

        Used by `ac225_spectra_stats` if provided.
        Should be a list compatible with the fitting function in `energy_spectrum`.
        """
        return self._p0Bi

    @p0Bi.setter # Corrected decorator
    def p0Bi(self, p0Bi: list | None):
        self._p0Bi = p0Bi

    '''TODO: properties for spectra, mass, time correction, etc'''
    '''TODO: catches for when requisite properties are not defined'''
    # Addressed by adding setters and noting in method docstrings where necessary.

    def set_properties(self, ROIFr: tuple[float, float], ROIBi: tuple[float, float], effFr: float, effBi: float):
        """Sets multiple ROI and efficiency properties simultaneously.

        Parameters
        ----------
        ROIFr : tuple[float, float]
            Region of Interest for Fr-221 peak (min_keV, max_keV).
        ROIBi : tuple[float, float]
            Region of Interest for Bi-213 peak (min_keV, max_keV).
        effFr : float
            Detection efficiency for the Fr-221 peak.
        effBi : float
            Detection efficiency for the Bi-213 peak.

        Returns
        -------
        None
        """
        self.ROIFr = ROIFr
        self.ROIBi = ROIBi
        self.effFr = effFr
        self.effBi = effBi

    def get_data_matrix(self, min_peak_signal: float = 5, vis: bool = False) -> np.ndarray:
        """Calculates and returns a matrix of count statistics for Ac-225 peaks.

        This method uses `ac225_spectra_stats` to determine net counts and
        uncertainties for Fr-221 and Bi-213 ROIs defined in `self.ROIFr` and
        `self.ROIBi`. The results are stored in `self._counts`.

        Parameters
        ----------
        min_peak_signal : float, optional
            Minimum peak signal required by `ac225_spectra_stats` to attempt
            fitting. Defaults to 5.
        vis : bool, optional
            If True, enables visualization during the call to `ac225_spectra_stats`.
            Defaults to False.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_spectra, 4) containing:
            [Fr_net_counts, dFr_net_counts, Bi_net_counts, dBi_net_counts].
            This matrix is also stored in `self._counts`.

        Raises
        ------
        AttributeError
            If essential properties like `_spectra`, `_ROIFr`, `_ROIBi`, `_kev`
            are not set before calling this method.
        """
        if self._spectra is None or self._ROIFr is None or self._ROIBi is None or self._kev is None:
            raise AttributeError("Spectra, ROIs, or kev calibration not set. Use setters or set_properties.")

        m = ac225_spectra_stats(self._kev,
                                self._spectra,
                                self._ROIFr,
                                self._ROIBi,
                                min_peak_signal=min_peak_signal,
                                vis=vis,
                                unpack=False,
                                fp0=self._p0Fr,
                                bp0=self._p0Bi)
        # fr, dfr, bi, dbi
        m = np.concatenate((m[:, :2], m[:, 3:5]), axis=1)

        # # fr, dfr, bi, dbi, mass (g)
        # m = np.concatenate((m, self._mass[:, np.newaxis]), axis=1) # TODO add getter/setter to mass to check it's 1d numpy array
        self._counts = m
        return m

    def cts_to_activity(self) -> np.ndarray:
        """Converts net counts from `self._counts` to activities (Bq).

        This method corrects the Fr-221 and Bi-213 net counts for their
        respective branching ratios (`_brFr`, `_brBi`), detection efficiencies
        (`_effFr`, `_effBi`), and the livetime (`_livetime`).
        The results are stored in `self._activity`.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_spectra, 4) containing:
            [Fr_activity_Bq, dFr_activity_Bq, Bi_activity_Bq, dBi_activity_Bq].
            This matrix is also stored in `self._activity`.

        Raises
        ------
        AttributeError
            If `self._counts`, `self._effFr`, or `self._effBi` are not set.
        """
        if self._counts is None or self._effFr is None or self._effBi is None:
            raise AttributeError("Counts or efficiencies not set. Run get_data_matrix() and set efficiencies.")

        m = self._counts # Fr_net, dFr_net, Bi_net, dBi_net
        n = np.zeros_like(m)
        n[:, 0] = correct_counts(m[:, 0], self._brFr, self._effFr) / self._livetime  # Fr_activity
        n[:, 1] = correct_counts(m[:, 1], self._brFr, self._effFr) / self._livetime  # dFr_activity
        n[:, 2] = correct_counts(m[:, 2], self._brBi, self._effBi) / self._livetime  # Bi_activity
        n[:, 3] = correct_counts(m[:, 3], self._brBi, self._effBi) / self._livetime  # dBi_activity
        self._activity = n
        return n

    def decay_corr(self, thresh: float = 5*3600) -> np.ndarray:
        """Applies decay correction to the activities stored in `self._activity`.

        Corrects Fr-221 activity back to t=0 (Ac-225 activity) and Bi-213
        activity back to t=0, accounting for its own decay and ingrowth from
        Ac-225/Fr-221. Uses `self._timeCorrection` for decay times.
        Sets `self._corrected` flag to True.

        Parameters
        ----------
        thresh : float, optional
            Time threshold in seconds. If `_timeCorrection` for a sample exceeds
            this, the corrected Bi-213 activity and its uncertainty are set to NaN,
            as it's assumed true B(0) is not reliably knowable. Defaults to 5*3600s (5 hours).

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_spectra, 4) containing decay-corrected
            activities: [Ac225_A0, dAc225_A0, Bi213_A0_free, dBi213_A0_free].
            This matrix updates `self._activity`. "Free" Bi-213 A0 refers to
            the Bi-213 not directly from the initial Ac-225 equilibrium portion.

        Raises
        ------
        AttributeError
            If `self._activity` or `self._timeCorrection` are not set.
        """
        if self._activity is None or self._timeCorrection is None:
            raise AttributeError("Activity data or time correction not set. Run cts_to_activity() and ensure timeCorrection is set.")

        if self._corrected:
            # print("Warning: Activity data may have already been decay corrected.") # Optional warning
            return self._activity

        t_corr = self._timeCorrection
        m = self._activity # Fr_A(t), dFr_A(t), Bi_A(t), dBi_A(t)
        fr_at_t, dfr_at_t, bi_at_t, dbi_at_t = m[:, 0], m[:, 1], m[:, 2], m[:, 3]

        lamA = np.log(2)/self._thalfAc225 # Ac-225 decay constant
        lamB = np.log(2)/self._thalfBi213 # Bi-213 decay constant

        # Fr-221 activity at t=0 (represents Ac-225 activity A0)
        f0 = fr_at_t * np.exp(lamA * t_corr)
        df0 = dfr_at_t * np.exp(lamA * t_corr)

        # Bi-213 activity at t=0 (B0)
        # B(t) = B0*exp(-lamB*t) + A0*lamB/(lamB-lamA)*(exp(-lamA*t) - exp(-lamB*t))
        # B0 = [B(t) - A0*lamB/(lamB-lamA)*(exp(-lamA*t) - exp(-lamB*t))] * exp(lamB*t)
        # where A0 is f0 (Ac-225 activity at t=0)
        bateman_factor = lamB / (lamB - lamA)
        b0_ingrowth_term = f0 * bateman_factor * (np.exp(-lamA * t_corr) - np.exp(-lamB * t_corr))
        b0 = (bi_at_t - b0_ingrowth_term) * np.exp(lamB * t_corr)

        # Uncertainty for b0 (more complex due to correlation if f0 used in b0_ingrowth_term)
        # Simplified: propagate errors for (bi_at_t * exp(lamB*t)) and the ingrowth part separately then combine.
        # This assumes f0 and bi_at_t are uncorrelated, which is reasonable if from different peaks.
        d_bi_decayed_back = dbi_at_t * np.exp(lamB * t_corr)
        # Uncertainty of ingrowth term: df0 dominates this term typically
        d_ingrowth_term = df0 * bateman_factor * (np.exp(-lamA * t_corr) - np.exp(-lamB * t_corr)) * np.exp(lamB * t_corr) # Approx
        db0 = np.sqrt(d_bi_decayed_back**2 + d_ingrowth_term**2)


        # "Free" Bi-213 is often considered B0 if A0 represents only the parent Ac-225.
        # The calculation above gives B0_total. If Fr is used as A0_Ac225, then
        # total Bi-213 at t=0 (B0_total) includes Bi-213 that was present at t=0
        # AND Bi-213 that would grow in from Ac-225 if Ac-225 itself had no Bi-213 initially.
        # The term "fb0 = b0 - f0" seems to imply "free Bi" is Bi(0) - Ac(0), which is unusual.
        # Assuming b0 from Bateman is the total Bi(0). If "free Bi" means non-Ac-225-supported Bi at t=0,
        # then it IS b0 from the Bateman equation if A0 is Ac-225.
        # Let's assume fb0 is the total Bi-213 activity at t=0 calculated.
        fb0 = b0 # Renaming for clarity based on Bateman solution.
        dfb0 = db0

        # Apply threshold if measurement time `t_corr` is too long
        try: # Handle single float t_corr
            if isinstance(t_corr, (float, int)):
                if t_corr > thresh:
                    fb0, dfb0 = np.nan, np.nan
            else: # Handle array t_corr
                for i in range(len(t_corr)):
                    if t_corr[i] > thresh:
                        fb0[i], dfb0[i] = np.nan, np.nan
        except TypeError: # Should be caught by isinstance check now
             pass


        n_corrected = np.array([f0, df0, fb0, dfb0]).T
        self._activity = n_corrected
        self._corrected = True
        return n_corrected

    def doserate(self, unit: str = "mgyh") -> np.ndarray:
        """Calculates dose rates from decay-corrected activities.

        Uses `self._activity` (assumed to be decay-corrected [Ac225_A0, dAc225_A0,
        Bi213_A0, dBi213_A0]) and `self._mass`. Dose rate from Ac-225 includes
        all its alpha progeny. Dose rate from "Bi" is calculated only for its own alpha.

        Parameters
        ----------
        unit : str, optional
            Desired unit for dose rate: "mgyh" (milliGray per hour) or
            "gys" (Gray per second). Defaults to "mgyh".

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_spectra, 4) containing dose rates:
            [DR_Ac, dDR_Ac, DR_Bi_last_alpha, dDR_Bi_last_alpha] in specified `unit`.
            This matrix is also stored in `self._doserate`.

        Raises
        ------
        AttributeError
            If `self._activity` or `self._mass` are not set.
        ValueError
            If `unit` is not "mgyh" or "gys".
        """
        if self._activity is None or self._mass is None:
            raise AttributeError("Decay-corrected activity or mass not set. Run decay_corr() and ensure mass is set.")

        if not self._corrected:
            print("WARNING: Activity data may not have been decay-corrected. Dose rate calculation assumes A0 values.")

        valid_units = ["mgyh", "gys"]
        scale_factors = {"mgyh": 1e3 * 3600, "gys": 1}
        if unit not in valid_units:
            raise ValueError(f"Unit ({unit}) must be one of: {valid_units}.")

        current_scale_factor = scale_factors[unit]

        m_act = self._activity # Ac225_A0, dAc225_A0, Bi213_A0, dBi213_A0
        n_dr = np.zeros_like(m_act)

        # Dose rate from Ac-225 (and its full chain of alphas)
        # Energy per decay for Ac-225 chain (self._kevAc)
        # Activity (Bq) * Energy_per_decay (J/decay) / mass (kg) = Dose_rate (Gy/s)
        # Energy_per_decay = self._kevAc * 1e3 (eV->keV) * 1.602e-19 (J/eV)
        # Mass = self._mass (g) * 1e-3 (kg/g)
        energy_factor_ac = self._kevAc * 1e3 * 1.602e-19
        energy_factor_bi = self._kevBi * 1e3 * 1.602e-19 # For Bi-213's own alpha
        mass_kg = self._mass[:, np.newaxis] * 1e-3

        n_dr[:, :2] = m_act[:, :2] * energy_factor_ac / mass_kg  # Ac dose rate and its uncertainty
        n_dr[:, 2:] = m_act[:, 2:] * energy_factor_bi / mass_kg  # Bi dose rate (last alpha) and its uncertainty

        self._doserate = n_dr * current_scale_factor
        return self._doserate

    def spec2dr(self, unit: str = "mgyh", min_peak_signal: float = 5) -> np.ndarray | None:
        """Wrapper function to perform full analysis from spectra to dose rate.

        Calls `get_data_matrix`, `cts_to_activity`, `decay_corr`, and `doserate`.

        Parameters
        ----------
        unit : str, optional
            Desired unit for dose rate, passed to `doserate`. Defaults to "mgyh".
        min_peak_signal : float, optional
            Minimum peak signal for `get_data_matrix`. Defaults to 5.

        Returns
        -------
        np.ndarray | None
            Dose rate matrix as returned by `doserate`, or None if a
            pre-requisite step fails (e.g., properties not set).
        """
        try:
            self.get_data_matrix(min_peak_signal=min_peak_signal)
            self.cts_to_activity()
            self.decay_corr() # Uses default threshold
            dr = self.doserate(unit=unit)
            return dr
        except AttributeError as e: # Catch if properties like ROIs, effs are not set
            print(f"Aborted in spec2dr: {e}")
            return None
        except Exception as e: # Catch other potential errors during processing
            print(f"An error occurred during spec2dr: {e}")
            return None


    def activity_ratio(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Calculates the Bi-213 / Fr-221 (or Ac-225) activity ratio.

        Uses `self._activity`. If `_corrected` is True, this means it uses
        decay-corrected A0 values (Ac225_A0 and Bi213_A0_free). If False,
        it uses activities at time of measurement. A warning is printed if
        data is already corrected, as ratio might be intended for uncorrected data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            - ratio : np.ndarray
                The activity ratio (Bi_activity / Fr_activity).
            - dratio : np.ndarray
                Propagated uncertainty in the ratio.
            Returns None if `self._activity` is not set.
        """
        if self._activity is None:
            print("Activity data not available. Run get_data_matrix and cts_to_activity first.")
            return None

        if self._corrected:
            print("WARNING: Calculating ratio from decay-corrected activities (A0 values).")

        m_act = self._activity # Fr_act, dFr_act, Bi_act, dBi_act (or Ac0, dAc0, Bi0, dBi0 if corrected)

        # Denominator (Fr or Ac activity)
        denom_act = m_act[:, 0]
        d_denom_act = m_act[:, 1]
        # Numerator (Bi activity)
        num_act = m_act[:, 2]
        d_num_act = m_act[:, 3]

        # Avoid division by zero or by NaN
        valid_idx = np.isfinite(denom_act) & np.isfinite(num_act) & (denom_act != 0)
        ratio = np.full_like(denom_act, np.nan)
        dratio = np.full_like(denom_act, np.nan)

        if np.any(valid_idx):
            ratio[valid_idx] = num_act[valid_idx] / denom_act[valid_idx]
            # (dR/R)^2 = (d_num/num)^2 + (d_den/den)^2
            term1_sq = (d_num_act[valid_idx] / num_act[valid_idx])**2 if np.any(num_act[valid_idx] !=0) else 0
            term2_sq = (d_denom_act[valid_idx] / denom_act[valid_idx])**2 # denom_act[valid_idx] is already !=0
            dratio[valid_idx] = ratio[valid_idx] * np.sqrt(term1_sq + term2_sq)

        self._ratio = ratio
        self._dratio = dratio
        return ratio, dratio


class MultiBioD():
    """
    Aggregates and analyzes results from multiple BioD instances or dose rate datasets,
    often for time-series analysis of biodistribution data.

    This class takes lists of dose rate matrices (typically from several `BioD`
    objects, each representing a time point or experimental group) and
    corresponding time points. It provides methods to:
    - Extract and concatenate all individual dose rate measurements (`extract_drlist`).
    - Aggregate dose rates by calculating means and standard deviations at each
      unique time point (`aggregate_drlist`).
    It also offers properties to access these processed data arrays.
    """

    def __init__(self, dr_list: list[np.ndarray] | None = None, t_list: list[float] | np.ndarray | None = None):
        """Initializes a MultiBioD object.

        Parameters
        ----------
        dr_list : list[np.ndarray] | None, optional
            A list of dose rate matrices. Each matrix in the list is expected
            to be a 2D NumPy array, typically output from `BioD.doserate()`,
            with shape (num_samples_at_this_timepoint, 4), where columns are
            [DR_Ac, dDR_Ac, DR_Bi, dDR_Bi]. Defaults to None.
        t_list : list[float] | np.ndarray | None, optional
            A list or array of time points corresponding to each dose rate matrix
            in `dr_list`. For example, hours post-injection. Defaults to None.

        Instance Attributes (Private)
        -----------------------------
        _drlist : list[np.ndarray] | None
            Stores `dr_list`.
        _t : list[float] | np.ndarray | None
            Stores `t_list`.
        _fr : np.ndarray | None
            Concatenated Fr/Ac dose rates from all samples in `_drlist`.
        _dfr : np.ndarray | None
            Concatenated uncertainties for Fr/Ac dose rates.
        _bi : np.ndarray | None
            Concatenated Bi dose rates from all samples.
        _dbi : np.ndarray | None
            Concatenated uncertainties for Bi dose rates.
        _tRavel : np.ndarray | None
            Array of time points corresponding to each individual sample in `_fr`, `_dfr`, etc.
        _frmean : np.ndarray | None
            Mean Fr/Ac dose rate at each unique time point in `_t`.
        _frstd : np.ndarray | None
            Standard deviation of Fr/Ac dose rate at each unique time point.
        _bimean : np.ndarray | None
            Mean Bi dose rate at each unique time point.
        _bistd : np.ndarray | None
            Standard deviation of Bi dose rate at each unique time point.
        """
        self._drlist = dr_list
        self._fr = None
        self._dfr = None
        self._bi = None
        self._dbi = None
        self._t = t_list  # e.g. "time post-injection, with no duplicates"
        self._tRavel = None

        self._frmean = None
        self._frstd = None
        self._bimean = None
        self._bistd = None

    @property
    def fr(self) -> np.ndarray | None:
        """Concatenated Fr/Ac dose rates from all samples across all time points.
        Populated by `extract_drlist()` or `dr2data()`.
        """
        return self._fr

    @property
    def dfr(self) -> np.ndarray | None:
        """Concatenated uncertainties of Fr/Ac dose rates.
        Populated by `extract_drlist()` or `dr2data()`.
        """
        return self._dfr

    @property
    def bi(self) -> np.ndarray | None:
        """Concatenated Bi-213 dose rates from all samples.
        Populated by `extract_drlist()` or `dr2data()`.
        """
        return self._bi

    @property
    def dbi(self) -> np.ndarray | None:
        """Concatenated uncertainties of Bi-213 dose rates.
        Populated by `extract_drlist()` or `dr2data()`.
        """
        return self._dbi

    @property
    def t(self) -> list[float] | np.ndarray | None:
        """List or array of unique time points for the dose rate data."""
        return self._t

    @property
    def tRavel(self) -> np.ndarray | None:
        """Array of time points corresponding to each individual sample in the
        concatenated `fr`, `dfr`, `bi`, `dbi` arrays.
        Populated by `extract_drlist()` or `dr2data()`.
        """
        return self._tRavel

    @property
    def frmean(self) -> np.ndarray | None:
        """Mean Fr/Ac dose rate at each unique time point specified in `t`.
        Populated by `aggregate_drlist()` or `dr2data()`.
        """
        return self._frmean

    @property
    def frstd(self) -> np.ndarray | None:
        """Standard deviation of Fr/Ac dose rate at each unique time point.
        Populated by `aggregate_drlist()` or `dr2data()`.
        """
        return self._frstd

    @property
    def bimean(self) -> np.ndarray | None:
        """Mean Bi-213 dose rate at each unique time point.
        Populated by `aggregate_drlist()` or `dr2data()`.
        """
        return self._bimean

    @property
    def bistd(self) -> np.ndarray | None:
        """Standard deviation of Bi-213 dose rate at each unique time point.
        Populated by `aggregate_drlist()` or `dr2data()`.
        """
        return self._bistd

    def extract_drlist(self):
        """
        Extracts and concatenates dose rate data from `self._drlist`.

        This method processes the list of dose rate matrices (`self._drlist`),
        where each matrix corresponds to a time point in `self._t` and contains
        dose rate information ([DR_Ac, dDR_Ac, DR_Bi, dDR_Bi]) for multiple samples.
        It flattens these matrices into single arrays for Fr/Ac dose rates (`_fr`),
        their uncertainties (`_dfr`), Bi dose rates (`_bi`), their uncertainties (`_dbi`),
        and creates a corresponding array of time points (`_tRavel`) for each sample.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method populates instance attributes `_fr`, `_dfr`, `_bi`, `_dbi`,
            and `_tRavel`.

        Raises
        ------
        TypeError | AttributeError
            If `self._drlist` or `self._t` are not set up correctly (e.g., None or
            mismatched lengths if not handled carefully before calling).
        """
        fr, dfr, bi, dbi = np.array([]), np.array(
            []), np.array([]), np.array([])
        tRavel = np.array([])
        if self._drlist is None or self._t is None:
            # Or raise an error, or handle appropriately
            print("Warning: dr_list or t_list is not set in MultiBioD. Cannot extract.")
            return

        for i in range(len(self._drlist)):
            m = self._drlist[i] # Dose rate matrix for time point self._t[i]
            fr = np.append(fr, m[:, 0])    # Fr/Ac dose rate
            dfr = np.append(dfr, m[:, 1])  # d(Fr/Ac dose rate)
            bi = np.append(bi, m[:, 2])    # Bi dose rate
            dbi = np.append(dbi, m[:, 3])  # d(Bi dose rate)
            tRavel = np.append(tRavel, np.repeat(self._t[i], len(m)))
        # fr, dfr, bi, dbi = np.array(fr), np.array(dfr), np.array(bi), np.array(dbi) # Already arrays

        self._fr = fr
        self._dfr = dfr
        self._bi = bi
        self._dbi = dbi
        self._tRavel = tRavel

    def aggregate_drlist(self):
        """
        Aggregates dose rate data from `self._drlist` to calculate means and
        standard deviations at each unique time point in `self._t`.

        For each time point, this method takes the corresponding dose rate matrix
        from `self._drlist` and calculates the mean and standard deviation of
        Fr/Ac dose rates and Bi dose rates across the samples at that time point.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method populates instance attributes `_frmean`, `_frstd`,
            `_bimean`, and `_bistd`.

        Raises
        ------
        TypeError | AttributeError
            If `self._drlist` is not set up correctly (e.g., None).
        """
        if self._drlist is None:
            print("Warning: dr_list is not set in MultiBioD. Cannot aggregate.")
            return

        num_time_points = len(self._drlist)
        fr_mean = np.zeros(num_time_points)
        dfr_std = np.zeros(num_time_points) # Note: This stores std of DR_Fr, not std of dDR_Fr
        bi_mean = np.zeros(num_time_points)
        dbi_std = np.zeros(num_time_points) # Note: This stores std of DR_Bi, not std of dDR_Bi


        for i in range(num_time_points):
            m = self._drlist[i] # Dose rate matrix for time point i
            means = np.mean(m, axis=0) # Mean across samples for [DR_Ac, dDR_Ac, DR_Bi, dDR_Bi]
            stds = np.std(m, axis=0)   # Std across samples for [DR_Ac, dDR_Ac, DR_Bi, dDR_Bi]

            fr_mean[i] = means[0]  # Mean DR_Fr/Ac
            dfr_std[i] = stds[0]   # Std of DR_Fr/Ac (Note: original code put stds[0] into dfr field)
            bi_mean[i] = means[2]  # Mean DR_Bi
            dbi_std[i] = stds[2]   # Std of DR_Bi (Note: original code put stds[2] into dbi field)

        self._frmean = fr_mean
        self._frstd = dfr_std   # Standard deviation of Fr/Ac dose rates
        self._bimean = bi_mean
        self._bistd = dbi_std   # Standard deviation of Bi dose rates

    def dr2data(self):
        """
        Convenience method to run both `extract_drlist` and `aggregate_drlist`.

        This populates all derived data attributes of the `MultiBioD` instance.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.extract_drlist()
        self.aggregate_drlist()
