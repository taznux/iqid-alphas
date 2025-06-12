import numpy as np

"""
This module provides functions for manipulating dose point kernels (DPKs)
and performing DPK convolution with iQID activity image stacks.

The functionalities include loading dose kernels from specific text file formats,
converting units (MeV to mGy), performing radial averaging on kernels,
padding kernels to match voxel sizes, and applying corrections related to
fast-decaying progeny in time-gated imaging scenarios.
"""


def load_txt_kernel(filename: str, dim: int, num_alpha_decays: float) -> np.ndarray:
    """Loads a 3D dose kernel from a .txt file into a NumPy array.

    The function assumes a specific file structure:
    - 5 header lines to be skipped.
    - Data columns are: x, y, z, Energy (E).
    The kernel is assumed to be cubic (dim x dim x dim).
    The output kernel values are normalized by `num_alpha_decays`.

    Parameters
    ----------
    filename : str
        The path to the .txt file containing the dose kernel data.
    dim : int
        The dimension of the cubic kernel (e.g., if kernel is 101x101x101, dim is 101).
    num_alpha_decays : float
        The total number of primary alpha particles simulated to generate the kernel.
        This is used for normalization, resulting in units like MeV/alpha_decay/voxel.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (dim, dim, dim) representing the dose kernel,
        with values in MeV/alpha_decay/voxel (or per source particle).
    """

    kernelData = np.genfromtxt(filename, delimiter=' ', skip_header=5)
    E = kernelData[:, -1]
    dims = (dim*np.ones(3)).astype(int)
    dose_kernel = E.reshape(dims[0], dims[1], dims[2])
    dose_kernel = dose_kernel/num_alpha_decays
    return (dose_kernel)


def mev_to_mgy(kernel: np.ndarray, vox_vol_m: float, dens_kgm: float = 1e3) -> np.ndarray:
    """Converts an energy deposition kernel from MeV (per source particle/event)
    per voxel to mGy (per source particle/event) per voxel.

    This conversion uses the voxel volume and material density.
    1 Gy = 1 J/kg.

    Parameters
    ----------
    kernel : np.ndarray
        The input 3D dose kernel, with values in MeV (e.g., MeV/source_particle).
        Shape can be (N, M, P).
    vox_vol_m : float
        The volume of a single voxel in cubic meters (m^3).
    dens_kgm : float, optional
        The density of the material (e.g., tissue) in kg/m^3.
        Defaults to 1000 kg/m^3 (density of water).

    Returns
    -------
    np.ndarray
        The dose kernel with values converted to mGy (e.g., mGy/source_particle).
        Shape is the same as the input `kernel`.
    """

    # mass_kg_px is mass per voxel in kg
    mass_kg_px = dens_kgm * vox_vol_m
    gy_kernel = kernel * 1e6 * 1.6021e-19 / mass_kg_px
    mgy_kernel = gy_kernel * 1e3
    return (mgy_kernel)


def radial_avg_kernel(kernel: np.ndarray, mode: str = "whole", bin_size: float = 0.5) -> np.ndarray | None:
    """Radially averages a 3D Monte Carlo dose kernel to reduce variance.

    This function calculates the average dose in concentric spherical shells
    (or rings, effectively, due to discretization) around the center of the kernel.
    It can return either a new 3D kernel where each voxel value is replaced
    by the average of its shell, or a 1D array representing the average dose
    as a function of radial distance.

    Parameters
    ----------
    kernel : np.ndarray
        The input 3D dose kernel, typically cubic (e.g., shape (N, N, N)).
        Assumes the center of the dose deposition is at the geometric center
        of the array.
    mode : str, optional
        Determines the output format:
        - "whole": Returns a 3D kernel of the same shape as input, where each
          voxel value is the radial average of its shell.
        - "segment": Returns a 1D array of average dose values for each
          radial bin.
        Defaults to "whole".
    bin_size : float, optional
        The thickness of each spherical shell (radial bin) used for averaging,
        in the same units as the voxel dimensions of the kernel (implicitly).
        For example, if kernel voxels are 1µm, bin_size=0.5 means shells are 0.5µm thick.
        Defaults to 0.5.

    Returns
    -------
    np.ndarray | None
        - If `mode` is "whole": A 3D NumPy array of the same shape as `kernel`,
          containing the radially averaged dose values.
        - If `mode` is "segment": A 1D NumPy array where each element is the
          average dose in a radial shell. The length is approximately `max_radius / 1`.
        - Returns `None` if an unsupported `mode` is specified.
    """

    if mode != 'whole' and mode != 'segment':
        print('Unsupported mode. Please select 3D "whole" or 1D "segment".')
        return (None)

    centerpt = len(kernel)//2
    a, b, c = kernel.shape

    # Grid of radial distances from the origin (centerpt)
    [X, Y, Z] = np.meshgrid(np.arange(a)-centerpt,
                            np.arange(b)-centerpt,
                            np.arange(c)-centerpt)
    R = np.sqrt(np.square(X) + np.square(Y) + np.square(Z))

    # Set spatial resolution of computation
    rad = np.arange(0, np.max(R), 1)

    segment_avg_rad = np.zeros(len(rad))
    kernel_avg_rad = np.zeros_like(kernel)
    idx = 0

    for i in rad:
        mask = (np.greater(R, i-bin_size) & np.less(R, i+bin_size))
        values = kernel[mask]
        segment_avg_rad[idx] = np.mean(values)
        kernel_avg_rad[mask] = np.mean(values)
        idx += 1

    if mode == 'whole':
        return (kernel_avg_rad)
    elif mode == 'segment':
        return (segment_avg_rad)
    else:
        return (None)


def pad_kernel_to_vsize(kernel: np.ndarray, vox_xy: int, slice_z: int = 12) -> np.ndarray:
    """Pads a 3D dose kernel with zeros so its dimensions become integer
    multiples of specified target voxel sizes.

    This is useful for preparing a high-resolution kernel (e.g., 1 µm isotropic)
    to be binned down to a coarser voxel grid (e.g., `vox_xy` µm in XY, `slice_z` µm in Z)
    using operations like `bin_ndarray` which typically require the input array
    dimensions to be divisible by the binning factors.

    Padding is applied symmetrically where possible.

    Parameters
    ----------
    kernel : np.ndarray
        The input 3D dose kernel (e.g., shape (N, M, P)). Assumed to have
        isotropic voxels (e.g., 1x1x1 µm^3) if not otherwise specified by context.
    vox_xy : int
        The target XY voxel size in the same units as the input kernel's voxel
        dimensions (typically µm). The kernel's X and Y dimensions will be
        padded to be multiples of this value.
    slice_z : int, optional
        The target Z voxel size (slice thickness) in the same units as the
        input kernel's voxel dimensions (typically µm). The kernel's Z dimension
        will be padded to be a multiple of this value. Defaults to 12.

    Returns
    -------
    np.ndarray
        The zero-padded kernel. Its dimensions will be slightly larger than
        the input kernel, such that they are integer multiples of `slice_z` (for Z)
        and `vox_xy` (for X and Y).
    """

    xy = np.round(vox_xy).astype(int)
    z = np.round(slice_z)

    rem_xy = np.shape(kernel)[1] % xy
    rem_z = np.shape(kernel)[0] % z
    pad_xy = int(xy - rem_xy)
    pad_z = int(z - rem_z)

    if rem_xy == 0:
        pad_xy = 0
    if rem_z == 0:
        pad_z = 0

    padded_kernel = np.pad(kernel,
                           ((pad_z//2, pad_z//2 + pad_z % 2),
                            (pad_xy//2, pad_xy//2 + pad_xy % 2),
                            (pad_xy//2, pad_xy//2 + pad_xy % 2)),
                           'constant')

    return padded_kernel


def get_pd(thalf: float, framerate: float) -> float:
    """Calculates the probability of a "double decay" event being missed
    within a single frame interval due to a fast-decaying progeny.

    This probability (`p`) represents the likelihood that if a parent nuclide
    decays, its short-lived progeny also decays before the next frame,
    potentially leading to missed counts or misinterpretation of events if
    the detection system cannot resolve these two decays.

    The derivation is often specific to the decay scheme and detection scenario,
    as mentioned "in accompanying manuscript."

    Parameters
    ----------
    thalf : float
        Half-life of the fast-decaying progeny radionuclide in seconds (s).
    framerate : float
        Frame rate of the imaging device in frames per second (fps).
        The frame duration `t1` is calculated as `1 / framerate`.

    Returns
    -------
    float
        The probability `p` (between 0 and 1) that the progeny decays within
        the same frame interval `t1` as its parent's observed decay.
        A higher value means more progeny decays are likely missed or merged.
    """
    lam = np.log(2)/thalf  # Decay constant of the progeny in s^-1
    t1 = 1 / framerate    # Frame duration in seconds
    # Formula for probability of decay within time t1: 1 - exp(-lam*t1)
    # The formula used: p = 1 - (1 - exp(-lam*t1)) / (lam*t1)
    # This is (1 - P(decay within t1)) / (lam*t1) if P is probability, which seems incorrect.
    # Let's assume the original formula from a source is:
    # P(progeny decays in [0, t1] | parent decayed at t=0 in [0, t1])
    # If parent decay is uniformly distributed in [0, t1], average time for progeny to decay is complex.
    # The provided formula p = 1 - (1/(lam*t1)) * (1 - np.exp(-lam * t1)) is used as-is.
    # This formula actually represents: 1 - (average lifetime of progeny that decay by t1) / t1, if they were all born at t=0.
    # Or, more likely: fraction of progeny that decay *after* time t1, if born uniformly in [0,t1].
    # The comment "Probability of double-decay, given observed parent decay" might imply a conditional probability.
    # For a progeny born at time t_birth, P(decay by t1) = 1 - exp(-lam * (t1 - t_birth)).
    # If parent decay (t_birth) is uniform in [0, t1], integral of P(decay by t1) dt_birth / t1
    # = integral_0^t1 (1-exp(-lam*(t1-u)))du / t1 = [u + (1/lam)*exp(-lam*(t1-u))]_0^t1 / t1
    # = (t1 - (1/lam)*(1 - exp(-lam*t1))) / t1 = 1 - (1 - exp(-lam*t1))/(lam*t1)
    # This is exactly the formula for `p`. So `p` is the probability that a progeny,
    # whose parent decayed uniformly within a frame, itself decays within that same frame.
    p = 1 - (1/(lam*t1)) * (1 - np.exp(-lam * t1))
    return p


def fr_corr_ac(framerate: float) -> tuple[float, float, float]:
    """Calculates a correction factor for Fr-221 counts in Ac-225 imaging
    due to the fast decay of At-217.

    At-217 is a progeny of Fr-221 (alpha decay branch) and has a very short
    half-life (32.3 ms). If Fr-221 decay is detected, there's a chance At-217
    also decays within the same imaging frame, potentially affecting event
    interpretation or counting if the system cannot distinguish them.
    This function calculates a scalar correction factor for the observed Fr-221 counts.

    The logic assumes a specific branching and detection scenario detailed elsewhere
    (e.g., "accompanying manuscript" or internal documentation).

    Parameters
    ----------
    framerate : float
        Frame rate of the imaging device in frames per second (fps).

    Returns
    -------
    tuple[float, float, float]
        - out : float
            The scalar correction factor (typically >= 1) to be multiplied by the
            observed Fr-221 counts to account for missed At-217 decays.
        - pd : float
            The probability of At-217 decaying within the same frame as its
            parent Fr-221, calculated by `get_pd`.
        - pf : float
            An intermediate probability factor used in the calculation of `out`.
            Its precise physical meaning depends on the underlying model assumptions.
    """
    thalf_at217 = 32.3 * 1e-3  # Half-life of At-217 in seconds
    pd = get_pd(thalf_at217, framerate) # Probability At-217 decays in same frame as Fr-221

    # pa, pf interpretation depends on the model (e.g., branching ratios, detection efficiencies)
    # Assuming Fr-221 has 4 main decay branches (alpha to At-217, and 3 EC branches to Rn-221)
    # If 'pa' is related to the probability of the At-217 branch not being "missed"
    # And 'pf' is a conditional probability related to detecting Fr-221 via one specific branch
    # given the possibility of missed At-217 decays.
    # This part is highly model-specific.
    pa = 0.25 * (1-pd) # Example: probability of observing an event from one of 4 branches if At-217 didn't obscure it.
    pf = 0.25 / (0.25 + 0.25 + 0.25 + pa)
    out = 1 + pd * pf
    return out, pd, pf
