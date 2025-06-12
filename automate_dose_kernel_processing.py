import os
import json
import logging
import numpy as np
from iqid.dpk import load_txt_kernel, mev_to_mgy, radial_avg_kernel, pad_kernel_to_vsize

"""
Automates the processing of dose point kernels (DPKs).

This script is configured via `config.json`. Its main purpose is to take a raw
dose kernel (typically from Monte Carlo simulations in a text file format),
process it through several steps, and save the resulting processed kernel as a
NumPy array file.

Workflow:
1.  Loads a raw dose kernel from a text file using `iqid.dpk.load_txt_kernel`.
    This kernel is usually in units like MeV/alpha_decay/voxel.
2.  Converts the kernel's units from MeV to mGy using `iqid.dpk.mev_to_mgy`,
    considering voxel volume and material density.
3.  Performs radial averaging on the kernel to reduce variance using
    `iqid.dpk.radial_avg_kernel`.
4.  Pads the kernel to dimensions suitable for specific voxel sizes (XY and Z)
    using `iqid.dpk.pad_kernel_to_vsize`.
5.  Saves the final processed 3D kernel as a `.npy` file.

Key inputs (from config.json under "automate_dose_kernel_processing"):
- `kernel_file`: Path to the raw dose kernel text file.
- `dim`: Dimension of the (assumed cubic) raw kernel.
- `num_alpha_decays`: Normalization factor for the raw kernel.
- `vox_vol_m`: Voxel volume in m^3 for unit conversion.
- `dens_kgm`: Material density in kg/m^3.
- `vox_xy`, `slice_z`: Target voxel dimensions for padding.
- `output_dir`: Directory to save the processed kernel.

Key outputs:
- `processed_kernel.npy`: The final processed dose kernel saved in the `output_dir`.

Logging of operations is performed to `automate_dose_kernel_processing.log`.
"""
import os
import json
import logging
import numpy as np
from iqid.dpk import load_txt_kernel, mev_to_mgy, radial_avg_kernel, pad_kernel_to_vsize

# Configure logging
logging.basicConfig(filename='automate_dose_kernel_processing.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s') # Corrected closing parenthesis

def process_dose_kernels(kernel_file: str, dim: int, num_alpha_decays: float,
                         vox_vol_m: float, dens_kgm: float,
                         vox_xy: int, slice_z: int, output_dir: str) -> None:
    """
    Loads, converts, averages, pads, and saves a dose kernel.

    Parameters
    ----------
    kernel_file : str
        Path to the raw dose kernel text file.
    dim : int
        Dimension of the cubic raw kernel.
    num_alpha_decays : float
        Normalization factor (number of simulated particles).
    vox_vol_m : float
        Voxel volume in cubic meters (m^3).
    dens_kgm : float
        Material density in kg/m^3.
    vox_xy : int
        Target XY voxel size for padding (in units consistent with kernel, e.g., µm).
    slice_z : int
        Target Z voxel size (slice thickness) for padding (e.g., µm).
    output_dir : str
        Directory to save the processed kernel.

    Returns
    -------
    None
        The processed kernel is saved to disk.

    Raises
    ------
    Exception
        Propagates exceptions from underlying DPK processing functions or file I/O.
    """
    try:
        logging.info(f"Loading dose kernel from: {kernel_file}")
        dose_kernel = load_txt_kernel(kernel_file, dim, num_alpha_decays)

        logging.info("Converting kernel from MeV to mGy.")
        mgy_kernel = mev_to_mgy(dose_kernel, vox_vol_m, dens_kgm)
        
        logging.info("Performing radial averaging on the kernel.")
        # Assuming mode="whole" and bin_size=0.5 are fixed for this automated script
        avg_kernel = radial_avg_kernel(mgy_kernel, mode="whole", bin_size=0.5)

        logging.info(f"Padding kernel to target voxel dimensions (XY: {vox_xy}, Z: {slice_z}).")
        padded_kernel = pad_kernel_to_vsize(avg_kernel, vox_xy, slice_z)
        
        logging.info(f"Saving processed kernel to: {output_dir}")
        save_processed_kernels(padded_kernel, output_dir)
        logging.info("Successfully processed and saved dose kernel.")
    except Exception as e:
        logging.error(f"Failed to process dose kernels: {e}", exc_info=True)
        raise

def save_processed_kernels(kernel: np.ndarray, output_dir: str) -> None:
    """Saves the processed kernel as a .npy file.

    Parameters
    ----------
    kernel : np.ndarray
        The processed dose kernel (3D NumPy array) to be saved.
    output_dir : str
        The directory where the kernel file (`processed_kernel.npy`) will be saved.
        This directory is created if it doesn't exist.

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
        save_path = os.path.join(output_dir, 'processed_kernel.npy')
        np.save(save_path, kernel)
        logging.info(f"Processed kernel saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save processed kernels: {e}", exc_info=True)
        raise

def main() -> None:
    """
    Main function to automate the processing of dose kernels.

    Workflow:
    1. Loads configuration parameters from `config.json`
       (from the "automate_dose_kernel_processing" section).
    2. Calls `process_dose_kernels` with these parameters to perform
       loading, unit conversion, radial averaging, padding, and saving.

    Parameters are sourced from `config.json`.
    Logs errors and progress to `automate_dose_kernel_processing.log`.
    """
    logging.info("Starting dose kernel processing workflow.")
    try:
        with open('config.json', 'r') as f:
            # Get the specific section for this script's parameters
            config = json.load(f)['automate_dose_kernel_processing']

        kernel_file = config['kernel_file']
        dim = config['automate_dose_kernel_processing']['dim']
        num_alpha_decays = config['automate_dose_kernel_processing']['num_alpha_decays']
        vox_vol_m = config['automate_dose_kernel_processing']['vox_vol_m']
        dens_kgm = config['automate_dose_kernel_processing']['dens_kgm']
        vox_xy = config['automate_dose_kernel_processing']['vox_xy']
        slice_z = config['automate_dose_kernel_processing']['slice_z']
        output_dir = config['automate_dose_kernel_processing']['output_dir']

        process_dose_kernels(kernel_file, dim, num_alpha_decays, vox_vol_m, dens_kgm, vox_xy, slice_z, output_dir)
    except Exception as e:
        logging.error("Failed to complete main dose kernel processing: %s", str(e))
        raise

if __name__ == "__main__":
    main()
