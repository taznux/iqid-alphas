# `config.json` Configuration File Documentation

This document describes the structure and parameters of the `config.json` file used to configure the iQID data processing pipeline.

## General Structure

The `config.json` file is organized into several main sections, each controlling different aspects of the processing pipeline or specific automation scripts. The current `config.json` primarily contains sections for each automation script. The structure outlined in `ucsf_ac225_iqid_processing_plan.md` provides a more comprehensive view, including planned top-level configurations.

**Example Structure (based on `ucsf_ac225_iqid_processing_plan.md` and current `config.json`):**

```json
{
  "experiment_id": "ucsf_ac225_exp1", // Planned
  "paths": { // Planned
    "root_data_dir": "/data/ucsf_ac225_exp1/",
    "raw_image_dir": "raw_iqid_images/",
    "quantitative_notes_doc": "quantitative_notes.docx",
    "dose_kernel_file": "kernels/ac225_water_10keV_kernel.txt",
    "output_base_dir": "output/",
    "processed_data_dir": "output/processed_data/",
    "aligned_images_dir": "output/aligned_images/",
    "dose_maps_dir": "output/dose_maps/",
    "log_dir": "output/logs/"
  },
  "quantitative_corrections": { // Planned
    "decay_correction_enabled": true,
    "half_life_seconds": 864000,
    "scanner_calibration_factor": 1.25,
    "unit_conversion_factor": 0.01
  },
  "automate_processing": { // Current name in config.json
    "file_name": "path/to/your/listmode_data.dat",
    "output_dir": "path/to/save/processed_data",
    "c_area_thresh": 15,
    "makedir": false,
    "ftype": "processed_lm",
    "binfac": 1,
    "ROI_area_thresh": 100,
    "t_binsize": 1000,
    "t_half": 3600,
    "subpx": 1,
    "event_fx": 0.1,
    "xlim": [0, null],
    "ylim": [0, null],
    "gauss": 15,
    "thresh": 0,
    "error_handling": { // Current in config.json
      "log_file": "path/to/log_file.log",
      "log_level": "ERROR"
    }
    // Parameters from plan like "roi_extraction_method" would go here if implemented
  },
  "automate_image_alignment": { // Current name in config.json
    "image_dir": "path/to/your/image_directory",
    "output_dir": "path/to/save/registered_images",
    "fformat": "tif",
    "pad": true, // Note: `pad` in `assemble_stack_hne` is True by default. This might be for other contexts.
    "deg": 2,
    "avg_over": 1,
    "subpx": 1,
    "color": [0, 0, 0],
    "convert_to_grayscale_for_ssd": true, // Added based on script's capability
    "error_handling": { // Current in config.json
      "log_file": "path/to/log_file.log",
      "log_level": "ERROR"
    }
    // Parameters from plan like "alignment_type" would go here if implemented
  },
  "automate_dose_kernel_processing": { // Current name in config.json
    "kernel_file": "path/to/your/kernel_file.txt",
    "dim": 128,
    "num_alpha_decays": 1000000,
    "vox_vol_m": 1e-9,
    "dens_kgm": 1000,
    "vox_xy": 1,
    "slice_z": 12,
    "output_dir": "path/to/save/processed_kernels",
    "error_handling": { // Current in config.json
      "log_file": "path/to/log_file.log",
      "log_level": "ERROR"
    }
    // Parameters from plan like "kernel_energy_mev" would go here if implemented
  },
  "logging": { // Planned
    "log_level": "INFO",
    "log_to_file": true,
    "log_filename_template": "processing_log_{script_name}_{timestamp}.log"
  },
  "parallelization": { // Planned
    "enabled": true,
    "max_workers_processing": 4,
    "max_workers_alignment": 2
  }
}
```

## Top-Level Parameters
*(Note: These are planned parameters based on `ucsf_ac225_iqid_processing_plan.md` and may not all be active in the current `config.json` version or used by current scripts.)*

*   `experiment_id`
    *   **Description:** A unique identifier for the experiment or dataset. Used for naming outputs or logs.
    *   **Type:** `string`
    *   **Example:** `"ucsf_ac225_exp1"`

## `paths` Section
*(Note: This section outlines planned configuration based on `ucsf_ac225_iqid_processing_plan.md`. Current scripts might hardcode some paths or derive them differently until this structure is fully implemented.)*

This section defines all relevant input and output paths for the pipeline.

*   `root_data_dir`
    *   **Description:** The main root directory for all experiment data. Other paths can be relative to this.
    *   **Type:** `string` (path)
    *   **Example:** `"/data/ucsf_ac225_exp1/"`
*   `raw_image_dir`
    *   **Description:** Subdirectory containing raw iQID image files (or other raw images for alignment).
    *   **Type:** `string` (path)
    *   **Example:** `"raw_iqid_images/"`
*   `quantitative_notes_doc`
    *   **Description:** Path to the `.docx` file containing quantitative correction notes.
    *   **Type:** `string` (path)
    *   **Example:** `"quantitative_notes.docx"`
*   `dose_kernel_file`
    *   **Description:** Path to the dose kernel text file.
    *   **Type:** `string` (path)
    *   **Example:** `"kernels/ac225_water_10keV_kernel.txt"`
*   `output_base_dir`
    *   **Description:** The base directory where all output subdirectories will be created.
    *   **Type:** `string` (path)
    *   **Example:** `"output/"`
*   `processed_data_dir`
    *   **Description:** Directory to save processed data from `automate_processing.py`.
    *   **Type:** `string` (path)
    *   **Example:** `"output/processed_data/"`
*   `aligned_images_dir`
    *   **Description:** Directory to save aligned images from `automate_image_alignment.py`.
    *   **Type:** `string` (path)
    *   **Example:** `"output/aligned_images/"`
*   `dose_maps_dir`
    *   **Description:** Directory to save generated dose maps from `automate_dose_kernel_processing.py`.
    *   **Type:** `string` (path)
    *   **Example:** `"output/dose_maps/"`
*   `log_dir`
    *   **Description:** Directory to store log files from the automation scripts.
    *   **Type:** `string` (path)
    *   **Example:** `"output/logs/"`

## `quantitative_corrections` Section
*(Note: This section outlines planned configuration based on `ucsf_ac225_iqid_processing_plan.md`. Current scripts might not use all these parameters directly from config yet.)*

Parameters for applying quantitative corrections to the iQID data.

*   `decay_correction_enabled`
    *   **Description:** Whether to apply radioactive decay correction.
    *   **Type:** `boolean`
    *   **Example:** `true`
*   `half_life_seconds`
    *   **Description:** Half-life of the isotope in seconds, used for decay correction.
    *   **Type:** `float` or `int`
    *   **Example:** `864000` (for Ac-225, 10 days)
*   `scanner_calibration_factor`
    *   **Description:** A global calibration factor for the scanner/detector.
    *   **Type:** `float`
    *   **Example:** `1.25`
*   `unit_conversion_factor`
    *   **Description:** Factor to convert raw image pixel values or counts to physical units (e.g., Bq/pixel or Bq/mL).
    *   **Type:** `float`
    *   **Example:** `0.01`

## `automate_processing` Section
(Corresponds to `automate_processing_params` in the plan)
Parameters specifically for the `automate_processing.py` script.

*   `file_name`
    *   **Description:** Path to the raw iQID listmode data file (`.dat`).
    *   **Type:** `string` (path)
    *   **Example:** `"path/to/your/listmode_data.dat"`
*   `output_dir`
    *   **Description:** Directory where processed data arrays (e.g., `xC.npy`, `yC.npy`) will be saved.
    *   **Type:** `string` (path)
    *   **Example:** `"path/to/save/processed_data"`
*   `c_area_thresh`
    *   **Description:** Cluster area threshold used in `ClusterData` initialization. Events from clusters smaller than this are typically filtered out.
    *   **Type:** `int`
    *   **Example:** `15`
*   `makedir`
    *   **Description:** Boolean flag passed to `ClusterData` to indicate if an analysis subdirectory should be automatically created by `ClusterData` (relative to the input file). Note: `automate_processing.py` itself uses `output_dir` for its outputs.
    *   **Type:** `boolean`
    *   **Example:** `false`
*   `ftype`
    *   **Description:** File type of the listmode data, passed to `ClusterData`. Options: "processed_lm", "offset_lm", "clusters".
    *   **Type:** `string`
    *   **Example:** `"processed_lm"`
*   `binfac`
    *   **Description:** Binning factor for image processing steps within `ClusterData`.
    *   **Type:** `int`
    *   **Example:** `1`
*   `ROI_area_thresh`
    *   **Description:** Area threshold (in pixels) for filtering ROIs identified by contour detection.
    *   **Type:** `int`
    *   **Example:** `100`
*   `t_binsize`
    *   **Description:** Time bin size in seconds for temporal analysis (e.g., creating time-activity curves).
    *   **Type:** `float` or `int`
    *   **Example:** `1000`
*   `t_half`
    *   **Description:** Half-life in seconds for decay correction or temporal fitting models within `ClusterData`.
    *   **Type:** `float` or `int`
    *   **Example:** `3600`
*   `subpx`
    *   **Description:** Subpixel factor for image reconstruction in `image_from_listmode`.
    *   **Type:** `int`
    *   **Example:** `1`
*   `event_fx`
    *   **Description:** Fraction of total events to load when using `image_from_big_listmode`.
    *   **Type:** `float` (0.0 to 1.0)
    *   **Example:** `0.1`
*   `xlim`
    *   **Description:** Tuple defining the (min_x, max_x) crop limits for `image_from_big_listmode`. `null` means no limit on that side.
    *   **Type:** `array` of two `int` or `null` values.
    *   **Example:** `[0, null]`
*   `ylim`
    *   **Description:** Tuple defining the (min_y, max_y) crop limits for `image_from_big_listmode`. `null` means no limit on that side.
    *   **Type:** `array` of two `int` or `null` values.
    *   **Example:** `[0, null]`
*   `gauss`
    *   **Description:** Gaussian blur kernel size used for contour preparation in `ClusterData`.
    *   **Type:** `int` (should be odd)
    *   **Example:** `15`
*   `thresh`
    *   **Description:** Threshold value used for binarizing the image before contour detection in `ClusterData`.
    *   **Type:** `int` or `float`
    *   **Example:** `0`
*   `error_handling`
    *   **Description:** Contains logging configuration for this specific script.
    *   **Type:** `object`
    *   `log_file`: Path to the log file.
        *   **Type:** `string`
        *   **Example:** `"path/to/log_file.log"` (Note: current script hardcodes `automate_processing.log`)
    *   `log_level`: Logging level (e.g., "ERROR", "INFO", "DEBUG").
        *   **Type:** `string`
        *   **Example:** `"ERROR"` (Note: current script hardcodes `INFO`)

*(Planned parameters from `ucsf_ac225_iqid_processing_plan.md` like `roi_extraction_method`, `threshold_value`, `minimum_roi_area_pixels` would be documented here if they were present in the actual `config.json`'s `automate_processing` section and used by the script.)*

## `automate_image_alignment` Section
(Corresponds to `automate_image_alignment_params` in the plan)
Parameters specifically for the `automate_image_alignment.py` script.

*   `image_dir`
    *   **Description:** Path to the directory containing image slices to be aligned. **Note:** This is typically overridden by a command-line argument when running the script.
    *   **Type:** `string` (path)
    *   **Example:** `"path/to/your/image_directory"`
*   `output_dir`
    *   **Description:** Directory where the aligned/registered images will be saved. **Note:** This is typically overridden by a command-line argument.
    *   **Type:** `string` (path)
    *   **Example:** `"path/to/save/registered_images"`
*   `fformat`
    *   **Description:** File format (extension) of the images in `image_dir`.
    *   **Type:** `string`
    *   **Example:** `"tif"`
*   `pad`
    *   **Description:** Boolean flag, likely intended for padding operations. Note: `assemble_stack_hne` used in the script has its own `pad=True` default for initial assembly. This config parameter might be for other padding steps if re-introduced.
    *   **Type:** `boolean`
    *   **Example:** `true`
*   `deg`
    *   **Description:** Angular increment in degrees for coarse rotation alignment step in `coarse_stack`.
    *   **Type:** `float` or `int`
    *   **Example:** `2`
*   `avg_over`
    *   **Description:** Number of previously aligned images to average for creating the reference in `coarse_stack`.
    *   **Type:** `int`
    *   **Example:** `1`
*   `subpx`
    *   **Description:** Subpixel factor. Noted as potentially unused by current functions in `automate_image_alignment.py` but kept for signature compatibility.
    *   **Type:** `int`
    *   **Example:** `1`
*   `color`
    *   **Description:** RGB tuple `[R, G, B]` used for padding color in `assemble_stack_hne`.
    *   **Type:** `array` of 3 `int` (0-255).
    *   **Example:** `[0, 0, 0]` (black)
*   `convert_to_grayscale_for_ssd`
    *   **Description:** If True, images are converted to grayscale before SSD calculation during coarse alignment.
    *   **Type:** `boolean`
    *   **Example:** `true` (This parameter might be added if not present, based on script capability)
*   `error_handling`
    *   **Description:** Logging configuration for this script.
    *   **Type:** `object`
    *   `log_file`: Path to the log file.
        *   **Type:** `string`
        *   **Example:** `"path/to/log_file.log"` (Note: current script hardcodes `automate_image_alignment.log`)
    *   `log_level`: Logging level.
        *   **Type:** `string`
        *   **Example:** `"ERROR"` (Note: current script hardcodes `INFO`)

*(Planned parameters like `alignment_type`, `reference_slice_index`, `coarse_alignment_factor`, `padding_value_he`, `crop_margins` from `ucsf_ac225_iqid_processing_plan.md` would be documented here if used directly from this config section by the script.)*

## `automate_dose_kernel_processing` Section
(Corresponds to `automate_dose_kernel_processing_params` in the plan)
Parameters specifically for the `automate_dose_kernel_processing.py` script.

*   `kernel_file`
    *   **Description:** Path to the raw dose kernel text file.
    *   **Type:** `string` (path)
    *   **Example:** `"path/to/your/kernel_file.txt"`
*   `dim`
    *   **Description:** Dimension of the (assumed cubic) raw input kernel loaded by `load_txt_kernel`.
    *   **Type:** `int`
    *   **Example:** `128`
*   `num_alpha_decays`
    *   **Description:** Normalization factor, typically the number of primary particles simulated for the kernel generation. Used by `load_txt_kernel`.
    *   **Type:** `float` or `int`
    *   **Example:** `1000000`
*   `vox_vol_m`
    *   **Description:** Voxel volume in cubic meters (m^3), used for `mev_to_mgy` conversion.
    *   **Type:** `float`
    *   **Example:** `1e-9` (for a 1mm x 1mm x 1mm voxel)
*   `dens_kgm`
    *   **Description:** Density of the material (e.g., tissue) in kg/m^3, used for `mev_to_mgy` conversion.
    *   **Type:** `float` or `int`
    *   **Example:** `1000` (for water)
*   `vox_xy`
    *   **Description:** Target XY voxel size (e.g., in µm) for padding the kernel using `pad_kernel_to_vsize`.
    *   **Type:** `int`
    *   **Example:** `1`
*   `slice_z`
    *   **Description:** Target Z voxel size (slice thickness, e.g., in µm) for padding the kernel using `pad_kernel_to_vsize`.
    *   **Type:** `int`
    *   **Example:** `12`
*   `output_dir`
    *   **Description:** Directory where the processed dose kernel (`processed_kernel.npy`) will be saved.
    *   **Type:** `string` (path)
    *   **Example:** `"path/to/save/processed_kernels"`
*   `error_handling`
    *   **Description:** Logging configuration for this script.
    *   **Type:** `object`
    *   `log_file`: Path to the log file.
        *   **Type:** `string`
        *   **Example:** `"path/to/log_file.log"` (Note: current script hardcodes `automate_dose_kernel_processing.log`)
    *   `log_level`: Logging level.
        *   **Type:** `string`
        *   **Example:** `"ERROR"` (Note: current script hardcodes `INFO`)

*(Planned parameters like `kernel_energy_mev`, `mev_to_mgy_conversion_factor`, `target_voxel_size_mm` from `ucsf_ac225_iqid_processing_plan.md` would be documented here if used by the script from this specific config section. Some overlap exists, e.g. `vox_vol_m`.)*

## `logging` Section
*(Note: This section outlines planned configuration based on `ucsf_ac225_iqid_processing_plan.md`. The current scripts use hardcoded log filenames and levels, but could be updated to use these global settings.)*

Global configuration for logging behavior across all scripts.

*   `log_level`
    *   **Description:** Default logging level for the scripts (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    *   **Type:** `string`
    *   **Example:** `"INFO"`
*   `log_to_file`
    *   **Description:** Boolean flag to enable or disable logging to a file.
    *   **Type:** `boolean`
    *   **Example:** `true`
*   `log_filename_template`
    *   **Description:** A template for log filenames, potentially including placeholders for script name and timestamp.
    *   **Type:** `string`
    *   **Example:** `"processing_log_{script_name}_{timestamp}.log"`

## `parallelization` Section
*(Note: This section outlines planned configuration based on `ucsf_ac225_iqid_processing_plan.md`. Parallelization is not yet implemented in the current scripts.)*

Parameters to control parallel processing capabilities (if implemented).

*   `enabled`
    *   **Description:** Global flag to enable or disable parallel processing features.
    *   **Type:** `boolean`
    *   **Example:** `true`
*   `max_workers_processing`
    *   **Description:** Maximum number of worker processes/threads for `automate_processing.py`.
    *   **Type:** `int`
    *   **Example:** `4`
*   `max_workers_alignment`
    *   **Description:** Maximum number of worker processes/threads for `automate_image_alignment.py`.
    *   **Type:** `int`
    *   **Example:** `2`
