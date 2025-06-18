# IQID-Alphas Data Structure and Processing Workflow - Complete Reference

## Executive Summary

This document provides a comprehensive understanding of the UCSF iQID and H&E data structure, processing workflows, and CLI development requirements. This information is critical for proper implementation of the batch processing CLI and pipeline integration.

The key strategic insight is that the **provided datasets contain manually processed ground truth results for most workflow stages**. The primary goal is therefore to develop an **automated pipeline**, leveraging the existing `iqid-alphas` framework, where each component can be **validated against this ground truth**.

> **📌 IMPORTANT UPDATE:** The ReUpload dataset structure has been corrected based on the actual dataset organization. The ReUpload data is organized as `ReUpload/3D/tissue/sample/` and `ReUpload/Sequential/tissue/sample/` (not the nested `ReUpload/iQID_reupload/iQID/3D/tissue/sample/` structure as previously documented). This correction affects all evaluation and processing logic.

## Overall Workflow Summary

The project is divided into two primary, sequential workflows:

1.  **iQID Alignment (Validation using `ReUpload` data):**
    * **Grouping:** Raw images are grouped by mouse (`M` id) and day (`D` id), as one raw image contains multiple tissue types (e.g., Kidney L/R, Tumor).
    * **Segmentation:** An automated module separates the different tissue regions from the single raw image into distinct sample stacks.
    * **Alignment:** Each separated tissue stack is then processed through an automated alignment module.
    * **Validation:** The outputs of the automated segmentation and alignment modules are validated against the ground truth in the `ReUpload` dataset.
2.  **Multi-Modal Co-Registration (Development using `DataPush1` data):**
    * **Pairing:** After the iQID alignment pipeline is validated, aligned iQID stacks from `DataPush1` are programmatically paired with their corresponding H&E image stacks based on matching conditions (mouse, day, tissue type).
    * **Co-Registration:** An automated module is developed to co-register these matched iQID and H&E pairs.
    * **Assessment:** Since no final ground truth exists for co-registration, this step is assessed qualitatively and with proxy metrics.

## Foundational Framework: `iqid-alphas`

[https://github.com/robin-peter/iqid-alphas](https://github.com/robin-peter/iqid-alphas)

A key component of this project is to build upon the existing and validated **`iqid-alphas` Python framework**. This framework, authored by the data providers, contains the original methods used for their scientific publications. We will not be starting from scratch; instead, we will adapt and extend this codebase.

* **Core Modules to Leverage:**
    * `iqid/process_object.py`: For listmode data processing, relevant to the `Raw` -> `Segmented` stage.
    * `iqid/align.py`: For alignment and registration, using `PyStackReg`. This is the foundation for both the iQID slice alignment and the iQID-H&E co-registration modules.
    * `iqid/dpk.py`: For dose kernel calculations and dosimetry. This will be used for downstream scientific validation of our final aligned data.
* **Strategy:** Our automated modules will be wrappers around, or extensions of, the functions within `iqid-alphas` to ensure our results are consistent with the established science.

## 1. Data Hierarchy and Structure

### 1.1 Dataset Overview

The UCSF data is organized into two distinct datasets that serve different purposes: validation and production analysis. The logical workflow proceeds from the `ReUpload` dataset to the `DataPush1` dataset.

#### **ReUpload**: Complete Workflow & Validation Dataset

* **iQID**: Contains the full workflow (`Raw` → `1_segmented` → `2_aligned`). The `segmented` and `aligned` stages serve as **ground truth** for validating automated algorithms.
* **H&E**: Not available in this dataset.
* **Status**: Single-modal iQID dataset with all processing stages.
* **Use Case**: Development and validation of the automated **segmentation** and **slice alignment** modules.

#### **DataPush1**: Production-Ready Dataset

* **iQID**: Already aligned and ready for 3D reconstruction. Represents ground truth for pre-processing.
* **H&E**: Processed histology images paired with iQID.
* **Status**: Complete multi-modal dataset (iQID + H&E).
* **Use Case**: Development and validation of the final **H&E and iQID co-registration** step.

### 1.2 Detailed Data Organization

#### **ReUpload Structure (Full Workflow Dataset):**

```
ReUpload/
└── iQID_reupload
    └── iQID
        ├── 3D/                      # 3D processing directory
        │   ├── kidney/              # Tissue type: kidney
        │   │   ├── D1M1(P1)_L/      # Sample: Day 1, Mouse 1, Part 1, Left kidney
        │   │   │   ├── 0_D1M1K(P1)_iqid_event_image.tif  # Raw iQID event data (INPUT)
        │   │   │   ├── 1_segmented/   # Ground Truth for Segmentation
        │   │   │   │   ├── mBq_0.tif    # Segmented slice 0
        │   │   │   │   └── ...
        │   │   │   └── 2_aligned/     # Ground Truth for Alignment
        │   │   │       ├── mBq_corr_0.tif   # Aligned slice 0
        │   │   │       └── ...
        │   │   └── ...              # Additional samples
        └── ...
```

#### **DataPush1 Structure (Production Dataset):**

```
DataPush1/
├── HE/                      # Ready-to-use H&E data
│   ├── 3D/                  # 3D reconstruction ready
│   │   ├── kidney/          # Tissue type: kidney
│   │   │   ├── D1M1_L/      # Sample: Day 1, Mouse 1, Left kidney
│   │   │   │   ├── P1L.tif      # H&E slice 1, Left (aligned)
│   │   │   │   ├── P2L.tif      # H&E slice 2, Left (aligned)
│   │   │   │   └── P3L.tif      # H&E slice 3, Left (aligned)
│   │   │   └── D1M1_R/      # Sample: Day 1, Mouse 1, Right kidney
│   │   └── tumor/           # Tissue type: tumor
│   ├── Sequential/          # Sequential section analysis
│   └── Upper/Lower/         # Specific anatomical sections
└── iQID/                      # Ready-to-use iQID data
    ├── 3D/                  # 3D reconstruction ready (aligned)
    │   ├── kidney/
    │   │   ├── D1M1(P1)_L/  # Sample: Day 1, Mouse 1, Protocol 1, Left
    │   │   │   ├── mBq_corr_0.tif   # Aligned slice 0
    │   │   │   ├── mBq_corr_1.tif   # Aligned slice 1
    │   │   │   └── mBq_corr_2.tif   # Aligned slice 2
    │   │   └── D1M1(P1)_R/
    │   └── tumor/
    ├── Sequential/
    └── Upper/Lower/
```

### 1.3 Sample Naming and Grouping

#### **iQID Sample Naming Format:** `D{day}M{mouse}(P{part})_{tissue}`

* **D{day}**: Time point (e.g., D1)
* **M{mouse}**: Mouse subject number (e.g., M1)
* **P{part}**: Acquisition part/session (e.g., P1)
* **\_{tissue}**: Tissue type suffix (`_L` for Left kidney, `_R` for Right kidney, `_T` for Tumor)

#### **Common ID Grouping for Tissue Separation:**

A critical concept is that samples with the same `D{day}M{mouse}(P{part})` prefix **share the same raw image** and represent different tissue types separated from a single acquisition.

* **Example:** `D1M1(P1)_L`, `D1M1(P1)_R`, and potentially `D1M1(P1)_T` would all be extracted from the *same* raw iQID image file.
* **Implication:** The first processing step is to group these sample directories to identify the single, shared raw image that serves as the input to the segmentation module.

### 1.4 Dataset Characteristics Summary

| Dataset     | iQID Status              | H&E Available | Processing Stages                 | Primary Use                               |
| ----------- | ------------------------ | ------------- | --------------------------------- | ----------------------------------------- |
| **ReUpload** | 🔄 Full workflow         | ❌ Not available | Raw → GT Segmented → GT Aligned   | **Validation of Segmentation & Alignment** |
| **DataPush1** | ✅ Aligned (Ground Truth) | ✅ Available   | Final stage only                  | **Validation of Co-Registration** |

## 2. iQID Processing Workflow

### 2.1 Three-Stage Processing & Validation Pipeline

The goal is to create an automated system that replicates the manual processing steps. The `ReUpload` dataset provides the inputs and the ground truth outputs for the first two stages.

```
(INPUT)          (AUTOMATION GOAL 1)    (AUTOMATION GOAL 2)
Raw iQID Image → Automated Segmentation → Automated Alignment → 3D Reconstruction
      ↓                    |                      |
      ↓            (VALIDATE AGAINST)     (VALIDATE AGAINST)
      ↓            `1_segmented/`         `2_aligned/`
      ↓             (Ground Truth)         (Ground Truth)
```

## 3. CLI and Automation Strategy

### 3.1 Automation & Validation Strategy

The core logic is to run an automated module and compare its output to the known ground truth.

#### **ReUpload (Segmentation & Alignment Validation):**

```
def validate_segmentation_module(raw_image_path, group_of_gt_dirs):
    # 1. Get the shared raw image input (already found by grouping)
    # raw_image_path = path/to/0_D1M1K(P1)_iqid_event_image.tif

    # 2. Run the automated segmentation module
    # This should output separate directories for each tissue type.
    automated_output_group = auto_segmentation_pipeline.run(raw_image_path)

    # 3. Compare each automated output against its corresponding ground truth dir
    # group_of_gt_dirs = {'L': path/to/D1M1_L/1_segmented, 'R': path/to/D1M1_R/1_segmented}
    all_metrics = {}
    for tissue_type, automated_dir in automated_output_group.items():
        gt_dir = group_of_gt_dirs[tissue_type]
        metrics = validation.compare_segmentation(automated_dir, gt_dir)
        all_metrics[tissue_type] = metrics
    return all_metrics

def validate_alignment_module(segmented_dir, aligned_gt_dir):
    # 1. Get segmented slices (input)
    # segmented_dir = path/to/D1M1_L/1_segmented

    # 2. Run the automated alignment module
    automated_alignment_dir = auto_alignment_pipeline.run(segmented_dir)

    # 3. Get the ground truth alignment
    # aligned_gt_dir = path/to/D1M1_L/2_aligned

    # 4. Compare the output against the ground truth
    metrics = validation.compare_alignment(automated_alignment_dir, aligned_gt_dir)
    return metrics
```

#### **DataPush1 (Co-Registration Development):**

```
def find_and_process_pairs(datpush1_root):
    # 1. Discover all iQID and H&E samples and create a lookup table.
    iqid_samples = find_samples(datpush1_root / "iQID")
    he_samples = find_samples(datpush1_root / "HE")

    # 2. Iterate and find matching pairs based on D, M, and tissue identifiers.
    paired_samples = match_samples(iqid_samples, he_samples)

    # 3. Process each pair
    for pair in paired_samples:
        develop_coregistration_module(pair)

def develop_coregistration_module(paired_sample):
    # 1. Get the prepared iQID and H&E stacks
    iqid_stack_dir = paired_sample['iqid_path']
    he_stack_dir = paired_sample['he_path']

    # 2. Run the automated co-registration module
    coregistered_output = auto_coregistration_pipeline.run(iqid_stack_dir, he_stack_dir)

    # 3. Assess the output
    assessment = assessment.quality_check_coregistration(coregistered_output)
    return assessment
```

## 5. Development Implementation Plan (Revised)

### 5.1 Phase 1: Core Data Understanding (COMPLETED)

* ✅ Document data hierarchy and ground truth availability.
* ✅ Understand the two primary validation workflows.

### 5.2 Phase 2: Automated Segmentation Module

* **Goal**: Develop an algorithm to process a raw multi-tissue image and extract separate, clean tissue slices for each type.
* **Input**: A single `.../0_*_iqid_event_image.tif` identified by grouping samples.
* **Ground Truth**: The corresponding set of `.../1_segmented/` directories for that group.
* **Validation**: Calculate IoU or Dice coefficient for segmentation masks against ground truth for each tissue type.

### 5.3 Phase 3: Automated Slice Alignment Module

* **Goal**: Develop an algorithm to spatially align a stack of segmented slices for a single tissue type.
* **Input**: Slices from `ReUpload/.../1_segmented/`.
* **Ground Truth**: `ReUpload/.../2_aligned/`.
* **Validation**: Calculate MSE or SSIM between the automated aligned stack and the ground truth stack.

### 5.4 Phase 4: Automated Co-Registration Module (H&E + iQID)

* **Goal**: Develop an algorithm to co-register the prepared H&E and iQID stacks.
* **Input**: Programmatically paired sample directories from `DataPush1/HE/` and `DataPush1/iQID/`.
* **Validation**: Qualitative visual inspection and quantitative proxy metrics (e.g., mutual information).

## 8. Comprehensive Validation and Testing Strategy

To ensure robustness, the automated pipeline will be tested at multiple entry points, allowing for both end-to-end validation and isolated module testing.

### 8.1 Modular and End-to-End Testing Pipelines

We will implement several testing pipelines that use different combinations of automated and ground-truth data:

* **Test Pipeline 1: Full Automation (`ReUpload`)**
    * **Workflow:** `Raw Image` → `Auto Segmented` → `Auto Aligned`
    * **Purpose:** Test the entire iQID pre-processing pipeline from end to end.
    * **Validation:**
        * Compare `Auto Segmented` output against `GT 1_segmented`.
        * Compare `Auto Aligned` output against `GT 2_aligned`.
* **Test Pipeline 2: Alignment Module Validation (`ReUpload`)**
    * **Workflow:** `Manual GT 1_segmented` → `Auto Aligned`
    * **Purpose:** Isolate and test the performance of the alignment module, independent of segmentation errors.
    * **Validation:** Compare `Auto Aligned` output against `GT 2_aligned`.
* **Test Pipeline 3: Co-Registration Module Development (`DataPush1`)**
    * **Workflow:** `Manual Aligned iQID` + `Manual Prepared H&E` → `Auto Co-Registered`
    * **Purpose:** Develop and assess the final co-registration step using clean, manually prepared inputs.
    * **Validation:** Qualitative assessment and proxy metrics (e.g., mutual information).

### 8.2 Downstream Scientific Analysis Validation

Once an aligned iQID stack is produced (either automatically or from ground truth), we can validate its scientific integrity.

* **Workflow:** `Aligned iQID Stack` → Run `iqid.dpk` for Dosimetry Analysis.
* **Purpose:** To ensure that the outputs from our automated pipeline are scientifically valid and can be used for the same downstream analyses as the original framework.
* **Validation:** Compare the resulting dose maps and dosimetry values against those from the original publications or expected scientific outcomes. This confirms that our automated process does not introduce artifacts that would corrupt the final scientific results.

## Conclusion

### **Development Strategy:**

1.  **Framework-First**: The core strategy is to build our automated pipeline by adapting and extending the existing, published `iqid-alphas` framework.
2.  **Validation-Driven**: Rigorously test each automated module against the provided ground truth (`ReUpload`) and through a series of modular and end-to-end testing pipelines.
3.  **Scientific Integrity**: Ensure the final outputs are compatible with and produce results consistent with the downstream scientific analyses (e.g., dosimetry) from the original framework.