### **IQID-Alphas Project Naming Conventions**

**Guiding Principle**: All Python code written for this project will adhere to the standards outlined in **PEP 8 -- Style Guide for Python Code**. The conventions listed below are specific applications of this standard based on the existing project plan.

---

#### **1. File and Directory Naming**

* **Python Modules**: Module filenames must be short, lowercase, and use underscores to separate words (`snake_case`).
    * **Examples**: `tissue_segmentation.py` , `slice_alignment.py` , `batch_processor.py` , `legacy_adapters.py`.

* **Configuration Files**: Configuration files should follow the same `snake_case` convention as modules.
    * **Examples**: `segmentation_config.json` , `alignment_config.json`.

* **Top-Level Directories**: Primary code directories should be lowercase.
    * **Examples**: `iqid_alphas/`, `pipelines/`, `docs/`, `tests/`.

* **CLI Interface Scripts**: The user-facing executable scripts in the top-level `pipelines/` directory must follow the `snake_case` convention.
    * **Examples**: `segmentation.py` , `alignment.py`.

---

#### **2. Code Naming (Variables, Classes, etc.)**

* **Class Names**: Class names must use the `PascalCase` (or `CapWords`) convention.
    * **Examples**: `TissueSeparator` , `EnhancedAligner` , `MultiModalAligner` , `SegmentationPipeline` , `ValidationSuite`.

* **Function and Method Names**: All functions and methods must be lowercase, with words separated by underscores (`snake_case`).
    * **Examples**: `separate_tissues` , `run_validation` , `generate_report` , `find_reference_slice` , `print_pipeline_summary`.

* **Variables and Arguments**: All variable names, function arguments, and instance attributes must be lowercase, with words separated by underscores (`snake_case`).
    * **Examples**: `raw_image_path` , `max_samples` , `ground_truth_dirs` , `alignment_method`.

* **Constants**: Module-level constants must be in all-uppercase, with words separated by underscores (`ALL_CAPS_SNAKE_CASE`).
    * **Example**: `DEFAULT_METHOD = 'adaptive_clustering'`

---

### **Summary Table**

| Element Type | Naming Convention | Example(s) from Plan |
| :--- | :--- | :--- |
| **Module/File** | `snake_case.py` | `tissue_segmentation.py`, `coregistration.py`  |
| **Class** | `PascalCase` | `TissueSeparator`, `AlignmentPipeline`  |
| **Function/Method**| `snake_case()` | `separate_tissues()`, `run_validation()`  |
| **Variable** | `snake_case` | `raw_image_path`, `max_samples`  |
| **Argument** | `snake_case` | `method='adaptive_clustering'`  |
| **Constant** | `ALL_CAPS_SNAKE_CASE` | `IOU_THRESHOLD = 0.5` |