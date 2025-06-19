# IQID-Alphas Modular Refactoring Plan & Progress Tracking

**Date**: June 19, 2025

## Objective

Refactor the `iqid_alphas` codebase into modular, function-focused packages as outlined in the project plans. This will improve maintainability, testability, and enable robust CLI and reporting integration.

---

## Refactoring Plan

### 1. Directory Structure & Modularization
- **Core Modules** (`iqid_alphas/core/`): Each major function (segmentation, alignment, coregistration, validation, batch processing, mapping) in its own file.
- **Pipelines** (`iqid_alphas/pipelines/`): Each pipeline (segmentation, alignment, coregistration) as a separate module, inheriting from a common `BasePipeline`.
- **Reporting & Visualization**:
    - `iqid_alphas/reporting/`: Markdown, Mermaid, and report generation logic.
    - `iqid_alphas/visualization/`: Plotting, overlays, and visual artifact generation.
- **Utils** (`iqid_alphas/utils/`): File I/O, math, adapters, transforms.

### 2. Function-Focused Design
- Each module exposes only the necessary classes/functions for its responsibility.
- Avoid monolithic scripts; split large files into smaller, focused modules if >500 lines.
- Use PEP8-compliant naming for all classes, functions, and variables.

### 3. Configuration-Driven
- All pipelines/modules accept configuration objects or JSON config files (in `iqid_alphas/configs/`).
- Implement a config loader/validator utility.

### 4. Unified CLI
- CLI entry points in `iqid_alphas/cli.py` and `pipelines/` scripts, using subcommands for each pipeline.
- CLI scripts are thin, delegating logic to pipeline classes.

### 5. Reporting Integration
- Pipelines call reporting/visualization hooks after each major step.
- Reports and visualizations saved in a structured output directory.

---

## Progress Tracking Table

| Task | Status | Notes |
|------|--------|-------|
| Create/verify modular directory structure | ⏳ In Progress |  |
| Refactor core modules into separate files | ⏳ In Progress |  |
| Implement/configure BasePipeline | ⏳ In Progress |  |
| Refactor segmentation pipeline | ⏳ In Progress |  |
| Refactor alignment pipeline | ⏳ In Progress |  |
| Refactor coregistration pipeline | ⏳ In Progress |  |
| Refactor utils and adapters | ⏳ In Progress |  |
| Implement config loader/validator | ⏳ In Progress |  |
| Refactor CLI scripts to use pipelines | ⏳ In Progress |  |
| Integrate reporting/visualization hooks | ⏳ In Progress |  |
| Update imports and documentation | ⏳ In Progress |  |

---

## Next Steps

1. Create missing directories and base files for modularization.
2. Move/split code from monolithic files into appropriate modules.
3. Refactor imports to use the new modular structure.
4. Update CLI scripts to use pipeline classes.
5. Add configuration loading and validation to all pipelines.
6. Integrate reporting and visualization hooks into pipeline steps.
7. Update this document as progress is made.

---

**Last updated:** June 19, 2025
