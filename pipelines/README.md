# IQID-Alphas Production Pipelines

This directory contains production-ready interface scripts for the three main IQID data processing pipelines. Each script provides a thin interface that delegates core processing to implementations in the `iqid_alphas.pipelines` package.

## 🚧 Implementation Status

**Current Status**: Pipeline interfaces are implemented with clean command-line interfaces. Core pipeline classes are created but need implementation of core modules.

## Pipeline Scripts

### 1. `segmentation.py` - Raw → Segmented Validation
**Dataset**: ReUpload  
**Purpose**: Validate automated tissue separation from multi-tissue raw iQID images

```bash
# Basic validation
python segmentation.py --data /path/to/ReUpload --output results/segmentation

# Test with limited samples
python segmentation.py --data /path/to/ReUpload --output results/test --max-samples 5
```

**Expected Features** (TODO):
- ✅ Discover raw image groups sharing same base ID (D1M1, etc.)
- ✅ Automated tissue separation (kidney L/R, tumor)
- ✅ Validation against ground truth in `1_segmented/` directories
- ✅ Quality metrics: IoU, Dice coefficient, precision, recall
- ✅ Comprehensive visualizations and reporting

### 2. `alignment.py` - Segmented → Aligned Validation
**Dataset**: ReUpload  
**Purpose**: Validate advanced slice alignment (Steps 9-13 bidirectional methods)

```bash
# Basic bidirectional alignment
python alignment.py --data /path/to/ReUpload --output results/alignment

# Try different methods
python alignment.py --data /path/to/ReUpload --alignment-method sequential
python alignment.py --data /path/to/ReUpload --reference-method quality_based
```

**Expected Features** (TODO):
- ✅ Step 9: Reference slice detection (largest slice strategy)
- ✅ Step 10: Bidirectional alignment (largest→up, largest→down)
- ✅ Step 11: Noise-robust methods (mask-based, template-based)
- ✅ Step 12: Enhanced multi-slice visualizations
- ✅ Step 13: 3D stack alignment with overlap analysis
- ✅ Validation against ground truth in `2_aligned/` directories
- ✅ Quality metrics: SSIM, mutual information, propagation quality

### 3. `coregistration.py` - iQID+H&E Co-registration
**Dataset**: DataPush1  
**Purpose**: Multi-modal co-registration with proxy quality assessment

```bash
# Basic mutual information registration
python coregistration.py --data /path/to/DataPush1 --output results/coregistration

# Try feature-based registration
python coregistration.py --data /path/to/DataPush1 --registration-method feature_based
python coregistration.py --data /path/to/DataPush1 --pairing-strategy fuzzy_matching
```

**Expected Features** (TODO):
- ✅ Intelligent iQID-H&E sample pairing based on naming patterns
- ✅ Multi-modal registration (mutual information, feature-based)
- ✅ Proxy quality assessment (no ground truth available)
- ✅ Quality metrics: mutual information, structural similarity, confidence
- ✅ Tissue-specific performance analysis
- ✅ Multi-modal fusion visualizations

## 🏗️ Architecture

```
pipelines/                      # Production pipeline interfaces (implemented)
├── segmentation.py
├── alignment.py
└── coregistration.py

iqid_alphas/pipelines/          # Core pipeline classes (TODO: implement)
├── segmentation.py
├── alignment.py
└── coregistration.py

iqid_alphas/core/               # Core processing modules (TODO: implement)
├── tissue_segmentation.py
├── slice_alignment.py
├── coregistration.py
├── validation.py
└── batch_processor.py
```

## 🎯 Usage Examples

### Quick Testing
```bash
# Test segmentation with 3 samples
cd pipelines/
python segmentation.py --data /path/to/ReUpload --max-samples 3 --output test_segmentation

# Test alignment with specific method
python alignment.py --data /path/to/ReUpload --max-samples 3 --alignment-method bidirectional

# Test co-registration with feature-based method
python coregistration.py --data /path/to/DataPush1 --max-samples 3 --registration-method feature_based
```

### Production Runs
```bash
# Full segmentation validation
python segmentation.py --data /path/to/ReUpload --output production/segmentation_results

# Full alignment validation with custom config
python alignment.py --data /path/to/ReUpload --output production/alignment_results --config alignment_config.json

# Full co-registration evaluation
python coregistration.py --data /path/to/DataPush1 --output production/coregistration_results
```

## 📋 Next Steps

### Phase 1: Core Infrastructure (Current)
1. **✅ Thin interface scripts** - Complete with command-line interfaces
2. **🚧 Core pipeline classes** - Created with TODO placeholders
3. **📋 TODO: Core modules** in `iqid_alphas/core/`:
   - `tissue_segmentation.py` - Multi-tissue separation
   - `slice_alignment.py` - Enhanced alignment with Steps 9-13
   - `coregistration.py` - Multi-modal registration
   - `validation.py` - Metrics and quality assessment
   - `batch_processor.py` - Efficient batch processing

### Phase 2: Implementation
1. Implement core processing modules
2. Connect pipeline classes to core modules
3. Add comprehensive validation and metrics
4. Implement visualization systems

### Phase 3: Testing & Optimization
1. End-to-end testing with real datasets
2. Performance optimization
3. Enhanced error handling and logging
4. Production deployment preparation

## 🔧 Configuration

Each pipeline supports JSON configuration files for customizing processing parameters:

```json
{
  "segmentation": {
    "methods": ["threshold", "watershed", "clustering"],
    "quality_threshold": 0.85,
    "visualization": true
  },
  "alignment": {
    "reference_method": "largest",
    "bidirectional": true,
    "noise_robust": true
  },
  "coregistration": {
    "registration_method": "mutual_information",
    "pairing_strategy": "automatic",
    "confidence_threshold": 0.7
  }
}
```

## 📊 Output Structure

Each pipeline creates standardized output directories:

```
output_directory/
├── results/
│   ├── metrics.json
│   ├── summary_report.json
│   └── detailed_results.csv
├── visualizations/
│   ├── overview_dashboard.png
│   ├── quality_metrics.png
│   └── sample_comparisons/
└── logs/
    ├── execution.log
    └── errors.log
```
