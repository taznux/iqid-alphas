# UCSF Pipeline Evaluation Report - Comprehensive

**Evaluation Date:** 2025-06-18 17:14:51
**Dataset:** UCSF Real Medical Imaging Data
**Total Samples Tested:** 10
**Overall Performance Score:** 55.6/100

## Executive Summary

**Status:** ❌ REQUIRES IMPROVEMENT

The pipeline needs significant improvements before production deployment.

## UCSF Dataset Analysis

- **Files Analyzed:** 10
- **Average File Size:** 0.2 MB
- **Data Types Found:** float64
- **Common Image Shapes:** (125, 181)

### Sample File Analysis
- **mBq_corr_11.tif:** (125, 181) float64, 0.2MB, Range: 0.000-0.143
- **mBq_corr_2.tif:** (125, 181) float64, 0.2MB, Range: 0.000-0.161
- **mBq_corr_2.tif:** (165, 161) float64, 0.2MB, Range: 0.000-0.101
- **mBq_corr_1.tif:** (165, 161) float64, 0.2MB, Range: 0.000-0.086
- **mBq_corr_11.tif:** (145, 161) float64, 0.2MB, Range: 0.000-0.143

## Pipeline Performance Results

### SimplePipeline
- **Success Rate:** 0.0%
- **Tests Run:** 5
- **Average Quality Score:** 0.0
- **Average Processing Time:** 0.00 seconds

### Core Components Performance

#### Processor
- **Success Rate:** 0.0%
- **Tests:** 0/3
- **Average Time:** 0.00s
- **Errors:** 3 errors encountered

#### Aligner
- **Success Rate:** 100.0%
- **Tests:** 2/2
- **Average Time:** 0.01s

#### Segmenter
- **Success Rate:** 100.0%
- **Tests:** 2/2
- **Average Time:** 0.00s

## Recommendations

- 🔧 Improve SimplePipeline reliability and error handling
- 📈 Enhance image processing quality algorithms
- ⚙️ Address processor component issues

## Technical Notes

- All tests performed with real UCSF medical imaging data
- Quality scores based on SNR, contrast, and edge preservation metrics
- Processing times measured on current hardware configuration
- Error handling and robustness tested with diverse file formats

## Files and Outputs

- **Detailed Results:** `ucsf_comprehensive_test_results.json`
- **Test Data:** Real UCSF DataPush1 dataset
- **Processing Logs:** Available in evaluation output directory

