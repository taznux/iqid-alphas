"""
Validation Utilities

Helper functions for file discovery, ground truth loading,
and validation result management.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from .types import GroundTruth, ValidationResult
from ...utils.io import load_image, natural_sort, list_directories

logger = logging.getLogger(__name__)


def find_images(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Find image files in a directory."""
    if extensions is None:
        extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    
    if not directory.exists():
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return natural_sort(image_files)


def find_corresponding_raw_image(seg_image_path: Path, raw_images: List[Path]) -> Optional[Path]:
    """Find the raw image corresponding to a segmented image."""
    seg_stem = seg_image_path.stem
    
    # Remove common segmentation suffixes
    for suffix in ["_seg", "_segmented", "_mask", "_binary"]:
        if seg_stem.endswith(suffix):
            seg_stem = seg_stem[:-len(suffix)]
            break
    
    # Look for matching raw image
    for raw_path in raw_images:
        if raw_path.stem == seg_stem:
            return raw_path
        
        # Also check if raw image stem is contained in seg stem
        if seg_stem.startswith(raw_path.stem):
            return raw_path
    
    return None


def find_corresponding_segmented_image(aligned_image_path: Path, segmented_images: List[Path]) -> Optional[Path]:
    """Find the segmented image corresponding to an aligned image."""
    aligned_stem = aligned_image_path.stem
    
    # Remove common alignment suffixes
    for suffix in ["_aligned", "_reg", "_registered"]:
        if aligned_stem.endswith(suffix):
            aligned_stem = aligned_stem[:-len(suffix)]
            break
    
    # Look for matching segmented image
    for seg_path in segmented_images:
        seg_stem = seg_path.stem
        
        # Remove common segmentation suffixes from segmented image
        for suffix in ["_seg", "_segmented", "_mask", "_binary"]:
            if seg_stem.endswith(suffix):
                seg_stem = seg_stem[:-len(suffix)]
                break
        
        if seg_stem == aligned_stem:
            return seg_path
        
        # Also check if aligned stem is contained in seg stem
        if seg_stem.startswith(aligned_stem):
            return seg_path
    
    return None


def load_ground_truth(ground_truth_path: Path) -> GroundTruth:
    """Load ground truth data from disk."""
    ground_truth = GroundTruth()
    
    try:
        # Load ground truth configuration
        config_path = ground_truth_path / "ground_truth.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                gt_config = json.load(f)
                ground_truth.expected_metrics = gt_config.get('expected_metrics', {})
        
        # Load segmentation masks
        masks_dir = ground_truth_path / "segmentation_masks"
        if masks_dir.exists():
            for mask_file in masks_dir.glob("*.tif*"):
                mask = load_image(mask_file, color_mode='grayscale')
                ground_truth.segmentation_masks[mask_file.stem] = mask
        
        # Load alignment transforms
        transforms_dir = ground_truth_path / "alignment_transforms"
        if transforms_dir.exists():
            for transform_file in transforms_dir.glob("*.npy"):
                transform = np.load(transform_file)
                ground_truth.alignment_transforms[transform_file.stem] = transform
        
        # Load registration transforms
        registration_dir = ground_truth_path / "registration_transforms"
        if registration_dir.exists():
            for transform_file in registration_dir.glob("*.npy"):
                transform = np.load(transform_file)
                ground_truth.registration_transforms[transform_file.stem] = transform
        
        logger.info(f"Loaded ground truth data from {ground_truth_path}")
        
    except Exception as e:
        logger.error(f"Failed to load ground truth data: {e}")
    
    return ground_truth


def save_validation_result(result: ValidationResult, output_path: Path):
    """Save validation result to disk."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert result to dictionary for JSON serialization
        result_dict = {
            'mode': result.mode.value,
            'metrics': result.metrics,
            'passed_tests': result.passed_tests,
            'total_tests': result.total_tests,
            'processing_time': result.processing_time,
            'success_rate': result.success_rate,
            'is_passing': result.is_passing,
            'errors': result.errors,
            'warnings': result.warnings,
            'metadata': result.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"Validation result saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save validation result: {e}")


def generate_validation_report(results: List[ValidationResult], output_path: Optional[Path] = None) -> str:
    """Generate a comprehensive validation report."""
    report = []
    
    # Header
    report.append("# Validation Report")
    report.append("=" * 50)
    report.append("")
    
    # Summary
    total_tests = sum(r.total_tests for r in results)
    total_passed = sum(r.passed_tests for r in results)
    overall_success = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
    
    report.append("## Summary")
    report.append(f"- Total validation runs: {len(results)}")
    report.append(f"- Total tests: {total_tests}")
    report.append(f"- Tests passed: {total_passed}")
    report.append(f"- Overall success rate: {overall_success:.1f}%")
    report.append(f"- Status: {'✅ PASSING' if overall_success >= 90 else '❌ FAILING'}")
    report.append("")
    
    # Individual results
    for i, result in enumerate(results, 1):
        report.append(f"## Validation Run {i}: {result.mode.value.title()}")
        report.append(f"- Tests passed: {result.passed_tests}/{result.total_tests}")
        report.append(f"- Processing time: {result.processing_time:.2f}s")
        report.append(f"- Status: {'✅ PASSING' if result.is_passing else '❌ FAILING'}")
        
        if result.metrics:
            report.append("- Key metrics:")
            for metric, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    report.append(f"  - {metric}: {value:.4f}")
                else:
                    report.append(f"  - {metric}: {value}")
        
        if result.errors:
            report.append("- Errors:")
            for error in result.errors:
                report.append(f"  - {error}")
        
        if result.warnings:
            report.append("- Warnings:")
            for warning in result.warnings:
                report.append(f"  - {warning}")
        
        report.append("")
    
    report_text = "\n".join(report)
    
    # Save to file if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Validation report saved to {output_path}")
    
    return report_text
