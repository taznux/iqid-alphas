"""
Validation System

Comprehensive validation framework for evaluating pipeline outputs,
comparing results against ground truth, and generating quality metrics.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, structural_similarity

from .mapping import ReverseMapper, CropLocationResult, AlignmentTransformResult
from .tissue_segmentation import TissueSeparator
from .alignment.aligner import UnifiedAligner, align_images
from .coregistration import MultiModalAligner, RegistrationMethod
from ..utils.io_utils import load_image, natural_sort, list_directories
from ..utils.math_utils import calculate_statistics


class ValidationMode(Enum):
    """Available validation modes."""
    SEGMENTATION = "segmentation"
    ALIGNMENT = "alignment"
    COREGISTRATION = "coregistration"
    COMPREHENSIVE = "comprehensive"


class MetricType(Enum):
    """Types of validation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    MUTUAL_INFORMATION = "mutual_information"
    CROSS_CORRELATION = "cross_correlation"
    TRANSFORMATION_ERROR = "transformation_error"
    DICE_COEFFICIENT = "dice_coefficient"


@dataclass
class ValidationResult:
    """Result from a validation run."""
    mode: ValidationMode
    metrics: Dict[str, float] = field(default_factory=dict)
    passed_tests: int = 0
    total_tests: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100.0
    
    @property
    def is_passing(self) -> bool:
        """Check if validation is passing (>= 90% success rate)."""
        return self.success_rate >= 90.0


@dataclass
class GroundTruth:
    """Container for ground truth data."""
    segmentation_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    alignment_transforms: Dict[str, np.ndarray] = field(default_factory=dict)
    registration_transforms: Dict[str, np.ndarray] = field(default_factory=dict)
    expected_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class ValidationSuite:
    """
    Comprehensive validation system for the iQID pipeline.
    
    Provides validation for segmentation, alignment, and coregistration
    components using ground truth data and automated quality metrics.
    """
    
    def __init__(self, 
                 ground_truth_path: Optional[Union[str, Path]] = None,
                 tolerance: float = 0.1,
                 min_success_rate: float = 90.0):
        """
        Initialize the validation suite.
        
        Parameters
        ----------
        ground_truth_path : str or Path, optional
            Path to ground truth data directory
        tolerance : float
            Tolerance for metric comparisons (0.1 = 10%)
        min_success_rate : float
            Minimum success rate to consider validation passing
        """
        self.ground_truth_path = Path(ground_truth_path) if ground_truth_path else None
        self.tolerance = tolerance
        self.min_success_rate = min_success_rate
        
        # Initialize components
        self.reverse_mapper = ReverseMapper()
        self.tissue_separator = TissueSeparator()
        self.aligner = UnifiedAligner()
        self.multimodal_aligner = MultiModalAligner()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load ground truth if available
        self.ground_truth = GroundTruth()
        if self.ground_truth_path and self.ground_truth_path.exists():
            self._load_ground_truth()
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
    
    def validate(self, 
                data_path: Union[str, Path],
                mode: ValidationMode = ValidationMode.COMPREHENSIVE,
                save_results: bool = True) -> ValidationResult:
        """
        Run comprehensive validation on the specified data.
        
        Parameters
        ----------
        data_path : str or Path
            Path to data directory to validate
        mode : ValidationMode
            Type of validation to perform
        save_results : bool
            Whether to save validation results
            
        Returns
        -------
        ValidationResult
            Comprehensive validation results
        """
        start_time = time.time()
        data_path = Path(data_path)
        
        self.logger.info(f"Starting {mode.value} validation on {data_path}")
        
        try:
            if mode == ValidationMode.SEGMENTATION:
                result = self._validate_segmentation(data_path)
            elif mode == ValidationMode.ALIGNMENT:
                result = self._validate_alignment(data_path)
            elif mode == ValidationMode.COREGISTRATION:
                result = self._validate_coregistration(data_path)
            elif mode == ValidationMode.COMPREHENSIVE:
                result = self._validate_comprehensive(data_path)
            else:
                raise ValueError(f"Unknown validation mode: {mode}")
            
            # Update timing
            result.processing_time = time.time() - start_time
            
            # Add to history
            self.validation_history.append(result)
            
            # Save results if requested
            if save_results:
                self._save_validation_result(result, data_path)
            
            self.logger.info(f"Validation completed: {result.success_rate:.1f}% success rate")
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            
            # Return failed result
            result = ValidationResult(
                mode=mode,
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
            self.validation_history.append(result)
            return result
    
    def _validate_segmentation(self, data_path: Path) -> ValidationResult:
        """Validate segmentation accuracy."""
        result = ValidationResult(mode=ValidationMode.SEGMENTATION)
        
        try:
            # Find raw and segmented images
            raw_images = self._find_images(data_path / "raw")
            segmented_images = self._find_images(data_path / "1_segmented")
            
            if not raw_images or not segmented_images:
                result.errors.append("Could not find raw or segmented images")
                return result
            
            result.total_tests = len(segmented_images)
            
            for seg_image_path in segmented_images:
                try:
                    # Load segmented image
                    seg_image = load_image(seg_image_path)
                    
                    # Find corresponding raw image
                    raw_image_path = self._find_corresponding_raw_image(seg_image_path, raw_images)
                    if not raw_image_path:
                        result.warnings.append(f"No corresponding raw image for {seg_image_path.name}")
                        continue
                    
                    raw_image = load_image(raw_image_path)
                    
                    # Use reverse mapping to find crop location
                    crop_result = self.reverse_mapper.find_crop_location(raw_image, seg_image)
                    
                    # Validate crop location
                    if crop_result.confidence > 0.8:  # High confidence threshold
                        result.passed_tests += 1
                        
                        # Calculate segmentation quality metrics
                        quality_metrics = self._calculate_segmentation_metrics(
                            raw_image, seg_image, crop_result
                        )
                        
                        # Add to overall metrics
                        for metric_name, value in quality_metrics.items():
                            if metric_name not in result.metrics:
                                result.metrics[metric_name] = []
                            result.metrics[metric_name].append(value)
                    else:
                        result.warnings.append(f"Low confidence crop location for {seg_image_path.name}")
                
                except Exception as e:
                    result.errors.append(f"Error processing {seg_image_path.name}: {e}")
            
            # Calculate average metrics
            for metric_name, values in result.metrics.items():
                if values:
                    result.metrics[metric_name] = np.mean(values)
            
        except Exception as e:
            result.errors.append(f"Segmentation validation failed: {e}")
        
        return result
    
    def _validate_alignment(self, data_path: Path) -> ValidationResult:
        """Validate alignment accuracy."""
        result = ValidationResult(mode=ValidationMode.ALIGNMENT)
        
        try:
            # Find segmented and aligned images
            segmented_images = self._find_images(data_path / "1_segmented")
            aligned_images = self._find_images(data_path / "2_aligned")
            
            if not segmented_images or not aligned_images:
                result.errors.append("Could not find segmented or aligned images")
                return result
            
            result.total_tests = len(aligned_images)
            
            for aligned_image_path in aligned_images:
                try:
                    # Load aligned image
                    aligned_image = load_image(aligned_image_path)
                    
                    # Find corresponding segmented image
                    seg_image_path = self._find_corresponding_segmented_image(aligned_image_path, segmented_images)
                    if not seg_image_path:
                        result.warnings.append(f"No corresponding segmented image for {aligned_image_path.name}")
                        continue
                    
                    seg_image = load_image(seg_image_path)
                    
                    # Calculate alignment transform
                    transform_result = self.reverse_mapper.calculate_alignment_transform(
                        seg_image, aligned_image
                    )
                    
                    # Validate alignment quality
                    if transform_result.match_quality > 0.7:  # Good alignment threshold
                        result.passed_tests += 1
                        
                        # Calculate alignment quality metrics
                        quality_metrics = self._calculate_alignment_metrics(
                            seg_image, aligned_image, transform_result
                        )
                        
                        # Add to overall metrics
                        for metric_name, value in quality_metrics.items():
                            if metric_name not in result.metrics:
                                result.metrics[metric_name] = []
                            result.metrics[metric_name].append(value)
                    else:
                        result.warnings.append(f"Poor alignment quality for {aligned_image_path.name}")
                
                except Exception as e:
                    result.errors.append(f"Error processing {aligned_image_path.name}: {e}")
            
            # Calculate average metrics
            for metric_name, values in result.metrics.items():
                if values:
                    result.metrics[metric_name] = np.mean(values)
            
        except Exception as e:
            result.errors.append(f"Alignment validation failed: {e}")
        
        return result
    
    def _validate_coregistration(self, data_path: Path) -> ValidationResult:
        """Validate coregistration accuracy."""
        result = ValidationResult(mode=ValidationMode.COREGISTRATION)
        
        try:
            # Find different modality images for coregistration testing
            iqid_images = self._find_images(data_path / "iqid")
            he_images = self._find_images(data_path / "he")
            
            if not iqid_images or not he_images:
                result.warnings.append("Could not find multi-modal images for coregistration validation")
                # Try with segmented images as different modalities
                iqid_images = self._find_images(data_path / "1_segmented")
                he_images = self._find_images(data_path / "2_aligned")
            
            if not iqid_images or not he_images:
                result.errors.append("Could not find images for coregistration validation")
                return result
            
            result.total_tests = min(len(iqid_images), len(he_images))
            
            for i in range(result.total_tests):
                try:
                    iqid_image = load_image(iqid_images[i])
                    he_image = load_image(he_images[i])
                    
                    # Test different registration methods
                    methods = [
                        RegistrationMethod.MUTUAL_INFORMATION,
                        RegistrationMethod.NORMALIZED_CROSS_CORRELATION,
                        RegistrationMethod.FEATURE_BASED
                    ]
                    
                    best_confidence = 0.0
                    best_result = None
                    
                    for method in methods:
                        try:
                            reg_result = self.multimodal_aligner.register_images(
                                iqid_image, he_image, method=method
                            )
                            
                            if reg_result.confidence > best_confidence:
                                best_confidence = reg_result.confidence
                                best_result = reg_result
                        
                        except Exception as e:
                            result.warnings.append(f"Method {method.value} failed: {e}")
                    
                    # Check if best result is acceptable
                    if best_result and best_result.confidence > 0.5:
                        result.passed_tests += 1
                        
                        # Add quality metrics
                        for metric_name, value in best_result.quality_metrics.items():
                            if metric_name not in result.metrics:
                                result.metrics[metric_name] = []
                            result.metrics[metric_name].append(value)
                
                except Exception as e:
                    result.errors.append(f"Error processing image pair {i}: {e}")
            
            # Calculate average metrics
            for metric_name, values in result.metrics.items():
                if values:
                    result.metrics[metric_name] = np.mean(values)
            
        except Exception as e:
            result.errors.append(f"Coregistration validation failed: {e}")
        
        return result
    
    def _validate_comprehensive(self, data_path: Path) -> ValidationResult:
        """Run comprehensive validation across all modes."""
        # Run individual validations
        seg_result = self._validate_segmentation(data_path)
        align_result = self._validate_alignment(data_path)
        coreg_result = self._validate_coregistration(data_path)
        
        # Combine results
        result = ValidationResult(mode=ValidationMode.COMPREHENSIVE)
        
        # Combine test counts
        result.total_tests = seg_result.total_tests + align_result.total_tests + coreg_result.total_tests
        result.passed_tests = seg_result.passed_tests + align_result.passed_tests + coreg_result.passed_tests
        
        # Combine metrics with prefixes
        for metric_name, value in seg_result.metrics.items():
            result.metrics[f"segmentation_{metric_name}"] = value
        
        for metric_name, value in align_result.metrics.items():
            result.metrics[f"alignment_{metric_name}"] = value
        
        for metric_name, value in coreg_result.metrics.items():
            result.metrics[f"coregistration_{metric_name}"] = value
        
        # Combine errors and warnings
        result.errors.extend(seg_result.errors)
        result.errors.extend(align_result.errors)
        result.errors.extend(coreg_result.errors)
        
        result.warnings.extend(seg_result.warnings)
        result.warnings.extend(align_result.warnings)
        result.warnings.extend(coreg_result.warnings)
        
        # Add individual results to metadata
        result.metadata = {
            'segmentation_result': seg_result,
            'alignment_result': align_result,
            'coregistration_result': coreg_result
        }
        
        return result
    
    def _find_images(self, directory: Path, extensions: List[str] = None) -> List[Path]:
        """Find image files in a directory."""
        if not directory.exists():
            return []
        
        if extensions is None:
            extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
        
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        
        return natural_sort([str(p) for p in images])
    
    def _find_corresponding_raw_image(self, seg_image_path: Path, raw_images: List[Path]) -> Optional[Path]:
        """Find the raw image corresponding to a segmented image."""
        # Simple matching based on filename similarity
        seg_name = seg_image_path.stem
        
        for raw_path in raw_images:
            if seg_name in raw_path.stem or raw_path.stem in seg_name:
                return raw_path
        
        return None
    
    def _find_corresponding_segmented_image(self, aligned_image_path: Path, segmented_images: List[Path]) -> Optional[Path]:
        """Find the segmented image corresponding to an aligned image."""
        # Simple matching based on filename similarity
        aligned_name = aligned_image_path.stem
        
        for seg_path in segmented_images:
            if aligned_name in seg_path.stem or seg_path.stem in aligned_name:
                return seg_path
        
        return None
    
    def _calculate_segmentation_metrics(self, 
                                      raw_image: np.ndarray, 
                                      seg_image: np.ndarray, 
                                      crop_result: CropLocationResult) -> Dict[str, float]:
        """Calculate segmentation quality metrics."""
        metrics = {}
        
        try:
            # Extract crop from raw image
            bbox = crop_result.bbox
            cropped = raw_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            
            # Resize to match segmented image if needed
            if cropped.shape != seg_image.shape:
                cropped = cv2.resize(cropped, seg_image.shape[:2][::-1])
            
            # Calculate metrics
            metrics['crop_confidence'] = crop_result.confidence
            metrics['size_ratio'] = (seg_image.shape[0] * seg_image.shape[1]) / (raw_image.shape[0] * raw_image.shape[1])
            
            # Convert to grayscale for comparison
            if len(cropped.shape) == 3:
                cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            else:
                cropped_gray = cropped
                
            if len(seg_image.shape) == 3:
                seg_gray = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
            else:
                seg_gray = seg_image
            
            # Structural similarity
            if cropped_gray.shape == seg_gray.shape:
                ssim_score = structural_similarity(cropped_gray, seg_gray)
                metrics['structural_similarity'] = ssim_score
            
            # Cross correlation
            correlation = cv2.matchTemplate(cropped_gray.astype(np.float32), 
                                          seg_gray.astype(np.float32), 
                                          cv2.TM_CCOEFF_NORMED)
            metrics['cross_correlation'] = np.max(correlation)
            
        except Exception as e:
            self.logger.warning(f"Error calculating segmentation metrics: {e}")
        
        return metrics
    
    def _calculate_alignment_metrics(self, 
                                   original_image: np.ndarray, 
                                   aligned_image: np.ndarray, 
                                   transform_result: AlignmentTransformResult) -> Dict[str, float]:
        """Calculate alignment quality metrics."""
        metrics = {}
        
        try:
            metrics['match_quality'] = transform_result.match_quality
            metrics['num_matches'] = transform_result.num_matches
            
            # Transform error (determinant should be close to 1 for good transforms)
            transform_matrix = np.array(transform_result.transform_matrix)
            if transform_matrix.shape == (2, 3):
                # Calculate determinant of the 2x2 rotation/scale part
                det = np.linalg.det(transform_matrix[:2, :2])
                metrics['transform_determinant'] = det
                metrics['transform_error'] = abs(1.0 - abs(det))
            
            # Image similarity after alignment
            if original_image.shape == aligned_image.shape:
                # Convert to grayscale
                if len(original_image.shape) == 3:
                    orig_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                else:
                    orig_gray = original_image
                
                if len(aligned_image.shape) == 3:
                    aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
                else:
                    aligned_gray = aligned_image
                
                # Structural similarity
                if orig_gray.shape == aligned_gray.shape:
                    ssim_score = structural_similarity(orig_gray, aligned_gray)
                    metrics['post_alignment_similarity'] = ssim_score
                
                # Mean squared error
                mse = mean_squared_error(orig_gray.flatten(), aligned_gray.flatten())
                metrics['mean_squared_error'] = mse
        
        except Exception as e:
            self.logger.warning(f"Error calculating alignment metrics: {e}")
        
        return metrics
    
    def _load_ground_truth(self):
        """Load ground truth data from disk."""
        try:
            # Load ground truth configuration
            config_path = self.ground_truth_path / "ground_truth.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    gt_config = json.load(f)
                    self.ground_truth.expected_metrics = gt_config.get('expected_metrics', {})
            
            # Load segmentation masks
            masks_dir = self.ground_truth_path / "segmentation_masks"
            if masks_dir.exists():
                for mask_file in masks_dir.glob("*.tif*"):
                    mask = load_image(mask_file, color_mode='grayscale')
                    self.ground_truth.segmentation_masks[mask_file.stem] = mask
            
            # Load alignment transforms
            transforms_dir = self.ground_truth_path / "alignment_transforms"
            if transforms_dir.exists():
                for transform_file in transforms_dir.glob("*.npy"):
                    transform = np.load(transform_file)
                    self.ground_truth.alignment_transforms[transform_file.stem] = transform
            
            self.logger.info(f"Loaded ground truth data from {self.ground_truth_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load ground truth data: {e}")
    
    def _save_validation_result(self, result: ValidationResult, data_path: Path):
        """Save validation result to disk."""
        try:
            # Create results directory
            results_dir = data_path / "validation_results"
            results_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"validation_{result.mode.value}_{timestamp}.json"
            
            # Convert result to dictionary
            result_dict = {
                'mode': result.mode.value,
                'metrics': result.metrics,
                'passed_tests': result.passed_tests,
                'total_tests': result.total_tests,
                'success_rate': result.success_rate,
                'processing_time': result.processing_time,
                'is_passing': result.is_passing,
                'errors': result.errors,
                'warnings': result.warnings,
                'timestamp': timestamp
            }
            
            # Save to file
            result_path = results_dir / filename
            with open(result_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            self.logger.info(f"Validation result saved to {result_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation result: {e}")
    
    def generate_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Generate a comprehensive validation report."""
        if not self.validation_history:
            return "No validation results available."
        
        report = []
        report.append("# iQID Pipeline Validation Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        total_validations = len(self.validation_history)
        passing_validations = sum(1 for r in self.validation_history if r.is_passing)
        
        report.append("## Summary")
        report.append(f"- Total validations: {total_validations}")
        report.append(f"- Passing validations: {passing_validations}")
        report.append(f"- Overall success rate: {(passing_validations/total_validations)*100:.1f}%")
        report.append("")
        
        # Individual results
        report.append("## Validation Results")
        for i, result in enumerate(self.validation_history, 1):
            report.append(f"### Validation {i} - {result.mode.value.title()}")
            report.append(f"- Success rate: {result.success_rate:.1f}%")
            report.append(f"- Tests passed: {result.passed_tests}/{result.total_tests}")
            report.append(f"- Processing time: {result.processing_time:.2f}s")
            report.append(f"- Status: {'✅ PASSING' if result.is_passing else '❌ FAILING'}")
            
            if result.metrics:
                report.append("- Key metrics:")
                for metric, value in result.metrics.items():
                    report.append(f"  - {metric}: {value:.4f}")
            
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
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Validation report saved to {output_path}")
        
        return report_text
