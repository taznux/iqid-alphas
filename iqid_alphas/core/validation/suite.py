"""
Main Validation Suite

Orchestrates validation across all pipeline components with
comprehensive reporting and analysis capabilities.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from .types import ValidationMode, ValidationResult, ValidationConfig, GroundTruth
from .validators import SegmentationValidator, AlignmentValidator, CoregistrationValidator
from .utils import load_ground_truth, save_validation_result, generate_validation_report
from .metrics import calculate_comprehensive_metrics


class ValidationSuite:
    """
    Comprehensive validation system for the iQID pipeline.
    
    Provides validation for segmentation, alignment, and coregistration
    components using ground truth data and automated quality metrics.
    """
    
    def __init__(self, 
                 ground_truth_path: Optional[Union[str, Path]] = None,
                 config: Optional[ValidationConfig] = None):
        """
        Initialize the validation suite.
        
        Parameters
        ----------
        ground_truth_path : Optional[Union[str, Path]]
            Path to ground truth data directory
        config : Optional[ValidationConfig]
            Validation configuration settings
        """
        self.config = config or ValidationConfig()
        self.ground_truth_path = Path(ground_truth_path) if ground_truth_path else None
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.seg_validator = SegmentationValidator(self.config)
        self.align_validator = AlignmentValidator(self.config)
        self.coreg_validator = CoregistrationValidator(self.config)
        
        # Load ground truth data
        self.ground_truth = GroundTruth()
        if self.ground_truth_path and self.ground_truth_path.exists():
            self.ground_truth = load_ground_truth(self.ground_truth_path)
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
    
    def validate(self, 
                data_path: Union[str, Path],
                mode: ValidationMode = ValidationMode.COMPREHENSIVE,
                save_results: bool = True) -> ValidationResult:
        """
        Run validation on a dataset.
        
        Parameters
        ----------
        data_path : Union[str, Path]
            Path to dataset directory
        mode : ValidationMode
            Type of validation to run
        save_results : bool
            Whether to save validation results to disk
            
        Returns
        -------
        ValidationResult
            Validation results and metrics
        """
        data_path = Path(data_path)
        self.logger.info(f"Starting {mode.value} validation on {data_path}")
        
        start_time = time.time()
        
        try:
            if mode == ValidationMode.SEGMENTATION:
                result = self.seg_validator.validate(data_path)
            elif mode == ValidationMode.ALIGNMENT:
                result = self.align_validator.validate(data_path)
            elif mode == ValidationMode.COREGISTRATION:
                result = self.coreg_validator.validate(data_path)
            elif mode == ValidationMode.COMPREHENSIVE:
                result = self._validate_comprehensive(data_path)
            else:
                raise ValueError(f"Unknown validation mode: {mode}")
            
            # Update processing time
            result.processing_time = time.time() - start_time
            
            # Add to history
            self.validation_history.append(result)
            
            # Save results if requested
            if save_results:
                output_path = data_path / f"validation_results_{mode.value}.json"
                save_validation_result(result, output_path)
            
            self.logger.info(f"Validation completed: {result.passed_tests}/{result.total_tests} tests passed")
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            result = ValidationResult(mode=mode)
            result.errors.append(f"Validation failed: {e}")
            result.processing_time = time.time() - start_time
            return result
    
    def _validate_comprehensive(self, data_path: Path) -> ValidationResult:
        """Run comprehensive validation across all modes."""
        # Run individual validations
        seg_result = self.seg_validator.validate(data_path) if self.config.enable_segmentation_validation else None
        align_result = self.align_validator.validate(data_path) if self.config.enable_alignment_validation else None
        coreg_result = self.coreg_validator.validate(data_path) if self.config.enable_coregistration_validation else None
        
        # Combine results
        result = ValidationResult(mode=ValidationMode.COMPREHENSIVE)
        
        # Combine test counts
        results_list = [r for r in [seg_result, align_result, coreg_result] if r is not None]
        result.total_tests = sum(r.total_tests for r in results_list)
        result.passed_tests = sum(r.passed_tests for r in results_list)
        
        # Combine metrics with prefixes
        if seg_result:
            for metric_name, value in seg_result.metrics.items():
                result.metrics[f"segmentation_{metric_name}"] = value
        
        if align_result:
            for metric_name, value in align_result.metrics.items():
                result.metrics[f"alignment_{metric_name}"] = value
        
        if coreg_result:
            for metric_name, value in coreg_result.metrics.items():
                result.metrics[f"coregistration_{metric_name}"] = value
        
        # Combine errors and warnings
        for r in results_list:
            result.errors.extend(r.errors)
            result.warnings.extend(r.warnings)
        
        # Add individual results to metadata
        result.metadata = {
            'segmentation_result': seg_result,
            'alignment_result': align_result,
            'coregistration_result': coreg_result
        }
        
        return result
    
    def batch_validate(self, 
                      dataset_paths: List[Union[str, Path]],
                      mode: ValidationMode = ValidationMode.COMPREHENSIVE) -> List[ValidationResult]:
        """
        Run validation on multiple datasets.
        
        Parameters
        ----------
        dataset_paths : List[Union[str, Path]]
            List of dataset directory paths
        mode : ValidationMode
            Type of validation to run
            
        Returns
        -------
        List[ValidationResult]
            List of validation results for each dataset
        """
        results = []
        
        for i, dataset_path in enumerate(dataset_paths, 1):
            self.logger.info(f"Validating dataset {i}/{len(dataset_paths)}: {dataset_path}")
            
            try:
                result = self.validate(dataset_path, mode=mode)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to validate {dataset_path}: {e}")
                # Create error result
                error_result = ValidationResult(mode=mode)
                error_result.errors.append(f"Validation failed: {e}")
                results.append(error_result)
        
        return results
    
    def generate_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Parameters
        ----------
        output_path : Optional[Union[str, Path]]
            Path to save the report
            
        Returns
        -------
        str
            Generated report text
        """
        if output_path:
            output_path = Path(output_path)
        
        return generate_validation_report(self.validation_history, output_path)
    
    def get_validation_statistics(self) -> Dict[str, float]:
        """Get comprehensive validation statistics."""
        if not self.validation_history:
            return {}
        
        # Collect all metrics from validation history
        all_metrics = []
        for result in self.validation_history:
            all_metrics.append(result.metrics)
        
        return calculate_comprehensive_metrics(all_metrics)
    
    def reset_history(self):
        """Clear validation history."""
        self.validation_history.clear()
        self.logger.info("Validation history reset")
    
    def export_results(self, output_path: Union[str, Path], format: str = "json"):
        """
        Export validation results to file.
        
        Parameters
        ----------
        output_path : Union[str, Path]
            Output file path
        format : str
            Export format ('json' or 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            import json
            # Convert results to serializable format
            results_data = []
            for result in self.validation_history:
                result_dict = {
                    'mode': result.mode.value,
                    'metrics': result.metrics,
                    'passed_tests': result.passed_tests,
                    'total_tests': result.total_tests,
                    'processing_time': result.processing_time,
                    'success_rate': result.success_rate,
                    'is_passing': result.is_passing,
                    'errors': result.errors,
                    'warnings': result.warnings
                }
                results_data.append(result_dict)
            
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
        
        elif format.lower() == "csv":
            import pandas as pd
            # Convert to DataFrame format
            rows = []
            for result in self.validation_history:
                row = {
                    'mode': result.mode.value,
                    'passed_tests': result.passed_tests,
                    'total_tests': result.total_tests,
                    'success_rate': result.success_rate,
                    'processing_time': result.processing_time,
                    'is_passing': result.is_passing,
                    'num_errors': len(result.errors),
                    'num_warnings': len(result.warnings)
                }
                # Add metrics as columns
                row.update(result.metrics)
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Validation results exported to {output_path}")
