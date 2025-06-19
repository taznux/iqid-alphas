"""
Tests for the modular validation package.

Tests the ValidationSuite and its components to ensure proper
functionality and backward compatibility.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from iqid_alphas.core.validation import (
    ValidationSuite,
    ValidationMode,
    ValidationResult,
    ValidationConfig,
    SegmentationValidator,
    AlignmentValidator,
    CoregistrationValidator
)


class TestValidationTypes:
    """Test validation data types and enums."""
    
    def test_validation_result_success_rate(self):
        """Test ValidationResult success rate calculation."""
        result = ValidationResult(mode=ValidationMode.SEGMENTATION)
        result.total_tests = 10
        result.passed_tests = 8
        
        assert result.success_rate == 80.0
        assert result.is_passing == False  # < 90%
        
        result.passed_tests = 9
        assert result.success_rate == 90.0
        assert result.is_passing == True  # >= 90%
    
    def test_validation_config_defaults(self):
        """Test ValidationConfig default values."""
        config = ValidationConfig()
        
        assert config.tolerance == 0.1
        assert config.min_success_rate == 90.0
        assert config.segmentation_confidence_threshold == 0.8
        assert config.alignment_quality_threshold == 0.7
        assert config.enable_segmentation_validation == True


class TestValidationSuite:
    """Test the main ValidationSuite class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "test_data"
        self.data_path.mkdir(parents=True)
        
        # Create test configuration
        self.config = ValidationConfig(
            tolerance=0.1,
            min_success_rate=90.0
        )
        
        self.validator = ValidationSuite(config=self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ValidationSuite initialization."""
        assert self.validator.config.tolerance == 0.1
        assert self.validator.config.min_success_rate == 90.0
        assert isinstance(self.validator.seg_validator, SegmentationValidator)
        assert isinstance(self.validator.align_validator, AlignmentValidator)
        assert isinstance(self.validator.coreg_validator, CoregistrationValidator)
    
    def test_initialization_with_ground_truth(self):
        """Test initialization with ground truth path."""
        gt_path = Path(self.temp_dir) / "ground_truth"
        gt_path.mkdir()
        
        # Create mock ground truth file
        gt_config = gt_path / "ground_truth.json"
        with open(gt_config, 'w') as f:
            f.write('{"expected_metrics": {"test": 0.5}}')
        
        validator = ValidationSuite(ground_truth_path=gt_path, config=self.config)
        assert validator.ground_truth_path == gt_path
    
    @patch('iqid_alphas.core.validation.validators.SegmentationValidator.validate')
    def test_segmentation_validation(self, mock_validate):
        """Test segmentation validation."""
        # Mock the validator response
        mock_result = ValidationResult(mode=ValidationMode.SEGMENTATION)
        mock_result.total_tests = 5
        mock_result.passed_tests = 4
        mock_result.metrics = {'accuracy': 0.85}
        mock_validate.return_value = mock_result
        
        result = self.validator.validate(self.data_path, mode=ValidationMode.SEGMENTATION)
        
        assert result.mode == ValidationMode.SEGMENTATION
        assert result.total_tests == 5
        assert result.passed_tests == 4
        assert 'accuracy' in result.metrics
        mock_validate.assert_called_once()
    
    @patch('iqid_alphas.core.validation.validators.AlignmentValidator.validate')
    def test_alignment_validation(self, mock_validate):
        """Test alignment validation."""
        mock_result = ValidationResult(mode=ValidationMode.ALIGNMENT)
        mock_result.total_tests = 3
        mock_result.passed_tests = 3
        mock_result.metrics = {'ssim_score': 0.92}
        mock_validate.return_value = mock_result
        
        result = self.validator.validate(self.data_path, mode=ValidationMode.ALIGNMENT)
        
        assert result.mode == ValidationMode.ALIGNMENT
        assert result.total_tests == 3
        assert result.passed_tests == 3
        assert 'ssim_score' in result.metrics
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation combining all modes."""
        # Mock individual validator results
        seg_result = ValidationResult(mode=ValidationMode.SEGMENTATION)
        seg_result.total_tests = 5
        seg_result.passed_tests = 4
        seg_result.metrics = {'accuracy': 0.85}
        
        align_result = ValidationResult(mode=ValidationMode.ALIGNMENT)
        align_result.total_tests = 3
        align_result.passed_tests = 3
        align_result.metrics = {'ssim_score': 0.92}
        
        coreg_result = ValidationResult(mode=ValidationMode.COREGISTRATION)
        coreg_result.total_tests = 2
        coreg_result.passed_tests = 1
        coreg_result.metrics = {'mutual_information': 0.75}
        
        # Configure mocks directly on the validators
        with patch.object(self.validator.seg_validator, 'validate', return_value=seg_result), \
             patch.object(self.validator.align_validator, 'validate', return_value=align_result), \
             patch.object(self.validator.coreg_validator, 'validate', return_value=coreg_result):
            
            result = self.validator.validate(self.data_path, mode=ValidationMode.COMPREHENSIVE)
            
            assert result.mode == ValidationMode.COMPREHENSIVE
            assert result.total_tests == 10  # 5 + 3 + 2
            assert result.passed_tests == 8   # 4 + 3 + 1
            assert 'segmentation_accuracy' in result.metrics
            assert 'alignment_ssim_score' in result.metrics
            assert 'coregistration_mutual_information' in result.metrics
    
    def test_batch_validate(self):
        """Test batch validation on multiple datasets."""
        # Create multiple test directories
        data_paths = []
        for i in range(3):
            path = Path(self.temp_dir) / f"dataset_{i}"
            path.mkdir()
            data_paths.append(path)
        
        with patch.object(self.validator, 'validate') as mock_validate:
            # Mock validation results
            mock_results = []
            for i in range(3):
                result = ValidationResult(mode=ValidationMode.SEGMENTATION)
                result.total_tests = 2
                result.passed_tests = 2
                mock_results.append(result)
            
            mock_validate.side_effect = mock_results
            
            results = self.validator.batch_validate(data_paths, mode=ValidationMode.SEGMENTATION)
            
            assert len(results) == 3
            assert all(r.mode == ValidationMode.SEGMENTATION for r in results)
            assert mock_validate.call_count == 3
    
    def test_generate_report(self):
        """Test validation report generation."""
        # Add some mock validation history
        result1 = ValidationResult(mode=ValidationMode.SEGMENTATION)
        result1.total_tests = 5
        result1.passed_tests = 4
        result1.processing_time = 10.0
        
        result2 = ValidationResult(mode=ValidationMode.ALIGNMENT)
        result2.total_tests = 3
        result2.passed_tests = 3
        result2.processing_time = 5.0
        
        self.validator.validation_history = [result1, result2]
        
        report = self.validator.generate_report()
        
        assert "Validation Report" in report
        assert "Summary" in report
        assert "Total validation runs: 2" in report
        assert "Total tests: 8" in report
        assert "Tests passed: 7" in report
    
    def test_statistics_calculation(self):
        """Test validation statistics calculation."""
        # Add some mock validation history with metrics
        result1 = ValidationResult(mode=ValidationMode.SEGMENTATION)
        result1.metrics = {'accuracy': 0.85, 'precision': 0.9}
        
        result2 = ValidationResult(mode=ValidationMode.ALIGNMENT)
        result2.metrics = {'accuracy': 0.92, 'ssim_score': 0.88}
        
        self.validator.validation_history = [result1, result2]
        
        stats = self.validator.get_validation_statistics()
        
        # Should have comprehensive metrics with mean, std, etc.
        assert 'accuracy_mean' in stats
        assert 'accuracy_std' in stats
        assert stats['accuracy_mean'] == 0.885  # (0.85 + 0.92) / 2
    
    def test_export_results_json(self):
        """Test exporting results to JSON format."""
        # Add mock validation history
        result = ValidationResult(mode=ValidationMode.SEGMENTATION)
        result.total_tests = 5
        result.passed_tests = 4
        result.metrics = {'accuracy': 0.85}
        
        self.validator.validation_history = [result]
        
        output_path = Path(self.temp_dir) / "results.json"
        self.validator.export_results(output_path, format="json")
        
        assert output_path.exists()
        
        # Verify JSON content
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]['mode'] == 'segmentation'
        assert data[0]['total_tests'] == 5
        assert data[0]['passed_tests'] == 4


class TestValidationComponents:
    """Test individual validation components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ValidationConfig()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_segmentation_validator_initialization(self):
        """Test SegmentationValidator initialization."""
        validator = SegmentationValidator(self.config)
        assert validator.config == self.config
        assert hasattr(validator, 'reverse_mapper')
    
    def test_alignment_validator_initialization(self):
        """Test AlignmentValidator initialization."""
        validator = AlignmentValidator(self.config)
        assert validator.config == self.config
        assert hasattr(validator, 'reverse_mapper')
    
    def test_coregistration_validator_initialization(self):
        """Test CoregistrationValidator initialization."""
        validator = CoregistrationValidator(self.config)
        assert validator.config == self.config
        assert hasattr(validator, 'multimodal_aligner')
        assert hasattr(validator, 'unified_aligner')


if __name__ == "__main__":
    pytest.main([__file__])
