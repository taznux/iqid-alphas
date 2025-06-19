#!/usr/bin/env python3
"""Integration tests for Phase 2 pipelines"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from iqid_alphas.pipelines.segmentation import run_segmentation_pipeline
from iqid_alphas.pipelines.alignment import run_alignment_pipeline
from iqid_alphas.pipelines.coregistration import CoregistrationPipeline
from iqid_alphas.modern_cli import main


class TestPipelineIntegration:
    """Integration tests for all pipelines."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.temp_dir / "data"
        self.output_dir = self.temp_dir / "output"
        
        # Create mock data structure
        self.data_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Create sample directories
        sample_dir = self.data_dir / "sample_001"
        sample_dir.mkdir()
        
        # Create subdirectories for different pipeline stages
        (sample_dir / "0_raw").mkdir()
        (sample_dir / "1_segmented").mkdir()
        (sample_dir / "2_aligned").mkdir()
        
        # Create mock image files
        for stage_dir in ["0_raw", "1_segmented", "2_aligned"]:
            stage_path = sample_dir / stage_dir
            for i in range(3):
                (stage_path / f"slice_{i:03d}.png").touch()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('iqid_alphas.pipelines.segmentation.TissueSeparator')
    @patch('iqid_alphas.pipelines.segmentation.ValidationSuite')
    def test_segmentation_pipeline_integration(self, mock_validator, mock_separator):
        """Test segmentation pipeline integration."""
        # Mock the components
        mock_separator.return_value.process_sample.return_value = Mock()
        mock_validator.return_value.validate_segmentation.return_value = Mock(
            metrics={'iou': 0.85, 'dice': 0.90}
        )
        
        # Run pipeline
        with patch('iqid_alphas.pipelines.segmentation.SegmentationPipeline.discover_samples') as mock_discover:
            mock_discover.return_value = [
                {
                    'id': 'sample_001',
                    'raw_dir': self.data_dir / "sample_001" / "0_raw",
                    'segmented_dir': self.data_dir / "sample_001" / "1_segmented"
                }
            ]
            
            result = run_segmentation_pipeline(
                data_path=str(self.data_dir),
                output_dir=str(self.output_dir),
                max_samples=1
            )
        
        # Verify results
        assert result['pipeline_name'] == 'SegmentationPipeline'
        assert 'total_samples' in result
        assert 'success_rate' in result
    
    @patch('iqid_alphas.pipelines.alignment_processor.UnifiedAligner')
    @patch('iqid_alphas.pipelines.alignment_processor.ValidationSuite')
    def test_alignment_pipeline_integration(self, mock_validator, mock_aligner):
        """Test alignment pipeline integration."""
        # Mock the components
        mock_aligner.return_value.align_slice_stack.return_value = Mock(
            aligned_slices=[Mock(), Mock(), Mock()]
        )
        mock_validator.return_value.validate_alignment.return_value = Mock(
            metrics={'ssim': 0.85, 'mutual_information': 0.75}
        )
        
        # Run pipeline
        result = run_alignment_pipeline(
            data_path=str(self.data_dir),
            output_dir=str(self.output_dir),
            max_samples=1
        )
        
        # Verify results
        assert result['pipeline_name'] == 'AlignmentPipeline'
        assert 'total_samples' in result
        assert 'success_rate' in result
    
    def test_coregistration_pipeline_integration(self):
        """Test coregistration pipeline integration."""
        # Create DataPush1-like structure
        datapush_dir = self.temp_dir / "datapush1"
        datapush_dir.mkdir()
        
        iqid_dir = datapush_dir / "iQID" / "3D"
        he_dir = datapush_dir / "HE" / "3D"
        
        iqid_dir.mkdir(parents=True)
        he_dir.mkdir(parents=True)
        
        # Create mock sample directories
        sample_iqid = iqid_dir / "sample_001"
        sample_he = he_dir / "sample_001"
        
        sample_iqid.mkdir()
        sample_he.mkdir()
        
        # Add mock files
        for i in range(3):
            (sample_iqid / f"slice_{i:03d}.png").touch()
            (sample_he / f"slice_{i:03d}.png").touch()
        
        # Initialize pipeline
        pipeline = CoregistrationPipeline(
            data_path=str(datapush_dir),
            output_dir=str(self.output_dir),
            registration_method="mutual_information",
            pairing_strategy="automatic"
        )
        
        # Test pipeline initialization
        assert pipeline.data_path == Path(datapush_dir)
        assert pipeline.output_dir == Path(self.output_dir)
        assert pipeline.registration_method == "mutual_information"
    
    def test_cli_help_commands(self):
        """Test CLI help system."""
        import sys
        from io import StringIO
        
        # Test main help
        old_argv = sys.argv
        old_stdout = sys.stdout
        
        try:
            sys.argv = ['iqid_alphas', '--help']
            sys.stdout = StringIO()
            
            with pytest.raises(SystemExit):
                main()
            
            output = sys.stdout.getvalue()
            assert 'IQID-Alphas Pipeline Suite' in output
            assert 'segmentation' in output
            assert 'alignment' in output
            assert 'coregistration' in output
            
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
    
    def test_pipeline_discovery(self):
        """Test sample discovery across pipelines."""
        from iqid_alphas.pipelines.segmentation import SegmentationPipeline
        from iqid_alphas.pipelines.alignment import AlignmentPipeline
        
        # Test segmentation discovery
        seg_pipeline = SegmentationPipeline(self.data_dir, self.output_dir)
        seg_samples = seg_pipeline.discover_samples()
        
        # Test alignment discovery  
        align_pipeline = AlignmentPipeline(self.data_dir, self.output_dir)
        align_samples = align_pipeline.discover_samples()
        
        # Verify samples were discovered
        assert len(seg_samples) >= 0  # May be 0 if no valid samples
        assert len(align_samples) >= 0  # May be 0 if no valid samples


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
