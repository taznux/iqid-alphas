"""
Simple tests for the batch processing package.
"""
import unittest
import tempfile
import os
from pathlib import Path

from iqid_alphas.core.batch_processing import (
    ProcessingMode, ProcessingStatus, PipelineStage,
    ProcessingConfig
)


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_processing_config(self):
        """Test ProcessingConfig creation."""
        config = ProcessingConfig()
        self.assertIsInstance(config, ProcessingConfig)
    
    def test_processing_mode_enum(self):
        """Test ProcessingMode enum."""
        self.assertTrue(hasattr(ProcessingMode, 'SEQUENTIAL'))
        self.assertTrue(hasattr(ProcessingMode, 'PARALLEL'))
    
    def test_processing_status_enum(self):
        """Test ProcessingStatus enum."""
        self.assertTrue(hasattr(ProcessingStatus, 'PENDING'))
        self.assertTrue(hasattr(ProcessingStatus, 'RUNNING'))
        self.assertTrue(hasattr(ProcessingStatus, 'COMPLETED'))
    
    def test_pipeline_stage_enum(self):
        """Test PipelineStage enum."""
        self.assertTrue(hasattr(PipelineStage, 'SEGMENTATION'))
        self.assertTrue(hasattr(PipelineStage, 'ALIGNMENT'))
        self.assertTrue(hasattr(PipelineStage, 'COREGISTRATION'))
    
    def test_imports_work(self):
        """Test that all imports work correctly."""
        from iqid_alphas.core.batch_processing import (
            BatchProcessor, JobScheduler, WorkerManager, ResultsStorage
        )
        # If we get here, imports worked
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
