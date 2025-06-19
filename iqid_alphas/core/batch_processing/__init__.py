"""
Batch Processing Package

Comprehensive batch processing system for high-throughput iQID data analysis.

Modules:
- types: Data structures and enums for batch processing
- scheduler: Job scheduling and task management
- workers: Worker processes for parallel execution
- monitor: Progress monitoring and resource tracking
- storage: Result storage and data management
- processor: Main BatchProcessor orchestrator

Usage:
    from iqid_alphas.core.batch_processing import BatchProcessor, ProcessingMode
    
    processor = BatchProcessor(mode=ProcessingMode.PARALLEL, max_workers=4)
    results = processor.process_dataset(input_paths, pipeline_config)
"""

from .types import (
    ProcessingMode,
    ProcessingStatus,
    BatchJob,
    BatchResult,
    ProcessingConfig
)
from .processor import BatchProcessor

__all__ = [
    'ProcessingMode',
    'ProcessingStatus', 
    'BatchJob',
    'BatchResult',
    'ProcessingConfig',
    'BatchProcessor'
]
