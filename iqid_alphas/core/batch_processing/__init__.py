"""
Batch Processing Package

Comprehensive batch processing system for high-throughput iQID data analysis.

The batch processing package is organized into several modules:
- types: Data structures and enums for batch processing
- scheduler: Job scheduling and resource management
- workers: Parallel worker processes for pipeline execution
- storage: Result storage and data management
- processor: Main BatchProcessor orchestrator

Usage:
    from iqid_alphas.core.batch_processing import BatchProcessor, ProcessingMode
    
    processor = BatchProcessor(max_workers=4, processing_mode=ProcessingMode.PARALLEL)
    processor.start()
    
    result = processor.process_dataset(
        dataset_paths="path/to/data",
        output_base_path="path/to/outputs",
        pipeline_stage=PipelineStage.FULL_PIPELINE
    )
    
    processor.stop()
"""

from .types import (
    ProcessingMode,
    ProcessingStatus,
    PipelineStage,
    BatchJob,
    BatchResult,
    ProcessingConfig,
    ResourceMetrics,
    WorkerStats
)
from .scheduler import JobScheduler, ResourceMonitor
from .workers import WorkerProcess, WorkerManager
from .storage import ResultsStorage
from .processor import BatchProcessor

__all__ = [
    # Types and enums
    'ProcessingMode',
    'ProcessingStatus',
    'PipelineStage',
    'BatchJob',
    'BatchResult',
    'ProcessingConfig',
    'ResourceMetrics',
    'WorkerStats',
    
    # Core components
    'JobScheduler',
    'ResourceMonitor',
    'WorkerProcess',
    'WorkerManager',
    'ResultsStorage',
    
    # Main processor
    'BatchProcessor'
]
