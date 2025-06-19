"""
Data Types and Enums for Batch Processing

Contains all data structures, enums, and configuration classes
used throughout the batch processing package.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import uuid


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel" 
    DISTRIBUTED = "distributed"


class ProcessingStatus(Enum):
    """Status of batch processing jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class PipelineStage(Enum):
    """Available pipeline stages."""
    SEGMENTATION = "segmentation"
    ALIGNMENT = "alignment"
    COREGISTRATION = "coregistration"
    VALIDATION = "validation"
    FULL_PIPELINE = "full_pipeline"


@dataclass
class ProcessingConfig:
    """Configuration for batch processing operations."""
    
    # Pipeline configuration
    pipeline_stage: PipelineStage = PipelineStage.FULL_PIPELINE
    enable_validation: bool = True
    save_intermediate: bool = True
    
    # Performance configuration
    processing_mode: ProcessingMode = ProcessingMode.PARALLEL
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    timeout_minutes: int = 60
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 5
    
    # Output configuration
    output_format: str = "numpy"  # numpy, tiff, hdf5
    compression: bool = True
    quality_threshold: float = 0.7
    
    # Resource monitoring
    monitor_resources: bool = True
    log_progress: bool = True
    progress_callback: Optional[Callable] = None
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchJob:
    """Individual batch processing job."""
    
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_path: Path = None
    output_path: Path = None
    config: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Status tracking
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Progress tracking
    progress: float = 0.0
    current_stage: Optional[str] = None
    estimated_completion: Optional[float] = None
    
    # Results and errors
    result_path: Optional[Path] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Resource usage
    memory_used_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def processing_time(self) -> Optional[float]:
        """Calculate processing time in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed (successfully or with failure)."""
        return self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]
    
    @property
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == ProcessingStatus.COMPLETED


@dataclass
class BatchResult:
    """Results from batch processing operation."""
    
    # Job tracking
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    
    # Performance metrics
    total_processing_time: float
    average_processing_time: float
    throughput_jobs_per_hour: float
    
    # Resource usage
    peak_memory_usage_gb: float
    total_cpu_time_seconds: float
    
    # Quality metrics
    average_quality_score: float
    quality_distribution: Dict[str, int]
    
    # Results
    successful_results: List[Path] = field(default_factory=list)
    failed_jobs: List[BatchJob] = field(default_factory=list)
    
    # Summary statistics
    stage_timings: Dict[str, float] = field(default_factory=dict)
    error_summary: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs / self.total_jobs) * 100.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_jobs == 0:
            return 0.0
        return (self.failed_jobs / self.total_jobs) * 100.0


@dataclass
class ResourceMetrics:
    """System resource usage metrics."""
    
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_available_gb: float = 0.0
    active_workers: int = 0
    
    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage."""
        total_memory = self.memory_used_gb + self.memory_available_gb
        if total_memory == 0:
            return 0.0
        return (self.memory_used_gb / total_memory) * 100.0


@dataclass
class WorkerStats:
    """Statistics for individual worker processes."""
    
    worker_id: str
    jobs_processed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    errors_encountered: int = 0
    last_job_completed: Optional[float] = None
    status: str = "idle"  # idle, working, error, terminated
    
    @property
    def throughput(self) -> float:
        """Calculate worker throughput (jobs per hour)."""
        if self.total_processing_time == 0:
            return 0.0
        return (self.jobs_processed / self.total_processing_time) * 3600.0
