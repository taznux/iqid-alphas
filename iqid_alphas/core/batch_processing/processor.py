"""
Main Batch Processor

Orchestrates the entire batch processing system, coordinating
scheduling, workers, storage, and monitoring for high-throughput
iQID data processing.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from concurrent.futures import as_completed
import threading

from .types import (
    ProcessingMode, ProcessingStatus, PipelineStage,
    BatchJob, BatchResult, ProcessingConfig
)
from .scheduler import JobScheduler, ResourceMonitor
from .workers import WorkerManager
from .storage import ResultsStorage
from ...utils.io_utils import list_directories, natural_sort

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Main batch processor for high-throughput iQID data analysis.
    
    Coordinates all aspects of batch processing including job scheduling,
    worker management, result storage, and monitoring.
    """
    
    def __init__(self,
                 max_workers: int = None,
                 processing_mode: ProcessingMode = ProcessingMode.PARALLEL,
                 storage_path: Union[str, Path] = "./batch_results",
                 storage_format: str = "sqlite",
                 config: Optional[ProcessingConfig] = None):
        """
        Initialize the batch processor.
        
        Parameters
        ----------
        max_workers : int, optional
            Maximum number of worker processes
        processing_mode : ProcessingMode
            Processing execution mode
        storage_path : Union[str, Path]
            Path for storing results
        storage_format : str
            Storage format ('sqlite', 'json', 'pickle')
        config : Optional[ProcessingConfig]
            Processing configuration
        """
        self.max_workers = max_workers
        self.processing_mode = processing_mode
        self.storage_path = Path(storage_path)
        self.storage_format = storage_format
        self.config = config or ProcessingConfig()
        
        # Core components
        self.scheduler = JobScheduler(
            max_workers=self.max_workers,
            processing_mode=self.processing_mode
        )
        
        self.worker_manager = WorkerManager(
            num_workers=self.max_workers or 4,
            config=self.config
        )
        
        self.storage = ResultsStorage(
            storage_path=self.storage_path / "results.db" if storage_format == "sqlite" else self.storage_path,
            storage_format=self.storage_format
        )
        
        # State management
        self.is_running = False
        self.current_batch_id = None
        self.active_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        
        # Performance tracking
        self.batch_start_time = None
        self.total_jobs_processed = 0
        self.processing_history = []
        
        # Monitoring
        self.monitor_thread = None
        self.monitoring_interval = 10.0  # seconds
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the batch processor."""
        if self.is_running:
            self.logger.warning("Batch processor already running")
            return
        
        self.logger.info("Starting batch processor...")
        
        try:
            # Start core components
            self.scheduler.start()
            self.worker_manager.start()
            
            # Start monitoring
            self._start_monitoring()
            
            self.is_running = True
            self.logger.info("Batch processor started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start batch processor: {e}")
            self.stop()
            raise
    
    def stop(self, timeout: float = 30.0):
        """Stop the batch processor."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping batch processor...")
        
        # Stop monitoring
        self._stop_monitoring()
        
        # Stop components
        if hasattr(self, 'scheduler'):
            self.scheduler.stop(timeout=timeout)
        if hasattr(self, 'worker_manager'):
            self.worker_manager.stop(timeout=timeout)
        
        self.is_running = False
        self.logger.info("Batch processor stopped")
    
    def process_dataset(self,
                       dataset_paths: Union[str, Path, List[Union[str, Path]]],
                       output_base_path: Union[str, Path],
                       pipeline_stage: PipelineStage = PipelineStage.FULL_PIPELINE,
                       job_priority: int = 1) -> BatchResult:
        """
        Process one or more datasets.
        
        Parameters
        ----------
        dataset_paths : Union[str, Path, List[Union[str, Path]]]
            Path(s) to dataset directories
        output_base_path : Union[str, Path]
            Base path for output results
        pipeline_stage : PipelineStage
            Which pipeline stage to execute
        job_priority : int
            Job priority (0 = highest)
            
        Returns
        -------
        BatchResult
            Batch processing results
        """
        if not self.is_running:
            raise RuntimeError("Batch processor not started. Call start() first.")
        
        # Normalize input paths
        if isinstance(dataset_paths, (str, Path)):
            dataset_paths = [Path(dataset_paths)]
        else:
            dataset_paths = [Path(p) for p in dataset_paths]
        
        output_base_path = Path(output_base_path)
        
        self.logger.info(f"Starting batch processing of {len(dataset_paths)} datasets")
        
        # Create batch ID
        self.current_batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.batch_start_time = time.time()
        
        # Discover and create jobs
        jobs = self._create_jobs(dataset_paths, output_base_path, pipeline_stage)
        
        if not jobs:
            self.logger.warning("No jobs created from dataset paths")
            return self._create_empty_batch_result()
        
        # Submit jobs for processing
        submitted_jobs = self._submit_jobs(jobs, job_priority)
        
        # Wait for completion and collect results
        batch_result = self._wait_for_completion(submitted_jobs)
        
        # Store batch result
        self.storage.store_batch_result(batch_result)
        self.processing_history.append(batch_result)
        
        self.logger.info(f"Batch processing completed: {batch_result.completed_jobs}/{batch_result.total_jobs} jobs successful")
        
        return batch_result
    
    def _create_jobs(self,
                    dataset_paths: List[Path],
                    output_base_path: Path,
                    pipeline_stage: PipelineStage) -> List[BatchJob]:
        """Create batch jobs from dataset paths."""
        jobs = []
        
        for dataset_path in dataset_paths:
            if not dataset_path.exists():
                self.logger.warning(f"Dataset path does not exist: {dataset_path}")
                continue
            
            # Discover samples in dataset
            if self.config.auto_discover_samples:
                sample_paths = self._discover_samples(dataset_path)
            else:
                sample_paths = [dataset_path]
            
            # Create jobs for each sample
            for sample_path in sample_paths:
                job_id = f"{self.current_batch_id}_{len(jobs):04d}"
                output_path = output_base_path / sample_path.name
                
                job = BatchJob(
                    job_id=job_id,
                    input_path=sample_path,
                    output_path=output_path,
                    pipeline_stage=pipeline_stage,
                    batch_id=self.current_batch_id
                )
                
                jobs.append(job)
        
        self.logger.info(f"Created {len(jobs)} jobs from {len(dataset_paths)} datasets")
        return jobs
    
    def _discover_samples(self, dataset_path: Path) -> List[Path]:
        """Discover individual samples within a dataset."""
        samples = []
        
        # Look for common iQID data structure patterns
        patterns = [
            "*/raw/*.tif*",  # Raw images
            "*/1_segmented/*.tif*",  # Segmented images
            "*/2_aligned/*.tif*",  # Aligned images
        ]
        
        for pattern in patterns:
            matches = list(dataset_path.glob(pattern))
            if matches:
                # Group by parent directory (sample)
                sample_dirs = {m.parent.parent for m in matches}
                samples.extend(sample_dirs)
                break
        
        # If no structured data found, treat each subdirectory as a sample
        if not samples:
            samples = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        # Remove duplicates and sort
        samples = natural_sort(list(set(samples)))
        
        self.logger.debug(f"Discovered {len(samples)} samples in {dataset_path}")
        return samples
    
    def _submit_jobs(self, jobs: List[BatchJob], priority: int) -> List[BatchJob]:
        """Submit jobs to the scheduler."""
        submitted_jobs = []
        
        for job in jobs:
            try:
                # Submit to scheduler
                if self.scheduler.submit_job(job, priority=priority):
                    self.active_jobs[job.job_id] = job
                    submitted_jobs.append(job)
                else:
                    self.logger.warning(f"Failed to submit job {job.job_id}")
                    job.status = ProcessingStatus.FAILED
                    job.error_message = "Failed to submit to scheduler"
                    self.failed_jobs[job.job_id] = job
                    
            except Exception as e:
                self.logger.error(f"Error submitting job {job.job_id}: {e}")
                job.status = ProcessingStatus.FAILED
                job.error_message = str(e)
                self.failed_jobs[job.job_id] = job
        
        self.logger.info(f"Submitted {len(submitted_jobs)}/{len(jobs)} jobs")
        return submitted_jobs
    
    def _wait_for_completion(self, jobs: List[BatchJob]) -> BatchResult:
        """Wait for all jobs to complete and collect results."""
        start_time = time.time()
        completed_count = 0
        failed_count = 0
        
        while completed_count + failed_count < len(jobs):
            # Process completed jobs from workers
            completed_jobs = self.worker_manager.get_results()
            
            for job in completed_jobs:
                # Update job tracking
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
                
                if job.success:
                    self.completed_jobs[job.job_id] = job
                    completed_count += 1
                else:
                    self.failed_jobs[job.job_id] = job
                    failed_count += 1
                
                # Store job result
                self.storage.store_job(job)
                
                self.logger.debug(f"Job {job.job_id} completed (success: {job.success})")
            
            # Check for stalled jobs
            current_time = time.time()
            for job in list(self.active_jobs.values()):
                if (job.started_at and 
                    current_time - job.started_at > self.config.job_timeout_seconds):
                    
                    self.logger.warning(f"Job {job.job_id} timed out")
                    job.status = ProcessingStatus.FAILED
                    job.error_message = "Job timeout"
                    job.completed_at = current_time
                    
                    del self.active_jobs[job.job_id]
                    self.failed_jobs[job.job_id] = job
                    failed_count += 1
                    
                    self.storage.store_job(job)
            
            # Process scheduler queue
            futures = self.scheduler.process_queue()
            
            # Submit jobs to workers
            for job in list(self.active_jobs.values()):
                if job.status == ProcessingStatus.PENDING:
                    if self.worker_manager.submit_job(job):
                        job.status = ProcessingStatus.RUNNING
            
            # Brief sleep to avoid busy waiting
            time.sleep(0.1)
        
        # Calculate batch statistics
        total_time = time.time() - start_time
        
        # Create batch result
        batch_result = BatchResult(
            total_jobs=len(jobs),
            completed_jobs=completed_count,
            failed_job_count=failed_count,
            cancelled_jobs=0,
            total_processing_time=total_time,
            average_processing_time=total_time / len(jobs) if jobs else 0,
            throughput_jobs_per_hour=(len(jobs) / max(total_time / 3600, 0.001)),
            peak_memory_usage_gb=0.0,  # Would be tracked by resource monitor
            total_cpu_time_seconds=0.0,  # Would be summed from jobs
            average_quality_score=0.0,  # Would be calculated from results
            quality_distribution={},
            successful_results=[Path(job.output_path) for job in self.completed_jobs.values()],
            failed_jobs=list(self.failed_jobs.values()),
            started_at=start_time,
            completed_at=time.time()
        )
        
        return batch_result
    
    def _create_empty_batch_result(self) -> BatchResult:
        """Create an empty batch result for when no jobs are created."""
        return BatchResult(
            total_jobs=0,
            completed_jobs=0,
            failed_job_count=0,
            cancelled_jobs=0,
            total_processing_time=0.0,
            average_processing_time=0.0,
            throughput_jobs_per_hour=0.0,
            peak_memory_usage_gb=0.0,
            total_cpu_time_seconds=0.0,
            average_quality_score=0.0,
            quality_distribution={},
            started_at=time.time(),
            completed_at=time.time()
        )
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("System monitoring started")
    
    def _stop_monitoring(self):
        """Stop background monitoring."""
        # Monitoring thread is daemon, so it will stop when main thread stops
        self.logger.info("System monitoring stopped")
    
    def _monitor_system(self):
        """Monitor system resources and job progress."""
        while self.is_running:
            try:
                # Get system statistics
                scheduler_stats = self.scheduler.get_statistics()
                worker_stats = self.worker_manager.get_worker_stats()
                storage_stats = self.storage.get_statistics()
                
                # Log progress
                if self.current_batch_id:
                    active_jobs = len(self.active_jobs)
                    completed_jobs = len(self.completed_jobs)
                    failed_jobs = len(self.failed_jobs)
                    total_jobs = active_jobs + completed_jobs + failed_jobs
                    
                    if total_jobs > 0:
                        progress = ((completed_jobs + failed_jobs) / total_jobs) * 100
                        self.logger.info(f"Batch progress: {progress:.1f}% "
                                       f"({completed_jobs}/{total_jobs} completed, "
                                       f"{failed_jobs} failed, {active_jobs} active)")
                
                # Store resource metrics if available
                if 'resource_metrics' in scheduler_stats:
                    self.storage.store_resource_metrics(scheduler_stats['resource_metrics'])
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingStatus]:
        """Get the status of a specific job."""
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].status
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            return ProcessingStatus.COMPLETED
        
        # Check failed jobs
        if job_id in self.failed_jobs:
            return ProcessingStatus.FAILED
        
        # Check scheduler
        return self.scheduler.get_job_status(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = ProcessingStatus.CANCELLED
            job.completed_at = time.time()
            
            del self.active_jobs[job_id]
            self.failed_jobs[job_id] = job
            
            self.storage.store_job(job)
            
            self.logger.info(f"Job {job_id} cancelled")
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'processor': {
                'is_running': self.is_running,
                'current_batch_id': self.current_batch_id,
                'total_jobs_processed': self.total_jobs_processed,
                'active_jobs': len(self.active_jobs),
                'completed_jobs': len(self.completed_jobs),
                'failed_jobs': len(self.failed_jobs)
            },
            'scheduler': self.scheduler.get_statistics() if hasattr(self, 'scheduler') else {},
            'workers': self.worker_manager.get_worker_stats() if hasattr(self, 'worker_manager') else {},
            'storage': self.storage.get_statistics() if hasattr(self, 'storage') else {}
        }
        
        return stats
    
    def generate_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Generate a comprehensive processing report."""
        if not self.processing_history:
            return "No batch processing history available."
        
        report = []
        
        # Header
        report.append("# Batch Processing Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        total_batches = len(self.processing_history)
        total_jobs = sum(b.total_jobs for b in self.processing_history)
        total_completed = sum(b.completed_jobs for b in self.processing_history)
        total_failed = sum(b.failed_jobs for b in self.processing_history)
        
        report.append("## Summary")
        report.append(f"- Total batch runs: {total_batches}")
        report.append(f"- Total jobs processed: {total_jobs}")
        report.append(f"- Successfully completed: {total_completed}")
        report.append(f"- Failed jobs: {total_failed}")
        report.append(f"- Overall success rate: {(total_completed/total_jobs)*100:.1f}%" if total_jobs > 0 else "- Overall success rate: 0.0%")
        report.append("")
        
        # Individual batch results
        report.append("## Batch Results")
        for i, batch_result in enumerate(self.processing_history, 1):
            report.append(f"### Batch {i}")
            report.append(f"- Jobs: {batch_result.completed_jobs}/{batch_result.total_jobs}")
            report.append(f"- Success rate: {batch_result.success_rate:.1f}%")
            report.append(f"- Processing time: {batch_result.total_processing_time:.2f}s")
            report.append(f"- Throughput: {batch_result.throughput_jobs_per_hour:.1f} jobs/hour")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Batch processing report saved to {output_path}")
        
        return report_text
