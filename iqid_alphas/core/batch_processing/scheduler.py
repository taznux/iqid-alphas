"""
Batch Processing Scheduler

Manages job scheduling, priority queues, and resource allocation
for batch processing operations.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union, Any
from queue import PriorityQueue, Queue, Empty
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import multiprocessing as mp

from .types import (
    ProcessingMode, ProcessingStatus, PipelineStage,
    BatchJob, BatchResult, ProcessingConfig, ResourceMetrics, WorkerStats
)

logger = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    """Job wrapper for scheduling with priority."""
    priority: int
    job: BatchJob
    submit_time: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """Enable priority queue ordering."""
        return self.priority < other.priority


class JobScheduler:
    """
    Advanced job scheduler with priority queues and resource management.
    
    Manages job queues, worker pools, and resource allocation for
    optimal batch processing performance.
    """
    
    def __init__(self,
                 max_workers: int = None,
                 processing_mode: ProcessingMode = ProcessingMode.PARALLEL,
                 max_queue_size: int = 1000,
                 priority_levels: int = 3):
        """
        Initialize the job scheduler.
        
        Parameters
        ----------
        max_workers : int, optional
            Maximum number of worker processes/threads
        processing_mode : ProcessingMode
            Processing execution mode
        max_queue_size : int
            Maximum size of job queue
        priority_levels : int
            Number of priority levels (0 = highest)
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.processing_mode = processing_mode
        self.max_queue_size = max_queue_size
        self.priority_levels = priority_levels
        
        # Job queues
        self.job_queue = PriorityQueue(maxsize=max_queue_size)
        self.result_queue = Queue()
        self.failed_queue = Queue()
        
        # Worker management
        self.workers = {}
        self.worker_stats = {}
        self.active_jobs = {}
        
        # Executor
        self.executor = None
        self.futures = {}
        
        # State management
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.stats_lock = threading.Lock()
        
        # Statistics
        self.total_jobs_submitted = 0
        self.total_jobs_completed = 0
        self.total_jobs_failed = 0
        self.start_time = None
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the job scheduler and worker pool."""
        if self.is_running:
            self.logger.warning("Scheduler already running")
            return
        
        self.logger.info(f"Starting job scheduler with {self.max_workers} workers")
        
        # Initialize executor based on processing mode
        if self.processing_mode == ProcessingMode.PARALLEL:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Initialize worker stats
        for i in range(self.max_workers):
            worker_id = f"worker_{i}"
            self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
        
        self.is_running = True
        self.start_time = time.time()
        self.shutdown_event.clear()
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        self.logger.info("Job scheduler started successfully")
    
    def stop(self, timeout: float = 30.0):
        """Stop the job scheduler and wait for completion."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping job scheduler...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.is_running = False
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True, timeout=timeout)
        
        # Stop resource monitoring
        self.resource_monitor.stop()
        
        self.logger.info("Job scheduler stopped")
    
    def submit_job(self, job: BatchJob, priority: int = 1) -> bool:
        """
        Submit a job for processing.
        
        Parameters
        ----------
        job : BatchJob
            Job to submit
        priority : int
            Job priority (0 = highest, higher numbers = lower priority)
            
        Returns
        -------
        bool
            True if job was successfully queued
        """
        if not self.is_running:
            self.logger.error("Cannot submit job: scheduler not running")
            return False
        
        if priority < 0 or priority >= self.priority_levels:
            priority = min(max(priority, 0), self.priority_levels - 1)
        
        try:
            scheduled_job = ScheduledJob(priority=priority, job=job)
            self.job_queue.put(scheduled_job, timeout=1.0)
            
            with self.stats_lock:
                self.total_jobs_submitted += 1
            
            job.status = ProcessingStatus.PENDING
            self.logger.debug(f"Job {job.job_id} queued with priority {priority}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit job {job.job_id}: {e}")
            return False
    
    def submit_jobs(self, jobs: List[BatchJob], priority: int = 1) -> int:
        """
        Submit multiple jobs for processing.
        
        Parameters
        ----------
        jobs : List[BatchJob]
            List of jobs to submit
        priority : int
            Job priority for all jobs
            
        Returns
        -------
        int
            Number of jobs successfully queued
        """
        submitted = 0
        for job in jobs:
            if self.submit_job(job, priority):
                submitted += 1
        
        self.logger.info(f"Submitted {submitted}/{len(jobs)} jobs for processing")
        return submitted
    
    def process_queue(self) -> List[Future]:
        """
        Process jobs from the queue.
        
        Returns
        -------
        List[Future]
            List of submitted futures
        """
        futures = []
        
        while self.is_running and not self.job_queue.empty():
            try:
                # Get job from queue
                scheduled_job = self.job_queue.get(timeout=0.1)
                job = scheduled_job.job
                
                # Submit job to executor
                future = self.executor.submit(self._process_job_wrapper, job)
                futures.append(future)
                self.futures[future] = job
                
                # Update job status
                job.status = ProcessingStatus.RUNNING
                job.started_at = time.time()
                self.active_jobs[job.job_id] = job
                
                self.logger.debug(f"Job {job.job_id} submitted to worker pool")
                
            except Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing queue: {e}")
                break
        
        return futures
    
    def _process_job_wrapper(self, job: BatchJob) -> BatchJob:
        """
        Wrapper function for processing jobs in worker processes.
        
        Parameters
        ----------
        job : BatchJob
            Job to process
            
        Returns
        -------
        BatchJob
            Processed job with results
        """
        try:
            # This would be implemented by the actual processor
            # For now, we'll simulate processing
            time.sleep(0.1)  # Simulate work
            
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = time.time()
            job.success = True
            
            with self.stats_lock:
                self.total_jobs_completed += 1
            
            return job
            
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.completed_at = time.time()
            job.success = False
            job.error_message = str(e)
            
            with self.stats_lock:
                self.total_jobs_failed += 1
            
            self.logger.error(f"Job {job.job_id} failed: {e}")
            return job
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingStatus]:
        """Get the status of a specific job."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].status
        return None
    
    def get_queue_size(self) -> int:
        """Get the current queue size."""
        return self.job_queue.qsize()
    
    def get_active_jobs(self) -> int:
        """Get the number of active jobs."""
        return len(self.active_jobs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self.stats_lock:
            current_time = time.time()
            runtime = current_time - self.start_time if self.start_time else 0
            
            stats = {
                'runtime_seconds': runtime,
                'total_jobs_submitted': self.total_jobs_submitted,
                'total_jobs_completed': self.total_jobs_completed,
                'total_jobs_failed': self.total_jobs_failed,
                'queue_size': self.get_queue_size(),
                'active_jobs': self.get_active_jobs(),
                'success_rate': (self.total_jobs_completed / max(self.total_jobs_submitted, 1)) * 100,
                'throughput_jobs_per_hour': (self.total_jobs_completed / max(runtime / 3600, 0.001)),
                'worker_stats': dict(self.worker_stats),
                'resource_metrics': self.resource_monitor.get_current_metrics()
            }
        
        return stats


class ResourceMonitor:
    """Monitor system resources during batch processing."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        """
        Initialize resource monitor.
        
        Parameters
        ----------
        monitoring_interval : float
            Interval between resource checks in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.current_metrics = ResourceMetrics()
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        while self.is_monitoring:
            try:
                # This would use psutil or similar for real resource monitoring
                # For now, we'll simulate metrics
                self.current_metrics = ResourceMetrics(
                    timestamp=time.time(),
                    cpu_percent=50.0,  # Simulated
                    memory_used_gb=2.0,  # Simulated
                    memory_available_gb=6.0,  # Simulated
                    disk_used_gb=100.0,  # Simulated
                    disk_available_gb=400.0,  # Simulated
                    active_workers=4  # Simulated
                )
                
                self.metrics_history.append(self.current_metrics)
                
                # Keep only recent history (last hour)
                cutoff_time = time.time() - 3600
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
            except Exception as e:
                self.logger.error(f"Error monitoring resources: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics."""
        return self.current_metrics
    
    def get_metrics_history(self) -> List[ResourceMetrics]:
        """Get resource metrics history."""
        return self.metrics_history.copy()
