"""
Batch Processing Workers

Worker processes and threads for executing pipeline operations
in parallel with proper error handling and progress tracking.
"""

import logging
import time
import multiprocessing as mp
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import traceback
import os
import signal

from .types import BatchJob, ProcessingStatus, PipelineStage, ProcessingConfig
from ..tissue_segmentation import TissueSeparator
from ..alignment.aligner import UnifiedAligner
from ..coregistration import MultiModalAligner
from ..validation import ValidationSuite
from ...utils.io import load_image, save_image

logger = logging.getLogger(__name__)


class WorkerProcess:
    """
    Individual worker process for batch processing.
    
    Handles pipeline operations in isolation with proper error
    handling and resource management.
    """
    
    def __init__(self, worker_id: str, config: ProcessingConfig):
        """
        Initialize worker process.
        
        Parameters
        ----------
        worker_id : str
            Unique identifier for this worker
        config : ProcessingConfig
            Processing configuration
        """
        self.worker_id = worker_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{worker_id}")
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Worker state
        self.jobs_processed = 0
        self.errors_encountered = 0
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        
        # Performance tracking
        self.processing_times = []
        self.success_count = 0
        self.failure_count = 0
    
    def _initialize_components(self):
        """Initialize pipeline components based on config."""
        try:
            # Segmentation
            if self.config.pipeline_stage in [PipelineStage.SEGMENTATION, PipelineStage.FULL_PIPELINE]:
                self.tissue_separator = TissueSeparator()
            
            # Alignment
            if self.config.pipeline_stage in [PipelineStage.ALIGNMENT, PipelineStage.FULL_PIPELINE]:
                self.aligner = UnifiedAligner()
            
            # Coregistration
            if self.config.pipeline_stage in [PipelineStage.COREGISTRATION, PipelineStage.FULL_PIPELINE]:
                self.coregistrator = MultiModalAligner()
            
            # Validation
            if self.config.enable_validation:
                self.validator = ValidationSuite()
            
            self.logger.info(f"Worker {self.worker_id} initialized for {self.config.pipeline_stage.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize worker components: {e}")
            raise
    
    def process_job(self, job: BatchJob) -> BatchJob:
        """
        Process a single job.
        
        Parameters
        ----------
        job : BatchJob
            Job to process
            
        Returns
        -------
        BatchJob
            Processed job with results
        """
        start_time = time.time()
        self.logger.info(f"Worker {self.worker_id} starting job {job.job_id}")
        
        try:
            # Update job status
            job.status = ProcessingStatus.RUNNING
            job.started_at = start_time
            job.worker_id = self.worker_id
            
            # Process based on pipeline stage
            if self.config.pipeline_stage == PipelineStage.SEGMENTATION:
                self._process_segmentation(job)
            elif self.config.pipeline_stage == PipelineStage.ALIGNMENT:
                self._process_alignment(job)
            elif self.config.pipeline_stage == PipelineStage.COREGISTRATION:
                self._process_coregistration(job)
            elif self.config.pipeline_stage == PipelineStage.FULL_PIPELINE:
                self._process_full_pipeline(job)
            elif self.config.pipeline_stage == PipelineStage.VALIDATION:
                self._process_validation(job)
            else:
                raise ValueError(f"Unknown pipeline stage: {self.config.pipeline_stage}")
            
            # Update success metrics
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = time.time()
            job.success = True
            
            self.success_count += 1
            self.jobs_processed += 1
            
        except Exception as e:
            # Handle processing errors
            job.status = ProcessingStatus.FAILED
            job.completed_at = time.time()
            job.success = False
            job.error_message = str(e)
            job.error_traceback = traceback.format_exc()
            
            self.failure_count += 1
            self.errors_encountered += 1
            
            self.logger.error(f"Job {job.job_id} failed: {e}")
            self.logger.debug(f"Traceback: {job.error_traceback}")
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        job.processing_time_seconds = processing_time
        
        # Update heartbeat
        self.last_heartbeat = time.time()
        
        self.logger.info(f"Worker {self.worker_id} completed job {job.job_id} "
                        f"(success: {job.success}, time: {processing_time:.2f}s)")
        
        return job
    
    def _process_segmentation(self, job: BatchJob):
        """Process segmentation stage."""
        input_image = load_image(job.input_path)
        
        # Perform tissue separation
        result = self.tissue_separator.separate_tissues(
            job.input_path,
            job.output_path / "segmented"
        )
        
        job.results['segmentation_paths'] = result
        job.results['num_tissues'] = len(result) if result else 0
    
    def _process_alignment(self, job: BatchJob):
        """Process alignment stage."""
        # Load segmented images
        segmented_dir = Path(job.input_path)
        image_files = list(segmented_dir.glob("*.tif*"))
        
        if not image_files:
            raise ValueError(f"No images found in {segmented_dir}")
        
        # Load images
        images = [load_image(f) for f in image_files]
        
        # Perform alignment
        aligned_images = self.aligner.align_slice_stack(images)
        
        # Save aligned images
        output_dir = job.output_path / "aligned"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        aligned_paths = []
        for i, aligned_img in enumerate(aligned_images):
            output_path = output_dir / f"aligned_{i:03d}.tif"
            save_image(aligned_img, output_path)
            aligned_paths.append(output_path)
        
        job.results['aligned_paths'] = aligned_paths
        job.results['num_slices'] = len(aligned_images)
    
    def _process_coregistration(self, job: BatchJob):
        """Process coregistration stage."""
        # This would implement multi-modal registration
        # For now, we'll simulate the process
        input_dir = Path(job.input_path)
        modality1_files = list(input_dir.glob("modality1/*.tif*"))
        modality2_files = list(input_dir.glob("modality2/*.tif*"))
        
        if not modality1_files or not modality2_files:
            raise ValueError("Missing modality images for coregistration")
        
        # Process pairs
        registered_pairs = []
        for m1_file, m2_file in zip(modality1_files, modality2_files):
            fixed_image = load_image(m1_file)
            moving_image = load_image(m2_file)
            
            # Register images
            result = self.coregistrator.register_images(fixed_image, moving_image)
            
            # Save registered image
            output_path = job.output_path / "registered" / f"reg_{m1_file.stem}_{m2_file.stem}.tif"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(result.registered_image, output_path)
            
            registered_pairs.append({
                'fixed': str(m1_file),
                'moving': str(m2_file),
                'registered': str(output_path),
                'confidence': result.confidence
            })
        
        job.results['registered_pairs'] = registered_pairs
        job.results['num_pairs'] = len(registered_pairs)
    
    def _process_full_pipeline(self, job: BatchJob):
        """Process full pipeline (segmentation -> alignment -> coregistration)."""
        # Segmentation
        self._process_segmentation(job)
        
        # Update input path for next stage
        job.input_path = job.output_path / "segmented"
        
        # Alignment
        self._process_alignment(job)
        
        # Coregistration (if multiple modalities available)
        if self.config.coregistration_enabled:
            job.input_path = job.output_path / "aligned"
            self._process_coregistration(job)
    
    def _process_validation(self, job: BatchJob):
        """Process validation stage."""
        # Run validation on processed data
        validation_result = self.validator.validate(
            job.input_path,
            save_results=True
        )
        
        job.results['validation_result'] = {
            'mode': validation_result.mode.value,
            'success_rate': validation_result.success_rate,
            'passed_tests': validation_result.passed_tests,
            'total_tests': validation_result.total_tests,
            'is_passing': validation_result.is_passing,
            'metrics': validation_result.metrics
        }
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker performance statistics."""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        stats = {
            'worker_id': self.worker_id,
            'jobs_processed': self.jobs_processed,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'errors_encountered': self.errors_encountered,
            'runtime_seconds': runtime,
            'last_heartbeat': self.last_heartbeat,
            'avg_processing_time': sum(self.processing_times) / max(len(self.processing_times), 1),
            'throughput_jobs_per_hour': (self.jobs_processed / max(runtime / 3600, 0.001)),
            'success_rate': (self.success_count / max(self.jobs_processed, 1)) * 100
        }
        
        return stats


class WorkerManager:
    """
    Manages multiple worker processes for batch processing.
    
    Handles worker lifecycle, load balancing, and fault tolerance.
    """
    
    def __init__(self, 
                 num_workers: int,
                 config: ProcessingConfig,
                 worker_timeout: float = 300.0):
        """
        Initialize worker manager.
        
        Parameters
        ----------
        num_workers : int
            Number of worker processes to create
        config : ProcessingConfig
            Processing configuration
        worker_timeout : float
            Timeout for worker processes in seconds
        """
        self.num_workers = num_workers
        self.config = config
        self.worker_timeout = worker_timeout
        
        self.workers = {}
        self.worker_processes = {}
        self.worker_queues = {}
        self.result_queue = mp.Queue()
        
        self.is_running = False
        self.shutdown_event = mp.Event()
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start all worker processes."""
        if self.is_running:
            return
        
        self.logger.info(f"Starting {self.num_workers} worker processes")
        
        for i in range(self.num_workers):
            worker_id = f"worker_{i}"
            self._start_worker(worker_id)
        
        self.is_running = True
        self.logger.info("All workers started successfully")
    
    def stop(self, timeout: float = 30.0):
        """Stop all worker processes."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping all worker processes")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop workers
        for worker_id in list(self.workers.keys()):
            self._stop_worker(worker_id, timeout)
        
        self.is_running = False
        self.logger.info("All workers stopped")
    
    def _start_worker(self, worker_id: str):
        """Start a single worker process."""
        try:
            # Create worker queue
            worker_queue = mp.Queue()
            self.worker_queues[worker_id] = worker_queue
            
            # Create and start worker process
            worker_process = mp.Process(
                target=self._worker_main,
                args=(worker_id, worker_queue, self.result_queue, self.config, self.shutdown_event)
            )
            worker_process.start()
            
            self.worker_processes[worker_id] = worker_process
            self.workers[worker_id] = {
                'process': worker_process,
                'queue': worker_queue,
                'start_time': time.time(),
                'jobs_assigned': 0
            }
            
            self.logger.info(f"Worker {worker_id} started (PID: {worker_process.pid})")
            
        except Exception as e:
            self.logger.error(f"Failed to start worker {worker_id}: {e}")
    
    def _stop_worker(self, worker_id: str, timeout: float):
        """Stop a single worker process."""
        try:
            if worker_id in self.worker_processes:
                process = self.worker_processes[worker_id]
                
                # Send stop signal
                if worker_id in self.worker_queues:
                    self.worker_queues[worker_id].put(None)  # Stop signal
                
                # Wait for process to finish
                process.join(timeout=timeout)
                
                if process.is_alive():
                    # Force terminate if still running
                    process.terminate()
                    process.join(timeout=5.0)
                    
                    if process.is_alive():
                        # Kill if still running
                        os.kill(process.pid, signal.SIGKILL)
                
                # Clean up
                del self.worker_processes[worker_id]
                del self.workers[worker_id]
                if worker_id in self.worker_queues:
                    del self.worker_queues[worker_id]
                
                self.logger.info(f"Worker {worker_id} stopped")
                
        except Exception as e:
            self.logger.error(f"Error stopping worker {worker_id}: {e}")
    
    @staticmethod
    def _worker_main(worker_id: str, 
                    job_queue: mp.Queue, 
                    result_queue: mp.Queue,
                    config: ProcessingConfig,
                    shutdown_event: mp.Event):
        """Main function for worker process."""
        try:
            # Create worker instance
            worker = WorkerProcess(worker_id, config)
            
            # Process jobs until shutdown
            while not shutdown_event.is_set():
                try:
                    # Get job from queue
                    job = job_queue.get(timeout=1.0)
                    
                    if job is None:  # Stop signal
                        break
                    
                    # Process job
                    processed_job = worker.process_job(job)
                    
                    # Send result back
                    result_queue.put(processed_job)
                    
                except Exception as e:
                    if not shutdown_event.is_set():
                        logger.error(f"Worker {worker_id} error: {e}")
            
            logger.info(f"Worker {worker_id} shutting down")
            
        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {e}")
    
    def submit_job(self, job: BatchJob) -> bool:
        """Submit job to least loaded worker."""
        if not self.is_running:
            return False
        
        # Find worker with smallest queue
        min_jobs = float('inf')
        target_worker = None
        
        for worker_id, worker_info in self.workers.items():
            if worker_info['jobs_assigned'] < min_jobs:
                min_jobs = worker_info['jobs_assigned']
                target_worker = worker_id
        
        if target_worker is None:
            return False
        
        try:
            # Submit job to worker
            self.worker_queues[target_worker].put(job)
            self.workers[target_worker]['jobs_assigned'] += 1
            
            self.logger.debug(f"Job {job.job_id} assigned to {target_worker}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit job {job.job_id}: {e}")
            return False
    
    def get_results(self, timeout: float = 0.1) -> List[BatchJob]:
        """Get completed job results."""
        results = []
        
        while True:
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
                
                # Update worker stats
                if result.worker_id in self.workers:
                    self.workers[result.worker_id]['jobs_assigned'] -= 1
                
            except:
                break
        
        return results
    
    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all workers."""
        stats = {}
        
        for worker_id, worker_info in self.workers.items():
            process = worker_info['process']
            stats[worker_id] = {
                'pid': process.pid,
                'is_alive': process.is_alive(),
                'start_time': worker_info['start_time'],
                'jobs_assigned': worker_info['jobs_assigned'],
                'queue_size': self.worker_queues[worker_id].qsize() if worker_id in self.worker_queues else 0
            }
        
        return stats
