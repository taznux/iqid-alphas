"""
Batch Processing System

Comprehensive batch processing framework for running pipeline operations
on multiple datasets, with parallel processing, progress tracking, and 
comprehensive error handling.
"""

import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import traceback

import numpy as np
from tqdm import tqdm

from .tissue_segmentation import TissueSeparator
from .alignment.aligner import UnifiedAligner
from .coregistration import MultiModalAligner, RegistrationMethod
from .validation import ValidationSuite, ValidationMode, ValidationResult
from .mapping import ReverseMapper
from ..utils.io_utils import load_image, save_image, natural_sort, list_directories
from ..utils.math_utils import calculate_statistics


class ProcessingMode(Enum):
    """Available processing modes."""
    SEGMENTATION = "segmentation"
    ALIGNMENT = "alignment" 
    COREGISTRATION = "coregistration"
    VALIDATION = "validation"
    FULL_PIPELINE = "full_pipeline"


class ParallelizationStrategy(Enum):
    """Parallelization strategies."""
    SEQUENTIAL = "sequential"
    THREAD_BASED = "thread_based"
    PROCESS_BASED = "process_based"
    HYBRID = "hybrid"


@dataclass
class ProcessingJob:
    """Individual processing job configuration."""
    job_id: str
    input_path: Path
    output_path: Path
    mode: ProcessingMode
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ProcessingResult:
    """Result from a processing job."""
    job_id: str
    mode: ProcessingMode
    success: bool = False
    processing_time: float = 0.0
    output_files: List[Path] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Results from a batch processing run."""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    processing_time: float = 0.0
    results: List[ProcessingResult] = field(default_factory=list)
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs / self.total_jobs) * 100.0


class BatchProcessor:
    """
    Comprehensive batch processing system for the iQID pipeline.
    
    Supports parallel processing of multiple datasets with configurable
    parallelization strategies, progress tracking, and error handling.
    """
    
    def __init__(self,
                 max_workers: Optional[int] = None,
                 parallelization: ParallelizationStrategy = ParallelizationStrategy.PROCESS_BASED,
                 temp_dir: Optional[Union[str, Path]] = None,
                 save_intermediate: bool = True,
                 validation_enabled: bool = True):
        """
        Initialize the batch processor.
        
        Parameters
        ----------
        max_workers : int, optional
            Maximum number of parallel workers (default: CPU count)
        parallelization : ParallelizationStrategy
            Strategy for parallelization
        temp_dir : str or Path, optional
            Directory for temporary files
        save_intermediate : bool
            Whether to save intermediate processing results
        validation_enabled : bool
            Whether to run validation after processing
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.parallelization = parallelization
        self.temp_dir = Path(temp_dir) if temp_dir else Path.cwd() / "temp"
        self.save_intermediate = save_intermediate
        self.validation_enabled = validation_enabled
        
        # Create temp directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tissue_separator = TissueSeparator()
        self.aligner = UnifiedAligner()
        self.multimodal_aligner = MultiModalAligner()
        self.validation_suite = ValidationSuite() if validation_enabled else None
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Job queue and results
        self.job_queue: List[ProcessingJob] = []
        self.processing_history: List[BatchResult] = []
    
    def add_job(self,
                input_path: Union[str, Path],
                output_path: Union[str, Path],
                mode: ProcessingMode,
                job_id: Optional[str] = None,
                parameters: Optional[Dict[str, Any]] = None,
                priority: int = 0) -> str:
        """
        Add a processing job to the queue.
        
        Parameters
        ----------
        input_path : str or Path
            Path to input data
        output_path : str or Path
            Path for output data
        mode : ProcessingMode
            Type of processing to perform
        job_id : str, optional
            Unique identifier for the job
        parameters : dict, optional
            Job-specific parameters
        priority : int
            Job priority (higher numbers processed first)
            
        Returns
        -------
        str
            Job ID
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not job_id:
            job_id = f"{mode.value}_{len(self.job_queue)}_{int(time.time())}"
        
        job = ProcessingJob(
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            mode=mode,
            parameters=parameters or {},
            priority=priority
        )
        
        self.job_queue.append(job)
        self.logger.info(f"Added job {job_id}: {mode.value} processing")
        
        return job_id
    
    def add_directory_batch(self,
                           input_root: Union[str, Path],
                           output_root: Union[str, Path],
                           mode: ProcessingMode,
                           pattern: str = "*",
                           recursive: bool = True) -> List[str]:
        """
        Add multiple jobs from directory structure.
        
        Parameters
        ----------
        input_root : str or Path
            Root directory containing input data
        output_root : str or Path
            Root directory for output data
        mode : ProcessingMode
            Type of processing to perform
        pattern : str
            Pattern to match directories/files
        recursive : bool
            Whether to search recursively
            
        Returns
        -------
        List[str]
            List of job IDs
        """
        input_root = Path(input_root)
        output_root = Path(output_root)
        
        job_ids = []
        
        # Find input directories
        if recursive:
            input_dirs = list(input_root.rglob(pattern))
        else:
            input_dirs = list(input_root.glob(pattern))
        
        # Filter to directories only
        input_dirs = [d for d in input_dirs if d.is_dir()]
        
        for input_dir in input_dirs:
            # Calculate relative path and create output path
            rel_path = input_dir.relative_to(input_root)
            output_dir = output_root / rel_path
            
            job_id = self.add_job(
                input_path=input_dir,
                output_path=output_dir,
                mode=mode
            )
            job_ids.append(job_id)
        
        self.logger.info(f"Added {len(job_ids)} jobs from directory batch")
        return job_ids
    
    def process_batch(self,
                     progress_callback: Optional[Callable] = None,
                     save_results: bool = True) -> BatchResult:
        """
        Process all jobs in the queue.
        
        Parameters
        ----------
        progress_callback : callable, optional
            Function to call with progress updates
        save_results : bool
            Whether to save batch results to disk
            
        Returns
        -------
        BatchResult
            Comprehensive batch processing results
        """
        if not self.job_queue:
            self.logger.warning("No jobs in queue")
            return BatchResult()
        
        start_time = time.time()
        
        # Sort jobs by priority (higher first)
        self.job_queue.sort(key=lambda x: x.priority, reverse=True)
        
        result = BatchResult(total_jobs=len(self.job_queue))
        
        self.logger.info(f"Starting batch processing of {result.total_jobs} jobs")
        
        try:
            if self.parallelization == ParallelizationStrategy.SEQUENTIAL:
                results = self._process_sequential(progress_callback)
            elif self.parallelization == ParallelizationStrategy.THREAD_BASED:
                results = self._process_threaded(progress_callback)
            elif self.parallelization == ParallelizationStrategy.PROCESS_BASED:
                results = self._process_multiprocessing(progress_callback)
            elif self.parallelization == ParallelizationStrategy.HYBRID:
                results = self._process_hybrid(progress_callback)
            else:
                raise ValueError(f"Unknown parallelization strategy: {self.parallelization}")
            
            # Collect results
            for job_result in results:
                result.results.append(job_result)
                if job_result.success:
                    result.completed_jobs += 1
                else:
                    result.failed_jobs += 1
            
            # Calculate summary metrics
            result.processing_time = time.time() - start_time
            result.summary_metrics = self._calculate_summary_metrics(result.results)
            
            # Add to history
            self.processing_history.append(result)
            
            # Save results
            if save_results:
                self._save_batch_result(result)
            
            # Clear job queue
            self.job_queue.clear()
            
            self.logger.info(f"Batch processing completed: {result.success_rate:.1f}% success rate")
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            result.processing_time = time.time() - start_time
            self.processing_history.append(result)
        
        return result
    
    def _process_sequential(self, progress_callback: Optional[Callable] = None) -> List[ProcessingResult]:
        """Process jobs sequentially."""
        results = []
        
        with tqdm(total=len(self.job_queue), desc="Processing jobs") as pbar:
            for i, job in enumerate(self.job_queue):
                result = self._process_single_job(job)
                results.append(result)
                
                pbar.update(1)
                if progress_callback:
                    progress_callback(i + 1, len(self.job_queue), result)
        
        return results
    
    def _process_threaded(self, progress_callback: Optional[Callable] = None) -> List[ProcessingResult]:
        """Process jobs using thread-based parallelization."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(self._process_single_job, job): job 
                           for job in self.job_queue}
            
            # Process completed jobs
            with tqdm(total=len(self.job_queue), desc="Processing jobs") as pbar:
                completed = 0
                for future in as_completed(future_to_job):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    pbar.update(1)
                    if progress_callback:
                        progress_callback(completed, len(self.job_queue), result)
        
        return results
    
    def _process_multiprocessing(self, progress_callback: Optional[Callable] = None) -> List[ProcessingResult]:
        """Process jobs using multiprocessing."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(_process_job_worker, job): job 
                           for job in self.job_queue}
            
            # Process completed jobs
            with tqdm(total=len(self.job_queue), desc="Processing jobs") as pbar:
                completed = 0
                for future in as_completed(future_to_job):
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(completed, len(self.job_queue), result)
                    
                    except Exception as e:
                        job = future_to_job[future]
                        error_result = ProcessingResult(
                            job_id=job.job_id,
                            mode=job.mode,
                            success=False,
                            errors=[f"Worker process failed: {e}"]
                        )
                        results.append(error_result)
                        completed += 1
                        pbar.update(1)
        
        return results
    
    def _process_hybrid(self, progress_callback: Optional[Callable] = None) -> List[ProcessingResult]:
        """Process jobs using hybrid parallelization (process for CPU-bound, thread for I/O)."""
        # Separate jobs by type
        cpu_intensive_jobs = []
        io_intensive_jobs = []
        
        for job in self.job_queue:
            if job.mode in [ProcessingMode.SEGMENTATION, ProcessingMode.ALIGNMENT, ProcessingMode.COREGISTRATION]:
                cpu_intensive_jobs.append(job)
            else:
                io_intensive_jobs.append(job)
        
        results = []
        completed = 0
        total_jobs = len(self.job_queue)
        
        with tqdm(total=total_jobs, desc="Processing jobs") as pbar:
            # Process CPU-intensive jobs with multiprocessing
            if cpu_intensive_jobs:
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_job = {executor.submit(_process_job_worker, job): job 
                                   for job in cpu_intensive_jobs}
                    
                    for future in as_completed(future_to_job):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            job = future_to_job[future]
                            error_result = ProcessingResult(
                                job_id=job.job_id,
                                mode=job.mode,
                                success=False,
                                errors=[f"Worker process failed: {e}"]
                            )
                            results.append(error_result)
                        
                        completed += 1
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(completed, total_jobs, results[-1])
            
            # Process I/O-intensive jobs with threading
            if io_intensive_jobs:
                with ThreadPoolExecutor(max_workers=self.max_workers * 2) as executor:
                    future_to_job = {executor.submit(self._process_single_job, job): job 
                                   for job in io_intensive_jobs}
                    
                    for future in as_completed(future_to_job):
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(completed, total_jobs, result)
        
        return results
    
    def _process_single_job(self, job: ProcessingJob) -> ProcessingResult:
        """Process a single job."""
        start_time = time.time()
        result = ProcessingResult(job_id=job.job_id, mode=job.mode)
        
        try:
            self.logger.info(f"Processing job {job.job_id}: {job.mode.value}")
            
            # Create output directory
            job.output_path.mkdir(parents=True, exist_ok=True)
            
            # Process based on mode
            if job.mode == ProcessingMode.SEGMENTATION:
                output_files = self._process_segmentation(job)
            elif job.mode == ProcessingMode.ALIGNMENT:
                output_files = self._process_alignment(job)
            elif job.mode == ProcessingMode.COREGISTRATION:
                output_files = self._process_coregistration(job)
            elif job.mode == ProcessingMode.VALIDATION:
                output_files = self._process_validation(job)
            elif job.mode == ProcessingMode.FULL_PIPELINE:
                output_files = self._process_full_pipeline(job)
            else:
                raise ValueError(f"Unknown processing mode: {job.mode}")
            
            result.output_files = output_files
            result.success = True
            
            # Run validation if enabled and not a validation job
            if self.validation_enabled and job.mode != ProcessingMode.VALIDATION:
                validation_result = self._run_validation(job.output_path, job.mode)
                if validation_result:
                    result.metrics.update(validation_result.metrics)
                    result.warnings.extend(validation_result.warnings)
            
        except Exception as e:
            error_msg = f"Job {job.job_id} failed: {e}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)
            result.errors.append(traceback.format_exc())
        
        result.processing_time = time.time() - start_time
        return result
    
    def _process_segmentation(self, job: ProcessingJob) -> List[Path]:
        """Process segmentation job."""
        output_files = []
        
        # Find raw images
        raw_images = self._find_images(job.input_path, "raw")
        if not raw_images:
            raise ValueError(f"No raw images found in {job.input_path}")
        
        # Create segmented output directory
        segmented_dir = job.output_path / "1_segmented"
        segmented_dir.mkdir(exist_ok=True)
        
        for raw_image_path in raw_images:
            # Load and process image
            raw_image = load_image(raw_image_path)
            
            # Apply tissue separation
            segmented_image = self.tissue_separator.separate_tissue(raw_image)
            
            # Save segmented image
            output_path = segmented_dir / f"seg_{raw_image_path.name}"
            save_image(segmented_image, output_path)
            output_files.append(output_path)
        
        return output_files
    
    def _process_alignment(self, job: ProcessingJob) -> List[Path]:
        """Process alignment job."""
        output_files = []
        
        # Find segmented images
        segmented_images = self._find_images(job.input_path, "1_segmented")
        if not segmented_images:
            raise ValueError(f"No segmented images found in {job.input_path}")
        
        # Create aligned output directory
        aligned_dir = job.output_path / "2_aligned"
        aligned_dir.mkdir(exist_ok=True)
        
        for seg_image_path in segmented_images:
            # Load image
            seg_image = load_image(seg_image_path)
            
            # Apply alignment
            aligned_image = self.aligner.align_image(seg_image)
            
            # Save aligned image
            output_path = aligned_dir / f"aligned_{seg_image_path.name}"
            save_image(aligned_image, output_path)
            output_files.append(output_path)
        
        return output_files
    
    def _process_coregistration(self, job: ProcessingJob) -> List[Path]:
        """Process coregistration job."""
        output_files = []
        
        # Find images for coregistration
        iqid_images = self._find_images(job.input_path, "iqid")
        he_images = self._find_images(job.input_path, "he")
        
        if not iqid_images or not he_images:
            # Try with standard pipeline directories
            iqid_images = self._find_images(job.input_path, "1_segmented")
            he_images = self._find_images(job.input_path, "2_aligned")
        
        if not iqid_images or not he_images:
            raise ValueError(f"Could not find image pairs for coregistration in {job.input_path}")
        
        # Create coregistered output directory
        coreg_dir = job.output_path / "3_coregistered"
        coreg_dir.mkdir(exist_ok=True)
        
        # Process image pairs
        for i, (iqid_path, he_path) in enumerate(zip(iqid_images, he_images)):
            iqid_image = load_image(iqid_path)
            he_image = load_image(he_path)
            
            # Apply coregistration
            reg_result = self.multimodal_aligner.register_images(
                iqid_image, he_image,
                method=job.parameters.get('method', RegistrationMethod.MUTUAL_INFORMATION)
            )
            
            # Save coregistered image
            output_path = coreg_dir / f"coreg_{i:03d}_{iqid_path.stem}_{he_path.stem}.tif"
            save_image(reg_result.registered_image, output_path)
            output_files.append(output_path)
        
        return output_files
    
    def _process_validation(self, job: ProcessingJob) -> List[Path]:
        """Process validation job."""
        if not self.validation_suite:
            raise ValueError("Validation suite not initialized")
        
        # Run validation
        validation_mode = job.parameters.get('validation_mode', ValidationMode.COMPREHENSIVE)
        validation_result = self.validation_suite.validate(
            job.input_path, 
            mode=validation_mode,
            save_results=True
        )
        
        # Generate report
        report_path = job.output_path / "validation_report.md"
        report_text = self.validation_suite.generate_report(report_path)
        
        return [report_path]
    
    def _process_full_pipeline(self, job: ProcessingJob) -> List[Path]:
        """Process full pipeline job."""
        output_files = []
        
        # Run segmentation
        seg_job = ProcessingJob(
            job_id=f"{job.job_id}_seg",
            input_path=job.input_path,
            output_path=job.output_path,
            mode=ProcessingMode.SEGMENTATION
        )
        seg_files = self._process_segmentation(seg_job)
        output_files.extend(seg_files)
        
        # Run alignment
        align_job = ProcessingJob(
            job_id=f"{job.job_id}_align",
            input_path=job.output_path,
            output_path=job.output_path,
            mode=ProcessingMode.ALIGNMENT
        )
        align_files = self._process_alignment(align_job)
        output_files.extend(align_files)
        
        # Run coregistration if parameters specify it
        if job.parameters.get('include_coregistration', False):
            coreg_job = ProcessingJob(
                job_id=f"{job.job_id}_coreg",
                input_path=job.output_path,
                output_path=job.output_path,
                mode=ProcessingMode.COREGISTRATION,
                parameters=job.parameters
            )
            coreg_files = self._process_coregistration(coreg_job)
            output_files.extend(coreg_files)
        
        return output_files
    
    def _find_images(self, base_path: Path, subdir: str) -> List[Path]:
        """Find images in a subdirectory."""
        search_path = base_path / subdir
        if not search_path.exists():
            search_path = base_path  # Try base path if subdir doesn't exist
        
        extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
        images = []
        
        for ext in extensions:
            images.extend(search_path.glob(f"*{ext}"))
            images.extend(search_path.glob(f"*{ext.upper()}"))
        
        return natural_sort([str(p) for p in images])
    
    def _run_validation(self, output_path: Path, mode: ProcessingMode) -> Optional[ValidationResult]:
        """Run validation on processed output."""
        if not self.validation_suite:
            return None
        
        try:
            if mode == ProcessingMode.SEGMENTATION:
                validation_mode = ValidationMode.SEGMENTATION
            elif mode == ProcessingMode.ALIGNMENT:
                validation_mode = ValidationMode.ALIGNMENT
            elif mode == ProcessingMode.COREGISTRATION:
                validation_mode = ValidationMode.COREGISTRATION
            else:
                validation_mode = ValidationMode.COMPREHENSIVE
            
            return self.validation_suite.validate(
                output_path,
                mode=validation_mode,
                save_results=False
            )
        
        except Exception as e:
            self.logger.warning(f"Validation failed: {e}")
            return None
    
    def _calculate_summary_metrics(self, results: List[ProcessingResult]) -> Dict[str, float]:
        """Calculate summary metrics from processing results."""
        if not results:
            return {}
        
        # Basic statistics
        total_time = sum(r.processing_time for r in results)
        successful_results = [r for r in results if r.success]
        
        summary = {
            'total_processing_time': total_time,
            'average_processing_time': total_time / len(results),
            'success_rate': len(successful_results) / len(results) * 100.0
        }
        
        # Aggregate metrics from successful results
        all_metrics = {}
        for result in successful_results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate statistics for each metric
        for metric_name, values in all_metrics.items():
            if values:
                stats = calculate_statistics(values)
                summary[f'{metric_name}_mean'] = stats['mean']
                summary[f'{metric_name}_std'] = stats['std']
                summary[f'{metric_name}_min'] = stats['min']
                summary[f'{metric_name}_max'] = stats['max']
        
        return summary
    
    def _save_batch_result(self, result: BatchResult):
        """Save batch result to disk."""
        try:
            # Create results directory
            results_dir = self.temp_dir / "batch_results"
            results_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"batch_result_{timestamp}.json"
            
            # Convert result to dictionary (excluding complex objects)
            result_dict = {
                'total_jobs': result.total_jobs,
                'completed_jobs': result.completed_jobs,
                'failed_jobs': result.failed_jobs,
                'success_rate': result.success_rate,
                'processing_time': result.processing_time,
                'summary_metrics': result.summary_metrics,
                'job_results': [
                    {
                        'job_id': r.job_id,
                        'mode': r.mode.value,
                        'success': r.success,
                        'processing_time': r.processing_time,
                        'output_files': [str(f) for f in r.output_files],
                        'metrics': r.metrics,
                        'errors': r.errors,
                        'warnings': r.warnings
                    }
                    for r in result.results
                ],
                'timestamp': timestamp
            }
            
            # Save to file
            result_path = results_dir / filename
            with open(result_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            self.logger.info(f"Batch result saved to {result_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save batch result: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            'jobs_in_queue': len(self.job_queue),
            'max_workers': self.max_workers,
            'parallelization_strategy': self.parallelization.value,
            'validation_enabled': self.validation_enabled,
            'temp_dir': str(self.temp_dir),
            'processing_history': len(self.processing_history)
        }
    
    def clear_queue(self):
        """Clear the job queue."""
        self.job_queue.clear()
        self.logger.info("Job queue cleared")
    
    def generate_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Generate a comprehensive batch processing report."""
        if not self.processing_history:
            return "No batch processing results available."
        
        report = []
        report.append("# iQID Batch Processing Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
            report.append(f"- Processing time: {batch_result.processing_time:.2f}s")
            
            if batch_result.summary_metrics:
                report.append("- Key metrics:")
                for metric, value in batch_result.summary_metrics.items():
                    report.append(f"  - {metric}: {value:.4f}")
            
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


def _process_job_worker(job: ProcessingJob) -> ProcessingResult:
    """
    Worker function for multiprocessing.
    
    This function needs to be at module level for pickle serialization.
    """
    # Create a minimal processor for this worker
    processor = BatchProcessor(
        max_workers=1,
        parallelization=ParallelizationStrategy.SEQUENTIAL,
        validation_enabled=False  # Disable validation in worker to avoid conflicts
    )
    
    return processor._process_single_job(job)
