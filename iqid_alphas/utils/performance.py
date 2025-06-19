#!/usr/bin/env python3
"""
Performance Optimization Module

Provides performance enhancements for IQID-Alphas pipelines:
- Parallel processing with configurable worker count
- Memory-efficient processing with streaming
- Resource monitoring and optimization
- Progress tracking and performance metrics
"""

import os
import time
import logging
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    start_time: float
    end_time: float
    processing_time: float
    peak_memory_mb: float
    avg_memory_mb: float
    cpu_usage_percent: float
    samples_processed: int
    samples_per_second: float
    errors_count: int
    success_rate: float


class ResourceMonitor:
    """Monitor system resources during processing."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.memory_samples = []
        self.cpu_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.debug("Resource monitoring started")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        metrics = {}
        if self.memory_samples:
            metrics['peak_memory_mb'] = max(self.memory_samples)
            metrics['avg_memory_mb'] = sum(self.memory_samples) / len(self.memory_samples)
        else:
            metrics['peak_memory_mb'] = 0.0
            metrics['avg_memory_mb'] = 0.0
        
        if self.cpu_samples:
            metrics['avg_cpu_percent'] = sum(self.cpu_samples) / len(self.cpu_samples)
        else:
            metrics['avg_cpu_percent'] = 0.0
        
        self.logger.debug(f"Resource monitoring stopped: {metrics}")
        return metrics
    
    def _monitor_loop(self):
        """Resource monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Memory usage in MB
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)
                
                # CPU usage percentage
                cpu_percent = process.cpu_percent(interval=None)
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.warning(f"Resource monitoring error: {e}")
                break


class ParallelProcessor:
    """Parallel processing manager for pipeline operations."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of workers (default: CPU count)
            use_processes: Use processes instead of threads for CPU-bound tasks
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.logger = logging.getLogger(__name__)
        
        # Adjust worker count based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 4:
            self.max_workers = min(self.max_workers, 2)
            self.logger.warning(f"Limited memory detected ({available_memory_gb:.1f}GB), reducing workers to {self.max_workers}")
    
    @contextmanager
    def get_executor(self):
        """Get appropriate executor (thread or process pool)."""
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            self.logger.debug(f"Created {executor_class.__name__} with {self.max_workers} workers")
            yield executor
    
    def process_samples_parallel(self, 
                                samples: List[Any], 
                                process_func: Callable,
                                progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        Process samples in parallel.
        
        Args:
            samples: List of samples to process
            process_func: Function to process each sample
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processing results
        """
        results = []
        errors = []
        
        with self.get_executor() as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(process_func, sample): sample 
                for sample in samples
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(completed_count, len(samples), result)
                        
                except Exception as e:
                    error_info = {
                        'sample': sample,
                        'error': str(e),
                        'type': type(e).__name__
                    }
                    errors.append(error_info)
                    self.logger.error(f"Error processing sample {sample}: {e}")
        
        self.logger.info(f"Parallel processing completed: {len(results)} successful, {len(errors)} errors")
        
        # Attach error information to results
        if hasattr(results, 'append'):
            for error in errors:
                results.append({'error': error})
        
        return results


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_processes: bool = False,
                 enable_monitoring: bool = True,
                 memory_limit_gb: Optional[float] = None):
        """
        Initialize performance optimizer.
        
        Args:
            max_workers: Maximum parallel workers
            use_processes: Use processes instead of threads
            enable_monitoring: Enable resource monitoring
            memory_limit_gb: Memory usage limit in GB
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.enable_monitoring = enable_monitoring
        self.memory_limit_gb = memory_limit_gb
        
        self.processor = ParallelProcessor(max_workers, use_processes)
        self.monitor = ResourceMonitor() if enable_monitoring else None
        self.logger = logging.getLogger(__name__)
    
    def optimize_pipeline_execution(self, 
                                   samples: List[Any],
                                   process_func: Callable,
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute pipeline with performance optimizations.
        
        Args:
            samples: List of samples to process
            process_func: Function to process each sample
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with results and performance metrics
        """
        start_time = time.time()
        
        # Start monitoring
        if self.monitor:
            self.monitor.start_monitoring()
        
        try:
            # Check memory constraints
            if self.memory_limit_gb:
                available_gb = psutil.virtual_memory().available / (1024**3)
                if available_gb < self.memory_limit_gb:
                    self.logger.warning(f"Available memory ({available_gb:.1f}GB) below limit ({self.memory_limit_gb}GB)")
            
            # Process samples
            results = self.processor.process_samples_parallel(
                samples, process_func, progress_callback
            )
            
            # Calculate metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Count successful and failed results
            success_count = sum(1 for r in results if not isinstance(r, dict) or 'error' not in r)
            error_count = len(results) - success_count
            
            # Create performance metrics
            resource_metrics = self.monitor.stop_monitoring() if self.monitor else {}
            
            metrics = PerformanceMetrics(
                start_time=start_time,
                end_time=end_time,
                processing_time=processing_time,
                peak_memory_mb=resource_metrics.get('peak_memory_mb', 0.0),
                avg_memory_mb=resource_metrics.get('avg_memory_mb', 0.0),
                cpu_usage_percent=resource_metrics.get('avg_cpu_percent', 0.0),
                samples_processed=len(samples),
                samples_per_second=len(samples) / processing_time if processing_time > 0 else 0.0,
                errors_count=error_count,
                success_rate=success_count / len(samples) if samples else 0.0
            )
            
            self.logger.info(f"Pipeline execution completed: {success_count}/{len(samples)} successful "
                           f"({metrics.success_rate:.1%}) in {processing_time:.1f}s "
                           f"({metrics.samples_per_second:.1f} samples/sec)")
            
            return {
                'results': results,
                'metrics': metrics,
                'success_count': success_count,
                'error_count': error_count,
                'success_rate': metrics.success_rate
            }
            
        except Exception as e:
            if self.monitor:
                self.monitor.stop_monitoring()
            
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def get_optimal_batch_size(self, sample_size_estimate: int = 1000000) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            sample_size_estimate: Estimated memory usage per sample in bytes
            
        Returns:
            Optimal batch size
        """
        available_memory = psutil.virtual_memory().available
        
        # Reserve 25% of memory for system operations
        usable_memory = int(available_memory * 0.75)
        
        # Calculate batch size
        batch_size = max(1, usable_memory // (sample_size_estimate * self.max_workers))
        
        # Reasonable limits
        batch_size = min(batch_size, 1000)  # Max 1000 samples per batch
        batch_size = max(batch_size, 1)     # Min 1 sample per batch
        
        self.logger.debug(f"Calculated optimal batch size: {batch_size} "
                         f"(available memory: {available_memory/(1024**3):.1f}GB)")
        
        return batch_size


# Convenience functions for easy integration
def create_performance_optimizer(parallel: bool = True, 
                               max_workers: Optional[int] = None,
                               processes: bool = False) -> Optional[PerformanceOptimizer]:
    """Create performance optimizer with sensible defaults."""
    if not parallel:
        return None
    
    return PerformanceOptimizer(
        max_workers=max_workers,
        use_processes=processes,
        enable_monitoring=True
    )


def optimize_pipeline_performance(samples: List[Any],
                                process_func: Callable,
                                parallel: bool = True,
                                max_workers: Optional[int] = None,
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Convenience function to optimize pipeline performance.
    
    Args:
        samples: Samples to process
        process_func: Processing function
        parallel: Enable parallel processing
        max_workers: Maximum workers
        progress_callback: Progress callback
        
    Returns:
        Results with performance metrics
    """
    if not parallel or len(samples) <= 1:
        # Sequential processing fallback
        start_time = time.time()
        results = []
        
        for i, sample in enumerate(samples):
            try:
                result = process_func(sample)
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(samples), result)
            except Exception as e:
                results.append({'error': {'sample': sample, 'error': str(e)}})
        
        processing_time = time.time() - start_time
        success_count = sum(1 for r in results if not isinstance(r, dict) or 'error' not in r)
        
        return {
            'results': results,
            'success_count': success_count,
            'error_count': len(results) - success_count,
            'success_rate': success_count / len(samples) if samples else 0.0,
            'processing_time': processing_time,
            'parallel': False
        }
    
    # Parallel processing
    optimizer = PerformanceOptimizer(max_workers=max_workers)
    return optimizer.optimize_pipeline_execution(samples, process_func, progress_callback)
