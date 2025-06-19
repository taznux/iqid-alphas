"""
Base pipeline classes for IQID-Alphas pipeline system.

This module provides the foundational pipeline framework that orchestrates core modules
with common functionality for logging, configuration, and progress tracking.
"""
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict

from ..utils.io import setup_logging
from ..core.validation import ValidationResult, ValidationMode


@dataclass
class PipelineConfig:
    """Base configuration class for all pipelines."""
    
    # Common configuration options
    verbose: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    memory_limit_gb: Optional[int] = None
    
    # Output configuration
    save_intermediate: bool = True
    output_format: str = "png"
    compression: str = "lzw"
    
    # Validation configuration
    generate_visualizations: bool = True
    save_metrics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


@dataclass
class PipelineResult:
    """Results from pipeline execution."""
    
    pipeline_name: str
    start_time: float
    end_time: float
    total_samples: int
    successful_samples: int
    failed_samples: int
    validation_results: List[ValidationResult]
    performance_metrics: Dict[str, Any]
    output_directory: Path
    
    @property
    def duration(self) -> float:
        """Total execution time in seconds."""
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.successful_samples / self.total_samples) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result_dict = asdict(self)
        result_dict['output_directory'] = str(self.output_directory)
        try:
            result_dict['validation_results'] = [
                result.to_dict() for result in self.validation_results
            ]
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to serialize validation results: {e}")
        return result_dict


class ProgressTracker:
    """Progress tracking system with console and file output."""
    
    def __init__(self, total_items: int, description: str = "Processing", 
                 enable_eta: bool = True):
        self.total_items = total_items
        self.description = description
        self.enable_eta = enable_eta
        self.current_item = 0
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
    def update(self, increment: int = 1, item_description: str = "") -> None:
        """Update progress tracker."""
        self.current_item += increment
        
        # Calculate progress percentage
        progress_pct = (self.current_item / self.total_items) * 100
        
        # Calculate ETA if enabled
        eta_str = ""
        if self.enable_eta and self.current_item > 0:
            elapsed_time = time.time() - self.start_time
            items_per_second = self.current_item / elapsed_time
            remaining_items = self.total_items - self.current_item
            eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
            eta_str = f" | ETA: {self._format_time(eta_seconds)}"
        
        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * self.current_item // self.total_items)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Log progress
        progress_msg = (
            f"{self.description}: [{bar}] "
            f"{self.current_item}/{self.total_items} ({progress_pct:.1f}%){eta_str}"
        )
        
        if item_description:
            progress_msg += f" | {item_description}"
        
        self.logger.info(progress_msg)
    
    def finish(self) -> None:
        """Mark progress as complete."""
        total_time = time.time() - self.start_time
        self.logger.info(
            f"{self.description} completed: {self.total_items} items in "
            f"{self._format_time(total_time)}"
        )
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class BasePipeline(ABC):
    """
    Abstract base class for all IQID-Alphas pipelines.
    
    Provides common functionality for logging, configuration, progress tracking,
    and pipeline orchestration.
    """
    
    def __init__(self, data_path: Union[str, Path], output_dir: Union[str, Path],
                 config: Optional[PipelineConfig] = None):
        """
        Initialize base pipeline.
        
        Args:
            data_path: Path to input data directory
            output_dir: Path to output directory
            config: Pipeline configuration (uses defaults if None)
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.config = config or PipelineConfig()
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config.log_level
        )
        
        # Initialize reporting
        from ..reporting import ReportGenerator
        self.report_generator = ReportGenerator(self.output_dir / "reports")
        
        # Initialize pipeline state
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.progress_tracker: Optional[ProgressTracker] = None
        
        # Validate inputs
        self._validate_inputs()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
        self.logger.info(f"Data path: {self.data_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _validate_inputs(self) -> None:
        """Validate pipeline inputs."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        if not self.data_path.is_dir():
            raise NotADirectoryError(f"Data path is not a directory: {self.data_path}")
    
    @abstractmethod
    def discover_samples(self) -> List[Dict[str, Any]]:
        """
        Discover samples to process from the data directory.
        
        Must be implemented by subclasses to define how samples are discovered
        and what metadata is associated with each sample.
        
        Returns:
            List of sample dictionaries with metadata
        """
        pass
    
    @abstractmethod
    def process_sample(self, sample: Dict[str, Any]) -> ValidationResult:
        """
        Process a single sample.
        
        Must be implemented by subclasses to define the core processing logic
        for individual samples.
        
        Args:
            sample: Sample dictionary with metadata
            
        Returns:
            ValidationResult with processing outcomes
        """
        pass
    
    def run_pipeline(self, max_samples: Optional[int] = None) -> PipelineResult:
        """
        Execute the complete pipeline.
        
        Args:
            max_samples: Maximum number of samples to process (None = all)
            
        Returns:
            PipelineResult with execution summary
        """
        self.logger.info(f"Starting {self.__class__.__name__} execution")
        self.start_time = time.time()
        
        try:
            # Discover samples
            self.logger.info("Discovering samples...")
            samples = self.discover_samples()
            
            if max_samples is not None:
                samples = samples[:max_samples]
                self.logger.info(f"Limited to {max_samples} samples")
            
            self.logger.info(f"Found {len(samples)} samples to process")
            
            # Initialize progress tracking
            self.progress_tracker = ProgressTracker(
                total_items=len(samples),
                description=f"{self.__class__.__name__} Processing"
            )
            
            # Process samples
            validation_results = []
            successful_samples = 0
            failed_samples = 0
            
            for i, sample in enumerate(samples):
                try:
                    self.logger.debug(f"Processing sample {i+1}/{len(samples)}: {sample}")
                    
                    # Process individual sample
                    result = self.process_sample(sample)
                    validation_results.append(result)
                    
                    if result.success:
                        successful_samples += 1
                    else:
                        failed_samples += 1
                    
                    # Update progress
                    self.progress_tracker.update(
                        item_description=f"Sample {sample.get('id', i+1)}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to process sample {i+1}: {e}")
                    failed_samples += 1
                    
                    # Create failure result
                    failure_result = ValidationResult(
                        mode=ValidationMode.SEGMENTATION,
                        metrics={},
                        errors=[str(e)],
                        processing_time=0.0,
                        passed_tests=0,
                        total_tests=1,
                        metadata={'sample_id': sample.get('id', f'sample_{i+1}')}
                    )
                    validation_results.append(failure_result)
                    
                    self.progress_tracker.update(
                        item_description=f"FAILED: {str(e)[:50]}..."
                    )
            
            # Finish progress tracking
            self.progress_tracker.finish()
            
            # Calculate performance metrics
            self.end_time = time.time()
            performance_metrics = self._calculate_performance_metrics(validation_results)
            
            # Create pipeline result
            result = PipelineResult(
                pipeline_name=self.__class__.__name__,
                start_time=self.start_time,
                end_time=self.end_time,
                total_samples=len(samples),
                successful_samples=successful_samples,
                failed_samples=failed_samples,
                validation_results=validation_results,
                performance_metrics=performance_metrics,
                output_directory=self.output_dir
            )
            
            # Generate reports if configured
            if self.config.save_metrics:
                self._save_pipeline_result(result)
            
            self.logger.info(f"Pipeline completed successfully!")
            self.logger.info(f"Processed {len(samples)} samples in {result.duration:.2f}s")
            self.logger.info(f"Success rate: {result.success_rate:.1f}%")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def _calculate_performance_metrics(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Calculate aggregate performance metrics."""
        if not results:
            return {}
        
        # Extract metrics from successful results
        successful_results = [r for r in results if r.is_passing]
        
        if not successful_results:
            return {"note": "No successful results for metric calculation"}
        
        # Calculate aggregate metrics
        metrics = {}
        
        # Processing time statistics
        processing_times = [r.processing_time for r in successful_results]
        metrics['avg_processing_time'] = sum(processing_times) / len(processing_times)
        metrics['min_processing_time'] = min(processing_times)
        metrics['max_processing_time'] = max(processing_times)
        
        # Extract common quality metrics if available
        common_metric_keys = set()
        for result in successful_results:
            common_metric_keys.update(result.metrics.keys())
        
        for metric_key in common_metric_keys:
            values = [
                r.metrics[metric_key] for r in successful_results 
                if metric_key in r.metrics and isinstance(r.metrics[metric_key], (int, float))
            ]
            if values:
                metrics[f'avg_{metric_key}'] = sum(values) / len(values)
                metrics[f'min_{metric_key}'] = min(values)
                metrics[f'max_{metric_key}'] = max(values)
        
        return metrics
    
    def _save_pipeline_result(self, result: PipelineResult) -> None:
        """Save pipeline result to JSON file."""
        result_file = self.output_dir / f"{result.pipeline_name}_result.json"
        
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Pipeline result saved to: {result_file}")
    
    def generate_report(self, result: PipelineResult) -> None:
        """
        Generate comprehensive pipeline report.
        
        Can be overridden by subclasses for pipeline-specific reporting.
        
        Args:
            result: Pipeline execution result
        """
        report_file = self.output_dir / f"{result.pipeline_name}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# {result.pipeline_name} Execution Report\n\n")
            f.write(f"**Execution Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Duration**: {result.duration:.2f} seconds\n")
            f.write(f"**Total Samples**: {result.total_samples}\n")
            f.write(f"**Successful**: {result.successful_samples}\n")
            f.write(f"**Failed**: {result.failed_samples}\n")
            f.write(f"**Success Rate**: {result.success_rate:.1f}%\n\n")
            
            # Performance metrics
            if result.performance_metrics:
                f.write("## Performance Metrics\n\n")
                for key, value in result.performance_metrics.items():
                    if isinstance(value, float):
                        f.write(f"- **{key}**: {value:.4f}\n")
                    else:
                        f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            # Sample results summary
            if result.validation_results:
                f.write("## Sample Results Summary\n\n")
                for i, validation_result in enumerate(result.validation_results):
                    status = "✅ SUCCESS" if validation_result.is_passing else "❌ FAILED"
                    f.write(f"{i+1}. **{validation_result.metadata['sample_id']}**: {status}\n")
                    
                    if validation_result.errors:
                        f.write(f"   - Error: {validation_result.errors}\n")
                    
                    if validation_result.metrics:
                        f.write("   - Metrics:\n")
                        for metric, value in validation_result.metrics.items():
                            if isinstance(value, float):
                                f.write(f"     - {metric}: {value:.4f}\n")
                            else:
                                f.write(f"     - {metric}: {value}\n")
                    f.write("\n")
        
        self.logger.info(f"Pipeline report saved to: {report_file}")
