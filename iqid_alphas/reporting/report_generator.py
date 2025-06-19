"""
Main Report Generator

Orchestrates the creation of comprehensive pipeline reports including:
- Per-sample detailed Markdown reports with visualizations
- Batch summary reports with aggregated metrics
- Performance analysis and trend reports
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .mermaid_generator import MermaidGenerator, PipelineStep, StepStatus
from ..utils.io import ensure_directory_exists


class SampleReport:
    """Container for individual sample report data."""
    
    def __init__(self, sample_id: str, pipeline_type: str):
        self.sample_id = sample_id
        self.pipeline_type = pipeline_type
        self.timestamp = datetime.now()
        self.steps: List[PipelineStep] = []
        self.metrics: Dict[str, Any] = {}
        self.visualizations: Dict[str, str] = {}
        self.overall_status: StepStatus = StepStatus.PENDING
        self.total_duration: float = 0.0
        self.error_message: Optional[str] = None


class BatchReport:
    """Container for batch processing report data."""
    
    def __init__(self, pipeline_type: str, batch_id: str):
        self.pipeline_type = pipeline_type
        self.batch_id = batch_id
        self.timestamp = datetime.now()
        self.sample_reports: List[SampleReport] = []
        self.summary_metrics: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}


class ReportGenerator:
    """Main report generator for pipeline results."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        self.mermaid_generator = MermaidGenerator()
        
        # Create output directories
        ensure_directory_exists(self.output_dir)
        ensure_directory_exists(self.output_dir / "samples")
        ensure_directory_exists(self.output_dir / "batches")
        ensure_directory_exists(self.output_dir / "visualizations")
    
    def generate_sample_report(self, sample_result: Dict[str, Any], 
                             visualizations: Optional[Dict[str, str]] = None) -> Path:
        """
        Generate a detailed Markdown report for a single sample.
        
        Args:
            sample_result: Sample processing result with metrics and status
            visualizations: Optional dictionary of visualization file paths
            
        Returns:
            Path to generated report file
        """
        sample_id = sample_result.get('sample_id', 'unknown')
        pipeline_type = sample_result.get('validation_type', 'unknown')
        
        # Create sample report object
        report = SampleReport(sample_id, pipeline_type)
        report.metrics = sample_result.get('metrics', {})
        report.total_duration = sample_result.get('processing_time', 0.0)
        report.overall_status = StepStatus.SUCCESS if sample_result.get('success', False) else StepStatus.FAILURE
        report.error_message = sample_result.get('error_message')
        
        if visualizations:
            report.visualizations = visualizations
        
        # Generate pipeline steps based on pipeline type
        report.steps = self._generate_pipeline_steps(sample_result, pipeline_type)
        
        # Generate Markdown report
        report_content = self._generate_markdown_report(report)
        
        # Save report
        report_filename = f"{sample_id}_{pipeline_type}_report.md"
        report_path = self.output_dir / "samples" / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Generated sample report: {report_path}")
        return report_path
    
    def generate_batch_report(self, batch_results: List[Dict[str, Any]], 
                            pipeline_type: str, batch_id: str) -> Dict[str, Path]:
        """
        Generate batch summary reports.
        
        Args:
            batch_results: List of sample results
            pipeline_type: Type of pipeline (segmentation, alignment, etc.)
            batch_id: Unique identifier for this batch
            
        Returns:
            Dictionary of report paths (markdown, json, csv)
        """
        # Create batch report object
        batch_report = BatchReport(pipeline_type, batch_id)
        
        # Process each sample
        for sample_result in batch_results:
            sample_id = sample_result.get('sample_id', 'unknown')
            sample_report = SampleReport(sample_id, pipeline_type)
            sample_report.metrics = sample_result.get('metrics', {})
            sample_report.total_duration = sample_result.get('processing_time', 0.0)
            sample_report.overall_status = StepStatus.SUCCESS if sample_result.get('success', False) else StepStatus.FAILURE
            sample_report.error_message = sample_result.get('error_message')
            
            batch_report.sample_reports.append(sample_report)
        
        # Calculate summary metrics
        batch_report.summary_metrics = self._calculate_batch_summary(batch_results)
        batch_report.performance_metrics = self._calculate_performance_metrics(batch_results)
        
        # Generate reports
        report_paths = {}
        
        # Markdown summary
        md_content = self._generate_batch_markdown(batch_report)
        md_path = self.output_dir / "batches" / f"{batch_id}_{pipeline_type}_summary.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        report_paths['markdown'] = md_path
        
        # JSON detailed report
        json_data = self._generate_batch_json(batch_report)
        json_path = self.output_dir / "batches" / f"{batch_id}_{pipeline_type}_detailed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        report_paths['json'] = json_path
        
        # CSV metrics export
        csv_path = self._generate_batch_csv(batch_report)
        report_paths['csv'] = csv_path
        
        self.logger.info(f"Generated batch reports for {len(batch_results)} samples")
        return report_paths
    
    def _generate_pipeline_steps(self, sample_result: Dict[str, Any], 
                               pipeline_type: str) -> List[PipelineStep]:
        """Generate pipeline steps based on result data."""
        steps = []
        
        if pipeline_type == 'segmentation':
            steps = [
                PipelineStep("load", "Load Raw Image", StepStatus.SUCCESS),
                PipelineStep("segment", "Segment Tissues", 
                           StepStatus.SUCCESS if sample_result.get('success') else StepStatus.FAILURE),
                PipelineStep("validate", "Validate Segmentation", 
                           StepStatus.SUCCESS if sample_result.get('success') else StepStatus.FAILURE)
            ]
        elif pipeline_type == 'alignment':
            steps = [
                PipelineStep("load", "Load Segmented Images", StepStatus.SUCCESS),
                PipelineStep("align", "Align Image Stack", 
                           StepStatus.SUCCESS if sample_result.get('success') else StepStatus.FAILURE),
                PipelineStep("validate", "Validate Alignment", 
                           StepStatus.SUCCESS if sample_result.get('success') else StepStatus.FAILURE)
            ]
        elif pipeline_type == 'coregistration':
            steps = [
                PipelineStep("load", "Load Multi-modal Images", StepStatus.SUCCESS),
                PipelineStep("register", "Register iQID-H&E", 
                           StepStatus.SUCCESS if sample_result.get('success') else StepStatus.FAILURE),
                PipelineStep("assess", "Assess Registration Quality", 
                           StepStatus.SUCCESS if sample_result.get('success') else StepStatus.FAILURE)
            ]
        
        return steps
    
    def _generate_markdown_report(self, report: SampleReport) -> str:
        """Generate Markdown content for sample report."""
        content = []
        
        # Header
        content.append(f"# Pipeline Report: {report.sample_id}")
        content.append(f"")
        content.append(f"**Pipeline Type**: {report.pipeline_type.title()}")
        content.append(f"**Status**: {'✅ Success' if report.overall_status == StepStatus.SUCCESS else '❌ Failed'}")
        content.append(f"**Timestamp**: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Duration**: {report.total_duration:.2f} seconds")
        content.append(f"")
        
        # Pipeline Flow Diagram
        if report.steps:
            content.append("## Pipeline Flow")
            content.append("")
            flowchart = self.mermaid_generator.generate_pipeline_flowchart(
                report.steps, f"{report.pipeline_type.title()} Pipeline"
            )
            content.append(flowchart)
            content.append("")
        
        # Step Details
        content.append("## Step Details")
        content.append("")
        
        for i, step in enumerate(report.steps, 1):
            status_icon = "✅" if step.status == StepStatus.SUCCESS else "❌"
            content.append(f"### {i}. {step.name}")
            content.append(f"* **Status**: {status_icon} {step.status.value.title()}")
            if step.duration:
                content.append(f"* **Duration**: {step.duration:.2f}s")
            if step.details:
                content.append(f"* **Details**: {step.details}")
            
            # Add visualization if available
            step_viz_key = step.id.lower()
            if step_viz_key in report.visualizations:
                viz_path = report.visualizations[step_viz_key]
                content.append(f"* **Visualization**:")
                content.append(f"  ![{step.name}]({viz_path})")
            
            content.append("")
        
        # Metrics
        if report.metrics:
            content.append("## Metrics")
            content.append("")
            content.append("| Metric | Value |")
            content.append("|--------|-------|")
            for key, value in report.metrics.items():
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                content.append(f"| {key.replace('_', ' ').title()} | {value_str} |")
            content.append("")
        
        # Error Information
        if report.error_message:
            content.append("## Error Details")
            content.append("")
            content.append(f"```")
            content.append(report.error_message)
            content.append(f"```")
            content.append("")
        
        return "\n".join(content)
    
    def _generate_batch_markdown(self, batch_report: BatchReport) -> str:
        """Generate Markdown content for batch summary."""
        content = []
        
        # Header
        content.append(f"# Batch Report: {batch_report.batch_id}")
        content.append(f"")
        content.append(f"**Pipeline Type**: {batch_report.pipeline_type.title()}")
        content.append(f"**Timestamp**: {batch_report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Total Samples**: {len(batch_report.sample_reports)}")
        content.append(f"")
        
        # Summary Statistics
        success_count = sum(1 for r in batch_report.sample_reports if r.overall_status == StepStatus.SUCCESS)
        failure_count = len(batch_report.sample_reports) - success_count
        success_rate = (success_count / len(batch_report.sample_reports)) * 100 if batch_report.sample_reports else 0
        
        content.append("## Summary")
        content.append("")
        content.append(f"* **Success Rate**: {success_rate:.1f}% ({success_count}/{len(batch_report.sample_reports)})")
        content.append(f"* **Successful Samples**: {success_count}")
        content.append(f"* **Failed Samples**: {failure_count}")
        content.append(f"")
        
        # Performance Metrics
        if batch_report.performance_metrics:
            content.append("## Performance Metrics")
            content.append("")
            content.append("| Metric | Value |")
            content.append("|--------|-------|")
            for key, value in batch_report.performance_metrics.items():
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                content.append(f"| {key.replace('_', ' ').title()} | {value_str} |")
            content.append("")
        
        # Sample Results Table
        content.append("## Sample Results")
        content.append("")
        content.append("| Sample ID | Status | Duration (s) | Error |")
        content.append("|-----------|--------|-------------|-------|")
        
        for sample in batch_report.sample_reports:
            status_icon = "✅" if sample.overall_status == StepStatus.SUCCESS else "❌"
            error_msg = sample.error_message[:50] + "..." if sample.error_message and len(sample.error_message) > 50 else (sample.error_message or "")
            content.append(f"| {sample.sample_id} | {status_icon} | {sample.total_duration:.2f} | {error_msg} |")
        
        content.append("")
        
        return "\n".join(content)
    
    def _generate_batch_json(self, batch_report: BatchReport) -> Dict[str, Any]:
        """Generate JSON data for batch report."""
        return {
            'batch_id': batch_report.batch_id,
            'pipeline_type': batch_report.pipeline_type,
            'timestamp': batch_report.timestamp.isoformat(),
            'total_samples': len(batch_report.sample_reports),
            'summary_metrics': batch_report.summary_metrics,
            'performance_metrics': batch_report.performance_metrics,
            'sample_results': [
                {
                    'sample_id': sample.sample_id,
                    'status': sample.overall_status.value,
                    'duration': sample.total_duration,
                    'metrics': sample.metrics,
                    'error_message': sample.error_message
                }
                for sample in batch_report.sample_reports
            ]
        }
    
    def _generate_batch_csv(self, batch_report: BatchReport) -> Path:
        """Generate CSV file for batch metrics."""
        import csv
        
        csv_path = self.output_dir / "batches" / f"{batch_report.batch_id}_{batch_report.pipeline_type}_metrics.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['sample_id', 'status', 'duration', 'success']
            
            # Add all unique metric keys
            all_metrics = set()
            for sample in batch_report.sample_reports:
                all_metrics.update(sample.metrics.keys())
            fieldnames.extend(sorted(all_metrics))
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for sample in batch_report.sample_reports:
                row = {
                    'sample_id': sample.sample_id,
                    'status': sample.overall_status.value,
                    'duration': sample.total_duration,
                    'success': sample.overall_status == StepStatus.SUCCESS
                }
                row.update(sample.metrics)
                writer.writerow(row)
        
        return csv_path
    
    def _calculate_batch_summary(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for batch."""
        if not batch_results:
            return {}
        
        success_count = sum(1 for r in batch_results if r.get('success', False))
        total_count = len(batch_results)
        
        # Calculate metric averages
        metric_sums = {}
        metric_counts = {}
        
        for result in batch_results:
            metrics = result.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_sums[key] = metric_sums.get(key, 0) + value
                    metric_counts[key] = metric_counts.get(key, 0) + 1
        
        metric_averages = {
            key: metric_sums[key] / metric_counts[key] 
            for key in metric_sums if metric_counts[key] > 0
        }
        
        return {
            'total_samples': total_count,
            'successful_samples': success_count,
            'failed_samples': total_count - success_count,
            'success_rate': (success_count / total_count) * 100 if total_count > 0 else 0,
            'average_metrics': metric_averages
        }
    
    def _calculate_performance_metrics(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance statistics for batch."""
        if not batch_results:
            return {}
        
        durations = [r.get('processing_time', 0) for r in batch_results]
        
        return {
            'total_processing_time': sum(durations),
            'average_processing_time': sum(durations) / len(durations),
            'min_processing_time': min(durations),
            'max_processing_time': max(durations),
            'samples_per_second': len(durations) / sum(durations) if sum(durations) > 0 else 0
        }
