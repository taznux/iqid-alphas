#!/usr/bin/env python3
"""
Segmentation Utilities Module

Data loading, metrics calculation, and other utility functions for segmentation pipeline.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from ...core.validation import ValidationResult


class SegmentationDataLoader:
    """Handles data loading operations for segmentation pipeline."""
    
    def __init__(self, config):
        self.config = config
    
    def discover_samples(self, data_path: Path) -> List[Dict[str, Any]]:
        """
        Discover raw image samples for segmentation processing.
        
        Args:
            data_path: Path to dataset directory
            
        Returns:
            List of sample dictionaries with metadata
        """
        samples = []
        
        # Look for raw image files (typically in raw/ or root directory)
        raw_patterns = ["raw/*.tiff", "raw/*.tif", "*.tiff", "*.tif"]
        
        raw_files = []
        for pattern in raw_patterns:
            found_files = list(data_path.glob(pattern))
            if found_files:
                raw_files.extend(found_files)
        
        if not raw_files:
            raise FileNotFoundError(f"No raw image files found in {data_path}")
        
        for raw_file in raw_files:
            # Extract sample ID (e.g., D1M1, D2M2, etc.)
            sample_id = raw_file.stem
            
            # Look for corresponding ground truth directory
            gt_dir = data_path / "1_segmented" / sample_id
            
            sample_info = {
                "id": sample_id,
                "raw_image_path": raw_file,
                "ground_truth_dir": gt_dir if gt_dir.exists() else None,
                "has_ground_truth": gt_dir.exists()
            }
            
            samples.append(sample_info)
        
        return samples


class SegmentationMetricsCalculator:
    """Calculates comprehensive metrics for segmentation results."""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_aggregate_metrics(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from individual results.
        
        Args:
            results: List of individual validation results
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                "total_samples": len(results),
                "successful_samples": 0,
                "success_rate": 0.0
            }
        
        # Extract metrics from successful results
        iou_scores = [r.metrics.get("avg_iou", 0.0) for r in successful_results 
                     if "avg_iou" in r.metrics and r.metrics["avg_iou"] > 0]
        dice_scores = [r.metrics.get("avg_dice", 0.0) for r in successful_results 
                      if "avg_dice" in r.metrics and r.metrics["avg_dice"] > 0]
        processing_times = [r.processing_time for r in successful_results]
        
        aggregate_metrics = {
            "total_samples": len(results),
            "successful_samples": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0.0,
        }
        
        # Add IoU metrics if available
        if iou_scores:
            aggregate_metrics.update({
                "avg_iou": sum(iou_scores) / len(iou_scores),
                "min_iou": min(iou_scores),
                "max_iou": max(iou_scores),
                "iou_std": self._calculate_std(iou_scores)
            })
        
        # Add Dice metrics if available
        if dice_scores:
            aggregate_metrics.update({
                "avg_dice": sum(dice_scores) / len(dice_scores),
                "min_dice": min(dice_scores),
                "max_dice": max(dice_scores),
                "dice_std": self._calculate_std(dice_scores)
            })
        
        # Add tissue count metrics
        tissue_counts = [r.metrics.get("total_tissues_found", 0) for r in successful_results 
                        if "total_tissues_found" in r.metrics]
        if tissue_counts:
            aggregate_metrics.update({
                "avg_tissues_per_sample": sum(tissue_counts) / len(tissue_counts),
                "min_tissues_per_sample": min(tissue_counts),
                "max_tissues_per_sample": max(tissue_counts)
            })
        
        return aggregate_metrics
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


class SegmentationReporter:
    """Handles report generation for segmentation results."""
    
    def __init__(self, config):
        self.config = config
    
    def generate_detailed_report(self, results: List[ValidationResult], output_path: Path) -> None:
        """
        Generate detailed segmentation pipeline report.
        
        Args:
            results: List of validation results
            output_path: Path to save the report
        """
        try:
            with open(output_path, 'w') as f:
                f.write("# Segmentation Pipeline Detailed Report\n\n")
                f.write(f"**Pipeline**: Raw → Segmented Validation\n")
                f.write(f"**Dataset**: {self.config.data_path}\n")
                f.write(f"**Total Samples**: {len(results)}\n")
                
                successful_results = [r for r in results if r.success]
                f.write(f"**Successful**: {len(successful_results)}\n")
                f.write(f"**Failed**: {len(results) - len(successful_results)}\n\n")
                
                # Summary statistics
                if successful_results:
                    self._write_summary_statistics(f, successful_results)
                
                # Individual sample results
                f.write("## Individual Sample Results\n\n")
                for result in results:
                    self._write_sample_result(f, result)
                    
        except Exception as e:
            print(f"Failed to generate detailed report: {str(e)}")
    
    def _write_summary_statistics(self, f, successful_results: List[ValidationResult]) -> None:
        """Write summary statistics section."""
        iou_scores = [r.metrics.get("avg_iou", 0.0) for r in successful_results 
                     if "avg_iou" in r.metrics and r.metrics["avg_iou"] > 0]
        
        if iou_scores:
            f.write("## Quality Metrics Summary\n\n")
            f.write(f"- **Average IoU**: {sum(iou_scores)/len(iou_scores):.3f}\n")
            f.write(f"- **Min IoU**: {min(iou_scores):.3f}\n")
            f.write(f"- **Max IoU**: {max(iou_scores):.3f}\n\n")
    
    def _write_sample_result(self, f, result: ValidationResult) -> None:
        """Write individual sample result."""
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        f.write(f"### {result.sample_id} - {status}\n\n")
        
        if result.success and result.metrics:
            f.write("**Metrics:**\n")
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    f.write(f"- {metric}: {value:.4f}\n")
                else:
                    f.write(f"- {metric}: {value}\n")
        
        if result.error_message:
            f.write(f"**Error:** {result.error_message}\n")
        
        f.write(f"**Processing Time:** {result.processing_time:.2f}s\n\n")
