#!/usr/bin/env python3
"""
Segmentation Quality Validation Module

Handles quality assessment and validation of segmentation results.
"""

from typing import Dict, List, Any
from ...core.validation import ValidationResult


class SegmentationQualityValidator:
    """Validates segmentation quality and success metrics."""
    
    def __init__(self, config):
        self.config = config
    
    def validate_segmentation_quality(self, segmentation_results: Dict[str, List], 
                                    sample_id: str) -> Dict[str, Any]:
        """
        Validate the quality of segmentation results.
        
        Args:
            segmentation_results: Results from tissue separator
            sample_id: Sample identifier
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}
        
        # Check tissue count
        total_tissues = len(segmentation_results)
        quality_metrics['total_tissues'] = total_tissues
        quality_metrics['has_tissues'] = total_tissues > 0
        
        # Validate tissue areas if available
        tissue_areas = []
        for tissue_type, tissue_paths in segmentation_results.items():
            tissue_area = len(tissue_paths)  # Simplified area measure
            tissue_areas.append(tissue_area)
            
            # Check against configured thresholds
            area_valid = (self.config.min_tissue_area <= tissue_area <= self.config.max_tissue_area)
            quality_metrics[f'{tissue_type}_area'] = tissue_area
            quality_metrics[f'{tissue_type}_area_valid'] = area_valid
        
        # Calculate aggregate area metrics
        if tissue_areas:
            quality_metrics['avg_tissue_area'] = sum(tissue_areas) / len(tissue_areas)
            quality_metrics['min_tissue_area'] = min(tissue_areas)
            quality_metrics['max_tissue_area'] = max(tissue_areas)
            quality_metrics['all_areas_valid'] = all(
                self.config.min_tissue_area <= area <= self.config.max_tissue_area 
                for area in tissue_areas
            )
        
        return quality_metrics
    
    def assess_validation_success(self, metrics: Dict[str, Any]) -> bool:
        """
        Assess if validation was successful based on metrics.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            bool: True if validation is considered successful
        """
        # Check basic success criteria
        if not metrics.get('has_tissues', False):
            return False
        
        # Check IoU threshold if available
        avg_iou = metrics.get('avg_iou', 0.0)
        if avg_iou > 0 and avg_iou < 0.3:  # Configurable threshold
            return False
        
        # Check Dice coefficient if available
        avg_dice = metrics.get('avg_dice', 0.0)
        if avg_dice > 0 and avg_dice < 0.3:  # Configurable threshold
            return False
        
        # Check validation status
        validation_status = metrics.get('validation_status', '')
        if validation_status == 'failed':
            return False
        
        return True
