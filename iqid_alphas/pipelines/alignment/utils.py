#!/usr/bin/env python3
"""
Alignment Utilities Module

Data loading, metrics calculation, and other utility functions for alignment pipeline.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional

from ...utils.io import load_image


class AlignmentDataLoader:
    """Handles data loading operations for alignment pipeline."""
    
    def __init__(self, config):
        self.config = config
    
    def load_slice_stack(self, directory: Path) -> List[Path]:
        """Load slice stack file paths from directory."""
        # Look for both .png and .tif files
        slice_files = list(directory.glob("*.png")) + list(directory.glob("*.tif"))
        slice_files = sorted(slice_files)
        return slice_files
    
    def load_image_stack(self, file_paths: List[Path]) -> np.ndarray:
        """Load images from file paths into a numpy array stack."""
        images = []
        target_shape = None
        
        # First pass: determine target shape (use the first valid image)
        for file_path in file_paths:
            try:
                image = load_image(file_path)
                # Ensure image is 2D (grayscale)
                if len(image.shape) == 3:
                    image = np.mean(image, axis=2)
                target_shape = image.shape
                break
            except Exception as e:
                print(f"Warning: Failed to load image {file_path}: {e}")
                continue
        
        if target_shape is None:
            raise ValueError("No valid images found to determine target shape")
        
        # Second pass: load and resize all images to target shape
        for file_path in file_paths:
            try:
                image = load_image(file_path)
                # Ensure image is 2D (grayscale)
                if len(image.shape) == 3:
                    image = np.mean(image, axis=2)
                
                # Resize to target shape if needed
                if image.shape != target_shape:
                    image = cv2.resize(
                        image, 
                        (target_shape[1], target_shape[0]), 
                        interpolation=cv2.INTER_LINEAR
                    )
                
                images.append(image)
            except Exception as e:
                print(f"Warning: Failed to load image {file_path}: {e}")
                continue
        
        if not images:
            raise ValueError("No images could be loaded from the file paths")
        
        # Stack images into 3D array (n_images, height, width)
        return np.stack(images, axis=0)


class AlignmentMetricsCalculator:
    """Calculates comprehensive metrics for alignment results."""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_metrics(self, alignment_result, validation_result, 
                         sample_id: str) -> Dict[str, Any]:
        """Calculate comprehensive alignment metrics."""
        metrics = {
            'slice_count': self._get_slice_count(alignment_result),
            'alignment_method': self.config.alignment_method,
            'reference_strategy': self.config.reference_strategy,
            'processing_stage': 'alignment_validation',
            'sample_id': sample_id
        }
        
        # Add validation metrics if available
        if hasattr(validation_result, 'metrics') and validation_result.metrics:
            metrics.update(validation_result.metrics)
        
        # Add alignment quality metrics if available
        if hasattr(alignment_result, 'quality_metrics') and alignment_result.quality_metrics:
            metrics.update(alignment_result.quality_metrics)
        
        # Add transform-based metrics
        metrics.update(self._calculate_transform_metrics(alignment_result))
        
        # Add confidence metrics
        metrics.update(self._calculate_confidence_metrics(alignment_result))
        
        return metrics
    
    def _get_slice_count(self, alignment_result) -> int:
        """Get the number of processed slices."""
        if hasattr(alignment_result, 'aligned_slices'):
            return len(alignment_result.aligned_slices)
        elif hasattr(alignment_result, 'aligned_images'):
            return len(alignment_result.aligned_images)
        elif hasattr(alignment_result, 'transforms'):
            return len(alignment_result.transforms)
        return 0
    
    def _calculate_transform_metrics(self, alignment_result) -> Dict[str, Any]:
        """Calculate metrics based on transformation data."""
        metrics = {}
        
        if hasattr(alignment_result, 'transforms') and alignment_result.transforms:
            transforms = alignment_result.transforms
            metrics['num_transforms'] = len(transforms)
            
            # Could add more sophisticated transform analysis:
            # - Average translation magnitude
            # - Average rotation angle
            # - Transform consistency
            
        return metrics
    
    def _calculate_confidence_metrics(self, alignment_result) -> Dict[str, Any]:
        """Calculate confidence and quality metrics."""
        metrics = {}
        
        if hasattr(alignment_result, 'confidence'):
            metrics['alignment_confidence'] = alignment_result.confidence
        
        if hasattr(alignment_result, 'quality_score'):
            metrics['quality_score'] = alignment_result.quality_score
            
        # Add threshold-based assessments
        if 'alignment_confidence' in metrics:
            confidence = metrics['alignment_confidence']
            if isinstance(confidence, (int, float)):
                metrics['high_confidence'] = confidence > 0.8
                metrics['acceptable_confidence'] = confidence > 0.5
        
        return metrics


class AlignmentVisualization:
    """Handles visualization generation for alignment results."""
    
    def __init__(self, config):
        self.config = config
    
    def should_generate_visualizations(self) -> bool:
        """Check if visualizations should be generated."""
        return getattr(self.config, 'generate_intermediate_visualizations', False)
    
    def generate_alignment_visualization(self, alignment_result, output_path: Path):
        """Generate visualization of alignment results."""
        if not self.should_generate_visualizations():
            return
        
        # Implementation would go here for generating:
        # - Before/after alignment comparisons
        # - Transform visualization
        # - Quality metric plots
        # etc.
        pass
