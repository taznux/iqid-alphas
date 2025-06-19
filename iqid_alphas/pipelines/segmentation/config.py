#!/usr/bin/env python3
"""
Segmentation Pipeline Configuration

Configuration management for the segmentation pipeline.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..base import PipelineConfig
from ...configs import load_pipeline_config


@dataclass
class SegmentationPipelineConfig(PipelineConfig):
    """Segmentation-specific pipeline configuration."""
    
    # Segmentation parameters
    segmentation_method: str = "adaptive_clustering"
    min_blob_area: int = 50
    max_blob_area: int = 50000
    preserve_blobs: bool = True
    use_clustering: bool = True
    
    # Quality control
    min_tissue_area: int = 100
    max_tissue_area: int = 1000000
    aspect_ratio_threshold: float = 10.0
    
    # Pipeline-specific paths
    data_path: str = ""
    output_dir: str = ""
    
    @classmethod
    def from_config_file(cls, config_path: Optional[Path] = None) -> 'SegmentationPipelineConfig':
        """Load configuration from file."""
        config_dict = load_pipeline_config("segmentation", config_path)
        
        # Extract configuration sections
        seg_config = config_dict.get("segmentation", {})
        pipeline_config = config_dict.get("pipeline", {})
        processing_config = config_dict.get("processing", {})
        validation_config = config_dict.get("validation", {})
        quality_config = config_dict.get("quality_control", {})
        
        return cls(
            # Base pipeline config
            max_workers=pipeline_config.get("max_workers", 4),
            memory_limit_gb=pipeline_config.get("memory_limit_gb"),
            log_level=pipeline_config.get("log_level", "INFO"),
            generate_visualizations=validation_config.get("generate_visualizations", True),
            save_metrics=validation_config.get("save_metrics", True),
            
            # Segmentation-specific config
            segmentation_method=seg_config.get("method", "adaptive_clustering"),
            min_blob_area=seg_config.get("min_blob_area", 50),
            max_blob_area=seg_config.get("max_blob_area", 50000),
            preserve_blobs=seg_config.get("preserve_blobs", True),
            use_clustering=seg_config.get("use_clustering", True),
            
            # Quality control config
            min_tissue_area=quality_config.get("min_tissue_area", 100),
            max_tissue_area=quality_config.get("max_tissue_area", 1000000),
            aspect_ratio_threshold=quality_config.get("aspect_ratio_threshold", 10.0)
        )
