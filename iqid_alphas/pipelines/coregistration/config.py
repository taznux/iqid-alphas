#!/usr/bin/env python3
"""
Coregistration Pipeline Configuration

Configuration management for the coregistration pipeline.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..base import PipelineConfig
from ...configs import load_pipeline_config


@dataclass
class CoregistrationPipelineConfig(PipelineConfig):
    """Coregistration-specific pipeline configuration."""
    
    # Registration parameters
    registration_method: str = "mutual_information"
    pairing_strategy: str = "automatic"
    
    # Quality assessment parameters
    mutual_info_threshold: float = 0.5
    ssim_threshold: float = 0.6
    feature_match_threshold: int = 10
    
    # Processing parameters
    max_image_size: int = 2048
    downsample_factor: float = 0.5
    
    # Pairing parameters
    pairing_confidence_threshold: float = 0.5
    fuzzy_matching_threshold: float = 0.7
    
    @classmethod
    def from_config_file(cls, config_path: Optional[Path] = None) -> 'CoregistrationPipelineConfig':
        """Load configuration from file."""
        config_dict = load_pipeline_config("coregistration", config_path)
        
        # Extract configuration sections
        coreg_config = config_dict.get("coregistration", {})
        pipeline_config = config_dict.get("pipeline", {})
        quality_config = config_dict.get("quality_assessment", {})
        
        return cls(
            # Base pipeline config
            max_workers=pipeline_config.get("max_workers", 4),
            memory_limit_gb=pipeline_config.get("memory_limit_gb"),
            log_level=pipeline_config.get("log_level", "INFO"),
            generate_visualizations=pipeline_config.get("generate_visualizations", True),
            save_metrics=pipeline_config.get("save_metrics", True),
            
            # Coregistration-specific config
            registration_method=coreg_config.get("method", "mutual_information"),
            pairing_strategy=coreg_config.get("pairing_strategy", "automatic"),
            
            # Quality parameters
            mutual_info_threshold=quality_config.get("mutual_info_threshold", 0.5),
            ssim_threshold=quality_config.get("ssim_threshold", 0.6),
            feature_match_threshold=quality_config.get("feature_match_threshold", 10),
            
            # Processing parameters
            max_image_size=coreg_config.get("max_image_size", 2048),
            downsample_factor=coreg_config.get("downsample_factor", 0.5),
            
            # Pairing parameters
            pairing_confidence_threshold=coreg_config.get("pairing_confidence_threshold", 0.5),
            fuzzy_matching_threshold=coreg_config.get("fuzzy_matching_threshold", 0.7)
        )
