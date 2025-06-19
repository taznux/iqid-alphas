#!/usr/bin/env python3
"""Alignment Pipeline Configuration"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .base import PipelineConfig
from ..configs import load_pipeline_config


@dataclass
class AlignmentPipelineConfig(PipelineConfig):
    """Alignment-specific pipeline configuration."""
    
    alignment_method: str = "bidirectional"
    reference_strategy: str = "largest_slice"
    alignment_algorithm: str = "feature_based"
    min_slice_count: int = 3
    max_slice_count: int = 100
    min_ssim_threshold: float = 0.5
    generate_intermediate_visualizations: bool = True
    save_alignment_matrices: bool = True
    
    def __post_init__(self):
        """Handle any additional unknown fields gracefully."""
        pass
    
    @classmethod
    def from_config_file(cls, config_path: Optional[Path] = None) -> 'AlignmentPipelineConfig':
        """Load configuration from file."""
        config_dict = load_pipeline_config("alignment", config_path)
        alignment_config = config_dict.get("alignment", {})
        pipeline_config = config_dict.get("pipeline", {})
        
        # Map config keys to class attributes and filter unknown ones
        if "method" in alignment_config:
            alignment_config["alignment_method"] = alignment_config.pop("method")
        if "registration_algorithm" in alignment_config:
            alignment_config["alignment_algorithm"] = alignment_config.pop("registration_algorithm")
        if "reference_method" in alignment_config:
            alignment_config["reference_strategy"] = alignment_config.pop("reference_method")
        
        # Filter out unknown configuration keys
        known_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_alignment = {k: v for k, v in alignment_config.items() if k in known_keys}
        filtered_pipeline = {k: v for k, v in pipeline_config.items() if k in known_keys}
        
        merged_config = {**filtered_pipeline, **filtered_alignment}
        return cls(**merged_config)
