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
    
    @classmethod
    def from_config_file(cls, config_path: Optional[Path] = None) -> 'AlignmentPipelineConfig':
        """Load configuration from file."""
        config_dict = load_pipeline_config("alignment", config_path)
        alignment_config = config_dict.get("alignment", {})
        pipeline_config = config_dict.get("pipeline", {})
        merged_config = {**pipeline_config, **alignment_config}
        return cls(**merged_config)
