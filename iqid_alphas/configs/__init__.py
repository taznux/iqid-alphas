"""
Configuration management system for IQID-Alphas pipelines.

This module provides configuration loading, validation, and management
functionality for all pipeline types.

Keeps under 500 lines by focusing on configuration management only.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field

from ..utils.io import ensure_directory_exists


@dataclass
class ConfigManager:
    """Configuration manager for pipeline configurations."""
    
    config_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    
    def __post_init__(self):
        """Initialize configuration manager."""
        self.logger = logging.getLogger(__name__)
        self.config_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_config(self, config_name: str, custom_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load configuration by name or from custom path.
        
        Args:
            config_name: Name of the configuration (e.g., 'segmentation', 'alignment')
            custom_path: Optional custom configuration file path
            
        Returns:
            Configuration dictionary
        """
        if custom_path:
            config_path = Path(custom_path)
            cache_key = str(config_path)
        else:
            config_path = self.config_dir / f"{config_name}_config.json"
            cache_key = config_name
        
        # Check cache first
        if cache_key in self.config_cache:
            self.logger.debug(f"Loading {config_name} config from cache")
            return self.config_cache[cache_key]
        
        # Load from file
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.logger.info(f"Loading {config_name} config from: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate basic structure
            self._validate_config_structure(config, config_name)
            
            # Cache the configuration
            self.config_cache[cache_key] = config
            
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}")
    
    def merge_configs(self, base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to merge/override
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self.merge_configs(merged[key], value)
            else:
                # Override value
                merged[key] = value
        
        return merged
    
    def save_config(self, config: Dict[str, Any], config_path: Path) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            config_path: Path where to save the configuration
        """
        ensure_directory_exists(config_path.parent)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, sort_keys=True)
            
            self.logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {config_path}: {e}")
    
    def validate_config(self, config: Dict[str, Any], config_type: str) -> bool:
        """
        Validate configuration structure and values.
        
        Args:
            config: Configuration to validate
            config_type: Type of configuration (segmentation, alignment, etc.)
            
        Returns:
            True if valid, raises exception if invalid
        """
        self.logger.info(f"Validating {config_type} configuration")
        
        # Basic structure validation
        self._validate_config_structure(config, config_type)
        
        # Type-specific validation
        if config_type == "segmentation":
            self._validate_segmentation_config(config)
        elif config_type == "alignment":
            self._validate_alignment_config(config)
        elif config_type == "coregistration":
            self._validate_coregistration_config(config)
        
        self.logger.info(f"Configuration validation passed for {config_type}")
        return True
    
    def _validate_config_structure(self, config: Dict[str, Any], config_type: str) -> None:
        """Validate basic configuration structure."""
        required_sections = ["pipeline", "output", "validation", "processing"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in {config_type} config")
        
        # Validate pipeline section
        pipeline_config = config["pipeline"]
        required_pipeline_keys = ["verbose", "log_level", "max_workers"]
        
        for key in required_pipeline_keys:
            if key not in pipeline_config:
                raise ValueError(f"Missing required key '{key}' in pipeline section")
    
    def _validate_segmentation_config(self, config: Dict[str, Any]) -> None:
        """Validate segmentation-specific configuration."""
        if "segmentation" not in config:
            raise ValueError("Missing 'segmentation' section in segmentation config")
        
        seg_config = config["segmentation"]
        required_keys = ["method", "min_blob_area", "max_blob_area"]
        
        for key in required_keys:
            if key not in seg_config:
                raise ValueError(f"Missing required key '{key}' in segmentation section")
        
        # Validate value ranges
        if seg_config["min_blob_area"] >= seg_config["max_blob_area"]:
            raise ValueError("min_blob_area must be less than max_blob_area")
        
        # Validate method
        valid_methods = ["adaptive_clustering", "thresholding", "watershed"]
        if seg_config["method"] not in valid_methods:
            raise ValueError(f"Invalid segmentation method. Must be one of: {valid_methods}")
    
    def _validate_alignment_config(self, config: Dict[str, Any]) -> None:
        """Validate alignment-specific configuration."""
        if "alignment" not in config:
            raise ValueError("Missing 'alignment' section in alignment config")
        
        align_config = config["alignment"]
        required_keys = ["method", "registration_algorithm", "reference_method"]
        
        for key in required_keys:
            if key not in align_config:
                raise ValueError(f"Missing required key '{key}' in alignment section")
        
        # Validate method choices
        valid_methods = ["bidirectional", "sequential", "reference_based"]
        if align_config["method"] not in valid_methods:
            raise ValueError(f"Invalid alignment method. Must be one of: {valid_methods}")
        
        valid_algorithms = ["stackreg", "opencv", "elastix"]
        if align_config["registration_algorithm"] not in valid_algorithms:
            raise ValueError(f"Invalid registration algorithm. Must be one of: {valid_algorithms}")
    
    def _validate_coregistration_config(self, config: Dict[str, Any]) -> None:
        """Validate coregistration-specific configuration."""
        if "coregistration" not in config:
            raise ValueError("Missing 'coregistration' section in coregistration config")
        
        coreg_config = config["coregistration"]
        required_keys = ["registration_method", "pairing_strategy"]
        
        for key in required_keys:
            if key not in coreg_config:
                raise ValueError(f"Missing required key '{key}' in coregistration section")
        
        # Validate method choices
        valid_methods = ["mutual_information", "feature_based", "intensity_based"]
        if coreg_config["registration_method"] not in valid_methods:
            raise ValueError(f"Invalid registration method. Must be one of: {valid_methods}")
        
        valid_strategies = ["automatic", "manual", "fuzzy_matching"]
        if coreg_config["pairing_strategy"] not in valid_strategies:
            raise ValueError(f"Invalid pairing strategy. Must be one of: {valid_strategies}")
    
    def get_default_config(self, config_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a pipeline type.
        
        Args:
            config_type: Type of configuration
            
        Returns:
            Default configuration dictionary
        """
        return self.load_config(config_type)
    
    def create_custom_config(self, base_config_type: str, 
                           overrides: Dict[str, Any],
                           output_path: Path) -> Dict[str, Any]:
        """
        Create a custom configuration based on a base configuration.
        
        Args:
            base_config_type: Base configuration type
            overrides: Configuration overrides
            output_path: Where to save the custom configuration
            
        Returns:
            Custom configuration dictionary
        """
        base_config = self.load_config(base_config_type)
        custom_config = self.merge_configs(base_config, overrides)
        
        # Validate the custom configuration
        self.validate_config(custom_config, base_config_type)
        
        # Save the custom configuration
        self.save_config(custom_config, output_path)
        
        return custom_config
    
    def list_available_configs(self) -> List[str]:
        """List all available configuration files."""
        config_files = list(self.config_dir.glob("*_config.json"))
        return [f.stem.replace("_config", "") for f in config_files]
    
    def get_config_schema(self, config_type: str) -> Dict[str, Any]:
        """
        Get configuration schema for documentation purposes.
        
        Args:
            config_type: Configuration type
            
        Returns:
            Configuration schema dictionary
        """
        # This would ideally be loaded from schema files
        # For now, return the default config as an example
        return self.load_config(config_type)


# Global configuration manager instance
config_manager = ConfigManager()


def load_pipeline_config(config_type: str, 
                        custom_path: Optional[Path] = None,
                        overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to load and optionally override pipeline configuration.
    
    Args:
        config_type: Type of configuration (segmentation, alignment, coregistration)
        custom_path: Optional path to custom configuration file
        overrides: Optional configuration overrides
        
    Returns:
        Complete configuration dictionary
    """
    config = config_manager.load_config(config_type, custom_path)
    
    if overrides:
        config = config_manager.merge_configs(config, overrides)
        config_manager.validate_config(config, config_type)
    
    return config
