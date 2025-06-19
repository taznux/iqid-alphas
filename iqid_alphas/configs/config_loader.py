"""
Configuration loader and validator for IQID-Alphas pipelines.
"""
import json
from typing import Dict

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    # TODO: Add validation logic
    return config
