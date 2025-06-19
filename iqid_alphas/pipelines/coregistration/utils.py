#!/usr/bin/env python3
"""
Coregistration Utilities Module

Data structures, loading, and utility functions for coregistration pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class CoregistrationResult:
    """Container for co-registration pipeline results"""
    pair_id: str
    iqid_dir: str
    he_dir: str
    success: bool
    registration_method: str
    pairing_confidence: float
    processing_time: float
    iqid_slice_count: int
    he_slice_count: int
    mutual_information: float
    structural_similarity: float
    registration_confidence: float
    feature_matches: int
    transform_parameters: Dict
    error_message: Optional[str] = None
    visualization_paths: Optional[List[str]] = None


class CoregistrationDataLoader:
    """Handles data loading operations for coregistration pipeline."""
    
    def __init__(self, config):
        self.config = config
    
    def discover_samples(self, data_path: Path) -> List[Dict[str, Any]]:
        """Discover and pair iQID and H&E samples for coregistration."""
        samples = []
        
        # Look for iQID data (assume it's in a subdirectory)
        iqid_base = data_path / "iqid" if (data_path / "iqid").exists() else data_path
        he_base = data_path / "he" if (data_path / "he").exists() else data_path
        
        # Find iQID directories
        iqid_dirs = [d for d in iqid_base.iterdir() if d.is_dir()]
        he_dirs = [d for d in he_base.iterdir() if d.is_dir()]
        
        # Pair samples based on strategy
        if self.config.pairing_strategy == "automatic":
            samples = self._pair_samples_automatic(iqid_dirs, he_dirs)
        elif self.config.pairing_strategy == "fuzzy_matching":
            samples = self._pair_samples_fuzzy(iqid_dirs, he_dirs)
        else:
            samples = self._pair_samples_manual(iqid_dirs, he_dirs)
                
        return samples
    
    def _pair_samples_automatic(self, iqid_dirs: List[Path], he_dirs: List[Path]) -> List[Dict[str, Any]]:
        """Simple automatic pairing strategy - match by name similarity."""
        samples = []
        
        for iqid_dir in iqid_dirs:
            # Find corresponding H&E directory
            he_dir = None
            for he_candidate in he_dirs:
                # Simple name matching - can be enhanced
                if self._names_match(iqid_dir.name, he_candidate.name):
                    he_dir = he_candidate
                    break
            
            if he_dir:
                samples.append({
                    'id': f"{iqid_dir.name}_{he_dir.name}",
                    'iqid_dir': iqid_dir,
                    'he_dir': he_dir,
                    'pairing_confidence': 0.8  # Mock confidence for automatic
                })
        
        return samples
    
    def _pair_samples_fuzzy(self, iqid_dirs: List[Path], he_dirs: List[Path]) -> List[Dict[str, Any]]:
        """Fuzzy matching pairing strategy."""
        samples = []
        
        for iqid_dir in iqid_dirs:
            best_match = None
            best_confidence = 0.0
            
            for he_candidate in he_dirs:
                confidence = self._calculate_name_similarity(iqid_dir.name, he_candidate.name)
                if confidence > best_confidence and confidence >= self.config.fuzzy_matching_threshold:
                    best_match = he_candidate
                    best_confidence = confidence
            
            if best_match:
                samples.append({
                    'id': f"{iqid_dir.name}_{best_match.name}",
                    'iqid_dir': iqid_dir,
                    'he_dir': best_match,
                    'pairing_confidence': best_confidence
                })
        
        return samples
    
    def _pair_samples_manual(self, iqid_dirs: List[Path], he_dirs: List[Path]) -> List[Dict[str, Any]]:
        """Manual pairing strategy - requires manual configuration."""
        # This would require a configuration file with explicit pairings
        # For now, fall back to automatic
        return self._pair_samples_automatic(iqid_dirs, he_dirs)
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Simple name matching logic for sample pairing."""
        # Extract base names (remove common suffixes/prefixes)
        base1 = name1.lower().replace("iqid", "").replace("he", "").replace("_", "").replace("-", "")
        base2 = name2.lower().replace("iqid", "").replace("he", "").replace("_", "").replace("-", "")
        
        # Check if they have significant overlap
        if len(base1) > 3 and len(base2) > 3:
            return base1 in base2 or base2 in base1
        return base1 == base2
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names for fuzzy matching."""
        # Simple Levenshtein-like similarity
        from difflib import SequenceMatcher
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def load_slice_stack(self, directory: Path) -> List[Path]:
        """Load slice stack file paths from directory."""
        # Look for common image formats
        extensions = ["*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"]
        slice_files = []
        
        for ext in extensions:
            slice_files.extend(directory.glob(ext))
        
        return sorted(slice_files)
    
    def load_image_safe(self, image_path: Path) -> Optional[np.ndarray]:
        """Safely load an image with error handling."""
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            # Resize if too large
            if max(image.shape) > self.config.max_image_size:
                scale = self.config.max_image_size / max(image.shape)
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            return image
        except Exception:
            return None


class CoregistrationMetricsCalculator:
    """Calculates comprehensive metrics for coregistration results."""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_summary_statistics(self, results: List[CoregistrationResult]) -> Dict[str, Any]:
        """Calculate summary statistics from coregistration results."""
        if not results:
            return {"error": "No results to analyze"}
        
        successful_results = [r for r in results if r.success]
        
        summary = {
            "total_pairs": len(results),
            "successful_pairs": len(successful_results),
            "failed_pairs": len(results) - len(successful_results),
            "registration_success_rate": len(successful_results) / len(results) if results else 0.0,
            "timestamp": str(Path.cwd()),  # Placeholder
        }
        
        if successful_results:
            # Registration quality metrics
            mi_scores = [r.mutual_information for r in successful_results]
            ssim_scores = [r.structural_similarity for r in successful_results]
            processing_times = [r.processing_time for r in successful_results]
            
            summary.update({
                "avg_mutual_information": sum(mi_scores) / len(mi_scores),
                "min_mutual_information": min(mi_scores),
                "max_mutual_information": max(mi_scores),
                
                "avg_structural_similarity": sum(ssim_scores) / len(ssim_scores),
                "min_structural_similarity": min(ssim_scores),
                "max_structural_similarity": max(ssim_scores),
                
                "avg_processing_time": sum(processing_times) / len(processing_times),
                "total_processing_time": sum(processing_times),
                
                "method_distribution": self._calculate_method_distribution(successful_results),
                "pairing_confidence_stats": self._calculate_pairing_stats(successful_results)
            })
        
        return summary
    
    def _calculate_method_distribution(self, results: List[CoregistrationResult]) -> Dict[str, int]:
        """Calculate distribution of registration methods used."""
        methods = {}
        for result in results:
            method = result.registration_method
            methods[method] = methods.get(method, 0) + 1
        return methods
    
    def _calculate_pairing_stats(self, results: List[CoregistrationResult]) -> Dict[str, float]:
        """Calculate pairing confidence statistics."""
        confidences = [r.pairing_confidence for r in results]
        
        return {
            "avg_pairing_confidence": sum(confidences) / len(confidences),
            "min_pairing_confidence": min(confidences),
            "max_pairing_confidence": max(confidences)
        }
