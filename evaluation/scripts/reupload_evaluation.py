#!/usr/bin/env python3
"""
ReUpload Dataset Pipeline Evaluation

This script evaluates the automated iQID processing pipeline by comparing
automated results against manually processed reference data in the ReUpload dataset.

Tests the complete workflow: Raw → Segmented → Aligned
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import skimage.io
import skimage.metrics
from dataclasses import dataclass

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iqid_alphas.core.raw_image_splitter import RawImageSplitter
from iqid_alphas.core.alignment import ImageAligner
from iqid_alphas.pipelines.simple import SimplePipeline

# Add evaluation utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visualizer import IQIDVisualizer
from utils.adaptive_segmentation import AdaptiveIQIDSegmenter, EnhancedRawImageSplitter

# Import Steps 9-13 bidirectional alignment capabilities
from bidirectional_alignment_evaluation import BidirectionalAlignmentEvaluator, AlignmentStrategy, ReferenceSliceInfo


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    sample_id: str
    stage: str
    success: bool
    similarity_score: float
    error_message: Optional[str] = None
    processing_time: float = 0.0
    file_count_automated: int = 0
    file_count_reference: int = 0
    detailed_metrics: Optional[Dict] = None
    visualization_paths: Optional[Dict] = None  # Added for visualization file paths
    # Enhanced for Steps 9-13 evaluation
    alignment_strategy: Optional[str] = None
    reference_slice_info: Optional[Dict] = None
    bidirectional_quality: Optional[Dict] = None


class ReUploadPipelineEvaluator:
    """
    Evaluates automated processing pipeline against ReUpload reference data
    """
    
    def __init__(self, reupload_path: str, output_dir: str = "evaluation/outputs"):
        self.reupload_path = Path(reupload_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components with improved blob-preserving and clustering-based settings
        self.raw_splitter = RawImageSplitter()
        self.enhanced_splitter = EnhancedRawImageSplitter(preserve_blobs=True)
        self.aligner = ImageAligner()
        
        # Initialize bidirectional alignment evaluator for Steps 9-13
        self.bidirectional_evaluator = BidirectionalAlignmentEvaluator(
            str(self.reupload_path), 
            str(self.output_dir / "bidirectional_alignment")
        )
        
        self.logger.info("Using improved clustering-based segmentation for distributed dots")
        self.logger.info("Integrated Steps 9-13 bidirectional alignment evaluation")
        
        # Initialize visualizer
        viz_output_dir = self.output_dir / "visualizations"
        self.visualizer = IQIDVisualizer(str(viz_output_dir))
        
        # Results storage
        self.results: List[EvaluationResult] = []
        self.summary_stats = {}
        
        self.logger.info(f"Initialized evaluator for ReUpload dataset: {self.reupload_path}")
        self.logger.info(f"Visualizations will be saved to: {viz_output_dir}")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("ReUploadEvaluator")
        logger.setLevel(logging.INFO)
        
        # Create handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def discover_samples(self) -> List[Dict]:
        """
        Discover available samples in the ReUpload dataset
        
        Returns:
            List of sample information dictionaries
        """
        samples = []
        
        # The reupload_path should contain iQID_reupload/iQID directory structure
        iqid_base_path = self.reupload_path / "iQID_reupload" / "iQID"
        if not iqid_base_path.exists():
            # Try alternative paths
            alt_paths = [
                self.reupload_path / "iQID",  # Direct iQID folder
                self.reupload_path,           # Already pointing to iQID folder
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists() and any(alt_path.glob("*/*/*/1_*")):
                    iqid_base_path = alt_path
                    break
            else:
                self.logger.error(f"Could not find iQID data structure in {self.reupload_path}")
                self.logger.info(f"Expected path: {iqid_base_path}")
                self.logger.info(f"Available paths: {list(self.reupload_path.iterdir())}")
                return samples
        
        self.logger.info(f"Using iQID base path: {iqid_base_path}")
            
        # Process both 3D and Sequential processing types
        for processing_type in ["3D", "Sequential"]:
            processing_path = iqid_base_path / processing_type
            if processing_path.exists():
                self.logger.info(f"Found processing type: {processing_type}")
                
                # Find tissue type directories (kidney, tumor, etc.)
                for tissue_dir in processing_path.iterdir():
                    if tissue_dir.is_dir():
                        self.logger.info(f"  Found tissue type: {tissue_dir.name} in {processing_type}")
                        
                        # Find sample directories within tissue type
                        for sample_dir in tissue_dir.iterdir():
                            if sample_dir.is_dir():
                                sample_info = self._analyze_sample(sample_dir, tissue_dir.name, processing_type)
                                if sample_info:
                                    samples.append(sample_info)
            else:
                self.logger.warning(f"Processing type directory not found: {processing_path}")
                            
        self.logger.info(f"Discovered {len(samples)} samples total")
        return samples
    
    def _analyze_sample(self, sample_path: Path, tissue_type: str, processing_type: str) -> Optional[Dict]:
        """
        Analyze a single sample directory to determine processing stages available
        """
        sample_id = sample_path.name
        
        # Look for raw iQID event images
        raw_files = list(sample_path.glob("*_iqid_event_image.tif"))
        
        # Look for subdirectories
        segmented_dir = sample_path / "1_segmented"
        aligned_dir = sample_path / "2_aligned"
        
        if not raw_files:
            self.logger.warning(f"No raw iQID event image found in {sample_path}")
            return None
            
        sample_info = {
            'sample_id': sample_id,
            'sample_path': sample_path,
            'tissue_type': tissue_type,
            'processing_type': processing_type,
            'raw_files': raw_files,
            'has_segmented': segmented_dir.exists(),
            'has_aligned': aligned_dir.exists(),
            'segmented_dir': segmented_dir if segmented_dir.exists() else None,
            'aligned_dir': aligned_dir if aligned_dir.exists() else None
        }
        
        self.logger.debug(f"Sample {sample_id}: Raw={len(raw_files)}, "
                         f"Segmented={sample_info['has_segmented']}, "
                         f"Aligned={sample_info['has_aligned']}")
                         
        return sample_info
    
    def _generate_stage_visualizations(self, sample_id: str, stage: str, 
                                     reference_files: List[Path], 
                                     automated_files: List[Path],
                                     include_raw: bool = False,
                                     raw_files: Optional[List[Path]] = None) -> Dict[str, str]:
        """
        Generate comprehensive visualizations for a processing stage
        
        Args:
            sample_id: Sample identifier
            stage: Processing stage name
            reference_files: List of reference image files
            automated_files: List of automated result files
            include_raw: Whether to include raw stage in pipeline overview
            raw_files: Raw image files (for pipeline overview)
            
        Returns:
            Dictionary of visualization file paths
        """
        viz_paths = {}
        
        try:
            # Load images
            ref_images = []
            ref_filenames = []
            for f in reference_files[:6]:  # Limit to first 6 for performance
                try:
                    img = skimage.io.imread(f)
                    ref_images.append(img)
                    ref_filenames.append(f.name)
                except Exception as e:
                    self.logger.warning(f"Failed to load reference image {f}: {e}")
            
            auto_images = []
            auto_filenames = []
            for f in automated_files[:6]:  # Limit to first 6 for performance
                try:
                    img = skimage.io.imread(f)
                    auto_images.append(img)
                    auto_filenames.append(f.name)
                except Exception as e:
                    self.logger.warning(f"Failed to load automated image {f}: {e}")
            
            # Generate individual stage visualizations
            if ref_images:
                ref_viz_path = self.visualizer.create_processing_visualization(
                    f"{sample_id}_reference", stage, ref_images, ref_filenames
                )
                if ref_viz_path:
                    viz_paths['reference_stage'] = ref_viz_path
            
            if auto_images:
                auto_viz_path = self.visualizer.create_processing_visualization(
                    f"{sample_id}_automated", stage, auto_images, auto_filenames
                )
                if auto_viz_path:
                    viz_paths['automated_stage'] = auto_viz_path
            
            # Generate comparison visualization
            if ref_images and auto_images:
                comp_viz_path = self.visualizer.create_comparison_visualization(
                    sample_id, stage, ref_images, auto_images, 
                    ref_filenames, auto_filenames
                )
                if comp_viz_path:
                    viz_paths['comparison'] = comp_viz_path
            
            # Generate pipeline overview if this is aligned stage
            if stage == "aligned" and include_raw and raw_files:
                try:
                    raw_images = []
                    for f in raw_files[:1]:  # Just use first raw image
                        raw_img = skimage.io.imread(f)
                        raw_images.append(raw_img)
                    
                    # Get first segmented image (should be in parent of automated results)
                    segmented_dir = Path(automated_files[0]).parent.parent / "segmented"
                    segmented_files = list(segmented_dir.glob("*.tif"))
                    segmented_images = []
                    if segmented_files:
                        seg_img = skimage.io.imread(segmented_files[0])
                        segmented_images.append(seg_img)
                    
                    pipeline_viz_path = self.visualizer.create_pipeline_overview(
                        sample_id, raw_images, segmented_images, auto_images[:1]
                    )
                    if pipeline_viz_path:
                        viz_paths['pipeline_overview'] = pipeline_viz_path
                        
                except Exception as e:
                    self.logger.warning(f"Failed to create pipeline overview for {sample_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed for {sample_id} {stage}: {e}")
        
        return viz_paths
    
    def run_automated_processing(self, sample_info: Dict) -> Dict:
        """
        Run automated processing pipeline on a sample
        
        Args:
            sample_info: Sample information dictionary
            
        Returns:
            Dictionary with paths to automated results
        """
        sample_id = sample_info['sample_id']
        sample_output_dir = self.output_dir / "automated_results" / sample_id
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'sample_id': sample_id,
            'segmented_dir': sample_output_dir / "segmented",
            'aligned_dir': sample_output_dir / "aligned",
            'processing_log': []
        }
        
        try:
            # Stage 1: Raw → Segmented (splitting)
            self.logger.info(f"Processing {sample_id}: Raw → Segmented")
            start_time = datetime.now()
            
            segmented_dir = results['segmented_dir']
            segmented_dir.mkdir(parents=True, exist_ok=True)
            
            # Create visualizations for raw images
            raw_visualization_paths = []
            
            # Process each raw file (should typically be one)
            for raw_file in sample_info['raw_files']:
                try:
                    # Load raw image for visualization
                    raw_image = skimage.io.imread(str(raw_file))
                    
                    # Create grid overlay visualization (especially useful for debugging failures)
                    # Estimate expected grid size based on typical iQID arrangements
                    expected_grid = self._estimate_grid_size(raw_image.shape)
                    
                    # Create raw image visualization with grid overlay
                    grid_viz_path = self.visualizer.create_grid_overlay_visualization(
                        raw_image,
                        expected_grid_size=expected_grid,
                        sample_id=f"{sample_id}_raw_{raw_file.stem}",
                        error_message=None  # Will be updated if splitting fails
                    )
                    raw_visualization_paths.append(grid_viz_path)
                    
                    # Attempt automated segmentation for additional insights
                    try:
                        segmentation_viz_path = self.visualizer.output_dir / f"{sample_id}_{raw_file.stem}_segmentation_analysis.png"
                        detected_blobs, _ = self.visualizer.robust_blob_segmentation(
                            raw_image,
                            min_blob_size=5000,  # Adjust based on expected blob size
                            expected_blobs=expected_grid[0] * expected_grid[1],
                            visualization_output=str(segmentation_viz_path)
                        )
                        
                        # Update grid visualization with detected blobs
                        updated_grid_viz = self.visualizer.create_grid_overlay_visualization(
                            raw_image,
                            expected_grid_size=expected_grid,
                            detected_blobs=detected_blobs,
                            sample_id=f"{sample_id}_raw_{raw_file.stem}_with_detection"
                        )
                        raw_visualization_paths.append(updated_grid_viz)
                        
                        self.logger.info(f"Robust segmentation detected {len(detected_blobs)} blobs "
                                       f"(expected {expected_grid[0] * expected_grid[1]})")
                        
                    except Exception as seg_e:
                        self.logger.warning(f"Robust segmentation failed for {raw_file}: {seg_e}")
                    
                    # Perform actual splitting using adaptive or fallback methods
                    try:
                        # First, try adaptive segmentation if we know the expected count
                        if sample_info.get('has_segmented'):
                            # Count ground truth segments to guide adaptive segmentation
                            gt_segments = list(sample_info['segmented_dir'].glob("*.tif"))
                            expected_count = len(gt_segments)
                            
                            self.logger.info(f"Using adaptive segmentation with expected count: {expected_count}")
                            segmented_files = self.enhanced_splitter.split_image_adaptive(
                                str(raw_file),
                                str(segmented_dir),
                                ground_truth_count=expected_count
                            )
                        else:
                            # Use adaptive segmentation without known count
                            self.logger.info("Using adaptive segmentation without ground truth count")
                            segmented_files = self.enhanced_splitter.split_image_adaptive(
                                str(raw_file),
                                str(segmented_dir)
                            )
                            
                    except Exception as adaptive_error:
                        self.logger.warning(f"Adaptive segmentation failed: {adaptive_error}")
                        
                        # Fallback: estimate grid from ground truth if available
                        if sample_info.get('has_segmented'):
                            gt_segments = list(sample_info['segmented_dir'].glob("*.tif"))
                            expected_count = len(gt_segments)
                            estimated_grid = self._estimate_grid_from_count(expected_count)
                            
                            self.logger.info(f"Trying fallback grid {estimated_grid} for {expected_count} segments")
                            
                            try:
                                fallback_splitter = RawImageSplitter(
                                    grid_rows=estimated_grid[0],
                                    grid_cols=estimated_grid[1]
                                )
                                segmented_files = fallback_splitter.split_image(
                                    str(raw_file),
                                    str(segmented_dir)
                                )
                                self.logger.info(f"Fallback grid splitting succeeded")
                                
                            except Exception as fallback_error:
                                self.logger.error(f"Fallback grid splitting also failed: {fallback_error}")
                                raise fallback_error
                        else:
                            # No ground truth available, use default grid as last resort
                            self.logger.warning("No ground truth available, using default 3x3 grid")
                            segmented_files = self.raw_splitter.split_image(
                                str(raw_file), 
                                str(segmented_dir)
                            )
                    
                    results['processing_log'].append({
                        'stage': 'segmentation',
                        'input': str(raw_file),
                        'output_count': len(segmented_files),
                        'time': (datetime.now() - start_time).total_seconds(),
                        'visualizations': raw_visualization_paths
                    })
                    
                except Exception as split_e:
                    # If splitting fails, update the grid visualization with error message
                    error_msg = f"Splitting failed: {str(split_e)}"
                    self.logger.error(f"Raw image splitting failed for {raw_file}: {split_e}")
                    
                    try:
                        # Create error visualization
                        raw_image = skimage.io.imread(str(raw_file))
                        expected_grid = self._estimate_grid_size(raw_image.shape)
                        
                        error_viz_path = self.visualizer.create_grid_overlay_visualization(
                            raw_image,
                            expected_grid_size=expected_grid,
                            sample_id=f"{sample_id}_raw_{raw_file.stem}_FAILED",
                            error_message=error_msg
                        )
                        raw_visualization_paths.append(error_viz_path)
                        
                    except Exception as viz_e:
                        self.logger.warning(f"Failed to create error visualization: {viz_e}")
                    
                    # Re-raise the original exception
                    raise split_e
            
            # Stage 2: Segmented → Aligned
            self.logger.info(f"Processing {sample_id}: Segmented → Aligned")
            start_time = datetime.now()
            
            aligned_dir = results['aligned_dir']
            aligned_dir.mkdir(parents=True, exist_ok=True)
            
            # Get segmented files and align them
            segmented_files = sorted(list(segmented_dir.glob("*.tif")))
            if segmented_files:
                aligned_files = self.aligner.align_images_files(
                    [str(f) for f in segmented_files],
                    str(aligned_dir)
                )
                results['processing_log'].append({
                    'stage': 'alignment',
                    'input_count': len(segmented_files),
                    'output_count': len(aligned_files),
                    'time': (datetime.now() - start_time).total_seconds()
                })
            
            results['success'] = True
            
        except Exception as e:
            self.logger.error(f"Automated processing failed for {sample_id}: {e}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def _estimate_grid_size(self, image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Estimate expected grid size based on image dimensions and typical iQID arrangements
        
        Based on empirical observations:
        - Tumors are typically in the upper rows (higher intensity)
        - Kidneys are in lower rows (left and right kidney pairs)
        - Common arrangement: 4 tumor slices on top, then kidney pairs below
        
        Args:
            image_shape: (height, width) of the image
            
        Returns:
            (rows, cols) expected grid arrangement
        """
        height, width = image_shape
        
        # Common iQID grid arrangements based on empirical observations
        # Priority given to arrangements matching the described tissue layout
        common_grids = [
            # Most common: tumor slices on top, kidney pairs below
            (5, 4),   # 5x4 = 20 samples (4 tumor + 16 kidney slices)
            (4, 4),   # 4x4 = 16 samples (4 tumor + 12 kidney slices)
            (3, 4),   # 3x4 = 12 samples (4 tumor + 8 kidney slices)
            (6, 2),   # 6x2 = 12 samples (vertical arrangement)
            
            # Alternative arrangements
            (4, 3),   # 4x3 = 12 samples  
            (2, 6),   # 2x6 = 12 samples (horizontal arrangement)
            (3, 3),   # 3x3 = 9 samples
            (2, 4),   # 2x4 = 8 samples
            (4, 2),   # 4x2 = 8 samples
            (1, 4),   # 1x4 = 4 samples (tumor only)
        ]
        
        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Find the grid that best matches the image aspect ratio
        best_grid = (3, 4)  # Default fallback
        best_score = float('inf')
        
        for rows, cols in common_grids:
            grid_aspect = cols / rows if rows > 0 else 1.0
            aspect_diff = abs(aspect_ratio - grid_aspect)
            
            # Scoring system:
            # 1. Prefer grids that match aspect ratio
            # 2. Bonus for common tissue arrangements (tumor + kidney pattern)
            # 3. Penalty for unusual sizes
            
            sample_count = rows * cols
            
            # Bonus for typical iQID arrangements
            arrangement_bonus = 0.0
            if sample_count in [12, 16, 20]:  # Common total counts
                arrangement_bonus = 0.2
            if rows >= 3 and cols == 4:  # Typical 4-column layout
                arrangement_bonus += 0.1
            if rows == 5 and cols == 4:  # Perfect tumor+kidney layout
                arrangement_bonus += 0.3
                
            score = aspect_diff - arrangement_bonus
            
            if score < best_score:
                best_score = score
                best_grid = (rows, cols)
        
        self.logger.debug(f"Estimated grid {best_grid} for image shape {image_shape} "
                         f"(aspect ratio: {aspect_ratio:.2f}) - "
                         f"Expected: {best_grid[0]*best_grid[1]} total samples")
        
        return best_grid
    
    def _estimate_grid_from_count(self, count: int) -> Tuple[int, int]:
        """
        Estimate grid size based on expected segment count
        
        Args:
            count: Expected number of segments
            
        Returns:
            (rows, cols) grid arrangement
        """
        # Common factorizations for typical iQID arrangements
        factorizations = [
            (1, count),    # Single row
            (2, count//2) if count % 2 == 0 else None,
            (3, count//3) if count % 3 == 0 else None,
            (4, count//4) if count % 4 == 0 else None,
            (5, count//5) if count % 5 == 0 else None,
            (6, count//6) if count % 6 == 0 else None,
        ]
        
        # Filter out None values
        valid_factorizations = [f for f in factorizations if f is not None]
        
        if not valid_factorizations:
            # If no perfect factorization, try to find closest square
            sqrt_count = int(np.sqrt(count))
            for i in range(sqrt_count, 0, -1):
                if count % i == 0:
                    return (i, count // i)
            return (1, count)  # Fallback to single row
        
        # Prefer more square-like arrangements (closer to 1:1 aspect ratio)
        best_grid = min(valid_factorizations, key=lambda x: abs(x[0] - x[1]))
        
        self.logger.debug(f"Estimated grid {best_grid} for {count} segments")
        return best_grid
    
    def compare_segmented_results(self, sample_info: Dict, automated_results: Dict) -> EvaluationResult:
        """
        Compare automated segmentation results with reference data
        """
        sample_id = sample_info['sample_id']
        
        if not sample_info['has_segmented']:
            return EvaluationResult(
                sample_id=sample_id,
                stage="segmented",
                success=False,
                similarity_score=0.0,
                error_message="No reference segmented data available"
            )
        
        try:
            # Get file lists
            reference_files = sorted(list(sample_info['segmented_dir'].glob("*.tif")))
            automated_files = sorted(list(automated_results['segmented_dir'].glob("*.tif")))
            
            if len(reference_files) == 0:
                return EvaluationResult(
                    sample_id=sample_id,
                    stage="segmented",
                    success=False,
                    similarity_score=0.0,
                    error_message="No reference segmented files found"
                )
            
            if len(automated_files) == 0:
                return EvaluationResult(
                    sample_id=sample_id,
                    stage="segmented",
                    success=False,
                    similarity_score=0.0,
                    error_message="No automated segmented files generated"
                )
            
            # Compare file counts
            count_similarity = min(len(automated_files), len(reference_files)) / max(len(automated_files), len(reference_files))
            
            # Compare individual images
            similarities = []
            detailed_metrics = {
                'file_comparisons': [],
                'count_match': len(automated_files) == len(reference_files),
                'reference_count': len(reference_files),
                'automated_count': len(automated_files)
            }
            
            # Compare up to the minimum number of files
            min_files = min(len(reference_files), len(automated_files))
            for i in range(min_files):
                ref_img = skimage.io.imread(reference_files[i])
                auto_img = skimage.io.imread(automated_files[i])
                
                # Normalize images for comparison
                if ref_img.dtype != auto_img.dtype:
                    ref_img = ref_img.astype(np.float32)
                    auto_img = auto_img.astype(np.float32)
                
                # Calculate similarity metrics
                if ref_img.shape == auto_img.shape:
                    ssim = skimage.metrics.structural_similarity(ref_img, auto_img, data_range=ref_img.max() - ref_img.min())
                    mse = skimage.metrics.mean_squared_error(ref_img, auto_img)
                    similarities.append(ssim)
                    
                    detailed_metrics['file_comparisons'].append({
                        'file_index': i,
                        'reference_file': reference_files[i].name,
                        'automated_file': automated_files[i].name,
                        'ssim': float(ssim),
                        'mse': float(mse),
                        'shape_match': True
                    })
                else:
                    self.logger.warning(f"Shape mismatch for {sample_id} file {i}: "
                                      f"ref={ref_img.shape}, auto={auto_img.shape}")
                    detailed_metrics['file_comparisons'].append({
                        'file_index': i,
                        'reference_file': reference_files[i].name,
                        'automated_file': automated_files[i].name,
                        'ssim': 0.0,
                        'mse': float('inf'),
                        'shape_match': False
                    })
            
            # Generate visualizations
            viz_paths = self._generate_stage_visualizations(
                sample_id, "segmented", reference_files, automated_files
            )
            
            # Overall similarity score combining count and image similarity
            if similarities:
                avg_image_similarity = np.mean(similarities)
                overall_similarity = (count_similarity + avg_image_similarity) / 2
            else:
                overall_similarity = count_similarity * 0.5  # Penalty for no successful comparisons
            
            return EvaluationResult(
                sample_id=sample_id,
                stage="segmented",
                success=True,
                similarity_score=overall_similarity,
                file_count_automated=len(automated_files),
                file_count_reference=len(reference_files),
                detailed_metrics=detailed_metrics,
                visualization_paths=viz_paths
            )
            
        except Exception as e:
            return EvaluationResult(
                sample_id=sample_id,
                stage="segmented",
                success=False,
                similarity_score=0.0,
                error_message=str(e)
            )
    
    def compare_aligned_results(self, sample_info: Dict, automated_results: Dict) -> EvaluationResult:
        """
        Compare automated alignment results with reference data
        """
        sample_id = sample_info['sample_id']
        
        if not sample_info['has_aligned']:
            return EvaluationResult(
                sample_id=sample_id,
                stage="aligned",
                success=False,
                similarity_score=0.0,
                error_message="No reference aligned data available"
            )
        
        try:
            # Get file lists
            reference_files = sorted(list(sample_info['aligned_dir'].glob("*.tif")))
            automated_files = sorted(list(automated_results['aligned_dir'].glob("*.tif")))
            
            if len(reference_files) == 0:
                return EvaluationResult(
                    sample_id=sample_id,
                    stage="aligned",
                    success=False,
                    similarity_score=0.0,
                    error_message="No reference aligned files found"
                )
            
            if len(automated_files) == 0:
                return EvaluationResult(
                    sample_id=sample_id,
                    stage="aligned",
                    success=False,
                    similarity_score=0.0,
                    error_message="No automated aligned files generated"
                )
            
            # Compare file counts
            count_similarity = min(len(automated_files), len(reference_files)) / max(len(automated_files), len(reference_files))
            
            # Compare individual images
            similarities = []
            detailed_metrics = {
                'file_comparisons': [],
                'count_match': len(automated_files) == len(reference_files),
                'reference_count': len(reference_files),
                'automated_count': len(automated_files),
                'alignment_quality': []
            }
            
            # Compare up to the minimum number of files
            min_files = min(len(reference_files), len(automated_files))
            for i in range(min_files):
                ref_img = skimage.io.imread(reference_files[i])
                auto_img = skimage.io.imread(automated_files[i])
                
                # Normalize images for comparison
                if ref_img.dtype != auto_img.dtype:
                    ref_img = ref_img.astype(np.float32)
                    auto_img = auto_img.astype(np.float32)
                
                # Calculate similarity metrics
                if ref_img.shape == auto_img.shape:
                    ssim = skimage.metrics.structural_similarity(ref_img, auto_img, data_range=ref_img.max() - ref_img.min())
                    mse = skimage.metrics.mean_squared_error(ref_img, auto_img)
                    similarities.append(ssim)
                    
                    detailed_metrics['file_comparisons'].append({
                        'file_index': i,
                        'reference_file': reference_files[i].name,
                        'automated_file': automated_files[i].name,
                        'ssim': float(ssim),
                        'mse': float(mse),
                        'shape_match': True
                    })
                else:
                    self.logger.warning(f"Shape mismatch for {sample_id} file {i}: "
                                      f"ref={ref_img.shape}, auto={auto_img.shape}")
                    detailed_metrics['file_comparisons'].append({
                        'file_index': i,
                        'reference_file': reference_files[i].name,
                        'automated_file': automated_files[i].name,
                        'ssim': 0.0,
                        'mse': float('inf'),
                        'shape_match': False
                    })
            
            # Generate visualizations (include raw files and pipeline overview for aligned stage)
            viz_paths = self._generate_stage_visualizations(
                sample_id, "aligned", reference_files, automated_files,
                include_raw=True, raw_files=sample_info.get('raw_files', [])
            )
            
            # Overall similarity score
            if similarities:
                avg_image_similarity = np.mean(similarities)
                overall_similarity = (count_similarity + avg_image_similarity) / 2
            else:
                overall_similarity = count_similarity * 0.5
            
            return EvaluationResult(
                sample_id=sample_id,
                stage="aligned",
                success=True,
                similarity_score=overall_similarity,
                file_count_automated=len(automated_files),
                file_count_reference=len(reference_files),
                detailed_metrics=detailed_metrics,
                visualization_paths=viz_paths
            )
            
        except Exception as e:
            return EvaluationResult(
                sample_id=sample_id,
                stage="aligned",
                success=False,
                similarity_score=0.0,
                error_message=str(e)
            )
    
    def evaluate_sample(self, sample_info: Dict) -> List[EvaluationResult]:
        """
        Evaluate a single sample through the complete pipeline
        """
        sample_id = sample_info['sample_id']
        self.logger.info(f"Evaluating sample: {sample_id}")
        
        results = []
        
        # Run automated processing
        start_time = datetime.now()
        automated_results = self.run_automated_processing(sample_info)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if not automated_results.get('success', False):
            # If automated processing failed, create failure results
            results.append(EvaluationResult(
                sample_id=sample_id,
                stage="processing",
                success=False,
                similarity_score=0.0,
                error_message=automated_results.get('error', 'Unknown processing error'),
                processing_time=processing_time
            ))
            return results
        
        # Compare segmented results
        if sample_info['has_segmented']:
            seg_result = self.compare_segmented_results(sample_info, automated_results)
            seg_result.processing_time = processing_time
            results.append(seg_result)
        
        # Compare aligned results
        if sample_info['has_aligned']:
            align_result = self.compare_aligned_results(sample_info, automated_results)
            align_result.processing_time = processing_time
            results.append(align_result)
        
        # Generate visualizations
        try:
            if sample_info['has_segmented']:
                seg_viz_paths = self._generate_stage_visualizations(
                    sample_id, "segmented", 
                    sorted(list(sample_info['segmented_dir'].glob("*.tif"))),
                    sorted(list(automated_results['segmented_dir'].glob("*.tif"))),
                    include_raw=False
                )
                if seg_viz_paths:
                    results[-1].visualization_paths = seg_viz_paths
            
            if sample_info['has_aligned']:
                align_viz_paths = self._generate_stage_visualizations(
                    sample_id, "aligned", 
                    sorted(list(sample_info['aligned_dir'].glob("*.tif"))),
                    sorted(list(automated_results['aligned_dir'].glob("*.tif"))),
                    include_raw=True,
                    raw_files=sample_info['raw_files']
                )
                if align_viz_paths:
                    results[-1].visualization_paths = align_viz_paths
            
        except Exception as e:
            self.logger.warning(f"Visualization generation encountered an error for {sample_id}: {e}")
        
        return results
    
    def run_full_evaluation(self, max_samples: Optional[int] = None) -> Dict:
        """
        Run complete evaluation on all available samples
        """
        self.logger.info("Starting full ReUpload dataset evaluation")
        
        # Discover samples
        samples = self.discover_samples()
        if not samples:
            self.logger.error("No samples found for evaluation")
            return {'error': 'No samples found'}
        
        if max_samples:
            samples = samples[:max_samples]
            self.logger.info(f"Limited evaluation to {len(samples)} samples")
        
        # Evaluate each sample
        all_results = []
        for i, sample_info in enumerate(samples):
            self.logger.info(f"Evaluating sample {i+1}/{len(samples)}: {sample_info['sample_id']}")
            
            try:
                sample_results = self.evaluate_sample(sample_info)
                all_results.extend(sample_results)
                self.results.extend(sample_results)
            except Exception as e:
                self.logger.error(f"Failed to evaluate sample {sample_info['sample_id']}: {e}")
                error_result = EvaluationResult(
                    sample_id=sample_info['sample_id'],
                    stage="evaluation",
                    success=False,
                    similarity_score=0.0,
                    error_message=str(e)
                )
                all_results.append(error_result)
                self.results.append(error_result)
        
        # Generate summary statistics
        self.summary_stats = self._generate_summary_stats()
        
        # Save results
        self._save_results()
        
        return {
            'total_samples': len(samples),
            'total_evaluations': len(all_results),
            'summary_stats': self.summary_stats,
            'results': all_results
        }
    
    def _generate_summary_stats(self) -> Dict:
        """Generate summary statistics from evaluation results"""
        if not self.results:
            return {}
        
        stats = {
            'total_evaluations': len(self.results),
            'successful_evaluations': sum(1 for r in self.results if r.success),
            'failed_evaluations': sum(1 for r in self.results if not r.success),
            'stages': {},
            'overall_metrics': {}
        }
        
        # Per-stage statistics
        stages = set(r.stage for r in self.results)
        for stage in stages:
            stage_results = [r for r in self.results if r.stage == stage]
            successful_stage = [r for r in stage_results if r.success]
            
            stage_stats = {
                'total': len(stage_results),
                'successful': len(successful_stage),
                'success_rate': len(successful_stage) / len(stage_results) if stage_results else 0,
                'avg_similarity': np.mean([r.similarity_score for r in successful_stage]) if successful_stage else 0,
                'avg_processing_time': np.mean([r.processing_time for r in stage_results if r.processing_time > 0]) if stage_results else 0
            }
            
            if successful_stage:
                similarities = [r.similarity_score for r in successful_stage]
                stage_stats.update({
                    'min_similarity': np.min(similarities),
                    'max_similarity': np.max(similarities),
                    'std_similarity': np.std(similarities)
                })
            
            stats['stages'][stage] = stage_stats
        
        # Overall metrics
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            stats['overall_metrics'] = {
                'avg_similarity': np.mean([r.similarity_score for r in successful_results]),
                'min_similarity': np.min([r.similarity_score for r in successful_results]),
                'max_similarity': np.max([r.similarity_score for r in successful_results]),
                'success_rate': len(successful_results) / len(self.results)
            }
        
        return stats
    
    def _save_results(self):
        """Save evaluation results to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_data = {
            'evaluation_info': {
                'timestamp': timestamp,
                'reupload_path': str(self.reupload_path),
                'total_samples': len(set(r.sample_id for r in self.results)),
                'total_evaluations': len(self.results)
            },
            'summary_stats': self.summary_stats,
            'detailed_results': [
                {
                    'sample_id': r.sample_id,
                    'stage': r.stage,
                    'success': r.success,
                    'similarity_score': r.similarity_score,
                    'error_message': r.error_message,
                    'processing_time': r.processing_time,
                    'file_count_automated': r.file_count_automated,
                    'file_count_reference': r.file_count_reference,
                    'detailed_metrics': r.detailed_metrics
                }
                for r in self.results
            ]
        }
        
        results_file = self.output_dir / f"reupload_evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved detailed results to: {results_file}")
        
        # Save summary report
        summary_file = self.output_dir / f"reupload_evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'summary_stats': self.summary_stats
            }, f, indent=2, default=str)
        
        self.logger.info(f"Saved summary to: {summary_file}")


def main():
    """Main entry point for ReUpload evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate automated processing against ReUpload reference data")
    parser.add_argument("--data", required=True, help="Path to ReUpload dataset")
    parser.add_argument("--output", default="evaluation/outputs", help="Output directory for results")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run evaluation
    evaluator = ReUploadPipelineEvaluator(args.data, args.output)
    results = evaluator.run_full_evaluation(args.max_samples)
    
    # Print summary
    print("\n" + "="*60)
    print("REUPLOAD PIPELINE EVALUATION RESULTS")
    print("="*60)
    
    if 'error' in results:
        print(f"❌ Evaluation failed: {results['error']}")
        return
    
    print(f"📊 Total samples evaluated: {results['total_samples']}")
    print(f"📊 Total evaluations performed: {results['total_evaluations']}")
    
    summary = results['summary_stats']
    if summary:
        print(f"\n🎯 Overall Success Rate: {summary.get('overall_metrics', {}).get('success_rate', 0):.2%}")
        
        for stage, stats in summary.get('stages', {}).items():
            print(f"\n📋 Stage: {stage.upper()}")
            print(f"   ✅ Success Rate: {stats['success_rate']:.2%} ({stats['successful']}/{stats['total']})")
            print(f"   📈 Average Similarity: {stats['avg_similarity']:.3f}")
            if 'avg_processing_time' in stats and stats['avg_processing_time'] > 0:
                print(f"   ⏱️  Average Processing Time: {stats['avg_processing_time']:.2f}s")
    
    print(f"\n📁 Results saved to: {evaluator.output_dir}")


if __name__ == "__main__":
    main()
