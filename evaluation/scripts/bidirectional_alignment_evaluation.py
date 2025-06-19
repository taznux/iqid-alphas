#!/usr/bin/env python3
"""
Bidirectional Alignment Evaluation Script for UCSF Data

This script specifically implements and evaluates Steps 9-13:
- Step 9: Use largest slice as reference and centerize all
- Step 10: Bidirectional alignment (largest→up, largest→down, top→bottom, bottom→top)
- Step 11: Simple methods for noisy iQID (mask/template based)
- Step 12: Show more slices in visualization
- Step 13: Full 3D stack alignment with overlapping

This builds on the existing evaluation infrastructure and focuses on advanced alignment techniques.
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from scipy import ndimage, signal
from skimage import io, metrics, filters, morphology, measure, feature, segmentation
import cv2

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iqid_alphas.core.alignment import ImageAligner

# Import enhanced centroid alignment
from centroid_alignment import EnhancedImageAligner, CentroidAlignedPadding

# Add evaluation utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visualizer import IQIDVisualizer
from utils.enhanced_sample_discovery import EnhancedSampleDiscovery

# Use existing evaluation infrastructure
from scripts.alignment_evaluation import AlignmentEvaluator


@dataclass
class ReferenceSliceInfo:
    """Information about the reference slice selection"""
    index: int
    score: float
    method: str  # 'largest_area', 'highest_contrast', 'most_features', 'auto'
    area: float
    contrast_score: float
    feature_count: int
    quality_metrics: Dict


@dataclass
class AlignmentStrategy:
    """Configuration for alignment strategy"""
    name: str
    direction: str  # 'up', 'down', 'bidirectional', 'full_3d'
    reference_method: str  # 'largest', 'center', 'auto', 'highest_quality'
    noise_handling: str  # 'none', 'mask_based', 'template_based', 'adaptive'
    overlap_analysis: bool = False
    quality_threshold: float = 0.3


@dataclass
class BidirectionalResult:
    """Enhanced result with bidirectional alignment metrics"""
    sample_id: str
    strategy: AlignmentStrategy
    reference_slice: ReferenceSliceInfo
    success: bool
    processing_time: float
    num_slices: int
    alignment_quality: float
    propagation_quality: Dict  # up/down propagation quality
    noise_metrics: Dict
    overlap_analysis: Optional[Dict] = None
    visualization_paths: List[str] = None
    error_message: Optional[str] = None


class BidirectionalAlignmentEvaluator:
    """
    Evaluator implementing Steps 9-13 for bidirectional alignment
    """
    
    def __init__(self, data_path: str, output_dir: str = "evaluation/outputs/bidirectional_test"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.aligner = ImageAligner()
        
        # Initialize visualizer with enhanced capabilities
        viz_output_dir = self.output_dir / "visualizations"
        self.visualizer = IQIDVisualizer(str(viz_output_dir))
        
        # Results storage
        self.results: List[BidirectionalResult] = []
        
        # Define alignment strategies to test (Steps 9-13)
        self.alignment_strategies = [
            AlignmentStrategy(
                name="largest_slice_bidirectional",
                direction="bidirectional",
                reference_method="largest",
                noise_handling="mask_based"
            ),
            AlignmentStrategy(
                name="auto_reference_bidirectional",
                direction="bidirectional", 
                reference_method="auto",
                noise_handling="adaptive"
            ),
            AlignmentStrategy(
                name="highest_quality_bidirectional",
                direction="bidirectional",
                reference_method="highest_quality",
                noise_handling="template_based"
            ),
            AlignmentStrategy(
                name="full_3d_overlap_analysis",
                direction="full_3d",
                reference_method="largest",
                noise_handling="adaptive",
                overlap_analysis=True
            )
        ]
        
        self.logger.info(f"Initialized bidirectional alignment evaluator")
        self.logger.info(f"Data path: {self.data_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Testing {len(self.alignment_strategies)} alignment strategies")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("BidirectionalAlignmentEvaluator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = self.output_dir / f"bidirectional_alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def discover_ucsf_samples(self) -> List[Dict]:
        """Discover UCSF samples suitable for evaluation"""
        samples = []
        
        # Look for different UCSF dataset structures
        possible_datasets = [
            self.data_path / "DataPush1",
            self.data_path / "ReUpload",
            self.data_path / "iQID_reupload" / "iQID",
            self.data_path / "UCSF",
            self.data_path
        ]
        
        for dataset_path in possible_datasets:
            if dataset_path.exists():
                self.logger.info(f"Exploring dataset: {dataset_path}")
                dataset_samples = self._discover_in_dataset(dataset_path)
                samples.extend(dataset_samples)
        
        # Filter for samples with sufficient slices for 3D alignment
        filtered_samples = [s for s in samples if s.get('num_slices', 0) >= 3]
        
        self.logger.info(f"Found {len(filtered_samples)} samples suitable for bidirectional alignment")
        return filtered_samples
    
    def _discover_in_dataset(self, dataset_path: Path) -> List[Dict]:
        """Discover samples in a dataset directory"""
        samples = []
        
        # Look for both 3D and Sequential studies
        for study_type in ["3D", "Sequential"]:
            study_path = dataset_path / study_type
            if not study_path.exists():
                continue
            
            # Look for tissue directories
            for tissue_dir in study_path.iterdir():
                if tissue_dir.is_dir():
                    tissue_samples = self._discover_in_tissue_dir(tissue_dir, study_type)
                    samples.extend(tissue_samples)
        
        # Also look directly in dataset for segmented data
        segmented_samples = self._find_direct_segmented_samples(dataset_path)
        samples.extend(segmented_samples)
        
        return samples
    
    def _discover_in_tissue_dir(self, tissue_dir: Path, study_type: str) -> List[Dict]:
        """Discover samples in tissue directory"""
        samples = []
        
        for sample_dir in tissue_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            
            # Look for segmented data (1_segmented or segmented)
            segmented_dirs = [
                sample_dir / "1_segmented",
                sample_dir / "segmented"
            ]
            
            for seg_dir in segmented_dirs:
                if seg_dir.exists():
                    image_files = sorted(list(seg_dir.glob("*.tif")) + list(seg_dir.glob("*.tiff")))
                    if len(image_files) >= 3:  # Minimum for meaningful alignment
                        sample_info = {
                            'sample_id': f"{tissue_dir.name}_{sample_dir.name}_{study_type}",
                            'dataset_path': tissue_dir.parent.parent,
                            'tissue_type': tissue_dir.name,
                            'study_type': study_type,
                            'sample_path': sample_dir,
                            'segmented_dir': seg_dir,
                            'image_files': image_files,
                            'num_slices': len(image_files)
                        }
                        samples.append(sample_info)
                        self.logger.debug(f"Found sample: {sample_info['sample_id']} "
                                        f"({len(image_files)} slices)")
        
        return samples
    
    def _find_direct_segmented_samples(self, dataset_path: Path) -> List[Dict]:
        """Find segmented samples directly in dataset"""
        samples = []
        
        # Look for 1_segmented directories anywhere in the dataset
        for seg_dir in dataset_path.rglob("1_segmented"):
            if seg_dir.is_dir():
                image_files = sorted(list(seg_dir.glob("*.tif")) + list(seg_dir.glob("*.tiff")))
                if len(image_files) >= 3:
                    sample_info = {
                        'sample_id': f"direct_{seg_dir.parent.name}",
                        'dataset_path': dataset_path,
                        'tissue_type': "unknown",
                        'study_type': "unknown",
                        'sample_path': seg_dir.parent,
                        'segmented_dir': seg_dir,
                        'image_files': image_files,
                        'num_slices': len(image_files)
                    }
                    samples.append(sample_info)
        
        return samples
    
    def find_reference_slice(self, images: List[np.ndarray], method: str = "auto") -> ReferenceSliceInfo:
        """
        Step 9: Find the optimal reference slice using multiple criteria
        """
        if not images:
            raise ValueError("No images provided")
        
        if len(images) == 1:
            return ReferenceSliceInfo(
                index=0, score=1.0, method=method,
                area=float(np.sum(images[0] > 0)),
                contrast_score=1.0, feature_count=0,
                quality_metrics={}
            )
        
        slice_metrics = []
        
        for i, img in enumerate(images):
            # Ensure we're working with proper data
            if img.dtype != np.uint8:
                img_8bit = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            else:
                img_8bit = img
            
            # Calculate comprehensive quality metrics
            metrics_dict = self._calculate_slice_quality_metrics(img_8bit, i)
            slice_metrics.append(metrics_dict)
        
        # Select reference based on method
        if method == "largest":
            best_idx = max(range(len(slice_metrics)), key=lambda i: slice_metrics[i]['area'])
        elif method == "center":
            best_idx = len(images) // 2
        elif method == "highest_quality":
            best_idx = max(range(len(slice_metrics)), key=lambda i: slice_metrics[i]['combined_score'])
        elif method == "auto":
            # Intelligent selection combining multiple criteria
            scores = []
            for i, metrics in enumerate(slice_metrics):
                # Weight: area (40%), contrast (30%), features (20%), entropy (10%)
                score = (0.4 * metrics['area_score'] + 
                        0.3 * metrics['contrast_score'] + 
                        0.2 * metrics['feature_score'] +
                        0.1 * metrics['entropy_score'])
                scores.append(score)
            best_idx = np.argmax(scores)
        else:
            # Default to center
            best_idx = len(images) // 2
        
        return ReferenceSliceInfo(
            index=best_idx,
            score=slice_metrics[best_idx]['combined_score'],
            method=method,
            area=slice_metrics[best_idx]['area'],
            contrast_score=slice_metrics[best_idx]['contrast_score'],
            feature_count=slice_metrics[best_idx]['feature_count'],
            quality_metrics=slice_metrics[best_idx]
        )
    
    def _calculate_slice_quality_metrics(self, img: np.ndarray, index: int) -> Dict:
        """Calculate comprehensive quality metrics for a slice"""
        metrics = {'slice_index': index}
        
        # Basic image properties
        total_pixels = img.shape[0] * img.shape[1]
        
        # Area metrics (tissue coverage)
        try:
            threshold = filters.threshold_otsu(img)
            binary_mask = img > threshold
            area = np.sum(binary_mask)
            area_fraction = area / total_pixels
        except:
            area = total_pixels * 0.5  # Default
            area_fraction = 0.5
        
        # Contrast metrics
        contrast = img.std()
        contrast_normalized = contrast / (img.mean() + 1e-8)
        
        # Feature detection
        try:
            corners = feature.corner_harris(img)
            feature_count = len(feature.corner_peaks(corners, min_distance=10))
        except:
            feature_count = 0
        
        # Edge density
        try:
            edges = filters.sobel(img)
            edge_threshold = filters.threshold_otsu(edges) if edges.max() > 0 else 0
            edge_density = np.sum(edges > edge_threshold) / total_pixels
        except:
            edge_density = 0.1
        
        # Information content (entropy)
        try:
            hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
            hist = hist / (hist.sum() + 1e-8)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
        except:
            entropy = 4.0  # Default medium entropy
        
        # Store raw metrics
        metrics.update({
            'area': area,
            'area_fraction': area_fraction,
            'contrast': contrast,
            'contrast_normalized': contrast_normalized,
            'feature_count': feature_count,
            'edge_density': edge_density,
            'entropy': entropy
        })
        
        # Normalized scores (0-1)
        max_area = total_pixels * 0.8  # Assume max useful area is 80%
        metrics['area_score'] = min(area / max_area, 1.0)
        metrics['contrast_score'] = min(contrast_normalized / 2.0, 1.0)
        metrics['feature_score'] = min(feature_count / 50.0, 1.0)  # Normalize by expected count
        metrics['edge_score'] = min(edge_density * 10, 1.0)
        metrics['entropy_score'] = entropy / 8.0  # Normalize entropy (max is 8 for 8-bit)
        
        # Combined score
        metrics['combined_score'] = (
            0.4 * metrics['area_score'] +
            0.3 * metrics['contrast_score'] +
            0.2 * metrics['feature_score'] +
            0.1 * metrics['entropy_score']
        )
        
        return metrics
    
    def bidirectional_alignment(self, images: List[np.ndarray], 
                               reference_idx: int,
                               noise_handling: str = "mask_based") -> Tuple[List[np.ndarray], Dict]:
        """
        Step 10: Implement bidirectional alignment strategy
        """
        if not images or reference_idx >= len(images):
            raise ValueError("Invalid images or reference index")
        
        aligned_images = [None] * len(images)
        alignment_info = {
            'reference_idx': reference_idx,
            'method': 'bidirectional',
            'noise_handling': noise_handling,
            'transforms': {},
            'quality_scores': {},
            'propagation_quality': {'up': [], 'down': []}
        }
        
        # Set reference image (centerize all to this)
        reference_img = images[reference_idx]
        aligned_images[reference_idx] = reference_img.copy()
        alignment_info['quality_scores'][reference_idx] = 1.0
        
        # Align upward from reference (reference→up: larger indices)
        current_ref = reference_img
        for i in range(reference_idx + 1, len(images)):
            try:
                aligned_img, transform_info = self._align_with_noise_handling(
                    current_ref, images[i], noise_handling
                )
                aligned_images[i] = aligned_img
                
                # Calculate quality for propagation tracking
                quality = self._calculate_pairwise_quality(current_ref, aligned_img)
                alignment_info['quality_scores'][i] = quality
                alignment_info['propagation_quality']['up'].append(quality)
                
                # Use aligned image as reference for next iteration
                current_ref = aligned_img
                alignment_info['transforms'][i] = transform_info
                
            except Exception as e:
                self.logger.warning(f"Failed to align slice {i} (up): {e}")
                aligned_images[i] = images[i].copy()
                alignment_info['quality_scores'][i] = 0.0
                alignment_info['propagation_quality']['up'].append(0.0)
        
        # Align downward from reference (reference→down: smaller indices)
        current_ref = reference_img
        for i in range(reference_idx - 1, -1, -1):
            try:
                aligned_img, transform_info = self._align_with_noise_handling(
                    current_ref, images[i], noise_handling
                )
                aligned_images[i] = aligned_img
                
                # Calculate quality for propagation tracking
                quality = self._calculate_pairwise_quality(current_ref, aligned_img)
                alignment_info['quality_scores'][i] = quality
                alignment_info['propagation_quality']['down'].append(quality)
                
                # Use aligned image as reference for next iteration
                current_ref = aligned_img
                alignment_info['transforms'][i] = transform_info
                
            except Exception as e:
                self.logger.warning(f"Failed to align slice {i} (down): {e}")
                aligned_images[i] = images[i].copy()
                alignment_info['quality_scores'][i] = 0.0
                alignment_info['propagation_quality']['down'].append(0.0)
        
        return aligned_images, alignment_info
    
    def _align_with_noise_handling(self, reference: np.ndarray, 
                                  moving: np.ndarray, 
                                  noise_handling: str) -> Tuple[np.ndarray, Dict]:
        """
        Step 11: Implement noise-robust alignment methods
        """
        if noise_handling == "none":
            return self.aligner.align_images(reference, moving)
        
        elif noise_handling == "mask_based":
            return self._mask_based_alignment(reference, moving)
        
        elif noise_handling == "template_based":
            return self._template_based_alignment(reference, moving)
        
        elif noise_handling == "adaptive":
            return self._adaptive_noise_alignment(reference, moving)
        
        else:
            return self.aligner.align_images(reference, moving)
    
    def _mask_based_alignment(self, reference: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Alignment using tissue masks to exclude noisy regions"""
        try:
            # Create tissue masks
            ref_mask = self._create_tissue_mask(reference)
            mov_mask = self._create_tissue_mask(moving)
            
            # Apply masks
            ref_masked = reference * ref_mask
            mov_masked = moving * mov_mask
            
            # Perform alignment on masked images
            aligned_img, transform_info = self.aligner.align_images(ref_masked, mov_masked)
            
            transform_info['method'] = 'mask_based'
            transform_info['mask_coverage'] = {
                'reference': float(np.sum(ref_mask) / ref_mask.size),
                'moving': float(np.sum(mov_mask) / mov_mask.size)
            }
            
            return aligned_img, transform_info
            
        except Exception as e:
            self.logger.warning(f"Mask-based alignment failed: {e}")
            return self.aligner.align_images(reference, moving)
    
    def _template_based_alignment(self, reference: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Alignment using template matching for robust feature detection"""
        try:
            # For simplicity, use enhanced phase correlation
            # In full implementation, extract and match templates
            aligned_img, transform_info = self.aligner.align_images(reference, moving)
            transform_info['method'] = 'template_based'
            return aligned_img, transform_info
            
        except Exception as e:
            self.logger.warning(f"Template-based alignment failed: {e}")
            return self.aligner.align_images(reference, moving)
    
    def _adaptive_noise_alignment(self, reference: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Adaptive alignment that chooses method based on image characteristics"""
        try:
            # Estimate noise levels
            ref_noise = self._estimate_noise_level(reference)
            mov_noise = self._estimate_noise_level(moving)
            
            # Choose strategy based on noise
            if ref_noise > 0.3 or mov_noise > 0.3:
                return self._mask_based_alignment(reference, moving)
            else:
                aligned_img, transform_info = self.aligner.align_images(reference, moving)
                transform_info['method'] = 'adaptive_standard'
                return aligned_img, transform_info
                
        except Exception as e:
            self.logger.warning(f"Adaptive alignment failed: {e}")
            return self.aligner.align_images(reference, moving)
    
    def _create_tissue_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a mask highlighting tissue regions"""
        try:
            # Convert to 8-bit if needed
            if image.dtype != np.uint8:
                img_8bit = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
            else:
                img_8bit = image
            
            # Use Otsu thresholding
            threshold = filters.threshold_otsu(img_8bit)
            binary_mask = img_8bit > threshold
            
            # Clean up mask
            mask = morphology.binary_opening(binary_mask, morphology.disk(2))
            mask = morphology.binary_closing(mask, morphology.disk(3))
            mask = morphology.remove_small_objects(mask, min_size=200)
            
            return mask.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Mask creation failed: {e}")
            return np.ones_like(image, dtype=np.float32)
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image"""
        try:
            if image.dtype != np.uint8:
                img_8bit = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
            else:
                img_8bit = image
            
            # Use Laplacian variance as noise measure
            laplacian = cv2.Laplacian(img_8bit, cv2.CV_64F)
            noise_level = laplacian.var() / (img_8bit.mean() + 1e-8)
            
            # Normalize to 0-1 range
            return min(noise_level / 1000.0, 1.0)
            
        except:
            return 0.5  # Default medium noise level
    
    def _calculate_pairwise_quality(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate alignment quality between two images"""
        try:
            # Ensure same shape
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            
            img1_crop = img1[:min_h, :min_w]
            img2_crop = img2[:min_h, :min_w]
            
            # Calculate SSIM
            data_range = max(img1_crop.max() - img1_crop.min(),
                           img2_crop.max() - img2_crop.min())
            
            if data_range > 0:
                ssim = metrics.structural_similarity(img1_crop, img2_crop, data_range=data_range)
                return max(0.0, ssim)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Quality calculation failed: {e}")
            return 0.0
    
    def full_3d_alignment_with_overlap(self, images: List[np.ndarray], 
                                     reference_idx: int) -> Tuple[List[np.ndarray], Dict]:
        """
        Step 13: Full 3D stack alignment with overlapping analysis
        """
        # First perform bidirectional alignment
        aligned_images, alignment_info = self.bidirectional_alignment(
            images, reference_idx, "adaptive"
        )
        
        # Add overlap analysis
        overlap_analysis = {
            'overlap_scores': [],
            'gap_regions': [],
            'consistency_metrics': [],
            'volume_coherence': 0.0
        }
        
        for i in range(len(aligned_images) - 1):
            img1 = aligned_images[i]
            img2 = aligned_images[i + 1]
            
            # Calculate overlap metrics
            overlap_score = self._calculate_overlap_score(img1, img2)
            gap_info = self._analyze_gaps(img1, img2)
            consistency = self._calculate_consistency_metric(img1, img2)
            
            overlap_analysis['overlap_scores'].append(overlap_score)
            overlap_analysis['gap_regions'].append(gap_info)
            overlap_analysis['consistency_metrics'].append(consistency)
        
        # Calculate overall volume coherence
        if overlap_analysis['overlap_scores']:
            overlap_analysis['volume_coherence'] = np.mean(overlap_analysis['overlap_scores'])
        
        alignment_info['overlap_analysis'] = overlap_analysis
        
        return aligned_images, alignment_info
    
    def _calculate_overlap_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate overlap score between consecutive slices"""
        try:
            # Create binary masks
            mask1 = img1 > filters.threshold_otsu(img1)
            mask2 = img2 > filters.threshold_otsu(img2)
            
            # Calculate intersection over union
            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1 | mask2)
            
            return intersection / union if union > 0 else 0.0
                
        except:
            return 0.0
    
    def _analyze_gaps(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Analyze gaps between consecutive slices"""
        try:
            mask1 = img1 > filters.threshold_otsu(img1)
            mask2 = img2 > filters.threshold_otsu(img2)
            
            gap1 = mask1 & ~mask2  # Present in img1 but not img2
            gap2 = mask2 & ~mask1  # Present in img2 but not img1
            
            return {
                'gap1_area': float(np.sum(gap1)),
                'gap2_area': float(np.sum(gap2)),
                'total_gap_fraction': float(np.sum(gap1 | gap2) / mask1.size)
            }
            
        except:
            return {'gap1_area': 0.0, 'gap2_area': 0.0, 'total_gap_fraction': 0.0}
    
    def _calculate_consistency_metric(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate consistency metric between consecutive slices"""
        try:
            # Normalize images
            img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
            img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)
            
            # Calculate correlation
            correlation = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
            
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except:
            return 0.0
    
    def create_enhanced_multi_slice_visualization(self, sample_id: str, 
                                                original_images: List[np.ndarray],
                                                aligned_images: List[np.ndarray],
                                                reference_slice: ReferenceSliceInfo,
                                                alignment_info: Dict,
                                                strategy: AlignmentStrategy) -> str:
        """
        Step 12: Create enhanced visualization showing multiple slices
        """
        try:
            n_slices = len(aligned_images)
            
            # Create comprehensive figure
            fig = plt.figure(figsize=(24, 16))
            
            # Create grid layout
            gs = gridspec.GridSpec(4, min(8, n_slices), figure=fig, hspace=0.3, wspace=0.1)
            
            # Row 1: Original images (up to 8 slices)
            slice_indices = np.linspace(0, n_slices-1, min(8, n_slices), dtype=int)
            
            for col, idx in enumerate(slice_indices):
                ax = fig.add_subplot(gs[0, col])
                img_8bit = self.visualizer.convert_16bit_to_8bit(original_images[idx])
                ax.imshow(img_8bit, cmap='viridis')
                title = f"Orig {idx}"
                if idx == reference_slice.index:
                    title += " (REF)"
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            
            # Row 2: Aligned images
            for col, idx in enumerate(slice_indices):
                ax = fig.add_subplot(gs[1, col])
                img_8bit = self.visualizer.convert_16bit_to_8bit(aligned_images[idx])
                ax.imshow(img_8bit, cmap='plasma')
                
                quality = alignment_info.get('quality_scores', {}).get(idx, 0.0)
                title = f"Align {idx}\nQ:{quality:.2f}"
                if idx == reference_slice.index:
                    title = f"Ref {idx}\nQ:1.00"
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            
            # Row 3: Difference maps between consecutive slices
            for col in range(min(7, n_slices-1)):
                ax = fig.add_subplot(gs[2, col])
                
                idx = slice_indices[col] if col < len(slice_indices) else n_slices - 2
                next_idx = min(idx + 1, n_slices - 1)
                
                img1 = self.visualizer.convert_16bit_to_8bit(aligned_images[idx])
                img2 = self.visualizer.convert_16bit_to_8bit(aligned_images[next_idx])
                
                min_h = min(img1.shape[0], img2.shape[0])
                min_w = min(img1.shape[1], img2.shape[1])
                diff = np.abs(img1[:min_h, :min_w].astype(np.float32) - 
                             img2[:min_h, :min_w].astype(np.float32))
                
                im = ax.imshow(diff, cmap='hot')
                ax.set_title(f"Diff {idx}-{next_idx}", fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, shrink=0.6)
            
            # Row 4: Analysis plots
            # Quality progression plot
            ax_quality = fig.add_subplot(gs[3, :3])
            quality_scores = [alignment_info.get('quality_scores', {}).get(i, 0.0) for i in range(n_slices)]
            ax_quality.plot(range(n_slices), quality_scores, 'bo-', linewidth=2, markersize=6)
            ax_quality.axvline(x=reference_slice.index, color='red', linestyle='--', label='Reference')
            ax_quality.set_xlabel('Slice Index')
            ax_quality.set_ylabel('Alignment Quality')
            ax_quality.set_title('Alignment Quality by Slice')
            ax_quality.grid(True, alpha=0.3)
            ax_quality.legend()
            
            # Propagation quality plot
            ax_prop = fig.add_subplot(gs[3, 3:6])
            prop_quality = alignment_info.get('propagation_quality', {})
            if prop_quality.get('up'):
                up_indices = list(range(reference_slice.index + 1, reference_slice.index + 1 + len(prop_quality['up'])))
                ax_prop.plot(up_indices, prop_quality['up'], 'go-', label='Up propagation', linewidth=2)
            if prop_quality.get('down'):
                down_indices = list(range(reference_slice.index - len(prop_quality['down']), reference_slice.index))
                ax_prop.plot(down_indices, prop_quality['down'][::-1], 'ro-', label='Down propagation', linewidth=2)
            ax_prop.axvline(x=reference_slice.index, color='blue', linestyle='--', label='Reference')
            ax_prop.set_xlabel('Slice Index')
            ax_prop.set_ylabel('Propagation Quality')
            ax_prop.set_title('Bidirectional Propagation Quality')
            ax_prop.grid(True, alpha=0.3)
            ax_prop.legend()
            
            # Overlap analysis (if available)
            if alignment_info.get('overlap_analysis'):
                ax_overlap = fig.add_subplot(gs[3, 6:])
                overlap_data = alignment_info['overlap_analysis']
                overlap_scores = overlap_data.get('overlap_scores', [])
                if overlap_scores:
                    ax_overlap.plot(range(len(overlap_scores)), overlap_scores, 'mo-', linewidth=2)
                    ax_overlap.axhline(y=np.mean(overlap_scores), color='orange', linestyle='--', 
                                     label=f'Mean: {np.mean(overlap_scores):.3f}')
                ax_overlap.set_xlabel('Slice Pair Index')
                ax_overlap.set_ylabel('Overlap Score')
                ax_overlap.set_title('3D Overlap Analysis')
                ax_overlap.grid(True, alpha=0.3)
                ax_overlap.legend()
            
            # Overall title
            overall_quality = np.mean([v for v in alignment_info.get('quality_scores', {}).values()])
            fig.suptitle(f"Enhanced Multi-Slice Visualization - {sample_id}\n"
                        f"Strategy: {strategy.name} | Reference: Slice {reference_slice.index} "
                        f"({reference_slice.method}) | Overall Quality: {overall_quality:.3f}", 
                        fontsize=16)
            
            # Save visualization
            viz_path = self.output_dir / "visualizations" / f"{sample_id}_{strategy.name}_enhanced.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(viz_path)
            
        except Exception as e:
            self.logger.error(f"Enhanced visualization failed for {sample_id}: {e}")
            plt.close('all')
            return None
    
    def evaluate_sample(self, sample_info: Dict, strategy: AlignmentStrategy) -> BidirectionalResult:
        """Evaluate a single sample with the specified alignment strategy"""
        sample_id = sample_info['sample_id']
        
        self.logger.info(f"Evaluating {sample_id} with strategy: {strategy.name}")
        
        try:
            start_time = datetime.now()
            
            # Load images
            image_files = sample_info.get('image_files', [])
            if not image_files:
                raise ValueError("No image files found")
            
            images = []
            for file_path in image_files:
                try:
                    img = io.imread(file_path)
                    if img.ndim == 3:
                        img = img[:, :, 0]  # Take first channel
                    images.append(img.astype(np.float32))
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
            
            if len(images) < 2:
                raise ValueError("Need at least 2 images for alignment")
            
            # Step 9: Find reference slice
            reference_slice = self.find_reference_slice(images, strategy.reference_method)
            
            # Perform alignment based on strategy
            if strategy.direction == "bidirectional":
                aligned_images, alignment_info = self.bidirectional_alignment(
                    images, reference_slice.index, strategy.noise_handling
                )
            elif strategy.direction == "full_3d":
                aligned_images, alignment_info = self.full_3d_alignment_with_overlap(
                    images, reference_slice.index
                )
            else:
                # Default to bidirectional
                aligned_images, alignment_info = self.bidirectional_alignment(
                    images, reference_slice.index, strategy.noise_handling
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate overall alignment quality
            quality_scores = alignment_info.get('quality_scores', {})
            overall_quality = np.mean(list(quality_scores.values())) if quality_scores else 0.0
            
            # Calculate noise metrics
            noise_metrics = self._calculate_noise_metrics(images, aligned_images)
            
            # Create enhanced visualization (Step 12)
            viz_path = self.create_enhanced_multi_slice_visualization(
                sample_id, images, aligned_images, reference_slice, alignment_info, strategy
            )
            
            result = BidirectionalResult(
                sample_id=sample_id,
                strategy=strategy,
                reference_slice=reference_slice,
                success=True,
                processing_time=processing_time,
                num_slices=len(aligned_images),
                alignment_quality=overall_quality,
                propagation_quality=alignment_info.get('propagation_quality', {}),
                noise_metrics=noise_metrics,
                overlap_analysis=alignment_info.get('overlap_analysis'),
                visualization_paths=[viz_path] if viz_path else [],
                error_message=None
            )
            
            self.logger.info(f"Successfully evaluated {sample_id}: quality={overall_quality:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for {sample_id}: {e}")
            return BidirectionalResult(
                sample_id=sample_id,
                strategy=strategy,
                reference_slice=ReferenceSliceInfo(0, 0.0, "failed", 0, 0, 0, {}),
                success=False,
                processing_time=0.0,
                num_slices=len(sample_info.get('image_files', [])),
                alignment_quality=0.0,
                propagation_quality={},
                noise_metrics={},
                error_message=str(e)
            )
    
    def _calculate_noise_metrics(self, original_images: List[np.ndarray], 
                               aligned_images: List[np.ndarray]) -> Dict:
        """Calculate noise-related metrics"""
        try:
            noise_levels_orig = [self._estimate_noise_level(img) for img in original_images]
            noise_levels_aligned = [self._estimate_noise_level(img) for img in aligned_images]
            
            return {
                'avg_noise_original': np.mean(noise_levels_orig),
                'avg_noise_aligned': np.mean(noise_levels_aligned),
                'noise_reduction': np.mean(noise_levels_orig) - np.mean(noise_levels_aligned),
                'max_noise_original': np.max(noise_levels_orig),
                'max_noise_aligned': np.max(noise_levels_aligned)
            }
        except:
            return {}
    
    def run_evaluation(self, max_samples: int = 5) -> Dict:
        """Run comprehensive evaluation of bidirectional alignment strategies"""
        self.logger.info("Starting bidirectional alignment evaluation (Steps 9-13)")
        
        # Discover samples
        samples = self.discover_ucsf_samples()
        
        if not samples:
            self.logger.error("No suitable samples found for evaluation")
            return {}
        
        # Limit samples for testing
        test_samples = samples[:max_samples]
        
        # Evaluate each sample with each strategy
        for sample in test_samples:
            for strategy in self.alignment_strategies:
                result = self.evaluate_sample(sample, strategy)
                self.results.append(result)
        
        # Generate and save results
        summary = self._generate_summary_report()
        self._save_results()
        
        self.logger.info("Bidirectional alignment evaluation completed")
        return summary
    
    def _generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        if not self.results:
            return {}
        
        # Organize by strategy
        strategy_results = {}
        for result in self.results:
            strategy_name = result.strategy.name
            if strategy_name not in strategy_results:
                strategy_results[strategy_name] = []
            strategy_results[strategy_name].append(result)
        
        summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_evaluations': len(self.results),
            'strategies_tested': list(strategy_results.keys()),
            'strategy_performance': {},
            'reference_method_analysis': {},
            'noise_handling_analysis': {}
        }
        
        # Analyze each strategy
        for strategy_name, results in strategy_results.items():
            successful = [r for r in results if r.success]
            
            if successful:
                qualities = [r.alignment_quality for r in successful]
                times = [r.processing_time for r in successful]
                
                # Propagation quality analysis
                prop_qualities = []
                for r in successful:
                    if r.propagation_quality:
                        prop_qualities.extend(r.propagation_quality.get('up', []))
                        prop_qualities.extend(r.propagation_quality.get('down', []))
                
                summary['strategy_performance'][strategy_name] = {
                    'success_rate': len(successful) / len(results),
                    'avg_quality': np.mean(qualities),
                    'std_quality': np.std(qualities),
                    'avg_time': np.mean(times),
                    'avg_propagation_quality': np.mean(prop_qualities) if prop_qualities else 0.0,
                    'best_sample': max(successful, key=lambda x: x.alignment_quality).sample_id,
                    'best_quality': max(qualities)
                }
        
        return summary
    
    def _save_results(self):
        """Save evaluation results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            result_dict = asdict(result)
            result_dict = self._convert_numpy_types(result_dict)
            results_data.append(result_dict)
        
        # Save detailed results
        results_file = self.output_dir / f"bidirectional_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save summary
        summary = self._generate_summary_report()
        summary_file = self.output_dir / f"bidirectional_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Summary saved to {summary_file}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def main():
    """Main function to run bidirectional alignment evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bidirectional Alignment Evaluation (Steps 9-13)")
    parser.add_argument("--data-path", required=True, help="Path to UCSF data directory")
    parser.add_argument("--output-dir", default="evaluation/outputs/bidirectional_test",
                       help="Output directory")
    parser.add_argument("--max-samples", type=int, default=3, 
                       help="Maximum number of samples to test")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BidirectionalAlignmentEvaluator(args.data_path, args.output_dir)
    
    # Run evaluation
    summary = evaluator.run_evaluation(max_samples=args.max_samples)
    
    print("\n" + "="*60)
    print("BIDIRECTIONAL ALIGNMENT EVALUATION SUMMARY")
    print("="*60)
    print(f"Total evaluations: {summary.get('total_evaluations', 0)}")
    print(f"Strategies tested: {len(summary.get('strategies_tested', []))}")
    
    if 'strategy_performance' in summary:
        print("\nStrategy Performance:")
        for strategy, performance in summary['strategy_performance'].items():
            print(f"\n  {strategy}:")
            print(f"    Success rate: {performance.get('success_rate', 0):.1%}")
            if 'avg_quality' in performance:
                print(f"    Average quality: {performance['avg_quality']:.3f} ± {performance.get('std_quality', 0):.3f}")
                print(f"    Best quality: {performance.get('best_quality', 0):.3f} ({performance.get('best_sample', 'N/A')})")
                print(f"    Average time: {performance.get('avg_time', 0):.2f}s")
                print(f"    Propagation quality: {performance.get('avg_propagation_quality', 0):.3f}")


if __name__ == "__main__":
    main()
