"""
Tissue Segmentation Module

Advanced tissue separation and segmentation for multi-tissue iQID raw images.
Migrated and refactored from legacy process_object.py and adaptive_segmentation.py.

This module implements the TissueSeparator class for automated tissue separation
from raw multi-tissue iQID images, with validation against ground truth data.
"""

import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import glob

# Image processing imports
try:
    from skimage import filters, morphology, measure, segmentation, feature
    from skimage.morphology import disk, opening, closing
    from scipy import ndimage
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    
try:
    from sklearn.cluster import DBSCAN, MeanShift
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Internal imports
from ..utils.io_utils import natural_sort, find_files_with_pattern, ensure_directory_exists
from ..utils.math_utils import normalize_array, calculate_statistics


class TissueSeparator:
    """
    Advanced tissue separation from multi-tissue raw iQID images.
    
    This class implements sophisticated segmentation algorithms to automatically
    separate multi-tissue raw iQID images into individual tissue types
    (e.g., kidney left/right, tumor, etc.).
    
    Key Features:
    - Adaptive clustering for low-contrast images
    - Multi-threshold segmentation
    - Noise-robust blob detection
    - Ground truth validation
    - Comprehensive quality metrics (IoU, Dice, precision, recall)
    """
    
    def __init__(self, 
                 method: str = 'adaptive_clustering',
                 min_blob_area: int = 50,
                 max_blob_area: int = 50000,
                 preserve_blobs: bool = True,
                 use_clustering: bool = True):
        """
        Initialize the tissue separator.
        
        Parameters
        ----------
        method : str, default 'adaptive_clustering'
            Segmentation method to use:
            - 'adaptive_clustering': Cluster distributed dots into tissue blobs
            - 'multi_threshold': Multiple threshold levels
            - 'watershed': Watershed-based segmentation
            - 'combined': Combination of multiple methods
        min_blob_area : int, default 50
            Minimum area for valid tissue blobs (pixels)
        max_blob_area : int, default 50000
            Maximum area for valid tissue blobs (pixels)
        preserve_blobs : bool, default True
            If True, use less aggressive morphological operations
        use_clustering : bool, default True
            If True, use clustering to group distributed dots
        """
        self.method = method
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.preserve_blobs = preserve_blobs
        self.use_clustering = use_clustering
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Validation data structures
        self._validation_results = {}
        self._processing_stats = {}
        
        # Check for required dependencies
        if not HAS_SKIMAGE:
            raise ImportError(
                "scikit-image is required for tissue segmentation. "
                "Install with: pip install scikit-image"
            )
    
    def separate_tissues(self, 
                        raw_image_path: Union[str, Path], 
                        output_dir: Optional[Union[str, Path]] = None) -> Dict[str, List[Path]]:
        """
        Separate tissues from a raw multi-tissue iQID image.
        
        This is the main public method that processes a raw image and returns
        separated tissue images, each containing a single tissue type.
        
        Parameters
        ----------
        raw_image_path : str or Path
            Path to the raw multi-tissue iQID image file
        output_dir : str or Path, optional
            Directory to save separated tissue images. If None, creates
            a subdirectory next to the input file.
            
        Returns
        -------
        Dict[str, List[Path]]
            Dictionary mapping tissue type names to lists of image file paths:
            {
                'kidney_left': [path1, path2, ...],
                'kidney_right': [path3, path4, ...], 
                'tumor': [path5, path6, ...],
                ...
            }
        """
        raw_image_path = Path(raw_image_path)
        
        if not raw_image_path.exists():
            raise FileNotFoundError(f"Raw image file not found: {raw_image_path}")
        
        self.logger.info(f"Starting tissue separation for: {raw_image_path}")
        
        # Set up output directory
        if output_dir is None:
            output_dir = raw_image_path.parent / f"{raw_image_path.stem}_separated"
        else:
            output_dir = Path(output_dir)
        
        ensure_directory_exists(output_dir)
        
        # Load and preprocess the raw image
        raw_image = self._load_raw_image(raw_image_path)
        preprocessed_image = self._preprocess_raw_image(raw_image)
        
        # Detect tissue regions
        tissue_regions = self._detect_tissue_regions(preprocessed_image)
        
        # Classify tissues by type
        classified_tissues = self._classify_tissue_types(tissue_regions, preprocessed_image)
        
        # Generate separated tissue images
        separated_paths = self._generate_separated_images(
            classified_tissues, preprocessed_image, output_dir, raw_image_path.stem
        )
        
        # Store processing statistics
        self._processing_stats[str(raw_image_path)] = {
            'num_regions_detected': len(tissue_regions),
            'num_tissues_classified': len(classified_tissues),
            'tissue_types': list(classified_tissues.keys()),
            'output_dir': str(output_dir)
        }
        
        self.logger.info(f"Tissue separation complete. Found {len(classified_tissues)} tissue types.")
        
        return separated_paths
    
    def validate_against_ground_truth(self, 
                                    automated_results: Dict[str, List[Path]], 
                                    ground_truth_dirs: Dict[str, Path]) -> Dict[str, Any]:
        """
        Validate automated tissue separation against ground truth data.
        
        Parameters
        ----------
        automated_results : Dict[str, List[Path]]
            Results from separate_tissues() method
        ground_truth_dirs : Dict[str, Path]
            Dictionary mapping tissue types to ground truth directory paths:
            {'kidney_left': Path('gt/kidney_left/'), ...}
            
        Returns
        -------
        Dict[str, Any]
            Validation results with metrics for each tissue type:
            {
                'overall': {'avg_iou': float, 'avg_dice': float, ...},
                'by_tissue': {
                    'kidney_left': {'iou': float, 'dice': float, ...},
                    'kidney_right': {'iou': float, 'dice': float, ...},
                    ...
                }
            }
        """
        self.logger.info("Starting ground truth validation")
        
        validation_results = {
            'overall': {},
            'by_tissue': {},
            'detailed': {}
        }
        
        all_ious = []
        all_dices = []
        all_precisions = []
        all_recalls = []
        
        for tissue_type, automated_paths in automated_results.items():
            if tissue_type not in ground_truth_dirs:
                self.logger.warning(f"No ground truth available for tissue type: {tissue_type}")
                continue
            
            gt_dir = Path(ground_truth_dirs[tissue_type])
            if not gt_dir.exists():
                self.logger.warning(f"Ground truth directory not found: {gt_dir}")
                continue
            
            # Compare automated results with ground truth
            tissue_metrics = self._compare_tissue_with_ground_truth(
                automated_paths, gt_dir, tissue_type
            )
            
            validation_results['by_tissue'][tissue_type] = tissue_metrics
            
            # Collect overall metrics
            if 'iou' in tissue_metrics:
                all_ious.append(tissue_metrics['iou'])
            if 'dice' in tissue_metrics:
                all_dices.append(tissue_metrics['dice'])
            if 'precision' in tissue_metrics:
                all_precisions.append(tissue_metrics['precision'])
            if 'recall' in tissue_metrics:
                all_recalls.append(tissue_metrics['recall'])
        
        # Calculate overall metrics
        if all_ious:
            validation_results['overall']['avg_iou'] = np.mean(all_ious)
            validation_results['overall']['std_iou'] = np.std(all_ious)
        if all_dices:
            validation_results['overall']['avg_dice'] = np.mean(all_dices)
            validation_results['overall']['std_dice'] = np.std(all_dices)
        if all_precisions:
            validation_results['overall']['avg_precision'] = np.mean(all_precisions)
        if all_recalls:
            validation_results['overall']['avg_recall'] = np.mean(all_recalls)
        
        # Store validation results
        self._validation_results.update(validation_results)
        
        self.logger.info(f"Validation complete. Overall IoU: {validation_results['overall'].get('avg_iou', 'N/A'):.3f}")
        
        return validation_results
    
    def calculate_segmentation_metrics(self, 
                                     predicted_mask: np.ndarray, 
                                     ground_truth_mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive segmentation quality metrics.
        
        Parameters
        ----------
        predicted_mask : np.ndarray
            Binary mask from automated segmentation
        ground_truth_mask : np.ndarray
            Binary ground truth mask
            
        Returns
        -------
        Dict[str, float]
            Dictionary with segmentation metrics:
            - 'iou': Intersection over Union (Jaccard index)
            - 'dice': Dice coefficient (F1 score)
            - 'precision': Precision score
            - 'recall': Recall score (sensitivity)
            - 'specificity': Specificity score
            - 'accuracy': Overall accuracy
        """
        # Ensure binary masks
        pred_binary = (predicted_mask > 0).astype(bool)
        gt_binary = (ground_truth_mask > 0).astype(bool)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        # True/False positives/negatives
        tp = intersection  # True positives
        fp = pred_binary.sum() - tp  # False positives
        fn = gt_binary.sum() - tp  # False negatives
        tn = (~pred_binary & ~gt_binary).sum()  # True negatives
        
        # Calculate metrics with safe division
        metrics = {}
        
        # IoU (Intersection over Union)
        metrics['iou'] = intersection / union if union > 0 else 0.0
        
        # Dice coefficient (F1 score)
        metrics['dice'] = (2 * intersection) / (pred_binary.sum() + gt_binary.sum()) if (pred_binary.sum() + gt_binary.sum()) > 0 else 0.0
        
        # Precision
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall (Sensitivity)
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Accuracy
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return metrics
    
    # Private methods for internal processing
    
    def _load_raw_image(self, image_path: Path) -> np.ndarray:
        """
        Load raw iQID image data.
        
        Migrated from legacy process_object.py load methods.
        """
        try:
            if image_path.suffix.lower() in ['.dat', '.raw']:
                # Handle binary iQID format (migrated from ClusterData.load_cluster_data)
                return self._load_iqid_binary_data(image_path)
            else:
                # Handle standard image formats
                from skimage import io
                return io.imread(str(image_path))
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def _load_iqid_binary_data(self, file_path: Path) -> np.ndarray:
        """
        Load binary iQID data format.
        
        Migrated from ClusterData.load_cluster_data() and init_header().
        """
        # Read header to get dimensions
        header = np.fromfile(str(file_path), dtype=np.int32, count=100)
        header_size = header[0]
        xdim = header[1]
        ydim = header[2]
        
        file_size_bytes = os.path.getsize(str(file_path))
        
        # Determine data type and number of elements based on file structure
        # For processed listmode data
        num_data_elements = 14  # Default for processed_lm format
        
        byte_size = 8  # Using float64
        byte_fac = 2
        
        num_clusters = np.floor(
            (file_size_bytes - 4 * header_size) / (byte_size * num_data_elements))
        
        # Load the data
        unshaped_data = np.fromfile(
            str(file_path), 
            dtype=np.float64, 
            count=header_size // byte_fac + int(num_clusters * num_data_elements)
        )
        
        data = unshaped_data[header_size // byte_fac:].reshape(
            int(num_data_elements), int(num_clusters), order='F'
        )
        
        # Extract coordinate data (migrated from init_metadata)
        yC_global = data[4, :]  # Y coordinates
        xC_global = data[5, :]  # X coordinates
        cluster_area = data[3, :]  # Cluster areas
        sum_cluster_signal = data[2, :]  # Signal intensities
        
        # Create 2D image from coordinate data
        image = np.zeros((ydim, xdim), dtype=np.float64)
        
        # Populate image with cluster data
        for i in range(len(xC_global)):
            x_coord = int(np.clip(xC_global[i], 0, xdim - 1))
            y_coord = int(np.clip(yC_global[i], 0, ydim - 1))
            
            # Use signal intensity if available, otherwise use area
            intensity = sum_cluster_signal[i] if len(sum_cluster_signal) > i else cluster_area[i]
            image[y_coord, x_coord] = intensity
        
        return image
    
    def _preprocess_raw_image(self, raw_image: np.ndarray) -> np.ndarray:
        """
        Preprocess raw image for segmentation.
        
        Applies noise reduction, normalization, and enhancement.
        """
        # Normalize the image
        preprocessed = normalize_array(raw_image, method='minmax')
        
        # Apply gentle smoothing to reduce noise while preserving edges
        if HAS_SKIMAGE:
            preprocessed = filters.gaussian(preprocessed, sigma=1.0, preserve_range=True)
        
        return preprocessed
    
    def _detect_tissue_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect tissue regions using the specified segmentation method.
        
        Migrated and enhanced from adaptive_segmentation.py
        """
        if self.method == 'adaptive_clustering':
            return self._clustering_based_segmentation(image)
        elif self.method == 'multi_threshold':
            return self._multi_threshold_segmentation(image)
        elif self.method == 'watershed':
            return self._watershed_segmentation(image)
        elif self.method == 'combined':
            return self._combined_segmentation(image)
        else:
            self.logger.warning(f"Unknown method {self.method}, using adaptive_clustering")
            return self._clustering_based_segmentation(image)
    
