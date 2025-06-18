#!/usr/bin/env python3
"""
Adaptive Segmentation for iQID Data

This module provides robust segmentation methods for iQID activity data
that can handle variable noise levels and intensity ranges across slices.
"""

import numpy as np
import cv2
from skimage import filters, morphology, measure, segmentation, feature
from skimage.morphology import disk, opening, closing
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
try:
    from sklearn.cluster import DBSCAN, MeanShift
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path


class AdaptiveIQIDSegmenter:
    """
    Adaptive segmentation for iQID activity blobs with noise robustness.
    Handles low contrast images with distributed dots forming blobs.
    """
    
    def __init__(self, min_blob_area: int = 50, max_blob_area: int = 50000, 
                 preserve_blobs: bool = True, use_clustering: bool = True):
        """
        Initialize the adaptive segmenter
        
        Args:
            min_blob_area: Minimum area for valid activity blobs
            max_blob_area: Maximum area for valid activity blobs
            preserve_blobs: If True, use less aggressive morphological operations
            use_clustering: If True, use clustering to group distributed dots into blobs
        """
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.preserve_blobs = preserve_blobs
        self.use_clustering = use_clustering
        self.logger = logging.getLogger(__name__)
        
    def detect_activity_regions(self, image: np.ndarray, 
                              method: str = "clustering") -> List[Dict]:
        """
        Detect activity regions using multiple robust methods
        
        Args:
            image: Input iQID activity image
            method: Segmentation method ('clustering', 'multi_threshold', 'watershed', 'adaptive', 'combined')
            
        Returns:
            List of detected region dictionaries with bbox, centroid, area, etc.
        """
        if method == "clustering":
            return self._clustering_based_segmentation(image)
        elif method == "multi_threshold":
            return self._multi_threshold_segmentation(image)
        elif method == "watershed":
            return self._watershed_segmentation(image)
        elif method == "adaptive":
            return self._adaptive_threshold_segmentation(image)
        elif method == "combined":
            return self._combined_segmentation(image)
        else:
            self.logger.warning(f"Unknown method {method}, using clustering")
            return self._clustering_based_segmentation(image)
    
    def _clustering_based_segmentation(self, image: np.ndarray) -> List[Dict]:
        """
        Use clustering to group distributed dots into blobs for low contrast images
        """
        self.logger.debug("Running clustering-based segmentation for distributed dots")
        
        # Step 1: Detect individual dots/peaks with very low threshold
        dot_positions = self._detect_activity_dots(image)
        
        if len(dot_positions) < 2:
            self.logger.debug("Not enough dots detected for clustering")
            return self._fallback_to_threshold_segmentation(image)
        
        # Step 2: Cluster nearby dots into blob groups
        blob_clusters = self._cluster_dots_into_blobs(dot_positions, image)
        
        # Step 3: Create blob masks from clusters
        regions = self._create_blob_regions_from_clusters(image, blob_clusters, dot_positions)
        
        self.logger.debug(f"Clustering detected {len(regions)} blob regions from {len(dot_positions)} dots")
        return regions
    
    def _detect_activity_dots(self, image: np.ndarray) -> np.ndarray:
        """
        Detect individual activity dots/peaks in low contrast image
        """
        # Use very low threshold to catch weak signals
        # Try multiple approaches to detect dots
        
        # Method 1: Local maxima detection
        local_maxima = feature.peak_local_max(
            image, 
            min_distance=3,  # Minimum distance between peaks
            threshold_abs=np.percentile(image[image > 0], 15),  # Very low threshold
            threshold_rel=0.1  # 10% of maximum
        )
        
        if len(local_maxima[0]) > 0:
            dots_maxima = np.column_stack(local_maxima)
        else:
            dots_maxima = np.array([]).reshape(0, 2)
        
        # Method 2: Adaptive threshold with connected components
        # Use very low threshold to catch weak signals
        low_threshold = np.percentile(image[image > 0], 25) if np.any(image > 0) else 0
        binary_low = image > low_threshold
        
        # Only remove single pixel noise (spike noise)
        if self.preserve_blobs:
            # Minimal spike noise removal - only remove isolated pixels
            kernel_spike = np.ones((2, 2), np.uint8)  # Very small kernel
            binary_cleaned = cv2.morphologyEx(binary_low.astype(np.uint8), 
                                            cv2.MORPH_OPEN, kernel_spike)
        else:
            binary_cleaned = binary_low.astype(np.uint8)
        
        # Find centroids of small connected components (individual dots)
        labeled = measure.label(binary_cleaned)
        dots_cc = []
        
        for region in measure.regionprops(labeled):
            if 1 <= region.area <= 50:  # Individual dots or small clusters
                dots_cc.append(region.centroid)
        
        dots_cc = np.array(dots_cc) if dots_cc else np.array([]).reshape(0, 2)
        
        # Combine both methods
        if len(dots_maxima) > 0 and len(dots_cc) > 0:
            all_dots = np.vstack([dots_maxima, dots_cc])
        elif len(dots_maxima) > 0:
            all_dots = dots_maxima
        elif len(dots_cc) > 0:
            all_dots = dots_cc
        else:
            all_dots = np.array([]).reshape(0, 2)
        
        # Remove duplicate dots that are too close
        if len(all_dots) > 1:
            distances = squareform(pdist(all_dots))
            np.fill_diagonal(distances, np.inf)  # Ignore self-distances
            
            # Remove dots that are too close to others
            to_remove = set()
            min_distance = 5  # Minimum distance between dots
            
            for i in range(len(all_dots)):
                if i in to_remove:
                    continue
                close_dots = np.where(distances[i] < min_distance)[0]
                for j in close_dots:
                    if j > i:  # Only remove later dots to avoid index issues
                        to_remove.add(j)
            
            if to_remove:
                keep_indices = [i for i in range(len(all_dots)) if i not in to_remove]
                all_dots = all_dots[keep_indices]
        
        self.logger.debug(f"Detected {len(all_dots)} activity dots")
        return all_dots
    
    def _cluster_dots_into_blobs(self, dot_positions: np.ndarray, image: np.ndarray) -> List[List[int]]:
        """
        Cluster nearby dots into blob groups using DBSCAN or distance-based clustering
        """
        if len(dot_positions) < 2:
            return [[0]] if len(dot_positions) == 1 else []
        
        # Use DBSCAN clustering if available, otherwise use distance-based clustering
        if HAS_SKLEARN:
            # Use DBSCAN clustering to group nearby dots
            image_diagonal = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
            eps = min(image_diagonal * 0.1, 50)  # Adaptive distance threshold
            min_samples = 1  # Allow single dots to form clusters
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dot_positions)
            labels = clustering.labels_
            
            # Group dots by cluster label
            clusters = {}
            for i, label in enumerate(labels):
                if label >= 0:  # Ignore noise points (label -1)
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(i)
            
            # Convert to list of lists
            blob_clusters = list(clusters.values())
            
            # Add isolated dots as single-dot clusters
            noise_points = [i for i, label in enumerate(labels) if label == -1]
            for noise_point in noise_points:
                blob_clusters.append([noise_point])
        else:
            # Fallback to distance-based clustering
            blob_clusters = self._distance_based_clustering(dot_positions, image)
        
        self.logger.debug(f"Clustered {len(dot_positions)} dots into {len(blob_clusters)} blob groups")
        return blob_clusters
    
    def _distance_based_clustering(self, dot_positions: np.ndarray, image: np.ndarray) -> List[List[int]]:
        """
        Fallback clustering method using distance thresholding
        """
        if len(dot_positions) == 0:
            return []
        if len(dot_positions) == 1:
            return [[0]]
        
        # Calculate pairwise distances
        distances = squareform(pdist(dot_positions))
        
        # Determine distance threshold
        image_diagonal = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
        distance_threshold = min(image_diagonal * 0.1, 50)
        
        # Simple clustering: group dots within threshold distance
        clusters = []
        used = set()
        
        for i in range(len(dot_positions)):
            if i in used:
                continue
                
            # Start new cluster
            cluster = [i]
            used.add(i)
            
            # Find all dots within threshold distance
            close_dots = np.where(distances[i] <= distance_threshold)[0]
            for j in close_dots:
                if j != i and j not in used:
                    cluster.append(j)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _create_blob_regions_from_clusters(self, image: np.ndarray, 
                                         blob_clusters: List[List[int]], 
                                         dot_positions: np.ndarray) -> List[Dict]:
        """
        Create blob regions from clustered dots
        """
        regions = []
        
        for cluster_idx, dot_indices in enumerate(blob_clusters):
            if not dot_indices:
                continue
                
            # Get positions of dots in this cluster
            cluster_dots = dot_positions[dot_indices]
            
            # Create a mask for this blob by expanding around clustered dots
            blob_mask = self._create_blob_mask_from_dots(image, cluster_dots)
            
            # Analyze the blob region
            if np.sum(blob_mask) >= self.min_blob_area:
                labeled_mask = measure.label(blob_mask)
                
                for region in measure.regionprops(labeled_mask):
                    if self.min_blob_area <= region.area <= self.max_blob_area:
                        # Calculate confidence based on dot density and intensity
                        blob_confidence = self._calculate_blob_confidence(
                            image, region, cluster_dots
                        )
                        
                        regions.append({
                            'bbox': region.bbox,
                            'centroid': region.centroid,
                            'area': region.area,
                            'method': 'clustering',
                            'confidence': blob_confidence,
                            'dot_count': len(dot_indices)
                        })
        
        return regions
    
    def _create_blob_mask_from_dots(self, image: np.ndarray, dot_positions: np.ndarray) -> np.ndarray:
        """
        Create a blob mask by expanding around clustered dots
        """
        mask = np.zeros(image.shape, dtype=bool)
        
        if len(dot_positions) == 0:
            return mask
        
        # For each dot, create a local region
        for dot_pos in dot_positions:
            y, x = int(dot_pos[0]), int(dot_pos[1])
            
            # Create expanding region around each dot based on local intensity
            radius = self._estimate_dot_influence_radius(image, y, x)
            
            # Create circular region around dot
            Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
            dist_from_dot = np.sqrt((Y - y)**2 + (X - x)**2)
            mask |= (dist_from_dot <= radius)
        
        # Connect nearby regions with morphological closing
        # Use minimal morphological operations to connect dots without destroying structure
        if len(dot_positions) > 1:
            # Calculate average distance between dots in cluster
            distances = pdist(dot_positions)
            avg_distance = np.mean(distances) if len(distances) > 0 else 10
            
            # Use closing to connect nearby regions
            closing_radius = int(min(avg_distance / 3, 10))
            if closing_radius > 0:
                mask = morphology.binary_closing(mask, morphology.disk(closing_radius))
        
        return mask
    
    def _estimate_dot_influence_radius(self, image: np.ndarray, y: int, x: int) -> float:
        """
        Estimate the influence radius of a dot based on local intensity profile
        """
        # Extract local region around dot
        radius_search = 15  # Search radius
        y_min = max(0, y - radius_search)
        y_max = min(image.shape[0], y + radius_search + 1)
        x_min = max(0, x - radius_search)
        x_max = min(image.shape[1], x + radius_search + 1)
        
        local_region = image[y_min:y_max, x_min:x_max]
        center_y = y - y_min
        center_x = x - x_min
        
        if local_region.size == 0:
            return 3.0  # Default radius
        
        # Find radius where intensity drops to background level
        center_intensity = local_region[center_y, center_x] if (
            0 <= center_y < local_region.shape[0] and 
            0 <= center_x < local_region.shape[1]
        ) else np.mean(local_region)
        
        background_level = np.percentile(local_region, 25)
        threshold = background_level + 0.3 * (center_intensity - background_level)
        
        # Find radius where intensity drops below threshold
        Y, X = np.ogrid[:local_region.shape[0], :local_region.shape[1]]
        distances = np.sqrt((Y - center_y)**2 + (X - center_x)**2)
        
        above_threshold = local_region > threshold
        if np.any(above_threshold):
            max_radius = np.max(distances[above_threshold])
            return min(max_radius, 10.0)  # Cap at reasonable size
        else:
            return 3.0  # Default radius
    
    def _calculate_blob_confidence(self, image: np.ndarray, region, dot_positions: np.ndarray) -> float:
        """
        Calculate confidence for a blob based on dots and intensity
        """
        # Extract region from image
        bbox = region.bbox
        region_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        region_mask = region.image
        
        if not np.any(region_mask):
            return 0.0
        
        # Intensity-based metrics
        mean_intensity = np.mean(region_image[region_mask])
        max_intensity = np.max(region_image[region_mask])
        
        # Shape-based metrics
        solidity = region.solidity
        extent = region.extent
        
        # Dot density metric
        dot_density = len(dot_positions) / region.area if region.area > 0 else 0
        
        # Combine metrics
        intensity_score = (mean_intensity + max_intensity) / 2
        shape_score = solidity * 0.6 + extent * 0.4
        density_score = min(1.0, dot_density * 100)  # Scale dot density
        
        confidence = intensity_score * 0.4 + shape_score * 0.3 + density_score * 0.3
        return confidence
    
    def _fallback_to_threshold_segmentation(self, image: np.ndarray) -> List[Dict]:
        """
        Fallback to threshold-based segmentation when clustering fails
        """
        self.logger.debug("Falling back to threshold-based segmentation")
        return self._multi_threshold_segmentation(image)

    def _multi_threshold_segmentation(self, image: np.ndarray) -> List[Dict]:
        """
        Use multiple threshold methods and combine results
        """
        regions_list = []
        
        # Method 1: Otsu thresholding
        try:
            otsu_thresh = filters.threshold_otsu(image)
            otsu_binary = image > otsu_thresh
            otsu_regions = self._extract_regions_from_binary(otsu_binary, "otsu")
            regions_list.extend(otsu_regions)
        except Exception as e:
            self.logger.warning(f"Otsu thresholding failed: {e}")
        
        # Method 2: Local thresholding
        try:
            local_thresh = filters.threshold_local(image, block_size=51, method='gaussian')
            local_binary = image > local_thresh
            local_regions = self._extract_regions_from_binary(local_binary, "local")
            regions_list.extend(local_regions)
        except Exception as e:
            self.logger.warning(f"Local thresholding failed: {e}")
        
        # Method 3: Multi-level Otsu
        try:
            multi_thresh = filters.threshold_multiotsu(image, classes=3)
            # Use the middle threshold to separate background from activity
            multi_binary = image > multi_thresh[0]
            multi_regions = self._extract_regions_from_binary(multi_binary, "multi_otsu")
            regions_list.extend(multi_regions)
        except Exception as e:
            self.logger.warning(f"Multi-Otsu thresholding failed: {e}")
        
        # Method 4: Low percentile-based thresholding (for low contrast)
        try:
            # Use much lower percentiles for low-contrast iQID data
            # Try multiple low thresholds
            for percentile in [25, 35, 50]:
                percentile_thresh = np.percentile(image[image > 0], percentile)
                percentile_binary = image > percentile_thresh
                percentile_regions = self._extract_regions_from_binary(percentile_binary, f"percentile_{percentile}")
                regions_list.extend(percentile_regions)
        except Exception as e:
            self.logger.warning(f"Percentile thresholding failed: {e}")
            
        # Method 5: Adaptive low-threshold (for very low contrast cases)
        try:
            # Use statistical approach for very low contrast
            mean_val = np.mean(image[image > 0])
            std_val = np.std(image[image > 0])
            # Use mean + small fraction of std as threshold
            low_thresh = mean_val + 0.1 * std_val
            low_binary = image > low_thresh
            low_regions = self._extract_regions_from_binary(low_binary, "low_adaptive")
            regions_list.extend(low_regions)
        except Exception as e:
            self.logger.warning(f"Low adaptive thresholding failed: {e}")
        
        # Combine and filter regions
        combined_regions = self._combine_overlapping_regions(regions_list)
        return combined_regions
    
    def _watershed_segmentation(self, image: np.ndarray) -> List[Dict]:
        """
        Use watershed segmentation for blob separation
        """
        # Preprocess image
        smoothed = filters.gaussian(image, sigma=1.0)
        
        # Find local maxima as seeds
        try:
            from skimage.feature import peak_local_maxima as peak_local_max
            local_maxima = peak_local_max(smoothed, min_distance=20, 
                                        threshold_abs=np.percentile(smoothed, 70))
        except ImportError:
            try:
                # Try the newer function name
                from skimage.feature import peak_local_max
                local_maxima = peak_local_max(smoothed, min_distance=20, 
                                            threshold_abs=np.percentile(smoothed, 70))
            except ImportError:
                # Use scipy.ndimage for local maxima detection
                from scipy.ndimage import maximum_filter
                neighborhood = np.ones((20, 20))
                local_maxima_mask = maximum_filter(smoothed, footprint=neighborhood) == smoothed
                threshold = np.percentile(smoothed, 70)
                local_maxima_mask &= (smoothed > threshold)
                local_maxima = np.where(local_maxima_mask)
            
        if len(local_maxima[0]) == 0:
            self.logger.warning("No local maxima found for watershed")
            return []
        
        # Create markers for watershed
        markers = np.zeros_like(image, dtype=int)
        for i, (y, x) in enumerate(zip(local_maxima[0], local_maxima[1])):
            markers[y, x] = i + 1
        
        # Apply watershed
        labels = segmentation.watershed(-smoothed, markers, mask=smoothed > 0)
        
        # Extract regions
        regions = []
        for region in measure.regionprops(labels):
            if self.min_blob_area <= region.area <= self.max_blob_area:
                bbox = region.bbox
                regions.append({
                    'bbox': bbox,
                    'centroid': region.centroid,
                    'area': region.area,
                    'method': 'watershed',
                    'confidence': self._calculate_region_confidence(image, region)
                })
        
        return regions
    
    def _adaptive_threshold_segmentation(self, image: np.ndarray) -> List[Dict]:
        """
        Use adaptive thresholding with morphological operations
        """
        # Convert to uint8 for OpenCV
        image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Adaptive thresholding
        adaptive_binary = cv2.adaptiveThreshold(
            image_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 2
        )
        
        # Morphological operations: ONLY for spike noise removal
        if self.preserve_blobs:
            # Minimal spike noise removal - only remove isolated single pixels
            self.logger.debug("Using minimal spike noise removal")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            # Only opening with very small kernel to remove single pixel noise
            adaptive_binary = cv2.morphologyEx(adaptive_binary, cv2.MORPH_OPEN, kernel)
        else:
            # Standard morphological operations
            self.logger.debug("Using standard morphological operations")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            adaptive_binary = cv2.morphologyEx(adaptive_binary, cv2.MORPH_OPEN, kernel)
            adaptive_binary = cv2.morphologyEx(adaptive_binary, cv2.MORPH_CLOSE, kernel)
        
        return self._extract_regions_from_binary(adaptive_binary > 0, "adaptive")
    
    def _combined_segmentation(self, image: np.ndarray) -> List[Dict]:
        """
        Combine multiple segmentation methods for robust results
        """
        all_regions = []
        
        # Get results from multiple methods
        multi_regions = self._multi_threshold_segmentation(image)
        watershed_regions = self._watershed_segmentation(image)
        adaptive_regions = self._adaptive_threshold_segmentation(image)
        
        all_regions.extend(multi_regions)
        all_regions.extend(watershed_regions)
        all_regions.extend(adaptive_regions)
        
        # Combine overlapping regions and rank by confidence
        combined_regions = self._combine_overlapping_regions(all_regions)
        
        # Sort by confidence and filter
        combined_regions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return combined_regions
    
    def _extract_regions_from_binary(self, binary_image: np.ndarray, method: str) -> List[Dict]:
        """
        Extract region properties from binary image
        """
        # Minimal cleanup: ONLY remove spike noise (single pixels)
        if self.preserve_blobs:
            # Only remove single pixel noise, preserve all actual blobs
            min_cleanup_size = 2  # Only remove 1-pixel noise
            self.logger.debug(f"Using spike noise removal only with min_size={min_cleanup_size}")
            cleaned = morphology.remove_small_objects(binary_image, min_size=min_cleanup_size)
            # No hole filling to preserve distributed dot patterns
        else:
            # Standard cleanup
            self.logger.debug(f"Using standard cleanup with min_size={self.min_blob_area}")
            cleaned = morphology.remove_small_objects(binary_image, min_size=self.min_blob_area)
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=self.min_blob_area)
        
        # Label connected components
        labeled = measure.label(cleaned)
        regions = []
        
        for region in measure.regionprops(labeled):
            if self.min_blob_area <= region.area <= self.max_blob_area:
                bbox = region.bbox
                regions.append({
                    'bbox': bbox,
                    'centroid': region.centroid,
                    'area': region.area,
                    'method': method,
                    'confidence': self._calculate_region_confidence_from_props(region)
                })
        
        return regions
    
    def _calculate_region_confidence(self, image: np.ndarray, region) -> float:
        """
        Calculate confidence score for a region
        """
        # Extract region pixels
        bbox = region.bbox
        region_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        region_mask = region.image
        
        # Calculate metrics
        mean_intensity = np.mean(region_image[region_mask])
        std_intensity = np.std(region_image[region_mask])
        
        # Shape metrics
        solidity = region.solidity
        extent = region.extent
        
        # Combine metrics (higher is better)
        confidence = (mean_intensity / 255.0) * 0.4 + \
                    solidity * 0.3 + \
                    extent * 0.2 + \
                    min(1.0, std_intensity / 50.0) * 0.1
        
        return confidence
    
    def _calculate_region_confidence_from_props(self, region) -> float:
        """
        Calculate confidence from region properties only
        """
        solidity = region.solidity
        extent = region.extent
        area = region.area
        
        # Normalize area to 0-1 range
        area_score = min(1.0, area / 1000.0)
        
        confidence = solidity * 0.5 + extent * 0.3 + area_score * 0.2
        return confidence
    
    def _combine_overlapping_regions(self, regions: List[Dict], 
                                   overlap_threshold: float = 0.3) -> List[Dict]:
        """
        Combine overlapping regions and keep the best ones
        """
        if not regions:
            return []
        
        # Sort by confidence
        regions_sorted = sorted(regions, key=lambda x: x.get('confidence', 0), reverse=True)
        
        combined = []
        used = set()
        
        for i, region in enumerate(regions_sorted):
            if i in used:
                continue
                
            # Find overlapping regions
            overlapping = [i]
            for j, other_region in enumerate(regions_sorted[i+1:], i+1):
                if j in used:
                    continue
                    
                if self._regions_overlap(region, other_region, overlap_threshold):
                    overlapping.append(j)
            
            # Mark as used
            used.update(overlapping)
            
            # If multiple overlapping regions, combine them or pick the best
            if len(overlapping) == 1:
                combined.append(region)
            else:
                # Pick the one with highest confidence
                best_region = max([regions_sorted[idx] for idx in overlapping], 
                                key=lambda x: x.get('confidence', 0))
                combined.append(best_region)
        
        return combined
    
    def _regions_overlap(self, region1: Dict, region2: Dict, threshold: float) -> bool:
        """
        Check if two regions overlap by more than threshold
        """
        bbox1 = region1['bbox']
        bbox2 = region2['bbox']
        
        # Calculate intersection
        y1_max = max(bbox1[0], bbox2[0])
        x1_max = max(bbox1[1], bbox2[1])
        y2_min = min(bbox1[2], bbox2[2])
        x2_min = min(bbox1[3], bbox2[3])
        
        if y1_max >= y2_min or x1_max >= x2_min:
            return False
        
        intersection_area = (y2_min - y1_max) * (x2_min - x1_max)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - intersection_area
        
        # Calculate overlap ratio
        overlap_ratio = intersection_area / union_area if union_area > 0 else 0
        
        return overlap_ratio > threshold
    
    def estimate_grid_from_regions(self, regions: List[Dict], 
                                  image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Estimate the grid arrangement from detected regions
        """
        if not regions:
            return (3, 3)  # Default fallback
        
        # Extract centroids
        centroids = [region['centroid'] for region in regions]
        y_coords = [c[0] for c in centroids]
        x_coords = [c[1] for c in centroids]
        
        # Cluster coordinates to find grid structure
        y_clusters = self._cluster_coordinates(y_coords, image_shape[0])
        x_clusters = self._cluster_coordinates(x_coords, image_shape[1])
        
        rows = len(y_clusters)
        cols = len(x_clusters)
        
        self.logger.info(f"Estimated grid: {rows}x{cols} from {len(regions)} regions")
        
        return (rows, cols)
    
    def _cluster_coordinates(self, coords: List[float], dimension: int) -> List[List[float]]:
        """
        Cluster coordinates to find grid lines
        """
        if not coords:
            return []
        
        coords_sorted = sorted(coords)
        clusters = []
        current_cluster = [coords_sorted[0]]
        
        # Group coordinates that are close together
        threshold = dimension * 0.1  # 10% of dimension
        
        for coord in coords_sorted[1:]:
            if coord - current_cluster[-1] < threshold:
                current_cluster.append(coord)
            else:
                clusters.append(current_cluster)
                current_cluster = [coord]
        
        clusters.append(current_cluster)
        
        return clusters


class EnhancedRawImageSplitter:
    """
    Enhanced raw image splitter that uses adaptive segmentation
    """
    
    def __init__(self, preserve_blobs: bool = True, use_clustering: bool = True):
        """Initialize the enhanced splitter"""
        self.segmenter = AdaptiveIQIDSegmenter(
            preserve_blobs=preserve_blobs,
            use_clustering=use_clustering
        )
        self.logger = logging.getLogger(__name__)
    
    def split_image_adaptive(self, raw_tiff_path: str, output_dir: str, 
                           ground_truth_count: Optional[int] = None) -> List[str]:
        """
        Split image using adaptive segmentation
        
        Args:
            raw_tiff_path: Path to raw TIFF file
            output_dir: Output directory for segments
            ground_truth_count: Expected number of segments (if known)
            
        Returns:
            List of output file paths
        """
        import skimage.io
        
        # Load image
        image = skimage.io.imread(raw_tiff_path)
        
        # Handle different image formats
        if image.ndim != 2:
            if image.ndim == 3 and image.shape[2] == 1:
                image = image.squeeze(axis=2)
            elif image.ndim == 3 and image.shape[0] == 1:
                image = image.squeeze(axis=0)
            else:
                raise ValueError(f"Cannot handle image shape: {image.shape}")
        
        # Detect activity regions using clustering for low contrast distributed dots
        regions = self.segmenter.detect_activity_regions(image, method="clustering")
        
        if ground_truth_count and len(regions) != ground_truth_count:
            self.logger.warning(f"Detected {len(regions)} regions, expected {ground_truth_count}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract and save segments
        output_paths = []
        for i, region in enumerate(regions):
            bbox = region['bbox']
            segment = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            
            # Save segment
            segment_path = output_path / f"segment_{i:02d}.tif"
            skimage.io.imsave(str(segment_path), segment, check_contrast=False)
            output_paths.append(str(segment_path))
        
        self.logger.info(f"Extracted {len(output_paths)} segments to {output_dir}")
        
        return output_paths
