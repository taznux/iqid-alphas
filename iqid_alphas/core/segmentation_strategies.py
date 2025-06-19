"""
Segmentation Strategy Implementations for TissueSegmentation
"""
import numpy as np
from skimage import filters, morphology, measure, segmentation, feature
from sklearn.cluster import DBSCAN
from skimage.morphology import disk


def clustering_based_segmentation(image, min_blob_area, max_blob_area, preserve_blobs, use_clustering, has_sklearn, logger=None):
    threshold = filters.threshold_otsu(image)
    significant_pixels = image > threshold
    if not np.any(significant_pixels):
        return []
    y_coords, x_coords = np.where(significant_pixels)
    coords = np.column_stack([x_coords, y_coords])
    if len(coords) == 0:
        return []
    regions = []
    if has_sklearn and use_clustering:
        clustering = DBSCAN(eps=10, min_samples=max(min_blob_area // 4, 1))
        cluster_labels = clustering.fit_predict(coords)
        for label in np.unique(cluster_labels):
            if label == -1:
                continue
            cluster_coords = coords[cluster_labels == label]
            cluster_mask = np.zeros_like(image, dtype=bool)
            cluster_mask[cluster_coords[:, 1], cluster_coords[:, 0]] = True
            if preserve_blobs:
                cluster_mask = morphology.binary_closing(cluster_mask, disk(3))
            else:
                cluster_mask = morphology.binary_closing(cluster_mask, disk(5))
                cluster_mask = morphology.binary_opening(cluster_mask, disk(3))
            region_area = np.sum(cluster_mask)
            if min_blob_area <= region_area <= max_blob_area:
                regions.append({
                    'mask': cluster_mask,
                    'area': region_area,
                    'centroid': np.mean(cluster_coords, axis=0),
                    'intensity': np.mean(image[cluster_mask])
                })
    else:
        labeled_regions = measure.label(significant_pixels)
        for region in measure.regionprops(labeled_regions):
            if min_blob_area <= region.area <= max_blob_area:
                mask = labeled_regions == region.label
                regions.append({
                    'mask': mask,
                    'area': region.area,
                    'centroid': np.array(region.centroid),
                    'intensity': np.mean(image[mask])
                })
    return regions

def multi_threshold_segmentation(image, min_blob_area, max_blob_area, HAS_SKIMAGE):
    """
    Multi-threshold segmentation using several thresholding strategies.
    """
    regions = []
    if not HAS_SKIMAGE:
        return simple_threshold_segmentation(image, min_blob_area, max_blob_area)
    thresholds = [
        filters.threshold_otsu(image),
        filters.threshold_yen(image),
        np.percentile(image[image > 0], 75),
        np.percentile(image[image > 0], 90)
    ]
    for i, threshold in enumerate(thresholds):
        binary_mask = image > threshold
        labeled_regions = measure.label(binary_mask)
        for region in measure.regionprops(labeled_regions):
            if min_blob_area <= region.area <= max_blob_area:
                mask = labeled_regions == region.label
                regions.append({
                    'mask': mask,
                    'area': region.area,
                    'centroid': np.array(region.centroid),
                    'intensity': np.mean(image[mask]),
                    'threshold_level': i
                })
    return regions

def watershed_segmentation(image, min_blob_area, max_blob_area, HAS_SKIMAGE):
    """
    Watershed-based segmentation for separating touching tissues.
    """
    if not HAS_SKIMAGE:
        return clustering_based_segmentation(image, min_blob_area, max_blob_area, True, True, False)
    smoothed = filters.gaussian(image, sigma=2.0)
    local_maxima = feature.peak_local_max(smoothed, min_distance=20, threshold_abs=np.percentile(smoothed, 80))
    if len(local_maxima) == 0:
        return clustering_based_segmentation(image, min_blob_area, max_blob_area, True, True, False)
    markers = np.zeros_like(image, dtype=int)
    for i, (y, x) in enumerate(local_maxima):
        markers[y, x] = i + 1
    labels = segmentation.watershed(-smoothed, markers, mask=smoothed > filters.threshold_otsu(smoothed))
    regions = []
    for region in measure.regionprops(labels):
        if min_blob_area <= region.area <= max_blob_area:
            mask = labels == region.label
            regions.append({
                'mask': mask,
                'area': region.area,
                'centroid': np.array(region.centroid),
                'intensity': np.mean(image[mask])
            })
    return regions

def combined_segmentation(image, min_blob_area, max_blob_area, preserve_blobs, use_clustering, HAS_SKIMAGE, HAS_SKLEARN, logger=None):
    """
    Combine clustering, threshold, and watershed segmentation results.
    """
    all_regions = []
    clustering_regions = clustering_based_segmentation(image, min_blob_area, max_blob_area, preserve_blobs, use_clustering, HAS_SKLEARN, logger=logger)
    threshold_regions = multi_threshold_segmentation(image, min_blob_area, max_blob_area, HAS_SKIMAGE)
    watershed_regions = watershed_segmentation(image, min_blob_area, max_blob_area, HAS_SKIMAGE)
    for regions in [clustering_regions, threshold_regions, watershed_regions]:
        all_regions.extend(regions)
    final_regions = []
    for region in all_regions:
        overlaps = False
        for existing in final_regions:
            overlap = np.sum(region['mask'] & existing['mask'])
            if overlap > 0.3 * min(region['area'], existing['area']):
                overlaps = True
                if region['intensity'] > existing['intensity']:
                    final_regions.remove(existing)
                    final_regions.append(region)
                break
        if not overlaps:
            final_regions.append(region)
    return final_regions

def simple_threshold_segmentation(image, min_blob_area, max_blob_area):
    """
    Simple threshold-based segmentation fallback.
    """
    threshold = np.mean(image) + 2 * np.std(image)
    binary_mask = image > threshold
    try:
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(binary_mask)
        regions = []
        for i in range(1, num_features + 1):
            mask = labeled_array == i
            area = np.sum(mask)
            if min_blob_area <= area <= max_blob_area:
                y_coords, x_coords = np.where(mask)
                centroid = np.array([np.mean(x_coords), np.mean(y_coords)])
                regions.append({
                    'mask': mask,
                    'area': area,
                    'centroid': centroid,
                    'intensity': np.mean(image[mask])
                })
        return regions
    except Exception:
        return []
