#!/usr/bin/env python3
"""
Visualization Utilities for iQID Processing Evaluation

Provides 16-bit to 8-bit conversion and color mapping for user-friendly
visualization of processing steps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import skimage.io
import skimage.exposure
from typing import Dict, List, Optional, Tuple
import cv2
from scipy import ndimage
from skimage import filters, morphology, segmentation, measure, feature
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import disk, remove_small_objects, binary_opening, binary_closing
from skimage.segmentation import watershed, clear_border
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans


class IQIDVisualizer:
    """
    Handles visualization of 16-bit iQID images for user display
    """
    
    def __init__(self, output_dir: str = "evaluation/outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color maps for different processing stages
        self.colormaps = {
            'raw': 'viridis',      # Blue-green for raw data
            'segmented': 'plasma',  # Purple-pink for segmented
            'aligned': 'hot',      # Red-yellow for aligned
            'comparison': 'RdYlBu' # Red-blue for comparison
        }
        
        # Grid overlay colors
        self.grid_colors = {
            'detected': (0, 255, 0),    # Green for detected blobs
            'failed': (255, 0, 0),      # Red for failed detection
            'grid': (255, 255, 0),      # Yellow for grid lines
            'expected': (0, 255, 255)   # Cyan for expected positions
        }
    
    def convert_16bit_to_8bit(self, image_16bit: np.ndarray, 
                             percentile_range: Tuple[float, float] = (1, 99)) -> np.ndarray:
        """
        Convert 16-bit image to 8-bit with contrast enhancement
        
        Args:
            image_16bit: Input 16-bit image
            percentile_range: Percentile range for contrast stretching
            
        Returns:
            8-bit converted image
        """
        # Handle different input types and ensure proper data type
        if image_16bit.dtype == np.float64:
            # Convert from float64 to uint16 range
            image_16bit = ((image_16bit - image_16bit.min()) / 
                          (image_16bit.max() - image_16bit.min()) * 65535).astype(np.uint16)
        elif image_16bit.dtype == np.float32:
            # Convert from float32 to uint16 range
            image_16bit = ((image_16bit - image_16bit.min()) / 
                          (image_16bit.max() - image_16bit.min()) * 65535).astype(np.uint16)
        elif image_16bit.dtype != np.uint16:
            # Convert other types to uint16
            if image_16bit.max() <= 255:
                # Likely 8-bit data, scale up
                image_16bit = (image_16bit * 257).astype(np.uint16)  # 257 = 65535/255
            else:
                # Scale to 16-bit range
                image_16bit = ((image_16bit - image_16bit.min()) / 
                              (image_16bit.max() - image_16bit.min()) * 65535).astype(np.uint16)
        
        # Apply percentile-based contrast stretching
        p_low, p_high = np.percentile(image_16bit, percentile_range)
        
        # Stretch contrast
        image_stretched = np.clip((image_16bit - p_low) / (p_high - p_low) * 255, 0, 255)
        
        return image_stretched.astype(np.uint8)
    
    def apply_colormap(self, image_8bit: np.ndarray, colormap: str = 'viridis') -> np.ndarray:
        """
        Apply colormap to 8-bit grayscale image
        
        Args:
            image_8bit: 8-bit grayscale image
            colormap: Matplotlib colormap name
            
        Returns:
            RGB colored image
        """
        # Normalize to 0-1 range
        normalized = image_8bit.astype(np.float32) / 255.0
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colored = cmap(normalized)
        
        # Convert back to 8-bit RGB
        rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)
        
        return rgb_image
    
    def robust_blob_segmentation(self, image: np.ndarray, 
                                min_blob_size: int = 1000,
                                expected_blobs: int = 12,
                                visualization_output: Optional[str] = None) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Robust blob segmentation from noisy dark background
        
        Args:
            image: Input 16-bit or 8-bit image
            min_blob_size: Minimum blob size in pixels
            expected_blobs: Expected number of blobs (for grid arrangement)
            visualization_output: Optional path to save segmentation visualization
            
        Returns:
            Tuple of (blob_boxes, segmentation_mask)
            blob_boxes: List of (x, y, width, height) for each detected blob
            segmentation_mask: Binary mask showing detected regions
        """
        # Convert to 8-bit if needed
        if image.dtype == np.uint16:
            image_8bit = self.convert_16bit_to_8bit(image)
        else:
            image_8bit = image.copy()
        
        # Step 1: Pre-processing - Noise reduction
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image_8bit, (3, 3), 1.0)
        
        # Step 2: Advanced thresholding
        # Try multiple thresholding approaches and combine
        
        # Global Otsu threshold
        otsu_thresh = threshold_otsu(blurred)
        binary_otsu = blurred > otsu_thresh
        
        # Local adaptive threshold for handling uneven illumination
        local_thresh = threshold_local(blurred, block_size=35, offset=10)
        binary_local = blurred > local_thresh
        
        # Intensity-based clustering (K-means) - downsample for efficiency on large images
        if blurred.size > 1000000:  # If image is larger than 1MP, downsample for clustering
            # Downsample by factor of 4 for clustering
            small_blurred = blurred[::4, ::4]
            pixels = small_blurred.reshape(-1, 1)
        else:
            pixels = blurred.reshape(-1, 1)
            
        # Use fewer clusters and iterations for speed
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=3, max_iter=100)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.flatten()
        
        # Find the cluster with highest intensity (foreground)
        fg_cluster = np.argmax(centers)
        if blurred.size > 1000000:
            # Upscale the result back to original size
            binary_kmeans_small = (labels == fg_cluster).reshape(small_blurred.shape)
            binary_kmeans = np.repeat(np.repeat(binary_kmeans_small, 4, axis=0), 4, axis=1)
            # Crop to match original size
            binary_kmeans = binary_kmeans[:blurred.shape[0], :blurred.shape[1]]
        else:
            binary_kmeans = (labels == fg_cluster).reshape(blurred.shape)
        
        # Combine thresholding results with voting
        # At least 2 of 3 methods should agree
        combined_binary = (binary_otsu.astype(int) + 
                          binary_local.astype(int) + 
                          binary_kmeans.astype(int)) >= 2
        
        # Step 3: Morphological operations
        # Remove small noise
        cleaned = remove_small_objects(combined_binary, min_size=min_blob_size//4)
        
        # Fill holes within objects
        filled = ndimage.binary_fill_holes(cleaned)
        
        # Smooth boundaries
        smoothed = binary_opening(filled, disk(2))
        smoothed = binary_closing(smoothed, disk(3))
        
        # Step 4: Watershed segmentation for touching objects
        # Distance transform
        distance = ndimage.distance_transform_edt(smoothed)
        
        # Find local maxima for seeds
        local_maxima = peak_local_max(distance, 
                                     min_distance=20,
                                     threshold_abs=distance.max()*0.3)
        
        # Create markers for watershed
        markers = np.zeros_like(smoothed, dtype=int)
        if len(local_maxima) > 0:
            markers[local_maxima[:, 0], local_maxima[:, 1]] = np.arange(1, len(local_maxima) + 1)
        
        # Apply watershed
        if markers.max() > 0:
            segmented = watershed(-distance, markers, mask=smoothed)
        else:
            segmented = smoothed.astype(int)
        
        # Step 5: Post-processing and validation
        # Remove objects touching border
        cleared = clear_border(segmented)
        
        # Filter by size and shape
        regions = measure.regionprops(cleared)
        
        valid_blobs = []
        final_mask = np.zeros_like(cleared, dtype=bool)
        
        for region in regions:
            # Size filter
            if region.area < min_blob_size:
                continue
            
            # Shape filter - reject very elongated objects
            if region.major_axis_length > 0:
                aspect_ratio = region.minor_axis_length / region.major_axis_length
                if aspect_ratio < 0.3:  # Too elongated
                    continue
            
            # Solidity filter - reject very irregular shapes
            if region.solidity < 0.7:
                continue
            
            # Store valid blob
            y, x, y2, x2 = region.bbox
            valid_blobs.append((x, y, x2-x, y2-y))
            
            # Add to final mask
            final_mask[cleared == region.label] = True
        
        # Step 6: Generate visualization if requested
        if visualization_output:
            self._create_segmentation_visualization(
                image_8bit, combined_binary, smoothed, final_mask,
                valid_blobs, visualization_output
            )
        
        return valid_blobs, final_mask
    
    def _create_segmentation_visualization(self, original: np.ndarray,
                                         initial_binary: np.ndarray,
                                         cleaned_binary: np.ndarray,
                                         final_mask: np.ndarray,
                                         detected_blobs: List[Tuple[int, int, int, int]],
                                         output_path: str):
        """Create step-by-step segmentation visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Ensure original is uint8 for OpenCV operations
        if original.dtype != np.uint8:
            if original.dtype == np.uint16:
                original_8bit = self.convert_16bit_to_8bit(original)
            else:
                # Handle float types
                original_norm = (original - original.min()) / (original.max() - original.min())
                original_8bit = (original_norm * 255).astype(np.uint8)
        else:
            original_8bit = original.copy()
        
        # Original image
        axes[0, 0].imshow(original_8bit, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Initial binary
        axes[0, 1].imshow(initial_binary, cmap='gray')
        axes[0, 1].set_title('Combined Thresholding')
        axes[0, 1].axis('off')
        
        # Cleaned binary
        axes[0, 2].imshow(cleaned_binary, cmap='gray')
        axes[0, 2].set_title('Morphological Cleaning')
        axes[0, 2].axis('off')
        
        # Final segmentation
        axes[1, 0].imshow(final_mask, cmap='gray')
        axes[1, 0].set_title('Final Segmentation')
        axes[1, 0].axis('off')
        
        # Overlay on original - ensure proper data type conversion
        overlay = original_8bit.copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        for x, y, w, h in detected_blobs:
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(overlay, f'{w}x{h}', (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title(f'Detected Blobs ({len(detected_blobs)})')
        axes[1, 1].axis('off')
        
        # Statistics
        axes[1, 2].axis('off')
        stats_text = f"""Segmentation Statistics:
        
Total blobs detected: {len(detected_blobs)}
        
Blob sizes:
{[f"{w}x{h}" for x, y, w, h in detected_blobs[:5]]}
{"..." if len(detected_blobs) > 5 else ""}

Image size: {original_8bit.shape}
Total foreground pixels: {np.sum(final_mask)}
Foreground percentage: {np.sum(final_mask)/original_8bit.size*100:.1f}%"""
        
        axes[1, 2].text(0.1, 0.9, stats_text, fontsize=10, va='top', ha='left',
                        transform=axes[1, 2].transAxes, family='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_grid_overlay_visualization(self, image: np.ndarray,
                                        expected_grid_size: Tuple[int, int],
                                        detected_blobs: Optional[List[Tuple[int, int, int, int]]] = None,
                                        sample_id: str = "unknown",
                                        error_message: Optional[str] = None) -> str:
        """
        Create visualization of raw image with segmentation grid overlay
        
        Args:
            image: Raw 16-bit image
            expected_grid_size: Expected (rows, cols) grid arrangement
            detected_blobs: Optional list of detected blob boxes
            sample_id: Sample identifier
            error_message: Optional error message to display
            
        Returns:
            Path to saved visualization
        """
        # Convert to 8-bit for visualization
        image_8bit = self.convert_16bit_to_8bit(image)
        
        # If no blobs provided, try to detect them (but skip for very large images to avoid timeout)
        if detected_blobs is None and image.size < 50000000:  # Skip segmentation for very large images (>50MP)
            try:
                segmentation_viz_path = self.output_dir / f"{sample_id}_segmentation_steps.png"
                detected_blobs, _ = self.robust_blob_segmentation(
                    image, visualization_output=str(segmentation_viz_path)
                )
            except Exception as e:
                print(f"Warning: Blob detection failed for {sample_id}: {e}")
                detected_blobs = []
        elif detected_blobs is None:
            print(f"Skipping blob detection for large image {sample_id} ({image.shape})")
            detected_blobs = []
        
        # Create RGB overlay - ensure 8-bit input for OpenCV
        if image_8bit.dtype != np.uint8:
            image_8bit = image_8bit.astype(np.uint8)
        
        overlay = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
        height, width = image_8bit.shape
        
        # Draw expected grid with tissue type annotations
        rows, cols = expected_grid_size
        if rows > 0 and cols > 0:
            # Calculate grid spacing
            row_spacing = height // rows
            col_spacing = width // cols
            
            # Draw grid lines
            for i in range(1, rows):
                y = i * row_spacing
                cv2.line(overlay, (0, y), (width, y), self.grid_colors['grid'], 2)
            
            for j in range(1, cols):
                x = j * col_spacing
                cv2.line(overlay, (x, 0), (x, height), self.grid_colors['grid'], 2)
            
            # Draw expected blob centers with tissue type labels
            for i in range(rows):
                for j in range(cols):
                    center_x = j * col_spacing + col_spacing // 2
                    center_y = i * row_spacing + row_spacing // 2
                    
                    # Determine tissue type based on position
                    # Based on user description: tumors on top (row 0), kidneys below
                    tissue_type = "Unknown"
                    tissue_color = self.grid_colors['expected']
                    
                    if i == 0 and cols == 4:  # First row, 4 columns = tumor slices
                        tissue_type = f"T{j+1}"  # T1, T2, T3, T4
                        tissue_color = (255, 100, 100)  # Light red for tumors
                    elif i > 0 and cols == 4:  # Lower rows = kidney pairs
                        kidney_side = "L" if j < 2 else "R"  # Left or Right kidney
                        slice_num = (i-1) * 2 + (j % 2) + 1
                        tissue_type = f"K{kidney_side}{slice_num}"
                        tissue_color = (100, 255, 100)  # Light green for kidneys
                    elif cols != 4:  # Other arrangements
                        sample_num = i * cols + j + 1
                        tissue_type = f"S{sample_num}"
                    
                    # Draw center circle
                    cv2.circle(overlay, (center_x, center_y), 8, tissue_color, 3)
                    
                    # Add tissue type label
                    label_pos = (center_x - 15, center_y + 25)
                    cv2.putText(overlay, tissue_type, label_pos,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, tissue_color, 2)
        
        # Draw detected blobs
        if detected_blobs:
            for i, (x, y, w, h) in enumerate(detected_blobs):
                # Draw bounding box
                cv2.rectangle(overlay, (x, y), (x+w, y+h), 
                            self.grid_colors['detected'], 3)
                
                # Draw center point
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(overlay, (center_x, center_y), 5, 
                         self.grid_colors['detected'], -1)
                
                # Label with blob number and size
                label = f"{i+1}: {w}x{h}"
                cv2.putText(overlay, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           self.grid_colors['detected'], 2)
        
        # Create visualization figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        axes[0].imshow(image_8bit, cmap='gray')
        axes[0].set_title(f'Original Raw Image\n{image.shape} ({image.dtype})')
        axes[0].axis('off')
        
        # Overlay with grid and detections
        axes[1].imshow(overlay)
        title = f'Grid Overlay Analysis\n'
        title += f'Expected: {expected_grid_size[0]}x{expected_grid_size[1]} grid'
        if detected_blobs:
            title += f'\nDetected: {len(detected_blobs)} blobs'
        else:
            title += f'\nDetected: 0 blobs'
        
        axes[1].set_title(title)
        axes[1].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=np.array(self.grid_colors['grid'])/255, label='Expected Grid'),
            Patch(facecolor=np.array(self.grid_colors['expected'])/255, label='Expected Centers'),
            Patch(facecolor=np.array(self.grid_colors['detected'])/255, label='Detected Blobs')
        ]
        axes[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Add error message if provided
        if error_message:
            fig.suptitle(f"{sample_id} - Grid Analysis (ERROR)\n{error_message}", 
                        fontsize=14, fontweight='bold', color='red')
        else:
            fig.suptitle(f"{sample_id} - Grid Analysis", 
                        fontsize=14, fontweight='bold')
        
        # Add analysis text with tissue type breakdown
        expected_total = expected_grid_size[0] * expected_grid_size[1]
        detected_total = len(detected_blobs) if detected_blobs else 0
        
        # Estimate tissue breakdown based on grid layout
        tissue_breakdown = ""
        if expected_grid_size[1] == 4:  # Standard 4-column layout
            tumor_count = 4 if expected_grid_size[0] > 0 else 0
            kidney_count = expected_total - tumor_count
            tissue_breakdown = f"""
• Expected tissues:
  - Tumor slices: {tumor_count} (row 1)
  - Kidney slices: {kidney_count} (rows 2-{expected_grid_size[0]})
  - Left kidney: ~{kidney_count//2} slices
  - Right kidney: ~{kidney_count//2} slices"""
        else:
            tissue_breakdown = f"\n• Mixed tissue arrangement"
        
        analysis_text = f"""Analysis Summary:
• Image size: {width} x {height}
• Expected grid: {expected_grid_size[0]} rows × {expected_grid_size[1]} cols
• Expected samples: {expected_total}
• Detected samples: {detected_total}
• Grid cell size: ~{col_spacing if expected_grid_size[1] > 0 else 'N/A'} × {row_spacing if expected_grid_size[0] > 0 else 'N/A'} pixels
• Detection rate: {detected_total/expected_total*100 if expected_total > 0 else 0:.1f}%{tissue_breakdown}

Legend:
• Yellow lines: Expected grid
• Cyan circles: Expected positions  
• Green boxes: Detected samples
• T1-T4: Tumor slices (higher intensity)
• KL/KR: Left/Right kidney slices"""
        
        plt.figtext(0.02, 0.02, analysis_text, fontsize=9, family='monospace',
                   verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        grid_file = self.output_dir / f"{sample_id}_grid_overlay.png"
        plt.savefig(grid_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(grid_file)
    
    def create_processing_visualization(self, sample_id: str, stage: str,
                                      images: List[np.ndarray], 
                                      filenames: List[str]) -> str:
        """
        Create visualization grid for a processing stage
        
        Args:
            sample_id: Sample identifier
            stage: Processing stage (raw, segmented, aligned)
            images: List of 16-bit images
            filenames: List of corresponding filenames
            
        Returns:
            Path to saved visualization
        """
        if not images:
            return None
        
        # Determine grid size
        num_images = len(images)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Process each image
        colormap = self.colormaps.get(stage, 'viridis')
        
        for i, (img, filename) in enumerate(zip(images, filenames)):
            if i >= len(axes):
                break
                
            # Convert to 8-bit and apply colormap
            img_8bit = self.convert_16bit_to_8bit(img)
            img_colored = self.apply_colormap(img_8bit, colormap)
            
            # Display
            axes[i].imshow(img_colored)
            axes[i].set_title(f"{filename}\n{img.shape} ({img.dtype})", fontsize=10)
            axes[i].axis('off')
            
            # Add intensity histogram as inset
            ax_hist = axes[i].inset_axes([0.7, 0.7, 0.28, 0.28])
            ax_hist.hist(img_8bit.flatten(), bins=50, alpha=0.7, color='white')
            ax_hist.set_xticks([])
            ax_hist.set_yticks([])
        
        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        # Add overall title
        fig.suptitle(f"{sample_id} - {stage.upper()} Stage", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / f"{sample_id}_{stage}_visualization.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(viz_file)
    
    def create_comparison_visualization(self, sample_id: str, stage: str,
                                      reference_images: List[np.ndarray],
                                      automated_images: List[np.ndarray],
                                      ref_filenames: List[str],
                                      auto_filenames: List[str]) -> str:
        """
        Create side-by-side comparison visualization
        
        Args:
            sample_id: Sample identifier
            stage: Processing stage
            reference_images: Reference (manual) images
            automated_images: Automated processing images
            ref_filenames: Reference filenames
            auto_filenames: Automated filenames
            
        Returns:
            Path to saved comparison visualization
        """
        num_pairs = min(len(reference_images), len(automated_images))
        if num_pairs == 0:
            return None
        
        # Limit to first 6 pairs for readability
        num_pairs = min(num_pairs, 6)
        
        # Create figure with 3 columns: reference, automated, difference
        fig, axes = plt.subplots(num_pairs, 3, figsize=(12, num_pairs * 3))
        if num_pairs == 1:
            axes = axes.reshape(1, -1)
        
        colormap = self.colormaps.get(stage, 'viridis')
        
        for i in range(num_pairs):
            ref_img = reference_images[i]
            auto_img = automated_images[i]
            
            # Convert to 8-bit
            ref_8bit = self.convert_16bit_to_8bit(ref_img)
            auto_8bit = self.convert_16bit_to_8bit(auto_img)
            
            # Apply colormaps
            ref_colored = self.apply_colormap(ref_8bit, colormap)
            auto_colored = self.apply_colormap(auto_8bit, colormap)
            
            # Calculate difference (if same shape)
            if ref_8bit.shape == auto_8bit.shape:
                diff = np.abs(ref_8bit.astype(np.float32) - auto_8bit.astype(np.float32))
                diff_colored = self.apply_colormap(diff.astype(np.uint8), 'RdBu')
            else:
                # Create a placeholder for shape mismatch
                diff_colored = np.zeros((max(ref_8bit.shape[0], auto_8bit.shape[0]),
                                       max(ref_8bit.shape[1], auto_8bit.shape[1]), 3), dtype=np.uint8)
                diff_colored[:, :, 0] = 255  # Red for shape mismatch
            
            # Display reference
            axes[i, 0].imshow(ref_colored)
            axes[i, 0].set_title(f"Reference\n{ref_filenames[i]}", fontsize=10)
            axes[i, 0].axis('off')
            
            # Display automated
            axes[i, 1].imshow(auto_colored)
            axes[i, 1].set_title(f"Automated\n{auto_filenames[i]}", fontsize=10)
            axes[i, 1].axis('off')
            
            # Display difference
            axes[i, 2].imshow(diff_colored)
            if ref_8bit.shape == auto_8bit.shape:
                mse = np.mean((ref_8bit.astype(np.float32) - auto_8bit.astype(np.float32)) ** 2)
                axes[i, 2].set_title(f"Difference\nMSE: {mse:.2f}", fontsize=10)
            else:
                axes[i, 2].set_title(f"Shape Mismatch\n{ref_8bit.shape} vs {auto_8bit.shape}", fontsize=10)
            axes[i, 2].axis('off')
        
        fig.suptitle(f"{sample_id} - {stage.upper()} Comparison", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save comparison
        comp_file = self.output_dir / f"{sample_id}_{stage}_comparison.png"
        plt.savefig(comp_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(comp_file)
    
    def create_pipeline_overview(self, sample_id: str, 
                               raw_images: List[np.ndarray],
                               segmented_images: List[np.ndarray],
                               aligned_images: List[np.ndarray]) -> str:
        """
        Create overview visualization showing the complete pipeline
        
        Args:
            sample_id: Sample identifier
            raw_images: Raw stage images
            segmented_images: Segmented stage images  
            aligned_images: Aligned stage images
            
        Returns:
            Path to saved overview visualization
        """
        # Take first image from each stage for overview
        stages = []
        stage_names = []
        
        if raw_images:
            stages.append(raw_images[0])
            stage_names.append('Raw')
        if segmented_images:
            stages.append(segmented_images[0])
            stage_names.append('Segmented')
        if aligned_images:
            stages.append(aligned_images[0])
            stage_names.append('Aligned')
        
        if not stages:
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, len(stages), figsize=(len(stages) * 5, 5))
        if len(stages) == 1:
            axes = [axes]
        
        # Process each stage
        for i, (img, stage_name) in enumerate(zip(stages, stage_names)):
            # Convert and colorize
            img_8bit = self.convert_16bit_to_8bit(img)
            colormap = self.colormaps.get(stage_name.lower(), 'viridis')
            img_colored = self.apply_colormap(img_8bit, colormap)
            
            # Display
            axes[i].imshow(img_colored)
            axes[i].set_title(f"{stage_name}\n{img.shape} ({img.dtype})", fontsize=12, fontweight='bold')
            axes[i].axis('off')
            
            # Add arrow between stages
            if i < len(stages) - 1:
                # Add arrow annotation
                axes[i].annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5),
                               xycoords='axes fraction', textcoords='axes fraction',
                               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        
        fig.suptitle(f"{sample_id} - Complete Processing Pipeline", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save overview
        overview_file = self.output_dir / f"{sample_id}_pipeline_overview.png"
        plt.savefig(overview_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(overview_file)
    
    def create_summary_dashboard(self, evaluation_results: List[Dict]) -> str:
        """
        Create summary dashboard with all evaluation results
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            Path to saved dashboard
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Extract data for plotting
        samples = [r['sample_id'] for r in evaluation_results]
        stages = list(set(r['stage'] for r in evaluation_results))
        
        # Success rate by stage
        ax1 = plt.subplot(2, 3, 1)
        stage_success = {}
        for stage in stages:
            stage_results = [r for r in evaluation_results if r['stage'] == stage]
            success_rate = sum(1 for r in stage_results if r['success']) / len(stage_results) if stage_results else 0
            stage_success[stage] = success_rate
        
        bars = ax1.bar(stage_success.keys(), [v * 100 for v in stage_success.values()])
        ax1.set_title('Success Rate by Stage')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        
        # Color bars by success rate
        for bar, rate in zip(bars, stage_success.values()):
            if rate > 0.8:
                bar.set_color('green')
            elif rate > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Similarity scores distribution
        ax2 = plt.subplot(2, 3, 2)
        successful_results = [r for r in evaluation_results if r['success']]
        if successful_results:
            similarities = [r['similarity_score'] for r in successful_results]
            ax2.hist(similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(similarities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(similarities):.3f}')
            ax2.set_title('Similarity Score Distribution')
            ax2.set_xlabel('Similarity Score')
            ax2.set_ylabel('Frequency')
            ax2.legend()
        
        # Processing time by sample
        ax3 = plt.subplot(2, 3, 3)
        sample_times = {}
        for sample in set(samples):
            sample_results = [r for r in evaluation_results if r['sample_id'] == sample]
            avg_time = np.mean([r.get('processing_time', 0) for r in sample_results])
            sample_times[sample] = avg_time
        
        if sample_times:
            ax3.bar(range(len(sample_times)), list(sample_times.values()))
            ax3.set_title('Average Processing Time per Sample')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_xticks(range(len(sample_times)))
            ax3.set_xticklabels([s[:10] + '...' if len(s) > 10 else s for s in sample_times.keys()], 
                               rotation=45, ha='right')
        
        # File count comparison
        ax4 = plt.subplot(2, 3, 4)
        ref_counts = [r.get('file_count_reference', 0) for r in evaluation_results if r['success']]
        auto_counts = [r.get('file_count_automated', 0) for r in evaluation_results if r['success']]
        
        if ref_counts and auto_counts:
            ax4.scatter(ref_counts, auto_counts, alpha=0.6)
            ax4.plot([0, max(max(ref_counts), max(auto_counts))], 
                    [0, max(max(ref_counts), max(auto_counts))], 'r--', label='Perfect Match')
            ax4.set_title('File Count Comparison')
            ax4.set_xlabel('Reference File Count')
            ax4.set_ylabel('Automated File Count')
            ax4.legend()
        
        # Tissue type performance
        ax5 = plt.subplot(2, 3, 5)
        tissue_performance = {}
        for result in evaluation_results:
            tissue = result.get('tissue_type', 'unknown')
            if tissue not in tissue_performance:
                tissue_performance[tissue] = {'success': 0, 'total': 0}
            tissue_performance[tissue]['total'] += 1
            if result['success']:
                tissue_performance[tissue]['success'] += 1
        
        if tissue_performance:
            tissues = list(tissue_performance.keys())
            success_rates = [tissue_performance[t]['success'] / tissue_performance[t]['total'] * 100 
                           for t in tissues]
            ax5.bar(tissues, success_rates)
            ax5.set_title('Success Rate by Tissue Type')
            ax5.set_ylabel('Success Rate (%)')
            ax5.set_ylim(0, 100)
        
        # Error summary
        ax6 = plt.subplot(2, 3, 6)
        error_results = [r for r in evaluation_results if not r['success']]
        error_types = {}
        for result in error_results:
            error_msg = result.get('error_message', 'Unknown error')
            # Categorize errors
            if 'shape' in error_msg.lower() or 'dimension' in error_msg.lower():
                error_types['Shape/Dimension'] = error_types.get('Shape/Dimension', 0) + 1
            elif 'file' in error_msg.lower() or 'not found' in error_msg.lower():
                error_types['File Issues'] = error_types.get('File Issues', 0) + 1
            elif 'processing' in error_msg.lower():
                error_types['Processing Error'] = error_types.get('Processing Error', 0) + 1
            else:
                error_types['Other'] = error_types.get('Other', 0) + 1
        
        if error_types:
            ax6.pie(error_types.values(), labels=error_types.keys(), autopct='%1.1f%%')
            ax6.set_title('Error Distribution')
        else:
            ax6.text(0.5, 0.5, 'No Errors!', ha='center', va='center', fontsize=16, 
                    color='green', weight='bold')
            ax6.set_title('Error Distribution')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_file = self.output_dir / "evaluation_dashboard.png"
        plt.savefig(dashboard_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(dashboard_file)


def test_visualizer():
    """Test the visualizer with sample data including new segmentation features"""
    viz = IQIDVisualizer()
    
    # Create test 16-bit image with blob-like structures
    np.random.seed(42)
    test_image = np.zeros((512, 512), dtype=np.uint16)
    
    # Add some blob-like structures in a grid pattern
    blob_centers = [(128, 128), (128, 384), (384, 128), (384, 384)]
    for cx, cy in blob_centers:
        # Create circular blob with noise
        y, x = np.ogrid[:512, :512]
        mask = (x - cx)**2 + (y - cy)**2 < 50**2
        test_image[mask] = np.random.normal(40000, 5000, np.sum(mask)).astype(np.uint16)
    
    # Add background noise
    test_image += np.random.normal(1000, 500, test_image.shape).astype(np.uint16)
    test_image = np.clip(test_image, 0, 65535)
    
    print(f"Test image shape: {test_image.shape}, dtype: {test_image.dtype}")
    print(f"Intensity range: {test_image.min()} - {test_image.max()}")
    
    # Test robust blob segmentation
    print("\nTesting robust blob segmentation...")
    segmentation_viz = viz.output_dir / "test_segmentation_steps.png"
    blobs, mask = viz.robust_blob_segmentation(
        test_image, 
        min_blob_size=1000,
        expected_blobs=4,
        visualization_output=str(segmentation_viz)
    )
    
    print(f"Detected {len(blobs)} blobs:")
    for i, (x, y, w, h) in enumerate(blobs):
        print(f"  Blob {i+1}: ({x}, {y}) size {w}x{h}")
    
    # Test grid overlay visualization
    print("\nTesting grid overlay visualization...")
    grid_viz = viz.create_grid_overlay_visualization(
        test_image,
        expected_grid_size=(2, 2),
        detected_blobs=blobs,
        sample_id="test_sample"
    )
    print(f"Grid overlay visualization saved to: {grid_viz}")
    
    # Test basic conversion functions
    img_8bit = viz.convert_16bit_to_8bit(test_image)
    img_colored = viz.apply_colormap(img_8bit, 'viridis')
    
    print(f"\n8-bit shape: {img_8bit.shape}, dtype: {img_8bit.dtype}")
    print(f"Colored shape: {img_colored.shape}, dtype: {img_colored.dtype}")
    
    # Save test visualization
    viz_file = viz.create_processing_visualization(
        "test_sample", "raw", [test_image], ["test_image.tif"]
    )
    print(f"Test visualization saved to: {viz_file}")
    
    print(f"\nAll test visualizations saved to: {viz.output_dir}")


if __name__ == "__main__":
    test_visualizer()
