#!/usr/bin/env python3
"""
Unified Reverse Mapping Evaluation Script

This script provides mode-driven evaluation for both segmentation and alignment
using the reverse mapping framework. It can:

1. Segmentation Mode: Locate cropped slices from `1_segmented` within `raw` images
2. Alignment Mode: Calculate transformations between `1_segmented` and `2_aligned` slices

The script generates both visualizations and JSON reports for automated analysis.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
from datetime import datetime

# Core imports
from iqid_alphas.core.mapping import ReverseMapper
from iqid_alphas.utils.io_utils import natural_sort
try:
    from iqid_alphas.utils.io_utils import load_image
except ImportError:
    # Fallback image loading
    import cv2
    def load_image(path):
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


class ReverseMapEvaluator:
    """
    Unified reverse mapping evaluator for both segmentation and alignment modes.
    """
    
    def __init__(self, 
                 data_root: Path,
                 output_dir: Path,
                 mode: str = 'segmentation'):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        data_root : Path
            Root directory containing the pipeline data
        output_dir : Path
            Directory to save results and visualizations
        mode : str
            Evaluation mode ('segmentation' or 'alignment')
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.mode = mode.lower()
        
        if self.mode not in ['segmentation', 'alignment']:
            raise ValueError(f"Mode must be 'segmentation' or 'alignment', got: {mode}")
        
        # Initialize components
        self.mapper = ReverseMapper()
        # Note: batch_processor and enhanced_plotter not yet implemented
        # Will add mock functionality for now
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful_mappings': 0,
            'failed_mappings': 0,
            'high_quality_mappings': 0,
            'average_confidence': 0.0,
            'processing_time': 0.0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f'reverse_map_evaluator_{self.mode}')
        logger.setLevel(logging.INFO)
        
        # Create handler
        log_file = self.output_dir / f"reverse_mapping_{self.mode}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def evaluate_segmentation_mode(self) -> Dict[str, Any]:
        """
        Evaluate segmentation by finding crop locations of segmented slices in raw images.
        
        Returns
        -------
        Dict containing evaluation results
        """
        self.logger.info("Starting segmentation evaluation...")
        
        try:
            # Discover segmentation samples
            samples = self.batch_processor.discover_segmentation_samples()
            self.logger.info(f"Found {len(samples)} segmentation samples")
            
            results = {
                'mode': 'segmentation',
                'timestamp': datetime.now().isoformat(),
                'samples': []
            }
            
            for sample in samples:
                try:
                    sample_result = self._evaluate_segmentation_sample(sample)
                    results['samples'].append(sample_result)
                    self.stats['total_processed'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process sample {sample.get('sample_id', 'unknown')}: {e}")
                    self.stats['failed_mappings'] += 1
            
            # Calculate final statistics
            results['statistics'] = self._calculate_statistics()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Segmentation evaluation failed: {e}")
            raise
    
    def _evaluate_segmentation_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single segmentation sample."""
        sample_id = sample.get('sample_id', 'unknown')
        raw_image_path = Path(sample['raw_image'])
        segmented_dir = Path(sample['segmented_dir'])
        
        self.logger.info(f"Processing segmentation sample: {sample_id}")
        
        # Load raw image
        raw_image = load_image(raw_image_path)
        
        # Get segmented slices
        slice_files = list(segmented_dir.glob("*.tif"))
        slice_files = natural_sort([str(f) for f in slice_files])
        
        sample_result = {
            'sample_id': sample_id,
            'raw_image': str(raw_image_path),
            'segmented_dir': str(segmented_dir),
            'mapped_slices': []
        }
        
        for slice_path in slice_files:
            try:
                # Load segmented slice
                slice_image = load_image(slice_path)
                
                # Find crop location
                crop_result = self.mapper.find_crop_location(raw_image, slice_image)
                
                # Store result
                slice_result = {
                    'slice': Path(slice_path).name,
                    'bbox': crop_result.bbox,
                    'confidence': crop_result.confidence,
                    'center': crop_result.center,
                    'method': crop_result.method
                }
                
                sample_result['mapped_slices'].append(slice_result)
                
                # Update statistics
                if crop_result.confidence > 0.8:
                    self.stats['high_quality_mappings'] += 1
                self.stats['successful_mappings'] += 1
                
                # Generate visualization
                self._create_segmentation_visualization(
                    raw_image, slice_image, crop_result, 
                    sample_id, Path(slice_path).name
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to process slice {slice_path}: {e}")
                self.stats['failed_mappings'] += 1
        
        return sample_result
    
    def evaluate_alignment_mode(self) -> Dict[str, Any]:
        """
        Evaluate alignment by calculating transformations between aligned and segmented slices.
        
        Returns
        -------
        Dict containing evaluation results
        """
        self.logger.info("Starting alignment evaluation...")
        
        try:
            # Discover alignment samples
            samples = self.batch_processor.discover_alignment_samples()
            self.logger.info(f"Found {len(samples)} alignment samples")
            
            results = {
                'mode': 'alignment',
                'timestamp': datetime.now().isoformat(),
                'samples': []
            }
            
            for sample in samples:
                try:
                    sample_result = self._evaluate_alignment_sample(sample)
                    results['samples'].append(sample_result)
                    self.stats['total_processed'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process sample {sample.get('sample_id', 'unknown')}: {e}")
                    self.stats['failed_mappings'] += 1
            
            # Calculate final statistics
            results['statistics'] = self._calculate_statistics()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Alignment evaluation failed: {e}")
            raise
    
    def _evaluate_alignment_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single alignment sample."""
        sample_id = sample.get('sample_id', 'unknown')
        segmented_dir = Path(sample['segmented_dir'])
        aligned_dir = Path(sample['aligned_dir'])
        
        self.logger.info(f"Processing alignment sample: {sample_id}")
        
        sample_result = {
            'sample_id': sample_id,
            'segmented_dir': str(segmented_dir),
            'aligned_dir': str(aligned_dir),
            'mapped_transforms': []
        }
        
        # Find corresponding slice pairs
        slice_pairs = self._find_corresponding_slices(segmented_dir, aligned_dir)
        
        for original_path, aligned_path in slice_pairs:
            try:
                # Load images
                original_image = load_image(original_path)
                aligned_image = load_image(aligned_path)
                
                # Calculate transformation
                transform_result = self.mapper.calculate_alignment_transform(
                    original_image, aligned_image
                )
                
                # Store result
                transform_data = {
                    'slice_pair': [Path(original_path).name, Path(aligned_path).name],
                    'transform_matrix': transform_result.transform_matrix.tolist(),
                    'match_quality': transform_result.match_quality,
                    'num_matches': transform_result.num_matches,
                    'inlier_ratio': transform_result.inlier_ratio,
                    'method': transform_result.method
                }
                
                sample_result['mapped_transforms'].append(transform_data)
                
                # Update statistics
                if transform_result.match_quality > 0.7:
                    self.stats['high_quality_mappings'] += 1
                self.stats['successful_mappings'] += 1
                
                # Generate visualization
                self._create_alignment_visualization(
                    original_image, aligned_image, transform_result,
                    sample_id, Path(original_path).name, Path(aligned_path).name
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to process pair {original_path}, {aligned_path}: {e}")
                self.stats['failed_mappings'] += 1
        
        return sample_result
    
    def _find_corresponding_slices(self, segmented_dir: Path, aligned_dir: Path) -> List[Tuple[str, str]]:
        """Find corresponding slice pairs between segmented and aligned directories."""
        # Get all slice files
        segmented_files = list(segmented_dir.glob("*.tif"))
        aligned_files = list(aligned_dir.glob("*.tif"))
        
        # Sort naturally
        segmented_files = natural_sort([str(f) for f in segmented_files])
        aligned_files = natural_sort([str(f) for f in aligned_files])
        
        # Match files based on common patterns
        pairs = []
        
        for seg_file in segmented_files:
            seg_name = Path(seg_file).stem
            
            # Look for corresponding aligned file
            # Common patterns: mBq_0.tif -> mBq_corr_0.tif
            for aligned_file in aligned_files:
                aligned_name = Path(aligned_file).stem
                
                # Remove common alignment suffixes
                aligned_base = aligned_name.replace('_corr', '').replace('_aligned', '')
                
                if seg_name == aligned_base:
                    pairs.append((seg_file, aligned_file))
                    break
        
        return pairs
    
    def _create_segmentation_visualization(self, 
                                         raw_image: np.ndarray, 
                                         slice_image: np.ndarray,
                                         crop_result: Any,
                                         sample_id: str,
                                         slice_name: str):
        """Create visualization for segmentation evaluation."""
        try:
            # Use the enhanced plotter to create segmentation visualization
            viz_path = self.output_dir / f"segmentation_{sample_id}_{slice_name}.png"
            
            self.plotter.plot_segmentation_on_raw(
                raw_image=raw_image,
                segmented_slice=slice_image,
                crop_location=crop_result,
                save_path=str(viz_path),
                title=f"Segmentation Mapping: {sample_id} - {slice_name}"
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to create segmentation visualization: {e}")
    
    def _create_alignment_visualization(self,
                                      original_image: np.ndarray,
                                      aligned_image: np.ndarray,
                                      transform_result: Any,
                                      sample_id: str,
                                      original_name: str,
                                      aligned_name: str):
        """Create visualization for alignment evaluation."""
        try:
            viz_path = self.output_dir / f"alignment_{sample_id}_{original_name}_{aligned_name}.png"
            
            # Create side-by-side comparison with overlay
            height = max(original_image.shape[0], aligned_image.shape[0])
            width = original_image.shape[1] + aligned_image.shape[1] + 20
            
            # Create combined image
            combined = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Convert to RGB if needed
            if len(original_image.shape) == 2:
                orig_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            else:
                orig_rgb = original_image
                
            if len(aligned_image.shape) == 2:
                aligned_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2RGB)
            else:
                aligned_rgb = aligned_image
            
            # Place images side by side
            h1, w1 = orig_rgb.shape[:2]
            h2, w2 = aligned_rgb.shape[:2]
            
            combined[:h1, :w1] = orig_rgb
            combined[:h2, w1+20:w1+20+w2] = aligned_rgb
            
            # Add quality metrics text
            quality_text = f"Quality: {transform_result.match_quality:.3f}, Matches: {transform_result.num_matches}"
            cv2.putText(combined, quality_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save visualization
            cv2.imwrite(str(viz_path), combined)
            
        except Exception as e:
            self.logger.warning(f"Failed to create alignment visualization: {e}")
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate final evaluation statistics."""
        total = self.stats['total_processed']
        successful = self.stats['successful_mappings']
        
        success_rate = (successful / total * 100) if total > 0 else 0
        quality_rate = (self.stats['high_quality_mappings'] / successful * 100) if successful > 0 else 0
        
        return {
            'total_processed': total,
            'successful_mappings': successful,
            'failed_mappings': self.stats['failed_mappings'],
            'success_rate_percent': success_rate,
            'high_quality_mappings': self.stats['high_quality_mappings'],
            'high_quality_rate_percent': quality_rate,
            'average_processing_time': self.stats['processing_time']
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the evaluation based on the configured mode.
        
        Returns
        -------
        Dict containing complete evaluation results
        """
        start_time = datetime.now()
        
        try:
            if self.mode == 'segmentation':
                results = self.evaluate_segmentation_mode()
            elif self.mode == 'alignment':
                results = self.evaluate_alignment_mode()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            # Calculate processing time
            end_time = datetime.now()
            self.stats['processing_time'] = (end_time - start_time).total_seconds()
            
            # Save results to JSON
            results_file = self.output_dir / f"reverse_mapping_{self.mode}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Evaluation completed. Results saved to {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Unified reverse mapping evaluation for segmentation and alignment"
    )
    
    parser.add_argument(
        "--mode",
        choices=['segmentation', 'alignment'],
        required=True,
        help="Evaluation mode: segmentation or alignment"
    )
    
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing pipeline data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for results and visualizations"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create evaluator
        evaluator = ReverseMapEvaluator(
            data_root=args.data_root,
            output_dir=args.output_dir,
            mode=args.mode
        )
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Print summary
        stats = results.get('statistics', {})
        print(f"\n{'='*60}")
        print(f"REVERSE MAPPING EVALUATION SUMMARY ({args.mode.upper()})")
        print(f"{'='*60}")
        print(f"Total Processed: {stats.get('total_processed', 0)}")
        print(f"Successful Mappings: {stats.get('successful_mappings', 0)}")
        print(f"Success Rate: {stats.get('success_rate_percent', 0):.1f}%")
        print(f"High Quality Mappings: {stats.get('high_quality_mappings', 0)}")
        print(f"High Quality Rate: {stats.get('high_quality_rate_percent', 0):.1f}%")
        print(f"Processing Time: {stats.get('average_processing_time', 0):.2f}s")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
