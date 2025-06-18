#!/usr/bin/env python3
"""
Alignment Pipeline Evaluation

This script evaluates the alignment step of the iQID processing pipeline
using existing 1_segmented data from the ReUpload dataset.
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iqid_alphas.core.alignment import ImageAligner
from skimage import io, metrics
import matplotlib.pyplot as plt

# Add evaluation utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visualizer import IQIDVisualizer


@dataclass
class AlignmentResult:
    """Container for alignment evaluation results"""
    sample_id: str
    success: bool
    alignment_quality: float  # SSIM score
    method_used: str
    processing_time: float
    num_slices: int
    error_message: Optional[str] = None
    visualization_path: Optional[str] = None


class AlignmentEvaluator:
    """
    Evaluates alignment step using existing 1_segmented data
    """
    
    def __init__(self, reupload_path: str, output_dir: str = "evaluation/outputs/alignment_test"):
        self.reupload_path = Path(reupload_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.aligner = ImageAligner()
        
        # Initialize visualizer
        viz_output_dir = self.output_dir / "visualizations"
        self.visualizer = IQIDVisualizer(str(viz_output_dir))
        
        # Results storage
        self.results: List[AlignmentResult] = []
        
        self.logger.info(f"Initialized alignment evaluator for: {self.reupload_path}")
        self.logger.info(f"Outputs will be saved to: {self.output_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("AlignmentEvaluator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def discover_segmented_samples(self) -> List[Dict]:
        """
        Discover samples with 1_segmented data
        """
        samples = []
        
        # Check different possible paths for ReUpload data
        possible_bases = [
            self.reupload_path / "3D",
            self.reupload_path / "Sequential", 
            self.reupload_path / "iQID_reupload" / "iQID" / "3D",
            self.reupload_path / "iQID_reupload" / "iQID" / "Sequential"
        ]
        
        for base_path in possible_bases:
            if base_path.exists():
                self.logger.info(f"Found processing base: {base_path}")
                
                # Find tissue directories
                for tissue_dir in base_path.iterdir():
                    if tissue_dir.is_dir():
                        # Find sample directories
                        for sample_dir in tissue_dir.iterdir():
                            if sample_dir.is_dir():
                                segmented_dir = sample_dir / "1_segmented"
                                aligned_dir = sample_dir / "2_aligned"
                                
                                if segmented_dir.exists():
                                    segmented_files = sorted(list(segmented_dir.glob("*.tif")))
                                    if segmented_files:
                                        sample_info = {
                                            'sample_id': sample_dir.name,
                                            'tissue_type': tissue_dir.name,
                                            'processing_type': base_path.name,
                                            'segmented_dir': segmented_dir,
                                            'aligned_dir': aligned_dir if aligned_dir.exists() else None,
                                            'segmented_files': segmented_files,
                                            'has_reference': aligned_dir.exists()
                                        }
                                        samples.append(sample_info)
                                        
                                        self.logger.debug(f"Found sample: {sample_dir.name} "
                                                        f"({len(segmented_files)} segmented slices)")
        
        self.logger.info(f"Discovered {len(samples)} samples with segmented data")
        return samples
    
    def align_sample(self, sample_info: Dict) -> AlignmentResult:
        """
        Align a single sample and evaluate results
        """
        sample_id = sample_info['sample_id']
        segmented_files = sample_info['segmented_files']
        
        self.logger.info(f"Aligning sample: {sample_id} ({len(segmented_files)} slices)")
        
        try:
            start_time = datetime.now()
            
            # Create output directory for this sample
            sample_output_dir = self.output_dir / "aligned_results" / sample_id
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load segmented images
            images = []
            for file_path in segmented_files:
                try:
                    img = io.imread(file_path)
                    if img.ndim == 3:
                        img = img[:, :, 0]  # Take first channel if RGB
                    images.append(img.astype(np.float32))
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
            
            if len(images) < 2:
                return AlignmentResult(
                    sample_id=sample_id,
                    success=False,
                    alignment_quality=0.0,
                    method_used="none",
                    processing_time=0.0,
                    num_slices=len(images),
                    error_message="Need at least 2 slices for alignment"
                )
            
            # Perform alignment using arrays directly
            aligned_images = []
            alignment_info = {'method_used': 'sequential_alignment', 'transforms': []}
            
            if len(images) == 1:
                aligned_images = images
                alignment_info['method_used'] = 'single_image'
            else:
                # Use first image as reference
                reference_img = images[0]
                aligned_images.append(reference_img)
                
                # Align each subsequent image to the reference
                for i in range(1, len(images)):
                    try:
                        aligned_img, transform_info = self.aligner.align_images(reference_img, images[i])
                        aligned_images.append(aligned_img)
                        alignment_info['transforms'].append({
                            'slice_index': i,
                            'method': transform_info.get('method', 'unknown'),
                            'transform': transform_info
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to align slice {i}: {e}")
                        # Fall back to original image
                        aligned_images.append(images[i])
                        alignment_info['transforms'].append({
                            'slice_index': i,
                            'method': 'fallback',
                            'error': str(e)
                        })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Save aligned images
            for i, aligned_img in enumerate(aligned_images):
                output_path = sample_output_dir / f"aligned_{i:03d}.tif"
                # Normalize to 0-255 for saving
                img_normalized = ((aligned_img - aligned_img.min()) / 
                                (aligned_img.max() - aligned_img.min() + 1e-8) * 255).astype(np.uint8)
                io.imsave(output_path, img_normalized)
            
            # Calculate alignment quality
            alignment_quality = self._calculate_alignment_quality(aligned_images)
            
            # Generate visualizations
            viz_path = self._create_alignment_visualization(
                sample_id, images, aligned_images, alignment_info
            )
            
            # Compare with reference if available
            reference_quality = None
            if sample_info['has_reference']:
                reference_quality = self._compare_with_reference(
                    sample_info, aligned_images
                )
            
            result = AlignmentResult(
                sample_id=sample_id,
                success=True,
                alignment_quality=alignment_quality,
                method_used=alignment_info.get('method_used', 'unknown'),
                processing_time=processing_time,
                num_slices=len(aligned_images),
                visualization_path=str(viz_path) if viz_path else None
            )
            
            self.logger.info(f"Alignment completed: {sample_id}, "
                           f"quality={alignment_quality:.3f}, "
                           f"method={alignment_info.get('method_used', 'unknown')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Alignment failed for {sample_id}: {e}")
            return AlignmentResult(
                sample_id=sample_id,
                success=False,
                alignment_quality=0.0,
                method_used="failed",
                processing_time=0.0,
                num_slices=len(segmented_files),
                error_message=str(e)
            )
    
    def _calculate_alignment_quality(self, aligned_images: List[np.ndarray]) -> float:
        """
        Calculate overall alignment quality using pairwise SSIM
        """
        if len(aligned_images) < 2:
            return 0.0
        
        ssim_scores = []
        for i in range(len(aligned_images) - 1):
            try:
                img1 = aligned_images[i]
                img2 = aligned_images[i + 1]
                
                # Ensure images have the same shape by cropping to minimum dimensions
                min_shape = [min(img1.shape[j], img2.shape[j]) for j in range(2)]
                img1_cropped = img1[:min_shape[0], :min_shape[1]]
                img2_cropped = img2[:min_shape[0], :min_shape[1]]
                
                # Calculate data range for SSIM
                data_range = max(img1_cropped.max() - img1_cropped.min(),
                               img2_cropped.max() - img2_cropped.min())
                
                if data_range > 0:
                    ssim = metrics.structural_similarity(
                        img1_cropped, img2_cropped, data_range=data_range
                    )
                    ssim_scores.append(ssim)
                else:
                    ssim_scores.append(0.0)
                    
            except Exception as e:
                self.logger.warning(f"SSIM calculation failed: {e}")
                ssim_scores.append(0.0)
        
        return np.mean(ssim_scores) if ssim_scores else 0.0
    
    def _compare_with_reference(self, sample_info: Dict, 
                              aligned_images: List[np.ndarray]) -> float:
        """
        Compare automated alignment with reference 2_aligned data
        """
        aligned_dir = sample_info['aligned_dir']
        if not aligned_dir or not aligned_dir.exists():
            return None
        
        reference_files = sorted(list(aligned_dir.glob("*.tif")))
        if not reference_files:
            return None
        
        try:
            # Load reference images
            reference_images = []
            for ref_file in reference_files:
                ref_img = io.imread(ref_file)
                if ref_img.ndim == 3:
                    ref_img = ref_img[:, :, 0]
                reference_images.append(ref_img.astype(np.float32))
            
            # Compare with automated results
            min_count = min(len(aligned_images), len(reference_images))
            ssim_scores = []
            
            for i in range(min_count):
                auto_img = aligned_images[i]
                ref_img = reference_images[i]
                
                # Resize if needed
                if auto_img.shape != ref_img.shape:
                    from skimage.transform import resize
                    auto_img = resize(auto_img, ref_img.shape, preserve_range=True)
                
                ssim = metrics.structural_similarity(
                    auto_img, ref_img,
                    data_range=max(auto_img.max(), ref_img.max()) - 
                             min(auto_img.min(), ref_img.min())
                )
                ssim_scores.append(ssim)
            
            comparison_score = np.mean(ssim_scores) if ssim_scores else 0.0
            self.logger.info(f"Reference comparison SSIM: {comparison_score:.3f}")
            return comparison_score
            
        except Exception as e:
            self.logger.warning(f"Reference comparison failed: {e}")
            return None
    
    def _create_alignment_visualization(self, sample_id: str, 
                                      original_images: List[np.ndarray],
                                      aligned_images: List[np.ndarray],
                                      alignment_info: Dict) -> Optional[Path]:
        """
        Create comprehensive alignment visualization
        """
        try:
            viz_dir = self.output_dir / "visualizations" / sample_id
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Create before/after comparison
            fig, axes = plt.subplots(2, min(4, len(original_images)), 
                                   figsize=(16, 8))
            if len(original_images) == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(min(4, len(original_images))):
                # Original image
                axes[0, i].imshow(original_images[i], cmap='viridis')
                axes[0, i].set_title(f'Original Slice {i}')
                axes[0, i].axis('off')
                
                # Aligned image
                axes[1, i].imshow(aligned_images[i], cmap='viridis')
                axes[1, i].set_title(f'Aligned Slice {i}')
                axes[1, i].axis('off')
            
            plt.suptitle(f'Alignment Results: {sample_id}\n'
                        f"Method: {alignment_info.get('method_used', 'unknown')}")
            plt.tight_layout()
            
            comparison_path = viz_dir / "alignment_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create alignment overlay for adjacent slices
            if len(aligned_images) >= 2:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Show first two aligned slices
                axes[0].imshow(aligned_images[0], cmap='Reds', alpha=0.7)
                axes[0].set_title('Slice 0')
                axes[0].axis('off')
                
                axes[1].imshow(aligned_images[1], cmap='Blues', alpha=0.7)
                axes[1].set_title('Slice 1')
                axes[1].axis('off')
                
                # Overlay
                axes[2].imshow(aligned_images[0], cmap='Reds', alpha=0.5)
                axes[2].imshow(aligned_images[1], cmap='Blues', alpha=0.5)
                axes[2].set_title('Overlay (Red: Slice 0, Blue: Slice 1)')
                axes[2].axis('off')
                
                plt.suptitle(f'Alignment Quality Check: {sample_id}')
                plt.tight_layout()
                
                overlay_path = viz_dir / "alignment_overlay.png"
                plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            return comparison_path
            
        except Exception as e:
            self.logger.error(f"Visualization creation failed for {sample_id}: {e}")
            return None
    
    def run_evaluation(self, max_samples: Optional[int] = None) -> Dict:
        """
        Run alignment evaluation on available samples
        """
        self.logger.info("Starting alignment evaluation...")
        
        # Discover samples
        samples = self.discover_segmented_samples()
        if not samples:
            self.logger.error("No samples found with segmented data")
            return {'error': 'No samples found'}
        
        if max_samples:
            samples = samples[:max_samples]
            self.logger.info(f"Limited evaluation to {len(samples)} samples")
        
        # Evaluate each sample
        for i, sample_info in enumerate(samples):
            self.logger.info(f"Processing sample {i+1}/{len(samples)}: {sample_info['sample_id']}")
            result = self.align_sample(sample_info)
            self.results.append(result)
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _generate_summary(self) -> Dict:
        """Generate evaluation summary statistics"""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        summary = {
            'total_samples': len(self.results),
            'successful_alignments': len(successful_results),
            'failed_alignments': len(failed_results),
            'success_rate': len(successful_results) / len(self.results) if self.results else 0,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        if successful_results:
            quality_scores = [r.alignment_quality for r in successful_results]
            processing_times = [r.processing_time for r in successful_results]
            
            summary.update({
                'alignment_quality': {
                    'mean': np.mean(quality_scores),
                    'std': np.std(quality_scores),
                    'min': np.min(quality_scores),
                    'max': np.max(quality_scores),
                    'median': np.median(quality_scores)
                },
                'processing_time': {
                    'mean': np.mean(processing_times),
                    'std': np.std(processing_times),
                    'total': np.sum(processing_times)
                },
                'methods_used': {
                    method: sum(1 for r in successful_results if r.method_used == method)
                    for method in set(r.method_used for r in successful_results)
                }
            })
        
        return summary
    
    def _save_results(self, summary: Dict):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_data = {
            'summary': summary,
            'detailed_results': [
                {
                    'sample_id': r.sample_id,
                    'success': r.success,
                    'alignment_quality': r.alignment_quality,
                    'method_used': r.method_used,
                    'processing_time': r.processing_time,
                    'num_slices': r.num_slices,
                    'error_message': r.error_message,
                    'visualization_path': r.visualization_path
                }
                for r in self.results
            ]
        }
        
        results_file = self.output_dir / f"alignment_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate alignment pipeline")
    parser.add_argument("--data", required=True, help="Path to ReUpload dataset")
    parser.add_argument("--output", default="evaluation/outputs/alignment_test", 
                       help="Output directory")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to process")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run evaluation
    evaluator = AlignmentEvaluator(args.data, args.output)
    summary = evaluator.run_evaluation(args.max_samples)
    
    # Print summary
    print("\n" + "="*60)
    print("ALIGNMENT EVALUATION RESULTS")
    print("="*60)
    
    if 'error' in summary:
        print(f"❌ Evaluation failed: {summary['error']}")
        return
    
    print(f"📊 Total samples: {summary['total_samples']}")
    print(f"✅ Successful alignments: {summary['successful_alignments']}")
    print(f"❌ Failed alignments: {summary['failed_alignments']}")
    print(f"📈 Success rate: {summary['success_rate']:.2%}")
    
    if 'alignment_quality' in summary:
        quality = summary['alignment_quality']
        print(f"\n🎯 Alignment Quality (SSIM):")
        print(f"   Mean: {quality['mean']:.3f} ± {quality['std']:.3f}")
        print(f"   Range: {quality['min']:.3f} - {quality['max']:.3f}")
        print(f"   Median: {quality['median']:.3f}")
    
    if 'processing_time' in summary:
        timing = summary['processing_time']
        print(f"\n⏱️  Processing Time:")
        print(f"   Mean per sample: {timing['mean']:.2f}s")
        print(f"   Total time: {timing['total']:.2f}s")
    
    if 'methods_used' in summary:
        methods = summary['methods_used']
        print(f"\n🔧 Alignment Methods:")
        for method, count in methods.items():
            print(f"   {method}: {count} samples")
    
    print(f"\n📁 Results saved to: {evaluator.output_dir}")


if __name__ == "__main__":
    main()
