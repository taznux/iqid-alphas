#!/usr/bin/env python3
"""
Test alignment functionality using existing 1_segmented data

This script tests the alignment step by starting with existing segmented data
and comparing automated alignment results with the ground truth 2_aligned data.
"""

import numpy as np
import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from iqid_alphas.core.alignment import ImageAligner
from evaluation.utils.visualizer import IQIDVisualizer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def find_segmented_samples(reupload_path: Path):
    """
    Find samples that have 1_segmented data for alignment testing
    """
    logger = logging.getLogger(__name__)
    samples = []
    
    # Look for iQID data structure
    iqid_paths = [
        reupload_path / "iQID_reupload" / "iQID",  # Expected structure
        reupload_path / "iQID",                    # Alternative structure
        reupload_path                             # Already pointing to iQID
    ]
    
    iqid_base_path = None
    for path in iqid_paths:
        if path.exists() and any(path.glob("*/*/*/1_*")):
            iqid_base_path = path
            break
    
    if not iqid_base_path:
        logger.error(f"Could not find iQID data structure in {reupload_path}")
        return samples
    
    logger.info(f"Using iQID base path: {iqid_base_path}")
    
    # Process both 3D and Sequential processing types
    for processing_type in ["3D", "Sequential"]:
        processing_path = iqid_base_path / processing_type
        if not processing_path.exists():
            continue
            
        logger.info(f"Found processing type: {processing_type}")
        
        # Find tissue type directories
        for tissue_dir in processing_path.iterdir():
            if not tissue_dir.is_dir():
                continue
                
            tissue_type = tissue_dir.name
            logger.info(f"  Found tissue type: {tissue_type}")
            
            # Find sample directories
            for sample_dir in tissue_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                    
                sample_id = sample_dir.name
                
                # Check for required directories
                segmented_dir = sample_dir / "1_segmented"
                aligned_dir = sample_dir / "2_aligned"
                
                if segmented_dir.exists() and aligned_dir.exists():
                    # Count files
                    segmented_files = list(segmented_dir.glob("*.tif"))
                    aligned_files = list(aligned_dir.glob("*.tif"))
                    
                    if len(segmented_files) > 0 and len(aligned_files) > 0:
                        samples.append({
                            'sample_id': sample_id,
                            'tissue_type': tissue_type,
                            'processing_type': processing_type,
                            'sample_path': sample_dir,
                            'segmented_dir': segmented_dir,
                            'aligned_dir': aligned_dir,
                            'segmented_count': len(segmented_files),
                            'aligned_count': len(aligned_files)
                        })
                        logger.info(f"    Added sample: {sample_id} ({len(segmented_files)} segmented, {len(aligned_files)} aligned)")
    
    logger.info(f"Found {len(samples)} samples with both segmented and aligned data")
    return samples

def test_alignment_single_sample(sample_info: dict, output_dir: Path, aligner: ImageAligner):
    """
    Test alignment on a single sample
    """
    logger = logging.getLogger(__name__)
    sample_id = sample_info['sample_id']
    
    logger.info(f"Testing alignment for sample: {sample_id}")
    
    # Create output directory for this sample
    sample_output_dir = output_dir / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get segmented files (input)
        segmented_files = sorted(list(sample_info['segmented_dir'].glob("*.tif")))
        
        if len(segmented_files) == 0:
            return {'success': False, 'error': 'No segmented files found'}
        
        logger.info(f"  Found {len(segmented_files)} segmented files")
        
        # Run automated alignment
        start_time = datetime.now()
        automated_aligned_dir = sample_output_dir / "automated_aligned"
        
        aligned_files = aligner.align_images_files(
            [str(f) for f in segmented_files],
            str(automated_aligned_dir)
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"  Alignment completed in {processing_time:.2f} seconds")
        logger.info(f"  Generated {len(aligned_files)} aligned files")
        
        # Load reference aligned files (ground truth)
        reference_files = sorted(list(sample_info['aligned_dir'].glob("*.tif")))
        logger.info(f"  Reference has {len(reference_files)} aligned files")
        
        # Compare results
        comparison_result = compare_alignment_results(
            automated_aligned_dir, sample_info['aligned_dir'], sample_id
        )
        
        return {
            'success': True,
            'sample_id': sample_id,
            'processing_time': processing_time,
            'input_count': len(segmented_files),
            'output_count': len(aligned_files),
            'reference_count': len(reference_files),
            'comparison': comparison_result,
            'automated_dir': str(automated_aligned_dir),
            'reference_dir': str(sample_info['aligned_dir'])
        }
        
    except Exception as e:
        logger.error(f"  Alignment failed for {sample_id}: {e}")
        return {
            'success': False,
            'sample_id': sample_id,
            'error': str(e)
        }

def compare_alignment_results(automated_dir: Path, reference_dir: Path, sample_id: str):
    """
    Compare automated alignment results with reference data
    """
    try:
        import skimage.io
        from skimage.metrics import structural_similarity as ssim
        
        # Get file lists
        automated_files = sorted(list(automated_dir.glob("*.tif")))
        reference_files = sorted(list(reference_dir.glob("*.tif")))
        
        if len(automated_files) == 0 or len(reference_files) == 0:
            return {'error': 'No files to compare'}
        
        similarities = []
        min_files = min(len(automated_files), len(reference_files))
        
        for i in range(min_files):
            try:
                auto_img = skimage.io.imread(automated_files[i])
                ref_img = skimage.io.imread(reference_files[i])
                
                # Ensure same shape for comparison
                if auto_img.shape != ref_img.shape:
                    # Resize if needed (simple approach)
                    min_h = min(auto_img.shape[0], ref_img.shape[0])
                    min_w = min(auto_img.shape[1], ref_img.shape[1])
                    auto_img = auto_img[:min_h, :min_w]
                    ref_img = ref_img[:min_h, :min_w]
                
                # Calculate SSIM
                if auto_img.dtype != ref_img.dtype:
                    auto_img = auto_img.astype(np.float32)
                    ref_img = ref_img.astype(np.float32)
                
                similarity = ssim(auto_img, ref_img, data_range=auto_img.max() - auto_img.min())
                similarities.append(similarity)
                
            except Exception as e:
                print(f"Failed to compare file {i}: {e}")
                similarities.append(0.0)
        
        return {
            'file_count_automated': len(automated_files),
            'file_count_reference': len(reference_files),
            'similarities': similarities,
            'average_similarity': np.mean(similarities) if similarities else 0.0,
            'min_similarity': np.min(similarities) if similarities else 0.0,
            'max_similarity': np.max(similarities) if similarities else 0.0
        }
        
    except Exception as e:
        return {'error': f'Comparison failed: {e}'}

def run_alignment_test(reupload_path: str, output_dir: str = "test_alignment_output", 
                      max_samples: int = None):
    """
    Run alignment test on ReUpload data
    """
    logger = setup_logging()
    logger.info("Starting alignment test using existing 1_segmented data")
    
    reupload_path = Path(reupload_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find samples with segmented data
    samples = find_segmented_samples(reupload_path)
    if not samples:
        logger.error("No samples found with segmented data")
        return
    
    if max_samples:
        samples = samples[:max_samples]
        logger.info(f"Limited test to {len(samples)} samples")
    
    # Initialize aligner
    aligner = ImageAligner(method='phase_correlation')
    
    # Test each sample
    results = []
    for i, sample_info in enumerate(samples):
        logger.info(f"Processing sample {i+1}/{len(samples)}: {sample_info['sample_id']}")
        
        result = test_alignment_single_sample(sample_info, output_dir, aligner)
        results.append(result)
        
        # Print progress
        if result['success']:
            comparison = result.get('comparison', {})
            avg_sim = comparison.get('average_similarity', 0)
            logger.info(f"  ✓ Success - Average similarity: {avg_sim:.3f}")
        else:
            logger.warning(f"  ✗ Failed - {result.get('error', 'Unknown error')}")
    
    # Generate summary
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(samples),
        'successful': len(successful_results),
        'failed': len(failed_results),
        'success_rate': len(successful_results) / len(samples) if samples else 0,
        'results': results
    }
    
    if successful_results:
        similarities = []
        processing_times = []
        
        for result in successful_results:
            comp = result.get('comparison', {})
            if 'average_similarity' in comp:
                similarities.append(comp['average_similarity'])
            if 'processing_time' in result:
                processing_times.append(result['processing_time'])
        
        if similarities:
            summary['overall_similarity'] = {
                'mean': np.mean(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities),
                'std': np.std(similarities)
            }
        
        if processing_times:
            summary['performance'] = {
                'mean_time': np.mean(processing_times),
                'total_time': np.sum(processing_times)
            }
    
    # Save results
    results_file = output_dir / f"alignment_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("ALIGNMENT TEST RESULTS")
    print("="*60)
    print(f"Total samples tested: {len(samples)}")
    print(f"Successful alignments: {len(successful_results)}")
    print(f"Failed alignments: {len(failed_results)}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    
    if 'overall_similarity' in summary:
        sim = summary['overall_similarity']
        print(f"\nAlignment Quality (SSIM):")
        print(f"  Average: {sim['mean']:.3f}")
        print(f"  Range: {sim['min']:.3f} - {sim['max']:.3f}")
        print(f"  Std Dev: {sim['std']:.3f}")
    
    if 'performance' in summary:
        perf = summary['performance']
        print(f"\nPerformance:")
        print(f"  Average time per sample: {perf['mean_time']:.2f} seconds")
        print(f"  Total processing time: {perf['total_time']:.2f} seconds")
    
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test alignment functionality using existing segmented data")
    parser.add_argument("--data", required=True, help="Path to ReUpload dataset")
    parser.add_argument("--output", default="test_alignment_output", help="Output directory")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the test
    summary = run_alignment_test(args.data, args.output, args.max_samples)
    
    print(f"\n✓ Alignment test completed!")
    print(f"Results are in: {args.output}")
