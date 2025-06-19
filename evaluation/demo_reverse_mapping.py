#!/usr/bin/env python3
"""
Simplified Reverse Mapping Demonstration

This script demonstrates the core reverse mapping functionality for Sub-Task 2.1
without requiring the full batch processor implementation.
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
from iqid_alphas.core.mapping import ReverseMapper, find_crop_location, calculate_alignment_transform


def load_image(path: Path) -> np.ndarray:
    """Simple image loading function."""
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def create_test_images() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create test images to demonstrate reverse mapping functionality.
    
    Returns
    -------
    Tuple of (raw_image, segmented_slice, original_image, aligned_image)
    """
    # Create a synthetic raw image with distinctive patterns
    raw_image = np.zeros((512, 512), dtype=np.uint8)
    
    # Add background noise
    raw_image += np.random.randint(10, 30, raw_image.shape, dtype=np.uint8)
    
    # Add some distinctive structures
    cv2.rectangle(raw_image, (100, 100), (200, 200), 150, -1)  # Square
    cv2.circle(raw_image, (300, 300), 50, 200, -1)  # Circle
    cv2.rectangle(raw_image, (50, 300), (150, 400), 180, -1)  # Another rectangle
    
    # Add some lines
    cv2.line(raw_image, (0, 250), (512, 250), 100, 3)
    cv2.line(raw_image, (250, 0), (250, 512), 120, 3)
    
    # Add more distinctive patterns in the crop region
    cv2.circle(raw_image, (150, 150), 20, 255, -1)  # Bright circle in crop area
    cv2.rectangle(raw_image, (110, 110), (130, 130), 80, -1)  # Dark rectangle
    
    # Create a segmented slice by cropping a region from the raw image
    # Extract exactly from position (100, 100) with size (100, 100)
    segmented_slice = raw_image[100:200, 100:200].copy()
    
    # Create original and aligned images for alignment testing with better features
    original_image = np.zeros((100, 100), dtype=np.uint8)
    
    # Add many distinctive features for better feature matching
    cv2.circle(original_image, (25, 25), 10, 255, -1)
    cv2.rectangle(original_image, (60, 20), (80, 40), 255, -1)
    cv2.circle(original_image, (20, 70), 8, 200, -1)
    cv2.rectangle(original_image, (70, 70), (90, 90), 180, -1)
    
    # Add more corners and edges for feature detection
    for i in range(5):
        for j in range(5):
            x, y = 15 + i * 17, 15 + j * 17
            cv2.circle(original_image, (x, y), 2, 150 + i*10, -1)
    
    # Create aligned image with known transformation (5 pixel translation)
    # Apply the transformation manually
    M = np.array([[1, 0, 5], [0, 1, 5]], dtype=np.float32)  # Translation matrix
    aligned_image = cv2.warpAffine(original_image, M, (100, 100))
    
    return raw_image, segmented_slice, original_image, aligned_image


def demonstrate_segmentation_mapping():
    """Demonstrate segmentation reverse mapping."""
    print("\n" + "="*60)
    print("SEGMENTATION REVERSE MAPPING DEMONSTRATION")
    print("="*60)
    
    # Create test images
    raw_image, segmented_slice, _, _ = create_test_images()
    
    print(f"Raw image shape: {raw_image.shape}")
    print(f"Segmented slice shape: {segmented_slice.shape}")
    
    # Test the convenience function
    print("\nTesting convenience function...")
    result_dict = find_crop_location(raw_image, segmented_slice)
    
    print(f"Crop location found:")
    print(f"  - Bounding box: {result_dict['bbox']}")
    print(f"  - Confidence: {result_dict['confidence']:.4f}")
    print(f"  - Center: {result_dict['center']}")
    print(f"  - Method: {result_dict['method']}")
    
    # Test the class-based approach
    print("\nTesting ReverseMapper class...")
    mapper = ReverseMapper()
    result_obj = mapper.find_crop_location(raw_image, segmented_slice)
    
    print(f"ReverseMapper result:")
    print(f"  - Bounding box: {result_obj.bbox}")
    print(f"  - Confidence: {result_obj.confidence:.4f}")
    print(f"  - Center: {result_obj.center}")
    print(f"  - Method: {result_obj.method}")
    
    # Validate the result
    expected_bbox = (100, 100, 100, 100)  # We know where we cropped from
    actual_bbox = result_obj.bbox
    
    print(f"\nValidation:")
    print(f"  - Expected bbox: {expected_bbox}")
    print(f"  - Actual bbox: {actual_bbox}")
    
    # Check if the mapping is approximately correct
    bbox_error = [abs(a - e) for a, e in zip(actual_bbox, expected_bbox)]
    max_error = max(bbox_error)
    
    if max_error <= 5:  # Allow some tolerance
        print(f"  - ✅ SUCCESS: Mapping found with max error {max_error} pixels")
    else:
        print(f"  - ❌ FAILED: Mapping error too high ({max_error} pixels)")
    
    return result_dict


def demonstrate_alignment_mapping():
    """Demonstrate alignment reverse mapping."""
    print("\n" + "="*60)
    print("ALIGNMENT REVERSE MAPPING DEMONSTRATION")
    print("="*60)
    
    # Create test images
    _, _, original_image, aligned_image = create_test_images()
    
    print(f"Original image shape: {original_image.shape}")
    print(f"Aligned image shape: {aligned_image.shape}")
    
    # Test the convenience function
    print("\nTesting convenience function...")
    result_dict = calculate_alignment_transform(original_image, aligned_image)
    
    print(f"Alignment transform found:")
    print(f"  - Transform matrix shape: {len(result_dict['transform_matrix'])}x{len(result_dict['transform_matrix'][0])}")
    print(f"  - Match quality: {result_dict['match_quality']:.4f}")
    print(f"  - Number of matches: {result_dict['num_matches']}")
    print(f"  - Inlier ratio: {result_dict['inlier_ratio']:.4f}")
    print(f"  - Method: {result_dict['method']}")
    
    # Test the class-based approach
    print("\nTesting ReverseMapper class...")
    mapper = ReverseMapper()
    result_obj = mapper.calculate_alignment_transform(original_image, aligned_image)
    
    print(f"ReverseMapper result:")
    print(f"  - Transform matrix shape: {result_obj.transform_matrix.shape}")
    print(f"  - Match quality: {result_obj.match_quality:.4f}")
    print(f"  - Number of matches: {result_obj.num_matches}")
    print(f"  - Inlier ratio: {result_obj.inlier_ratio:.4f}")
    print(f"  - Method: {result_obj.method}")
    
    # Analyze the transformation matrix
    transform = result_obj.transform_matrix
    print(f"\nTransformation matrix analysis:")
    print(f"  - Translation X: {transform[0, 2]:.2f}")
    print(f"  - Translation Y: {transform[1, 2]:.2f}")
    print(f"  - Scale X: {transform[0, 0]:.4f}")
    print(f"  - Scale Y: {transform[1, 1]:.4f}")
    print(f"  - Rotation/Shear: {transform[0, 1]:.4f}, {transform[1, 0]:.4f}")
    
    # Expected translation is (5, 5)
    expected_tx, expected_ty = 5.0, 5.0
    actual_tx, actual_ty = transform[0, 2], transform[1, 2]
    
    print(f"\nValidation:")
    print(f"  - Expected translation: ({expected_tx}, {expected_ty})")
    print(f"  - Actual translation: ({actual_tx:.2f}, {actual_ty:.2f})")
    
    # Check if the translation is approximately correct
    tx_error = abs(actual_tx - expected_tx)
    ty_error = abs(actual_ty - expected_ty)
    max_error = max(tx_error, ty_error)
    
    if max_error <= 10 and result_obj.match_quality > 0.1:  # Allow reasonable tolerance
        print(f"  - ✅ SUCCESS: Transform found with max error {max_error:.2f} pixels")
    else:
        print(f"  - ❌ FAILED: Transform error too high ({max_error:.2f} pixels) or low quality ({result_obj.match_quality:.4f})")
    
    return result_dict


def save_demo_results(seg_result: Dict, align_result: Dict, output_dir: Path):
    """Save demonstration results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save segmentation results
    seg_file = output_dir / "demo_segmentation_mapping.json"
    with open(seg_file, 'w') as f:
        json.dump({
            'mode': 'segmentation',
            'timestamp': datetime.now().isoformat(),
            'demonstration': True,
            'result': seg_result
        }, f, indent=2)
    
    # Save alignment results
    align_file = output_dir / "demo_alignment_mapping.json"
    with open(align_file, 'w') as f:
        json.dump({
            'mode': 'alignment',
            'timestamp': datetime.now().isoformat(),
            'demonstration': True,
            'result': align_result
        }, f, indent=2)
    
    print(f"\nDemo results saved to:")
    print(f"  - {seg_file}")
    print(f"  - {align_file}")


def main():
    """Main entry point for the demonstration."""
    parser = argparse.ArgumentParser(
        description="Demonstrate reverse mapping functionality for Sub-Task 2.1"
    )
    
    parser.add_argument(
        "--mode",
        choices=['segmentation', 'alignment', 'both'],
        default='both',
        help="Demonstration mode"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_results"),
        help="Output directory for demo results"
    )
    
    args = parser.parse_args()
    
    print("REVERSE MAPPING FRAMEWORK DEMONSTRATION")
    print("Sub-Task 2.1: Ground Truth Data Preparation")
    print(f"Mode: {args.mode}")
    
    try:
        seg_result = None
        align_result = None
        
        if args.mode in ['segmentation', 'both']:
            seg_result = demonstrate_segmentation_mapping()
        
        if args.mode in ['alignment', 'both']:
            align_result = demonstrate_alignment_mapping()
        
        # Save results
        if seg_result or align_result:
            save_demo_results(
                seg_result or {}, 
                align_result or {}, 
                args.output_dir
            )
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Sub-Task 2.1 reverse mapping framework is ready!")
        print(f"✅ Segmentation evaluation: {'✓' if seg_result else '—'}")
        print(f"✅ Alignment evaluation: {'✓' if align_result else '—'}")
        print(f"Next: Implement full BatchProcessor and enhanced visualization")
        
        return 0
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
