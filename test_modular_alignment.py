#!/usr/bin/env python3
"""
Test the modular alignment package structure
"""

import sys
import os
sys.path.insert(0, '/home/wxc151/iqid-alphas')

import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from iqid_alphas.core.alignment import (
            EnhancedAligner, 
            AlignmentMethod, 
            ReferenceStrategy,
            FeatureBasedAligner,
            IntensityBasedAligner,
            PhaseCorrelationAligner,
            calculate_ssd,
            create_affine_transform
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the alignment package."""
    print("\nTesting basic functionality...")
    
    try:
        from iqid_alphas.core.alignment import EnhancedAligner, AlignmentMethod
        
        # Create test images
        reference = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        moving = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Create aligner
        aligner = EnhancedAligner()
        print(f"✅ EnhancedAligner created (default method: {aligner.default_method.value})")
        
        # Test single slice alignment
        result = aligner.align_slice(moving, reference, AlignmentMethod.FEATURE_BASED)
        print(f"✅ Single slice alignment completed (confidence: {result.confidence:.3f})")
        
        # Test stack alignment
        image_stack = np.random.randint(0, 255, (5, 100, 100), dtype=np.uint8)
        stack_result = aligner.align_stack(image_stack)
        print(f"✅ Stack alignment completed ({len(stack_result.aligned_images)} images)")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_individual_algorithms():
    """Test individual alignment algorithms."""
    print("\nTesting individual algorithms...")
    
    try:
        from iqid_alphas.core.alignment import (
            FeatureBasedAligner, 
            IntensityBasedAligner, 
            PhaseCorrelationAligner
        )
        
        # Create test images
        reference = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        moving = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        # Test feature-based alignment
        feature_aligner = FeatureBasedAligner()
        result = feature_aligner.align(moving, reference)
        print(f"✅ Feature-based alignment: {result.method.value} (confidence: {result.confidence:.3f})")
        
        # Test intensity-based alignment  
        intensity_aligner = IntensityBasedAligner()
        result = intensity_aligner.align(moving, reference)
        print(f"✅ Intensity-based alignment: {result.method.value} (confidence: {result.confidence:.3f})")
        
        # Test phase correlation alignment
        phase_aligner = PhaseCorrelationAligner()
        result = phase_aligner.align(moving, reference)
        print(f"✅ Phase correlation alignment: {result.method.value} (confidence: {result.confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Algorithm test failed: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from iqid_alphas.core.alignment import calculate_ssd, create_affine_transform
        
        # Test SSD calculation
        img1 = np.ones((10, 10))
        img2 = np.ones((10, 10)) * 2
        ssd = calculate_ssd(img1, img2)
        print(f"✅ SSD calculation: {ssd}")
        
        # Test affine transform creation
        transform = create_affine_transform(rotation=45, translation=(10, 20))
        print(f"✅ Affine transform created: shape {transform.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MODULAR ALIGNMENT PACKAGE VALIDATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_individual_algorithms,
        test_utilities
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"✅ Imports: {'PASS' if results[0] else 'FAIL'}")
    print(f"✅ Basic functionality: {'PASS' if results[1] else 'FAIL'}")
    print(f"✅ Individual algorithms: {'PASS' if results[2] else 'FAIL'}")
    print(f"✅ Utilities: {'PASS' if results[3] else 'FAIL'}")
    print("=" * 60)
    
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("✅ Modular alignment package is working correctly")
        print("✅ Phase 1 Task 3 (slice alignment migration) COMPLETED")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
