#!/usr/bin/env python3
"""
Focused UCSF Pipeline Test
Quick test of specific pipeline components with UCSF data.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from tifffile import imread

# Add to path
sys.path.insert(0, '/home/wxc151/iqid-alphas')

def test_iqid_processor():
    """Test IQIDProcessor with UCSF data."""
    print("🧪 Testing IQIDProcessor with UCSF Data")
    
    # Load UCSF data info
    discovery_file = '/home/wxc151/iqid-alphas/evaluation/reports/simple_ucsf_evaluation.json'
    
    if not os.path.exists(discovery_file):
        print("❌ No UCSF discovery data found")
        return False
    
    with open(discovery_file, 'r') as f:
        data = json.load(f)
        discovery = data['discovery']
    
    # Get a test file
    if not discovery['sample_files']:
        print("❌ No sample files found")
        return False
    
    sample_info = discovery['sample_files'][0]
    test_file = os.path.join(sample_info['path'], sample_info['sample_files'][0])
    
    print(f"  📁 Test file: {test_file}")
    
    # Test basic loading
    try:
        img_array = imread(test_file)
        print(f"  ✅ Image loaded: {img_array.shape} {img_array.dtype}")
    except Exception as e:
        print(f"  ❌ Image loading failed: {e}")
        return False
    
    # Test IQIDProcessor
    try:
        from iqid_alphas.core.processor import IQIDProcessor
        processor = IQIDProcessor()
        print("  ✅ IQIDProcessor imported and created")
        
        # Test processing
        result = processor.process_image(img_array)
        
        if result is not None:
            print(f"  ✅ Processing successful: {type(result)}")
            return True
        else:
            print("  ⚠️ Processing returned None")
            return False
            
    except Exception as e:
        print(f"  ❌ IQIDProcessor failed: {e}")
        return False

def test_simple_pipeline():
    """Test SimplePipeline with UCSF data."""
    print("\n🔧 Testing SimplePipeline with UCSF Data")
    
    # Load discovery data
    discovery_file = '/home/wxc151/iqid-alphas/evaluation/reports/simple_ucsf_evaluation.json'
    with open(discovery_file, 'r') as f:
        data = json.load(f)
        discovery = data['discovery']
    
    sample_info = discovery['sample_files'][0]
    test_file = os.path.join(sample_info['path'], sample_info['sample_files'][0])
    img_array = imread(test_file)
    
    try:
        from iqid_alphas.pipelines.simple import SimplePipeline
        pipeline = SimplePipeline()
        print("  ✅ SimplePipeline imported and created")
        
        # Check available methods
        methods = [method for method in dir(pipeline) if not method.startswith('_')]
        print(f"  📋 Available methods: {methods}")
        
        # Try different processing methods
        result = None
        
        if hasattr(pipeline, 'process'):
            try:
                result = pipeline.process(img_array)
                print("  ✅ pipeline.process() worked")
            except Exception as e:
                print(f"  ❌ pipeline.process() failed: {e}")
        
        if result is None and hasattr(pipeline, 'process_iqid_image'):
            try:
                result = pipeline.process_iqid_image(img_array)
                print("  ✅ pipeline.process_iqid_image() worked")
            except Exception as e:
                print(f"  ❌ pipeline.process_iqid_image() failed: {e}")
        
        if result is None and hasattr(pipeline, 'run'):
            try:
                result = pipeline.run({'iqid_image': img_array})
                print("  ✅ pipeline.run() worked")
            except Exception as e:
                print(f"  ❌ pipeline.run() failed: {e}")
        
        if result is not None:
            print(f"  ✅ SimplePipeline processing successful: {type(result)}")
            return True
        else:
            print("  ❌ All SimplePipeline methods failed or returned None")
            return False
            
    except Exception as e:
        print(f"  ❌ SimplePipeline failed: {e}")
        return False

def main():
    """Run focused tests."""
    print("🚀 Focused UCSF Pipeline Testing")
    print("=" * 40)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Test IQIDProcessor
    processor_success = test_iqid_processor()
    results['tests']['iqid_processor'] = processor_success
    
    # Test SimplePipeline
    pipeline_success = test_simple_pipeline()
    results['tests']['simple_pipeline'] = pipeline_success
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    print(f"IQIDProcessor: {'✅ PASS' if processor_success else '❌ FAIL'}")
    print(f"SimplePipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    
    overall_success = processor_success and pipeline_success
    print(f"\nOverall: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    # Save results
    results_path = '/home/wxc151/iqid-alphas/evaluation/reports/focused_ucsf_test.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return overall_success

if __name__ == "__main__":
    main()
