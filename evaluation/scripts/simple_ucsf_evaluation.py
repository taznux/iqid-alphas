#!/usr/bin/env python3
"""
Simple UCSF Data Evaluation
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

def main():
    print("🔍 UCSF Pipeline Evaluation")
    print("=" * 50)
    
    # Check data paths
    base_path = '/home/wxc151/iqid-alphas'
    data_path = f"{base_path}/data"
    
    print(f"Base path: {base_path}")
    print(f"Data path: {data_path}")
    print(f"Data path exists: {os.path.exists(data_path)}")
    
    if not os.path.exists(data_path):
        print("❌ UCSF data not found!")
        return
    
    print(f"Data contents: {os.listdir(data_path)}")
    
    # Discover UCSF structure
    discovery = {
        'total_files': 0,
        'iqid_dirs': 0,
        'he_dirs': 0,
        'sample_files': []
    }
    
    # Check DataPush1/iQID
    iqid_path = f"{data_path}/DataPush1/iQID"
    if os.path.exists(iqid_path):
        print(f"\n📁 Scanning {iqid_path}")
        for root, dirs, files in os.walk(iqid_path):
            if files:
                tif_files = [f for f in files if f.endswith('.tif')]
                if tif_files:
                    discovery['iqid_dirs'] += 1
                    discovery['total_files'] += len(tif_files)
                    discovery['sample_files'].append({
                        'path': root,
                        'file_count': len(tif_files),
                        'sample_files': tif_files[:3]  # First 3 for reference
                    })
                    print(f"  Found {len(tif_files)} files in {os.path.relpath(root, iqid_path)}")
    
    # Check DataPush1/HE
    he_path = f"{data_path}/DataPush1/HE"
    if os.path.exists(he_path):
        print(f"\n📁 Scanning {he_path}")
        for root, dirs, files in os.walk(he_path):
            if files:
                tif_files = [f for f in files if f.endswith('.tif')]
                if tif_files:
                    discovery['he_dirs'] += 1
                    discovery['total_files'] += len(tif_files)
                    print(f"  Found {len(tif_files)} files in {os.path.relpath(root, he_path)}")
    
    # Summary
    print(f"\n📊 Discovery Summary:")
    print(f"  Total files: {discovery['total_files']}")
    print(f"  iQID directories: {discovery['iqid_dirs']}")
    print(f"  H&E directories: {discovery['he_dirs']}")
    
    # Test data loading
    print(f"\n🧪 Testing Data Loading...")
    
    if discovery['sample_files']:
        test_sample = discovery['sample_files'][0]
        sample_path = os.path.join(test_sample['path'], test_sample['sample_files'][0])
        
        try:
            # Try to import required libraries
            from tifffile import imread
            import numpy as np
            
            print(f"  Loading: {os.path.basename(sample_path)}")
            img_array = imread(sample_path)
            
            print(f"  ✅ Loaded successfully!")
            print(f"  Shape: {img_array.shape}")
            print(f"  Dtype: {img_array.dtype}")
            print(f"  Size: {img_array.nbytes / 1024 / 1024:.1f} MB")
            print(f"  Range: {np.min(img_array):.3f} to {np.max(img_array):.3f}")
            
        except ImportError as e:
            print(f"  ❌ Import error: {e}")
        except Exception as e:
            print(f"  ❌ Loading error: {e}")
    
    # Test pipeline imports
    print(f"\n🔧 Testing Pipeline Imports...")
    
    sys.path.insert(0, base_path)
    
    try:
        from iqid_alphas.pipelines.simple import SimplePipeline
        print("  ✅ SimplePipeline imported")
    except Exception as e:
        print(f"  ❌ SimplePipeline import failed: {e}")
    
    try:
        from iqid_alphas.core.processor import IQIDProcessor
        print("  ✅ IQIDProcessor imported")
    except Exception as e:
        print(f"  ❌ IQIDProcessor import failed: {e}")
    
    try:
        from iqid_alphas.core.alignment import ImageAligner
        print("  ✅ ImageAligner imported")
    except Exception as e:
        print(f"  ❌ ImageAligner import failed: {e}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'discovery': discovery,
        'status': 'completed'
    }
    
    results_path = f"{base_path}/evaluation/reports/simple_ucsf_evaluation.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Simple evaluation completed!")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
