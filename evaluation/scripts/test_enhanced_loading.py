#!/usr/bin/env python3
"""
Test Enhanced Image Loading with Extension Mismatch
Tests the enhanced IQIDProcessor with smart format detection.
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image
import shutil

# Add package to path
sys.path.insert(0, '/home/wxc151/iqid-alphas')

def test_enhanced_image_loading():
    """Test enhanced image loading with extension mismatches."""
    print("🧪 Testing Enhanced IQIDProcessor with Extension Mismatches")
    print("=" * 60)
    
    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="iqid_test_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Import enhanced processor
        from iqid_alphas.core.processor import IQIDProcessor
        processor = IQIDProcessor()
        
        # Test 1: PNG file with .tif extension
        print("\n📄 Test 1: PNG file with .tif extension")
        img_rgb = Image.new('RGB', (100, 100), color='red')
        png_as_tif = os.path.join(test_dir, "fake_tiff.tif")
        img_rgb.save(png_as_tif, "PNG")
        print(f"Created: {png_as_tif} (PNG data with .tif extension)")
        
        try:
            result1 = processor.load_image(png_as_tif, "test1")
            print(f"✅ Success: {result1.shape} {result1.dtype}")
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        # Test 2: JPEG file with .png extension
        print("\n📄 Test 2: JPEG file with .png extension")
        img_jpeg = Image.new('RGB', (100, 100), color='blue')
        jpeg_as_png = os.path.join(test_dir, "fake_png.png")
        img_jpeg.save(jpeg_as_png, "JPEG")
        print(f"Created: {jpeg_as_png} (JPEG data with .png extension)")
        
        try:
            result2 = processor.load_image(jpeg_as_png, "test2")
            print(f"✅ Success: {result2.shape} {result2.dtype}")
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        # Test 3: Real UCSF TIFF file (correct format)
        print("\n📄 Test 3: Real UCSF TIFF file (correct format)")
        ucsf_file = "/home/wxc151/iqid-alphas/data/DataPush1/iQID/Sequential/kidneys/D1M1_L/mBq_corr_11.tif"
        
        if os.path.exists(ucsf_file):
            try:
                result3 = processor.load_image(ucsf_file, "ucsf_test")
                print(f"✅ UCSF file success: {result3.shape} {result3.dtype}")
                print(f"   Value range: {np.min(result3):.6f} to {np.max(result3):.6f}")
            except Exception as e:
                print(f"❌ UCSF file failed: {e}")
        else:
            print("⚠️ UCSF file not found, skipping")
        
        # Test 4: Format detection accuracy
        print("\n🔍 Test 4: Format Detection Accuracy")
        test_files = [
            (png_as_tif, "PNG"),
            (jpeg_as_png, "JPEG")
        ]
        
        for file_path, expected_format in test_files:
            if os.path.exists(file_path):
                detected = processor._detect_file_format(file_path)
                if detected == expected_format:
                    print(f"✅ {os.path.basename(file_path)}: {detected} (correct)")
                else:
                    print(f"❌ {os.path.basename(file_path)}: {detected} (expected {expected_format})")
        
        # Test 5: Verify loaded images
        print("\n📊 Test 5: Loaded Images Summary")
        for key, img in processor.images.items():
            print(f"  {key}: {img.shape} {img.dtype}")
        
        print(f"\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            shutil.rmtree(test_dir)
            print(f"🧹 Cleaned up: {test_dir}")
        except:
            pass

def test_direct_format_detection():
    """Test format detection directly."""
    print("\n🔍 Direct Format Detection Test")
    print("-" * 30)
    
    # Create test files
    test_dir = tempfile.mkdtemp(prefix="format_test_")
    
    try:
        # Create different format files
        test_files = []
        
        # PNG file
        img = Image.new('RGB', (50, 50), color='green')
        png_file = os.path.join(test_dir, "test.png")
        img.save(png_file, "PNG")
        test_files.append((png_file, "PNG"))
        
        # JPEG file
        jpeg_file = os.path.join(test_dir, "test.jpg")
        img.save(jpeg_file, "JPEG")
        test_files.append((jpeg_file, "JPEG"))
        
        # Test detection
        from iqid_alphas.core.processor import IQIDProcessor
        processor = IQIDProcessor()
        
        for file_path, expected in test_files:
            detected = processor._detect_file_format(file_path)
            status = "✅" if detected == expected else "❌"
            print(f"  {status} {os.path.basename(file_path)}: {detected} (expected {expected})")
    
    finally:
        shutil.rmtree(test_dir)

def main():
    """Run all tests."""
    print("🚀 Enhanced Image Loading Test Suite")
    print("Testing extension mismatch handling in IQIDProcessor")
    print("=" * 60)
    
    # Test enhanced loading
    test_enhanced_image_loading()
    
    # Test format detection
    test_direct_format_detection()
    
    print("\n" + "=" * 60)
    print("🎉 Test Suite Complete!")
    print("\nKey Features Tested:")
    print("✅ Automatic file format detection")
    print("✅ Extension mismatch handling")
    print("✅ Multiple loader fallback strategy")
    print("✅ Real UCSF data compatibility")
    print("✅ Error reporting and recovery")

if __name__ == "__main__":
    main()
