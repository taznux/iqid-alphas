#!/usr/bin/env python3
"""
Smart Image Loader with Format Detection
Automatically detects actual file format regardless of extension and uses appropriate loader.
"""

import os
import numpy as np
from PIL import Image
import tifffile
from skimage import io as skimage_io
import warnings

class SmartImageLoader:
    """Smart image loader that detects actual file format regardless of extension."""
    
    def __init__(self):
        # File format signatures (magic numbers)
        self.format_signatures = {
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'\xff\xd8\xff': 'JPEG',
            b'GIF8': 'GIF',
            b'BM': 'BMP',
            b'II*\x00': 'TIFF_LE',  # Little-endian TIFF
            b'MM\x00*': 'TIFF_BE',  # Big-endian TIFF
            b'RIFF': 'WEBP',  # Could be WEBP (need to check further)
        }
    
    def detect_file_format(self, file_path):
        """Detect actual file format by reading file header."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)  # Read first 16 bytes
            
            # Check against known signatures
            for signature, format_name in self.format_signatures.items():
                if header.startswith(signature):
                    return format_name
            
            # Special case for WEBP (needs more specific check)
            if header.startswith(b'RIFF') and b'WEBP' in header:
                return 'WEBP'
            
            return 'UNKNOWN'
            
        except Exception as e:
            print(f"Error detecting format for {file_path}: {e}")
            return 'ERROR'
    
    def load_image_smart(self, file_path):
        """Load image using smart format detection."""
        actual_format = self.detect_file_format(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        print(f"File: {os.path.basename(file_path)}")
        print(f"  Extension: {file_extension}")
        print(f"  Actual format: {actual_format}")
        
        # Strategy 1: Try format-specific loader based on detected format
        if actual_format in ['PNG', 'JPEG', 'GIF', 'BMP', 'WEBP']:
            try:
                img = Image.open(file_path)
                img_array = np.array(img)
                print(f"  ✅ PIL loader (format-aware): {img_array.shape} {img_array.dtype}")
                return img_array
            except Exception as e:
                print(f"  ❌ PIL loader failed: {e}")
        
        elif actual_format in ['TIFF_LE', 'TIFF_BE']:
            try:
                img_array = tifffile.imread(file_path)
                print(f"  ✅ tifffile loader (TIFF-specific): {img_array.shape} {img_array.dtype}")
                return img_array
            except Exception as e:
                print(f"  ❌ tifffile loader failed: {e}")
        
        # Strategy 2: Fallback to PIL (most versatile)
        try:
            img = Image.open(file_path)
            img_array = np.array(img)
            print(f"  ✅ PIL fallback loader: {img_array.shape} {img_array.dtype}")
            return img_array
        except Exception as e:
            print(f"  ❌ PIL fallback failed: {e}")
        
        # Strategy 3: Last resort - skimage
        try:
            img_array = skimage_io.imread(file_path)
            print(f"  ✅ skimage fallback loader: {img_array.shape} {img_array.dtype}")
            return img_array
        except Exception as e:
            print(f"  ❌ skimage fallback failed: {e}")
        
        print(f"  ❌ All loaders failed for {file_path}")
        return None
    
    def load_with_format_correction(self, file_path):
        """Load image with automatic format correction."""
        actual_format = self.detect_file_format(file_path)
        
        # If format mismatch detected, suggest correction
        file_extension = os.path.splitext(file_path)[1].lower()
        expected_extensions = {
            'PNG': '.png',
            'JPEG': '.jpg',
            'GIF': '.gif',
            'BMP': '.bmp',
            'TIFF_LE': '.tif',
            'TIFF_BE': '.tif',
            'WEBP': '.webp'
        }
        
        if actual_format in expected_extensions:
            expected_ext = expected_extensions[actual_format]
            if file_extension != expected_ext:
                print(f"  ⚠️ Format mismatch: {file_extension} extension but {actual_format} format")
                print(f"     Recommended: rename to {expected_ext}")
        
        return self.load_image_smart(file_path)

def test_extension_mismatch_scenarios():
    """Test various extension mismatch scenarios."""
    print("🧪 Testing Extension Mismatch Scenarios")
    print("=" * 50)
    
    loader = SmartImageLoader()
    
    # Create test directory
    test_dir = "/tmp/extension_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test images with mismatched extensions
    test_scenarios = []
    
    # Scenario 1: PNG with .tif extension
    try:
        from PIL import Image
        # Create a small test PNG
        img = Image.new('RGB', (50, 50), color='red')
        png_as_tif = os.path.join(test_dir, "fake_tiff.tif")
        img.save(png_as_tif, "PNG")  # Save as PNG but with .tif extension
        test_scenarios.append(("PNG as .tif", png_as_tif))
        print(f"Created: {png_as_tif} (PNG data with .tif extension)")
    except Exception as e:
        print(f"Could not create PNG test file: {e}")
    
    # Scenario 2: JPEG with .png extension  
    try:
        img = Image.new('RGB', (50, 50), color='blue')
        jpeg_as_png = os.path.join(test_dir, "fake_png.png")
        img.save(jpeg_as_png, "JPEG")  # Save as JPEG but with .png extension
        test_scenarios.append(("JPEG as .png", jpeg_as_png))
        print(f"Created: {jpeg_as_png} (JPEG data with .png extension)")
    except Exception as e:
        print(f"Could not create JPEG test file: {e}")
    
    # Test each scenario
    print("\n🔍 Testing Smart Loading...")
    for scenario_name, file_path in test_scenarios:
        print(f"\n--- {scenario_name} ---")
        result = loader.load_with_format_correction(file_path)
        
        if result is not None:
            print(f"  ✅ Successfully loaded: {result.shape} {result.dtype}")
        else:
            print(f"  ❌ Failed to load")
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test directory: {test_dir}")
    except:
        pass

def enhance_iqid_processor_with_smart_loading():
    """Show how to enhance IQIDProcessor with smart loading."""
    print("\n🔧 Enhanced IQIDProcessor Example")
    print("=" * 40)
    
    enhanced_code = '''
# Enhanced IQIDProcessor with smart image loading
class EnhancedIQIDProcessor(IQIDProcessor):
    def __init__(self):
        super().__init__()
        self.smart_loader = SmartImageLoader()
    
    def load_image(self, image_path):
        """Load image with smart format detection."""
        try:
            # Use smart loader instead of direct tifffile.imread
            img_array = self.smart_loader.load_image_smart(image_path)
            
            if img_array is not None:
                return self._ensure_float_array(img_array)
            else:
                raise ValueError(f"Could not load image: {image_path}")
                
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def _ensure_float_array(self, img_array):
        """Ensure image array is in proper format for processing."""
        # Convert to float if needed
        if img_array.dtype == np.uint8:
            return img_array.astype(np.float64) / 255.0
        elif img_array.dtype == np.uint16:
            return img_array.astype(np.float64) / 65535.0
        else:
            return img_array.astype(np.float64)
'''
    
    print(enhanced_code)

def test_with_real_ucsf_files():
    """Test smart loading with real UCSF files."""
    print("\n🔬 Testing with Real UCSF Files")
    print("=" * 35)
    
    loader = SmartImageLoader()
    
    # Test with a real UCSF file
    ucsf_file = "/home/wxc151/iqid-alphas/data/DataPush1/iQID/Sequential/kidneys/D1M1_L/mBq_corr_11.tif"
    
    if os.path.exists(ucsf_file):
        print(f"Testing real UCSF file: {os.path.basename(ucsf_file)}")
        result = loader.load_with_format_correction(ucsf_file)
        
        if result is not None:
            print(f"✅ Real UCSF file loaded successfully")
            print(f"   Shape: {result.shape}")
            print(f"   Data type: {result.dtype}")
            print(f"   Value range: {np.min(result):.6f} to {np.max(result):.6f}")
        else:
            print("❌ Failed to load real UCSF file")
    else:
        print(f"UCSF file not found: {ucsf_file}")

def main():
    """Main function to run all tests."""
    print("🚀 Smart Image Loader - Extension Mismatch Handler")
    print("=" * 55)
    
    # Test extension mismatch scenarios
    test_extension_mismatch_scenarios()
    
    # Show enhanced processor code
    enhance_iqid_processor_with_smart_loading()
    
    # Test with real files
    test_with_real_ucsf_files()
    
    print("\n" + "=" * 55)
    print("✅ Smart image loading tests complete!")
    print("\nKey Benefits:")
    print("- Automatically detects actual file format")
    print("- Uses appropriate loader for each format")
    print("- Handles extension mismatches gracefully")
    print("- Provides helpful format correction suggestions")

if __name__ == "__main__":
    main()
