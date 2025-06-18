#!/usr/bin/env python3
"""
Extension Mismatch Testing for IQID-Alphas Pipeline
Tests pipeline robustness with files that have incorrect extensions.
"""

import os
import sys
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime
import json

# Add package to path
sys.path.insert(0, '/home/wxc151/iqid-alphas')

try:
    from skimage import io
    from PIL import Image
    import matplotlib.pyplot as plt
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False
    print("Warning: Imaging libraries not available")

class ExtensionMismatchTester:
    """Test pipeline behavior with incorrect file extensions."""
    
    def __init__(self):
        self.test_dir = '/home/wxc151/iqid-alphas/evaluation/outputs/extension_mismatch_test'
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'extension_mismatch_evaluation',
            'test_files': {},
            'loading_results': {},
            'pipeline_results': {}
        }
        
        # Create test directory
        os.makedirs(self.test_dir, exist_ok=True)
    
    def create_test_files(self):
        """Create test files with mismatched extensions."""
        print("🔧 Creating test files with mismatched extensions...")
        
        test_files = {}
        
        # Create sample data
        # 1. PNG data with .tif extension
        png_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        png_file = f"{self.test_dir}/png_as_tif.tif"
        Image.fromarray(png_data).save(png_file, 'PNG')  # Save as PNG but with .tif extension
        test_files['png_as_tif'] = {
            'file': png_file,
            'actual_format': 'PNG',
            'extension': '.tif',
            'description': 'PNG image saved with .tif extension'
        }
        
        # 2. JPEG data with .tif extension  
        jpeg_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        jpeg_file = f"{self.test_dir}/jpeg_as_tif.tif"
        Image.fromarray(jpeg_data).save(jpeg_file, 'JPEG')  # Save as JPEG but with .tif extension
        test_files['jpeg_as_tif'] = {
            'file': jpeg_file,
            'actual_format': 'JPEG',
            'extension': '.tif',
            'description': 'JPEG image saved with .tif extension'
        }
        
        # 3. TIFF data with .png extension
        tiff_data = np.random.rand(100, 100).astype(np.float32)
        tiff_file = f"{self.test_dir}/tiff_as_png.png"
        io.imsave(tiff_file, tiff_data, check_contrast=False)  # This will warn but still save
        # Rename to create mismatch
        actual_tiff_file = f"{self.test_dir}/tiff_as_png_actual.png"
        shutil.copy(tiff_file, actual_tiff_file)
        test_files['tiff_as_png'] = {
            'file': actual_tiff_file,
            'actual_format': 'TIFF-like',
            'extension': '.png',
            'description': 'TIFF-like data saved with .png extension'
        }
        
        # 4. Binary data with .tif extension (complete mismatch)
        binary_file = f"{self.test_dir}/binary_as_tif.tif"
        with open(binary_file, 'wb') as f:
            f.write(b'This is not an image file at all, just binary data that looks like text.')
        test_files['binary_as_tif'] = {
            'file': binary_file,
            'actual_format': 'Binary/Text',
            'extension': '.tif',
            'description': 'Non-image binary data with .tif extension'
        }
        
        # 5. Real UCSF file copied with wrong extension
        ucsf_source = '/home/wxc151/iqid-alphas/data/DataPush1/iQID/Sequential/kidneys/D1M1_L/mBq_corr_11.tif'
        if os.path.exists(ucsf_source):
            ucsf_wrong_ext = f"{self.test_dir}/ucsf_real_as_png.png"
            shutil.copy(ucsf_source, ucsf_wrong_ext)
            test_files['ucsf_as_png'] = {
                'file': ucsf_wrong_ext,
                'actual_format': 'Scientific TIFF',
                'extension': '.png',
                'description': 'Real UCSF scientific TIFF with .png extension'
            }
        
        self.results['test_files'] = test_files
        print(f"   ✅ Created {len(test_files)} test files with mismatched extensions")
        return test_files
    
    def test_basic_loading(self, test_files):
        """Test basic image loading with different loaders."""
        print("\n🔍 Testing basic image loading capabilities...")
        
        loading_results = {}
        
        for test_name, file_info in test_files.items():
            file_path = file_info['file']
            print(f"  Testing: {test_name} ({file_info['description']})")
            
            results = {
                'file_info': file_info,
                'skimage_imread': {'success': False, 'error': None, 'shape': None, 'dtype': None},
                'pil_open': {'success': False, 'error': None, 'size': None, 'mode': None},
                'tifffile_imread': {'success': False, 'error': None, 'shape': None, 'dtype': None}
            }
            
            # Test 1: skimage.io.imread (current pipeline default)
            try:
                img_skimage = io.imread(file_path)
                results['skimage_imread']['success'] = True
                results['skimage_imread']['shape'] = img_skimage.shape
                results['skimage_imread']['dtype'] = str(img_skimage.dtype)
                print(f"    ✅ skimage.imread: {img_skimage.shape} {img_skimage.dtype}")
            except Exception as e:
                results['skimage_imread']['error'] = str(e)
                print(f"    ❌ skimage.imread: {e}")
            
            # Test 2: PIL Image.open
            try:
                from PIL import Image
                img_pil = Image.open(file_path)
                results['pil_open']['success'] = True
                results['pil_open']['size'] = img_pil.size
                results['pil_open']['mode'] = img_pil.mode
                print(f"    ✅ PIL.Image.open: {img_pil.size} {img_pil.mode}")
            except Exception as e:
                results['pil_open']['error'] = str(e)
                print(f"    ❌ PIL.Image.open: {e}")
            
            # Test 3: tifffile.imread (for scientific TIFF)
            try:
                from tifffile import imread
                img_tifffile = imread(file_path)
                results['tifffile_imread']['success'] = True
                results['tifffile_imread']['shape'] = img_tifffile.shape
                results['tifffile_imread']['dtype'] = str(img_tifffile.dtype)
                print(f"    ✅ tifffile.imread: {img_tifffile.shape} {img_tifffile.dtype}")
            except Exception as e:
                results['tifffile_imread']['error'] = str(e)
                print(f"    ❌ tifffile.imread: {e}")
            
            loading_results[test_name] = results
        
        self.results['loading_results'] = loading_results
        return loading_results
    
    def test_iqid_processor(self, test_files):
        """Test IQIDProcessor with mismatched extensions."""
        print("\n🔧 Testing IQIDProcessor with mismatched extensions...")
        
        processor_results = {}
        
        try:
            from iqid_alphas.core.processor import IQIDProcessor
            processor = IQIDProcessor()
            print("  ✅ IQIDProcessor initialized successfully")
            
            for test_name, file_info in test_files.items():
                file_path = file_info['file']
                print(f"  Testing IQIDProcessor with: {test_name}")
                
                result = {
                    'file_info': file_info,
                    'load_success': False,
                    'load_error': None,
                    'process_success': False,
                    'process_error': None,
                    'image_shape': None,
                    'image_dtype': None
                }
                
                try:
                    # Test loading
                    img_array = processor.load_image(file_path, key=test_name)
                    result['load_success'] = True
                    result['image_shape'] = img_array.shape
                    result['image_dtype'] = str(img_array.dtype)
                    print(f"    ✅ Loaded: {img_array.shape} {img_array.dtype}")
                    
                    # Test processing
                    try:
                        processed = processor.process_image(img_array)
                        result['process_success'] = True
                        print(f"    ✅ Processed successfully")
                    except Exception as e:
                        result['process_error'] = str(e)
                        print(f"    ⚠️ Processing failed: {e}")
                    
                except Exception as e:
                    result['load_error'] = str(e)
                    print(f"    ❌ Loading failed: {e}")
                
                processor_results[test_name] = result
                
        except Exception as e:
            print(f"  ❌ IQIDProcessor initialization failed: {e}")
            processor_results['initialization_error'] = str(e)
        
        self.results['pipeline_results']['iqid_processor'] = processor_results
        return processor_results
    
    def test_simple_pipeline(self, test_files):
        """Test SimplePipeline with mismatched extensions."""
        print("\n🔧 Testing SimplePipeline with mismatched extensions...")
        
        pipeline_results = {}
        
        try:
            from iqid_alphas.pipelines.simple import SimplePipeline
            pipeline = SimplePipeline()
            print("  ✅ SimplePipeline initialized successfully")
            
            for test_name, file_info in test_files.items():
                file_path = file_info['file']
                print(f"  Testing SimplePipeline with: {test_name}")
                
                result = {
                    'file_info': file_info,
                    'load_success': False,
                    'load_error': None,
                    'pipeline_success': False,
                    'pipeline_error': None
                }
                
                try:
                    # First try to load the image
                    img_array = io.imread(file_path)
                    result['load_success'] = True
                    print(f"    ✅ Image loaded: {img_array.shape}")
                    
                    # Test pipeline processing
                    try:
                        # Try different pipeline methods
                        pipeline_result = None
                        
                        if hasattr(pipeline, 'process'):
                            pipeline_result = pipeline.process(img_array)
                        elif hasattr(pipeline, 'process_iqid_image'):
                            pipeline_result = pipeline.process_iqid_image(img_array)
                        
                        if pipeline_result is not None:
                            result['pipeline_success'] = True
                            print(f"    ✅ Pipeline processing successful")
                        else:
                            result['pipeline_error'] = "Pipeline returned None"
                            print(f"    ⚠️ Pipeline returned None")
                    
                    except Exception as e:
                        result['pipeline_error'] = str(e)
                        print(f"    ❌ Pipeline processing failed: {e}")
                
                except Exception as e:
                    result['load_error'] = str(e)
                    print(f"    ❌ Image loading failed: {e}")
                
                pipeline_results[test_name] = result
                
        except Exception as e:
            print(f"  ❌ SimplePipeline initialization failed: {e}")
            pipeline_results['initialization_error'] = str(e)
        
        self.results['pipeline_results']['simple_pipeline'] = pipeline_results
        return pipeline_results
    
    def create_robust_loader(self):
        """Create an enhanced image loader that handles extension mismatches."""
        print("\n🛠️ Creating robust image loader...")
        
        robust_loader_code = '''
def robust_image_loader(file_path):
    """
    Robust image loader that tries multiple methods and ignores extensions.
    
    Parameters
    ----------
    file_path : str
        Path to image file (extension may be incorrect)
        
    Returns
    -------
    np.ndarray
        Loaded image array
    dict
        Metadata about successful loading method
    """
    import os
    import numpy as np
    from pathlib import Path
    
    methods = []
    
    # Method 1: tifffile (best for scientific data)
    try:
        from tifffile import imread as tiff_imread
        img = tiff_imread(file_path)
        return img, {'method': 'tifffile', 'shape': img.shape, 'dtype': str(img.dtype)}
    except Exception as e:
        methods.append(('tifffile', str(e)))
    
    # Method 2: skimage (general purpose)
    try:
        from skimage import io
        img = io.imread(file_path)
        return img, {'method': 'skimage', 'shape': img.shape, 'dtype': str(img.dtype)}
    except Exception as e:
        methods.append(('skimage', str(e)))
    
    # Method 3: PIL with format detection
    try:
        from PIL import Image
        img_pil = Image.open(file_path)
        img = np.array(img_pil)
        return img, {'method': 'PIL', 'shape': img.shape, 'dtype': str(img.dtype), 'original_mode': img_pil.mode}
    except Exception as e:
        methods.append(('PIL', str(e)))
    
    # Method 4: OpenCV (if available)
    try:
        import cv2
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img, {'method': 'opencv', 'shape': img.shape, 'dtype': str(img.dtype)}
        else:
            methods.append(('opencv', 'imread returned None'))
    except Exception as e:
        methods.append(('opencv', str(e)))
    
    # Method 5: Raw binary reading with format detection
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
            
        # Check for common image signatures
        if header.startswith(b'\\x89PNG'):
            from PIL import Image
            img_pil = Image.open(file_path)
            img = np.array(img_pil)
            return img, {'method': 'PIL_forced_PNG', 'shape': img.shape, 'dtype': str(img.dtype)}
        elif header.startswith(b'\\xff\\xd8\\xff'):
            from PIL import Image
            img_pil = Image.open(file_path)
            img = np.array(img_pil)
            return img, {'method': 'PIL_forced_JPEG', 'shape': img.shape, 'dtype': str(img.dtype)}
        elif header.startswith(b'II*') or header.startswith(b'MM\\x00*'):
            from tifffile import imread as tiff_imread
            img = tiff_imread(file_path)
            return img, {'method': 'tifffile_forced', 'shape': img.shape, 'dtype': str(img.dtype)}
            
    except Exception as e:
        methods.append(('format_detection', str(e)))
    
    # All methods failed
    raise ValueError(f"Could not load image from {file_path}. Tried methods: {methods}")
'''
        
        # Save the robust loader
        loader_file = f"{self.test_dir}/robust_image_loader.py"
        with open(loader_file, 'w') as f:
            f.write(robust_loader_code)
        
        print(f"  ✅ Robust loader saved to: {loader_file}")
        return loader_file
    
    def test_robust_loader(self, test_files):
        """Test the robust image loader."""
        print("\n🧪 Testing robust image loader...")
        
        # Create and import robust loader
        loader_file = self.create_robust_loader()
        
        robust_results = {}
        
        for test_name, file_info in test_files.items():
            file_path = file_info['file']
            print(f"  Testing robust loader with: {test_name}")
            
            try:
                # Execute the robust loader function
                exec(open(f"{self.test_dir}/robust_image_loader.py").read())
                img, metadata = locals()['robust_image_loader'](file_path)
                
                robust_results[test_name] = {
                    'success': True,
                    'metadata': metadata,
                    'file_info': file_info
                }
                print(f"    ✅ Success: {metadata['method']} - {metadata['shape']} {metadata['dtype']}")
                
            except Exception as e:
                robust_results[test_name] = {
                    'success': False,
                    'error': str(e),
                    'file_info': file_info
                }
                print(f"    ❌ Failed: {e}")
        
        self.results['robust_loader_results'] = robust_results
        return robust_results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n📄 Generating extension mismatch evaluation report...")
        
        # Calculate success rates
        summary = {
            'total_test_files': len(self.results.get('test_files', {})),
            'loading_methods': {},
            'pipeline_components': {},
            'robust_loader': {},
            'recommendations': []
        }
        
        # Analyze loading results
        if 'loading_results' in self.results:
            for method in ['skimage_imread', 'pil_open', 'tifffile_imread']:
                successes = sum(1 for result in self.results['loading_results'].values() 
                               if result[method]['success'])
                total = len(self.results['loading_results'])
                summary['loading_methods'][method] = {
                    'success_rate': (successes / total) * 100 if total > 0 else 0,
                    'successes': successes,
                    'total': total
                }
        
        # Analyze pipeline results
        if 'pipeline_results' in self.results:
            for component in ['iqid_processor', 'simple_pipeline']:
                if component in self.results['pipeline_results']:
                    component_results = self.results['pipeline_results'][component]
                    if isinstance(component_results, dict):
                        successes = sum(1 for result in component_results.values() 
                                       if isinstance(result, dict) and 
                                       (result.get('load_success', False) or result.get('pipeline_success', False)))
                        total = len([r for r in component_results.values() if isinstance(r, dict)])
                        summary['pipeline_components'][component] = {
                            'success_rate': (successes / total) * 100 if total > 0 else 0,
                            'successes': successes,
                            'total': total
                        }
        
        # Analyze robust loader
        if 'robust_loader_results' in self.results:
            successes = sum(1 for result in self.results['robust_loader_results'].values() 
                           if result.get('success', False))
            total = len(self.results['robust_loader_results'])
            summary['robust_loader'] = {
                'success_rate': (successes / total) * 100 if total > 0 else 0,
                'successes': successes,
                'total': total
            }
        
        # Generate recommendations
        if summary['loading_methods'].get('skimage_imread', {}).get('success_rate', 0) < 100:
            summary['recommendations'].append("Consider implementing fallback loading methods")
        
        if summary['robust_loader'].get('success_rate', 0) > 80:
            summary['recommendations'].append("Robust loader shows promise for handling extension mismatches")
        
        summary['recommendations'].append("Implement file format detection based on content, not extension")
        summary['recommendations'].append("Add comprehensive error handling for malformed files")
        
        # Create detailed report
        report_content = f"""# Extension Mismatch Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Type:** File Extension Mismatch Tolerance
**Total Test Files:** {summary['total_test_files']}

## Executive Summary

This evaluation tests the pipeline's ability to handle files with incorrect extensions,
which is a common issue in medical imaging workflows where files may be renamed or
have inconsistent naming conventions.

## Test Results Summary

### Loading Method Performance
"""
        
        for method, stats in summary['loading_methods'].items():
            status = "✅" if stats['success_rate'] >= 80 else "⚠️" if stats['success_rate'] >= 50 else "❌"
            report_content += f"- **{method}:** {status} {stats['success_rate']:.1f}% ({stats['successes']}/{stats['total']})\n"
        
        report_content += "\n### Pipeline Component Performance\n"
        for component, stats in summary['pipeline_components'].items():
            status = "✅" if stats['success_rate'] >= 80 else "⚠️" if stats['success_rate'] >= 50 else "❌"
            report_content += f"- **{component}:** {status} {stats['success_rate']:.1f}% ({stats['successes']}/{stats['total']})\n"
        
        if summary['robust_loader']:
            status = "✅" if summary['robust_loader']['success_rate'] >= 80 else "⚠️" if summary['robust_loader']['success_rate'] >= 50 else "❌"
            report_content += f"\n### Robust Loader Performance\n- **Enhanced Loader:** {status} {summary['robust_loader']['success_rate']:.1f}% ({summary['robust_loader']['successes']}/{summary['robust_loader']['total']})\n"
        
        report_content += f"""
## Test File Details

The following test scenarios were evaluated:
"""
        
        for test_name, file_info in self.results.get('test_files', {}).items():
            report_content += f"- **{test_name}:** {file_info['description']}\n"
        
        report_content += f"""
## Recommendations

"""
        for rec in summary['recommendations']:
            report_content += f"- {rec}\n"
        
        report_content += f"""
## Technical Implementation

### Enhanced Image Loading Function
A robust image loader was developed that:
1. Tries multiple loading libraries (tifffile, skimage, PIL, OpenCV)
2. Detects file format from binary signatures, not extensions
3. Provides fallback mechanisms for edge cases
4. Returns detailed metadata about successful loading method

### Integration with IQID-Alphas
The enhanced loader can be integrated into the pipeline by:
1. Replacing the current `load_image` method in IQIDProcessor
2. Adding format detection capabilities
3. Implementing comprehensive error handling
4. Maintaining backward compatibility

## Conclusion

The evaluation demonstrates the pipeline's current limitations with extension mismatches
and provides solutions for enhanced robustness in real-world medical imaging scenarios.
"""
        
        # Save report
        report_path = f"{self.test_dir}/extension_mismatch_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Save detailed results
        results_path = f"{self.test_dir}/extension_mismatch_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"  ✅ Report saved: {report_path}")
        print(f"  ✅ Results saved: {results_path}")
        
        return summary
    
    def run_comprehensive_test(self):
        """Run complete extension mismatch evaluation."""
        print("🚀 Starting Extension Mismatch Evaluation")
        print("=" * 60)
        
        try:
            # 1. Create test files
            test_files = self.create_test_files()
            
            # 2. Test basic loading
            loading_results = self.test_basic_loading(test_files)
            
            # 3. Test pipeline components
            processor_results = self.test_iqid_processor(test_files)
            pipeline_results = self.test_simple_pipeline(test_files)
            
            # 4. Test robust loader
            robust_results = self.test_robust_loader(test_files)
            
            # 5. Generate summary
            summary = self.generate_summary_report()
            
            print("\n" + "=" * 60)
            print("🎉 EXTENSION MISMATCH EVALUATION COMPLETE")
            print("=" * 60)
            
            # Display summary
            print(f"\n📊 RESULTS SUMMARY:")
            print(f"   🗂️ Test Files: {summary['total_test_files']}")
            
            for method, stats in summary['loading_methods'].items():
                status = "✅" if stats['success_rate'] >= 80 else "⚠️" if stats['success_rate'] >= 50 else "❌"
                print(f"   {status} {method}: {stats['success_rate']:.1f}%")
            
            if summary['robust_loader']:
                status = "✅" if summary['robust_loader']['success_rate'] >= 80 else "⚠️"
                print(f"   {status} Robust Loader: {summary['robust_loader']['success_rate']:.1f}%")
            
            print(f"\n📁 OUTPUT FILES:")
            print(f"   📄 Report: {self.test_dir}/extension_mismatch_evaluation_report.md")
            print(f"   📋 Results: {self.test_dir}/extension_mismatch_results.json")
            print(f"   🛠️ Robust Loader: {self.test_dir}/robust_image_loader.py")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Extension mismatch evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function."""
    tester = ExtensionMismatchTester()
    results = tester.run_comprehensive_test()
    return results

if __name__ == "__main__":
    main()
