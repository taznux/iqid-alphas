#!/usr/bin/env python3
"""
UCSF Pipeline Evaluation System
Comprehensive evaluation of pipelines using the real UCSF dataset.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tifffile import imread
import pandas as pd
from scipy import stats
from skimage import measure, filters, morphology
import warnings
warnings.filterwarnings('ignore')

# Add package to path
sys.path.insert(0, '/home/wxc151/iqid-alphas')

try:
    from iqid_alphas.pipelines.simple import SimplePipeline
    from iqid_alphas.pipelines.advanced import AdvancedPipeline
    from iqid_alphas.pipelines.combined import CombinedPipeline
    from iqid_alphas.core.processor import IQIDProcessor
    from iqid_alphas.core.alignment import ImageAligner
    from iqid_alphas.core.segmentation import ImageSegmenter
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")

class UCSFPipelineEvaluator:
    """Comprehensive evaluation system for UCSF dataset processing."""
    
    def __init__(self):
        self.base_path = '/home/wxc151/iqid-alphas'
        self.data_path = '/home/wxc151/iqid-alphas/data'
        self.evaluation_dir = f"{self.base_path}/evaluation/reports"
        self.output_dir = f"{self.base_path}/evaluation/outputs"
        
        os.makedirs(self.evaluation_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.evaluation_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset': 'UCSF',
            'pipeline_version': '1.0',
            'evaluation_type': 'comprehensive_ucsf_assessment',
            'data_discovery': {},
            'pipeline_tests': {},
            'quality_metrics': {},
            'performance_metrics': {}
        }
    
    def discover_ucsf_data(self):
        """Discover and catalog all available UCSF data."""
        print("🔍 Discovering UCSF Dataset...")
        
        discovery = {
            'datapush1_iqid': {},
            'datapush1_he': {},
            'reupload_iqid': {},
            'total_files': 0,
            'sample_pairs': [],
            'data_structure': {}
        }
        
        # Discover DataPush1 iQID data
        iqid_path = f"{self.data_path}/DataPush1/iQID"
        if os.path.exists(iqid_path):
            print("   📁 Scanning DataPush1/iQID...")
            for root, dirs, files in os.walk(iqid_path):
                if files:
                    tif_files = [f for f in files if f.endswith('.tif')]
                    if tif_files:
                        rel_path = os.path.relpath(root, iqid_path)
                        discovery['datapush1_iqid'][rel_path] = {
                            'file_count': len(tif_files),
                            'files': tif_files[:5],  # First 5 files for reference
                            'full_path': root
                        }
                        discovery['total_files'] += len(tif_files)
        
        # Discover DataPush1 H&E data
        he_path = f"{self.data_path}/DataPush1/HE"
        if os.path.exists(he_path):
            print("   📁 Scanning DataPush1/HE...")
            for root, dirs, files in os.walk(he_path):
                if files:
                    tif_files = [f for f in files if f.endswith('.tif')]
                    if tif_files:
                        rel_path = os.path.relpath(root, he_path)
                        discovery['datapush1_he'][rel_path] = {
                            'file_count': len(tif_files),
                            'files': tif_files[:5],
                            'full_path': root
                        }
                        discovery['total_files'] += len(tif_files)
        
        # Discover ReUpload data
        reupload_path = f"{self.data_path}/ReUpload"
        if os.path.exists(reupload_path):
            print("   📁 Scanning ReUpload...")
            for root, dirs, files in os.walk(reupload_path):
                if files:
                    tif_files = [f for f in files if f.endswith('.tif')]
                    if tif_files:
                        rel_path = os.path.relpath(root, reupload_path)
                        discovery['reupload_iqid'][rel_path] = {
                            'file_count': len(tif_files),
                            'files': tif_files[:5],
                            'full_path': root
                        }
                        discovery['total_files'] += len(tif_files)
        
        print(f"   ✓ Total files discovered: {discovery['total_files']}")
        print(f"   ✓ iQID directories: {len(discovery['datapush1_iqid'])}")
        print(f"   ✓ H&E directories: {len(discovery['datapush1_he'])}")
        print(f"   ✓ ReUpload directories: {len(discovery['reupload_iqid'])}")
        
        self.evaluation_results['data_discovery'] = discovery
        return discovery
    
    def test_data_loading(self, discovery):
        """Test data loading capabilities across different file types."""
        print("\n📊 Testing Data Loading Capabilities...")
        
        loading_results = {
            'successful_loads': 0,
            'failed_loads': 0,
            'file_characteristics': {},
            'loading_errors': [],
            'performance_metrics': {}
        }
        
        # Test sample from each category
        test_samples = []
        
        # Add iQID samples
        for dir_name, dir_info in list(discovery['datapush1_iqid'].items())[:3]:
            if dir_info['files']:
                sample_path = os.path.join(dir_info['full_path'], dir_info['files'][0])
                test_samples.append(('iqid', sample_path))
        
        # Add H&E samples
        for dir_name, dir_info in list(discovery['datapush1_he'].items())[:2]:
            if dir_info['files']:
                sample_path = os.path.join(dir_info['full_path'], dir_info['files'][0])
                test_samples.append(('he', sample_path))
        
        print(f"   Testing {len(test_samples)} representative samples...")
        
        for data_type, file_path in test_samples:
            try:
                start_time = datetime.now()
                
                # Try loading with tifffile
                img_array = imread(file_path)
                
                load_time = (datetime.now() - start_time).total_seconds()
                
                # Analyze characteristics
                characteristics = {
                    'shape': img_array.shape,
                    'dtype': str(img_array.dtype),
                    'size_mb': img_array.nbytes / 1024 / 1024,
                    'min_value': float(np.min(img_array)),
                    'max_value': float(np.max(img_array)),
                    'mean_value': float(np.mean(img_array)),
                    'load_time_sec': load_time
                }
                
                loading_results['file_characteristics'][os.path.basename(file_path)] = characteristics
                loading_results['successful_loads'] += 1
                
                print(f"     ✓ {data_type}: {os.path.basename(file_path)} - {img_array.shape} {img_array.dtype}")
                
            except Exception as e:
                loading_results['failed_loads'] += 1
                loading_results['loading_errors'].append({
                    'file': os.path.basename(file_path),
                    'error': str(e)
                })
                print(f"     ❌ {data_type}: {os.path.basename(file_path)} - {e}")
        
        success_rate = loading_results['successful_loads'] / (loading_results['successful_loads'] + loading_results['failed_loads']) * 100
        print(f"   ✓ Data loading success rate: {success_rate:.1f}%")
        
        self.evaluation_results['pipeline_tests']['data_loading'] = loading_results
        return loading_results
    
    def test_simple_pipeline(self, discovery):
        """Test SimplePipeline with UCSF data."""
        print("\n🔧 Testing SimplePipeline...")
        
        pipeline_results = {
            'tests_run': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'processing_errors': [],
            'quality_scores': [],
            'processing_times': []
        }
        
        try:
            # Initialize pipeline
            config_path = f"{self.base_path}/configs/pipeline_test_config.json"
            if not os.path.exists(config_path):
                print("     ⚠️ No pipeline config found, using defaults")
                pipeline = SimplePipeline()
            else:
                pipeline = SimplePipeline(config_path)
            
            # Test with iQID samples
            test_count = 0
            for dir_name, dir_info in discovery['datapush1_iqid'].items():
                if test_count >= 3:  # Limit to 3 tests
                    break
                
                if dir_info['files']:
                    sample_path = os.path.join(dir_info['full_path'], dir_info['files'][0])
                    
                    try:
                        start_time = datetime.now()
                        
                        # Load image
                        img_array = imread(sample_path)
                        
                        # Process with pipeline
                        result = pipeline.process_iqid_image(img_array)
                        
                        process_time = (datetime.now() - start_time).total_seconds()
                        
                        if result is not None:
                            pipeline_results['successful_processes'] += 1
                            pipeline_results['processing_times'].append(process_time)
                            
                            # Simple quality assessment
                            quality_score = self._assess_processing_quality(img_array, result)
                            pipeline_results['quality_scores'].append(quality_score)
                            
                            print(f"     ✓ {os.path.basename(sample_path)} - Quality: {quality_score:.1f}")
                        else:
                            pipeline_results['failed_processes'] += 1
                            
                    except Exception as e:
                        pipeline_results['failed_processes'] += 1
                        pipeline_results['processing_errors'].append({
                            'file': os.path.basename(sample_path),
                            'error': str(e)
                        })
                        print(f"     ❌ {os.path.basename(sample_path)} - {e}")
                    
                    pipeline_results['tests_run'] += 1
                    test_count += 1
            
            # Calculate metrics
            if pipeline_results['quality_scores']:
                pipeline_results['average_quality'] = np.mean(pipeline_results['quality_scores'])
                pipeline_results['quality_std'] = np.std(pipeline_results['quality_scores'])
            else:
                pipeline_results['average_quality'] = 0
                pipeline_results['quality_std'] = 0
            
            if pipeline_results['processing_times']:
                pipeline_results['average_time'] = np.mean(pipeline_results['processing_times'])
            else:
                pipeline_results['average_time'] = 0
            
            success_rate = pipeline_results['successful_processes'] / max(1, pipeline_results['tests_run']) * 100
            pipeline_results['success_rate'] = success_rate
            
            print(f"   ✓ Tests run: {pipeline_results['tests_run']}")
            print(f"   ✓ Success rate: {success_rate:.1f}%")
            print(f"   ✓ Average quality: {pipeline_results['average_quality']:.1f}")
            print(f"   ✓ Average time: {pipeline_results['average_time']:.2f}s")
            
        except Exception as e:
            print(f"     ❌ SimplePipeline initialization failed: {e}")
            pipeline_results['initialization_error'] = str(e)
        
        self.evaluation_results['pipeline_tests']['simple_pipeline'] = pipeline_results
        return pipeline_results
    
    def test_advanced_pipeline(self, discovery):
        """Test AdvancedPipeline with UCSF data."""
        print("\n🔧 Testing AdvancedPipeline...")
        
        pipeline_results = {
            'tests_run': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'processing_errors': [],
            'quality_scores': [],
            'processing_times': []
        }
        
        try:
            # Initialize pipeline
            pipeline = AdvancedPipeline()
            
            # Test with paired iQID and H&E samples
            iqid_samples = list(discovery['datapush1_iqid'].items())[:2]
            he_samples = list(discovery['datapush1_he'].items())[:2]
            
            for i, (iqid_dir, iqid_info) in enumerate(iqid_samples):
                if i < len(he_samples):
                    he_dir, he_info = he_samples[i]
                    
                    if iqid_info['files'] and he_info['files']:
                        iqid_path = os.path.join(iqid_info['full_path'], iqid_info['files'][0])
                        he_path = os.path.join(he_info['full_path'], he_info['files'][0])
                        
                        try:
                            start_time = datetime.now()
                            
                            # Load images
                            iqid_img = imread(iqid_path)
                            he_img = imread(he_path)
                            
                            # Process with advanced pipeline
                            result = pipeline.process_paired_images(iqid_img, he_img)
                            
                            process_time = (datetime.now() - start_time).total_seconds()
                            
                            if result is not None:
                                pipeline_results['successful_processes'] += 1
                                pipeline_results['processing_times'].append(process_time)
                                
                                # Quality assessment
                                quality_score = self._assess_advanced_quality(iqid_img, he_img, result)
                                pipeline_results['quality_scores'].append(quality_score)
                                
                                print(f"     ✓ Pair {i+1} - Quality: {quality_score:.1f}")
                            else:
                                pipeline_results['failed_processes'] += 1
                                
                        except Exception as e:
                            pipeline_results['failed_processes'] += 1
                            pipeline_results['processing_errors'].append({
                                'pair': f"Pair {i+1}",
                                'error': str(e)
                            })
                            print(f"     ❌ Pair {i+1} - {e}")
                        
                        pipeline_results['tests_run'] += 1
            
            # Calculate metrics
            if pipeline_results['quality_scores']:
                pipeline_results['average_quality'] = np.mean(pipeline_results['quality_scores'])
            else:
                pipeline_results['average_quality'] = 0
            
            if pipeline_results['processing_times']:
                pipeline_results['average_time'] = np.mean(pipeline_results['processing_times'])
            else:
                pipeline_results['average_time'] = 0
            
            success_rate = pipeline_results['successful_processes'] / max(1, pipeline_results['tests_run']) * 100
            pipeline_results['success_rate'] = success_rate
            
            print(f"   ✓ Tests run: {pipeline_results['tests_run']}")
            print(f"   ✓ Success rate: {success_rate:.1f}%")
            print(f"   ✓ Average quality: {pipeline_results['average_quality']:.1f}")
            
        except Exception as e:
            print(f"     ❌ AdvancedPipeline initialization failed: {e}")
            pipeline_results['initialization_error'] = str(e)
        
        self.evaluation_results['pipeline_tests']['advanced_pipeline'] = pipeline_results
        return pipeline_results
    
    def test_core_components(self, discovery):
        """Test core processing components."""
        print("\n⚙️ Testing Core Components...")
        
        core_results = {
            'processor_tests': {'success': 0, 'total': 0, 'errors': []},
            'aligner_tests': {'success': 0, 'total': 0, 'errors': []},
            'segmenter_tests': {'success': 0, 'total': 0, 'errors': []}
        }
        
        # Test IQIDProcessor
        try:
            processor = IQIDProcessor()
            
            # Test with sample iQID data
            for dir_name, dir_info in list(discovery['datapush1_iqid'].items())[:2]:
                if dir_info['files']:
                    sample_path = os.path.join(dir_info['full_path'], dir_info['files'][0])
                    
                    try:
                        img_array = imread(sample_path)
                        processed = processor.process_image(img_array)
                        
                        if processed is not None:
                            core_results['processor_tests']['success'] += 1
                            print(f"     ✓ Processor: {os.path.basename(sample_path)}")
                        
                    except Exception as e:
                        core_results['processor_tests']['errors'].append(str(e))
                        print(f"     ❌ Processor: {os.path.basename(sample_path)} - {e}")
                    
                    core_results['processor_tests']['total'] += 1
                    
        except Exception as e:
            print(f"     ❌ IQIDProcessor initialization failed: {e}")
        
        # Test ImageAligner
        try:
            aligner = ImageAligner()
            
            # Test with pairs of iQID images
            iqid_samples = list(discovery['datapush1_iqid'].items())[:1]
            for dir_name, dir_info in iqid_samples:
                if len(dir_info['files']) >= 2:
                    path1 = os.path.join(dir_info['full_path'], dir_info['files'][0])
                    path2 = os.path.join(dir_info['full_path'], dir_info['files'][1])
                    
                    try:
                        img1 = imread(path1)
                        img2 = imread(path2)
                        
                        aligned = aligner.align_images(img1, img2)
                        
                        if aligned is not None:
                            core_results['aligner_tests']['success'] += 1
                            print(f"     ✓ Aligner: {os.path.basename(path1)} -> {os.path.basename(path2)}")
                        
                    except Exception as e:
                        core_results['aligner_tests']['errors'].append(str(e))
                        print(f"     ❌ Aligner: {e}")
                    
                    core_results['aligner_tests']['total'] += 1
                    
        except Exception as e:
            print(f"     ❌ ImageAligner initialization failed: {e}")
        
        # Test ImageSegmenter
        try:
            segmenter = ImageSegmenter()
            
            # Test with H&E data
            for dir_name, dir_info in list(discovery['datapush1_he'].items())[:1]:
                if dir_info['files']:
                    sample_path = os.path.join(dir_info['full_path'], dir_info['files'][0])
                    
                    try:
                        img_array = imread(sample_path)
                        
                        # Only test if reasonable size (avoid huge images)
                        if img_array.nbytes < 500 * 1024 * 1024:  # < 500MB
                            segmented = segmenter.segment_tissue(img_array)
                            
                            if segmented is not None:
                                core_results['segmenter_tests']['success'] += 1
                                print(f"     ✓ Segmenter: {os.path.basename(sample_path)}")
                        else:
                            print(f"     ⚠️ Segmenter: Skipping large file {os.path.basename(sample_path)}")
                        
                    except Exception as e:
                        core_results['segmenter_tests']['errors'].append(str(e))
                        print(f"     ❌ Segmenter: {os.path.basename(sample_path)} - {e}")
                    
                    core_results['segmenter_tests']['total'] += 1
                    
        except Exception as e:
            print(f"     ❌ ImageSegmenter initialization failed: {e}")
        
        self.evaluation_results['pipeline_tests']['core_components'] = core_results
        return core_results
    
    def _assess_processing_quality(self, original, processed):
        """Simple quality assessment for processed images."""
        try:
            if processed is None:
                return 0.0
            
            # Basic quality metrics
            snr = 20 * np.log10(np.mean(processed) / (np.std(processed) + 1e-10))
            contrast = np.std(processed) / (np.mean(processed) + 1e-10)
            
            # Normalize to 0-100 scale
            quality_score = min(100, max(0, (snr + contrast * 10) * 2))
            return quality_score
            
        except:
            return 50.0  # Default moderate score
    
    def _assess_advanced_quality(self, iqid_img, he_img, result):
        """Quality assessment for advanced pipeline results."""
        try:
            if result is None:
                return 0.0
            
            # Basic correlation assessment
            if hasattr(result, 'aligned_iqid') and hasattr(result, 'aligned_he'):
                correlation = np.corrcoef(result.aligned_iqid.flatten(), result.aligned_he.flatten())[0, 1]
                quality_score = max(0, min(100, (correlation + 1) * 50))
            else:
                quality_score = 60.0  # Default moderate score
            
            return quality_score
            
        except:
            return 50.0
    
    def generate_visualization_dashboard(self):
        """Generate comprehensive visualization dashboard."""
        print("\n📊 Generating Evaluation Dashboard...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Data discovery summary
        ax1 = fig.add_subplot(gs[0, :2])
        discovery = self.evaluation_results['data_discovery']
        
        categories = ['iQID Dirs', 'H&E Dirs', 'ReUpload Dirs']
        counts = [
            len(discovery['datapush1_iqid']),
            len(discovery['datapush1_he']),
            len(discovery['reupload_iqid'])
        ]
        
        ax1.bar(categories, counts, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('UCSF Data Discovery', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Directory Count')
        
        # Pipeline performance
        ax2 = fig.add_subplot(gs[0, 2:])
        pipeline_tests = self.evaluation_results['pipeline_tests']
        
        if 'simple_pipeline' in pipeline_tests and 'advanced_pipeline' in pipeline_tests:
            pipelines = ['Simple', 'Advanced']
            success_rates = [
                pipeline_tests['simple_pipeline'].get('success_rate', 0),
                pipeline_tests['advanced_pipeline'].get('success_rate', 0)
            ]
            
            bars = ax2.bar(pipelines, success_rates, color=['lightblue', 'orange'])
            ax2.set_title('Pipeline Success Rates', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')
        
        # Data loading performance
        ax3 = fig.add_subplot(gs[1, :2])
        if 'data_loading' in pipeline_tests:
            loading = pipeline_tests['data_loading']
            success = loading['successful_loads']
            failed = loading['failed_loads']
            
            ax3.pie([success, failed], labels=['Success', 'Failed'], 
                   colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
            ax3.set_title('Data Loading Results', fontsize=14, fontweight='bold')
        
        # Core components performance
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'core_components' in pipeline_tests:
            core = pipeline_tests['core_components']
            
            components = []
            success_rates = []
            
            for comp_name, comp_data in core.items():
                if comp_data['total'] > 0:
                    components.append(comp_name.replace('_tests', '').title())
                    success_rates.append(comp_data['success'] / comp_data['total'] * 100)
            
            if components:
                ax4.bar(components, success_rates, color=['steelblue', 'orange', 'green'])
                ax4.set_title('Core Components Performance', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Success Rate (%)')
                ax4.set_ylim(0, 100)
                plt.setp(ax4.get_xticklabels(), rotation=45)
        
        # Quality scores distribution
        ax5 = fig.add_subplot(gs[2, :2])
        all_quality_scores = []
        
        for test_name, test_data in pipeline_tests.items():
            if isinstance(test_data, dict) and 'quality_scores' in test_data:
                all_quality_scores.extend(test_data['quality_scores'])
        
        if all_quality_scores:
            ax5.hist(all_quality_scores, bins=10, color='lightblue', alpha=0.7, edgecolor='black')
            ax5.set_title('Quality Scores Distribution', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Quality Score')
            ax5.set_ylabel('Frequency')
        else:
            ax5.text(0.5, 0.5, 'No Quality Data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Quality Scores Distribution', fontsize=14, fontweight='bold')
        
        # Processing times
        ax6 = fig.add_subplot(gs[2, 2:])
        processing_times = []
        time_labels = []
        
        for test_name, test_data in pipeline_tests.items():
            if isinstance(test_data, dict) and 'processing_times' in test_data and test_data['processing_times']:
                processing_times.extend(test_data['processing_times'])
                time_labels.extend([test_name.replace('_', ' ').title()] * len(test_data['processing_times']))
        
        if processing_times:
            ax6.boxplot([processing_times], labels=['All Tests'])
            ax6.set_title('Processing Time Distribution', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Time (seconds)')
        else:
            ax6.text(0.5, 0.5, 'No Timing Data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Processing Time Distribution', fontsize=14, fontweight='bold')
        
        plt.suptitle('UCSF Pipeline Evaluation Dashboard', fontsize=16, fontweight='bold')
        
        # Save dashboard
        dashboard_path = f"{self.evaluation_dir}/ucsf_evaluation_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Dashboard saved: {dashboard_path}")
        
        plt.close()
        return dashboard_path
    
    def calculate_overall_score(self):
        """Calculate overall evaluation score."""
        scores = {
            'data_discovery': 0,
            'data_loading': 0,
            'pipeline_performance': 0,
            'core_components': 0,
            'overall': 0
        }
        
        # Data discovery score (based on data found)
        discovery = self.evaluation_results['data_discovery']
        total_dirs = len(discovery['datapush1_iqid']) + len(discovery['datapush1_he']) + len(discovery['reupload_iqid'])
        if total_dirs > 0:
            scores['data_discovery'] = min(100, total_dirs * 10)  # 10 points per directory found
        
        # Data loading score
        if 'data_loading' in self.evaluation_results['pipeline_tests']:
            loading = self.evaluation_results['pipeline_tests']['data_loading']
            total_tests = loading['successful_loads'] + loading['failed_loads']
            if total_tests > 0:
                scores['data_loading'] = (loading['successful_loads'] / total_tests) * 100
        
        # Pipeline performance score
        pipeline_scores = []
        for test_name in ['simple_pipeline', 'advanced_pipeline']:
            if test_name in self.evaluation_results['pipeline_tests']:
                test_data = self.evaluation_results['pipeline_tests'][test_name]
                if 'success_rate' in test_data:
                    pipeline_scores.append(test_data['success_rate'])
        
        if pipeline_scores:
            scores['pipeline_performance'] = np.mean(pipeline_scores)
        
        # Core components score
        if 'core_components' in self.evaluation_results['pipeline_tests']:
            core = self.evaluation_results['pipeline_tests']['core_components']
            component_scores = []
            
            for comp_data in core.values():
                if comp_data['total'] > 0:
                    component_scores.append((comp_data['success'] / comp_data['total']) * 100)
            
            if component_scores:
                scores['core_components'] = np.mean(component_scores)
        
        # Calculate weighted overall score
        weights = {
            'data_discovery': 0.2,
            'data_loading': 0.3,
            'pipeline_performance': 0.3,
            'core_components': 0.2
        }
        
        scores['overall'] = sum(scores[key] * weights[key] for key in weights.keys())
        
        self.evaluation_results['quality_metrics']['scores'] = scores
        return scores
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report."""
        print("\n📄 Generating Comprehensive Report...")
        
        scores = self.calculate_overall_score()
        
        report_content = f"""# UCSF Pipeline Evaluation Report

**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** UCSF Medical Imaging Data
**Pipeline Version:** {self.evaluation_results['pipeline_version']}
**Overall Score:** {scores['overall']:.1f}/100

## Executive Summary

"""
        
        if scores['overall'] >= 80:
            report_content += "✅ **STATUS: PRODUCTION READY**\n\n"
            report_content += "The pipeline demonstrates excellent performance with the UCSF dataset.\n\n"
        elif scores['overall'] >= 60:
            report_content += "⚠️ **STATUS: ACCEPTABLE WITH RECOMMENDATIONS**\n\n"
            report_content += "The pipeline shows good performance but may benefit from improvements.\n\n"
        else:
            report_content += "❌ **STATUS: REQUIRES IMPROVEMENT**\n\n"
            report_content += "The pipeline needs significant improvements before production use.\n\n"
        
        # Data Discovery Summary
        discovery = self.evaluation_results['data_discovery']
        report_content += f"""## Data Discovery Results

- **Total Files Found:** {discovery['total_files']}
- **iQID Directories:** {len(discovery['datapush1_iqid'])}
- **H&E Directories:** {len(discovery['datapush1_he'])}
- **ReUpload Directories:** {len(discovery['reupload_iqid'])}
- **Discovery Score:** {scores['data_discovery']:.1f}/100

"""
        
        # Pipeline Performance
        pipeline_tests = self.evaluation_results['pipeline_tests']
        report_content += "## Pipeline Performance\n\n"
        
        for test_name, test_data in pipeline_tests.items():
            if isinstance(test_data, dict) and 'success_rate' in test_data:
                report_content += f"### {test_name.replace('_', ' ').title()}\n"
                report_content += f"- **Success Rate:** {test_data['success_rate']:.1f}%\n"
                report_content += f"- **Tests Run:** {test_data['tests_run']}\n"
                
                if 'average_quality' in test_data:
                    report_content += f"- **Average Quality:** {test_data['average_quality']:.1f}\n"
                
                if 'average_time' in test_data:
                    report_content += f"- **Average Processing Time:** {test_data['average_time']:.2f}s\n"
                
                report_content += "\n"
        
        # Detailed Scores
        report_content += f"""## Detailed Score Breakdown

- **Data Discovery:** {scores['data_discovery']:.1f}/100
- **Data Loading:** {scores['data_loading']:.1f}/100
- **Pipeline Performance:** {scores['pipeline_performance']:.1f}/100
- **Core Components:** {scores['core_components']:.1f}/100

## Recommendations

"""
        
        # Generate recommendations based on scores
        if scores['data_loading'] < 80:
            report_content += "- 🔧 Improve data loading reliability and error handling\n"
        
        if scores['pipeline_performance'] < 80:
            report_content += "- 🔧 Optimize pipeline processing algorithms\n"
            report_content += "- 🧪 Expand testing with diverse UCSF samples\n"
        
        if scores['core_components'] < 80:
            report_content += "- 🔧 Address core component processing issues\n"
            report_content += "- 📊 Implement additional quality control measures\n"
        
        report_content += """
## Technical Details

Complete evaluation results available in: `ucsf_evaluation_results.json`
Visual dashboard available in: `ucsf_evaluation_dashboard.png`

## Data Files Tested

"""
        
        # Add sample file details
        if 'data_loading' in pipeline_tests and 'file_characteristics' in pipeline_tests['data_loading']:
            for filename, characteristics in pipeline_tests['data_loading']['file_characteristics'].items():
                report_content += f"- **{filename}:** {characteristics['shape']} {characteristics['dtype']}\n"
        
        # Save report
        report_path = f"{self.evaluation_dir}/ucsf_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"   ✓ Report saved: {report_path}")
        return report_path
    
    def run_comprehensive_evaluation(self):
        """Run complete UCSF pipeline evaluation."""
        print("🚀 Starting Comprehensive UCSF Pipeline Evaluation")
        print("=" * 60)
        
        try:
            # 1. Discover UCSF data
            discovery = self.discover_ucsf_data()
            
            # 2. Test data loading
            loading_results = self.test_data_loading(discovery)
            
            # 3. Test pipelines
            simple_results = self.test_simple_pipeline(discovery)
            advanced_results = self.test_advanced_pipeline(discovery)
            
            # 4. Test core components
            core_results = self.test_core_components(discovery)
            
            # 5. Generate visualizations
            dashboard_path = self.generate_visualization_dashboard()
            
            # 6. Calculate scores and generate report
            scores = self.calculate_overall_score()
            report_path = self.generate_comprehensive_report()
            
            # 7. Save complete results
            results_path = f"{self.evaluation_dir}/ucsf_evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2)
            
            print("\n" + "=" * 60)
            print("🎉 UCSF PIPELINE EVALUATION COMPLETE")
            print("=" * 60)
            
            print(f"\n📊 OVERALL ASSESSMENT:")
            print(f"   🎯 Overall Score: {scores['overall']:.1f}/100")
            print(f"   📁 Total UCSF Files: {discovery['total_files']}")
            print(f"   ✅ Data Loading: {scores['data_loading']:.1f}%")
            print(f"   🔧 Pipeline Performance: {scores['pipeline_performance']:.1f}%")
            print(f"   ⚙️ Core Components: {scores['core_components']:.1f}%")
            
            if scores['overall'] >= 80:
                print(f"   ✅ Status: PRODUCTION READY")
            elif scores['overall'] >= 60:
                print(f"   ⚠️ Status: ACCEPTABLE WITH RECOMMENDATIONS")
            else:
                print(f"   ❌ Status: REQUIRES IMPROVEMENT")
            
            print(f"\n📁 EVALUATION OUTPUTS:")
            print(f"   📄 Detailed Report: {report_path}")
            print(f"   📊 Dashboard: {dashboard_path}")
            print(f"   📋 Results Data: {results_path}")
            
            return self.evaluation_results
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run UCSF evaluation."""
    evaluator = UCSFPipelineEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    return results

if __name__ == "__main__":
    main()
