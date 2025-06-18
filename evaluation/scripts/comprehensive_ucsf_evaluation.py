#!/usr/bin/env python3
"""
Comprehensive UCSF Dataset Evaluation System

This script provides a complete evaluation framework for the IQID-Alphas system
using the full UCSF dataset (both DataPush1 and ReUpload datasets).

Features:
- Complete dataset discovery and analysis
- Multi-pipeline evaluation (Simple, Advanced, Combined)
- Production readiness assessment
- Quality metrics and performance analysis
- Comprehensive reporting with visualizations
- Dataset-specific workflow testing
"""

import sys
import os
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import time

# Add iqid_alphas to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iqid_alphas.cli import IQIDCLIProcessor
from iqid_alphas.pipelines.simple import SimplePipeline
from iqid_alphas.pipelines.advanced import AdvancedPipeline
from iqid_alphas.pipelines.combined import CombinedPipeline

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    sample_id: str
    dataset_type: str
    pipeline_type: str
    processing_stage: str
    success: bool
    processing_time: float
    quality_score: Optional[float] = None
    alignment_score: Optional[float] = None
    error_message: Optional[str] = None
    output_files_count: int = 0
    memory_usage_mb: Optional[float] = None

@dataclass
class DatasetSummary:
    """Container for dataset summary statistics"""
    dataset_type: str
    total_samples: int
    iqid_samples: int
    he_samples: int
    paired_samples: int
    stage_distribution: Dict[str, int]
    tissue_distribution: Dict[str, int]
    ready_for_3d: int

class ComprehensiveUCSFEvaluator:
    """Comprehensive evaluation system for UCSF datasets"""
    
    def __init__(self, data_path: str, output_path: str = None):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path) if output_path else Path("evaluation/outputs")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize CLI processor for data discovery
        self.cli_processor = IQIDCLIProcessor()
        
        # Storage for evaluation results
        self.evaluation_metrics: List[EvaluationMetrics] = []
        self.dataset_summaries: List[DatasetSummary] = []
        self.pipeline_configs = {}
        
        # Load evaluation configurations
        self.load_evaluation_configs()
        
        self.logger.info(f"Initialized UCSF Evaluator with data path: {self.data_path}")
        self.logger.info(f"Output path: {self.output_path}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.output_path / f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_evaluation_configs(self):
        """Load evaluation configurations for different pipelines"""
        config_dir = Path(__file__).parent.parent / "configs"
        
        # Default configurations for evaluation
        self.pipeline_configs = {
            'simple': {
                "processing": {
                    "normalize": True,
                    "remove_outliers": True,
                    "gaussian_filter": True,
                    "filter_sigma": 1.0
                },
                "segmentation": {
                    "method": "otsu",
                    "min_size": 100,
                    "remove_small_objects": True
                },
                "alignment": {
                    "method": "phase_correlation",
                    "max_shift": 50,
                    "subpixel_precision": True
                },
                "visualization": {
                    "save_plots": True,
                    "output_dir": str(self.output_path),
                    "dpi": 300,
                    "format": "png"
                }
            },
            'advanced': {
                "processing": {
                    "normalize": True,
                    "remove_outliers": True,
                    "gaussian_filter": True,
                    "filter_sigma": 1.5,
                    "advanced_preprocessing": True
                },
                "segmentation": {
                    "method": "watershed",
                    "min_size": 50,
                    "remove_small_objects": True,
                    "morphological_operations": True
                },
                "alignment": {
                    "method": "feature_based",
                    "max_shift": 100,
                    "subpixel_precision": True,
                    "quality_assessment": True
                },
                "quality_control": {
                    "enable_validation": True,
                    "min_alignment_score": 0.5,
                    "max_processing_time": 300
                },
                "visualization": {
                    "save_plots": True,
                    "output_dir": str(self.output_path),
                    "dpi": 300,
                    "format": "png",
                    "comprehensive_plots": True
                }
            },
            'combined': {
                "processing": {
                    "normalize": True,
                    "remove_outliers": True,
                    "gaussian_filter": True,
                    "filter_sigma": 1.2
                },
                "multimodal": {
                    "enable_coregistration": True,
                    "iqid_he_alignment": True,
                    "cross_modal_validation": True
                },
                "segmentation": {
                    "method": "multi_modal",
                    "min_size": 75,
                    "use_he_guidance": True
                },
                "alignment": {
                    "method": "multi_modal",
                    "max_shift": 75,
                    "subpixel_precision": True,
                    "cross_modal_refinement": True
                },
                "visualization": {
                    "save_plots": True,
                    "output_dir": str(self.output_path),
                    "dpi": 300,
                    "format": "png",
                    "multimodal_plots": True
                }
            }
        }
        
        # Save configs to files
        for pipeline_name, config in self.pipeline_configs.items():
            config_file = config_dir / f"evaluation_{pipeline_name}_config.json"
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
    
    def discover_all_datasets(self) -> Dict[str, Dict]:
        """Discover and analyze all available datasets"""
        self.logger.info("🔍 Starting comprehensive dataset discovery...")
        
        datasets = {}
        
        # Check for DataPush1 (Production dataset)
        datapush1_path = self.data_path / "DataPush1"
        if datapush1_path.exists():
            self.logger.info("📁 Discovering DataPush1 (Production) dataset...")
            try:
                discovery_result = self.cli_processor.discover_data(datapush1_path)
                datasets['DataPush1'] = discovery_result
                self.create_dataset_summary(discovery_result, 'DataPush1')
                self.logger.info(f"✅ DataPush1 discovery complete: {len(discovery_result['iqid_samples'])} iQID, {len(discovery_result['he_samples'])} H&E samples")
            except Exception as e:
                self.logger.error(f"❌ DataPush1 discovery failed: {e}")
                datasets['DataPush1'] = None
        
        # Check for ReUpload (Workflow dataset)
        reupload_path = self.data_path / "ReUpload"
        if reupload_path.exists():
            self.logger.info("📁 Discovering ReUpload (Workflow) dataset...")
            try:
                discovery_result = self.cli_processor.discover_data(reupload_path)
                datasets['ReUpload'] = discovery_result
                self.create_dataset_summary(discovery_result, 'ReUpload')
                self.logger.info(f"✅ ReUpload discovery complete: {len(discovery_result['iqid_samples'])} workflow samples")
            except Exception as e:
                self.logger.error(f"❌ ReUpload discovery failed: {e}")
                datasets['ReUpload'] = None
        
        # Check for any other datasets
        for item in self.data_path.iterdir():
            if item.is_dir() and item.name not in ['DataPush1', 'ReUpload', 'Visualization']:
                self.logger.info(f"📁 Discovering additional dataset: {item.name}...")
                try:
                    discovery_result = self.cli_processor.discover_data(item)
                    datasets[item.name] = discovery_result
                    self.create_dataset_summary(discovery_result, item.name)
                    self.logger.info(f"✅ {item.name} discovery complete")
                except Exception as e:
                    self.logger.error(f"❌ {item.name} discovery failed: {e}")
                    datasets[item.name] = None
        
        self.logger.info(f"🎯 Total datasets discovered: {len([d for d in datasets.values() if d is not None])}")
        return datasets
    
    def create_dataset_summary(self, discovery_result: Dict, dataset_name: str):
        """Create summary statistics for a dataset"""
        if not discovery_result:
            return
        
        dataset_info = discovery_result.get('dataset_info', {})
        iqid_samples = discovery_result.get('iqid_samples', [])
        he_samples = discovery_result.get('he_samples', [])
        paired_samples = discovery_result.get('paired_samples', [])
        
        # Stage distribution
        stage_dist = {}
        for sample in iqid_samples:
            stage = sample.get('processing_stage', 'unknown')
            stage_dist[stage] = stage_dist.get(stage, 0) + 1
        
        # Tissue distribution
        tissue_dist = {}
        for sample in iqid_samples:
            tissue = sample.get('tissue_type', 'unknown')
            tissue_dist[tissue] = tissue_dist.get(tissue, 0) + 1
        
        # 3D ready count
        ready_for_3d = sum(1 for sample in iqid_samples if sample.get('can_reconstruct_3d', False))
        
        summary = DatasetSummary(
            dataset_type=dataset_info.get('type', 'unknown'),
            total_samples=len(iqid_samples) + len(he_samples),
            iqid_samples=len(iqid_samples),
            he_samples=len(he_samples),
            paired_samples=len(paired_samples),
            stage_distribution=stage_dist,
            tissue_distribution=tissue_dist,
            ready_for_3d=ready_for_3d
        )
        
        self.dataset_summaries.append(summary)
    
    def evaluate_pipeline_on_dataset(self, pipeline_type: str, dataset_name: str, 
                                   discovery_result: Dict, max_samples: int = None) -> List[EvaluationMetrics]:
        """Evaluate a specific pipeline on a dataset"""
        self.logger.info(f"🧪 Evaluating {pipeline_type} pipeline on {dataset_name} dataset...")
        
        if not discovery_result:
            self.logger.error(f"No discovery result for {dataset_name}")
            return []
        
        metrics = []
        dataset_type = discovery_result.get('dataset_info', {}).get('type', 'unknown')
        iqid_samples = discovery_result.get('iqid_samples', [])
        
        # Limit samples if specified
        if max_samples:
            iqid_samples = iqid_samples[:max_samples]
        
        # Get appropriate pipeline class
        pipeline_classes = {
            'simple': SimplePipeline,
            'advanced': AdvancedPipeline,
            'combined': CombinedPipeline
        }
        
        if pipeline_type not in pipeline_classes:
            self.logger.error(f"Unknown pipeline type: {pipeline_type}")
            return []
        
        PipelineClass = pipeline_classes[pipeline_type]
        config = self.pipeline_configs[pipeline_type]
        
        # Create pipeline-specific output directory
        pipeline_output = self.output_path / f"{dataset_name}_{pipeline_type}_output"
        pipeline_output.mkdir(parents=True, exist_ok=True)
        
        # Update config with correct output path
        config['visualization']['output_dir'] = str(pipeline_output)
        
        for i, sample in enumerate(iqid_samples):
            sample_id = sample.get('sample_id', f'sample_{i}')
            sample_dir = sample.get('sample_dir')
            processing_stage = sample.get('processing_stage', 'unknown')
            
            self.logger.info(f"  📊 Processing sample {i+1}/{len(iqid_samples)}: {sample_id}")
            
            try:
                # Create sample-specific output directory
                sample_output = pipeline_output / f"sample_{sample_id}"
                sample_output.mkdir(parents=True, exist_ok=True)
                
                # Update config for this sample
                sample_config = config.copy()
                sample_config['visualization']['output_dir'] = str(sample_output)
                
                # Initialize pipeline
                pipeline = PipelineClass(sample_config)
                
                # Record start time and memory
                start_time = time.time()
                
                # Process based on pipeline type and dataset
                if pipeline_type == 'combined' and dataset_type == 'production':
                    # Try to find paired H&E sample
                    paired_samples = discovery_result.get('paired_samples', [])
                    paired_sample = None
                    for pair in paired_samples:
                        if pair.get('iqid_sample', {}).get('sample_id') == sample_id:
                            paired_sample = pair
                            break
                    
                    if paired_sample:
                        he_sample = paired_sample.get('he_sample', {})
                        he_sample_dir = he_sample.get('sample_dir')
                        
                        # Get representative slices
                        iqid_files = sample.get('slice_files', [])
                        he_files = he_sample.get('slice_files', [])
                        
                        if iqid_files and he_files:
                            # Use middle slices as representatives
                            iqid_file = iqid_files[len(iqid_files)//2]
                            he_file = he_files[len(he_files)//2]
                            
                            result = pipeline.process_image_pair(
                                he_path=he_file,
                                iqid_path=iqid_file,
                                output_dir=str(sample_output)
                            )
                        else:
                            # Fall back to single modal if no paired files
                            result = pipeline.process_iqid_stack(sample_dir, str(sample_output))
                    else:
                        # No paired sample, process as single modal
                        result = pipeline.process_iqid_stack(sample_dir, str(sample_output))
                
                elif pipeline_type == 'advanced':
                    # Advanced pipeline uses single representative slice
                    slice_files = sample.get('slice_files', [])
                    if slice_files:
                        representative_slice = slice_files[len(slice_files)//2]
                        result = pipeline.process_image(representative_slice, str(sample_output))
                    else:
                        result = pipeline.process_iqid_stack(sample_dir, str(sample_output))
                
                else:  # simple pipeline
                    result = pipeline.process_iqid_stack(sample_dir, str(sample_output))
                
                # Record end time
                processing_time = time.time() - start_time
                
                # Extract metrics from result
                success = result.get('status') == 'success' if result else False
                quality_score = result.get('quality_score') if result else None
                alignment_score = result.get('alignment_score') if result else None
                error_message = result.get('error') if result and not success else None
                
                # Count output files
                output_files_count = len(list(sample_output.glob('**/*'))) if sample_output.exists() else 0
                
                # Create metrics entry
                metric = EvaluationMetrics(
                    sample_id=sample_id,
                    dataset_type=dataset_type,
                    pipeline_type=pipeline_type,
                    processing_stage=processing_stage,
                    success=success,
                    processing_time=processing_time,
                    quality_score=quality_score,
                    alignment_score=alignment_score,
                    error_message=error_message,
                    output_files_count=output_files_count
                )
                
                metrics.append(metric)
                
                if success:
                    self.logger.info(f"    ✅ Success in {processing_time:.2f}s")
                else:
                    self.logger.warning(f"    ❌ Failed: {error_message}")
                
            except Exception as e:
                self.logger.error(f"    💥 Exception processing {sample_id}: {e}")
                self.logger.debug(traceback.format_exc())
                
                metric = EvaluationMetrics(
                    sample_id=sample_id,
                    dataset_type=dataset_type,
                    pipeline_type=pipeline_type,
                    processing_stage=processing_stage,
                    success=False,
                    processing_time=0.0,
                    error_message=str(e)
                )
                metrics.append(metric)
        
        success_count = sum(1 for m in metrics if m.success)
        self.logger.info(f"🎯 {pipeline_type} on {dataset_name}: {success_count}/{len(metrics)} successful")
        
        return metrics
    
    def run_comprehensive_evaluation(self, max_samples_per_dataset: int = None) -> Dict:
        """Run comprehensive evaluation across all datasets and pipelines"""
        self.logger.info("🚀 Starting comprehensive UCSF evaluation...")
        
        # Discover all datasets
        datasets = self.discover_all_datasets()
        
        # Evaluate each pipeline on each dataset
        pipelines_to_test = ['simple', 'advanced', 'combined']
        
        for dataset_name, discovery_result in datasets.items():
            if discovery_result is None:
                continue
                
            self.logger.info(f"\n📊 Evaluating dataset: {dataset_name}")
            
            for pipeline_type in pipelines_to_test:
                metrics = self.evaluate_pipeline_on_dataset(
                    pipeline_type, dataset_name, discovery_result, max_samples_per_dataset
                )
                self.evaluation_metrics.extend(metrics)
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save all results
        self.save_evaluation_results()
        
        self.logger.info("✅ Comprehensive evaluation completed!")
        return report
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        self.logger.info("📋 Generating comprehensive evaluation report...")
        
        # Convert metrics to DataFrame for easier analysis
        metrics_data = []
        for metric in self.evaluation_metrics:
            metrics_data.append({
                'sample_id': metric.sample_id,
                'dataset_type': metric.dataset_type,
                'pipeline_type': metric.pipeline_type,
                'processing_stage': metric.processing_stage,
                'success': metric.success,
                'processing_time': metric.processing_time,
                'quality_score': metric.quality_score,
                'alignment_score': metric.alignment_score,
                'output_files_count': metric.output_files_count,
                'error_message': metric.error_message
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Generate summary statistics
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_samples_evaluated': len(self.evaluation_metrics),
            'datasets_evaluated': len(set(m.dataset_type for m in self.evaluation_metrics)),
            'pipelines_evaluated': len(set(m.pipeline_type for m in self.evaluation_metrics)),
            'overall_success_rate': df['success'].mean() if not df.empty else 0,
            'dataset_summaries': [
                {
                    'dataset_type': ds.dataset_type,
                    'total_samples': ds.total_samples,
                    'iqid_samples': ds.iqid_samples,
                    'he_samples': ds.he_samples,
                    'paired_samples': ds.paired_samples,
                    'ready_for_3d': ds.ready_for_3d,
                    'stage_distribution': ds.stage_distribution,
                    'tissue_distribution': ds.tissue_distribution
                } for ds in self.dataset_summaries
            ],
            'pipeline_performance': {},
            'dataset_performance': {},
            'processing_stage_performance': {},
            'timing_analysis': {},
            'quality_analysis': {}
        }
        
        if not df.empty:
            # Pipeline performance
            for pipeline in df['pipeline_type'].unique():
                pipeline_df = df[df['pipeline_type'] == pipeline]
                report['pipeline_performance'][pipeline] = {
                    'success_rate': pipeline_df['success'].mean(),
                    'avg_processing_time': pipeline_df['processing_time'].mean(),
                    'total_samples': len(pipeline_df),
                    'successful_samples': pipeline_df['success'].sum(),
                    'avg_quality_score': pipeline_df['quality_score'].mean() if pipeline_df['quality_score'].notna().any() else None,
                    'avg_alignment_score': pipeline_df['alignment_score'].mean() if pipeline_df['alignment_score'].notna().any() else None
                }
            
            # Dataset performance
            for dataset in df['dataset_type'].unique():
                dataset_df = df[df['dataset_type'] == dataset]
                report['dataset_performance'][dataset] = {
                    'success_rate': dataset_df['success'].mean(),
                    'avg_processing_time': dataset_df['processing_time'].mean(),
                    'total_samples': len(dataset_df),
                    'successful_samples': dataset_df['success'].sum()
                }
            
            # Processing stage performance
            for stage in df['processing_stage'].unique():
                stage_df = df[df['processing_stage'] == stage]
                report['processing_stage_performance'][stage] = {
                    'success_rate': stage_df['success'].mean(),
                    'avg_processing_time': stage_df['processing_time'].mean(),
                    'total_samples': len(stage_df)
                }
            
            # Timing analysis
            report['timing_analysis'] = {
                'fastest_processing_time': df['processing_time'].min(),
                'slowest_processing_time': df['processing_time'].max(),
                'median_processing_time': df['processing_time'].median(),
                'avg_processing_time': df['processing_time'].mean()
            }
            
            # Quality analysis (if available)
            if df['quality_score'].notna().any():
                report['quality_analysis'] = {
                    'avg_quality_score': df['quality_score'].mean(),
                    'min_quality_score': df['quality_score'].min(),
                    'max_quality_score': df['quality_score'].max(),
                    'median_quality_score': df['quality_score'].median()
                }
        
        return report
    
    def save_evaluation_results(self):
        """Save all evaluation results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed metrics as JSON
        metrics_file = self.output_path / f"detailed_metrics_{timestamp}.json"
        metrics_data = [
            {
                'sample_id': m.sample_id,
                'dataset_type': m.dataset_type,
                'pipeline_type': m.pipeline_type,
                'processing_stage': m.processing_stage,
                'success': m.success,
                'processing_time': m.processing_time,
                'quality_score': m.quality_score,
                'alignment_score': m.alignment_score,
                'output_files_count': m.output_files_count,
                'error_message': m.error_message,
                'memory_usage_mb': m.memory_usage_mb
            } for m in self.evaluation_metrics
        ]
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"💾 Detailed metrics saved to: {metrics_file}")
        
        # Save as CSV for easier analysis
        df = pd.DataFrame(metrics_data)
        csv_file = self.output_path / f"evaluation_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        self.logger.info(f"💾 CSV results saved to: {csv_file}")
        
        # Generate and save visualizations
        self.create_evaluation_visualizations(df, timestamp)
    
    def create_evaluation_visualizations(self, df: pd.DataFrame, timestamp: str):
        """Create comprehensive visualization plots"""
        if df.empty:
            self.logger.warning("No data available for visualizations")
            return
        
        self.logger.info("📊 Creating evaluation visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Success rate by pipeline
        plt.subplot(3, 4, 1)
        if 'pipeline_type' in df.columns and 'success' in df.columns:
            pipeline_success = df.groupby('pipeline_type')['success'].mean()
            pipeline_success.plot(kind='bar', color='skyblue')
            plt.title('Success Rate by Pipeline')
            plt.ylabel('Success Rate')
            plt.xticks(rotation=45)
        
        # 2. Success rate by dataset
        plt.subplot(3, 4, 2)
        if 'dataset_type' in df.columns and 'success' in df.columns:
            dataset_success = df.groupby('dataset_type')['success'].mean()
            dataset_success.plot(kind='bar', color='lightgreen')
            plt.title('Success Rate by Dataset')
            plt.ylabel('Success Rate')
            plt.xticks(rotation=45)
        
        # 3. Processing time distribution
        plt.subplot(3, 4, 3)
        if 'processing_time' in df.columns:
            plt.hist(df['processing_time'], bins=20, color='coral', alpha=0.7)
            plt.title('Processing Time Distribution')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency')
        
        # 4. Pipeline vs Dataset heatmap
        plt.subplot(3, 4, 4)
        if 'pipeline_type' in df.columns and 'dataset_type' in df.columns and 'success' in df.columns:
            pivot_table = df.pivot_table(values='success', index='pipeline_type', columns='dataset_type', aggfunc='mean')
            sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0.5)
            plt.title('Success Rate Heatmap')
        
        # 5. Processing stage performance
        plt.subplot(3, 4, 5)
        if 'processing_stage' in df.columns and 'success' in df.columns:
            stage_success = df.groupby('processing_stage')['success'].mean()
            stage_success.plot(kind='bar', color='plum')
            plt.title('Success Rate by Processing Stage')
            plt.ylabel('Success Rate')
            plt.xticks(rotation=45)
        
        # 6. Quality scores distribution
        plt.subplot(3, 4, 6)
        if 'quality_score' in df.columns and df['quality_score'].notna().any():
            plt.hist(df['quality_score'].dropna(), bins=15, color='gold', alpha=0.7)
            plt.title('Quality Score Distribution')
            plt.xlabel('Quality Score')
            plt.ylabel('Frequency')
        
        # 7. Processing time by pipeline
        plt.subplot(3, 4, 7)
        if 'pipeline_type' in df.columns and 'processing_time' in df.columns:
            df.boxplot(column='processing_time', by='pipeline_type', ax=plt.gca())
            plt.title('Processing Time by Pipeline')
            plt.suptitle('')  # Remove automatic title
            plt.ylabel('Time (seconds)')
        
        # 8. Success vs Processing Time scatter
        plt.subplot(3, 4, 8)
        if 'processing_time' in df.columns and 'success' in df.columns:
            success_colors = ['red' if not s else 'green' for s in df['success']]
            plt.scatter(df['processing_time'], df['success'], c=success_colors, alpha=0.6)
            plt.title('Success vs Processing Time')
            plt.xlabel('Processing Time (seconds)')
            plt.ylabel('Success')
        
        # 9. Output files count distribution
        plt.subplot(3, 4, 9)
        if 'output_files_count' in df.columns:
            plt.hist(df['output_files_count'], bins=15, color='lightblue', alpha=0.7)
            plt.title('Output Files Count Distribution')
            plt.xlabel('Number of Output Files')
            plt.ylabel('Frequency')
        
        # 10. Error analysis
        plt.subplot(3, 4, 10)
        if 'error_message' in df.columns:
            error_df = df[df['error_message'].notna()]
            if not error_df.empty:
                error_counts = error_df['error_message'].value_counts().head(5)
                error_counts.plot(kind='barh', color='salmon')
                plt.title('Top 5 Error Types')
                plt.xlabel('Frequency')
        
        # 11. Sample count by dataset and pipeline
        plt.subplot(3, 4, 11)
        if 'dataset_type' in df.columns and 'pipeline_type' in df.columns:
            sample_counts = df.groupby(['dataset_type', 'pipeline_type']).size().unstack(fill_value=0)
            sample_counts.plot(kind='bar', stacked=True)
            plt.title('Sample Count by Dataset and Pipeline')
            plt.ylabel('Number of Samples')
            plt.xticks(rotation=45)
            plt.legend(title='Pipeline Type')
        
        # 12. Success rate over time (if we had timestamps)
        plt.subplot(3, 4, 12)
        # For now, show cumulative success rate
        if 'success' in df.columns:
            cumulative_success = df['success'].expanding().mean()
            plt.plot(cumulative_success, color='purple')
            plt.title('Cumulative Success Rate')
            plt.xlabel('Sample Index')
            plt.ylabel('Cumulative Success Rate')
        
        plt.tight_layout()
        
        # Save the dashboard
        dashboard_file = self.output_path / f"evaluation_dashboard_{timestamp}.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Evaluation dashboard saved to: {dashboard_file}")
        
        # Create additional detailed plots
        self.create_detailed_plots(df, timestamp)
    
    def create_detailed_plots(self, df: pd.DataFrame, timestamp: str):
        """Create additional detailed analysis plots"""
        
        # Pipeline comparison plot
        if 'pipeline_type' in df.columns and 'success' in df.columns and 'processing_time' in df.columns:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Success rate comparison
            pipeline_stats = df.groupby('pipeline_type').agg({
                'success': 'mean',
                'processing_time': 'mean',
                'quality_score': 'mean',
                'output_files_count': 'mean'
            }).round(3)
            
            pipeline_stats['success'].plot(kind='bar', ax=ax1, color='lightblue')
            ax1.set_title('Success Rate by Pipeline')
            ax1.set_ylabel('Success Rate')
            ax1.tick_params(axis='x', rotation=45)
            
            # Processing time comparison
            pipeline_stats['processing_time'].plot(kind='bar', ax=ax2, color='lightcoral')
            ax2.set_title('Average Processing Time by Pipeline')
            ax2.set_ylabel('Time (seconds)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Quality score comparison (if available)
            if pipeline_stats['quality_score'].notna().any():
                pipeline_stats['quality_score'].plot(kind='bar', ax=ax3, color='lightgreen')
                ax3.set_title('Average Quality Score by Pipeline')
                ax3.set_ylabel('Quality Score')
                ax3.tick_params(axis='x', rotation=45)
            
            # Output files comparison
            pipeline_stats['output_files_count'].plot(kind='bar', ax=ax4, color='gold')
            ax4.set_title('Average Output Files by Pipeline')
            ax4.set_ylabel('File Count')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            pipeline_comparison_file = self.output_path / f"pipeline_comparison_{timestamp}.png"
            plt.savefig(pipeline_comparison_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Pipeline comparison saved to: {pipeline_comparison_file}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive UCSF Dataset Evaluation")
    parser.add_argument('--data', type=str, default='data', 
                       help='Path to UCSF data directory')
    parser.add_argument('--output', type=str, default='evaluation/outputs',
                       help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per dataset to evaluate')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup paths
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Data path does not exist: {data_path}")
        return 1
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    try:
        evaluator = ComprehensiveUCSFEvaluator(str(data_path), str(output_path))
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        report = evaluator.run_comprehensive_evaluation(args.max_samples)
        
        # Save final report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f"comprehensive_evaluation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Comprehensive evaluation completed!")
        print(f"📁 Results saved to: {output_path}")
        print(f"📋 Final report: {report_file}")
        print(f"\n📊 Summary:")
        print(f"   - Total samples evaluated: {report['total_samples_evaluated']}")
        print(f"   - Overall success rate: {report['overall_success_rate']:.2%}")
        print(f"   - Datasets evaluated: {report['datasets_evaluated']}")
        print(f"   - Pipelines evaluated: {report['pipelines_evaluated']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
