#!/usr/bin/env python3
"""
Segmentation Pipeline Interface

Thin interface for ReUpload Raw → Segmented validation pipeline.
Delegates core implementation to iqid_alphas.pipelines.SegmentationPipeline.

Usage:
    python segmentation.py --data /path/to/ReUpload --output results/segmentation
"""

import sys
import argparse
from pathlib import Path

# Add iqid_alphas to path
sys.path.insert(0, str(Path(__file__).parent))

# TODO: Import core pipeline when implemented
try:
    from iqid_alphas.pipelines.segmentation import SegmentationPipeline
except ImportError as e:
    print(f"⚠️  TODO: Core pipeline not yet implemented: {e}")
    print("     This interface will delegate to iqid_alphas.pipelines.SegmentationPipeline")
    SegmentationPipeline = None


def print_pipeline_summary(results: dict):
    """Print pipeline execution summary"""
    print("\n" + "="*60)
    print("RAW → SEGMENTED PIPELINE RESULTS")
    print("="*60)
    
    if 'error' in results:
        print(f"❌ Pipeline failed: {results['error']}")
        return
    
    print(f"📊 Total sample groups processed: {results.get('total_groups', 0)}")
    print(f"✅ Successful segmentations: {results.get('successful_segmentations', 0)}")
    print(f"❌ Failed segmentations: {results.get('failed_segmentations', 0)}")
    
    success_rate = results.get('success_rate', 0.0)
    print(f"📈 Success rate: {success_rate:.1%}")
    
    if 'avg_iou' in results:
        print(f"📈 Average IoU score: {results['avg_iou']:.3f}")
    if 'avg_dice' in results:
        print(f"📈 Average Dice score: {results['avg_dice']:.3f}")
    
    total_time = results.get('total_time', 0.0)
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    
    if 'tissue_type_performance' in results and results['tissue_type_performance']:
        print(f"\n🧬 Tissue Type Performance:")
        for tissue, perf in results['tissue_type_performance'].items():
            print(f"   {tissue}: IoU {perf.get('avg_iou', 0):.3f}, "
                  f"Dice {perf.get('avg_dice', 0):.3f}")


def main():
    """Main entry point for segmentation pipeline interface"""
    parser = argparse.ArgumentParser(
        description="Segmentation Pipeline - Raw iQID to Segmented Tissues Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with default settings
  python segmentation.py --data /path/to/ReUpload --output results/segmentation
  
  # Limit to 5 sample groups for testing
  python segmentation.py --data /path/to/ReUpload --output results/test --max-samples 5
  
  # Use custom configuration
  python segmentation.py --data /path/to/ReUpload --config custom_config.json
        """
    )
    
    parser.add_argument("--data", required=True, 
                       help="Path to ReUpload dataset root directory")
    parser.add_argument("--output", default="outputs/segmentation", 
                       help="Output directory for results and visualizations")
    parser.add_argument("--config", 
                       help="Path to configuration file (JSON)")
    parser.add_argument("--max-samples", type=int, 
                       help="Maximum number of sample groups to process (for testing)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Validate inputs
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data path does not exist: {data_path}")
        return 1
    
    output_path = Path(args.output)
    
    print("🔬 IQID-Alphas Segmentation Pipeline")
    print("="*50)
    print(f"📁 Data path: {data_path}")
    print(f"📁 Output path: {output_path}")
    if args.config:
        print(f"⚙️  Config: {args.config}")
    if args.max_samples:
        print(f"🔢 Max samples: {args.max_samples}")
    print()
    
    # Check if core pipeline is available
    if SegmentationPipeline is None:
        print("⚠️  TODO: Core SegmentationPipeline not yet implemented")
        print("     This interface script is ready to use once the core pipeline is available.")
        print()
        print("📋 Expected functionality:")
        print("   ✓ Discover raw iQID images in ReUpload dataset")
        print("   ✓ Group samples by shared raw image (D1M1, etc.)")
        print("   ✓ Run automated tissue separation on each raw image")
        print("   ✓ Compare results against ground truth in 1_segmented/ directories")
        print("   ✓ Calculate IoU, Dice, precision, recall metrics")
        print("   ✓ Generate visualizations and comprehensive report")
        print()
        print("🚧 Implementation status: Core pipeline classes with TODO markers created")
        return 0
    
    try:
        # Initialize pipeline
        print("🚀 Initializing segmentation pipeline...")
        pipeline = SegmentationPipeline(
            data_path=str(data_path),
            output_dir=str(output_path),
            config_path=args.config
        )
        
        # Execute validation
        print("⚡ Running segmentation validation...")
        results = pipeline.run_validation(max_samples=args.max_samples)
        
        # Generate comprehensive report
        print("📊 Generating comprehensive report...")
        pipeline.generate_report(results)
        
        # Print summary
        print_pipeline_summary(results)
        
        # Show output location
        print(f"\n📁 Results saved to: {output_path}")
        print(f"📊 Visualizations available in: {output_path}/visualizations/")
        
        return 0 if results.get('status') != 'failed' else 1
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
