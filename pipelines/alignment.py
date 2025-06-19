#!/usr/bin/env python3
"""
Alignment Pipeline Interface

Thin interface for ReUpload Segmented → Aligned validation pipeline.
Delegates core implementation to iqid_alphas.pipelines.AlignmentPipeline.

Usage:
    python alignment.py --data /path/to/ReUpload --output results/alignment
"""

import sys
import argparse
from pathlib import Path

# Add iqid_alphas to path
sys.path.insert(0, str(Path(__file__).parent))

# TODO: Import core pipeline when implemented
try:
    from iqid_alphas.pipelines.alignment import AlignmentPipeline
except ImportError as e:
    print(f"⚠️  TODO: Core pipeline not yet implemented: {e}")
    print("     This interface will delegate to iqid_alphas.pipelines.AlignmentPipeline")
    AlignmentPipeline = None


def print_alignment_summary(results: dict):
    """Print alignment pipeline execution summary"""
    print("\n" + "="*60)
    print("SEGMENTED → ALIGNED PIPELINE RESULTS")
    print("="*60)
    
    if 'error' in results:
        print(f"❌ Pipeline failed: {results['error']}")
        return
    
    print(f"📊 Total samples processed: {results.get('total_samples', 0)}")
    print(f"✅ Successful alignments: {results.get('successful_alignments', 0)}")
    print(f"❌ Failed alignments: {results.get('failed_alignments', 0)}")
    
    success_rate = results.get('success_rate', 0.0)
    print(f"📈 Success rate: {success_rate:.1%}")
    
    if 'avg_ssim' in results:
        print(f"📈 Average SSIM score: {results['avg_ssim']:.3f}")
    if 'avg_mi' in results:
        print(f"📈 Average mutual information: {results['avg_mi']:.3f}")
    if 'avg_propagation_quality' in results:
        print(f"🎯 Average propagation quality: {results['avg_propagation_quality']:.3f}")
    
    total_time = results.get('total_time', 0.0)
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    
    if 'method_performance' in results and results['method_performance']:
        print(f"\n🔬 Method Performance:")
        for method, perf in results['method_performance'].items():
            print(f"   {method}: Success Rate {perf.get('success_rate', 0):.1%}, "
                  f"Quality {perf.get('avg_quality', 0):.3f}")


def main():
    """Main entry point for alignment pipeline interface"""
    parser = argparse.ArgumentParser(
        description="Alignment Pipeline - Segmented to Aligned Slices Validation (Steps 9-13)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with bidirectional alignment
  python alignment.py --data /path/to/ReUpload --output results/alignment
  
  # Test different alignment methods
  python alignment.py --data /path/to/ReUpload --alignment-method sequential
  
  # Use quality-based reference slice selection
  python alignment.py --data /path/to/ReUpload --reference-method quality_based
  
  # Limit to 3 samples for testing
  python alignment.py --data /path/to/ReUpload --output results/test --max-samples 3
        """
    )
    
    parser.add_argument("--data", required=True, 
                       help="Path to ReUpload dataset root directory")
    parser.add_argument("--output", default="outputs/alignment", 
                       help="Output directory for results and visualizations")
    parser.add_argument("--alignment-method", default="bidirectional", 
                       choices=["bidirectional", "sequential", "reference_based"],
                       help="Alignment strategy to use")
    parser.add_argument("--reference-method", default="largest", 
                       choices=["largest", "central", "quality_based"],
                       help="Reference slice selection method")
    parser.add_argument("--config", 
                       help="Path to configuration file (JSON)")
    parser.add_argument("--max-samples", type=int, 
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Validate inputs
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data path does not exist: {data_path}")
        return 1
    
    output_path = Path(args.output)
    
    print("🔬 IQID-Alphas Alignment Pipeline")
    print("="*50)
    print(f"📁 Data path: {data_path}")
    print(f"📁 Output path: {output_path}")
    print(f"⚙️  Alignment method: {args.alignment_method}")
    print(f"🎯 Reference method: {args.reference_method}")
    if args.config:
        print(f"⚙️  Config: {args.config}")
    if args.max_samples:
        print(f"🔢 Max samples: {args.max_samples}")
    print()
    
    # Check if core pipeline is available
    if AlignmentPipeline is None:
        print("⚠️  TODO: Core AlignmentPipeline not yet implemented")
        print("     This interface script is ready to use once the core pipeline is available.")
        print()
        print("📋 Expected functionality (Steps 9-13):")
        print("   ✓ Step 9: Find reference slice (largest slice as reference)")
        print("   ✓ Step 10: Bidirectional alignment (largest→up, largest→down)")
        print("   ✓ Step 11: Noise-robust methods (mask-based, template-based)")
        print("   ✓ Step 12: Enhanced multi-slice visualizations")
        print("   ✓ Step 13: 3D stack alignment with overlapping analysis")
        print("   ✓ Compare results against ground truth in 2_aligned/ directories")
        print("   ✓ Calculate SSIM, mutual information, propagation quality")
        print("   ✓ Generate comprehensive visualizations and report")
        print()
        print("🚧 Implementation status: Core pipeline classes with TODO markers created")
        return 0
    
    try:
        # Initialize pipeline
        print("🚀 Initializing alignment pipeline...")
        pipeline = AlignmentPipeline(
            data_path=str(data_path),
            output_dir=str(output_path),
            alignment_method=args.alignment_method,
            reference_method=args.reference_method,
            config_path=args.config
        )
        
        # Execute validation
        print("⚡ Running alignment validation...")
        results = pipeline.run_validation(max_samples=args.max_samples)
        
        # Generate comprehensive report
        print("📊 Generating comprehensive report...")
        pipeline.generate_report(results)
        
        # Print summary
        print_alignment_summary(results)
        
        # Show output location
        print(f"\n📁 Results saved to: {output_path}")
        print(f"📊 Visualizations available in: {output_path}/visualizations/")
        
        # Show method-specific insights
        if results.get('method_performance'):
            print(f"\n🔍 Key Findings:")
            print(f"   ✓ Step 9: Reference slice detection implemented")
            print(f"   ✓ Step 10: Bidirectional alignment ({args.alignment_method})")
            print(f"   ✓ Step 11: Noise-robust alignment methods")
            print(f"   ✓ Step 12: Enhanced multi-slice visualizations")
            print(f"   ✓ Step 13: 3D stack alignment with overlap analysis")
        
        return 0 if results.get('status') != 'failed' else 1
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
