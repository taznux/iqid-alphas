#!/usr/bin/env python3
"""
Co-registration Pipeline Interface

Thin interface for DataPush1 iQID+H&E co-registration pipeline.
Delegates core implementation to iqid_alphas.pipelines.CoregistrationPipeline.

Usage:
    python coregistration.py --data /path/to/DataPush1 --output results/coregistration
"""

import sys
import argparse
from pathlib import Path

# Add iqid_alphas to path
sys.path.insert(0, str(Path(__file__).parent))

# TODO: Import core pipeline when implemented
try:
    from iqid_alphas.pipelines.coregistration import CoregistrationPipeline
except ImportError as e:
    print(f"⚠️  TODO: Core pipeline not yet implemented: {e}")
    print("     This interface will delegate to iqid_alphas.pipelines.CoregistrationPipeline")
    CoregistrationPipeline = None


def print_coregistration_summary(results: dict):
    """Print co-registration pipeline execution summary"""
    print("\n" + "="*60)
    print("iQID+H&E CO-REGISTRATION PIPELINE RESULTS")
    print("="*60)
    
    if 'error' in results:
        print(f"❌ Pipeline failed: {results['error']}")
        return
    
    print(f"📊 Total sample pairs processed: {results.get('total_pairs', 0)}")
    print(f"✅ Successfully paired samples: {results.get('successful_pairings', 0)}")
    print(f"🔗 Successfully registered pairs: {results.get('successful_registrations', 0)}")
    
    registration_success_rate = results.get('registration_success_rate', 0.0)
    print(f"📈 Registration success rate: {registration_success_rate:.1%}")
    
    if 'avg_mutual_information' in results:
        print(f"📈 Average mutual information: {results['avg_mutual_information']:.3f}")
    if 'avg_structural_similarity' in results:
        print(f"📈 Average structural similarity: {results['avg_structural_similarity']:.3f}")
    if 'avg_confidence' in results:
        print(f"🎯 Average registration confidence: {results['avg_confidence']:.3f}")
    
    total_time = results.get('total_time', 0.0)
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    
    if 'tissue_performance' in results and results['tissue_performance']:
        print(f"\n🧬 Tissue Type Performance:")
        for tissue, perf in results['tissue_performance'].items():
            pairs = perf.get('total_pairs', 0)
            success_rate = perf.get('success_rate', 0.0)
            print(f"   {tissue}: {pairs} pairs, Success Rate {success_rate:.1%}")
    
    if 'method_performance' in results and results['method_performance']:
        print(f"\n🔬 Registration Method Performance:")
        for method, perf in results['method_performance'].items():
            success_rate = perf.get('success_rate', 0.0)
            avg_quality = perf.get('avg_quality', 0.0)
            print(f"   {method}: Success Rate {success_rate:.1%}, Quality {avg_quality:.3f}")


def main():
    """Main entry point for co-registration pipeline interface"""
    parser = argparse.ArgumentParser(
        description="Co-registration Pipeline - iQID and H&E Multi-modal Registration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic co-registration with mutual information
  python coregistration.py --data /path/to/DataPush1 --output results/coregistration
  
  # Try feature-based registration
  python coregistration.py --data /path/to/DataPush1 --registration-method feature_based
  
  # Use fuzzy matching for sample pairing
  python coregistration.py --data /path/to/DataPush1 --pairing-strategy fuzzy_matching
  
  # Limit to 3 pairs for testing
  python coregistration.py --data /path/to/DataPush1 --output results/test --max-samples 3
        """
    )
    
    parser.add_argument("--data", required=True, 
                       help="Path to DataPush1 dataset root directory")
    parser.add_argument("--output", default="outputs/coregistration", 
                       help="Output directory for results and visualizations")
    parser.add_argument("--registration-method", default="mutual_information",
                       choices=["mutual_information", "feature_based", "intensity_based"],
                       help="Multi-modal registration method")
    parser.add_argument("--pairing-strategy", default="automatic",
                       choices=["automatic", "manual", "fuzzy_matching"],
                       help="Strategy for pairing iQID and H&E samples")
    parser.add_argument("--config", 
                       help="Path to configuration file (JSON)")
    parser.add_argument("--max-samples", type=int, 
                       help="Maximum number of sample pairs to process (for testing)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Validate inputs
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data path does not exist: {data_path}")
        return 1
    
    output_path = Path(args.output)
    
    print("🔬 IQID-Alphas Co-registration Pipeline")
    print("="*50)
    print(f"📁 Data path: {data_path}")
    print(f"📁 Output path: {output_path}")
    print(f"⚙️  Registration method: {args.registration_method}")
    print(f"🔗 Pairing strategy: {args.pairing_strategy}")
    if args.config:
        print(f"⚙️  Config: {args.config}")
    if args.max_samples:
        print(f"🔢 Max samples: {args.max_samples}")
    print()
    
    # Check if core pipeline is available
    if CoregistrationPipeline is None:
        print("⚠️  TODO: Core CoregistrationPipeline not yet implemented")
        print("     This interface script is ready to use once the core pipeline is available.")
        print()
        print("📋 Expected functionality:")
        print("   ✓ Discover iQID samples in DataPush1/iQID/3D/")
        print("   ✓ Discover H&E samples in DataPush1/HE/3D/")
        print("   ✓ Intelligent sample pairing based on naming patterns")
        print("   ✓ Multi-modal registration (mutual information, feature-based)")
        print("   ✓ Quality assessment using proxy metrics (no ground truth)")
        print("   ✓ Confidence scoring for registration results")
        print("   ✓ Tissue-specific performance analysis")
        print("   ✓ Generate multi-modal fusion visualizations")
        print()
        print("📊 Assessment Metrics (No Ground Truth Available):")
        print("   • Mutual Information: Cross-modal information content")
        print("   • Structural Similarity: Preserved anatomical features")
        print("   • Feature Matching: Keypoint correspondence quality")
        print("   • Registration Confidence: Transform stability assessment")
        print()
        print("🚧 Implementation status: Core pipeline classes with TODO markers created")
        return 0
    
    try:
        # Initialize pipeline
        print("🚀 Initializing co-registration pipeline...")
        pipeline = CoregistrationPipeline(
            data_path=str(data_path),
            output_dir=str(output_path),
            registration_method=args.registration_method,
            pairing_strategy=args.pairing_strategy,
            config_path=args.config
        )
        
        # Execute evaluation
        print("⚡ Running co-registration evaluation...")
        results = pipeline.run_evaluation(max_samples=args.max_samples)
        
        # Generate comprehensive report
        print("📊 Generating comprehensive report...")
        pipeline.generate_report(results)
        
        # Print summary
        print_coregistration_summary(results)
        
        # Show output location
        print(f"\n📁 Results saved to: {output_path}")
        print(f"📊 Visualizations available in: {output_path}/visualizations/")
        
        # Show method-specific insights
        if results.get('method_performance'):
            print(f"\n🔍 Key Insights:")
            print(f"   🔗 Sample pairing strategy: {args.pairing_strategy}")
            print(f"   ⚙️  Registration method: {args.registration_method}")
            print(f"   📊 Assessment: Proxy metrics (no ground truth)")
            print(f"   🧬 Multi-modal: iQID (functional) + H&E (structural)")
        
        # Clinical interpretation notes
        print(f"\n💡 Clinical Interpretation Notes:")
        print(f"   • High mutual information indicates good cross-modal alignment")
        print(f"   • Structural similarity preserves anatomical landmarks")
        print(f"   • Registration confidence indicates transform reliability")
        print(f"   • Tissue-specific performance guides clinical applications")
        
        return 0 if results.get('status') != 'failed' else 1
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
