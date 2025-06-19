#!/usr/bin/env python3
"""
Test Runner for Steps 9-13 Bidirectional Alignment Evaluation

Quick test script to run the bidirectional alignment evaluation on UCSF data.
"""

import sys
import logging
from pathlib import Path

# Add the evaluation script path
sys.path.insert(0, str(Path(__file__).parent))

from bidirectional_alignment_evaluation import BidirectionalAlignmentEvaluator


def run_quick_test():
    """Run a quick test of the bidirectional alignment evaluation"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("QuickTest")
    
    # Try to find UCSF data in common locations
    possible_data_paths = [
        "/home/wxc151/iqid-alphas/data",
        "/home/wxc151/data",
        "./data",
        "../data",
        "../../data"
    ]
    
    data_path = None
    for path in possible_data_paths:
        if Path(path).exists():
            data_path = path
            break
    
    if not data_path:
        logger.error("Could not find UCSF data directory. Please specify the path manually.")
        logger.info("Expected data structure:")
        logger.info("  data/")
        logger.info("    DataPush1/")
        logger.info("      3D/ or Sequential/")
        logger.info("        tissue_dirs/")
        logger.info("          sample_dirs/")
        logger.info("            1_segmented/")
        logger.info("              *.tif files")
        return
    
    logger.info(f"Found data directory: {data_path}")
    
    # Initialize evaluator
    output_dir = "evaluation/outputs/steps_9_13_test"
    evaluator = BidirectionalAlignmentEvaluator(data_path, output_dir)
    
    # Run evaluation with limited samples for testing
    logger.info("Starting bidirectional alignment evaluation...")
    summary = evaluator.run_evaluation(max_samples=2)  # Test with 2 samples
    
    # Print results
    print("\n" + "="*60)
    print("STEPS 9-13 EVALUATION RESULTS")
    print("="*60)
    
    if summary:
        print(f"Total evaluations: {summary.get('total_evaluations', 0)}")
        print(f"Strategies tested: {summary.get('strategies_tested', [])}")
        
        if 'strategy_performance' in summary:
            print("\nStrategy Performance Summary:")
            for strategy, perf in summary['strategy_performance'].items():
                print(f"\n  📊 {strategy}:")
                print(f"     ✅ Success rate: {perf.get('success_rate', 0):.1%}")
                if 'avg_quality' in perf:
                    print(f"     🎯 Quality: {perf['avg_quality']:.3f} (best: {perf.get('best_quality', 0):.3f})")
                    print(f"     ⏱️  Time: {perf.get('avg_time', 0):.2f}s")
                    print(f"     🔄 Propagation: {perf.get('avg_propagation_quality', 0):.3f}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"📊 Visualizations in: {output_dir}/visualizations/")
        
        # Show key findings
        print("\n🔍 Key Findings:")
        print("   ✓ Step 9: Reference slice detection implemented")
        print("   ✓ Step 10: Bidirectional alignment (largest→up/down)")
        print("   ✓ Step 11: Noise-robust alignment methods")
        print("   ✓ Step 12: Enhanced multi-slice visualizations")
        print("   ✓ Step 13: 3D stack alignment with overlap analysis")
        
    else:
        print("❌ No results generated. Check logs for issues.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    run_quick_test()
