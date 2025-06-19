#!/usr/bin/env python3
"""
Co-registration Pipeline CLI Wrapper (package version)
"""
import sys
import argparse
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iqid_alphas.pipelines.coregistration.pipeline import CoregistrationPipeline
from iqid_alphas.pipelines.coregistration.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="IQID-Alphas Co-registration Pipeline - iQID + H&E Multi-modal Registration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pipelines/coregistration.py --data /path/to/datapush1 --output results/coregistration
    python pipelines/coregistration.py --data /path/to/datapush1 --output results/coregistration \
        --config configs/coregistration_custom.json
    python pipelines/coregistration.py --data /path/to/datapush1 --output results/coregistration \
        --max-samples 10
        """,
    )
    parser.add_argument(
        "--data", required=True, help="Path to DataPush1 dataset directory"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for coregistration results"
    )
    parser.add_argument(
        "--config", help="Path to custom configuration JSON file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of sample pairs to process (useful for testing)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging output"
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data path does not exist: {data_path}")
        sys.exit(1)
    if not data_path.is_dir():
        print(f"❌ Error: Data path is not a directory: {data_path}")
        sys.exit(1)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("🔬 IQID-Alphas Co-registration Pipeline")
    print("=" * 50)
    print(f"📁 Input Data: {data_path}")
    print(f"📁 Output Dir: {output_path}")
    if args.config:
        print(f"⚙️  Config File: {args.config}")
    if args.max_samples:
        print(f"🔢 Max Samples: {args.max_samples}")
    print("=" * 50)

    try:
        config = load_config(args.config) if args.config else None
        result = CoregistrationPipeline(
            config=config,
            data_path=str(data_path),
            output_dir=str(output_path),
            max_samples=args.max_samples,
        ).run()
        print("\n✅ CO-REGISTRATION COMPLETED")
        print("=" * 50)
        print(f"🔬 Total Pairs: {result['total_pairs']}")
        print(f"✅ Successful: {result['successful_pairings']}")
        print(f"❌ Failed: {result['failed_pairs']}")
        print(f"📈 Success Rate: {result['success_rate']:.1%}")
        print(f"⏱️  Duration: {result['duration']:.2f}s")
        print(f"📁 Results: {result['output_dir']}")
        metrics = result.get("metrics")
        if metrics:
            print("\n📊 Quality Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
