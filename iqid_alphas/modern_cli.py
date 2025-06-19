#!/usr/bin/env python3
"""
IQID-Alphas Unified CLI

Modern CLI interface with subcommands for all pipelines:
- python -m iqid_alphas segmentation <args>
- python -m iqid_alphas alignment <args>  
- python -m iqid_alphas coregistration <args>
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from .pipelines.segmentation import run_segmentation_pipeline
from .pipelines.alignment import run_alignment_pipeline
from .pipelines.coregistration import CoregistrationPipeline


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_segmentation(args) -> int:
    """Run segmentation pipeline."""
    try:
        results = run_segmentation_pipeline(
            data_path=args.data,
            output_dir=args.output,
            max_samples=args.max_samples,
            config_path=args.config
        )
        
        print(f"✅ Segmentation completed: {results['success_rate']:.1%} success rate")
        return 0
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        return 1


def cmd_alignment(args) -> int:
    """Run alignment pipeline."""
    try:
        results = run_alignment_pipeline(
            data_path=args.data,
            output_dir=args.output,
            max_samples=args.max_samples,
            config_path=args.config
        )
        
        print(f"✅ Alignment completed: {results['success_rate']:.1%} success rate")
        return 0
        
    except Exception as e:
        print(f"❌ Alignment failed: {e}")
        return 1


def cmd_coregistration(args) -> int:
    """Run coregistration pipeline."""
    try:
        pipeline = CoregistrationPipeline(
            data_path=args.data,
            output_dir=args.output,
            registration_method=args.method,
            pairing_strategy=args.pairing,
            config_path=args.config
        )
        
        results = pipeline.run_evaluation(max_samples=args.max_samples)
        success_rate = results.get('registration_success_rate', 0.0)
        
        print(f"✅ Coregistration completed: {success_rate:.1%} success rate")
        return 0
        
    except Exception as e:
        print(f"❌ Coregistration failed: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create unified CLI parser."""
    parser = argparse.ArgumentParser(
        prog='iqid_alphas',
        description='IQID-Alphas Pipeline Suite'
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Segmentation subcommand
    seg_parser = subparsers.add_parser('segmentation', help='Run segmentation pipeline')
    seg_parser.add_argument('--data', required=True, help='Path to ReUpload dataset')
    seg_parser.add_argument('--output', required=True, help='Output directory')
    seg_parser.add_argument('--config', help='Configuration file path')
    seg_parser.add_argument('--max-samples', type=int, help='Maximum samples to process')
    seg_parser.set_defaults(func=cmd_segmentation)
    
    # Alignment subcommand
    align_parser = subparsers.add_parser('alignment', help='Run alignment pipeline')
    align_parser.add_argument('--data', required=True, help='Path to ReUpload dataset')
    align_parser.add_argument('--output', required=True, help='Output directory')
    align_parser.add_argument('--config', help='Configuration file path')
    align_parser.add_argument('--max-samples', type=int, help='Maximum samples to process')
    align_parser.set_defaults(func=cmd_alignment)
    
    # Coregistration subcommand
    coreg_parser = subparsers.add_parser('coregistration', help='Run coregistration pipeline')
    coreg_parser.add_argument('--data', required=True, help='Path to DataPush1 dataset')
    coreg_parser.add_argument('--output', required=True, help='Output directory')
    coreg_parser.add_argument('--method', default='mutual_information', 
                              choices=['mutual_information', 'feature_based', 'intensity_based'],
                              help='Registration method')
    coreg_parser.add_argument('--pairing', default='automatic',
                              choices=['automatic', 'manual', 'fuzzy_matching'],
                              help='Sample pairing strategy')
    coreg_parser.add_argument('--config', help='Configuration file path')
    coreg_parser.add_argument('--max-samples', type=int, help='Maximum samples to process')
    coreg_parser.set_defaults(func=cmd_coregistration)
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
