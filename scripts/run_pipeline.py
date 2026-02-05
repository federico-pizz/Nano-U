"""CLI tool to run Nano-U training pipelines (see README and REFACTURING_DOCUMENTATION.md)."""

import os
import sys
import argparse
import json
import traceback
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path for imports
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.pipeline import run_training_pipeline, run_pipeline_sweep, run_end_to_end_pipeline

def run_pipelines_from_file(pipelines_file: str, 
                               config_path: str = "config/experiments.yaml",
                               output_dir: str = "results/") -> List[Dict[str, Any]]:
    """Run pipelines listed in a file."""
    try:
        with open(pipelines_file, 'r') as f:
            pipeline_configs = [line.strip() for line in f if line.strip()]
        
        if not pipeline_configs:
            raise ValueError("No pipelines found in file")
        
        return run_pipeline_sweep(pipeline_configs, config_path, output_dir)
        
    except Exception as e:
        return [{
            'status': 'failed',
            'error': f"Failed to read pipelines file: {e}",
            'traceback': traceback.format_exc()
        }]

def get_available_pipelines(config_path: str = "config/experiments.yaml") -> List[str]:
    """Get list of available pipeline names (experiments)."""
    try:
        config = load_config(config_path)
        # Assuming config has an 'experiments' key or similar, but previously it was top level or under 'experiments'
        # Let's check how load_config presents it. Usually it resolves it.
        # For listing, we can just look at the yaml structure.
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            return list(data.get('experiments', {}).keys())
    except Exception:
        return []

def main():
    """Command line interface for training pipeline."""
    parser = argparse.ArgumentParser(description='Run Nano-U training pipelines')
    parser.add_argument('--config', default='config/experiments.yaml', help='Configuration file path')
    parser.add_argument('--pipeline', help='Single pipeline name to run')
    parser.add_argument('--pipelines-file', help='File with pipeline names (one per line)')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--list', action='store_true', help='List available pipelines')
    parser.add_argument('--full', action='store_true', help='Run full end-to-end pipeline (Teacher -> student -> Distill -> Benchmark)')
    
    args = parser.parse_args()
    
    if args.list:
        pipelines = get_available_pipelines(args.config)
        if pipelines:
            print("📋 Available pipelines:")
            for pipe in pipelines:
                print(f"  - {pipe}")
        else:
            print("❌ No pipelines found in configuration or error reading file")
        return
    
    if args.pipeline:
        print(f"🚀 Running training pipeline: {args.pipeline}")
        result = run_training_pipeline(args.pipeline, args.config, args.output)
        
        if result['status'] == 'success':
            print(f"\n✅ Pipeline completed successfully!")
            print(f"Results saved to: {result['results_path']}")
            print(f"Pipeline directory: {result['pipeline_dir']}")
        else:
            print(f"\n❌ Pipeline failed!")
            print(f"Error: {result['error']}")
        return
    
    if args.pipelines_file:
        print(f"🚀 Running pipelines from file: {args.pipelines_file}")
        results = run_pipelines_from_file(args.pipelines_file, args.config, args.output)
        
        return
    
    if args.full:
        result = run_end_to_end_pipeline(args.config, args.output)
        if result['status'] == 'success':
            print("\n🏁 Integration run finished successfully.")
        else:
            print(f"\n❌ Integration run failed: {result.get('error', 'Unknown Error')}")
        return
    
    print("⚠️  Please specify either --pipeline, --pipelines-file, or --full")
    parser.print_help()

if __name__ == "__main__":
    main()
