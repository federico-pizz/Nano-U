"""Single entry point to run experiments from config (see README and REFACTURING_DOCUMENTATION.md)."""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
import yaml
from typing import Dict, List, Any

# Add project root to path for imports
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.train import train_model


def run_experiment(config_name: str, config_path: str = "config/experiments.yaml",
                  output_dir: str = "results/") -> Dict[str, Any]:
    """Run single experiment with comprehensive logging.
    
    Args:
        config_name: Name of experiment configuration to run
        config_path: Path to configuration file
        output_dir: Base output directory
    
    Returns:
        Dictionary with experiment results and status
    """
    try:
        # Load full configuration
        full_config = load_config(config_path)
        if config_name in full_config:
            experiment_config = full_config[config_name]
        elif "experiments" in full_config and config_name in full_config["experiments"]:
            experiment_config = full_config["experiments"][config_name]
        else:
            raise ValueError(f"Experiment '{config_name}' not found in configuration")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path(output_dir) / f"{config_name}_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment config for reproducibility
        save_config = {config_name: dict(experiment_config)}
        if "data" in full_config and "input_shape" in full_config["data"]:
            save_config[config_name]["input_shape"] = full_config["data"]["input_shape"]
        with open(experiment_dir / "config.yaml", "w") as f:
            yaml.dump(save_config, f)
        
        # Run training with main config and output under experiment_dir
        result = train_model(config_path=config_path, experiment_name=config_name, output_dir=str(experiment_dir))
        
        results_path = experiment_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)
        
        return {
            "status": "success",
            "config_name": config_name,
            "experiment_dir": result.get("experiment_dir", str(experiment_dir)),
            "results_path": str(results_path),
            "timestamp": timestamp,
            **result,
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'config_name': config_name,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }


def run_experiment_sweep(experiment_configs: List[str], 
                        config_path: str = "config/experiments.yaml",
                        output_dir: str = "results/sweeps/") -> List[Dict[str, Any]]:
    """Run multiple experiments with parallel processing.
    
    Args:
        experiment_configs: List of experiment names to run
        config_path: Path to configuration file
        output_dir: Base output directory for sweep
    
    Returns:
        List of experiment results
    """
    results = []
    
    for config_name in experiment_configs:
        print(f"\n🚀 Running experiment: {config_name}")
        result = run_experiment(config_name, config_path, output_dir)
        results.append(result)
        
        # Early stopping on repeated failures
        failed_count = sum(1 for r in results if r['status'] == 'failed')
        if failed_count > len(results) * 0.5:  # Stop if >50% fail
            print(f"\n⚠️  Stopping sweep due to high failure rate: {failed_count}/{len(results)}")
            break
        
        # Small delay between experiments to avoid resource contention
        if config_name != experiment_configs[-1]:
            print("⏳  Waiting 2 seconds before next experiment...")
            import time
            time.sleep(2)
    
    return results


def run_experiments_from_file(experiments_file: str, 
                              config_path: str = "config/experiments.yaml",
                              output_dir: str = "results/") -> List[Dict[str, Any]]:
    """Run experiments listed in a file.
    
    Args:
        experiments_file: Path to file containing experiment names (one per line)
        config_path: Path to configuration file
        output_dir: Base output directory
    
    Returns:
        List of experiment results
    """
    try:
        with open(experiments_file, 'r') as f:
            experiment_configs = [line.strip() for line in f if line.strip()]
        
        if not experiment_configs:
            raise ValueError("No experiments found in file")
        
        return run_experiment_sweep(experiment_configs, config_path, output_dir)
        
    except Exception as e:
        return [{
            'status': 'failed',
            'error': f"Failed to read experiments file: {e}",
            'traceback': traceback.format_exc()
        }]


def get_available_experiments(config_path: str = "config/experiments.yaml") -> List[str]:
    """Get list of available experiment names.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        List of experiment names
    """
    try:
        config = load_config(config_path)
        return list(config.keys()) if isinstance(config, dict) else []
    except Exception:
        return []


def main():
    """Command line interface for experiment runner."""
    parser = argparse.ArgumentParser(description='Run Nano-U experiments')
    parser.add_argument('--config', default='config/experiments.yaml', help='Configuration file path')
    parser.add_argument('--experiment', help='Single experiment name to run')
    parser.add_argument('--experiments-file', help='File with experiment names (one per line)')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--list', action='store_true', help='List available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        experiments = get_available_experiments(args.config)
        if experiments:
            print("📋 Available experiments:")
            for exp in experiments:
                print(f"  - {exp}")
        else:
            print("❌ No experiments found in configuration")
        return
    
    if args.experiment:
        print(f"🚀 Running single experiment: {args.experiment}")
        result = run_experiment(args.experiment, args.config, args.output)
        
        if result['status'] == 'success':
            print(f"\n✅ Experiment completed successfully!")
            print(f"Results saved to: {result['results_path']}")
            print(f"Experiment directory: {result['experiment_dir']}")
        else:
            print(f"\n❌ Experiment failed!")
            print(f"Error: {result['error']}")
        return
    
    if args.experiments_file:
        print(f"🚀 Running experiments from file: {args.experiments_file}")
        results = run_experiments_from_file(args.experiments_file, args.config, args.output)
        
        # Print summary
        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = sum(1 for r in results if r['status'] == 'failed')
        
        print(f"\n📊 Summary:")
        print(f"  Success: {success_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {len(results)}")
        
        if failed_count > 0:
            print(f"\n⚠️  Failed experiments:")
            for result in results:
                if result['status'] == 'failed':
                    print(f"  - {result.get('config_name', 'Unknown')}: {result.get('error', 'Error')}")
        return
    
    print("⚠️  Please specify either --experiment or --experiments-file")
    parser.print_help()


if __name__ == "__main__":
    main()
