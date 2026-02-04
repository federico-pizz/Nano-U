#!/usr/bin/env python3
"""
Automated experiment runner for Phase 4 optimization.

Runs hyperparameter sweeps with tracking and analysis.

Usage:
    python scripts/run_experiments.py --phase 4.1 --output results/phase_4_1/
    python scripts/run_experiments.py --phase 4.2 --resume --checkpoint checkpoint.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Project root
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"


class ExperimentRunner:
    """Run and track hyperparameter experiments."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.checkpoint_file = self.output_dir / "checkpoint.json"
    
    def load_checkpoint(self) -> Dict:
        """Load previous checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {}
    
    def save_checkpoint(self, completed_experiments: List[Dict]):
        """Save checkpoint of completed experiments."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "completed": len(completed_experiments),
            "results": completed_experiments
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def run_experiment(self, exp_config: Dict) -> Dict:
        """
        Run single training experiment.
        
        Args:
            exp_config: Dictionary with training parameters
        
        Returns:
            Dictionary with results
        """
        # DEBUG: Log config before mutation
        print(f"[DEBUG] run_experiment() received config: {exp_config}")
        
        # Create a copy to avoid mutating the input config dictionary
        config_copy = exp_config.copy()
        exp_id = config_copy.pop("exp_id", f"exp_{len(self.results)}")
        
        # DEBUG: Log config after pop mutation
        print(f"[DEBUG] After pop('exp_id'), config_copy is now: {config_copy}")
        
        exp_name = f"{exp_id}_" + "_".join(
            f"{k}_{str(v).replace('.', '')}" for k, v in config_copy.items()
        )
        
        # Build command
        cmd = [
            str(Path(sys.executable).resolve()),
            str(SRC / "train.py"),
            "--model", "nano_u",
            "--distill",
            "--teacher-weights", str(ROOT / "models" / "bu_net.keras"),
            "--enable-nas",
            "--nas-csv-path", str(self.output_dir / f"{exp_name}_nas.csv"),
        ]
        
        # Add hyperparameters
        param_map = {
            "lr": "--lr",
            "batch_size": "--batch-size",
            "epochs": "--epochs",
            "temperature": "--temperature",
            "alpha": "--alpha",
            "weight_decay": "--weight-decay",
        }
        
        for key, flag in param_map.items():
            if key in exp_config:
                cmd.extend([flag, str(exp_config[key])])
        
        print(f"\n{'='*70}")
        print(f"Running: {exp_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*70}")
        
        # Run experiment
        result = {
            "exp_id": exp_id,
            "exp_name": exp_name,
            "config": exp_config,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        try:
            ret = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=3600*6)
            
            if ret.returncode == 0:
                result["status"] = "completed"
                
                # Extract metrics from NAS CSV if available
                nas_csv = self.output_dir / f"{exp_name}_nas.csv"
                if nas_csv.exists():
                    try:
                        df = pd.read_csv(nas_csv)
                        if len(df) == 0:
                            print(f"⚠️  Warning: NAS CSV is empty: {nas_csv}")
                        elif "redundancy_score" not in df.columns:
                            print(f"⚠️  Warning: 'redundancy_score' column not found in {nas_csv}")
                            print(f"   Available columns: {list(df.columns)}")
                        else:
                            try:
                                result["final_redundancy"] = float(df["redundancy_score"].iloc[-1])
                                result["mean_redundancy"] = float(df["redundancy_score"].mean())
                                result["metrics_file"] = str(nas_csv)
                            except (ValueError, TypeError) as e:
                                print(f"⚠️  Warning: Failed to convert redundancy_score to float: {e}")
                    except pd.errors.EmptyDataError:
                        print(f"⚠️  Warning: NAS CSV is empty or malformed: {nas_csv}")
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to read NAS metrics from {nas_csv}: {e}")
                
                print(f"✅ Experiment completed: {exp_name}")
            else:
                result["status"] = "failed"
                result["error"] = ret.stderr[-2000:]  # Last 2000 chars of error for more context
                print(f"❌ Experiment failed: {exp_name}")
                if ret.stderr:
                    print(f"Error (last 500 chars): {ret.stderr[-500:]}")
        
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            print(f"⏱️  Experiment timeout: {exp_name}")
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"⚠️  Experiment error: {exp_name} - {e}")
        
        result["end_time"] = datetime.now().isoformat()
        self.results.append(result)
        self.save_checkpoint(self.results)
        
        return result
    
    def run_phase_4_1(self):
        """Run Phase 4.1: Learning Rate & Batch Size Sweep."""
        print("\n" + "="*70)
        print("PHASE 4.1: Learning Rate & Batch Size Sweep")
        print("="*70)
        
        learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        batch_sizes = [4, 8, 16, 32]
        epochs = 50
        
        configs = [
            {"lr": lr, "batch_size": bs, "epochs": epochs, "exp_id": f"lr{lr:.0e}_bs{bs}"}
            for lr in learning_rates
            for bs in batch_sizes
        ]
        
        completed = 0
        for config in configs:
            self.run_experiment(config)
            completed += 1
            print(f"\nProgress: {completed}/{len(configs)} experiments")
        
        # Analyze results
        self.analyze_phase_4_1()
    
    def run_phase_4_2(self):
        """Run Phase 4.2: Distillation Hyperparameters."""
        print("\n" + "="*70)
        print("PHASE 4.2: Distillation Hyperparameters")
        print("="*70)
        
        # Default: Use best from Phase 4.1 (or baseline)
        best_lr = 1e-4
        best_bs = 8
        
        temperatures = [2.0, 3.0, 4.0, 5.0, 6.0]
        alphas = [0.2, 0.3, 0.4, 0.5]
        epochs = 100
        
        configs = [
            {
                "lr": best_lr,
                "batch_size": best_bs,
                "temperature": temp,
                "alpha": alpha,
                "epochs": epochs,
                "exp_id": f"temp{temp}_alpha{alpha}"
            }
            for temp in temperatures
            for alpha in alphas
        ]
        
        completed = 0
        for config in configs:
            self.run_experiment(config)
            completed += 1
            print(f"\nProgress: {completed}/{len(configs)} experiments")
        
        self.analyze_phase_4_2()
    
    def run_phase_4_3(self):
        """Run Phase 4.3: Regularization & Dropout."""
        print("\n" + "="*70)
        print("PHASE 4.3: Regularization & Dropout")
        print("="*70)
        
        # Best hyperparameters from Phase 4.2
        best_config = {"lr": 1e-4, "batch_size": 8, "temperature": 4.0, "alpha": 0.3}
        
        weight_decays = [0, 1e-4, 1e-3, 1e-2]
        # Note: Dropout requires model architecture changes - skip for now
        epochs = 100
        
        configs = [
            {
                "lr": best_config["lr"],
                "batch_size": best_config["batch_size"],
                "temperature": best_config["temperature"],
                "alpha": best_config["alpha"],
                "weight_decay": wd,
                "epochs": epochs,
                "exp_id": f"wd{wd:.0e}"
            }
            for wd in weight_decays
        ]
        
        completed = 0
        for config in configs:
            self.run_experiment(config)
            completed += 1
            print(f"\nProgress: {completed}/{len(configs)} experiments")
        
        self.analyze_phase_4_3()
    
    def analyze_phase_4_1(self):
        """Analyze Phase 4.1 results."""
        df = pd.DataFrame([r for r in self.results if r["status"] == "completed"])
        
        if df.empty:
            print("No completed experiments to analyze")
            return
        
        print("\n" + "="*70)
        print("PHASE 4.1 ANALYSIS: Learning Rate & Batch Size")
        print("="*70)
        
        # Extract configs
        df["lr"] = df["config"].apply(lambda x: x.get("lr", 0))
        df["batch_size"] = df["config"].apply(lambda x: x.get("batch_size", 0))
        
        # Check if final_redundancy column exists
        if "final_redundancy" not in df.columns:
            print("⚠️  Warning: No redundancy metrics available for analysis")
            return
        
        # Summary statistics
        print("\nTop 5 experiments by redundancy score (lower is better):")
        top = df.nsmallest(5, "final_redundancy")[["exp_id", "lr", "batch_size", "final_redundancy"]]
        print(top.to_string(index=False))
        
        # Save analysis
        df.to_csv(self.output_dir / "phase_4_1_summary.csv", index=False)
        print(f"\n✅ Analysis saved to {self.output_dir / 'phase_4_1_summary.csv'}")
    
    def analyze_phase_4_2(self):
        """Analyze Phase 4.2 results."""
        df = pd.DataFrame([r for r in self.results if r["status"] == "completed"])
        
        if df.empty:
            print("No completed experiments to analyze")
            return
        
        print("\n" + "="*70)
        print("PHASE 4.2 ANALYSIS: Distillation Hyperparameters")
        print("="*70)
        
        # Extract configs
        df["temperature"] = df["config"].apply(lambda x: x.get("temperature", 0))
        df["alpha"] = df["config"].apply(lambda x: x.get("alpha", 0))
        
        # Check if final_redundancy column exists
        if "final_redundancy" not in df.columns:
            print("⚠️  Warning: No redundancy metrics available for analysis")
            return
        
        print("\nTop 5 experiments by redundancy score:")
        top = df.nsmallest(5, "final_redundancy")[["exp_id", "temperature", "alpha", "final_redundancy"]]
        print(top.to_string(index=False))
        
        df.to_csv(self.output_dir / "phase_4_2_summary.csv", index=False)
        print(f"\n✅ Analysis saved to {self.output_dir / 'phase_4_2_summary.csv'}")
    
    def analyze_phase_4_3(self):
        """Analyze Phase 4.3 results."""
        df = pd.DataFrame([r for r in self.results if r["status"] == "completed"])
        
        if df.empty:
            print("No completed experiments to analyze")
            return
        
        print("\n" + "="*70)
        print("PHASE 4.3 ANALYSIS: Regularization & Dropout")
        print("="*70)
        
        df["weight_decay"] = df["config"].apply(lambda x: x.get("weight_decay", 0))
        
        # Check if final_redundancy column exists
        if "final_redundancy" not in df.columns:
            print("⚠️  Warning: No redundancy metrics available for analysis")
            return
        
        print("\nExperiments by weight decay:")
        summary = df.groupby("weight_decay").agg({
            "final_redundancy": ["mean", "min", "max"]
        }).round(4)
        print(summary)
        
        df.to_csv(self.output_dir / "phase_4_3_summary.csv", index=False)
        print(f"\n✅ Analysis saved to {self.output_dir / 'phase_4_3_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 experiments")
    parser.add_argument("--phase", type=str, choices=["4", "4.1", "4.2", "4.3"],
                       default="4.1", help="Which phase to run")
    parser.add_argument("--output", type=str, default="results/phase_4/",
                       help="Output directory")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str,
                       help="Checkpoint file path")
    
    args = parser.parse_args()
    
    # DEBUG: Log parsed arguments
    print(f"[DEBUG] Parsed args: phase={args.phase}, output={args.output}, resume={args.resume}, checkpoint={args.checkpoint}")
    
    runner = ExperimentRunner(args.output)
    
    if args.resume and args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        # DEBUG: Log checkpoint loading attempt
        print(f"[DEBUG] Attempting to load checkpoint from {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            runner.results = checkpoint.get("results", [])
            print(f"[DEBUG] Restored {len(runner.results)} completed experiments from checkpoint")
        else:
            print(f"[DEBUG] Checkpoint file not found: {args.checkpoint}")
    
    try:
        # DEBUG: Log which phases will execute
        print(f"[DEBUG] Determining which phases to run...")
        print(f"[DEBUG] Phase check: args.phase={args.phase}")
        print(f"[DEBUG] Will run 4.1: {args.phase in ['4', '4.1']}")
        print(f"[DEBUG] Will run 4.2: {args.phase in ['4', '4.2']}")
        print(f"[DEBUG] Will run 4.3: {args.phase in ['4', '4.3']}")
        
        if args.phase in ["4", "4.1"]:
            runner.run_phase_4_1()
        if args.phase in ["4", "4.2"]:
            runner.run_phase_4_2()
        if args.phase in ["4", "4.3"]:
            runner.run_phase_4_3()
        
        print("\n" + "="*70)
        print("✅ All experiments completed!")
        print(f"Results saved to: {runner.output_dir}")
        print("="*70)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        print(f"Progress saved to: {runner.checkpoint_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
