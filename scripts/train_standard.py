"""Standard training: one experiment from experiments.yaml, NAS monitoring, quantize, benchmark.

Usage:
    python scripts/train_standard.py --experiment default
    python scripts/train_standard.py --experiment bu_net_nas --output results/teacher/
    python scripts/train_standard.py --list
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import shutil

# Allow running directly from project root
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_training_pipeline, quantize_and_benchmark


def list_experiments(config_path: str):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return list(data.get("experiments", {}).keys())


def main():
    parser = argparse.ArgumentParser(
        description="Standard training with NAS monitoring → quantize → benchmark"
    )
    parser.add_argument("--experiment", "-e", help="Experiment name from experiments.yaml")
    parser.add_argument("--config", default="config/experiments.yaml", help="Config file path")
    parser.add_argument("--output", default="results/", help="Output base directory")
    parser.add_argument("--models-dir", default="models/", help="Where to copy final .keras + .tflite")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit")
    args = parser.parse_args()

    if args.list:
        experiments = list_experiments(args.config)
        print("Available experiments:")
        for e in experiments:
            print(f"  - {e}")
        return

    if not args.experiment:
        parser.error("--experiment is required (or use --list to see options)")

    # ── 1. Train ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"STANDARD TRAINING  —  experiment: {args.experiment}")
    print(f"{'='*55}")

    result = run_training_pipeline(args.experiment, args.config, args.output)

    if result["status"] != "success":
        print(f"\nTraining failed: {result.get('error')}")
        sys.exit(1)

    print(f"\nTraining complete  →  {result['model_path']}")

    # ── 2. Copy to models/ ────────────────────────────────────────────────────
    print(f"\n─── Copy Models + Quantize + Benchmark ───")
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    src_model = Path(result["model_path"])
    dst_model = models_dir / src_model.name
    
    if src_model.exists():
        shutil.copy(src_model, dst_model)
        print(f"Model weights copied  →  {dst_model}")
    else:
        print(f"Completed model not found at {src_model}")
        sys.exit(1)

    # ── 3. Quantize + Benchmark ───────────────────────────────────────────────
    if result.get("model_name", "") != "bu_net":
        print(f"\n{'─'*55}")
        print("POST-TRAINING: Quantize + Benchmark")
        print(f"{'─'*55}")

        qb = quantize_and_benchmark(str(dst_model), models_dir=args.models_dir)

        # ── 4. Summary ────────────────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print("DONE")
        print(f"   Model:       {result['model_path']}")
        print(f"   TFLite:      {qb['quantization'].get('tflite_path', 'N/A')}")
        print(f"   Size:        {qb['quantization'].get('size_kb', 'N/A')} KB")
        if qb.get("benchmark") and qb["benchmark"].get("inference"):
            inf = qb["benchmark"].get("inference", {})
            print(f"   Latency:     {inf.get('avg_latency_ms', '?'):.2f} ms")
            print(f"   Throughput:  {inf.get('throughput_fps', '?'):.1f} FPS")
        print(f"{'='*55}\n")
    else:
        print(f"\n{'='*55}")
        print("DONE (BU-Net Trained; Quantization Skipped)")
        print(f"   Model:       {result['model_path']}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
