"""Standard training from config.yaml without KD, with quantize, benchmark.

Usage:
    python scripts/train_standard.py bu_net
    python scripts/train_standard.py nano_u
"""

import os
import sys
import argparse
from pathlib import Path

import shutil

# Allow running directly from project root
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_training_pipeline, quantize_model_pipeline, benchmark_model_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Standard training without KD → quantize → benchmark"
    )
    parser.add_argument("model", choices=["bu_net", "nano_u"], help="Model to train (bu_net or nano_u)")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = args.config
    
    from src.utils.config import load_config
    config = load_config(config_path)
    output_dir = config.get("data", {}).get("paths", {}).get("results_dir", "results/")

    # ── 1. Train ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"STANDARD TRAINING: {args.model.upper()} (No KD)")
    print(f"{'='*55}")

    # Explicitly load config to bypass KD for the standard training run
    import yaml
    
    # We must explicitly turn off distillation for standard training
    if "models" in config and args.model in config["models"]:
        config["models"][args.model]["use_distillation"] = False
    
    # Save a temporary config override
    temp_config_path = f"config/temp_{args.model}_standard.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    try:
        result = run_training_pipeline(args.model, temp_config_path, output_dir)
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    if result["status"] != "success":
        print(f"\nTraining failed: {result.get('error')}")
        sys.exit(1)

    print(f"\nTraining complete  →  {result['model_path']}")

    # ── 2. Post-processing ────────────────────────────────────────────────────
    print(f"\n─── Quantize + Benchmark ───")
    src_model = Path(result["model_path"])
    models_dir_path = str(src_model.parent)
    
    if not src_model.exists():
        print(f"Completed model not found at {src_model}")
        sys.exit(1)

    # ── 3. Quantize + Benchmark ───────────────────────────────────────────────
    if result.get("model_name", "") != "bu_net":
        print(f"\n{'─'*55}")
        print("POST-TRAINING: Quantize + Benchmark")
        print(f"{'─'*55}")

        quant_res = quantize_model_pipeline(str(src_model), models_dir=models_dir_path)
        
        bench_res = {}
        if quant_res.get("status") == "success" and quant_res.get("tflite_path"):
            bench_res = benchmark_model_pipeline(quant_res["tflite_path"])
        else:
            print("Skipping benchmark as quantization failed.")

        # ── 4. Summary ────────────────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print("DONE")
        print(f"   Model:       {result['model_path']}")
        print(f"   TFLite:      {quant_res.get('tflite_path', 'N/A')}")
        print(f"   Size:        {quant_res.get('size_kb', 'N/A')} KB")
        if bench_res and bench_res.get("inference"):
            inf = bench_res.get("inference", {})
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
