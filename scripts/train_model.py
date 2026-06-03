"""Train a single model from config without knowledge distillation.

Useful for training BU-Net as a standalone teacher, or as an ablation baseline.

Usage:
    python scripts/train_model.py bu_net
    python scripts/train_model.py nano_u --config config/TinyAgri_config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Allow running directly from project root
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_training, export_int8


def main():
    parser = argparse.ArgumentParser(
        description="Standard training without KD: train → quantize → evaluate"
    )
    parser.add_argument("model", choices=["bu_net", "nano_u"], help="Model to train")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = args.config

    from src.utils.config import load_config
    config = load_config(config_path)
    output_dir = config.get("data", {}).get("paths", {}).get("results_dir", "results/")

    print(f"\n{'='*55}")
    print(f"STANDARD TRAINING: {args.model.upper()} (No KD)")
    print(f"{'='*55}")

    # Phase 1: Train
    print("\nPhase 1: Training")

    result = run_training(
        args.model, config_path, output_dir,
        config_overrides={"use_distillation": False},
    )

    if result["status"] != "success":
        print(f"\nTraining failed: {result.get('error')}")
        sys.exit(1)

    print(f"Training complete  →  {result['model_path']}")

    src_model = Path(result["model_path"])
    models_dir_path = str(src_model.parent)

    if not src_model.exists():
        print(f"Model not found at {src_model}")
        sys.exit(1)

    eval_results = {}
    quant_res = {}

    if result.get("model_name", "") != "bu_net":

        # Phase 2: Quantize to INT8 TFLite
        print("\nPhase 2: Quantize to INT8 TFLite")
        quant_res = export_int8(str(src_model), models_dir=models_dir_path)

        if quant_res.get("quant_params_path"):
            print(f"\n  ⚠  Firmware dependency: {quant_res['quant_params_path']}")
            print(f"     firmware/build.rs reads this file at compile time.")
            print(f"     Run 'cargo build' only after this file exists.\n")

        if quant_res.get("status") != "success":
            print("Quantization failed — skipping evaluation.")
        else:
            # Phase 3: Evaluate
            # evaluate_and_plot resolves .tflite before .h5 — evaluates INT8 model
            print("\nPhase 3: Evaluation")
            try:
                from src.evaluate import evaluate_and_plot
                eval_out = Path(result["pipeline_dir"]) / "eval_predictions.png"
                print(f"Evaluating {args.model} (INT8 TFLite)...")
                eval_results = evaluate_and_plot(
                    model_name=args.model,
                    config_path=config_path,
                    out_path=str(eval_out),
                )
            except Exception as e:
                print(f"Evaluation failed: {e}")

    # Summary JSON
    summary = {
        "config_path": config_path,
        "model": {
            "model_path": result["model_path"],
            "pipeline_dir": result["pipeline_dir"],
            "tflite_path": quant_res.get("tflite_path"),
            "tflite_size_kb": quant_res.get("size_kb"),
            "quant_params_path": quant_res.get("quant_params_path"),
            "eval_int8": eval_results,
        },
    }
    summary_path = Path(result["pipeline_dir"]) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print("DONE")
    print(f"   Model:        {result['model_path']}")
    if quant_res.get("tflite_path"):
        print(f"   TFLite:       {quant_res['tflite_path']}")
        print(f"   Size:         {quant_res.get('size_kb', 'N/A')} KB")
        print(f"   Quant params: {quant_res.get('quant_params_path', 'N/A')}  ← required by firmware/build.rs")
    if eval_results:
        print(f"   mIoU (INT8):  {eval_results.get('miou', float('nan')):.4f}")
        print(f"   F1   (INT8):  {eval_results.get('f1', float('nan')):.4f}")
    print(f"   Summary:      {summary_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
