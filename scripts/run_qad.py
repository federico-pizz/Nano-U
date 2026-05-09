"""Quantization-Aware Distillation (QAD) pipeline: Teacher → Student → INT8 TFLite.

  Phase 1 — Train Teacher (bu_net)
  Phase 2 — Train Student (nano_u) via knowledge distillation + QAT from epoch 1
  Phase 3 — Export Student to INT8 TFLite + produce quant_params.json for firmware
  Phase 4 — Evaluate Teacher (float) and Student (INT8 TFLite) on test set
             Saves pipeline_summary.json with all metrics

Usage:
    python scripts/run_qad.py
    python scripts/run_qad.py --config config/TinyAgri_config.yaml
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_training, export_int8


def main():
    parser = argparse.ArgumentParser(description="QAD pipeline: Teacher → Student → INT8 TFLite")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = args.config

    from src.utils.config import load_config
    config = load_config(config_path)
    output_dir = config.get("data", {}).get("paths", {}).get("results_dir", "results/")

    print(f"\n{'='*55}")
    print("TEACHER → STUDENT DISTILLATION PIPELINE")
    print(f"{'='*55}")

    # Phase 1: Train Teacher
    print("\nPhase 1: Training Teacher")
    teacher_res = run_training("bu_net", config_path, output_dir)

    if teacher_res["status"] != "success":
        print(f"Teacher training failed: {teacher_res.get('error')}")
        sys.exit(1)

    print(f"Teacher trained  →  {teacher_res['model_path']}")

    # Phase 2: Train Student (QAD)
    print("\nPhase 2: Training Student")
    # Inject the teacher path resolved at runtime so the student always distils
    # from the model trained in Phase 1, regardless of what is in the YAML.
    student_res = run_training(
        "nano_u", config_path, output_dir,
        config_overrides={"teacher_weights": teacher_res["model_path"]},
    )

    if student_res["status"] != "success":
        print(f"Student training failed: {student_res.get('error')}")
        sys.exit(1)

    print(f"Student trained  →  {student_res['model_path']}")

    # Phase 3: Quantize to INT8 TFLite
    print("\nPhase 3: Quantize to INT8 TFLite")
    student_src = Path(student_res["model_path"])
    models_dir_path = str(student_src.parent)

    if not student_src.exists():
        print(f"Student model not found at {student_src}")
        sys.exit(1)

    quant_res = export_int8(str(student_src), models_dir=models_dir_path)

    if quant_res.get("quant_params_path"):
        print(f"\n  ⚠  Firmware dependency: {quant_res['quant_params_path']}")
        print(f"     firmware/build.rs reads this file at compile time.")
        print(f"     Run 'cargo build' only after this file exists.\n")

    if quant_res.get("status") != "success":
        print("Quantization failed — aborting.")
        sys.exit(1)

    # Phase 4: Evaluate on Test Set
    # evaluate_and_plot resolves .tflite before .h5, so the student is
    # evaluated as INT8 TFLite — the same format that runs on the MCU.
    print("\nPhase 4: Evaluation and Visualization")
    eval_results_all = {}
    try:
        from src.evaluate import evaluate_and_plot

        # Evaluate Teacher (float Keras — used only as distillation reference)
        teacher_path = Path(teacher_res["model_path"])
        teacher_eval_out = Path(teacher_res["pipeline_dir"]) / "eval_predictions.png"
        print(f"\nEvaluating Teacher ({teacher_path.stem})...")
        eval_results_all['teacher'] = evaluate_and_plot(
            model_name=teacher_path.stem,
            config_path=config_path,
            out_path=str(teacher_eval_out),
        )

        # Evaluate Student — picks nano_u.tflite (INT8) via candidate resolution
        student_eval_out = Path(student_res["pipeline_dir"]) / "eval_predictions.png"
        print(f"\nEvaluating Student INT8 TFLite ({student_src.stem})...")
        eval_results_all['student_int8'] = evaluate_and_plot(
            model_name=student_src.stem,
            config_path=config_path,
            out_path=str(student_eval_out),
        )

    except Exception as e:
        print(f"Evaluation failed: {e}")

    # Pipeline summary JSON
    summary = {
        "config_path": config_path,
        "teacher": {
            "model_path": teacher_res["model_path"],
            "pipeline_dir": teacher_res["pipeline_dir"],
            "eval": eval_results_all.get("teacher", {}),
        },
        "student": {
            "model_path": student_res["model_path"],
            "pipeline_dir": student_res["pipeline_dir"],
            "tflite_path": quant_res.get("tflite_path"),
            "tflite_size_kb": quant_res.get("size_kb"),
            "quant_params_path": quant_res.get("quant_params_path"),
            "eval_int8": eval_results_all.get("student_int8", {}),
        },
    }
    summary_path = Path(output_dir) / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    teacher_miou = eval_results_all.get("teacher", {}).get("miou", float("nan"))
    student_miou = eval_results_all.get("student_int8", {}).get("miou", float("nan"))

    print(f"\n{'='*55}")
    print("PIPELINE COMPLETE")
    print(f"   Teacher path:     {teacher_res['model_path']}")
    print(f"   Student path:     {student_res['model_path']}")
    print(f"   TFLite:           {quant_res.get('tflite_path', 'N/A')}")
    print(f"   Size:             {quant_res.get('size_kb', 'N/A')} KB")
    print(f"   Quant params:     {quant_res.get('quant_params_path', 'N/A')}  ← required by firmware/build.rs")
    print(f"   Teacher mIoU:     {teacher_miou:.4f}")
    print(f"   Student mIoU:     {student_miou:.4f}  (INT8 TFLite)")
    print(f"   Summary JSON:     {summary_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
