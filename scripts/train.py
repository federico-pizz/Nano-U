"""Train Nano-U, from configuration to an evaluated INT8 TFLite model.

Two paths, one program:

  default          Quantization-Aware Distillation (QAD).
                   Phase 1 - train the teacher (bu_net).
                   Phase 2 - train the student (nanou) by knowledge distillation
                             with QAT applied from epoch 1.
                   Phase 3 - export the student to INT8 TFLite + quant_params.json.
                   Phase 4 - evaluate teacher (float), student (FP32) and
                             student (INT8) on the test split.

  --no-distill     Train one model on its own - the teacher as a standalone
                   baseline, or a student without KD as an ablation. Quantization
                   and evaluation still run for anything that is not `bu_net`.

Usage:
    python scripts/train.py
    python scripts/train.py --preset tinyagri
    python scripts/train.py --no-distill --model bu_net
    python scripts/train.py --no-distill --model nanou --preset tinyagri
"""

import argparse
import json
import sys
from pathlib import Path

from src.pipeline import run_training, export_int8
from src.utils.config import load_config
from nanou.config import preset_path


def _warn_firmware_dependency(quant_res):
    if quant_res.get("quant_params_path"):
        print(f"\n  Firmware dependency: {quant_res['quant_params_path']}")
        print("     firmware/build.rs reads this file at compile time.")
        print("     Run 'cargo build' only after this file exists.\n")


def run_qad(config_path, output_dir):
    """Teacher -> student(KD+QAT) -> INT8 -> evaluate. Writes pipeline_summary.json."""
    print(f"\n{'='*55}")
    print("TEACHER -> STUDENT DISTILLATION PIPELINE")
    print(f"{'='*55}")

    # Phase 1: Train Teacher
    print("\nPhase 1: Training Teacher")
    teacher_res = run_training("bu_net", config_path, output_dir)

    if teacher_res["status"] != "success":
        print(f"Teacher training failed: {teacher_res.get('error')}")
        sys.exit(1)

    print(f"Teacher trained  ->  {teacher_res['model_path']}")

    # Phase 2: Train Student (QAD)
    print("\nPhase 2: Training Student")
    # Inject the teacher path resolved at runtime so the student always distils
    # from the model trained in Phase 1, regardless of what is in the YAML.
    student_res = run_training(
        "nanou", config_path, output_dir,
        config_overrides={"teacher_weights": teacher_res["model_path"]},
    )

    if student_res["status"] != "success":
        print(f"Student training failed: {student_res.get('error')}")
        sys.exit(1)

    print(f"Student trained  ->  {student_res['model_path']}")

    # Phase 3: Quantize to INT8 TFLite
    print("\nPhase 3: Quantize to INT8 TFLite")
    student_src = Path(student_res["model_path"])

    if not student_src.exists():
        print(f"Student model not found at {student_src}")
        sys.exit(1)

    quant_res = export_int8(str(student_src), models_dir=str(student_src.parent),
                            config_path=config_path)
    _warn_firmware_dependency(quant_res)

    if quant_res.get("status") != "success":
        print("Quantization failed - aborting.")
        sys.exit(1)

    # Phase 4: Evaluate on Test Set
    # evaluate_and_plot resolves .tflite before .h5, so the student is
    # evaluated as INT8 TFLite - the same format that runs on the MCU.
    print("\nPhase 4: Evaluation and Visualization")
    eval_results_all = {}
    try:
        from src.evaluate import evaluate_and_plot

        # Evaluate Teacher (float Keras - used only as distillation reference)
        teacher_path = Path(teacher_res["model_path"])
        teacher_eval_out = Path(teacher_res["pipeline_dir"]) / "eval_predictions.png"
        print(f"\nEvaluating Teacher ({teacher_path.stem})...")
        eval_results_all['teacher'] = evaluate_and_plot(
            model_name=teacher_path.stem,
            config_path=config_path,
            out_path=str(teacher_eval_out),
        )

        # Evaluate Student in BOTH formats so the quantization gap is real:
        #   - FP32 Keras (.h5)  -> prefer="float"
        #   - INT8 TFLite       -> prefer="tflite" (the on-device format)
        # Without forcing prefer="float" the resolver picks .tflite for both,
        # making the "float" row silently identical to INT8.
        student_dir = Path(student_res["pipeline_dir"])
        print(f"\nEvaluating Student FP32 Keras ({student_src.stem})...")
        eval_results_all['student_fp32'] = evaluate_and_plot(
            model_name=student_src.stem,
            config_path=config_path,
            out_path=str(student_dir / "eval_predictions_fp32.png"),
            prefer="float",
        )

        print(f"\nEvaluating Student INT8 TFLite ({student_src.stem})...")
        eval_results_all['student_int8'] = evaluate_and_plot(
            model_name=student_src.stem,
            config_path=config_path,
            out_path=str(student_dir / "eval_predictions_int8.png"),
            prefer="tflite",
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
            "eval_fp32": eval_results_all.get("student_fp32", {}),
            "eval_int8": eval_results_all.get("student_int8", {}),
        },
    }
    summary_path = Path(output_dir) / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    teacher_miou = eval_results_all.get("teacher", {}).get("miou", float("nan"))
    student_fp32_miou = eval_results_all.get("student_fp32", {}).get("miou", float("nan"))
    student_miou = eval_results_all.get("student_int8", {}).get("miou", float("nan"))

    print(f"\n{'='*55}")
    print("PIPELINE COMPLETE")
    print(f"   Teacher path:     {teacher_res['model_path']}")
    print(f"   Student path:     {student_res['model_path']}")
    print(f"   TFLite:           {quant_res.get('tflite_path', 'N/A')}")
    print(f"   Size:             {quant_res.get('size_kb', 'N/A')} KB")
    print(f"   Quant params:     {quant_res.get('quant_params_path', 'N/A')}  <- required by firmware/build.rs")
    print(f"   Teacher mIoU:     {teacher_miou:.4f}")
    print(f"   Student mIoU:     {student_fp32_miou:.4f}  (FP32 Keras)")
    print(f"   Student mIoU:     {student_miou:.4f}  (INT8 TFLite)")
    print(f"   Summary JSON:     {summary_path}")
    print(f"{'='*55}\n")


def run_single(model, config_path, output_dir):
    """Train one model without KD, then quantize and evaluate it. Writes summary.json."""
    print(f"\n{'='*55}")
    print(f"STANDARD TRAINING: {model.upper()} (No KD)")
    print(f"{'='*55}")

    # Phase 1: Train
    print("\nPhase 1: Training")

    result = run_training(
        model, config_path, output_dir,
        config_overrides={"use_distillation": False},
    )

    if result["status"] != "success":
        print(f"\nTraining failed: {result.get('error')}")
        sys.exit(1)

    print(f"Training complete  ->  {result['model_path']}")

    src_model = Path(result["model_path"])

    if not src_model.exists():
        print(f"Model not found at {src_model}")
        sys.exit(1)

    eval_results = {}
    quant_res = {}

    # bu_net is the teacher: it never runs on the MCU, so there is nothing to
    # quantize and no INT8 number to report for it.
    if result.get("model_name", "") != "bu_net":

        # Phase 2: Quantize to INT8 TFLite
        print("\nPhase 2: Quantize to INT8 TFLite")
        quant_res = export_int8(str(src_model), models_dir=str(src_model.parent),
                                config_path=config_path)
        _warn_firmware_dependency(quant_res)

        if quant_res.get("status") != "success":
            print("Quantization failed - skipping evaluation.")
        else:
            # Phase 3: Evaluate
            # evaluate_and_plot resolves .tflite before .h5 - evaluates INT8 model
            print("\nPhase 3: Evaluation")
            try:
                from src.evaluate import evaluate_and_plot
                eval_out = Path(result["pipeline_dir"]) / "eval_predictions.png"
                print(f"Evaluating {model} (INT8 TFLite)...")
                eval_results = evaluate_and_plot(
                    model_name=model,
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
        print(f"   Quant params: {quant_res.get('quant_params_path', 'N/A')}  <- required by firmware/build.rs")
    if eval_results:
        print(f"   mIoU (INT8):  {eval_results.get('miou', float('nan')):.4f}")
        print(f"   F1   (INT8):  {eval_results.get('f1', float('nan')):.4f}")
    print(f"   Summary:      {summary_path}")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Nano-U: QAD by default, or a single model with --no-distill"
    )
    parser.add_argument("--preset", default="tinyagri",
                        help="config preset from config/presets/ (default: tinyagri)")
    parser.add_argument("--config", default=None,
                        help="explicit config file, used instead of --preset")
    parser.add_argument("--no-distill", action="store_true",
                        help="Train one model without knowledge distillation")
    # Free-form rather than a choice list: this is the config section name that
    # run_training passes through as the experiment name, so a config may define
    # sections beyond bu_net/nanou.
    parser.add_argument("--model", default=None,
                        help="With --no-distill: the model/experiment section to train")
    args = parser.parse_args()

    if args.model and not args.no_distill:
        parser.error("--model applies only to --no-distill; the QAD path trains both models")
    if args.no_distill and not args.model:
        parser.error("--no-distill requires --model (e.g. --model bu_net)")

    config_path = args.config or str(preset_path(args.preset))
    config = load_config(config_path)
    output_dir = config.get("data", {}).get("paths", {}).get("results_dir", "results/")

    if args.no_distill:
        run_single(args.model, config_path, output_dir)
    else:
        run_qad(config_path, output_dir)


if __name__ == "__main__":
    main()
