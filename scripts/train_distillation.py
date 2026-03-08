"""Distillation pipeline: Teacher (BU-Net) → Student (Nano-U) with NAS monitoring, quantize, benchmark.

This pipeline performs the complete knowledge distillation workflow:
  Phase 1 — Train Teacher (bu_net) with high capacity to establish a strong baseline
  Phase 2 — Train Student (nano_u) using Distillation loss to match the teacher
  Phase 3 — Quantize Student → INT8 TFLite for embedded deployment
  Phase 4 — Benchmark the quantized student
  Phase 5 — Test and visually evaluate the student

The output of this script is the fully trained, distilled, and quantized `nano_u` model,
because distillation is specifically designed to transfer the knowledge from the heavy 
`bu_net` (teacher) into the ultra-efficient `nano_u` (student) architecture.

Usage:
    python scripts/train_distillation.py
"""

import os
import sys
import shutil
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_training_pipeline, quantize_model_pipeline, benchmark_model_pipeline


def main():
    config_path = "config/config.yaml"
    output_dir = "results/"
    models_dir_path = "models/"

    models_dir = Path(models_dir_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print("TEACHER → STUDENT DISTILLATION PIPELINE")
    print(f"{'='*55}")

    # ── Phase 1: Train Teacher ─────────────────────────────────────────────
    print(f"\n─── Phase 1: Training Teacher (from config) ───")
    teacher_res = run_training_pipeline("bu_net", config_path, output_dir)

    if teacher_res["status"] != "success":
        print(f"Teacher training failed: {teacher_res.get('error')}")
        sys.exit(1)

    print(f"Teacher trained  →  {teacher_res['model_path']}")

    # ── Phase 2: Train Student ─────────────────────────────────────────────
    print(f"\n─── Phase 2: Training Student (from config) ───")
    student_res = run_training_pipeline("nano_u", config_path, output_dir)

    if student_res["status"] != "success":
        print(f"Student training failed: {student_res.get('error')}")
        sys.exit(1)

    print(f"Student trained  →  {student_res['model_path']}")

    # ── Phase 3 & 4: Quantize + Benchmark ───
    print(f"\n─── Phase 3 & 4: Quantize + Benchmark ───")
    student_src = Path("models/nano_u.h5")
    if not student_src.exists():
        print(f"Student model not found at {student_src}")
    else:
        print(f"Student model found at {student_src}")

    quant_res = quantize_model_pipeline(str(student_src), models_dir=models_dir_path)
    
    bench_res = {}
    if quant_res.get("status") == "success" and quant_res.get("tflite_path"):
        bench_res = benchmark_model_pipeline(quant_res["tflite_path"])
    else:
        print("Skipping benchmark as quantization failed.")

    # ── Phase 5: Visual Evaluation on Test Set ─────────────
    print(f"\n─── Phase 5: Evaluation & Visualization ───")
    eval_results_all = {}
    try:
        from src.evaluate import evaluate_and_plot
        
        # Evaluate Teacher
        teacher_path = Path(teacher_res["model_path"])
        teacher_stem = teacher_path.stem
        teacher_eval_out = Path(teacher_res["pipeline_dir"]) / "eval_predictions.png"
        print(f"\nEvaluating Teacher ({teacher_stem})...")
        eval_results_teacher = evaluate_and_plot(
            model_name=teacher_stem,
            config_path=config_path,
            out_path=str(teacher_eval_out)
        )
        print(f"Teacher metrics:")
        for k, v in eval_results_teacher.items():
            print(f"  {k}: {v:.4f}")
        eval_results_all['teacher'] = eval_results_teacher
            
        # Evaluate Student
        student_stem = student_src.stem
        student_eval_out = Path(student_res["pipeline_dir"]) / "eval_predictions.png"
        print(f"\nEvaluating Student ({student_stem})...")
        eval_results_student = evaluate_and_plot(
            model_name=student_stem,
            config_path=config_path,
            out_path=str(student_eval_out)
        )
        
        print(f"Student metrics:")
        for k, v in eval_results_student.items():
            print(f"  {k}: {v:.4f}")
        eval_results_all['student'] = eval_results_student
            
    except Exception as e:
        print(f"Evaluation failed: {e}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("PIPELINE COMPLETE - NANO_U STUDENT TRAINED SUCCESSFULLY")
    print("The final model produced by this pipeline is the nano_u (student).")
    print("Knowledge has been distilled from the bu_net (teacher).")
    print(f"   Teacher Path:     {teacher_res['model_path']}")
    print(f"   Student Path:     {student_res['model_path']}")
    print(f"   TFLite:      {quant_res.get('tflite_path', 'N/A')}")
    print(f"   Size:        {quant_res.get('size_kb', 'N/A')} KB")
    
    if bench_res and bench_res.get("inference"):
        inf = bench_res.get("inference", {})
        print(f"   Latency:     {inf.get('avg_latency_ms', '?'):.2f} ms")
        print(f"   Throughput:  {inf.get('throughput_fps', '?'):.1f} FPS")
        
    if eval_results_all and 'student' in eval_results_all:
        print(f"   Test IoU:    {eval_results_all['student'].get('iou', 0):.4f}")
        print(f"   Teacher IoU: {eval_results_all.get('teacher', {}).get('iou', 0):.4f}")
        print(f"   Plots saved: {teacher_res['pipeline_dir']}/eval_predictions.png, {student_res['pipeline_dir']}/eval_predictions.png")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
