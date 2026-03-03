"""Distillation pipeline: Teacher (BU-Net) → Student (Nano-U) with NAS monitoring, quantize, benchmark.

This pipeline performs the complete knowledge distillation workflow:
  Phase 1 — Train Teacher (bu_net) with high capacity to establish a strong baseline
  Phase 2 — Copy teacher weights to models/ folder to act as target for student
  Phase 3 — Train Student (nano_u) using Distillation loss to match the teacher
  Phase 4 — Copy student weights and Quantize Student → INT8 TFLite for embedded deployment
  Phase 5 — Benchmark the quantized student
  Phase 6 — Test and visually evaluate the student

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

    # ── Phase 2: Copy teacher weights ──────────────────────────────────────
    print(f"\n─── Phase 2: Copying Teacher Weights ───")
    teacher_src = Path(teacher_res["model_path"])
    teacher_dst = models_dir / teacher_src.name

    if teacher_src.exists():
        shutil.copy(teacher_src, teacher_dst)
        print(f"Teacher weights copied  →  {teacher_dst}")
    else:
        print(f"Teacher model not found at {teacher_src}")
        sys.exit(1)

    # ── Phase 3: Train Student ─────────────────────────────────────────────
    print(f"\n─── Phase 3: Training Student (from config) ───")
    student_res = run_training_pipeline("nano_u", config_path, output_dir)

    if student_res["status"] != "success":
        print(f"Student training failed: {student_res.get('error')}")
        sys.exit(1)

    print(f"Student trained  →  {student_res['model_path']}")

    # ── Phase 4: Copy student weights & Quantize ───
    print(f"\n─── Phase 4 & 5: Copy Weights + Quantize + Benchmark ───")
    student_src = Path(student_res["model_path"])
    student_dst = models_dir / student_src.name
    if student_src.exists():
        shutil.copy(student_src, student_dst)
        print(f"Student weights copied  →  {student_dst}")
    else:
        print(f"Student model not found at {student_src}")

    quant_res = quantize_model_pipeline(str(student_dst), models_dir=models_dir_path)
    
    bench_res = {}
    if quant_res.get("status") == "success" and quant_res.get("tflite_path"):
        bench_res = benchmark_model_pipeline(quant_res["tflite_path"])
    else:
        print("Skipping benchmark as quantization failed.")

    # ── Phase 6: Visual Evaluation on Test Set ─────────────
    print(f"\n─── Phase 6: Evaluation & Visualization ───")
    try:
        from src.evaluate import evaluate_and_plot
        
        student_stem = student_src.stem
        eval_out_path = Path(output_dir) / f"{student_stem}_eval_plot.png"
        
        eval_results = evaluate_and_plot(
            model_name=student_stem,
            config_path=config_path,
            out_path=str(eval_out_path)
        )
        
        print(f"Evaluation metrics:")
        for k, v in eval_results.items():
            print(f"  {k}: {v:.4f}")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("PIPELINE COMPLETE - NANO_U STUDENT TRAINED SUCCESSFULLY")
    print("The final model produced by this pipeline is the nano_u (student).")
    print("Knowledge has been distilled from the bu_net (teacher).")
    print(f"   Teacher Path:     {teacher_dst}")
    print(f"   Student Path:     {student_res['model_path']}")
    print(f"   TFLite:      {quant_res.get('tflite_path', 'N/A')}")
    print(f"   Size:        {quant_res.get('size_kb', 'N/A')} KB")
    
    if bench_res and bench_res.get("inference"):
        inf = bench_res.get("inference", {})
        print(f"   Latency:     {inf.get('avg_latency_ms', '?'):.2f} ms")
        print(f"   Throughput:  {inf.get('throughput_fps', '?'):.1f} FPS")
    if 'eval_results' in locals() and eval_results:
        print(f"   Test IoU:    {eval_results.get('iou', 0):.4f}")
        print(f"   Plot saved:  {eval_out_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
