"""Distillation pipeline: Teacher (BU-Net) → Student (Nano-U) with NAS monitoring, quantize, benchmark.

Replaces the old --full end-to-end pipeline. Runs the full sequence automatically:
  Phase 1 — Train Teacher (BU-Net) with NAS monitoring
  Phase 2 — Copy teacher weights to models/
  Phase 3 — Train Student (Nano-U) with Distillation + NAS monitoring
  Phase 4 — Quantize Student → INT8 TFLite
  Phase 5 — Benchmark

Usage:
    python scripts/train_distillation.py
    python scripts/train_distillation.py --output results/ --teacher-experiment bu_net_nas --student-experiment distillation_nas
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_training_pipeline, quantize_and_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Teacher→Student distillation pipeline with NAS monitoring → quantize → benchmark"
    )
    parser.add_argument("--config", default="config/experiments.yaml")
    parser.add_argument("--output", default="results/", help="Base output directory")
    parser.add_argument("--models-dir", default="models/", help="Where to place final artifacts")
    parser.add_argument(
        "--teacher-experiment", default="bu_net_nas",
        help="Experiment name for teacher training (default: bu_net_nas)"
    )
    parser.add_argument(
        "--student-experiment", default="distillation_nas",
        help="Experiment name for student distillation (default: distillation_nas)"
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print("🎓 TEACHER → STUDENT DISTILLATION PIPELINE")
    print(f"{'='*55}")

    # ── Phase 1: Train Teacher ─────────────────────────────────────────────
    print(f"\n─── Phase 1: Training Teacher ({args.teacher_experiment}) ───")
    teacher_res = run_training_pipeline(args.teacher_experiment, args.config, args.output)

    if teacher_res["status"] != "success":
        print(f"❌ Teacher training failed: {teacher_res.get('error')}")
        sys.exit(1)

    print(f"✅ Teacher trained  →  {teacher_res['model_path']}")

    # ── Phase 2: Copy teacher weights ──────────────────────────────────────
    print(f"\n─── Phase 2: Copying Teacher Weights ───")
    teacher_src = Path(teacher_res["model_path"])
    teacher_dst = models_dir / teacher_src.name

    if teacher_src.exists():
        shutil.copy(teacher_src, teacher_dst)
        print(f"✅ Teacher weights copied  →  {teacher_dst}")
    else:
        print(f"❌ Teacher model not found at {teacher_src}")
        sys.exit(1)

    # ── Phase 3: Train Student ─────────────────────────────────────────────
    print(f"\n─── Phase 3: Training Student ({args.student_experiment}) ───")
    student_res = run_training_pipeline(args.student_experiment, args.config, args.output)

    if student_res["status"] != "success":
        print(f"❌ Student training failed: {student_res.get('error')}")
        sys.exit(1)

    print(f"✅ Student trained  →  {student_res['model_path']}")

    # ── Phase 4: Copy student weights & Quantize ───
    print(f"\n─── Phase 4 & 5: Copy Weights + Quantize + Benchmark ───")
    student_src = Path(student_res["model_path"])
    student_dst = models_dir / student_src.name
    if student_src.exists():
        shutil.copy(student_src, student_dst)
        print(f"✅ Student weights copied  →  {student_dst}")
    else:
        print(f"❌ Student model not found at {student_src}")

    qb = quantize_and_benchmark(str(student_dst), models_dir=args.models_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("✨ PIPELINE COMPLETE")
    print(f"   Teacher:     {teacher_dst}")
    print(f"   Student:     {student_res['model_path']}")
    print(f"   TFLite:      {qb['quantization'].get('tflite_path', 'N/A')}")
    print(f"   Size:        {qb['quantization'].get('size_kb', 'N/A')} KB")
    inf = qb["benchmark"].get("inference", {})
    print(f"   Latency:     {inf.get('avg_latency_ms', '?'):.2f} ms")
    print(f"   Throughput:  {inf.get('throughput_fps', '?'):.1f} FPS")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
