"""Evolutionary NAS search: finds best architecture → trains it → quantizes → benchmarks.

Usage:
    python scripts/nas_search.py --model nano_u
    python scripts/nas_search.py --model bu_net --output results/nas_bu_net/
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_nas_search, run_training_pipeline, quantize_and_benchmark
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Evolutionary NAS search → train best arch → quantize → benchmark"
    )
    parser.add_argument(
        "--model", choices=["nano_u", "bu_net"], default="nano_u",
        help="Which model to search (default: nano_u)"
    )
    parser.add_argument(
        "--experiment", default="default",
        help="Experiment name defined in experiments.yaml (default: default)"
    )
    parser.add_argument("--config", default="config/experiments.yaml")

    parser.add_argument("--output", default="results/nas_search/", help="NAS search output dir")
    parser.add_argument("--models-dir", default="models/", help="Where to place final artifacts")
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Only run the search — do not train the best arch afterward"
    )
    args = parser.parse_args()

    # ── Phase 1: Evolutionary Search ──────────────────────────────────────
    search_results = run_nas_search(
        config_path=args.config,
        output_dir=args.output,
        model_name=args.model,
    )
    best_arch = search_results["best_arch"]

    if args.skip_train:
        print(f"\n⏭  --skip-train set. Stopping after search.")
        print(f"   Best arch: {best_arch}")
        print(f"   Results:   {Path(args.output) / 'best_arch.json'}")
        return

    # ── Phase 2: Train best architecture ──────────────────────────────────
    print(f"\n─── Training Best Architecture: {best_arch} ───")

    # Inject best arch_seq into the appropriate base experiment
    base_experiment = "default" if args.model == "nano_u" else "bu_net_nas"
    full_config = load_config(args.config)

    # Build an ephemeral experiment config that overrides arch_seq
    # We write the best_arch back into config by passing it via train_model directly
    import json, yaml, tempfile
    from src.train import train_model

    experiment_cfg = dict(full_config.get("experiments", {}).get(base_experiment, {}))
    experiment_cfg["arch_seq"] = best_arch
    experiment_cfg["model_name"] = args.model
    experiment_cfg["use_nas"] = True  # keep NAS monitoring on during full train

    # Inject into a temporary config
    tmp_cfg = dict(full_config)
    tmp_cfg.setdefault("experiments", {})[f"nas_best_{args.model}"] = experiment_cfg

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(tmp_cfg, tmp)
        tmp_path = tmp.name

    train_output = str(Path(args.output) / f"best_{args.model}_trained")
    result = run_training_pipeline(
        f"nas_best_{args.model}",
        config_path=tmp_path,
        output_dir=train_output,
    )

    if result["status"] != "success":
        print(f"❌ Training best arch failed: {result.get('error')}")
        sys.exit(1)

    print(f"✅ Best arch trained  →  {result['model_path']}")

    # ── Phase 3: Copy weights and Quantize ────────────────────────────────
    print(f"\n─── Copy Models + Quantize + Benchmark ───")
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    src_model = Path(result["model_path"])
    dst_model = models_dir / src_model.name
    
    if src_model.exists():
        shutil.copy(src_model, dst_model)
        print(f"✅ Best arch copied  →  {dst_model}")
    else:
        print(f"❌ Completed model not found at {src_model}")
        sys.exit(1)

    qb = quantize_and_benchmark(str(dst_model), models_dir=args.models_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("✨ NAS SEARCH COMPLETE")
    print(f"   Best arch:   {best_arch}")
    print(f"   Fitness:     {search_results['best_fitness']:.4f}")
    print(f"   Model:       {result['model_path']}")
    print(f"   TFLite:      {qb['quantization'].get('tflite_path', 'N/A')}")
    print(f"   Size:        {qb['quantization'].get('size_kb', 'N/A')} KB")
    inf = qb["benchmark"].get("inference", {})
    print(f"   Latency:     {inf.get('avg_latency_ms', '?'):.2f} ms")
    print(f"   Throughput:  {inf.get('throughput_fps', '?'):.1f} FPS")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
