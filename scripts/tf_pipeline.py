#!/usr/bin/env python3
"""
Python replacement for scripts/tf_pipeline.sh with NAS integration.

Provides a lightweight CLI with subcommands that mirror the shell pipeline:
- prepare: run data preparation
- build: build models
- train: train models with optional NAS regularization
- distill: distillation pipeline
- analyze: run NAS analysis using RunningCovariance or DistillationAwareNAS
- quantize: call quantize script
- infer: run inference on models
- eval: evaluate models

This script intentionally delegates heavy work to existing src/* modules and
adds orchestration, argument parsing, logging, and NAS-enabled train options.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENV = ROOT / ".venv-tf"
PYTHON = VENV / "bin" / "python"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("tf_pipeline")


def ensure_venv(install_deps: bool = False):
    if not VENV.exists():
        log.error("virtual environment %s not found", VENV)
        sys.exit(1)
    if not PYTHON.exists():
        log.error("python binary not found in venv: %s", PYTHON)
        sys.exit(1)
    if install_deps:
        subprocess.check_call([str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])


def run_module(module_path: Path, args=None):
    args = args or []
    cmd = [str(PYTHON), str(module_path)] + args
    log.info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_build(args):
    # Prefer the NAS-aware wrapper when present
    build_wrapper = SRC / "build_nas.py"
    if build_wrapper.exists():
        return run_module(build_wrapper, args=[])

    # Fallback to original build script
    build = SRC / "build_net.py"
    if not build.exists():
        log.warning("build_net.py not found, skipping")
        return 0
    return run_module(build, args=[])


def cmd_train(args):
    # Prefer train_with_nas.py when NAS is enabled, fallback to train.py
    train = SRC / ("train_with_nas.py" if args.enable_nas else "train.py")
    if not train.exists():
        log.warning("train.py not found, skipping")
        return 0

    # Build base command for subprocess fallback
    cmd = [str(PYTHON), str(train), "--model", args.model]
    if args.distill and args.teacher_weights:
        cmd += ["--distill", "--teacher-weights", args.teacher_weights]
    if args.enable_nas:
        # pass NAS flags through env or CLI variable
        cmd += ["--enable-nas"]
        if args.nas_layers:
            cmd += ["--nas-layers", args.nas_layers]
        if args.nas_weight:
            cmd += ["--nas-weight", str(args.nas_weight)]

    # direct execution of selected training module
    log.info("Executing: %s", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_distill(args):
    return cmd_train(args)


def cmd_analyze(args):
    # call a small analysis script inside src or run inline
    analyze = SRC / "nas_analyze.py"
    if not analyze.exists():
        log.warning("nas_analyze.py not found in src; creating a minimal analysis runner")
        # fallback: import DistillationAwareNAS and run simple analyze
        # keep minimal to avoid heavy imports here
        log.info("Please run a dedicated analysis script for full results")
        return 0
    return run_module(analyze, args=["--num-batches", str(args.num_batches)])


def cmd_quantize(args):
    quant = SRC / "quantize.py"
    if not quant.exists():
        log.warning("quantize.py not found, skipping")
        return 0
    return run_module(quant, args=["--model-name", args.model_name, "--output", args.output])


def cmd_infer(args):
    infer = SRC / "infer.py"
    if not infer.exists():
        log.warning("infer.py not found, skipping")
        return 0
    # Some infer scripts accept a positional model path; forward as positional arg.
    return run_module(infer, args=[args.model_path])


def cmd_eval(args):
    evaluate = SRC / "evaluate.py"
    if not evaluate.exists():
        log.warning("evaluate.py not found, skipping")
        return 0
    return run_module(evaluate, args=["--model-name", args.model_name, "--out", args.out])


def cmd_pipeline(args):
    """Orchestrate end-to-end: 1) train bu_net with NAS, 2) distill nano_u from bu_net, 3) quantize nano_u to int8, 4) evaluate models."""
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1) Train bu_net with NAS
    bu_train = SRC / "train_with_nas.py"
    if not bu_train.exists():
        log.error("train_with_nas.py not found in src")
        return 1
    r = run_module(bu_train, args=["--model", "bu_net", "--enable-nas"])
    if r != 0:
        log.error("bu_net training failed: %s", r)
        return r

    teacher_ckpt = models_dir / "bu_net.keras"
    if not teacher_ckpt.exists():
        log.error("Expected teacher checkpoint not found: %s", teacher_ckpt)
        return 1

    # 2) Distill nano_u using trained bu_net
    r = run_module(bu_train, args=["--model", "nano_u", "--distill", "--teacher-weights", str(teacher_ckpt)])
    if r != 0:
        log.error("Distillation failed: %s", r)
        return r

    # 3) Quantize nano_u to int8
    quant = SRC / "quantize.py"
    int8_out = models_dir / "nano_u-int8.tflite"
    if quant.exists():
        r = run_module(quant, args=["--model-name", "nano_u", "--output", str(int8_out)])
        if r != 0:
            log.error("Quantization failed: %s", r)
            return r
    else:
        log.warning("quantize.py not found, skipping quantization")

    # 4) Evaluate models (float students)
    evaluate = SRC / "evaluate.py"
    if evaluate.exists():
        # evaluate bu_net
        r1 = run_module(evaluate, args=["--model-name", "bu_net", "--out", str(models_dir / "bu_net_metrics.json")])
        # evaluate nano_u
        r2 = run_module(evaluate, args=["--model-name", "nano_u", "--out", str(models_dir / "nano_u_metrics.json")])
        if r1 != 0 or r2 != 0:
            log.error("Evaluation returned non-zero exit codes: %s %s", r1, r2)
            return 1
    else:
        log.warning("evaluate.py not found, skipping evaluation")

    log.info("Pipeline finished. Artifacts in %s", models_dir)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tf_pipeline")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies into .venv-tf")

    sub = parser.add_subparsers(dest="cmd")

    s_build = sub.add_parser("build")

    s_train = sub.add_parser("train")
    s_train.add_argument("--model", required=True)
    s_train.add_argument("--distill", action="store_true")
    s_train.add_argument("--teacher-weights")
    s_train.add_argument("--enable-nas", action="store_true")
    s_train.add_argument("--nas-layers")
    s_train.add_argument("--nas-weight", type=float)

    s_distill = sub.add_parser("distill")
    s_distill.add_argument("--model", required=True)
    s_distill.add_argument("--teacher-weights")

    s_analyze = sub.add_parser("analyze")
    s_analyze.add_argument("--num-batches", type=int, default=50)

    s_quantize = sub.add_parser("quantize")
    s_quantize.add_argument("--model-name", required=True)
    s_quantize.add_argument("--output", required=True)

    s_infer = sub.add_parser("infer")
    s_infer.add_argument("model_path", help="Path to model to run inference with")

    s_eval = sub.add_parser("eval")
    s_eval.add_argument("--model-name", required=True)
    s_eval.add_argument("--out", required=True)

    s_pipeline = sub.add_parser("pipeline")
    s_pipeline.add_argument("--enable-nas-for-student", action="store_true", help="Enable NAS for student during distillation")

    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    ensure_venv(args.install_deps)

    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "distill":
        return cmd_distill(args)
    if args.cmd == "analyze":
        return cmd_analyze(args)
    if args.cmd == "quantize":
        return cmd_quantize(args)
    if args.cmd == "infer":
        return cmd_infer(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "pipeline":
        return cmd_pipeline(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
