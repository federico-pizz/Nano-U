"""Leakage-safe grouped cross-validation search.

Runs unattended for hours; every fold is split by whole capture sequence, so no
scene appears on both sides of a fold. See `nanou.search` for the logic.

    python scripts/sweep.py --preset tinyagri --k 5 --epochs 50 \
        --temperatures 2 4 8 --alphas 0.3 0.5 0.7 \
        --regimes none geometric photometric full --ce on off
"""

import argparse
import os

from nanou.config import load_config, preset_path
from nanou.preflight import PreflightError, require_dataset
from nanou.search import AUGMENT_REGIMES, expand_grid, run_cv


def main():
    p = argparse.ArgumentParser(description="Leakage-safe grouped CV hyperparameter search")
    p.add_argument(
        "--preset",
        default="tinyagri",
        help="config preset from config/presets/ (default: tinyagri)",
    )
    p.add_argument("--config", default=None, help="explicit config file, used instead of --preset")
    p.add_argument("--k", type=int, default=5, help="number of grouped folds")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--temperatures", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument(
        "--regimes",
        nargs="+",
        default=["none", "geometric", "photometric", "full"],
        choices=sorted(AUGMENT_REGIMES),
    )
    p.add_argument(
        "--ce",
        nargs="+",
        default=["on", "off"],
        choices=["on", "off"],
        help="CE-loss ablation: 'on' keeps CE, 'off' sets alpha=1.0",
    )
    p.add_argument(
        "--tversky",
        type=float,
        nargs="+",
        default=[0.0],
        help="conservative-loss ablation: student supervised loss is "
        "(1-w)*BCE + w*Tversky(alpha_fp, beta_fn) for each weight w. "
        "Default [0.0] = pure BCE (no sweep).",
    )
    p.add_argument(
        "--tversky-alpha",
        type=float,
        nargs="+",
        default=[0.7],
        help="Tversky alpha_fp (false-positive weight) values to sweep. "
        "By default beta_fn is derived as 1-alpha (stay on the "
        "alpha+beta=1 simplex); override with --tversky-beta. Only active "
        "for tversky weights > 0.",
    )
    p.add_argument(
        "--tversky-beta",
        type=float,
        nargs="+",
        default=None,
        help="explicit beta_fn values, paired element-wise with "
        "--tversky-alpha (must match its length). Omit to couple "
        "beta = 1 - alpha.",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument(
        "--fixed-teacher",
        default=None,
        help="reuse one teacher .h5 for speed (logs a leakage caveat)",
    )
    p.add_argument(
        "--select",
        default="miou",
        choices=["f0.5", "f1", "f2", "miou"],
        help="primary selection metric for Nano-U (default mIoU; F0.5 is "
        "the tiebreak/safety axis). The teacher is always plain.",
    )
    p.add_argument(
        "--min-miou",
        type=float,
        default=0.0,
        help="only relevant with a non-mIoU --select: restrict the choice "
        "to configs whose mean mIoU clears this floor (e.g. 0.70); "
        "falls back to best available if none qualify",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of STUDENT pipelines to run concurrently (each in "
        "its own process/GPU context). Set to how many tiny Nano-U "
        "QAD pipelines fit in VRAM (e.g. 3). Default 1 = sequential.",
    )
    p.add_argument(
        "--teacher-jobs",
        type=int,
        default=1,
        help="concurrent BU-Net teacher trainings. The teacher is ~12.85M "
        "params and nearly fills a 6 GB GPU, so keep this low (default "
        "1 = sequential); raise only on a larger GPU.",
    )
    p.add_argument("--output", default=None, help="defaults to <results_dir>/cv")
    args = p.parse_args()

    if args.fixed_teacher:
        print(
            "[CV] WARNING: --fixed-teacher reuses one teacher across folds; "
            "the teacher may have seen some folds' validation frames (leakage)."
        )

    config_path = args.config or str(preset_path(args.preset))
    config = load_config(config_path)
    try:
        require_dataset(config["data"]["paths"]["processed"]["root"], preset=args.preset)
    except PreflightError as e:
        raise SystemExit(f"\n{e}") from e
    output = args.output or os.path.join(
        config.get("data", {}).get("paths", {}).get("results_dir", "results"), "cv"
    )
    ce_options = [c == "on" for c in args.ce]

    # Build the (weight, alpha_fp, beta_fn) Tversky specs. alpha/beta are paired
    # element-wise (not a 2D product) to stay on the chosen simplex; beta defaults
    # to 1-alpha. Weight 0.0 is pure BCE, so its alpha/beta are irrelevant.
    if args.tversky_beta is not None:
        if len(args.tversky_beta) != len(args.tversky_alpha):
            p.error("--tversky-beta must have the same length as --tversky-alpha")
        ab_pairs = list(zip(args.tversky_alpha, args.tversky_beta, strict=True))
    else:
        ab_pairs = [(a, round(1.0 - a, 6)) for a in args.tversky_alpha]
    tversky_specs: list[tuple[float, float, float]] = []
    for w in args.tversky:
        if w == 0.0:
            tversky_specs.append((0.0, 0.7, 0.3))  # alpha/beta unused at w=0
        else:
            tversky_specs.extend((w, a, b) for a, b in ab_pairs)

    grid = expand_grid(args.temperatures, args.alphas, args.regimes, ce_options, tversky_specs)

    run_cv(
        config_path,
        k=args.k,
        epochs=args.epochs,
        grid=grid,
        output_dir=output,
        threshold=args.threshold,
        seed=args.seed,
        fixed_teacher=args.fixed_teacher,
        select_primary=f"{args.select}_mean",
        min_miou=args.min_miou,
        jobs=args.jobs,
        teacher_jobs=args.teacher_jobs,
    )


if __name__ == "__main__":
    main()
