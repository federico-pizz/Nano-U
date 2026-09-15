"""Seed variance: an error bar for every headline number.

Claim: the headline mIoU and F0.5 come from single
runs, so a gap like 0.870 vs 0.876 cannot be told apart from run-to-run noise.
This retrains the student with N seeds, everything else as the preset sets it,
and reports mean +/- std (sample std) of each headline metric in both formats.

The first seed is the preset's own, so row 0 is the shipped configuration
(up to GPU nondeterminism). The teacher is the preset's shipped one and does
not vary: this is the student's variance, not the whole pipeline's.

    python experiments/seed_variance.py --preset tinyagri --seeds 5
    python experiments/seed_variance.py --preset tinyagri --seeds 2 --epochs 1   # does it run

Writes results/experiments/seed_variance/summary.json: one row per seed, and
the mean and std of each metric.
"""

import argparse

import numpy as np

from nanou.config import get_project_root, preset_path, resolve
from nanou.evaluate import write_report
from nanou.pipeline import STUDENT, train_and_score
from nanou.preflight import PreflightError, require_dataset, require_teacher

NAME = "seed_variance"

HEADLINE = ("miou", "dice", "precision", "recall", "f0.5")


def main():
    p = argparse.ArgumentParser(description="Mean +/- std of the headline metrics over seeds")
    p.add_argument("--preset", default="tinyagri", help="config preset (default: tinyagri)")
    p.add_argument("--seeds", type=int, default=5, help="how many seeds to train (default: 5)")
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the preset's epochs; 1 checks the experiment runs end to end",
    )
    args = p.parse_args()
    if args.seeds < 2:
        p.error("--seeds needs at least 2 for a standard deviation")

    student = resolve(preset=args.preset).for_model(STUDENT)
    try:
        require_dataset(student.data_root, preset=args.preset)
        require_teacher(student.teacher_weights, preset=args.preset)
    except PreflightError as e:
        raise SystemExit(f"\n{e}") from e

    first = student.seed if student.seed is not None else 0
    out = get_project_root() / "results" / "experiments" / NAME
    rows = []
    for seed in range(first, first + args.seeds):
        run_dir = out / "runs" / f"seed{seed}"
        overrides = {"seed": seed, "models_dir": str(run_dir / "model")}
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        entry = train_and_score(STUDENT, str(preset_path(args.preset)), str(run_dir), overrides)
        row = {"seed": seed}
        for fmt in ("fp32", "int8"):
            for metric in HEADLINE:
                row[f"{metric}_{fmt}"] = entry[f"eval_{fmt}"][metric]
        rows.append(row)

    stats = {}
    for key in rows[0]:
        if key == "seed":
            continue
        values = np.array([r[key] for r in rows], dtype=np.float64)
        stats[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}

    print(f"\n{args.seeds} seeds, mean +/- std")
    for key, s in stats.items():
        print(f"  {key:<16}{s['mean']:.4f} +/- {s['std']:.4f}")
    write_report(
        {"rows": rows, "mean_std": stats},
        out / "summary.json",
        experiment=NAME,
        preset=args.preset,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
