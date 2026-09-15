"""Leakage-safe grouped cross-validation search, one sweep file at a time.

The folds, the epochs and the grid live in the sweep file (see sweeps/), so a
sweep is committed next to the preset it produced. Only what describes the
machine stays on the command line. Every fold is split by whole capture
sequence, so no scene appears on both sides of a fold; see `nanou.search`.

    python scripts/sweep.py sweeps/tinyagri-stage3.yaml --preset tinyagri --jobs 3

Runs unattended for hours. The table lands in <results_dir>/cv/<sweep name>/,
next to a copy of the sweep file that produced it.
"""

import argparse
import os
import shutil
from pathlib import Path

from nanou.config import preset_path, resolve
from nanou.preflight import PreflightError, require_dataset
from nanou.search import load_sweep, run_cv


def main():
    p = argparse.ArgumentParser(description="Leakage-safe grouped CV hyperparameter search")
    p.add_argument("sweep", help="sweep file with the folds, epochs and grid (see sweeps/)")
    p.add_argument(
        "--preset",
        default="tinyagri",
        help="config preset from config/presets/ (default: tinyagri)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="student pipelines run at once, each in its own process (default 1). "
        "Set it to how many Nano-U runs fit in VRAM.",
    )
    p.add_argument(
        "--teacher-jobs",
        type=int,
        default=1,
        help="BU-Net teachers trained at once (default 1). The teacher nearly "
        "fills a 6 GB GPU, so raise this only on a larger one.",
    )
    p.add_argument("--output", default=None, help="defaults to <results_dir>/cv/<sweep name>")
    args = p.parse_args()

    try:
        sweep = load_sweep(args.sweep)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"\n{e}") from e

    config_path = str(preset_path(args.preset))
    config = resolve(path=config_path)
    try:
        require_dataset(config.data_root, preset=args.preset)
    except PreflightError as e:
        raise SystemExit(f"\n{e}") from e

    results_dir = config.output_dir or "results"
    output = args.output or os.path.join(results_dir, "cv", Path(args.sweep).stem)
    os.makedirs(output, exist_ok=True)
    shutil.copy(args.sweep, output)

    run_cv(
        config_path,
        output_dir=output,
        jobs=args.jobs,
        teacher_jobs=args.teacher_jobs,
        **sweep,
    )


if __name__ == "__main__":
    main()
