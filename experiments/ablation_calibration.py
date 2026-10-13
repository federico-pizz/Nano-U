"""Ablation: how much each training regime needs good calibration data.

Claim under test: quantization-aware training earns its cost by making the
weights tolerate INT8, so a QAT-trained model should survive a bad calibration
where a float-trained one falls apart. The QAD ablation exports every variant
with the whole val split as calibration data, the friendly case, and there the
four variants land within a point of each other. This one removes the crutch:
each model is exported twice, once calibrated on real images and once on random
noise, and both are scored on the test split.

The noise case is not academic. The paper's TinyAgri INT8 number collapsed to
0.700 because `export_int8` calibrated on whatever the template config pointed
at, which was another domain's images or nothing at all.

Nothing is trained here. It re-exports models that `ablation_qad.py` already
produced, so run that first:

    python experiments/ablation_qad.py --preset tinyagri --seeds 3
    python experiments/ablation_calibration.py --preset tinyagri

    # models from an archived run, or any other directory of runs
    python experiments/ablation_calibration.py --preset tinyagri --runs <dir>

Writes results/experiments/ablation_calibration/summary.json: one row per
model and calibration source, with the drop from images to noise.
"""

import argparse
import statistics
from pathlib import Path

from nanou.config import get_project_root, preset_path, resolve
from nanou.evaluate import evaluate, write_report
from nanou.pipeline import STUDENT, export_int8
from nanou.preflight import PreflightError, require_dataset

NAME = "ablation_calibration"
SOURCES = ("images", "noise")


def find_models(runs_dir: Path) -> list[tuple[str, str, Path]]:
    """Every student .h5 under a run directory, as (variant, seed, path).

    Handles both layouts `ablation_qad.py` writes: `runs/<variant>/model/` for a
    single seed, and `runs/<variant>/seed<N>/model/` for repeats.
    """
    found = []
    for path in sorted(runs_dir.glob("*/**/model/*.h5")):
        parts = path.relative_to(runs_dir).parts
        variant = parts[0]
        seed = next((p for p in parts if p.startswith("seed")), "")
        found.append((variant, seed, path))
    return found


def main():
    p = argparse.ArgumentParser(description="INT8 with real calibration images against noise")
    p.add_argument("--preset", default="tinyagri", help="config preset (default: tinyagri)")
    p.add_argument(
        "--runs",
        default=None,
        help="directory of trained runs (default: results/experiments/ablation_qad/runs)",
    )
    args = p.parse_args()

    student = resolve(preset=args.preset).for_model(STUDENT)
    try:
        require_dataset(student.data_root, preset=args.preset)
    except PreflightError as e:
        raise SystemExit(f"\n{e}") from e

    root = get_project_root()
    runs_dir = (
        Path(args.runs) if args.runs else root / "results" / "experiments" / "ablation_qad" / "runs"
    )
    if not runs_dir.is_dir():
        raise SystemExit(f"\nno runs at {runs_dir}. Run experiments/ablation_qad.py first.")
    models = find_models(runs_dir)
    if not models:
        raise SystemExit(f"\nno student models under {runs_dir}")

    out = root / "results" / "experiments" / NAME
    config_path = str(preset_path(args.preset))
    rows = []
    for variant, seed, model_path in models:
        scores = {}
        for source in SOURCES:
            # Each export goes to its own directory: same file name, and the
            # quant params of one calibration must not overwrite the other's.
            export_dir = out / "runs" / variant / (seed or "seed") / source
            exported = export_int8(model_path, export_dir, config_path, calibration=source)
            # out_dir keeps the figures and report next to the export, instead
            # of overwriting the preset's own evaluation of the shipped model.
            scores[source] = evaluate(
                exported["tflite_path"],
                config_path=config_path,
                split="test",
                out_dir=export_dir,
            )
        row = {
            "variant": variant,
            "seed": seed,
            "model": str(model_path.relative_to(root))
            if root in model_path.parents
            else str(model_path),
        }
        for source in SOURCES:
            for metric in ("miou", "f0.5", "precision", "recall"):
                row[f"{metric}_{source}"] = scores[source][metric]
        row["miou_drop"] = scores["images"]["miou"] - scores["noise"]["miou"]
        rows.append(row)

    stats = {}
    for variant in sorted({r["variant"] for r in rows}):
        drops = [r["miou_drop"] for r in rows if r["variant"] == variant]
        stats[variant] = {
            "runs": len(drops),
            "miou_drop_mean": statistics.fmean(drops),
            "miou_drop_std": statistics.stdev(drops) if len(drops) > 1 else None,
        }

    print(f"\n{'variant':<10}{'runs':>6}{'mIoU images':>13}{'mIoU noise':>12}{'drop':>9}")
    for variant, st in stats.items():
        got = [r for r in rows if r["variant"] == variant]
        print(
            f"{variant:<10}{st['runs']:>6}"
            f"{statistics.fmean(r['miou_images'] for r in got):>13.4f}"
            f"{statistics.fmean(r['miou_noise'] for r in got):>12.4f}"
            f"{st['miou_drop_mean']:>+9.4f}"
        )
    write_report(
        {"rows": rows, "per_variant": stats},
        out / "summary.json",
        experiment=NAME,
        preset=args.preset,
        runs_dir=str(runs_dir),
    )


if __name__ == "__main__":
    main()
