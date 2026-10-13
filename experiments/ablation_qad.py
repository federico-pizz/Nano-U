"""Ablation: QAD against its own parts.

Claim: distillation and quantization-aware training
together carry the student through INT8 better than either one alone. The
paper has no ablation, so the four cells are measured here:

    variant    teacher term   QAT    INT8 weights from
    qad        yes            yes    the QAT-trained model
    qat_only   no             yes    the QAT-trained model
    kd_only    yes            no     post-training quantization
    ptq        no             no     post-training quantization

Every variant trains through the distillation step, so all four share the
preset's supervised loss (BCE plus its Tversky blend) and only the two switches
move: alpha=0 drops the teacher term, qat_enabled=false trains in float. The
teacher is the preset's shipped one, not retrained.

    python experiments/ablation_qad.py --preset tinyagri
    python experiments/ablation_qad.py --preset tinyagri --seeds 3   # 3 runs per cell
    python experiments/ablation_qad.py --preset tinyagri --epochs 1   # does it run

With --seeds N every cell is trained N times, from the preset's seed upward.
One run per cell cannot separate the cells: the seed-to-seed spread of this
student is about 0.002 mIoU on TinyAgri and 0.023 on Botanic Garden, against
differences between cells of about 0.01. Repeats are what make the comparison
mean anything.

Writes results/experiments/ablation_qad/summary.json: one row per run, and
per-variant mean and sample std when there is more than one seed.
"""

import argparse
import statistics

from nanou.config import get_project_root, preset_path, resolve
from nanou.evaluate import write_report
from nanou.pipeline import STUDENT, train_and_score
from nanou.preflight import PreflightError, require_dataset, require_teacher

NAME = "ablation_qad"

VARIANTS = {
    "qad": {},
    "qat_only": {"alpha": 0.0},
    "kd_only": {"qat_enabled": False},
    "ptq": {"alpha": 0.0, "qat_enabled": False},
}


def main():
    p = argparse.ArgumentParser(description="QAD vs QAT-only vs KD-only vs PTQ")
    p.add_argument("--preset", default="tinyagri", help="config preset (default: tinyagri)")
    p.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="runs per variant, from the preset's seed upward (default: 1)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the preset's epochs; 1 checks the experiment runs end to end",
    )
    args = p.parse_args()
    if args.seeds < 1:
        p.error("--seeds needs at least 1")

    student = resolve(preset=args.preset).for_model(STUDENT)
    try:
        require_dataset(student.data_root, preset=args.preset)
        require_teacher(student.teacher_weights, preset=args.preset)
    except PreflightError as e:
        raise SystemExit(f"\n{e}") from e

    out = get_project_root() / "results" / "experiments" / NAME
    first = student.seed if student.seed is not None else 0
    rows = []
    for variant, switches in VARIANTS.items():
        for seed in range(first, first + args.seeds):
            # One seed keeps the old run layout; repeats get a seed folder each.
            run_dir = (
                out / "runs" / variant
                if args.seeds == 1
                else out / "runs" / variant / f"seed{seed}"
            )
            # Weights go under the run, never over the preset's shipped model.
            overrides = {**switches, "seed": seed, "models_dir": str(run_dir / "model")}
            if args.epochs is not None:
                overrides["epochs"] = args.epochs
            entry = train_and_score(STUDENT, str(preset_path(args.preset)), str(run_dir), overrides)
            fp32, int8 = entry["eval_fp32"], entry["eval_int8"]
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "teacher_term": switches.get("alpha", student.alpha) > 0,
                    "qat": switches.get("qat_enabled", student.qat_enabled),
                    "miou_fp32": fp32["miou"],
                    "miou_int8": int8["miou"],
                    "int8_gap": int8["miou"] - fp32["miou"],
                    "f0.5_int8": int8["f0.5"],
                    "precision_int8": int8["precision"],
                    "recall_int8": int8["recall"],
                }
            )

    stats = {}
    for variant in VARIANTS:
        seen = [r["miou_int8"] for r in rows if r["variant"] == variant]
        stats[variant] = {
            "runs": len(seen),
            "miou_int8_mean": statistics.fmean(seen),
            # One run has no spread of its own: the seed ablation measures it.
            "miou_int8_std": statistics.stdev(seen) if len(seen) > 1 else None,
        }

    print(f"\n{'variant':<10}{'teacher':>9}{'qat':>6}{'runs':>6}{'mIoU int8':>11}{'std':>9}")
    for variant, st in stats.items():
        row = next(r for r in rows if r["variant"] == variant)
        std = "n/a" if st["miou_int8_std"] is None else f"{st['miou_int8_std']:.4f}"
        print(
            f"{variant:<10}{str(row['teacher_term']):>9}{str(row['qat']):>6}"
            f"{st['runs']:>6}{st['miou_int8_mean']:>11.4f}{std:>9}"
        )
    write_report(
        {"rows": rows, "per_variant": stats},
        out / "summary.json",
        experiment=NAME,
        preset=args.preset,
        seeds=args.seeds,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
