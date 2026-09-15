"""Ablation: the distillation temperature, with and without the T**2 factor.

Claim: the T**2 rescaling of the teacher term is the
classical Hinton correction, which assumes T > 1. The tuned TinyAgri T is 0.7,
where T**2 = 0.49 turns the correction into a down-weighting of the teacher.
This scores the student at each T with the factor on ("squared") and off
("none"), everything else as the preset sets it. The teacher is the preset's
shipped one, not retrained.

T = 1.0 is run once: the factor is 1 there, so both settings are the same run.
These are test-split numbers, a sensitivity curve around the tuned T; the T
itself was picked by the CV sweep (sweeps/), not by this table.

    python experiments/ablation_temperature.py --preset tinyagri
    python experiments/ablation_temperature.py --preset tinyagri --epochs 1   # does it run

Writes results/experiments/ablation_temperature/summary.json, one row per run.
"""

import argparse

from nanou.config import get_project_root, preset_path, resolve
from nanou.evaluate import write_report
from nanou.pipeline import STUDENT, train_and_score
from nanou.preflight import PreflightError, require_dataset, require_teacher

NAME = "ablation_temperature"

TEMPERATURES = (0.7, 1.0, 2.0, 4.0, 8.0)
T_SCALES = ("squared", "none")


def main():
    p = argparse.ArgumentParser(description="Distillation temperature, with and without T**2")
    p.add_argument("--preset", default="tinyagri", help="config preset (default: tinyagri)")
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the preset's epochs; 1 checks the experiment runs end to end",
    )
    args = p.parse_args()

    student = resolve(preset=args.preset).for_model(STUDENT)
    try:
        require_dataset(student.data_root, preset=args.preset)
        require_teacher(student.teacher_weights, preset=args.preset)
    except PreflightError as e:
        raise SystemExit(f"\n{e}") from e

    runs = [(t, s) for t in TEMPERATURES for s in T_SCALES if not (t == 1.0 and s == "none")]
    out = get_project_root() / "results" / "experiments" / NAME
    rows = []
    for temperature, t_scale in runs:
        run_dir = out / "runs" / f"T{temperature}_{t_scale}"
        overrides = {
            "temperature": temperature,
            "t_scale": t_scale,
            "models_dir": str(run_dir / "model"),
        }
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        entry = train_and_score(STUDENT, str(preset_path(args.preset)), str(run_dir), overrides)
        fp32, int8 = entry["eval_fp32"], entry["eval_int8"]
        rows.append(
            {
                "temperature": temperature,
                "t_scale": t_scale,
                "tuned": temperature == student.temperature and t_scale == student.t_scale,
                "miou_fp32": fp32["miou"],
                "miou_int8": int8["miou"],
                "f0.5_int8": int8["f0.5"],
                "precision_int8": int8["precision"],
                "recall_int8": int8["recall"],
            }
        )

    print(f"\n{'T':>5}  {'t_scale':<8}{'mIoU fp32':>11}{'mIoU int8':>11}{'F0.5 int8':>11}")
    for r in rows:
        mark = "  <- preset" if r["tuned"] else ""
        print(
            f"{r['temperature']:>5}  {r['t_scale']:<8}{r['miou_fp32']:>11.4f}"
            f"{r['miou_int8']:>11.4f}{r['f0.5_int8']:>11.4f}{mark}"
        )
    write_report(
        {"rows": rows},
        out / "summary.json",
        experiment=NAME,
        preset=args.preset,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
