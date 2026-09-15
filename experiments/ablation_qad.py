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
    python experiments/ablation_qad.py --preset tinyagri --epochs 1   # does it run

Writes results/experiments/ablation_qad/summary.json, one row per variant.
"""

import argparse

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

    out = get_project_root() / "results" / "experiments" / NAME
    rows = []
    for variant, switches in VARIANTS.items():
        run_dir = out / "runs" / variant
        # Weights go under the run, never over the preset's shipped model.
        overrides = {**switches, "models_dir": str(run_dir / "model")}
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        entry = train_and_score(STUDENT, str(preset_path(args.preset)), str(run_dir), overrides)
        fp32, int8 = entry["eval_fp32"], entry["eval_int8"]
        rows.append(
            {
                "variant": variant,
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

    print(f"\n{'variant':<10}{'teacher':>9}{'qat':>6}{'mIoU fp32':>11}{'mIoU int8':>11}{'gap':>9}")
    for r in rows:
        print(
            f"{r['variant']:<10}{str(r['teacher_term']):>9}{str(r['qat']):>6}"
            f"{r['miou_fp32']:>11.4f}{r['miou_int8']:>11.4f}{r['int8_gap']:>+9.4f}"
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
