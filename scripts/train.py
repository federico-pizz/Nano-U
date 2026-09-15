"""Train Nano-U, from configuration to an evaluated INT8 TFLite model.

Two paths, one program:

  default          Quantization-Aware Distillation (QAD).
                   Phase 1 - train the teacher (bu_net).
                   Phase 2 - train the student (nanou) by knowledge distillation
                             with QAT applied from epoch 1.
                   Then     export the student to INT8 TFLite + quant_params.json,
                            and evaluate the teacher (FP32) and the student (FP32
                            and INT8) on the test split.

  --no-distill     Train one model on its own - the teacher as a standalone
                   baseline, or a student without KD as an ablation. It is
                   evaluated as FP32, and anything that is not `bu_net` is also
                   exported and evaluated as INT8.

Both write <results_dir>/summary.json, with the same schema.

Usage:
    python scripts/train.py
    python scripts/train.py --preset tinyagri
    python scripts/train.py --no-distill --model bu_net
    python scripts/train.py --no-distill --model nanou --preset tinyagri
"""

import argparse
import sys
from pathlib import Path

from nanou.config import preset_path, resolve
from nanou.pipeline import SUMMARY_FILE, run_pipeline
from nanou.preflight import PreflightError, require_dataset


def _miou(report):
    return f"{report['miou']:.4f}" if report else "-"


def _print_summary(summary, summary_path):
    print(f"\n{'=' * 55}\nDONE")
    for name, entry in summary["models"].items():
        print(f"   {name}")
        print(f"      model:         {entry['model_path']}")
        if entry["tflite_path"]:
            print(f"      TFLite:        {entry['tflite_path']} ({entry['tflite_size_kb']} KB)")
            print(
                f"      quant params:  {entry['quant_params_path']}"
                "  <- required by firmware/build.rs"
            )
        print(f"      mIoU FP32:     {_miou(entry['eval_fp32'])}")
        print(f"      mIoU INT8:     {_miou(entry['eval_int8'])}")
    print(f"   Summary JSON:  {summary_path}")
    print(f"{'=' * 55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Nano-U: QAD by default, or a single model with --no-distill"
    )
    parser.add_argument(
        "--preset",
        default="tinyagri",
        help="config preset from config/presets/ (default: tinyagri)",
    )
    parser.add_argument(
        "--config", default=None, help="explicit config file, used instead of --preset"
    )
    parser.add_argument(
        "--no-distill", action="store_true", help="Train one model without knowledge distillation"
    )
    # Free-form rather than a choice list: this is the config section name that
    # run_training passes through as the experiment name, so a config may define
    # sections beyond bu_net/nanou.
    parser.add_argument(
        "--model", default=None, help="With --no-distill: the model/experiment section to train"
    )
    args = parser.parse_args()

    if args.model and not args.no_distill:
        parser.error("--model applies only to --no-distill; the QAD path trains both models")
    if args.no_distill and not args.model:
        parser.error("--no-distill requires --model (e.g. --model bu_net)")

    config_path = args.config or str(preset_path(args.preset))
    config = resolve(path=config_path)
    output_dir = config.output_dir or "results/"

    # The one expected failure, reported as its fix. Anything else is a bug or
    # a broken config, and is left to arrive with its traceback.
    try:
        require_dataset(config.data_root, preset=args.preset)
    except PreflightError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

    summary = run_pipeline(config_path, output_dir, distil=not args.no_distill, model=args.model)
    _print_summary(summary, Path(output_dir) / SUMMARY_FILE)


if __name__ == "__main__":
    main()
