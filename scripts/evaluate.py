"""Evaluate a trained model on a dataset split, or report its structure.

Two modes:

  default   Score the model on a split and write, under
            results/<dataset>/<model>/, a prediction montage
            eval_<fmt>_<split>.png, a threshold-sweep figure
            eval_<fmt>_<split>_metrics.png and one report eval_<fmt>_<split>.json.
            --format picks the file scored: int8 is the exported .tflite, fp32
            the Keras checkpoint. Without it, the .tflite is scored if it has
            been exported and the Keras model otherwise - the line starting
            [eval] says which.

  --info    Do not evaluate. Load the FP32 Keras model and print parameter
            counts, layer counts, the Keras summary and a serialization
            round-trip check, plus the size of the INT8 .tflite sibling if one
            exists.

Usage:
    python scripts/evaluate.py                                     # nanou, test split
    python scripts/evaluate.py nanou --format fp32
    python scripts/evaluate.py bu_net --preset botanicgarden
    python scripts/evaluate.py nanou --split val --threshold 0.6
    python scripts/evaluate.py nanou --info
"""

import argparse
import os

from nanou.config import preset_path, resolve
from nanou.evaluate import evaluate
from nanou.preflight import (
    PreflightError,
    require_keras_model,
    require_model,
    require_tflite_model,
)


def show_info(model_path, model_name, models_dir):
    """Print parameter counts, layer counts, summary and serialization check."""
    import tensorflow_model_optimization as tfmot
    import tf_keras as keras

    from nanou.introspect import (
        count_parameters,
        get_model_config,
        get_model_summary,
        validate_model_serialization,
    )
    from nanou.predictor import CUSTOM_OBJECTS

    # quantize_scope + PadToMatch are both required: a QAT-wrapped checkpoint
    # carries tfmot wrapper layers, and every architecture here carries PadToMatch.
    with tfmot.quantization.keras.quantize_scope():
        with keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            model = keras.models.load_model(model_path, compile=False)

    info = get_model_config(model)
    print(f"\n{'=' * 55}")
    print(f"MODEL INFO: {model_name}")
    print(f"{'=' * 55}")
    print(f"   Path:            {model_path}")
    print(f"   Input shape:     {info['input_shape']}")
    print(f"   Parameters:      {count_parameters(model):,}")
    print(f"   Layers:          {info['layer_count']}")
    print(f"   Serializable:    {validate_model_serialization(model)}")

    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")
    if os.path.exists(tflite_path):
        print(f"   INT8 TFLite:     {tflite_path} ({os.path.getsize(tflite_path) / 1024:.2f} KB)")
    else:
        print("   INT8 TFLite:     not exported - python scripts/export.py")

    print(f"\n{get_model_summary(model)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model, or report its structure")
    parser.add_argument(
        "model",
        nargs="?",
        default="nanou",
        help="Model to evaluate (.h5 or .tflite basename without extension)",
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
        "--format",
        choices=["int8", "fp32"],
        default=None,
        help="Score the exported INT8 .tflite or the FP32 Keras model "
        "(default: INT8 if it has been exported, FP32 otherwise)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate (default test; val is what selection may look at)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Operating decision threshold for the headline metrics",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Report model structure and parameter counts instead of evaluating",
    )
    args = parser.parse_args()
    config_path = args.config or str(preset_path(args.preset))
    models_dir = resolve(path=config_path).models_dir

    try:
        if args.info or args.format == "fp32":
            model_path = require_keras_model(models_dir, args.model, preset=args.preset)
        elif args.format == "int8":
            model_path = require_tflite_model(models_dir, args.model, preset=args.preset)
        else:
            model_path = require_model(models_dir, args.model, preset=args.preset)
    except PreflightError as e:
        raise SystemExit(f"\n{e}") from e

    if args.info:
        show_info(model_path, args.model, models_dir)
        return

    evaluate(model_path, config_path, split=args.split, threshold=args.threshold)


if __name__ == "__main__":
    main()
