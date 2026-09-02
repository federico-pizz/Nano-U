"""Evaluate a trained model on a dataset split, or report its structure.

Two modes:

  default   Score the model on a split and write the metric artefacts:
            eval_results.json, a format-tagged eval_results_{fp32,int8}.json,
            a prediction montage and a threshold-sweep report.

  --info    Do not evaluate. Load the FP32 Keras model and print parameter
            counts, layer counts, the Keras summary and a serialization
            round-trip check, plus the size of the INT8 .tflite sibling if one
            exists.

Usage:
    python scripts/evaluate.py                                     # nano_u, test split
    python scripts/evaluate.py bu_net --preset botanicgarden
    python scripts/evaluate.py nano_u --split val --threshold 0.6
    python scripts/evaluate.py nano_u --info
"""

import argparse
import os

from src.evaluate import evaluate_and_plot
from src.utils import get_project_root
from src.utils.config import load_config
from nanou.config import preset_path


def show_info(model_name, config_path):
    """Print parameter counts, layer counts, summary and serialization check."""
    import tensorflow_model_optimization as tfmot
    import tf_keras as keras

    from src.models import (
        PadToMatch,
        count_parameters,
        get_model_config,
        get_model_summary,
        validate_model_serialization,
    )
    from src.utils import BinaryIoU

    config = load_config(config_path)
    root_dir = str(get_project_root())
    models_dir = config['data']['paths']['models_dir']
    if not os.path.isabs(models_dir):
        models_dir = os.path.join(root_dir, models_dir)

    candidates = [os.path.join(models_dir, f"{model_name}{ext}") for ext in (".h5", ".keras")]
    model_path = next((c for c in candidates if os.path.exists(c)), None)
    if model_path is None:
        raise FileNotFoundError(
            f"No Keras model for {model_name!r} in {models_dir}. "
            f"Train one with: python scripts/train.py --config {config_path}"
        )

    # quantize_scope + PadToMatch are both required: a QAT-wrapped checkpoint
    # carries tfmot wrapper layers, and every architecture here carries PadToMatch.
    custom_objects = {"PadToMatch": PadToMatch, "BinaryIoU": BinaryIoU}
    with tfmot.quantization.keras.quantize_scope():
        with keras.utils.custom_object_scope(custom_objects):
            model = keras.models.load_model(model_path, compile=False)

    info = get_model_config(model)
    print(f"\n{'='*55}")
    print(f"MODEL INFO: {model_name}")
    print(f"{'='*55}")
    print(f"   Path:            {model_path}")
    print(f"   Input shape:     {info['input_shape']}")
    print(f"   Parameters:      {count_parameters(model):,}")
    print(f"   Layers:          {info['layer_count']}")
    print(f"   Serializable:    {validate_model_serialization(model)}")

    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")
    if os.path.exists(tflite_path):
        print(f"   INT8 TFLite:     {tflite_path} "
              f"({os.path.getsize(tflite_path) / 1024:.2f} KB)")
    else:
        print("   INT8 TFLite:     not exported - python scripts/export.py")

    print(f"\n{get_model_summary(model)}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model, or report its structure')
    parser.add_argument('model', nargs='?', default='nano_u',
                        help='Model to evaluate (.h5 or .tflite basename without extension)')
    parser.add_argument("--preset", default="tinyagri",
                        help="config preset from config/presets/ (default: tinyagri)")
    parser.add_argument("--config", default=None,
                        help="explicit config file, used instead of --preset")
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help="Split to evaluate (default test; val is what selection may look at)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Operating decision threshold for the headline metrics')
    parser.add_argument('--info', action='store_true',
                        help='Report model structure and parameter counts instead of evaluating')
    args = parser.parse_args()
    config_path = args.config or str(preset_path(args.preset))

    if args.info:
        show_info(args.model, config_path)
        return

    evaluate_and_plot(args.model, config_path, batch_size=8, threshold=args.threshold,
                      samples_to_plot=6, split=args.split)


if __name__ == '__main__':
    main()
