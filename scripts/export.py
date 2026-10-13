"""Convert a trained Keras model to a calibrated INT8 TFLite file.

Also writes the companion <model>_quant_params.json, which firmware/build.rs
reads at compile time to bake the input/output scales, zero-points and the
normalization mean/std into the binary. A firmware build without this file
cannot produce correct on-device preprocessing.

--config must be the SAME config the model was trained with: it supplies the
calibration image directory, the input shape and the normalization constants.
Without paths, the model is <models_dir>/nanou.h5 from that config and the
output goes next to it.

Usage:
    python scripts/export.py
    python scripts/export.py --preset botanicgarden
    python scripts/export.py runs/nanou.h5 runs/nanou.tflite --preset tinyagri
"""

import argparse
from pathlib import Path

from nanou.config import preset_path, resolve
from nanou.quantize import quantize_model


def main():
    parser = argparse.ArgumentParser(description="Export a Keras model to INT8 TFLite")
    parser.add_argument(
        "model_path", nargs="?", default=None, help="default: <models_dir>/nanou.h5"
    )
    parser.add_argument(
        "output_path", nargs="?", default=None, help="default: the model path with .tflite"
    )
    parser.add_argument(
        "--preset",
        default="tinyagri",
        help="config preset from config/presets/ (default: tinyagri)",
    )
    parser.add_argument(
        "--config", default=None, help="explicit config file, used instead of --preset"
    )
    args = parser.parse_args()

    config_path = args.config or str(preset_path(args.preset))
    models_dir = Path(resolve(path=config_path).models_dir or "models")
    model_path = Path(args.model_path or models_dir / "nanou.h5")
    output_path = args.output_path or str(model_path.with_suffix(".tflite"))

    if not model_path.exists():
        raise SystemExit(
            f"\nNo trained model at {model_path}\n\n"
            f"  Fix it with:  python scripts/train.py --preset {args.preset}\n"
        )

    quantize_model(str(model_path), output_path, config_path=config_path)


if __name__ == "__main__":
    main()
