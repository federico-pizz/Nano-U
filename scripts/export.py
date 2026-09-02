"""Convert a trained Keras model to a calibrated INT8 TFLite file.

Also writes the companion <model>_quant_params.json, which firmware/build.rs
reads at compile time to bake the input/output scales, zero-points and the
normalization mean/std into the binary. A firmware build without this file
cannot produce correct on-device preprocessing.

--config must be the SAME config the model was trained with: it supplies the
calibration image directory, the input shape and the normalization constants.

Usage:
    python scripts/export.py
    python scripts/export.py models/TinyAgri/nano_u.h5 models/TinyAgri/nano_u.tflite \
        --config config/TinyAgri_config.yaml
"""

import argparse
import sys

from src.quantize_model import quantize_model


def main():
    parser = argparse.ArgumentParser(description="Export a Keras model to INT8 TFLite")
    parser.add_argument("model_path", nargs="?", default="models/nano_u.h5")
    parser.add_argument("output_path", nargs="?", default="models/nano_u.tflite")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    ok = quantize_model(args.model_path, args.output_path, config_path=args.config)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
