"""INT8 export: Keras model -> quantized .tflite plus the quant-params JSON.

`convert_to_tflite_quantized` lives here rather than beside the architectures
because this is its only caller, and because what it produces - an INT8 graph
with INT8 input and output - is an export decision, not a modelling one.
"""

import json
import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from nanou.architectures import PadToMatch
from nanou.config import load_config
from nanou.metrics import BinaryIoU

_custom_objects = {"BinaryIoU": BinaryIoU}


def convert_to_tflite_quantized(
    model: keras.Model,
    output_path: str,
    representative_data_gen=None,
    allow_select_tf_ops: bool = False,
) -> str:
    """Convert Keras model to Int8 Quantized TFLite model.

    Args:
        model: The Keras model to convert (plain or QAT-annotated).
        output_path: Destination path for the .tflite file.
        representative_data_gen: Calibration generator for full-integer quant.
        allow_select_tf_ops: When True, adds SELECT_TF_OPS as a fallback for
            ops not natively supported in TFLITE_BUILTINS_INT8 mode. Keeps
            the primary ops list as INT8-only so that the microcontroller
            path is still fully quantized.

    Returns:
        ``output_path``. A converter failure propagates as the converter's own
        exception.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_data_gen:
        converter.representative_dataset = representative_data_gen

    # Always enforce full-integer quantization as the primary target.
    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if allow_select_tf_ops:
        # Allow TF Select ops as a fallback (e.g. for any ops the INT8
        # kernel set doesn't cover). This will not affect ops that ARE
        # covered natively.
        supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    converter.target_spec.supported_ops = supported_ops
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Quantized TFLite model saved to {output_path}")
    return output_path


def extract_quant_params(
    tflite_path: str, config_path: str = "config/presets/tinyagri.yaml"
) -> dict:
    """Return a quantized TFLite model's input/output scale and zero_point.

    The firmware bakes these into the binary at compile time, so they are the
    contract between the exported model and the on-device pre/post-processing.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_quant = input_details.get("quantization_parameters", {})
    out_quant = output_details.get("quantization_parameters", {})

    # "quantization_parameters" is the rich dict (scales/zero_points arrays);
    # fall back to the legacy scalar tuple "quantization" if absent.
    def _parse(rich, legacy):
        scales = rich.get("scales", [])
        zeros = rich.get("zero_points", [])
        if len(scales) > 0 and len(zeros) > 0:
            return float(scales[0]), int(zeros[0])
        # Legacy: (scale, zero_point) tuple
        if legacy and legacy != (0.0, 0):
            return float(legacy[0]), int(legacy[1])
        return None, None

    in_scale, in_zero = _parse(in_quant, input_details.get("quantization"))
    out_scale, out_zero = _parse(out_quant, output_details.get("quantization"))

    params = {
        "input": {
            "dtype": input_details["dtype"].__name__,
            "scale": in_scale,
            "zero_point": in_zero,
            "shape": [int(s) for s in input_details["shape"]],
        },
        "output": {
            "dtype": output_details["dtype"].__name__,
            "scale": out_scale,
            "zero_point": out_zero,
            "shape": [int(s) for s in output_details["shape"]],
        },
        "normalization": {
            "mean": list(
                load_config(config_path)
                .get("data", {})
                .get("normalization", {})
                .get("mean", [0.5, 0.5, 0.5])
            ),
            "std": list(
                load_config(config_path)
                .get("data", {})
                .get("normalization", {})
                .get("std", [0.5, 0.5, 0.5])
            ),
        },
    }
    return params


def quantize_model(
    model_path: str,
    output_path: str,
    input_shape: tuple[int, ...] | None = None,
    config_path: str = "config/presets/tinyagri.yaml",
) -> str:
    """Load a Keras model, convert it to a quantized TFLite model, and record its scales.

    The quantization parameters are read back out of the produced .tflite and
    written to a companion JSON file, so the Rust firmware picks up the real
    values rather than hardcoded ones.

    Returns:
        The path of the companion ``_quant_params.json``.

    Raises:
        FileNotFoundError: if ``model_path`` does not exist, or if there are no
            calibration images and noise calibration was not asked for.
        ValueError: if ``quantization.calibration_split`` is not val or train.
    """
    config = load_config(config_path)
    if input_shape is None:
        input_shape = (1, *config.get("data", {}).get("input_shape", (60, 80, 3)))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model to quantize at {model_path}")

    # QAT models carry tfmot wrappers, so they only load inside quantize_scope.
    custom_objects = {"BinaryIoU": BinaryIoU, "PadToMatch": PadToMatch}
    with tfmot.quantization.keras.quantize_scope():
        with keras.utils.custom_object_scope(custom_objects):
            model = keras.models.load_model(model_path, compile=False)
    print(f"Model loaded successfully from {model_path}")

    data_paths = config.get("data", {}).get("paths", {})
    data_cfg = data_paths.get("processed", {})

    # Which split calibrates the INT8 scales and zero points. "val" reproduces
    # the published results, but val also picks the checkpoint, so calibration
    # is not independent of model selection. "train" is the strictly
    # independent choice. The test split is deliberately not accepted.
    calib_split = str(config.get("quantization", {}).get("calibration_split", "val")).lower()
    if calib_split not in ("val", "train"):
        raise ValueError(
            f"quantization.calibration_split must be 'val' or 'train', got {calib_split!r}"
        )
    calib_img_dir = data_cfg.get(calib_split, {}).get("img", "")

    mean = np.array(
        config.get("data", {}).get("normalization", {}).get("mean", [0.5, 0.5, 0.5]),
        dtype=np.float32,
    )
    std = np.array(
        config.get("data", {}).get("normalization", {}).get("std", [0.5, 0.5, 0.5]),
        dtype=np.float32,
    )

    img_files = []
    if calib_img_dir and os.path.isdir(calib_img_dir):
        img_files = [
            os.path.join(calib_img_dir, f) for f in os.listdir(calib_img_dir) if f.endswith(".png")
        ]
        img_files = sorted(img_files)  # Use the entire split for calibration

    # A noise-calibrated INT8 model loads, runs, and is quietly worthless: the
    # activation ranges bear no relation to real images. Fail loudly instead,
    # unless a noise run is explicitly requested.
    allow_noise = bool(config.get("quantization", {}).get("allow_noise_calibration", False))
    if not img_files and not allow_noise:
        raise FileNotFoundError(
            f"No calibration images found for the '{calib_split}' split "
            f"(looked in {calib_img_dir!r}). Check that config_path points at the "
            "config the model was trained with. Set "
            "quantization.allow_noise_calibration: true to calibrate on random "
            "noise anyway (debug only - the resulting INT8 model is not usable)."
        )

    def representative_data_gen():
        shape_to_gen = list(input_shape)
        if len(shape_to_gen) == 3:
            shape_to_gen = [1] + shape_to_gen

        if img_files:
            print(
                f"Calibration using {len(img_files)} images from {calib_img_dir} "
                f"({calib_split} split)"
            )
            for img_path in img_files:
                img = cv2.imread(img_path)
                if img is not None:
                    img = img.astype(np.float32)[:, :, ::-1] / 255.0  # BGR->RGB, 0-1
                    h, w = shape_to_gen[1], shape_to_gen[2]
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(img, (w, h))  # cv2 takes (width, height)
                    img = (img - mean) / std
                    img = np.expand_dims(img, axis=0)
                    yield [img]
        else:
            # Only reachable with quantization.allow_noise_calibration: true.
            print(
                "Warning: No calibration data found. Using random noise - "
                "the resulting INT8 model is for debugging only."
            )
            for _ in range(100):
                data = np.random.rand(*shape_to_gen).astype(np.float32)
                data = (data - mean) / std
                yield [data]

    print(f"Quantizing model {model_path} to {output_path}...")
    convert_to_tflite_quantized(model, output_path, representative_data_gen)

    # Extract actual quant params from the produced .tflite and persist them
    # for the Rust build script.
    params = extract_quant_params(output_path, config_path)

    params_path = os.path.splitext(output_path)[0] + "_quant_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(
        f"\n[QUANT] Saved quantization metadata to {params_path} "
        f"(input_scale={params['input']['scale']}, output_scale={params['output']['scale']})"
    )

    return params_path
