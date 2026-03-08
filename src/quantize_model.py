import os
import sys
import tensorflow as tf
import tf_keras as keras
import numpy as np
import tensorflow_model_optimization as tfmot
import json

# Allow running the script directly (python src/quantize_model.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.utils import convert_to_tflite_quantized
from typing import Optional, Type, Tuple
from src.utils.config import load_config
import cv2

# Import the input shape from the config file globally
_GLOBAL_CONFIG = load_config("config/config.yaml")
INPUT_SHAPE = tuple(_GLOBAL_CONFIG["data"]["input_shape"])

try:
    from src.utils.metrics import BinaryIoU as _BinaryIoU
    _custom_objects = {"BinaryIoU": _BinaryIoU}
except ImportError:
    _BinaryIoU = None
    _custom_objects = {}
    print("Warning: Could not import BinaryIoU")


def extract_quant_params(tflite_path: str) -> dict:
    """
    Inspect a quantized TFLite model and return its input/output quantization
    parameters (scale and zero_point). These are needed by the on-device
    inference code to pre/post-process tensors correctly.
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
        zeros  = rich.get("zero_points", [])
        if len(scales) > 0 and len(zeros) > 0:
            return float(scales[0]), int(zeros[0])
        # Legacy: (scale, zero_point) tuple
        if legacy and legacy != (0.0, 0):
            return float(legacy[0]), int(legacy[1])
        return None, None

    in_scale,  in_zero  = _parse(in_quant,  input_details.get("quantization"))
    out_scale, out_zero = _parse(out_quant, output_details.get("quantization"))

    params = {
        "input": {
            "dtype":      input_details["dtype"].__name__,
            "scale":      in_scale,
            "zero_point": in_zero,
            "shape":      [int(s) for s in input_details["shape"]],
        },
        "output": {
            "dtype":      output_details["dtype"].__name__,
            "scale":      out_scale,
            "zero_point": out_zero,
            "shape":      [int(s) for s in output_details["shape"]],
        },
        "normalization": {
            "mean": list(load_config().get("data", {}).get("normalization", {}).get("mean", [0.5, 0.5, 0.5])),
            "std":  list(load_config().get("data", {}).get("normalization", {}).get("std", [0.5, 0.5, 0.5])),
        }
    }
    return params


def quantize_model(model_path: str, output_path: str, input_shape: Optional[Tuple[int, ...]] = None):
    """
    Load a Keras model and convert it to a quantized TFLite model.
    After conversion, extract the actual quantization parameters from the
    produced .tflite and save them to a companion JSON file so that the
    on-device Rust code can use correct (non-hardcoded) values.
    """
    if input_shape is None:
        input_shape = (1, *INPUT_SHAPE)
        
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return False

    try:
        # 1. Custom objects for metrics
        custom_objects = {"BinaryIoU": _BinaryIoU} if _BinaryIoU else {}
        
        # 2. Add tfmot objects to custom_objects for QAT models
        with tfmot.quantization.keras.quantize_scope():
            with keras.utils.custom_object_scope(custom_objects):
                model = keras.models.load_model(model_path, compile=False)
        
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    config = load_config()
    
    # Use secondary dataset if quantizing nano_u2
    is_secondary = "nano_u2" in model_path
    data_paths = config.get("data", {}).get("paths", {})
    data_cfg = data_paths.get("secondary" if is_secondary else "processed", {})
    
    if is_secondary:
        train_img_dir = data_cfg.get("train", {}).get("img", "")
    else:
        train_img_dir = data_cfg.get("train", {}).get("img", "")
        
    mean = np.array(config.get("data", {}).get("normalization", {}).get("mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.array(config.get("data", {}).get("normalization", {}).get("std", [0.5, 0.5, 0.5]), dtype=np.float32)

    img_files = []
    if train_img_dir and os.path.isdir(train_img_dir):
        img_files = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.png')]
        img_files = sorted(img_files)[:200]  # Use up to 200 samples from train set

    def representative_data_gen():
        shape_to_gen = list(input_shape)
        if len(shape_to_gen) == 3:
            shape_to_gen = [1] + shape_to_gen
            
        if img_files:
            print(f"Calibration using {len(img_files)} images from {train_img_dir}")
            for img_path in img_files:
                img = cv2.imread(img_path)
                if img is not None:
                    img = img.astype(np.float32)[:, :, ::-1] / 255.0  # BGR→RGB, 0-1
                    h, w = shape_to_gen[1], shape_to_gen[2]
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(img, (w, h))  # cv2 takes (width, height)
                    img = (img - mean) / std
                    img = np.expand_dims(img, axis=0)
                    yield [img]
        else:
            # Fallback to random data if no training data is found
            print("Warning: No training data found. Using random noise for quantization.")
            for _ in range(100):
                data = np.random.rand(*shape_to_gen).astype(np.float32)
                data = (data - mean) / std
                yield [data]

    print(f"Quantizing model {model_path} to {output_path}...")
    success = convert_to_tflite_quantized(model, output_path, representative_data_gen)

    if success:
        # ── Extract actual quant params from the produced .tflite ────────────
        params = extract_quant_params(output_path)

        params_path = os.path.splitext(output_path)[0] + "_quant_params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

        print(f"\nQuantization parameters extracted and saved to {params_path}")
        print(f"   input  → scale={params['input']['scale']}, zero_point={params['input']['zero_point']}")
        print(f"   output → scale={params['output']['scale']}, zero_point={params['output']['zero_point']}")
        print(f"\nThese values will be picked up by esp_flash/build.rs automatically.")

    return success


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        model_path = sys.argv[1]
        output_path = sys.argv[2]
        quantize_model(model_path, output_path)
    elif len(sys.argv) > 1 and sys.argv[1] == "nano_u2":
        quantize_model("models/nano_u2.h5", "models/nano_u2.tflite")
    else:
        # verification/manual run
        quantize_model("models/nano_u.h5", "models/nano_u.tflite")
