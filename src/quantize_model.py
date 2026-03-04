import tensorflow as tf
import numpy as np
import os
from src.models.utils import convert_to_tflite_quantized
from typing import Optional, Type, Tuple
from src.utils.config import load_config
import cv2

# Import the input shape from the config file globally
_GLOBAL_CONFIG = load_config("config/config.yaml")
INPUT_SHAPE = tuple(_GLOBAL_CONFIG["data"]["input_shape"])

try:
    from src.utils.metrics import BinaryIoU as _BinaryIoU
    _custom_objects: Optional[dict] = {"BinaryIoU": _BinaryIoU}
except ImportError:
    _custom_objects = None
    print("Warning: Could not import BinaryIoU")

def quantize_model(model_path: str, output_path: str, input_shape: Optional[Tuple[int, ...]] = None):
    """
    Load a Keras model and convert it to a quantized TFLite model.
    """
    if input_shape is None:
        input_shape = (1, *INPUT_SHAPE)
        
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return False

    print(f"Loading {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=_custom_objects)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False
    
    config = load_config()
        
    val_img_dir = config.get("data", {}).get("paths", {}).get("processed", {}).get("val", {}).get("img", "")
    mean = np.array(config.get("data", {}).get("normalization", {}).get("mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.array(config.get("data", {}).get("normalization", {}).get("std", [0.5, 0.5, 0.5]), dtype=np.float32)

    img_files = []
    if val_img_dir and os.path.isdir(val_img_dir):
        img_files = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith(('.png', '.jpg'))]
        img_files = sorted(img_files)[:100]  # Use up to 100 samples

    def representative_data_gen():
        shape_to_gen = list(input_shape)
        if len(shape_to_gen) == 3:
            shape_to_gen = [1] + shape_to_gen
            
        if img_files:
            for img_path in img_files:
                img = cv2.imread(img_path)
                if img is not None:
                    img = img.astype(np.float32)[:, :, ::-1] / 255.0
                    if img.shape != tuple(shape_to_gen[1:]):
                        img = cv2.resize(img, (shape_to_gen[2], shape_to_gen[1]))
                    img = (img - mean) / std
                    img = np.expand_dims(img, axis=0)
                    yield [img]
        else:
            # Fallback to random data if no validation data is found
            print("Warning: No validation data found. Using random noise for quantization.")
            for _ in range(100):
                # Generate base 0-1 random noise and shift to expected normalized range
                data = np.random.rand(*shape_to_gen).astype(np.float32)
                data = (data - mean) / std
                yield [data]

    print(f"Quantizing model to {output_path}...")
    return convert_to_tflite_quantized(model, output_path, representative_data_gen)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        model_path = sys.argv[1]
        output_path = sys.argv[2]
        quantize_model(model_path, output_path)
    else:
        # verification/manual run
        quantize_model("models/nano_u.keras", "models/nano_u.tflite")
