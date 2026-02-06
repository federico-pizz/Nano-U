
import tensorflow as tf
import numpy as np
import os
from src.models.utils import convert_to_tflite_quantized
try:
    from src.utils.metrics import BinaryIoU
except ImportError:
    print("Warning: Could not import BinaryIoU")
    BinaryIoU = None

def quantize_model(model_path: str, output_path: str, input_shape=(1, 48, 64, 3)):
    """
    Load a Keras model and convert it to a quantized TFLite model.
    """
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return False

    print(f"Loading {model_path}...")
    try:
        custom_objects = {'BinaryIoU': BinaryIoU} if BinaryIoU else None
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

    # Define representative dataset generator
    def representative_data_gen():
        # TODO: Use real validation data if available
        # Input shape typically includes batch dim in Keras config, but generator expects [1, H, W, C] list
        # We assume input_shape is (1, 48, 64, 3) or (48, 64, 3)
        shape_to_gen = list(input_shape)
        if len(shape_to_gen) == 3:
            shape_to_gen = [1] + shape_to_gen
            
        for _ in range(100):
            data = np.random.rand(*shape_to_gen).astype(np.float32)
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
