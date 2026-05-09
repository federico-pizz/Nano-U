"""Utility functions for model operations."""

import tensorflow as tf
import tf_keras as keras
Model = keras.Model
import io
import tempfile
import os


def count_parameters(model: Model) -> int:
    """Return total trainable parameter count."""
    return int(model.count_params())


def get_model_summary(model: Model) -> str:
    """Return model summary as string (captures model.summary())."""
    buf = io.StringIO()
    model.summary(print_fn=lambda s: buf.write(s + "\n"))
    return buf.getvalue()


def get_model_config(model: Model) -> dict:
    """Return dict with model_name, input_shape, parameter_count, layer_count."""
    input_shape = list(model.input_shape[1:]) if model.input_shape else []
    return {
        "model_name": model.name,
        "input_shape": input_shape,
        "parameter_count": count_parameters(model),
        "layer_count": len(model.layers),
    }


def validate_model_serialization(model: Model) -> bool:
    """Save and load model; return True if successful."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        model.save(path, save_format="h5")
        loaded = keras.models.load_model(path)
        return loaded is not None and loaded.count_params() == model.count_params()
    except Exception:
        return False
    finally:
        if os.path.exists(path):
            os.unlink(path)


def convert_to_tflite_quantized(
    model: Model,
    output_path: str,
    representative_data_gen=None,
    allow_select_tf_ops: bool = False,
) -> bool:
    """
    Convert Keras model to Int8 Quantized TFLite model.

    Args:
        model: The Keras model to convert (plain or QAT-annotated).
        output_path: Destination path for the .tflite file.
        representative_data_gen: Calibration generator for full-integer quant.
        allow_select_tf_ops: When True, adds SELECT_TF_OPS as a fallback for
            ops not natively supported in TFLITE_BUILTINS_INT8 mode. Keeps
            the primary ops list as INT8-only so that the microcontroller
            path is still fully quantized.
    """
    try:
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
        return True
    except Exception as e:
        print(f"Quantization failed: {e}")
        return False
