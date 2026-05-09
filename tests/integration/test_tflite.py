"""Integration tests for TFLite export compatibility.

Verifies that the Nano-U model converts cleanly to TFLite and only uses
operations supported by the microflow-rs firmware inference engine.
"""

import tensorflow as tf
import pytest

from src.models import create_nano_u

# Ops supported by microflow-rs (based on firmware source analysis)
MICROFLOW_SUPPORTED_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "RELU",
    "RELU6",
    "MAX_POOL_2D",
    "AVERAGE_POOL_2D",
    "FULLY_CONNECTED",
    "RESHAPE",
    "SOFTMAX",
    "RESIZE_BILINEAR",
    "RESIZE_NEAREST_NEIGHBOR",  # UpSampling2D(interpolation='nearest')
    "DELEGATE",
}


def _tflite_ops(model_content: bytes) -> set:
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    return {op["op_name"] for op in interpreter._get_ops_details()}


@pytest.fixture(scope="module")
def nano_u_tflite():
    model = create_nano_u(input_shape=(60, 80, 3), filters=[4, 8, 16], bottleneck=16)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def test_tflite_conversion_succeeds(nano_u_tflite):
    assert len(nano_u_tflite) > 0


def test_tflite_only_supported_ops(nano_u_tflite):
    """Nano-U must not introduce ops unsupported by microflow-rs."""
    ops_used = _tflite_ops(nano_u_tflite)
    unsupported = ops_used - MICROFLOW_SUPPORTED_OPS
    assert not unsupported, f"Nano-U uses unsupported ops: {unsupported}"


def test_tflite_inference_runs(nano_u_tflite):
    """Converted model must produce a (1, 60, 80, 1) output tensor."""
    import numpy as np

    interpreter = tf.lite.Interpreter(model_content=nano_u_tflite)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    interpreter.set_tensor(inp["index"], np.zeros(inp["shape"], dtype=np.float32))
    interpreter.invoke()
    result = interpreter.get_tensor(out["index"])

    assert result.shape == (1, 60, 80, 1)
    assert result.dtype == np.float32
