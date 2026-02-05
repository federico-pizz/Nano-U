import tensorflow as tf
import pytest
import numpy as np
import os
from src.models import create_nano_u

# List of operations supported by microflow-rs based on source analysis
MICROFLOW_SUPPORTED_OPS = {
    'CONV_2D',
    'DEPTHWISE_CONV_2D',
    'RELU',
    'RELU6',
    'AVERAGE_POOL_2D',
    'FULLY_CONNECTED',
    'RESHAPE',
    'SOFTMAX',
    'RESIZE_BILINEAR',
    'DELEGATE'
}

def get_tflite_ops(tflite_model_content):
    """Extract list of operations from a TFLite model."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
    interpreter.allocate_tensors()
    
    # Get all op names used in the model
    op_details = interpreter._get_ops_details()
    ops_used = {op['op_name'] for op in op_details}
    return ops_used

def test_nanou_tflite_compatibility():
    """Check if the Nano-U model is compatible with microflow-rs."""
    model = create_nano_u(input_shape=(48, 48, 3))
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    ops_used = get_tflite_ops(tflite_model)
    print(f"\nOps used in Nano-U: {ops_used}")
    
    unsupported = ops_used - MICROFLOW_SUPPORTED_OPS
    
    if unsupported:
        print(f"Unsupported ops found: {unsupported}")
    
    assert not unsupported, f"Nano-U uses unsupported ops: {unsupported}"

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
