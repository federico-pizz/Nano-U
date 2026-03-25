import tensorflow as tf
from src.models import create_nano_u
from src.utils.qat import apply_qat_to_model
import numpy as np

model = create_nano_u(input_shape=(60, 80, 3))
qat_model = apply_qat_to_model(model)

# dummy input
def representative_data_gen():
    for _ in range(10):
        yield [np.random.rand(1, 60, 80, 3).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

ops = interpreter._get_ops_details()
print(f"Total ops in TFLite: {len(ops)}")
op_types = {}
for op in ops:
    op_name = op['op_name']
    op_types[op_name] = op_types.get(op_name, 0) + 1

for k, v in op_types.items():
    print(f"{k}: {v}")
