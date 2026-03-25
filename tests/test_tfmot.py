import tensorflow_model_optimization as tfmot
import tf_keras as keras
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_transforms
import inspect

# Let's see what class Conv2DBatchNormQuantize expects
transform = default_8bit_transforms.Conv2DBatchNormQuantize()
print(transform.pattern().class_name)

import sys
print(sys.modules['tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_transforms'].keras)

