import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_transforms

t = default_8bit_transforms.Conv2DBatchNormQuantize()
print("Pattern:", t.pattern())
print("Expected Conv2D class:", t.pattern()._nodes[0].class_name)
