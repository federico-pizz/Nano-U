import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot

inp = keras.layers.Input((10, 10, 3))
x = keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False, activation='linear')(inp)
out = keras.layers.BatchNormalization()(x)
model = keras.Model(inp, out)

qat_model = tfmot.quantization.keras.quantize_model(model)

for layer in qat_model.layers:
    print(layer.name, layer.__class__.__name__)
