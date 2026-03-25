import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot

inp = keras.layers.Input((10, 10, 3))
x = keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False)(inp)
model = keras.Model(inp, x)

qat_model = tfmot.quantization.keras.quantize_model(model)
wrapper = qat_model.layers[2]  # depthwise_conv2d
print("Weights:", wrapper.weights)

@tf.function
def train_step(x):
    qat_model(x, training=True)

x = tf.random.normal((32, 10, 10, 3))
train_step(x)

# Print min/max to see if they updated
for w in wrapper.weights:
    if 'min' in w.name or 'max' in w.name:
        print(w.name, w.numpy())

