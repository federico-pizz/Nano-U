import tensorflow as tf
import tf_keras as keras

in_layer = keras.layers.Input((10, 10, 3))
x = keras.layers.Conv2D(3, 3)(in_layer)
bn = keras.layers.BatchNormalization()
out = bn(x)
model = keras.Model(in_layer, out)

moving_mean = bn.moving_mean

print("Before:", moving_mean.numpy()[0])

@tf.function
def train_step(x):
    model(x, training=True)

x = tf.random.normal((32, 10, 10, 3))
train_step(x)

print("After:", moving_mean.numpy()[0])
