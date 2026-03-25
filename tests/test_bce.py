import tensorflow as tf
import tf_keras as keras

bce = keras.losses.BinaryCrossentropy(from_logits=True)
y_true = tf.random.uniform((16, 60, 80, 1), 0, 2, dtype=tf.float32)
y_pred = tf.random.normal((16, 60, 80, 1))

loss1 = bce(y_true, y_pred)
print("BCE loss:", loss1.numpy())

mse = tf.reduce_mean(tf.math.squared_difference(tf.nn.sigmoid(y_true), tf.nn.sigmoid(y_pred)))
print("MSE loss:", mse.numpy())

