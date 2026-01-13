#!/usr/bin/env python3
"""
Tiny dummy segmentation model for allocation measurement.

Model: 48x48x3 input -> small conv stack -> 48x48x1 output.
Designed to be minimal and export an INT8 TFLite file. Not intended
for accuracy — only to measure memory / allocation behavior.
"""

import os
import numpy as np
import tensorflow as tf

IMG_H, IMG_W, IMG_C = 48, 48, 3

def simple_conv_block(x, out_channels):
    x = tf.keras.layers.Conv2D(out_channels, (3, 3), padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    return x

def build_tiny_segmentation_model():
    inputs = tf.keras.Input(shape=(IMG_H, IMG_W, IMG_C), name="input")
    
    # Simple encoder-decoder without skip connections
    
    # Encoder
    # Level 1
    conv1 = simple_conv_block(inputs, 12)
    pool1 = tf.keras.layers.AveragePooling2D((2, 2))(conv1)
    
    # Level 2
    conv2 = simple_conv_block(pool1, 48)
    pool2 = tf.keras.layers.AveragePooling2D((2, 2))(conv2)
    
    # Level 3 (Bottleneck)
    conv3 = simple_conv_block(pool2, 64)
    
    # Decoder
    # Level 2
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(conv3)
    conv4 = simple_conv_block(up2, 48)
    
    # Level 1
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(conv4)
    conv5 = simple_conv_block(up1, 12)
    
    # Output
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", use_bias=True, name="output")(conv5)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def representative_dataset_gen():
    # Small representative set; values are floats in [0,255]
    for _ in range(50):
        data = np.random.randint(0, 255, size=(1, IMG_H, IMG_W, IMG_C)).astype(np.float32)
        yield [data]


def to_tflite_int8(model: tf.keras.Model, out_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved int8 TFLite to {out_path}")


def main():
    model = build_tiny_segmentation_model()
    # Lightweight initialization (no real training required)
    x = np.random.randn(16, IMG_H, IMG_W, IMG_C).astype(np.float32)
    y = np.random.randn(16, IMG_H, IMG_W, 1).astype(np.float32)
    # A short epoch to initialize weights; conversion uses the weights only.
    model.fit(x, y, epochs=1, batch_size=8, verbose=0)

    out_dir = os.path.dirname(__file__)
    # Small model output name
    out_path = os.path.join(out_dir, "models/dummy5.tflite")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    to_tflite_int8(model, out_path)


if __name__ == "__main__":
    main()
