"""Model builder functions for Nano-U architectures."""

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model
from typing import List, Tuple, Optional, Dict, Callable

import sys
import os

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from .layers import depthwise_sep_conv, triple_conv, double_conv, bottleneck_dw_conv
from src.utils.config import load_config

_GLOBAL_CONFIG = load_config("config/config.yaml")
_GLOBAL_INPUT_SHAPE = tuple(_GLOBAL_CONFIG.get("data", {}).get("input_shape", [120, 160, 3]))


@tf.keras.utils.register_keras_serializable(package="NanoU")
class PadToMatch(layers.Layer):
    """Center-crop the skip tensor to match the spatial dims of the upsampled tensor.

    Cropping the (larger) encoder skip instead of zero-padding the upsampled
    decoder feature avoids introducing synthetic border values and keeps the
    decoder feature map spatially aligned with the encoder features.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x_up, skip = inputs[0], inputs[1]
        x_h = tf.shape(x_up)[1]
        x_w = tf.shape(x_up)[2]
        crop_top  = (tf.shape(skip)[1] - x_h) // 2
        crop_left = (tf.shape(skip)[2] - x_w) // 2
        return skip[:, crop_top:crop_top + x_h, crop_left:crop_left + x_w, :]

def create_nano_u(
    input_shape: Tuple[int, int, int] = _GLOBAL_INPUT_SHAPE,
    filters: Optional[List[int]] = None,
    bottleneck: Optional[int] = None,
    name: str = 'nano_u',
) -> Model:
    """Strictly sequential Nano-U encoder-decoder (no skip connections).

    Kept parameter-minimal by design; skip connections are omitted to reduce
    parameter count and avoid overfitting on small agricultural datasets.
    TFLite export is trivial — MCU-inference details live in ``firmware/``.
    """
    inputs = layers.Input(shape=input_shape, name='input_image')

    if filters is None:
        filters = list(_GLOBAL_CONFIG.get("models", {}).get("nano_u", {}).get("filters", [16, 32, 64]))
    if bottleneck is None:
        bottleneck = int(_GLOBAL_CONFIG.get("models", {}).get("nano_u", {}).get("bottleneck", 64))

    x = depthwise_sep_conv(inputs, filters[0], name='enc1a')
    x = depthwise_sep_conv(x, filters[0], name='enc1b')
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = depthwise_sep_conv(x, filters[1], name='enc2a')
    x = depthwise_sep_conv(x, filters[1], name='enc2b')
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = depthwise_sep_conv(x, filters[2], name='enc3a')
    x = depthwise_sep_conv(x, filters[2], name='enc3b')
    x = layers.MaxPooling2D(pool_size=(3, 2))(x)
    x = depthwise_sep_conv(x, bottleneck, name='bottleneck_a')
    x = depthwise_sep_conv(x, bottleneck, name='bottleneck_b')
    x = layers.UpSampling2D(size=(3, 2), interpolation='nearest', name='dec1_up')(x)
    x = depthwise_sep_conv(x, filters[2], name='dec1_conv_a')
    x = depthwise_sep_conv(x, filters[2], name='dec1_conv_b')
    x = layers.UpSampling2D(size=2, interpolation='nearest', name='dec2_up')(x)
    x = depthwise_sep_conv(x, filters[1], name='dec2_conv_a')
    x = depthwise_sep_conv(x, filters[1], name='dec2_conv_b')
    x = layers.UpSampling2D(size=2, interpolation='nearest', name='dec3_up')(x)
    x = depthwise_sep_conv(x, filters[0], name='dec3_conv_a')
    x = depthwise_sep_conv(x, filters[0], name='dec3_conv_b')
    outputs = layers.Conv2D(1, 1, padding='same', activation='linear', name='output')(x)

    return Model(inputs=inputs, outputs=outputs, name=name)


def create_bu_net(
    input_shape: Tuple[int, int, int] = _GLOBAL_INPUT_SHAPE,
    filters: Optional[List[int]] = None,
    bottleneck: Optional[int] = None,
    name: str = 'bu_net',
) -> Model:
    """BU-Net teacher model (U-Net with skip connections) using Functional API."""
    
    if filters is None:
        filters = list(_GLOBAL_CONFIG.get("models", {}).get("bu_net", {}).get("filters", [64, 128, 256, 512, 1024, 2048]))
    if bottleneck is None:
        bottleneck = int(_GLOBAL_CONFIG.get("models", {}).get("bu_net", {}).get("bottleneck", 2048))
        
    inputs = layers.Input(shape=input_shape, name='input_image')
    num_stages = len(filters)

    encoder_outputs: List[tf.Tensor] = []
    x = inputs

    x = triple_conv(x, filters[0], name='enc1')
    encoder_outputs.append(x)
    x = layers.MaxPooling2D()(x)

    for i in range(1, num_stages):
        x = triple_conv(x, filters[i], name=f'enc{i + 1}')
        encoder_outputs.append(x)
        if i < num_stages - 1:
            x = layers.MaxPooling2D()(x)

    x = triple_conv(layers.MaxPooling2D()(x), bottleneck, name='bottleneck')

    def up_block(x_in, skip, out_filters, block_name):
        x_in = layers.UpSampling2D(size=2, interpolation='nearest', name=f'{block_name}_up')(x_in)
        skip_cropped = PadToMatch(name=f'{block_name}_pad')([x_in, skip])
        x_in = layers.Concatenate(name=f'{block_name}_concat')([skip_cropped, x_in])
        x_in = triple_conv(x_in, out_filters, name=f'{block_name}_conv')
        return x_in

    for i in range(num_stages - 1, -1, -1):
        x = up_block(x, encoder_outputs[i], filters[i], f'dec{num_stages - i}')

    outputs = layers.Conv2D(1, 1, activation='linear', padding='same', name='output_segmentation')(x)
    h, w = input_shape[0], input_shape[1]
    outputs = layers.Resizing(h, w, name='output_resize')(outputs)
    return Model(inputs=inputs, outputs=outputs, name=name)
