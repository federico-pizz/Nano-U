"""Layer definitions for Nano-U models."""

import tensorflow as tf
import tf_keras as keras
layers = keras.layers
from typing import Optional


def depthwise_sep_conv(x: tf.Tensor, filters: int, stride: int = 1, name: Optional[str] = None) -> tf.Tensor:
    """Depthwise separable convolution block with batch normalization and ReLU.
    
    Args:
        x: Input tensor
        filters: Number of output filters
        stride: Stride for depthwise convolution
        name: Optional name prefix for layers
    
    Returns:
        Output tensor after depthwise separable convolution
    """
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=stride, padding='same', use_bias=False,
        name=f'{name}_dw' if name else None
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn1' if name else None)(x)
    x = layers.ReLU(name=f'{name}_relu1' if name else None)(x)
    
    x = layers.Conv2D(
        filters, kernel_size=1, strides=1, padding='same', use_bias=False,
        name=f'{name}_pw' if name else None
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn2' if name else None)(x)
    x = layers.ReLU(name=f'{name}_relu2' if name else None)(x)
    
    return x


def triple_conv(x: tf.Tensor, out_channels: int,
                mid_channels: Optional[int] = None, name: Optional[str] = None) -> tf.Tensor:
    """Three depthwise separable convolution blocks with batch normalization.

    Args:
        x: Input tensor
        out_channels: Number of output channels
        mid_channels: Number of channels in middle blocks (default: out_channels)
        name: Optional name prefix for layers

    Returns:
        Output tensor after triple convolution
    """
    if mid_channels is None:
        mid_channels = out_channels
    
    # Block 1
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=1, padding='same', use_bias=False,
        name=f'{name}_dw1' if name else None
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn1a' if name else None)(x)
    x = layers.Conv2D(
        mid_channels, kernel_size=1, strides=1, padding='same', use_bias=False,
        name=f'{name}_pw1' if name else None
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn1b' if name else None)(x)
    x = layers.ReLU(name=f'{name}_relu1' if name else None)(x)
    
    # Block 2
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=1, padding='same', use_bias=False,
        name=f'{name}_dw2' if name else None
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn2a' if name else None)(x)
    x = layers.Conv2D(
        mid_channels, kernel_size=1, strides=1, padding='same', use_bias=False,
        name=f'{name}_pw2' if name else None
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn2b' if name else None)(x)
    x = layers.ReLU(name=f'{name}_relu2' if name else None)(x)
    
    # Block 3
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=1, padding='same', use_bias=False,
        name=f'{name}_dw3' if name else None
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn3a' if name else None)(x)
    x = layers.Conv2D(
        out_channels, kernel_size=1, strides=1, padding='same', use_bias=False,
        name=f'{name}_pw3' if name else None
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn3b' if name else None)(x)
    x = layers.ReLU(name=f'{name}_relu3' if name else None)(x)
    
    return x


def double_conv(x: tf.Tensor, out_channels: int,
                mid_channels: Optional[int] = None, name: Optional[str] = None) -> tf.Tensor:
    """Two depthwise separable convolution blocks with ReLU (no BatchNorm).

    Optimized for microcontroller deployment with reduced parameter overhead.

    Args:
        x: Input tensor
        out_channels: Number of output channels
        mid_channels: Number of channels in middle block (default: out_channels)
        name: Optional name prefix for layers

    Returns:
        Output tensor after double convolution
    """
    if mid_channels is None:
        mid_channels = out_channels
    
    # Pointwise to reduce channels
    x = layers.Conv2D(
        mid_channels, kernel_size=1, strides=1, padding='same', use_bias=True,
        name=f'{name}_pw1' if name else None
    )(x)
    x = layers.ReLU(name=f'{name}_relu1' if name else None)(x)
    
    # Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=1, padding='same', use_bias=True,
        name=f'{name}_dw' if name else None
    )(x)
    x = layers.ReLU(name=f'{name}_relu2' if name else None)(x)
    
    # Pointwise to expand channels
    x = layers.Conv2D(
        out_channels, kernel_size=1, strides=1, padding='same', use_bias=True,
        name=f'{name}_pw2' if name else None
    )(x)
    x = layers.ReLU(name=f'{name}_relu3' if name else None)(x)
    
    return x


def bottleneck_dw_conv(x: tf.Tensor, filters: int, name: Optional[str] = None) -> tf.Tensor:
    """Bottleneck with Depthwise Separable convolution.
    
    Args:
        x: Input tensor
        filters: Number of output filters
        name: Optional name prefix for layers
        
    Returns:
        Output tensor after bottleneck block
    """
    in_channels = x.shape[-1]
    
    # Expansion
    expand_filters = filters * 2
    x_res = layers.Conv2D(expand_filters, 1, use_bias=False, name=f'{name}_expand' if name else None)(x)
    x_res = layers.BatchNormalization(name=f'{name}_bn1' if name else None)(x_res)
    x_res = layers.ReLU(name=f'{name}_relu1' if name else None)(x_res)
    
    # Depthwise
    x_res = layers.DepthwiseConv2D(3, padding='same', use_bias=False, name=f'{name}_dw' if name else None)(x_res)
    x_res = layers.BatchNormalization(name=f'{name}_bn2' if name else None)(x_res)
    x_res = layers.ReLU(name=f'{name}_relu2' if name else None)(x_res)
    
    # Projection
    x_res = layers.Conv2D(filters, 1, use_bias=False, name=f'{name}_proj' if name else None)(x_res)
    x_res = layers.BatchNormalization(name=f'{name}_bn3' if name else None)(x_res)
    
    # Skip connection if shapes match
    if in_channels == filters:
        return layers.Add(name=f'{name}_add' if name else None)([x, x_res])
    return x_res
