"""Unit tests for src/models/layers.py — output shape contracts."""

import numpy as np
import pytest
import tensorflow as tf
import tf_keras as keras

from src.models.layers import (
    depthwise_sep_conv,
    triple_conv,
    double_conv,
    bottleneck_dw_conv,
)

H, W = 60, 80


def _build(fn, input_shape, **kwargs):
    """Wrap a layer function in a Keras functional model and run a forward pass."""
    inp = keras.Input(shape=input_shape)
    out = fn(inp, **kwargs)
    model = keras.Model(inputs=inp, outputs=out)
    x = np.zeros((1,) + input_shape, dtype=np.float32)
    return model(x, training=False)


# ── depthwise_sep_conv ────────────────────────────────────────────────────────

def test_depthwise_sep_conv_output_shape():
    out = _build(depthwise_sep_conv, (H, W, 3), filters=16)
    assert out.shape == (1, H, W, 16)


def test_depthwise_sep_conv_stride2_halves_spatial():
    out = _build(depthwise_sep_conv, (H, W, 3), filters=16, stride=2)
    assert out.shape == (1, H // 2, W // 2, 16)


def test_depthwise_sep_conv_output_dtype():
    out = _build(depthwise_sep_conv, (H, W, 3), filters=8)
    assert out.dtype == tf.float32


# ── triple_conv ───────────────────────────────────────────────────────────────

def test_triple_conv_output_shape():
    out = _build(triple_conv, (H, W, 8), out_channels=16)
    assert out.shape == (1, H, W, 16)


def test_triple_conv_preserves_spatial():
    out = _build(triple_conv, (30, 40, 4), out_channels=8)
    assert out.shape[1:3] == (30, 40)


# ── double_conv ───────────────────────────────────────────────────────────────

def test_double_conv_output_shape():
    out = _build(double_conv, (H, W, 4), out_channels=8)
    assert out.shape == (1, H, W, 8)


def test_double_conv_mid_channels():
    out = _build(double_conv, (H, W, 8), out_channels=16, mid_channels=4)
    assert out.shape == (1, H, W, 16)


# ── bottleneck_dw_conv ────────────────────────────────────────────────────────

def test_bottleneck_dw_conv_output_shape():
    out = _build(bottleneck_dw_conv, (H, W, 8), filters=16)
    assert out.shape == (1, H, W, 16)


def test_bottleneck_dw_conv_skip_connection_when_channels_match():
    """When in_channels == filters the skip add must not change the shape."""
    out = _build(bottleneck_dw_conv, (H, W, 16), filters=16)
    assert out.shape == (1, H, W, 16)


def test_bottleneck_dw_conv_no_skip_when_channels_differ():
    """When in_channels != filters there is no residual add, but shape is still correct."""
    out = _build(bottleneck_dw_conv, (H, W, 8), filters=16)
    assert out.shape == (1, H, W, 16)
