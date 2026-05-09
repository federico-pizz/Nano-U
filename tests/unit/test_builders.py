"""Unit tests for src/models/builders.py — create_nano_u and create_bu_net."""

import tensorflow as tf
import pytest

from src.models.builders import create_nano_u, create_bu_net

H, W = 60, 80

# MCU flash budget: Nano-U must stay well under 100 KB of parameters
# (each param is 4 bytes float32 → 25 000 params ≈ 100 KB)
NANO_U_PARAM_BUDGET = 25_000


# ── create_nano_u ─────────────────────────────────────────────────────────────

def test_nano_u_output_shape():
    m = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    out = m(tf.random.normal((1, H, W, 3)), training=False)
    assert out.shape == (1, H, W, 1)


def test_nano_u_output_shape_batch():
    m = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    out = m(tf.random.normal((3, H, W, 3)), training=False)
    assert out.shape == (3, H, W, 1)


def test_nano_u_output_dtype():
    m = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    out = m(tf.random.normal((1, H, W, 3)), training=False)
    assert out.dtype == tf.float32


def test_nano_u_param_count_within_mcu_budget():
    """Production config must stay under the MCU flash budget."""
    m = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    params = m.count_params()
    assert params < NANO_U_PARAM_BUDGET, (
        f"Nano-U has {params} params — exceeds MCU budget of {NANO_U_PARAM_BUDGET}"
    )


def test_nano_u_param_count_stable():
    """Parameter count must not change between builds (catches accidental layer additions)."""
    m = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    assert m.count_params() == 4212


def test_nano_u_default_config_matches_expected():
    """create_nano_u() with no args must use the production config (60×80, 4212 params)."""
    m = create_nano_u()
    assert m.input_shape == (None, H, W, 3)
    assert m.count_params() == 4212


# ── create_bu_net ─────────────────────────────────────────────────────────────

def test_bu_net_output_shape():
    m = create_bu_net(input_shape=(H, W, 3), filters=[16, 32, 64], bottleneck=64)
    out = m(tf.random.normal((1, H, W, 3)), training=False)
    assert out.shape == (1, H, W, 1)


def test_bu_net_output_dtype():
    m = create_bu_net(input_shape=(H, W, 3), filters=[16, 32, 64], bottleneck=64)
    out = m(tf.random.normal((1, H, W, 3)), training=False)
    assert out.dtype == tf.float32


def test_bu_net_larger_than_nano_u():
    """BU-Net (teacher) must have more parameters than Nano-U (student) for the same filters."""
    nano = create_nano_u(input_shape=(H, W, 3), filters=[16, 32, 64], bottleneck=64)
    bu = create_bu_net(input_shape=(H, W, 3), filters=[16, 32, 64], bottleneck=64)
    assert bu.count_params() > nano.count_params()
