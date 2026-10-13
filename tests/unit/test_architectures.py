"""Unit tests for nanou.architectures - create_nano_u and create_bu_net."""

import pytest
import tensorflow as tf

from nanou.architectures import create_bu_net, create_nano_u

H, W = 60, 80

# MCU flash budget: Nano-U must stay well under 100 KB of parameters
# (each param is 4 bytes float32 -> 25 000 params ~ 100 KB)
NANO_U_PARAM_BUDGET = 25_000


# -- create_nano_u -------------------------------------------------------------


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
        f"Nano-U has {params} params - exceeds MCU budget of {NANO_U_PARAM_BUDGET}"
    )


def test_nano_u_param_count_stable():
    """Parameter count must not change between builds (catches accidental layer additions)."""
    m = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    assert m.count_params() == 4212


def test_nano_u_production_config_matches_expected():
    """The shipped shape (60x80, [4,8,16], bottleneck 16) must give 4212 params."""
    m = create_nano_u(input_shape=(60, 80, 3), filters=[4, 8, 16], bottleneck=16)
    assert m.input_shape == (None, 60, 80, 3)
    assert m.count_params() == 4212


def test_nano_u_requires_explicit_shape():
    """The builders take no default architecture; a bare call is a TypeError."""
    with pytest.raises(TypeError):
        create_nano_u()


# -- create_bu_net -------------------------------------------------------------


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
