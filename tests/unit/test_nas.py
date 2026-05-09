"""Unit tests for src/nas.py — compute_layer_redundancy."""

import tensorflow as tf

from src.nas import compute_layer_redundancy


def test_redundancy_wide_matrix():
    t = tf.random.normal((32, 64))
    res = compute_layer_redundancy(t)
    assert 0.0 <= res["redundancy_score"] <= 1.0
    assert res["condition_number"] > 0
    assert res["rank"] > 0
    assert res["num_channels"] > 0


def test_redundancy_tall_matrix():
    t = tf.random.normal((64, 32))
    res = compute_layer_redundancy(t)
    assert 0.0 <= res["redundancy_score"] <= 1.0
    assert res["condition_number"] > 0
    assert res["rank"] > 0


def test_redundancy_conv_activations():
    """4-D conv-like tensor (B, H, W, C) must be handled without error."""
    t = tf.random.normal((16, 24, 32, 64))
    res = compute_layer_redundancy(t)
    assert 0.0 <= res["redundancy_score"] <= 1.0
    assert res["rank"] > 0


def test_redundancy_rank_deficient():
    """Rank-deficient input (all cols identical) must return high redundancy score."""
    t = tf.tile(tf.random.normal((32, 1)), [1, 64])
    res = compute_layer_redundancy(t)
    assert 0.0 <= res["redundancy_score"] <= 1.0
    assert res["redundancy_score"] > 0.5


def test_redundancy_returns_required_keys():
    t = tf.random.normal((16, 32))
    res = compute_layer_redundancy(t)
    for key in ("redundancy_score", "condition_number", "rank", "num_channels"):
        assert key in res, f"Missing key: {key}"
