"""Unit tests for nanou.redundancy.

The first five also carry the three cases that used to live in the library as a
self-check returning True/False (section 7.4): the 4-D conv case, the rank-1
case and the independent-channels case. Here a failure is reported instead of
swallowed.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from nanou.architectures import create_nano_u
from nanou.redundancy import RedundancyMonitor, compute_layer_redundancy, extract_activations


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
    assert res["condition_number"] > 0
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


# -- extract_activations and the monitor ---------------------------------------

H, W = 60, 80


def _tiny_model():
    return create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)


def test_extract_activations_handles_a_single_layer():
    """One layer name must map to that layer's tensor, not to a batch slice."""
    model = _tiny_model()
    x = tf.random.normal((3, H, W, 3))
    acts = extract_activations(model, ["enc1a_dw"], x)
    assert list(acts) == ["enc1a_dw"]
    assert acts["enc1a_dw"].shape[0] == 3
    assert acts["enc1a_dw"].shape[1:] == model.get_layer("enc1a_dw").output.shape[1:]


def test_extract_activations_handles_several_layers():
    model = _tiny_model()
    x = tf.random.normal((2, H, W, 3))
    acts = extract_activations(model, ["enc1a_dw", "enc1a_pw"], x)
    assert list(acts) == ["enc1a_dw", "enc1a_pw"]
    for name, act in acts.items():
        assert act.shape[1:] == model.get_layer(name).output.shape[1:]


def test_monitor_writes_the_agreed_columns(tmp_path):
    """The monitor measures through extract_activations and keeps the CSV schema."""
    model = _tiny_model()
    x = np.random.default_rng(0).standard_normal((4, H, W, 3)).astype(np.float32)
    monitor = RedundancyMonitor(
        layers_to_monitor=["enc1a_dw", "enc1a_pw"],
        log_frequency=1,
        output_dir=str(tmp_path),
        validation_data=(x, np.zeros((4, H, W, 1), np.float32)),
    )
    monitor.set_model(model)
    monitor.on_epoch_end(0)

    df = pd.read_csv(tmp_path / "metrics.csv")
    for layer in ("enc1a_dw", "enc1a_pw"):
        for suffix in ("redundancy_score", "condition_number", "rank", "num_channels"):
            assert f"{layer}_{suffix}" in df.columns
    assert 0.0 <= df["enc1a_dw_redundancy_score"].iloc[0] <= 1.0
