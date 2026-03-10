import tensorflow as tf
from src.nas import compute_layer_redundancy


def test_compute_layer_redundancy_shapes_and_ranges():
    """compute_layer_redundancy runs on 2D and 4D tensors and returns sane ranges."""
    for t in [
        tf.random.normal((32, 64)),           # wide
        tf.random.normal((64, 32)),           # tall
        tf.random.normal((16, 24, 32, 64)),   # conv-like activations
        tf.tile(tf.random.normal((32, 1)), [1, 64]),  # rank-deficient
    ]:
        res = compute_layer_redundancy(t)
        assert 0.0 <= res["redundancy_score"] <= 1.0
        assert res["condition_number"] > 0
        assert res["rank"] > 0
        assert res["num_channels"] > 0
