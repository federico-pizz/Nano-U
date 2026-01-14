import numpy as np
import tensorflow as tf
from src.nas_covariance import (
    _collapse_spatial_to_channels,
    covariance_redundancy,
    RunningCovariance,
)


def test_collapse_spatial_to_channels_rank4():
    # Create batch of 2 images, H=2,W=2,C=3
    x = tf.constant(np.arange(2*2*2*3).reshape((2,2,2,3)), dtype=tf.float32)
    out = _collapse_spatial_to_channels(x, data_format='channels_last')
    assert out.shape == (2, 3)
    # Manual mean over H,W for the first batch and channel 0
    expected0 = 4.5
    assert np.isclose(out.numpy()[0,0], expected0)


def test_covariance_redundancy_perfect_correlation():
    # Use linearly related samples so channels have covariance
    x = tf.constant([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=tf.float32)
    score = covariance_redundancy(x, normalize=True)
    # should be a finite positive scalar
    assert float(score.numpy()) > 0.0
    assert np.isfinite(float(score.numpy()))


def test_running_covariance_update_and_reset():
    # Prepare simple data: two channels, small batch
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    rc = RunningCovariance(channels=2)
    # update with batch
    rc.update(x)
    # after update, n should be > 0 and covariance finite
    assert int(rc.n.numpy()) > 0
    cov = rc.covariance().numpy()
    assert cov.shape == (2,2)
    assert np.all(np.isfinite(cov))
    # compute redundancy score
    score = rc.redundancy_score().numpy()
    assert np.isfinite(score)

    # reset and verify zeroed state
    rc.reset()
    assert int(rc.n.numpy()) == 0
    cov_after = rc.covariance().numpy()
    assert np.allclose(cov_after, np.zeros_like(cov_after))
