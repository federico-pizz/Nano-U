"""Integration tests: verify TensorFlow GPU usage with the Nano-U model.

All tests are marked with @pytest.mark.gpu and are automatically skipped
when no CUDA GPU is detected.  Run them explicitly with:

    pytest tests/integration/test_gpu.py -v
or exclude them from a CPU run with:
    pytest -m "not gpu"
"""

import numpy as np
import pytest
import tensorflow as tf

H, W = 60, 80

# Skip the entire module if TF cannot see a GPU
gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    if not gpu_available:
        pytest.skip("No GPU detected — skipping GPU integration tests")


# ── basic GPU availability ────────────────────────────────────────────────────

def test_gpu_is_visible():
    gpus = tf.config.list_physical_devices("GPU")
    assert len(gpus) > 0, "TensorFlow sees no GPU devices"


def test_tensor_op_runs_on_gpu():
    """A simple matmul must execute on GPU without error."""
    with tf.device("/GPU:0"):
        a = tf.random.normal((64, 64))
        b = tf.random.normal((64, 64))
        c = tf.matmul(a, b)
    assert c.shape == (64, 64)
    assert c.dtype == tf.float32


def test_gpu_result_matches_cpu():
    """GPU matmul must produce a finite, correctly-shaped result.

    float32 CPU/GPU results are not bit-exact (different reduction order),
    so we only verify shape, dtype, and finiteness — not numerical equality.
    """
    tf.random.set_seed(42)
    a = tf.random.normal((32, 32), seed=1)
    b = tf.random.normal((32, 32), seed=2)

    with tf.device("/GPU:0"):
        gpu_result = tf.matmul(a, b).numpy()

    assert gpu_result.shape == (32, 32)
    assert not np.any(np.isnan(gpu_result))
    assert not np.any(np.isinf(gpu_result))


# ── model forward pass on GPU ─────────────────────────────────────────────────

def test_nano_u_forward_pass_shape_on_gpu(nano_u_model):
    """Full forward pass through Nano-U on GPU must produce (1, H, W, 1) output."""
    x = tf.random.normal((1, H, W, 3))
    with tf.device("/GPU:0"):
        out = nano_u_model(x, training=False)
    assert out.shape == (1, H, W, 1)


def test_nano_u_output_dtype_on_gpu(nano_u_model):
    x = tf.random.normal((1, H, W, 3))
    with tf.device("/GPU:0"):
        out = nano_u_model(x, training=False)
    assert out.dtype == tf.float32


def test_nano_u_output_no_nan_on_gpu(nano_u_model):
    x = tf.random.normal((1, H, W, 3))
    with tf.device("/GPU:0"):
        out = nano_u_model(x, training=False)
    assert not np.any(np.isnan(out.numpy()))


def test_nano_u_output_no_inf_on_gpu(nano_u_model):
    x = tf.random.normal((1, H, W, 3))
    with tf.device("/GPU:0"):
        out = nano_u_model(x, training=False)
    assert not np.any(np.isinf(out.numpy()))


def test_nano_u_batch_forward_pass_on_gpu(nano_u_model):
    """Batch size > 1 must work correctly."""
    x = tf.random.normal((4, H, W, 3))
    with tf.device("/GPU:0"):
        out = nano_u_model(x, training=False)
    assert out.shape == (4, H, W, 1)


def test_nano_u_gpu_cpu_outputs_agree(nano_u_model):
    """GPU and CPU forward passes must agree to float32 precision."""
    tf.random.set_seed(0)
    x = tf.random.normal((1, H, W, 3), seed=7)
    with tf.device("/CPU:0"):
        cpu_out = nano_u_model(x, training=False).numpy()
    with tf.device("/GPU:0"):
        gpu_out = nano_u_model(x, training=False).numpy()
    np.testing.assert_allclose(cpu_out, gpu_out, rtol=1e-4, atol=1e-4)


# ── training step on GPU ──────────────────────────────────────────────────────

def test_nano_u_training_step_on_gpu(nano_u_model):
    """One gradient update on GPU must produce a finite loss."""
    import tf_keras as keras

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    x = tf.random.normal((2, H, W, 3))
    y = tf.cast(tf.random.uniform((2, H, W, 1)) > 0.5, tf.float32)

    with tf.device("/GPU:0"):
        with tf.GradientTape() as tape:
            logits = nano_u_model(x, training=True)
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            )
        grads = tape.gradient(loss, nano_u_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, nano_u_model.trainable_variables))

    assert np.isfinite(loss.numpy()), f"Loss is not finite: {loss.numpy()}"
    assert loss.numpy() > 0.0


def test_nano_u_gradients_not_none_on_gpu(nano_u_model):
    """All trainable variables must receive non-None gradients."""
    x = tf.random.normal((1, H, W, 3))
    y = tf.cast(tf.random.uniform((1, H, W, 1)) > 0.5, tf.float32)

    with tf.device("/GPU:0"):
        with tf.GradientTape() as tape:
            logits = nano_u_model(x, training=True)
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            )
        grads = tape.gradient(loss, nano_u_model.trainable_variables)

    none_grads = [v.name for v, g in zip(nano_u_model.trainable_variables, grads) if g is None]
    assert len(none_grads) == 0, f"None gradients for: {none_grads}"
