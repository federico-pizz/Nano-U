"""End-to-end pipeline integration tests.

Covers the full path: synthetic on-disk dataset → make_dataset → Nano-U
forward pass → loss computation → gradient update → BinaryIoU evaluation.
No GPU required; runs on CPU in a few seconds.
"""

import numpy as np
import pytest
import tensorflow as tf
import tf_keras as keras

from src.data import make_dataset
from src.evaluate import bce_loss_from_logits, dice_coef, sigmoid
from src.models.builders import create_nano_u
from src.utils.metrics import BinaryIoU

H, W = 60, 80


@pytest.fixture(scope="module")
def tiny_model():
    return create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)


# ── data → model forward pass ─────────────────────────────────────────────────

def test_pipeline_output_shape(tmp_png_dataset, tiny_model):
    """make_dataset batches must flow through the model to the correct output shape."""
    ds = make_dataset(
        tmp_png_dataset["imgs"], tmp_png_dataset["masks"],
        batch_size=2, shuffle=False, augment=False, target_size=(H, W),
    )
    img_batch, _ = next(iter(ds))
    logits = tiny_model(img_batch, training=False)
    assert logits.shape == (2, H, W, 1)


def test_pipeline_output_dtype(tmp_png_dataset, tiny_model):
    ds = make_dataset(
        tmp_png_dataset["imgs"], tmp_png_dataset["masks"],
        batch_size=2, shuffle=False, augment=False, target_size=(H, W),
    )
    img_batch, _ = next(iter(ds))
    logits = tiny_model(img_batch, training=False)
    assert logits.dtype == tf.float32


def test_pipeline_output_finite(tmp_png_dataset, tiny_model):
    ds = make_dataset(
        tmp_png_dataset["imgs"], tmp_png_dataset["masks"],
        batch_size=2, shuffle=False, augment=False, target_size=(H, W),
    )
    img_batch, _ = next(iter(ds))
    logits = tiny_model(img_batch, training=False)
    assert not np.any(np.isnan(logits.numpy()))
    assert not np.any(np.isinf(logits.numpy()))


# ── loss computation ──────────────────────────────────────────────────────────

def test_pipeline_bce_loss_finite(tmp_png_dataset, tiny_model):
    ds = make_dataset(
        tmp_png_dataset["imgs"], tmp_png_dataset["masks"],
        batch_size=2, shuffle=False, augment=False, target_size=(H, W),
    )
    img_batch, mask_batch = next(iter(ds))
    logits = tiny_model(img_batch, training=False)
    loss = bce_loss_from_logits(mask_batch, logits).numpy()
    assert np.isfinite(loss)
    assert loss > 0.0


def test_pipeline_dice_coef_in_range(tmp_png_dataset, tiny_model):
    ds = make_dataset(
        tmp_png_dataset["imgs"], tmp_png_dataset["masks"],
        batch_size=2, shuffle=False, augment=False, target_size=(H, W),
    )
    img_batch, mask_batch = next(iter(ds))
    logits = tiny_model(img_batch, training=False)
    probs = sigmoid(logits)
    dice = dice_coef(mask_batch, probs).numpy()
    assert 0.0 <= dice <= 1.0


# ── gradient update ───────────────────────────────────────────────────────────

def test_pipeline_training_step_reduces_loss(tmp_png_dataset):
    """Overfitting a single batch for several steps must reduce the loss."""
    model = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)

    ds = make_dataset(
        tmp_png_dataset["imgs"], tmp_png_dataset["masks"],
        batch_size=4, shuffle=False, augment=False, target_size=(H, W),
    )
    img_batch, mask_batch = next(iter(ds))

    def step():
        with tf.GradientTape() as tape:
            logits = model(img_batch, training=True)
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_batch, logits=logits)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return float(loss.numpy())

    initial_loss = step()
    final_loss = initial_loss
    for _ in range(30):
        final_loss = step()

    assert np.isfinite(final_loss)
    assert final_loss < initial_loss, (
        f"loss did not decrease over 30 steps: {initial_loss:.4f} -> {final_loss:.4f}"
    )


def test_pipeline_gradients_not_none(tmp_png_dataset):
    """Every trainable variable must receive a gradient on a normal batch."""
    model = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)

    ds = make_dataset(
        tmp_png_dataset["imgs"], tmp_png_dataset["masks"],
        batch_size=2, shuffle=False, augment=False, target_size=(H, W),
    )
    img_batch, mask_batch = next(iter(ds))

    with tf.GradientTape() as tape:
        logits = model(img_batch, training=True)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_batch, logits=logits)
        )
    grads = tape.gradient(loss, model.trainable_variables)

    none_vars = [v.name for v, g in zip(model.trainable_variables, grads) if g is None]
    assert len(none_vars) == 0, f"None gradients for: {none_vars}"


# ── metric evaluation ─────────────────────────────────────────────────────────

def test_pipeline_iou_stateful_over_full_dataset(tmp_png_dataset, tiny_model):
    """Accumulate BinaryIoU over all batches then call result() — must be in [0, 1]."""
    ds = make_dataset(
        tmp_png_dataset["imgs"], tmp_png_dataset["masks"],
        batch_size=2, shuffle=False, augment=False, target_size=(H, W),
    )
    metric = BinaryIoU(threshold=0.5)
    for img_batch, mask_batch in ds:
        logits = tiny_model(img_batch, training=False)
        probs = sigmoid(logits)
        metric.update_state(mask_batch, probs)

    iou = metric.result().numpy()
    assert 0.0 <= iou <= 1.0


def test_pipeline_mask_and_image_shapes_match(tmp_png_dataset):
    """Dataset must yield image and mask batches with consistent spatial dimensions."""
    ds = make_dataset(
        tmp_png_dataset["imgs"], tmp_png_dataset["masks"],
        batch_size=2, shuffle=False, augment=False, target_size=(H, W),
    )
    for img_batch, mask_batch in ds:
        assert img_batch.shape[1:3] == mask_batch.shape[1:3]
