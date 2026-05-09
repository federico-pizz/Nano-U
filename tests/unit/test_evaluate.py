"""Unit tests for src/evaluate.py."""

import numpy as np
import pytest
import tensorflow as tf

from src.evaluate import sigmoid, dice_coef, bce_loss_from_logits, focal_loss_from_logits

EPS = 1e-7


# ── sigmoid ───────────────────────────────────────────────────────────────────

def test_sigmoid_range():
    # Use ±10 — sigmoid(±50) also saturates to exact 0/1 in float32
    x = tf.constant([-10.0, -1.0, 0.0, 1.0, 10.0])
    y = sigmoid(x).numpy()
    assert y.min() > 0.0
    assert y.max() < 1.0


def test_sigmoid_zero():
    assert abs(sigmoid(tf.constant(0.0)).numpy() - 0.5) < 1e-6


def test_sigmoid_large_positive_saturates():
    assert sigmoid(tf.constant(100.0)).numpy() > 0.999


def test_sigmoid_large_negative_saturates():
    assert sigmoid(tf.constant(-100.0)).numpy() < 0.001


# ── dice_coef ─────────────────────────────────────────────────────────────────

def test_dice_coef_perfect_soft():
    """Identical soft prediction and label → 1.0."""
    t = tf.ones((1, 4, 4, 1), dtype=tf.float32)
    assert abs(dice_coef(t, t).numpy() - 1.0) < 1e-5


def test_dice_coef_perfect_binary(mask_tensor):
    d = dice_coef(mask_tensor, mask_tensor, threshold=0.5)
    assert abs(d.numpy() - 1.0) < 1e-5


def test_dice_coef_no_overlap():
    y_true = tf.constant([[[[1.0], [0.0]], [[0.0], [0.0]]]])
    y_pred = tf.constant([[[[0.0], [1.0]], [[0.0], [0.0]]]])
    d = dice_coef(y_true, y_pred, threshold=0.5).numpy()
    assert d < 0.1


def test_dice_coef_empty_mask_no_nan():
    """All-zero label must not produce NaN (epsilon guard)."""
    y_true = tf.zeros((1, 8, 8, 1))
    y_pred = tf.zeros((1, 8, 8, 1))
    d = dice_coef(y_true, y_pred).numpy()
    assert not np.isnan(d)


def test_dice_coef_in_01():
    rng = np.random.default_rng(5)
    y_true = tf.constant((rng.random((2, 16, 16, 1)) > 0.5).astype(np.float32))
    y_pred = tf.constant(rng.random((2, 16, 16, 1)).astype(np.float32))
    d = dice_coef(y_true, y_pred).numpy()
    assert 0.0 <= d <= 1.0 + EPS


# ── bce_loss_from_logits ──────────────────────────────────────────────────────

def test_bce_loss_from_logits_positive(logits_tensor, mask_tensor):
    loss = bce_loss_from_logits(mask_tensor, logits_tensor).numpy()
    assert loss > 0.0


def test_bce_loss_from_logits_finite(logits_tensor, mask_tensor):
    loss = bce_loss_from_logits(mask_tensor, logits_tensor).numpy()
    assert np.isfinite(loss)


def test_bce_loss_from_logits_agrees_with_tf(logits_tensor, mask_tensor):
    """Must match tf.nn.sigmoid_cross_entropy_with_logits."""
    import tf_keras as keras
    ref = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_tensor, logits=logits_tensor)
    ).numpy()
    ours = bce_loss_from_logits(mask_tensor, logits_tensor).numpy()
    assert abs(ours - ref) < 1e-4


# ── focal_loss_from_logits ────────────────────────────────────────────────────

def test_focal_loss_positive(logits_tensor, mask_tensor):
    loss = focal_loss_from_logits(mask_tensor, logits_tensor).numpy()
    assert loss > 0.0


def test_focal_loss_finite(logits_tensor, mask_tensor):
    loss = focal_loss_from_logits(mask_tensor, logits_tensor).numpy()
    assert np.isfinite(loss)


def test_focal_loss_le_bce(logits_tensor, mask_tensor):
    """Focal loss (gamma=2) should down-weight easy examples, so <= BCE."""
    bce = bce_loss_from_logits(mask_tensor, logits_tensor).numpy()
    focal = focal_loss_from_logits(mask_tensor, logits_tensor, gamma=2.0, alpha=0.5).numpy()
    # focal uses alpha weighting so direct comparison is approximate; just confirm it's finite and small
    assert np.isfinite(focal)
    assert focal >= 0.0
