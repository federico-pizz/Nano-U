"""Unit tests for src/utils/metrics.py — BinaryIoU."""

import numpy as np
import pytest
import tensorflow as tf

from src.utils.metrics import BinaryIoU


def _make_iou(threshold=0.5):
    return BinaryIoU(threshold=threshold)


# ── correctness ───────────────────────────────────────────────────────────────

def test_binary_iou_perfect():
    """Mix of fg/bg, perfect prediction → mIoU == 1.0."""
    m = _make_iou()
    rng = np.random.default_rng(0)
    y = tf.constant((rng.random((1, 8, 8, 1)) > 0.5).astype(np.float32))
    m.update_state(y, y)
    assert abs(m.result().numpy() - 1.0) < 1e-5


def test_binary_iou_all_foreground():
    """All pixels foreground, perfect pred → fg IoU=1, bg IoU=0 → mIoU=0.5."""
    m = _make_iou()
    y = tf.ones((1, 8, 8, 1))
    m.update_state(y, y)
    assert abs(m.result().numpy() - 0.5) < 1e-5


def test_binary_iou_no_overlap():
    """Pred all foreground, label all background → IoU 0 for both classes → ~0."""
    m = _make_iou()
    y_true = tf.zeros((1, 4, 4, 1))
    y_pred = tf.ones((1, 4, 4, 1))
    m.update_state(y_true, y_pred)
    assert m.result().numpy() < 0.1


def test_binary_iou_half_overlap():
    m = _make_iou()
    y_true = tf.constant([[[[1.], [1.], [0.], [0.]]]])  # shape (1,1,4,1)
    y_pred = tf.constant([[[[1.], [0.], [1.], [0.]]]])
    m.update_state(y_true, y_pred)
    result = m.result().numpy()
    assert 0.0 < result < 1.0


# ── threshold ─────────────────────────────────────────────────────────────────

def test_binary_iou_threshold_binarizes_correctly():
    m = BinaryIoU(threshold=0.7)
    # pred = 0.6 is below 0.7 → treated as 0; label = 1 → FN
    y_true = tf.ones((1, 4, 4, 1))
    y_pred = tf.ones((1, 4, 4, 1)) * 0.6
    m.update_state(y_true, y_pred)
    result = m.result().numpy()
    # All positives are false negatives → low score
    assert result < 0.5


def test_binary_iou_threshold_just_above():
    """All-foreground perfect prediction → fg_iou=1, bg_iou=0 → mIoU=0.5."""
    m = BinaryIoU(threshold=0.5)
    y_true = tf.ones((1, 4, 4, 1))
    y_pred = tf.ones((1, 4, 4, 1)) * 0.51  # just above threshold → all TP
    m.update_state(y_true, y_pred)
    assert abs(m.result().numpy() - 0.5) < 1e-4


# ── stateful accumulation ─────────────────────────────────────────────────────

def test_binary_iou_stateful_two_batches():
    """Calling update_state twice then result() must average across all pixels."""
    m = _make_iou()
    y_perfect = tf.ones((1, 4, 4, 1))
    y_worst = tf.zeros((1, 4, 4, 1))    # pred=0, true=1 → all FN
    y_true = tf.ones((1, 4, 4, 1))

    m.update_state(y_true, y_perfect)   # perfect batch
    m.update_state(y_true, y_worst)     # worst batch
    result = m.result().numpy()
    # Should be between 0 and 1
    assert 0.0 < result < 1.0


def test_binary_iou_reset_state():
    m = _make_iou()
    rng = np.random.default_rng(10)
    y = tf.constant((rng.random((1, 4, 4, 1)) > 0.5).astype(np.float32))
    m.update_state(y, y)
    before_reset = m.result().numpy()
    assert before_reset > 0.9  # perfect prediction with mixed fg/bg

    m.reset_state()
    # After reset, update with all-zero → bg IoU only
    m.update_state(tf.zeros((1, 4, 4, 1)), tf.zeros((1, 4, 4, 1)))
    result_after_reset = m.result().numpy()
    assert result_after_reset >= 0.0  # sanity — must not crash or go negative


# ── from_logits ───────────────────────────────────────────────────────────────

def test_binary_iou_from_logits_flag():
    """from_logits=True: sigmoid(5)≈0.993 → above 0.5 → all-fg perfect → mIoU=0.5."""
    logits = tf.ones((1, 4, 4, 1)) * 5.0
    y_true = tf.ones((1, 4, 4, 1))
    m = BinaryIoU(threshold=0.5, from_logits=True)
    m.update_state(y_true, logits)
    assert abs(m.result().numpy() - 0.5) < 1e-4
