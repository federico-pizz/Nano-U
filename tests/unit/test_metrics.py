"""Unit tests for nanou.metrics - the metric functions and BinaryIoU."""

import numpy as np
import pytest
import tensorflow as tf

from nanou.metrics import (
    BinaryIoU,
    bce_loss_from_logits,
    compute_segmentation_metrics,
    dice_coef,
    fbeta_score,
    focal_loss_from_logits,
    sigmoid,
)

EPS = 1e-7


# -- sigmoid -------------------------------------------------------------------


def test_sigmoid_range():
    # Use +/-10 - sigmoid(+/-50) also saturates to exact 0/1 in float32
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


# -- dice_coef -----------------------------------------------------------------


def test_dice_coef_perfect_soft():
    """Identical soft prediction and label -> 1.0."""
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


# -- bce_loss_from_logits ------------------------------------------------------


def test_bce_loss_from_logits_positive(logits_tensor, mask_tensor):
    loss = bce_loss_from_logits(mask_tensor, logits_tensor).numpy()
    assert loss > 0.0


def test_bce_loss_from_logits_finite(logits_tensor, mask_tensor):
    loss = bce_loss_from_logits(mask_tensor, logits_tensor).numpy()
    assert np.isfinite(loss)


def test_bce_loss_from_logits_agrees_with_tf(logits_tensor, mask_tensor):
    """Must match tf.nn.sigmoid_cross_entropy_with_logits."""
    ref = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_tensor, logits=logits_tensor)
    ).numpy()
    ours = bce_loss_from_logits(mask_tensor, logits_tensor).numpy()
    assert abs(ours - ref) < 1e-4


# -- focal_loss_from_logits ----------------------------------------------------


def test_focal_loss_positive(logits_tensor, mask_tensor):
    loss = focal_loss_from_logits(mask_tensor, logits_tensor).numpy()
    assert loss > 0.0


def test_focal_loss_finite(logits_tensor, mask_tensor):
    loss = focal_loss_from_logits(mask_tensor, logits_tensor).numpy()
    assert np.isfinite(loss)


def test_focal_loss_with_alpha_is_finite_and_nonnegative(logits_tensor, mask_tensor):
    """Alpha weighting rules out a direct comparison against BCE, so check the range."""
    focal = focal_loss_from_logits(mask_tensor, logits_tensor, gamma=2.0, alpha=0.5).numpy()
    assert np.isfinite(focal)
    assert focal >= 0.0


# -- fbeta_score ---------------------------------------------------------------


def test_fbeta_equals_f1_at_beta_1():
    p, r = 0.6, 0.4
    assert abs(fbeta_score(p, r, 1.0) - (2 * p * r / (p + r))) < 1e-6


def test_fbeta_equals_value_when_precision_equals_recall():
    # F_beta == P == R whenever P == R, for any beta.
    for beta in (0.5, 1.0, 2.0):
        assert abs(fbeta_score(0.7, 0.7, beta) - 0.7) < 1e-6


def test_fbeta_beta_lt_1_weights_precision():
    """recall>precision: F0.5 (precision-weighted) < F1 < F2 (recall-weighted)."""
    p, r = 0.4, 0.8
    assert fbeta_score(p, r, 0.5) < fbeta_score(p, r, 1.0) < fbeta_score(p, r, 2.0)


def test_fbeta_beta_lt_1_rewards_high_precision():
    """precision>recall: the conservative F0.5 exceeds F1 (and F2)."""
    p, r = 0.9, 0.5
    assert fbeta_score(p, r, 0.5) > fbeta_score(p, r, 1.0) > fbeta_score(p, r, 2.0)


# -- compute_segmentation_metrics ----------------------------------------------


def _probs_masks(n=6, h=8, w=8, seed=0):
    rng = np.random.default_rng(seed)
    masks = (rng.random((n, h, w, 1)) > 0.5).astype(np.float32)
    return masks


def test_metrics_perfect_prediction():
    masks = _probs_masks()
    res = compute_segmentation_metrics(masks.copy(), masks)
    for k in ("miou", "dice", "precision", "recall", "f0.5", "f1", "f2"):
        assert abs(res[k] - 1.0) < 1e-4, (k, res[k])


def test_metrics_in_unit_range_and_keys():
    masks = _probs_masks(seed=1)
    rng = np.random.default_rng(2)
    probs = rng.random(masks.shape).astype(np.float32)
    res = compute_segmentation_metrics(probs, masks)
    for k in ("miou", "dice", "precision", "recall", "f0.5", "f1", "f2"):
        assert 0.0 <= res[k] <= 1.0 + 1e-6
    # sweep length matches the requested thresholds
    sw = res["threshold_sweep"]
    assert len(sw["thresholds"]) == len(sw["miou"]) == len(sw["precision"])


def test_metrics_per_sequence_breakdown():
    masks = _probs_masks(n=4, seed=3)
    res = compute_segmentation_metrics(masks.copy(), masks, groups=["a", "a", "b", "b"])
    ps = res["per_sequence"]
    assert ps["n_groups"] == 2
    assert ps["groups"] == ["a", "b"]
    assert "miou_mean" in ps and "miou_std" in ps


def test_metrics_groups_length_mismatch_raises():
    masks = _probs_masks(n=4, seed=4)
    with pytest.raises(ValueError):
        compute_segmentation_metrics(masks.copy(), masks, groups=["a", "b"])


def test_metrics_custom_operating_threshold():
    # All probs = 0.6: at threshold 0.5 everything is 'road', at 0.7 nothing is.
    masks = np.ones((3, 4, 4, 1), dtype=np.float32)
    probs = np.full_like(masks, 0.6)
    hi = compute_segmentation_metrics(probs, masks, operating_threshold=0.5)
    lo = compute_segmentation_metrics(probs, masks, operating_threshold=0.7)
    assert hi["recall"] > lo["recall"]  # stricter threshold -> fewer positives


def test_numpy_miou_matches_binary_iou():
    """The numpy mIoU must equal the streaming BinaryIoU the models train against.

    compute_segmentation_metrics reimplements mIoU in numpy (for the sweep and
    per-sequence breakdown); this pins it to nanou.metrics.BinaryIoU so the two can
    never silently diverge. Random probs/masks at the default 0.5 threshold.
    """
    rng = np.random.default_rng(7)
    masks = (rng.random((5, 8, 8, 1)) > 0.5).astype(np.float32)
    probs = rng.random(masks.shape).astype(np.float32)

    np_miou = compute_segmentation_metrics(probs, masks, operating_threshold=0.5)["miou"]

    iou = BinaryIoU(threshold=0.5)
    iou.update_state(masks, probs)
    keras_miou = float(iou.result().numpy())

    assert abs(np_miou - keras_miou) < 1e-4, (np_miou, keras_miou)


# -- BinaryIoU (the Keras metric) ---------------------------------------------


def _make_iou(threshold=0.5):
    return BinaryIoU(threshold=threshold)


# -- correctness ---------------------------------------------------------------


def test_binary_iou_perfect():
    """Mix of fg/bg, perfect prediction -> mIoU == 1.0."""
    m = _make_iou()
    rng = np.random.default_rng(0)
    y = tf.constant((rng.random((1, 8, 8, 1)) > 0.5).astype(np.float32))
    m.update_state(y, y)
    assert abs(m.result().numpy() - 1.0) < 1e-5


def test_binary_iou_all_foreground():
    """All pixels foreground, perfect pred -> fg IoU=1, bg IoU=0 -> mIoU=0.5."""
    m = _make_iou()
    y = tf.ones((1, 8, 8, 1))
    m.update_state(y, y)
    assert abs(m.result().numpy() - 0.5) < 1e-5


def test_binary_iou_no_overlap():
    """Pred all foreground, label all background -> IoU 0 for both classes -> ~0."""
    m = _make_iou()
    y_true = tf.zeros((1, 4, 4, 1))
    y_pred = tf.ones((1, 4, 4, 1))
    m.update_state(y_true, y_pred)
    assert m.result().numpy() < 0.1


def test_binary_iou_half_overlap():
    m = _make_iou()
    y_true = tf.constant([[[[1.0], [1.0], [0.0], [0.0]]]])  # shape (1,1,4,1)
    y_pred = tf.constant([[[[1.0], [0.0], [1.0], [0.0]]]])
    m.update_state(y_true, y_pred)
    result = m.result().numpy()
    assert 0.0 < result < 1.0


# -- threshold -----------------------------------------------------------------


def test_binary_iou_threshold_binarizes_correctly():
    m = BinaryIoU(threshold=0.7)
    # pred = 0.6 is below 0.7 -> treated as 0; label = 1 -> FN
    y_true = tf.ones((1, 4, 4, 1))
    y_pred = tf.ones((1, 4, 4, 1)) * 0.6
    m.update_state(y_true, y_pred)
    result = m.result().numpy()
    # All positives are false negatives -> low score
    assert result < 0.5


def test_binary_iou_threshold_just_above():
    """All-foreground perfect prediction -> fg_iou=1, bg_iou=0 -> mIoU=0.5."""
    m = BinaryIoU(threshold=0.5)
    y_true = tf.ones((1, 4, 4, 1))
    y_pred = tf.ones((1, 4, 4, 1)) * 0.51  # just above threshold -> all TP
    m.update_state(y_true, y_pred)
    assert abs(m.result().numpy() - 0.5) < 1e-4


# -- stateful accumulation -----------------------------------------------------


def test_binary_iou_stateful_two_batches():
    """Calling update_state twice then result() must average across all pixels."""
    m = _make_iou()
    y_perfect = tf.ones((1, 4, 4, 1))
    y_worst = tf.zeros((1, 4, 4, 1))  # pred=0, true=1 -> all FN
    y_true = tf.ones((1, 4, 4, 1))

    m.update_state(y_true, y_perfect)  # perfect batch
    m.update_state(y_true, y_worst)  # worst batch
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
    # After reset, update with all-zero -> bg IoU only
    m.update_state(tf.zeros((1, 4, 4, 1)), tf.zeros((1, 4, 4, 1)))
    result_after_reset = m.result().numpy()
    assert result_after_reset >= 0.0  # sanity - must not crash or go negative


# -- from_logits ---------------------------------------------------------------


def test_binary_iou_from_logits_flag():
    """from_logits=True: sigmoid(5)~0.993 -> above 0.5 -> all-fg perfect -> mIoU=0.5."""
    logits = tf.ones((1, 4, 4, 1)) * 5.0
    y_true = tf.ones((1, 4, 4, 1))
    m = BinaryIoU(threshold=0.5, from_logits=True)
    m.update_state(y_true, logits)
    assert abs(m.result().numpy() - 0.5) < 1e-4
