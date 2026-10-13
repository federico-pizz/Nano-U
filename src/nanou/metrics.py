"""Segmentation metrics, and the two Keras objects that carry them.

Everything here is a pure function of arrays: no model is loaded, no file is
read. That is what lets the host evaluator, the sweep and the on-device
evaluator all score through the same code, which is the only way their numbers
are comparable by construction.
"""

import json

import numpy as np
import tensorflow as tf
import tf_keras as keras


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars and arrays."""

    def default(self, obj):
        """Return a JSON-serialisable stand-in for a NumPy value."""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BinaryIoU(keras.metrics.Metric):
    """Mean IoU over the two classes, accumulated as a streaming confusion matrix.

    Foreground and background IoU are averaged, so a model that predicts all
    background scores around 0.5 rather than 0.0 - which is what the reported
    mIoU means throughout this project.
    """

    def __init__(self, threshold=0.5, from_logits=False, name="binary_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.from_logits = from_logits
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate tp/fp/fn/tn for one batch. `sample_weight` is ignored."""
        # Convert logits -> probabilities if the caller said so explicitly,
        # otherwise use the values as probabilities directly.
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true > 0.5, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
        self.tn.assign_add(tn)

    def result(self):
        """Return the mean of foreground and background IoU.

        `(iou_fg + iou_bg) / 2`, not foreground-only IoU. Background IoU is high
        and stable on this task, so this mean sits above foreground IoU and can
        sit *below* F1 - which surprises readers who assume "mIoU" means the
        foreground class alone. Every mIoU this project reports is this formula.
        """
        # Calculate Intersection over Union for foreground and background, returning the mean (mIoU)
        iou_fg = self.tp / (self.tp + self.fp + self.fn + 1e-7)
        iou_bg = self.tn / (self.tn + self.fp + self.fn + 1e-7)
        return (iou_fg + iou_bg) / 2.0

    def reset_state(self):
        """Zero the four counters between epochs."""
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
        self.tn.assign(0.0)

    def get_config(self):
        """Return the constructor arguments, so the metric survives save/load."""
        config = super().get_config()
        config.update({"threshold": self.threshold, "from_logits": self.from_logits})
        return config


EPS = 1e-7


def sigmoid(x):
    """Elementwise logistic sigmoid."""
    return tf.math.sigmoid(x)


def dice_coef(y_true, y_pred, threshold=None):
    """Dice coefficient. `threshold` binarises the prediction first; None keeps it soft."""
    # y_pred may be logits or probabilities; assume probabilities
    if threshold is not None:
        y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * inter + EPS) / (denom + EPS)


def bce_loss_from_logits(y_true, logits):
    """Binary cross-entropy computed straight from logits."""
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(y_true, logits)


def focal_loss_from_logits(y_true, logits, gamma=2.0, alpha=0.25):
    """Focal loss with correct asymmetric alpha weighting.

    alpha weights the positive class; (1 - alpha) weights the negative class.
    This provides class-balance control, not just a loss scale.
    """
    prob = tf.math.sigmoid(logits)
    y_true = tf.cast(y_true, tf.float32)
    pt = tf.where(tf.equal(y_true, 1.0), prob, 1.0 - prob)
    # alpha_t: alpha for positives, (1-alpha) for negatives
    alpha_t = tf.where(
        tf.equal(y_true, 1.0),
        tf.fill(tf.shape(y_true), alpha),
        tf.fill(tf.shape(y_true), 1.0 - alpha),
    )
    loss = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt + EPS)
    return tf.reduce_mean(loss)


def fbeta_score(precision, recall, beta=1.0, eps=EPS):
    """F-beta = (1+b^2)*P*R / (b^2*P + R).

    beta>1 weights recall (e.g. F2), beta<1 weights precision (e.g. F0.5).
    For autonomous navigation the positive class is 'road/drivable', so a
    precision-weighted F0.5 (don't call it road unless sure) is the conservative
    selection criterion.
    """
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (b2 * precision + recall + eps)


def _metrics_from_counts(tp, fp, fn, tn, eps=EPS):
    """Pixel-confusion counts -> the metric dict reported everywhere."""
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    miou = 0.5 * (iou_fg + iou_bg)  # same formula as utils.metrics.BinaryIoU
    return {
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "miou": float(miou),
        "f0.5": float(fbeta_score(precision, recall, 0.5, eps)),
        "f1": float(fbeta_score(precision, recall, 1.0, eps)),
        "f2": float(fbeta_score(precision, recall, 2.0, eps)),
    }


def compute_segmentation_metrics(
    probs, masks, thresholds=None, groups=None, operating_threshold=0.5
):
    """Rich, interpretable metrics from probability maps and binary masks.

    Args:
        probs:  float array (N, H, W[, 1]) of road/drivable probabilities in [0,1].
        masks:  array same leading shape; thresholded at 0.5 to a binary label.
        thresholds: thresholds for the sweep (default 0.05..0.95). The sweep gives
            a PR curve and lets the operating point be *chosen*, not assumed.
        groups: optional length-N sequence ids -> per-sequence mean+/-std, exposing
            the variance the pooled headline hides.
        operating_threshold: decision threshold for the headline numbers - the
            conservativeness knob (higher = stricter 'road').

    Returns a JSON-serializable dict.
    """
    probs = np.asarray(probs, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    mask_bin = masks > 0.5
    n = probs.shape[0]
    axis = tuple(range(1, probs.ndim))

    # Per-image counts at the operating threshold -> fast pooled/bootstrap/group sums.
    pred_bin = probs > operating_threshold
    tp_i = np.sum(mask_bin & pred_bin, axis=axis).astype(np.float64)
    fp_i = np.sum((~mask_bin) & pred_bin, axis=axis).astype(np.float64)
    fn_i = np.sum(mask_bin & (~pred_bin), axis=axis).astype(np.float64)
    tn_i = np.sum((~mask_bin) & (~pred_bin), axis=axis).astype(np.float64)

    res = _metrics_from_counts(tp_i.sum(), fp_i.sum(), fn_i.sum(), tn_i.sum())
    res["threshold"] = float(operating_threshold)
    res["n_images"] = int(n)

    # Threshold sweep (PR curve + metric-vs-threshold).
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 19), 3)
    sweep = {
        "thresholds": [float(t) for t in thresholds],
        "precision": [],
        "recall": [],
        "f0.5": [],
        "f1": [],
        "f2": [],
        "miou": [],
    }
    for t in thresholds:
        pb = probs > t
        m = _metrics_from_counts(
            float(np.sum(mask_bin & pb)),
            float(np.sum((~mask_bin) & pb)),
            float(np.sum(mask_bin & (~pb))),
            float(np.sum((~mask_bin) & (~pb))),
        )
        for k in ("precision", "recall", "f0.5", "f1", "f2", "miou"):
            sweep[k].append(m[k])
    res["threshold_sweep"] = sweep

    # Per-sequence breakdown (leakage-relevant: variance across scenes).
    if groups is not None:
        groups = list(groups)
        if len(groups) != n:
            raise ValueError(f"groups length {len(groups)} != n_images {n}")
        g_arr = np.array(groups)
        per = {"groups": [], "miou": [], "f0.5": []}
        for g in sorted(set(groups)):
            idx = np.where(g_arr == g)[0]
            m = _metrics_from_counts(
                tp_i[idx].sum(), fp_i[idx].sum(), fn_i[idx].sum(), tn_i[idx].sum()
            )
            per["groups"].append(str(g))
            per["miou"].append(m["miou"])
            per["f0.5"].append(m["f0.5"])
        per["n_groups"] = len(per["groups"])
        for k in ("miou", "f0.5"):
            arr = np.array(per[k], dtype=np.float64)
            per[f"{k}_mean"] = float(arr.mean())
            per[f"{k}_std"] = float(arr.std())
        res["per_sequence"] = per

    return res
