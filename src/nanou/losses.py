"""Loss terms used by the training step.

The binary cross-entropy the students train against is stock Keras; what lives
here is the shaping term that is not.
"""

import tensorflow as tf


def tversky_loss(
    y_true: tf.Tensor,
    y_logits: tf.Tensor,
    alpha_fp: float = 0.7,
    beta_fn: float = 0.3,
    eps: float = 1e-7,
) -> tf.Tensor:
    """Soft Tversky loss for the path (positive=1) class.

    Tversky index T = TP / (TP + alpha_fp*FP + beta_fn*FN); loss = 1 - T. With
    ``alpha_fp > beta_fn`` (default 0.7 vs 0.3) false *positives* are penalized
    harder than false negatives. For path segmentation a false positive labels a
    non-traversable pixel as path, a potential collision, while a false negative
    only shrinks the navigable region: over-cautious but safe. So this dial puts
    Nano-U's conservative (precision-favoring) behaviour into the gradient
    instead of manufacturing it through model selection, which is also why the
    CV sweep selects on mIoU and keeps F0.5 as the tiebreak.
    ``alpha_fp=beta_fn=0.5`` recovers the symmetric Dice loss.

    Computed from sigmoid probabilities (the student emits logits), pooled over
    the whole batch (soft TP/FP/FN), so it is differentiable end-to-end.
    """
    p = tf.nn.sigmoid(y_logits)
    yt = tf.cast(y_true > 0.5, tf.float32)
    tp = tf.reduce_sum(p * yt)
    fp = tf.reduce_sum(p * (1.0 - yt))
    fn = tf.reduce_sum((1.0 - p) * yt)
    tversky = (tp + eps) / (tp + alpha_fp * fp + beta_fn * fn + eps)
    return 1.0 - tversky
