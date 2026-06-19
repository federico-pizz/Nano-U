"""Ablation-toggle mechanics (#7 CE-loss on/off, #8 augmentation on/off).

These pin the behavior the CV sweep relies on:
  * CE off is exactly ``alpha=1.0`` in the KD loss ``(1-alpha)·CE + alpha·distill``;
  * augmentation off is deterministic, augmentation on changes pixels.
No production code changes are needed — these lock the existing mechanics.
"""

import numpy as np
import pytest
import tensorflow as tf
import tf_keras as keras

from src.train import train_step, tversky_loss
from src.data import make_dataset
from src.models.builders import create_nano_u

H, W = 60, 80


def _student_teacher():
    s = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    t = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    return s, t


# ── #7: CE-loss on/off via alpha ──────────────────────────────────────────────

def test_ce_off_alpha_one_total_equals_distillation(img_tensor, mask_tensor):
    """alpha=1.0 → CE coefficient (1-alpha)=0, so total loss == distillation loss."""
    student, teacher = _student_teacher()
    opt = keras.optimizers.Adam(1e-3)
    student(img_tensor)  # materialize variables
    opt.build(student.trainable_variables)  # build optimizer outside tf.function
    out = train_step(student, teacher, img_tensor, mask_tensor, opt,
                     alpha=1.0, temperature=4.0)
    assert abs(float(out["loss"]) - float(out["distillation_loss"])) < 1e-5
    assert float(out["student_loss"]) > 0.0  # CE still computed, just not weighted in


def test_ce_on_contributes_to_total(img_tensor, mask_tensor):
    """alpha=0.5 → total == 0.5·CE + 0.5·distill, distinct from distill alone."""
    student, teacher = _student_teacher()
    opt = keras.optimizers.Adam(1e-3)
    student(img_tensor)  # materialize variables
    opt.build(student.trainable_variables)  # build optimizer outside tf.function
    out = train_step(student, teacher, img_tensor, mask_tensor, opt,
                     alpha=0.5, temperature=4.0)
    expected = 0.5 * float(out["student_loss"]) + 0.5 * float(out["distillation_loss"])
    assert abs(float(out["loss"]) - expected) < 1e-4
    assert abs(float(out["loss"]) - float(out["distillation_loss"])) > 1e-6


# ── conservative Tversky supervised loss ──────────────────────────────────────

def test_tversky_penalizes_false_positive_more_than_false_negative():
    """alpha_fp>beta_fn ⇒ a false 'path' pixel costs more than a missed one.

    This is the safety property: for path=1, a false positive is a potential
    collision, a false negative is just over-caution. Construct a single-FP case
    and a single-FN case with everything else correct and assert FP loss > FN.
    """
    # logits → high positive ≈ predict path, high negative ≈ predict non-path.
    # Both cases hold TP=1 fixed with exactly one error, so the only difference
    # is FP vs FN — an apples-to-apples comparison of the two penalties.
    HI, LO = 10.0, -10.0
    # Case A: TP=1, one false positive (truth 0, predicted path).
    y_fp = tf.constant([[1.0, 0.0]])
    p_fp = tf.constant([[HI, HI]])
    # Case B: TP=1, one false negative (truth 1, predicted non-path).
    y_fn = tf.constant([[1.0, 1.0]])
    p_fn = tf.constant([[HI, LO]])
    fp_loss = float(tversky_loss(y_fp, p_fp, alpha_fp=0.7, beta_fn=0.3))
    fn_loss = float(tversky_loss(y_fn, p_fn, alpha_fp=0.7, beta_fn=0.3))
    assert fp_loss > fn_loss


def test_tversky_symmetric_at_half_treats_fp_and_fn_equally():
    """alpha_fp=beta_fn=0.5 (Dice-equivalent) ⇒ FP and FN cost the same."""
    HI, LO = 10.0, -10.0
    y_fp = tf.constant([[1.0, 0.0]])
    p_fp = tf.constant([[HI, HI]])
    y_fn = tf.constant([[1.0, 1.0]])
    p_fn = tf.constant([[HI, LO]])
    fp_loss = float(tversky_loss(y_fp, p_fp, alpha_fp=0.5, beta_fn=0.5))
    fn_loss = float(tversky_loss(y_fn, p_fn, alpha_fp=0.5, beta_fn=0.5))
    assert abs(fp_loss - fn_loss) < 1e-4


# ── #8: augmentation on/off ───────────────────────────────────────────────────

def _first_batch_imgs(ds):
    for imgs, _ in ds.take(1):
        return imgs.numpy()


def test_augment_false_is_deterministic(tmp_png_dataset):
    imgs, masks = tmp_png_dataset["imgs"], tmp_png_dataset["masks"]
    kw = dict(batch_size=4, shuffle=False, augment=False, target_size=(H, W))
    a = _first_batch_imgs(make_dataset(imgs, masks, **kw))
    b = _first_batch_imgs(make_dataset(imgs, masks, **kw))
    assert np.allclose(a, b)  # no randomness → identical across builds


def test_augment_true_changes_pixels(tmp_png_dataset):
    imgs, masks = tmp_png_dataset["imgs"], tmp_png_dataset["masks"]
    base = _first_batch_imgs(make_dataset(
        imgs, masks, batch_size=4, shuffle=False, augment=False, target_size=(H, W)))
    tf.random.set_seed(0)
    # Drive the change with a guaranteed flip + colour jitter. max_rotation_deg=0
    # avoids mutating augment_pair's process-global cached rotation layer, which
    # other tests rely on being an identity rotation.
    aug = _first_batch_imgs(make_dataset(
        imgs, masks, batch_size=4, shuffle=False, augment=True, target_size=(H, W),
        flip_prob=1.0, max_rotation_deg=0.0, brightness=0.2, contrast=0.2))
    assert not np.allclose(base, aug)  # augmentation must actually alter the input
