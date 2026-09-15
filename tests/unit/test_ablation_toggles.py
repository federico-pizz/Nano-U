"""Ablation-toggle mechanics (#7 CE-loss on/off, #8 augmentation on/off).

These pin the behavior the CV sweep relies on:
  * CE off is exactly ``alpha=1.0`` in the KD loss ``(1-alpha)*CE + alpha*distill``;
  * augmentation off is deterministic, augmentation on changes pixels.

They also hold the ``DistillModel`` contract: the teacher stays frozen, the
``T**2`` factor is exactly what the ``t_scale`` knob switches, ``test_step``
agrees with ``train_step``, and the four history keys do not move.
"""

import numpy as np
import pytest
import tensorflow as tf
import tf_keras as keras

from nanou.architectures import create_nano_u
from nanou.data import make_dataset
from nanou.distill import DistillModel
from nanou.losses import tversky_loss

H, W = 60, 80


def _student_teacher():
    s = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    t = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    return s, t


def _distiller(alpha=0.3, temperature=4.0, t_scale="squared"):
    """A compiled DistillModel over two throwaway students."""
    student, teacher = _student_teacher()
    d = DistillModel(
        student=student, teacher=teacher, alpha=alpha, temperature=temperature, t_scale=t_scale
    )
    d.compile(optimizer=keras.optimizers.Adam(1e-3))
    return d


def _one_step(distiller, x, y):
    """Run one training step eagerly and return the metric dict as floats."""
    logs = distiller.train_step((x, y))
    return {k: float(v) for k, v in logs.items()}


# -- #7: CE-loss on/off via alpha ----------------------------------------------


def test_ce_off_alpha_one_total_equals_distillation(img_tensor, mask_tensor):
    """alpha=1.0 -> CE coefficient (1-alpha)=0, so total loss == distillation loss."""
    out = _one_step(_distiller(alpha=1.0), img_tensor, mask_tensor)
    assert abs(out["loss"] - out["distillation_loss"]) < 1e-5
    assert out["student_loss"] > 0.0  # CE still computed, just not weighted in


def test_ce_on_contributes_to_total(img_tensor, mask_tensor):
    """alpha=0.5 -> total == 0.5*CE + 0.5*distill, distinct from distill alone."""
    out = _one_step(_distiller(alpha=0.5), img_tensor, mask_tensor)
    expected = 0.5 * out["student_loss"] + 0.5 * out["distillation_loss"]
    assert abs(out["loss"] - expected) < 1e-4
    assert abs(out["loss"] - out["distillation_loss"]) > 1e-6


# -- the T^2 factor on the distillation term (PAPER_QUESTIONS B2) ---------------


def test_distillation_term_carries_the_t_squared_factor(img_tensor, mask_tensor):
    """For fixed logits the KD term scales as T^2 relative to the unscaled term.

    Both branches of the `t_scale` knob are pinned here: `squared` must be
    exactly `T**2` times `none` on the same inputs.
    """
    teacher_logits = tf.constant([[2.0, -1.0, 0.5]])
    student_logits = tf.constant([[-0.5, 1.5, 0.0]])

    for temperature in (0.7, 4.0):
        scaled = _distiller(temperature=temperature, t_scale="squared")
        plain = _distiller(temperature=temperature, t_scale="none")
        with_factor = float(scaled._distillation_loss(teacher_logits, student_logits))
        without = float(plain._distillation_loss(teacher_logits, student_logits))
        assert abs(with_factor - without * temperature**2) < 1e-6 * max(1.0, without)


def test_t_scale_below_one_shrinks_the_distillation_term(img_tensor, mask_tensor):
    """T=0.7 sharpens the teacher, so T^2 down-weights KD instead of compensating."""
    teacher_logits = tf.constant([[2.0, -1.0, 0.5]])
    student_logits = tf.constant([[-0.5, 1.5, 0.0]])
    scaled = _distiller(temperature=0.7, t_scale="squared")
    plain = _distiller(temperature=0.7, t_scale="none")
    assert float(scaled._distillation_loss(teacher_logits, student_logits)) < float(
        plain._distillation_loss(teacher_logits, student_logits)
    )


def test_unknown_t_scale_is_rejected():
    """A typo in the knob must fail at construction, not train silently."""
    student, teacher = _student_teacher()
    with pytest.raises(ValueError, match="t_scale"):
        DistillModel(student=student, teacher=teacher, t_scale="quadratic")


# -- the teacher stays frozen ---------------------------------------------------


def test_teacher_is_not_trainable(img_tensor, mask_tensor):
    """The teacher must not appear in the variables the optimizer updates."""
    d = _distiller()
    teacher_ids = {id(v) for v in d.teacher.weights}
    assert not any(id(v) in teacher_ids for v in d.trainable_variables)
    assert len(d.trainable_variables) == len(d.student.trainable_variables)


def test_teacher_weights_unchanged_by_a_step(img_tensor, mask_tensor):
    """One training step moves the student and leaves the teacher exactly as it was."""
    d = _distiller()
    before_teacher = [w.numpy().copy() for w in d.teacher.weights]
    before_student = [w.numpy().copy() for w in d.student.trainable_variables]
    _one_step(d, img_tensor, mask_tensor)
    assert all(
        np.array_equal(a, b.numpy()) for a, b in zip(before_teacher, d.teacher.weights, strict=True)
    )
    assert any(
        not np.allclose(a, b.numpy())
        for a, b in zip(before_student, d.student.trainable_variables, strict=True)
    )


# -- train and validation share one objective -----------------------------------


def test_validation_objective_is_the_training_objective(img_tensor, mask_tensor):
    """test_step must report the documented objective, recomputed here by hand.

    This is the whole point of the class. The validation loss used to be a
    second, hand-written implementation of the same maths in eager code, free to
    drift from what training optimized; now both come from `_losses_for` and the
    only difference between them is the gradient and the BatchNorm mode.
    """
    alpha, temperature = 0.4, 2.0
    d = _distiller(alpha=alpha, temperature=temperature)
    d.reset_metrics()
    reported = {k: float(v) for k, v in d.test_step((img_tensor, mask_tensor)).items()}

    student_logits = d.student(img_tensor, training=False)
    teacher_logits = d.teacher(img_tensor, training=False)
    bce = float(keras.losses.BinaryCrossentropy(from_logits=True)(mask_tensor, student_logits))
    distill = float(
        tf.reduce_mean(
            tf.math.squared_difference(
                tf.nn.sigmoid(teacher_logits / temperature),
                tf.nn.sigmoid(student_logits / temperature),
            )
        )
    ) * (temperature**2)

    assert abs(reported["student_loss"] - bce) < 1e-5
    assert abs(reported["distillation_loss"] - distill) < 1e-5
    assert abs(reported["loss"] - ((1 - alpha) * bce + alpha * distill)) < 1e-5


def test_train_and_test_step_report_the_same_keys(img_tensor, mask_tensor):
    """Both steps feed the same trackers, so history and val-history line up.

    The values differ - BatchNorm runs in training mode on one side - but the
    set of keys is what `pipeline` and `results.json` depend on.
    """
    d = _distiller()
    d.reset_metrics()
    val_keys = set(d.test_step((img_tensor, mask_tensor)).keys())
    d.reset_metrics()
    train_keys = set(_one_step(d, img_tensor, mask_tensor).keys())
    assert val_keys == train_keys


def test_history_keys_are_the_contract(img_tensor, mask_tensor):
    """`loss`, `student_loss`, `distillation_loss`, `binary_iou` - read by results.json."""
    d = _distiller()
    assert [m.name for m in d.metrics] == [
        "loss",
        "student_loss",
        "distillation_loss",
        "binary_iou",
    ]


# -- conservative Tversky supervised loss --------------------------------------


def test_tversky_penalizes_false_positive_more_than_false_negative():
    """alpha_fp>beta_fn => a false 'path' pixel costs more than a missed one.

    This is the safety property: for path=1, a false positive is a potential
    collision, a false negative is just over-caution. Construct a single-FP case
    and a single-FN case with everything else correct and assert FP loss > FN.
    """
    # logits -> high positive ~ predict path, high negative ~ predict non-path.
    # Both cases hold TP=1 fixed with exactly one error, so the only difference
    # is FP vs FN - an apples-to-apples comparison of the two penalties.
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
    """alpha_fp=beta_fn=0.5 (Dice-equivalent) => FP and FN cost the same."""
    HI, LO = 10.0, -10.0
    y_fp = tf.constant([[1.0, 0.0]])
    p_fp = tf.constant([[HI, HI]])
    y_fn = tf.constant([[1.0, 1.0]])
    p_fn = tf.constant([[HI, LO]])
    fp_loss = float(tversky_loss(y_fp, p_fp, alpha_fp=0.5, beta_fn=0.5))
    fn_loss = float(tversky_loss(y_fn, p_fn, alpha_fp=0.5, beta_fn=0.5))
    assert abs(fp_loss - fn_loss) < 1e-4


# -- #8: augmentation on/off ---------------------------------------------------


def _first_batch_imgs(ds):
    for imgs, _ in ds.take(1):
        return imgs.numpy()


def test_augment_false_is_deterministic(tmp_png_dataset):
    imgs, masks = tmp_png_dataset["imgs"], tmp_png_dataset["masks"]
    kw = dict(batch_size=4, shuffle=False, augment=False, target_size=(H, W))
    a = _first_batch_imgs(make_dataset(imgs, masks, **kw))
    b = _first_batch_imgs(make_dataset(imgs, masks, **kw))
    assert np.allclose(a, b)  # no randomness -> identical across builds


def test_augment_true_changes_pixels(tmp_png_dataset):
    imgs, masks = tmp_png_dataset["imgs"], tmp_png_dataset["masks"]
    base = _first_batch_imgs(
        make_dataset(imgs, masks, batch_size=4, shuffle=False, augment=False, target_size=(H, W))
    )
    tf.random.set_seed(0)
    # Drive the change with a guaranteed flip + colour jitter. max_rotation_deg=0
    # avoids mutating augment_pair's process-global cached rotation layer, which
    # other tests rely on being an identity rotation.
    aug = _first_batch_imgs(
        make_dataset(
            imgs,
            masks,
            batch_size=4,
            shuffle=False,
            augment=True,
            target_size=(H, W),
            flip_prob=1.0,
            max_rotation_deg=0.0,
            brightness=0.2,
            contrast=0.2,
        )
    )
    assert not np.allclose(base, aug)  # augmentation must actually alter the input
