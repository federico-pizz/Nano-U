"""Smoke tests for the knowledge-distillation training step (src/train.py).

The custom GradientTape distillation loop is the most intricate piece of the
training code, so these tests exercise `train_step` directly on the synthetic
on-disk dataset: losses must be finite, the distillation term well-behaved, and
the student's weights must actually move under optimization.
"""

import numpy as np
import pytest
import tf_keras as keras

from src.data import make_dataset
from src.models.builders import create_nano_u
from src.train import train_step

H, W = 60, 80


def _student():
    return create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)


def _teacher():
    return create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16, name="teacher")


def _first_batch(tmp_png_dataset, batch_size=4):
    ds = make_dataset(
        tmp_png_dataset["imgs"], tmp_png_dataset["masks"],
        batch_size=batch_size, shuffle=False, augment=False, target_size=(H, W),
    )
    return next(iter(ds))


def _built_optimizer(model, lr):
    """Adam with its slot variables pre-built.

    `train_step` is a shared module-level ``@tf.function``; building the
    optimizer's variables here (outside the traced function) keeps each test
    independent of trace order, since the function then creates no variables.
    """
    optimizer = keras.optimizers.Adam(lr)
    optimizer.build(model.trainable_variables)
    return optimizer


def test_train_step_returns_finite_losses(tmp_png_dataset):
    student, teacher = _student(), _teacher()
    optimizer = _built_optimizer(student, 1e-3)
    x, y = _first_batch(tmp_png_dataset)

    out = train_step(student, teacher, x, y, optimizer, alpha=0.5, temperature=4.0)

    for key in ("loss", "student_loss", "distillation_loss"):
        assert np.isfinite(out[key].numpy()), f"{key} not finite"
    assert out["distillation_loss"].numpy() >= 0.0
    # Confusion-matrix counts must sum to the number of pixels in the batch.
    total = sum(out[k].numpy() for k in ("tp", "fp", "fn", "tn"))
    assert total == pytest.approx(x.shape[0] * H * W, rel=1e-4)


def test_train_step_updates_student_weights(tmp_png_dataset):
    student, teacher = _student(), _teacher()
    optimizer = _built_optimizer(student, 1e-2)
    x, y = _first_batch(tmp_png_dataset)

    before = [w.numpy().copy() for w in student.trainable_variables]
    for _ in range(3):
        train_step(student, teacher, x, y, optimizer, alpha=0.5, temperature=4.0)
    after = [w.numpy() for w in student.trainable_variables]

    assert any(not np.allclose(a, b) for a, b in zip(before, after)), \
        "no student weight changed after distillation steps"


def test_train_step_without_teacher_is_pure_bce(tmp_png_dataset):
    """teacher=None must zero the distillation term and still train."""
    student = _student()
    optimizer = _built_optimizer(student, 1e-3)
    x, y = _first_batch(tmp_png_dataset)

    out = train_step(student, None, x, y, optimizer, alpha=0.5, temperature=4.0)

    assert out["distillation_loss"].numpy() == 0.0
    assert np.isfinite(out["loss"].numpy())
