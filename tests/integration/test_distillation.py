"""Integration tests for the distillation path: `DistillModel` under a real `fit`.

Distillation is the most intricate piece of the training code, so these run it
the way training does - through `train_with_distillation` on the synthetic
on-disk dataset - and check the things downstream stages depend on: finite
losses, weights that move, the eight history keys, and a checkpoint on disk
holding the student rather than the wrapper.
"""

import numpy as np
import tf_keras as keras

from nanou.architectures import create_nano_u
from nanou.data import make_dataset
from nanou.distill import DistillModel
from nanou.train import train_with_distillation

H, W = 60, 80

HISTORY_KEYS = {
    "loss",
    "student_loss",
    "distillation_loss",
    "binary_iou",
    "val_loss",
    "val_student_loss",
    "val_distillation_loss",
    "val_binary_iou",
}


def _student():
    return create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)


def _teacher():
    return create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16, name="teacher")


def _dataset(tmp_png_dataset, batch_size=4):
    return make_dataset(
        tmp_png_dataset["imgs"],
        tmp_png_dataset["masks"],
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        target_size=(H, W),
    )


def _first_batch(tmp_png_dataset, batch_size=4):
    return next(iter(_dataset(tmp_png_dataset, batch_size)))


def _config(tmp_path, **overrides):
    config = {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "alpha": 0.5,
        "temperature": 4.0,
        "models_dir": str(tmp_path / "models"),
    }
    config.update(overrides)
    return config


def test_train_step_returns_finite_losses(tmp_png_dataset):
    distiller = DistillModel(_student(), _teacher(), alpha=0.5, temperature=4.0)
    distiller.compile(optimizer=keras.optimizers.Adam(1e-3))
    x, y = _first_batch(tmp_png_dataset)

    out = {k: float(v) for k, v in distiller.train_step((x, y)).items()}

    for key in ("loss", "student_loss", "distillation_loss"):
        assert np.isfinite(out[key]), f"{key} not finite"
    assert out["distillation_loss"] >= 0.0
    assert 0.0 <= out["binary_iou"] <= 1.0


def test_train_step_updates_student_weights(tmp_png_dataset):
    distiller = DistillModel(_student(), _teacher(), alpha=0.5, temperature=4.0)
    distiller.compile(optimizer=keras.optimizers.Adam(1e-2))
    x, y = _first_batch(tmp_png_dataset)

    before = [w.numpy().copy() for w in distiller.student.trainable_variables]
    for _ in range(3):
        distiller.train_step((x, y))
    after = [w.numpy() for w in distiller.student.trainable_variables]

    assert any(not np.allclose(a, b) for a, b in zip(before, after, strict=True)), (
        "no student weight changed after distillation steps"
    )


def test_fit_produces_the_contract_history_keys(tmp_png_dataset, tmp_path):
    """The eight keys `pipeline` and `results.json` read must all be present."""
    dataset = _dataset(tmp_png_dataset)
    history = train_with_distillation(
        _student(),
        _teacher(),
        _config(tmp_path),
        dataset,
        dataset,
        experiment_dir=str(tmp_path / "run"),
    )

    assert HISTORY_KEYS <= set(history)
    assert all(len(history[k]) == 2 for k in HISTORY_KEYS)
    assert all(np.isfinite(v) for k in HISTORY_KEYS for v in history[k])


def test_fit_checkpoints_the_student_not_the_wrapper(tmp_png_dataset, tmp_path):
    """`temp_model.h5` has to be loadable as the student, since that is what gets exported."""
    dataset = _dataset(tmp_png_dataset)
    student = _student()
    train_with_distillation(
        student,
        _teacher(),
        _config(tmp_path),
        dataset,
        dataset,
        experiment_dir=str(tmp_path / "run"),
    )

    checkpoint = tmp_path / "models" / "temp_model.h5"
    assert checkpoint.exists()
    # Loading into a freshly built student proves the file holds the student's
    # topology and not the DistillModel wrapper's.
    _student().load_weights(str(checkpoint))


def test_fit_leaves_the_teacher_untouched(tmp_png_dataset, tmp_path):
    dataset = _dataset(tmp_png_dataset)
    teacher = _teacher()
    before = [w.numpy().copy() for w in teacher.weights]

    train_with_distillation(
        _student(),
        teacher,
        _config(tmp_path),
        dataset,
        dataset,
        experiment_dir=str(tmp_path / "run"),
    )

    assert all(
        np.array_equal(a, b.numpy()) for a, b in zip(before, teacher.weights, strict=True)
    ), "distillation modified the teacher"
