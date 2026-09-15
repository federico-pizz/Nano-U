"""Training entry points: a plain fit loop and the distillation loop."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tf_keras as keras
import yaml

from nanou.architectures import build_from_config
from nanou.config import TrainConfig, resolve
from nanou.data import as_dataset, discover_processed_pairs, make_dataset
from nanou.distill import DistillModel
from nanou.metrics import BinaryIoU, NumpyEncoder
from nanou.qat import apply_qat_to_model
from nanou.redundancy import RedundancyMonitor

# Grow GPU memory on demand instead of reserving the whole card up front, so a
# training run and an evaluation run can share one device.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

# The CV harness passes each fold's file lists along with the overrides. They
# pick the data rather than configure the run, so they are not schema keys:
# `train_model` takes them out before resolving, and uses them when present.
FOLD_KEYS = ("cv_train_img", "cv_train_mask", "cv_val_img", "cv_val_mask")


def set_global_seed(seed: int | None) -> None:
    """Seed Python, NumPy and TensorFlow RNGs for reproducible runs.

    No-op when ``seed`` is None. Combined with a seeded ``make_dataset`` shuffle,
    this makes a training run repeatable given the same config.
    """
    if seed is None:
        return
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_single_model(
    model: keras.Model,
    config: TrainConfig,
    train_data: tf.data.Dataset | tuple[tf.Tensor, tf.Tensor],
    val_data: tf.data.Dataset | tuple[tf.Tensor, tf.Tensor] | None = None,
    experiment_dir: str = "results/",
    extra_callbacks: list | None = None,
) -> keras.callbacks.History:
    """Train a standalone Keras model using standard fit routines.

    This function establishes the environment, optimizer, and callbacks (such as
    ModelCheckpoint and EarlyStopping) to train a single model without distillation.

    Args:
        model: Compiled Keras model instance.
        config: The model's resolved configuration.
        train_data: Dataset mapped as a tf.data.Dataset or an (x, y) tensor tuple.
        val_data: Evaluation dataset mapped as a tf.data.Dataset or (x, y) tensor tuple.
        experiment_dir: Directory the TensorBoard logs and redundancy CSVs are written to.
        extra_callbacks: Callbacks appended after the ones this function builds.

    Returns:
        The keras.callbacks.History object detailing the metrics per epoch.
    """
    # The casts stay: YAML reads an exponent without a dot (5e-4) as a string.
    epochs = int(config.epochs)
    batch_size = int(config.batch_size)
    learning_rate = float(config.learning_rate)

    # Create optimizer
    weight_decay = float(config.weight_decay)
    optimizer_type = config.optimizer.lower()

    if optimizer_type == "adamw" or weight_decay > 0:
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[BinaryIoU(threshold=0.5, from_logits=True)],
    )

    # Setup callbacks
    callbacks = []

    # Early stopping
    if config.early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=int(config.patience), restore_best_weights=True
            )
        )

    # Reduce LR on Plateau
    if config.lr_scheduler.lower() == "plateau":
        patience = int(config.lr_plateau_patience)
        factor = float(config.lr_plateau_factor)
        min_lr = float(config.min_lr)
        print(
            f"Adding ReduceLROnPlateau callback (patience={patience}, factor={factor}, min_lr={min_lr})"
        )
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=factor, patience=patience, min_lr=min_lr, verbose=1
            )
        )

    output_dir = experiment_dir
    # Model checkpoint - always save as temp_model.h5 in models_dir
    models_dir = Path(config.models_dir or "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(models_dir / "temp_model.h5")

    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_best_only=True, monitor="val_loss", mode="min", verbose=1
        )
    )

    # TensorBoard
    if config.tensorboard:
        log_dir = os.path.join(experiment_dir, "logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

    # SVD channel redundancy. Observation only - see nanou.redundancy.
    if config.track_redundancy:
        callbacks.append(
            RedundancyMonitor(
                layers_to_monitor=list(config.redundancy_layers),
                log_frequency=config.redundancy_every_epochs,
                validation_data=val_data,
                output_dir=output_dir,
            )
        )

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # Brief training header (Keras will handle per-epoch progress).
    print(f"\n[TRAIN] {model.name}: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    if weight_decay > 0:
        print(f"[TRAIN] weight_decay={weight_decay}")

    # Fit - accept both tf.data.Dataset and (x, y) tuple
    if isinstance(train_data, tf.data.Dataset):
        return model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1,
        )
    return model.fit(
        train_data[0],
        train_data[1],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=1,
    )


class StudentCheckpoint(keras.callbacks.ModelCheckpoint):
    """`ModelCheckpoint` that writes the wrapped student, not the `DistillModel`.

    `fit` runs on the wrapper, but the artifact every downstream stage wants -
    quantization, export, the firmware - is the student alone. Everything else
    about the callback (monitor, best-only, verbosity) is stock.
    """

    def __init__(self, student: keras.Model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._student = student

    def set_model(self, model):
        """Point the callback at the student, whatever `fit` handed it."""
        super().set_model(self._student)


def _distillation_callbacks(
    student: keras.Model,
    config: TrainConfig,
    checkpoint_path: str,
    has_validation: bool,
    val_dataset: tf.data.Dataset | None,
    experiment_dir: str,
) -> list[keras.callbacks.Callback]:
    """Build the callback list for a distillation run.

    Checkpointing, early stopping and the LR schedule are all stock Keras. They
    replace a hand-written scheduler that had to be kept in step with Keras's by
    hand; `cooldown=0` and `min_delta=0` are spelled out because they are what
    make the stock behaviour match what that code did.
    """
    callbacks: list[keras.callbacks.Callback] = []

    if has_validation:
        callbacks.append(
            StudentCheckpoint(
                student,
                filepath=checkpoint_path,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1,
            )
        )
        if config.early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=int(config.patience),
                    min_delta=0.0,
                    restore_best_weights=True,
                    verbose=1,
                )
            )
        if config.lr_scheduler.lower() == "plateau":
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=float(config.lr_plateau_factor),
                    patience=int(config.lr_plateau_patience),
                    min_lr=float(config.min_lr),
                    min_delta=0.0,
                    cooldown=0,
                    verbose=1,
                )
            )

    if config.track_redundancy:
        callbacks.append(
            RedundancyMonitor(
                layers_to_monitor=list(config.redundancy_layers),
                log_frequency=config.redundancy_every_epochs,
                validation_data=val_dataset,
                output_dir=experiment_dir,
            )
        )

    return callbacks


def train_with_distillation(
    student: keras.Model,
    teacher: keras.Model,
    config: TrainConfig,
    train_data: tuple[tf.Tensor, tf.Tensor] | tf.data.Dataset,
    val_data: tuple[tf.Tensor, tf.Tensor] | tf.data.Dataset | None = None,
    experiment_dir: str = "results/",
) -> dict[str, list[float]]:
    """Train a student by knowledge distillation, through a single `fit` on `DistillModel`.

    The objective lives in `DistillModel`; this function only assembles the
    optimizer, the callbacks and the datasets around it.

    Args:
        student: The lightweight model being trained (QAT-wrapped, if enabled).
        teacher: The pre-trained high-capacity model providing soft targets.
        config: The student's resolved configuration - alpha, temperature,
            t_scale, the Tversky blend, the optimizer settings and the
            callback thresholds.
        train_data: Training dataset, or an (x, y) tuple.
        val_data: Validation dataset, or an (x, y) tuple. Checkpointing, early
            stopping and the LR schedule all need it and are skipped without it.
        experiment_dir: Where per-run logs are written.

    Returns:
        The `fit` history as a plain dict of metric name -> per-epoch values.
    """
    alpha = float(config.alpha)
    temperature = float(config.temperature)
    # Conservative (precision-favoring) supervised loss for the student. Off by
    # default (weight 0.0 -> pure BCE); set `tversky_weight` in the preset to
    # blend in the Tversky term. Only the student reads these - BU-Net trains
    # through the plain Keras path and never sees them.
    tversky_weight = float(config.tversky_weight)
    tversky_alpha = float(config.tversky_alpha)
    tversky_beta = float(config.tversky_beta)
    t_scale = str(config.t_scale)

    learning_rate = float(config.learning_rate)
    weight_decay = float(config.weight_decay)
    optimizer_type = config.optimizer.lower()
    if optimizer_type == "adamw" or weight_decay > 0:
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    epochs = int(config.epochs)
    batch_size = int(config.batch_size)

    models_dir = Path(config.models_dir or "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(models_dir / "temp_model.h5")

    print(
        f"\n[TRAIN-KD] student='{student.name}' teacher='{teacher.name}' "
        f"epochs={epochs} batch_size={batch_size} "
        f"alpha={alpha} T={temperature} lr={learning_rate}"
    )
    if weight_decay > 0:
        print(f"[TRAIN-KD] weight_decay={weight_decay}")
    if t_scale != "squared":
        print(f"[TRAIN-KD] distillation term scaling: {t_scale} (paper uses T^2)")
    if tversky_weight > 0.0:
        print(
            f"[TRAIN-KD] supervised loss = {(1 - tversky_weight):.2f}*BCE + "
            f"{tversky_weight:.2f}*Tversky(alpha_fp={tversky_alpha}, "
            f"beta_fn={tversky_beta}) [conservative student]"
        )

    train_dataset, val_dataset = as_dataset(train_data, val_data, batch_size)

    distiller = DistillModel(
        student=student,
        teacher=teacher,
        alpha=alpha,
        temperature=temperature,
        tversky_weight=tversky_weight,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        t_scale=t_scale,
    )
    distiller.compile(optimizer=optimizer)

    callbacks = _distillation_callbacks(
        student=student,
        config=config,
        checkpoint_path=checkpoint_path,
        has_validation=val_dataset is not None,
        val_dataset=val_dataset,
        experiment_dir=experiment_dir,
    )

    history = distiller.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1,
    )

    history_dict = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    if history_dict.get("loss"):
        msg = (
            f"[TRAIN-KD] completed {len(history_dict['loss'])} epochs, "
            f"final_loss={history_dict['loss'][-1]:.4f}"
        )
        if history_dict.get("val_loss"):
            msg += f", final_val_loss={history_dict['val_loss'][-1]:.4f}"
        print(msg)

    return history_dict


def train_model(
    config_path: str = "config/presets/tinyagri.yaml",
    experiment_name: str = "nanou",
    output_dir: str | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one model block of a preset and save it.

    Args:
        config_path: The preset file.
        experiment_name: The model block to train. It also names the output files.
        output_dir: Where the history and the run's config are written. Defaults
            to ``<output_dir>/<model name>/``, with the preset's output_dir.
        extra_config: Keys applied on top of the preset, so they win over it
            (e.g. teacher_weights resolved at runtime). They must be schema
            keys, apart from the fold file lists in `FOLD_KEYS`.

    Returns:
        The final history, the saved model's path and name, the experiment directory
        and whether QAT was applied. Any failure propagates as an exception.
    """
    overrides = dict(extra_config or {})
    fold = {key: overrides.pop(key) for key in FOLD_KEYS if key in overrides}
    resolved = resolve(path=config_path, overrides=overrides)
    config = resolved.for_model(experiment_name)

    # Seed all RNGs up front so the run is reproducible. A caller's seed is an
    # override like any other, so it is already in the resolved config.
    set_global_seed(config.seed)

    if output_dir is not None:
        experiment_dir = Path(output_dir)
    else:
        experiment_dir = Path(config.output_dir or "results/") / config.model_name

    experiment_dir.mkdir(parents=True, exist_ok=True)

    # A caller can point models_dir elsewhere - the CV harness gives every fold
    # its own - so concurrent runs never clobber a shared temp_model.h5 or
    # <model>.h5.
    if config.models_dir is None:
        config = replace(config, models_dir="models")
    models_dir = Path(config.models_dir)

    # Injected fold file lists bypass directory discovery, so a leakage-safe
    # grouped split trains through the exact same code path as a normal run.
    # Guarded by key presence: a normal run never sets them.
    if fold.get("cv_train_img") and fold.get("cv_train_mask"):
        train_img_files = list(fold["cv_train_img"])
        train_mask_files = list(fold["cv_train_mask"])
        val_img_files = list(fold.get("cv_val_img") or [])
        val_mask_files = list(fold.get("cv_val_mask") or [])
        print(
            f"[CV] using injected fold file-lists: "
            f"{len(train_img_files)} train / {len(val_img_files)} val pairs."
        )
    else:
        print("Loading data from processed paths...")
        (train_img_files, train_mask_files, val_img_files, val_mask_files) = (
            discover_processed_pairs(config.split_paths)
        )

    print(f"Found {len(train_img_files)} training pairs.")
    if val_img_files:
        print(f"Found {len(val_img_files)} validation pairs.")
    else:
        print("Validation data skipped (mismatch, empty, or no val dirs).")

    norm = resolved.normalization
    target_size = config.input_shape[:2]

    # augment_params lets the CV harness sweep the *kind* of augmentation
    # (geometric / photometric / full); ignored when augment is False.
    train_ds = make_dataset(
        train_img_files,
        train_mask_files,
        batch_size=config.batch_size,
        augment=config.augment,
        target_size=target_size,
        mean=norm["mean"],
        std=norm["std"],
        seed=config.seed,
        **config.augment_params,
    )
    val_ds: tf.data.Dataset | None = None
    if val_img_files:
        val_ds = make_dataset(
            val_img_files,
            val_mask_files,
            batch_size=config.batch_size,
            augment=False,
            shuffle=False,
            target_size=target_size,
            mean=norm["mean"],
            std=norm["std"],
            seed=config.seed,
        )

    model_to_save = None

    # Build models
    if config.use_distillation:
        # The teacher is built from its own block in the same preset.
        teacher_name = config.teacher_model_name
        print(f"[KD] Building teacher '{teacher_name}'")
        teacher = build_from_config(resolved.for_model(teacher_name))

        teacher_weights = config.teacher_weights
        if teacher_weights and Path(teacher_weights).exists():
            print(f"[KD] Loading teacher weights from {teacher_weights}")
            teacher.load_weights(teacher_weights)
        elif teacher_weights:
            print(
                f"[KD] Teacher weights not found at {teacher_weights}; using random initialization."
            )

        student = build_from_config(config)

        # Apply QAT (Quantization-Aware Training) to the student model.
        # apply_qat_to_model raises on failure: QAD requires fake-quant nodes
        # from epoch 1, so a silent float fallback must not be possible.
        is_qat_model = config.qat_enabled
        if is_qat_model:
            print("Applying Quantization-Aware Training (QAT) to student model...")
            student = apply_qat_to_model(student)

        history = train_with_distillation(
            student, teacher, config, train_ds, val_ds, experiment_dir=str(experiment_dir)
        )
        model_to_save = student
    else:
        model = build_from_config(config)

        # Apply QAT only to nanou (the student / deployment model).
        # TFMOT requires a Sequential or Functional model; bu_net and other
        # custom subclasses are not compatible and must skip QAT.
        is_qat_model = config.qat_enabled and config.model_name == "nanou"
        if is_qat_model:
            print("Applying Quantization-Aware Training (QAT) to model...")
            model = apply_qat_to_model(model)

        history = train_single_model(
            model, config, train_ds, val_ds, experiment_dir=str(experiment_dir)
        )
        model_to_save = model

    # Restore best weights before saving if they exist
    temp_checkpoint = str(models_dir / "temp_model.h5")
    if os.path.exists(temp_checkpoint):
        print(f"Loading best weights from {temp_checkpoint} for final save...")
        model_to_save.load_weights(temp_checkpoint)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{config.model_name}.h5"

    # Save the trained model. QAT models are saved with their quantization
    # wrappers intact; the TFLite converter in quantize_model.py then applies
    # PTQ on top of the QAT-trained weights to produce the final INT8 model.
    model_to_save.save(model_path)
    if is_qat_model:
        print(f"QAT model saved to {model_path}")

    if os.path.exists(temp_checkpoint):
        os.remove(temp_checkpoint)

    history_path = experiment_dir / "history.json"
    history_dict = history.history if hasattr(history, "history") else history

    # Ensure all values are standard Python floats for JSON serialization
    serializable_history = {}
    for key, values in history_dict.items():
        serializable_history[key] = [float(v) for v in values]

    with open(history_path, "w") as f:
        json.dump(serializable_history, f, indent=2, cls=NumpyEncoder)

    # What the run actually trained with, fold lists included, so a run
    # directory says on its own where its numbers came from.
    record = {**asdict(config), "input_shape": list(config.input_shape), **fold}
    with open(experiment_dir / "config.yaml", "w") as f:
        yaml.safe_dump(record, f)

    return {
        "final_metrics": serializable_history,
        "model_path": str(model_path),
        "model_name": config.model_name,
        "experiment_dir": str(experiment_dir),
        # Recorded so results.json states whether the saved .h5 actually
        # carries fake-quantization wrappers, rather than leaving it to be
        # inferred from the config.
        "qat_applied": bool(is_qat_model),
    }
