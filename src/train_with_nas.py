"""
NAS-enabled training entrypoint based on src/train.py.

This file mirrors the canonical training flow but adds optional NAS integration
via wrapping models in training-time subclasses that include the covariance
regularizer loss (computed using ActivationExtractor and covariance_regularizer_loss)
so that training can proceed with model.fit while keeping CLI parity with train.py.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Ensure project root is importable when executed directly
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.Nano_U.model_tf import build_nano_u
from src.models.BU_Net.model_tf import build_bu_net
from src.utils import make_dataset, BinaryIoU, get_project_root
from src.utils.config import load_config
from src.nas_covariance import ActivationExtractor, covariance_regularizer_loss

# GPU memory growth note
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s). Training will use GPU.")
    except RuntimeError as e:
        print(f"⚠ GPU configuration error: {e}")
else:
    print("⚠ WARNING: No GPU detected. Training will use CPU.")


class NASWrapper(keras.Model):
    """Wrap a base Keras model to add NAS covariance regularizer into train_step.

    This wrapper keeps the normal model behavior but computes the covariance
    regularizer on activations extracted by ActivationExtractor and adds it to
    the supervised loss using nas_weight.
    """
    def __init__(self, base_model, extractor: ActivationExtractor, nas_weight: float, loss_fn, metrics=None):
        super(NASWrapper, self).__init__()
        self.base_model = base_model
        self.extractor = extractor
        self.nas_weight = tf.cast(nas_weight, tf.float32)
        self.loss_fn = loss_fn
        self._user_metrics = list(metrics) if metrics is not None else []
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")

    @property
    def metrics(self):
        # Exclude Keras internal CompileMetrics wrapper if present (it has name 'compile_metrics')
        user_metrics = [m for m in self._user_metrics if getattr(m, 'name', '') != 'compile_metrics']
        return [self.total_loss_tracker, self.student_loss_tracker] + list(user_metrics)

    def call(self, x):
        return self.base_model(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            student_loss = self.loss_fn(y, y_pred)
            student_loss = tf.reduce_mean(student_loss)
            acts = self.extractor(x, training=True)
            reg_loss = covariance_regularizer_loss(acts, weights=None, normalize=True)
            total_loss = student_loss + self.nas_weight * reg_loss
        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))

        self.student_loss_tracker.update_state(student_loss)
        self.total_loss_tracker.update_state(total_loss)
        for m in self._user_metrics:
            try:
                m.update_state(y, y_pred)
            except Exception:
                pass
        # Debug: inspect metrics before returning results
        for mm in self.metrics:
            try:
                print(f"METRIC DEBUG: name={mm.name}, type={type(mm)}, is_metric={isinstance(mm, tf.keras.metrics.Metric)}, variables={getattr(mm, 'variables', None)}")
            except Exception as _e:
                print(f"METRIC DEBUG: failed to inspect metric {mm}: {_e}")
        results = {}
        for m in self.metrics:
            if getattr(m, 'name', '') == 'compile_metrics':
                continue
            try:
                results[m.name] = m.result()
            except Exception as _e:
                print(f"METRIC RESULT ERROR: {m} -> {_e}")
                results[m.name] = None
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self.base_model(x, training=False)
        student_loss = self.loss_fn(y, y_pred)
        self.total_loss_tracker.update_state(student_loss)
        for m in self._user_metrics:
            try:
                m.update_state(y, y_pred)
            except Exception:
                pass
        # Debug: inspect metrics before returning results
        for mm in self.metrics:
            try:
                print(f"METRIC DEBUG: name={mm.name}, type={type(mm)}, is_metric={isinstance(mm, tf.keras.metrics.Metric)}, variables={getattr(mm, 'variables', None)}")
            except Exception as _e:
                print(f"METRIC DEBUG: failed to inspect metric {mm}: {_e}")
        results = {}
        for m in self.metrics:
            if getattr(m, 'name', '') == 'compile_metrics':
                continue
            try:
                results[m.name] = m.result()
            except Exception as _e:
                print(f"METRIC RESULT ERROR: {m} -> {_e}")
                results[m.name] = None
        return results


class DistillerWithNAS(keras.Model):
    """Extension of Distiller that adds NAS regularizer during distillation training."""
    def __init__(self, student, teacher, extractor: ActivationExtractor, nas_weight: float):
        super(DistillerWithNAS, self).__init__()
        self.student = student
        self.teacher = teacher
        self.extractor = extractor
        self.nas_weight = tf.cast(nas_weight, tf.float32)
        self._user_metrics = []
        # trackers
        self.distillation_loss_tracker = keras.metrics.Mean(name="distill_loss")
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.5, temperature=2.0):
        # Store user metrics but avoid passing them to super().compile to prevent Keras from
        # wrapping them in CompileMetrics which isn't compatible with our custom Model wrappers.
        self._user_metrics = list(metrics) if metrics is not None else []
        super(DistillerWithNAS, self).compile(optimizer=optimizer, metrics=None)
        # Manually ensure metric variables are created by calling update_state once with zeros
        for m in self._user_metrics:
            try:
                # Use safe initialization: if metric has update_state, call with zeros to create variables
                m.update_state(tf.zeros((1, 1)), tf.zeros((1, 1)))
            except Exception:
                pass
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    @property
    def metrics(self):
        # Exclude Keras internal CompileMetrics wrapper if present
        user_metrics = [m for m in self._user_metrics if getattr(m, 'name', '') != 'compile_metrics']
        return [self.total_loss_tracker, self.distillation_loss_tracker, self.student_loss_tracker] + list(user_metrics)

    def train_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)

            student_soft = tf.math.sigmoid(student_predictions / self.temperature)
            teacher_soft = tf.math.sigmoid(teacher_predictions / self.temperature)
            dist_loss = self.distillation_loss_fn(student_soft, teacher_soft)

            loss = self.alpha * student_loss + (1 - self.alpha) * dist_loss

            # NAS regularizer contribution
            acts = self.extractor(x, training=True)
            reg_loss = covariance_regularizer_loss(acts, weights=None, normalize=True)
            loss = loss + self.nas_weight * reg_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(dist_loss)
        self.total_loss_tracker.update_state(loss)

        for m in self._user_metrics:
            try:
                m.update_state(y, student_predictions)
            except Exception:
                pass

        results = {}
        for m in self.metrics:
            if getattr(m, 'name', '') == 'compile_metrics':
                continue
            try:
                results[m.name] = m.result()
            except Exception as _e:
                print(f"METRIC RESULT ERROR: {m} -> {_e}")
                results[m.name] = None
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)
        self.total_loss_tracker.update_state(student_loss)
        for m in self._user_metrics:
            try:
                m.update_state(y, y_pred)
            except Exception:
                pass
        results = {}
        for m in self.metrics:
            if getattr(m, 'name', '') == 'compile_metrics':
                continue
            try:
                results[m.name] = m.result()
            except Exception as _e:
                print(f"METRIC RESULT ERROR: {m} -> {_e}")
                results[m.name] = None
        return results

    def call(self, x):
        return self.student(x)


class SaveStudentCallback(keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_binary_iou', save_best_only=True, mode='max'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = -np.inf if mode == 'max' else np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = None if logs is None else logs.get(self.monitor)
        if current is None:
            return
        is_better = (current > self.best) if self.mode == 'max' else (current < self.best)
        if (self.save_best_only and is_better) or not self.save_best_only:
            if self.save_best_only:
                self.best = current
            # model may be DistillerWithNAS or NASWrapper; save student if present
            try:
                if hasattr(self.model, 'student'):
                    self.model.student.save(self.filepath)
                elif hasattr(self.model, 'base_model'):
                    self.model.base_model.save(self.filepath)
                else:
                    self.model.save(self.filepath)
                print(f"\nSaved model to {self.filepath}")
            except Exception as e:
                print(f"Warning: failed to save model {e}")


def build_model_from_config(name: str, config: dict):
    input_shape = tuple(config["data"]["input_shape"])
    if name.lower() == "nano_u":
        cfg = config["models"]["nano_u"]
        return build_nano_u(
            input_shape=input_shape,
            filters=cfg["filters"],
            bottleneck=cfg["bottleneck"],
            decoder_filters=cfg["decoder_filters"]
        )
    elif name.lower() == "bu_net":
        cfg = config["models"]["bu_net"]
        return build_bu_net(
            input_shape=input_shape,
            filters=cfg["filters"],
            bottleneck=cfg["bottleneck"],
            decoder_filters=cfg["decoder_filters"]
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def train(model_name="nano_u", epochs=None, batch_size=None, lr=None,
          distill=False, teacher_weights=None, alpha=None, temperature=None,
          augment=True, config_path="config/config.yaml",
          enable_nas=False, nas_layers=None, nas_weight=0.01):

    config = load_config(config_path)
    train_cfg = config["training"].get(model_name, {})
    epochs = epochs if epochs is not None else train_cfg.get("epochs", 50)
    batch_size = batch_size if batch_size is not None else train_cfg.get("batch_size", 8)
    lr = lr if lr is not None else float(train_cfg.get("learning_rate", 1e-4))

    if distill:
        distill_cfg = train_cfg.get("distillation", {})
        if alpha is None: alpha = distill_cfg.get("alpha", 0.3)
        if temperature is None: temperature = distill_cfg.get("temperature", 3.0)
        if teacher_weights is None: teacher_weights = distill_cfg.get("teacher_weights", None)

    # Resolve teacher_weights if provided as relative path; attempt to auto-detect canonical bu_net checkpoint when distilling
    if teacher_weights and not os.path.exists(teacher_weights):
        root_teacher = os.path.join(str(get_project_root()), teacher_weights)
        if os.path.exists(root_teacher):
            teacher_weights = root_teacher

    if distill and not teacher_weights:
        root_dir = str(get_project_root())
        candidate_models_dir = config["data"]["paths"].get("models_dir", "models")
        candidate_models_dir = candidate_models_dir if os.path.isabs(candidate_models_dir) else os.path.join(root_dir, candidate_models_dir)
        candidate_teacher = os.path.join(candidate_models_dir, "bu_net.keras")
        if os.path.exists(candidate_teacher):
            teacher_weights = candidate_teacher

    root_dir = str(get_project_root())
    processed_paths = config["data"]["paths"]["processed"]
    def resolve_path(p): return p if os.path.isabs(p) else os.path.join(root_dir, p)

    train_img_dir = resolve_path(processed_paths["train"]["img"])
    train_mask_dir = resolve_path(processed_paths["train"]["mask"])
    val_img_dir = resolve_path(processed_paths["val"]["img"])
    val_mask_dir = resolve_path(processed_paths["val"]["mask"])

    train_img_files = sorted([os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.png')])
    train_mask_files = sorted([os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir) if f.endswith('.png')])
    val_img_files = sorted([os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.png')])
    val_mask_files = sorted([os.path.join(val_mask_dir, f) for f in os.listdir(val_mask_dir) if f.endswith('.png')])

    assert len(train_img_files) > 0, "No training data found"

    norm_cfg = config["data"]["normalization"]
    train_ds = make_dataset(train_img_files, train_mask_files, batch_size=batch_size, shuffle=True, augment=augment,
                            mean=norm_cfg["mean"], std=norm_cfg["std"],
                            flip_prob=0.5, max_rotation_deg=45,
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    val_ds = make_dataset(val_img_files, val_mask_files, batch_size=batch_size, shuffle=False,
                          mean=norm_cfg["mean"], std=norm_cfg["std"])

    student_model = build_model_from_config(model_name, config)

    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if optimizer_name == "adamw":
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    iou_metric = BinaryIoU(threshold=0.5, name='binary_iou')

    models_dir = resolve_path(config["data"]["paths"]["models_dir"])
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    ckpt_path = os.path.join(models_dir, f"{model_name}.keras")

    print(f"Starting training for {model_name}...")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}, Distill: {distill}, NAS: {enable_nas}")

    extractor = None
    selectors = None
    if enable_nas:
        if nas_layers:
            selectors = [s.strip() for s in nas_layers.split(',')]
        else:
            selectors = ['/conv/']
        extractor = ActivationExtractor(student_model, selectors)

    if distill:
        if not teacher_weights: raise ValueError("Teacher weights required for distillation")
        print(f"Loading teacher from {teacher_weights}")
        teacher = keras.models.load_model(teacher_weights, compile=False)
        teacher.trainable = False

        if enable_nas:
            model = DistillerWithNAS(student=student_model, teacher=teacher, extractor=extractor, nas_weight=nas_weight)
        else:
            # Use original Distiller defined in train.py style; reimplement minimal here
            class Distiller(keras.Model):
                def __init__(self, student, teacher):
                    super(Distiller, self).__init__()
                    self.student = student
                    self.teacher = teacher
                def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.5, temperature=2.0):
                    # Store user metrics but avoid passing them to super().compile to prevent Keras
                    # wrapping them in CompileMetrics, which can be incompatible with our nested custom Model.
                    self._user_metrics = list(metrics) if metrics is not None else []
                    super(Distiller, self).compile(optimizer=optimizer, metrics=None)
                    # Initialize metric variables safely by calling update_state with zeros where possible
                    for m in self._user_metrics:
                        try:
                            m.update_state(tf.zeros((1, 1)), tf.zeros((1, 1)))
                        except Exception:
                            pass
                    self.student_loss_fn = student_loss_fn
                    self.distillation_loss_fn = distillation_loss_fn
                    self.alpha = alpha
                    self.temperature = temperature
                def train_step(self, data):
                    x, y = data
                    teacher_predictions = self.teacher(x, training=False)
                    with tf.GradientTape() as tape:
                        student_predictions = self.student(x, training=True)
                        student_loss = self.student_loss_fn(y, student_predictions)
                        student_soft = tf.math.sigmoid(student_predictions / self.temperature)
                        teacher_soft = tf.math.sigmoid(teacher_predictions / self.temperature)
                        dist_loss = self.distillation_loss_fn(student_soft, teacher_soft)
                        loss = self.alpha * student_loss + (1 - self.alpha) * dist_loss
                    trainable_vars = self.student.trainable_variables
                    gradients = tape.gradient(loss, trainable_vars)
                    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                    for m in getattr(self, '_user_metrics', []):
                        try:
                            m.update_state(y, student_predictions)
                        except Exception:
                            pass
                    # Debug: inspect metrics before returning results
                    for mm in self.metrics:
                        try:
                            print(f"METRIC DEBUG INNER DISTILLER: name={mm.name}, type={type(mm)}, is_metric={isinstance(mm, tf.keras.metrics.Metric)}, variables={getattr(mm, 'variables', None)}")
                        except Exception as _e:
                            print(f"METRIC DEBUG INNER DISTILLER: failed to inspect metric {mm}: {_e}")
                    # Return only metrics safe to call result() on (exclude Keras internal compile wrapper)
                    results = {}
                    for m in self.metrics:
                        if getattr(m, 'name', '') == 'compile_metrics':
                            continue
                        try:
                            results[m.name] = m.result()
                        except Exception as _e:
                            print(f"METRIC RESULT ERROR: {m} -> {_e}")
                            results[m.name] = None
                    return results
                def test_step(self, data):
                    x, y = data
                    y_pred = self.student(x, training=False)
                    student_loss = self.student_loss_fn(y, y_pred)
                    for m in getattr(self, '_user_metrics', []):
                        try:
                            m.update_state(y, y_pred)
                        except Exception:
                            pass
                    results = {}
                    for m in self.metrics:
                        if getattr(m, 'name', '') == 'compile_metrics':
                            continue
                        try:
                            results[m.name] = m.result()
                        except Exception as _e:
                            print(f"METRIC RESULT ERROR: {m} -> {_e}")
                            results[m.name] = None
                    return results
                def call(self, x):
                    return self.student(x)

            model = Distiller(student=student_model, teacher=teacher)

        def distillation_loss_fn(y_stud, y_teach):
            return tf.reduce_mean(tf.square(y_stud - y_teach))

        model.compile(
            optimizer=optimizer,
            metrics=[iou_metric],
            student_loss_fn=bce_loss,
            distillation_loss_fn=distillation_loss_fn,
            alpha=alpha,
            temperature=temperature
        )
        callbacks = [
            SaveStudentCallback(filepath=ckpt_path, monitor="val_binary_iou", mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_binary_iou", mode="max", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_binary_iou", mode="max", patience=20, restore_best_weights=True)
        ]
    else:
        # Non-distillation path
        if enable_nas:
            # Wrap student in NAS wrapper and save student model via SaveStudentCallback to avoid
            # serializing the wrapper object during ModelCheckpoint saves.
            model = NASWrapper(base_model=student_model, extractor=extractor, nas_weight=nas_weight, loss_fn=bce_loss, metrics=[iou_metric])
            model.compile(optimizer=optimizer, metrics=[iou_metric])
            callbacks = [
                SaveStudentCallback(filepath=ckpt_path, monitor="val_binary_iou", mode="max"),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_binary_iou", mode="max", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor="val_binary_iou", mode="max", patience=20, restore_best_weights=True)
            ]
        else:
            model = student_model
            model.compile(optimizer=optimizer, loss=bce_loss, metrics=[iou_metric])
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_best_only=True, monitor="val_binary_iou", mode="max", save_weights_only=False),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_binary_iou", mode="max", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor="val_binary_iou", mode="max", patience=20, restore_best_weights=True)
            ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    if distill:
        # ensure student saved
        if hasattr(model, 'student'):
            model.student.save(ckpt_path)
    else:
        if hasattr(model, 'base_model'):
            model.base_model.save(ckpt_path)
        else:
            model.save(ckpt_path)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", default="nano_u", choices=["nano_u", "bu_net"], help="Model to train")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config (optional)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--teacher-weights", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--enable-nas", action="store_true")
    parser.add_argument("--nas-layers", type=str, default=None)
    parser.add_argument("--nas-weight", type=float, default=0.01)
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        distill=args.distill,
        teacher_weights=args.teacher_weights,
        alpha=args.alpha,
        temperature=args.temperature,
        augment=not args.no_augment,
        config_path=args.config,
        enable_nas=args.enable_nas,
        nas_layers=args.nas_layers,
        nas_weight=args.nas_weight
    )
