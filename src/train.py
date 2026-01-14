"""train.py
Lightweight training entrypoint for Nano-U models.

Overview for non-TensorFlow developers:
- This module builds a Keras model (student) and optionally a larger teacher.
- Knowledge distillation is supported via the Distiller class: the student is trained
  with a combination of the standard supervised loss (binary crossentropy) and
  a distillation loss computed between softened teacher and student outputs.
- Training uses tf.data datasets produced by make_dataset; outputs are logits
  (raw scores) and the code applies sigmoid where needed to get probabilities.

Key design notes:
- Keep file I/O and dataset construction outside model code (see src/utils/data.py).
- Metrics operate on probabilities (sigmoid applied internally where appropriate).
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Allow running the script directly (python src/train.py)
# If executed directly, add project root so absolute imports from `src` work.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.Nano_U.model_tf import build_nano_u
from src.models.BU_Net.model_tf import build_bu_net
from src.utils import make_dataset, BinaryIoU, get_project_root
from src.utils.config import load_config

# Enable GPU memory growth to avoid OOM errors
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


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.5, temperature=2.0):
        # Store user metrics explicitly to avoid relying on Keras compiled_metrics() internals
        self._user_metrics = list(metrics) if metrics is not None else []
        super(Distiller, self).compile(optimizer=optimizer, metrics=self._user_metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.distillation_loss_tracker = keras.metrics.Mean(name="distill_loss")
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        # Return internal trackers first, then user-provided metrics so Keras can reset them.
        metrics = [self.total_loss_tracker, self.distillation_loss_tracker, self.student_loss_tracker]
        metrics += list(getattr(self, "_user_metrics", []))
        return metrics

    def train_step(self, data):
        x, y = data
        # Teacher prediction (inference only)
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Distillation
            student_soft = tf.math.sigmoid(student_predictions / self.temperature)
            teacher_soft = tf.math.sigmoid(teacher_predictions / self.temperature)
            dist_loss = self.distillation_loss_fn(student_soft, teacher_soft)
            
            loss = self.alpha * student_loss + (1 - self.alpha) * dist_loss
            
        # Compute gradients (Standard float32 precision)
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(dist_loss)
        self.total_loss_tracker.update_state(loss)

        # Update user metrics directly to avoid deprecated compiled_metrics() calls in Keras internals
        for m in getattr(self, "_user_metrics", []):
            try:
                m.update_state(y, student_predictions)
            except Exception:
                # Ignore metric update failures to avoid stopping training; log minimally
                print(f"Warning: failed to update metric {m}")

        return {m.name: m.result() for m in self.metrics}

    # validation, keras wants the alias test_step
    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)
        self.total_loss_tracker.update_state(student_loss)

        for m in getattr(self, "_user_metrics", []):
            try:
                m.update_state(y, y_pred)
            except Exception:
                print(f"Warning: failed to update metric {m} (validation)")

        return {m.name: m.result() for m in self.metrics}
    
    def call(self, x):
        return self.student(x)

class SaveStudentCallback(keras.callbacks.Callback):
    """
    Custom Keras callback to save the student model during distillation training.
    
    This callback monitors a specified validation metric and saves the student model
    (from the Distiller wrapper) whenever the metric improves. It's designed for
    knowledge distillation where we want to save only the lightweight student model,
    not the entire Distiller object.
    """
    def __init__(self, filepath, monitor='val_binary_iou', save_best_only=True, mode='max'):
        """
        Initialize the callback.
        
        Args:
            filepath (str): Path where to save the student model (e.g., 'models/nano_u_tf_best.keras').
            monitor (str): Name of the metric to monitor (e.g., 'val_binary_iou').
            save_best_only (bool): If True, save only when the metric improves.
            mode (str): 'max' for metrics that should increase (e.g., IoU), 'min' for decrease.
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = -np.inf if mode == 'max' else np.inf

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch. Check if the monitored metric improved and save if needed.
        
        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary of metrics from the current epoch.
        """
        current = logs.get(self.monitor)
        if current is None:
            # Fallback if metric not found (e.g., string mismatch in logs)
            keys = list(logs.keys())
            print(f"\nWarning: '{self.monitor}' not found in logs {keys}. Skipping save check.")
            return
        
        # Determine if current value is better than best
        is_better = (current > self.best) if self.mode == 'max' else (current < self.best)
        
        # Save if always saving or if improved
        if (self.save_best_only and is_better) or not self.save_best_only:
            if self.save_best_only:
                self.best = current  # Update best value
            # Save the student model (not the Distiller wrapper)
            self.model.student.save(self.filepath)
            print(f"\nMetric {self.monitor} improved. Saved student model to {self.filepath}")


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
             augment=True, config_path="config/config.yaml"):
    
    # Load configuration
    config = load_config(config_path)
    
    # Resolve parameters (CLI overrides Config)
    train_cfg = config["training"].get(model_name, {})
    
    epochs = epochs if epochs is not None else train_cfg.get("epochs", 50)
    batch_size = batch_size if batch_size is not None else train_cfg.get("batch_size", 8)
    lr = lr if lr is not None else float(train_cfg.get("learning_rate", 1e-4))
    
    # Distillation params
    if distill:
        distill_cfg = train_cfg.get("distillation", {})
        # Favor teacher more by default: lower alpha means student relies more on distillation loss
        if alpha is None: alpha = distill_cfg.get("alpha", 0.3)
        # Higher temperature softens teacher probabilities to provide richer gradients
        if temperature is None: temperature = distill_cfg.get("temperature", 3.0)
        if teacher_weights is None: teacher_weights = distill_cfg.get("teacher_weights", None)
        
    # Resolve relative path for teacher weights if needed
    if teacher_weights and not os.path.exists(teacher_weights):
         root_teacher = os.path.join(str(get_project_root()), teacher_weights)
         if os.path.exists(root_teacher):
             teacher_weights = root_teacher

    # Data paths from config
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
    
    # Increased augmentation strength to combat overfitting on limited dataset
    train_ds = make_dataset(train_img_files, train_mask_files, batch_size=batch_size, shuffle=True, augment=augment,
                            mean=norm_cfg["mean"], std=norm_cfg["std"],
                            flip_prob=0.5, max_rotation_deg=45, # Increased rotation
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1) # Increased color jitter
    val_ds = make_dataset(val_img_files, val_mask_files, batch_size=batch_size, shuffle=False,
                          mean=norm_cfg["mean"], std=norm_cfg["std"])

    # Build Student
    student_model = build_model_from_config(model_name, config)
    
    # Optimizer setup with Weight Decay for regularization
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    
    if optimizer_name == "adamw":
        # Use AdamW for proper weight decay (decoupled)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr) # Fallback


    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # BinaryIoU expects a probability threshold (applied after sigmoid). Use 0.5 for standard 50% cutoff.
    iou_metric = BinaryIoU(threshold=0.5, name='binary_iou')

    models_dir = resolve_path(config["data"]["paths"]["models_dir"])
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    # Use a single canonical model filename (no '_tf' suffix) and overwrite the best/final into this file.
    ckpt_path = os.path.join(models_dir, f"{model_name}.keras")

    print(f"Starting training for {model_name}...")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}, Distill: {distill}")

    if distill:
        if not teacher_weights: raise ValueError("Teacher weights required for distillation")
        print(f"Loading teacher from {teacher_weights}")
        teacher = keras.models.load_model(teacher_weights, compile=False)
        teacher.trainable = False
        
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
        model = student_model
        model.compile(optimizer=optimizer, loss=bce_loss, metrics=[iou_metric])
        callbacks = [
            # Save the best model to a single canonical path (overwrite previous).
            tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_best_only=True, monitor="val_binary_iou", mode="max", save_weights_only=False),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_binary_iou", mode="max", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_binary_iou", mode="max", patience=20, restore_best_weights=True)
        ]

    # Unified Training Loop
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Ensure final student/model saved to the canonical path (may overwrite; ModelCheckpoint already saved best)
    if distill:
        model.student.save(ckpt_path)
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
        config_path=args.config
    )
