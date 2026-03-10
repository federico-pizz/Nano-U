import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import tf_keras as keras
from typing import Dict, Optional, Tuple, List, Any, Union
from pathlib import Path
from datetime import datetime
import yaml
import json
import traceback
import tensorflow_model_optimization as tfmot
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Allow running the script directly (python src/train.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import create_nano_u, create_bu_net, create_model_from_config
from src.utils import BinaryIoU
from src.utils.config import load_config
from src.utils.qat import apply_qat_to_model
from src.data import make_dataset
from src.nas import NASCallback as NASMonitorCallback


def _get_config(full_config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    """Retrieve the experiment configuration from the full config dictionary.
    
    Searches for the experiment configuration in the following order:
    1. Inside the 'training' block (preferred for hyperparameter sets).
    2. Inside the 'experiments' block (legacy support).
    3. Custom top-level match by exact experiment name.
    
    Args:
        full_config: The complete configuration dictionary loaded from YAML.
        experiment_name: The target experiment name to search for.
        
    Returns:
        The dictionary containing the experiment-specific configuration.
        
    Raises:
        KeyError: If the experiment name cannot be found in the supported blocks.
    """
    if "training" in full_config and experiment_name in full_config["training"]:
        return full_config["training"][experiment_name]
    if "experiments" in full_config and experiment_name in full_config["experiments"]:
        return full_config["experiments"][experiment_name]
    if experiment_name in full_config:
        return full_config[experiment_name]
    
    raise KeyError(
        f"Experiment/Model '{experiment_name}' not found in config. "
        f"Top-level keys: {list(full_config.keys())}"
    )


def _get_experiment_config(full_config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    """Backward-compatible alias for older code/tests expecting `_get_experiment_config`.

    New code should call `_get_config` directly; this wrapper exists so that
    external callers and the test suite can continue to import the previous
    helper name without breaking.
    """
    return _get_config(full_config, experiment_name)



@tf.function(reduce_retracing=True)
def train_step(student: keras.Model, teacher: Optional[keras.Model], x: tf.Tensor, y: tf.Tensor,
               optimizer: keras.optimizers.Optimizer, alpha: float = 0.3, temperature: float = 4.0,
               bce_loss_fn=None, mse_loss_fn=None) -> Dict[str, tf.Tensor]:
    """Custom training step for knowledge distillation.
    
    Args:
        student: Student model
        teacher: Optional teacher model
        x: Input batch
        y: Ground truth masks
        optimizer: Optimizer
        alpha: Weight for student loss (1-alpha for distillation)
        temperature: Distillation temperature
        bce_loss_fn: BinaryCrossentropy loss object
        mse_loss_fn: MeanSquaredError loss object
        
    Returns:
        Dictionary of loss components
    """
    if bce_loss_fn is None:
        bce_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    if mse_loss_fn is None:
        mse_loss_fn = keras.losses.MeanSquaredError()

    with tf.GradientTape() as tape:
        # Forward pass
        if teacher is not None:
            teacher_pred = teacher(x, training=False)
        student_pred = student(x, training=True)
        
        # Compute losses
        student_loss = bce_loss_fn(y, student_pred)
        
        if teacher is not None:
            # Distillation loss: MSE between temperature-scaled sigmoided outputs.
            # Multiplied by temperature^2 to correct magnitude shrinkage from logit scaling.
            distill_loss = tf.reduce_mean(
                tf.math.squared_difference(
                    tf.nn.sigmoid(teacher_pred / temperature),
                    tf.nn.sigmoid(student_pred / temperature)
                )
            ) * (temperature ** 2)
            total_loss = alpha * student_loss + (1 - alpha) * distill_loss
        else:
            distill_loss = tf.constant(0.0, dtype=tf.float32)
            total_loss = student_loss
        
        # Scale total loss for better training stability if needed
        # (old implementation used sum reduction implicitly in BCEWithLogitsLoss sometimes, 
        # but here we use mean for consistency with current pipeline)
    
    # Apply gradients
    gradients = tape.gradient(total_loss, student.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student.trainable_variables))
    
    # Compute batch-level binary_iou functionally for progress logging
    # (stateful Metric objects cannot be initialized inside tf.function)
    y_pred_sig = tf.nn.sigmoid(student_pred)
    y_pred_bin = tf.cast(y_pred_sig > 0.5, tf.float32)
    y_true_bin = tf.cast(y > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true_bin * y_pred_bin)
    union = tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin) - intersection
    iou = intersection / (union + 1e-7)

    return {
        'loss': total_loss,
        'student_loss': student_loss,
        'distillation_loss': distill_loss,
        'binary_iou': iou
    }


def train_single_model(
    model: keras.Model,
    config: Dict[str, Any],
    train_data: Union[tf.data.Dataset, Tuple[tf.Tensor, tf.Tensor]],
    val_data: Optional[Union[tf.data.Dataset, Tuple[tf.Tensor, tf.Tensor]]] = None,
    experiment_dir: str = "results/",
    extra_callbacks: Optional[list] = None,
) -> keras.callbacks.History:
    """Train a standalone Keras model using standard fit routines.
    
    This function establishes the environment, optimizer, and callbacks (such as
    ModelCheckpoint and EarlyStopping) to train a single model without distillation.
    
    Args:
        model: Compiled Keras model instance.
        config: Training hyperparameters (epochs, batch_size, learning_rate).
        train_data: Dataset mapped as a tf.data.Dataset or an (x, y) tensor tuple.
        val_data: Evaluation dataset mapped as a tf.data.Dataset or (x, y) tensor tuple.
        
    Returns:
        The keras.callbacks.History object detailing the metrics per epoch.
    """
    # Setup training parameters
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 16)
    learning_rate = float(config.get('learning_rate', 0.001))
    
    # Create optimizer
    weight_decay = float(config.get('weight_decay', 0.0))
    optimizer_type = config.get('optimizer', 'adam').lower()
    
    if optimizer_type == 'adamw' or weight_decay > 0:
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[BinaryIoU(threshold=0.5, from_logits=True)]
    )
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping
    if config.get('early_stopping', False):
        patience = int(config.get('patience', 10))
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )
        )
    
    # Reduce LR on Plateau
    if config.get('lr_scheduler', '').lower() == 'plateau':
        patience = int(config.get('lr_plateau_patience', 5))
        factor = float(config.get('lr_plateau_factor', 0.2))
        min_lr = float(config.get('min_lr', 1e-6))
        print(f"Adding ReduceLROnPlateau callback (patience={patience}, factor={factor}, min_lr={min_lr})")
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=1
            )
        )
    
    output_dir = experiment_dir
    # Model checkpoint - always save as temp_model.h5 in models/
    Path("models").mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join("models", "temp_model.h5")
    
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
    )
    
    # TensorBoard
    if config.get('tensorboard', False):
        log_dir = os.path.join(experiment_dir, 'logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.TensorBoard(log_dir=log_dir)
        )
    
    # NAS monitoring if enabled
    if config.get('use_nas', False):
        nas_layers = config.get('layers_to_monitor', ['enc1a_dw', 'enc1a_pw'])
        nas_freq = config.get('nas_frequency', 10)
        
        callbacks.append(
            NASMonitorCallback(
                layers_to_monitor=nas_layers,
                log_frequency=nas_freq,
                validation_data=val_data,
                output_dir=output_dir
            )
        )
    
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    
    # Brief training header (Keras will handle per-epoch progress).
    print(f"\n[TRAIN] {model.name}: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    if weight_decay > 0:
        print(f"[TRAIN] weight_decay={weight_decay}")
    
    # Fit — accept both tf.data.Dataset and (x, y) tuple
    is_dataset = isinstance(train_data, tf.data.Dataset)
    try:
        if is_dataset:
            history = model.fit(
                train_data,
                epochs=epochs,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1,
            )
        else:
            history = model.fit(
                train_data[0], train_data[1],
                batch_size=batch_size,
                epochs=epochs,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1,
            )
        
        return history
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        print(f"Traceback:")
        print(traceback.format_exc())
        raise


def build_distillation_datasets(
    train_data: Union[tf.data.Dataset, Tuple[tf.Tensor, tf.Tensor]],
    val_data: Optional[Union[tf.data.Dataset, Tuple[tf.Tensor, tf.Tensor]]],
    batch_size: int,
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
    """Normalize train/val inputs into tf.data.Datasets for distillation."""
    if isinstance(train_data, tf.data.Dataset):
        train_dataset = train_data
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if val_data is None:
        return train_dataset, None

    if isinstance(val_data, tf.data.Dataset):
        val_dataset = val_data
    else:
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset


def update_plateau_and_early_stopping(
    epoch: int,
    config: Dict[str, Any],
    optimizer: keras.optimizers.Optimizer,
    current_val_loss: float,
    best_val_loss: float,
    patience_counter: int,
    lr_patience_counter: int,
) -> Tuple[float, int, int, bool]:
    """Apply ReduceLROnPlateau + EarlyStopping logic, returning updated state."""
    stopped_early = False

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0
        lr_patience_counter = 0
    else:
        patience_counter += 1
        lr_patience_counter += 1

    use_plateau = config.get("lr_scheduler", "").lower() == "plateau"
    lr_patience = int(config.get("lr_plateau_patience", 5))
    lr_factor = float(config.get("lr_plateau_factor", 0.2))
    min_lr = float(config.get("min_lr", 1e-6))

    if use_plateau and lr_patience_counter >= lr_patience:
        old_lr = float(optimizer.learning_rate.numpy())
        new_lr = max(old_lr * lr_factor, min_lr)
        if new_lr < old_lr:
            print(f"  Epoch {epoch+1}: ReduceLROnPlateau reducing learning rate to {new_lr:.8f}.")
            optimizer.learning_rate.assign(new_lr)
        lr_patience_counter = 0

    if config.get("early_stopping", False) and patience_counter >= int(config.get("patience", 10)):
        print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
        stopped_early = True

    return best_val_loss, patience_counter, lr_patience_counter, stopped_early


def train_with_distillation(
    student: keras.Model,
    teacher: keras.Model,
    config: Dict[str, Any],
    train_data: Union[Tuple[tf.Tensor, tf.Tensor], tf.data.Dataset],
    val_data: Optional[Union[Tuple[tf.Tensor, tf.Tensor], tf.data.Dataset]] = None,
    experiment_dir: str = "results/",
) -> Dict[str, List[float]]:
    """Train a student model via knowledge distillation using a custom GradientTape loop.
    
    The loss function minimizes a weighted sum of:
      1. Binary Crossentropy against the hard ground truth labels.
      2. Mean Squared Error (MSE) against the temperature-softened outputs of the teacher.
      
    Args:
        student: The lightweight model architecture being trained.
        teacher: The high-capacity, pre-trained model providing target probabilities.
        config: Hyperparameter dictionary containing alpha and temperature settings.
        train_data: The dataset for training iterations.
        val_data: The evaluation dataset used for Checkpointing and Early Stopping.
        
    Returns:
        A dictionary mapping metric names to lists of floats recorded per epoch.
    """
    # Setup distillation parameters and optimizer
    alpha = float(config.get("alpha", 0.3))
    temperature = float(config.get("temperature", 4.0))
    learning_rate = float(config.get("learning_rate", 0.0005))
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer_type = config.get("optimizer", "adam").lower()
    
    if optimizer_type == "adamw" or weight_decay > 0:
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Model checkpoint - always save as temp_model.h5 in models/
    Path("models").mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join("models", "temp_model.h5")
    output_dir = experiment_dir
    
    print(
        f"\n[TRAIN-KD] student='{student.name}' teacher='{teacher.name}' "
        f"epochs={config.get('epochs', 100)} batch_size={config.get('batch_size', 16)} "
        f"alpha={alpha} T={temperature} lr={learning_rate}"
    )
    if weight_decay > 0:
        print(f"[TRAIN-KD] weight_decay={weight_decay}")
    
    # dummy loss since we use custom loop
    student.compile(optimizer=optimizer, loss='mse')
    
    try:
        epochs = config.get("epochs", 100)
        batch_size = config.get("batch_size", 16)
        
        # Normalize data inputs into tf.data.Datasets
        train_dataset, val_dataset = build_distillation_datasets(train_data, val_data, batch_size)
            
        # Initialize NAS Monitor for manual loop
        nas_callback = None
        if config.get('use_nas', False):
            nas_layers = config.get('layers_to_monitor', ['enc1a_dw', 'enc1a_pw'])
            nas_freq = config.get('nas_frequency', 10)
            nas_callback = NASMonitorCallback(
                layers_to_monitor=nas_layers,
                log_frequency=nas_freq,
                validation_data=val_dataset,
                output_dir=output_dir
            )
            # Simulate Keras callback initialization
            nas_callback.set_model(student)
        
        # Loss objects and metrics
        bce_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        mse_loss_fn = keras.losses.MeanSquaredError()
        val_iou_metric = BinaryIoU(threshold=0.5, from_logits=True)

        history: Dict[str, List[float]] = {
            'loss': [],
            'student_loss': [],
            'distillation_loss': [],
            'binary_iou': [],
            'val_loss': [],
            'val_student_loss': [],
            'val_distillation_loss': [],
            'val_binary_iou': []
        }
        
        
        best_val_loss = float("inf")
        patience_counter = 0
        lr_patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            epoch_losses: Dict[str, List[float]] = {
                "loss": [],
                "student_loss": [],
                "distillation_loss": [],
                "binary_iou": []
            }
            
            print(f"Epoch {epoch + 1}/{epochs}")
            progbar = keras.utils.Progbar(target=len(train_dataset), stateful_metrics=['loss', 'binary_iou', 'lr'])
            
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                losses = train_step(
                    student, teacher, x_batch, y_batch, optimizer, alpha, temperature,
                    bce_loss_fn, mse_loss_fn
                )
                
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.numpy())
                
                # Update progress bar
                lr = float(optimizer.learning_rate.numpy())
                current_metrics = [
                    ('loss', losses['loss'].numpy()),
                    ('binary_iou', losses['binary_iou'].numpy()),
                    ('lr', lr)
                ]
                progbar.update(step + 1, values=current_metrics)
            
            # Calculate epoch averages - convert to standard float for JSON serialization
            for key, values in epoch_losses.items():
                history.setdefault(key, [])
                history[key].append(float(np.mean(values)))
            
            # Validation phase
            if val_dataset:
                val_losses: Dict[str, List[float]] = {
                    "loss": [],
                    "student_loss": [],
                    "distillation_loss": [],
                }
                val_iou_metric.reset_state()
                
                for x_batch, y_batch in val_dataset:
                    if teacher is not None:
                        teacher_pred = teacher(x_batch, training=False)
                    student_pred = student(x_batch, training=False)
                    
                    student_loss = bce_loss_fn(y_batch, student_pred).numpy()
                    
                    if teacher is not None:
                        # Distillation loss: MSE between sigmoided outputs
                        distill_loss = tf.reduce_mean(
                            tf.math.squared_difference(
                                tf.nn.sigmoid(teacher_pred / temperature),
                                tf.nn.sigmoid(student_pred / temperature)
                            )
                        ).numpy() * (temperature ** 2)
                        total_loss = float(alpha * student_loss + (1 - alpha) * distill_loss)
                    else:
                        distill_loss = 0.0
                        total_loss = float(student_loss)
                    
                    val_losses['loss'].append(total_loss)
                    val_losses['student_loss'].append(float(student_loss))
                    val_losses['distillation_loss'].append(float(distill_loss))
                    
                    # Update IoU
                    val_iou_metric.update_state(y_batch, student_pred)
                
                for key, values in val_losses.items():
                    history.setdefault(f"val_{key}", [])
                    history[f"val_{key}"].append(float(np.mean(values)))

                history.setdefault("val_binary_iou", [])
                val_iou_final = float(val_iou_metric.result().numpy())
                history["val_binary_iou"].append(val_iou_final)

                # Final progbar update for the epoch including validation metrics
                lr = float(optimizer.learning_rate.numpy())
                final_values = [
                    ('loss', history['loss'][-1]),
                    ('binary_iou', history['binary_iou'][-1]),
                    ('val_loss', history['val_loss'][-1]),
                    ('val_binary_iou', val_iou_final),
                    ('lr', lr)
                ]
                progbar.update(len(train_dataset), values=final_values, finalize=True)
            
            # --- Manual Logic for custom training loop callbacks ---
            if val_dataset:
                current_val_loss = history["val_loss"][-1]

                # Model checkpointing + LR scheduling + early stopping
                improved = current_val_loss < best_val_loss
                best_val_loss, patience_counter, lr_patience_counter, stopped = update_plateau_and_early_stopping(
                    epoch,
                    config,
                    optimizer,
                    current_val_loss,
                    best_val_loss,
                    patience_counter,
                    lr_patience_counter,
                )

                if improved:
                    print(
                        f"Epoch {epoch + 1}: val_loss improved from {best_val_loss:.5f} to {current_val_loss:.5f}, "
                        f"saving model to {checkpoint_path}"
                    )
                    student.save(checkpoint_path)

                if stopped:
                    if os.path.exists(checkpoint_path):
                        print(f"Restoring best weights from {checkpoint_path}")
                        student.load_weights(checkpoint_path)
                    break
            
            # Trigger NAS callback manually at epoch end
            if nas_callback:
                nas_callback.on_epoch_end(epoch, logs=history)
        
        # Single-line completion summary
        completed_epochs = len(history.get("loss", []))
        if history.get("loss"):
            final_loss = history["loss"][-1]
            msg = f"[TRAIN-KD] completed {completed_epochs} epochs, final_loss={final_loss:.4f}"
            if history.get("val_loss"):
                msg += f", final_val_loss={history['val_loss'][-1]:.4f}"
            print(msg)

        return history
        
    except Exception as e:
        print(f"\nDistillation training failed: {e}")
        print(f"Traceback:")
        print(traceback.format_exc())
        raise


def train_model(config_path: str = "config/config.yaml", experiment_name: str = "default",
                output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Execute training pipeline coordinating dataset construction and model selection.
    
    This overarching orchestrator loads the configuration definitions, parses and prepares
    normalized tf.data image datasets, constructs either a generic single network or initializes
    a teacher/student distillation pair, and dispatches the execution to the relevant training loop.
    
    Args:
        config_path: Path to the main YAML configuration file.
        experiment_name: Internal name pointing to a hyperparameter block in the configuration.
        output_dir: Manual override directory to store the resulting checkpoints and metric jsons.
        
    Returns:
        A dictionary containing termination status, path strings, and the final history structure.
    """
    try:
        # Load full configuration and resolve experiment
        full_config = load_config(config_path)
        config = dict(_get_config(full_config, experiment_name))
        
        # Override the model_name to be the experiment_name since we call run_training_pipeline("bu_net")
        config["model_name"] = experiment_name
        
        # Merge model architecture config if available
        if "models" in full_config and experiment_name in full_config["models"]:
            model_config = full_config["models"][experiment_name]
            # Update config with model defaults if not overridden in experiment
            for key, value in model_config.items():
                if key not in config:
                    config[key] = value

        # If a nested `distillation` block is present in the experiment config
        # (as in config/config.yaml), normalize it into the flat keys expected
        # by the training pipeline so that knowledge distillation is actually
        # activated when `enabled: true` is set.
        dist_cfg = config.get("distillation")
        if isinstance(dist_cfg, dict) and dist_cfg.get("enabled"):
            # Enable distillation for this experiment
            config.setdefault("use_distillation", True)
            # Teacher model name can differ from the student model_name
            teacher_model_name = dist_cfg.get("teacher_model_name", "bu_net")
            config.setdefault("teacher_model_name", teacher_model_name)
            # Carry over optional knobs if they are provided
            if "teacher_weights" in dist_cfg:
                config.setdefault("teacher_weights", dist_cfg["teacher_weights"])
            if "alpha" in dist_cfg:
                config.setdefault("alpha", dist_cfg["alpha"])
            if "temperature" in dist_cfg:
                config.setdefault("temperature", dist_cfg["temperature"])

        if "data" in full_config and "input_shape" in full_config["data"]:
            config.setdefault("input_shape", full_config["data"]["input_shape"])
        
        if output_dir is not None:
            experiment_dir = Path(output_dir)
        else:
            base_output = Path(config.get("output_dir", "results/"))
            model_name_for_dir = config.get('model_name', experiment_name)
            experiment_dir = base_output / model_name_for_dir
        
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Data loading
        data_paths = full_config.get("data", {}).get("paths", {})
        processed = data_paths.get("processed", {})
        
        if not isinstance(processed, dict) or "train" not in processed:
            raise ValueError("'data.paths.processed.train' not found in configuration. Processed data is required.")

        print("Loading data from processed paths...")
        train_cfg = processed.get("train", {})
        val_cfg = processed.get("val", {})
        
        t_img_dir = Path(train_cfg.get("img", ""))
        t_mask_dir = Path(train_cfg.get("mask", ""))
        v_img_dir = Path(val_cfg.get("img", ""))
        v_mask_dir = Path(val_cfg.get("mask", ""))
        
        if not t_img_dir.exists() or not t_mask_dir.exists():
            raise FileNotFoundError(f"Training directories not found: {t_img_dir} or {t_mask_dir}")

        train_img_files = [str(f) for f in t_img_dir.glob("*.png")]
        train_mask_files = [str(f) for f in t_mask_dir.glob("*.png")]
        
        if not train_img_files:
            raise FileNotFoundError(f"No training images found in {t_img_dir}")
        if len(train_img_files) != len(train_mask_files):
            raise ValueError(f"Mismatch in training images ({len(train_img_files)}) and masks ({len(train_mask_files)})")
            
        print(f"Found {len(train_img_files)} training pairs.")
        
        val_img_files: List[str] = []
        val_mask_files: List[str] = []
        if v_img_dir.exists() and v_mask_dir.exists():
            val_img_files = [str(f) for f in v_img_dir.glob("*.png")]
            val_mask_files = [str(f) for f in v_mask_dir.glob("*.png")]
            if len(val_img_files) == len(val_mask_files) and val_img_files:
                print(f"Found {len(val_img_files)} validation pairs.")
            else:
                print("Validation data skipped (mismatch or empty).")
                val_img_files, val_mask_files = [], []

        norm_mean = full_config.get("data", {}).get("normalization", {}).get("mean", [0.5, 0.5, 0.5])
        norm_std = full_config.get("data", {}).get("normalization", {}).get("std", [0.5, 0.5, 0.5])

        batch_size = config.get("batch_size", 16)
        
        input_shape_cfg = config.get("input_shape")
        target_size = (input_shape_cfg[0], input_shape_cfg[1]) if input_shape_cfg else None

        train_ds = make_dataset(
            train_img_files, train_mask_files,
            batch_size=batch_size, augment=config.get("augment", False),
            target_size=target_size,
            mean=norm_mean, std=norm_std
        )
        val_ds: Optional[tf.data.Dataset] = None
        if val_img_files:
            val_ds = make_dataset(
                val_img_files, val_mask_files,
                batch_size=batch_size, augment=False,
                target_size=target_size,
                mean=norm_mean, std=norm_std
            )

        train_data = train_ds
        val_data = val_ds

        model_to_save = None
        
        # Build models
        if config.get("use_distillation", False):
            # The 'teacher_experiment' is basically an experiment config or model name
            # Let's see if we can extract its structure.
            teacher_name = config.get("teacher_model_name", "bu_net")
            teacher_config = config.copy()
            teacher_config["model_name"] = teacher_name
            
            # Re-apply teacher defaults from config presets, preferring `models` over `experiments`.
            if "models" in full_config and teacher_name in full_config["models"]:
                teacher_defaults = full_config["models"][teacher_name]
                teacher_config.update(teacher_defaults)
            elif "experiments" in full_config and teacher_name in full_config["experiments"]:
                teacher_defaults = full_config["experiments"][teacher_name]
                teacher_config.update(teacher_defaults)
            
            print(f"[KD] Building teacher '{teacher_name}'")
            teacher = create_model_from_config(teacher_config)
            
            teacher_weights = config.get("teacher_weights")
            if teacher_weights and Path(teacher_weights).exists():
                print(f"[KD] Loading teacher weights from {teacher_weights}")
                teacher.load_weights(teacher_weights)
            else:
                if teacher_weights:
                    print(f"[KD] Teacher weights not found at {teacher_weights}; using random initialization.")
                
            student = create_model_from_config(config)

            # Apply QAT (Quantization-Aware Training) to the student model
            qat_enabled = config.get("qat_enabled", True)
            if qat_enabled:
                print("Applying Quantization-Aware Training (QAT) to student model...")
                student = apply_qat_to_model(student)

            # Distillation can now correctly handle tf.data.Dataset transparently
            history = train_with_distillation(student, teacher, config, train_data, val_data, experiment_dir=str(experiment_dir))
            model_to_save = student
            is_qat_model = qat_enabled
        else:
            model = create_model_from_config(config)

            # Apply QAT only to nano_u (the student / deployment model).
            # TFMOT requires a Sequential or Functional model; bu_net and other
            # custom subclasses are not compatible and must skip QAT.
            model_name = config.get("model_name", "nano_u")
            qat_enabled = config.get("qat_enabled", True) and model_name == "nano_u"
            if qat_enabled:
                print("Applying Quantization-Aware Training (QAT) to model...")
                model = apply_qat_to_model(model)

            # fit handles both (x, y) and dataset
            history = train_single_model(model, config, train_data, val_data, experiment_dir=str(experiment_dir))
            model_to_save = model
            is_qat_model = qat_enabled

        
        # Restore best weights before saving if they exist
        temp_checkpoint = "models/temp_model.h5"
        if os.path.exists(temp_checkpoint):
            print(f"Loading best weights from {temp_checkpoint} for final save...")
            model_to_save.load_weights(temp_checkpoint)

        Path("models").mkdir(parents=True, exist_ok=True)
        model_name = config.get('model_name', 'model')
        model_path = Path("models") / f"{model_name}.h5"

        if is_qat_model:
            # Strip the QAT annotation wrappers and save a plain float model.
            # The TFLite converter in quantize_model.py will then apply PTQ
            # (post-training quantization) on top of the QAT-trained weights,
            if hasattr(model_to_save, "_is_qat_model") and model_to_save._is_qat_model:
                print("QAT detected — saving model with quantization wrappers...")
                model_to_save.save(model_path)
                print(f"QAT model saved to {model_path}")
            else:
                model_to_save.save(model_path)
        else:
            model_to_save.save(model_path)

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
        
        with open(experiment_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        return {
            "status": "success",
            "final_metrics": serializable_history,
            "model_path": str(model_path),
            "model_name": config.get("model_name", "model"),
            "experiment_dir": str(experiment_dir),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    """Command-line interface entrypoint for triggering training pipelines."""
    parser = argparse.ArgumentParser(description="Train Nano-U models via single-pass or Distillation")
    parser.add_argument("--config", default="config/experiments.yaml", help="Configuration file path")
    parser.add_argument("--experiment", default="default", help="Experiment name to run")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()
    result = train_model(config_path=args.config, experiment_name=args.experiment, output_dir=args.output)
    
    if result['status'] == 'success':
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {result['model_path']}")
        print(f"Experiment directory: {result['experiment_dir']}")
        
        # Print final metrics
        print(f"\nFinal Metrics:")
        for metric, values in result['final_metrics'].items():
            print(f"  {metric}: {values[-1]:.4f}")
    else:
        print(f"\nTraining failed!")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
