"""Unified training pipeline: single model, distillation, and experiment entry point."""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
from datetime import datetime
import yaml
import json
import traceback

# Allow running the script directly (python src/train.py)
# If executed directly, add project root so absolute imports from `src` work.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import create_nano_u, create_bu_net, create_model_from_config
from src.utils import BinaryIoU, get_project_root
from src.utils.config import load_config
from src.data import make_dataset, get_synthetic_data
from src.nas import NASCallback as NASMonitorCallback


def _get_experiment_config(full_config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    """Resolve experiment config from full config (supports top-level or experiments section)."""
    if experiment_name in full_config:
        return full_config[experiment_name]
    if "experiments" in full_config and experiment_name in full_config["experiments"]:
        return full_config["experiments"][experiment_name]
    raise KeyError(f"Experiment '{experiment_name}' not found in config. "
                   f"Top-level keys: {list(full_config.keys())}")


def _get_train_val_data_synthetic(config: Dict[str, Any], num_train: int = 64, num_val: int = 16) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Build synthetic (train, val) data for training when no dataset paths are used."""
    input_shape = tuple(config.get("input_shape", [48, 64, 3]))
    if len(input_shape) == 2:
        input_shape = (input_shape[0], input_shape[1], 3)
    h, w, c = input_shape
    x_train = np.random.rand(num_train, h, w, c).astype(np.float32)
    y_train = np.random.randint(0, 2, (num_train, h, w, 1)).astype(np.float32)
    x_val = np.random.rand(num_val, h, w, c).astype(np.float32)
    y_val = np.random.randint(0, 2, (num_val, h, w, 1)).astype(np.float32)
    return (x_train, y_train), (x_val, y_val)


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
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    if mse_loss_fn is None:
        mse_loss_fn = tf.keras.losses.MeanSquaredError()

    with tf.GradientTape() as tape:
        # Forward pass
        if teacher is not None:
            teacher_pred = teacher(x, training=False)
        student_pred = student(x, training=True)
        
        # Compute losses
        student_loss = bce_loss_fn(y, student_pred)
        
        if teacher is not None:
            # Distillation loss: MSE between sigmoided (softened) outputs
            # This matches the "logic" of the old implementation
            # We use mean of squared differences for robustness across shapes
            distill_loss = tf.reduce_mean(
                tf.math.squared_difference(
                    tf.nn.sigmoid(teacher_pred / temperature),
                    tf.nn.sigmoid(student_pred / temperature)
                )
            )
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
    
    return {
        'loss': total_loss,
        'student_loss': student_loss,
        'distillation_loss': distill_loss
    }


def train_single_model(model: keras.Model, config: Dict, train_data: Tuple[tf.Tensor, tf.Tensor],
                      val_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> keras.callbacks.History:
    """Train a single model without distillation.
    
    Args:
        model: Model to train
        config: Training configuration
        train_data: Training data (x, y)
        val_data: Optional validation data (x, y)
    
    Returns:
        Training history
    """
    # Setup training parameters
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 16)
    learning_rate = config.get('learning_rate', 0.001)
    
    # Create optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[BinaryIoU(threshold=0.5)]
    )
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping
    if config.get('early_stopping', False):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        )
    
    # Model checkpoint - always save as best_model.keras in output_dir
    output_dir = config.get('output_dir', 'results/')
    checkpoint_path = os.path.join(output_dir, "best_model.keras")
    
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    )
    
    # TensorBoard
    if config.get('tensorboard', False):
        log_dir = config.get('log_dir', 'logs/')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.TensorBoard(log_dir=log_dir)
        )
    
    # NAS monitoring if enabled
    if config.get('use_nas', False):
        nas_layers = config.get('layers_to_monitor', ['conv2d', 'conv2d_1'])
        nas_freq = config.get('nas_frequency', 10)
        
        callbacks.append(
            NASMonitorCallback(
                layers_to_monitor=nas_layers,
                log_frequency=nas_freq
            )
        )
    
    # Training
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    try:
        history = model.fit(
            train_data[0], train_data[1],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print(f"Traceback:")
        print(traceback.format_exc())
        raise


def train_with_distillation(student: keras.Model, teacher: keras.Model, config: Dict,
                            train_data: Tuple[tf.Tensor, tf.Tensor],
                            val_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> keras.callbacks.History:
    """Train student model with knowledge distillation from teacher.
    
    Args:
        student: Student model to train
        teacher: Teacher model for distillation
        config: Training configuration
        train_data: Training data (x, y)
        val_data: Optional validation data (x, y)
    
    Returns:
        Training history
    """
    # Setup distillation parameters
    alpha = config.get('alpha', 0.3)
    temperature = config.get('temperature', 4.0)
    
    # Create optimizer
    learning_rate = config.get('learning_rate', 0.0005)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping
    if config.get('early_stopping', False):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        )
    
    # Model checkpoint - always save as best_model.keras in output_dir
    checkpoint_path = os.path.join(config.get('output_dir', 'results/'), "best_model.keras")
    
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    )
    
    # Training
    print(f"\nStarting distillation training for {config.get('epochs', 100)} epochs...")
    print(f"Student: {student.name}")
    print(f"Teacher: {teacher.name}")
    print(f"Parameters: {student.count_params():,}")
    print(f"Alpha: {alpha}, Temperature: {temperature}")
    print(f"Learning rate: {learning_rate}")
    
    try:
        # Custom training loop for distillation
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', 16)
        
        # Create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset
        if val_data:
            val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            val_dataset = None
        
        # Training loop
        # Loss objects and metrics
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        mse_loss_fn = tf.keras.losses.MeanSquaredError()
        val_iou_metric = BinaryIoU(threshold=0.5)

        history = {
            'loss': [],
            'student_loss': [],
            'distillation_loss': [],
            'val_loss': [],
            'val_student_loss': [],
            'val_distillation_loss': [],
            'val_iou': []
        }
        
        for epoch in range(epochs):
            # Training phase
            epoch_losses = {'loss': [], 'student_loss': [], 'distillation_loss': []}
            
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                losses = train_step(
                    student, teacher, x_batch, y_batch, optimizer, alpha, temperature,
                    bce_loss_fn, mse_loss_fn
                )
                
                for key, value in losses.items():
                    epoch_losses[key].append(value.numpy())
            
            # Calculate epoch averages - convert to standard float for JSON serialization
            for key in epoch_losses:
                history[key].append(float(np.mean(epoch_losses[key])))
            
            # Validation phase
            if val_dataset:
                val_losses = {'loss': [], 'student_loss': [], 'distillation_loss': []}
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
                        ).numpy()
                        total_loss = alpha * student_loss + (1 - alpha) * distill_loss
                    else:
                        total_loss = student_loss
                    
                    val_losses['loss'].append(float(total_loss))
                    val_losses['student_loss'].append(float(student_loss))
                    val_losses['distillation_loss'].append(float(distill_loss) if teacher is not None else 0.0)
                    
                    # Update IoU
                    val_iou_metric.update_state(y_batch, student_pred)
                
                for key in val_losses:
                    history[f'val_{key}'].append(float(np.mean(val_losses[key])))
                history['val_iou'].append(float(val_iou_metric.result().numpy()))
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}:")
                print(f"  Loss: {history['loss'][-1]:.4f}")
                print(f"  Student Loss: {history['student_loss'][-1]:.4f}")
                if teacher is not None:
                    print(f"  Distillation Loss: {history['distillation_loss'][-1]:.4f}")
                if val_dataset:
                    print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
                    print(f"  Val Student Loss: {history['val_student_loss'][-1]:.4f}")
                    if teacher is not None:
                        print(f"  Val Distillation Loss: {history['val_distillation_loss'][-1]:.4f}")
        
        return history
        
    except Exception as e:
        print(f"\n❌ Distillation training failed: {e}")
        print(f"Traceback:")
        print(traceback.format_exc())
        raise


def train_model(config_path: str = "config/experiments.yaml", experiment_name: str = "default",
                output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Main training function with automatic teacher/student handling.
    
    Args:
        config_path: Path to configuration file
        experiment_name: Name of experiment to run
        output_dir: Override base output directory (optional)
    
    Returns:
        Dictionary with training results and status
    """
    try:
        # Load full configuration and resolve experiment
        full_config = load_config(config_path)
        config = dict(_get_experiment_config(full_config, experiment_name))
        
        # Merge model-specific config if available
        model_name = config.get("model_name", "nano_u")
        if "models" in full_config and model_name in full_config["models"]:
            model_config = full_config["models"][model_name]
            # Update config with model defaults if not overridden in experiment
            for key, value in model_config.items():
                if key not in config:
                    config[key] = value

        if "data" in full_config and "input_shape" in full_config["data"]:
            config.setdefault("input_shape", full_config["data"]["input_shape"])
        
        if output_dir is not None:
            experiment_dir = Path(output_dir)
        else:
            base_output = Path(config.get("output_dir", "results/"))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_dir = base_output / f"{experiment_name}_{config.get('model_name', 'nano_u')}_{timestamp}"
        
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Data loading
        data_paths = full_config.get("data", {}).get("paths", {})
        processed = data_paths.get("processed", {})
        
        if not isinstance(processed, dict) or "train" not in processed:
            raise ValueError("❌ 'data.paths.processed.train' not found in configuration. Processed data is required.")

        print("📂 Loading data from processed paths...")
        train_cfg = processed.get("train", {})
        val_cfg = processed.get("val", {})
        
        t_img_dir = Path(train_cfg.get("img", ""))
        t_mask_dir = Path(train_cfg.get("mask", ""))
        v_img_dir = Path(val_cfg.get("img", ""))
        v_mask_dir = Path(val_cfg.get("mask", ""))
        
        if not t_img_dir.exists() or not t_mask_dir.exists():
            raise FileNotFoundError(f"❌ Training directories not found: {t_img_dir} or {t_mask_dir}")

        train_img_files = [str(f) for f in t_img_dir.glob("*.png")]
        train_mask_files = [str(f) for f in t_mask_dir.glob("*.png")]
        
        if not train_img_files:
            raise FileNotFoundError(f"❌ No training images found in {t_img_dir}")
        if len(train_img_files) != len(train_mask_files):
            raise ValueError(f"❌ Mismatch in training images ({len(train_img_files)}) and masks ({len(train_mask_files)})")
            
        print(f"✅ Found {len(train_img_files)} training pairs.")
        
        if v_img_dir.exists() and v_mask_dir.exists():
            val_img_files = [str(f) for f in v_img_dir.glob("*.png")]
            val_mask_files = [str(f) for f in v_mask_dir.glob("*.png")]
            if len(val_img_files) == len(val_mask_files) and val_img_files:
                print(f"✅ Found {len(val_img_files)} validation pairs.")
            else:
                print("⚠️ Validation data skipped (mismatch or empty).")
                val_img_files, val_mask_files = [], []

        batch_size = config.get("batch_size", 16)
        train_ds = make_dataset(
            train_img_files, train_mask_files, 
            batch_size=batch_size, augment=config.get("augment", False)
        )
        if val_img_files:
            val_ds = make_dataset(
                val_img_files, val_mask_files, 
                batch_size=batch_size, augment=False
            )
        else:
            val_ds = None
            
        train_data = (train_ds, None)
        val_data = val_ds

       
        # Build models
        if config.get("use_distillation", False):
            # Create teacher model - use specific name if provided, otherwise default to bu_net
            teacher_name = config.get("teacher_model_name", "bu_net")
            teacher_config = config.copy()
            teacher_config["model_name"] = teacher_name
            
            # Re-apply teacher defaults from full_config to override student settings
            if "models" in full_config and teacher_name in full_config["models"]:
                print(f"🔄 Re-applying default configuration for teacher: {teacher_name}")
                teacher_defaults = full_config["models"][teacher_name]
                teacher_config.update(teacher_defaults)
            
            print(f"👨‍🏫 Creating teacher model: {teacher_name}")
            teacher = create_model_from_config(teacher_config)
            
            teacher_weights = config.get("teacher_weights")
            if teacher_weights and Path(teacher_weights).exists():
                print(f"📦 Loading teacher weights from: {teacher_weights}")
                teacher.load_weights(teacher_weights)
            else:
                print(f"⚠️ Teacher weights not found at: {teacher_weights}. Using random initialization.")
                
            student = create_model_from_config(config)
            
            # Handle dataset vs numpy tuple for distillation
            if isinstance(train_data[0], tf.data.Dataset):
                # Distillation currently expects (x, y) tuple
                print("⚠️ Distillation with tf.data.Dataset not fully optimized, converting to single batch.")
                for x, y in train_data[0].take(1):
                    train_data = (x, y)
                if val_data:
                    for x, y in val_data.take(1):
                        val_data = (x, y)
            
            history = train_with_distillation(student, teacher, config, train_data, val_data)
            model_to_save = student
        else:
            model = create_model_from_config(config)
            # fit handles both (x, y) and dataset
            history = train_single_model(model, config, train_data, val_data)
            model_to_save = model
        
        model_path = experiment_dir / f"{config.get('model_name', 'model')}.keras"
        model_to_save.save(model_path)
        
        # Clean up best_model.keras if it exists since we just saved the final one
        checkpoint_path = experiment_dir / "best_model.keras"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        history_path = experiment_dir / "history.json"
        history_dict = history.history if hasattr(history, "history") else history
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)
        
        with open(experiment_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        return {
            "status": "success",
            "final_metrics": history_dict,
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
    """Command line interface for training."""
    parser = argparse.ArgumentParser(description="Train Nano-U models")
    parser.add_argument("--config", default="config/experiments.yaml", help="Configuration file path")
    parser.add_argument("--experiment", default="default", help="Experiment name to run")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()
    result = train_model(config_path=args.config, experiment_name=args.experiment, output_dir=args.output)
    
    if result['status'] == 'success':
        print(f"\n✅ Training completed successfully!")
        print(f"Model saved to: {result['model_path']}")
        print(f"Experiment directory: {result['experiment_dir']}")
        
        # Print final metrics
        print(f"\nFinal Metrics:")
        for metric, values in result['final_metrics'].items():
            print(f"  {metric}: {values[-1]:.4f}")
    else:
        print(f"\n❌ Training failed!")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
