import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import get_project_root
from src_tf.models.Nano_U.model_tf import build_nano_u
from src_tf.models.BU_Net.model_tf import build_bu_net
from src_tf.utils.data_tf import make_dataset
from src_tf.utils.metrics_tf import BinaryIoU
from src_tf.utils.config import load_config

# Enable GPU memory growth to avoid OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠ GPU configuration error: {e}")
else:
    print("⚠ No GPU detected - using CPU (training will be slow)")


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


def train_tf(model_name="nano_u", epochs=None, batch_size=None, lr=None,
             distill=False, teacher_weights=None, alpha=None, temperature=None,
             augment=True, config_path="config/config.yaml"):
    
    # Load configuration
    config = load_config(config_path)
    
    # Resolve parameters (CLI overrides Config)
    train_cfg = config["training"].get(model_name, {})
    gen_train_cfg = config["training"]
    
    epochs = epochs if epochs is not None else train_cfg.get("epochs", 50)
    batch_size = batch_size if batch_size is not None else train_cfg.get("batch_size", 8)
    lr = lr if lr is not None else float(train_cfg.get("learning_rate", 1e-4))
    
    # Distillation params
    if distill:
        distill_cfg = train_cfg.get("distillation", {})
        if alpha is None:
            alpha = distill_cfg.get("alpha", 0.5)
        if temperature is None:
            temperature = distill_cfg.get("temperature", 2.0)
        if teacher_weights is None:
            teacher_weights = distill_cfg.get("teacher_weights", None)
            
    # Resolve relative path for teacher weights if needed
    if teacher_weights and not os.path.exists(teacher_weights):
         # Try relative to project root
         root_teacher = os.path.join(str(get_project_root()), teacher_weights)
         if os.path.exists(root_teacher):
             teacher_weights = root_teacher

    # Data paths from config
    root_dir = str(get_project_root())
    processed_paths = config["data"]["paths"]["processed"]
    
    # Helpers to resolve paths (absolute or relative to project root)
    def resolve_path(p):
        return p if os.path.isabs(p) else os.path.join(root_dir, p)

    train_img_dir = resolve_path(processed_paths["train"]["img"])
    train_mask_dir = resolve_path(processed_paths["train"]["mask"])
    val_img_dir = resolve_path(processed_paths["val"]["img"])
    val_mask_dir = resolve_path(processed_paths["val"]["mask"])

    train_img_files = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.png')]
    train_mask_files = [os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir) if f.endswith('.png')]
    val_img_files = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.png')]
    val_mask_files = [os.path.join(val_mask_dir, f) for f in os.listdir(val_mask_dir) if f.endswith('.png')]

    assert len(train_img_files) == len(train_mask_files) and len(train_img_files) > 0, "No training data found"
    assert len(val_img_files) == len(val_mask_files) and len(val_img_files) > 0, "No validation data found"

    # Normalization params
    norm_cfg = config["data"]["normalization"]
    mean = norm_cfg["mean"]
    std = norm_cfg["std"]

    train_ds = make_dataset(train_img_files, train_mask_files, batch_size=batch_size, shuffle=True, augment=augment,
                            mean=mean, std=std)
    val_ds = make_dataset(val_img_files, val_mask_files, batch_size=batch_size, shuffle=False,
                          mean=mean, std=std)

    model = build_model_from_config(model_name, config)
    
    # Optimizer
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr) # Fallback

    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    iou_metric = BinaryIoU()

    models_dir = resolve_path(config["data"]["paths"]["models_dir"])
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    ckpt_path = os.path.join(models_dir, f"{model_name}_tf_best.keras")

    print(f"Starting training for {model_name}...")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"Distillation: {distill}")

    if not distill:
        model.compile(optimizer=optimizer, loss=bce_loss, metrics=[iou_metric])
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path, save_best_only=True, monitor="val_binary_iou", mode="max"
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_binary_iou", mode="max", patience=15, restore_best_weights=True)
        ]
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        model.save(os.path.join(models_dir, f"{model_name}_tf_final.keras"))
        return model, history

    # Distillation Logic
    if teacher_weights is None or not os.path.exists(teacher_weights):
        raise ValueError(f"Distillation requires valid teacher_weights path. Got: {teacher_weights}")
    
    print(f"Loading teacher model from {teacher_weights}")
    # Note: Keras loading might fail if custom layers aren't registered. 
    # Since we use @register_keras_serializable in layers_tf.py, it should work.
    teacher = keras.models.load_model(teacher_weights, compile=False)
    teacher.trainable = False

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            student_logits = model(x, training=True)
            loss_stud = bce_loss(y, student_logits)
            
            teacher_logits = teacher(x, training=False)
            
            # Sigmoid softening
            stud_soft = tf.math.sigmoid(student_logits / temperature)
            teach_soft = tf.math.sigmoid(teacher_logits / temperature)
            
            loss_teacher = tf.reduce_mean(tf.square(stud_soft - teach_soft))
            loss = alpha * loss_stud + (1.0 - alpha) * loss_teacher
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, student_logits

    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        loss = bce_loss(y, logits)
        return loss, logits

    best_val_iou = -1.0
    history = {"loss": [], "val_loss": [], "val_iou": []}
    
    for epoch in range(epochs):
        # Train
        train_losses = []
        for x, y in train_ds:
            loss, logits = train_step(x, y)
            train_losses.append(loss.numpy())
        
        # Validate
        val_losses = []
        iou_metric.reset_states()
        for x, y in val_ds:
            vloss, logits = val_step(x, y)
            val_losses.append(vloss.numpy())
            iou_metric.update_state(y, logits)
            
        val_iou = float(iou_metric.result().numpy())
        tr_loss = float(np.mean(train_losses)) if train_losses else 0.0
        va_loss = float(np.mean(val_losses)) if val_losses else 0.0
        
        history["loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_iou"].append(val_iou)
        
        print(f"Epoch {epoch+1}/{epochs} - loss: {tr_loss:.4f} - val_loss: {va_loss:.4f} - val_iou: {val_iou:.4f}")
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            model.save(ckpt_path)
            print("  Values improved, saved checkpoint.")

    model.save(os.path.join(models_dir, f"{model_name}_tf_final.keras"))
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model-name", default="nano_u", choices=["nano_u", "bu_net"]) 
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--teacher-weights", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    train_tf(
        model_name=args.model_name,
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
