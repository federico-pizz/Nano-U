import os
import sys
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from pathlib import Path

# Allow running the script directly
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.qat import NoOpQuantizeConfig
from src.utils.metrics import BinaryIoU
from src.data import make_dataset
from src.train import train_single_model

def fine_tune_model(base_model_path: str, output_model_path: str, epochs: int = 10):
    """
    Load an existing Nano-U model and fine-tune it on the secondary dataset.
    """
    print(f"\n--- Fine-tuning {base_model_path} ---")
    
    # 1. Load Configuration
    config = load_config("config/config.yaml")
    
    # 2. Prepare Secondary Dataset
    secondary_data = config.get("data", {}).get("paths", {}).get("secondary", {})
    if not secondary_data:
        print("Error: 'data.paths.secondary' not found in config.yaml")
        return False
    
    train_cfg = secondary_data.get("train", {})
    val_cfg = secondary_data.get("val", {})
    
    t_img_dir = Path(train_cfg.get("img", ""))
    t_mask_dir = Path(train_cfg.get("mask", ""))
    v_img_dir = Path(val_cfg.get("img", ""))
    v_mask_dir = Path(val_cfg.get("mask", ""))
    
    if not t_img_dir.exists() or not t_mask_dir.exists():
        print(f"Error: Training directories not found: {t_img_dir} or {t_mask_dir}")
        return False

    train_img_files = sorted([str(f) for f in t_img_dir.glob("*.png")])
    train_mask_files = sorted([str(f) for f in t_mask_dir.glob("*.png")])
    val_img_files = sorted([str(f) for f in v_img_dir.glob("*.png")])
    val_mask_files = sorted([str(f) for f in v_mask_dir.glob("*.png")])

    print(f"Found {len(train_img_files)} training samples and {len(val_img_files)} validation samples.")

    norm_mean = config.get("data", {}).get("normalization", {}).get("mean", [0.5, 0.5, 0.5])
    norm_std = config.get("data", {}).get("normalization", {}).get("std", [0.5, 0.5, 0.5])
    input_shape = config.get("data", {}).get("input_shape", [60, 80, 3])
    target_size = (input_shape[0], input_shape[1])
    batch_size = config.get("training", {}).get("nano_u", {}).get("batch_size", 8)

    train_ds = make_dataset(
        train_img_files, train_mask_files,
        batch_size=batch_size, augment=True,
        target_size=target_size,
        mean=norm_mean, std=norm_std
    )
    val_ds = None
    if val_img_files:
        val_ds = make_dataset(
            val_img_files, val_mask_files,
            batch_size=batch_size, augment=False,
            target_size=target_size,
            mean=norm_mean, std=norm_std
        )

    # 3. Load Model with QAT Scope
    custom_objects = {
        "BinaryIoU": BinaryIoU,
        "NoOpQuantizeConfig": NoOpQuantizeConfig
    }
    
    with tfmot.quantization.keras.quantize_scope(custom_objects):
        model = keras.models.load_model(base_model_path, custom_objects=custom_objects, compile=False)
    
    # 4. Fine-tuning
    # Load config parameters from training.nano_u_fine_tune
    fine_tune_cfg = config.get("training", {}).get("nano_u_fine_tune", {})
    
    training_cfg = config.get("training", {}).get("nano_u", {}).copy()
    training_cfg.update(fine_tune_cfg) # Override defaults with fine-tuning specific params
    
    print(f"Fine-tuning Configuration:")
    print(f"  Epochs: {training_cfg.get('epochs')}")
    print(f"  Learning Rate: {training_cfg.get('learning_rate')}")
    print(f"  Batch Size: {training_cfg.get('batch_size')}")
    
    experiment_dir = Path("results/nano_u_fine_tuned")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    history = train_single_model(
        model=model,
        config=training_cfg,
        train_data=train_ds,
        val_data=val_ds,
        experiment_dir=str(experiment_dir)
    )
    
    # 5. Save the fine-tuned model
    print(f"Saving fine-tuned model to {output_model_path}")
    model.save(output_model_path)
    
    return True

if __name__ == "__main__":
    base_model = "models/nano_u.h5"
    output_model = "models/nano_u2.h5"
    fine_tune_model(base_model, output_model, epochs=10)
