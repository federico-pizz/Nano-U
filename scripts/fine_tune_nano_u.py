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

class DataRehearsalCallback(keras.callbacks.Callback):
    """Monitor validation performance on separate datasets during fine-tuning."""
    def __init__(self, botanic_val_ds, tinyagri_val_ds):
        super().__init__()
        self.botanic_val_ds = botanic_val_ds
        self.tinyagri_val_ds = tinyagri_val_ds
        self.botanic_iou = BinaryIoU(name="val_iou_botanic", from_logits=True)
        self.tinyagri_iou = BinaryIoU(name="val_iou_tinyagri", from_logits=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Evaluate Botanic Garden
        self.botanic_iou.reset_state()
        for x, y in self.botanic_val_ds:
            y_pred = self.model(x, training=False)
            self.botanic_iou.update_state(y, y_pred)
        logs["val_iou_botanic"] = self.botanic_iou.result().numpy()
        
        # Evaluate TinyAgri
        self.tinyagri_iou.reset_state()
        for x, y in self.tinyagri_val_ds:
            y_pred = self.model(x, training=False)
            self.tinyagri_iou.update_state(y, y_pred)
        logs["val_iou_tinyagri"] = self.tinyagri_iou.result().numpy()
        
        print(f" - val_iou_botanic: {logs['val_iou_botanic']:.4f} - val_iou_tinyagri: {logs['val_iou_tinyagri']:.4f}")

def fine_tune_model(base_model_path: str, output_model_path: str, epochs: int = 10):
    """
    Load an existing Nano-U model and fine-tune it on the secondary dataset.
    """
    print(f"\n--- Fine-tuning {base_model_path} ---")
    
    # 1. Load Configuration
    config = load_config("config/config.yaml")
    
    # 2. Prepare Datasets (Data Rehearsal)
    # Primary dataset (botanic_garden)
    primary_data = config.get("data", {}).get("paths", {}).get("processed", {})
    p_train_cfg = primary_data.get("train", {})
    p_val_cfg = primary_data.get("val", {})
    pt_img_dir = Path(p_train_cfg.get("img", ""))
    pt_mask_dir = Path(p_train_cfg.get("mask", ""))
    pv_img_dir = Path(p_val_cfg.get("img", ""))
    pv_mask_dir = Path(p_val_cfg.get("mask", ""))

    # Secondary dataset (tinyagri)
    secondary_data = config.get("data", {}).get("paths", {}).get("secondary", {})
    if not secondary_data:
        print("Error: 'data.paths.secondary' not found in config.yaml")
        return False
    
    s_train_cfg = secondary_data.get("train", {})
    s_val_cfg = secondary_data.get("val", {})
    st_img_dir = Path(s_train_cfg.get("img", ""))
    st_mask_dir = Path(s_train_cfg.get("mask", ""))
    sv_img_dir = Path(s_val_cfg.get("img", ""))
    sv_mask_dir = Path(s_val_cfg.get("mask", ""))
    
    # MIX TRAINING: Combine botanic_garden and tinyagri
    train_img_files = []
    train_mask_files = []
    
    if pt_img_dir.exists() and pt_mask_dir.exists():
        train_img_files.extend(sorted([str(f) for f in pt_img_dir.glob("*.png")]))
        train_mask_files.extend(sorted([str(f) for f in pt_mask_dir.glob("*.png")]))
        print(f"Loaded {len(train_img_files)} training samples from botanic_garden.")
    
    if st_img_dir.exists() and st_mask_dir.exists():
        s_train_imgs = sorted([str(f) for f in st_img_dir.glob("*.png")])
        s_train_masks = sorted([str(f) for f in st_mask_dir.glob("*.png")])
        train_img_files.extend(s_train_imgs)
        train_mask_files.extend(s_train_masks)
        print(f"Added {len(s_train_imgs)} training samples from tinyagri (Total: {len(train_img_files)}).")

    # VALIDATION: Separate and Combined
    val_img_botanic = sorted([str(f) for f in pv_img_dir.glob("*.png")])
    val_mask_botanic = sorted([str(f) for f in pv_mask_dir.glob("*.png")])
    val_img_tinyagri = sorted([str(f) for f in sv_img_dir.glob("*.png")])
    val_mask_tinyagri = sorted([str(f) for f in sv_mask_dir.glob("*.png")])
    
    val_img_all = val_img_botanic + val_img_tinyagri
    val_mask_all = val_mask_botanic + val_mask_tinyagri

    print(f"Validation: {len(val_img_botanic)} botanic, {len(val_img_tinyagri)} tinyagri (Total: {len(val_img_all)})")

    norm_mean = config.get("data", {}).get("normalization", {}).get("mean", [0.5, 0.5, 0.5])
    norm_std = config.get("data", {}).get("normalization", {}).get("std", [0.5, 0.5, 0.5])
    input_shape = config.get("data", {}).get("input_shape", [60, 80, 3])
    target_size = (input_shape[0], input_shape[1])
    batch_size = config.get("training", {}).get("nano_u", {}).get("batch_size", 8)

    train_ds = make_dataset(
        train_img_files, train_mask_files,
        batch_size=batch_size, augment=True, shuffle=True,
        target_size=target_size,
        mean=norm_mean, std=norm_std
    )
    
    # Combined val set for checkpointing
    val_ds = make_dataset(
        val_img_all, val_mask_all,
        batch_size=batch_size, augment=False,
        target_size=target_size,
        mean=norm_mean, std=norm_std
    )
    
    # Separate val sets for metric tracking
    botanic_val_ds = make_dataset(
        val_img_botanic, val_mask_botanic,
        batch_size=batch_size, augment=False,
        target_size=target_size,
        mean=norm_mean, std=norm_std
    )
    tinyagri_val_ds = make_dataset(
        val_img_tinyagri, val_mask_tinyagri,
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
    
    # Add Rehearsal monitoring
    rehearsal_cb = DataRehearsalCallback(botanic_val_ds, tinyagri_val_ds)
    
    history = train_single_model(
        model=model,
        config=training_cfg,
        train_data=train_ds,
        val_data=val_ds,
        experiment_dir=str(experiment_dir),
        extra_callbacks=[rehearsal_cb]
    )
    
    # 5. Save the fine-tuned model
    print(f"Saving fine-tuned model to {output_model_path}")
    model.save(output_model_path)
    
    return True

if __name__ == "__main__":
    base_model = "models/nano_u.h5"
    output_model = "models/nano_u2.h5"
    fine_tune_model(base_model, output_model, epochs=10)
