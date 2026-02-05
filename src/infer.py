import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.utils import get_project_root
from src.models import create_nano_u, create_bu_net
from src.data import make_dataset
from src.utils.config import load_config


def load_model_by_path(model_path):
    # Robust model loading with clearer errors
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if model_path.endswith('.keras') or model_path.endswith('.h5'):
        try:
            return keras.models.load_model(model_path, compile=False)
        except Exception as e:
            raise RuntimeError(f"Failed loading model at {model_path}: {e}")
    raise ValueError(f"Unsupported model file: {model_path}")

def build_model_from_config(name: str, config: dict):
    input_shape = tuple(config["data"]["input_shape"])
    
    if name.lower() == "nano_u":
        cfg = config["models"]["nano_u"]
        return create_nano_u(
            input_shape=input_shape,
            filters=cfg["filters"],
            bottleneck=cfg["bottleneck"]
        )
    elif name.lower() == "bu_net":
        cfg = config["models"]["bu_net"]
        return create_bu_net(
            input_shape=input_shape,
            filters=cfg["filters"],
            bottleneck=cfg["bottleneck"]
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

def infer(model_name="nano_u", weights_path=None, batch_size=None, config_path="config/config.yaml"):
    config = load_config(config_path)
    root_dir = str(get_project_root())
    
    # Resolve paths
    def resolve_path(p):
        return p if os.path.isabs(p) else os.path.join(root_dir, p)
    
    processed_paths = config["data"]["paths"]["processed"]
    val_img_dir = resolve_path(processed_paths["val"]["img"])
    val_mask_dir = resolve_path(processed_paths["val"]["mask"])

    val_img_files = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.png')]
    val_mask_files = [os.path.join(val_mask_dir, f) for f in os.listdir(val_mask_dir) if f.endswith('.png')]

    if batch_size is None:
        batch_size = 8 # Default

    # Normalization params
    norm_cfg = config["data"]["normalization"]
    mean = norm_cfg["mean"]
    std = norm_cfg["std"]

    val_ds = make_dataset(val_img_files, val_mask_files, batch_size=batch_size, shuffle=False,
                          mean=mean, std=std)

    if weights_path:
        model = load_model_by_path(weights_path)
    else:
        # Build fresh model structure and try to load weights
        # Note: If loading by path (.keras) it usually contains architecture. 
        # But if we just want to load weights into a built model, we need the structure.
        
        # Try to find default best model
        default_path = resolve_path(os.path.join(config["data"]["paths"]["models_dir"], f"{model_name}_tf_best.keras"))
        
        if os.path.exists(default_path):
            print(f"Loading best available model from {default_path}")
            model = keras.models.load_model(default_path, compile=False)
        else:
            print("No weights found; using randomly initialized model (structure from config).")
            model = build_model_from_config(model_name, config)

    preds = []
    print("Running inference...")
    for batch_imgs, _ in val_ds:
        logits = model(batch_imgs, training=False)
        probs = tf.math.sigmoid(logits)
        preds.append(probs.numpy())
    
    if len(preds) > 0:
        preds = np.concatenate(preds, axis=0)
        print(f"Predictions shape: {preds.shape}")
        return preds
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model-name", default="nano_u") 
    parser.add_argument("--weights", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    infer(args.model_name, args.weights, args.batch_size, args.config)
