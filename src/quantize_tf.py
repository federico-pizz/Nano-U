import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras

# Allow running the script directly (python src/quantize_tf.py)
# If executed directly, add project root so imports from `src` work.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_project_root
from src.utils.config import load_config
from src.utils.data_tf import make_dataset

def convert_to_tflite(model_path, out_path=None, int8=False, config=None):
    model = keras.models.load_model(model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Default: float32 conversion
    if int8:
        # Request DEFAULT optimizations (allows full integer when representative dataset provided)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset is required for full integer quantization
        if config:
            print("Preparing representative dataset for full-int8 quantization...")
            quant_cfg = config.get("quantization", {})
            norm_cfg = config["data"]["normalization"]

            # Use validation data for calibration
            root_dir = str(get_project_root())
            val_path_cfg = config["data"]["paths"]["processed"]["val"]
            val_img_dir = os.path.join(root_dir, val_path_cfg["img"]) if not os.path.isabs(val_path_cfg["img"]) else val_path_cfg["img"]
            val_mask_dir = os.path.join(root_dir, val_path_cfg["mask"]) if not os.path.isabs(val_path_cfg["mask"]) else val_path_cfg["mask"]

            val_img_files = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.png')]
            val_mask_files = [os.path.join(val_mask_dir, f) for f in os.listdir(val_mask_dir) if f.endswith('.png')]

            ds = make_dataset(
                val_img_files, val_mask_files,
                batch_size=1,
                shuffle=False,
                mean=norm_cfg["mean"],
                std=norm_cfg["std"]
            )

            limit = int(quant_cfg.get("representative_dataset_size", 100))
            images_np = []
            for img, _ in ds.take(limit):
                images_np.append(img.numpy())

            def rep_ds_gen():
                for img in images_np:
                    yield [img]

            converter.representative_dataset = rep_ds_gen

            # Force int8 I/O (recommended for embedded runtimes)
            try:
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            except Exception:
                pass

            # Restrict to TFLite builtins int8 for minimal runtime footprint
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            # Encourage per-channel quantization for weights
            try:
                converter.target_spec.supported_types = [tf.int8]
            except Exception:
                pass
        else:
            print("Error: Representative dataset required for full-int8 quantization. Provide --config or set config in call.")
            # fall back to dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()
    except Exception as e:
        err_msg = str(e)
        print("Initial TFLite conversion failed:", err_msg)
        # Some models require TF Select (TF kernel fallback) for ops not supported by pure TFLite.
        # Retry with SELECT_TF_OPS enabled which allows TensorFlow ops to be used at runtime.
        try:
            print("Retrying conversion enabling TF Select (SELECT_TF_OPS)...")
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_model = converter.convert()
        except Exception as e2:
            print("Conversion failed even with TF Select enabled:", str(e2))
            raise

    if out_path is None:
        root_dir = str(get_project_root())
        out_path = os.path.join(root_dir, "models", os.path.basename(model_path).replace(".keras", ".tflite"))

    # Create dir if not exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024.0
    print(f"Saved TFLite model to {out_path} ({size_kb:.1f} KB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite format")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .keras model file")
    parser.add_argument("--model-name", type=str, default="nano_u", choices=["nano_u", "bu_net"], help="Model name to auto-find")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--output", type=str, default=None, help="Output TFLite path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    root_dir = str(get_project_root())
    
    model_path = args.model_path
    if not model_path:
        # Try to find from config models_dir
        models_dir = config["data"]["paths"]["models_dir"]
        if not os.path.isabs(models_dir):
            models_dir = os.path.join(root_dir, models_dir)
        model_path = os.path.join(models_dir, f"{args.model_name}_tf_best.keras")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit(1)
    
    # Use output path from args or default next to model
    output_path = args.output
    
    # If int8 is requested via CLI, ensure we pass config
    convert_to_tflite(model_path, out_path=output_path, int8=args.int8, config=config)
