import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras
from src.utils.config import load_config
from src.utils.data_tf import make_dataset

# Add project root to sys.path to resolve src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def convert_to_tflite(model_path, out_path=None, int8=False, config=None):
    model = keras.models.load_model(model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Setup representative dataset if config is available
        if config:
            print("Preparing representative dataset for quantization...")
            quant_cfg = config["quantization"]
            norm_cfg = config["data"]["normalization"]
            
            # Use validation data for calibration
            root_dir = str(get_project_root())
            val_path_cfg = config["data"]["paths"]["processed"]["val"]
            val_img_dir = os.path.join(root_dir, val_path_cfg["img"]) if not os.path.isabs(val_path_cfg["img"]) else val_path_cfg["img"]
            val_mask_dir = os.path.join(root_dir, val_path_cfg["mask"]) if not os.path.isabs(val_path_cfg["mask"]) else val_path_cfg["mask"]
            
            val_img_files = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.png')]
            val_mask_files = [os.path.join(val_mask_dir, f) for f in os.listdir(val_mask_dir) if f.endswith('.png')]
            
            # Create dataset
            # We don't need augmentation or shuffling, just raw valid samples
            ds = make_dataset(
                val_img_files, val_mask_files, 
                batch_size=1, # Process one by one for calibration
                shuffle=False, 
                mean=norm_cfg["mean"], 
                std=norm_cfg["std"]
            )
            
            limit = quant_cfg.get("representative_dataset_size", 100)
            
            def rep_ds_gen():
                count = 0
                for img, _ in ds:
                    if count >= limit:
                        break
                    # img is already (1, H, W, C) because batch_size=1
                    yield [img]
                    count += 1
            
            converter.representative_dataset = rep_ds_gen
            
            # Configurable IO types
            if quant_cfg.get("input_type") == "int8":
                converter.inference_input_type = tf.int8
            elif quant_cfg.get("input_type") == "uint8":
                converter.inference_input_type = tf.uint8
                
            if quant_cfg.get("output_type") == "int8":
                converter.inference_output_type = tf.int8
            elif quant_cfg.get("output_type") == "uint8":
                converter.inference_output_type = tf.uint8
                
            # Supported Ops
            ops = quant_cfg.get("supported_ops", ["TFLITE_BUILTINS_INT8"])
            target_ops = []
            if "TFLITE_BUILTINS_INT8" in ops:
                target_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS_INT8)
            if "SELECT_TF_OPS" in ops:
                target_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
            converter.target_spec.supported_ops = target_ops
            
        else:
            print("Warning: No config provided for INT8 quantization. Using defaults without representative dataset (may fail or give poor results).")

    tflite_model = converter.convert()
    
    if out_path is None:
        root_dir = str(get_project_root())
        out_path = os.path.join(root_dir, "models", os.path.basename(model_path).replace(".keras", ".tflite"))
        
    # Create dir if not exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {out_path}")
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
