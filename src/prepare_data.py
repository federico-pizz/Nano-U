"""
Data Preparation Script (Config Driven)

This script processes raw images and masks specified in the config/config.yaml file.
It supports multiple datasets and merges them into a single training/validation/test split.
For now, images are separated randomly, but later sequences will not be mixed to ensure
the model is not trained on previous or following frames of a video sequence.
"""

import os
import sys

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import random
import glob
import argparse
import numpy as np
from src.utils.config import load_config
from src.utils import get_project_root

def sort_by_frame(files):
    import re
    def get_frame_number(filename):
        m = re.search(r'(\d+)', os.path.basename(filename))
        return int(m.group(1)) if m else 0
    return sorted(files, key=get_frame_number)

def prepare_data(config_path="config/config.yaml"):
    config = load_config(config_path)
    root_dir = str(get_project_root())

    # Create output directories defined in config
    processed_paths = config["data"]["paths"]["processed"]
    out_dirs = {
        "train": {
            "img": os.path.join(root_dir, processed_paths["train"]["img"]),
            "mask": os.path.join(root_dir, processed_paths["train"]["mask"])
        },
        "val": {
            "img": os.path.join(root_dir, processed_paths["val"]["img"]),
            "mask": os.path.join(root_dir, processed_paths["val"]["mask"])
        },
        "test": {
            "img": os.path.join(root_dir, processed_paths["test"]["img"]),
            "mask": os.path.join(root_dir, processed_paths["test"]["mask"])
        }
    }

    for split in out_dirs:
        os.makedirs(out_dirs[split]["img"], exist_ok=True)
        os.makedirs(out_dirs[split]["mask"], exist_ok=True)

    target_shape = tuple(config["data"]["input_shape"][:2]) # (H, W) -> (48, 64)
    # CV2 resize expects (W, H)
    resize_dim = (target_shape[1], target_shape[0]) 

    # Collect all valid pairs from all datasets
    all_pairs = []

    raw_datasets = config["data"]["paths"].get("raw_datasets", [])
    
    if not raw_datasets:
        print("No 'raw_datasets' list found in config.")
        return

    dataset_idx = 0

    for ds in raw_datasets:
        ds_name = ds.get("name", f"ds_{dataset_idx}")
        img_dir = ds["images"]
        mask_dir = ds["masks"]
        
        # Resolve relative paths
        if not os.path.isabs(img_dir): img_dir = os.path.join(root_dir, img_dir)
        if not os.path.isabs(mask_dir): mask_dir = os.path.join(root_dir, mask_dir)

        if not os.path.exists(img_dir):
            print(f"Skipping missing dataset directory: {img_dir}")
            continue

        print(f"Processing dataset: {ds_name}...")
        
        img_files = glob.glob(os.path.join(img_dir, "*.png"))
        img_files = sort_by_frame(img_files)
        
        count = 0
        
        for img_path in img_files:
            basename = os.path.basename(img_path)
            
            # Robust matching logic
            if "frame" in basename:
                frame_part = basename.split("frame")[1]
                frame_num = frame_part.split(".")[0]
                mask_name = f"mask{frame_num}.png"
            else:
                mask_name = basename.replace("frame", "mask")
                
            mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, basename.replace("frame", "mask"))
                
            if os.path.exists(mask_path):
                all_pairs.append({
                    "img": img_path,
                    "mask": mask_path,
                    "ds_name": ds_name
                })
                count += 1
        
        print(f"  Found {count} pairs (scanned {len(img_files)} images).")
        dataset_idx += 1

    # Shuffle and Split
    random.seed(config["training"]["seed"])
    random.shuffle(all_pairs)
    
    splits = config["data"]["split"]
    n_total = len(all_pairs)
    n_train = int(n_total * splits["train"])
    n_val = int(n_total * splits["validation"])
    # Rest to test

    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train+n_val]
    test_pairs = all_pairs[n_train+n_val:]
    
    print(f"Total pairs: {n_total}")
    print(f"Split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

    def save_split(pairs, split_name):
        idx = 0
        for p in pairs:
            # Read
            img = cv2.imread(p["img"])
            mask = cv2.imread(p["mask"], cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None: continue
            
            # Resize
            img_res = cv2.resize(img, resize_dim)
            mask_res = cv2.resize(mask, resize_dim, interpolation=cv2.INTER_NEAREST) # Nearest for masks
            
            # Save
            # Naming convention: {split}_frame_{idx:05d}.png
            out_name = f"{split_name}_frame_{idx:05d}.png"
            
            cv2.imwrite(os.path.join(out_dirs[split_name]["img"], out_name), img_res)
            cv2.imwrite(os.path.join(out_dirs[split_name]["mask"], out_name), mask_res)
            idx += 1
            if idx % 100 == 0:
                print(f"  Saved {idx} images in {split_name}", end='\r')
        print(f"  Done saving {split_name} ({idx} images).")

    save_split(train_pairs, "train")
    save_split(val_pairs, "val")
    save_split(test_pairs, "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    prepare_data(args.config)
