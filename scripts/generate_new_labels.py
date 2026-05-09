"""Run full inference with bu_net on all TinyAgri images and save predicted masks to data/new_labels.

Source structure:  data/TinyAgri/{category}/{scene}/*.png
Output structure:  data/new_labels/{category}/{scene}/*.png
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tf_keras as keras
import tensorflow_model_optimization as tfmot

from src.utils import BinaryIoU
from src.models import PadToMatch

CONFIG = {
    "model_path": "models/tinyagri/bu_net.h5",
    "input_shape": (60, 80),
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "threshold": 0.5,
    "src_root": "data/TinyAgri",
    "dst_root": "data/new_labels",
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_model(model_path: str):
    with tfmot.quantization.keras.quantize_scope():
        with keras.utils.custom_object_scope({"BinaryIoU": BinaryIoU, "PadToMatch": PadToMatch}):
            return keras.models.load_model(model_path, compile=False)


def preprocess(img_path: str, input_shape, mean, std) -> np.ndarray:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return img[np.newaxis]  # [1, H, W, 3]


def predict_mask(model, img_batch: np.ndarray, threshold: float, orig_size) -> np.ndarray:
    out = model(img_batch, training=False).numpy()[0, ..., 0]  # [H, W]
    if out.min() < -1e-3 or out.max() > 1.0 + 1e-3:
        out = 1.0 / (1.0 + np.exp(-out))
    binary = (out > threshold).astype(np.uint8) * 255
    if (binary.shape[0], binary.shape[1]) != (orig_size[0], orig_size[1]):
        binary = cv2.resize(binary, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    return binary


def collect_image_folders(src_root: str):
    """Walk src_root and return list of (rel_subdir, [img_paths]) for every leaf dir with images."""
    results = []
    for dirpath, dirnames, filenames in os.walk(src_root):
        # skip the masks subtree
        if os.path.basename(dirpath) == "masks":
            dirnames.clear()
            continue
        imgs = sorted([os.path.join(dirpath, f) for f in filenames if f.lower().endswith(".png")])
        if imgs:
            rel = os.path.relpath(dirpath, src_root)
            results.append((rel, imgs))
    return results


def main():
    model_path = os.path.join(PROJECT_ROOT, CONFIG["model_path"])
    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    print("Model loaded.")

    src_root = os.path.join(PROJECT_ROOT, CONFIG["src_root"])
    dst_root = os.path.join(PROJECT_ROOT, CONFIG["dst_root"])

    folders = collect_image_folders(src_root)
    print(f"Found {len(folders)} image folder(s) under {src_root}")

    for rel, img_paths in folders:
        out_dir = os.path.join(dst_root, rel)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{rel}] {len(img_paths)} images -> {out_dir}")

        for img_path in img_paths:
            orig = cv2.imread(img_path)
            orig_h, orig_w = orig.shape[:2]
            batch = preprocess(img_path, CONFIG["input_shape"], CONFIG["mean"], CONFIG["std"])
            mask = predict_mask(model, batch, CONFIG["threshold"], (orig_h, orig_w))
            out_path = os.path.join(out_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, mask)

        print(f"[{rel}] done.")

    print("All done. Labels saved to", dst_root)


if __name__ == "__main__":
    main()
