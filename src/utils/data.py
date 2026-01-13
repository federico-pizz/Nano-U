import tensorflow as tf
import numpy as np
import cv2
import os
import math
from tensorflow.keras import layers


def sorted_by_frame(files):
    import re
    def get_frame_number(filename):
        if 'frame' in filename:
            m = re.search(r'frame(\d+)', filename)
        else:
            m = re.search(r'mask(\d+)', filename)
        return int(m.group(1)) if m else 0
    return sorted(files, key=get_frame_number)


def _augment_pair(img, mask,
                  flip_prob=0.5,
                  max_rotation_deg=20,
                  brightness=0.2,
                  contrast=0.2,
                  saturation=0.2,
                  hue=0.05):
    # Build preprocessing layers once (as static attributes) to reuse graph
    if not hasattr(_augment_pair, "_flip"):
        _augment_pair._flip = layers.RandomFlip("horizontal")
        _augment_pair._rotate = layers.RandomRotation(factor=max_rotation_deg/360.0, fill_mode="reflect")
    # Apply identical geometric transforms by concatenating image and mask
    concat = tf.concat([img, mask], axis=-1)
    # Random flipping controlled via flip_prob
    do_flip = tf.less(tf.random.uniform([]), flip_prob)
    concat = tf.cond(do_flip, lambda: _augment_pair._flip(concat, training=True), lambda: concat)
    # Random rotation
    concat = _augment_pair._rotate(concat, training=True)
    # Split back
    img = concat[..., :3]
    mask = concat[..., 3:4]
    # Threshold mask after interpolation to keep binary
    mask = tf.cast(mask > 0.5, tf.float32)
    # Color jitter on image only
    b_delta = tf.random.uniform([], -brightness, brightness)
    img = tf.image.adjust_brightness(img, b_delta)
    c_factor = tf.random.uniform([], 1.0 - contrast, 1.0 + contrast)
    img = tf.image.adjust_contrast(img, c_factor)
    s_factor = tf.random.uniform([], 1.0 - saturation, 1.0 + saturation)
    img = tf.image.adjust_saturation(img, s_factor)
    h_delta = tf.random.uniform([], -hue, hue)
    img = tf.image.adjust_hue(img, h_delta)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, mask


def make_dataset(img_files, mask_files, batch_size=8, shuffle=True, augment=False,
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                 flip_prob=0.5, max_rotation_deg=20,
                 brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
    # Validate inputs early and keep behavior stable
    if len(img_files) == 0:
        raise ValueError("make_dataset: img_files list is empty")
    if len(mask_files) == 0:
        raise ValueError("make_dataset: mask_files list is empty")
    if len(img_files) != len(mask_files):
        # Allow mismatch but warn and truncate to shortest
        min_len = min(len(img_files), len(mask_files))
        print(f"Warning: img_files and mask_files length mismatch ({len(img_files)} vs {len(mask_files)}). Truncating to {min_len} pairs.")
        img_files = img_files[:min_len]
        mask_files = mask_files[:min_len]

    img_files = sorted_by_frame(img_files)
    mask_files = sorted_by_frame(mask_files)

    # Ensure mean/std are numpy arrays for broadcasting
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    def _load_pair(img_path, mask_path):
        # Load and normalize using existing cv2 pipeline (maintains current dependencies)
        img = cv2.imread(img_path.decode())
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = img.astype(np.float32)[:, :, ::-1] / 255.0
        img = (img - mean) / std  # Normalize to specified range

        mask = cv2.imread(mask_path.decode(), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, -1)
        return img, mask

    def _load_pair_tf(img_path, mask_path):
        # Use numpy loader via numpy_function to preserve current behavior and avoid adding dependencies
        img, mask = tf.numpy_function(_load_pair, [img_path, mask_path], [tf.float32, tf.float32])
        img.set_shape([None, None, 3])
        mask.set_shape([None, None, 1])
        return img, mask

    ds = tf.data.Dataset.from_tensor_slices((img_files, mask_files))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_files))
    ds = ds.map(_load_pair_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda i, m: _augment_pair(i, m, flip_prob, max_rotation_deg, brightness, contrast, saturation, hue),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
