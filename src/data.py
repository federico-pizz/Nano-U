"""Consolidated data pipeline: dataset preparation, augmentation, and synthetic data."""

import os
import re
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Tuple, Optional, Union

def sorted_by_frame(files: List[str]) -> List[str]:
    """Sort files by frame number in filename."""
    def get_frame_number(filename: str) -> int:
        if 'frame' in filename:
            m = re.search(r'frame(\d+)', filename)
        else:
            m = re.search(r'mask(\d+)', filename)
        return int(m.group(1)) if m else 0
    return sorted(files, key=get_frame_number)

def _augment_pair(img, mask, flip_prob=0.5, max_rotation_deg=20, 
                  brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
    """Apply random augmentations to an image-mask pair."""
    if not hasattr(_augment_pair, "_flip"):
        _augment_pair._flip = layers.RandomFlip("horizontal")
        _augment_pair._rotate = layers.RandomRotation(factor=max_rotation_deg/360.0, fill_mode="reflect")
    
    concat = tf.concat([img, mask], axis=-1)
    do_flip = tf.less(tf.random.uniform([]), flip_prob)
    concat = tf.cond(do_flip, lambda: _augment_pair._flip(concat, training=True), lambda: concat)
    concat = _augment_pair._rotate(concat, training=True)
    
    img = concat[..., :3]
    mask = concat[..., 3:4]
    mask = tf.cast(mask > 0.5, tf.float32)
    
    # Color jitter
    img = tf.image.random_brightness(img, brightness)
    img = tf.image.random_contrast(img, 1.0 - contrast, 1.0 + contrast)
    img = tf.image.random_saturation(img, 1.0 - saturation, 1.0 + saturation)
    img = tf.image.random_hue(img, hue)
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, mask

def make_dataset(img_files: List[str], mask_files: List[str], batch_size: int = 8, 
                 shuffle: bool = True, augment: bool = False,
                 mean: List[float] = [0.5, 0.5, 0.5], std: List[float] = [0.5, 0.5, 0.5],
                 **augment_kwargs) -> tf.data.Dataset:
    """Create a TensorFlow dataset from image and mask files."""
    if len(img_files) != len(mask_files):
        min_len = min(len(img_files), len(mask_files))
        img_files = img_files[:min_len]
        mask_files = mask_files[:min_len]

    img_files = sorted_by_frame(img_files)
    mask_files = sorted_by_frame(mask_files)
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    def _load_pair(img_path, mask_path):
        img = cv2.imread(img_path.decode())
        if img is None: return np.zeros((1,1,3), dtype=np.float32), np.zeros((1,1,1), dtype=np.float32)
        img = img.astype(np.float32)[:, :, ::-1] / 255.0
        img = (img - mean) / std
        
        mask = cv2.imread(mask_path.decode(), cv2.IMREAD_GRAYSCALE)
        if mask is None: return np.zeros((1,1,3), dtype=np.float32), np.zeros((1,1,1), dtype=np.float32)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, -1)
        return img, mask

    def _load_pair_tf(img_path, mask_path):
        img, mask = tf.numpy_function(_load_pair, [img_path, mask_path], [tf.float32, tf.float32])
        img.set_shape([None, None, 3])
        mask.set_shape([None, None, 1])
        return img, mask

    ds = tf.data.Dataset.from_tensor_slices((img_files, mask_files))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_files))
    ds = ds.map(_load_pair_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda i, m: _augment_pair(i, m, **augment_kwargs), 
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def get_synthetic_data(input_shape: Tuple[int, int, int] = (48, 64, 3), 
                       num_samples: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for testing and quick runs."""
    h, w, c = input_shape
    x = np.random.rand(num_samples, h, w, c).astype(np.float32)
    y = np.random.randint(0, 2, (num_samples, h, w, 1)).astype(np.float32)
    return x, y
