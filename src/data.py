"""Consolidated data pipeline: dataset preparation and augmentation."""

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
                 target_size: Optional[Tuple[int, int]] = None,
                 mean: Union[List[float], np.ndarray] = [0.5, 0.5, 0.5],
                 std: Union[List[float], np.ndarray] = [0.5, 0.5, 0.5],
                 **augment_kwargs) -> tf.data.Dataset:
    """Create a TensorFlow dataset from image and mask files."""
    img_files = sorted_by_frame(img_files)
    mask_files = sorted_by_frame(mask_files)
    
    if len(img_files) != len(mask_files):
        min_len = min(len(img_files), len(mask_files))
        img_files = img_files[:min_len]
        mask_files = mask_files[:min_len]

    mean_tf = tf.constant(mean, dtype=tf.float32)
    std_tf = tf.constant(std, dtype=tf.float32)
    
    # Store standard specific values to avoid kwargs pass-through conflicts
    augment_args = augment_kwargs.copy()

    def _load_pair_tf(img_path, mask_path):
        img_bytes = tf.io.read_file(img_path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        
        mask_bytes = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask_bytes, channels=1)
        mask = tf.cast(mask, tf.float32) / 255.0
        
        if target_size is not None:
            img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.BILINEAR)
            mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
        return img, mask

    def _normalize(img, mask):
        return (img - mean_tf) / std_tf, mask

    ds = tf.data.Dataset.from_tensor_slices((img_files, mask_files))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_files))
        
    ds = ds.map(_load_pair_tf, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        ds = ds.map(lambda i, m: _augment_pair(i, m, **augment_args), 
                    num_parallel_calls=tf.data.AUTOTUNE)
                    
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

