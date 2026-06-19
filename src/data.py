"""Consolidated data pipeline: dataset preparation and augmentation."""

import os
import re
import cv2
import numpy as np
import tensorflow as tf
import tf_keras as keras
layers = keras.layers
from typing import List, Tuple, Optional, Union


def sorted_by_frame(files: List[str]) -> List[str]:
    """Sort files by frame number. Matches build.rs logic: extracts the last underscore-separated integer."""
    def get_frame_number(filename: str) -> int:
        # Ignore directory path and extension
        name = os.path.splitext(os.path.basename(filename))[0]
        # Split by underscore and try to parse the last part
        parts = name.split('_')
        if parts:
            try:
                # Try the last part (e.g. 014001)
                return int(parts[-1])
            except ValueError:
                # Fallback to searching for any digits if the last part isn't an int
                digits = re.findall(r'\d+', name)
                if digits:
                    return int(digits[-1])
        return 0
    return sorted(files, key=get_frame_number)


def sequence_group(path: str) -> str:
    """Return the capture-sequence / scene id a frame belongs to.

    Leakage-safe cross-validation must keep all frames of one sequence in the
    same fold (adjacent frames are near-duplicates → temporal+spatial leakage),
    and evaluation reports per-sequence variance. This parses the grouping key
    from the basename for the two dataset naming conventions in use:

      TinyAgri:      ``d6_s1_frame100.png``                       → ``d6_s1``
      BotanicGarden: ``img_c54d7a_22290063136_seq_000000_000301`` → ``seq_000000``

    The key is everything up to (and excluding) the trailing frame number. If no
    trailing ``_<int>`` is present the whole stem is returned, so each file forms
    its own singleton group rather than being silently merged.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    # BotanicGarden: explicit ``seq_<id>`` token wins regardless of position.
    m = re.search(r"(seq_\d+)", name)
    if m:
        return m.group(1)
    # Otherwise strip a trailing frame index — a bare ``_<int>`` or a marked
    # ``_frame<int>`` / ``_f<int>`` / ``_img<int>`` token — keeping the prefix.
    m = re.match(r"(?i)^(.+)_(?:frame|f|img)?\d+$", name)
    if m:
        return m.group(1)
    return name


def grouped_kfold(groups: List[str], k: int, seed: int = 0):
    """K-fold split that keeps every group entirely within one fold.

    This is the leakage-safe partition for the CV hyperparameter search: a
    *whole* capture sequence (see :func:`sequence_group`) goes to either the
    train or the validation side of a fold, never both — so temporally/spatially
    adjacent near-duplicate frames can't leak validation signal into training.

    Args:
        groups: length-N group label per item (e.g. ``sequence_group`` of each
            image path). Order defines the returned indices.
        k: number of folds. Must be ≥2 and ≤ the number of distinct groups.
        seed: shuffles the *group* assignment to folds (not the items).

    Returns:
        ``[(train_idx, val_idx), ...]`` of length k, each a pair of int ndarrays
        indexing into ``groups``. The k validation index sets are disjoint and
        together cover every item exactly once.
    """
    groups = list(groups)
    n = len(groups)
    uniq = sorted(set(groups))
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if len(uniq) < k:
        raise ValueError(
            f"need at least k={k} distinct groups to fold, got {len(uniq)}. "
            "Too few sequences for this many folds."
        )
    rng = np.random.default_rng(seed)
    shuffled = list(uniq)
    rng.shuffle(shuffled)
    # Round-robin the (shuffled) groups onto folds → balanced group counts.
    fold_of = {g: i % k for i, g in enumerate(shuffled)}

    idx = np.arange(n)
    fold_id = np.array([fold_of[g] for g in groups])
    splits = []
    for f in range(k):
        val_idx = idx[fold_id == f]
        train_idx = idx[fold_id != f]
        splits.append((train_idx, val_idx))
    return splits


def augment_pair(
    img,
    mask,
    flip_prob: float = 0.5,
    max_rotation_deg: float = 20.0,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.05,
):
    """Apply random augmentations to an image–mask pair.

    Supported knobs (kept in sync with `make_dataset`):
      - flip_prob: probability of horizontal flip
      - max_rotation_deg: maximum rotation in degrees
      - brightness, contrast, saturation, hue: color jitter parameters
    """
    # Cache the augmentation layers once on the function object. They hold no
    # trainable weights (only RNG state), so sharing the same instances across
    # tf.data's parallel `map` calls is safe; the alternative — rebuilding layers
    # per element — would be far slower. NOTE: max_rotation_deg is read only on
    # first call, so changing it between calls in one process has no effect.
    if not hasattr(augment_pair, "_flip"):
        augment_pair._flip = layers.RandomFlip("horizontal")
        augment_pair._rotate = layers.RandomRotation(factor=max_rotation_deg/360.0, fill_mode="reflect")
    
    concat = tf.concat([img, mask], axis=-1)
    do_flip = tf.less(tf.random.uniform([]), flip_prob)
    concat = tf.cond(do_flip, lambda: augment_pair._flip(concat, training=True), lambda: concat)
    concat = augment_pair._rotate(concat, training=True)
    
    img = concat[..., :3]
    mask = concat[..., 3:4]
    mask = tf.cast(mask > 0.5, tf.float32)
    
    # Color jitter. Each knob is guarded: a value of 0 means "disabled" (e.g. the
    # geometric-only regime zeroes all of these). Calling the TF ops with a 0 knob
    # would pass lower == upper and raise "upper must be > lower" (contrast /
    # saturation) — these are static Python floats at trace time, so the guards
    # fold away cleanly inside the tf.data map.
    if brightness > 0:
        img = tf.image.random_brightness(img, brightness)
    if contrast > 0:
        img = tf.image.random_contrast(img, 1.0 - contrast, 1.0 + contrast)
    if saturation > 0:
        img = tf.image.random_saturation(img, 1.0 - saturation, 1.0 + saturation)
    if hue > 0:
        img = tf.image.random_hue(img, hue)
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, mask

def make_dataset(img_files: List[str], mask_files: List[str], batch_size: int = 8,
                 shuffle: bool = True, augment: bool = False,
                 target_size: Optional[Tuple[int, int]] = None,
                 mean: Union[List[float], np.ndarray] = [0.5, 0.5, 0.5],
                 std: Union[List[float], np.ndarray] = [0.5, 0.5, 0.5],
                 seed: Optional[int] = None,
                 **augment_kwargs) -> tf.data.Dataset:
    """Create a TensorFlow dataset from image and mask files.

    Any extra keyword arguments are forwarded to `augment_pair` when
    `augment=True`. Unknown keys raise a ValueError early so that typos in
    config do not silently get ignored.

    `seed` seeds the shuffle so that, given the same global seed, the epoch
    order is reproducible across runs.
    """
    # Pair each image to its mask by identical basename. The extraction tools
    # give every img/mask pair the same unique filename, so this is exact and
    # independent of directory enumeration order. Positional zipping after two
    # independent sorts is fragile: sorted_by_frame keys only on the trailing
    # frame number, which collides across scenes/subfolders, and stable-sort
    # tie-breaking then leaks the arbitrary glob order of each directory.
    img_files = sorted_by_frame(img_files)  # determines split / shuffle order
    mask_by_name = {os.path.basename(m): m for m in mask_files}
    paired = [(i, mask_by_name[os.path.basename(i)])
              for i in img_files if os.path.basename(i) in mask_by_name]

    if not paired:
        raise ValueError(
            "No image/mask pairs share a basename; cannot build dataset "
            f"({len(img_files)} images, {len(mask_files)} masks)."
        )
    missing = len(img_files) - len(paired)
    if missing:
        raise ValueError(
            f"{missing} image(s) have no mask with a matching filename; "
            "refusing to fall back to positional pairing."
        )
    img_files = [i for i, _ in paired]
    mask_files = [m for _, m in paired]

    mean_tf = tf.constant(mean, dtype=tf.float32)
    std_tf = tf.constant(std, dtype=tf.float32)
    
    # Only allow kwargs that `_augment_pair` actually supports.
    allowed_augment_keys = {
        "flip_prob",
        "max_rotation_deg",
        "brightness",
        "contrast",
        "saturation",
        "hue",
    }
    unknown_keys = set(augment_kwargs) - allowed_augment_keys
    if unknown_keys:
        raise ValueError(
            f"Unknown augmentation kwargs {sorted(unknown_keys)}; "
            f"allowed keys are {sorted(allowed_augment_keys)}"
        )
    augment_args = {k: v for k, v in augment_kwargs.items() if k in allowed_augment_keys}

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
        ds = ds.shuffle(buffer_size=len(img_files), seed=seed)
        
    ds = ds.map(_load_pair_tf, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        ds = ds.map(lambda i, m: augment_pair(i, m, **augment_args), 
                    num_parallel_calls=tf.data.AUTOTUNE)
                    
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

