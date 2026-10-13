"""Consolidated data pipeline: dataset preparation and augmentation."""

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tf_keras as keras

layers = keras.layers


def sorted_by_frame(files: list[str]) -> list[str]:
    """Sort files by frame number, then by file name - the order build.rs embeds the test images in.

    The frame number is the last underscore-separated integer. It is not unique:
    sequences reuse frame numbers (``Crops_scene1_frame490`` and
    ``Tomatoes_scene2_frame490``), so the file name breaks ties. Without it the
    order among tied frames is whatever the directory listing gave, and the
    firmware's image i is no longer the host's image i.
    """

    def get_frame_number(filename: str) -> int:
        # Ignore directory path and extension
        name = os.path.splitext(os.path.basename(filename))[0]
        # Split by underscore and try to parse the last part
        parts = name.split("_")
        if parts:
            try:
                # Try the last part (e.g. 014001)
                return int(parts[-1])
            except ValueError:
                # Fallback to searching for any digits if the last part isn't an int
                digits = re.findall(r"\d+", name)
                if digits:
                    return int(digits[-1])
        return 0

    return sorted(files, key=lambda f: (get_frame_number(f), os.path.basename(f)))


def sequence_group(path: str) -> str:
    """Return the capture-sequence / scene id a frame belongs to.

    Leakage-safe cross-validation must keep all frames of one sequence in the
    same fold (adjacent frames are near-duplicates -> temporal+spatial leakage),
    and evaluation reports per-sequence variance. This parses the grouping key
    from the basename for the two dataset naming conventions in use:

      TinyAgri:      ``Crops_scene1_frame100.png``                -> ``Crops_scene1``
      BotanicGarden: ``img_c54d7a_22290063136_seq_000000_000301`` -> ``seq_000000``

    The key is everything up to (and excluding) the trailing frame number. If no
    trailing ``_<int>`` is present the whole stem is returned, so each file forms
    its own singleton group rather than being silently merged.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    # BotanicGarden: explicit ``seq_<id>`` token wins regardless of position.
    m = re.search(r"(seq_\d+)", name)
    if m:
        return m.group(1)
    # Otherwise strip a trailing frame index - a bare ``_<int>`` or a marked
    # ``_frame<int>`` / ``_f<int>`` / ``_img<int>`` token - keeping the prefix.
    m = re.match(r"(?i)^(.+)_(?:frame|f|img)?\d+$", name)
    if m:
        return m.group(1)
    return name


def grouped_kfold(groups: list[str], k: int, seed: int = 0):
    """K-fold split that keeps every group entirely within one fold.

    This is the leakage-safe partition for the CV hyperparameter search: a
    *whole* capture sequence (see :func:`sequence_group`) goes to either the
    train or the validation side of a fold, never both - so temporally/spatially
    adjacent near-duplicate frames can't leak validation signal into training.

    Args:
        groups: length-N group label per item (e.g. ``sequence_group`` of each
            image path). Order defines the returned indices.
        k: number of folds. Must be >=2 and <= the number of distinct groups.
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
    # Round-robin the (shuffled) groups onto folds -> balanced group counts.
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
    """Apply random augmentations to an image-mask pair.

    Supported knobs (kept in sync with `make_dataset`):
      - flip_prob: probability of horizontal flip
      - max_rotation_deg: maximum rotation in degrees
      - brightness, contrast, saturation, hue: color jitter parameters
    """
    # Cache the augmentation layers on the function object. They hold no
    # trainable weights (only RNG state), so sharing the same instances across
    # tf.data's parallel `map` calls is safe; the alternative - rebuilding layers
    # per element - would be far slower. The rotation layer bakes in
    # `max_rotation_deg`, so it is cached per angle: an in-process sweep over
    # augmentation regimes (e.g. a sweep with --jobs 1) would otherwise
    # silently reuse the first regime's rotation for every later regime.
    if not hasattr(augment_pair, "_flip"):
        augment_pair._flip = layers.RandomFlip("horizontal")
        augment_pair._rotate_cache = {}
    rotate = augment_pair._rotate_cache.get(max_rotation_deg)
    if rotate is None and max_rotation_deg > 0:
        rotate = layers.RandomRotation(factor=max_rotation_deg / 360.0, fill_mode="reflect")
        augment_pair._rotate_cache[max_rotation_deg] = rotate

    concat = tf.concat([img, mask], axis=-1)
    do_flip = tf.less(tf.random.uniform([]), flip_prob)
    concat = tf.cond(do_flip, lambda: augment_pair._flip(concat, training=True), lambda: concat)
    # max_rotation_deg == 0 disables rotation outright (a factor-0 RandomRotation
    # is a no-op but still costs a resample). Static Python float at trace time,
    # so the guard folds away inside the tf.data map.
    if rotate is not None:
        concat = rotate(concat, training=True)

    img = concat[..., :3]
    mask = concat[..., 3:4]
    mask = tf.cast(mask > 0.5, tf.float32)

    # Color jitter. Each knob is guarded: a value of 0 means "disabled" (e.g. the
    # geometric-only regime zeroes all of these). Calling the TF ops with a 0 knob
    # would pass lower == upper and raise "upper must be > lower" (contrast /
    # saturation) - these are static Python floats at trace time, so the guards
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


def make_dataset(
    img_files: list[str],
    mask_files: list[str],
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = False,
    target_size: tuple[int, int] | None = None,
    mean: Sequence[float] | np.ndarray = (0.5, 0.5, 0.5),
    std: Sequence[float] | np.ndarray = (0.5, 0.5, 0.5),
    seed: int | None = None,
    **augment_kwargs,
) -> tf.data.Dataset:
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
    paired = [
        (i, mask_by_name[os.path.basename(i)])
        for i in img_files
        if os.path.basename(i) in mask_by_name
    ]

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
        ds = ds.map(
            lambda i, m: augment_pair(i, m, **augment_args), num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def as_dataset(
    train_data: tf.data.Dataset | tuple[tf.Tensor, tf.Tensor],
    val_data: tf.data.Dataset | tuple[tf.Tensor, tf.Tensor] | None,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset | None]:
    """Normalize train/val inputs into tf.data.Datasets for distillation."""
    if isinstance(train_data, tf.data.Dataset):
        train_dataset = train_data
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if val_data is None:
        return train_dataset, None

    if isinstance(val_data, tf.data.Dataset):
        val_dataset = val_data
    else:
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset


def discover_processed_pairs(
    processed: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Resolve processed train/val image and mask file lists from the split paths.

    Returns ``(train_img_files, train_mask_files, val_img_files, val_mask_files)``.
    Validation lists come back empty when the val directories are absent or their
    image/mask counts disagree (validation is treated as optional).

    Raises:
        ValueError: 'train' missing from the processed config, or the train
            image/mask counts disagree (a malformed split - caught early rather
            than silently truncated downstream).
        FileNotFoundError: training directories missing, or no training images.

    Note: the actual image<->mask pairing is done by basename in ``make_dataset``;
    the count checks here are only a cheap guard against an obviously broken split.
    """
    if not isinstance(processed, dict) or "train" not in processed:
        raise ValueError("no 'train' split in the split paths. Processed data is required.")

    train_cfg = processed.get("train", {})
    val_cfg = processed.get("val", {})

    t_img_dir = Path(train_cfg.get("img", ""))
    t_mask_dir = Path(train_cfg.get("mask", ""))
    v_img_dir = Path(val_cfg.get("img", ""))
    v_mask_dir = Path(val_cfg.get("mask", ""))

    if not t_img_dir.exists() or not t_mask_dir.exists():
        raise FileNotFoundError(f"Training directories not found: {t_img_dir} or {t_mask_dir}")

    train_img_files = [str(f) for f in t_img_dir.glob("*.png")]
    train_mask_files = [str(f) for f in t_mask_dir.glob("*.png")]

    if not train_img_files:
        raise FileNotFoundError(f"No training images found in {t_img_dir}")
    if len(train_img_files) != len(train_mask_files):
        raise ValueError(
            f"Mismatch in training images ({len(train_img_files)}) "
            f"and masks ({len(train_mask_files)})"
        )

    val_img_files: list[str] = []
    val_mask_files: list[str] = []
    if v_img_dir.exists() and v_mask_dir.exists():
        val_img_files = [str(f) for f in v_img_dir.glob("*.png")]
        val_mask_files = [str(f) for f in v_mask_dir.glob("*.png")]
        if not (val_img_files and len(val_img_files) == len(val_mask_files)):
            val_img_files, val_mask_files = [], []

    return train_img_files, train_mask_files, val_img_files, val_mask_files
