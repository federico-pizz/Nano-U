"""Unit tests for src/data.py."""

import numpy as np
import pytest
import tensorflow as tf

from src.data import sorted_by_frame, make_dataset, augment_pair


# ── sorted_by_frame ──────────────────────────────────────────────────────────

def test_sorted_by_frame_numeric_not_lexicographic():
    files = ["frame_10.png", "frame_2.png", "frame_9.png"]
    assert sorted_by_frame(files) == ["frame_2.png", "frame_9.png", "frame_10.png"]


def test_sorted_by_frame_leading_zeros():
    files = ["img_014.png", "img_003.png", "img_100.png"]
    result = sorted_by_frame(files)
    assert result == ["img_003.png", "img_014.png", "img_100.png"]


def test_sorted_by_frame_with_path_prefix():
    files = ["/data/test/frame_020.png", "/data/test/frame_005.png"]
    result = sorted_by_frame(files)
    assert result[0].endswith("005.png")
    assert result[1].endswith("020.png")


# ── make_dataset ─────────────────────────────────────────────────────────────

def test_make_dataset_output_shape(tmp_png_dataset):
    imgs, masks = tmp_png_dataset["imgs"], tmp_png_dataset["masks"]
    ds = make_dataset(imgs, masks, batch_size=2, shuffle=False, augment=False,
                      target_size=(60, 80))
    batch_img, batch_mask = next(iter(ds))
    assert batch_img.shape == (2, 60, 80, 3)
    assert batch_mask.shape == (2, 60, 80, 1)


def test_make_dataset_normalization_range(tmp_png_dataset):
    """After normalization with mean=0.5, std=0.5 pixel values should be in ~[-1, 1]."""
    imgs, masks = tmp_png_dataset["imgs"], tmp_png_dataset["masks"]
    ds = make_dataset(imgs, masks, batch_size=4, shuffle=False, augment=False,
                      target_size=(60, 80), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    batch_img, _ = next(iter(ds))
    vals = batch_img.numpy()
    assert vals.min() >= -1.1
    assert vals.max() <= 1.1


def test_make_dataset_no_shuffle_order(tmp_png_dataset):
    """With shuffle=False the dataset must yield images in sorted order."""
    imgs, masks = tmp_png_dataset["imgs"], tmp_png_dataset["masks"]
    ds = make_dataset(imgs, masks, batch_size=1, shuffle=False, augment=False,
                      target_size=(60, 80))
    first_run = [b[0].numpy() for b, _ in ds]
    second_run = [b[0].numpy() for b, _ in ds]
    for a, b in zip(first_run, second_run):
        np.testing.assert_array_equal(a, b)


def test_make_dataset_dtype(tmp_png_dataset):
    imgs, masks = tmp_png_dataset["imgs"], tmp_png_dataset["masks"]
    ds = make_dataset(imgs, masks, batch_size=2, shuffle=False, augment=False,
                      target_size=(60, 80))
    img_batch, mask_batch = next(iter(ds))
    assert img_batch.dtype == tf.float32
    assert mask_batch.dtype == tf.float32


def test_make_dataset_unknown_augment_kwarg_raises(tmp_png_dataset):
    imgs, masks = tmp_png_dataset["imgs"], tmp_png_dataset["masks"]
    with pytest.raises(ValueError, match="Unknown augmentation kwargs"):
        make_dataset(imgs, masks, batch_size=2, shuffle=False, augment=True,
                     target_size=(60, 80), nonexistent_param=0.5)


# ── augment_pair ─────────────────────────────────────────────────────────────

def test_augment_pair_same_spatial_transform():
    """Image and mask must receive identical spatial transforms.

    Strategy: set both image and mask to the SAME binary left-strip pattern
    (cols 0-19 = 1, rest = 0) with color jitter disabled.  After any flip the
    binarized image foreground and the mask foreground must be at exactly the
    same pixel positions.
    """
    # Left-strip pattern shared between image (all channels) and mask
    img_np = np.zeros((60, 80, 3), dtype=np.float32)
    img_np[:, :20, :] = 1.0
    mask_np = np.zeros((60, 80, 1), dtype=np.float32)
    mask_np[:, :20, 0] = 1.0

    img = tf.constant(img_np)
    mask = tf.constant(mask_np)

    for seed in range(10):
        tf.random.set_seed(seed)
        # No rotation — only flip and mild color jitter (bimodal 0/1 image survives)
        aug_img, aug_mask = augment_pair(img, mask, flip_prob=0.5, max_rotation_deg=0.0)
        # Binarize image (channel 0) and mask the same way
        img_fg = (aug_img.numpy()[..., 0] > 0.5)
        mask_fg = (aug_mask.numpy()[..., 0] > 0.5)
        np.testing.assert_array_equal(
            img_fg, mask_fg,
            err_msg=f"seed={seed}: image and mask spatial transforms differ"
        )


def test_augment_pair_mask_stays_binary():
    rng = np.random.default_rng(7)
    img = tf.constant(rng.random((60, 80, 3)).astype(np.float32))
    mask = tf.constant((rng.random((60, 80, 1)) > 0.5).astype(np.float32))
    _, aug_mask = augment_pair(img, mask)
    unique = set(np.unique(aug_mask.numpy()))
    assert unique <= {0.0, 1.0}
