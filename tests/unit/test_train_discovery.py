"""Tests for train.py processed-data discovery and the end-to-end img↔mask
alignment of the real discovery → make_dataset path.

These guard the file-listing logic that feeds make_dataset: the glob, the
optional-validation handling, and the count-mismatch guard that must error
rather than silently truncate a malformed split.
"""

import cv2
import numpy as np
import pytest
import tensorflow as tf

from src.data import make_dataset
from src.train import discover_processed_pairs


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_solid_png(path, value, channels=3):
    shape = (60, 80, 3) if channels == 3 else (60, 80)
    cv2.imwrite(str(path), np.full(shape, value, dtype=np.uint8))


def _make_split(root, split, names, ids):
    """Create root/<split>/{img,mask}/*.png, painting each pair its id."""
    img_dir = root / split / "img"
    mask_dir = root / split / "mask"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for name, id_ in zip(names, ids):
        _write_solid_png(img_dir / f"{name}.png", id_, channels=3)
        _write_solid_png(mask_dir / f"{name}.png", id_, channels=1)
    return img_dir, mask_dir


def _processed_cfg(root, with_val=True):
    cfg = {"train": {"img": str(root / "train" / "img"),
                     "mask": str(root / "train" / "mask")}}
    if with_val:
        cfg["val"] = {"img": str(root / "val" / "img"),
                      "mask": str(root / "val" / "mask")}
    return cfg


# ── discover_processed_pairs: happy paths ────────────────────────────────────

def test_discover_happy_path(tmp_path):
    _make_split(tmp_path, "train", [f"f{k}" for k in range(5)], list(range(5)))
    _make_split(tmp_path, "val", [f"f{k}" for k in range(2)], list(range(2)))
    ti, tm, vi, vm = discover_processed_pairs(_processed_cfg(tmp_path))
    assert len(ti) == len(tm) == 5
    assert len(vi) == len(vm) == 2


def test_discover_no_val_dirs_returns_empty_val(tmp_path):
    _make_split(tmp_path, "train", [f"f{k}" for k in range(3)], list(range(3)))
    ti, tm, vi, vm = discover_processed_pairs(_processed_cfg(tmp_path, with_val=False))
    assert len(ti) == len(tm) == 3
    assert vi == [] and vm == []


def test_discover_val_count_mismatch_skips_val(tmp_path):
    _make_split(tmp_path, "train", [f"f{k}" for k in range(3)], list(range(3)))
    _, mask_dir = _make_split(tmp_path, "val", [f"f{k}" for k in range(3)], list(range(3)))
    (mask_dir / "f0.png").unlink()  # val now 3 imgs / 2 masks → skipped, not fatal
    ti, _, vi, vm = discover_processed_pairs(_processed_cfg(tmp_path))
    assert len(ti) == 3
    assert vi == [] and vm == []


# ── discover_processed_pairs: error paths ────────────────────────────────────

def test_discover_missing_train_key_raises():
    with pytest.raises(ValueError, match="processed.train"):
        discover_processed_pairs({"val": {}})


def test_discover_missing_train_dirs_raises(tmp_path):
    cfg = {"train": {"img": str(tmp_path / "nope" / "img"),
                     "mask": str(tmp_path / "nope" / "mask")}}
    with pytest.raises(FileNotFoundError, match="Training directories not found"):
        discover_processed_pairs(cfg)


def test_discover_empty_train_dir_raises(tmp_path):
    (tmp_path / "train" / "img").mkdir(parents=True)
    (tmp_path / "train" / "mask").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No training images"):
        discover_processed_pairs(_processed_cfg(tmp_path, with_val=False))


def test_discover_train_count_mismatch_raises(tmp_path):
    """A train img/mask count mismatch must error, not silently truncate."""
    _, mask_dir = _make_split(tmp_path, "train", [f"f{k}" for k in range(4)], list(range(4)))
    (mask_dir / "f0.png").unlink()  # 4 images, 3 masks
    with pytest.raises(ValueError, match="Mismatch in training images"):
        discover_processed_pairs(_processed_cfg(tmp_path, with_val=False))


# ── end-to-end: discovery → make_dataset alignment ───────────────────────────

def test_discovery_to_make_dataset_alignment(tmp_path):
    """The real path — train.py discovery feeding make_dataset — must pair the
    right image with the right mask even when frame numbers collide across
    'scenes' (the exact condition present in BotanicGarden / TinyAgri)."""
    names = [f"sceneA_frame{k}" for k in range(4)] + [f"sceneB_frame{k}" for k in range(4)]
    ids = list(range(4)) + [10 + k for k in range(4)]
    _make_split(tmp_path, "train", names, ids)

    ti, tm, _, _ = discover_processed_pairs(_processed_cfg(tmp_path, with_val=False))
    ds = make_dataset(ti, tm, batch_size=3, shuffle=False, augment=False,
                      mean=[0, 0, 0], std=[1, 1, 1])

    pairs = []
    for img_b, mask_b in ds:
        for i in range(int(img_b.shape[0])):
            img_id = int(round(float(tf.reduce_mean(img_b[i])) * 255.0))
            mask_id = int(round(float(tf.reduce_mean(mask_b[i])) * 255.0))
            pairs.append((img_id, mask_id))

    assert len(pairs) == len(names)
    for img_id, mask_id in pairs:
        assert img_id == mask_id, f"mispaired: image {img_id} with mask {mask_id}"
