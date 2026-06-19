"""Unit tests for src/data.py."""

import os
import random
import re

import cv2
import numpy as np
import pytest
import tensorflow as tf

from src.data import (
    sorted_by_frame, make_dataset, augment_pair, sequence_group, grouped_kfold,
)


# ── grouped_kfold (leakage-safe folds) ───────────────────────────────────────

def _groups_from(seqs_with_counts):
    g = []
    for name, cnt in seqs_with_counts:
        g.extend([name] * cnt)
    return g


def test_grouped_kfold_no_group_spans_train_and_val():
    groups = _groups_from([("s1", 5), ("s2", 3), ("s3", 4), ("s4", 2), ("s5", 6)])
    g = np.array(groups)
    for tr, va in grouped_kfold(groups, k=3, seed=0):
        tr_groups, va_groups = set(g[tr]), set(g[va])
        assert tr_groups.isdisjoint(va_groups)  # the leakage guarantee


def test_grouped_kfold_val_sets_partition_all_indices():
    groups = _groups_from([("a", 4), ("b", 4), ("c", 4), ("d", 4)])
    splits = grouped_kfold(groups, k=4, seed=1)
    seen = np.concatenate([va for _, va in splits])
    assert sorted(seen.tolist()) == list(range(len(groups)))  # exact cover, no dups


def test_grouped_kfold_train_is_complement_of_val():
    groups = _groups_from([("a", 3), ("b", 3), ("c", 3)])
    n = len(groups)
    for tr, va in grouped_kfold(groups, k=3, seed=2):
        assert sorted(np.concatenate([tr, va]).tolist()) == list(range(n))
        assert set(tr).isdisjoint(set(va))


def test_grouped_kfold_raises_when_too_few_groups():
    with pytest.raises(ValueError):
        grouped_kfold(_groups_from([("a", 5), ("b", 5)]), k=5)


def test_grouped_kfold_raises_on_k_lt_2():
    with pytest.raises(ValueError):
        grouped_kfold(_groups_from([("a", 2), ("b", 2)]), k=1)


# ── sequence_group (leakage-safe CV grouping) ────────────────────────────────

def test_sequence_group_tinyagri():
    assert sequence_group("d6_s1_frame100.png") == "d6_s1"


def test_sequence_group_botanicgarden():
    assert sequence_group(
        "img_c54d7a_22290063136_seq_000000_000301.tif"
    ) == "seq_000000"


def test_sequence_group_bare_trailing_int_with_path():
    assert sequence_group("/data/train/img/scene3_005.png") == "scene3"


def test_sequence_group_no_number_is_whole_stem():
    assert sequence_group("weird_name.png") == "weird_name"


def test_sequence_group_keeps_distinct_scenes_apart():
    a = sequence_group("d6_s1_frame10.png")
    b = sequence_group("d6_s2_frame10.png")
    assert a != b  # different scenes must not collapse into one CV group


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


# ── image ↔ mask alignment ───────────────────────────────────────────────────
#
# make_dataset must pair each image with the mask sharing its basename, NOT by
# position. Independent sorting + positional zipping is fragile because
# sorted_by_frame keys only on the trailing frame number, which collides across
# scenes/subfolders; the surviving order then depends on arbitrary directory
# enumeration, so a plain re-copy can silently mispair >50% of the data.
#
# Encoding trick: each pair's image AND mask are painted a solid value equal to
# the pair's id, so after loading we can recover both ids and assert they match.

def _write_solid_png(path, value, channels):
    shape = (60, 80, 3) if channels == 3 else (60, 80)
    cv2.imwrite(str(path), np.full(shape, value, dtype=np.uint8))


def _build_pair_dir(tmp_path, names, ids):
    """Write img/ and mask/ PNGs where pair ``names[k]`` is painted ``ids[k]`` in
    both the image and the mask. Returns (img_paths, mask_paths)."""
    img_dir = tmp_path / "img"
    mask_dir = tmp_path / "mask"
    img_dir.mkdir()
    mask_dir.mkdir()
    img_paths, mask_paths = [], []
    for name, id_ in zip(names, ids):
        ip, mp = img_dir / f"{name}.png", mask_dir / f"{name}.png"
        _write_solid_png(ip, id_, channels=3)
        _write_solid_png(mp, id_, channels=1)
        img_paths.append(str(ip))
        mask_paths.append(str(mp))
    return img_paths, mask_paths


def _recover_pairs(ds):
    """Recover per-item (img_id, mask_id) from the solid pixel values.

    Built with mean=0, std=1 so a value v survives as v/255; ×255 recovers v.
    """
    pairs = []
    for img_b, mask_b in ds:
        for i in range(int(img_b.shape[0])):
            img_id = int(round(float(tf.reduce_mean(img_b[i])) * 255.0))
            mask_id = int(round(float(tf.reduce_mean(mask_b[i])) * 255.0))
            pairs.append((img_id, mask_id))
    return pairs


def test_make_dataset_pairs_by_basename_not_position(tmp_path):
    """Right image ↔ right mask even when the two lists are enumerated in
    different orders and frame numbers collide across 'scenes'."""
    # sceneA_frameK and sceneB_frameK both reduce to frame number K under
    # sorted_by_frame — the exact collision present in the real datasets.
    names = [f"sceneA_frame{k}" for k in range(4)] + [f"sceneB_frame{k}" for k in range(4)]
    ids = list(range(4)) + [10 + k for k in range(4)]
    imgs, masks = _build_pair_dir(tmp_path, names, ids)

    # Enumerate the two directories in unrelated orders (the re-copy scenario).
    random.Random(1).shuffle(imgs)
    random.Random(2).shuffle(masks)

    ds = make_dataset(imgs, masks, batch_size=3, shuffle=False, augment=False,
                      mean=[0, 0, 0], std=[1, 1, 1])
    pairs = _recover_pairs(ds)
    assert len(pairs) == len(names)
    for img_id, mask_id in pairs:
        assert img_id == mask_id, f"mispaired: image {img_id} with mask {mask_id}"


def test_make_dataset_missing_mask_raises(tmp_path):
    """An image without a same-name mask must error, not silently truncate."""
    names = [f"frame_{k:03d}" for k in range(5)]
    imgs, masks = _build_pair_dir(tmp_path, names, list(range(5)))
    with pytest.raises(ValueError, match="no mask"):
        make_dataset(imgs, masks[:-1], batch_size=2, shuffle=False)


def test_make_dataset_no_shared_basename_raises(tmp_path):
    imgs, _ = _build_pair_dir(tmp_path, [f"img_{k}" for k in range(3)], list(range(3)))
    other = tmp_path / "other"
    other.mkdir()
    bad_masks = []
    for k in range(3):
        p = other / f"unrelated_{k}.png"
        _write_solid_png(p, k, channels=1)
        bad_masks.append(str(p))
    with pytest.raises(ValueError, match="share a basename"):
        make_dataset(imgs, bad_masks, batch_size=2, shuffle=False)


def test_make_dataset_extra_masks_ignored(tmp_path):
    """More masks than images is fine — pairing is driven by the images."""
    names = [f"frame_{k:03d}" for k in range(5)]
    imgs, masks = _build_pair_dir(tmp_path, names, list(range(5)))
    ds = make_dataset(imgs[:3], masks, batch_size=2, shuffle=False,
                      mean=[0, 0, 0], std=[1, 1, 1])
    pairs = _recover_pairs(ds)
    assert len(pairs) == 3
    for img_id, mask_id in pairs:
        assert img_id == mask_id


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


# ── seeded shuffle / reproducibility ─────────────────────────────────────────

def test_make_dataset_seeded_shuffle_is_reproducible(tmp_path):
    """Same seed → identical epoch order, and pairing survives the shuffle."""
    names = [f"frame_{k:03d}" for k in range(12)]
    ids = list(range(12))
    imgs, masks = _build_pair_dir(tmp_path, names, ids)

    def order(seed):
        ds = make_dataset(imgs, masks, batch_size=4, shuffle=True, seed=seed,
                          mean=[0, 0, 0], std=[1, 1, 1])
        return _recover_pairs(ds)

    run_a = order(123)
    run_b = order(123)
    assert run_a == run_b, "same seed must give the same order"
    # every image is still with its own mask after shuffling
    for img_id, mask_id in run_a:
        assert img_id == mask_id
    # it is a genuine permutation of all items
    assert sorted(img_id for img_id, _ in run_a) == ids


def test_make_dataset_different_seed_changes_order(tmp_path):
    names = [f"frame_{k:03d}" for k in range(16)]
    imgs, masks = _build_pair_dir(tmp_path, names, list(range(16)))

    def order(seed):
        ds = make_dataset(imgs, masks, batch_size=4, shuffle=True, seed=seed,
                          mean=[0, 0, 0], std=[1, 1, 1])
        return [img_id for img_id, _ in _recover_pairs(ds)]

    assert order(1) != order(2), "different seeds should shuffle differently"


# ── Python ↔ Rust frame-ordering parity ──────────────────────────────────────

def _rust_frame_key(filename):
    """Faithful transcription of the sort key in firmware/build.rs:118-125:

        file_stem -> split('_').last()
                  -> trim_start_matches(non-ascii-digit)
                  -> parse::<u32>().ok()
                  -> unwrap_or(0)

    Kept in lockstep with build.rs so the parity test below detects drift on
    either side. eval_esp32 compares Python and firmware outputs frame-by-frame,
    so the two orderings MUST agree on the real processed filenames.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    last = stem.split('_')[-1]
    stripped = re.sub(r'^\D+', '', last)          # strip leading non-digits
    return int(stripped) if stripped.isdigit() else 0


def test_sorted_by_frame_matches_build_rs_ordering():
    """sorted_by_frame must order files identically to firmware/build.rs across
    the real processed naming schemes of both datasets (plus a few variants)."""
    names = [
        # BotanicGarden processed: folder prefix + zero-padded trailing counter
        "1005_05_img_c54d7a_22290063136_seq_000000_000101.png",
        "1005_07_img_c54d7a_22290071278_seq_000000_000001.png",
        "1006_01_img_c54d7a_22291022414_seq_000000_001201.png",
        "1008_01_img_c54d7a_22293041248_seq_000000_000301.png",
        # TinyAgri processed: scene prefix + frameN
        "Crops_scene1_frame100.png",
        "Crops_scene2_frame9.png",
        "Tomatoes_scene1_frame100.png",
        "Tomatoes_scene2_frame2.png",
        # bare-numeric and frameN variants
        "frame500.png", "frame20.png", "0421.png",
    ]
    shuffled = names[:]
    random.Random(0).shuffle(shuffled)

    py_order = sorted_by_frame(shuffled)
    rust_order = sorted(shuffled, key=_rust_frame_key)  # stable, like Rust sort_by_key
    assert py_order == rust_order
