"""Unit tests for the dataset extraction tools in tools/.

These scripts are the source of truth for the processed splits, but had no
coverage. Tests here lock the split arithmetic, the road-color masking, and the
numeric frame ordering that keeps the sequential split in true capture order.
"""

import numpy as np
import pytest
from PIL import Image

# The tools/ extraction scripts are kept out of the repo (gitignored), so they
# are only importable in a local dev checkout. Skip this module cleanly when they
# are absent (e.g. on CI / a fresh clone) instead of failing collection.
_eb = pytest.importorskip("tools.extract_botanicgarden")
_et = pytest.importorskip("tools.extract_tinyagri")

ROAD_COLOR = _eb.ROAD_COLOR
extract_binary_road = _eb.extract_binary_road
frame_key = _eb.frame_key
sequential_split = _eb.sequential_split
get_frame_num = _et.get_frame_num


# ── sequential_split ─────────────────────────────────────────────────────────

def test_sequential_split_fractions():
    train, val, test = sequential_split(list(range(10)), 0.70, 0.20)
    assert train == list(range(7))
    assert val == [7, 8]
    assert test == [9]


def test_sequential_split_is_an_ordered_partition():
    items = list(range(137))
    train, val, test = sequential_split(items, 0.70, 0.20)
    # sequential (not shuffled): concatenation reproduces the input exactly
    assert train + val + test == items
    assert not (set(train) & set(val))
    assert not (set(val) & set(test))


def test_sequential_split_tiny_input():
    train, val, test = sequential_split([1, 2], 0.70, 0.20)
    assert train + val + test == [1, 2]


# ── frame_key (numeric ordering, the lexicographic-sort fix) ─────────────────

def test_frame_key_numeric_not_lexicographic():
    files = ["seq_2.tif", "seq_10.tif", "seq_9.tif"]
    assert sorted(files, key=frame_key) == ["seq_2.tif", "seq_9.tif", "seq_10.tif"]


def test_frame_key_padded_and_unpadded_and_paths():
    assert frame_key("x_000101.tif") == 101
    assert frame_key("/a/b/c_5.png") == 5
    assert frame_key("img_c54d7a_22290063136_seq_000000_000301.tif") == 301


def test_frame_key_no_digits_is_zero():
    assert frame_key("nodigits.tif") == 0


# ── extract_binary_road ──────────────────────────────────────────────────────

def test_extract_binary_road_matches_only_target_color(tmp_path):
    h, w = 4, 6
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[0, 0] = ROAD_COLOR            # road
    arr[1, 1] = ROAD_COLOR            # road
    arr[2, 2] = (248, 249, 173)       # off-by-one in B → must NOT match
    p = tmp_path / "mask.png"
    Image.fromarray(arr, "RGB").save(p)

    out = extract_binary_road(str(p), ROAD_COLOR)
    assert out.shape == (h, w)
    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}
    assert out[0, 0] == 255 and out[1, 1] == 255
    assert out[2, 2] == 0
    assert int((out == 255).sum()) == 2


# ── TinyAgri get_frame_num ───────────────────────────────────────────────────

def test_get_frame_num_image_and_mask():
    assert get_frame_num("d6_s1_frame100.png") == 100
    assert get_frame_num("mask100.png") == 100
    assert get_frame_num("no_number_here.png") is None
