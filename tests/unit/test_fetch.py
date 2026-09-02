"""Unit tests for src/nanou/fetch.py.

Nothing here touches the network: the split arithmetic and the layout check are
the parts that can be wrong silently, and both are pure.
"""

import pytest

from nanou.fetch import (
    BOTANICGARDEN,
    DATASETS,
    TINYAGRI,
    FetchError,
    sequential_split,
    spec_for,
    verify,
)


# -- the split, which is the part that can leak ----------------------------------

def test_split_is_sequential_not_random():
    """Consecutive frames are near-duplicates, so the split must be contiguous."""
    train, val, test = sequential_split(list(range(10)), 0.70, 0.20)
    assert train == list(range(7))
    assert val == [7, 8]
    assert test == [9]


def test_split_keeps_every_item():
    for n in range(0, 200):
        parts = sequential_split(list(range(n)), 0.70, 0.20)
        assert sum(len(p) for p in parts) == n


def test_split_parts_do_not_overlap():
    train, val, test = sequential_split(list(range(97)), 0.70, 0.20)
    assert max(train) < min(val) and max(val) < min(test)


def test_split_reproduces_the_shipped_tinyagri_sizes():
    """The four TinyAgri sequences, as released, split to the shipped counts."""
    per_sequence = {"Crops_scene1": 274, "Crops_scene2": 380,
                    "Tomatoes_scene1": 692, "Tomatoes_scene2": 464}
    totals = [0, 0, 0]
    for n in per_sequence.values():
        for i, part in enumerate(sequential_split(list(range(n)), 0.70, 0.20)):
            totals[i] += len(part)
    assert totals == [1267, 362, 181]


# -- dataset registry ------------------------------------------------------------

def test_spec_is_found_by_the_data_root_directory_name():
    assert spec_for("data/TinyAgri") is TINYAGRI
    assert spec_for("/anywhere/else/BotanicGarden") is BOTANICGARDEN


def test_an_unknown_dataset_lists_the_known_ones():
    with pytest.raises(FetchError, match="TinyAgri"):
        spec_for("data/Mystery")


def test_botanicgarden_has_no_download_and_says_where_to_get_it():
    assert BOTANICGARDEN.repo_id is None
    assert "robot-pesg/BotanicGarden" in BOTANICGARDEN.note


def test_every_registered_spec_is_keyed_by_its_own_name():
    assert all(name == spec.name for name, spec in DATASETS.items())


# -- layout verification ---------------------------------------------------------

def _dataset(root, counts=(2, 2, 2)):
    for split, n in zip(("train", "val", "test"), counts):
        (root / split / "img").mkdir(parents=True)
        (root / split / "mask").mkdir(parents=True)
        for i in range(n):
            (root / split / "img" / f"f{i}.png").touch()
            (root / split / "mask" / f"f{i}.png").touch()
    return root


def test_verify_returns_the_pair_counts(tmp_path):
    assert verify(_dataset(tmp_path / "ds", (5, 2, 1))) == {"train": 5, "val": 2, "test": 1}


def test_verify_rejects_a_missing_split(tmp_path):
    root = _dataset(tmp_path / "ds")
    for f in (root / "test" / "mask").iterdir():
        f.unlink()
    (root / "test" / "mask").rmdir()
    with pytest.raises(FetchError, match="missing test/img or test/mask"):
        verify(root)


def test_verify_rejects_an_image_without_its_mask(tmp_path):
    root = _dataset(tmp_path / "ds")
    (root / "train" / "img" / "orphan.png").touch()
    with pytest.raises(FetchError, match="do not pair up"):
        verify(root)
