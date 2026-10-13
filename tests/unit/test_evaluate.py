"""The three evaluation steps - score, plot, write_report - and how a split is listed."""

import json

import numpy as np
import pytest

from nanou.config import resolve
from nanou.evaluate import frame_groups, plot, score, split_files, write_report
from nanou.metrics import compute_segmentation_metrics


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    probs = rng.random((4, 6, 8, 1)).astype(np.float32)
    masks = (rng.random((4, 6, 8, 1)) > 0.5).astype(np.float32)
    return probs, masks


def test_score_is_the_segmentation_metrics_plus_the_two_losses(arrays):
    probs, masks = arrays
    results = score(probs, masks, threshold=0.4)
    expected = compute_segmentation_metrics(probs, masks, operating_threshold=0.4)
    assert results["miou"] == expected["miou"]
    assert results["threshold"] == 0.4
    assert np.isfinite(results["bce"]) and np.isfinite(results["focal"])


def test_the_report_is_one_file_that_says_what_it_describes(arrays, tmp_path):
    probs, masks = arrays
    results = score(probs, masks)
    path = write_report(results, tmp_path / "eval_int8_test.json", split="test", format="int8")
    report = json.loads((tmp_path / "eval_int8_test.json").read_text())
    assert path.endswith("eval_int8_test.json")
    assert list(report)[:2] == ["split", "format"]
    assert report["miou"] == results["miou"]


@pytest.mark.parametrize("groups", [None, ["a", "a", "b", "b"]])
def test_plot_writes_the_two_figures(arrays, tmp_path, groups):
    probs, masks = arrays
    imgs = np.zeros((4, 6, 8, 3), dtype=np.float32)
    paths = plot(probs, masks, imgs, score(probs, masks, groups), tmp_path, "eval_fp32_test")
    assert [p.split("/")[-1] for p in paths] == ["eval_fp32_test.png", "eval_fp32_test_metrics.png"]
    assert all((tmp_path / p.split("/")[-1]).stat().st_size > 0 for p in paths)


def _split_config(tmp_path, names):
    for sub in ("img", "mask"):
        (tmp_path / "test" / sub).mkdir(parents=True)
        for n in names:
            (tmp_path / "test" / sub / n).touch()
    preset = tmp_path / "preset.yaml"
    preset.write_text(f"extends: base\ndata_root: {tmp_path}\nnanou: {{}}\n")
    return resolve(path=preset)


def test_split_files_come_in_the_firmware_order(tmp_path):
    names = ["Tomatoes_scene2_frame490.png", "Crops_scene1_frame490.png", "Crops_scene1_frame7.png"]
    imgs, masks = split_files(_split_config(tmp_path, names), "test")
    order = ["Crops_scene1_frame7.png", "Crops_scene1_frame490.png", "Tomatoes_scene2_frame490.png"]
    assert [p.split("/")[-1] for p in imgs] == order
    assert [p.split("/")[-1] for p in masks] == order


def test_an_unknown_split_raises(tmp_path):
    with pytest.raises(ValueError, match="split"):
        split_files(_split_config(tmp_path, ["a_1.png"]), "holdout")


def test_frame_groups_are_skipped_when_rows_do_not_line_up():
    imgs = ["d6_s1_frame1.png", "d6_s1_frame2.png", "d7_s2_frame1.png"]
    assert frame_groups(imgs, 3) == ["d6_s1", "d6_s1", "d7_s2"]
    assert frame_groups(imgs, 2) is None
