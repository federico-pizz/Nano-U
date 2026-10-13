"""Which images calibrate an INT8 export, and what a noise export means.

`calibration_images` decides that: it is the one place where a bad or missing
calibration set has to be refused rather than quietly turned into noise, and it
is what the calibration ablation varies per export.
"""

import pytest

from nanou.config import resolve
from nanou.quantize import calibration_images


def _config(tmp_path, images=("a.png", "b.png"), **quant):
    """A preset rooted at tmp_path, with `images` in its val split."""
    for split in ("train", "val", "test"):
        (tmp_path / split / "img").mkdir(parents=True)
        (tmp_path / split / "mask").mkdir(parents=True)
    for name in images:
        (tmp_path / "val" / "img" / name).write_bytes(b"")
    lines = "\n".join(f"  {key}: {value}" for key, value in quant.items())
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        f"extends: base\ndata_root: {tmp_path}\nnanou: {{}}\nquantization:\n{lines}\n"
        if quant
        else f"extends: base\ndata_root: {tmp_path}\nnanou: {{}}\n"
    )
    return resolve(path=preset)


def test_the_val_split_calibrates_by_default(tmp_path):
    files, source = calibration_images(_config(tmp_path))

    assert source == "val"
    assert [f.rsplit("/", 1)[-1] for f in files] == ["a.png", "b.png"]


def test_the_train_split_is_used_when_the_config_asks_for_it(tmp_path):
    config = _config(tmp_path, calibration_split="train")
    (tmp_path / "train" / "img" / "c.png").write_bytes(b"")

    files, source = calibration_images(config)

    assert source == "train"
    assert [f.rsplit("/", 1)[-1] for f in files] == ["c.png"]


def test_asking_for_noise_skips_the_images_that_are_there(tmp_path):
    files, source = calibration_images(_config(tmp_path), calibration="noise")

    assert (files, source) == ([], "noise")


def test_the_config_can_ask_for_noise_too(tmp_path):
    files, source = calibration_images(_config(tmp_path, calibration_source="noise"))

    assert (files, source) == ([], "noise")


def test_an_export_asking_for_images_overrides_a_noise_config(tmp_path):
    config = _config(tmp_path, calibration_source="noise")

    files, source = calibration_images(config, calibration="images")

    assert source == "val"
    assert len(files) == 2


def test_a_split_with_no_images_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="No calibration images"):
        calibration_images(_config(tmp_path, images=()))


def test_a_split_with_no_images_passes_when_noise_is_allowed(tmp_path):
    config = _config(tmp_path, images=(), allow_noise_calibration="true")

    assert calibration_images(config) == ([], "noise")


def test_no_source_asked_for_falls_back_to_the_config(tmp_path):
    """None and "" both mean "whatever the config says", so callers can pass a
    variable straight through without a branch."""
    config = _config(tmp_path, calibration_source="noise")

    assert calibration_images(config, calibration=None) == ([], "noise")
    assert calibration_images(config, calibration="") == ([], "noise")


@pytest.mark.parametrize("source", ["val", "train", "random"])
def test_an_unknown_calibration_source_is_refused(tmp_path, source):
    with pytest.raises(ValueError, match="images.*noise"):
        calibration_images(_config(tmp_path), calibration=source)
