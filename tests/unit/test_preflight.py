"""Unit tests for src/nanou/preflight.py.

The contract these lock is not just "it raises" but "the message names the
command that fixes it" - that discipline is the whole reason the module exists.
"""

import os
from pathlib import Path

import pytest

from nanou.preflight import (
    PreflightError,
    quant_params_are_stale,
    quant_params_path,
    require_dataset,
    require_esp_toolchain,
    require_keras_model,
    require_model,
    require_serial_port,
    require_tflite_model,
)


def make_dataset(root: Path, counts=(2, 2, 2), masks=None):
    """Build a dataset skeleton of empty PNGs and return its root."""
    masks = counts if masks is None else masks
    for split, n_img, n_mask in zip(("train", "val", "test"), counts, masks):
        (root / split / "img").mkdir(parents=True)
        (root / split / "mask").mkdir(parents=True)
        for i in range(n_img):
            (root / split / "img" / f"f{i}.png").touch()
        for i in range(n_mask):
            (root / split / "mask" / f"f{i}.png").touch()
    return root


# -- every error names its fix ---------------------------------------------------

def test_missing_dataset_names_get_data(tmp_path):
    with pytest.raises(PreflightError, match="scripts/get_data.py --preset tinyagri"):
        require_dataset(tmp_path / "absent", preset="tinyagri")


def test_missing_model_names_train(tmp_path):
    with pytest.raises(PreflightError, match="scripts/train.py --preset tinyagri"):
        require_keras_model(tmp_path, "nanou", preset="tinyagri")


def test_missing_tflite_names_export(tmp_path):
    with pytest.raises(PreflightError, match="scripts/export.py --preset tinyagri"):
        require_tflite_model(tmp_path, "nanou", preset="tinyagri")


def test_missing_any_model_names_train(tmp_path):
    with pytest.raises(PreflightError, match="scripts/train.py"):
        require_model(tmp_path, "nanou")


def test_unsourced_esp_env_names_the_script(monkeypatch):
    monkeypatch.delenv("LIBCLANG_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/cargo")
    with pytest.raises(PreflightError, match=r"source ~/export-esp\.sh"):
        require_esp_toolchain()


def test_missing_cargo_names_rustup(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(PreflightError, match="rustup"):
        require_esp_toolchain()


def test_absent_serial_port_names_the_listing_command(tmp_path):
    with pytest.raises(PreflightError, match=r"ls /dev/tty"):
        require_serial_port(str(tmp_path / "ttyNOPE"))


# -- dataset layout --------------------------------------------------------------

def test_a_complete_dataset_passes(tmp_path):
    require_dataset(make_dataset(tmp_path / "ds"))


def test_unpaired_images_and_masks_raise(tmp_path):
    make_dataset(tmp_path / "ds", counts=(3, 2, 2), masks=(2, 2, 2))
    with pytest.raises(PreflightError, match="3 images but 2 masks"):
        require_dataset(tmp_path / "ds")


def test_an_empty_split_raises(tmp_path):
    make_dataset(tmp_path / "ds", counts=(0, 2, 2), masks=(0, 2, 2))
    with pytest.raises(PreflightError, match="No .png files"):
        require_dataset(tmp_path / "ds")


def test_a_dataset_with_no_val_split_is_allowed(tmp_path):
    """Training runs without validation data; a half-present split does not."""
    root = make_dataset(tmp_path / "ds")
    for sub in ("img", "mask"):
        for f in (root / "val" / sub).iterdir():
            f.unlink()
        (root / "val" / sub).rmdir()
    (root / "val").rmdir()
    require_dataset(root)


def test_a_half_present_val_split_raises(tmp_path):
    root = make_dataset(tmp_path / "ds")
    for f in (root / "val" / "mask").iterdir():
        f.unlink()
    (root / "val" / "mask").rmdir()
    with pytest.raises(PreflightError, match="missing"):
        require_dataset(root)


# -- model resolution ------------------------------------------------------------

def test_require_model_prefers_the_int8_file(tmp_path):
    (tmp_path / "nanou.h5").touch()
    (tmp_path / "nanou.tflite").touch()
    assert require_model(tmp_path, "nanou").suffix == ".tflite"


def test_require_keras_model_accepts_either_keras_format(tmp_path):
    (tmp_path / "nanou.keras").touch()
    assert require_keras_model(tmp_path, "nanou").suffix == ".keras"


# -- quant params ----------------------------------------------------------------

def test_absent_quant_params_are_stale(tmp_path):
    assert quant_params_are_stale(tmp_path, "nanou")


def test_quant_params_older_than_the_tflite_are_stale(tmp_path):
    quant_params_path(tmp_path, "nanou").write_text("{}")
    tflite = tmp_path / "nanou.tflite"
    tflite.touch()
    os.utime(quant_params_path(tmp_path, "nanou"), (1, 1))
    assert quant_params_are_stale(tmp_path, "nanou")


def test_quant_params_newer_than_the_tflite_are_current(tmp_path):
    (tmp_path / "nanou.tflite").touch()
    params = quant_params_path(tmp_path, "nanou")
    params.write_text("{}")
    os.utime(params, None)
    assert not quant_params_are_stale(tmp_path, "nanou")
