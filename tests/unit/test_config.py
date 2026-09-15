"""Unit tests for src/nanou/config.py - the schema resolver.

No TensorFlow here: the resolver is plain YAML and dataclasses, and keeping the
import list short is what makes these runnable in a second.
"""

import dataclasses
from dataclasses import fields

import pytest
import yaml

from nanou.config import (
    DATASET_KEYS,
    TRAIN_KEYS,
    TrainConfig,
    available_presets,
    get_project_root,
    resolve,
    resolve_path,
)

ROOT = get_project_root()
BASE_YAML = ROOT / "config" / "base.yaml"
SHIPPED = ["tinyagri", "tinyagri-paper", "botanicgarden", "botanicgarden-paper"]


# -- base.yaml and the dataclass must agree ------------------------------------


def test_base_yaml_has_no_key_outside_the_schema():
    training = yaml.safe_load(BASE_YAML.read_text())["training"]
    assert set(training) <= TRAIN_KEYS


def test_base_yaml_covers_every_optional_field():
    """Every field with a default is spelled out in base.yaml.

    The two required fields (model_name, data_root) are the exceptions: they
    come from the preset.
    """
    training = yaml.safe_load(BASE_YAML.read_text())["training"]
    optional = {f.name for f in fields(TrainConfig)} - {"model_name", "data_root"}
    assert optional == set(training)


def test_base_yaml_values_match_the_dataclass_defaults():
    """A default written twice is a default that can drift. This is the guard."""
    training = yaml.safe_load(BASE_YAML.read_text())["training"]
    reference = TrainConfig(model_name="probe", data_root="data/probe")
    for key, value in training.items():
        expected = getattr(reference, key)
        if isinstance(expected, tuple):
            expected = list(expected)
        assert value == expected, f"base.yaml {key}={value!r}, dataclass has {expected!r}"


# -- the shipped presets --------------------------------------------------------


def test_the_four_presets_are_present():
    assert available_presets() == sorted(SHIPPED)


@pytest.mark.parametrize("name", SHIPPED)
def test_every_preset_resolves_both_models(name):
    resolved = resolve(name)
    assert set(resolved.models) == {"nanou", "bu_net"}
    for model_name, cfg in resolved.models.items():
        assert cfg.model_name == model_name
        assert cfg.data_root
        assert cfg.models_dir and cfg.output_dir


@pytest.mark.parametrize("name", SHIPPED)
def test_split_paths_are_derived_from_data_root(name):
    cfg = resolve(name).for_model("nanou")
    for split, pair in cfg.split_paths.items():
        assert pair["img"] == f"{cfg.data_root}/{split}/img"
        assert pair["mask"] == f"{cfg.data_root}/{split}/mask"


def test_only_the_student_distils():
    resolved = resolve("tinyagri")
    assert resolved.for_model("nanou").use_distillation is True
    assert resolved.for_model("bu_net").use_distillation is False


def test_preset_overrides_base():
    # base.yaml says epochs 100 and optimizer adam; every preset says otherwise.
    cfg = resolve("tinyagri").for_model("nanou")
    assert cfg.epochs == 200
    assert cfg.optimizer == "adamw"


def test_model_block_overrides_preset_level():
    resolved = resolve("tinyagri")
    assert resolved.for_model("nanou").batch_size == 16
    assert resolved.for_model("bu_net").batch_size == 32


def test_overrides_win_over_the_file():
    cfg = resolve("tinyagri", overrides={"epochs": 3}).for_model("nanou")
    assert cfg.epochs == 3


# -- rejection ------------------------------------------------------------------


def test_unknown_key_in_a_model_block_raises(tmp_path):
    preset = tmp_path / "typo.yaml"
    preset.write_text("extends: base\ndata_root: data/x\nnanou:\n  epocs: 5\n")
    with pytest.raises(ValueError, match="epocs"):
        resolve(path=preset)


def test_the_error_suggests_the_key_that_was_meant(tmp_path):
    preset = tmp_path / "typo.yaml"
    preset.write_text("extends: base\ndata_root: data/x\nnanou:\n  epocs: 5\n")
    with pytest.raises(ValueError, match="epochs"):
        resolve(path=preset)


def test_unknown_scalar_at_the_top_level_raises(tmp_path):
    preset = tmp_path / "typo.yaml"
    preset.write_text("extends: base\ndata_root: data/x\nlearnign_rate: 0.1\nnanou: {}\n")
    with pytest.raises(ValueError, match="learnign_rate"):
        resolve(path=preset)


def test_missing_data_root_raises(tmp_path):
    preset = tmp_path / "rootless.yaml"
    preset.write_text("extends: base\nnanou:\n  epochs: 5\n")
    with pytest.raises(ValueError, match="data_root"):
        resolve(path=preset)


def test_a_preset_with_no_model_block_raises(tmp_path):
    preset = tmp_path / "empty.yaml"
    preset.write_text("extends: base\ndata_root: data/x\n")
    with pytest.raises(ValueError, match="no model blocks"):
        resolve(path=preset)


def test_unknown_preset_name_lists_the_known_ones():
    with pytest.raises(FileNotFoundError, match="tinyagri"):
        resolve("no-such-preset")


def test_asking_for_a_model_the_preset_lacks_raises():
    with pytest.raises(KeyError, match="mystery_net"):
        resolve("tinyagri").for_model("mystery_net")


# -- frozen ---------------------------------------------------------------------


def test_trainconfig_is_frozen():
    cfg = resolve("tinyagri").for_model("nanou")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.epochs = 1


# -- paths ----------------------------------------------------------------------


def test_relative_paths_resolve_against_the_project_root():
    assert resolve_path("models/TinyAgri") == str(ROOT / "models" / "TinyAgri")


def test_absolute_paths_are_left_alone(tmp_path):
    # tmp_path is outside the repo, so this warns; the point here is that the
    # path comes back unchanged rather than rebased onto the project root.
    with pytest.warns(UserWarning):
        assert resolve_path(str(tmp_path)) == str(tmp_path.resolve())


def test_a_dataset_outside_the_repo_warns_but_is_allowed(tmp_path):
    """A multi-gigabyte image set normally lives on another disk."""
    outside = tmp_path / "elsewhere" / "data"
    with pytest.warns(UserWarning, match="outside the project root"):
        assert resolve_path(str(outside)) == str(outside)


def test_non_strings_pass_through():
    assert resolve_path(None) is None
    assert resolve_path(7) == 7


def test_a_relative_preset_path_is_taken_from_the_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve(path="config/presets/tinyagri.yaml").name == "tinyagri"


# -- overrides ------------------------------------------------------------------


def test_an_unknown_override_is_rejected_like_a_preset_typo():
    with pytest.raises(ValueError, match="Did you mean 'epochs'"):
        resolve("tinyagri", overrides={"epocs": 3})


# -- the dataset, one per preset ------------------------------------------------


@pytest.mark.parametrize("name", SHIPPED)
def test_the_preset_answers_for_its_dataset(name):
    resolved = resolve(name)
    for cfg in resolved.models.values():
        for key in DATASET_KEYS:
            assert getattr(resolved, key) == getattr(cfg, key), key
    assert resolved.split_paths == resolved.for_model("nanou").split_paths
    assert len(resolved.normalization["mean"]) == len(resolved.normalization["std"]) == 3


def test_a_model_block_that_moves_a_dataset_key_is_rejected(tmp_path):
    preset = tmp_path / "split-root.yaml"
    preset.write_text(
        "extends: base\ndata_root: data/x\nnanou:\n  epochs: 5\nbu_net:\n  data_root: data/y\n"
    )
    with pytest.raises(ValueError, match="different 'data_root'"):
        resolve(path=preset)
