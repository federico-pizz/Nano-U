"""Configuration schema and resolver.

One typed object, built once, instead of the config dictionary being re-read and
re-defaulted at every call site.

`TrainConfig` carries every training key with its default in one place.
`resolve()` merges `config/base.yaml`, then a preset, then caller overrides, and
**rejects unknown keys** - a typo in a preset is an error, not a silent default.

The layering is:

    base.yaml `training:`  ->  preset top-level keys  ->  preset `<model>:` block
                           ->  overrides passed by the caller

Presets look like this::

    extends: base
    data_root: data/TinyAgri     # the five split paths are derived from it
    models_dir: models/TinyAgri
    output_dir: results/TinyAgri
    seed: 98

    nanou:                       # a model block; its name is the model name
      use_distillation: true
      epochs: 200

    bu_net:
      epochs: 200

Reserved top-level names are `extends`, `data` and `quantization`. Any other
mapping at the top level is a model block; any other scalar is a preset-level
default shared by every model in the file.
"""

import warnings
from dataclasses import dataclass, field, fields, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import yaml

# Top-level preset keys that are not model blocks and not training defaults.
RESERVED_KEYS = ("extends", "data", "quantization")

# Split directory layout derived from `data_root`. Changing this changes what a
# preset means, so it lives here rather than being spelled out in five places.
SPLITS = ("train", "val", "test")

# Keys that describe the dataset rather than one model's run. They must agree
# across a preset's model blocks, which is what lets the preset answer for them.
DATASET_KEYS = ("data_root", "models_dir", "output_dir", "input_shape", "seed")


def get_project_root() -> Path:
    """Return the repository root (the directory holding `config/` and `src/`)."""
    # src/nanou/config.py -> nanou -> src -> root
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(value, project_root: Path | None = None):
    """Make a path absolute against the project root, leaving non-strings alone.

    A path above the repository is allowed: a multi-gigabyte image set normally
    lives on another disk. It only warns, because pointing at the wrong place is
    a mistake worth noticing, not one worth refusing.
    """
    if not isinstance(value, str) or not value:
        return value
    root = Path(project_root or get_project_root()).resolve()
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if root not in resolved.parents and resolved != root:
        warnings.warn(
            f"path '{value}' resolves outside the project root ({resolved})", stacklevel=2
        )
    return str(resolved)


@dataclass(frozen=True)
class TrainConfig:
    """Every key one training run reads, with its default.

    Frozen: a run's configuration cannot change halfway through it. Use
    `dataclasses.replace` (or `resolve(overrides=...)`) to derive a variant.
    """

    # Identity. `model_name` names the output files, so it has no default; it is
    # taken from the name of the preset block that defines the model.
    model_name: str
    data_root: str

    # Optimization
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"

    # Augmentation. Empty `augment_params` means make_dataset's own defaults,
    # which are the "full" regime - set them explicitly to pin a regime.
    augment: bool = False
    augment_params: dict[str, Any] = field(default_factory=dict)

    # Stopping and learning-rate schedule
    early_stopping: bool = False
    patience: int = 10
    lr_scheduler: str = ""
    lr_plateau_patience: int = 5
    lr_plateau_factor: float = 0.2
    min_lr: float = 1e-6

    # Quantization-aware training
    qat_enabled: bool = True

    # Knowledge distillation
    use_distillation: bool = False
    teacher_model_name: str = "bu_net"
    teacher_weights: str | None = None
    alpha: float = 0.3
    temperature: float = 4.0
    t_scale: str = "squared"  # "squared" applies the T^2 Hinton factor; "none" drops it

    # Conservative supervised loss: (1-w)*BCE + w*Tversky(alpha_fp, beta_fn)
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3

    # SVD channel-redundancy monitoring
    track_redundancy: bool = False
    redundancy_every_epochs: int = 10
    redundancy_layers: list[str] = field(default_factory=lambda: ["enc1a_dw", "enc1a_pw"])

    tensorboard: bool = False
    seed: int | None = None

    # Architecture. None means "the builder's own default for this model".
    input_shape: tuple[int, int, int] = (60, 80, 3)
    filters: list[int] | None = None
    bottleneck: int | None = None
    decoder_filters: list[int] | None = None

    # Output locations
    models_dir: str | None = None
    output_dir: str | None = None

    @property
    def split_paths(self) -> dict[str, dict[str, str]]:
        """The six dataset directories derived from `data_root`."""
        root = Path(self.data_root)
        return {s: {"img": str(root / s / "img"), "mask": str(root / s / "mask")} for s in SPLITS}


TRAIN_KEYS = frozenset(f.name for f in fields(TrainConfig))


def _reject_unknown(mapping: dict[str, Any], where: str, allowed=TRAIN_KEYS) -> None:
    """Raise on the first key that is not part of the schema."""
    for key in mapping:
        if key in allowed:
            continue
        hint = get_close_matches(key, sorted(allowed), n=1)
        suffix = f" Did you mean '{hint[0]}'?" if hint else ""
        raise ValueError(f"unknown config key '{key}' in {where}.{suffix}")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class ResolvedConfig:
    """A preset, fully merged: one `TrainConfig` per model plus the shared blocks.

    `data` and `quantization` describe the dataset and the export rather than a
    training run, so they keep their nested shape and are not part of
    `TrainConfig`.
    """

    name: str
    path: Path
    models: dict[str, TrainConfig]
    data: dict[str, Any]
    quantization: dict[str, Any]

    def for_model(self, model_name: str) -> TrainConfig:
        """Return the resolved config for one model block."""
        try:
            return self.models[model_name]
        except KeyError:
            known = ", ".join(sorted(self.models)) or "none"
            raise KeyError(
                f"preset '{self.name}' defines no model '{model_name}' (has: {known})"
            ) from None

    # Dataset-level facts. `resolve()` refuses a preset whose model blocks
    # disagree on any of `DATASET_KEYS`, so the first block answers for all.

    @property
    def _first(self) -> TrainConfig:
        return next(iter(self.models.values()))

    @property
    def data_root(self) -> str:
        """The dataset directory, holding `<split>/{img,mask}`."""
        return self._first.data_root

    @property
    def split_paths(self) -> dict[str, dict[str, str]]:
        """The six split directories derived from `data_root`."""
        return self._first.split_paths

    @property
    def models_dir(self) -> str | None:
        """Where the dataset's trained models and exports live."""
        return self._first.models_dir

    @property
    def output_dir(self) -> str | None:
        """Where the dataset's run outputs and reports go."""
        return self._first.output_dir

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """(height, width, channels) every model of the preset takes."""
        return self._first.input_shape

    @property
    def seed(self) -> int | None:
        """The run seed shared by every model of the preset."""
        return self._first.seed

    @property
    def normalization(self) -> dict[str, list[float]]:
        """The `data.normalization` block: `mean` and `std`, one value per channel."""
        return self.data["normalization"]


def preset_path(preset: str) -> Path:
    """Map a preset name to its file under `config/presets/`."""
    return get_project_root() / "config" / "presets" / f"{preset}.yaml"


def available_presets() -> list[str]:
    """Names of every preset shipped in `config/presets/`, sorted."""
    directory = get_project_root() / "config" / "presets"
    return sorted(p.stem for p in directory.glob("*.yaml")) if directory.is_dir() else []


def resolve(
    preset: str | None = None, path=None, overrides: dict[str, Any] | None = None
) -> ResolvedConfig:
    """Merge base, preset and overrides into one `ResolvedConfig`.

    Args:
        preset: preset name, resolved to `config/presets/<name>.yaml`.
        path: an explicit preset file, used instead of *preset*. A relative
            path is taken from the repository root, not the working directory.
        overrides: keys applied last, on top of every model block. These come
            from the command line, so they win over anything on disk.

    Raises:
        ValueError: a key (overrides included) is not part of the schema,
            `data_root` is missing, or two model blocks disagree on one of
            `DATASET_KEYS`.
        FileNotFoundError: the preset or the file it extends does not exist.
    """
    root = get_project_root()
    if path is None:
        if preset is None:
            raise ValueError("resolve() needs either a preset name or a path")
        path = preset_path(preset)
        if not path.exists():
            known = ", ".join(available_presets()) or "none"
            raise FileNotFoundError(f"no preset '{preset}' (available: {known})")
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    raw = _load_yaml(path)
    name = preset or path.stem

    base_name = raw.get("extends", "base")
    base = _load_yaml(root / "config" / f"{base_name}.yaml")

    base_training = dict(base.get("training") or {})
    _reject_unknown(base_training, f"config/{base_name}.yaml training block")

    # Preset top level splits three ways: reserved blocks, shared training
    # defaults, and one block per model.
    shared: dict[str, Any] = {}
    blocks: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if key in RESERVED_KEYS:
            continue
        if key in TRAIN_KEYS:
            shared[key] = value
        elif isinstance(value, dict):
            blocks[key] = value
        else:
            _reject_unknown({key: value}, str(path))

    if not blocks:
        raise ValueError(f"{path} defines no model blocks")
    _reject_unknown(overrides or {}, "the overrides")

    merged_data = {**(base.get("data") or {}), **(raw.get("data") or {})}
    merged_quant = {**(base.get("quantization") or {}), **(raw.get("quantization") or {})}

    models: dict[str, TrainConfig] = {}
    for model_name, block in blocks.items():
        _reject_unknown(block, f"{path} block '{model_name}'")
        merged = {**base_training, **shared, **block, **(overrides or {})}
        # The block name is the model name: it is the key the pipeline looks the
        # model up by, so deriving it here means the two cannot disagree.
        merged["model_name"] = model_name
        if not merged.get("data_root"):
            raise ValueError(f"{path} sets no 'data_root'; it has no default")
        for key in ("data_root", "models_dir", "output_dir", "teacher_weights"):
            if merged.get(key):
                merged[key] = resolve_path(merged[key], root)
        if merged.get("input_shape") is not None:
            merged["input_shape"] = tuple(merged["input_shape"])
        models[model_name] = TrainConfig(**merged)

    # One dataset per preset. A block that moved one of these would train its
    # model on different data, and the preset could no longer answer for it.
    first = next(iter(models.values()))
    for model_name, cfg in models.items():
        for key in DATASET_KEYS:
            if getattr(cfg, key) != getattr(first, key):
                raise ValueError(
                    f"preset '{name}': model '{model_name}' sets a different '{key}' "
                    f"than '{first.model_name}'. Dataset-level keys belong at the top "
                    "of the preset, not inside a model block."
                )

    return ResolvedConfig(
        name=name, path=path, models=models, data=merged_data, quantization=merged_quant
    )


__all__ = [
    "TrainConfig",
    "ResolvedConfig",
    "resolve",
    "get_project_root",
    "resolve_path",
    "preset_path",
    "available_presets",
    "DATASET_KEYS",
    "TRAIN_KEYS",
    "replace",
]
