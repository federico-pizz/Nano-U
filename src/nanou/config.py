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

import os
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

    def to_legacy_dict(self) -> dict[str, Any]:
        """Emit the nested dictionary the current `src/` modules still expect.

        A bridge, not a design: it exists so `--preset` works end to end while
        the library beneath still takes a config path and re-reads keys itself.
        It goes away once those modules take a `TrainConfig` directly, and it is
        the single place where the old `results_dir` spelling survives.
        """
        # The legacy shape has one `data.paths` block for the whole file, so
        # the dataset-level keys have to agree across the model blocks. They do
        # in every shipped preset (all four set them at the preset level); this
        # fails loudly rather than silently picking whichever block came first.
        shared_keys = ("data_root", "models_dir", "output_dir", "input_shape", "seed")
        any_model = next(iter(self.models.values()))
        for name, cfg in self.models.items():
            for key in shared_keys:
                if getattr(cfg, key) != getattr(any_model, key):
                    raise ValueError(
                        f"preset '{self.name}': model '{name}' sets a different "
                        f"'{key}' than '{any_model.model_name}'. Dataset-level keys "
                        f"belong at the top of the preset, not inside a model block."
                    )
        splits = any_model.split_paths

        legacy: dict[str, Any] = {
            "data": {
                "input_shape": list(any_model.input_shape),
                "num_classes": self.data.get("num_classes", 1),
                "normalization": self.data.get("normalization", {}),
                "split": self.data.get("split", {}),
                "paths": {
                    "processed": {"root": any_model.data_root, **splits},
                    "models_dir": any_model.models_dir,
                    "results_dir": any_model.output_dir,
                },
            },
            "models": {},
            "training": {"seed": any_model.seed},
            "quantization": dict(self.quantization),
        }

        for name, cfg in self.models.items():
            legacy["models"][name] = {
                "filters": cfg.filters,
                "bottleneck": cfg.bottleneck,
                "decoder_filters": cfg.decoder_filters,
            }
            legacy["training"][name] = {
                "model_name": cfg.model_name,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "optimizer": cfg.optimizer,
                "augment": cfg.augment,
                "augment_params": dict(cfg.augment_params),
                "early_stopping": cfg.early_stopping,
                "patience": cfg.patience,
                "lr_scheduler": cfg.lr_scheduler,
                "lr_plateau_patience": cfg.lr_plateau_patience,
                "lr_plateau_factor": cfg.lr_plateau_factor,
                "min_lr": cfg.min_lr,
                "qat_enabled": cfg.qat_enabled,
                "use_distillation": cfg.use_distillation,
                "teacher_model_name": cfg.teacher_model_name,
                "teacher_weights": cfg.teacher_weights,
                "alpha": cfg.alpha,
                "temperature": cfg.temperature,
                "t_scale": cfg.t_scale,
                "tversky_weight": cfg.tversky_weight,
                "tversky_alpha": cfg.tversky_alpha,
                "tversky_beta": cfg.tversky_beta,
                "track_redundancy": cfg.track_redundancy,
                "redundancy_every_epochs": cfg.redundancy_every_epochs,
                "redundancy_layers": list(cfg.redundancy_layers),
                "tensorboard": cfg.tensorboard,
                "seed": cfg.seed,
                "input_shape": list(cfg.input_shape),
                "filters": cfg.filters,
                "bottleneck": cfg.bottleneck,
                "decoder_filters": cfg.decoder_filters,
                "models_dir": cfg.models_dir,
                "output_dir": cfg.output_dir,
            }

        return legacy


def preset_path(preset: str) -> Path:
    """Map a preset name to its file under `config/presets/`."""
    return get_project_root() / "config" / "presets" / f"{preset}.yaml"


def available_presets() -> list[str]:
    """Names of every preset shipped in `config/presets/`, sorted."""
    directory = get_project_root() / "config" / "presets"
    return sorted(p.stem for p in directory.glob("*.yaml")) if directory.is_dir() else []


def is_preset_file(path) -> bool:
    """True when *path* is a preset (as opposed to a pre-v2 flat config file)."""
    try:
        loaded = _load_yaml(Path(path))
    except (FileNotFoundError, yaml.YAMLError):
        return False
    return isinstance(loaded, dict) and "extends" in loaded


def resolve(
    preset: str | None = None, path=None, overrides: dict[str, Any] | None = None
) -> ResolvedConfig:
    """Merge base, preset and overrides into one `ResolvedConfig`.

    Args:
        preset: preset name, resolved to `config/presets/<name>.yaml`.
        path: an explicit preset file, used instead of *preset*.
        overrides: keys applied last, on top of every model block. These come
            from the command line, so they win over anything on disk.

    Raises:
        ValueError: a key is not part of the schema, or `data_root` is missing.
        FileNotFoundError: the preset or the file it extends does not exist.
    """
    if path is None:
        if preset is None:
            raise ValueError("resolve() needs either a preset name or a path")
        path = preset_path(preset)
        if not path.exists():
            known = ", ".join(available_presets()) or "none"
            raise FileNotFoundError(f"no preset '{preset}' (available: {known})")
    path = Path(path)
    raw = _load_yaml(path)
    name = preset or path.stem
    root = get_project_root()

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

    return ResolvedConfig(
        name=name, path=path, models=models, data=merged_data, quantization=merged_quant
    )


def load_config(config_path: str = "tinyagri") -> dict[str, Any]:
    """Resolve a config and return it in the pre-v2 nested dictionary shape.

    Accepts a preset name (`"tinyagri"`), a path to a preset, or a path to a
    pre-v2 flat config file. Presets go through `resolve()` - so `base.yaml` is
    merged under them and unknown keys still raise - and are then flattened by
    `ResolvedConfig.to_legacy_dict`.

    This is a bridge, not a design: it exists so the modules that still read a
    dictionary keep working while they are moved, and it disappears when they
    take a `TrainConfig` directly.
    """
    root = Path(get_project_root())

    # A bare name (no separator, no suffix) is a preset name.
    if (
        isinstance(config_path, str)
        and not os.path.splitext(config_path)[1]
        and os.sep not in config_path
    ):
        config_path = str(preset_path(config_path))

    target = Path(config_path) if os.path.isabs(config_path) else root / config_path
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {target}")

    if is_preset_file(target):
        return resolve(path=target).to_legacy_dict()

    with open(target) as fh:
        config = yaml.safe_load(fh)

    # Flat pre-v2 file: keep the structure as written, absolutize the paths.
    paths = config.get("data", {}).get("paths", {})
    raw = paths.get("raw_datasets")
    if isinstance(raw, list):
        for entry in raw:
            for key in ("images", "masks"):
                if isinstance(entry, dict) and key in entry:
                    entry[key] = resolve_path(entry[key], root)
    processed = paths.get("processed")
    if isinstance(processed, dict):
        for key, value in processed.items():
            if isinstance(value, str):
                processed[key] = resolve_path(value, root)
            elif isinstance(value, dict):
                for sub, sub_value in value.items():
                    if isinstance(sub_value, str):
                        value[sub] = resolve_path(sub_value, root)
    if "models_dir" in paths:
        paths["models_dir"] = resolve_path(paths["models_dir"], root)

    return config


__all__ = [
    "TrainConfig",
    "ResolvedConfig",
    "resolve",
    "load_config",
    "get_project_root",
    "resolve_path",
    "preset_path",
    "available_presets",
    "is_preset_file",
    "TRAIN_KEYS",
    "replace",
]
