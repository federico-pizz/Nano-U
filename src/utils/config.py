import os
from pathlib import Path

import yaml

from nanou.config import is_preset_file, preset_path, resolve

def _resolve_path(p, project_root):
    if not isinstance(p, str):
        return p
    path = Path(p)
    resolved = path if path.is_absolute() else (project_root / path)
    try:
        resolved = resolved.resolve()
        project_root_resolved = Path(project_root).resolve()
        if not str(resolved).startswith(str(project_root_resolved)):
            raise ValueError(f"Path '{p}' resolves outside project root")
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path '{p}': {e}")
    return str(resolved)


def load_config(config_path="config/presets/tinyagri.yaml"):
    """Load a config and normalize its paths to absolute strings.

    Accepts three things:

      * a preset name, e.g. "tinyagri";
      * a path to a preset under `config/presets/`;
      * a path to a pre-v2 flat config file.

    A preset is resolved through `nanou.config.resolve` - which merges
    `config/base.yaml` under it and rejects unknown keys - and then flattened
    back into the nested dictionary the rest of `src/` still reads. That
    flattening is a bridge: it disappears once those modules take a
    `TrainConfig` directly.

    For a flat config file the behaviour is unchanged: the structure is kept as
    written and only the path strings under `data.paths` become absolute.
    """
    # Get project root (assuming this file is in src/utils/)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent

    # A bare name (no separator, no suffix) is a preset name.
    if isinstance(config_path, str) and not os.path.splitext(config_path)[1] \
            and os.sep not in config_path:
        config_path = str(preset_path(config_path))

    # Allow absolute paths or paths relative to project root
    if os.path.isabs(config_path):
        target_path = Path(config_path)
    else:
        target_path = project_root / config_path

    if not target_path.exists():
        raise FileNotFoundError(f"Config file not found: {target_path}")

    if is_preset_file(target_path):
        return resolve(path=target_path).to_legacy_dict()

    with open(target_path, 'r') as f:
        config = yaml.safe_load(f)

    # Normalize commonly used paths so callers can assume absolute paths
    data_paths = config.get("data", {}).get("paths", {})
    if data_paths:
        # Raw datasets: list of dicts with images/masks
        raw = data_paths.get("raw_datasets")
        if isinstance(raw, list):
            for ds in raw:
                if isinstance(ds, dict):
                    if "images" in ds:
                        ds["images"] = _resolve_path(ds["images"], project_root)
                    if "masks" in ds:
                        ds["masks"] = _resolve_path(ds["masks"], project_root)
        # Processed paths (train/val/test folders)
        processed = data_paths.get("processed")
        if isinstance(processed, dict):
            # processed may contain nested train/val/test dicts
            for k, v in processed.items():
                if isinstance(v, str):
                    processed[k] = _resolve_path(v, project_root)
                elif isinstance(v, dict):
                    for subk, subv in v.items():
                        if isinstance(subv, str):
                            v[subk] = _resolve_path(subv, project_root)
        # models_dir
        if "models_dir" in data_paths:
            data_paths["models_dir"] = _resolve_path(data_paths["models_dir"], project_root)

    return config
