import yaml
import os
from pathlib import Path

def _resolve_path(p, project_root):
    if not isinstance(p, str):
        return p
    path = Path(p)
    resolved = path if path.is_absolute() else (project_root / path)
    return str(resolved)


def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML configuration file and normalizes common paths to absolute strings
    relative to the repository root (src/.. -> project root).

    Returned config keeps the same structure but replaces path strings under
    `data.paths` and `data.paths.processed` and `data.paths.raw_datasets` with
    absolute paths to simplify callers.
    """
    # Get project root (assuming this file is in src/utils/)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent

    # Allow absolute paths or paths relative to project root
    if os.path.isabs(config_path):
        target_path = Path(config_path)
    else:
        target_path = project_root / config_path

    if not target_path.exists():
        raise FileNotFoundError(f"Config file not found: {target_path}")

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
