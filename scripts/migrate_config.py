"""
Migrate old flat config to new experiments.yaml format.

Usage:
    python scripts/migrate_config.py old_config.yaml new_config.yaml
"""

import sys
import yaml
from pathlib import Path


def migrate_old_config(old_path: str, new_path: str) -> None:
    """Convert old config (flat model/epochs/batch_size/...) to new experiments format."""
    with open(old_path) as f:
        old = yaml.safe_load(f) or {}

    new = {
        "experiments": {
            "migrated": {
                "model_name": old.get("model", "nano_u"),
                "epochs": old.get("epochs", 50),
                "batch_size": old.get("batch_size", 16),
                "learning_rate": old.get("learning_rate", 0.001),
                "use_distillation": old.get("distillation", False),
                "teacher_weights": old.get("teacher_weights"),
                "alpha": old.get("alpha", 0.3),
                "temperature": old.get("temperature", 4.0),
                "use_nas": old.get("nas", False),
                "layers_to_monitor": old.get("layers_to_monitor", ["conv2d", "conv2d_1"]),
                "nas_frequency": old.get("nas_frequency", 10),
            }
        }
    }

    Path(new_path).parent.mkdir(parents=True, exist_ok=True)
    with open(new_path, "w") as f:
        yaml.dump(new, f, default_flow_style=False, sort_keys=False)

    print(f"Migrated config written to: {new_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/migrate_config.py <old_config.yaml> <new_config.yaml>")
        sys.exit(1)
    migrate_old_config(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
