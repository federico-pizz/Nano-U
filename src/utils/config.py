import yaml
import os
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML configuration file.
    Resolves the path relative to the project root.
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
        
    return config
