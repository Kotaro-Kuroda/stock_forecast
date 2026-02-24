import json
from pathlib import Path

def load_config() -> dict:
    """Load configuration from config.json in the project root."""
    # current file is in shared/config_loader.py
    # project root is ../
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.json"
    
    if not config_path.exists():
        return {}
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

import sys

def apply_config(config: dict) -> None:
    """Apply configuration to the current environment."""
    for path in config.get("extra_paths", []):
        if path not in sys.path:
            sys.path.append(path)

