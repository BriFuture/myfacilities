"""some basic configurations
"""
from pathlib import Path

_BASE_DIR = Path.home() / ".shells" # Basic Path
_CONFIG_DIR = _BASE_DIR / "configs"  # Default Config Path

_CONFIG_DIR.mkdir(parents=True, exist_ok=True)