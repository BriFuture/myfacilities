from pathlib import Path

_BASE_DIR = Path.home() / ".shells" # Basic Path
_LOG_DIR = _BASE_DIR / "logs"   # Log Path
_CONFIG_DIR = _BASE_DIR / "configs"  # Default Config Path

_LOG_DIR.mkdir(parents=True, exist_ok=True)
_CONFIG_DIR.mkdir(parents=True, exist_ok=True)