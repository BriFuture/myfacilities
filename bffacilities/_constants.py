from pathlib import Path

_BASE_DIR = Path.home() / ".shells"
_LOG_DIR = _BASE_DIR / "logs"
_CONFIG_DIR = _BASE_DIR / "configs"

_LOG_DIR.mkdir(parents=True, exist_ok=True)
_CONFIG_DIR.mkdir(parents=True, exist_ok=True)