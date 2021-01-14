"""some basic configurations
"""
from pathlib import Path
import os
import os.path as osp

_BASE_DIR = Path.home() / ".shells" # Basic Path
_CONFIG_DIR = _BASE_DIR / "configs"  # Default Config Path

_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

BFF_ROOT_PATH = Path(osp.abspath(osp.dirname(__file__)))
BFF_OTHER_PATH = BFF_ROOT_PATH / "other"
BFF_TEMPLATE_PATH = BFF_ROOT_PATH / "template"