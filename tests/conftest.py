import os
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", "/Users/wswi/ComfyUI-Installs/ComfyUI/ComfyUI"))

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture(scope="session")
def comfy_root():
    """Put the installed ComfyUI on sys.path (read-only use) or skip the test."""
    if not (COMFYUI_ROOT / "comfy" / "model_patcher.py").is_file():
        pytest.skip(f"ComfyUI not found at {COMFYUI_ROOT}; set COMFYUI_ROOT")
    if str(COMFYUI_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFYUI_ROOT))
    return COMFYUI_ROOT
