import os
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _find_comfyui():
    """Locate a ComfyUI checkout: $COMFYUI_ROOT, an ancestor of this package (when it is
    installed inside custom_nodes), or a ComfyUI directory under the home folder."""
    env = os.environ.get("COMFYUI_ROOT")
    if env:
        return Path(env)
    candidates = list(PACKAGE_ROOT.parents)
    for pattern in ("ComfyUI", "ComfyUI/ComfyUI", "ComfyUI*/ComfyUI", "ComfyUI*/*/ComfyUI"):
        candidates.extend(sorted(Path.home().glob(pattern)))
    for candidate in candidates:
        if (candidate / "comfy" / "model_patcher.py").is_file():
            return candidate
    return None


COMFYUI_ROOT = _find_comfyui()


@pytest.fixture(scope="session")
def comfy_root():
    """Put the installed ComfyUI on sys.path (read-only use) or skip the test."""
    if COMFYUI_ROOT is None or not (COMFYUI_ROOT / "comfy" / "model_patcher.py").is_file():
        pytest.skip("ComfyUI not found; set COMFYUI_ROOT to a ComfyUI checkout")
    if str(COMFYUI_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFYUI_ROOT))
    return COMFYUI_ROOT
