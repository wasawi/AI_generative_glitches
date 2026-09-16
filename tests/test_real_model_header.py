import os
from pathlib import Path

import pytest

GGUF_PATH = Path(os.environ.get("KREA2_GGUF", "/Volumes/DATA/ComfyUI/models/unet/KREA/museByStableYogi_v25GGUF.gguf"))


def test_real_krea2_gguf_exposes_the_glitch_sites():
    if not GGUF_PATH.is_file():
        pytest.skip(f"{GGUF_PATH} not found; set KREA2_GGUF")
    gguf = pytest.importorskip("gguf")
    names = {tensor.name for tensor in gguf.GGUFReader(str(GGUF_PATH)).tensors}
    assert {int(name.split(".")[1]) for name in names if name.startswith("blocks.")} == set(range(28))
    for block in range(28):
        for family in ("attn", "mlp"):
            assert any(name.startswith(f"blocks.{block}.{family}.") for name in names), f"blocks.{block}.{family}"
