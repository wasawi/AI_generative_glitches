import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_images import compare, format_report  # noqa: E402


def png(array):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_identical_images():
    image = np.full((4, 4, 3), 100)
    result = compare(png(image), png(image))
    assert result == {"identical": True, "max_abs": 0, "mean_abs": 0.0, "changed_fraction": 0.0}
    assert format_report(result) == "identical"


def test_different_images():
    a = np.full((4, 4, 3), 100)
    b = a.copy()
    b[0, 0, 0] = 110
    result = compare(png(a), png(b))
    assert result["identical"] is False
    assert result["max_abs"] == 10
    assert result["changed_fraction"] == 1 / 16
    assert format_report(result) == "different: max_abs=10 mean_abs=0.2083 changed_pixels=6.25%"


def test_size_mismatch():
    result = compare(png(np.zeros((4, 4, 3))), png(np.zeros((4, 5, 3))))
    assert result == {"identical": False, "reason": "size differs: 4x4 vs 5x4"}
    assert format_report(result) == "different: size differs: 4x4 vs 5x4"
