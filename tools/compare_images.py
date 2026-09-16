"""Compare two images pixel by pixel.

Usage: python tools/compare_images.py A.png B.png   (use the Python that runs your ComfyUI)
Exit code: 0 identical, 1 different, 2 usage error.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image


def compare(a, b) -> dict:
    with Image.open(a) as image_a, Image.open(b) as image_b:
        pixels_a = np.asarray(image_a.convert("RGB"), dtype=np.int16)
        pixels_b = np.asarray(image_b.convert("RGB"), dtype=np.int16)
    if pixels_a.shape != pixels_b.shape:
        size_a = f"{pixels_a.shape[1]}x{pixels_a.shape[0]}"
        size_b = f"{pixels_b.shape[1]}x{pixels_b.shape[0]}"
        return {"identical": False, "reason": f"size differs: {size_a} vs {size_b}"}
    diff = np.abs(pixels_a - pixels_b)
    return {
        "identical": bool(diff.max() == 0),
        "max_abs": int(diff.max()),
        "mean_abs": float(diff.mean()),
        "changed_fraction": float((diff.max(axis=-1) > 0).mean()),
    }


def format_report(result: dict) -> str:
    if "reason" in result:
        return f"different: {result['reason']}"
    if result["identical"]:
        return "identical"
    return (
        f"different: max_abs={result['max_abs']} mean_abs={result['mean_abs']:.4f} "
        f"changed_pixels={result['changed_fraction']:.2%}"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    result = compare(argv[0], argv[1])
    print(format_report(result))
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
