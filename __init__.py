"""Glitches: reproducible activation glitches for Krea2 models in ComfyUI."""

from .glitches.node import GlitchModelKrea2, GlitchShapeKrea2


class _LesionModelKrea2(GlitchModelKrea2):
    """Old name of Glitch Model (Krea2). Kept so workflows and PNGs saved before the
    rename still load; hidden from the node search because DEPRECATED is set."""

    DEPRECATED = True


class _LesionShapeKrea2(GlitchShapeKrea2):
    """Old name of Glitch Shape (Krea2); see _LesionModelKrea2."""

    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "GlitchModelKrea2": GlitchModelKrea2,
    "GlitchShapeKrea2": GlitchShapeKrea2,
    "LesionModelKrea2": _LesionModelKrea2,
    "LesionShapeKrea2": _LesionShapeKrea2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GlitchModelKrea2": "Glitch Model (Krea2)",
    "GlitchShapeKrea2": "Glitch Shape (Krea2)",
    "LesionModelKrea2": "Lesion Model (Krea2) (renamed to Glitch Model)",
    "LesionShapeKrea2": "Lesion Shape (Krea2) (renamed to Glitch Shape)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
