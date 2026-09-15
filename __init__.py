"""ComfyUI-LesionLab: reproducible activation lesions for Krea2 models."""

from .lesion_lab.node import LesionModelKrea2, LesionShapeKrea2

NODE_CLASS_MAPPINGS = {"LesionModelKrea2": LesionModelKrea2, "LesionShapeKrea2": LesionShapeKrea2}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LesionModelKrea2": "Lesion Model (Krea2)",
    "LesionShapeKrea2": "Lesion Shape (Krea2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
