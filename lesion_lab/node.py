"""ComfyUI node: Lesion Model (Krea2). The only module in this package that imports ComfyUI."""

from __future__ import annotations

import comfy.patcher_extension
from comfy.ldm.krea2.model import SingleStreamDiT

from .recipe import MODES, SEED_MAX, STRENGTH_MAX, TARGETS, LesionRecipe, validate_against_model
from .runtime import LesionWrapper

WRAPPER_KEY = "lesion_lab"


class LesionModelKrea2:
    DESCRIPTION = (
        "Applies reproducible activation lesions to the image tokens of a Krea2 model during sampling. "
        "Never modifies weights or checkpoint files."
    )
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "recipe")
    FUNCTION = "apply"
    CATEGORY = "experimental/lesion-lab"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "mode": (list(MODES), {"default": "noise"}),
                "strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": STRENGTH_MAX, "step": 0.01}),
                "probability": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target": (list(TARGETS), {"default": "both"}),
                "block_start": ("INT", {"default": 0, "min": 0, "max": 999}),
                "block_end": ("INT", {"default": 27, "min": 0, "max": 999}),
                "step_start": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "step_end": ("INT", {"default": 999, "min": 0, "max": 10000}),
                "lesion_seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
            }
        }

    def apply(self, model, enabled, mode, strength, probability, target,
              block_start, block_end, step_start, step_end, lesion_seed):
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
        if not isinstance(diffusion_model, SingleStreamDiT):
            raise ValueError(
                f"Lesion Model (Krea2) requires a Krea2 model; got {type(diffusion_model).__name__}"
            )
        recipe = LesionRecipe.build(
            enabled=enabled, mode=mode, strength=strength, probability=probability, target=target,
            block_start=block_start, block_end=block_end, step_start=step_start, step_end=step_end,
            lesion_seed=lesion_seed,
        )
        validate_against_model(recipe, n_blocks=len(diffusion_model.blocks))
        clone = model.clone()
        if recipe.noop_reason() is None:
            clone.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, LesionWrapper(recipe)
            )
        return (clone, recipe.describe())
