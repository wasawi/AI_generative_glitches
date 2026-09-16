"""ComfyUI node: Glitch Model (Krea2). The only module in this package that imports ComfyUI."""

from __future__ import annotations

import comfy.patcher_extension

from .architectures import architecture_for, supported_names
from .recipe import MODES, SEED_MAX, STRENGTH_WIDGET_LIMIT, TARGETS, GlitchRecipe, clamp_to_model
from .runtime import GlitchWrapper
from .shape import DISTRIBUTIONS, NOISE_SCALE_MAX, GlitchShape

WRAPPER_KEY = "glitches"


class GlitchModelKrea2:
    DESCRIPTION = (
        "Applies reproducible activation glitches to the image tokens of a Krea2 model during sampling. "
        "Never modifies weights or checkpoint files."
    )
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "recipe")
    FUNCTION = "apply"
    CATEGORY = "experimental/glitches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "mode": (list(MODES), {"default": "noise"}),
                "strength": ("FLOAT", {"default": 0.15, "min": -STRENGTH_WIDGET_LIMIT, "max": STRENGTH_WIDGET_LIMIT, "step": 0.01}),
                "probability": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.0001, "round": 0.0001}),
                "target": (list(TARGETS), {"default": "both"}),
                "block_start": ("INT", {"default": 0, "min": 0, "max": 27}),
                "block_end": ("INT", {"default": 27, "min": 0, "max": 27}),
                "step_start": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "step_end": ("INT", {"default": 999, "min": 0, "max": 10000}),
                "glitch_seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
            },
            "optional": {
                "shape": ("GLITCH_SHAPE",),
            },
        }

    def apply(self, model, enabled, mode, strength, probability, target,
              block_start, block_end, step_start, step_end, glitch_seed, shape=None):
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
        architecture = architecture_for(diffusion_model)
        if architecture is None:
            raise ValueError(
                f"Glitch Model (Krea2) requires a {supported_names()} model; "
                f"got {type(diffusion_model).__name__}"
            )
        recipe = GlitchRecipe.build(
            enabled=enabled, mode=mode, strength=strength, probability=probability, target=target,
            block_start=block_start, block_end=block_end, step_start=step_start, step_end=step_end,
            glitch_seed=glitch_seed,
        )
        recipe = clamp_to_model(recipe, n_blocks=architecture.block_count(diffusion_model))
        if shape is not None and not isinstance(shape, GlitchShape):
            raise TypeError("shape must come from a Glitch Shape (Krea2) node")
        clone = model.clone()
        if recipe.noop_reason() is None:
            clone.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, GlitchWrapper(recipe, shape, architecture)
            )
        text = recipe.describe()
        if shape is not None:
            text += " | " + shape.describe(recipe.mode)
        return (clone, text)


class GlitchShapeKrea2:
    DESCRIPTION = (
        "Shapes a Glitch Model (Krea2): noise distribution and blob size (noise mode), plus optional strength "
        "curves over sampling steps and blocks and a spatial mask (all modes)."
    )
    RETURN_TYPES = ("GLITCH_SHAPE",)
    RETURN_NAMES = ("shape",)
    FUNCTION = "build"
    CATEGORY = "experimental/glitches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "distribution": (list(DISTRIBUTIONS), {"default": "gaussian"}),
                "spike_density": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001}),
                "noise_scale": ("INT", {"default": 1, "min": 1, "max": NOISE_SCALE_MAX}),
            },
            "optional": {
                "step_curve": ("FLOAT", {"forceInput": True}),
                "block_curve": ("FLOAT", {"forceInput": True}),
                "spatial_mask": ("MASK",),
            },
        }

    def build(self, distribution, spike_density, noise_scale, step_curve=None, block_curve=None, spatial_mask=None):
        return (GlitchShape.build(distribution, spike_density, noise_scale, step_curve, block_curve, spatial_mask),)
