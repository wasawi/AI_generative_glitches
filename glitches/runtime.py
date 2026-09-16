"""Per-call glitch wrapper for Krea2's DIFFUSION_MODEL wrapper slot (torch only, duck-typed).

ComfyUI 0.35.1 calls DIFFUSION_MODEL wrappers from SingleStreamDiT.forward as
``wrapper(executor, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)``
with ``executor.class_obj`` set to the SingleStreamDiT instance.
"""

from __future__ import annotations

import math

import torch

from .architectures import Krea2Architecture
from .effects import apply_glitch
from .recipe import GlitchRecipe
from .steps import step_from_sigmas

MISSING_SIGMAS = (
    "Glitch Model (Krea2): sampler did not provide sample_sigmas; "
    "use KSampler, KSamplerAdvanced or SamplerCustom"
)


def image_token_grid(x: torch.Tensor, patch: int, architecture=Krea2Architecture) -> tuple[int, int]:
    """Kept for callers that pass a patch size directly; the adapter owns the rule."""
    return architecture.image_token_grid(_PatchOnly(patch), x)


class _PatchOnly:
    """Minimal stand-in exposing just the attribute the grid rule reads."""

    def __init__(self, patch):
        self.patch = patch


def image_token_count(x: torch.Tensor, patch: int) -> int:
    h, w = image_token_grid(x, patch)
    return h * w


def _make_hook(recipe: GlitchRecipe, block: int, family: str, step: int, txt: int, n_img: int, shaping: dict):
    def hook(module, args, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[1] < txt + n_img:
            got = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
            raise RuntimeError(
                f"Glitch Model (Krea2): unexpected {family} output at block {block}: "
                f"expected [B, >= {txt + n_img}, D], got {got}"
            )
        return apply_glitch(output, recipe, block, family, step, txt, n_img, **shaping)

    return hook


class GlitchWrapper:
    def __init__(self, recipe: GlitchRecipe, shape=None, architecture=Krea2Architecture):
        self.recipe = recipe
        self.shape = shape
        self.architecture = architecture

    def __call__(self, executor, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs):
        if "sigmas" not in transformer_options or "sample_sigmas" not in transformer_options:
            raise RuntimeError(MISSING_SIGMAS)
        step = step_from_sigmas(transformer_options["sigmas"], transformer_options["sample_sigmas"])
        if not self.recipe.step_active(step):
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

        dit = executor.class_obj
        architecture = self.architecture
        grid = architecture.image_token_grid(dit, x)
        n_img = grid[0] * grid[1]
        txt = architecture.text_token_count(context)
        shaping = {}
        if self.shape is not None:
            shaping = {
                "shape": self.shape,
                "grid": grid,
                "n_steps": torch.as_tensor(transformer_options["sample_sigmas"]).numel() - 1,
                "n_blocks": architecture.block_count(dit),
            }
        handles = []
        try:
            for block in self.recipe.blocks:
                for family in self.recipe.families:
                    hook = _make_hook(self.recipe, block, family, step, txt, n_img, shaping)
                    handles.append(architecture.site_module(dit, block, family).register_forward_hook(hook))
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
