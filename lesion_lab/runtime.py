"""Per-call lesion wrapper for Krea2's DIFFUSION_MODEL wrapper slot (torch only, duck-typed).

ComfyUI 0.35.1 calls DIFFUSION_MODEL wrappers from SingleStreamDiT.forward as
``wrapper(executor, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)``
with ``executor.class_obj`` set to the SingleStreamDiT instance.
"""

from __future__ import annotations

import math

import torch

from .lesions import apply_lesion
from .recipe import LesionRecipe
from .steps import step_from_sigmas

MISSING_SIGMAS = (
    "Lesion Model (Krea2): sampler did not provide sample_sigmas; "
    "use KSampler, KSamplerAdvanced or SamplerCustom"
)


def image_token_grid(x: torch.Tensor, patch: int) -> tuple[int, int]:
    # comfy/sample.py:58-59 unsqueezes a 4-D image latent to 5-D [B, C, 1, H, W] whenever
    # the model's latent_format reports latent_dimensions == 3 (Krea2 uses Wan21, which does),
    # so real Krea2 latents arrive 5-D with T == 1. SingleStreamDiT._forward flattens T into
    # the batch dimension without repeating context, so only T == 1 is coherent.
    if x.ndim == 5:
        if x.shape[2] != 1:
            raise ValueError(
                "Lesion Model (Krea2) supports single-frame image latents [B, C, H, W] or "
                f"[B, C, 1, H, W] only; multi-frame (T > 1) latents are unsupported, got {tuple(x.shape)}"
            )
    elif x.ndim != 4:
        raise ValueError(f"Lesion Model (Krea2) supports image latents [B, C, H, W] only; got {x.ndim}-D input")
    return math.ceil(x.shape[-2] / patch), math.ceil(x.shape[-1] / patch)


def image_token_count(x: torch.Tensor, patch: int) -> int:
    h, w = image_token_grid(x, patch)
    return h * w


def _make_hook(recipe: LesionRecipe, block: int, family: str, step: int, txt: int, n_img: int, shaping: dict):
    def hook(module, args, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[1] < txt + n_img:
            got = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
            raise RuntimeError(
                f"Lesion Model (Krea2): unexpected {family} output at block {block}: "
                f"expected [B, >= {txt + n_img}, D], got {got}"
            )
        return apply_lesion(output, recipe, block, family, step, txt, n_img, **shaping)

    return hook


class LesionWrapper:
    def __init__(self, recipe: LesionRecipe, shape=None):
        self.recipe = recipe
        self.shape = shape

    def __call__(self, executor, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs):
        if "sigmas" not in transformer_options or "sample_sigmas" not in transformer_options:
            raise RuntimeError(MISSING_SIGMAS)
        step = step_from_sigmas(transformer_options["sigmas"], transformer_options["sample_sigmas"])
        if not self.recipe.step_active(step):
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

        dit = executor.class_obj
        grid = image_token_grid(x, dit.patch)
        n_img = grid[0] * grid[1]
        txt = context.shape[1]
        shaping = {}
        if self.shape is not None:
            shaping = {
                "shape": self.shape,
                "grid": grid,
                "n_steps": torch.as_tensor(transformer_options["sample_sigmas"]).numel() - 1,
                "n_blocks": len(dit.blocks),
            }
        handles = []
        try:
            for block in self.recipe.blocks:
                modules = {"attention": dit.blocks[block].attn, "mlp": dit.blocks[block].mlp}
                for family in self.recipe.families:
                    hook = _make_hook(self.recipe, block, family, step, txt, n_img, shaping)
                    handles.append(modules[family].register_forward_hook(hook))
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
