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


def image_token_count(x: torch.Tensor, patch: int) -> int:
    if x.ndim != 4:
        raise ValueError(f"Lesion Model (Krea2) supports image latents [B, C, H, W] only; got {x.ndim}-D input")
    return math.ceil(x.shape[-2] / patch) * math.ceil(x.shape[-1] / patch)


def _make_hook(recipe: LesionRecipe, block: int, family: str, step: int, txt: int, n_img: int):
    def hook(module, args, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[1] < txt + n_img:
            got = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
            raise RuntimeError(
                f"Lesion Model (Krea2): unexpected {family} output at block {block}: "
                f"expected [B, >= {txt + n_img}, D], got {got}"
            )
        return apply_lesion(output, recipe, block, family, step, txt, n_img)

    return hook


class LesionWrapper:
    def __init__(self, recipe: LesionRecipe):
        self.recipe = recipe

    def __call__(self, executor, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs):
        if "sigmas" not in transformer_options or "sample_sigmas" not in transformer_options:
            raise RuntimeError(MISSING_SIGMAS)
        step = step_from_sigmas(transformer_options["sigmas"], transformer_options["sample_sigmas"])
        if not self.recipe.step_active(step):
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

        dit = executor.class_obj
        n_img = image_token_count(x, dit.patch)
        txt = context.shape[1]
        handles = []
        try:
            for block in self.recipe.blocks:
                modules = {"attention": dit.blocks[block].attn, "mlp": dit.blocks[block].mlp}
                for family in self.recipe.families:
                    hook = _make_hook(self.recipe, block, family, step, txt, n_img)
                    handles.append(modules[family].register_forward_hook(hook))
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
