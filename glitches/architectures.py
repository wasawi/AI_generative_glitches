"""Where the glitch sites live in a diffusion model (torch only, no ComfyUI import).

Everything architecture-specific sits behind one small adapter: which submodules are glitch
sites, how many blocks there are, and how to find the generated-image tokens in an activation.
Krea2 is the only architecture supported today; adding another (Flux's double/single blocks,
for example) means adding one adapter and registering its model class, not changing the node,
the recipe, the shape or the maths.
"""

from __future__ import annotations

import math

import torch

FAMILIES = ("attention", "mlp")


class Krea2Architecture:
    """Krea2 (comfy.ldm.krea2.model.SingleStreamDiT).

    A stack of ``blocks``, each with ``attn`` and ``mlp``, run over one sequence of
    ``[text | generated image | reference images]`` tokens.
    """

    name = "Krea2"
    model_class_path = ("comfy.ldm.krea2.model", "SingleStreamDiT")

    @staticmethod
    def block_count(diffusion_model) -> int:
        return len(diffusion_model.blocks)

    @staticmethod
    def site_module(diffusion_model, block: int, family: str):
        return diffusion_model.blocks[block].attn if family == "attention" else diffusion_model.blocks[block].mlp

    @staticmethod
    def image_token_grid(diffusion_model, x: torch.Tensor) -> tuple[int, int]:
        """Token-grid height and width of the generated image, from the latent handed to the model.

        comfy/sample.py unsqueezes a 4-D image latent to 5-D [B, C, 1, H, W] whenever the model's
        latent_format reports latent_dimensions == 3 (Krea2 uses Wan21, which does), so real Krea2
        latents arrive 5-D with T == 1. SingleStreamDiT._forward flattens T into the batch dimension
        without repeating context, so only T == 1 is coherent.
        """
        if x.ndim == 5:
            if x.shape[2] != 1:
                raise ValueError(
                    "Glitch Model (Krea2) supports single-frame image latents [B, C, H, W] or "
                    f"[B, C, 1, H, W] only; multi-frame (T > 1) latents are unsupported, got {tuple(x.shape)}"
                )
        elif x.ndim != 4:
            raise ValueError(
                f"Glitch Model (Krea2) supports image latents [B, C, H, W] only; got {x.ndim}-D input"
            )
        patch = diffusion_model.patch
        return math.ceil(x.shape[-2] / patch), math.ceil(x.shape[-1] / patch)

    @staticmethod
    def text_token_count(context: torch.Tensor) -> int:
        """Rows before the generated-image tokens: Krea2 concatenates (text, image, references)."""
        return context.shape[1]


ARCHITECTURES = (Krea2Architecture,)


def architecture_for(diffusion_model):
    """The adapter for this model, or None when no supported architecture matches."""
    for architecture in ARCHITECTURES:
        module_name, class_name = architecture.model_class_path
        module = __import__(module_name, fromlist=[class_name])
        if isinstance(diffusion_model, getattr(module, class_name)):
            return architecture
    return None


def supported_names() -> str:
    return ", ".join(architecture.name for architecture in ARCHITECTURES)
