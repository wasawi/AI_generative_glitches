"""Deterministic channel selection and activation transforms (torch only)."""

from __future__ import annotations

import math

import torch

from .recipe import LesionRecipe

FAMILY_IDS = {"attention": 0, "mlp": 1}
PURPOSE_SELECT = 0
PURPOSE_NOISE = 1

_MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def mix_seed(lesion_seed: int, block: int, family: str, step: int, purpose: int) -> int:
    """Stable 63-bit generator seed for one site, step and purpose (independent of PYTHONHASHSEED)."""
    state = 0
    for part in (lesion_seed, block, FAMILY_IDS[family], step, purpose):
        state = _splitmix64(state ^ (int(part) & _MASK64))
    return state & ((1 << 63) - 1)


def channel_count(probability: float, width: int) -> int:
    return min(width, max(1, round(probability * width)))


def select_channels(recipe: LesionRecipe, block: int, family: str, step: int, width: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(mix_seed(recipe.lesion_seed, block, family, step, PURPOSE_SELECT))
    return torch.randperm(width, generator=generator)[: channel_count(recipe.probability, width)]


def apply_lesion(out: torch.Tensor, recipe: LesionRecipe, block: int, family: str, step: int,
                 txt: int, n_img: int) -> torch.Tensor:
    """Return a copy of ``out`` [B, L, D] with the recipe applied to rows ``txt:txt+n_img``."""
    selected = select_channels(recipe, block, family, step, out.shape[-1]).to(out.device)
    rows = slice(txt, txt + n_img)
    region = out[:, rows, :]
    values = region.index_select(-1, selected)
    strength = recipe.strength

    if recipe.mode == "dropout":
        lesioned = values * (1.0 - strength)
    elif recipe.mode == "amplify":
        lesioned = values * (1.0 + strength)
    elif recipe.mode == "sign_flip":
        lesioned = values * (1.0 - 2.0 * strength)
    elif recipe.mode == "noise":
        rms = torch.linalg.vector_norm(region, dim=-1, keepdim=True, dtype=torch.float32) / math.sqrt(region.shape[-1])
        generator = torch.Generator(device=out.device)
        generator.manual_seed(mix_seed(recipe.lesion_seed, block, family, step, PURPOSE_NOISE))
        noise = torch.randn((n_img, selected.numel()), generator=generator, device=out.device, dtype=torch.float32)
        lesioned = values.float() + strength * rms * noise
    else:
        raise ValueError(f"unsupported lesion mode: {recipe.mode!r}")

    result = out.clone()
    result[:, rows, :].index_copy_(2, selected, lesioned.to(out.dtype))
    return result
