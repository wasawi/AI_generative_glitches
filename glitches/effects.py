"""Deterministic channel selection and activation transforms (torch only)."""

from __future__ import annotations

import math

import torch

from .recipe import GlitchRecipe

FAMILY_IDS = {"attention": 0, "mlp": 1}
PURPOSE_SELECT = 0
PURPOSE_NOISE = 1

_MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def mix_seed(glitch_seed: int, block: int, family: str, step: int, purpose: int) -> int:
    """Stable 63-bit generator seed for one site, step and purpose (independent of PYTHONHASHSEED)."""
    state = 0
    for part in (glitch_seed, block, FAMILY_IDS[family], step, purpose):
        state = _splitmix64(state ^ (int(part) & _MASK64))
    return state & ((1 << 63) - 1)


def channel_count(probability: float, width: int) -> int:
    return min(width, max(1, round(probability * width)))


def select_channels(recipe: GlitchRecipe, block: int, family: str, step: int, width: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(mix_seed(recipe.glitch_seed, block, family, step, PURPOSE_SELECT))
    return torch.randperm(width, generator=generator)[: channel_count(recipe.probability, width)]


def apply_glitch(out: torch.Tensor, recipe: GlitchRecipe, block: int, family: str, step: int,
                 txt: int, n_img: int, *, shape=None, grid=None, n_steps=None, n_blocks=None) -> torch.Tensor:
    """Return a copy of ``out`` [B, L, D] with the recipe applied to rows ``txt:txt+n_img``.

    The four-mode dispatch (dropout/amplify/sign_flip/noise) below is intentionally duplicated in
    ``_apply_shaped`` rather than shared, so the unshaped path here stays bit-identical to the base
    spec's implementation regardless of how the shaped path changes.
    """
    if shape is not None:
        return _apply_shaped(out, recipe, block, family, step, txt, n_img, shape, grid, n_steps, n_blocks)
    selected = select_channels(recipe, block, family, step, out.shape[-1]).to(out.device)
    rows = slice(txt, txt + n_img)
    region = out[:, rows, :]
    values = region.index_select(-1, selected)
    strength = recipe.strength

    if recipe.mode == "dropout":
        glitched = values * (1.0 - strength)
    elif recipe.mode == "amplify":
        glitched = values * (1.0 + strength)
    elif recipe.mode == "sign_flip":
        glitched = values * (1.0 - 2.0 * strength)
    elif recipe.mode == "noise":
        rms = torch.linalg.vector_norm(region, dim=-1, keepdim=True, dtype=torch.float32) / math.sqrt(region.shape[-1])
        generator = torch.Generator(device=out.device)
        generator.manual_seed(mix_seed(recipe.glitch_seed, block, family, step, PURPOSE_NOISE))
        noise = torch.randn((n_img, selected.numel()), generator=generator, device=out.device, dtype=torch.float32)
        glitched = values.float() + strength * rms * noise
    else:
        raise ValueError(f"unsupported glitch mode: {recipe.mode!r}")

    result = out.clone()
    result[:, rows, :].index_copy_(2, selected, glitched.to(out.dtype))
    return result


def _apply_shaped(out, recipe, block, family, step, txt, n_img, shape, grid, n_steps, n_blocks):
    h, w = grid
    if h * w != n_img:
        raise ValueError(f"grid {h}x{w} does not match {n_img} image tokens")
    scalar = shape.step_multiplier(step, n_steps) * shape.block_multiplier(block, n_blocks)
    mask = shape.token_mask(step, n_steps, h, w, out.device)
    if scalar == 0.0 and mask is None:
        # Returning `out` by reference (not a clone) is deliberate: the only caller is the
        # forward hook, which passes this return value straight back as the module's output.
        # Nothing below this branch writes to `out` in place, unlike the base path above, which
        # clones because it does.
        return out

    selected = select_channels(recipe, block, family, step, out.shape[-1]).to(out.device)
    rows = slice(txt, txt + n_img)
    region = out[:, rows, :]
    values = region.index_select(-1, selected).float()
    if mask is None:
        multiplier = torch.full((1, n_img, 1), scalar, dtype=torch.float32, device=out.device)
    else:
        multiplier = mask * scalar
    dose = recipe.strength * multiplier

    if recipe.mode == "dropout":
        glitched = values * (1.0 - dose)
    elif recipe.mode == "amplify":
        glitched = values * (1.0 + dose)
    elif recipe.mode == "sign_flip":
        glitched = values * (1.0 - 2.0 * dose)
    elif recipe.mode == "noise":
        rms = torch.linalg.vector_norm(region, dim=-1, keepdim=True, dtype=torch.float32) / math.sqrt(region.shape[-1])
        generator = torch.Generator(device=out.device)
        generator.manual_seed(mix_seed(recipe.glitch_seed, block, family, step, PURPOSE_NOISE))
        noise = shape.draw_noise(selected.numel(), h, w, generator, out.device)
        glitched = values + dose * rms * noise
    else:
        raise ValueError(f"unsupported glitch mode: {recipe.mode!r}")

    result = out.clone()
    result[:, rows, :].index_copy_(2, selected, glitched.to(out.dtype))
    return result
