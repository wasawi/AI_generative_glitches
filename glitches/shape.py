"""Optional glitch shaping: noise distribution, spatial mask and step/block curves (torch only)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

DISTRIBUTIONS = ("gaussian", "uniform", "laplace", "cauchy", "spikes", "binary")
CAUCHY_CLIP = 20.0
NOISE_SCALE_MAX = 64
_EDGE = 1e-7


def _curve(name, value):
    message = f"{name} must be a finite number or a non-empty list of finite numbers"
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        items = value.detach().flatten().cpu().tolist()
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif hasattr(value, "tolist") and not isinstance(value, str):
        items = value.tolist()
    else:
        raise ValueError(f"{message}; got {type(value).__name__}")
    if not isinstance(items, list):
        items = [items]
    flat = []
    for item in items:
        flat.extend(item if isinstance(item, (list, tuple)) else [item])
    if not flat or any(isinstance(x, (str, bool)) or not isinstance(x, (int, float)) for x in flat):
        raise ValueError(message)
    numbers = tuple(float(x) for x in flat)
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError(message)
    return numbers


def _mask(value):
    message = "spatial_mask must be a finite MASK of shape [H, W] or [frames, H, W]"
    if value is None:
        return None
    if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3) or value.numel() == 0:
        raise ValueError(message)
    mask = value.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(mask).all():
        raise ValueError(message)
    return mask.unsqueeze(0) if mask.ndim == 2 else mask.contiguous()


def curve_at(curve: tuple[float, ...], index: int, count: int) -> float:
    """Value of ``curve`` stretched over ``count`` positions, linearly interpolated at ``index``."""
    if len(curve) == 1 or count == 1:
        return curve[0]
    position = index * (len(curve) - 1) / (count - 1)
    low = math.floor(position)
    high = min(low + 1, len(curve) - 1)
    return curve[low] + (curve[high] - curve[low]) * (position - low)


@dataclass(frozen=True, eq=False)
class GlitchShape:
    distribution: str
    spike_density: float
    noise_scale: int
    step_curve: tuple[float, ...] | None
    block_curve: tuple[float, ...] | None
    spatial_mask: torch.Tensor | None
    _mask_cache: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def build(cls, distribution, spike_density, noise_scale, step_curve=None, block_curve=None,
              spatial_mask=None) -> GlitchShape:
        if distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {', '.join(DISTRIBUTIONS)}; got {distribution!r}")
        try:
            density = float(spike_density)
        except (TypeError, ValueError):
            density = float("nan")
        if not (math.isfinite(density) and 0.0 < density <= 1.0):
            raise ValueError(f"spike_density must be in (0, 1]; got {spike_density!r}")
        scale = noise_scale
        if isinstance(scale, float) and scale.is_integer():
            scale = int(scale)
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError(f"noise_scale must be an integer >= 1; got {noise_scale!r}")
        return cls(distribution, density, scale, _curve("step_curve", step_curve),
                   _curve("block_curve", block_curve), _mask(spatial_mask))

    def step_multiplier(self, step: int, n_steps: int) -> float:
        return 1.0 if self.step_curve is None else curve_at(self.step_curve, step, n_steps)

    def block_multiplier(self, block: int, n_blocks: int) -> float:
        return 1.0 if self.block_curve is None else curve_at(self.block_curve, block, n_blocks)

    def token_mask(self, step: int, n_steps: int, h: int, w: int, device) -> torch.Tensor | None:
        """Mask frame for ``step`` resized to the ``h × w`` token grid, shaped [1, h*w, 1] (row-major)."""
        if self.spatial_mask is None:
            return None
        frames = self.spatial_mask.shape[0]
        if frames == 1 or n_steps == 1:
            frame = 0
        else:
            frame = min(frames - 1, math.floor(step * (frames - 1) / (n_steps - 1) + 0.5))
        key = (frame, h, w, str(device))
        if key not in self._mask_cache:
            # Unbounded by design: entries accumulate for every distinct (frame, h, w, device)
            # seen, bounded in practice only by (frames × grids × devices). A many-frame mask
            # (KJNodes allows up to 4096) can therefore pin a resized copy per frame in memory
            # until this shape node re-executes and a fresh GlitchShape (and cache) is built.
            resized = F.interpolate(
                self.spatial_mask[frame][None, None], size=(h, w), mode="bilinear", align_corners=False, antialias=True
            )
            self._mask_cache[key] = resized.reshape(1, h * w, 1).to(device)
        return self._mask_cache[key]

    def draw_noise(self, k: int, h: int, w: int, generator: torch.Generator, device) -> torch.Tensor:
        """Noise field of shape [h*w, k] from the chosen distribution and blob size."""
        scale = self.noise_scale
        size = (k, math.ceil(h / scale), math.ceil(w / scale))
        if self.distribution == "gaussian":
            values = torch.randn(size, generator=generator, device=device, dtype=torch.float32)
        else:
            u = torch.rand(size, generator=generator, device=device, dtype=torch.float32)
            if self.distribution == "uniform":
                values = (2.0 * u - 1.0) * math.sqrt(3.0)
            elif self.distribution == "laplace":
                t = torch.clamp(u - 0.5, -0.5 + _EDGE, 0.5 - _EDGE)
                values = -(1.0 / math.sqrt(2.0)) * torch.sign(t) * torch.log1p(-2.0 * t.abs())
            elif self.distribution == "cauchy":
                values = torch.clamp(
                    torch.tan(math.pi * (torch.clamp(u, _EDGE, 1.0 - _EDGE) - 0.5)), -CAUCHY_CLIP, CAUCHY_CLIP
                )
            elif self.distribution == "spikes":
                signs = torch.rand(size, generator=generator, device=device, dtype=torch.float32)
                spike = torch.where(signs < 0.5, -1.0, 1.0) / math.sqrt(self.spike_density)
                values = torch.where(u < self.spike_density, spike, torch.zeros((), device=device))
            elif self.distribution == "binary":
                values = torch.where(u < 0.5, -1.0, 1.0)
            else:
                raise ValueError(f"unsupported distribution: {self.distribution!r}")
        if scale > 1:
            values = F.interpolate(values[None], size=(h, w), mode="bilinear", align_corners=False)[0]
            rms = values.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
            values = torch.where(rms > 0, values / rms.clamp_min(1e-12), values)
        return values.reshape(k, h * w).T

    def describe(self, mode: str) -> str:
        def points(curve):
            return "-" if curve is None else f"{len(curve)}pts"

        mask = "-" if self.spatial_mask is None else "x".join(str(d) for d in self.spatial_mask.shape)
        text = (
            f"shape dist={self.distribution} density={self.spike_density:g} scale={self.noise_scale} "
            f"step_curve={points(self.step_curve)} block_curve={points(self.block_curve)} mask={mask}"
        )
        if mode != "noise":
            text += f" (dist/density/scale unused in mode {mode})"
        return text
