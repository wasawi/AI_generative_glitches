"""Validated, immutable description of one glitch experiment (no torch, no ComfyUI)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

MODES = ("dropout", "amplify", "sign_flip", "noise")
TARGETS = ("attention", "mlp", "both")
FAMILIES = ("attention", "mlp")
STRENGTH_WIDGET_LIMIT = 1_000_000.0  # widget bound only; any finite strength is valid
SEED_MAX = 2**63 - 1


def _finite_float(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number; got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    return number


def _int(name, value):
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer; got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer; got {value!r}") from None


def _int_range(name, start, end):
    start_value = _int(f"{name}_start", start)
    end_value = _int(f"{name}_end", end)
    if start_value < 0:
        raise ValueError(f"{name}_start must be >= 0; got {start_value}")
    if end_value < start_value:
        raise ValueError(f"{name}_end ({end_value}) must be >= {name}_start ({start_value})")
    return start_value, end_value


@dataclass(frozen=True)
class GlitchRecipe:
    enabled: bool
    mode: str
    strength: float
    probability: float
    target: str
    block_start: int
    block_end: int
    step_start: int
    step_end: int
    glitch_seed: int

    @classmethod
    def build(cls, *, enabled, mode, strength, probability, target,
              block_start, block_end, step_start, step_end, glitch_seed) -> GlitchRecipe:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
        if target not in TARGETS:
            raise ValueError(f"target must be one of {', '.join(TARGETS)}; got {target!r}")
        strength_value = _finite_float("strength", strength)
        probability_value = _finite_float("probability", probability)
        if not 0.0 <= probability_value <= 1.0:
            raise ValueError(f"probability must be between 0 and 1; got {probability_value:g}")
        block_start_value, block_end_value = _int_range("block", block_start, block_end)
        step_start_value, step_end_value = _int_range("step", step_start, step_end)
        seed = _int("glitch_seed", glitch_seed)
        if not 0 <= seed <= SEED_MAX:
            raise ValueError(f"glitch_seed must be between 0 and {SEED_MAX}; got {seed}")
        return cls(bool(enabled), mode, strength_value, probability_value, target,
                   block_start_value, block_end_value, step_start_value, step_end_value, seed)

    @property
    def families(self) -> tuple[str, ...]:
        return FAMILIES if self.target == "both" else (self.target,)

    @property
    def blocks(self) -> range:
        return range(self.block_start, self.block_end + 1)

    @property
    def site_count(self) -> int:
        return len(self.blocks) * len(self.families)

    def noop_reason(self) -> str | None:
        if not self.enabled:
            return "disabled"
        if self.strength == 0:
            return "strength=0"
        if self.probability == 0:
            return "probability=0"
        return None

    def step_active(self, step: int) -> bool:
        return self.step_start <= step <= self.step_end

    def describe(self) -> str:
        body = (
            f"krea2-glitch v1 | mode={self.mode} strength={self.strength:g} probability={self.probability:g} | "
            f"target={self.target} blocks={self.block_start}-{self.block_end} sites={self.site_count} | "
            f"steps={self.step_start}-{self.step_end} | tokens=image | glitch_seed={self.glitch_seed}"
        )
        reason = self.noop_reason()
        return body if reason is None else f"no-op ({reason}) | {body}"


def clamp_to_model(recipe: GlitchRecipe, n_blocks: int) -> GlitchRecipe:
    """Return ``recipe`` with its block range clamped to the model's blocks.

    Clamping rather than raising: a block_end past the last block (from a widget, a randomizer
    or a shorter model) must never abort a generation. The recipe string then shows the range
    actually used.
    """
    last = n_blocks - 1
    start, end = min(recipe.block_start, last), min(recipe.block_end, last)
    if (start, end) == (recipe.block_start, recipe.block_end):
        return recipe
    return replace(recipe, block_start=start, block_end=end)
