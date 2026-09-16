# Glitch Shape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `Glitch Shape (Krea2)` node (noise distribution, noise blob size, step/block strength curves, spatial mask) and an optional `shape` input on `Glitch Model (Krea2)`, without changing behaviour when no shape is connected.

**Architecture:** A frozen `GlitchShape` value object in `glitches/shape.py` (torch only) validates inputs and provides curve multipliers, a cached token mask and the noise field. `apply_glitch` gains a separate shaped path used only when a shape is passed; the existing path stays byte-for-byte the same. The runtime passes the token grid, step count and block count through. `node.py` adds the shape node and the optional input.

**Tech Stack:** Python 3.12.11, torch 2.9.1, ComfyUI 0.35.1 (read-only), pytest in `./.test-deps`.

**Spec:** `docs/superpowers/specs/2026-09-15-glitch-shape-design.md` (extends `docs/superpowers/specs/2026-09-14-krea2-glitch-lab-design.md`)

## Global Constraints

- Write only inside `/path/to/AI_generative_glitches`. Other paths are read-only. Tests must not use pytest's `tmp_path`/`tmpdir`.
- Python: `python`. Test command, from the repo root, always with `tests` or explicit test paths: `PYTHONPATH=.test-deps PYTHONDONTWRITEBYTECODE=1 python -m pytest <paths> -v` (abbreviated `$PYTEST <paths>`).
- Only `glitches/node.py` imports `comfy`; relative imports inside the package.
- Without a shape, `apply_glitch`, the recipe string and every existing test stay exactly as they are.
- Registry keys: `GlitchModelKrea2` (existing), `GlitchShapeKrea2` (display `Glitch Shape (Krea2)`), category `experimental/glitches`, custom type `GLITCH_SHAPE`.
- Every commit message ends with a `Co-Authored-By: <model that authored the commit> <noreply@anthropic.com>` line and `Claude-Session: https://claude.ai/code/session_01J7HDJFz4GpCtv6Nnr8Wqvk`.

---

## File Structure

| Path | Change | Responsibility |
|---|---|---|
| `glitches/shape.py` | create | `GlitchShape`, `curve_at`, `DISTRIBUTIONS`, validation, token mask, noise field, summary text |
| `glitches/effects.py` | modify | shaped path in `apply_glitch` |
| `glitches/runtime.py` | modify | `image_token_grid`, pass shape/grid/step and block counts |
| `glitches/node.py` | modify | `GlitchShapeKrea2`; optional `shape` on `GlitchModelKrea2` |
| `__init__.py` | modify | register the shape node |
| `tests/test_shape.py` | create | validation, curves, summary |
| `tests/test_shape_fields.py` | create | token mask, noise field |
| `tests/test_effects_shape.py` | create | shaped `apply_glitch` |
| `tests/test_runtime.py` | modify | grid helper, shape pass-through |
| `tests/test_shape_integration.py` | create | real ComfyUI + tiny Krea2 |
| `tests/test_comfyui_loading.py` | modify | shape node registered, `shape` optional |
| `tests/workflow_helpers.py` | create | shared workflow test helpers |
| `tests/test_random_workflow.py` | modify | use shared helpers |
| `tests/test_shape_workflow.py` | create | example workflow structure |
| `workflows/krea2-glitch-shape.json` | create | example UI-format workflow |
| `README.md` | modify | "Shaping the glitch" section |

---

### Task 1: GlitchShape settings, curves and summary

**Files:**
- Create: `glitches/shape.py`
- Test: `tests/test_shape.py`

**Interfaces:**
- Consumes: nothing.
- Produces (in `glitches/shape.py`): `DISTRIBUTIONS`, `CAUCHY_CLIP = 20.0`, `NOISE_SCALE_MAX = 64`, `curve_at(curve: tuple[float, ...], index: int, count: int) -> float`, `GlitchShape.build(distribution, spike_density, noise_scale, step_curve=None, block_curve=None, spatial_mask=None) -> GlitchShape`, fields `distribution, spike_density, noise_scale, step_curve, block_curve, spatial_mask`, methods `step_multiplier(step, n_steps) -> float`, `block_multiplier(block, n_blocks) -> float`, `describe(mode: str) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_shape.py`:
```python
import pytest
import torch

from glitches.shape import DISTRIBUTIONS, GlitchShape, curve_at


def build(**overrides):
    args = dict(distribution="gaussian", spike_density=0.05, noise_scale=1)
    args.update(overrides)
    return GlitchShape.build(**args)


class HasToList:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def test_distributions_are_listed_in_spec_order():
    assert DISTRIBUTIONS == ("gaussian", "uniform", "laplace", "cauchy", "spikes", "binary")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.5, (0.5,)),
        (2, (2.0,)),
        ([0, 0.5, 1], (0.0, 0.5, 1.0)),
        ((1, 2), (1.0, 2.0)),
        ([[0, 1], [2]], (0.0, 1.0, 2.0)),
        (torch.tensor([[0.0, 1.0], [2.0, 3.0]]), (0.0, 1.0, 2.0, 3.0)),
        (HasToList([3, 4]), (3.0, 4.0)),
    ],
)
def test_curves_are_normalized_to_float_tuples(value, expected):
    assert build(step_curve=value).step_curve == expected
    assert build(block_curve=value).block_curve == expected


def test_unconnected_inputs_stay_none():
    shape = build()
    assert shape.step_curve is None and shape.block_curve is None and shape.spatial_mask is None


def test_2d_mask_becomes_a_single_float32_cpu_frame():
    mask = build(spatial_mask=torch.ones(4, 6, dtype=torch.float64)).spatial_mask
    assert mask.shape == (1, 4, 6) and mask.dtype == torch.float32 and mask.device.type == "cpu"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"distribution": "perlin"}, "distribution must be one of gaussian, uniform, laplace, cauchy, spikes, binary"),
        ({"spike_density": 0.0}, r"spike_density must be in \(0, 1\]"),
        ({"spike_density": 1.5}, r"spike_density must be in \(0, 1\]"),
        ({"spike_density": float("nan")}, r"spike_density must be in \(0, 1\]"),
        ({"spike_density": "x"}, r"spike_density must be in \(0, 1\]"),
        ({"noise_scale": 0}, "noise_scale must be an integer >= 1"),
        ({"noise_scale": 2.5}, "noise_scale must be an integer >= 1"),
        ({"noise_scale": True}, "noise_scale must be an integer >= 1"),
        ({"step_curve": []}, "step_curve must be a finite number or a non-empty list of finite numbers"),
        ({"step_curve": [1.0, float("nan")]}, "step_curve must be a finite number or a non-empty list of finite numbers"),
        ({"step_curve": "abc"}, "step_curve must be a finite number or a non-empty list of finite numbers"),
        ({"step_curve": ["1"]}, "step_curve must be a finite number or a non-empty list of finite numbers"),
        ({"block_curve": [float("inf")]}, "block_curve must be a finite number or a non-empty list of finite numbers"),
        ({"spatial_mask": torch.zeros(1, 1, 2, 2)}, r"spatial_mask must be a finite MASK of shape \[H, W\] or \[frames, H, W\]"),
        ({"spatial_mask": torch.zeros(0, 4)}, r"spatial_mask must be a finite MASK of shape \[H, W\] or \[frames, H, W\]"),
        ({"spatial_mask": torch.tensor([[float("nan")]])}, r"spatial_mask must be a finite MASK of shape \[H, W\] or \[frames, H, W\]"),
        ({"spatial_mask": [[1.0]]}, r"spatial_mask must be a finite MASK of shape \[H, W\] or \[frames, H, W\]"),
    ],
)
def test_validation_messages(overrides, message):
    with pytest.raises(ValueError, match=message):
        build(**overrides)


def test_float_noise_scale_with_integer_value_is_accepted():
    assert build(noise_scale=4.0).noise_scale == 4


def test_curve_at_single_value_and_single_step():
    assert curve_at((0.7,), 5, 8) == 0.7
    assert curve_at((0.2, 0.9), 0, 1) == 0.2


def test_two_point_curve_interpolates_linearly_across_steps():
    assert [curve_at((0.0, 1.0), i, 8) for i in range(8)] == pytest.approx([i / 7 for i in range(8)])


def test_sixteen_point_curve_spans_steps_and_blocks():
    curve = tuple(float(v) for v in range(16))
    assert curve_at(curve, 0, 8) == 0.0 and curve_at(curve, 7, 8) == 15.0
    assert curve_at(curve, 0, 28) == 0.0 and curve_at(curve, 27, 28) == 15.0
    assert curve_at(curve, 9, 28) == pytest.approx(5.0)


def test_multipliers_default_to_one_and_follow_curves():
    assert build().step_multiplier(3, 8) == 1.0
    assert build().block_multiplier(4, 28) == 1.0
    shape = build(step_curve=[0.0, 1.0], block_curve=[2.0])
    assert shape.step_multiplier(7, 8) == 1.0
    assert shape.step_multiplier(0, 8) == 0.0
    assert shape.block_multiplier(13, 28) == 2.0


def test_describe_lists_every_setting():
    shape = build(distribution="spikes", noise_scale=4, step_curve=list(range(16)), spatial_mask=torch.ones(16, 32, 24))
    assert shape.describe("noise") == "shape dist=spikes density=0.05 scale=4 step_curve=16pts block_curve=- mask=16x32x24"


def test_describe_marks_noise_settings_unused_outside_noise_mode():
    assert build().describe("dropout") == (
        "shape dist=gaussian density=0.05 scale=1 step_curve=- block_curve=- mask=- "
        "(dist/density/scale unused in mode dropout)"
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `$PYTEST tests/test_shape.py`
Expected: collection error `ModuleNotFoundError: No module named 'glitches.shape'`.

- [ ] **Step 3: Implement `glitches/shape.py` (settings part)**

```python
"""Optional glitch shaping: noise distribution, spatial mask and step/block curves (torch only)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

DISTRIBUTIONS = ("gaussian", "uniform", "laplace", "cauchy", "spikes", "binary")
CAUCHY_CLIP = 20.0
NOISE_SCALE_MAX = 64


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
```

- [ ] **Step 4: Run to verify pass**

Run: `$PYTEST tests/test_shape.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add glitches/shape.py tests/test_shape.py
git commit -m "Add GlitchShape settings, curves and summary"
```
(with the trailer lines from Global Constraints)

---

### Task 2: Token mask and noise field

**Files:**
- Modify: `glitches/shape.py`
- Test: `tests/test_shape_fields.py`

**Interfaces:**
- Consumes: `GlitchShape` from Task 1.
- Produces: `GlitchShape.token_mask(step: int, n_steps: int, h: int, w: int, device) -> Tensor[1, h*w, 1] | None` (float32, cached per `(frame, h, w, str(device))`); `GlitchShape.draw_noise(k: int, h: int, w: int, generator: torch.Generator, device) -> Tensor[h*w, k]` (float32).

- [ ] **Step 1: Write the failing tests**

`tests/test_shape_fields.py`:
```python
import math

import pytest
import torch

from glitches.shape import CAUCHY_CLIP, GlitchShape

CPU = torch.device("cpu")


def build(**overrides):
    args = dict(distribution="gaussian", spike_density=0.05, noise_scale=1)
    args.update(overrides)
    return GlitchShape.build(**args)


def draw(distribution, k=4, h=250, w=200, scale=1, density=0.05, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return build(distribution=distribution, spike_density=density, noise_scale=scale).draw_noise(k, h, w, generator, CPU)


def test_no_mask_gives_none():
    assert build().token_mask(0, 8, 5, 3, CPU) is None


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(1, [0] * 8), (3, [0, 0, 1, 1, 1, 1, 2, 2]), (16, [0, 2, 4, 6, 9, 11, 13, 15])],
)
def test_mask_frames_play_across_steps(frames, expected):
    mask = torch.stack([torch.full((4, 4), float(f)) for f in range(frames)])
    shape = build(spatial_mask=mask)
    assert [round(shape.token_mask(i, 8, 2, 2, CPU)[0, 0, 0].item()) for i in range(8)] == expected


def test_mask_at_grid_size_keeps_row_major_layout():
    mask = torch.zeros(5, 3)
    mask[:, 0] = 1.0
    mask[0, 2] = 0.5
    tokens = build(spatial_mask=mask).token_mask(0, 8, 5, 3, CPU)
    assert tokens.shape == (1, 15, 1) and tokens.dtype == torch.float32
    assert torch.equal(tokens.reshape(5, 3), mask)


def test_mask_is_resized_to_a_non_square_grid():
    mask = torch.zeros(512, 512)
    mask[:, :256] = 1.0
    grid = build(spatial_mask=mask).token_mask(0, 8, 5, 4, CPU).reshape(5, 4)
    torch.testing.assert_close(grid[:, 0], torch.ones(5), rtol=0, atol=1e-6)
    torch.testing.assert_close(grid[:, 3], torch.zeros(5), rtol=0, atol=1e-6)


def test_resized_mask_is_cached():
    shape = build(spatial_mask=torch.rand(3, 16, 16))
    first = shape.token_mask(0, 8, 5, 3, CPU)
    assert shape.token_mask(0, 8, 5, 3, CPU) is first
    assert shape.token_mask(1, 8, 5, 3, CPU) is first  # step 1 still uses frame 0 of 3
    assert shape.token_mask(7, 8, 5, 3, CPU) is not first


@pytest.mark.parametrize("distribution", ["gaussian", "uniform", "laplace", "cauchy", "spikes", "binary"])
def test_noise_layout_dtype_and_determinism(distribution):
    values = draw(distribution)
    assert values.shape == (50000, 4) and values.dtype == torch.float32
    assert torch.isfinite(values).all()
    assert torch.equal(values, draw(distribution))
    assert not torch.equal(values, draw(distribution, seed=1))


@pytest.mark.parametrize("distribution", ["gaussian", "uniform", "laplace", "spikes", "binary"])
def test_noise_has_unit_rms(distribution):
    rms = draw(distribution).pow(2).mean().sqrt().item()
    assert abs(rms - 1.0) < 0.05


def test_uniform_and_binary_value_ranges():
    assert draw("uniform").abs().max().item() <= math.sqrt(3.0) + 1e-6
    assert set(draw("binary").unique().tolist()) == {-1.0, 1.0}


def test_spike_density_is_respected():
    values = draw("spikes", density=0.05)
    fraction = (values != 0).float().mean().item()
    assert 0.045 <= fraction <= 0.055
    assert values.abs().max().item() == pytest.approx(1 / math.sqrt(0.05))


def test_cauchy_is_clipped_but_heavy_tailed():
    values = draw("cauchy")
    assert values.abs().max().item() <= CAUCHY_CLIP
    assert (values.abs() > 5).any()


def test_noise_scale_gives_smooth_unit_rms_channels():
    k, h, w = 6, 64, 48
    smooth = draw("gaussian", k=k, h=h, w=w, scale=4).T.reshape(k, h, w)
    grainy = draw("gaussian", k=k, h=h, w=w, scale=1).T.reshape(k, h, w)
    torch.testing.assert_close(smooth.pow(2).mean(dim=(1, 2)).sqrt(), torch.ones(k), rtol=0, atol=1e-4)

    def neighbour_correlation(field):
        pairs = torch.stack([field[:, :, :-1].reshape(-1), field[:, :, 1:].reshape(-1)])
        return torch.corrcoef(pairs)[0, 1].item()

    assert neighbour_correlation(smooth) > 0.8
    assert abs(neighbour_correlation(grainy)) < 0.1


def test_scaled_sparse_spikes_with_empty_channels_stay_finite():
    values = draw("spikes", k=16, h=8, w=8, scale=8, density=0.001, seed=2)
    assert torch.isfinite(values).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `$PYTEST tests/test_shape_fields.py`
Expected: FAIL with `AttributeError: 'GlitchShape' object has no attribute 'token_mask'` / `'draw_noise'`.

- [ ] **Step 3: Implement in `glitches/shape.py`**

Add `import torch.nn.functional as F` below `import torch`, add `_EDGE = 1e-7` below `NOISE_SCALE_MAX = 64`, and add these methods to `GlitchShape` (after `block_multiplier`):

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `$PYTEST tests/test_shape.py tests/test_shape_fields.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add glitches/shape.py tests/test_shape_fields.py
git commit -m "Add shape token mask and noise field"
```

---

### Task 3: Shaped glitch path and runtime pass-through

**Files:**
- Modify: `glitches/effects.py`, `glitches/runtime.py`
- Test: `tests/test_effects_shape.py` (create), `tests/test_runtime.py` (append)

**Interfaces:**
- Consumes: `GlitchShape.step_multiplier`, `block_multiplier`, `token_mask`, `draw_noise` (Tasks 1–2); existing `select_channels`, `mix_seed`, `PURPOSE_NOISE`.
- Produces: `apply_glitch(out, recipe, block, family, step, txt, n_img, *, shape=None, grid=None, n_steps=None, n_blocks=None)`; `runtime.image_token_grid(x, patch) -> tuple[int, int]`; `GlitchWrapper(recipe, shape=None)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_effects_shape.py`:
```python
import pytest
import torch

from glitches.effects import apply_glitch, select_channels
from glitches.shape import GlitchShape
from test_effects import IMG, N_IMG, TXT, WIDTH, activations, glitch, recipe

MODES = ["dropout", "amplify", "sign_flip", "noise"]
GRID = (2, 2)  # N_IMG == 4


def shape(**overrides):
    args = dict(distribution="gaussian", spike_density=0.05, noise_scale=1)
    args.update(overrides)
    return GlitchShape.build(**args)


def shaped(out, r, s, step=2, block=3, family="mlp", n_steps=8, n_blocks=28, grid=GRID):
    return apply_glitch(out, r, block, family, step, TXT, N_IMG, shape=s, grid=grid, n_steps=n_steps, n_blocks=n_blocks)


@pytest.mark.parametrize("mode", MODES)
def test_passing_no_shape_is_bit_identical_to_the_base_path(mode):
    out = activations()
    r = recipe(mode, 0.5)
    result = apply_glitch(out, r, 3, "mlp", 2, TXT, N_IMG, shape=None, grid=GRID, n_steps=8, n_blocks=28)
    assert torch.equal(result, glitch(out, r))


@pytest.mark.parametrize("mode", ["dropout", "amplify", "sign_flip"])
def test_all_ones_shape_matches_the_unshaped_result(mode):
    out = activations()
    r = recipe(mode, 0.5)
    ones = shape(step_curve=[1.0], block_curve=[1.0], spatial_mask=torch.ones(2, 2))
    torch.testing.assert_close(shaped(out, r, ones), glitch(out, r))


@pytest.mark.parametrize("mode", MODES)
def test_zero_step_multiplier_leaves_the_activation_untouched(mode):
    out = activations()
    assert torch.equal(shaped(out, recipe(mode, 0.5), shape(step_curve=[0.0])), out)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_mask_leaves_the_activation_untouched(mode, dtype):
    out = activations(dtype=dtype)
    result = shaped(out, recipe(mode, 0.5), shape(distribution="cauchy", spatial_mask=torch.zeros(2, 2)))
    assert result.dtype == dtype
    assert torch.equal(result, out)


def test_left_column_mask_only_glitches_left_column_tokens():
    out = activations()
    r = recipe("dropout", 1.0, probability=1.0)
    result = shaped(out, r, shape(spatial_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]])))
    image_before, image_after = out[:, IMG], result[:, IMG]
    assert torch.count_nonzero(image_after[:, [0, 2]]) == 0  # row-major tokens 0 and 2 are the left column
    assert torch.equal(image_after[:, [1, 3]], image_before[:, [1, 3]])
    assert torch.equal(result[:, :TXT], out[:, :TXT])


def test_curves_scale_the_dose():
    out = activations()
    r = recipe("dropout", 1.0)
    selected = select_channels(r, 3, "mlp", 2, WIDTH)
    result = shaped(out, r, shape(step_curve=[0.5], block_curve=[1.0]))
    torch.testing.assert_close(result[:, IMG][..., selected], out[:, IMG][..., selected] * 0.5)


def test_shaped_noise_touches_only_selected_channels_and_is_repeatable():
    out = activations()
    r = recipe("noise", 0.5)
    s = shape(distribution="spikes", spike_density=1.0, noise_scale=2)  # density 1: every selected channel spikes
    result = shaped(out, r, s)
    selected = select_channels(r, 3, "mlp", 2, WIDTH)
    unselected = torch.ones(WIDTH, dtype=torch.bool)
    unselected[selected] = False
    assert torch.equal(result[:, IMG][..., unselected], out[:, IMG][..., unselected])
    assert not torch.equal(result[:, IMG][..., selected], out[:, IMG][..., selected])
    assert torch.equal(result, shaped(out, r, s))


def test_grid_must_match_the_image_token_count():
    with pytest.raises(ValueError, match="grid 3x3 does not match 4 image tokens"):
        shaped(activations(), recipe("dropout", 0.5), shape(), grid=(3, 3))
```

Append to `tests/test_runtime.py`:
```python
def test_image_token_grid_returns_height_and_width():
    from glitches.runtime import image_token_grid

    assert image_token_grid(torch.zeros(1, 4, 5, 6), 2) == (3, 3)
    assert image_token_grid(torch.zeros(1, 4, 1, 5, 6), 2) == (3, 3)


def test_wrapper_passes_shape_grid_step_and_block_counts(monkeypatch):
    import glitches.runtime as runtime
    from glitches.shape import GlitchShape

    seen = []

    def spy(output, recipe_, block, family, step, txt, n_img, **kwargs):
        seen.append((block, family, step, txt, n_img, kwargs))
        return output

    monkeypatch.setattr(runtime, "apply_glitch", spy)
    s = GlitchShape.build("gaussian", 0.05, 1)
    call(GlitchWrapper(recipe(target="mlp", block_start=1, block_end=1), s), FakeExecutor(FakeDiT()), sigma=0.5)
    assert seen == [(1, "mlp", 1, TXT, 6, {"shape": s, "grid": (2, 3), "n_steps": 2, "n_blocks": 2})]
```

- [ ] **Step 2: Run to verify failure**

Run: `$PYTEST tests/test_effects_shape.py tests/test_runtime.py`
Expected: FAIL (`TypeError: apply_glitch() got an unexpected keyword argument 'shape'`, `ImportError: cannot import name 'image_token_grid'`, `TypeError: GlitchWrapper.__init__() takes 2 positional arguments`).

- [ ] **Step 3: Implement the shaped path in `glitches/effects.py`**

Replace the `apply_glitch` signature line pair

```python
def apply_glitch(out: torch.Tensor, recipe: GlitchRecipe, block: int, family: str, step: int,
                 txt: int, n_img: int) -> torch.Tensor:
    """Return a copy of ``out`` [B, L, D] with the recipe applied to rows ``txt:txt+n_img``."""
```

with

```python
def apply_glitch(out: torch.Tensor, recipe: GlitchRecipe, block: int, family: str, step: int,
                 txt: int, n_img: int, *, shape=None, grid=None, n_steps=None, n_blocks=None) -> torch.Tensor:
    """Return a copy of ``out`` [B, L, D] with the recipe applied to rows ``txt:txt+n_img``."""
    if shape is not None:
        return _apply_shaped(out, recipe, block, family, step, txt, n_img, shape, grid, n_steps, n_blocks)
```

and add below `apply_glitch` (leave its existing body unchanged):

```python
def _apply_shaped(out, recipe, block, family, step, txt, n_img, shape, grid, n_steps, n_blocks):
    h, w = grid
    if h * w != n_img:
        raise ValueError(f"grid {h}x{w} does not match {n_img} image tokens")
    scalar = shape.step_multiplier(step, n_steps) * shape.block_multiplier(block, n_blocks)
    mask = shape.token_mask(step, n_steps, h, w, out.device)
    if scalar == 0.0 and mask is None:
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
```

- [ ] **Step 4: Implement the runtime pass-through in `glitches/runtime.py`**

1. Rename `def image_token_count(x: torch.Tensor, patch: int) -> int:` to `def image_token_grid(x: torch.Tensor, patch: int) -> tuple[int, int]:` (keep its comment and validation), change its final line to `return math.ceil(x.shape[-2] / patch), math.ceil(x.shape[-1] / patch)`, and add after it:

```python
def image_token_count(x: torch.Tensor, patch: int) -> int:
    h, w = image_token_grid(x, patch)
    return h * w
```

2. Replace `_make_hook` with:

```python
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
```

3. Replace the `GlitchWrapper` class with:

```python
class GlitchWrapper:
    def __init__(self, recipe: GlitchRecipe, shape=None):
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
```

- [ ] **Step 5: Run to verify pass**

Run: `$PYTEST tests/test_effects_shape.py tests/test_runtime.py tests/test_effects.py`
Expected: all PASS (existing glitch and runtime tests unchanged and passing).

- [ ] **Step 6: Commit**

```bash
git add glitches/effects.py glitches/runtime.py tests/test_effects_shape.py tests/test_runtime.py
git commit -m "Apply glitch shapes in the glitch path and runtime"
```

---

### Task 4: Nodes, registration and ComfyUI integration tests

**Files:**
- Modify: `glitches/node.py`, `__init__.py`, `tests/test_comfyui_loading.py`
- Test: `tests/test_shape_integration.py` (create)

**Interfaces:**
- Consumes: `GlitchShape`, `DISTRIBUTIONS`, `NOISE_SCALE_MAX` (Tasks 1–2); `GlitchWrapper(recipe, shape)` (Task 3); fixtures/helpers in `tests/test_node_integration.py` (`comfy`, `node`, `patcher`, `apply`, `run`, `hook_count`, `H`, `W`, `SCHEDULE`).
- Produces: `glitches.node.GlitchShapeKrea2` (`build(distribution, spike_density, noise_scale, step_curve=None, block_curve=None, spatial_mask=None) -> (GlitchShape,)`); `GlitchModelKrea2.apply(..., shape=None)`; registry key `GlitchShapeKrea2`.

- [ ] **Step 1: Write the failing tests**

`tests/test_shape_integration.py`:
```python
import pytest
import torch

from test_node_integration import H, W, apply, comfy, hook_count, node, patcher, run  # noqa: F401 (fixtures)

TXT = 5
GRID_H, GRID_W = (H + 1) // 2, (W + 1) // 2  # latent 9x6 -> 5x3 image tokens


@pytest.fixture
def shape_node(comfy):
    from glitches.node import GlitchShapeKrea2

    return GlitchShapeKrea2()


def make_shape(shape_node, **overrides):
    args = dict(distribution="gaussian", spike_density=0.05, noise_scale=1)
    args.update(overrides)
    (shape,) = shape_node.build(**args)
    return shape


def test_shape_node_declares_inputs_and_builds_a_shape(shape_node):
    from glitches.shape import GlitchShape

    spec = type(shape_node).INPUT_TYPES()
    assert list(spec["required"]) == ["distribution", "spike_density", "noise_scale"]
    assert list(spec["optional"]) == ["step_curve", "block_curve", "spatial_mask"]
    assert spec["optional"]["step_curve"] == ("FLOAT", {"forceInput": True})
    assert type(shape_node).RETURN_TYPES == ("GLITCH_SHAPE",)
    shape = make_shape(shape_node, distribution="spikes", noise_scale=4, step_curve=[0.0, 1.0], spatial_mask=torch.ones(3, 8, 8))
    assert isinstance(shape, GlitchShape)


def test_recipe_gains_shape_summary_only_when_connected(shape_node, node, patcher):
    shape = make_shape(shape_node)
    _, plain = apply(node, patcher)
    _, shaped = apply(node, patcher, shape=shape)
    assert " | shape " not in plain
    assert shaped == plain + " | " + shape.describe("dropout")


def test_wrong_shape_type_is_rejected(node, patcher):
    with pytest.raises(TypeError, match=r"shape must come from a Glitch Shape \(Krea2\) node"):
        apply(node, patcher, shape="not a shape")


def test_mask_limits_the_glitch_to_masked_tokens(shape_node, node, patcher):
    mask = torch.zeros(GRID_H, GRID_W)
    mask[:, 0] = 1.0
    shape = make_shape(shape_node, spatial_mask=mask)
    clone, _ = apply(node, patcher, probability=1.0, block_start=0, block_end=0, step_start=0, step_end=3, shape=shape)
    dit = patcher.model.diffusion_model
    captured = []
    handle = dit.blocks[0].register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        run(patcher, 0.75)
        run(clone, 0.75)
    finally:
        handle.remove()
    base, glitched = captured
    n_img = GRID_H * GRID_W
    base_img = base[:, TXT:TXT + n_img].reshape(2, GRID_H, GRID_W, -1)
    glitched_img = glitched[:, TXT:TXT + n_img].reshape(2, GRID_H, GRID_W, -1)
    assert not torch.equal(base_img[:, :, 0], glitched_img[:, :, 0])
    assert torch.equal(base_img[:, :, 1:], glitched_img[:, :, 1:])
    assert torch.equal(base[:, :TXT], glitched[:, :TXT])
    assert hook_count(clone) == 0


def test_step_curve_zeros_leave_those_steps_untouched(shape_node, node, patcher):
    shape = make_shape(shape_node, step_curve=[0.0, 1.0, 0.0, 1.0])
    clone, _ = apply(node, patcher, step_start=0, step_end=3, shape=shape)
    for sigma, changed in [(1.0, False), (0.75, True), (0.5, False), (0.25, True)]:
        assert (not torch.equal(run(patcher, sigma), run(clone, sigma))) == changed, f"sigma={sigma}"


def test_one_shape_can_drive_two_chained_glitch_nodes(shape_node, node, patcher):
    shape = make_shape(shape_node, distribution="binary", noise_scale=2)
    first, _ = apply(node, patcher, mode="noise", strength=0.5, target="attention", block_end=0, shape=shape)
    second, _ = apply(node, first, mode="amplify", strength=2.0, target="mlp", block_start=1, shape=shape)
    base, one, two = run(patcher, 0.75), run(first, 0.75), run(second, 0.75)
    assert torch.isfinite(two).all()
    assert not torch.equal(base, one) and not torch.equal(one, two)
    assert hook_count(second) == 0
```

In `tests/test_comfyui_loading.py`, after the line `assert required["strength"][1]["max"] == 1000.0`, add:
```python
        assert node_class.INPUT_TYPES()["optional"] == {"shape": ("GLITCH_SHAPE",)}
        shape_class = module.NODE_CLASS_MAPPINGS["GlitchShapeKrea2"]
        assert module.NODE_DISPLAY_NAME_MAPPINGS["GlitchShapeKrea2"] == "Glitch Shape (Krea2)"
        assert shape_class.CATEGORY == "experimental/glitches"
        assert shape_class.RETURN_TYPES == ("GLITCH_SHAPE",)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PYTEST tests/test_shape_integration.py tests/test_comfyui_loading.py`
Expected: FAIL (`ImportError: cannot import name 'GlitchShapeKrea2'`, `KeyError: 'optional'`).

- [ ] **Step 3: Implement `glitches/node.py`**

1. Add below the existing `from .runtime import GlitchWrapper` line:
```python
from .shape import DISTRIBUTIONS, NOISE_SCALE_MAX, GlitchShape
```
2. In `GlitchModelKrea2.INPUT_TYPES`, add an `"optional"` key next to `"required"`:
```python
            "optional": {
                "shape": ("GLITCH_SHAPE",),
            },
```
3. Change the `apply` signature to end with `glitch_seed, shape=None):`, and replace everything from the line `clone = model.clone()` through the line `return (clone, recipe.describe())` with:
```python
        if shape is not None and not isinstance(shape, GlitchShape):
            raise TypeError("shape must come from a Glitch Shape (Krea2) node")
        clone = model.clone()
        if recipe.noop_reason() is None:
            clone.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, GlitchWrapper(recipe, shape)
            )
        text = recipe.describe()
        if shape is not None:
            text += " | " + shape.describe(recipe.mode)
        return (clone, text)
```
4. Append the new node class:
```python
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
```

- [ ] **Step 4: Register the node in `__init__.py`**

```python
"""ComfyUI-LesionLab: reproducible activation glitches for Krea2 models."""

from .glitches.node import GlitchModelKrea2, GlitchShapeKrea2

NODE_CLASS_MAPPINGS = {"GlitchModelKrea2": GlitchModelKrea2, "GlitchShapeKrea2": GlitchShapeKrea2}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GlitchModelKrea2": "Glitch Model (Krea2)",
    "GlitchShapeKrea2": "Glitch Shape (Krea2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

- [ ] **Step 5: Run to verify pass**

Run: `$PYTEST tests`
Expected: all PASS, 0 skipped, no warnings summary.

- [ ] **Step 6: Commit**

```bash
git add glitches/node.py __init__.py tests/test_shape_integration.py tests/test_comfyui_loading.py
git commit -m "Add Glitch Shape (Krea2) node and optional shape input"
```

---

### Task 5: Example workflow, shared workflow test helpers and README

**Files:**
- Create: `tests/workflow_helpers.py`, `tests/test_shape_workflow.py`, `workflows/krea2-glitch-shape.json`
- Modify: `tests/test_random_workflow.py`, `README.md`

**Interfaces:**
- Consumes: registry keys and input names from Task 4; KJNodes `CreateShapeMask` (verified 2026-09-15: required widgets in order `shape` [circle/square/triangle], `frames`, `location_x`, `location_y`, `grow`, `frame_width`, `frame_height`, `shape_width`, `shape_height`; outputs `mask`, `mask_inverted`).
- Produces: `tests/workflow_helpers.py` with `load_workflow(name)`, `only(workflow, key, value)`, `smoke_inputs(class_type)`, `assert_links_consistent(workflow)`.

- [ ] **Step 1: Create the shared helpers and switch the random workflow test to them**

`tests/workflow_helpers.py`:
```python
import json
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


def load_workflow(name):
    return json.loads((WORKFLOWS / name).read_text())


def only(workflow, key, value):
    found = [n for n in workflow["nodes"] if n.get(key) == value]
    assert len(found) == 1, f"expected exactly one node with {key}={value!r}, found {len(found)}"
    return found[0]


def smoke_inputs(class_type):
    smoke = load_workflow("krea2-glitch-smoke.json")
    return next(node for node in smoke.values() if node["class_type"] == class_type)["inputs"]


def assert_links_consistent(workflow):
    nodes = {n["id"]: n for n in workflow["nodes"]}
    link_ids = set()
    for link_id, origin, origin_slot, target, target_slot, _type in workflow["links"]:
        assert link_id not in link_ids
        link_ids.add(link_id)
        assert link_id in (nodes[origin]["outputs"][origin_slot]["links"] or [])
        assert nodes[target]["inputs"][target_slot]["link"] == link_id
    for node in workflow["nodes"]:
        for slot in node.get("inputs", []):
            assert slot.get("link") is None or slot["link"] in link_ids
        for slot in node.get("outputs", []):
            assert set(slot.get("links") or []) <= link_ids
    assert workflow["last_link_id"] == max(link_ids)
    assert workflow["last_node_id"] == max(nodes)
```

In `tests/test_random_workflow.py`:
1. Replace the import block and module constants up to (not including) `DRIVEN_INPUTS = [` with:
```python
import random

import pytest

from glitches.effects import _splitmix64
from glitches.recipe import MODES, TARGETS, GlitchRecipe
from workflow_helpers import assert_links_consistent, load_workflow, only, smoke_inputs

EASY_SEED_MAX = 1125899906842624  # comfyui-easy-use py/config.py MAX_SEED_NUM
```
2. Delete its local `load`, `only` and `smoke_inputs` functions and add in their place:
```python
def load():
    return load_workflow("krea2-glitch-random.json")
```
3. Replace the body of `test_links_are_consistent_in_both_directions` with `assert_links_consistent(load())`.

Run: `$PYTEST tests/test_random_workflow.py`
Expected: 6 passed (behaviour unchanged).

- [ ] **Step 2: Write the failing example-workflow tests**

`tests/test_shape_workflow.py`:
```python
from workflow_helpers import assert_links_consistent, load_workflow, only, smoke_inputs


def load():
    return load_workflow("krea2-glitch-shape.json")


def source_of(workflow, node, input_name):
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    slot = next(i for i in node["inputs"] if i["name"] == input_name)
    assert slot["link"] is not None, f"{node['type']}.{input_name} is not linked"
    _, origin, origin_slot, *_ = links[slot["link"]]
    return nodes[origin], origin_slot


def test_links_are_consistent():
    assert_links_consistent(load())


def test_shape_node_feeds_the_glitch_node_and_the_mask_feeds_the_shape():
    workflow = load()
    glitch = only(workflow, "type", "GlitchModelKrea2")
    shape = only(workflow, "type", "GlitchShapeKrea2")
    assert source_of(workflow, glitch, "shape") == (shape, 0)
    mask_node, mask_slot = source_of(workflow, shape, "spatial_mask")
    assert mask_node["type"] == "CreateShapeMask" and mask_slot == 0
    unlinked = {i["name"]: i["link"] for i in shape["inputs"]}
    assert unlinked["step_curve"] is None and unlinked["block_curve"] is None
    assert shape["widgets_values"] == ["spikes", 0.05, 4]
    assert glitch["widgets_values"][:2] == [True, "noise"]


def test_loaders_and_sampler_match_the_smoke_workflow():
    workflow = load()
    sampler = smoke_inputs("KSampler")
    assert only(workflow, "type", "KSampler")["widgets_values"] == [
        sampler["seed"], "fixed", sampler["steps"], sampler["cfg"],
        sampler["sampler_name"], sampler["scheduler"], sampler["denoise"],
    ]
    assert only(workflow, "type", "LoaderGGUF")["widgets_values"] == [smoke_inputs("LoaderGGUF")["gguf_name"]]
    clip = smoke_inputs("CLIPLoader")
    assert only(workflow, "type", "CLIPLoader")["widgets_values"] == [clip["clip_name"], clip["type"], clip["device"]]
    assert only(workflow, "type", "VAELoader")["widgets_values"] == [smoke_inputs("VAELoader")["vae_name"]]
```

Run: `$PYTEST tests/test_shape_workflow.py`
Expected: FAIL with `FileNotFoundError` for `krea2-glitch-shape.json`.

- [ ] **Step 3: Generate `workflows/krea2-glitch-shape.json`**

Run once from the repo root (writes only that file):
```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'EOF'
import json
from pathlib import Path

nodes, links = [], []

def node(type_, title, pos, size, inputs, outputs, widgets_values=None):
    n = {"id": len(nodes) + 1, "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": len(nodes),
         "mode": 0, "properties": {"Node name for S&R": type_},
         "inputs": [dict({"localized_name": i[0], "name": i[0], "type": i[1], "link": None},
                         **({"widget": {"name": i[0]}} if i[2] == "w" else {}),
                         **({"shape": 7} if i[2] == "o" else {})) for i in inputs],
         "outputs": [{"localized_name": name, "name": name, "type": kind, "links": []} for name, kind in outputs]}
    if title:
        n["title"] = title
    if widgets_values is not None:
        n["widgets_values"] = widgets_values
    nodes.append(n)
    return n

def link(origin, origin_slot, target, input_name):
    target_slot = next(k for k, i in enumerate(target["inputs"]) if i["name"] == input_name)
    link_id = len(links) + 1
    origin["outputs"][origin_slot]["links"].append(link_id)
    target["inputs"][target_slot]["link"] = link_id
    links.append([link_id, origin["id"], origin_slot, target["id"], target_slot, origin["outputs"][origin_slot]["type"]])

loader = node("LoaderGGUF", "Krea2 GGUF", (100, 130), (300, 58), [("gguf_name", "COMBO", "w")], [("MODEL", "MODEL")],
              ["KREA/museByStableYogi_v25GGUF.gguf"])
mask = node("CreateShapeMask", "Spatial mask (circle)", (-340, 470), (320, 270),
            [("shape", "COMBO", "w"), ("frames", "INT", "w"), ("location_x", "INT", "w"), ("location_y", "INT", "w"),
             ("grow", "INT", "w"), ("frame_width", "INT", "w"), ("frame_height", "INT", "w"),
             ("shape_width", "INT", "w"), ("shape_height", "INT", "w")],
            [("mask", "MASK"), ("mask_inverted", "MASK")], ["circle", 1, 512, 512, 0, 1024, 1024, 512, 512])
shape = node("GlitchShapeKrea2", "Glitch Shape (Krea2)", (100, 470), (320, 170),
             [("distribution", "COMBO", "w"), ("spike_density", "FLOAT", "w"), ("noise_scale", "INT", "w"),
              ("step_curve", "FLOAT", "o"), ("block_curve", "FLOAT", "o"), ("spatial_mask", "MASK", "o")],
             [("shape", "GLITCH_SHAPE")], ["spikes", 0.05, 4])
glitch = node("GlitchModelKrea2", "Glitch Model (Krea2)", (470, 130), (300, 314),
              [("model", "MODEL", "l"), ("enabled", "BOOLEAN", "w"), ("mode", "COMBO", "w"), ("strength", "FLOAT", "w"),
               ("probability", "FLOAT", "w"), ("target", "COMBO", "w"), ("block_start", "INT", "w"), ("block_end", "INT", "w"),
               ("step_start", "INT", "w"), ("step_end", "INT", "w"), ("glitch_seed", "INT", "w"), ("shape", "GLITCH_SHAPE", "o")],
              [("model", "MODEL"), ("recipe", "STRING")], [True, "noise", 0.5, 0.25, "both", 0, 27, 0, 3, 0])
clip = node("CLIPLoader", "Krea2 text encoder", (100, 700), (300, 106),
            [("clip_name", "COMBO", "w"), ("type", "COMBO", "w"), ("device", "COMBO", "w")], [("CLIP", "CLIP")],
            ["qwen3-vl-4b-instruct-abliterated.safetensors", "krea2", "default"])
prompt = node("CLIPTextEncode", "Prompt", (470, 700), (400, 200), [("clip", "CLIP", "l"), ("text", "STRING", "w")],
              [("CONDITIONING", "CONDITIONING")],
              ["a photograph of a red bicycle leaning against a white brick wall, soft morning daylight, sharp focus"])
negative = node("ConditioningZeroOut", "Empty negative", (970, 700), (220, 26), [("conditioning", "CONDITIONING", "l")],
                [("CONDITIONING", "CONDITIONING")])
latent = node("EmptyLatentImage", "Latent", (100, 950), (300, 106),
              [("width", "INT", "w"), ("height", "INT", "w"), ("batch_size", "INT", "w")], [("LATENT", "LATENT")], [1024, 1024, 1])
sampler = node("KSampler", "KSampler (seed fixed)", (1280, 130), (300, 262),
               [("model", "MODEL", "l"), ("positive", "CONDITIONING", "l"), ("negative", "CONDITIONING", "l"),
                ("latent_image", "LATENT", "l"), ("seed", "INT", "w"), ("steps", "INT", "w"), ("cfg", "FLOAT", "w"),
                ("sampler_name", "COMBO", "w"), ("scheduler", "COMBO", "w"), ("denoise", "FLOAT", "w")],
               [("LATENT", "LATENT")], [123456789, "fixed", 8, 1.0, "er_sde", "simple", 1.0])
vae = node("VAELoader", "VAE", (100, 1100), (300, 58), [("vae_name", "COMBO", "w")], [("VAE", "VAE")], ["qwen_image_vae.safetensors"])
decode = node("VAEDecode", "Decode", (1650, 130), (160, 46), [("samples", "LATENT", "l"), ("vae", "VAE", "l")], [("IMAGE", "IMAGE")])
save = node("SaveImage", "Save", (1880, 130), (320, 320), [("images", "IMAGE", "l"), ("filename_prefix", "STRING", "w")], [],
            ["glitchlab/shape"])
recipe = node("PreviewAny", "Recipe", (830, -140), (420, 160), [("source", "*", "l")], [("STRING", "STRING")], [])

link(loader, 0, glitch, "model"); link(mask, 0, shape, "spatial_mask"); link(shape, 0, glitch, "shape")
link(glitch, 0, sampler, "model"); link(glitch, 1, recipe, "source")
link(clip, 0, prompt, "clip"); link(prompt, 0, negative, "conditioning"); link(prompt, 0, sampler, "positive")
link(negative, 0, sampler, "negative"); link(latent, 0, sampler, "latent_image")
link(sampler, 0, decode, "samples"); link(vae, 0, decode, "vae"); link(decode, 0, save, "images")

for n in nodes:
    for out in n["outputs"]:
        out["links"] = out["links"] or None
workflow = {"id": "8d2e4b61-3f5a-4c0e-9a7b-2e6f1c9d4a58", "revision": 0, "last_node_id": len(nodes),
            "last_link_id": len(links), "nodes": nodes, "links": links, "groups": [], "config": {},
            "extra": {"ds": {"scale": 0.7, "offset": [500, 300]}}, "version": 0.4}
Path("workflows/krea2-glitch-shape.json").write_text(json.dumps(workflow, indent=1) + "\n")
print("wrote workflows/krea2-glitch-shape.json", len(nodes), "nodes", len(links), "links")
EOF
```
Expected: `wrote workflows/krea2-glitch-shape.json 13 nodes 13 links`.

- [ ] **Step 4: Run to verify pass**

Run: `$PYTEST tests/test_shape_workflow.py tests/test_random_workflow.py`
Expected: all PASS.

- [ ] **Step 5: Add the README section**

In `README.md`, insert this section immediately before the line `## Limits`:

````markdown
## Shaping the glitch

`Glitch Shape (Krea2)` is an optional companion node. Connect its `shape` output to the glitch node's
`shape` input; without it the glitch node behaves exactly as described above.

| Input | Meaning |
|---|---|
| `distribution` | Noise values: `gaussian`, `uniform`, `laplace` (occasional large values), `cauchy` (rare huge spikes, clipped at ±20), `spikes` (most values 0, a few big ones), `binary` (±1). Noise mode only |
| `spike_density` | Fraction of values that spike with `spikes` |
| `noise_scale` | Noise blob size in image tokens (1 token = 16×16 px). 1 = fine grain, 8 = large blobs. Noise mode only |
| `step_curve` | Strength multiplier across sampling steps (every mode) |
| `block_curve` | Strength multiplier across the 28 blocks (every mode) |
| `spatial_mask` | Strength multiplier over the picture (every mode): white = full glitch, black = untouched |

All noise distributions except `cauchy` have the same average size, so `strength` means the same across them.

**Curves.** Any node with a `FLOAT` value or list output works, for example KJNodes **Spline Editor**
(add it with a double-click search, then connect its `float` output). A curve is always stretched over the
whole run: its first point is step 0 (or block 0), its last point the final step (or block 27), with
straight lines in between, whatever `points_to_sample` is. Values above 1 boost the glitch; negative
values reverse it. The glitch node's step and block windows still limit where it acts, so open them fully
(`step_start` 0, `step_end` 999, `block_start` 0, `block_end` 27) when you let a curve do the shaping.

**Masks.** Any `MASK` works: KJNodes `CreateShapeMask`, `CreateGradientMask`, `CreateVoronoiMask`,
`CreateFluidMask`, or a mask painted with ComfyUI's mask editor on a `Load Image` node. The mask is
stretched to the image, so draw it at the image's aspect ratio. A mask with several frames plays across
the sampling steps (first frame at step 0, last frame at the final step).

`workflows/krea2-glitch-shape.json` is a ready example: the smoke workflow with `noise` mode shaped by
`spikes` (density 0.05, blob size 4) inside a centred circle from `CreateShapeMask`. It needs
comfyui-kjnodes.

````

- [ ] **Step 6: Run the whole suite**

Run: `$PYTEST tests`
Expected: all PASS, 0 skipped, no warnings summary.

- [ ] **Step 7: Commit**

```bash
git add tests/workflow_helpers.py tests/test_random_workflow.py tests/test_shape_workflow.py workflows/krea2-glitch-shape.json README.md
git commit -m "Add Glitch Shape example workflow, shared workflow test helpers and README section"
```

- [ ] **Step 8: Hand off the manual check to the user**

Do not touch ComfyUI. Tell the user: restart ComfyUI; open `workflows/krea2-glitch-shape.json`; queue once with the glitch node off and once on; the glitch should be confined to the centred circle; then try adding a KJNodes Spline Editor to `step_curve` as the README describes.
