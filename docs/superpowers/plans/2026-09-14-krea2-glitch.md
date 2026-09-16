# Krea2 Glitches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `Glitch Model (Krea2)` ComfyUI node, which applies reproducible, step-gated glitches to the image-token activations of a Krea2 model's attention/MLP outputs without touching weights or files.

**Architecture:** This folder is the custom-node package. Pure modules (`recipe`, `glitches`, `steps`, `runtime`) hold all logic and are unit-tested without ComfyUI. `node.py` clones the incoming MODEL and registers a keyed `DIFFUSION_MODEL` wrapper; per model call the wrapper computes the sampling step from sigmas, attaches forward hooks to the selected `blocks[i].attn` / `blocks[i].mlp`, runs the model, and removes the hooks in `finally`. Integration tests drive a tiny real Krea2 `SingleStreamDiT` inside a real `ModelPatcher` from the installed ComfyUI.

**Tech Stack:** Python 3.12.11, torch 2.9.1, ComfyUI 0.35.1 (`comfy.model_patcher`, `comfy.patcher_extension`, `comfy.ldm.krea2.model`), pytest (installed into `./.test-deps`), numpy/PIL (already in the ComfyUI venv).

**Spec:** `docs/superpowers/specs/2026-09-14-krea2-glitch-lab-design.md`

## Global Constraints

- Write only inside `/Users/wswi/Desktop/CLAUDE/ComfyUI-LesionLab` (the repo root, "this folder"). Other paths are read-only. Tests must not write outside this folder: do not use pytest's `tmp_path`/`tmpdir`; use `io.BytesIO`.
- Python for everything: `/Volumes/DATA/ComfyUI/.venv/bin/python` (Python 3.12.11, torch 2.9.1).
- Test command (run from the repo root): `PYTHONPATH=.test-deps PYTHONDONTWRITEBYTECODE=1 /Volumes/DATA/ComfyUI/.venv/bin/python -m pytest <paths> -v`. Below this is abbreviated as `$PYTEST <paths>`. Always pass tests or explicit test file paths; never run pytest without a path from the repo root (pytest would import the root __init__.py, which imports ComfyUI).
- ComfyUI code (read-only): `/Users/wswi/ComfyUI-Installs/ComfyUI/ComfyUI`, overridable with env `COMFYUI_ROOT`.
- Never write, rename or replace a checkpoint; never patch weights; never modify ComfyUI or other custom nodes.
- Inside the package use relative imports only (`from .recipe import …`). Only `glitches/node.py` imports `comfy`.
- Node: class/registry key `GlitchModelKrea2`, display name `Glitch Model (Krea2)`, category `experimental/glitches`, outputs `("MODEL", "STRING")` named `("model", "recipe")`, wrapper key `"glitches"`.
- Inputs, in order: `model`, `enabled` (True), `mode` (`dropout`/`amplify`/`sign_flip`/`noise`, default `noise`), `strength` (0.15, 0–10), `probability` (0.25, 0–1), `target` (`attention`/`mlp`/`both`, default `both`), `block_start` (0), `block_end` (27), `step_start` (0), `step_end` (999), `glitch_seed` (0, 0–2^63−1). No input may be named `seed` or `noise_seed`.
- Every commit message ends with these two lines:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01J7HDJFz4GpCtv6Nnr8Wqvk
  ```

---

## File Structure

| Path | Responsibility |
|---|---|
| `__init__.py` | ComfyUI registration (`NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`) |
| `glitches/__init__.py` | Package marker; imports nothing (keeps unit tests ComfyUI-free) |
| `glitches/recipe.py` | `GlitchRecipe` frozen dataclass, input validation, no-op reason, `describe()`, `validate_against_model()` |
| `glitches/effects.py` | `mix_seed`, `channel_count`, `select_channels`, `apply_glitch` |
| `glitches/steps.py` | `step_from_sigmas` |
| `glitches/runtime.py` | `image_token_count`, `GlitchWrapper` (hooks lifecycle) |
| `glitches/node.py` | `GlitchModelKrea2` ComfyUI node |
| `tests/pytest.ini` | pytest options (rootdir = tests/), filters torch's pynvml FutureWarning |
| `tests/conftest.py` | package on `sys.path`; `comfy_root` fixture |
| `tests/test_recipe.py`, `tests/test_effects.py`, `tests/test_steps.py`, `tests/test_runtime.py` | unit tests |
| `tests/test_node_integration.py` | tiny real Krea2 through real `ModelPatcher` |
| `tests/test_comfyui_loading.py` | import exactly like ComfyUI's `load_custom_node` |
| `tests/test_real_model_header.py` | tensor names of the real GGUF |
| `tests/test_compare_images.py`, `tests/test_smoke_workflow.py` | tooling checks |
| `tools/compare_images.py` | pixel comparison CLI |
| `workflows/krea2-glitch-smoke.json` | API-format smoke workflow |
| `README.md` | install, usage, experiment guide |

---

### Task 1: Test tooling and the glitch recipe

**Files:**
- Create: `pytest.ini`, `tests/conftest.py`, `glitches/__init__.py`, `glitches/recipe.py`
- Test: `tests/test_recipe.py`

**Interfaces:**
- Consumes: nothing.
- Produces (in `glitches/recipe.py`):
  - `MODES = ("dropout", "amplify", "sign_flip", "noise")`, `TARGETS = ("attention", "mlp", "both")`, `FAMILIES = ("attention", "mlp")`, `STRENGTH_MAX: dict[str, float]`, `SEED_MAX = 2**63 - 1`
  - `GlitchRecipe.build(*, enabled, mode, strength, probability, target, block_start, block_end, step_start, step_end, glitch_seed) -> GlitchRecipe` (keyword-only)
  - fields `enabled: bool, mode: str, strength: float, probability: float, target: str, block_start: int, block_end: int, step_start: int, step_end: int, glitch_seed: int`
  - properties `families -> tuple[str, ...]`, `blocks -> range`, `site_count -> int`
  - methods `noop_reason() -> str | None`, `step_active(step: int) -> bool`, `describe() -> str`
  - `validate_against_model(recipe: GlitchRecipe, n_blocks: int) -> None`
- Fixture in `tests/conftest.py`: `comfy_root` (session) → `pathlib.Path`, skips if ComfyUI is absent.

- [ ] **Step 1: Install pytest into this folder**

Run:
```bash
PIP_DISABLE_PIP_VERSION_CHECK=1 /Volumes/DATA/ComfyUI/.venv/bin/python -m pip install --no-cache-dir --target ./.test-deps pytest
```
Expected: `Successfully installed … pytest-…`. `.test-deps/` is already in `.gitignore`.

- [ ] **Step 2: Create `pytest.ini` and `tests/conftest.py`**

`pytest.ini`:
```ini
[pytest]
testpaths = tests
addopts = -p no:cacheprovider
```

`tests/conftest.py`:
```python
import os
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", "/Users/wswi/ComfyUI-Installs/ComfyUI/ComfyUI"))

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture(scope="session")
def comfy_root():
    """Put the installed ComfyUI on sys.path (read-only use) or skip the test."""
    if not (COMFYUI_ROOT / "comfy" / "model_patcher.py").is_file():
        pytest.skip(f"ComfyUI not found at {COMFYUI_ROOT}; set COMFYUI_ROOT")
    if str(COMFYUI_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFYUI_ROOT))
    return COMFYUI_ROOT
```

`glitches/__init__.py`:
```python
"""Core logic for ComfyUI-LesionLab. Import submodules directly; this package imports nothing."""
```

- [ ] **Step 3: Write the failing recipe tests**

`tests/test_recipe.py`:
```python
import pytest

from glitches.recipe import GlitchRecipe, validate_against_model

DEFAULTS = dict(
    enabled=True, mode="noise", strength=0.15, probability=0.25, target="both",
    block_start=0, block_end=27, step_start=0, step_end=999, glitch_seed=0,
)


def build(**overrides):
    return GlitchRecipe.build(**{**DEFAULTS, **overrides})


def test_defaults_build_an_active_recipe():
    recipe = build()
    assert recipe.noop_reason() is None
    assert recipe.families == ("attention", "mlp")
    assert list(recipe.blocks) == list(range(28))
    assert recipe.site_count == 56


def test_single_family_target_counts_sites():
    recipe = build(target="mlp", block_start=2, block_end=5)
    assert recipe.families == ("mlp",)
    assert recipe.site_count == 4


def test_step_active_is_inclusive():
    recipe = build(step_start=2, step_end=4)
    assert [recipe.step_active(s) for s in range(6)] == [False, False, True, True, True, False]


def test_describe_active_recipe():
    assert build().describe() == (
        "krea2-glitch v1 | mode=noise strength=0.15 probability=0.25 | "
        "target=both blocks=0-27 sites=56 | steps=0-999 | tokens=image | glitch_seed=0"
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [({"enabled": False}, "disabled"), ({"strength": 0.0}, "strength=0"), ({"probability": 0.0}, "probability=0")],
)
def test_noop_reasons(overrides, reason):
    recipe = build(**overrides)
    assert recipe.noop_reason() == reason
    assert recipe.describe().startswith(f"no-op ({reason}) | krea2-glitch v1 | ")


@pytest.mark.parametrize(("mode", "limit"), [("dropout", 1), ("sign_flip", 1), ("amplify", 10), ("noise", 10)])
def test_strength_limit_per_mode(mode, limit):
    assert build(mode=mode, strength=limit).strength == limit
    with pytest.raises(ValueError, match=f"strength for mode {mode} must be between 0 and {limit}"):
        build(mode=mode, strength=limit + 0.01)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mode": "scale"}, "mode must be one of dropout, amplify, sign_flip, noise; got 'scale'"),
        ({"target": "ff"}, "target must be one of attention, mlp, both; got 'ff'"),
        ({"strength": float("nan")}, "strength must be a finite number"),
        ({"strength": -0.1}, "strength for mode noise must be between 0 and 10"),
        ({"probability": 1.5}, "probability must be between 0 and 1"),
        ({"probability": "x"}, "probability must be a finite number"),
        ({"block_start": -1}, "block_start must be >= 0"),
        ({"block_start": 5, "block_end": 4}, r"block_end \(4\) must be >= block_start \(5\)"),
        ({"step_start": 3, "step_end": 2}, r"step_end \(2\) must be >= step_start \(3\)"),
        ({"step_end": 2.5}, "step_end must be an integer"),
        ({"glitch_seed": -1}, "glitch_seed must be between 0 and 9223372036854775807"),
    ],
)
def test_validation_messages(overrides, message):
    with pytest.raises(ValueError, match=message):
        build(**overrides)


def test_validate_against_model():
    validate_against_model(build(block_end=27), n_blocks=28)
    with pytest.raises(ValueError, match="block_end 28 exceeds last block 27"):
        validate_against_model(build(block_end=28), n_blocks=28)
```

- [ ] **Step 4: Run to verify failure**

Run: `$PYTEST tests/test_recipe.py`
Expected: collection error `ModuleNotFoundError: No module named 'glitches.recipe'`.

- [ ] **Step 5: Implement `glitches/recipe.py`**

```python
"""Validated, immutable description of one glitch experiment (no torch, no ComfyUI)."""

from __future__ import annotations

import math
from dataclasses import dataclass

MODES = ("dropout", "amplify", "sign_flip", "noise")
TARGETS = ("attention", "mlp", "both")
FAMILIES = ("attention", "mlp")
STRENGTH_MAX = {"dropout": 1.0, "amplify": 10.0, "sign_flip": 1.0, "noise": 10.0}
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
        limit = STRENGTH_MAX[mode]
        if not 0.0 <= strength_value <= limit:
            raise ValueError(f"strength for mode {mode} must be between 0 and {limit:g}; got {strength_value:g}")
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


def validate_against_model(recipe: GlitchRecipe, n_blocks: int) -> None:
    if recipe.block_end > n_blocks - 1:
        raise ValueError(f"block_end {recipe.block_end} exceeds last block {n_blocks - 1}")
```

- [ ] **Step 6: Run to verify pass**

Run: `$PYTEST tests/test_recipe.py`
Expected: all tests PASS (23 passed).

- [ ] **Step 7: Commit**

```bash
git add pytest.ini tests/conftest.py tests/test_recipe.py glitches/__init__.py glitches/recipe.py
git commit -m "Add validated glitch recipe and test tooling" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J7HDJFz4GpCtv6Nnr8Wqvk"
```

---

### Task 2: Channel selection and glitch transforms

**Files:**
- Create: `glitches/effects.py`
- Test: `tests/test_effects.py`

**Interfaces:**
- Consumes: `GlitchRecipe` (fields `mode`, `strength`, `probability`, `glitch_seed`) from Task 1.
- Produces (in `glitches/effects.py`):
  - `FAMILY_IDS = {"attention": 0, "mlp": 1}`, `PURPOSE_SELECT = 0`, `PURPOSE_NOISE = 1`
  - `mix_seed(glitch_seed: int, block: int, family: str, step: int, purpose: int) -> int` (0 ≤ result < 2^63)
  - `channel_count(probability: float, width: int) -> int`
  - `select_channels(recipe, block: int, family: str, step: int, width: int) -> torch.LongTensor` (CPU, length `channel_count`)
  - `apply_glitch(out: Tensor[B, L, D], recipe, block: int, family: str, step: int, txt: int, n_img: int) -> Tensor` (new tensor, same dtype/device)

- [ ] **Step 1: Write the failing tests**

`tests/test_effects.py`:
```python
import pytest
import torch

from glitches.effects import apply_glitch, channel_count, mix_seed, select_channels
from glitches.recipe import GlitchRecipe

TXT, N_IMG, REF, WIDTH = 3, 4, 2, 20
IMG = slice(TXT, TXT + N_IMG)


def recipe(mode="dropout", strength=1.0, probability=0.25, seed=7):
    return GlitchRecipe.build(
        enabled=True, mode=mode, strength=strength, probability=probability, target="both",
        block_start=0, block_end=27, step_start=0, step_end=999, glitch_seed=seed,
    )


def activations(batch=2, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, TXT + N_IMG + REF, WIDTH, generator=generator).to(dtype)


def glitch(out, r, block=3, family="mlp", step=2):
    return apply_glitch(out, r, block, family, step, TXT, N_IMG)


def assert_text_and_reference_rows_untouched(before, after):
    assert torch.equal(after[:, :TXT], before[:, :TXT])
    assert torch.equal(after[:, TXT + N_IMG:], before[:, TXT + N_IMG:])


def test_mix_seed_golden_values_never_change():
    assert mix_seed(0, 0, "attention", 0, 0) == 8695987549771912286
    assert mix_seed(7, 3, "mlp", 2, 1) == 5231173937662915042


def test_mix_seed_changes_with_every_component():
    base = mix_seed(7, 3, "mlp", 2, 0)
    others = [
        mix_seed(8, 3, "mlp", 2, 0), mix_seed(7, 4, "mlp", 2, 0), mix_seed(7, 3, "attention", 2, 0),
        mix_seed(7, 3, "mlp", 3, 0), mix_seed(7, 3, "mlp", 2, 1),
    ]
    assert base not in others


@pytest.mark.parametrize(("probability", "expected"), [(0.25, 5), (0.01, 1), (0.5, 10), (1.0, 20)])
def test_channel_count(probability, expected):
    assert channel_count(probability, WIDTH) == expected


def test_select_channels_is_deterministic_distinct_and_site_specific():
    r = recipe()
    first = select_channels(r, 3, "mlp", 2, WIDTH)
    assert torch.equal(first, select_channels(r, 3, "mlp", 2, WIDTH))
    assert first.numel() == 5 and first.unique().numel() == 5
    others = [
        select_channels(r, 4, "mlp", 2, WIDTH), select_channels(r, 3, "attention", 2, WIDTH),
        select_channels(r, 3, "mlp", 3, WIDTH), select_channels(recipe(seed=8), 3, "mlp", 2, WIDTH),
    ]
    assert all(not torch.equal(first, other) for other in others)


@pytest.mark.parametrize(
    ("mode", "strength", "factor"),
    [("dropout", 1.0, 0.0), ("dropout", 0.25, 0.75), ("amplify", 1.0, 2.0), ("sign_flip", 1.0, -1.0), ("sign_flip", 0.5, 0.0)],
)
def test_scaling_modes_apply_their_formula_to_selected_image_channels(mode, strength, factor):
    out = activations()
    r = recipe(mode, strength)
    result = glitch(out, r)
    selected = select_channels(r, 3, "mlp", 2, WIDTH)
    torch.testing.assert_close(result[:, IMG][..., selected], out[:, IMG][..., selected] * factor)
    unselected = torch.ones(WIDTH, dtype=torch.bool)
    unselected[selected] = False
    assert torch.equal(result[:, IMG][..., unselected], out[:, IMG][..., unselected])
    assert_text_and_reference_rows_untouched(out, result)


def test_noise_is_rms_scaled_selected_only_and_shared_across_batch():
    out = activations()
    r = recipe("noise", 0.5)
    result = glitch(out, r)
    selected = select_channels(r, 3, "mlp", 2, WIDTH)
    changed = torch.nonzero((result != out).any(dim=0).any(dim=0)).flatten()
    assert changed.tolist() == sorted(selected.tolist())
    assert_text_and_reference_rows_untouched(out, result)
    image = out[:, IMG]
    rms = image.pow(2).mean(dim=-1, keepdim=True).sqrt()
    unit_noise = (result[:, IMG][..., selected] - image[..., selected]) / (0.5 * rms)
    torch.testing.assert_close(unit_noise[0], unit_noise[1])


def test_noise_is_repeatable_and_step_specific():
    out = activations()
    r = recipe("noise", 0.5)
    assert torch.equal(glitch(out, r, step=2), glitch(out, r, step=2))
    assert not torch.equal(glitch(out, r, step=2), glitch(out, r, step=3))


@pytest.mark.parametrize("mode", ["dropout", "amplify", "sign_flip", "noise"])
def test_result_per_item_does_not_depend_on_batch_size(mode):
    out = activations(batch=2)
    r = recipe(mode, 0.5)
    torch.testing.assert_close(glitch(out, r)[:1], glitch(out[:1], r), rtol=0, atol=1e-6)


def test_input_is_not_mutated():
    out = activations()
    snapshot = out.clone()
    glitch(out, recipe("noise", 0.5))
    glitch(out, recipe("sign_flip", 1.0))
    assert torch.equal(out, snapshot)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["dropout", "noise"])
def test_dtype_is_preserved(dtype, mode):
    result = glitch(activations(dtype=dtype), recipe(mode, 0.5))
    assert result.dtype == dtype
    assert torch.isfinite(result.float()).all()


def test_full_probability_dropout_silences_every_image_channel():
    out = activations()
    result = glitch(out, recipe("dropout", 1.0, probability=1.0))
    assert torch.count_nonzero(result[:, IMG]) == 0
    assert_text_and_reference_rows_untouched(out, result)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PYTEST tests/test_effects.py`
Expected: `ModuleNotFoundError: No module named 'glitches.glitches'`.

- [ ] **Step 3: Implement `glitches/effects.py`**

```python
"""Deterministic channel selection and activation transforms (torch only)."""

from __future__ import annotations

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
                 txt: int, n_img: int) -> torch.Tensor:
    """Return a copy of ``out`` [B, L, D] with the recipe applied to rows ``txt:txt+n_img``."""
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
        rms = region.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        generator = torch.Generator(device=out.device)
        generator.manual_seed(mix_seed(recipe.glitch_seed, block, family, step, PURPOSE_NOISE))
        noise = torch.randn((n_img, selected.numel()), generator=generator, device=out.device, dtype=torch.float32)
        glitched = values.float() + strength * rms * noise
    else:
        raise ValueError(f"unsupported glitch mode: {recipe.mode!r}")

    result = out.clone()
    result[:, rows, :].index_copy_(2, selected, glitched.to(out.dtype))
    return result
```

- [ ] **Step 4: Run to verify pass**

Run: `$PYTEST tests/test_effects.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add glitches/effects.py tests/test_effects.py
git commit -m "Add deterministic channel selection and glitch transforms" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J7HDJFz4GpCtv6Nnr8Wqvk"
```

---

### Task 3: Step index from sigmas

**Files:**
- Create: `glitches/steps.py`
- Test: `tests/test_steps.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `step_from_sigmas(sigmas: torch.Tensor, sample_sigmas: torch.Tensor) -> int` in `glitches/steps.py`. Raises `ValueError("sample_sigmas must contain at least 2 values")`.

- [ ] **Step 1: Write the failing tests**

`tests/test_steps.py`:
```python
import pytest
import torch

from glitches.steps import step_from_sigmas

SCHEDULE = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])


@pytest.mark.parametrize(
    ("sigma", "step"),
    [(1.0, 0), (0.75, 1), (0.5, 2), (0.25, 3), (0.9, 0), (0.6, 1), (0.3, 2), (0.1, 3), (1.5, 0), (0.0, 3)],
)
def test_schedule_points_and_in_between_sigmas(sigma, step):
    assert step_from_sigmas(torch.tensor([sigma, sigma]), SCHEDULE) == step


def test_float_error_just_above_a_schedule_point_counts_as_that_step():
    assert step_from_sigmas(torch.tensor([0.5 + 1e-8], dtype=torch.float64), SCHEDULE.double()) == 2


def test_uses_largest_sigma_in_the_batch():
    assert step_from_sigmas(torch.tensor([0.5, 0.75]), SCHEDULE) == 1


def test_shortened_schedule_counts_from_its_first_entry():
    assert step_from_sigmas(torch.tensor([0.5]), torch.tensor([0.5, 0.25, 0.0])) == 0


def test_single_step_schedule():
    assert step_from_sigmas(torch.tensor([1.0]), torch.tensor([1.0, 0.0])) == 0


def test_schedule_too_short():
    with pytest.raises(ValueError, match="at least 2 values"):
        step_from_sigmas(torch.tensor([1.0]), torch.tensor([1.0]))
```

- [ ] **Step 2: Run to verify failure**

Run: `$PYTEST tests/test_steps.py`
Expected: `ModuleNotFoundError: No module named 'glitches.steps'`.

- [ ] **Step 3: Implement `glitches/steps.py`**

```python
"""Map ComfyUI's per-call sigma to a sampling step index (torch only)."""

from __future__ import annotations

import torch


def step_from_sigmas(sigmas: torch.Tensor, sample_sigmas: torch.Tensor) -> int:
    """Index of the schedule interval containing the current sigma.

    ``sample_sigmas`` is the descending schedule of N+1 values that the sampler runs; a call at
    ``sample_sigmas[i]`` is step ``i`` and calls between two entries belong to the earlier one.
    """
    schedule = [float(value) for value in torch.as_tensor(sample_sigmas).detach().flatten().cpu()]
    if len(schedule) < 2:
        raise ValueError("sample_sigmas must contain at least 2 values")
    sigma = float(torch.as_tensor(sigmas).detach().max().cpu())
    n_steps = len(schedule) - 1
    eps = 1e-6 * max(schedule[0], 1.0)
    step = sum(1 for value in schedule[1:n_steps] if value >= sigma - eps)
    return min(max(step, 0), n_steps - 1)
```

- [ ] **Step 4: Run to verify pass**

Run: `$PYTEST tests/test_steps.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add glitches/steps.py tests/test_steps.py
git commit -m "Add sampling step lookup from sigmas" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J7HDJFz4GpCtv6Nnr8Wqvk"
```

---

### Task 4: Glitch wrapper and hook lifecycle

**Files:**
- Create: `glitches/runtime.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `GlitchRecipe.blocks`, `.families`, `.step_active()` (Task 1); `apply_glitch` (Task 2); `step_from_sigmas` (Task 3).
- Produces (in `glitches/runtime.py`):
  - `image_token_count(x: torch.Tensor, patch: int) -> int` (raises `ValueError` mentioning `image latents` for non-4-D input)
  - `class GlitchWrapper(recipe)` with `__call__(executor, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)`. `executor.class_obj` must expose `.patch: int` and `.blocks[i].attn` / `.blocks[i].mlp` modules; `executor(...)` is called with the same positional arguments.
  - `MISSING_SIGMAS` error text containing `sample_sigmas`.

- [ ] **Step 1: Write the failing tests**

`tests/test_runtime.py`:
```python
import pytest
import torch
from torch import nn

from glitches.recipe import GlitchRecipe
from glitches.runtime import GlitchWrapper, image_token_count

SCHEDULE = torch.tensor([1.0, 0.5, 0.0])
TXT = 3
WIDTH = 8


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, x):
        return self.mlp(self.attn(x))


class FakeDiT(nn.Module):
    patch = 2

    def __init__(self, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(n_blocks))


def hook_counts(dit):
    return [len(b.attn._forward_hooks) + len(b.mlp._forward_hooks) for b in dit.blocks]


class FakeExecutor:
    """Stands in for comfy.patcher_extension.WrapperExecutor: runs all blocks over [B, txt + img, D] ones."""

    def __init__(self, dit):
        self.class_obj = dit
        self.seen_hooks = None

    def sequence_length(self, x, context):
        return context.shape[1] + (x.shape[-2] // 2) * (x.shape[-1] // 2)

    def __call__(self, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs):
        self.seen_hooks = hook_counts(self.class_obj)
        seq = torch.ones(x.shape[0], self.sequence_length(x, context), WIDTH)
        for block in self.class_obj.blocks:
            seq = block(seq)
        return seq


def recipe(**overrides):
    args = dict(enabled=True, mode="dropout", strength=1.0, probability=1.0, target="both",
                block_start=0, block_end=1, step_start=1, step_end=1, glitch_seed=0)
    args.update(overrides)
    return GlitchRecipe.build(**args)


def call(wrapper, executor, sigma, options=None):
    x = torch.zeros(1, 4, 4, 6)
    context = torch.zeros(1, TXT, 12)
    if options is None:
        options = {"sigmas": torch.tensor([sigma]), "sample_sigmas": SCHEDULE}
    return wrapper(executor, x, torch.tensor([sigma]), context, None, None, options)


def test_image_token_count_rounds_up_odd_sizes():
    assert image_token_count(torch.zeros(1, 4, 5, 6), 2) == 9


def test_video_latents_are_rejected():
    with pytest.raises(ValueError, match="image latents"):
        image_token_count(torch.zeros(1, 4, 2, 4, 6), 2)


def test_outside_step_window_runs_untouched_without_hooks():
    executor = FakeExecutor(FakeDiT())
    out = call(GlitchWrapper(recipe()), executor, sigma=1.0)
    assert executor.seen_hooks == [0, 0]
    assert torch.all(out == 1)


def test_inside_window_hooks_only_selected_sites_glitches_image_rows_and_cleans_up():
    executor = FakeExecutor(FakeDiT())
    out = call(GlitchWrapper(recipe(target="mlp", block_start=1, block_end=1)), executor, sigma=0.5)
    assert executor.seen_hooks == [0, 1]
    assert torch.all(out[:, :TXT] == 1)
    assert torch.all(out[:, TXT:] == 0)
    assert hook_counts(executor.class_obj) == [0, 0]


def test_hooks_are_removed_when_the_model_raises():
    class Boom(FakeExecutor):
        def __call__(self, *args, **kwargs):
            super().__call__(*args, **kwargs)
            raise RuntimeError("boom")

    executor = Boom(FakeDiT())
    with pytest.raises(RuntimeError, match="boom"):
        call(GlitchWrapper(recipe()), executor, sigma=0.5)
    assert executor.seen_hooks == [2, 2]
    assert hook_counts(executor.class_obj) == [0, 0]


def test_missing_sample_sigmas_is_a_clear_error():
    with pytest.raises(RuntimeError, match="sample_sigmas"):
        call(GlitchWrapper(recipe()), FakeExecutor(FakeDiT()), sigma=0.5, options={"sigmas": torch.tensor([0.5])})


def test_activation_shorter_than_image_slice_is_a_clear_error():
    class Short(FakeExecutor):
        def sequence_length(self, x, context):
            return 2

    executor = Short(FakeDiT())
    with pytest.raises(RuntimeError, match="unexpected attention output at block 0"):
        call(GlitchWrapper(recipe()), executor, sigma=0.5)
    assert hook_counts(executor.class_obj) == [0, 0]
```

- [ ] **Step 2: Run to verify failure**

Run: `$PYTEST tests/test_runtime.py`
Expected: `ModuleNotFoundError: No module named 'glitches.runtime'`.

- [ ] **Step 3: Implement `glitches/runtime.py`**

```python
"""Per-call glitch wrapper for Krea2's DIFFUSION_MODEL wrapper slot (torch only, duck-typed).

ComfyUI 0.35.1 calls DIFFUSION_MODEL wrappers from SingleStreamDiT.forward as
``wrapper(executor, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)``
with ``executor.class_obj`` set to the SingleStreamDiT instance.
"""

from __future__ import annotations

import math

import torch

from .effects import apply_glitch
from .recipe import GlitchRecipe
from .steps import step_from_sigmas

MISSING_SIGMAS = (
    "Glitch Model (Krea2): sampler did not provide sample_sigmas; "
    "use KSampler, KSamplerAdvanced or SamplerCustom"
)


def image_token_count(x: torch.Tensor, patch: int) -> int:
    if x.ndim != 4:
        raise ValueError(f"Glitch Model (Krea2) supports image latents [B, C, H, W] only; got {x.ndim}-D input")
    return math.ceil(x.shape[-2] / patch) * math.ceil(x.shape[-1] / patch)


def _make_hook(recipe: GlitchRecipe, block: int, family: str, step: int, txt: int, n_img: int):
    def hook(module, args, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[1] < txt + n_img:
            got = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
            raise RuntimeError(
                f"Glitch Model (Krea2): unexpected {family} output at block {block}: "
                f"expected [B, >= {txt + n_img}, D], got {got}"
            )
        return apply_glitch(output, recipe, block, family, step, txt, n_img)

    return hook


class GlitchWrapper:
    def __init__(self, recipe: GlitchRecipe):
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
```

- [ ] **Step 4: Run to verify pass**

Run: `$PYTEST tests/test_runtime.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add glitches/runtime.py tests/test_runtime.py
git commit -m "Add step-gated glitch wrapper with scoped forward hooks" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J7HDJFz4GpCtv6Nnr8Wqvk"
```

---

### Task 5: ComfyUI node, registration and integration tests

**Files:**
- Create: `glitches/node.py`, `__init__.py`
- Test: `tests/test_node_integration.py`, `tests/test_comfyui_loading.py`, `tests/test_real_model_header.py`

**Interfaces:**
- Consumes: `GlitchRecipe.build`, `noop_reason`, `describe`, `validate_against_model`, `MODES`, `TARGETS`, `SEED_MAX` (Task 1); `GlitchWrapper` (Task 4); fixture `comfy_root` (Task 1). ComfyUI (read-only): `comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL` (`"diffusion_model"`), `ModelPatcher.clone()`, `ModelPatcher.add_wrapper_with_key(type, key, fn)`, `ModelPatcher.get_wrappers(type, key)`, `comfy.ldm.krea2.model.SingleStreamDiT`.
- Produces: `glitches.node.GlitchModelKrea2` with `INPUT_TYPES()`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION = "apply"`, `CATEGORY`, `DESCRIPTION`, and `apply(model, enabled, mode, strength, probability, target, block_start, block_end, step_start, step_end, glitch_seed) -> tuple[ModelPatcher, str]`; `glitches.node.WRAPPER_KEY = "glitches"`; root `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`.

- [ ] **Step 1: Write the failing integration tests**

`tests/test_node_integration.py`:
```python
import pytest
import torch

SCHEDULE = [1.0, 0.75, 0.5, 0.25, 0.0]


@pytest.fixture(scope="module")
def comfy(comfy_root):
    import comfy.ldm.krea2.model as krea2_model
    import comfy.model_patcher as model_patcher
    import comfy.patcher_extension as patcher_extension

    return krea2_model, model_patcher, patcher_extension


@pytest.fixture
def node(comfy):
    from glitches.node import GlitchModelKrea2

    return GlitchModelKrea2()


def make_patcher(comfy, diffusion_model):
    _, model_patcher, _ = comfy
    holder = torch.nn.Module()
    holder.diffusion_model = diffusion_model
    return model_patcher.ModelPatcher(holder, torch.device("cpu"), torch.device("cpu"))


@pytest.fixture
def patcher(comfy):
    krea2_model = comfy[0]
    torch.manual_seed(0)
    dit = krea2_model.SingleStreamDiT(
        features=64, tdim=32, txtdim=32, heads=4, kvheads=2, multiplier=2, layers=2, patch=2, channels=4,
        txtlayers=3, txtheads=4, txtkvheads=4, operations=torch.nn,
    )
    for parameter in dit.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return make_patcher(comfy, dit.eval())


def apply(node, model, **overrides):
    args = dict(enabled=True, mode="dropout", strength=1.0, probability=0.5, target="both",
                block_start=0, block_end=1, step_start=1, step_end=2, glitch_seed=7)
    args.update(overrides)
    return node.apply(model, **args)


def run(model, sigma, ref=False, without=()):
    """Call the diffusion model the way ComfyUI's sampler does, with model.wrappers in transformer_options."""
    dit = model.model.diffusion_model
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(2, 4, 8, 6, generator=generator)
    context = torch.randn(2, 5, 3 * 32, generator=generator)
    sigmas = torch.full((2,), sigma)
    options = {"wrappers": model.wrappers, "sigmas": sigmas, "sample_sigmas": torch.tensor(SCHEDULE)}
    for key in without:
        options.pop(key)
    kwargs = {}
    if ref:
        kwargs = {"ref_latents": [torch.randn(1, 4, 8, 6, generator=generator)], "ref_latents_method": "index_timestep_zero"}
    with torch.no_grad():
        return dit(x, sigmas, context, transformer_options=options, **kwargs)


def hook_count(model):
    return sum(len(b.attn._forward_hooks) + len(b.mlp._forward_hooks) for b in model.model.diffusion_model.blocks)


def glitch_wrappers(comfy, model):
    return model.get_wrappers(comfy[2].WrappersMP.DIFFUSION_MODEL, "glitches")


def test_rejects_non_krea2_model(comfy, node):
    with pytest.raises(ValueError, match="requires a Krea2 model; got Linear"):
        apply(node, make_patcher(comfy, torch.nn.Linear(2, 2)))


def test_block_end_beyond_the_model_is_rejected(node, patcher):
    with pytest.raises(ValueError, match="block_end 2 exceeds last block 1"):
        apply(node, patcher, block_end=2)


def test_clone_carries_the_wrapper_and_source_stays_clean(comfy, node, patcher):
    clone, recipe = apply(node, patcher)
    assert clone is not patcher
    assert glitch_wrappers(comfy, patcher) == []
    assert len(glitch_wrappers(comfy, clone)) == 1
    assert "sites=4" in recipe and recipe.startswith("krea2-glitch v1 | mode=dropout")


def test_noop_attaches_no_wrapper(comfy, node, patcher):
    clone, recipe = apply(node, patcher, enabled=False)
    assert recipe.startswith("no-op (disabled) | ")
    assert glitch_wrappers(comfy, clone) == []


def test_output_changes_only_inside_the_step_window(node, patcher):
    clone, _ = apply(node, patcher)
    for sigma, inside in [(1.0, False), (0.75, True), (0.5, True), (0.25, False)]:
        base = run(patcher, sigma)
        glitched = run(clone, sigma)
        assert (not torch.equal(base, glitched)) == inside, f"sigma={sigma}"


def test_hooks_are_removed_after_success_and_after_an_exception(comfy, node, patcher):
    clone, _ = apply(node, patcher)
    run(clone, 0.75)
    assert hook_count(clone) == 0

    seen = []

    def boom(executor, *args, **kwargs):
        seen.append(hook_count(clone))
        raise RuntimeError("boom")

    clone.add_wrapper_with_key(comfy[2].WrappersMP.DIFFUSION_MODEL, "boom", boom)
    with pytest.raises(RuntimeError, match="boom"):
        run(clone, 0.75)
    assert seen == [4]
    assert hook_count(clone) == 0


def test_chained_nodes_stack(comfy, node, patcher):
    first, _ = apply(node, patcher, target="attention", block_end=0, glitch_seed=1)
    second, _ = apply(node, first, mode="amplify", strength=2.0, target="mlp", block_start=1, glitch_seed=2)
    assert len(glitch_wrappers(comfy, first)) == 1
    assert len(glitch_wrappers(comfy, second)) == 2
    base, one, two = run(patcher, 0.75), run(first, 0.75), run(second, 0.75)
    assert not torch.equal(base, one)
    assert not torch.equal(one, two)


def test_reference_latent_path(node, patcher):
    clone, _ = apply(node, patcher)
    base = run(patcher, 0.75, ref=True)
    glitched = run(clone, 0.75, ref=True)
    assert glitched.shape == base.shape == (2, 4, 8, 6)
    assert torch.isfinite(glitched).all()
    assert not torch.equal(base, glitched)


def test_missing_sample_sigmas_raises(node, patcher):
    clone, _ = apply(node, patcher)
    with pytest.raises(RuntimeError, match="sample_sigmas"):
        run(clone, 0.75, without=("sample_sigmas",))
    assert hook_count(clone) == 0
```

`tests/test_comfyui_loading.py`:
```python
import importlib.util
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUTS = [
    "model", "enabled", "mode", "strength", "probability", "target",
    "block_start", "block_end", "step_start", "step_end", "glitch_seed",
]


def test_package_imports_the_way_comfyui_loads_custom_nodes(comfy_root):
    # Mirrors nodes.load_custom_node in ComfyUI 0.35.1 for a directory module.
    module_path = str(PACKAGE_ROOT)
    sys_module_name = module_path.replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(sys_module_name, os.path.join(module_path, "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[sys_module_name] = module
    try:
        spec.loader.exec_module(module)
        node_class = module.NODE_CLASS_MAPPINGS["GlitchModelKrea2"]
        assert module.NODE_DISPLAY_NAME_MAPPINGS["GlitchModelKrea2"] == "Glitch Model (Krea2)"
        assert node_class.CATEGORY == "experimental/glitches"
        assert node_class.RETURN_TYPES == ("MODEL", "STRING")
        assert node_class.RETURN_NAMES == ("model", "recipe")
        required = node_class.INPUT_TYPES()["required"]
        assert list(required) == EXPECTED_INPUTS
        assert required["mode"][0] == ["dropout", "amplify", "sign_flip", "noise"]
        assert required["mode"][1]["default"] == "noise"
        assert required["glitch_seed"][1]["max"] == 2**63 - 1
    finally:
        for name in [n for n in sys.modules if n == sys_module_name or n.startswith(sys_module_name + ".")]:
            del sys.modules[name]
```

`tests/test_real_model_header.py`:
```python
import os
from pathlib import Path

import pytest

GGUF_PATH = Path(os.environ.get("KREA2_GGUF", "/Volumes/DATA/ComfyUI/models/unet/KREA/museByStableYogi_v25GGUF.gguf"))


def test_real_krea2_gguf_exposes_the_glitch_sites():
    if not GGUF_PATH.is_file():
        pytest.skip(f"{GGUF_PATH} not found; set KREA2_GGUF")
    gguf = pytest.importorskip("gguf")
    names = {tensor.name for tensor in gguf.GGUFReader(str(GGUF_PATH)).tensors}
    assert {int(name.split(".")[1]) for name in names if name.startswith("blocks.")} == set(range(28))
    for block in range(28):
        for family in ("attn", "mlp"):
            assert any(name.startswith(f"blocks.{block}.{family}.") for name in names), f"blocks.{block}.{family}"
```

- [ ] **Step 2: Run to verify failure**

Run: `$PYTEST tests/test_node_integration.py tests/test_comfyui_loading.py tests/test_real_model_header.py`
Expected: node and loading tests FAIL (`ModuleNotFoundError: No module named 'glitches.node'` / `FileNotFoundError` for `__init__.py`); the header test PASSES already (it only reads the real file). If the node tests are SKIPPED instead, ComfyUI was not found: check `COMFYUI_ROOT` before continuing.

- [ ] **Step 3: Implement `glitches/node.py`**

```python
"""ComfyUI node: Glitch Model (Krea2). The only module in this package that imports ComfyUI."""

from __future__ import annotations

import comfy.patcher_extension
from comfy.ldm.krea2.model import SingleStreamDiT

from .recipe import MODES, SEED_MAX, TARGETS, GlitchRecipe, validate_against_model
from .runtime import GlitchWrapper

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
                "strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01}),
                "probability": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target": (list(TARGETS), {"default": "both"}),
                "block_start": ("INT", {"default": 0, "min": 0, "max": 999}),
                "block_end": ("INT", {"default": 27, "min": 0, "max": 999}),
                "step_start": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "step_end": ("INT", {"default": 999, "min": 0, "max": 10000}),
                "glitch_seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
            }
        }

    def apply(self, model, enabled, mode, strength, probability, target,
              block_start, block_end, step_start, step_end, glitch_seed):
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
        if not isinstance(diffusion_model, SingleStreamDiT):
            raise ValueError(
                f"Glitch Model (Krea2) requires a Krea2 model; got {type(diffusion_model).__name__}"
            )
        recipe = GlitchRecipe.build(
            enabled=enabled, mode=mode, strength=strength, probability=probability, target=target,
            block_start=block_start, block_end=block_end, step_start=step_start, step_end=step_end,
            glitch_seed=glitch_seed,
        )
        validate_against_model(recipe, n_blocks=len(diffusion_model.blocks))
        clone = model.clone()
        if recipe.noop_reason() is None:
            clone.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, GlitchWrapper(recipe)
            )
        return (clone, recipe.describe())
```

- [ ] **Step 4: Implement root `__init__.py`**

```python
"""ComfyUI-LesionLab: reproducible activation glitches for Krea2 models."""

from .glitches.node import GlitchModelKrea2

NODE_CLASS_MAPPINGS = {"GlitchModelKrea2": GlitchModelKrea2}
NODE_DISPLAY_NAME_MAPPINGS = {"GlitchModelKrea2": "Glitch Model (Krea2)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

- [ ] **Step 5: Run to verify pass**

Run: `$PYTEST tests/test_node_integration.py tests/test_comfyui_loading.py tests/test_real_model_header.py`
Expected: all PASS, none skipped. Lines such as `Exception ignored in: <function ModelPatcher.__del__ …> AttributeError: 'NoneType' object has no attribute 'ON_DETACH'` printed at interpreter exit are ComfyUI teardown noise, not failures.

- [ ] **Step 6: Run the whole suite**

Run: `$PYTEST tests`
Expected: all PASS, 0 skipped.

- [ ] **Step 7: Commit**

```bash
git add __init__.py glitches/node.py tests/test_node_integration.py tests/test_comfyui_loading.py tests/test_real_model_header.py
git commit -m "Add Glitch Model (Krea2) node with ComfyUI integration tests" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J7HDJFz4GpCtv6Nnr8Wqvk"
```

---

### Task 6: Smoke workflow, image comparison tool and README

**Files:**
- Create: `tools/compare_images.py`, `workflows/krea2-glitch-smoke.json`, `README.md`
- Test: `tests/test_compare_images.py`, `tests/test_smoke_workflow.py`

**Interfaces:**
- Consumes: node registry key and input names (Global Constraints). Model/encoder/VAE names verified on 2026-09-14 in `/Volumes/DATA/ComfyUI/user/default/workflows/Krea_2_clean.json`: `LoaderGGUF` (`gguf_name`: `KREA/museByStableYogi_v25GGUF.gguf`), `CLIPLoader` (`qwen3-vl-4b-instruct-abliterated.safetensors`, type `krea2`), `VAELoader` (`qwen_image_vae.safetensors`), `KSampler` 8 steps, cfg 1, `er_sde`, `simple`.
- Produces: `tools/compare_images.py` with `compare(a, b) -> dict`, `format_report(result: dict) -> str`, `main(argv: list[str]) -> int` (0 identical, 1 different, 2 usage).

- [ ] **Step 1: Write the failing tests**

`tests/test_compare_images.py`:
```python
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_images import compare, format_report  # noqa: E402


def png(array):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_identical_images():
    image = np.full((4, 4, 3), 100)
    result = compare(png(image), png(image))
    assert result == {"identical": True, "max_abs": 0, "mean_abs": 0.0, "changed_fraction": 0.0}
    assert format_report(result) == "identical"


def test_different_images():
    a = np.full((4, 4, 3), 100)
    b = a.copy()
    b[0, 0, 0] = 110
    result = compare(png(a), png(b))
    assert result["identical"] is False
    assert result["max_abs"] == 10
    assert result["changed_fraction"] == 1 / 16
    assert format_report(result) == "different: max_abs=10 mean_abs=0.2083 changed_pixels=6.25%"


def test_size_mismatch():
    result = compare(png(np.zeros((4, 4, 3))), png(np.zeros((4, 5, 3))))
    assert result == {"identical": False, "reason": "size differs: 4x4 vs 5x4"}
    assert format_report(result) == "different: size differs: 4x4 vs 5x4"
```

`tests/test_smoke_workflow.py`:
```python
import json
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "krea2-glitch-smoke.json"
LESION_WIDGETS = [
    "enabled", "mode", "strength", "probability", "target",
    "block_start", "block_end", "step_start", "step_end", "glitch_seed",
]


def load():
    return json.loads(WORKFLOW.read_text())


def test_every_node_is_api_format_and_every_link_resolves():
    workflow = load()
    for node_id, node in workflow.items():
        assert isinstance(node["class_type"], str) and isinstance(node["inputs"], dict), node_id
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in workflow, f"{node_id} links to missing node {value[0]}"


def test_glitch_node_sits_between_loader_and_sampler_and_starts_disabled():
    workflow = load()
    glitch_id, glitch = next((k, v) for k, v in workflow.items() if v["class_type"] == "GlitchModelKrea2")
    assert workflow[glitch["inputs"]["model"][0]]["class_type"] == "LoaderGGUF"
    assert sorted(k for k in glitch["inputs"] if k != "model") == sorted(LESION_WIDGETS)
    assert glitch["inputs"]["enabled"] is False
    sampler = next(v for v in workflow.values() if v["class_type"] == "KSampler")
    assert sampler["inputs"]["model"] == [glitch_id, 0]
```

- [ ] **Step 2: Run to verify failure**

Run: `$PYTEST tests/test_compare_images.py tests/test_smoke_workflow.py`
Expected: FAIL (`ModuleNotFoundError: No module named 'compare_images'`, `FileNotFoundError` for the workflow).

- [ ] **Step 3: Implement `tools/compare_images.py`**

```python
"""Compare two images pixel by pixel.

Usage: /Volumes/DATA/ComfyUI/.venv/bin/python tools/compare_images.py A.png B.png
Exit code: 0 identical, 1 different, 2 usage error.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image


def compare(a, b) -> dict:
    with Image.open(a) as image_a, Image.open(b) as image_b:
        pixels_a = np.asarray(image_a.convert("RGB"), dtype=np.int16)
        pixels_b = np.asarray(image_b.convert("RGB"), dtype=np.int16)
    if pixels_a.shape != pixels_b.shape:
        size_a = f"{pixels_a.shape[1]}x{pixels_a.shape[0]}"
        size_b = f"{pixels_b.shape[1]}x{pixels_b.shape[0]}"
        return {"identical": False, "reason": f"size differs: {size_a} vs {size_b}"}
    diff = np.abs(pixels_a - pixels_b)
    return {
        "identical": bool(diff.max() == 0),
        "max_abs": int(diff.max()),
        "mean_abs": float(diff.mean()),
        "changed_fraction": float((diff.max(axis=-1) > 0).mean()),
    }


def format_report(result: dict) -> str:
    if "reason" in result:
        return f"different: {result['reason']}"
    if result["identical"]:
        return "identical"
    return (
        f"different: max_abs={result['max_abs']} mean_abs={result['mean_abs']:.4f} "
        f"changed_pixels={result['changed_fraction']:.2%}"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    result = compare(argv[0], argv[1])
    print(format_report(result))
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Create `workflows/krea2-glitch-smoke.json`**

```json
{
  "1": {"class_type": "LoaderGGUF", "_meta": {"title": "Krea2 GGUF"},
        "inputs": {"gguf_name": "KREA/museByStableYogi_v25GGUF.gguf"}},
  "2": {"class_type": "GlitchModelKrea2", "_meta": {"title": "Glitch Model (Krea2)"},
        "inputs": {"model": ["1", 0], "enabled": false, "mode": "noise", "strength": 0.5, "probability": 0.25,
                   "target": "both", "block_start": 0, "block_end": 27, "step_start": 0, "step_end": 3,
                   "glitch_seed": 0}},
  "3": {"class_type": "CLIPLoader", "_meta": {"title": "Krea2 text encoder"},
        "inputs": {"clip_name": "qwen3-vl-4b-instruct-abliterated.safetensors", "type": "krea2", "device": "default"}},
  "4": {"class_type": "CLIPTextEncode", "_meta": {"title": "Prompt"},
        "inputs": {"clip": ["3", 0],
                   "text": "a photograph of a red bicycle leaning against a white brick wall, soft morning daylight, sharp focus"}},
  "5": {"class_type": "ConditioningZeroOut", "_meta": {"title": "Empty negative"},
        "inputs": {"conditioning": ["4", 0]}},
  "6": {"class_type": "EmptyLatentImage", "_meta": {"title": "Latent"},
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
  "7": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
        "inputs": {"model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                   "seed": 123456789, "steps": 8, "cfg": 1.0, "sampler_name": "er_sde", "scheduler": "simple",
                   "denoise": 1.0}},
  "8": {"class_type": "VAELoader", "_meta": {"title": "VAE"},
        "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
  "9": {"class_type": "VAEDecode", "_meta": {"title": "Decode"},
        "inputs": {"samples": ["7", 0], "vae": ["8", 0]}},
  "10": {"class_type": "SaveImage", "_meta": {"title": "Save"},
         "inputs": {"images": ["9", 0], "filename_prefix": "glitchlab/smoke"}},
  "11": {"class_type": "PreviewAny", "_meta": {"title": "Recipe"},
         "inputs": {"source": ["2", 1]}}
}
```

- [ ] **Step 5: Run to verify pass**

Run: `$PYTEST tests/test_compare_images.py tests/test_smoke_workflow.py`
Expected: all PASS.

- [ ] **Step 6: Write `README.md`**

````markdown
# ComfyUI-LesionLab

`Glitch Model (Krea2)` damages a Krea2 model's internal activations while it generates, in a controlled
and repeatable way, so you can see how the picture changes. It never modifies a checkpoint file or the
model weights: remove the node and the model is back to normal.

## Install

This folder is the custom-node package. Link it into ComfyUI once, then restart ComfyUI:

```bash
ln -s /Users/wswi/Desktop/CLAUDE/ComfyUI-LesionLab /Volumes/DATA/ComfyUI/custom_nodes/ComfyUI-LesionLab
```

The node appears under **experimental → glitches**. Edits in this folder take effect after a restart.

## Wiring

```
LoaderGGUF / Load Diffusion Model → Glitch Model (Krea2) → KSampler → VAE Decode → Save Image
```

Works with any Krea2 model (GGUF or safetensors). Other architectures are rejected with a clear error.
The `recipe` output is a one-line summary of the settings; connect it to `Preview Any` or save it with
your image metadata.

## Controls

| Input | Meaning |
|---|---|
| `enabled` | Off = unmodified model |
| `mode` | `dropout`, `amplify`, `sign_flip` or `noise` (below) |
| `strength` | Dose. 0 is always no change |
| `probability` | Fraction of the hidden channels hit at each site (at least one channel when > 0) |
| `target` | Glitch the output of `attention`, `mlp` or `both` in each block |
| `block_start`, `block_end` | Which of the 28 main blocks (0–27), inclusive |
| `step_start`, `step_end` | Which sampling steps, inclusive; step 0 is the first step KSampler runs |
| `glitch_seed` | Picks the channels and the noise. Same seed + same settings = same glitch |

Only the image being generated is glitched. Prompt tokens, the text-fusion stage and reference images are
never touched, so the model still reads the prompt; it just draws it wrongly.

At each site and step, the same randomly chosen channels are hit for every image token. With strength `s`:

| Mode | Selected channels become | `strength` range |
|---|---|---|
| `dropout` | `x × (1 − s)` (1 = silenced) | 0–1 |
| `amplify` | `x × (1 + s)` | 0–10 |
| `sign_flip` | `x × (1 − 2s)` (0.5 = zeroed, 1 = negated) | 0–1 |
| `noise` | `x + s × rms × Gaussian noise`, where rms is the token's own activation size | 0–10 |

Steps: with 8 sampling steps, `step_start=0, step_end=3` glitches the first half, where composition is
decided; later steps mostly affect detail and texture. With `denoise < 1` or KSampler Advanced
`start_at_step`, step 0 is the first step actually run.

## First experiment

`workflows/krea2-glitch-smoke.json` (open it with **Workflow → Open**): fixed prompt and seed,
8 steps, glitch node set to `noise`, strength 0.5, probability 0.25, target both, blocks 0–27, steps 0–3.

1. Set the KSampler's *control after generate* to **fixed** if it shows `randomize`.
2. Run with `enabled` **off** → image A.
3. Run with `enabled` **on** → image B.
4. Run with `enabled` **off** → image C.
5. Run with `enabled` **on** → image D.

Alternate like this: ComfyUI caches results, so re-queuing identical settings returns the cached image
instead of generating again. Then compare (images are in ComfyUI's output folder under `glitchlab/`):

```bash
/Volumes/DATA/ComfyUI/.venv/bin/python tools/compare_images.py A.png C.png   # how much your setup varies run to run
/Volumes/DATA/ComfyUI/.venv/bin/python tools/compare_images.py B.png D.png   # the glitch repeats
/Volumes/DATA/ComfyUI/.venv/bin/python tools/compare_images.py A.png B.png   # the glitch's effect
```

A/C and B/D should be identical, or differ no more than A/C does (Apple GPU kernels are not always
bit-exact). A/B should differ clearly.

Things to try next: `dropout` strength 1 on `mlp` only; a narrow block range such as 10–14; only late
steps (5–7); compare `glitch_seed` 0, 1 and 2 at the same settings.

## Limits

- Put this node before any torch.compile node; a compiled model can skip the glitch hooks.
- Image latents only (no video); single GPU only.
- The loaders' own limits still apply: the installed ComfyUI-GGUF rejects GGUF files tagged with arch
  `krea2` (for example `krea2_turbo_Q4_0.gguf`); `museByStableYogi_v25GGUF.gguf` loads.

## Development

Tests run with the ComfyUI venv's Python; pytest lives in `./.test-deps` (git-ignored):

```bash
/Volumes/DATA/ComfyUI/.venv/bin/python -m pip install --no-cache-dir --target ./.test-deps pytest
PYTHONPATH=.test-deps PYTHONDONTWRITEBYTECODE=1 /Volumes/DATA/ComfyUI/.venv/bin/python -m pytest tests -v
```

Integration tests import ComfyUI read-only from `/Users/wswi/ComfyUI-Installs/ComfyUI/ComfyUI`
(override with `COMFYUI_ROOT`) and read tensor names from the real GGUF (override with `KREA2_GGUF`).
Design and plan: `docs/superpowers/`.
````

- [ ] **Step 7: Run the whole suite**

Run: `$PYTEST tests`
Expected: all PASS, 0 skipped.

- [ ] **Step 8: Commit**

```bash
git add tools/compare_images.py workflows/krea2-glitch-smoke.json README.md tests/test_compare_images.py tests/test_smoke_workflow.py
git commit -m "Add smoke workflow, image comparison tool and README" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J7HDJFz4GpCtv6Nnr8Wqvk"
```

- [ ] **Step 9: Hand off the manual smoke test to the user**

Do not create the symlink or restart ComfyUI (both are outside this folder). Tell the user:
1. Run `ln -s /Users/wswi/Desktop/CLAUDE/ComfyUI-LesionLab /Volumes/DATA/ComfyUI/custom_nodes/ComfyUI-LesionLab` and restart ComfyUI.
2. Follow README "First experiment" (A off, B on, C off, D on) and run the three comparisons.
3. Report the three comparison lines; pass = A/C and B/D identical or within the A/C level, A/B clearly different.
