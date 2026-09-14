# Krea2 Lesion Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ComfyUI node that applies reproducible, parameterized runtime activation lesions to Krea2 models loaded through ComfyUI-GGUF.

**Architecture:** Create a standalone `ComfyUI-LesionLab` custom-node package. The node validates an immutable lesion recipe, clones the incoming `MODEL`, and installs a model-scoped U-Net forward wrapper that temporarily registers hooks on selected attention/MLP modules for each denoising call. The hooks are removed in `finally`, preserving the source GGUF, the base workflow, and ComfyUI-GGUF's on-demand dequantization.

**Tech Stack:** Python 3.12, PyTorch, ComfyUI custom-node API, City96 ComfyUI-GGUF.

**Spec:** `/Users/wswi/Documents/Codex/2026-09-14/co/docs/superpowers/specs/2026-09-14-krea2-lesion-lab-design.md`

## Global Constraints

- Implement and version the source package only under `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab`; deploy the completed package to `/Users/wswi/Desktop/CHATGPT/ComfyUI-LesionLab` without Git metadata. Do not modify `/Volumes/DATA/ComfyUI/custom_nodes/ComfyUI-GGUF`.
- Never write to, rename, or replace any GGUF checkpoint.
- Support only MODEL-in/MODEL-out Krea2 experiments in the first release.
- Use deterministic per-block/per-step PyTorch generators derived from the UI seed.
- Preserve ComfyUI-GGUF's device and offload behavior; do not materialize all quantized weights.
- Node controls are: enabled, mode, strength, target block range, target family, step start/end, probability, and seed.
- Validation failures must be explicit; disabled or zero-strength configurations are no-ops.

---

## File Structure

- `custom_nodes/ComfyUI-LesionLab/__init__.py` — ComfyUI registration and display names.
- `custom_nodes/ComfyUI-LesionLab/lesion_lab/config.py` — immutable recipe, parsing, and validation with no ComfyUI imports.
- `custom_nodes/ComfyUI-LesionLab/lesion_lab/lesions.py` — deterministic tensor transformations and module-target predicates.
- `custom_nodes/ComfyUI-LesionLab/lesion_lab/runtime.py` — forward-wrapper adapter and temporary hook lifecycle.
- `custom_nodes/ComfyUI-LesionLab/lesion_lab/nodes.py` — `LesionModelGGUF` node declaration and MODEL cloning.
- `custom_nodes/ComfyUI-LesionLab/tests/test_config.py` — pure recipe validation tests.
- `custom_nodes/ComfyUI-LesionLab/tests/test_lesions.py` — deterministic transform and no-op tests.
- `custom_nodes/ComfyUI-LesionLab/tests/test_runtime.py` — fake-module hook cleanup tests.
- `custom_nodes/ComfyUI-LesionLab/README.md` — installation, node placement, parameter guidance, and baseline workflow.

### Task 1: Scaffold the package and define recipes

**Files:**
- Create: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/__init__.py`
- Create: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/lesion_lab/__init__.py`
- Create: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/lesion_lab/config.py`
- Create: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/tests/test_config.py`

**Interfaces:**
- Consumes: Primitive values from ComfyUI widgets.
- Produces: `LesionRecipe`, `parse_block_range`, and `validate_recipe` for runtime and node code.

- [ ] **Step 1: Write the failing validation tests**

```python
import pytest
from lesion_lab.config import LesionRecipe, parse_block_range

def test_range_and_recipe_are_normalized():
    recipe = LesionRecipe.build("noise", 0.25, "2-5", "both", 1, 4, 0.5, 17)
    assert recipe.blocks == (2, 5)
    assert recipe.mode == "noise"
    assert recipe.seed == 17

@pytest.mark.parametrize("value", ["", "5-2", "a-4", "1-2-3"])
def test_invalid_block_range_is_rejected(value):
    with pytest.raises(ValueError, match="block range"):
        parse_block_range(value)

def test_invalid_probability_is_rejected():
    with pytest.raises(ValueError, match="probability"):
        LesionRecipe.build("dropout", 0.1, "0-1", "attention", 0, 1, 1.1, 1)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python -m pytest custom_nodes/ComfyUI-LesionLab/tests/test_config.py -v`

Expected: FAIL because `lesion_lab.config` does not exist.

- [ ] **Step 3: Implement immutable recipe validation**

```python
@dataclass(frozen=True)
class LesionRecipe:
    mode: Literal["dropout", "scale", "sign_flip", "noise"]
    strength: float
    blocks: tuple[int, int]
    target: Literal["attention", "mlp", "both"]
    step_start: int
    step_end: int
    probability: float
    seed: int

    @classmethod
    def build(cls, mode, strength, block_range, target, step_start, step_end, probability, seed):
        if mode not in {"dropout", "scale", "sign_flip", "noise"}:
            raise ValueError("mode must be dropout, scale, sign_flip, or noise")
        if target not in {"attention", "mlp", "both"}:
            raise ValueError("target must be attention, mlp, or both")
        if not math.isfinite(float(strength)):
            raise ValueError("strength must be finite")
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        start, end = parse_block_range(block_range)
        if int(step_start) < 0 or int(step_end) < int(step_start):
            raise ValueError("step range must satisfy 0 <= start <= end")
        return cls(mode, float(strength), (start, end), target, int(step_start), int(step_end), float(probability), int(seed))
```

Import `math` and implement `parse_block_range` with `text.partition("-")`; reject a missing separator, non-integer values, negatives, and an end below start with a `ValueError` message beginning `block range`.

- [ ] **Step 4: Run the validation tests**

Run: `./.venv/bin/python -m pytest custom_nodes/ComfyUI-LesionLab/tests/test_config.py -v`

Expected: PASS.

### Task 2: Implement deterministic activation transforms

**Files:**
- Create: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/lesion_lab/lesions.py`
- Create: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/tests/test_lesions.py`

**Interfaces:**
- Consumes: `LesionRecipe`, an activation `torch.Tensor`, `block_index`, and `step_index`.
- Produces: `apply_lesion(tensor, recipe, block_index, step_index) -> torch.Tensor`.

- [ ] **Step 1: Write transform tests**

```python
import torch
from lesion_lab.config import LesionRecipe
from lesion_lab.lesions import apply_lesion

def recipe(mode, strength=0.5, probability=1.0):
    return LesionRecipe.build(mode, strength, "0-3", "both", 0, 3, probability, 9)

def test_zero_strength_is_a_value_preserving_noop():
    x = torch.arange(8, dtype=torch.float32)
    out = apply_lesion(x, recipe("noise", 0.0), 1, 1)
    assert torch.equal(out, x)

def test_noise_is_repeatable_per_seed_block_and_step():
    x = torch.ones(16)
    assert torch.equal(apply_lesion(x, recipe("noise"), 2, 1), apply_lesion(x, recipe("noise"), 2, 1))
    assert not torch.equal(apply_lesion(x, recipe("noise"), 2, 1), apply_lesion(x, recipe("noise"), 2, 2))

def test_sign_flip_at_probability_one_negates_all_values():
    x = torch.tensor([1.0, -2.0])
    assert torch.equal(apply_lesion(x, recipe("sign_flip", probability=1.0), 0, 0), -x)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python -m pytest custom_nodes/ComfyUI-LesionLab/tests/test_lesions.py -v`

Expected: FAIL because `lesion_lab.lesions` does not exist.

- [ ] **Step 3: Implement modes using local generators**

Implement `make_generator(seed, block_index, step_index, device)` and derive its integer seed with a stable arithmetic combination, not Python's randomized `hash()`. `dropout` zeroes mask-selected values; `scale` multiplies selected values by `1 - strength`; `sign_flip` negates selected values; `noise` adds `torch.randn` multiplied by `strength`. Do not mutate the input tensor.

- [ ] **Step 4: Run the transform tests**

Run: `./.venv/bin/python -m pytest custom_nodes/ComfyUI-LesionLab/tests/test_lesions.py -v`

Expected: PASS.

### Task 3: Add a model-scoped hook runtime

**Files:**
- Create: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/lesion_lab/runtime.py`
- Create: `/Volumes/DATA/ComfyUI-LesionLab/tests/test_runtime.py`

**Interfaces:**
- Consumes: `torch.nn.Module`, `LesionRecipe`, current denoising step, and module names.
- Produces: `find_targets`, `install_hooks`, and `run_with_lesions`.

- [ ] **Step 1: Inspect the currently running ComfyUI API before coding the adapter**

Run in the ComfyUI Python environment:

```python
import comfy.model_patcher
import inspect
print(inspect.signature(comfy.model_patcher.ModelPatcher.set_model_unet_function_wrapper))
```

If that method is absent, inspect `dir(comfy.model_patcher.ModelPatcher)` and select its current model forward-wrapper API. Record the exact method name and callback signature in a comment at the top of `runtime.py`. Do not patch `ComfyUI-GGUF` globally.

- [ ] **Step 2: Write hook lifecycle tests using a fake model**

```python
import torch
from torch import nn
from lesion_lab.config import LesionRecipe
from lesion_lab.runtime import run_with_lesions

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.transformer_blocks[0].attn = nn.Identity()
    def forward(self, x):
        return self.transformer_blocks[0].attn(self.transformer_blocks[0](x))

def test_hooks_are_removed_after_success_and_failure():
    model = Toy()
    recipe = LesionRecipe.build("scale", 1.0, "0-0", "attention", 0, 0, 1.0, 1)
    run_with_lesions(model, recipe, 0, lambda: model(torch.ones(1, 2)))
    assert not model.transformer_blocks[0].attn._forward_hooks
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `./.venv/bin/python -m pytest custom_nodes/ComfyUI-LesionLab/tests/test_runtime.py -v`

Expected: FAIL because `lesion_lab.runtime` does not exist.

- [ ] **Step 4: Implement target discovery and scoped hooks**

Implement `find_targets(model, recipe)` by enumerating `model.named_modules()`, extracting numeric transformer-block indices from names containing `transformer_blocks.<N>`, and classifying names containing `attn` as attention and names containing `mlp`, `ff`, or `feed_forward` as MLP. `run_with_lesions` must return the executor result and use `try/finally` to remove every `RemovableHandle`, even if the executor raises. Each hook must pass through outputs that are not tensors and transform tensor outputs via `apply_lesion` only when `step_start <= step_index <= step_end`.

- [ ] **Step 5: Run runtime tests**

Run: `./.venv/bin/python -m pytest custom_nodes/ComfyUI-LesionLab/tests/test_runtime.py -v`

Expected: PASS.

### Task 4: Register the ComfyUI node and wire its controls

**Files:**
- Create: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/lesion_lab/nodes.py`
- Modify: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/__init__.py`
- Modify: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/tests/test_runtime.py`

**Interfaces:**
- Consumes: a ComfyUI `MODEL` and primitive widget values.
- Produces: `LesionModelGGUF.apply(model, enabled, mode, strength, block_range, target, step_start, step_end, probability, seed) -> tuple[MODEL, STRING]` and registry key `LesionModelGGUF`.

- [ ] **Step 1: Write a node-shape test with a fake patcher**

```python
from lesion_lab.nodes import LesionModelGGUF

class FakePatcher:
    def clone(self):
        return FakePatcher()

def test_node_returns_a_clone_and_recipe_status():
    clone, status = LesionModelGGUF().apply(FakePatcher(), True, "noise", 0.2, "0-1", "both", 0, 2, 0.5, 7)
    assert isinstance(clone, FakePatcher)
    assert "noise" in status and "seed=7" in status
```

- [ ] **Step 2: Run the node test to verify it fails**

Run: `./.venv/bin/python -m pytest custom_nodes/ComfyUI-LesionLab/tests/test_runtime.py -v`

Expected: FAIL because `LesionModelGGUF` does not exist.

- [ ] **Step 3: Implement `LesionModelGGUF`**

Define `INPUT_TYPES` with defaults: `enabled=True`, `mode="noise"`, `strength=0.15`, `block_range="0-999"`, `target="both"`, `step_start=0`, `step_end=7`, `probability=0.25`, `seed=0`. Set `RETURN_TYPES = ("MODEL", "STRING")`, `RETURN_NAMES = ("model", "recipe")`, `FUNCTION = "apply"`, `CATEGORY = "experimental/lesion-lab"`, and `TITLE = "Lesion Model (GGUF)"`. Clone the model, attach only the validated recipe and wrapper state to the clone, and return a status string with mode, range, target, strength, probability, and seed. If disabled, return a clone with no wrapper state and `"Lesion disabled"`.

- [ ] **Step 4: Register the node**

```python
from .lesion_lab.nodes import LesionModelGGUF

NODE_CLASS_MAPPINGS = {"LesionModelGGUF": LesionModelGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"LesionModelGGUF": "Lesion Model (GGUF)"}
```

- [ ] **Step 5: Run all package tests**

Run: `./.venv/bin/python -m pytest custom_nodes/ComfyUI-LesionLab/tests -v`

Expected: PASS.

### Task 5: Validate ComfyUI discovery and document the experiment

**Files:**
- Create: `/Users/wswi/Documents/Codex/2026-09-14/co/work/ComfyUI-LesionLab/README.md`

**Interfaces:**
- Consumes: existing `Unet Loader (GGUF)` output.
- Produces: user instructions for an executable Krea2 workflow and baseline comparison.

- [ ] **Step 1: Write the README workflow instructions**

Document this exact wiring: `Unet Loader (GGUF) → Lesion Model (GGUF) → KSampler → VAE Decode → Save Image`. Explain each widget, state that the source GGUF is never modified, and provide baseline settings (`enabled=False`) plus a first experiment (`noise`, `strength=0.15`, `block_range=0-999`, `target=both`, `step_start=2`, `step_end=5`, `probability=0.15`, fixed seed).

- [ ] **Step 2: Start or restart ComfyUI and verify discovery**

Open the custom-node list and confirm `Lesion Model (GGUF)` appears under `experimental/lesion-lab`. Save a workflow containing the node. If startup logs a missing ComfyUI wrapper API, use the signature captured in Task 3 to adjust only `runtime.py` and rerun the package tests.

- [ ] **Step 3: Run the manual smoke workflow**

Use one fixed prompt, model, sampler seed, and KSampler configuration. Generate:

1. a baseline with the node disabled;
2. a repeated baseline to confirm no change;
3. a lesion image using the documented first experiment;
4. a repeated lesion image to confirm deterministic output.

Verify all four runs complete and that the two pairs match within the normal deterministic behavior of the user's setup.

- [ ] **Step 4: Record the outcome in the README**

Add the installed ComfyUI version, the exact selected GGUF filename, the working forward-wrapper API signature, and the first successful lesion recipe. Do not include generated images or copy model files.

## Self-review

- Spec coverage: Tasks 1–2 implement validation, deterministic recipes, four requested lesion modes, and no-op behavior; Task 3 implements model-scoped, step-gated activation hooks and cleanup; Task 4 provides every approved ComfyUI control and MODEL-to-MODEL flow; Task 5 verifies discovery, a smoke workflow, reproducibility, and documentation.
- Placeholder scan: no unresolved tasks, filenames, or validation behaviors remain. The one API inspection is deliberately version-sensitive and records the local exact signature before its adapter is written.
- Type consistency: `LesionRecipe.build` produces the object consumed by `apply_lesion`, `run_with_lesions`, and `LesionModelGGUF.apply`; `LesionModelGGUF.apply` outputs `(MODEL, STRING)` consistently.
