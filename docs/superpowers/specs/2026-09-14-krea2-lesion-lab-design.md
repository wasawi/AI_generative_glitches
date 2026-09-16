# Krea2 Lesion Lab — design

Status: approved in brainstorming, 2026-09-14. Supersedes the original
`2026-09-14-krea2-lesion-lab-design.md` (kept in git history).

## Purpose

A ComfyUI custom node that applies reproducible, parameterized lesions to the
internal activations of a Krea2 diffusion model while it generates, so the
*picture itself* is distorted in controlled, repeatable ways. It never
modifies a checkpoint file or the model weights. It is an experimental tool
and makes no claims beyond what the images show.

## Target environment (verified 2026-09-14)

| Item | Value |
|---|---|
| ComfyUI code | `/Users/wswi/ComfyUI-Installs/ComfyUI/ComfyUI`, version 0.35.1 |
| Base directory / custom nodes | `/Volumes/DATA/ComfyUI`, `/Volumes/DATA/ComfyUI/custom_nodes` |
| Python / torch | `/Volumes/DATA/ComfyUI/.venv`, Python 3.12.11, torch 2.9.1, MPS |
| Krea2 model code | `comfy/ldm/krea2/model.py`, class `SingleStreamDiT` |
| Known-loadable Krea2 GGUF | `models/unet/KREA/museByStableYogi_v25GGUF.gguf` (GGUF arch tag `qwen_image`, 28 blocks) |
| Not loadable by installed ComfyUI-GGUF | `krea2_turbo_Q4_0.gguf` (arch tag `krea2` is not in its `IMG_ARCH_LIST`) |

A second ComfyUI install (`~/ComfyUI-Installs/ComfyUI - LOCAL`, Python 3.13)
exists but is not a target.

## Constraints

- All source, tests, docs and tooling live in this folder
  (`/Users/wswi/Desktop/CLAUDE/ComfyUI-LesionLab`). Nothing outside it is
  written. Other locations are read only.
- ComfyUI discovers the node through a symlink the **user** creates:
  `ln -s /Users/wswi/Desktop/CLAUDE/ComfyUI-LesionLab /Volumes/DATA/ComfyUI/custom_nodes/ComfyUI-LesionLab`.
  This folder is therefore itself the package root.
- Never write, rename or replace any checkpoint; never patch model weights.
- Do not modify ComfyUI or other custom nodes; use only public ComfyUI
  extension APIs.

## Scope

In: any Krea2 `MODEL` (GGUF via ComfyUI-GGUF, or safetensors via the core
loader); lesions on the output of the attention and MLP sub-modules of the
main transformer blocks, restricted to generated-image token positions.

Out (first release): other architectures; text-fusion blocks; text-token or
reference-image-token lesions; weight lesions or permanent GGUF variants;
multi-frame (T > 1) latents; multi-GPU; torch.compile compatibility.

## Node

Class `LesionModelKrea2`, registry key `LesionModelKrea2`, display name
`Lesion Model (Krea2)`, category `experimental/lesion-lab`.

Wiring: `Load Diffusion Model` or `Unet Loader (GGUF)` → `Lesion Model (Krea2)` → `KSampler`.

### Inputs

| Input | Widget | Default | Range | Meaning |
|---|---|---|---|---|
| `model` | MODEL | — | — | Krea2 model |
| `enabled` | BOOLEAN | `True` | — | `False` returns an unmodified clone |
| `mode` | combo | `noise` | `dropout`, `amplify`, `sign_flip`, `noise` | Lesion type |
| `strength` | FLOAT | `0.15` | any finite value, step 0.01 (widget bounds ±1,000,000) | Dose, sign included; `0` is always a no-op |
| `probability` | FLOAT | `0.25` | 0–1, step 0.0001 | Fraction of hidden channels hit per site |
| `target` | combo | `both` | `attention`, `mlp`, `both` | Which sub-module outputs |
| `block_start` | INT | `0` | 0–27 | First block, inclusive; clamped to the model |
| `block_end` | INT | `27` | 0–27 | Last block, inclusive; clamped to the model, never an error |
| `step_start` | INT | `0` | 0–10000 | First sampling step, inclusive |
| `step_end` | INT | `999` | 0–10000 | Last sampling step, inclusive |
| `lesion_seed` | INT | `0` | 0–2^63−1 | Seed for channel selection and noise |

The seed input is deliberately **not** named `seed`: ComfyUI's frontend
attaches an auto-randomizing "control after generate" widget to inputs named
`seed` or `noise_seed`, which would break repeatability.

### Outputs

`model` (MODEL) and `recipe` (STRING). Format for an active recipe:

```
krea2-lesion v1 | mode=noise strength=0.15 probability=0.25 | target=both blocks=0-27 sites=56 | steps=0-999 | tokens=image | lesion_seed=0
```

For a no-op the string starts with `no-op (<reason>) | ` followed by the same
fields, where reason is `disabled`, `strength=0` or `probability=0`.

## Lesion semantics

A **site** is the output of `blocks[i].attn` or `blocks[i].mlp` for a block
`i` in `[block_start, block_end]` and a family selected by `target`.
`sites = (block_end − block_start + 1) × (2 if target == both else 1)`.

Activations at a site have shape `[B, L, D]`, token order
`[text | generated image | reference images]`. The lesion touches only rows
`txt : txt + n_img` (generated image) and only the selected channels.

**Channel selection.** For each (lesion_seed, block, family, step), choose
exactly `k = max(1, round(probability × D))` distinct channels using
`torch.randperm(D, generator=g)[:k]` with a CPU generator. The generator seed
is a stable 64-bit mix (splitmix64-style, not Python `hash`) of
`(lesion_seed, block, family_id, step, purpose)` with `family_id`
attention=0/mlp=1 and `purpose` selection=0/noise=1. The same channels are used
for every image token and every batch item, so results do not depend on how
ComfyUI batches cond/uncond.

**Transforms** on the selected slice `a = out[:, txt:txt+n_img, sel]`, with
strength `s`:

| Mode | Replacement | Allowed `s` |
|---|---|---|
| `dropout` | `a × (1 − s)` | any finite (1 = silenced, 2 = negated, −1 = doubled) |
| `amplify` | `a × (1 + s)` | any finite (−2 = inverted) |
| `sign_flip` | `a × (1 − 2s)` | any finite (0.5 = zeroed, 1 = negated, −1 = tripled) |
| `noise` | `a + s × rms × n` | any finite |

For `noise`: `rms` is each token's root-mean-square over all `D` channels of
the original output (shape `[B, n_img, 1]`), computed in float32. `n` is a
float32 standard-normal field of shape `[n_img, k]` drawn from a generator on
the activation's device, seeded with `purpose=1`, broadcast over the batch.
`a + s × rms × n` is computed in float32 and the result cast to the
activation's dtype.

The hook returns a new tensor (clone + index assignment); the module's output
tensor and any input tensor are never modified in place.

**No-op recipes.** `enabled=False`, `strength == 0` or `probability == 0`:
the node returns a clone with no wrapper attached.

## Step index

ComfyUI provides, per model call, `transformer_options["sigmas"]` (current
noise level, one value per batch item) and
`transformer_options["sample_sigmas"]` (the schedule being run, descending,
length `N+1`, last entry typically 0).

`step_from_sigmas(sigma, sample_sigmas)`: take `σ = max(sigmas)`; the step is
the number of entries in `sample_sigmas[1:N]` that are `≥ σ − ε`, with
`ε = 1e-6 × max(sample_sigmas[0], 1)`, clamped to `[0, N−1]`. So a call at
`sample_sigmas[i]` is step `i`, and in-between calls from multi-evaluation
samplers belong to the step whose interval contains them. Step 0 is the first
step of the schedule actually passed to the sampler (after `denoise < 1` or
KSampler Advanced `start_at_step` have shortened it).

## Runtime architecture

```
__init__.py              NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS (relative imports only)
lesion_lab/__init__.py
lesion_lab/recipe.py     LesionRecipe (frozen dataclass), build, noop_reason(), describe(), validate_against_model()
lesion_lab/lesions.py    mix_seed, select_channels, apply_lesion (torch only)
lesion_lab/steps.py      step_from_sigmas (torch only)
lesion_lab/runtime.py    LesionWrapper, lesion hooks, image-slice computation (torch only, duck-typed)
lesion_lab/node.py       LesionModelKrea2 (the only module importing comfy)
tests/                   unit + integration tests
tools/compare_images.py  pixel comparison for the manual smoke test
workflows/krea2-lesion-smoke.json
README.md
docs/superpowers/{specs,plans}/
```

Units depend only downward: `recipe` ← `lesions`, `steps` ← `runtime` ← `node`.
Every module except `node.py` imports without ComfyUI. Inside the package,
imports are relative (`from .recipe import …`), because ComfyUI loads custom
nodes by file path under a hyphenated module name and does not put the
package folder on `sys.path`.

### At node execution (no weights loaded)

1. If `model.model.diffusion_model` is not an instance of
   `comfy.ldm.krea2.model.SingleStreamDiT`, raise
   `ValueError("Lesion Model (Krea2) requires a Krea2 model; got <class name>")`.
2. `LesionRecipe.build(...)` validates inputs; then
   `validate_against_model(recipe, n_blocks=len(diffusion_model.blocks))`
   rejects `block_end` beyond the last block.
3. `clone = model.clone()`.
4. No-op recipe: return `(clone, recipe.describe())`.
5. Otherwise `clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "lesion_lab", LesionWrapper(recipe))`
   and return `(clone, recipe.describe())`.

`ModelPatcher.clone()` copies wrapper lists, so the source model never
carries the wrapper. Chained lesion nodes append to the same key in chain
order and their effects stack.

### Per model call

`SingleStreamDiT.forward` runs all `DIFFUSION_MODEL` wrappers as
`wrapper(executor, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)`,
where `executor.class_obj` is the `SingleStreamDiT`. `LesionWrapper.__call__`:

1. Require `transformer_options["sample_sigmas"]` and `["sigmas"]`; if absent
   raise `RuntimeError("Lesion Model (Krea2): sampler did not provide sample_sigmas; use KSampler, KSamplerAdvanced or SamplerCustom")`.
2. `step = step_from_sigmas(...)`. If outside `[step_start, step_end]`, return
   `executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)`.
3. Accept `x` of shape `[B, C, H, W]` or `[B, C, 1, H, W]` (ComfyUI passes Krea2 image latents
   5-D with T = 1, because Krea2's latent_format, Wan21, has `latent_dimensions = 3` and
   `comfy/sample.py` unsqueezes a 4-D image latent to 5-D before it reaches the diffusion
   model); raise `ValueError` for any other rank, and for 5-D with T > 1 (multi-frame latents
   are unsupported). `patch = executor.class_obj.patch`;
   `n_img = ceil(H/patch) × ceil(W/patch)` using the last two dims of `x` in both cases;
   `txt = context.shape[1]`.
4. Register `register_forward_hook` on each selected `blocks[i].attn` /
   `blocks[i].mlp`; call the executor; remove every handle in `finally`
   (covers exceptions and ComfyUI interrupts).
5. Each hook: output must be a `torch.Tensor` with `shape[1] ≥ txt + n_img`,
   otherwise `RuntimeError` naming the block and family; returns
   `apply_lesion(out, recipe, block, family, step, txt, n_img)`.

## Error handling summary

| Situation | Behavior |
|---|---|
| Non-Krea2 model | `ValueError` at node execution |
| Unknown mode/target, non-finite or out-of-range strength/probability, start > end, negative values, `block_end` past last block | `ValueError` at node execution, message names the field |
| No-op recipe | Clone without wrapper, recipe string says why |
| Missing sigma info | `RuntimeError` during sampling |
| Multi-frame (T > 1) latent | `ValueError` during sampling |
| Unexpected activation shape/type | `RuntimeError` during sampling |
| Any exception inside the model | Hooks removed, exception propagates |

## Testing

Tests run with the DATA venv's Python and pytest installed into
`./.test-deps` (git-ignored):

```
/Volumes/DATA/ComfyUI/.venv/bin/python -m pip install --no-cache-dir --target ./.test-deps pytest
PYTHONPATH=.test-deps PYTHONDONTWRITEBYTECODE=1 /Volumes/DATA/ComfyUI/.venv/bin/python -m pytest -p no:cacheprovider tests
```

**Unit tests (no ComfyUI):**
- recipe: defaults, every validation message, strength accepting any finite value (negatives included), block-range clamping to the model, no-op reasons, `describe()` format.
- lesions: each mode's formula on known values; exactly `k` channels changed; determinism; different step/block/family/seed → different selection; per-item identical result for batch 1 vs 2; text/reference rows and unselected channels unchanged; input not mutated; dtype preserved (float16/bfloat16).
- steps: exact schedule points, in-between sigmas, final step, sigma above `sample_sigmas[0]`, shortened schedules.

**Integration tests** (import ComfyUI read-only from `COMFYUI_ROOT`, default `/Users/wswi/ComfyUI-Installs/ComfyUI/ComfyUI`; skipped if absent):
- Tiny real `SingleStreamDiT` (2 blocks, small width) inside a real `ModelPatcher`, driven through the real `WrapperExecutor` with synthetic `sigmas` / `sample_sigmas`: output differs inside the window and is bit-identical outside; hooks removed after success and after a forced exception; source patcher has no wrapper; chained nodes stack; reference-latent path works; non-Krea2 model rejected.
- GGUF header check (skipped if the file is absent): the tensor names of `museByStableYogi_v25GGUF.gguf` contain `blocks.{0..27}.attn.*` and `blocks.{0..27}.mlp.*`; no tensor data is read.
- Load-as-ComfyUI check: import this folder exactly as `nodes.load_custom_node` does (`spec_from_file_location(<folder path with "." replaced by "_x_">, "<folder>/__init__.py")`) and assert `NODE_CLASS_MAPPINGS["LesionModelKrea2"]`, the input names and the outputs.

**Manual smoke test (user, in the running ComfyUI):** after creating the
symlink and restarting, load `workflows/krea2-lesion-smoke.json` (derived
read-only from an existing Krea2 workflow of the user's), fixed prompt and
sampler seed, and generate alternately: A off, B on, C off, D on (alternating
because ComfyUI caches results for unchanged inputs; A/C is the baseline pair, B/D the lesion pair). `tools/compare_images.py A.png B.png` reports identical
/ max and mean absolute pixel difference. Pass: all four runs complete, the
pairs match (or differ only at the level of the setup's normal MPS
nondeterminism, which the baseline pair measures), and baseline vs lesion
differ visibly.

## Documentation

`README.md`: symlink command, wiring, every input, mode formulas, step
meaning, keep `lesion_seed` fixed, limits (torch.compile unsupported; no
multi-frame latents; no multi-GPU), a first experiment
(`noise`, strength 0.5, probability 0.25, target both, blocks 0-27, steps 0-3),
and how to run the tests.

## Risks

- Strengths near 1000 can overflow float16 activations (NaN / black image); bfloat16 and float32 have
  far more headroom. The limit is not dtype-aware.
- ComfyUI's Krea2 code or wrapper API may change in later versions; the
  integration tests against the installed ComfyUI detect this.
- MPS kernels may not be bit-deterministic between runs; the baseline pair in
  the smoke test measures this before judging the lesion pair.
- torch.compile is not supported: ComfyUI's torch.compile wrapper is an
  APPLY_MODEL wrapper that swaps `diffusion_model` at call time, and both
  wrappers end up on the final patcher regardless of node order, so no
  ordering avoids it; results are undefined with torch.compile in the graph.
  Documented, not handled.
- F5: the integration tests build `transformer_options["wrappers"]` by
  copying `model.wrappers` directly rather than by calling ComfyUI's own
  sampler-side merge (`comfy.sampler_helpers.prepare_model_patcher`) for
  every test; one test now exercises the real merge function directly, but a
  future ComfyUI that stopped merging `model.wrappers` this way would still
  only be caught by that one test, not by the others in the file.
