# ComfyUI-LesionLab

`Lesion Model (Krea2)` damages a Krea2 model's internal activations while it generates, in a controlled
and repeatable way, so you can see how the picture changes. It never modifies a checkpoint file or the
model weights: remove the node and the model is back to normal.

## Install

This folder is the custom-node package. Link it into ComfyUI once, then restart ComfyUI:

```bash
ln -s /Users/wswi/Desktop/CLAUDE/ComfyUI-LesionLab /Volumes/DATA/ComfyUI/custom_nodes/ComfyUI-LesionLab
```

The node appears under **experimental → lesion-lab**. Edits in this folder take effect after a restart.

## Wiring

```
GGUF Loader / Unet Loader (GGUF) / Load Diffusion Model → Lesion Model (Krea2) → KSampler → VAE Decode → Save Image
```

`GGUF Loader` (`LoaderGGUF`, from the `gguf` custom-node pack) and `Unet Loader (GGUF)` (from ComfyUI-GGUF) are
alternatives for loading a Krea2 GGUF checkpoint; `Load Diffusion Model` is the core loader for safetensors.

Works with any Krea2 model (GGUF or safetensors). Other architectures are rejected with a clear error.
The `recipe` output is a one-line summary of the settings; connect it to `Preview Any` or save it with
your image metadata.

## Controls

| Input | Meaning |
|---|---|
| `enabled` | Off = unmodified model |
| `mode` | `dropout`, `amplify`, `sign_flip` or `noise` (below) |
| `strength` | Dose. Any finite value, negative included: a negative dose runs the formula backwards (negative `dropout` amplifies, negative `amplify` attenuates and inverts). 0 is always no change |
| `probability` | Fraction of the 6144 hidden channels hit at each site (at least one when > 0, so 0.0002 ≈ a single channel). Step 0.0001 |
| `target` | Lesion the output of `attention`, `mlp` or `both` in each block |
| `block_start`, `block_end` | Which of the 28 main blocks (0–27), inclusive. A value past the model's last block is clamped, never an error |
| `step_start`, `step_end` | Which sampling steps, inclusive; step 0 is the first step KSampler runs |
| `lesion_seed` | Picks the channels and the noise. Same seed + same settings = same lesion. Keep it fixed while comparing runs; change it deliberately to get a different lesion pattern |

Only the image being generated is lesioned. Prompt tokens, the text-fusion stage and reference images are
never touched, so the model still reads the prompt; it just draws it wrongly.

At each site and step, the same randomly chosen channels are hit for every image token. With strength `s`:

| Mode | Selected channels become | `strength` range |
|---|---|---|
| `dropout` | `x × (1 − s)` (1 = silenced, 2 = negated, 5 = −4x, −1 = doubled) | any finite |
| `amplify` | `x × (1 + s)` (−2 = inverted) | any finite |
| `sign_flip` | `x × (1 − 2s)` (0.5 = zeroed, 1 = negated, 1.5 = −2x, −1 = tripled) | any finite |
| `noise` | `x + s × rms × Gaussian noise`, where rms is the token's own activation size | any finite |

Steps: with 8 sampling steps, `step_start=0, step_end=3` lesions the first half, where composition is
decided; later steps mostly affect detail and texture. With `denoise < 1` or KSampler Advanced
`start_at_step`, step 0 is the first step actually run.

## First experiment

`workflows/krea2-lesion-smoke.json` (open it with **Workflow → Open**): fixed prompt and seed,
8 steps, lesion node set to `noise`, strength 0.5, probability 0.25, target both, blocks 0–27, steps 0–3.

1. Set the KSampler's *control after generate* to **fixed** if it shows `randomize`.
2. Run with `enabled` **off** → image A.
3. Run with `enabled` **on** → image B.
4. Run with `enabled` **off** → image C.
5. Run with `enabled` **on** → image D.

Alternate like this: ComfyUI caches results, so re-queuing identical settings returns the cached image
instead of generating again. Then compare (images are in ComfyUI's output folder under `lesionlab/`):

```bash
/Volumes/DATA/ComfyUI/.venv/bin/python tools/compare_images.py A.png C.png   # how much your setup varies run to run
/Volumes/DATA/ComfyUI/.venv/bin/python tools/compare_images.py B.png D.png   # the lesion repeats
/Volumes/DATA/ComfyUI/.venv/bin/python tools/compare_images.py A.png B.png   # the lesion's effect
```

A/C and B/D should be identical, or differ no more than A/C does (Apple GPU kernels are not always
bit-exact). A/B should differ clearly.

Things to try next: `dropout` strength 1 on `mlp` only; a narrow block range such as 10–14; only late
steps (5–7); compare `lesion_seed` 0, 1 and 2 at the same settings.

## Random exploration

`workflows/krea2-lesion-random.json` is the smoke workflow with every lesion input except `enabled` driven by
one random **master seed** (the `Lesion master seed` node, set to *randomize*). Every queue draws a new recipe
that is always valid:

| Input | Random range |
|---|---|
| `mode` | dropout, amplify, sign_flip, noise |
| `target` | attention, mlp, both |
| `strength` | signed: magnitude 0.05–2.0 for dropout and sign_flip, 0.05–4.0 for amplify and noise, either sign |
| `probability` | 0.05–1.0 |
| `block_start`, `block_end` | two draws in 0–27, lower one is the start |
| `step_start`, `step_end` | two draws in 0–7 (the 8 KSampler steps), lower one is the start |
| `lesion_seed` | the master seed itself |

How it works, with no custom code: three core `Math Expression` nodes mix the master seed (splitmix64), eight more
read separate bit fields of the result, and two `easy anythingIndexSwitch` nodes turn the mode and target indexes
into the combo values. It needs comfyui-easy-use (`easy seed`, `easy anythingIndexSwitch`).

- The KSampler seed stays **fixed**, so only the lesion changes between runs. The `Recipe` preview shows what was drawn.
- To repeat a result, set `Lesion master seed` to **fixed** and enter that seed. The seed is also stored in the
  workflow embedded in each saved PNG.
- To change a range, edit the expression in the matching node: `2.0` and `1.0` in `strength`, `0.05` in
  `strength`/`probability`, `% 28` in the block nodes. The step nodes use `& 7` (0–7); for a different KSampler
  step count `N`, use `min(((a >> 46) & 255) % N, ((a >> 54) & 255) % N)` in `step_start` and the same with
  `max` in `step_end` — this also widens each field from 3 bits to 8, which is why the second shift moves
  from 49 to 54 (bits 46–53 and 54–61 are otherwise unused; do not use `(a >> 49) & 255`, since bits 49–56
  would overlap the first field and correlate the two draws).

## Shaping the lesion

`Lesion Shape (Krea2)` is an optional companion node. Connect its `shape` output to the lesion node's
`shape` input; without it the lesion node behaves exactly as described above.

| Input | Meaning |
|---|---|
| `distribution` | Noise values: `gaussian`, `uniform`, `laplace` (occasional large values), `cauchy` (rare huge spikes, clipped at ±20), `spikes` (most values 0, a few big ones), `binary` (±1). Noise mode only |
| `spike_density` | Fraction of values that spike with `spikes` |
| `noise_scale` | Noise blob size in image tokens (1 token = 16×16 px). 1 = fine grain … up to 64 = very large blobs; at the maximum, the coarse grid can collapse to a single cell (at 1024×1024, a 64×64 token grid, `noise_scale` 64 gives each channel one constant value across the whole image — a per-channel offset rather than noise). Noise mode only |
| `step_curve` | Strength multiplier across sampling steps (every mode) |
| `block_curve` | Strength multiplier across the 28 blocks (every mode) |
| `spatial_mask` | Strength multiplier over the picture (every mode): white = full lesion, black = untouched |

At `noise_scale` 1 every distribution except `cauchy` has the same average size, so `strength` means the
same across them. At `noise_scale` above 1 the field is rescaled per channel, which brings `cauchy` to
that size too; with `spikes` and a large blob size on a small image, some channels can come out empty and
add nothing.

**Curves.** Any node with a `FLOAT` value or list output works, for example KJNodes **Spline Editor**
(add it with a double-click search, then connect its `float` output). For one control point per sampling
step, set the Spline Editor's `points_to_sample` to your KSampler step count (8 in the example workflows)
and connect it to `step_curve`; for one control point per Krea2 block, set `points_to_sample` to 28 and
connect it to `block_curve`. A curve is always stretched over the whole run regardless of that count: its
first point is step 0 (or block 0), its last point the final step (or block 27), with straight lines in
between, whatever `points_to_sample` is — so any other length still works. Values above 1 boost the
lesion; negative values reverse it. KJNodes' Spline Editor defaults its own `min_value`/`max_value` to
0/1, so raise `max_value` above 1 to boost or lower `min_value` below 0 to reverse. The lesion node's
step and block windows still limit where it acts,
so open them fully (`step_start` 0, `step_end` 999, `block_start` 0, `block_end` 27) when you let a curve
do the shaping.

**Masks.** Any `MASK` works: KJNodes `CreateShapeMask`, `CreateGradientMask`, `CreateVoronoiMask`,
`CreateFluidMask`, or a mask painted with ComfyUI's mask editor on a `Load Image` node. The mask is
stretched to the image, so draw it at the image's aspect ratio. A mask with several frames plays across
the sampling steps (first frame at step 0, last frame at the final step).

`workflows/krea2-lesion-shape.json` is a ready example: the smoke workflow with `noise` mode shaped by
`spikes` (density 0.05, blob size 4) inside a centred circle from `CreateShapeMask`. It needs
comfyui-kjnodes.

### Randomized version

`workflows/krea2-lesion-shape-random.json` is the same graph with one `Master seed` node (set to
*randomize*) driving **every** lesion, shape and mask input through `Math Expression` nodes and index
switches — the same splitmix64 mixer as the random workflow, extended with two further mixed values so
each setting reads its own bits. Unplug any randomizer to pin that input by hand; `enabled` and the
KSampler seed stay manual, and `step_curve`/`block_curve` stay free for a Spline Editor.

| Input | Random range |
|---|---|
| `mode`, `target` | all four modes / all three targets |
| `strength` | signed: magnitude 0.05–2.0 for dropout and sign_flip, 0.05–4.0 for amplify and noise, either sign |
| `probability` | 0.05–1.0 |
| blocks, steps | two draws each (0–27, 0–7), lower one is the start |
| `distribution` | all six |
| `spike_density` | 0.01–0.5 |
| `noise_scale` | 1–16 |
| mask `shape` | circle, square, triangle |
| mask `frames` | 1–8 (each frame is a full-size mask tensor, so this is deliberately far below the node's 4096) |
| mask canvas | 512, 768 or 1024 square |
| mask position and size | derived from the canvas, so the shape always stays on it (10–60 % of the canvas) |

## Limits

- torch.compile is not supported: ComfyUI's compile wrapper swaps `diffusion_model` at call time regardless
  of node order, so putting this node before or after a torch.compile node does not help; results with
  torch.compile in the graph are undefined.
- Very high strengths can overflow models that run in float16 (black image or NaN errors); bfloat16 and
  float32 models have far more headroom.
- Single-frame image latents only (multi-frame T > 1 latents are rejected); single GPU only.
- The loaders' own limits still apply: ComfyUI-GGUF's `Unet Loader (GGUF)` rejects GGUF files tagged with
  arch `krea2` (for example `krea2_turbo_Q4_0.gguf`); the smoke workflow's `museByStableYogi_v25GGUF.gguf`
  loads with the `gguf` pack's `GGUF Loader`, which is what the user's own Krea2 workflow uses.

## Development

Tests run with the ComfyUI venv's Python; pytest lives in `./.test-deps` (git-ignored):

```bash
/Volumes/DATA/ComfyUI/.venv/bin/python -m pip install --no-cache-dir --target ./.test-deps pytest
PYTHONPATH=.test-deps PYTHONDONTWRITEBYTECODE=1 /Volumes/DATA/ComfyUI/.venv/bin/python -m pytest tests -v
```

Integration tests import ComfyUI read-only from `/Users/wswi/ComfyUI-Installs/ComfyUI/ComfyUI`
(override with `COMFYUI_ROOT`) and read tensor names from the real GGUF (override with `KREA2_GGUF`).
Design and plan: `docs/superpowers/`.
