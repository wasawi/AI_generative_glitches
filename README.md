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
LoaderGGUF / Load Diffusion Model → Lesion Model (Krea2) → KSampler → VAE Decode → Save Image
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
| `target` | Lesion the output of `attention`, `mlp` or `both` in each block |
| `block_start`, `block_end` | Which of the 28 main blocks (0–27), inclusive |
| `step_start`, `step_end` | Which sampling steps, inclusive; step 0 is the first step KSampler runs |
| `lesion_seed` | Picks the channels and the noise. Same seed + same settings = same lesion |

Only the image being generated is lesioned. Prompt tokens, the text-fusion stage and reference images are
never touched, so the model still reads the prompt; it just draws it wrongly.

At each site and step, the same randomly chosen channels are hit for every image token. With strength `s`:

| Mode | Selected channels become | `strength` range |
|---|---|---|
| `dropout` | `x × (1 − s)` (1 = silenced) | 0–1 |
| `amplify` | `x × (1 + s)` | 0–10 |
| `sign_flip` | `x × (1 − 2s)` (0.5 = zeroed, 1 = negated) | 0–1 |
| `noise` | `x + s × rms × Gaussian noise`, where rms is the token's own activation size | 0–10 |

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

## Limits

- Put this node before any torch.compile node; a compiled model can skip the lesion hooks.
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
