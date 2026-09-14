# Krea2 Lesion Lab design

## Purpose

Create a local ComfyUI custom-node package for reproducible inference-time
lesions in Krea2 GGUF diffusion models. The tool is an experimental way to
systematically distort the denoiser's computation. It does not modify the
source GGUF checkpoint and makes no claims about human mental states.

## Scope

The first release targets the user's Krea2 model loaded by the installed
City96 `ComfyUI-GGUF` package at `/Volumes/data/comfyUI/custom_nodes/ComfyUI-GGUF`.
It will be installed as a separate sibling package, `ComfyUI-LesionLab`.

It will not alter LM Studio, create permanent GGUF mutations, or support
other image architectures in the first release.

## Workflow and interface

`Unet Loader (GGUF)` loads the model, followed by a `Lesion Model (GGUF)`
node, followed by the existing KSampler. The lesion node consumes and emits
`MODEL`, allowing all existing sampler and decoder nodes to remain unchanged.

The node exposes:

- enabled switch
- lesion mode: channel dropout, scale, sign flip, or Gaussian noise
- strength
- target block range
- target family: attention, MLP, or both
- denoising step start and end
- probability
- seed

It also emits a concise recipe/status string for use in workflow metadata.

## Runtime design

The lesion node clones the `GGUFModelPatcher` returned by ComfyUI-GGUF and
associates a validated lesion recipe with that clone. Per-model forward hooks
or wrappers identify eligible transformer submodules and apply the recipe only
when a matching denoising step is active.

Random masks and noise are generated deterministically from the user seed,
block identifier, and denoising step. This permits exact repeatability for a
fixed ComfyUI workflow and seed. The hook acts on runtime activations; the
quantized GGUF's on-disk tensors are not written or replaced.

## Validation and failure behavior

The node rejects non-finite strengths, invalid probabilities, malformed block
ranges, and reversed step windows. On architectures without matching target
modules it raises a clear error. A disabled node or a zero-strength recipe is
a no-op baseline.

## Verification

1. Unit checks cover validation, deterministic recipe-derived random state,
   and no-op behavior.
2. A ComfyUI import/discovery check verifies the custom node is registered.
3. A minimal workflow places it between the GGUF loader and KSampler and
   confirms a run completes with both disabled and enabled recipes.
4. A fixed-prompt, fixed-seed visual comparison is retained as a manual
   experiment, not a quality assertion.

## Risks and decisions

The installed ComfyUI-GGUF loader dequantizes layer weights during execution;
the node must preserve its device/offload behavior. Hooks must therefore be
attached per cloned model, never globally, and released with the model.

The initial version favors runtime activation lesions over weight patches:
they are reversible, parameterized in ComfyUI, and avoid invalidating a
13.57 GB checkpoint. Permanent GGUF variants can be a separate later project
once an interesting recipe is established.
