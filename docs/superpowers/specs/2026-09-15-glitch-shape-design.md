# Glitch Shape — design

Status: approved in brainstorming, 2026-09-15. Extends
`docs/superpowers/specs/2026-09-14-krea2-glitch-lab-design.md` (the base spec); everything
not changed here stays as the base spec defines it.

## Purpose

Give finer, graphical control over a glitch by reusing ComfyUI's existing interactive editor
nodes instead of writing new frontend code:

- **Value distribution** of the noise mode (not only Gaussian).
- **Spatial pattern**: where on the picture the glitch acts, and how smooth the noise field is.
- **Curves** of glitch strength across sampling steps and across the model's blocks.

## Scope

In: a new node `Glitch Shape (Krea2)`; a new optional `shape` input on `Glitch Model (Krea2)`;
curves from any node that outputs a FLOAT value or list (e.g. KJNodes `Spline Editor`);
masks from any node that outputs `MASK` (KJNodes `CreateShapeMask`, `CreateGradientMask`,
`CreateVoronoiMask`, `CreateFluidMask`, core mask editor / `Load Image` mask); an example workflow.

Out: channel-structure controls; new JavaScript widgets or live previews; changes to the smoke
and random workflows; dtype-aware strength limits.

Constraints from the base spec still hold: all writes stay in this repository; ComfyUI and other
custom nodes are read-only; only `glitches/node.py` imports `comfy`; relative imports only.

## Nodes

### `Glitch Shape (Krea2)`

Class and registry key `GlitchShapeKrea2`, display name `Glitch Shape (Krea2)`, category
`experimental/glitches`, `FUNCTION = "build"`, `RETURN_TYPES = ("GLITCH_SHAPE",)`,
`RETURN_NAMES = ("shape",)`.

| Input | Kind | Default | Range | Meaning |
|---|---|---|---|---|
| `distribution` | combo | `gaussian` | `gaussian`, `uniform`, `laplace`, `cauchy`, `spikes`, `binary` | Noise value distribution (noise mode only) |
| `spike_density` | FLOAT | 0.05 | 0.001–1.0, step 0.001 | Fraction of noise values that spike (`spikes` only) |
| `noise_scale` | INT | 1 | 1–64 | Noise blob size in image tokens (1 token = 16×16 px); 1 = independent per token (noise mode only) |
| `step_curve` | optional FLOAT, `forceInput` | — | finite values | Strength multiplier across sampling steps |
| `block_curve` | optional FLOAT, `forceInput` | — | finite values | Strength multiplier across blocks |
| `spatial_mask` | optional MASK | — | finite values | Strength multiplier over the picture |

A curve input accepts a Python number, a list or tuple of numbers (one level of nested lists is
flattened, as KJNodes' own float nodes do), a `torch.Tensor` (flattened), or any object with a
`tolist()` method (e.g. a pandas Series). Values outside 0–1 are allowed: above 1 boosts, negative
reverses a mode's effect.

### `Glitch Model (Krea2)` change

One new optional input `shape` (`GLITCH_SHAPE`). Adding a link-only optional input does not change
the node's widget order, so existing workflows and saved PNGs are unaffected.

- Not connected: behaviour, output and recipe string are exactly as in the base spec (bit-identical).
- Connected: the node attaches `GlitchWrapper(recipe, shape)`; the no-op rules of the base spec are
  unchanged (a shape never turns a no-op recipe into an active one).
- A value that is not a `GlitchShape` raises `TypeError("shape must come from a Glitch Shape (Krea2) node")`.

### Recipe string with a shape

The base recipe string is followed by one segment:

```
 | shape dist=spikes density=0.05 scale=4 step_curve=16pts block_curve=- mask=16x512x512
```

`-` marks an unconnected input; `mask=FxHxW`. When the recipe's mode is not `noise`, the segment
ends with ` (dist/density/scale unused in mode <mode>)`.

## Data model: `GlitchShape`

`glitches/shape.py`, torch only, no ComfyUI.

```
@dataclass(frozen=True, eq=False)
class GlitchShape:
    distribution: str
    spike_density: float
    noise_scale: int
    step_curve: tuple[float, ...] | None
    block_curve: tuple[float, ...] | None
    spatial_mask: torch.Tensor | None   # float32, CPU, [frames, H, W]
    _mask_cache: dict = field(default_factory=dict, compare=False, repr=False)
```

`eq=False` because a tensor field has no single-boolean equality; identity comparison is enough.

`GlitchShape.build(distribution, spike_density, noise_scale, step_curve=None, block_curve=None,
spatial_mask=None)` validates and normalizes (2-D mask → one frame). A `ValueError` names the field:

| Condition | Message starts with |
|---|---|
| unknown distribution | `distribution must be one of gaussian, uniform, laplace, cauchy, spikes, binary` |
| `spike_density` not in (0, 1] or not finite | `spike_density must be in (0, 1]` |
| `noise_scale` < 1 or not an integer | `noise_scale must be an integer >= 1` |
| curve empty, non-numeric, or non-finite | `step_curve must be a finite number or a non-empty list of finite numbers` (same for `block_curve`) |
| mask not 2-D/3-D, empty, or non-finite | `spatial_mask must be a finite MASK of shape [H, W] or [frames, H, W]` |

## Mapping onto steps, blocks and tokens

For a model call at step `i` of a schedule with `N` steps (`N = len(sample_sigmas) − 1`), block `b`
of `B` blocks (28 for Krea2) and an image-token grid `h × w` (`n_img = h·w`):

**Curve value** `curve_at(c, i, N)` for a curve `c` of length `L`: if `L == 1` or `N == 1` the value
is `c[0]`; otherwise position `p = i·(L−1)/(N−1)`,
`lo = floor(p)`, `hi = min(lo + 1, L − 1)`, value `c[lo] + (c[hi] − c[lo])·(p − lo)`.
`step_multiplier(i, N) = curve_at(step_curve, i, N)` (1.0 if unconnected);
`block_multiplier(b, B) = curve_at(block_curve, b, B)` (1.0 if unconnected).
Curves always span the whole schedule and all blocks; the recipe's step and block windows still
decide where hooks attach.

**Token mask** `token_mask(i, N, h, w, device)`: `None` if unconnected. Otherwise frame
`f = floor(i·(F−1)/(N−1) + 0.5)` (`f = 0` when `F == 1` or `N == 1`), resized with
`torch.nn.functional.interpolate(frame[None, None], size=(h, w), mode="bilinear",
align_corners=False, antialias=True)`, flattened row-major (matching Krea2's
`b c (h ph) (w pw) -> b (h w)` patch order), shaped `[1, n_img, 1]`, float32, on `device`.
Results are cached per `(f, h, w, device)` inside the shape (the cache is not part of equality).

**Multiplier** `m = step_multiplier · block_multiplier · token_mask` (the mask factor is 1 when
unconnected). If the scalar part `step_multiplier · block_multiplier` is exactly 0 and there is no
mask, the site returns the activation unchanged.

**Modes with a shape**, on the selected slice `a` (float32 arithmetic, cast back to the activation dtype):

| Mode | Replacement |
|---|---|
| `dropout` | `a × (1 − s·m)` |
| `amplify` | `a × (1 + s·m)` |
| `sign_flip` | `a × (1 − 2·s·m)` |
| `noise` | `a + s·m·rms·n` |

Without a shape the base spec's code path runs unchanged.

## Noise field with a shape

`draw_noise(k, h, w, generator, device) -> Tensor[n_img, k]` (float32), using the same seeded device
generator as the base spec (`purpose = 1`), so results are deterministic per seed, site and step.

1. Coarse grid `gh = ceil(h/S)`, `gw = ceil(w/S)` with `S = noise_scale`; draw `v` of shape `[k, gh, gw]`:

| distribution | construction (`u = torch.rand`, `z = torch.randn`, all from the generator) |
|---|---|
| `gaussian` | `z` |
| `uniform` | `(2u − 1)·√3` |
| `laplace` | `t = clamp(u − ½, −½ + 1e−7, ½ − 1e−7)`; `−(1/√2)·sign(t)·log1p(−2|t|)` |
| `cauchy` | `clamp(tan(π·(clamp(u, 1e−7, 1 − 1e−7) − ½)), −20, 20)` |
| `spikes` | `u₁ < d ? (u₂ < ½ ? −1 : 1)/√d : 0` with `d = spike_density` |
| `binary` | `u < ½ ? −1 : 1` |

2. If `S > 1`: upsample to `[k, h, w]` (`interpolate(mode="bilinear", align_corners=False)`), then
   divide each channel by its RMS over tokens when that RMS is > 0 (all-zero channels stay zero).
3. Return `v.reshape(k, n_img).T` — the same `[n_img, k]` layout the base spec broadcasts over the batch.

At `noise_scale` 1 (step 2 skipped), all distributions except `cauchy` have RMS 1 in expectation, so
strength means the same across them. At `noise_scale > 1`, step 2's per-channel renormalization makes
every distribution — `cauchy` included — unit-RMS, and can leave an all-zero `spikes` channel empty
(RMS 0, left unrescaled by design) rather than renormalized.

## Runtime changes

- `image_token_grid(x, patch) -> (h, w)` computes the grid from 4-D or single-frame 5-D latents with the
  base spec's rules; `image_token_count` returns `h·w`.
- `GlitchWrapper(recipe, shape=None)`; per call it additionally computes `n_steps = len(sample_sigmas) − 1`,
  the grid `(h, w)`, and `n_blocks = len(dit.blocks)`, and passes them with `shape` to `apply_glitch`.
- `apply_glitch(out, recipe, block, family, step, txt, n_img, *, shape=None, grid=None, n_steps=None, n_blocks=None)`:
  with `shape is None` it is exactly the base implementation.

## Errors summary

| Situation | Behaviour |
|---|---|
| Invalid Glitch Shape input | `ValueError` when the shape node runs, message names the field |
| `shape` input not a `GlitchShape` | `TypeError` when the glitch node runs |
| Anything during sampling | No new failure paths; base spec errors unchanged |

## Testing

Commands and tooling as in the base spec (`$PYTEST tests`, pytest in `./.test-deps`, no `tmp_path`).

Unit (no ComfyUI):
- `GlitchShape.build`: every validation message; curve normalization from number, list, nested list,
  tuple, tensor, and an object with `tolist()`; 2-D mask becomes one frame.
- `curve_at`: exact values for lengths 1, 2 and 16 over 8 steps and 28 blocks, endpoints hit exactly.
- `token_mask`: frame selection across steps for 1, 3 and 16 frames; resize to a non-square grid;
  row-major order (a mask whose left half is 1 marks exactly the left-half tokens); cache reuse.
- `draw_noise`: RMS within 5 % of 1 over ≥ 200k samples for gaussian, uniform, laplace, spikes, binary;
  spikes non-zero fraction within 10 % of the density; cauchy bounded by ±20; `noise_scale = 4`
  output is smooth (neighbouring tokens correlate) and has per-channel RMS 1; deterministic per generator seed.
- `apply_glitch`: no shape → bit-identical to the base result; all-ones shape (curves `[1.0]`,
  mask of ones) → matches the no-shape result within float tolerance (`assert_close`) for dropout,
  amplify and sign_flip (the shape path does its arithmetic in float32, so the last bit may differ); zero step
  multiplier or all-zero mask → bit-identical to the unglitched activation; left-half mask → right-half
  image tokens unchanged.
- Recipe segment format, including the "unused in mode" suffix.

Integration (real ComfyUI read-only, tiny real `SingleStreamDiT`, latents from
`comfy.sample.fix_empty_latent_channels`):
- Shape node output type and registration; `shape` is optional on the glitch node.
- Left-half mask: only left-half image tokens differ from baseline (recording hook on a block).
- Step curve with 0 at some steps: those steps' outputs bit-identical to baseline.
- One shape shared by two chained glitch nodes.
- Wrong `shape` type → `TypeError`.

Example workflow structure test: links consistent, `Glitch Shape` wired into the glitch node's
`shape` input, `CreateShapeMask` wired into `spatial_mask`, loaders and sampler equal to the smoke workflow.

## Documentation and example

- README section "Shaping the glitch": the node's inputs, the mapping rules in plain words, and wiring
  recipes (Spline Editor → `step_curve`/`block_curve` with `points_to_sample` = steps or 28; a mask
  node → `spatial_mask`; multi-frame masks play across steps).
- `workflows/krea2-glitch-shape.json`: the smoke workflow plus `Glitch Shape` (`spikes`, density 0.05,
  `noise_scale` 4) with KJNodes `CreateShapeMask` feeding `spatial_mask`; glitch mode `noise`. The
  Spline Editor is not pre-placed (its curve lives in a JavaScript widget with no known-good saved
  state to copy); the README explains adding it.

## Risks

- Very large curve or mask values multiply strength further; the base spec's float16 overflow note applies.
- Masks with a different aspect ratio than the image are stretched to the token grid.
- KJNodes' node names or outputs may change in future versions; the glitch nodes only depend on the
  generic `FLOAT` and `MASK` types.
