# I Broke the Model on Purpose and It Drew Me a Bicycle Anyway

*Or: what happens when you reach inside a diffusion model while it's thinking and start flipping switches.*

---

## The short version

Every image in this post came from the same prompt, the same model, and **the same sampler seed**:

> *a photograph of a red bicycle leaning against a white brick wall, soft morning daylight, sharp focus*

Seed `123456789`. Eight steps. Never changed once. Not for a single image.

So why does each one look like a different universe had a go at the same bicycle?

Because between the sampler and the model I wired in a node that **damages the model's own thoughts while it generates** — and the only number I changed between runs is the seed for *that* damage.

No LoRA. No img2img. No post-processing filter. Nothing was done to the picture after it was made. The picture came out like that.

---

## What the node actually does

A diffusion model doesn't paint. It thinks, in numbers, and the picture is a side effect.

Krea2 chops your canvas into little 16×16-pixel patches called **tokens**, and hands each one a list of about **6,144 numbers** describing everything it currently believes about that patch — is it metal, is it shadowed, is it part of a wheel, is it in focus. Then it passes those numbers through **28 blocks**, each one nudging them a bit closer to a coherent image. Do that eight times and you get a bicycle.

My node sits in the middle of that and messes with the numbers *in flight*.

Specifically: at every one of those 28 blocks, it picks a random slice of the 6,144 channels and corrupts them. Four ways to do it —

| Mode | What it does to the chosen numbers |
|---|---|
| `dropout` | silences them (or over-silences into negative territory) |
| `amplify` | cranks them up |
| `sign_flip` | inverts them — the model's "definitely bright" becomes "definitely dark" |
| `noise` | adds random noise, scaled to how loud that patch already was |

The images here all use **`noise`**.

Two things that matter, and matter a lot:

**Nothing is written to your checkpoint.** The damage lives in memory for the duration of one generation. Delete the node and the model is exactly as it was. No files touched, ever.

**Only the picture gets hit.** Krea2 processes your prompt and your image side by side in the same stream of tokens. The node steps carefully around the prompt tokens and only corrupts the image ones. The model still *reads* your request perfectly. It just can no longer *draw* it properly. That distinction is the entire aesthetic — this isn't a model that forgot what a bicycle is. It's a model that knows exactly what a bicycle is and has lost fine motor control.

---

## The recipe behind these images

Here it is, the whole thing, nothing hidden:

| Setting | Value | Translation |
|---|---|---|
| `mode` | `noise` | add noise to the activations |
| `strength` | **5.0** | five times the patch's own natural loudness. This is a lot. The default starter value is 0.5 |
| `probability` | **0.25** | corrupt a quarter of the 6,144 channels — about 1,500 of them |
| `target` | `both` | hit the attention stage *and* the feed-forward stage in each block |
| `block_start` / `block_end` | **0 → 27** | all 28 blocks, no survivors |
| `step_start` / `step_end` | **0 → 2** | **only the first three of eight steps** |
| `distribution` | `binary` | every noise value is exactly +1 or −1. Hard-edged, digital, no gentle gradients |
| `noise_scale` | **4** | noise comes in 4-token blobs — chunks of roughly 64×64 pixels, not per-pixel fuzz |
| `spatial_mask` | 512×512 square or circle, centred | the damage is confined to the middle of the frame |

Model was `museByStableYogi_v25` (a Krea2 GGUF), 8 steps, `er_sde` / `simple`, CFG 1.0, 1664×1248, batches of four.

---

## Why `steps 0–2` is the whole trick

This is the setting I'd argue about with strangers.

Early sampling steps decide **composition** — where the mass is, where the light comes from, what the big shapes are. Later steps decide **texture and detail** — the grain of the brick, the highlight on the chrome.

Glitch the late steps and you get crunchy artifacts sprinkled over a perfectly normal photo of a bicycle. Mildly interesting. Looks like a JPEG having a bad day.

Glitch the **first three steps** and you corrupt the decision-making *before the model has committed to anything* — and then you hand the wreckage to five clean steps that try, in complete sincerity, to render it as a beautiful photograph. Soft morning daylight. Sharp focus. Every bit of the model's craft, applied to a structure that was never a bicycle in the first place.

That's the good stuff. The model isn't glitching. The model is doing its absolute best. The glitch happened three steps ago and it's too late for anyone.

---

## Why there's a mask

`spatial_mask` restricts the damage to a region of the frame. Everything outside it generates completely untouched.

Twenty-two of these use a **square** mask; seven use a **circle**. Both are 512×512, centred on a 1024 canvas, stretched to fit the frame — so in both cases the middle half of the picture is in the blast zone and the edges are pristine.

The point isn't subtlety. It's **contrast**. A picture that's glitched everywhere just reads as "broken image, scroll on". A picture where the outer third is a calm, competent, soft-morning-light photograph of a white brick wall, and the middle has *come apart*, reads as something happening to a real place. Your eye needs the undamaged part to understand what it's losing.

The square and circle land differently, too. The square's straight edges fight the organic shapes and you get a visible rectangular *frame* of damage — a screen within the screen. The circle's edge dissolves more readily into the composition, so the damage feels like it's welling up from inside the scene rather than being projected onto it.

---

## What changes between the images: one number

Every single setting above is identical across all 29 images. So is the prompt. So is the sampler seed.

The only thing that moves is **`glitch_seed`** — which decides *which* 1,500 channels get hit and *what* the noise looks like. I just ran consecutive integers, `…455` through `…479`, and kept what I liked.

(A few images share a glitch seed — those came out of the same batch of four. Same corruption, same everything; they differ only because each slot in a batch starts from different initial noise. Nice illustration of how much of the result is the glitch and how much is ordinary diffusion luck.)

Same seed plus same settings always gives the same glitch, which is the part I care about most. This is a repeatable instrument, not a slot machine. Find something you like, write down the seed, get it back tomorrow.

---

## Two bits of honesty about the metadata

If you pull these PNGs into ComfyUI and go looking, you'll find two settings that do nothing:

- `spike_density 0.05` — only applies to the `spikes` distribution. These use `binary`. It's a leftover from an earlier session, sitting there inert.
- Two **Spline Editor** nodes are present in the graph but **unplugged**. They drive `step_curve` and `block_curve` — strength multipliers that ramp the glitch across the sampling steps or across the 28 blocks. I left them disconnected for this set to keep the recipe flat and easy to reason about. They're a whole other rabbit hole.

---

## Try it yourself

The nodes are on GitHub, MIT licensed:

**https://github.com/wasawi/AI_generative_glitches**

Clone it into `custom_nodes`, restart ComfyUI, and look under **experimental → glitches**.

```bash
git clone https://github.com/wasawi/AI_generative_glitches.git /path/to/ComfyUI/custom_nodes/AI_generative_glitches
```

You'll need a **Krea2** model (GGUF or safetensors — it checks the architecture, not the file format). Four example workflows ship with it:

- **`krea2-glitch-smoke.json`** — start here. Toggle `enabled` on and off with a fixed seed and watch what one setting does.
- **`krea2-glitch-shape.json`** — adds the shape node: distributions, blob size, masks, curves. This is the one these images came from.
- **`krea2-glitch-random.json`** and **`krea2-glitch-shape-random.json`** — every input driven from a single master seed. Hit queue, get a completely new recipe every time, built entirely from stock math nodes. This is how you find settings you'd never have thought to type. Unplug any randomizer to pin that value by hand.

If you want to land somewhere near these images fast: `noise`, strength **5**, probability **0.25**, target **both**, blocks **0–27**, steps **0–2**, `binary`, blob size **4**, and a centred shape mask. Then just walk the glitch seed.

And if it comes out ugly — turn the strength up. It gets better around the point where it should have gotten worse.
