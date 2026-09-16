import pytest
import torch

from lesion_lab.runtime import image_token_count

SCHEDULE = [1.0, 0.75, 0.5, 0.25, 0.0]


@pytest.fixture(scope="module")
def comfy(comfy_root):
    import comfy.ldm.krea2.model as krea2_model
    import comfy.model_patcher as model_patcher
    import comfy.patcher_extension as patcher_extension

    return krea2_model, model_patcher, patcher_extension


@pytest.fixture
def node(comfy):
    from lesion_lab.node import LesionModelKrea2

    return LesionModelKrea2()


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
                block_start=0, block_end=1, step_start=1, step_end=2, lesion_seed=7)
    args.update(overrides)
    return node.apply(model, **args)


H, W = 9, 6  # odd H exercises the ceil() padding in image_token_count / process_img (F1)


class _LatentFormatHolder:
    """Exposes exactly the one attribute comfy.sample.fix_empty_latent_channels reads for a
    non-empty latent: model.get_model_object("latent_format")."""

    def get_model_object(self, name):
        assert name == "latent_format"
        from comfy.latent_formats import Wan21  # Krea2.latent_format (comfy/supported_models.py)

        return Wan21()


def build_image_latent(batch, channels, h, w, generator):
    """Build a latent the way ComfyUI actually hands one to the diffusion model instead of
    hand-writing its rank: comfy/sample.py fix_empty_latent_channels (used by common_ksampler,
    SamplerCustom and SamplerCustomAdvanced) unconditionally unsqueezes a 4-D latent to 5-D
    once the model's latent_format reports latent_dimensions == 3, which Krea2's format
    (Wan21) does. The tensor is non-zero so fix_empty_latent_channels's is_empty branch (which
    would resize channels to latent_format.latent_channels == 16) does not run and channels
    stay at the tiny test model's channels=4 -- only the unsqueeze(2), the part F1 is about,
    is exercised."""
    import comfy.sample as comfy_sample

    x = torch.randn(batch, channels, h, w, generator=generator)
    return comfy_sample.fix_empty_latent_channels(_LatentFormatHolder(), x)


def run(model, sigma, ref=False, without=()):
    """Call the diffusion model the way ComfyUI's sampler does, with model.wrappers in transformer_options."""
    dit = model.model.diffusion_model
    generator = torch.Generator().manual_seed(1)
    x = build_image_latent(2, 4, H, W, generator)
    context = torch.randn(2, 5, 3 * 32, generator=generator)
    sigmas = torch.full((2,), sigma)
    options = {"wrappers": model.wrappers, "sigmas": sigmas, "sample_sigmas": torch.tensor(SCHEDULE)}
    for key in without:
        options.pop(key)
    kwargs = {}
    if ref:
        kwargs = {"ref_latents": [torch.randn(1, 4, H, W, generator=generator)], "ref_latents_method": "index_timestep_zero"}
    with torch.no_grad():
        return dit(x, sigmas, context, transformer_options=options, **kwargs)


def hook_count(model):
    return sum(len(b.attn._forward_hooks) + len(b.mlp._forward_hooks) for b in model.model.diffusion_model.blocks)


def lesion_wrappers(comfy, model):
    return model.get_wrappers(comfy[2].WrappersMP.DIFFUSION_MODEL, "lesion_lab")


def test_rejects_non_krea2_model(comfy, node):
    with pytest.raises(ValueError, match="requires a Krea2 model; got Linear"):
        apply(node, make_patcher(comfy, torch.nn.Linear(2, 2)))


def test_block_end_beyond_the_model_is_clamped_not_rejected(node, patcher):
    # the tiny test model has 2 blocks; a larger block_end used to raise and abort the run
    clone, recipe = apply(node, patcher, block_end=27)
    assert "blocks=0-1" in recipe and "sites=4" in recipe
    assert not torch.equal(run(patcher, 0.75), run(clone, 0.75))
    assert hook_count(clone) == 0


def test_clone_carries_the_wrapper_and_source_stays_clean(comfy, node, patcher):
    clone, recipe = apply(node, patcher)
    assert clone is not patcher
    assert lesion_wrappers(comfy, patcher) == []
    assert len(lesion_wrappers(comfy, clone)) == 1
    assert "sites=4" in recipe and recipe.startswith("krea2-lesion v1 | mode=dropout")


def test_noop_attaches_no_wrapper(comfy, node, patcher):
    clone, recipe = apply(node, patcher, enabled=False)
    assert recipe.startswith("no-op (disabled) | ")
    assert lesion_wrappers(comfy, clone) == []


def test_output_changes_only_inside_the_step_window(node, patcher):
    # x is real 5-D [B, C, 1, H, W] (see build_image_latent / F1), the shape ComfyUI actually
    # passes for Krea2 since Wan21.latent_dimensions == 3.
    clone, _ = apply(node, patcher)
    for sigma, inside in [(1.0, False), (0.75, True), (0.5, True), (0.25, False)]:
        base = run(patcher, sigma)
        lesioned = run(clone, sigma)
        assert lesioned.shape == base.shape == (2, 4, 1, H, W)
        assert (not torch.equal(base, lesioned)) == inside, f"sigma={sigma}"
    assert hook_count(clone) == 0


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
    first, _ = apply(node, patcher, target="attention", block_end=0, lesion_seed=1)
    second, _ = apply(node, first, mode="amplify", strength=2.0, target="mlp", block_start=1, lesion_seed=2)
    assert len(lesion_wrappers(comfy, first)) == 1
    assert len(lesion_wrappers(comfy, second)) == 2
    base, one, two = run(patcher, 0.75), run(first, 0.75), run(second, 0.75)
    assert not torch.equal(base, one)
    assert not torch.equal(one, two)


def test_reference_latent_path(node, patcher):
    # Real 5-D [B, C, 1, H, W] image latent (odd H) plus a reference latent (F1); also checks,
    # by recording block 0's own output with a forward hook the test adds and removes, that the
    # lesion only changes the generated-image rows of the activation and leaves the text and
    # reference-image rows bit-identical to the unlesioned run (F6). block 0's own attn/mlp are
    # lesioned (block_start=0), so its output already reflects the lesion; block 0 operates on
    # the full [text | image | reference] sequence, so this is equivalent to hooking
    # blocks[0].mlp directly without racing the lesion wrapper's own (per-call) hook on that
    # submodule for hook-execution order.
    clone, _ = apply(node, patcher)
    dit = patcher.model.diffusion_model
    assert clone.model.diffusion_model is dit

    captured = []

    def record(module, args, output):
        captured.append(output)

    handle = dit.blocks[0].register_forward_hook(record)
    try:
        for sigma, inside in [(1.0, False), (0.75, True)]:
            base = run(patcher, sigma, ref=True)
            lesioned = run(clone, sigma, ref=True)
            assert lesioned.shape == base.shape == (2, 4, 1, H, W)
            assert torch.isfinite(lesioned).all()
            assert (not torch.equal(base, lesioned)) == inside, f"sigma={sigma}"
    finally:
        handle.remove()
    assert hook_count(clone) == 0

    # captured holds, in order, [base(1.0), lesioned(1.0), base(0.75), lesioned(0.75)]; use the
    # inside-window (0.75) pair.
    base_act, lesioned_act = captured[2], captured[3]
    txt = 5
    n_img = image_token_count(torch.zeros(2, 4, 1, H, W), dit.patch)
    n_ref = image_token_count(torch.zeros(1, 4, H, W), dit.patch)
    assert base_act.shape[1] == lesioned_act.shape[1] == txt + n_img + n_ref
    assert torch.equal(base_act[:, :txt], lesioned_act[:, :txt])  # text rows unchanged
    assert torch.equal(base_act[:, txt + n_img:], lesioned_act[:, txt + n_img:])  # reference rows unchanged
    assert not torch.equal(base_act[:, txt:txt + n_img], lesioned_act[:, txt:txt + n_img])  # image rows differ


def test_missing_sample_sigmas_raises(node, patcher):
    clone, _ = apply(node, patcher)
    with pytest.raises(RuntimeError, match="sample_sigmas"):
        run(clone, 0.75, without=("sample_sigmas",))
    assert hook_count(clone) == 0


def test_wrappers_reach_transformer_options_through_comfys_own_merge(comfy, node, patcher):
    # Every other test in this file hand-builds transformer_options["wrappers"] = model.wrappers,
    # which is how the wrapper reaches the model in production too, but only because ComfyUI's
    # real sampler merge step (comfy.sampler_helpers.prepare_model_patcher, called from
    # KSampler/SamplerCustom/SamplerCustomAdvanced sampling) currently does exactly that. This
    # test calls that real function instead, so a future ComfyUI that stopped merging
    # model.wrappers this way would fail here even though the hand-built route above would keep
    # passing (F5).
    import comfy.sampler_helpers as sampler_helpers

    _, _, patcher_extension = comfy
    clone, _ = apply(node, patcher)

    model_options = {"transformer_options": {}}
    sampler_helpers.prepare_model_patcher(clone, {}, model_options)
    merged_wrappers = model_options["transformer_options"]["wrappers"]
    assert "lesion_lab" in merged_wrappers.get(patcher_extension.WrappersMP.DIFFUSION_MODEL, {})

    dit = clone.model.diffusion_model
    generator = torch.Generator().manual_seed(1)
    x = build_image_latent(2, 4, H, W, generator)
    context = torch.randn(2, 5, 3 * 32, generator=generator)
    sigmas = torch.full((2,), 0.75)
    sample_sigmas = torch.tensor(SCHEDULE)

    lesioned_options = dict(model_options["transformer_options"])
    lesioned_options["sigmas"] = sigmas
    lesioned_options["sample_sigmas"] = sample_sigmas
    with torch.no_grad():
        lesioned = dit(x, sigmas, context, transformer_options=lesioned_options)
        baseline = dit(x, sigmas, context, transformer_options={"sigmas": sigmas, "sample_sigmas": sample_sigmas})
    assert not torch.equal(lesioned, baseline)
    assert hook_count(clone) == 0
