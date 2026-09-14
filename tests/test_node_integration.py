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


def lesion_wrappers(comfy, model):
    return model.get_wrappers(comfy[2].WrappersMP.DIFFUSION_MODEL, "lesion_lab")


def test_rejects_non_krea2_model(comfy, node):
    with pytest.raises(ValueError, match="requires a Krea2 model; got Linear"):
        apply(node, make_patcher(comfy, torch.nn.Linear(2, 2)))


def test_block_end_beyond_the_model_is_rejected(node, patcher):
    with pytest.raises(ValueError, match="block_end 2 exceeds last block 1"):
        apply(node, patcher, block_end=2)


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
    clone, _ = apply(node, patcher)
    for sigma, inside in [(1.0, False), (0.75, True), (0.5, True), (0.25, False)]:
        base = run(patcher, sigma)
        lesioned = run(clone, sigma)
        assert (not torch.equal(base, lesioned)) == inside, f"sigma={sigma}"


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
    clone, _ = apply(node, patcher)
    base = run(patcher, 0.75, ref=True)
    lesioned = run(clone, 0.75, ref=True)
    assert lesioned.shape == base.shape == (2, 4, 8, 6)
    assert torch.isfinite(lesioned).all()
    assert not torch.equal(base, lesioned)


def test_missing_sample_sigmas_raises(node, patcher):
    clone, _ = apply(node, patcher)
    with pytest.raises(RuntimeError, match="sample_sigmas"):
        run(clone, 0.75, without=("sample_sigmas",))
    assert hook_count(clone) == 0
