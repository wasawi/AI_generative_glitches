import pytest
import torch

from test_node_integration import H, W, apply, comfy, hook_count, node, patcher, run  # noqa: F401 (fixtures)

TXT = 5
GRID_H, GRID_W = (H + 1) // 2, (W + 1) // 2  # latent 9x6 -> 5x3 image tokens


@pytest.fixture
def shape_node(comfy):
    from glitches.node import GlitchShapeKrea2

    return GlitchShapeKrea2()


def make_shape(shape_node, **overrides):
    args = dict(distribution="gaussian", spike_density=0.05, noise_scale=1)
    args.update(overrides)
    (shape,) = shape_node.build(**args)
    return shape


def test_shape_node_declares_inputs_and_builds_a_shape(shape_node):
    from glitches.shape import GlitchShape

    spec = type(shape_node).INPUT_TYPES()
    assert list(spec["required"]) == ["distribution", "spike_density", "noise_scale"]
    assert list(spec["optional"]) == ["step_curve", "block_curve", "spatial_mask"]
    assert spec["optional"]["step_curve"] == ("FLOAT", {"forceInput": True})
    assert type(shape_node).RETURN_TYPES == ("GLITCH_SHAPE",)
    shape = make_shape(shape_node, distribution="spikes", noise_scale=4, step_curve=[0.0, 1.0], spatial_mask=torch.ones(3, 8, 8))
    assert isinstance(shape, GlitchShape)


def test_recipe_gains_shape_summary_only_when_connected(shape_node, node, patcher):
    shape = make_shape(shape_node)
    _, plain = apply(node, patcher)
    _, shaped = apply(node, patcher, shape=shape)
    assert " | shape " not in plain
    assert shaped == plain + " | " + shape.describe("dropout")


def test_wrong_shape_type_is_rejected(node, patcher):
    with pytest.raises(TypeError, match=r"shape must come from a Glitch Shape \(Krea2\) node"):
        apply(node, patcher, shape="not a shape")


def test_mask_limits_the_glitch_to_masked_tokens(shape_node, node, patcher):
    mask = torch.zeros(GRID_H, GRID_W)
    mask[:, 0] = 1.0
    shape = make_shape(shape_node, spatial_mask=mask)
    clone, _ = apply(node, patcher, probability=1.0, block_start=0, block_end=0, step_start=0, step_end=3, shape=shape)
    dit = patcher.model.diffusion_model
    captured = []
    handle = dit.blocks[0].register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        run(patcher, 0.75)
        run(clone, 0.75)
    finally:
        handle.remove()
    base, glitched = captured
    n_img = GRID_H * GRID_W
    base_img = base[:, TXT:TXT + n_img].reshape(2, GRID_H, GRID_W, -1)
    glitched_img = glitched[:, TXT:TXT + n_img].reshape(2, GRID_H, GRID_W, -1)
    assert not torch.equal(base_img[:, :, 0], glitched_img[:, :, 0])
    assert torch.equal(base_img[:, :, 1:], glitched_img[:, :, 1:])
    assert torch.equal(base[:, :TXT], glitched[:, :TXT])
    assert hook_count(clone) == 0


def test_step_curve_zeros_leave_those_steps_untouched(shape_node, node, patcher):
    shape = make_shape(shape_node, step_curve=[0.0, 1.0, 0.0, 1.0])
    clone, _ = apply(node, patcher, step_start=0, step_end=3, shape=shape)
    for sigma, changed in [(1.0, False), (0.75, True), (0.5, False), (0.25, True)]:
        assert (not torch.equal(run(patcher, sigma), run(clone, sigma))) == changed, f"sigma={sigma}"


def test_one_shape_can_drive_two_chained_glitch_nodes(shape_node, node, patcher):
    shape = make_shape(shape_node, distribution="binary", noise_scale=2)
    first, _ = apply(node, patcher, mode="noise", strength=0.5, target="attention", block_end=0, shape=shape)
    second, _ = apply(node, first, mode="amplify", strength=2.0, target="mlp", block_start=1, shape=shape)
    base, one, two = run(patcher, 0.75), run(first, 0.75), run(second, 0.75)
    assert torch.isfinite(two).all()
    assert not torch.equal(base, one) and not torch.equal(one, two)
    assert hook_count(second) == 0
