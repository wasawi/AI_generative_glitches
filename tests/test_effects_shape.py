import pytest
import torch

from glitches.effects import apply_glitch, select_channels
from glitches.shape import GlitchShape
from test_effects import IMG, N_IMG, TXT, WIDTH, activations, glitch, recipe

MODES = ["dropout", "amplify", "sign_flip", "noise"]
GRID = (2, 2)  # N_IMG == 4


def shape(**overrides):
    args = dict(distribution="gaussian", spike_density=0.05, noise_scale=1)
    args.update(overrides)
    return GlitchShape.build(**args)


def shaped(out, r, s, step=2, block=3, family="mlp", n_steps=8, n_blocks=28, grid=GRID):
    return apply_glitch(out, r, block, family, step, TXT, N_IMG, shape=s, grid=grid, n_steps=n_steps, n_blocks=n_blocks)


@pytest.mark.parametrize("mode", MODES)
def test_passing_no_shape_is_bit_identical_to_the_base_path(mode):
    out = activations()
    r = recipe(mode, 0.5)
    result = apply_glitch(out, r, 3, "mlp", 2, TXT, N_IMG, shape=None, grid=GRID, n_steps=8, n_blocks=28)
    assert torch.equal(result, glitch(out, r))


@pytest.mark.parametrize("mode", ["dropout", "amplify", "sign_flip"])
def test_all_ones_shape_matches_the_unshaped_result(mode):
    out = activations()
    r = recipe(mode, 0.5)
    ones = shape(step_curve=[1.0], block_curve=[1.0], spatial_mask=torch.ones(2, 2))
    torch.testing.assert_close(shaped(out, r, ones), glitch(out, r))


@pytest.mark.parametrize("mode", MODES)
def test_zero_step_multiplier_leaves_the_activation_untouched(mode):
    out = activations()
    assert torch.equal(shaped(out, recipe(mode, 0.5), shape(step_curve=[0.0])), out)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_mask_leaves_the_activation_untouched(mode, dtype):
    out = activations(dtype=dtype)
    result = shaped(out, recipe(mode, 0.5), shape(distribution="cauchy", spatial_mask=torch.zeros(2, 2)))
    assert result.dtype == dtype
    assert torch.equal(result, out)


def test_left_column_mask_only_glitches_left_column_tokens():
    out = activations()
    r = recipe("dropout", 1.0, probability=1.0)
    result = shaped(out, r, shape(spatial_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]])))
    image_before, image_after = out[:, IMG], result[:, IMG]
    assert torch.count_nonzero(image_after[:, [0, 2]]) == 0  # row-major tokens 0 and 2 are the left column
    assert torch.equal(image_after[:, [1, 3]], image_before[:, [1, 3]])
    assert torch.equal(result[:, :TXT], out[:, :TXT])


def test_curves_scale_the_dose():
    out = activations()
    r = recipe("dropout", 1.0)
    selected = select_channels(r, 3, "mlp", 2, WIDTH)
    result = shaped(out, r, shape(step_curve=[0.5], block_curve=[1.0]))
    torch.testing.assert_close(result[:, IMG][..., selected], out[:, IMG][..., selected] * 0.5)


def test_shaped_noise_touches_only_selected_channels_and_is_repeatable():
    out = activations()
    r = recipe("noise", 0.5)
    s = shape(distribution="spikes", spike_density=1.0, noise_scale=2)  # density 1: every selected channel spikes
    result = shaped(out, r, s)
    selected = select_channels(r, 3, "mlp", 2, WIDTH)
    unselected = torch.ones(WIDTH, dtype=torch.bool)
    unselected[selected] = False
    assert torch.equal(result[:, IMG][..., unselected], out[:, IMG][..., unselected])
    assert not torch.equal(result[:, IMG][..., selected], out[:, IMG][..., selected])
    assert torch.equal(result, shaped(out, r, s))


def test_grid_must_match_the_image_token_count():
    with pytest.raises(ValueError, match="grid 3x3 does not match 4 image tokens"):
        shaped(activations(), recipe("dropout", 0.5), shape(), grid=(3, 3))
