import math

import pytest
import torch

from glitches.shape import CAUCHY_CLIP, GlitchShape

CPU = torch.device("cpu")


def build(**overrides):
    args = dict(distribution="gaussian", spike_density=0.05, noise_scale=1)
    args.update(overrides)
    return GlitchShape.build(**args)


def draw(distribution, k=4, h=250, w=200, scale=1, density=0.05, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return build(distribution=distribution, spike_density=density, noise_scale=scale).draw_noise(k, h, w, generator, CPU)


def test_no_mask_gives_none():
    assert build().token_mask(0, 8, 5, 3, CPU) is None


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(1, [0] * 8), (3, [0, 0, 1, 1, 1, 1, 2, 2]), (16, [0, 2, 4, 6, 9, 11, 13, 15])],
)
def test_mask_frames_play_across_steps(frames, expected):
    mask = torch.stack([torch.full((4, 4), float(f)) for f in range(frames)])
    shape = build(spatial_mask=mask)
    assert [round(shape.token_mask(i, 8, 2, 2, CPU)[0, 0, 0].item()) for i in range(8)] == expected


def test_mask_at_grid_size_keeps_row_major_layout():
    mask = torch.zeros(5, 3)
    mask[:, 0] = 1.0
    mask[0, 2] = 0.5
    tokens = build(spatial_mask=mask).token_mask(0, 8, 5, 3, CPU)
    assert tokens.shape == (1, 15, 1) and tokens.dtype == torch.float32
    assert torch.equal(tokens.reshape(5, 3), mask)


def test_mask_is_resized_to_a_non_square_grid():
    mask = torch.zeros(512, 512)
    mask[:, :256] = 1.0
    grid = build(spatial_mask=mask).token_mask(0, 8, 5, 4, CPU).reshape(5, 4)
    torch.testing.assert_close(grid[:, 0], torch.ones(5), rtol=0, atol=1e-6)
    torch.testing.assert_close(grid[:, 3], torch.zeros(5), rtol=0, atol=1e-6)


def test_resized_mask_is_cached():
    shape = build(spatial_mask=torch.rand(3, 16, 16))
    first = shape.token_mask(0, 8, 5, 3, CPU)
    assert shape.token_mask(0, 8, 5, 3, CPU) is first
    assert shape.token_mask(1, 8, 5, 3, CPU) is first  # step 1 still uses frame 0 of 3
    assert shape.token_mask(7, 8, 5, 3, CPU) is not first


@pytest.mark.parametrize("distribution", ["gaussian", "uniform", "laplace", "cauchy", "spikes", "binary"])
def test_noise_layout_dtype_and_determinism(distribution):
    values = draw(distribution)
    assert values.shape == (50000, 4) and values.dtype == torch.float32
    assert torch.isfinite(values).all()
    assert torch.equal(values, draw(distribution))
    assert not torch.equal(values, draw(distribution, seed=1))


@pytest.mark.parametrize("distribution", ["gaussian", "uniform", "laplace", "spikes", "binary"])
def test_noise_has_unit_rms(distribution):
    rms = draw(distribution).pow(2).mean().sqrt().item()
    assert abs(rms - 1.0) < 0.05


def test_uniform_and_binary_value_ranges():
    assert draw("uniform").abs().max().item() <= math.sqrt(3.0) + 1e-6
    assert set(draw("binary").unique().tolist()) == {-1.0, 1.0}


def test_spike_density_is_respected():
    values = draw("spikes", density=0.05)
    fraction = (values != 0).float().mean().item()
    assert 0.045 <= fraction <= 0.055
    assert values.abs().max().item() == pytest.approx(1 / math.sqrt(0.05))


def test_cauchy_is_clipped_but_heavy_tailed():
    values = draw("cauchy")
    assert values.abs().max().item() <= CAUCHY_CLIP
    assert (values.abs() > 5).any()


def test_noise_scale_gives_smooth_unit_rms_channels():
    k, h, w = 6, 64, 48
    smooth = draw("gaussian", k=k, h=h, w=w, scale=4).T.reshape(k, h, w)
    grainy = draw("gaussian", k=k, h=h, w=w, scale=1).T.reshape(k, h, w)
    torch.testing.assert_close(smooth.pow(2).mean(dim=(1, 2)).sqrt(), torch.ones(k), rtol=0, atol=1e-4)

    def neighbour_correlation(field):
        pairs = torch.stack([field[:, :, :-1].reshape(-1), field[:, :, 1:].reshape(-1)])
        return torch.corrcoef(pairs)[0, 1].item()

    assert neighbour_correlation(smooth) > 0.8
    assert abs(neighbour_correlation(grainy)) < 0.1


def test_scaled_sparse_spikes_with_empty_channels_stay_finite():
    values = draw("spikes", k=16, h=8, w=8, scale=8, density=0.001, seed=2)
    assert torch.isfinite(values).all()
