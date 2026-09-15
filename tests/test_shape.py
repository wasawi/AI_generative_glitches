import pytest
import torch

from lesion_lab.shape import DISTRIBUTIONS, LesionShape, curve_at


def build(**overrides):
    args = dict(distribution="gaussian", spike_density=0.05, noise_scale=1)
    args.update(overrides)
    return LesionShape.build(**args)


class HasToList:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def test_distributions_are_listed_in_spec_order():
    assert DISTRIBUTIONS == ("gaussian", "uniform", "laplace", "cauchy", "spikes", "binary")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.5, (0.5,)),
        (2, (2.0,)),
        ([0, 0.5, 1], (0.0, 0.5, 1.0)),
        ((1, 2), (1.0, 2.0)),
        ([[0, 1], [2]], (0.0, 1.0, 2.0)),
        (torch.tensor([[0.0, 1.0], [2.0, 3.0]]), (0.0, 1.0, 2.0, 3.0)),
        (HasToList([3, 4]), (3.0, 4.0)),
    ],
)
def test_curves_are_normalized_to_float_tuples(value, expected):
    assert build(step_curve=value).step_curve == expected
    assert build(block_curve=value).block_curve == expected


def test_unconnected_inputs_stay_none():
    shape = build()
    assert shape.step_curve is None and shape.block_curve is None and shape.spatial_mask is None


def test_2d_mask_becomes_a_single_float32_cpu_frame():
    mask = build(spatial_mask=torch.ones(4, 6, dtype=torch.float64)).spatial_mask
    assert mask.shape == (1, 4, 6) and mask.dtype == torch.float32 and mask.device.type == "cpu"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"distribution": "perlin"}, "distribution must be one of gaussian, uniform, laplace, cauchy, spikes, binary"),
        ({"spike_density": 0.0}, r"spike_density must be in \(0, 1\]"),
        ({"spike_density": 1.5}, r"spike_density must be in \(0, 1\]"),
        ({"spike_density": float("nan")}, r"spike_density must be in \(0, 1\]"),
        ({"spike_density": "x"}, r"spike_density must be in \(0, 1\]"),
        ({"noise_scale": 0}, "noise_scale must be an integer >= 1"),
        ({"noise_scale": 2.5}, "noise_scale must be an integer >= 1"),
        ({"noise_scale": True}, "noise_scale must be an integer >= 1"),
        ({"step_curve": []}, "step_curve must be a finite number or a non-empty list of finite numbers"),
        ({"step_curve": [1.0, float("nan")]}, "step_curve must be a finite number or a non-empty list of finite numbers"),
        ({"step_curve": "abc"}, "step_curve must be a finite number or a non-empty list of finite numbers"),
        ({"step_curve": ["1"]}, "step_curve must be a finite number or a non-empty list of finite numbers"),
        ({"block_curve": [float("inf")]}, "block_curve must be a finite number or a non-empty list of finite numbers"),
        ({"spatial_mask": torch.zeros(1, 1, 2, 2)}, r"spatial_mask must be a finite MASK of shape \[H, W\] or \[frames, H, W\]"),
        ({"spatial_mask": torch.zeros(0, 4)}, r"spatial_mask must be a finite MASK of shape \[H, W\] or \[frames, H, W\]"),
        ({"spatial_mask": torch.tensor([[float("nan")]])}, r"spatial_mask must be a finite MASK of shape \[H, W\] or \[frames, H, W\]"),
        ({"spatial_mask": [[1.0]]}, r"spatial_mask must be a finite MASK of shape \[H, W\] or \[frames, H, W\]"),
    ],
)
def test_validation_messages(overrides, message):
    with pytest.raises(ValueError, match=message):
        build(**overrides)


def test_float_noise_scale_with_integer_value_is_accepted():
    assert build(noise_scale=4.0).noise_scale == 4


def test_curve_at_single_value_and_single_step():
    assert curve_at((0.7,), 5, 8) == 0.7
    assert curve_at((0.2, 0.9), 0, 1) == 0.2


def test_two_point_curve_interpolates_linearly_across_steps():
    assert [curve_at((0.0, 1.0), i, 8) for i in range(8)] == pytest.approx([i / 7 for i in range(8)])


def test_sixteen_point_curve_spans_steps_and_blocks():
    curve = tuple(float(v) for v in range(16))
    assert curve_at(curve, 0, 8) == 0.0 and curve_at(curve, 7, 8) == 15.0
    assert curve_at(curve, 0, 28) == 0.0 and curve_at(curve, 27, 28) == 15.0
    assert curve_at(curve, 9, 28) == pytest.approx(5.0)


def test_multipliers_default_to_one_and_follow_curves():
    assert build().step_multiplier(3, 8) == 1.0
    assert build().block_multiplier(4, 28) == 1.0
    shape = build(step_curve=[0.0, 1.0], block_curve=[2.0])
    assert shape.step_multiplier(7, 8) == 1.0
    assert shape.step_multiplier(0, 8) == 0.0
    assert shape.block_multiplier(13, 28) == 2.0


def test_describe_lists_every_setting():
    shape = build(distribution="spikes", noise_scale=4, step_curve=list(range(16)), spatial_mask=torch.ones(16, 32, 24))
    assert shape.describe("noise") == "shape dist=spikes density=0.05 scale=4 step_curve=16pts block_curve=- mask=16x32x24"


def test_describe_marks_noise_settings_unused_outside_noise_mode():
    assert build().describe("dropout") == (
        "shape dist=gaussian density=0.05 scale=1 step_curve=- block_curve=- mask=- "
        "(dist/density/scale unused in mode dropout)"
    )
