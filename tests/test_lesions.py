import pytest
import torch

from lesion_lab.lesions import apply_lesion, channel_count, mix_seed, select_channels
from lesion_lab.recipe import LesionRecipe

TXT, N_IMG, REF, WIDTH = 3, 4, 2, 20
IMG = slice(TXT, TXT + N_IMG)


def recipe(mode="dropout", strength=1.0, probability=0.25, seed=7):
    return LesionRecipe.build(
        enabled=True, mode=mode, strength=strength, probability=probability, target="both",
        block_start=0, block_end=27, step_start=0, step_end=999, lesion_seed=seed,
    )


def activations(batch=2, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, TXT + N_IMG + REF, WIDTH, generator=generator).to(dtype)


def lesion(out, r, block=3, family="mlp", step=2):
    return apply_lesion(out, r, block, family, step, TXT, N_IMG)


def assert_text_and_reference_rows_untouched(before, after):
    assert torch.equal(after[:, :TXT], before[:, :TXT])
    assert torch.equal(after[:, TXT + N_IMG:], before[:, TXT + N_IMG:])


def test_mix_seed_golden_values_never_change():
    assert mix_seed(0, 0, "attention", 0, 0) == 8695987549771912286
    assert mix_seed(7, 3, "mlp", 2, 1) == 5231173937662915042


def test_mix_seed_changes_with_every_component():
    base = mix_seed(7, 3, "mlp", 2, 0)
    others = [
        mix_seed(8, 3, "mlp", 2, 0), mix_seed(7, 4, "mlp", 2, 0), mix_seed(7, 3, "attention", 2, 0),
        mix_seed(7, 3, "mlp", 3, 0), mix_seed(7, 3, "mlp", 2, 1),
    ]
    assert base not in others


@pytest.mark.parametrize(("probability", "expected"), [(0.25, 5), (0.01, 1), (0.5, 10), (1.0, 20)])
def test_channel_count(probability, expected):
    assert channel_count(probability, WIDTH) == expected


def test_select_channels_is_deterministic_distinct_and_site_specific():
    r = recipe()
    first = select_channels(r, 3, "mlp", 2, WIDTH)
    assert torch.equal(first, select_channels(r, 3, "mlp", 2, WIDTH))
    assert first.numel() == 5 and first.unique().numel() == 5
    others = [
        select_channels(r, 4, "mlp", 2, WIDTH), select_channels(r, 3, "attention", 2, WIDTH),
        select_channels(r, 3, "mlp", 3, WIDTH), select_channels(recipe(seed=8), 3, "mlp", 2, WIDTH),
    ]
    assert all(not torch.equal(first, other) for other in others)


@pytest.mark.parametrize(
    ("mode", "strength", "factor"),
    [
        ("dropout", 1.0, 0.0), ("dropout", 0.25, 0.75), ("amplify", 1.0, 2.0), ("sign_flip", 1.0, -1.0), ("sign_flip", 0.5, 0.0),
        # strengths above 1 continue the same formulas
        ("dropout", 2.0, -1.0), ("dropout", 5.0, -4.0), ("sign_flip", 1.5, -2.0), ("amplify", 1000.0, 1001.0),
    ],
)
def test_scaling_modes_apply_their_formula_to_selected_image_channels(mode, strength, factor):
    out = activations()
    r = recipe(mode, strength)
    result = lesion(out, r)
    selected = select_channels(r, 3, "mlp", 2, WIDTH)
    torch.testing.assert_close(result[:, IMG][..., selected], out[:, IMG][..., selected] * factor)
    unselected = torch.ones(WIDTH, dtype=torch.bool)
    unselected[selected] = False
    assert torch.equal(result[:, IMG][..., unselected], out[:, IMG][..., unselected])
    assert_text_and_reference_rows_untouched(out, result)


def test_noise_is_rms_scaled_selected_only_and_shared_across_batch():
    out = activations()
    r = recipe("noise", 0.5)
    result = lesion(out, r)
    selected = select_channels(r, 3, "mlp", 2, WIDTH)
    changed = torch.nonzero((result != out).any(dim=0).any(dim=0)).flatten()
    assert changed.tolist() == sorted(selected.tolist())
    assert_text_and_reference_rows_untouched(out, result)
    image = out[:, IMG]
    rms = image.pow(2).mean(dim=-1, keepdim=True).sqrt()
    unit_noise = (result[:, IMG][..., selected] - image[..., selected]) / (0.5 * rms)
    torch.testing.assert_close(unit_noise[0], unit_noise[1])


def test_noise_is_repeatable_and_step_specific():
    out = activations()
    r = recipe("noise", 0.5)
    assert torch.equal(lesion(out, r, step=2), lesion(out, r, step=2))
    assert not torch.equal(lesion(out, r, step=2), lesion(out, r, step=3))


@pytest.mark.parametrize("mode", ["dropout", "amplify", "sign_flip", "noise"])
def test_result_per_item_does_not_depend_on_batch_size(mode):
    out = activations(batch=2)
    r = recipe(mode, 0.5)
    torch.testing.assert_close(lesion(out, r)[:1], lesion(out[:1], r), rtol=0, atol=1e-6)


def test_input_is_not_mutated():
    out = activations()
    snapshot = out.clone()
    lesion(out, recipe("noise", 0.5))
    lesion(out, recipe("sign_flip", 1.0))
    assert torch.equal(out, snapshot)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["dropout", "noise"])
def test_dtype_is_preserved(dtype, mode):
    result = lesion(activations(dtype=dtype), recipe(mode, 0.5))
    assert result.dtype == dtype
    assert torch.isfinite(result.float()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["dropout", "amplify", "sign_flip", "noise"])
def test_maximum_strength_stays_finite(dtype, mode):
    result = lesion(activations(dtype=dtype), recipe(mode, 1000.0))
    assert result.dtype == dtype
    assert torch.isfinite(result.float()).all()


def test_full_probability_dropout_silences_every_image_channel():
    out = activations()
    result = lesion(out, recipe("dropout", 1.0, probability=1.0))
    assert torch.count_nonzero(result[:, IMG]) == 0
    assert_text_and_reference_rows_untouched(out, result)
