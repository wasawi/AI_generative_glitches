import pytest

from glitches.recipe import GlitchRecipe, clamp_to_model

DEFAULTS = dict(
    enabled=True, mode="noise", strength=0.15, probability=0.25, target="both",
    block_start=0, block_end=27, step_start=0, step_end=999, glitch_seed=0,
)


def build(**overrides):
    return GlitchRecipe.build(**{**DEFAULTS, **overrides})


def test_defaults_build_an_active_recipe():
    recipe = build()
    assert recipe.noop_reason() is None
    assert recipe.families == ("attention", "mlp")
    assert list(recipe.blocks) == list(range(28))
    assert recipe.site_count == 56


def test_single_family_target_counts_sites():
    recipe = build(target="mlp", block_start=2, block_end=5)
    assert recipe.families == ("mlp",)
    assert recipe.site_count == 4


def test_step_active_is_inclusive():
    recipe = build(step_start=2, step_end=4)
    assert [recipe.step_active(s) for s in range(6)] == [False, False, True, True, True, False]


def test_describe_active_recipe():
    assert build().describe() == (
        "krea2-glitch v1 | mode=noise strength=0.15 probability=0.25 | "
        "target=both blocks=0-27 sites=56 | steps=0-999 | tokens=image | glitch_seed=0"
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [({"enabled": False}, "disabled"), ({"strength": 0.0}, "strength=0"), ({"probability": 0.0}, "probability=0")],
)
def test_noop_reasons(overrides, reason):
    recipe = build(**overrides)
    assert recipe.noop_reason() == reason
    assert recipe.describe().startswith(f"no-op ({reason}) | krea2-glitch v1 | ")


@pytest.mark.parametrize("mode", ["dropout", "amplify", "sign_flip", "noise"])
def test_strength_accepts_any_finite_value_including_negatives(mode):
    assert build(mode=mode, strength=2.0).strength == 2.0
    assert build(mode=mode, strength=250000.0).strength == 250000.0
    assert build(mode=mode, strength=-3.5).strength == -3.5
    assert build(mode=mode, strength=-0.001).noop_reason() is None
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="strength must be a finite number"):
            build(mode=mode, strength=bad)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mode": "scale"}, "mode must be one of dropout, amplify, sign_flip, noise; got 'scale'"),
        ({"target": "ff"}, "target must be one of attention, mlp, both; got 'ff'"),
        ({"strength": float("nan")}, "strength must be a finite number"),
        ({"probability": 1.5}, "probability must be between 0 and 1"),
        ({"probability": "x"}, "probability must be a finite number"),
        ({"block_start": -1}, "block_start must be >= 0"),
        ({"block_start": 5, "block_end": 4}, r"block_end \(4\) must be >= block_start \(5\)"),
        ({"step_start": 3, "step_end": 2}, r"step_end \(2\) must be >= step_start \(3\)"),
        ({"step_end": 2.5}, "step_end must be an integer"),
        ({"glitch_seed": -1}, "glitch_seed must be between 0 and 9223372036854775807"),
    ],
)
def test_validation_messages(overrides, message):
    with pytest.raises(ValueError, match=message):
        build(**overrides)


def test_block_range_is_clamped_to_the_model_never_raising():
    # a too-high block_end used to abort the generation; it now clamps to the model
    clamped = clamp_to_model(build(block_start=5, block_end=999), n_blocks=28)
    assert (clamped.block_start, clamped.block_end) == (5, 27)
    both = clamp_to_model(build(block_start=40, block_end=60), n_blocks=28)
    assert (both.block_start, both.block_end) == (27, 27)
    inside = build(block_start=2, block_end=27)
    assert clamp_to_model(inside, n_blocks=28) is inside
    assert clamp_to_model(build(block_end=3), n_blocks=2).block_end == 1
