import pytest

from lesion_lab.recipe import LesionRecipe, validate_against_model

DEFAULTS = dict(
    enabled=True, mode="noise", strength=0.15, probability=0.25, target="both",
    block_start=0, block_end=27, step_start=0, step_end=999, lesion_seed=0,
)


def build(**overrides):
    return LesionRecipe.build(**{**DEFAULTS, **overrides})


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
        "krea2-lesion v1 | mode=noise strength=0.15 probability=0.25 | "
        "target=both blocks=0-27 sites=56 | steps=0-999 | tokens=image | lesion_seed=0"
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [({"enabled": False}, "disabled"), ({"strength": 0.0}, "strength=0"), ({"probability": 0.0}, "probability=0")],
)
def test_noop_reasons(overrides, reason):
    recipe = build(**overrides)
    assert recipe.noop_reason() == reason
    assert recipe.describe().startswith(f"no-op ({reason}) | krea2-lesion v1 | ")


@pytest.mark.parametrize(("mode", "limit"), [("dropout", 1), ("sign_flip", 1), ("amplify", 10), ("noise", 10)])
def test_strength_limit_per_mode(mode, limit):
    assert build(mode=mode, strength=limit).strength == limit
    with pytest.raises(ValueError, match=f"strength for mode {mode} must be between 0 and {limit}"):
        build(mode=mode, strength=limit + 0.01)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mode": "scale"}, "mode must be one of dropout, amplify, sign_flip, noise; got 'scale'"),
        ({"target": "ff"}, "target must be one of attention, mlp, both; got 'ff'"),
        ({"strength": float("nan")}, "strength must be a finite number"),
        ({"strength": -0.1}, "strength for mode noise must be between 0 and 10"),
        ({"probability": 1.5}, "probability must be between 0 and 1"),
        ({"probability": "x"}, "probability must be a finite number"),
        ({"block_start": -1}, "block_start must be >= 0"),
        ({"block_start": 5, "block_end": 4}, r"block_end \(4\) must be >= block_start \(5\)"),
        ({"step_start": 3, "step_end": 2}, r"step_end \(2\) must be >= step_start \(3\)"),
        ({"step_end": 2.5}, "step_end must be an integer"),
        ({"lesion_seed": -1}, "lesion_seed must be between 0 and 9223372036854775807"),
    ],
)
def test_validation_messages(overrides, message):
    with pytest.raises(ValueError, match=message):
        build(**overrides)


def test_validate_against_model():
    validate_against_model(build(block_end=27), n_blocks=28)
    with pytest.raises(ValueError, match="block_end 28 exceeds last block 27"):
        validate_against_model(build(block_end=28), n_blocks=28)
