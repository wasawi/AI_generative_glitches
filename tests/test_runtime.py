import pytest
import torch
from torch import nn

from glitches.recipe import GlitchRecipe
from glitches.runtime import GlitchWrapper, image_token_count

SCHEDULE = torch.tensor([1.0, 0.5, 0.0])
TXT = 3
WIDTH = 8


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, x):
        return self.mlp(self.attn(x))


class FakeDiT(nn.Module):
    patch = 2

    def __init__(self, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(n_blocks))


def hook_counts(dit):
    return [len(b.attn._forward_hooks) + len(b.mlp._forward_hooks) for b in dit.blocks]


class FakeExecutor:
    """Stands in for comfy.patcher_extension.WrapperExecutor: runs all blocks over [B, txt + img, D] ones."""

    def __init__(self, dit):
        self.class_obj = dit
        self.seen_hooks = None

    def sequence_length(self, x, context):
        return context.shape[1] + (x.shape[-2] // 2) * (x.shape[-1] // 2)

    def __call__(self, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs):
        self.seen_hooks = hook_counts(self.class_obj)
        seq = torch.ones(x.shape[0], self.sequence_length(x, context), WIDTH)
        for block in self.class_obj.blocks:
            seq = block(seq)
        return seq


def recipe(**overrides):
    args = dict(enabled=True, mode="dropout", strength=1.0, probability=1.0, target="both",
                block_start=0, block_end=1, step_start=1, step_end=1, glitch_seed=0)
    args.update(overrides)
    return GlitchRecipe.build(**args)


def call(wrapper, executor, sigma, options=None):
    x = torch.zeros(1, 4, 4, 6)
    context = torch.zeros(1, TXT, 12)
    if options is None:
        options = {"sigmas": torch.tensor([sigma]), "sample_sigmas": SCHEDULE}
    return wrapper(executor, x, torch.tensor([sigma]), context, None, None, options)


def test_image_token_count_rounds_up_odd_sizes():
    assert image_token_count(torch.zeros(1, 4, 5, 6), 2) == 9


def test_image_token_count_treats_4d_and_5d_singleframe_the_same():
    # Real Krea2 latents are 5-D with T=1 (comfy/sample.py:58-59 unsqueezes a 4-D image
    # latent because Krea2's latent_format, Wan21, has latent_dimensions == 3). Both ranks
    # must give the same token count for the same H, W; odd H exercises the ceil() rounding.
    assert image_token_count(torch.zeros(1, 4, 5, 6), 2) == image_token_count(torch.zeros(1, 4, 1, 5, 6), 2) == 9


def test_multiframe_latents_are_rejected():
    with pytest.raises(ValueError, match="multi-frame"):
        image_token_count(torch.zeros(1, 4, 2, 4, 6), 2)


def test_3d_latents_are_rejected():
    with pytest.raises(ValueError, match="image latents"):
        image_token_count(torch.zeros(4, 5, 6), 2)


def test_outside_step_window_runs_untouched_without_hooks():
    executor = FakeExecutor(FakeDiT())
    out = call(GlitchWrapper(recipe()), executor, sigma=1.0)
    assert executor.seen_hooks == [0, 0]
    assert torch.all(out == 1)


def test_inside_window_hooks_only_selected_sites_glitches_image_rows_and_cleans_up():
    executor = FakeExecutor(FakeDiT())
    out = call(GlitchWrapper(recipe(target="mlp", block_start=1, block_end=1)), executor, sigma=0.5)
    assert executor.seen_hooks == [0, 1]
    assert torch.all(out[:, :TXT] == 1)
    assert torch.all(out[:, TXT:] == 0)
    assert hook_counts(executor.class_obj) == [0, 0]


def test_hooks_are_removed_when_the_model_raises():
    class Boom(FakeExecutor):
        def __call__(self, *args, **kwargs):
            super().__call__(*args, **kwargs)
            raise RuntimeError("boom")

    executor = Boom(FakeDiT())
    with pytest.raises(RuntimeError, match="boom"):
        call(GlitchWrapper(recipe()), executor, sigma=0.5)
    assert executor.seen_hooks == [2, 2]
    assert hook_counts(executor.class_obj) == [0, 0]


def test_missing_sample_sigmas_is_a_clear_error():
    with pytest.raises(RuntimeError, match="sample_sigmas"):
        call(GlitchWrapper(recipe()), FakeExecutor(FakeDiT()), sigma=0.5, options={"sigmas": torch.tensor([0.5])})


def test_activation_shorter_than_image_slice_is_a_clear_error():
    class Short(FakeExecutor):
        def sequence_length(self, x, context):
            return 2

    executor = Short(FakeDiT())
    with pytest.raises(RuntimeError, match="unexpected attention output at block 0"):
        call(GlitchWrapper(recipe()), executor, sigma=0.5)
    assert hook_counts(executor.class_obj) == [0, 0]


def test_image_token_grid_returns_height_and_width():
    from glitches.runtime import image_token_grid

    assert image_token_grid(torch.zeros(1, 4, 5, 6), 2) == (3, 3)
    assert image_token_grid(torch.zeros(1, 4, 1, 5, 6), 2) == (3, 3)


def test_wrapper_passes_shape_grid_step_and_block_counts(monkeypatch):
    import glitches.runtime as runtime
    from glitches.shape import GlitchShape

    seen = []

    def spy(output, recipe_, block, family, step, txt, n_img, **kwargs):
        seen.append((block, family, step, txt, n_img, kwargs))
        return output

    monkeypatch.setattr(runtime, "apply_glitch", spy)
    s = GlitchShape.build("gaussian", 0.05, 1)
    call(GlitchWrapper(recipe(target="mlp", block_start=1, block_end=1), s), FakeExecutor(FakeDiT()), sigma=0.5)
    assert seen == [(1, "mlp", 1, TXT, 6, {"shape": s, "grid": (2, 3), "n_steps": 2, "n_blocks": 2})]
