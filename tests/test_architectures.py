import pytest
import torch

from glitches.architectures import ARCHITECTURES, Krea2Architecture, architecture_for, supported_names


def test_only_krea2_is_supported_today():
    assert ARCHITECTURES == (Krea2Architecture,)
    assert supported_names() == "Krea2"


def test_architecture_lookup_rejects_unknown_models(comfy_root):
    assert architecture_for(torch.nn.Linear(2, 2)) is None
    assert architecture_for(None) is None


def test_krea2_adapter_finds_blocks_and_sites(comfy_root):
    import comfy.ldm.krea2.model as krea2_model

    torch.manual_seed(0)
    dit = krea2_model.SingleStreamDiT(
        features=64, tdim=32, txtdim=32, heads=4, kvheads=2, multiplier=2, layers=3, patch=2, channels=4,
        txtlayers=3, txtheads=4, txtkvheads=4, operations=torch.nn,
    )
    assert architecture_for(dit) is Krea2Architecture
    assert Krea2Architecture.block_count(dit) == 3
    for block in range(3):
        assert Krea2Architecture.site_module(dit, block, "attention") is dit.blocks[block].attn
        assert Krea2Architecture.site_module(dit, block, "mlp") is dit.blocks[block].mlp


def test_krea2_adapter_reads_the_token_grid_and_text_length(comfy_root):
    import comfy.ldm.krea2.model as krea2_model

    dit = krea2_model.SingleStreamDiT(
        features=64, tdim=32, txtdim=32, heads=4, kvheads=2, multiplier=2, layers=2, patch=2, channels=4,
        txtlayers=3, txtheads=4, txtkvheads=4, operations=torch.nn,
    )
    assert Krea2Architecture.image_token_grid(dit, torch.zeros(1, 4, 9, 6)) == (5, 3)  # odd size rounds up
    assert Krea2Architecture.image_token_grid(dit, torch.zeros(1, 4, 1, 9, 6)) == (5, 3)  # real 5-D latent
    assert Krea2Architecture.text_token_count(torch.zeros(2, 7, 96)) == 7
    with pytest.raises(ValueError, match="multi-frame"):
        Krea2Architecture.image_token_grid(dit, torch.zeros(1, 4, 2, 9, 6))
    with pytest.raises(ValueError, match=r"\[B, C, H, W\] only"):
        Krea2Architecture.image_token_grid(dit, torch.zeros(4, 9, 6))
