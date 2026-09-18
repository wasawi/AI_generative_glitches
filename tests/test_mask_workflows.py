import pytest

from workflow_helpers import assert_links_consistent, load_workflow, only, smoke_inputs

SAM3 = "krea2-glitch-mask-sam3.json"
STENCIL = "krea2-glitch-mask-stencil.json"
BOTH = [SAM3, STENCIL]


def source_of(workflow, node, input_name):
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    slot = next(i for i in node["inputs"] if i["name"] == input_name)
    assert slot["link"] is not None, f"{node['type']}.{input_name} is not linked"
    _, origin, origin_slot, *_ = links[slot["link"]]
    return nodes[origin], origin_slot


@pytest.mark.parametrize("name", BOTH)
def test_links_are_consistent(name):
    assert_links_consistent(load_workflow(name))


@pytest.mark.parametrize("name", BOTH)
def test_the_mask_reaches_the_shape_node_through_grow_and_blur(name):
    workflow = load_workflow(name)
    glitch = only(workflow, "type", "GlitchModelKrea2")
    shape = only(workflow, "type", "GlitchShapeKrea2")
    grow = only(workflow, "type", "GrowMaskWithBlur")
    segment = only(workflow, "type", "SAM3_Detect")
    assert source_of(workflow, glitch, "shape") == (shape, 0)
    assert source_of(workflow, shape, "spatial_mask") == (grow, 0)
    assert source_of(workflow, grow, "mask") == (segment, 0)


@pytest.mark.parametrize("name", BOTH)
def test_the_segmenter_has_its_own_checkpoint_and_text_encoder(name):
    workflow = load_workflow(name)
    segment = only(workflow, "type", "SAM3_Detect")
    checkpoint, slot = source_of(workflow, segment, "model")
    assert checkpoint["type"] == "CheckpointLoaderSimple" and slot == 0
    encoder, _ = source_of(workflow, segment, "conditioning")
    assert encoder["type"] == "CLIPTextEncode"
    # the phrase must be encoded by SAM3's own CLIP, never the Krea2 text encoder
    assert source_of(workflow, encoder, "clip") == (checkpoint, 1)


@pytest.mark.parametrize("name", BOTH)
def test_individual_masks_stays_off(name):
    # on, SAM3 returns one mask per object, which GlitchShape reads as frames across steps
    segment = only(load_workflow(name), "type", "SAM3_Detect")
    assert segment["widgets_values"][2] is False


@pytest.mark.parametrize("name", BOTH)
def test_the_mask_is_grown_and_blurred_but_never_thresholded(name):
    workflow = load_workflow(name)
    expand, _, _, _, blur_radius, *_ = only(workflow, "type", "GrowMaskWithBlur")["widgets_values"]
    assert expand > 0 and blur_radius > 0
    assert not any(node["type"] == "RoundMask" for node in workflow["nodes"])


@pytest.mark.parametrize("name", BOTH)
def test_loaders_match_the_smoke_workflow(name):
    workflow = load_workflow(name)
    assert only(workflow, "type", "LoaderGGUF")["widgets_values"] == [smoke_inputs("LoaderGGUF")["gguf_name"]]
    clip = smoke_inputs("CLIPLoader")
    assert only(workflow, "type", "CLIPLoader")["widgets_values"] == [clip["clip_name"], clip["type"], clip["device"]]
    assert only(workflow, "type", "VAELoader")["widgets_values"] == [smoke_inputs("VAELoader")["vae_name"]]


@pytest.mark.parametrize("name,title", [(SAM3, "Pass 1 - clean"), (STENCIL, "KSampler (seed fixed)")])
def test_full_denoise_samplers_match_the_smoke_workflow(name, title):
    sampler = smoke_inputs("KSampler")
    assert only(load_workflow(name), "title", title)["widgets_values"] == [
        sampler["seed"], "fixed", sampler["steps"], sampler["cfg"],
        sampler["sampler_name"], sampler["scheduler"], sampler["denoise"],
    ]


def test_both_passes_share_one_seed_and_one_prompt():
    workflow = load_workflow(SAM3)
    clean = only(workflow, "title", "Pass 1 - clean")
    glitched = only(workflow, "title", "Pass 2 - glitched")
    assert source_of(workflow, clean, "seed")[0]["type"] == "PrimitiveInt"
    for slot in ("seed", "positive", "negative"):
        assert source_of(workflow, clean, slot) == source_of(workflow, glitched, slot)


def test_the_second_pass_redenoises_only_the_masked_region():
    # a full second generation diverges — attention spreads the damage outside the mask, so the
    # mask has to gate the sampler (SetLatentNoiseMask), not merely the per-token dose
    workflow = load_workflow(SAM3)
    clean = only(workflow, "title", "Pass 1 - clean")
    glitched = only(workflow, "title", "Pass 2 - glitched")
    masked, slot = source_of(workflow, glitched, "latent_image")
    assert masked["type"] == "SetLatentNoiseMask" and slot == 0
    assert source_of(workflow, masked, "samples") == (clean, 0)
    assert source_of(workflow, masked, "mask") == (only(workflow, "type", "GrowMaskWithBlur"), 0)


def test_the_glitched_pass_matches_the_clean_one_except_for_denoise():
    workflow = load_workflow(SAM3)
    clean = only(workflow, "title", "Pass 1 - clean")["widgets_values"]
    glitched = only(workflow, "title", "Pass 2 - glitched")["widgets_values"]
    assert glitched[:6] == clean[:6]
    assert clean[6] == 1.0
    assert 0.0 < glitched[6] < 1.0


def test_the_empty_latent_feeds_only_the_clean_pass():
    workflow = load_workflow(SAM3)
    clean = only(workflow, "title", "Pass 1 - clean")
    empty = only(workflow, "type", "EmptyLatentImage")
    assert source_of(workflow, clean, "latent_image") == (empty, 0)
    assert len(empty["outputs"][0]["links"]) == 1


def test_only_the_second_pass_is_glitched():
    workflow = load_workflow(SAM3)
    clean = only(workflow, "title", "Pass 1 - clean")
    glitched = only(workflow, "title", "Pass 2 - glitched")
    assert source_of(workflow, clean, "model")[0]["type"] == "LoaderGGUF"
    assert source_of(workflow, glitched, "model") == (only(workflow, "type", "GlitchModelKrea2"), 0)


def test_the_segmented_image_is_the_clean_pass():
    workflow = load_workflow(SAM3)
    segment = only(workflow, "type", "SAM3_Detect")
    decode, slot = source_of(workflow, segment, "image")
    assert decode["title"] == "Decode (clean)" and slot == 0
    assert source_of(workflow, decode, "samples")[0]["title"] == "Pass 1 - clean"


def test_the_glitch_targets_the_early_steps():
    glitch = only(load_workflow(SAM3), "type", "GlitchModelKrea2")
    enabled, mode, *_ = glitch["widgets_values"]
    assert enabled is True and mode == "noise"
    # block_start, block_end, step_start, step_end
    assert glitch["widgets_values"][5:9] == [0, 27, 0, 2]


def test_the_stencil_workflow_samples_once_from_a_reference_picture():
    workflow = load_workflow(STENCIL)
    segment = only(workflow, "type", "SAM3_Detect")
    reference, slot = source_of(workflow, segment, "image")
    assert reference["type"] == "LoadImage" and slot == 0
    assert len([n for n in workflow["nodes"] if n["type"] == "KSampler"]) == 1
