from workflow_helpers import assert_links_consistent, load_workflow, only, smoke_inputs


def load():
    return load_workflow("krea2-lesion-shape.json")


def source_of(workflow, node, input_name):
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    slot = next(i for i in node["inputs"] if i["name"] == input_name)
    assert slot["link"] is not None, f"{node['type']}.{input_name} is not linked"
    _, origin, origin_slot, *_ = links[slot["link"]]
    return nodes[origin], origin_slot


def test_links_are_consistent():
    assert_links_consistent(load())


def test_shape_node_feeds_the_lesion_node_and_the_mask_feeds_the_shape():
    workflow = load()
    lesion = only(workflow, "type", "LesionModelKrea2")
    shape = only(workflow, "type", "LesionShapeKrea2")
    assert source_of(workflow, lesion, "shape") == (shape, 0)
    mask_node, mask_slot = source_of(workflow, shape, "spatial_mask")
    assert mask_node["type"] == "CreateShapeMask" and mask_slot == 0
    unlinked = {i["name"]: i["link"] for i in shape["inputs"]}
    assert unlinked["step_curve"] is None and unlinked["block_curve"] is None
    assert shape["widgets_values"] == ["spikes", 0.05, 4]
    assert lesion["widgets_values"][:2] == [True, "noise"]


def test_loaders_and_sampler_match_the_smoke_workflow():
    workflow = load()
    sampler = smoke_inputs("KSampler")
    assert only(workflow, "type", "KSampler")["widgets_values"] == [
        sampler["seed"], "fixed", sampler["steps"], sampler["cfg"],
        sampler["sampler_name"], sampler["scheduler"], sampler["denoise"],
    ]
    assert only(workflow, "type", "LoaderGGUF")["widgets_values"] == [smoke_inputs("LoaderGGUF")["gguf_name"]]
    clip = smoke_inputs("CLIPLoader")
    assert only(workflow, "type", "CLIPLoader")["widgets_values"] == [clip["clip_name"], clip["type"], clip["device"]]
    assert only(workflow, "type", "VAELoader")["widgets_values"] == [smoke_inputs("VAELoader")["vae_name"]]
