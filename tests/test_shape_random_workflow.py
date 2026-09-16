import random

import pytest

from glitches.recipe import MODES, TARGETS, GlitchRecipe
from glitches.shape import DISTRIBUTIONS, GlitchShape
from workflow_helpers import assert_links_consistent, load_workflow, only, smoke_inputs

EASY_SEED_MAX = 1125899906842624  # comfyui-easy-use py/config.py MAX_SEED_NUM
MASK_SHAPES = ["circle", "square", "triangle"]  # KJNodes CreateShapeMask 'shape' options
LESION_DRIVEN = [
    "mode", "strength", "probability", "target",
    "block_start", "block_end", "step_start", "step_end", "glitch_seed",
]
SHAPE_DRIVEN = ["distribution", "spike_density", "noise_scale"]
MASK_DRIVEN = [
    "shape", "frames", "location_x", "location_y", "grow",
    "frame_width", "frame_height", "shape_width", "shape_height",
]


def load():
    return load_workflow("krea2-glitch-shape-random.json")


class Graph:
    """Evaluates the randomizer for one master seed.

    Math Expression nodes run through ComfyUI's real node implementation; the other three
    upstream types are pass-throughs (seed value, string widget, index switch).
    """

    def __init__(self, workflow, math_node, seed):
        self.nodes = {n["id"]: n for n in workflow["nodes"]}
        self.links = {link[0]: link for link in workflow["links"]}
        self.math_node = math_node
        self.seed = seed
        self.cache = {}

    def input_value(self, node, name):
        slot = next(i for i in node["inputs"] if i["name"] == name)
        assert slot["link"] is not None, f"{node['type']} input {name} is not linked"
        _, origin, origin_slot, *_ = self.links[slot["link"]]
        return self.output(origin, origin_slot)

    def output(self, node_id, slot):
        key = (node_id, slot)
        if key not in self.cache:
            node = self.nodes[node_id]
            kind = node["type"]
            if kind == "easy seed":
                value = self.seed
            elif kind == "PrimitiveString":
                value = node["widgets_values"][0]
            elif kind == "ComfyMathExpression":
                values = {
                    i["name"].split(".", 1)[1]: self.input_value(node, i["name"])
                    for i in node["inputs"]
                    if i["name"].startswith("values.") and i["link"] is not None
                }
                value = self.math_node.execute(expression=node["widgets_values"][0], values=values).args[slot]
            elif kind == "easy anythingIndexSwitch":
                value = self.input_value(node, f"value{self.input_value(node, 'index')}")
            else:
                raise AssertionError(f"unexpected upstream node type: {kind}")
            self.cache[key] = value
        return self.cache[key]

    def linked_inputs(self, node_type, names):
        node = next(n for n in self.nodes.values() if n["type"] == node_type)
        return {name: self.input_value(node, name) for name in names}


@pytest.fixture(scope="module")
def math_node(comfy_root):
    from comfy_extras.nodes_math import MathExpressionNode

    return MathExpressionNode


def test_links_are_consistent():
    assert_links_consistent(load())


def test_every_randomizable_input_is_driven_and_enabled_stays_manual():
    workflow = load()
    glitch = {i["name"]: i for i in only(workflow, "type", "GlitchModelKrea2")["inputs"]}
    assert glitch["enabled"]["link"] is None
    assert glitch["shape"]["link"] is not None
    for name in LESION_DRIVEN:
        assert glitch[name]["link"] is not None, name
    shape = {i["name"]: i for i in only(workflow, "type", "GlitchShapeKrea2")["inputs"]}
    for name in SHAPE_DRIVEN:
        assert shape[name]["link"] is not None, name
    assert shape["step_curve"]["link"] is None and shape["block_curve"]["link"] is None
    assert shape["spatial_mask"]["link"] is not None
    mask = {i["name"]: i for i in only(workflow, "type", "CreateShapeMask")["inputs"]}
    for name in MASK_DRIVEN:
        assert mask[name]["link"] is not None, name


def test_master_seed_randomizes_and_sampler_stays_fixed():
    workflow = load()
    assert only(workflow, "type", "easy seed")["widgets_values"][1] == "randomize"
    sampler = smoke_inputs("KSampler")
    assert only(workflow, "type", "KSampler")["widgets_values"] == [
        sampler["seed"], "fixed", sampler["steps"], sampler["cfg"],
        sampler["sampler_name"], sampler["scheduler"], sampler["denoise"],
    ]
    assert only(workflow, "type", "LoaderGGUF")["widgets_values"] == [smoke_inputs("LoaderGGUF")["gguf_name"]]
    clip = smoke_inputs("CLIPLoader")
    assert only(workflow, "type", "CLIPLoader")["widgets_values"] == [clip["clip_name"], clip["type"], clip["device"]]
    assert only(workflow, "type", "VAELoader")["widgets_values"] == [smoke_inputs("VAELoader")["vae_name"]]


@pytest.mark.parametrize(
    ("title", "options"),
    [
        ("mode switch", list(MODES)),
        ("target switch", list(TARGETS)),
        ("distribution switch", list(DISTRIBUTIONS)),
        ("mask shape switch", MASK_SHAPES),
    ],
)
def test_switches_list_their_options_in_node_order(title, options):
    workflow = load()
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    switch = only(workflow, "title", title)
    for index, name in enumerate(options):
        slot = next(i for i in switch["inputs"] if i["name"] == f"value{index}")
        source = nodes[links[slot["link"]][1]]
        assert source["type"] == "PrimitiveString"
        assert source["widgets_values"] == [name]
    # one spare empty slot, no extra wired options
    assert switch["inputs"][len(options)]["link"] is None or switch["inputs"][len(options)]["name"] == "index"


def test_every_master_seed_yields_valid_glitch_shape_and_mask_settings(math_node):
    workflow = load()
    rng = random.Random(20260916)
    seeds = [0, 1, EASY_SEED_MAX] + [rng.randrange(EASY_SEED_MAX + 1) for _ in range(3000)]
    seen = {"mode": set(), "target": set(), "distribution": set(), "shape": set(), "noise_scale": set(), "frames": set()}
    for seed in seeds:
        graph = Graph(workflow, math_node, seed)

        glitch = graph.linked_inputs("GlitchModelKrea2", LESION_DRIVEN)
        recipe = GlitchRecipe.build(enabled=True, **glitch)
        assert recipe.noop_reason() is None
        assert 0.05 <= abs(recipe.strength) <= (2.0 if recipe.mode in ("dropout", "sign_flip") else 4.0), (seed, recipe)
        assert 0.05 <= recipe.probability <= 1.0
        assert recipe.block_end <= 27 and recipe.step_end <= 7
        assert glitch["glitch_seed"] == seed

        shape_values = graph.linked_inputs("GlitchShapeKrea2", SHAPE_DRIVEN)
        shape = GlitchShape.build(**shape_values)
        assert shape.distribution in DISTRIBUTIONS
        assert 0.01 <= shape.spike_density <= 0.5
        assert 1 <= shape.noise_scale <= 16

        mask = graph.linked_inputs("CreateShapeMask", MASK_DRIVEN)
        assert mask["shape"] in MASK_SHAPES
        assert 1 <= mask["frames"] <= 8, (seed, mask)  # each frame is a full-size mask tensor
        assert mask["frame_width"] in (512, 768, 1024) and mask["frame_height"] in (512, 768, 1024)
        assert 0 <= mask["location_x"] < mask["frame_width"], (seed, mask)  # centre stays on canvas
        assert 0 <= mask["location_y"] < mask["frame_height"], (seed, mask)
        assert 8 <= mask["shape_width"] <= mask["frame_width"], (seed, mask)  # KJNodes min is 8
        assert 8 <= mask["shape_height"] <= mask["frame_height"], (seed, mask)
        assert -512 <= mask["grow"] <= 512

        for name, value in (("mode", recipe.mode), ("target", recipe.target), ("distribution", shape.distribution),
                            ("shape", mask["shape"]), ("noise_scale", shape.noise_scale), ("frames", mask["frames"])):
            seen[name].add(value)

    assert seen["mode"] == set(MODES)
    assert seen["target"] == set(TARGETS)
    assert seen["distribution"] == set(DISTRIBUTIONS)
    assert seen["shape"] == set(MASK_SHAPES)
    assert seen["frames"] == set(range(1, 9))
    assert len(seen["noise_scale"]) == 16


def test_distribution_and_target_are_not_correlated(math_node):
    """distribution must not reuse the target randomizer's bit field: v % 6 would determine v % 3."""
    workflow = load()
    rng = random.Random(7)
    pairs = set()
    for seed in [rng.randrange(EASY_SEED_MAX + 1) for _ in range(600)]:
        graph = Graph(workflow, math_node, seed)
        distribution = graph.linked_inputs("GlitchShapeKrea2", ["distribution"])["distribution"]
        target = graph.linked_inputs("GlitchModelKrea2", ["target"])["target"]
        pairs.add((distribution, target))
    assert len(pairs) == len(DISTRIBUTIONS) * len(TARGETS), sorted(pairs)
