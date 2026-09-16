import random

import pytest

from glitches.effects import _splitmix64
from glitches.recipe import MODES, TARGETS, GlitchRecipe
from workflow_helpers import assert_links_consistent, load_workflow, only, smoke_inputs

EASY_SEED_MAX = 1125899906842624  # comfyui-easy-use py/config.py MAX_SEED_NUM
DRIVEN_INPUTS = [
    "mode", "strength", "probability", "target",
    "block_start", "block_end", "step_start", "step_end", "glitch_seed",
]


def load():
    return load_workflow("krea2-glitch-random.json")


class Graph:
    """Evaluates the randomizer part of the UI-format workflow for one master seed.

    Math Expression nodes run through ComfyUI's real node implementation; the other
    upstream nodes are plain pass-throughs (seed value, string widget, index switch).
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
                raise AssertionError(f"unexpected node type upstream of the glitch node: {kind}")
            self.cache[key] = value
        return self.cache[key]

    def glitch_inputs(self):
        glitch = next(n for n in self.nodes.values() if n["type"] == "GlitchModelKrea2")
        return {name: self.input_value(glitch, name) for name in DRIVEN_INPUTS}


@pytest.fixture(scope="module")
def math_node(comfy_root):
    from comfy_extras.nodes_math import MathExpressionNode

    return MathExpressionNode


def test_links_are_consistent_in_both_directions():
    assert_links_consistent(load())


def test_every_glitch_input_except_enabled_is_driven():
    glitch = only(load(), "type", "GlitchModelKrea2")
    inputs = {i["name"]: i for i in glitch["inputs"]}
    assert inputs["enabled"]["link"] is None
    for name in DRIVEN_INPUTS:
        assert inputs[name]["link"] is not None, name


def test_master_seed_randomizes_while_the_rest_matches_the_smoke_workflow():
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


def test_index_switches_list_modes_and_targets_in_recipe_order():
    workflow = load()
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    for title, expected in [("mode switch", MODES), ("target switch", TARGETS)]:
        switch = only(workflow, "title", title)
        for index, name in enumerate(expected):
            slot = next(i for i in switch["inputs"] if i["name"] == f"value{index}")
            source = nodes[links[slot["link"]][1]]
            assert source["type"] == "PrimitiveString"
            assert source["widgets_values"] == [name]


def test_seed_mixer_is_splitmix64(math_node):
    workflow = load()
    mixer = only(workflow, "title", "seed mixer 3/3")
    rng = random.Random(1)
    for seed in [0, 1, EASY_SEED_MAX] + [rng.randrange(EASY_SEED_MAX + 1) for _ in range(300)]:
        assert Graph(workflow, math_node, seed).output(mixer["id"], 1) == _splitmix64(seed)


def test_every_master_seed_yields_a_valid_active_recipe_covering_all_choices(math_node):
    workflow = load()
    rng = random.Random(20260915)
    seeds = [0, 1, EASY_SEED_MAX] + [rng.randrange(EASY_SEED_MAX + 1) for _ in range(20000)]
    seen = {name: set() for name in ("mode", "target", "block_start", "block_end", "step_start", "step_end")}
    top_strength = {mode: 0.0 for mode in MODES}
    strengths = []
    for seed in seeds:
        values = Graph(workflow, math_node, seed).glitch_inputs()
        assert values["glitch_seed"] == seed
        recipe = GlitchRecipe.build(enabled=True, **values)
        assert recipe.noop_reason() is None
        cap = 2.0 if recipe.mode in ("dropout", "sign_flip") else 4.0
        assert 0.05 <= abs(recipe.strength) <= cap, (seed, recipe)  # signed draw
        assert 0.05 <= recipe.probability <= 1.0, (seed, recipe)
        assert recipe.block_end <= 27 and recipe.step_end <= 7, (seed, recipe)
        top_strength[recipe.mode] = max(top_strength[recipe.mode], abs(recipe.strength))
        strengths.append(recipe.strength)
        for name in seen:
            seen[name].add(getattr(recipe, name))
    assert seen["mode"] == set(MODES)
    assert seen["target"] == set(TARGETS)
    assert seen["block_start"] == seen["block_end"] == set(range(28))
    assert seen["step_start"] == seen["step_end"] == set(range(8))
    assert top_strength["amplify"] > 3.9 and top_strength["noise"] > 3.9
    assert top_strength["dropout"] > 1.9 and top_strength["sign_flip"] > 1.9
    assert min(strengths) < 0 < max(strengths), "the signed draw must produce both signs"
