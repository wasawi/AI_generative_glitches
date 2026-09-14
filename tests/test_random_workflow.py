import json
import random
from pathlib import Path

import pytest

from lesion_lab.lesions import _splitmix64
from lesion_lab.recipe import MODES, TARGETS, LesionRecipe

ROOT = Path(__file__).resolve().parents[1]
RANDOM_WORKFLOW = ROOT / "workflows" / "krea2-lesion-random.json"
SMOKE_WORKFLOW = ROOT / "workflows" / "krea2-lesion-smoke.json"
EASY_SEED_MAX = 1125899906842624  # comfyui-easy-use py/config.py MAX_SEED_NUM
DRIVEN_INPUTS = [
    "mode", "strength", "probability", "target",
    "block_start", "block_end", "step_start", "step_end", "lesion_seed",
]


def load():
    return json.loads(RANDOM_WORKFLOW.read_text())


def only(workflow, key, value):
    found = [n for n in workflow["nodes"] if n.get(key) == value]
    assert len(found) == 1, f"expected exactly one node with {key}={value!r}, found {len(found)}"
    return found[0]


def smoke_inputs(class_type):
    smoke = json.loads(SMOKE_WORKFLOW.read_text())
    return next(node for node in smoke.values() if node["class_type"] == class_type)["inputs"]


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
                raise AssertionError(f"unexpected node type upstream of the lesion node: {kind}")
            self.cache[key] = value
        return self.cache[key]

    def lesion_inputs(self):
        lesion = next(n for n in self.nodes.values() if n["type"] == "LesionModelKrea2")
        return {name: self.input_value(lesion, name) for name in DRIVEN_INPUTS}


@pytest.fixture(scope="module")
def math_node(comfy_root):
    from comfy_extras.nodes_math import MathExpressionNode

    return MathExpressionNode


def test_links_are_consistent_in_both_directions():
    workflow = load()
    nodes = {n["id"]: n for n in workflow["nodes"]}
    link_ids = set()
    for link_id, origin, origin_slot, target, target_slot, _type in workflow["links"]:
        assert link_id not in link_ids
        link_ids.add(link_id)
        assert link_id in (nodes[origin]["outputs"][origin_slot]["links"] or [])
        assert nodes[target]["inputs"][target_slot]["link"] == link_id
    for node in workflow["nodes"]:
        for slot in node.get("inputs", []):
            assert slot.get("link") is None or slot["link"] in link_ids
        for slot in node.get("outputs", []):
            assert set(slot.get("links") or []) <= link_ids
    assert workflow["last_link_id"] == max(link_ids)
    assert workflow["last_node_id"] == max(nodes)


def test_every_lesion_input_except_enabled_is_driven():
    lesion = only(load(), "type", "LesionModelKrea2")
    inputs = {i["name"]: i for i in lesion["inputs"]}
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
    for seed in seeds:
        values = Graph(workflow, math_node, seed).lesion_inputs()
        assert values["lesion_seed"] == seed
        recipe = LesionRecipe.build(enabled=True, **values)
        assert recipe.noop_reason() is None
        cap = 1.0 if recipe.mode in ("dropout", "sign_flip") else 2.0
        assert 0.05 <= recipe.strength <= cap, (seed, recipe)
        assert 0.05 <= recipe.probability <= 1.0, (seed, recipe)
        assert recipe.block_end <= 27 and recipe.step_end <= 7, (seed, recipe)
        top_strength[recipe.mode] = max(top_strength[recipe.mode], recipe.strength)
        for name in seen:
            seen[name].add(getattr(recipe, name))
    assert seen["mode"] == set(MODES)
    assert seen["target"] == set(TARGETS)
    assert seen["block_start"] == seen["block_end"] == set(range(28))
    assert seen["step_start"] == seen["step_end"] == set(range(8))
    assert top_strength["amplify"] > 1.9 and top_strength["noise"] > 1.9
    assert top_strength["dropout"] > 0.95 and top_strength["sign_flip"] > 0.95
