import json
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


def load_workflow(name):
    return json.loads((WORKFLOWS / name).read_text())


def only(workflow, key, value):
    found = [n for n in workflow["nodes"] if n.get(key) == value]
    assert len(found) == 1, f"expected exactly one node with {key}={value!r}, found {len(found)}"
    return found[0]


def smoke_inputs(class_type):
    smoke = load_workflow("krea2-glitch-smoke.json")
    return next(node for node in smoke.values() if node["class_type"] == class_type)["inputs"]


def assert_links_consistent(workflow):
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
