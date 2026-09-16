import json
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "krea2-glitch-smoke.json"
LESION_WIDGETS = [
    "enabled", "mode", "strength", "probability", "target",
    "block_start", "block_end", "step_start", "step_end", "glitch_seed",
]


def load():
    return json.loads(WORKFLOW.read_text())


def test_every_node_is_api_format_and_every_link_resolves():
    workflow = load()
    for node_id, node in workflow.items():
        assert isinstance(node["class_type"], str) and isinstance(node["inputs"], dict), node_id
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in workflow, f"{node_id} links to missing node {value[0]}"


def test_glitch_node_sits_between_loader_and_sampler_and_starts_disabled():
    workflow = load()
    glitch_id, glitch = next((k, v) for k, v in workflow.items() if v["class_type"] == "GlitchModelKrea2")
    assert workflow[glitch["inputs"]["model"][0]]["class_type"] == "LoaderGGUF"
    assert sorted(k for k in glitch["inputs"] if k != "model") == sorted(LESION_WIDGETS)
    assert glitch["inputs"]["enabled"] is False
    sampler = next(v for v in workflow.values() if v["class_type"] == "KSampler")
    assert sampler["inputs"]["model"] == [glitch_id, 0]
