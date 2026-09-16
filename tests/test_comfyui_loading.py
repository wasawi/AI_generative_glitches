import importlib.util
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUTS = [
    "model", "enabled", "mode", "strength", "probability", "target",
    "block_start", "block_end", "step_start", "step_end", "glitch_seed",
]


def test_package_imports_the_way_comfyui_loads_custom_nodes(comfy_root):
    # Mirrors nodes.load_custom_node in ComfyUI 0.35.1 for a directory module.
    module_path = str(PACKAGE_ROOT)
    sys_module_name = module_path.replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(sys_module_name, os.path.join(module_path, "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[sys_module_name] = module
    try:
        spec.loader.exec_module(module)
        node_class = module.NODE_CLASS_MAPPINGS["GlitchModelKrea2"]
        assert module.NODE_DISPLAY_NAME_MAPPINGS["GlitchModelKrea2"] == "Glitch Model (Krea2)"
        assert node_class.CATEGORY == "experimental/glitches"
        assert node_class.RETURN_TYPES == ("MODEL", "STRING")
        assert node_class.RETURN_NAMES == ("model", "recipe")
        required = node_class.INPUT_TYPES()["required"]
        assert list(required) == EXPECTED_INPUTS
        assert required["mode"][0] == ["dropout", "amplify", "sign_flip", "noise"]
        assert required["mode"][1]["default"] == "noise"
        assert required["glitch_seed"][1]["max"] == 2**63 - 1
        assert required["strength"][1]["min"] < 0 and required["strength"][1]["max"] >= 1_000_000
        assert required["probability"][1]["step"] <= 0.0001
        assert required["block_start"][1]["max"] == 27 and required["block_end"][1]["max"] == 27
        assert node_class.INPUT_TYPES()["optional"] == {"shape": ("GLITCH_SHAPE",)}
        shape_class = module.NODE_CLASS_MAPPINGS["GlitchShapeKrea2"]
        assert module.NODE_DISPLAY_NAME_MAPPINGS["GlitchShapeKrea2"] == "Glitch Shape (Krea2)"
        assert shape_class.CATEGORY == "experimental/glitches"
        assert shape_class.RETURN_TYPES == ("GLITCH_SHAPE",)
    finally:
        for name in [n for n in sys.modules if n == sys_module_name or n.startswith(sys_module_name + ".")]:
            del sys.modules[name]


def test_old_node_names_still_load_but_are_hidden(comfy_root):
    """Workflows and PNGs saved before the rename reference the old registry keys; they must keep
    working, while only the new names show up in the node search (DEPRECATED hides them)."""
    module_path = str(PACKAGE_ROOT)
    sys_module_name = module_path.replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(sys_module_name, os.path.join(module_path, "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[sys_module_name] = module
    try:
        spec.loader.exec_module(module)
        mappings = module.NODE_CLASS_MAPPINGS
        for old, new in (("LesionModelKrea2", "GlitchModelKrea2"), ("LesionShapeKrea2", "GlitchShapeKrea2")):
            assert issubclass(mappings[old], mappings[new])
            assert mappings[old].DEPRECATED is True
            assert getattr(mappings[new], "DEPRECATED", False) is False
            assert mappings[old].RETURN_TYPES == mappings[new].RETURN_TYPES
            assert old in module.NODE_DISPLAY_NAME_MAPPINGS
    finally:
        for name in [n for n in sys.modules if n == sys_module_name or n.startswith(sys_module_name + ".")]:
            del sys.modules[name]
