import importlib.util
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUTS = [
    "model", "enabled", "mode", "strength", "probability", "target",
    "block_start", "block_end", "step_start", "step_end", "lesion_seed",
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
        node_class = module.NODE_CLASS_MAPPINGS["LesionModelKrea2"]
        assert module.NODE_DISPLAY_NAME_MAPPINGS["LesionModelKrea2"] == "Lesion Model (Krea2)"
        assert node_class.CATEGORY == "experimental/lesion-lab"
        assert node_class.RETURN_TYPES == ("MODEL", "STRING")
        assert node_class.RETURN_NAMES == ("model", "recipe")
        required = node_class.INPUT_TYPES()["required"]
        assert list(required) == EXPECTED_INPUTS
        assert required["mode"][0] == ["dropout", "amplify", "sign_flip", "noise"]
        assert required["mode"][1]["default"] == "noise"
        assert required["lesion_seed"][1]["max"] == 2**63 - 1
    finally:
        for name in [n for n in sys.modules if n == sys_module_name or n.startswith(sys_module_name + ".")]:
            del sys.modules[name]
