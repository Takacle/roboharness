"""GMR integration utilities — motion loading, body matching, GMR path resolution.

Consolidates the former ``examples/_gmr_shared.py`` into a proper package module
with unified GMR_ROOT resolution (no sys.path hacks) and standard package imports.

Requires GMR as an external repo (non-pip dependency). See README for setup.
"""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from gmr_harness.alignment._gmr_path import find_gmr_root
from gmr_harness.alignment.skeleton_maps import (
    BVH_SKELETON,
    SMPLX_SKELETON,
)


def get_gmr_root() -> Path:
    """Lazily resolve GMR_ROOT. Safe to call at module level in non-GMR environments."""
    return find_gmr_root()


CAM_AZIMUTHS = {
    "inspect_front": 0.0,
    "inspect_side": 90.0,
    "inspect_back": 180.0,
}

CAM_ELEVATION = -20.0

HUMAN_RGBA = (0.2, 0.85, 1.0, 0.75)
HUMAN_SPHERE_RADIUS = 0.035
HUMAN_EDGE_WIDTH = 0.012

SPEC_TPOSE_RGBA = (0.2, 1.0, 0.3, 0.65)
SPEC_TPOSE_SPHERE_RADIUS = 0.03
SPEC_TPOSE_EDGE_WIDTH = 0.01

_SKELETON_EDGES = SMPLX_SKELETON.edges + BVH_SKELETON.edges

_GMR_INSTALL_HELP = """
GMR (general_motion_retargeting) not found.

Setup options:
  1. Clone GMR next to your project:
     git clone <GMR_URL> ../GMR
  2. Set environment variable:
     export GMR_ROOT=/path/to/GMR

GMR must contain: general_motion_retargeting/params.py
"""


def _require_gmr(feature: str = "") -> Any:
    """Import GMR safely without polluting sys.path.

    Returns the ``general_motion_retargeting`` top-level module.
    Raises ``SystemExit`` with install instructions on failure.
    """
    gmr_root = find_gmr_root()
    gmr_pkg = gmr_root / "general_motion_retargeting"
    init_py = gmr_pkg / "__init__.py"
    if not init_py.exists():
        hint = f" (required for: {feature})" if feature else ""
        raise SystemExit(f"{_GMR_INSTALL_HELP}\n{hint}")

    if "general_motion_retargeting" in __import__("sys").modules:
        return __import__("sys").modules["general_motion_retargeting"]

    spec = importlib.util.spec_from_file_location(
        "general_motion_retargeting",
        str(init_py),
        submodule_search_locations=[str(gmr_pkg)],
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load GMR from {gmr_pkg}")
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules["general_motion_retargeting"] = module
    spec.loader.exec_module(module)
    return module


def _require_gmr_submodule(submodule_path: str, feature: str = "") -> Any:
    """Import a GMR submodule via importlib, avoiding sys.path insertion."""
    _require_gmr(feature)
    gmr_root = find_gmr_root()
    parts = submodule_path.split(".")
    py_file = gmr_root / "general_motion_retargeting" / "/".join(parts[1:]) / ".py"
    if not py_file.exists():
        py_file = (
            gmr_root / "general_motion_retargeting" / "/".join(parts[1:-1]) / f"{parts[-1]}.py"
        )
    mod = __import__("importlib").import_module(submodule_path)
    return mod


def find_root_body(xml_path: Path) -> str:
    from gmr_harness.alignment.orientation_aligner import _resolve_includes

    root = ET.parse(str(xml_path)).getroot()
    _resolve_includes(root, xml_path.parent)
    worldbody = root.find("worldbody")
    root_body = worldbody.find("body") if worldbody is not None else None
    if root_body is not None:
        name = root_body.attrib.get("name")
        if name:
            return name
    return "pelvis"


def check_smplx_config_before_retarget(robot: str, src: str) -> None:
    """Validate SMPL-X IK config before constructing a GMR retargeter."""
    if src != "smplx":
        return

    from gmr_harness.alignment.smplx_coordinate import validate_smplx_runtime_config

    try:
        gmr = _require_gmr("SMPL-X config validation")
        params = gmr.params
        cfg_path = params.IK_CONFIG_DICT.get("smplx", {}).get(robot, "")
    except (ImportError, AttributeError, SystemExit):
        return
    if not cfg_path:
        return
    p = Path(str(cfg_path))
    if not p.exists():
        return
    with p.open() as f:
        validate_smplx_runtime_config(json.load(f), p)


def load_bvh(bvh_file: str, bvh_format: str) -> tuple[list, float, int]:
    _require_gmr("BVH motion loading")
    from general_motion_retargeting.utils.lafan1 import load_lafan1_file
    from general_motion_retargeting.utils.soma import detect_soma_bvh, load_soma_bvh_file

    if bvh_format == "soma":
        frames, h, fps = load_soma_bvh_file(bvh_file)
        print(f"[bvh] Loaded SOMA format: {len(frames)} frames @ {fps} fps")
        return frames, h, fps

    if bvh_format == "lafan1":
        frames, h = load_lafan1_file(bvh_file)
        print(f"[bvh] Loaded LAFAN1 format: {len(frames)} frames @ 30 fps")
        return frames, h, 30

    if detect_soma_bvh(bvh_file):
        frames, h, fps = load_soma_bvh_file(bvh_file)
        print(f"[bvh] Auto-detected SOMA format: {len(frames)} frames @ {fps} fps")
        return frames, h, fps

    frames, h = load_lafan1_file(bvh_file)
    print(f"[bvh] Auto-detected LAFAN1 format: {len(frames)} frames @ 30 fps")
    return frames, h, 30


def load_smplx(npz_file: str) -> tuple[list, float, int]:
    _require_gmr("SMPL-X motion loading")
    from general_motion_retargeting.utils.smpl import (
        get_smplx_data_offline_fast,
        load_smplx_file,
    )

    from gmr_harness.alignment.smplx_coordinate import (
        classify_smplx_frame_convention,
        smpl_to_mujoco_frame,
    )

    smplx_body_model_path = get_gmr_root() / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        npz_file, smplx_body_model_path
    )
    tgt_fps = 30
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    convention = classify_smplx_frame_convention(frames)
    if convention == "y":
        frames = [smpl_to_mujoco_frame(f) for f in frames]
        print(
            f"[smplx] Loaded: {len(frames)} frames @ {aligned_fps} fps"
            f"  height={human_height:.2f} m  (Y-up -> Z-up converted)"
        )
    else:
        print(
            f"[smplx] Loaded: {len(frames)} frames @ {aligned_fps} fps"
            f"  height={human_height:.2f} m  (AMASS Z-up, no conversion)"
        )

    return frames, human_height, aligned_fps


def load_fbx_offline(pkl_file: str) -> tuple[list, float, int]:
    _require_gmr("FBX offline loading")
    from general_motion_retargeting.utils.fbx_offline import load_fbx_offline_file

    frames, human_height, fps = load_fbx_offline_file(pkl_file)
    print(f"[fbx_offline] Loaded: {len(frames)} frames @ {fps} fps  height={human_height:.2f} m")
    return frames, human_height, fps


def load_motion(src: str, motion_file: str, bvh_format: str = "auto") -> tuple[list, float, int]:
    if src == "bvh":
        return load_bvh(motion_file, bvh_format)
    if src == "smplx":
        return load_smplx(motion_file)
    if src == "fbx_offline":
        return load_fbx_offline(motion_file)
    raise ValueError(f"Unknown src: {src!r}; choose from bvh, smplx, fbx_offline")


def scaled_human_reference(retargeter: Any, raw_frame: dict) -> dict:
    data = retargeter.to_numpy(raw_frame)
    data = retargeter.scale_human_data(
        data, retargeter.human_root_name, retargeter.human_scale_table
    )
    data = retargeter.apply_world_rotation(data)
    return {k: (p.copy(), q.copy()) for k, (p, q) in data.items()}
