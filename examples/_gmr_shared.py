"""Shared GMR integration utilities for example scripts.

Internal helper for ``examples/gmr_*.py`` scripts. Extracts common logic
so that retarget loaders, replay backends, and path setup are defined once.

Requires GMR as a sibling directory and ``mujoco`` (lazy imports).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from roboharness.alignment.skeleton_maps import (
    BVH_SKELETON,
    SMPLX_SKELETON,
)

_HERE = Path(__file__).resolve().parent
GMR_ROOT = _HERE.parent.parent / "GMR"

if not GMR_ROOT.exists():
    raise RuntimeError(
        f"GMR not found at {GMR_ROOT}.\nPlace GMR/ next to roboharness/, or edit GMR_ROOT."
    )

sys.path.insert(0, str(GMR_ROOT))

_ROBOHARNESS_SRC = _HERE.parent / "src"
if _ROBOHARNESS_SRC.exists() and str(_ROBOHARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(_ROBOHARNESS_SRC))

CAM_AZIMUTHS = {
    "inspect_front": 0.0,
    "inspect_side": 90.0,
    "inspect_back": 180.0,
}

CAM_ELEVATION = -20.0

# SMPL-X world-frame convention: the SMPL model uses Y-up right-handed
# with +X facing the model's left, while MuJoCo uses Z-up.  Pre-multiplying
# human model bone orientations by this quaternion (equivalent to a 180°
# rotation about the <-1,-1,-1> axis) converts them into the robot's Z-up
# frame, so that per-body quaternion offsets and T-pose specs are consistent.
HUMAN_RGBA = (0.2, 0.85, 1.0, 0.75)
HUMAN_SPHERE_RADIUS = 0.035
HUMAN_EDGE_WIDTH = 0.012

SPEC_TPOSE_RGBA = (0.2, 1.0, 0.3, 0.65)
SPEC_TPOSE_SPHERE_RADIUS = 0.03
SPEC_TPOSE_EDGE_WIDTH = 0.01

# Combined skeleton edges for overlay rendering (both BVH and SMPL-X conventions).
_SKELETON_EDGES = SMPLX_SKELETON.edges + BVH_SKELETON.edges


def find_root_body(xml_path: Path) -> str:
    content = xml_path.read_text()
    matches = re.findall(r'<body\s+name="([^"]+)"', content)
    return matches[0] if matches else "pelvis"


def load_bvh(bvh_file: str, bvh_format: str) -> tuple[list, float, int]:
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
    from general_motion_retargeting.utils.smpl import (
        get_smplx_data_offline_fast,
        load_smplx_file,
    )

    from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_frame

    smplx_body_model_path = GMR_ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        npz_file, smplx_body_model_path
    )
    tgt_fps = 30
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    frames = [smpl_to_mujoco_frame(f) for f in frames]
    print(
        f"[smplx] Loaded: {len(frames)} frames @ {aligned_fps} fps"
        f"  height={human_height:.2f} m  (Z-up)"
    )
    return frames, human_height, aligned_fps


def load_fbx_offline(pkl_file: str) -> tuple[list, float, int]:
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


class GMRReplayBackend:
    """Replays a pre-computed qpos sequence through MuJoCo without physics stepping."""

    def __init__(
        self,
        xml_path: Path,
        qpos_seq: np.ndarray,
        cameras: list[str],
        root_body_name: str = "pelvis",
        cam_distance: float = 2.5,
        use_meshcat: bool = False,
        human_seq: list[dict] | None = None,
        tpose_spec: dict | None = None,
    ) -> None:
        import mujoco

        from roboharness.core.capture import CameraView

        self._qpos_seq = qpos_seq
        self._frame = 0
        self._cam_distance = cam_distance
        self._CameraView = CameraView
        self._human_seq = human_seq
        self._skeleton_debug_logged = False

        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=480, width=640)

        self._root_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, root_body_name
        )
        if self._root_body_id < 0:
            self._root_body_id = 1

        if use_meshcat:
            try:
                import meshcat

                self._meshcat_vis = meshcat.Visualizer()
                print(f"[meshcat] Interactive viewer: {self._meshcat_vis.url()}")
                print("[meshcat] Open the URL above in your browser to orbit the robot.")
            except ImportError:
                print("[meshcat] meshcat not installed — skipping interactive viewer.")
                self._meshcat_vis = None
        else:
            self._meshcat_vis = None

        self._tpose_spec = tpose_spec
        self._spec_body_edges: list[tuple[str, str]] = []
        self._spec_body_positions: dict[str, np.ndarray] = {}
        if tpose_spec is not None and "links" in tpose_spec and "qpos" in tpose_spec:
            self._build_spec_overlay_data(tpose_spec)

    def _lookat(self) -> np.ndarray:
        return self._data.xpos[self._root_body_id].copy()

    def _build_spec_overlay_data(self, tpose_spec: dict) -> None:
        import mujoco

        spec_qpos = np.asarray(tpose_spec["qpos"], dtype=np.float64)

        # Build a name->parent map from the already-loaded model
        body_name_to_parent: dict[str, str | None] = {}
        for i in range(1, self._model.nbody):  # skip world (id 0)
            body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name is None:
                continue
            parent_id = self._model.body_parentid[i]
            if parent_id > 0:
                parent_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
                body_name_to_parent[body_name] = parent_name
            else:
                body_name_to_parent[body_name] = None

        spec_links: dict[str, dict] = tpose_spec.get("links", {})
        # Use spec["links"]["pos"] directly (already world-frame positions)
        for link_name, frame in spec_links.items():
            self._spec_body_positions[link_name] = np.asarray(frame["pos"], dtype=np.float64)

        # Build edges: connect each spec link to its parent if both are in the spec
        for link_name in spec_links:
            parent = body_name_to_parent.get(link_name)
            if parent is not None and parent in spec_links:
                self._spec_body_edges.append((parent, link_name))

        spec_qpos_num = len(spec_qpos)
        model_nq = self._model.nq
        if spec_qpos_num == model_nq:
            print(f"[spec_overlay] {len(spec_links)} bodies, {len(self._spec_body_edges)} edges")
        else:
            print(
                f"[spec_overlay] qpos length mismatch (spec={spec_qpos_num}, model={model_nq}) — "
                "using spec link positions directly"
            )

    def _add_spec_tpose_geoms(self) -> None:
        import mujoco

        if not self._spec_body_positions or not self._tpose_spec:
            return

        scene = self._renderer.scene
        rgba = np.array(SPEC_TPOSE_RGBA, dtype=np.float32)
        identity = np.eye(3).flatten()
        sphere_size = np.array([SPEC_TPOSE_SPHERE_RADIUS, 0.0, 0.0])

        for _link_name, pos in self._spec_body_positions.items():
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=sphere_size,
                pos=pos.copy(),
                mat=identity,
                rgba=rgba,
            )
            scene.ngeom += 1

        for parent, child in self._spec_body_edges:
            if parent not in self._spec_body_positions or child not in self._spec_body_positions:
                continue
            if scene.ngeom >= scene.maxgeom:
                return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=sphere_size,
                pos=np.zeros(3),
                mat=identity,
                rgba=rgba,
            )
            mujoco.mjv_connector(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=SPEC_TPOSE_EDGE_WIDTH,
                from_=self._spec_body_positions[parent],
                to=self._spec_body_positions[child],
            )
            scene.ngeom += 1

    def _add_human_skeleton_geoms(self) -> None:
        import mujoco

        if self._human_seq is None:
            return
        idx = min(self._frame - 1, len(self._human_seq) - 1)
        if idx < 0:
            return
        frame = self._human_seq[idx]
        scene = self._renderer.scene
        rgba = np.array(HUMAN_RGBA, dtype=np.float32)
        identity = np.eye(3).flatten()
        sphere_size = np.array([HUMAN_SPHERE_RADIUS, 0.0, 0.0])

        for _bone_name, (pos, _quat) in frame.items():
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=sphere_size,
                pos=np.asarray(pos, dtype=np.float64),
                mat=identity,
                rgba=rgba,
            )
            scene.ngeom += 1

        drawn_edges = 0
        for parent, child in _SKELETON_EDGES:
            if parent not in frame or child not in frame:
                continue
            if scene.ngeom >= scene.maxgeom:
                return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=sphere_size,
                pos=np.zeros(3),
                mat=identity,
                rgba=rgba,
            )
            mujoco.mjv_connector(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=HUMAN_EDGE_WIDTH,
                from_=np.asarray(frame[parent][0], dtype=np.float64),
                to=np.asarray(frame[child][0], dtype=np.float64),
            )
            scene.ngeom += 1
            drawn_edges += 1

        if not self._skeleton_debug_logged:
            self._skeleton_debug_logged = True
            print(
                f"[skeleton] bones={len(frame)} edges_drawn={drawn_edges} "
                f"sample_bones={list(frame.keys())[:6]}"
            )

    def _capture_free_cam(self, camera_name: str):
        import mujoco

        azimuth = CAM_AZIMUTHS.get(camera_name, 90.0)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self._lookat()
        cam.distance = self._cam_distance
        cam.azimuth = azimuth
        cam.elevation = CAM_ELEVATION

        self._renderer.update_scene(self._data, camera=cam)
        if self._tpose_spec is not None:
            self._add_spec_tpose_geoms()
        else:
            self._add_human_skeleton_geoms()
        rgb = self._renderer.render().copy()

        return self._CameraView(name=camera_name, rgb=rgb)

    def step(self, action: Any) -> dict[str, Any]:
        import mujoco

        idx = min(self._frame, len(self._qpos_seq) - 1)
        qpos = self._qpos_seq[idx]
        np.copyto(self._data.qpos, qpos)
        self._data.time = idx / 30.0
        mujoco.mj_forward(self._model, self._data)
        self._frame += 1
        return self.get_state()

    def reset(self) -> dict[str, Any]:
        import mujoco

        self._frame = 0
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        return {
            "time": float(self._data.time),
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
        }

    def save_state(self) -> dict[str, Any]:
        import mujoco

        state_size = mujoco.mj_stateSize(self._model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        state = np.empty(state_size, dtype=np.float64)
        mujoco.mj_getState(self._model, self._data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return {"mujoco_state": state, "time": float(self._data.time)}

    def restore_state(self, state: dict[str, Any]) -> None:
        import mujoco

        mujoco.mj_setState(
            self._model, self._data, state["mujoco_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS
        )

    def capture_camera(self, camera_name: str):
        return self._capture_free_cam(camera_name)

    def get_sim_time(self) -> float:
        return float(self._data.time)

    def cleanup(self) -> None:
        """No-op — no temp files written in this implementation."""
