"""GMR Replay Backend — replays qpos sequences through MuJoCo for visual inspection.

Extracted from ``examples/_gmr_shared.py`` as a standalone module with proper
package imports and no sys.path hacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gmr_harness._capture import CameraView

from gmr_harness.gmr_integration import (
    _SKELETON_EDGES,
    CAM_AZIMUTHS,
    CAM_ELEVATION,
    HUMAN_EDGE_WIDTH,
    HUMAN_RGBA,
    HUMAN_SPHERE_RADIUS,
    SPEC_TPOSE_EDGE_WIDTH,
    SPEC_TPOSE_RGBA,
    SPEC_TPOSE_SPHERE_RADIUS,
)


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

        self._qpos_seq = qpos_seq
        self._frame = 0
        self._cam_distance = cam_distance
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
            except ImportError:
                print("[meshcat] meshcat not installed - skipping interactive viewer.")
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

        body_name_to_parent: dict[str, str | None] = {}
        for i in range(1, self._model.nbody):
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
        for link_name, frame in spec_links.items():
            self._spec_body_positions[link_name] = np.asarray(frame["pos"], dtype=np.float64)

        for link_name in spec_links:
            parent = body_name_to_parent.get(link_name)
            if parent is not None and parent in spec_links:
                self._spec_body_edges.append((parent, link_name))

        spec_qpos_num = len(spec_qpos)
        model_nq = self._model.nq
        if spec_qpos_num == model_nq:
            print(f"[spec_overlay] {len(spec_links)} bodies, {len(self._spec_body_edges)} edges")
        else:
            print(f"[spec_overlay] qpos length mismatch (spec={spec_qpos_num}, model={model_nq})")

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

    def _capture_free_cam(self, camera_name: str) -> Any:
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

        return CameraView(name=camera_name, rgb=rgb)

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
        pass
