"""Canonical SMPL-X template T-pose frame generation.

Creates a synthetic calibration frame from the SMPL-X body model's zero-pose,
providing joint world positions and world quaternions that serve as the ground
truth for solving per-body rotation offsets in GMR IK configs.

Unlike motion-sequence frames (which carry root orientation from the capture),
the template frame uses identity root orientation and zero body pose — this is
the canonical T-pose that the SMPL-X model defines mathematically.

Requires ``smplx``, ``torch``, and ``scipy`` (lazy imports).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_body_model_path(model_path: Path | str | None = None) -> Path:
    """Resolve a SMPL-X body model path for ``smplx.create()``.

    Returns a path that ``smplx.create(path, "smplx")`` can consume directly.
    Because ``smplx.create`` appends the model type subfolder when *path* is a
    directory, this function always returns the **parent** directory (i.e.
    ``body_models/``) rather than the ``smplx/`` subfolder itself.

    Resolution rules:

    - A ``.npz`` model file: returned as-is — ``smplx.create`` detects the
      model type from the filename and loads it directly regardless of gender.
    - A directory containing a ``smplx/`` subfolder: returns the directory
      itself (``smplx.create`` appends ``smplx/`` internally).
    - A directory named ``smplx``: returns its parent (same reason).
    - Any other directory: returned as-is (``smplx.create`` will error if no
      matching model file exists).
    - ``None``: auto-discovers via ``GMR_ROOT/assets/body_models`` using
      :func:`roboharness.alignment._gmr_path.find_gmr_root`.

    Raises ``FileNotFoundError`` if resolution fails.
    """
    if model_path is not None:
        p = Path(model_path).resolve()
        if p.is_file():
            return p
        if p.is_dir():
            smplx_sub = p / "smplx"
            if smplx_sub.is_dir():
                return p
            if p.name == "smplx":
                return p.parent
            return p
        raise FileNotFoundError(f"Body model path does not exist: {p}")

    from roboharness.alignment._gmr_path import find_gmr_root

    gmr_root = find_gmr_root()
    candidate = gmr_root / "assets" / "body_models"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Cannot auto-discover SMPL-X body models. "
        f"Tried {candidate}. "
        f"Set GMR_ROOT or pass --smplx_template_model explicitly."
    )


_REQUIRED_JOINTS = frozenset(
    {
        "pelvis",
        "spine3",
        "left_hip",
        "right_hip",
        "left_shoulder",
        "right_shoulder",
        "left_foot",
        "right_foot",
    }
)


def load_smplx_template_tpose(
    body_model_path: Path | str | None = None,
    gender: str = "male",
    betas: np.ndarray | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Create a GMR-compatible SMPL-X frame from the body model zero-pose.

    Parameters
    ----------
    body_model_path:
        Path to the SMPL-X body model. Accepts:
        - A directory (e.g. ``body_models/``) containing ``smplx/`` subfolder.
        - The ``smplx/`` subfolder itself.
        - A ``.npz`` model file (e.g. ``SMPLX_MALE.npz``).
        - ``None`` to auto-discover via ``GMR_ROOT/assets/body_models``.
    gender:
        ``"male"`` or ``"female"`` or ``"neutral"``.
    betas:
        Optional shape coefficients (length 10). Defaults to zeros.

    Returns
    -------
    frame:
        ``{joint_name: (position_3d, quat_wxyz)}`` where quaternions are
        scalar-first ``[w, x, y, z]`` and positions are in **Z-up MuJoCo
        coordinates** (X=forward, Y=left, Z=up).  The Y-up → Z-up conversion
        is applied internally so callers receive data already in the robot's
        frame convention.
    human_height:
        Deterministic height estimate for scaling.
    """
    import smplx  # type: ignore[import-untyped]
    import torch
    from scipy.spatial.transform import Rotation as R
    from smplx.joint_names import JOINT_NAMES  # type: ignore[import-untyped]

    body_model_resolved = resolve_body_model_path(body_model_path)
    if body_model_resolved.is_file():
        bm = smplx.SMPLX(
            str(body_model_resolved),
            gender=gender,
            use_pca=False,
        )
    else:
        bm = smplx.create(
            str(body_model_resolved),
            "smplx",
            gender=gender,
            use_pca=False,
        )

    if betas is None:
        betas = np.zeros(getattr(bm, "num_betas", 10), dtype=np.float32)
    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)

    num_frames = 1
    out = bm(
        betas=betas_t,
        global_orient=torch.zeros(num_frames, 3),
        body_pose=torch.zeros(num_frames, 63),
        transl=torch.zeros(num_frames, 3),
        left_hand_pose=torch.zeros(num_frames, 45),
        right_hand_pose=torch.zeros(num_frames, 45),
        jaw_pose=torch.zeros(num_frames, 3),
        leye_pose=torch.zeros(num_frames, 3),
        reye_pose=torch.zeros(num_frames, 3),
        return_full_pose=True,
    )

    joints = out.joints[0].detach().numpy()
    full_body_pose = out.full_pose[0].reshape(-1, 3).detach().numpy()
    global_orient = full_body_pose[0]

    min_y = float(joints[:, 1].min())
    joints[:, 1] -= min_y

    parents = bm.parents
    joint_names = JOINT_NAMES[: len(parents)]

    frame: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    joint_orientations: list[R] = []

    for i, joint_name in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(global_orient)
        else:
            rot = joint_orientations[int(parents[i])] * R.from_rotvec(full_body_pose[i].squeeze())
        joint_orientations.append(rot)
        pos = joints[i].copy()
        quat = rot.as_quat(scalar_first=True)
        frame[joint_name] = (pos, quat)

    from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_frame

    frame = smpl_to_mujoco_frame(frame)

    height = 1.66 + 0.1 * float(betas[0])

    return frame, float(height)
