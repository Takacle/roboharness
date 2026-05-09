"""Scale normalization for SMPL-X template frames.

Applies per-bone scale factors (``human_scale_table``) and the height-ratio
correction to a template frame's positions so that effective bone lengths
match the IK solver's expectations.

Orientations are preserved unchanged.
"""

from __future__ import annotations

import numpy as np


def apply_human_scale(
    frame: dict[str, tuple[np.ndarray, np.ndarray]],
    scale_table: dict[str, float],
    *,
    human_root_name: str = "pelvis",
    height_assumption: float = 1.8,
    human_height: float = 1.66,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Scale template frame positions by bone scale factors.

    The root joint position is scaled in-place. All other joints are scaled
    relative to the root: the vector from root to child is multiplied by the
    child's scale factor.

    Parameters
    ----------
    frame:
        ``{joint_name: (position, quat)}`` — typically a Y-up template frame.
    scale_table:
        Per-bone scale factors (``{bone_name: float}``).
    human_root_name:
        Name of the root joint in the frame.
    height_assumption:
        Config ``human_height_assumption`` value.
    human_height:
        Actual human height from the template model.

    Returns
    -------
    New frame dict with scaled positions and unchanged orientations.
    Joints not in ``scale_table`` are dropped from the output.
    """
    height_ratio = human_height / height_assumption if height_assumption > 1e-9 else 1.0
    scaled_table = {str(k): float(v) * height_ratio for k, v in scale_table.items()}

    if human_root_name not in frame or human_root_name not in scaled_table:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (pos, quat) in frame.items():
            if name in scaled_table:
                out[name] = (
                    np.asarray(pos, dtype=np.float64).copy(),
                    np.asarray(quat, dtype=np.float64).copy(),
                )
        return out

    root_pos_raw, root_quat = frame[human_root_name]
    root_pos = np.asarray(root_pos_raw, dtype=np.float64)
    root_quat_arr = np.asarray(root_quat, dtype=np.float64)

    scaled_root_pos = root_pos * scaled_table[human_root_name]
    scaled: dict[str, tuple[np.ndarray, np.ndarray]] = {
        human_root_name: (scaled_root_pos.copy(), root_quat_arr.copy()),
    }

    for name, (pos, quat) in frame.items():
        if name == human_root_name:
            continue
        if name not in scaled_table:
            continue
        p = np.asarray(pos, dtype=np.float64)
        q = np.asarray(quat, dtype=np.float64)
        local_pos = p - root_pos
        scaled_pos = scaled_root_pos + local_pos * scaled_table[name]
        scaled[name] = (scaled_pos, q.copy())

    return scaled
