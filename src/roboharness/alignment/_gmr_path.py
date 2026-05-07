"""Centralized GMR path resolution.

Replaces the ad-hoc ``sys.path.insert`` + ``GMR_ROOT`` discovery logic that was
copy-pasted across ``examples/_gmr_shared.py``, ``scripts/setup_robot.py``, and
``scripts/stage_tpose.py``.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_gmr_root(hint: Path | None = None) -> Path:
    """Locate the GMR repository root.

    Resolution order:
    1. ``GMR_ROOT`` environment variable.
    2. Explicit *hint* parameter (checked for ``params.py``).
    3. Sibling of the roboharness project root (``../GMR``).
    4. Current working directory.
    """
    env = os.environ.get("GMR_ROOT")
    if env:
        candidate = Path(env)
        if (candidate / "general_motion_retargeting" / "params.py").exists():
            return candidate

    if hint is not None and (hint / "general_motion_retargeting" / "params.py").exists():
        return hint

    # Derive roboharness root from this file's location.
    roboharness_root = Path(__file__).resolve().parents[3]  # src/roboharness/alignment → ... → repo
    sibling = roboharness_root.parent / "GMR"
    if (sibling / "general_motion_retargeting" / "params.py").exists():
        return sibling

    sibling2 = roboharness_root / "GMR"
    if (sibling2 / "general_motion_retargeting" / "params.py").exists():
        return sibling2

    # Current working directory as last resort.
    cwd = Path.cwd()
    if (cwd / "GMR" / "general_motion_retargeting" / "params.py").exists():
        return cwd / "GMR"

    raise FileNotFoundError(
        "GMR not found. Set GMR_ROOT env var or place GMR/ next to roboharness/."
    )
