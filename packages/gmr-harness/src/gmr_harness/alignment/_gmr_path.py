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

    # Walk up the directory tree checking siblings for GMR.
    _file = Path(__file__).resolve()
    for _parent in _file.parents:
        for _sibling in (_parent.parent / "GMR", _parent / "GMR"):
            if (_sibling / "general_motion_retargeting" / "params.py").exists():
                return _sibling

    # Current working directory as last resort.
    cwd = Path.cwd()
    if (cwd / "GMR" / "general_motion_retargeting" / "params.py").exists():
        return cwd / "GMR"

    raise FileNotFoundError(
        "GMR not found. Set GMR_ROOT env var or place GMR/ next to roboharness/."
    )
