"""Lightweight GMR params.py loader.

Avoids triggering GMR's heavy ``__init__.py`` (which imports ``mink``).
Loads ``params.py`` as a file-level module via ``importlib`` so the caller
can read ``ROBOT_XML_DICT``, ``IK_CONFIG_DICT``, etc. without pulling in the
full GMR dependency tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BLOCKED_MODS = ("general_motion_retargeting",)


def load_gmr_params(gmr_root: Path) -> object:
    """Load GMR's ``params.py`` as a data module.

    Returns the module object so callers can access ``ROBOT_XML_DICT``,
    ``ROBOT_BASE_DICT``, ``VIEWER_CAM_DISTANCE_DICT``, and ``IK_CONFIG_DICT``.
    """
    params_path = gmr_root / "general_motion_retargeting" / "params.py"
    if not params_path.exists():
        raise FileNotFoundError(f"params.py not found at {params_path}")
    spec = importlib.util.spec_from_file_location("_gmr_params", params_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {params_path}")
    # Prevent GMR's __init__.py from being auto-imported when params.py does
    # ``from . import ...``.
    saved: dict[str, object] = {}
    for name in _BLOCKED_MODS:
        if name not in sys.modules:
            saved[name] = sys.modules.get(name)
            sys.modules[name] = type(sys)(name)  # placeholder
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name in _BLOCKED_MODS:
            if name in saved:
                sys.modules[name] = saved[name]
            elif name in sys.modules and isinstance(sys.modules[name], type(sys)):
                del sys.modules[name]
    return module
