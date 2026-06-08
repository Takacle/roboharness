"""Lazy import guards with actionable error messages for optional dependencies."""

from __future__ import annotations

from typing import Any

_REQUIREMENTS: dict[str, str] = {
    "smplx": "pip install gmr-harness[smplx]",
    "torch": "pip install gmr-harness[smplx]",
    "scipy": "pip install scipy  (or gmr-harness[smplx])",
    "mujoco": "pip install gmr-harness[mujoco]",
    "openai": "pip install gmr-harness[vlm]",
    "anthropic": "pip install anthropic",
    "httpx": "pip install gmr-harness[vlm]",
    "PIL": "pip install Pillow  (or gmr-harness[mujoco])",
    "meshcat": "pip install meshcat",
}


def require(module_name: str, feature: str = "") -> Any:
    """Import *module_name* or raise ``SystemExit`` with install instructions.

    Parameters
    ----------
    module_name:
        Top-level module name (e.g. ``"scipy"``, ``"mujoco"``).
    feature:
        Human-readable description of what needs this dependency.

    Returns
    -------
    The imported module object.
    """
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        pkg = _REQUIREMENTS.get(module_name, f"pip install {module_name}")
        hint = f"\n  Required for: {feature}" if feature else ""
        raise SystemExit(f"Missing dependency: {module_name}. Install with: {pkg}{hint}") from None
