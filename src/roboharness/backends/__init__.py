"""Simulator backend adapters for Roboharness."""

from roboharness.backends.visualizer import (
    MeshcatVisualizer,
    MuJoCoNativeVisualizer,
    Visualizer,
)

__all__ = [
    "MeshcatVisualizer",
    "MuJoCoNativeVisualizer",
    "Visualizer",
]

# MjlabBackend is an optional heavy dependency (requires mjlab + GPU).
# Import lazily so that ``import roboharness.backends`` works without mjlab.
try:
    from roboharness.backends.mjlab_backend import MjlabBackend  # noqa: F401 — re-export

    __all__.append("MjlabBackend")
except ImportError:
    pass  # mjlab not installed; users must import from the submodule directly
