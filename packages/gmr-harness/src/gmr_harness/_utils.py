"""Shared internal utilities — image saving, JSON I/O, media helpers."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def save_image(arr: np.ndarray, path: Path) -> None:
    """Save RGB array as PNG. Uses PIL if available, falls back to raw numpy.

    When PIL is not installed the array is saved as ``.npy`` next to the
    requested *path* and a warning is emitted so callers know the ``.png``
    was not created.
    """
    try:
        from PIL import Image

        img = Image.fromarray(arr)
        img.save(path)
    except ImportError:
        npy_path = path.with_suffix(".npy")
        np.save(npy_path, arr)
        logger.warning(
            "PIL not installed — saved %s as %s instead of PNG",
            path.name,
            npy_path.name,
        )


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save dict as JSON, converting numpy types."""
    with path.open("w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return as dict."""
    with path.open() as f:
        result: dict[str, Any] = json.load(f)
    return result


def to_float(value: Any) -> float:
    """Convert a tensor/numpy/scalar value to float.

    Handles PyTorch tensors (CPU/CUDA), numpy scalars/arrays, and Python numbers.
    Multi-element tensors/arrays return the mean. Unconvertible values return 0.0.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.number):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(value.mean())
    # Handle torch tensors and similar array-like objects via duck typing
    if hasattr(value, "item"):
        if hasattr(value, "numel") and value.numel() > 1:
            return float(value.float().mean().item())
        if hasattr(value, "size") and isinstance(value.size, int) and value.size > 1:
            return float(value.mean())
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def encode_image_base64(path: Path) -> str:
    """Read a file and return its base64-encoded string."""
    return base64.standard_b64encode(path.read_bytes()).decode()


def select_image_files(image_paths: list[Path], max_images: int) -> list[Path]:
    """Select up to *max_images* files, preferring front/side/back views."""
    if len(image_paths) <= max_images:
        return image_paths
    fronts = sorted(p for p in image_paths if "front" in p.name)
    sides = sorted(p for p in image_paths if "side" in p.name)
    backs = sorted(p for p in image_paths if "back" in p.name)
    selected = (fronts + sides + backs)[:max_images]
    return sorted(selected)
