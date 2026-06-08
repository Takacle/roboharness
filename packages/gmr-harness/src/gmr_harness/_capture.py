from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CameraView:
    name: str
    rgb: np.ndarray
    depth: np.ndarray | None = None
    segmentation: np.ndarray | None = None

    def save(self, directory: Path) -> dict[str, str]:
        directory.mkdir(parents=True, exist_ok=True)
        saved: dict[str, str] = {}

        rgb_path = directory / f"{self.name}_rgb.png"
        try:
            from PIL import Image

            Image.fromarray(self.rgb).save(rgb_path)
        except ImportError:
            npy_path = rgb_path.with_suffix(".npy")
            np.save(npy_path, self.rgb)
        saved["rgb"] = str(rgb_path)

        if self.depth is not None:
            depth_path = directory / f"{self.name}_depth.npy"
            np.save(depth_path, self.depth)
            saved["depth"] = str(depth_path)

        if self.segmentation is not None:
            seg_path = directory / f"{self.name}_seg.npy"
            np.save(seg_path, self.segmentation)
            saved["segmentation"] = str(seg_path)

        return saved
