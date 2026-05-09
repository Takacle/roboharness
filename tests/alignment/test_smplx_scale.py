"""Tests for SMPL-X scale normalization module.

Validates:
- Root scaling with height_ratio
- Child relative-to-root scaling
- Joints not in scale_table are dropped
- Empty scale_table handling
"""

from __future__ import annotations

import numpy as np


class TestApplyHumanScale:
    def _make_frame(self):
        return {
            "pelvis": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            "spine3": (np.array([0.0, 1.5, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            "left_shoulder": (np.array([0.2, 1.5, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            "right_shoulder": (np.array([-0.2, 1.5, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        }

    def test_scales_root_position(self):
        from roboharness.alignment.smplx_scale import apply_human_scale

        frame = self._make_frame()
        scale_table = {"pelvis": 0.9, "spine3": 0.85}
        result = apply_human_scale(frame, scale_table, human_height=1.66, height_assumption=1.66)
        root_pos = result["pelvis"][0]
        expected_root = np.array([0.0, 1.0, 0.0]) * 0.9
        np.testing.assert_allclose(root_pos, expected_root, atol=1e-8)

    def test_scales_child_relative_to_root(self):
        from roboharness.alignment.smplx_scale import apply_human_scale

        frame = self._make_frame()
        scale_table = {"pelvis": 1.0, "spine3": 0.8}
        result = apply_human_scale(frame, scale_table, human_height=1.66, height_assumption=1.66)
        root_pos = result["pelvis"][0]
        spine_pos = result["spine3"][0]
        local_spine = np.array([0.0, 1.5, 0.0]) - np.array([0.0, 1.0, 0.0])
        expected_spine = root_pos + local_spine * 0.8
        np.testing.assert_allclose(spine_pos, expected_spine, atol=1e-8)

    def test_applies_height_ratio(self):
        from roboharness.alignment.smplx_scale import apply_human_scale

        frame = self._make_frame()
        scale_table = {"pelvis": 1.0}
        result = apply_human_scale(frame, scale_table, human_height=1.8, height_assumption=1.8)
        root_pos = result["pelvis"][0]
        np.testing.assert_allclose(root_pos, np.array([0.0, 1.0, 0.0]), atol=1e-8)

        result2 = apply_human_scale(frame, scale_table, human_height=1.8, height_assumption=1.6)
        root_pos2 = result2["pelvis"][0]
        ratio = 1.8 / 1.6
        np.testing.assert_allclose(root_pos2, np.array([0.0, 1.0, 0.0]) * ratio, atol=1e-8)

    def test_drops_joints_not_in_scale_table(self):
        from roboharness.alignment.smplx_scale import apply_human_scale

        frame = self._make_frame()
        scale_table = {"pelvis": 1.0}
        result = apply_human_scale(frame, scale_table, human_height=1.66, height_assumption=1.66)
        assert "pelvis" in result
        assert "spine3" not in result

    def test_preserves_orientations(self):
        from roboharness.alignment.smplx_scale import apply_human_scale

        frame = self._make_frame()
        quat = np.array([0.707, 0.707, 0.0, 0.0])
        frame["spine3"] = (frame["spine3"][0], quat.copy())
        scale_table = {"pelvis": 1.0, "spine3": 0.9}
        result = apply_human_scale(frame, scale_table, human_height=1.66, height_assumption=1.66)
        np.testing.assert_allclose(result["spine3"][1], quat, atol=1e-10)

    def test_does_not_modify_input(self):
        from roboharness.alignment.smplx_scale import apply_human_scale

        frame = self._make_frame()
        orig_pelvis_pos = frame["pelvis"][0].copy()
        scale_table = {"pelvis": 0.5}
        apply_human_scale(frame, scale_table, human_height=1.66, height_assumption=1.66)
        np.testing.assert_array_equal(frame["pelvis"][0], orig_pelvis_pos)

    def test_zero_height_assumption_fallback(self):
        from roboharness.alignment.smplx_scale import apply_human_scale

        frame = self._make_frame()
        scale_table = {"pelvis": 1.0}
        result = apply_human_scale(frame, scale_table, human_height=1.66, height_assumption=0.0)
        root_pos = result["pelvis"][0]
        np.testing.assert_allclose(root_pos, np.array([0.0, 1.0, 0.0]), atol=1e-8)

    def test_root_not_in_frame_passes_through(self):
        from roboharness.alignment.smplx_scale import apply_human_scale

        frame = {
            "joint_a": (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        scale_table = {"joint_a": 0.8}
        result = apply_human_scale(
            frame, scale_table, human_root_name="pelvis", human_height=1.66, height_assumption=1.66
        )
        assert "joint_a" in result
        np.testing.assert_allclose(result["joint_a"][0], np.array([1.0, 0.0, 0.0]), atol=1e-8)

    def test_empty_scale_table_returns_empty(self):
        from roboharness.alignment.smplx_scale import apply_human_scale

        frame = self._make_frame()
        result = apply_human_scale(frame, {}, human_height=1.66, height_assumption=1.66)
        assert len(result) == 0
