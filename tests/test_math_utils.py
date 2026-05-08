"""Unit tests for roboharness._math_utils — shared quaternion/vector/math utilities."""

from __future__ import annotations

import numpy as np
import pytest

from roboharness._math_utils import (
    IDENTITY_QUAT,
    normalize_quat,
    normalize_vector,
    quat_multiply,
    rotation_matrix_to_axis_angle,
    rotation_matrix_to_quat,
)


class TestNormalizeQuat:
    def test_unit_quat_preserved(self) -> None:
        result = normalize_quat([1.0, 0.0, 0.0, 0.0])
        assert result == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_normalizes_scaled_quat(self) -> None:
        result = normalize_quat([2.0, 0.0, 0.0, 0.0])
        assert result == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_zero_returns_identity(self) -> None:
        result = normalize_quat([0.0, 0.0, 0.0, 0.0])
        assert result == [1.0, 0.0, 0.0, 0.0]

    def test_near_zero_returns_identity(self) -> None:
        result = normalize_quat([1e-12, 0.0, 0.0, 0.0])
        assert result == [1.0, 0.0, 0.0, 0.0]

    def test_non_unit_generic(self) -> None:
        result = normalize_quat([1.0, 2.0, 3.0, 4.0])
        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0)


class TestNormalizeVector:
    def test_unit_vector_unchanged(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        result = normalize_vector(v)
        assert result == pytest.approx(v)

    def test_normalizes_scaled_vector(self) -> None:
        result = normalize_vector(np.array([3.0, 0.0, 0.0]))
        assert result == pytest.approx(np.array([1.0, 0.0, 0.0]))

    def test_zero_returns_input(self) -> None:
        v = np.array([0.0, 0.0, 0.0])
        result = normalize_vector(v)
        assert np.array_equal(result, v)

    def test_with_fallback(self) -> None:
        fallback = np.array([0.0, 0.0, 1.0])
        result = normalize_vector(np.array([0.0, 0.0, 0.0]), fallback)
        assert result == pytest.approx(fallback)

    def test_fallback_not_used_for_valid_vector(self) -> None:
        fallback = np.array([0.0, 0.0, 1.0])
        result = normalize_vector(np.array([2.0, 0.0, 0.0]), fallback)
        assert result == pytest.approx(np.array([1.0, 0.0, 0.0]))


class TestQuatMultiply:
    def test_identity_multiplication(self) -> None:
        q = [1.0, 0.0, 0.0, 0.0]
        result = quat_multiply(q, q)
        assert result == pytest.approx(q)

    def test_90_deg_x_axis(self) -> None:
        # 90° around X: w=cos(45°), x=sin(45°)
        cos45 = np.cos(np.pi / 4)
        q_x = [cos45, cos45, 0.0, 0.0]
        # q_x * q_x = 180° around X = [0, 0, 0, -1] or [0, 1, 0, 0]
        result = quat_multiply(q_x, q_x)
        assert result == pytest.approx([0.0, 1.0, 0.0, 0.0])


class TestRotationMatrixToAxisAngle:
    def test_identity(self) -> None:
        axis, angle = rotation_matrix_to_axis_angle(np.eye(3))
        assert angle == 0.0
        assert np.allclose(axis, [0.0, 0.0, 1.0])

    def test_90_deg_z(self) -> None:
        theta = np.pi / 2
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        axis, angle = rotation_matrix_to_axis_angle(R)
        assert angle == pytest.approx(theta)
        assert np.allclose(np.abs(axis), [0.0, 0.0, 1.0], atol=1e-6)

    def test_180_deg_x(self) -> None:
        R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        axis, angle = rotation_matrix_to_axis_angle(R)
        assert angle == pytest.approx(np.pi)
        assert np.allclose(np.abs(axis), [1.0, 0.0, 0.0], atol=1e-6)

    def test_180_deg_diagonal(self) -> None:
        v = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + 2.0 * (K @ K)
        axis, angle = rotation_matrix_to_axis_angle(R)
        assert angle == pytest.approx(np.pi, abs=1e-6)
        assert np.allclose(np.abs(axis), np.abs(v), atol=1e-6)

    def test_near_identity(self) -> None:
        # 1e-6 rad rotation around Z — angle well below 1e-8 rad threshold
        eps = 1e-9
        R = np.array(
            [[np.cos(eps), -np.sin(eps), 0.0], [np.sin(eps), np.cos(eps), 0.0], [0.0, 0.0, 1.0]],
        )
        axis, angle = rotation_matrix_to_axis_angle(R)
        assert angle == 0.0
        assert np.allclose(axis, [0.0, 0.0, 1.0])


class TestRotationMatrixToQuat:
    def test_identity(self) -> None:
        q = rotation_matrix_to_quat(np.eye(3))
        assert q == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_90_deg_z(self) -> None:
        theta = np.pi / 2
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        q = rotation_matrix_to_quat(R)
        assert len(q) == 4
        assert np.linalg.norm(q) == pytest.approx(1.0)
        # 90° around Z: scalar-first [cos(45°), 0, 0, sin(45°)]
        expected = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
        assert q == pytest.approx(expected)


class TestIdentityQuat:
    def test_identity_quat_value(self) -> None:
        assert IDENTITY_QUAT == [1.0, 0.0, 0.0, 0.0]
