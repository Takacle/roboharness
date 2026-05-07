"""Tests for roboharness.alignment.skeleton_maps."""

from __future__ import annotations

import pytest

from roboharness.alignment.skeleton_maps import (
    BVH_SKELETON,
    ROLES,
    SMPLX_SKELETON,
    get_skeleton,
)


class TestSkeletonData:
    def test_smplx_has_all_roles(self):
        for role in ROLES:
            assert role in SMPLX_SKELETON.role_to_joint, f"SMPL-X missing role: {role}"

    def test_bvh_has_all_roles(self):
        for role in ROLES:
            assert role in BVH_SKELETON.role_to_joint, f"BVH missing role: {role}"

    def test_smplx_root_name(self):
        assert SMPLX_SKELETON.root_name == "pelvis"

    def test_bvh_root_name(self):
        assert BVH_SKELETON.root_name == "Hips"

    def test_smplx_scale_keys_are_joint_names(self):
        for joint in SMPLX_SKELETON.scale_defaults:
            assert joint in SMPLX_SKELETON.role_to_joint.values(), (
                f"SMPL-X scale key {joint!r} not in role_to_joint values"
            )

    def test_bvh_scale_keys_are_joint_names(self):
        for joint in BVH_SKELETON.scale_defaults:
            assert joint in BVH_SKELETON.role_to_joint.values(), (
                f"BVH scale key {joint!r} not in role_to_joint values"
            )

    def test_no_duplicate_joint_names_in_roles(self):
        for skeleton in (SMPLX_SKELETON, BVH_SKELETON):
            joints = list(skeleton.role_to_joint.values())
            assert len(joints) == len(set(joints)), (
                f"{skeleton.name} has duplicate joint names in role_to_joint"
            )

    def test_smplx_fallback_joints_exist_in_skeleton(self):
        all_joints = set(SMPLX_SKELETON.role_to_joint.values())
        for child, parent in SMPLX_SKELETON.fallback_map.items():
            assert parent in all_joints, f"SMPL-X fallback parent {parent!r} not in joints"

    def test_bvh_fallback_joints_exist_in_skeleton(self):
        all_joints = set(BVH_SKELETON.role_to_joint.values())
        for child, parent in BVH_SKELETON.fallback_map.items():
            assert parent in all_joints, f"BVH fallback parent {parent!r} not in joints"


class TestGetSkeleton:
    def test_smplx(self):
        assert get_skeleton("smplx") is SMPLX_SKELETON

    def test_bvh(self):
        assert get_skeleton("bvh") is BVH_SKELETON

    def test_fbx_uses_bvh(self):
        assert get_skeleton("fbx") is BVH_SKELETON

    def test_fbx_offline_uses_bvh(self):
        assert get_skeleton("fbx_offline") is BVH_SKELETON

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown src_format"):
            get_skeleton("unknown_format")
