"""Tests for roboharness.alignment.gmr_register."""

from __future__ import annotations

import pytest

from roboharness.alignment.gmr_register import (
    register_in_params,
    update_script_choices,
)

_PARAMS_TEMPLATE = """\
import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",
    "unitree_h1": ASSET_ROOT / "unitree_h1" / "h1.xml",
}

IK_CONFIG_DICT = {
    "smplx": {
        "unitree_g1": IK_CONFIG_ROOT / "smplx_to_g1.json",
        "unitree_h1": IK_CONFIG_ROOT / "smplx_to_h1.json",
    },
    "bvh": {
        "unitree_g1": IK_CONFIG_ROOT / "bvh_to_g1.json",
    },
}

ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
    "unitree_h1": "pelvis",
}

VIEWER_CAM_DISTANCE_DICT = {
    "unitree_g1": 2.0,
    "unitree_h1": 3.0,
}
"""

_SCRIPT_TEMPLATE = """\
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot",
    choices=["unitree_g1", "unitree_h1", "booster_t1"],
    default="unitree_g1",
)
args = parser.parse_args()
"""


class TestRegisterInParams:
    @pytest.fixture()
    def gmr_root(self, tmp_path):
        pkg = tmp_path / "general_motion_retargeting"
        pkg.mkdir()
        (pkg / "params.py").write_text(_PARAMS_TEMPLATE)
        return tmp_path

    def test_dry_run_does_not_modify(self, gmr_root):
        params_path = gmr_root / "general_motion_retargeting" / "params.py"
        original = params_path.read_text()
        register_in_params(
            gmr_root,
            "new_robot",
            "new_robot.xml",
            "pelvis",
            2.5,
            ["smplx"],
            dry_run=True,
        )
        assert params_path.read_text() == original

    def test_adds_robot_xml(self, gmr_root):
        register_in_params(
            gmr_root,
            "new_robot",
            "new_robot.xml",
            "pelvis",
            2.5,
            ["smplx"],
            dry_run=False,
        )
        text = (gmr_root / "general_motion_retargeting" / "params.py").read_text()
        assert '"new_robot"' in text
        assert "new_robot.xml" in text
        assert "smplx_to_new_robot.json" in text

    def test_syntax_valid_after_insert(self, gmr_root):
        register_in_params(
            gmr_root,
            "new_robot",
            "new_robot.xml",
            "pelvis",
            2.5,
            ["smplx", "bvh"],
            dry_run=False,
        )
        text = (gmr_root / "general_motion_retargeting" / "params.py").read_text()
        compile(text, "params.py", "exec")

    def test_creates_backup(self, gmr_root):
        register_in_params(
            gmr_root,
            "new_robot",
            "new_robot.xml",
            "pelvis",
            2.5,
            ["smplx"],
            dry_run=False,
        )
        assert (gmr_root / "general_motion_retargeting" / "params.py.bak").exists()

    def test_skip_existing_key(self, gmr_root):
        result = register_in_params(
            gmr_root,
            "unitree_g1",
            "g1.xml",
            "pelvis",
            2.5,
            ["smplx"],
            dry_run=True,
        )
        assert any("SKIP" in line for line in result)

    def test_multiple_formats(self, gmr_root):
        register_in_params(
            gmr_root,
            "new_robot",
            "new_robot.xml",
            "pelvis",
            2.5,
            ["smplx", "bvh"],
            dry_run=False,
        )
        text = (gmr_root / "general_motion_retargeting" / "params.py").read_text()
        assert "smplx_to_new_robot.json" in text
        assert "bvh_to_new_robot.json" in text

    def test_nested_ik_config_inserted_in_correct_block(self, gmr_root):
        register_in_params(
            gmr_root,
            "new_robot",
            "new_robot.xml",
            "pelvis",
            2.5,
            ["smplx", "bvh"],
            dry_run=False,
        )
        text = (gmr_root / "general_motion_retargeting" / "params.py").read_text()
        smplx_block = text.split('"smplx": {')[1].split('"bvh": {')[0]
        assert "smplx_to_new_robot.json" in smplx_block
        bvh_block = text.split('"bvh": {')[1].split('"fbx": {')[0]
        assert "bvh_to_new_robot.json" in bvh_block

    def test_nested_skip_already_in_block(self, gmr_root):
        result = register_in_params(
            gmr_root,
            "unitree_g1",
            "g1.xml",
            "pelvis",
            2.5,
            ["smplx"],
            dry_run=True,
        )
        assert any("IK_CONFIG_DICT[smplx]" in line and "SKIP" in line for line in result)


class TestUpdateScriptChoices:
    @pytest.fixture()
    def gmr_root(self, tmp_path):
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "test_script.py").write_text(_SCRIPT_TEMPLATE)
        return tmp_path

    def test_dry_run_no_change(self, gmr_root):
        script = gmr_root / "scripts" / "test_script.py"
        original = script.read_text()
        update_script_choices(gmr_root, "new_robot", dry_run=True)
        assert script.read_text() == original

    def test_appends_choice(self, gmr_root):
        update_script_choices(gmr_root, "new_robot", dry_run=False)
        text = (gmr_root / "scripts" / "test_script.py").read_text()
        assert "new_robot" in text

    def test_syntax_valid_after_update(self, gmr_root):
        update_script_choices(gmr_root, "new_robot", dry_run=False)
        text = (gmr_root / "scripts" / "test_script.py").read_text()
        compile(text, "test_script.py", "exec")

    def test_skip_already_present(self, gmr_root):
        result = update_script_choices(gmr_root, "unitree_g1", dry_run=True)
        assert any("already present" in r for r in result)
