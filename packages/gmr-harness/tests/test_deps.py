"""Tests for gmr_harness._deps — lazy import guards."""

from __future__ import annotations

import pytest

from gmr_harness._deps import require


def test_require_numpy_succeeds():
    mod = require("numpy")
    assert hasattr(mod, "array")


def test_require_missing_module_exits():
    with pytest.raises(SystemExit, match="Missing dependency: definitely_not_a_real_pkg"):
        require("definitely_not_a_real_pkg")


def test_require_missing_with_feature_hint():
    with pytest.raises(SystemExit, match="Required for: time travel"):
        require("definitely_not_a_real_pkg", "time travel")
