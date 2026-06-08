"""Root conftest for gmr-harness standalone tests.

Filters out ROS pytest plugins (``launch_testing``,
``launch_testing_ros``) that crash due to missing ``lark`` dependency.
"""

from __future__ import annotations

import os

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
