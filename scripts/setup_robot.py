"""Compatibility wrapper — delegates to ``gmr_harness.cli.setup_robot``.

.. deprecated::
    Use ``gmr-harness setup`` directly. This script is kept for backward
    compatibility and will be removed in a future release.

Re-exports key names so existing tests that mock ``scripts.setup_robot.xxx``
continue to work during the migration period.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "scripts/setup_robot.py is deprecated; use 'gmr-harness setup' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward-compatible mocking in tests.
from gmr_harness.alignment.body_matcher import match_bodies  # noqa: E402,F401
from gmr_harness.alignment.config_gen import (  # noqa: E402,F401
    clone_ik_config,
    generate_ik_config,
    write_ik_config,
)
from gmr_harness.alignment.gmr_register import (  # noqa: E402,F401
    register_in_params,
    update_script_choices,
)
from gmr_harness.alignment.orientation_aligner import (  # noqa: E402,F401
    extract_xml_body_names,
    parse_world_rotation_arg,
)
from gmr_harness.alignment.skeleton_maps import get_skeleton  # noqa: E402,F401
from gmr_harness.alignment.smplx_offset_solver import (  # noqa: E402,F401
    solve_smplx_offsets_from_template,
    write_solved_config,
)
from gmr_harness.cli.setup_robot import (  # noqa: E402,F401
    _find_clone_source,
    _get_gmr_root,
    _solve_smplx_offsets,
    _solve_via_agent,
    find_gmr_root,
    load_gmr_params,
    main,
)

if __name__ == "__main__":
    main()
