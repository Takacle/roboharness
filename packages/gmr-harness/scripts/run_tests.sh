#!/usr/bin/env bash
# Run gmr-harness tests with ROS plugin interference disabled.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m pytest tests -q -p pytest_cov "$@"
