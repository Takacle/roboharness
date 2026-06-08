#!/usr/bin/env bash
# E2E verification for gmr-harness agent CLI.
#
# Requires:
#   - GMR_ROOT (env var or GMR_HARNESS_E2E_GMR_ROOT)
#   - Robot with T-pose spec and motion file
#
# Usage:
#   GMR_HARNESS_E2E_ROBOT=engineai_pm01 \
#   GMR_HARNESS_E2E_SPEC=specs/tpose/engineai_pm01.json \
#   GMR_HARNESS_E2E_MOTION=/path/to/tpose.bvh \
#   GMR_HARNESS_E2E_GMR_ROOT=/path/to/GMR \
#   bash scripts/verify_e2e.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

# ── Resolve GMR_ROOT ──
GMR_ROOT="${GMR_HARNESS_E2E_GMR_ROOT:-${GMR_ROOT:-}}"
if [ -z "$GMR_ROOT" ]; then
    echo "ERROR: GMR_ROOT not set."
    echo "  export GMR_ROOT=/path/to/GMR"
    echo "  or:  GMR_HARNESS_E2E_GMR_ROOT=/path/to/GMR bash $0"
    exit 1
fi
if [ ! -d "$GMR_ROOT/general_motion_retargeting" ]; then
    echo "ERROR: GMR not found at $GMR_ROOT/general_motion_retargeting"
    exit 1
fi
export GMR_ROOT

# ── Resolve robot / spec / motion ──
ROBOT="${GMR_HARNESS_E2E_ROBOT:-}"
SPEC="${GMR_HARNESS_E2E_SPEC:-}"
MOTION="${GMR_HARNESS_E2E_MOTION:-}"

if [ -z "$ROBOT" ]; then
    # Pick first available robot from GMR params
    PARAMS="$GMR_ROOT/general_motion_retargeting/params.py"
    if [ -f "$PARAMS" ]; then
        ROBOT=$(grep -oP '"\K[^"]+(?=": ASSET_ROOT)' "$PARAMS" | head -1 || true)
    fi
fi
if [ -z "$ROBOT" ]; then
    echo "ERROR: no robot found. Set GMR_HARNESS_E2E_ROBOT=<name>"
    exit 1
fi
echo "[e2e] Robot: $ROBOT"

if [ -z "$SPEC" ]; then
    SPEC="$PROJECT/specs/tpose/${ROBOT}.json"
    if [ ! -f "$SPEC" ]; then
        SPEC="$GMR_ROOT/specs/tpose/${ROBOT}.json"
    fi
fi
if [ ! -f "$SPEC" ]; then
    echo "ERROR: spec not found at $SPEC"
    echo "  Set GMR_HARNESS_E2E_SPEC=<path>"
    exit 1
fi
echo "[e2e] Spec: $SPEC"

if [ -z "$MOTION" ]; then
    echo "[e2e] WARNING: no motion file set (GMR_HARNESS_E2E_MOTION)."
    echo "  solve_mode --dry_run will skip the retargeting step without a motion."
fi

export PYTHONPATH="${PROJECT}/src${PYTHONPATH:+:$PYTHONPATH}"

errors=0

# ── Test 1: --help ──
echo ""
echo "=== Test 1: gmr-harness agent --help ==="
if python -m gmr_harness.cli.main agent --help > /dev/null 2>&1; then
    echo "PASS"
else
    echo "FAIL"
    errors=$((errors + 1))
fi

# ── Test 2: solve_mode --dry_run ──
echo ""
echo "=== Test 2: agent --solve_mode --dry_run ==="
if [ -n "$MOTION" ]; then
    if python -m gmr_harness.cli.agent \
        --robot "$ROBOT" \
        --solve_mode \
        --dry_run \
        --tpose_spec "$SPEC" \
        --tpose_motion "$MOTION" \
        > /dev/null 2>&1; then
        echo "PASS"
    else
        echo "FAIL"
        errors=$((errors + 1))
    fi
else
    echo "SKIP (no motion file)"
fi

# ── Test 3: validate --help ──
echo ""
echo "=== Test 3: gmr-harness validate --help ==="
if python -m gmr_harness.cli.main validate --help > /dev/null 2>&1; then
    echo "PASS"
else
    echo "FAIL"
    errors=$((errors + 1))
fi

# ── Test 4: setup --help ──
echo ""
echo "=== Test 4: gmr-harness setup --help ==="
if python -m gmr_harness.cli.main setup --help > /dev/null 2>&1; then
    echo "PASS"
else
    echo "FAIL"
    errors=$((errors + 1))
fi

# ── Summary ──
echo ""
echo "===== E2E Summary ====="
if [ "$errors" -eq 0 ]; then
    echo "All tests passed."
    exit 0
else
    echo "${errors} test(s) failed."
    exit 1
fi
