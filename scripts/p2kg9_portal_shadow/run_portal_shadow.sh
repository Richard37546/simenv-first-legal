#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/noetic/setup.bash
source "$ROOT/devel/setup.bash" 2>/dev/null || true
exec python3 "$ROOT/scripts/p2kg9_portal_shadow/portal_shadow_node.py" "$@"
