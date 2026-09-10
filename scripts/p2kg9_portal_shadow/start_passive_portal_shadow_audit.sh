#!/usr/bin/env bash
# Default-off launcher for the isolated audit-only Portal Shadow.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID=""
OUTPUT_ROOT="$ROOT/debug/odom_accuracy_audit_v1/p2kg9_current_door_portal_shadow_048/online_run"

usage() {
  printf '%s\n' "usage: $0 --run-id RUN_ID [--output-root DIRECTORY] [--check]"
}

CHECK_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:?missing run id}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing output root}"; shift 2 ;;
    --check) CHECK_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 64 ;;
  esac
done

# This launcher can be started by the strict-mode audit pane before any ROS
# shell setup exists.  ROS Noetic's setup script is not nounset-safe when
# ROS_DISTRO is absent, so constrain the compatibility relaxation to sourcing.
source_ros_workspace() {
  set +u
  source /opt/ros/noetic/setup.bash
  source "$ROOT/devel/setup.bash" 2>/dev/null || true
  set -u
}

if [[ "$CHECK_ONLY" == true ]]; then
  [[ -n "$RUN_ID" ]] || { printf '%s\n' "--run-id is required" >&2; exit 64; }
  [[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf '%s\n' "invalid run id" >&2; exit 64; }
  [[ ! -e "$OUTPUT_ROOT/$RUN_ID" ]] || { printf '%s\n' "output already exists" >&2; exit 65; }
  printf '%s\n' "P2KG9 audit launcher check passed; no ROS process started."
  exit 0
fi

[[ -n "$RUN_ID" ]] || { printf '%s\n' "--run-id is required" >&2; exit 64; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf '%s\n' "invalid run id" >&2; exit 64; }
[[ ! -e "$OUTPUT_ROOT/$RUN_ID" ]] || { printf '%s\n' "output already exists" >&2; exit 65; }
source_ros_workspace
rosnode list >/dev/null 2>&1 || { printf '%s\n' "ROS master unavailable; audit Shadow was not started." >&2; exit 69; }
mkdir -p "$OUTPUT_ROOT"
export P2KG9_RUN_ID="$RUN_ID"
export P2KG9_OUTPUT_ROOT="$OUTPUT_ROOT"
exec "$ROOT/scripts/p2kg9_portal_shadow/run_portal_shadow.sh"
