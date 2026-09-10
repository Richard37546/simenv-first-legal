#!/usr/bin/env bash
set -eo pipefail

ROOT=/home/richard/simenv_official_clean
source /opt/ros/noetic/setup.bash
source "$ROOT/devel/setup.bash"

mode=${1:?usage: run_acceptance.sh fixedstand|full|perception-delay run_id}
run_id=${2:?usage: run_acceptance.sh fixedstand|full|perception-delay run_id}
case "$mode" in
  fixedstand) extra=(--fixedstand-only) ;;
  full) extra=() ;;
  perception-delay) extra=(--perception-delay-sim-sec 5.0) ;;
  *) echo "mode must be fixedstand, full, or perception-delay" >&2; exit 64 ;;
esac

exec python3 "$ROOT/scripts/a1_safe_startup_supervisor/a1_safe_startup_supervisor.py" \
  --run-id "$run_id" --offline-truth-capture "${extra[@]}"
