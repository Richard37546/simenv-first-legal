#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
  set -u
fi

if [[ -f devel/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source devel/setup.bash
  set -u
fi

python3 scripts/local_subgoal_runner_mvp/enter_building_centerline_once.py "$@"
