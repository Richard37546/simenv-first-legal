#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /opt/ros/noetic/setup.bash
if [[ -f devel/setup.bash ]]; then
  source devel/setup.bash
fi
exec python3 scripts/rgbd_danger_perception/world_result_coordinate.py "$@"
