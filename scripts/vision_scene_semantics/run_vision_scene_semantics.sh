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

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

if [[ -f config/vision_api.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source config/vision_api.env
  set +a
fi

python3 scripts/vision_scene_semantics/vision_scene_semantics_node.py "$@"
