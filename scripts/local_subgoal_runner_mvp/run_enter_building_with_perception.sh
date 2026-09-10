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

mkdir -p debug/navigation_stage debug/perception_pipeline
POST_STAGE_OBSERVATION_SEC="${POST_STAGE_OBSERVATION_SEC:-3.0}"
VISION_API_TIMEOUT_SEC="${VISION_API_TIMEOUT_SEC:-6.0}"
cat > debug/navigation_stage/current_stage.json <<'JSON'
{
  "stage": "building_entry",
  "source": "run_enter_building_with_perception.sh"
}
JSON

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT

python3 scripts/vision_scene_semantics/vision_scene_semantics_node.py \
  --api-timeout-sec "$VISION_API_TIMEOUT_SEC" \
  > debug/perception_pipeline/vision_scene_semantics.log 2>&1 &
pids+=("$!")

python3 scripts/doorway_candidate_detector/doorway_candidate_detector.py \
  > debug/perception_pipeline/doorway_candidate_detector.log 2>&1 &
pids+=("$!")

python3 scripts/room_frontier_viewpoint_selector/room_frontier_viewpoint_selector.py \
  > debug/perception_pipeline/room_frontier_viewpoint_selector.log 2>&1 &
pids+=("$!")

set +e
python3 scripts/local_subgoal_runner_mvp/enter_building_centerline_once.py "$@"
runner_status=$?
set -e

cat > debug/navigation_stage/current_stage.json <<'JSON'
{
  "stage": "inside_corridor",
  "source": "run_enter_building_with_perception.sh"
}
JSON

sleep "$POST_STAGE_OBSERVATION_SEC"

echo "perception_outputs:"
echo "  vision: debug/vision_scene_semantics/latest_project_scene_analysis.json"
echo "  doorway: debug/doorway_candidate_detector/latest_doorway_candidate.json"
echo "  room_viewpoint: debug/room_frontier_viewpoint_selector/latest_room_viewpoint.json"

exit "$runner_status"
