#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
RUN_COMMAND="$0 $*"

RUN_ARCHIVE_ROOT="debug/state_machine_navigation/run_archives"
mkdir -p "$RUN_ARCHIVE_ROOT"
exec 8>"$RUN_ARCHIVE_ROOT/.state_machine_execution.lock"
if ! flock -n 8; then
  echo "fatal: another state-machine archive run is active" >&2
  exit 75
fi
RUN_START_WALL_SEC="$(date +%s.%N)"
RUN_GIT_HEAD="$(git rev-parse HEAD)"
RUN_DIRTY_DIFF_HASH="$(git diff --binary | sha256sum | awk '{print $1}')"
RUN_STATUS_HASH="$(git status --short | sha256sum | awk '{print $1}')"
exec 9>"$RUN_ARCHIVE_ROOT/.run_id.lock"
flock 9
next_run_id=1
for existing_run_dir in "$RUN_ARCHIVE_ROOT"/run_*; do
  [[ -d "$existing_run_dir" ]] || continue
  existing_name="$(basename "$existing_run_dir")"
  existing_id="${existing_name#run_}"
  existing_id="${existing_id%%_*}"
  if [[ "$existing_id" =~ ^[0-9]+$ ]] && ((10#$existing_id >= next_run_id)); then
    next_run_id=$((10#$existing_id + 1))
  fi
done
RUN_ID="$(printf "%04d" "$next_run_id")"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S_%N)"
RUN_NAME="run_${RUN_ID}_${RUN_TIMESTAMP}_pid$$"
RUN_ID_SOURCE="ARCHIVE_SEQUENCE"
if [[ -n "${ROOM_LOCAL_VALIDATION_RUN_ID:-}" ]]; then
  [[ "${ROOM_LOCAL_VALIDATION_RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "fatal: invalid ROOM_LOCAL_VALIDATION_RUN_ID" >&2; exit 64;
  }
  RUN_NAME="${ROOM_LOCAL_VALIDATION_RUN_ID}"
  RUN_ID_SOURCE="EXTERNAL_ROOM_LOCAL_VALIDATION_RUN_ID"
fi
RUN_ARCHIVE_DIR="$RUN_ARCHIVE_ROOT/$RUN_NAME"
if ! mkdir "$RUN_ARCHIVE_DIR"; then
  echo "fatal: unique run archive already exists: $RUN_ARCHIVE_DIR" >&2
  exit 73
fi
flock -u 9
export STATE_MACHINE_RUN_ID="$RUN_NAME"
export STATE_MACHINE_RUN_ARCHIVE_DIR="$RUN_ARCHIVE_DIR"
RUN_TERMINAL_LOG="$RUN_ARCHIVE_DIR/terminal_output.log"
exec > >(tee -a "$RUN_TERMINAL_LOG") 2>&1

python3 scripts/local_subgoal_runner_mvp/capture_navigation_provenance.py \
  --repo-root "$ROOT_DIR" \
  --output-dir "$RUN_ARCHIVE_DIR" \
  --run-command "$RUN_COMMAND"

echo "run_archive:"
echo "  run_id: $RUN_NAME"
echo "  run_id_source: $RUN_ID_SOURCE"
if [[ "$RUN_ID_SOURCE" == "ARCHIVE_SEQUENCE" ]]; then
  echo "  archive_sequence: $RUN_ID"
fi
echo "  dir: $RUN_ARCHIVE_DIR"

SHARED_OUTPUTS=(
  "debug/state_machine_navigation/state_machine_navigation_summary.json"
  "audit_reports/state_machine_navigation_report.md"
  "debug/block_astar_dwa_mature/block_astar_dwa_mature_summary.json"
  "debug/short_horizon_target_selection/short_horizon_target_override.json"
  "debug/navigation_stage/current_stage.json"
  "debug/rgbd_danger_perception/latest_danger_tracks.json"
  "debug/rgbd_danger_perception/world_result_coordinate_status.json"
  "results/detected_danger.json"
  "debug/rgbd_danger_perception/shadow_validation.json"
  "debug/room_search/latest_room_search_diagnostic.json"
  "debug/local_free_space_entry_target/local_free_space_entry_target_debug.json"
  "debug/forced_room_entry_mvp/forced_room_entry_summary.json"
  "debug/forced_room_entry_mvp/trigger_event.json"
  "debug/forced_room_entry_mvp/before_entry_rgb.png"
  "debug/forced_room_entry_mvp/after_turn_rgb.png"
  "debug/forced_room_entry_mvp/after_entry_rgb.png"
  "debug/forced_room_entry_mvp/before_entry_depth_colormap.png"
  "debug/forced_room_entry_mvp/after_turn_depth_colormap.png"
  "debug/forced_room_entry_mvp/after_entry_depth_colormap.png"
)

: > "$RUN_ARCHIVE_DIR/shared_output_baseline.sha256"
for shared_path in "${SHARED_OUTPUTS[@]}"; do
  if [[ -f "$shared_path" ]]; then
    printf "%s  %s\n" "$(sha256sum "$shared_path" | awk '{print $1}')" "$shared_path"
  else
    printf "MISSING  %s\n" "$shared_path"
  fi
done > "$RUN_ARCHIVE_DIR/shared_output_baseline.sha256"

python3 - "$RUN_ARCHIVE_DIR/run_manifest.json" <<PY
import json, pathlib
path = pathlib.Path("$RUN_ARCHIVE_DIR/run_manifest.json")
path.write_text(json.dumps({
    "schema_version": 1,
    "run_id": "$RUN_NAME",
    "archive_dir": "$RUN_ARCHIVE_DIR",
    "git_head": "$RUN_GIT_HEAD",
    "dirty_diff_sha256": "$RUN_DIRTY_DIFF_HASH",
    "git_status_sha256": "$RUN_STATUS_HASH",
    "command": "$RUN_COMMAND",
    "start_wall_time_sec": float("$RUN_START_WALL_SEC"),
    "complete": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

archive_if_changed() {
  local src="$1"
  local dst_name="$2"
  [[ -f "$src" ]] || return 0
  if [[ "$src" == "debug/state_machine_navigation/state_machine_navigation_summary.json" ]]; then
    if ! python3 - "$src" "$RUN_NAME" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if data.get("run_id") == sys.argv[2] else 1)
PY
    then
      printf "%s\n" "$src:run_id_mismatch" >> "$RUN_ARCHIVE_DIR/rejected_shared_outputs.txt"
      return 0
    fi
  fi
  local after_hash before_hash
  after_hash="$(sha256sum "$src" | awk '{print $1}')"
  before_hash="$(awk -v path="$src" '$2 == path {print $1}' "$RUN_ARCHIVE_DIR/shared_output_baseline.sha256")"
  if [[ -z "$before_hash" || "$before_hash" == "MISSING" || "$after_hash" != "$before_hash" ]]; then
    cp --no-clobber "$src" "$RUN_ARCHIVE_DIR/$dst_name"
    printf "%s  %s\n" "$after_hash" "$dst_name" >> "$RUN_ARCHIVE_DIR/archived_changed_files.sha256"
  else
    printf "%s\n" "$src" >> "$RUN_ARCHIVE_DIR/skipped_unchanged_shared_outputs.txt"
  fi
}

archive_run_outputs() {
  local status="$1"
  archive_if_changed "debug/state_machine_navigation/state_machine_navigation_summary.json" "state_machine_navigation_summary.json"
  archive_if_changed "audit_reports/state_machine_navigation_report.md" "state_machine_navigation_report.md"
  archive_if_changed "debug/block_astar_dwa_mature/block_astar_dwa_mature_summary.json" "block_astar_dwa_mature_summary.json"
  archive_if_changed "debug/short_horizon_target_selection/short_horizon_target_override.json" "short_horizon_target_override.json"
  archive_if_changed "debug/navigation_stage/current_stage.json" "current_stage.json"
  archive_if_changed "debug/rgbd_danger_perception/latest_danger_tracks.json" "latest_danger_tracks.json"
  archive_if_changed "debug/rgbd_danger_perception/shadow_validation.json" "rgbd_shadow_validation.json"
  archive_if_changed "debug/room_search/latest_room_search_diagnostic.json" "latest_room_search_diagnostic.json"
  archive_if_changed "debug/local_free_space_entry_target/local_free_space_entry_target_debug.json" "local_free_space_entry_target_debug.json"
  archive_if_changed "debug/forced_room_entry_mvp/forced_room_entry_summary.json" "forced_room_entry_summary.json"
  archive_if_changed "debug/forced_room_entry_mvp/trigger_event.json" "forced_room_entry_trigger_event.json"
  archive_if_changed "debug/forced_room_entry_mvp/before_entry_rgb.png" "forced_room_entry_before_rgb.png"
  archive_if_changed "debug/forced_room_entry_mvp/after_turn_rgb.png" "forced_room_entry_after_turn_rgb.png"
  archive_if_changed "debug/forced_room_entry_mvp/after_entry_rgb.png" "forced_room_entry_after_rgb.png"
  archive_if_changed "debug/forced_room_entry_mvp/before_entry_depth_colormap.png" "forced_room_entry_before_depth_colormap.png"
  archive_if_changed "debug/forced_room_entry_mvp/after_turn_depth_colormap.png" "forced_room_entry_after_depth_colormap.png"
  archive_if_changed "debug/forced_room_entry_mvp/after_entry_depth_colormap.png" "forced_room_entry_after_depth_colormap.png"
  cat > "$RUN_ARCHIVE_DIR/run_manifest.txt" <<EOF
run_id=$RUN_NAME
run_id_source=$RUN_ID_SOURCE
archive_sequence=$RUN_ID
timestamp=$RUN_TIMESTAMP
exit_status=$status
command=$RUN_COMMAND
terminal_log=terminal_output.log
git_head=$RUN_GIT_HEAD
dirty_diff_sha256=$RUN_DIRTY_DIFF_HASH
EOF
  if [[ -f "$RUN_ARCHIVE_DIR/state_machine_navigation_summary.json" ]]; then
    python3 - "$RUN_ARCHIVE_DIR/state_machine_navigation_summary.json" "$RUN_ARCHIVE_DIR/resolved_parameters.json" <<'PY'
import json, pathlib, sys
summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pathlib.Path(sys.argv[2]).write_text(json.dumps(summary.get("config", {}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  RUN_END_WALL_SEC="$(date +%s.%N)"
  python3 - "$RUN_ARCHIVE_DIR/run_manifest.json" <<PY
import json, pathlib
path = pathlib.Path("$RUN_ARCHIVE_DIR/run_manifest.json")
data = json.loads(path.read_text(encoding="utf-8"))
data.update({"exit_status": int("$status"), "end_wall_time_sec": float("$RUN_END_WALL_SEC"), "complete": True, "resolved_parameters": "resolved_parameters.json" if pathlib.Path("$RUN_ARCHIVE_DIR/resolved_parameters.json").exists() else None, "source_manifest": "source_manifest.json" if pathlib.Path("$RUN_ARCHIVE_DIR/source_manifest.json").exists() else None, "file_hashes": "file_hashes.sha256"})
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
  if [[ " $* " == *" --enable-room-local-guarded-online-validation "* ]]; then
    python3 scripts/local_subgoal_runner_mvp/finalize_room_local_online_validation_archive.py \
      --repo-root "$ROOT_DIR" \
      --archive-dir "$RUN_ARCHIVE_DIR" \
      --run-id "$RUN_NAME" \
      --exit-status "$status" \
      --launch-command "$RUN_COMMAND" \
      --validation-enabled true
  fi
  printf "%s\n" "$RUN_NAME" > "$RUN_ARCHIVE_DIR/RUN_COMPLETE"
  find "$RUN_ARCHIVE_DIR" -type f ! -name 'file_hashes.sha256' -print0 | sort -z | xargs -0 sha256sum > "$RUN_ARCHIVE_DIR/file_hashes.sha256"
  echo "run_archive_saved:"
  echo "  dir: $RUN_ARCHIVE_DIR"
}

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

mkdir -p debug/navigation_stage debug/perception_pipeline debug/state_machine_navigation
VISION_API_TIMEOUT_SEC="${VISION_API_TIMEOUT_SEC:-6.0}"

cat > debug/navigation_stage/current_stage.json <<'JSON'
{
  "stage": "state_machine_starting",
  "source": "run_state_machine_navigation.sh"
}
JSON

pids=()
VALIDATION_OBSERVER_PID=""
cleanup() {
  # Give the validation-only ROS node its normal shutdown callback before the
  # archive finalizer evaluates its telemetry file.
  if [[ -n "$VALIDATION_OBSERVER_PID" ]] && kill -0 "$VALIDATION_OBSERVER_PID" >/dev/null 2>&1; then
    kill -INT "$VALIDATION_OBSERVER_PID" >/dev/null 2>&1 || true
    wait "$VALIDATION_OBSERVER_PID" 2>/dev/null || true
  fi
  for pid in "${pids[@]:-}"; do
    [[ "$pid" == "$VALIDATION_OBSERVER_PID" ]] && continue
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
on_exit() {
  local status=$?
  cleanup
  archive_run_outputs "$status"
}
trap on_exit EXIT

if [[ " $* " == *" --enable-room-local-guarded-online-validation "* ]]; then
  VALIDATION_TELEMETRY_DIR="$RUN_ARCHIVE_DIR/telemetry"
  VALIDATION_READY_FILE="$VALIDATION_TELEMETRY_DIR/observer.ready.json"
  mkdir -p "$VALIDATION_TELEMETRY_DIR"
  python3 scripts/local_subgoal_runner_mvp/room_local_online_validation_observer.py \
    --run-id "$RUN_NAME" \
    --output "$VALIDATION_TELEMETRY_DIR/command_odom_correlation.json" \
    --ready-file "$VALIDATION_READY_FILE" \
    >"$RUN_ARCHIVE_DIR/room_local_validation_observer.log" 2>&1 &
  pids+=("$!")
  VALIDATION_OBSERVER_PID="${pids[-1]}"
  for _ in $(seq 1 50); do
    [[ -s "$VALIDATION_READY_FILE" ]] && break
    sleep 0.1
  done
  [[ -s "$VALIDATION_READY_FILE" ]] || { echo "fatal: ROOM_LOCAL validation observer not ready" >&2; exit 76; }
fi

scripts/rgbd_danger_perception/run_rgbd_danger_perception.sh \
  > "$RUN_ARCHIVE_DIR/rgbd_danger_perception.log" 2>&1 &
pids+=("$!")

set +e
# Forward all state-machine options, including forced-room-entry MVP flags.
python3 scripts/local_subgoal_runner_mvp/navigation_state_machine.py "$@"
state_machine_status=$?
set -e

echo "state_machine_output:"
echo "  summary: debug/state_machine_navigation/state_machine_navigation_summary.json"
echo "  report: audit_reports/state_machine_navigation_report.md"
echo "perception_outputs:"
echo "  rgbd_danger: debug/rgbd_danger_perception/latest_danger_tracks.json"
echo "  room_search: debug/room_search/latest_room_search_diagnostic.json"

exit "$state_machine_status"
