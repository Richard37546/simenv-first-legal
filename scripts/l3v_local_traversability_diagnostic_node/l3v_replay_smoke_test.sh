#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/richard/simenv_official_clean
SCRIPT_DIR="$ROOT/scripts/l3v_local_traversability_diagnostic_node"
OUT="$ROOT/debug/l3v_local_traversability_diagnostic_node"
REPORT="$ROOT/audit_reports/l3v_local_traversability_diagnostic_node_report.md"
BAG="$ROOT/debug/l3t_fresh_sensor_coverage_diagnostic/fresh_diagnostic.bag"
PLAY_DURATION=80
PLAY_RATE=0.7
PIDS=()
STARTED_ROSCORE=0

for arg in "$@"; do
  case "$arg" in
    --bag=*) BAG="${arg#*=}" ;;
    --play-duration=*) PLAY_DURATION="${arg#*=}" ;;
    --play-rate=*) PLAY_RATE="${arg#*=}" ;;
    *) echo "[l3v] unknown argument: $arg" >&2; exit 2 ;;
  esac
done

set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/devel/setup.bash" 2>/dev/null || true
set -u

mkdir -p "$OUT" "$ROOT/audit_reports"
rm -f "$OUT"/*.json "$OUT"/*.pgm "$OUT"/*.log "$OUT"/*.pid 2>/dev/null || true

log() { printf '[l3v] %s\n' "$*"; }

start_bg() {
  local name="$1"
  local logfile="$2"
  shift 2
  log "start $name"
  "$@" >"$logfile" 2>&1 &
  PIDS+=("$!")
  echo "$!" >"$OUT/${name}.pid"
}

cleanup() {
  local status=$?
  log "cleanup status=$status"
  rosnode kill /l3v_local_traversability_node >/dev/null 2>&1 || true
  for pid in "${PIDS[@]:-}"; do
    kill -INT "$pid" >/dev/null 2>&1 || true
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  if [ "$STARTED_ROSCORE" = "1" ]; then
    pkill -f "roscore" >/dev/null 2>&1 || true
    pkill -f "rosmaster" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [ ! -s "$BAG" ]; then
  echo "[l3v] missing bag: $BAG" >&2
  exit 1
fi

if ! timeout 5s rostopic list >/dev/null 2>&1; then
  start_bg roscore "$OUT/roscore.log" roscore
  STARTED_ROSCORE=1
  sleep 5
fi

rosparam set use_sim_time true

start_bg l3v_node "$OUT/l3v_node.log" \
  python3 "$SCRIPT_DIR/l3v_local_traversability_node.py" \
  _publish_hz:=2.0 \
  _max_points_per_cloud:=3500 \
  _depth_stride:=20

sleep 5

start_bg rosbag_play "$OUT/rosbag_play.log" \
  rosbag play --clock -r "$PLAY_RATE" -u "$PLAY_DURATION" \
  --topics \
  /team/livox/icp_odom_gated \
  /team/livox/scan_cloud_filtered \
  /real_sense/depth/image_raw \
  /real_sense/depth/points \
  /real_sense/rgb/camera_info \
  /tf \
  /tf_static \
  /clock \
  --bags "$BAG"

sleep 35

python3 "$SCRIPT_DIR/l3v_diagnostic_readiness_check.py" \
  --output-dir "$OUT" \
  --report "$REPORT" \
  --timeout 60

log "done"
