#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/richard/simenv_official_clean
SCRIPT_DIR="$ROOT/scripts/l2_livox_icp_rtabmap_readiness"
CONFIG="$SCRIPT_DIR/l2_livox_icp_rtabmap_config.yaml"
OUT="$ROOT/debug/l2_livox_icp_rtabmap_readiness"
LOGS="$ROOT/logs"
START_L1S=false
FORCE_STATIC_TF=false
BASE_FRAME=base
RUN_DURATION=25
PIDS=()

for arg in "$@"; do
  case "$arg" in
    --start-l1s)
      START_L1S=true
      ;;
    --force-static-tf)
      FORCE_STATIC_TF=true
      ;;
    --base-frame=*)
      BASE_FRAME="${arg#*=}"
      ;;
    --duration=*)
      RUN_DURATION="${arg#*=}"
      ;;
    *)
      echo "[l2] unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT" "$LOGS"

log() {
  printf '[l2] %s\n' "$*"
}

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
  for node in \
    /l2_livox_readiness_monitor \
    /l2_livox_odom_gate \
    /l2_livox_static_tf \
    /l2_livox_icp_odometry_raw \
    /l2_livox_rtabmap \
    /l2_livox_l1s_provider; do
    rosnode kill "$node" >/dev/null 2>&1 || true
  done
  for pid in "${PIDS[@]:-}"; do
    kill -INT "$pid" >/dev/null 2>&1 || true
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_topic_once() {
  local topic="$1"
  local timeout_sec="$2"
  log "wait topic $topic"
  timeout "${timeout_sec}s" rostopic echo -n 1 "$topic" >/dev/null
}

precheck() {
  log "precheck branch $(git -C "$ROOT" branch --show-current)"
  rostopic list >/dev/null
  log "check /clock info"
  rostopic info /clock >/dev/null
  wait_topic_once /scan 10
  log "check /tf info"
  rostopic info /tf >/dev/null
  log "check /tf_static info"
  rostopic info /tf_static >/dev/null
  log "check TF $BASE_FRAME -> laser_livox"
  timeout 5s rosrun tf tf_echo "$BASE_FRAME" laser_livox >"$OUT/tf_${BASE_FRAME}_to_laser_livox.txt" 2>&1 || true
  if ! grep -q "Translation:" "$OUT/tf_${BASE_FRAME}_to_laser_livox.txt"; then
    log "TF $BASE_FRAME -> laser_livox unavailable"
    return 1
  fi
  if [ "$START_L1S" = true ]; then
    rosnode kill /l2_livox_l1s_provider >/dev/null 2>&1 || true
    start_bg l2_livox_l1s_provider "$LOGS/l2_livox_l1s_provider.log" \
      rosrun team_livox_scan_preprocess l1s_scan_clean_cloud_node.py __name:=l2_livox_l1s_provider
    sleep 3
  fi
  log "check clean cloud type"
  if ! timeout 3s rostopic type /team/livox/scan_cloud_filtered >/dev/null 2>&1; then
    log "/team/livox/scan_cloud_filtered is missing; start L1S first or pass --start-l1s"
    return 1
  fi
  local cloud_type
  cloud_type="$(timeout 5s rostopic type /team/livox/scan_cloud_filtered || true)"
  echo "$cloud_type" >"$OUT/clean_cloud_type.txt"
  if [ "$cloud_type" != "sensor_msgs/PointCloud2" ]; then
    log "clean cloud type is $cloud_type, expected sensor_msgs/PointCloud2"
    return 1
  fi
  log "sample clean cloud header"
  timeout 5s rostopic echo -n 1 /team/livox/scan_cloud_filtered/header >"$OUT/clean_cloud_header.txt"
  log "sample clean cloud hz"
  timeout 6s rostopic hz /team/livox/scan_cloud_filtered >"$OUT/clean_cloud_hz.txt" 2>&1 || true
}

precheck

rm -f "$OUT/readiness_events.jsonl" "$OUT/summary.json" "$OUT/l2_readiness.bag" "$OUT/l2_readiness.bag.active"

if [ "$FORCE_STATIC_TF" = true ]; then
  start_bg l2_livox_static_tf "$LOGS/l2_livox_static_tf.log" \
    rosrun tf static_transform_publisher 0.2 0.0 0.08 0.0 0.785 0.0 "$BASE_FRAME" laser_livox 100
  sleep 1
fi

start_bg l2_livox_icp_odometry_raw "$LOGS/l2_livox_icp_odometry_raw.log" \
  rosrun rtabmap_odom icp_odometry \
  __name:=l2_livox_icp_odometry_raw \
  _frame_id:="$BASE_FRAME" \
  _odom_frame_id:=team_livox_odom \
  _publish_tf:=false \
  _wait_for_transform:=true \
  _wait_for_transform_duration:=0.2 \
  _subscribe_scan_cloud:=true \
  _scan_cloud_max_points:=0 \
  _Odom/GuessMotion:=true \
  _Odom/ResetCountdown:=1 \
  _Icp/PointToPlane:=true \
  _Icp/VoxelSize:=0.0 \
  _Icp/PMOutlierRatio:=0.65 \
  scan:=/team/livox/unused_scan \
  scan_cloud:=/team/livox/scan_cloud_filtered \
  odom:=/team/livox/icp_odom_raw \
  odom_info:=/team/livox/icp_odom_info

sleep 3

start_bg l2_livox_odom_gate "$LOGS/l2_livox_odom_gate.log" \
  python3 "$SCRIPT_DIR/l2_livox_odom_gate.py" \
  _input_topic:=/team/livox/icp_odom_raw \
  _output_topic:=/team/livox/icp_odom_gated \
  _status_topic:=/team/livox/icp_odom_gate_status \
  _events_path:="$OUT/readiness_events.jsonl" \
  _max_delta_translation:=0.5 \
  _max_delta_yaw_deg:=30.0

sleep 2

start_bg l2_livox_rtabmap "$LOGS/l2_livox_rtabmap.log" \
  rosrun rtabmap_slam rtabmap \
  __name:=l2_livox_rtabmap \
  _frame_id:="$BASE_FRAME" \
  _map_frame_id:=team_livox_map \
  _odom_frame_id:=team_livox_odom \
  _publish_tf:=false \
  _subscribe_scan_cloud:=true \
  _subscribe_rgb:=false \
  _subscribe_depth:=false \
  _subscribe_scan:=false \
  _subscribe_odom_info:=false \
  _approx_sync:=false \
  _database_path:="$OUT/l2_livox_rtabmap.db" \
  _Mem/IncrementalMemory:=true \
  _Reg/Strategy:=1 \
  _Icp/VoxelSize:=0.0 \
  odom:=/team/livox/icp_odom_gated \
  scan_cloud:=/team/livox/scan_cloud_filtered

sleep 2

start_bg l2_livox_rosbag_record "$LOGS/l2_livox_rosbag_record.log" \
  timeout -s INT 15s rosbag record -O "$OUT/l2_readiness.bag" \
  /team/livox/scan_cloud_filtered \
  /team/livox/icp_odom_raw \
  /team/livox/icp_odom_gated \
  /l2_livox_rtabmap/info \
  /l2_livox_rtabmap/mapGraph \
  /l2_livox_rtabmap/mapPath \
  /tf \
  /tf_static

SOURCE_BRANCH="$(git -C "$ROOT" branch --show-current)"
python3 "$SCRIPT_DIR/l2_livox_readiness_monitor.py" \
  --config "$CONFIG" \
  --duration "$RUN_DURATION" \
  --source-branch "$SOURCE_BRANCH" \
  >"$LOGS/l2_livox_readiness_monitor.log" 2>&1 || true

if [ -f "$OUT/l2_readiness.bag.active" ] && [ ! -f "$OUT/l2_readiness.bag" ]; then
  log "rosbag left active file; attempting reindex is intentionally skipped"
fi

if [ -f "$OUT/l2_readiness.bag" ]; then
  rosbag info "$OUT/l2_readiness.bag" >"$OUT/l2_readiness_bag_info.txt" 2>&1 || true
fi

log "done"
