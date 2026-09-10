#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/richard/simenv_official_clean
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

source /opt/ros/noetic/setup.bash
if [ -f "$ROOT/devel/setup.bash" ]; then
  source "$ROOT/devel/setup.bash"
fi

PIDS=()

stop_prior_odom_gates() {
  # The result-coordinate writer accepts only one current-run continuity
  # identity.  A gate left over from a prior runtime can otherwise keep
  # publishing a valid-looking but foreign identity on the shared topic.
  for node in /slam_bev_odom_gate /l2_livox_odom_gate /safe_startup_odom_gate; do
    rosnode kill "$node" >/dev/null 2>&1 || true
  done
}

start_bg() {
  local name="$1"
  local logfile="$2"
  shift 2
  echo "[slam_bev_runtime] start $name"
  "$@" >"$logfile" 2>&1 &
  PIDS+=("$!")
  echo "$!" >"$LOGS/${name}.pid"
}

cleanup() {
  local status=$?
  for node in /bev_perception /slam_bev_l1s_provider /slam_bev_icp_odometry_raw /slam_bev_odom_gate; do
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

stop_prior_odom_gates

start_bg bev_perception "$LOGS/slam_bev_runtime_bev_perception.log" \
  roslaunch bev_perception bev_perception.launch

start_bg slam_bev_l1s_provider "$LOGS/slam_bev_runtime_l1s.log" \
  rosrun team_livox_scan_preprocess l1s_scan_clean_cloud_node.py __name:=slam_bev_l1s_provider

sleep 2

start_bg slam_bev_icp_odometry_raw "$LOGS/slam_bev_runtime_icp_odometry_raw.log" \
  rosrun rtabmap_odom icp_odometry \
  __name:=slam_bev_icp_odometry_raw \
  _frame_id:=base \
  _odom_frame_id:=team_livox_odom \
  _publish_tf:=false \
  _wait_for_transform:=true \
  _wait_for_transform_duration:=0.2 \
  _subscribe_scan_cloud:=true \
  _scan_cloud_max_points:=0 \
  _Odom/GuessMotion:=true \
  _wait_imu_to_init:=true \
  _Odom/ResetCountdown:=1 \
  _Icp/PointToPlane:=true \
  _Icp/VoxelSize:=0.0 \
  _Icp/PMOutlierRatio:=0.65 \
  scan:=/team/livox/unused_scan \
  scan_cloud:=/team/livox/scan_cloud_filtered \
  imu:=/trunk_imu \
  odom:=/team/livox/icp_odom_raw \
  odom_info:=/team/livox/icp_odom_info

sleep 2

start_bg slam_bev_odom_gate "$LOGS/slam_bev_runtime_odom_gate.log" \
  python3 "$ROOT/scripts/l2_livox_icp_rtabmap_readiness/l2_livox_odom_gate.py" \
  __name:=slam_bev_odom_gate \
  _input_topic:=/team/livox/icp_odom_raw \
  _output_topic:=/team/livox/icp_odom_gated \
  _status_topic:=/team/livox/icp_odom_gate_status \
  _output_frame_id:=team_livox_odom \
  _output_child_frame_id:=base \
  _events_path:="$ROOT/debug/slam_bev_runtime/odom_gate_events.jsonl" \
  _audit_run_epoch_id:="${STARTUP_ANCHOR_RUN_ID:-UNSCOPED_RUN}" \
  _audit_event_topic:=/audit/startup_anchor/odom_epoch_event \
  _max_delta_translation:=0.5 \
  _max_delta_yaw_deg:=30.0 \
  _startup_stable_samples:=10 \
  _imu_topic:=/trunk_imu \
  _max_input_gap_sec:=0.25 \
  _recovery_imu_max_age_sec:=0.05 \
  _recovery_consistent_samples:=10 \
  _recovery_max_yaw_error_deg:=2.5

echo "[slam_bev_runtime] running. Ctrl-C to stop."
wait
