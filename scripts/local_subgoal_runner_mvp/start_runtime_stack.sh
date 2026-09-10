#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/richard/simenv_official_clean"
LOG_DIR="$ROOT/debug/runtime_stack"
PID_FILE="$LOG_DIR/runtime_stack.pids"

GUI="${GUI:-true}"
PAUSED="${PAUSED:-false}"
UNITREE_CTRL_DT="${UNITREE_CTRL_DT:-0.006}"
ROS_SETUP="/opt/ros/noetic/setup.bash"
DEVEL_SETUP="$ROOT/devel/setup.bash"

mkdir -p "$LOG_DIR"
: > "$PID_FILE"

source_ros() {
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
  # shellcheck disable=SC1090
  source "$DEVEL_SETUP" 2>/dev/null || true
}

record_pid() {
  local name="$1"
  local pid="$2"
  echo "$name $pid" >> "$PID_FILE"
}

cleanup() {
  echo
  echo "[runtime_stack] stopping background processes..."
  if [[ -f "$PID_FILE" ]]; then
    tac "$PID_FILE" | while read -r name pid; do
      if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[runtime_stack] stopping $name pid=$pid"
        kill "$pid" 2>/dev/null || true
      fi
    done
  fi
}
trap cleanup INT TERM

wait_for_ros() {
  local timeout_sec="${1:-90}"
  local start
  start="$(date +%s)"
  echo "[runtime_stack] waiting for ROS master..."
  while true; do
    if rostopic list >/dev/null 2>&1; then
      echo "[runtime_stack] ROS master is ready"
      return 0
    fi
    if (( "$(date +%s)" - start >= timeout_sec )); then
      echo "[runtime_stack] ERROR: ROS master not ready after ${timeout_sec}s" >&2
      return 1
    fi
    sleep 1
  done
}

wait_for_service() {
  local service_name="$1"
  local timeout_sec="${2:-90}"
  local start
  start="$(date +%s)"
  echo "[runtime_stack] waiting for service $service_name..."
  while true; do
    if rosservice list 2>/dev/null | grep -qx "$service_name"; then
      echo "[runtime_stack] service $service_name is ready"
      return 0
    fi
    if (( "$(date +%s)" - start >= timeout_sec )); then
      echo "[runtime_stack] ERROR: service $service_name not ready after ${timeout_sec}s" >&2
      return 1
    fi
    sleep 1
  done
}

cd "$ROOT"
source_ros

echo "[runtime_stack] logs: $LOG_DIR"
echo "[runtime_stack] starting simulator: GUI=$GUI PAUSED=$PAUSED UNITREE_CTRL_DT=$UNITREE_CTRL_DT"
(
  cd "$ROOT"
  source_ros
  GUI="$GUI" PAUSED="$PAUSED" UNITREE_CTRL_DT="$UNITREE_CTRL_DT" ./auto.sh
) >"$LOG_DIR/auto.log" 2>&1 &
record_pid "auto" "$!"

wait_for_ros 120
wait_for_service "/set_door_state" 120

echo "[runtime_stack] opening main entrance"
rosservice call /set_door_state "{door_id: 'main_entrance', open: true}" \
  >"$LOG_DIR/open_main_entrance.log" 2>&1 || {
  echo "[runtime_stack] ERROR: failed to open main entrance, see $LOG_DIR/open_main_entrance.log" >&2
  exit 1
}

echo "[runtime_stack] starting slam_bev runtime"
(
  cd "$ROOT"
  source_ros
  bash scripts/slam_bev_runtime/run_slam_bev_runtime.sh
) >"$LOG_DIR/slam_bev_runtime.log" 2>&1 &
record_pid "slam_bev_runtime" "$!"

echo "[runtime_stack] starting L3V local traversability node"
(
  cd "$ROOT"
  source_ros
  python3 scripts/l3v_local_traversability_diagnostic_node/l3v_local_traversability_node.py
) >"$LOG_DIR/l3v_local_traversability_node.log" 2>&1 &
record_pid "l3v_local_traversability_node" "$!"

echo "[runtime_stack] started. PID file: $PID_FILE"
echo "[runtime_stack] keep this terminal open. Press Ctrl+C to stop auto/slam/L3V."

wait
