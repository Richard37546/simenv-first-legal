#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/richard/simenv_official_clean"
SESSION="${SESSION:-simenv_stack}"
GUI="${GUI:-true}"
PAUSED="${PAUSED:-false}"
UNITREE_CTRL_DT="${UNITREE_CTRL_DT:-0.006}"
UNITREE_RL_POLICY="${UNITREE_RL_POLICY:-stair}"
P2KG15_ANGULAR_ACTUATION_TELEMETRY="${P2KG15_ANGULAR_ACTUATION_TELEMETRY:-0}"
P2KG15_ANGULAR_ACTUATION_TELEMETRY_RATE_HZ="${P2KG15_ANGULAR_ACTUATION_TELEMETRY_RATE_HZ:-10.0}"
STARTUP_ANCHOR_AUDIT_ENABLED="${STARTUP_ANCHOR_AUDIT_ENABLED:-0}"
STARTUP_ANCHOR_RUN_ID="${STARTUP_ANCHOR_RUN_ID:-}"
STARTUP_ANCHOR_ARCHIVE_DIR="${STARTUP_ANCHOR_ARCHIVE_DIR:-}"
STARTUP_ANCHOR_READY_FILE="${STARTUP_ANCHOR_READY_FILE:-}"
LOG_DIR="$ROOT/debug/runtime_stack_tmux"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[runtime_stack_tmux] ERROR: tmux is not installed." >&2
  echo "[runtime_stack_tmux] Use scripts/local_subgoal_runner_mvp/start_runtime_stack.sh instead, or install tmux." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[runtime_stack_tmux] ERROR: tmux session '$SESSION' already exists." >&2
  echo "[runtime_stack_tmux] Attach with: tmux attach -t $SESSION" >&2
  echo "[runtime_stack_tmux] Or stop it with: tmux kill-session -t $SESSION" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

AUTO_CMD="cd '$ROOT'; source /opt/ros/noetic/setup.bash; source ./devel/setup.bash 2>/dev/null || true; echo '[auto] type 2 then 6 here after controller is ready'; GUI='$GUI' PAUSED='$PAUSED' UNITREE_CTRL_DT='$UNITREE_CTRL_DT' UNITREE_RL_POLICY='$UNITREE_RL_POLICY' P2KG15_ANGULAR_ACTUATION_TELEMETRY='$P2KG15_ANGULAR_ACTUATION_TELEMETRY' P2KG15_ANGULAR_ACTUATION_TELEMETRY_RATE_HZ='$P2KG15_ANGULAR_ACTUATION_TELEMETRY_RATE_HZ' STARTUP_ANCHOR_AUDIT_ENABLED='$STARTUP_ANCHOR_AUDIT_ENABLED' STARTUP_ANCHOR_RUN_ID='$STARTUP_ANCHOR_RUN_ID' STARTUP_ANCHOR_ARCHIVE_DIR='$STARTUP_ANCHOR_ARCHIVE_DIR' STARTUP_ANCHOR_READY_FILE='$STARTUP_ANCHOR_READY_FILE' ./auto.sh"

SLAM_CMD="cd '$ROOT'; source /opt/ros/noetic/setup.bash; source ./devel/setup.bash 2>/dev/null || true; export STARTUP_ANCHOR_RUN_ID='$STARTUP_ANCHOR_RUN_ID'; echo '[slam] waiting for ROS master...'; until rostopic list >/dev/null 2>&1; do sleep 1; done; echo '[slam] waiting for /set_door_state...'; until rosservice list 2>/dev/null | grep -qx '/set_door_state'; do sleep 1; done; echo '[slam] opening main entrance'; rosservice call /set_door_state \"{door_id: 'main_entrance', open: true}\"; echo '[slam] starting slam_bev runtime'; bash scripts/slam_bev_runtime/run_slam_bev_runtime.sh"

L3V_CMD="cd '$ROOT'; source /opt/ros/noetic/setup.bash; source ./devel/setup.bash 2>/dev/null || true; echo '[l3v] waiting for ROS master...'; until rostopic list >/dev/null 2>&1; do sleep 1; done; echo '[l3v] starting L3V local traversability node'; python3 scripts/l3v_local_traversability_diagnostic_node/l3v_local_traversability_node.py"

tmux new-session -d -s "$SESSION" -n runtime "$AUTO_CMD"
tmux split-window -t "$SESSION:runtime" -h "$SLAM_CMD"
tmux split-window -t "$SESSION:runtime.1" -v "$L3V_CMD"
tmux select-pane -t "$SESSION:runtime.0"

echo "[runtime_stack_tmux] started tmux session: $SESSION"
echo "[runtime_stack_tmux] active pane is auto.sh. Type 2 then 6 in that pane."
echo "[runtime_stack_tmux] detach: Ctrl-b then d"
echo "[runtime_stack_tmux] stop all panes: tmux kill-session -t $SESSION"

tmux attach -t "$SESSION"
