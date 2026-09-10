#!/usr/bin/env bash
# Gracefully stops only the recorded P2KG9U audit tmux session.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVE_ENV="$REPO_ROOT/debug/odom_accuracy_audit_v1/p2kg9_current_door_portal_shadow_048/online_run/p2kg9u_active_env.sh"
ENV_FILE="$ACTIVE_ENV"
DRY_RUN=false
# PortalShadow.close() can wait up to 30 s for its compute worker before it
# writes the two final audit records.  Leave a small margin for those atomic
# writes so the stop supervisor does not race a healthy graceful shutdown.
STOP_WAIT_SEC="${P2KG9U_STOP_WAIT_SEC:-45}"

usage() { echo "usage: $0 [--env-file FILE] [--dry-run] [--timeout-sec SECONDS]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="${2:?missing environment file}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --timeout-sec) STOP_WAIT_SEC="${2:?missing timeout}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 64 ;;
  esac
done

[[ "$STOP_WAIT_SEC" =~ ^[1-9][0-9]*$ ]] || { echo "timeout must be a positive integer" >&2; exit 64; }
[[ -f "$ENV_FILE" ]] || { echo "P2KG9U NOT_PREPARED"; exit 65; }
# shellcheck disable=SC1090
source "$ENV_FILE"
[[ -s "$P2KG9U_PANE_FILE" ]] || { echo "P2KG9U PARTIAL_START: pane record missing"; exit 66; }

pane_for_title() { awk -v title="$1" '$3 == title {print $1; exit}' "$P2KG9U_PANE_FILE"; }
ROS_BAG_PANE="$(pane_for_title p2kg9u-rosbag)"
SHADOW_PANE="$(pane_for_title p2kg9u-shadow)"
GATE_PANE="$(pane_for_title p2kg12-room-zone-effect-gate)"
CONTINUOUS_YAW_PANE="$(pane_for_title continuous-odom-imu-yaw-shadow)"
[[ -n "$ROS_BAG_PANE" && -n "$SHADOW_PANE" && -n "$CONTINUOUS_YAW_PANE" ]] || { echo "P2KG9U PARTIAL_START: recorded rosbag/shadow pane identities are incomplete"; exit 66; }

verify_bag_complete() {
  [[ -d "$BAG_DIR" ]] || { echo "P2KG9U ROSBAG_FAILED: bag directory missing" >&2; return 1; }
  if compgen -G "$BAG_DIR/*.bag.active" >/dev/null; then
    echo "P2KG9U ROSBAG_FAILED: .bag.active remains after graceful stop" >&2
    return 1
  fi
  compgen -G "$BAG_DIR/*.bag" >/dev/null || { echo "P2KG9U ROSBAG_FAILED: completed .bag missing" >&2; return 1; }
  echo "P2KG9U rosbag active-file check passed"
}

verify_shadow_output() {
  python3 - "$SHADOW_RUN_DIR" <<'PY'
import json
import pathlib
import sys
run_dir = pathlib.Path(sys.argv[1])
frame = run_dir / "frame_accounting.json"
shutdown = run_dir / "queue_and_shutdown.json"
missing = [str(p) for p in (frame, shutdown) if not p.is_file()]
if missing:
    raise SystemExit("missing Shadow output: " + ", ".join(missing))
data = json.loads(shutdown.read_text(encoding="utf-8"))
checks = {
    "compute_queue_remaining": data.get("compute_queue_remaining") == 0,
    "writer_queue_remaining": data.get("writer_queue_remaining", data.get("queue_remaining")) == 0,
    "compute_worker_stopped": data.get("compute_worker_alive") is False,
    "writer_worker_stopped": data.get("writer_worker_alive", data.get("worker_alive")) is False,
    "writer_errors_empty": data.get("writer_errors") == [],
}

failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("Shadow shutdown verification failed: " + ", ".join(failed))
print("Shadow shutdown verified: " + json.dumps(checks, sort_keys=True))
PY
}

verify_continuous_yaw_shadow_output() {
  python3 - "$CONTINUOUS_YAW_SHADOW_DIR" <<'PY'
import json
import pathlib
import sys
directory = pathlib.Path(sys.argv[1])
ready = directory / "ready.json"
events = directory / "shadow_events.jsonl"
result = directory / "shadow_result.json"
missing = [str(path) for path in (ready, events, result) if not path.is_file()]
if missing:
    raise SystemExit("missing continuous yaw Shadow output: " + ", ".join(missing))
data = json.loads(result.read_text(encoding="utf-8"))
checks = {
    "closed": data.get("status") == "CLOSED",
    "audit_only": data.get("audit_authority") is True,
    "command_authority_false": data.get("command_authority") is False,
    "odom_authority_false": data.get("odom_authority") is False,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("continuous yaw Shadow shutdown verification failed: " + ", ".join(failed))
print("Continuous yaw Shadow shutdown verified: " + json.dumps(checks, sort_keys=True))
PY
}

pane_live() {
  tmux list-panes -t "$P2KG9U_TMUX_SESSION" -F '#{pane_id} #{pane_dead}' 2>/dev/null |
    awk -v id="$1" '$1 == id && ($2 == "" || $2 == "0") {found=1} END {exit !found}'
}

wait_for_pane_exit() {
  local pane="$1" label="$2" deadline=$(( $(date +%s) + STOP_WAIT_SEC ))
  while pane_live "$pane"; do
    if (( $(date +%s) >= deadline )); then
      echo "P2KG9U partial stop timeout for $label pane $pane; preserving audit session" >&2
      return 1
    fi
    sleep 0.1
  done
}

stop_supervised_rosbag_if_needed() {
  local pid_file="$SHADOW_RUN_DIR/rosbag.pid" pid deadline
  [[ -r "$pid_file" ]] || return 0
  pid="$(cat "$pid_file")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || { echo "P2KG9U ROSBAG_FAILED: invalid supervisor pid file" >&2; return 1; }
  process_running() {
    kill -0 "$1" 2>/dev/null && ! ps -o stat= -p "$1" 2>/dev/null | grep -q '^[[:space:]]*Z'
  }
  if process_running "$pid"; then
    kill -INT "$pid" 2>/dev/null || true
    deadline=$(( $(date +%s) + STOP_WAIT_SEC ))
    while process_running "$pid"; do
      (( $(date +%s) < deadline )) || { echo "P2KG9U ROSBAG_FAILED: supervisor did not exit after SIGINT" >&2; return 1; }
      sleep 0.1
    done
  fi
}

if [[ "$DRY_RUN" == true ]]; then
  echo "DRY RUN: Ctrl-C rosbag pane $ROS_BAG_PANE"
  echo "DRY RUN: wait for rosbag exit; verify no .bag.active"
  echo "DRY RUN: Ctrl-C Shadow pane $SHADOW_PANE"
  echo "DRY RUN: wait for Shadow exit; verify frame_accounting and queue_and_shutdown"
  echo "DRY RUN: Ctrl-C continuous yaw Shadow pane $CONTINUOUS_YAW_PANE"
  echo "DRY RUN: wait for continuous yaw Shadow exit; verify ready/events/result with zero command authority"
  echo "DRY RUN: Ctrl-C Portal effect gate pane ${GATE_PANE:-none}"
  echo "DRY RUN: remove only audit session $P2KG9U_TMUX_SESSION after all three audit panes exit"
  exit 0
fi

if ! tmux has-session -t "$P2KG9U_TMUX_SESSION" 2>/dev/null; then
  echo "P2KG9U audit tmux session missing; attempting supervised rosbag-only graceful close"
  stop_supervised_rosbag_if_needed
  verify_bag_complete
  echo "P2KG9U audit session was already absent; bag closure verified"
  exit 0
fi

if pane_live "$ROS_BAG_PANE"; then
  tmux send-keys -t "$ROS_BAG_PANE" C-c
  wait_for_pane_exit "$ROS_BAG_PANE" "rosbag"
fi
verify_bag_complete
if pane_live "$SHADOW_PANE"; then
  tmux send-keys -t "$SHADOW_PANE" C-c
  wait_for_pane_exit "$SHADOW_PANE" "Shadow"
fi
verify_shadow_output
if pane_live "$CONTINUOUS_YAW_PANE"; then
  tmux send-keys -t "$CONTINUOUS_YAW_PANE" C-c
  wait_for_pane_exit "$CONTINUOUS_YAW_PANE" "continuous yaw Shadow"
fi
verify_continuous_yaw_shadow_output
if [[ -n "$GATE_PANE" ]] && pane_live "$GATE_PANE"; then
  tmux send-keys -t "$GATE_PANE" C-c
  wait_for_pane_exit "$GATE_PANE" "Portal effect gate"
fi
tmux kill-session -t "$P2KG9U_TMUX_SESSION" 2>/dev/null || true
echo "P2KG9U audit session stopped gracefully; bag and Shadow shutdown were verified under $SHADOW_RUN_DIR"
