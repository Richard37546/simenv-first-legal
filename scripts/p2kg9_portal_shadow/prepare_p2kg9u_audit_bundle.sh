#!/usr/bin/env bash
# Starts only the P2K-G9U audit Shadow and rosbag, with explicit readiness.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_OUTPUT_ROOT="$REPO_ROOT/debug/odom_accuracy_audit_v1/p2kg9_current_door_portal_shadow_048/online_run"
DEFAULT_BAG_OUTPUT_ROOT="/mnt/d/data/simenv_audit_bags/p2kg9_current_door_portal_shadow_048/online_run"
OUTPUT_ROOT="${P2KG9U_OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
BAG_OUTPUT_ROOT="${P2KG9U_BAG_OUTPUT_ROOT:-$DEFAULT_BAG_OUTPUT_ROOT}"
OUTPUT_ROOT_EXPLICIT=false
BAG_OUTPUT_ROOT_EXPLICIT=false
RUN_ID="online_$(date +%Y%m%d_%H%M%S)_p2kg9u"
CHECK_ONLY=false
MODE=""
ENV_FILE=""
# Reuses the former bag-worker budget: 300 polls x 0.1 s.
START_WAIT_TENTHS="${P2KG9U_START_WAIT_TENTHS:-300}"

usage() { echo "usage: $0 [--run-id RUN_ID] [--output-root DIRECTORY] [--bag-output-root DIRECTORY] [--check]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:?missing run id}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing output root}"; OUTPUT_ROOT_EXPLICIT=true; shift 2 ;;
    --bag-output-root) BAG_OUTPUT_ROOT="${2:?missing bag output root}"; BAG_OUTPUT_ROOT_EXPLICIT=true; shift 2 ;;
    --check) CHECK_ONLY=true; shift ;;
    --shadow-pane|--gate-pane|--geometry-pane|--continuous-yaw-shadow-pane|--rosbag-pane) MODE="$1"; ENV_FILE="${2:?missing environment file}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 64 ;;
  esac
done

# Explicit --output-root calls are used by isolated legacy tools/tests.  Keep
# their bag location local unless they explicitly request a different one.
if [[ "$OUTPUT_ROOT_EXPLICIT" == true && "$BAG_OUTPUT_ROOT_EXPLICIT" == false && -z "${P2KG9U_BAG_OUTPUT_ROOT:-}" ]]; then
  BAG_OUTPUT_ROOT="$OUTPUT_ROOT"
fi

load_env() { [[ -f "$ENV_FILE" ]] || exit 65; source "$ENV_FILE"; }

# ROS Noetic's setup fragment reads ROS_DISTRO before initializing it.  This
# bundle intentionally uses `set -u`, and pane processes may start with no ROS
# environment inherited.  Disable nounset only while sourcing setup files, then
# restore the bundle's strict-mode contract before starting an audit process.
source_ros_workspace() {
  set +u
  source /opt/ros/noetic/setup.bash
  source "$REPO_ROOT/devel/setup.bash" 2>/dev/null || true
  set -u
}
if [[ "$MODE" == "--shadow-pane" ]]; then
  load_env
  startup_log="$SHADOW_OUTPUT_ROOT/${SHADOW_RUN_ID}.portal_shadow.startup.log"
  launcher="${P2KG9U_SHADOW_LAUNCHER:-$REPO_ROOT/scripts/p2kg9_portal_shadow/start_passive_portal_shadow_audit.sh}"
  child=""
  stop_shadow_child() {
    trap - INT TERM HUP
    if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
      # The pane owns a wrapper shell, not the ROS node.  Relay one graceful
      # signal and wait so PortalShadow.close() can persist its final
      # frame_accounting and queue_and_shutdown records before pane exit.
      kill -INT "$child" 2>/dev/null || true
      wait "$child" 2>/dev/null || true
    fi
    exit 0
  }
  trap stop_shadow_child INT TERM HUP
  # Background shell children inherit ignored terminal signals.  Reset that
  # disposition in a dedicated session so the relay above reaches rospy.
  setsid bash -c 'trap - INT HUP TERM; exec "$@"' bash "$launcher" \
    --run-id "$SHADOW_RUN_ID" --output-root "$SHADOW_OUTPUT_ROOT" >"$startup_log" 2>&1 &
  child=$!
  for _ in $(seq 1 "$START_WAIT_TENTHS"); do
    if [[ -d "$SHADOW_RUN_DIR" ]]; then
      mkdir -p "$SHADOW_RUN_DIR/logs"
      mv "$startup_log" "$SHADOW_RUN_DIR/logs/portal_shadow.log"
      wait "$child"
      child_status=$?
      trap - INT TERM HUP
      exit "$child_status"
    fi
    kill -0 "$child" 2>/dev/null || exit 70
    sleep 0.1
  done
  stop_shadow_child
  exit 70
fi
if [[ "$MODE" == "--gate-pane" ]]; then
  load_env
  mkdir -p "$SHADOW_RUN_DIR/logs"
  source_ros_workspace
  exec python3 "$REPO_ROOT/scripts/p2kg9_portal_shadow/portal_room_zone_effect_gate.py" \
    >"$SHADOW_RUN_DIR/logs/portal_room_zone_effect_gate.log" 2>&1
fi
if [[ "$MODE" == "--geometry-pane" ]]; then
  load_env
  mkdir -p "$SHADOW_RUN_DIR/logs"
  source_ros_workspace
  exec python3 "$REPO_ROOT/scripts/minimum_zero_progress_evidence/geometry_sidecar.py" \
    --building-sdf "$REPO_ROOT/generated_building/model.sdf" \
    >"$SHADOW_RUN_DIR/logs/nearby_model_geometry.log" 2>&1
fi
if [[ "$MODE" == "--continuous-yaw-shadow-pane" ]]; then
  load_env
  mkdir -p "$CONTINUOUS_YAW_SHADOW_DIR/logs"
  source_ros_workspace
  exec python3 "$REPO_ROOT/scripts/l2_livox_icp_rtabmap_readiness/continuous_odom_imu_yaw_shadow.py" \
    _output_dir:="$CONTINUOUS_YAW_SHADOW_DIR" \
    _status_topic:="$CONTINUOUS_YAW_SHADOW_STATUS_TOPIC" \
    _odom_topic:=/team/livox/icp_odom_raw \
    _imu_topic:=/trunk_imu \
    _max_imu_age_sec:=0.05 \
    _max_yaw_error_deg:=2.9 \
    _required_consecutive_samples:=10 \
    >"$CONTINUOUS_YAW_SHADOW_DIR/logs/continuous_odom_imu_yaw_shadow.log" 2>&1
fi
if [[ "$MODE" == "--rosbag-pane" ]]; then
  load_env
  source_ros_workspace
  mkdir -p "$SHADOW_RUN_DIR/logs"
  mkdir -p "$BAG_DIR"
  launcher="${P2KG9U_ROSBAG_LAUNCHER:-rosbag}"
  pid_file="$SHADOW_RUN_DIR/rosbag.pid"
  rosbag_pid=""
  stop_rosbag_gracefully() {
    if [[ -n "$rosbag_pid" ]] && kill -0 "$rosbag_pid" 2>/dev/null; then
      # SIGINT is the signal rosbag record uses to flush its index and rename
      # the temporary .bag.active output to .bag.
      kill -INT "$rosbag_pid" 2>/dev/null || true
      wait "$rosbag_pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  }
  # Keep rosbag in its own session.  If the tmux pane receives HUP/TERM, this
  # supervisor receives the signal first and can request a graceful rosbag
  # close instead of leaving an orphaned .bag.active file.
  trap 'stop_rosbag_gracefully; exit 0' HUP INT TERM
  # A shell starts asynchronous children with SIGINT ignored.  Reset that
  # disposition before exec so rosbag can receive the supervisor's SIGINT.
  setsid bash -c 'trap - INT HUP TERM; exec "$@"' bash "$launcher" record -O "$BAG_DIR/$SHADOW_RUN_ID.bag" \
    /clock /gazebo/model_states /team/livox/scan_cloud_filtered /team/livox/icp_odom_raw \
    /team/livox/icp_odom_info /team/livox/icp_odom_gated /team/livox/icp_odom_gate_status \
    /team/local_traversability_grid /team/traversability_status /audit/startup_anchor/odom_epoch_event /tf /tf_static /cmd_vel \
    /cmd_vel_raw /audit/stair_cmd_receipt /audit/stair_policy_cmd_input \
    /imu_velocity_follower/status /trunk_imu /team/doorway_candidate \
    /team/danger_tracks /team/danger_hypotheses \
    /audit/p2kg9/portal_candidate /audit/p2kg9/portal_status /audit/p2kg9/event_journal \
    /audit/p2kg7r/ray_evidence_status /audit/p2kg11/portal_frame \
    /audit/p2kg12/room_zone_state /audit/p2kg12/portal_effect_gate \
    /audit/p2kg15/selected_rl_policy /audit/p2kg15/angular_actuation_state \
    /audit/p2kg15/doorway_raw_occupancy_provenance /audit/nearby_model_geometry \
    /audit/continuous_odom_imu_yaw_shadow_status \
    /unitree/rl_mode_ready >"$SHADOW_RUN_DIR/logs/rosbag.log" 2>&1 &
  rosbag_pid="$!"
  printf '%s\n' "$rosbag_pid" > "$pid_file"
  if wait "$rosbag_pid"; then
    rm -f "$pid_file"
    exit 0
  fi
  status=$?
  rm -f "$pid_file"
  exit "$status"
fi

[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid run id" >&2; exit 64; }
[[ "$START_WAIT_TENTHS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid startup wait budget" >&2; exit 64; }
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
BAG_OUTPUT_ROOT="$(realpath -m "$BAG_OUTPUT_ROOT")"
for script in "$REPO_ROOT/scripts/p2kg9_portal_shadow/start_passive_portal_shadow_audit.sh" "$REPO_ROOT/scripts/p2kg9_portal_shadow/portal_room_zone_effect_gate.py" "$REPO_ROOT/scripts/p2kg9_portal_shadow/status_p2kg9u_audit_bundle.sh" "$REPO_ROOT/scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh"; do
  [[ -f "$script" && -x "$script" ]] || { echo "required executable missing: $script" >&2; exit 65; }
done
[[ -f "$REPO_ROOT/scripts/minimum_zero_progress_evidence/geometry_sidecar.py" ]] || { echo "geometry sidecar missing" >&2; exit 65; }
[[ -f "$REPO_ROOT/scripts/l2_livox_icp_rtabmap_readiness/continuous_odom_imu_yaw_shadow.py" ]] || { echo "continuous yaw shadow missing" >&2; exit 65; }
[[ ! -e "$OUTPUT_ROOT/$RUN_ID" ]] || { echo "run output already exists: $OUTPUT_ROOT/$RUN_ID" >&2; exit 66; }
if [[ "$CHECK_ONLY" == true ]]; then
  echo "P2KG9U bundle check passed; bag_output_root=$BAG_OUTPUT_ROOT; no ROS, Gazebo, runner, or robot process started."
  exit 0
fi
rostopic list >/dev/null 2>&1 || { echo "P2KG9U STARTING_SHADOW: ROS master unavailable" >&2; exit 69; }
for topic in /clock /gazebo/model_states /team/livox/scan_cloud_filtered /team/livox/icp_odom_raw /team/livox/icp_odom_gated /team/local_traversability_grid /team/traversability_status /tf /tf_static; do
  rostopic type "$topic" >/dev/null 2>&1 || { echo "P2KG9U STARTING_SHADOW: H1 topic unavailable: $topic" >&2; exit 70; }
done
# Angular-actuation telemetry is optional diagnostics.  The official Stair
# baseline does not enable it by default, so only policy identity and RL
# readiness qualify the audit bundle.  The optional topic remains in rosbag.
for topic in /audit/p2kg15/selected_rl_policy /unitree/rl_mode_ready; do
  rostopic type "$topic" >/dev/null 2>&1 || { echo "P2KG15_STAIR_PREFLIGHT_FAILED: required policy/RL topic unavailable: $topic" >&2; exit 76; }
done
policy_record="$(timeout 5 rostopic echo -n 1 /audit/p2kg15/selected_rl_policy 2>/dev/null || true)"
[[ "$policy_record" == *"stair"* && "$policy_record" == *"src/unitree_guide/logs/policy_act_inference_stair.pt"* && "$policy_record" == *"2d5aa72511c0c6609c02f4105845eee6974d3d73431497f8f35306da9588fe14"* ]] || {
  echo "P2KG15_STAIR_PREFLIGHT_FAILED: selected policy record is not the required stair asset/hash" >&2
  exit 77
}
rl_ready_record="$(timeout 5 rostopic echo -n 1 /unitree/rl_mode_ready 2>/dev/null || true)"
[[ "${rl_ready_record,,}" == *"data: true"* ]] || { echo "P2KG15_STAIR_PREFLIGHT_FAILED: /unitree/rl_mode_ready is not true" >&2; exit 78; }

mkdir -p "$OUTPUT_ROOT" "$BAG_OUTPUT_ROOT"
ENV_FILE="$OUTPUT_ROOT/p2kg9u_${RUN_ID}.env.sh"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"; BAG_DIR="$BAG_OUTPUT_ROOT/$RUN_ID/bag"; SESSION="p2kg9u_$RUN_ID"; PANE_FILE="$OUTPUT_ROOT/${RUN_ID}.audit_panes.txt"
CONTINUOUS_YAW_SHADOW_DIR="$RUN_DIR/continuous_odom_imu_yaw_shadow"; CONTINUOUS_YAW_SHADOW_STATUS_TOPIC="/audit/continuous_odom_imu_yaw_shadow_status"
[[ ! -e "$ENV_FILE" ]] || { echo "environment file already exists" >&2; exit 66; }
umask 077
{
  printf 'export REPO_ROOT=%q\n' "$REPO_ROOT"; printf 'export SHADOW_RUN_ID=%q\n' "$RUN_ID"; printf 'export SHADOW_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"; printf 'export P2KG9U_BAG_OUTPUT_ROOT=%q\n' "$BAG_OUTPUT_ROOT"
  printf 'export SHADOW_RUN_DIR=%q\n' "$RUN_DIR"; printf 'export BAG_DIR=%q\n' "$BAG_DIR"; printf 'export P2KG9U_TMUX_SESSION=%q\n' "$SESSION"; printf 'export P2KG9U_PANE_FILE=%q\n' "$PANE_FILE"
  printf 'export CONTINUOUS_YAW_SHADOW_DIR=%q\n' "$CONTINUOUS_YAW_SHADOW_DIR"; printf 'export CONTINUOUS_YAW_SHADOW_STATUS_TOPIC=%q\n' "$CONTINUOUS_YAW_SHADOW_STATUS_TOPIC"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"; cp "$ENV_FILE" "$OUTPUT_ROOT/p2kg9u_active_env.sh"
tmux new-session -d -s "$SESSION" -n audit "bash '$0' --shadow-pane '$ENV_FILE'"
tmux set-option -t "$SESSION:audit" remain-on-exit on
tmux select-pane -t "$SESSION:audit.0" -T p2kg9u-shadow

state_log() { mkdir -p "$RUN_DIR/logs"; printf '%s state=%s elapsed_tenths=%s detail=%s\n' "$(date -Is)" "$1" "$2" "$3" | tee -a "$RUN_DIR/logs/startup.log"; }
for attempt in $(seq 1 "$START_WAIT_TENTHS"); do
  [[ -d "$RUN_DIR" ]] || { sleep 0.1; continue; }
  missing=""
  for topic in /audit/p2kg9/portal_candidate /audit/p2kg9/portal_status /audit/p2kg9/event_journal /audit/p2kg11/portal_frame; do rostopic type "$topic" >/dev/null 2>&1 || missing="$topic"; done
  [[ -z "$missing" ]] && break
  state_log WAITING_FOR_SHADOW_TOPICS "$attempt" "waiting_for=$missing"
  sleep 0.1
done
[[ -d "$RUN_DIR" ]] || { echo "P2KG9U SHADOW_FAILED: run directory was not created; log=$OUTPUT_ROOT/${RUN_ID}.portal_shadow.startup.log" >&2; exit 71; }
missing=""; for topic in /audit/p2kg9/portal_candidate /audit/p2kg9/portal_status /audit/p2kg9/event_journal /audit/p2kg11/portal_frame; do rostopic type "$topic" >/dev/null 2>&1 || missing="$topic"; done
[[ -z "$missing" ]] || { state_log SHADOW_FAILED "$START_WAIT_TENTHS" "topic_timeout=$missing"; echo "P2KG9U SHADOW_FAILED: $missing; log=$RUN_DIR/logs/portal_shadow.log" >&2; exit 72; }
state_log STARTING_GATE 0 "shadow_topics_ready"
tmux split-window -h -t "$SESSION:audit" "bash '$0' --gate-pane '$ENV_FILE'"
tmux select-pane -t "$SESSION:audit.1" -T p2kg12-room-zone-effect-gate
tmux select-layout -t "$SESSION:audit" even-horizontal
for attempt in $(seq 1 "$START_WAIT_TENTHS"); do
  gate_dead="$(tmux list-panes -t "$SESSION:audit" -F '#{pane_title} #{pane_dead}' | awk '$1 == "p2kg12-room-zone-effect-gate" {print $2; exit}')"
  rostopic type /audit/p2kg12/portal_effect_gate >/dev/null 2>&1 && [[ "$gate_dead" != "1" ]] && break
  state_log STARTING_GATE "$attempt" "waiting_for=/audit/p2kg12/portal_effect_gate"
  sleep 0.1
done
gate_dead="$(tmux list-panes -t "$SESSION:audit" -F '#{pane_title} #{pane_dead}' | awk '$1 == "p2kg12-room-zone-effect-gate" {print $2; exit}')"
rostopic type /audit/p2kg12/portal_effect_gate >/dev/null 2>&1 && [[ "$gate_dead" != "1" ]] || { state_log SHADOW_FAILED "$START_WAIT_TENTHS" "portal_effect_gate_unavailable"; echo "P2KG9U SHADOW_FAILED: portal effect gate; log=$RUN_DIR/logs/portal_room_zone_effect_gate.log" >&2; exit 73; }
state_log STARTING_GEOMETRY 0 "shadow_and_portal_effect_gate_ready"
tmux split-window -h -t "$SESSION:audit" "bash '$0' --geometry-pane '$ENV_FILE'"
tmux select-pane -t "$SESSION:audit.2" -T minimum-zero-progress-geometry
tmux select-layout -t "$SESSION:audit" even-horizontal
for attempt in $(seq 1 "$START_WAIT_TENTHS"); do
  geometry_dead="$(tmux list-panes -t "$SESSION:audit" -F '#{pane_title} #{pane_dead}' | awk '$1 == "minimum-zero-progress-geometry" {print $2; exit}')"
  rostopic type /audit/nearby_model_geometry >/dev/null 2>&1 && [[ "$geometry_dead" != "1" ]] && break
  state_log STARTING_GEOMETRY "$attempt" "waiting_for=/audit/nearby_model_geometry"
  sleep 0.1
done
geometry_dead="$(tmux list-panes -t "$SESSION:audit" -F '#{pane_title} #{pane_dead}' | awk '$1 == "minimum-zero-progress-geometry" {print $2; exit}')"
rostopic type /audit/nearby_model_geometry >/dev/null 2>&1 && [[ "$geometry_dead" != "1" ]] || { state_log SHADOW_FAILED "$START_WAIT_TENTHS" "nearby_model_geometry_unavailable"; echo "P2KG9U SHADOW_FAILED: geometry sidecar; log=$RUN_DIR/logs/nearby_model_geometry.log" >&2; exit 73; }
state_log STARTING_CONTINUOUS_YAW_SHADOW 0 "shadow_portal_effect_gate_and_geometry_ready"
CONTINUOUS_YAW_PANE="$(tmux split-window -h -P -F '#{pane_id}' -t "$SESSION:audit" "bash '$0' --continuous-yaw-shadow-pane '$ENV_FILE'")"
tmux select-pane -t "$CONTINUOUS_YAW_PANE" -T continuous-odom-imu-yaw-shadow
tmux select-layout -t "$SESSION:audit" tiled
for attempt in $(seq 1 "$START_WAIT_TENTHS"); do
  continuous_dead="$(tmux list-panes -t "$SESSION:audit" -F '#{pane_title} #{pane_dead}' | awk '$1 == "continuous-odom-imu-yaw-shadow" {print $2; exit}')"
  [[ -s "$CONTINUOUS_YAW_SHADOW_DIR/ready.json" ]] && rostopic type "$CONTINUOUS_YAW_SHADOW_STATUS_TOPIC" >/dev/null 2>&1 && [[ "$continuous_dead" != "1" ]] && break
  state_log STARTING_CONTINUOUS_YAW_SHADOW "$attempt" "waiting_for=$CONTINUOUS_YAW_SHADOW_STATUS_TOPIC"
  sleep 0.1
done
continuous_dead="$(tmux list-panes -t "$SESSION:audit" -F '#{pane_title} #{pane_dead}' | awk '$1 == "continuous-odom-imu-yaw-shadow" {print $2; exit}')"
[[ -s "$CONTINUOUS_YAW_SHADOW_DIR/ready.json" ]] && rostopic type "$CONTINUOUS_YAW_SHADOW_STATUS_TOPIC" >/dev/null 2>&1 && [[ "$continuous_dead" != "1" ]] || { state_log SHADOW_FAILED "$START_WAIT_TENTHS" "continuous_yaw_shadow_unavailable"; echo "P2KG9U SHADOW_FAILED: continuous yaw shadow; log=$CONTINUOUS_YAW_SHADOW_DIR/logs/continuous_odom_imu_yaw_shadow.log" >&2; exit 73; }
state_log STARTING_ROSBAG 0 "shadow_portal_effect_gate_geometry_and_continuous_yaw_ready"
ROSBAG_PANE="$(tmux split-window -h -P -F '#{pane_id}' -t "$SESSION:audit" "bash '$0' --rosbag-pane '$ENV_FILE'")"
tmux select-pane -t "$ROSBAG_PANE" -T p2kg9u-rosbag
tmux select-layout -t "$SESSION:audit" tiled
for attempt in $(seq 1 "$START_WAIT_TENTHS"); do
  [[ -e "$BAG_DIR/$RUN_ID.bag.active" || -e "$BAG_DIR/$RUN_ID.bag" ]] && break
  state_log STARTING_ROSBAG "$attempt" "waiting_for_bag_file"
  sleep 0.1
done
[[ -e "$BAG_DIR/$RUN_ID.bag.active" || -e "$BAG_DIR/$RUN_ID.bag" ]] || { state_log ROSBAG_FAILED "$START_WAIT_TENTHS" "bag_file_not_created"; echo "P2KG9U ROSBAG_FAILED: log=$RUN_DIR/logs/rosbag.log" >&2; exit 74; }
bag_dead="$(tmux list-panes -t "$SESSION:audit" -F '#{pane_title} #{pane_dead}' | awk '$1 == "p2kg9u-rosbag" {print $2; exit}')"
[[ "$bag_dead" != "1" ]] || { state_log ROSBAG_FAILED 0 "rosbag_pane_exited"; echo "P2KG9U ROSBAG_FAILED: log=$RUN_DIR/logs/rosbag.log" >&2; exit 75; }
tmux list-panes -t "$SESSION:audit" -F '#{pane_id} #{pane_pid} #{pane_title} #{pane_dead} #{pane_dead_status}' > "$PANE_FILE"
state_log READY 0 "shadow_topics_continuous_yaw_rosbag_and_stair_rl_preflight_healthy"
echo "P2KG9U audit bundle READY"
