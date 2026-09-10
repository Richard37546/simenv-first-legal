#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$REPO_ROOT/debug/odom_accuracy_audit_v1/p2kg9_current_door_portal_shadow_048/online_run/p2kg9u_active_env.sh"
[[ -f "$ENV_FILE" ]] || { echo "P2KG9U NOT_PREPARED"; exit 65; }
source "$ENV_FILE"
[[ -d "$SHADOW_RUN_DIR" && -s "$P2KG9U_PANE_FILE" ]] || { echo "P2KG9U STARTING_SHADOW"; exit 66; }
state="$(tail -1 "$SHADOW_RUN_DIR/logs/startup.log" 2>/dev/null | sed -n 's/.*state=\([^ ]*\).*/\1/p')"
[[ "$state" == "READY" ]] || { echo "P2KG9U ${state:-PARTIAL_START}"; exit 67; }
tmux has-session -t "$P2KG9U_TMUX_SESSION" 2>/dev/null || { echo "P2KG9U PARTIAL_START: tmux session missing"; exit 68; }
pane_dead() { tmux list-panes -t "$P2KG9U_TMUX_SESSION" -F '#{pane_title} #{pane_dead}' 2>/dev/null | awk -v title="$1" '$1 == title {print $2; exit}'; }
[[ "$(pane_dead p2kg9u-shadow)" != "1" ]] || { echo "P2KG9U SHADOW_FAILED: see $SHADOW_RUN_DIR/logs/portal_shadow.log"; exit 68; }
[[ "$(pane_dead p2kg12-room-zone-effect-gate)" != "1" ]] || { echo "P2KG9U SHADOW_FAILED: see $SHADOW_RUN_DIR/logs/portal_room_zone_effect_gate.log"; exit 68; }
continuous_yaw_dead="$(pane_dead continuous-odom-imu-yaw-shadow)"
[[ -n "$continuous_yaw_dead" && "$continuous_yaw_dead" != "1" ]] || { echo "P2KG9U SHADOW_FAILED: see $CONTINUOUS_YAW_SHADOW_DIR/logs/continuous_odom_imu_yaw_shadow.log"; exit 68; }
[[ "$(pane_dead p2kg9u-rosbag)" != "1" ]] || { echo "P2KG9U ROSBAG_FAILED: see $SHADOW_RUN_DIR/logs/rosbag.log"; exit 68; }
# Angular-actuation telemetry is optional diagnostics.  It is intentionally
# not a readiness condition because the official Stair baseline disables it.
for topic in /audit/p2kg9/portal_candidate /audit/p2kg9/portal_status /audit/p2kg9/event_journal /audit/p2kg11/portal_frame /audit/p2kg12/portal_effect_gate /audit/continuous_odom_imu_yaw_shadow_status /audit/p2kg15/selected_rl_policy /unitree/rl_mode_ready; do rostopic type "$topic" >/dev/null 2>&1 || { echo "P2KG9U PARTIAL_START: missing $topic"; exit 69; }; done
[[ -s "$CONTINUOUS_YAW_SHADOW_DIR/ready.json" ]] || { echo "P2KG9U PARTIAL_START: continuous yaw shadow READY file missing"; exit 69; }
policy_record="$(timeout 5 rostopic echo -n 1 /audit/p2kg15/selected_rl_policy 2>/dev/null || true)"
[[ "$policy_record" == *"stair"* && "$policy_record" == *"src/unitree_guide/logs/policy_act_inference_stair.pt"* && "$policy_record" == *"2d5aa72511c0c6609c02f4105845eee6974d3d73431497f8f35306da9588fe14"* ]] || { echo "P2KG15_STAIR_PREFLIGHT_FAILED: selected policy record is not the required stair asset/hash"; exit 69; }
rl_ready_record="$(timeout 5 rostopic echo -n 1 /unitree/rl_mode_ready 2>/dev/null || true)"
[[ "${rl_ready_record,,}" == *"data: true"* ]] || { echo "P2KG15_STAIR_PREFLIGHT_FAILED: /unitree/rl_mode_ready is not true"; exit 69; }
[[ -e "$BAG_DIR/$SHADOW_RUN_ID.bag.active" || -e "$BAG_DIR/$SHADOW_RUN_ID.bag" ]] || { echo "P2KG9U ROSBAG_FAILED: bag file missing"; exit 70; }
echo "P2KG9U READY"
