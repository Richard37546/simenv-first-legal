#!/usr/bin/env bash
# Internal worker for the P2K-G9U audit bundle. It never starts H1 or navigation.
set -euo pipefail

MODE="${1:-}"
ENV_FILE="${2:-}"
[[ "$MODE" == "shadow" || "$MODE" == "bag" ]] || { echo "usage: $0 {shadow|bag} ENV_FILE" >&2; exit 64; }
[[ -f "$ENV_FILE" ]] || { echo "environment file missing: $ENV_FILE" >&2; exit 65; }
# shellcheck disable=SC1090
source "$ENV_FILE"
source /opt/ros/noetic/setup.bash
source "$REPO_ROOT/devel/setup.bash" 2>/dev/null || true

if [[ "$MODE" == "shadow" ]]; then
  exec bash "$REPO_ROOT/scripts/p2kg9_portal_shadow/start_passive_portal_shadow_audit.sh" \
    --run-id "$SHADOW_RUN_ID" --output-root "$SHADOW_OUTPUT_ROOT"
fi

# The Shadow creates SHADOW_RUN_DIR. Do not create it here because the isolated
# Shadow launcher deliberately rejects pre-existing run directories.
for _ in $(seq 1 300); do
  [[ -d "$SHADOW_RUN_DIR" ]] && break
  sleep 0.1
done
[[ -d "$SHADOW_RUN_DIR" ]] || { echo "Shadow output directory was not created" >&2; exit 70; }

for topic in /audit/p2kg9/portal_candidate /audit/p2kg9/portal_status /audit/p2kg9/event_journal; do
  rostopic type "$topic" >/dev/null 2>&1 || { echo "required Shadow topic unavailable: $topic" >&2; exit 71; }
done

mkdir "$BAG_DIR"
exec rosbag record -O "$BAG_DIR/$SHADOW_RUN_ID.bag" \
  /clock \
  /gazebo/model_states \
  /team/livox/scan_cloud_filtered \
  /team/livox/icp_odom_raw \
  /team/livox/icp_odom_gated \
  /team/local_traversability_grid \
  /team/traversability_status \
  /tf \
  /tf_static \
  /cmd_vel \
  /cmd_vel_raw \
  /audit/stair_cmd_receipt \
  /audit/stair_policy_cmd_input \
  /imu_velocity_follower/status \
  /trunk_imu \
  /team/doorway_candidate \
  /team/danger_tracks \
  /team/danger_hypotheses \
  /audit/p2kg9/portal_candidate \
  /audit/p2kg9/portal_status \
  /audit/p2kg9/event_journal \
  /audit/p2kg7r/ray_evidence_status \
  /audit/p2kg15/selected_rl_policy \
  /audit/p2kg15/angular_actuation_state \
  /audit/nearby_model_geometry \
  /unitree/rl_mode_ready
