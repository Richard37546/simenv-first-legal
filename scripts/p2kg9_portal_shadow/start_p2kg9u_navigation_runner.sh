#!/usr/bin/env bash
# User-invoked wrapper for the audited Portal-entry validation.  The explicit
# flag is required: without it the state machine stays on the corridor target
# even when the Portal effect gate has committed a candidate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$REPO_ROOT/scripts/p2kg9_portal_shadow/status_p2kg9u_audit_bundle.sh"
# READY above confirms the audit bundle and its topic contracts.  Navigation
# itself also requires an actual gated-odom message; wait for it before target
# preparation so an early runtime sample cannot turn into a zero-iteration
# state-machine failure.
python3 "$REPO_ROOT/scripts/p2kg9_portal_shadow/wait_for_navigation_odom_ready.py" \
  --timeout-sec "${P2KG9U_NAVIGATION_ODOM_WAIT_SEC:-45}"
# Stage B is behavior-neutral, but its capture writer and external sidecar must
# agree on one exact run root.  Fail before starting production navigation if a
# stale shell environment would route the snapshot outside the live sidecar's
# watch directory.
bash "$REPO_ROOT/scripts/room_search_stage_b_shadow/verify_room_search_stage_b_navigation_contract.sh"
ROOM_SEARCH_ARGS=()
if [[ "${ROOM_SEARCH_V2:-false}" == "true" ]]; then
  ROOM_SEARCH_ARGS+=(--enable-room-search-v2)
fi
P_PRE_ARGS=()
if [[ -n "${P_PRE_UPSTREAM_TANGENT_M:-}" ]]; then
  P_PRE_ARGS+=(--p-pre-upstream-tangent-m "$P_PRE_UPSTREAM_TANGENT_M")
fi
ROOM_LOCAL_VALIDATION_ARGS=()
if [[ "${ROOM_LOCAL_GUARDED_ONLINE_VALIDATION:-false}" == "true" ]]; then
  if [[ -z "${ROOM_LOCAL_VALIDATION_RUN_ID:-}" ]]; then
    export ROOM_LOCAL_VALIDATION_RUN_ID="room_local_validation_$(date +%Y%m%d_%H%M%S)"
  fi
  ROOM_LOCAL_VALIDATION_ARGS+=(--enable-room-local-guarded-online-validation)
fi
exec bash "$REPO_ROOT/scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh" \
  --execute \
  --enable-hierarchical-portal-local-autonomy \
  "${ROOM_SEARCH_ARGS[@]}" \
  "${P_PRE_ARGS[@]}" \
  "${ROOM_LOCAL_VALIDATION_ARGS[@]}"
