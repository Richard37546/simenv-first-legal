#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/richard/simenv_official_clean"
PID_FILE="${ROOM_SEARCH_STAGE_B_SHADOW_PID_FILE:-${ROOT}/debug/room_search/stage_b_shadow/sidecar.pid}"
READY_FILE="${ROOM_SEARCH_STAGE_B_SHADOW_READY_FILE:-${ROOT}/debug/room_search/stage_b_shadow/sidecar.ready.json}"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "ROOM_SEARCH Stage B sidecar NOT RUNNING (pid file absent)"
  exit 1
fi
SIDECAR_PID="$(<"${PID_FILE}")"
if kill -0 "${SIDECAR_PID}" 2>/dev/null; then
  echo "ROOM_SEARCH Stage B sidecar RUNNING: PID ${SIDECAR_PID}"
  if [[ -s "${READY_FILE}" ]]; then
    echo "ROOM_SEARCH Stage B audit state: READY"
    cat "${READY_FILE}"
  else
    echo "ROOM_SEARCH Stage B audit state: STARTING_OR_INCOMPLETE"
  fi
  exit 0
fi
echo "ROOM_SEARCH Stage B sidecar NOT RUNNING (stale PID ${SIDECAR_PID})"
exit 1
