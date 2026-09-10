#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/richard/simenv_official_clean"
PID_FILE="${ROOM_SEARCH_STAGE_B_SHADOW_PID_FILE:-${ROOT}/debug/room_search/stage_b_shadow/sidecar.pid}"
READY_FILE="${ROOM_SEARCH_STAGE_B_SHADOW_READY_FILE:-${ROOT}/debug/room_search/stage_b_shadow/sidecar.ready.json}"
if [[ ! -f "${PID_FILE}" ]]; then
  rm -f -- "${READY_FILE}"
  echo "ROOM_SEARCH Stage B sidecar already stopped"
  exit 0
fi
SIDECAR_PID="$(<"${PID_FILE}")"
if kill -0 "${SIDECAR_PID}" 2>/dev/null; then
  kill -TERM "${SIDECAR_PID}"
  for _ in $(seq 1 40); do
    kill -0 "${SIDECAR_PID}" 2>/dev/null || break
    sleep 0.05
  done
  if kill -0 "${SIDECAR_PID}" 2>/dev/null; then
    kill -KILL "${SIDECAR_PID}"
  fi
fi
rm -f -- "${PID_FILE}" "${READY_FILE}"
echo "ROOM_SEARCH Stage B sidecar/evaluator stopped; navigation runtime was not touched"
