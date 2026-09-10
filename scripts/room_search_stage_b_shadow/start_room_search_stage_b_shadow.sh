#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/richard/simenv_official_clean"
FLAG="${ROOM_SEARCH_STAGE_B_SHADOW:-}"
MODE="${ROOM_SEARCH_STAGE_B_MODE:-ONE_SHOT}"
FROZEN_FLAG="${ROOM_SEARCH_FROZEN_DECISION_CAPTURE:-}"
WATCH_DIR="${ROOM_SEARCH_STAGE_B_SHADOW_DIR:-}"
RESULT_DIR="${ROOM_SEARCH_STAGE_B_SHADOW_RESULT_DIR:-}"
PID_FILE="${ROOM_SEARCH_STAGE_B_SHADOW_PID_FILE:-${ROOT}/debug/room_search/stage_b_shadow/sidecar.pid}"
LOG_FILE="${ROOM_SEARCH_STAGE_B_SHADOW_LOG_FILE:-${ROOT}/debug/room_search/stage_b_shadow/sidecar.log}"
READY_FILE="${ROOM_SEARCH_STAGE_B_SHADOW_READY_FILE:-${ROOT}/debug/room_search/stage_b_shadow/sidecar.ready.json}"
POLICY_PATH="${ROOT}/src/unitree_guide/logs/policy_act_inference_stair.pt"
POLICY_SHA256="2d5aa72511c0c6609c02f4105845eee6974d3d73431497f8f35306da9588fe14"
REQUIRED_SOURCES=(
  "${ROOT}/scripts/local_subgoal_runner_mvp/navigation_state_machine.py"
  "${ROOT}/scripts/local_subgoal_runner_mvp/room_search_stage_b_shadow_capture.py"
  "${ROOT}/scripts/local_subgoal_runner_mvp/room_search_stage_b_shadow_sidecar.py"
  "${ROOT}/scripts/local_subgoal_runner_mvp/room_search_stage_a_contract.py"
)

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if ! truthy "${FLAG}"; then
  echo "ROOM_SEARCH Stage B shadow is disabled; set ROOM_SEARCH_STAGE_B_SHADOW=true"
  exit 2
fi
case "${MODE}" in
  ONE_SHOT|MULTI_DECISION) ;;
  *) echo "ROOM_SEARCH_STAGE_B_MODE must be ONE_SHOT or MULTI_DECISION"; exit 2 ;;
esac
if truthy "${FROZEN_FLAG}"; then
  echo "ROOM_SEARCH Stage B shadow conflicts with ROOM_SEARCH_FROZEN_DECISION_CAPTURE"
  exit 2
fi
if [[ -z "${WATCH_DIR}" || -z "${RESULT_DIR}" ]]; then
  echo "ROOM_SEARCH_STAGE_B_SHADOW_DIR and ROOM_SEARCH_STAGE_B_SHADOW_RESULT_DIR are required"
  exit 2
fi
for source_path in "${REQUIRED_SOURCES[@]}"; do
  if [[ ! -f "${source_path}" ]]; then
    echo "Stage B source identity incomplete: ${source_path}"
    exit 2
  fi
done
if [[ "${UNITREE_RL_POLICY:-stair}" != "stair" ]]; then
  echo "Stage B online review is stair-only"
  exit 2
fi
if [[ ! -f "${POLICY_PATH}" ]]; then
  echo "Frozen stair policy missing: ${POLICY_PATH}"
  exit 2
fi
ACTUAL_POLICY_SHA256="$(sha256sum "${POLICY_PATH}" | awk '{print $1}')"
if [[ "${ACTUAL_POLICY_SHA256}" != "${POLICY_SHA256}" ]]; then
  echo "Frozen stair policy SHA256 mismatch: ${ACTUAL_POLICY_SHA256}"
  exit 2
fi

mkdir -p "${WATCH_DIR}" "${RESULT_DIR}" "$(dirname "${PID_FILE}")" "$(dirname "${LOG_FILE}")"
if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
  echo "ROOM_SEARCH Stage B sidecar already running: PID $(<"${PID_FILE}")"
  exit 2
fi
rm -f -- "${READY_FILE}"

SIDECAR_MODE_ARGS=()
if [[ "${MODE}" == "ONE_SHOT" ]]; then
  SIDECAR_MODE_ARGS+=(--one-shot)
else
  SIDECAR_MODE_ARGS+=(--mode MULTI_DECISION)
fi

nohup python3 "${ROOT}/scripts/local_subgoal_runner_mvp/room_search_stage_b_shadow_sidecar.py" \
  --watch-dir "${WATCH_DIR}" \
  --result-dir "${RESULT_DIR}" \
  --timeout-sec "${ROOM_SEARCH_STAGE_B_EVALUATOR_TIMEOUT_SEC:-30}" \
  --ros-grid-observer \
  --ready-file "${READY_FILE}" \
  "${SIDECAR_MODE_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
SIDECAR_PID=$!
printf '%s\n' "${SIDECAR_PID}" >"${PID_FILE}"
for _ in $(seq 1 100); do
  if [[ -s "${READY_FILE}" ]]; then
    echo "ROOM_SEARCH Stage B ${MODE} sidecar READY: PID ${SIDECAR_PID}"
    exit 0
  fi
  if ! kill -0 "${SIDECAR_PID}" 2>/dev/null; then
    echo "ROOM_SEARCH Stage B sidecar exited before READY"
    tail -n 40 "${LOG_FILE}" 2>/dev/null || true
    rm -f -- "${PID_FILE}"
    exit 2
  fi
  sleep 0.05
done
kill -TERM "${SIDECAR_PID}" 2>/dev/null || true
rm -f -- "${PID_FILE}"
echo "ROOM_SEARCH Stage B sidecar READY timeout"
exit 2
