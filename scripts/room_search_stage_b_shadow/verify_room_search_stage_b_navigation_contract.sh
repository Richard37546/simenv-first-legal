#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/richard/simenv_official_clean"
FLAG="${ROOM_SEARCH_STAGE_B_SHADOW:-}"
FROZEN_FLAG="${ROOM_SEARCH_FROZEN_DECISION_CAPTURE:-}"
WATCH_DIR="${ROOM_SEARCH_STAGE_B_SHADOW_DIR:-}"
RESULT_DIR="${ROOM_SEARCH_STAGE_B_SHADOW_RESULT_DIR:-}"
PID_FILE="${ROOM_SEARCH_STAGE_B_SHADOW_PID_FILE:-${ROOT}/debug/room_search/stage_b_shadow/sidecar.pid}"
READY_FILE="${ROOM_SEARCH_STAGE_B_SHADOW_READY_FILE:-${ROOT}/debug/room_search/stage_b_shadow/sidecar.ready.json}"

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

fail_contract() {
  echo "ROOM_SEARCH_STAGE_B_STARTUP_CONTRACT_FAILED: $*" >&2
  exit 78
}

if ! truthy "${FLAG}"; then
  echo "ROOM_SEARCH Stage B navigation contract SKIPPED: shadow disabled"
  exit 0
fi
if truthy "${FROZEN_FLAG}"; then
  fail_contract "conflicting ROOM_SEARCH_FROZEN_DECISION_CAPTURE is enabled"
fi
[[ -n "${WATCH_DIR}" ]] || fail_contract "ROOM_SEARCH_STAGE_B_SHADOW_DIR is empty"
[[ -n "${RESULT_DIR}" ]] || fail_contract "ROOM_SEARCH_STAGE_B_SHADOW_RESULT_DIR is empty"
[[ -d "${WATCH_DIR}" ]] || fail_contract "capture directory does not exist: ${WATCH_DIR}"
[[ -d "${RESULT_DIR}" ]] || fail_contract "result directory does not exist: ${RESULT_DIR}"
[[ -s "${PID_FILE}" ]] || fail_contract "sidecar pid file is missing or empty: ${PID_FILE}"
[[ -s "${READY_FILE}" ]] || fail_contract "sidecar READY file is missing or empty: ${READY_FILE}"

SIDECAR_PID="$(<"${PID_FILE}")"
[[ "${SIDECAR_PID}" =~ ^[1-9][0-9]*$ ]] || fail_contract "invalid sidecar pid: ${SIDECAR_PID}"
kill -0 "${SIDECAR_PID}" 2>/dev/null || fail_contract "sidecar process is not running: PID ${SIDECAR_PID}"

python3 - "${WATCH_DIR}" "${RESULT_DIR}" "${READY_FILE}" "${SIDECAR_PID}" <<'PY'
import json
from pathlib import Path
import sys


def fail(reason: str) -> None:
    print(f"ROOM_SEARCH_STAGE_B_STARTUP_CONTRACT_FAILED: {reason}", file=sys.stderr)
    raise SystemExit(78)


expected_watch = Path(sys.argv[1]).resolve()
expected_result = Path(sys.argv[2]).resolve()
ready_file = Path(sys.argv[3]).resolve()
expected_pid = int(sys.argv[4])
try:
    ready = json.loads(ready_file.read_text(encoding="utf-8"))
except Exception as exc:
    fail(f"READY JSON unreadable: {type(exc).__name__}:{exc}")

if ready.get("schema_version") != "room_search_stage_b_shadow_result_v1":
    fail(f"unexpected READY schema: {ready.get('schema_version')!r}")
if ready.get("status") != "SIDECAR_READY":
    fail(f"sidecar READY status is {ready.get('status')!r}")
if ready.get("pid") != expected_pid:
    fail(f"READY pid {ready.get('pid')!r} != live pid {expected_pid}")
mode = ready.get("mode", "ONE_SHOT")
if mode not in {"ONE_SHOT", "MULTI_DECISION"}:
    fail(f"unsupported sidecar mode: {mode!r}")
if ready.get("one_shot") is not (mode == "ONE_SHOT"):
    fail("sidecar mode/one_shot contract mismatch")
if ready.get("read_only_grid_status_observer") is not True:
    fail("read-only Grid/status observer is not enabled")
for authority in (
    "production_authority",
    "selection_authority",
    "command_authority",
    "completion_authority",
    "recoverability_authority",
    "fallback_authority",
):
    if ready.get(authority) is not False:
        fail(f"authority isolation violated: {authority}={ready.get(authority)!r}")

ready_watch_raw = ready.get("watch_dir")
ready_result_raw = ready.get("result_dir")
if not isinstance(ready_watch_raw, str) or not ready_watch_raw:
    fail("READY watch_dir is absent")
if not isinstance(ready_result_raw, str) or not ready_result_raw:
    fail("READY result_dir is absent")
ready_watch = Path(ready_watch_raw).resolve()
ready_result = Path(ready_result_raw).resolve()
if ready_watch != expected_watch:
    fail(f"capture dir {expected_watch} != sidecar watch_dir {ready_watch}")
if ready_result != expected_result:
    fail(f"configured result dir {expected_result} != sidecar result_dir {ready_result}")

try:
    command = [
        value.decode("utf-8", errors="replace")
        for value in Path(f"/proc/{expected_pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]
except OSError as exc:
    fail(f"cannot inspect live sidecar command: {type(exc).__name__}:{exc}")
if not any(value.endswith("/room_search_stage_b_shadow_sidecar.py") for value in command):
    fail("live pid is not room_search_stage_b_shadow_sidecar.py")


def option_value(option: str) -> str:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        fail(f"live sidecar command is missing {option}")
        raise AssertionError("unreachable")


if Path(option_value("--watch-dir")).resolve() != expected_watch:
    fail("live sidecar --watch-dir differs from configured capture directory")
if Path(option_value("--result-dir")).resolve() != expected_result:
    fail("live sidecar --result-dir differs from configured result directory")
if Path(option_value("--ready-file")).resolve() != ready_file:
    fail("live sidecar --ready-file differs from configured READY file")
if "--ros-grid-observer" not in command:
    fail("live sidecar command lacks read-only observer flag")
if mode == "ONE_SHOT":
    if "--one-shot" not in command:
        fail("ONE_SHOT sidecar command lacks --one-shot")
else:
    if "--one-shot" in command:
        fail("MULTI_DECISION sidecar command must not contain --one-shot")
    if option_value("--mode") != "MULTI_DECISION":
        fail("MULTI_DECISION sidecar command lacks --mode MULTI_DECISION")

print("ROOM_SEARCH Stage B navigation contract READY")
print(json.dumps({
    "capture_dir": str(expected_watch),
    "result_dir": str(expected_result),
    "ready_file": str(ready_file),
    "sidecar_pid": expected_pid,
    "status": "PATHS_AND_AUTHORITY_MATCH",
}, sort_keys=True))
PY
