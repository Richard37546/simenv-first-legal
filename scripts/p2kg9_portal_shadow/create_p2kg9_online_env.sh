#!/usr/bin/env bash
# Creates a sourceable, audit-only run environment. It never starts ROS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_OUTPUT_ROOT="$REPO_ROOT/debug/odom_accuracy_audit_v1/p2kg9_current_door_portal_shadow_048/online_run"
DEFAULT_BAG_OUTPUT_ROOT="/mnt/d/data/simenv_audit_bags/p2kg9_current_door_portal_shadow_048/online_run"
OUTPUT_ROOT="${P2KG9U_OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
BAG_OUTPUT_ROOT="${P2KG9U_BAG_OUTPUT_ROOT:-$DEFAULT_BAG_OUTPUT_ROOT}"
OUTPUT_ROOT_EXPLICIT=false
BAG_OUTPUT_ROOT_EXPLICIT=false
RUN_ID=""
ENV_FILE=""

usage() {
  printf '%s\n' "usage: $0 --run-id RUN_ID [--output-root DIRECTORY] [--bag-output-root DIRECTORY] [--env-file FILE]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:?missing run id}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing output root}"; OUTPUT_ROOT_EXPLICIT=true; shift 2 ;;
    --bag-output-root) BAG_OUTPUT_ROOT="${2:?missing bag output root}"; BAG_OUTPUT_ROOT_EXPLICIT=true; shift 2 ;;
    --env-file) ENV_FILE="${2:?missing environment file}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 64 ;;
  esac
done

if [[ "$OUTPUT_ROOT_EXPLICIT" == true && "$BAG_OUTPUT_ROOT_EXPLICIT" == false && -z "${P2KG9U_BAG_OUTPUT_ROOT:-}" ]]; then
  BAG_OUTPUT_ROOT="$OUTPUT_ROOT"
fi

[[ -n "$RUN_ID" ]] || { printf '%s\n' "--run-id is required" >&2; exit 64; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf '%s\n' "invalid run id" >&2; exit 64; }
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
BAG_OUTPUT_ROOT="$(realpath -m "$BAG_OUTPUT_ROOT")"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
BAG_DIR="$BAG_OUTPUT_ROOT/$RUN_ID/bag"
[[ -n "$ENV_FILE" ]] || ENV_FILE="$OUTPUT_ROOT/p2kg9_online_env.sh"
ENV_FILE="$(realpath -m "$ENV_FILE")"
[[ ! -e "$RUN_DIR" ]] || { printf '%s\n' "run output already exists: $RUN_DIR" >&2; exit 65; }
[[ ! -e "$ENV_FILE" ]] || { printf '%s\n' "environment file already exists: $ENV_FILE" >&2; exit 65; }
mkdir -p "$(dirname "$ENV_FILE")"
umask 077
{
  printf 'export REPO_ROOT=%q\n' "$REPO_ROOT"
  printf 'export SHADOW_RUN_ID=%q\n' "$RUN_ID"
  printf 'export SHADOW_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
  printf 'export SHADOW_RUN_DIR=%q\n' "$RUN_DIR"
  printf 'export P2KG9U_BAG_OUTPUT_ROOT=%q\n' "$BAG_OUTPUT_ROOT"
  printf 'export BAG_DIR=%q\n' "$BAG_DIR"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"
printf '%s\n' "Created audit environment file: $ENV_FILE"
printf '%s\n' "Source this exact path in every terminal before continuing."
