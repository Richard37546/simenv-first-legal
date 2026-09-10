#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/richard/simenv_official_clean}"
ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-noetic}"
BUILD_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  scripts/bootstrap_build_workspace.sh [--clean] [--catkin-arg ARG ...]

Rebuild the catkin workspace from source in a reproducible local environment.

Options:
  --clean             Remove build/, devel/, install/, and log/ before building.
  --catkin-arg ARG    Pass one argument through to catkin_make.

Environment:
  ROOT                Workspace root. Default: /home/richard/simenv_official_clean
  ROS_DISTRO_NAME     ROS distro. Default: noetic
USAGE
}

CLEAN=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --clean)
      CLEAN=true
      shift
      ;;
    --catkin-arg)
      if [ "$#" -lt 2 ]; then
        echo "--catkin-arg requires a value" >&2
        exit 2
      fi
      BUILD_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"

if [ ! -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]; then
  echo "missing ROS setup: /opt/ros/${ROS_DISTRO_NAME}/setup.bash" >&2
  exit 1
fi

source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

if [ "$CLEAN" = true ]; then
  rm -rf "$ROOT/build" "$ROOT/devel" "$ROOT/install" "$ROOT/log"
fi

catkin_make -DCATKIN_WHITELIST_PACKAGES= "${BUILD_ARGS[@]}"

if [ -f "$ROOT/devel/setup.bash" ]; then
  echo "bootstrap complete: source $ROOT/devel/setup.bash"
else
  echo "catkin_make completed but devel/setup.bash was not generated" >&2
  exit 1
fi
