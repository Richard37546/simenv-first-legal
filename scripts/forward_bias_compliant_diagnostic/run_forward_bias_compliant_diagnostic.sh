#!/usr/bin/env bash
set -eo pipefail

cd /home/richard/simenv_official_clean
source /opt/ros/noetic/setup.bash
source devel/setup.bash 2>/dev/null || true
set -u

python3 scripts/forward_bias_compliant_diagnostic/forward_bias_compliant_diagnostic.py "$@"
