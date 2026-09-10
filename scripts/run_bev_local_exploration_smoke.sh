#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/richard/simenv_official_clean
BEV_DIR=/home/richard/.ros/results/bev_maps

source /opt/ros/noetic/setup.bash
if [ -f "$ROOT/devel/setup.bash" ]; then
  source "$ROOT/devel/setup.bash"
fi

echo "[smoke] check package"
rospack find bev_perception >/dev/null

echo "[smoke] check topics"
test "$(timeout 5s rostopic type /bev/occupancy_grid || true)" = "nav_msgs/OccupancyGrid"
test "$(timeout 5s rostopic type /team/livox/icp_odom_gated || true)" = "nav_msgs/Odometry"

echo "[smoke] check latest accgrid semantics"
python3 - <<'PY'
import glob
import os
import sys
import numpy as np
files = sorted(glob.glob('/home/richard/.ros/results/bev_maps/*_accgrid.npy'), key=os.path.getmtime)
if not files:
    raise SystemExit('no *_accgrid.npy under /home/richard/.ros/results/bev_maps')
path = files[-1]
a = np.load(path)
values = sorted(np.unique(a).tolist())
print('latest_accgrid', os.path.basename(path), values, a.shape, a.dtype)
if any(v not in (0, 1, 2) for v in values):
    raise SystemExit('accgrid contains values outside 0/1/2')
PY

echo "[smoke] run L3ZE"
python3 "$ROOT/scripts/l3ze_adapter_update/l3ze_adapter_update.py" --input-dir "$BEV_DIR"
echo "[smoke] run L3ZF"
python3 "$ROOT/scripts/l3zf_multiframe_replay/l3zf_multiframe_replay.py"
echo "[smoke] run L3ZM"
python3 "$ROOT/scripts/l3zm_end_to_end_shadow_replay/l3zm_end_to_end_shadow_replay.py"
echo "[smoke] run L3ZN"
python3 "$ROOT/scripts/l3zn_navigation_handoff_freeze/l3zn_navigation_handoff_freeze.py"

echo "SLAM_BEV_MINIMAL_INTEGRATION_READY"
