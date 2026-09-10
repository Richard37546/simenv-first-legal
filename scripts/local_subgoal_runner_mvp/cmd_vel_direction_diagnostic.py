#!/usr/bin/env python3
"""Compliant cmd_vel direction diagnostic using /team/livox/icp_odom_gated only."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "local_subgoal_runner_mvp" / "cmd_vel_direction_diagnostic_summary.json"
ODOM_TOPIC = "/team/livox/icp_odom_gated"
CMD_TOPIC = "/cmd_vel"


def yaw_from_quat(q: Any) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def read_pose(timeout_sec: float) -> Tuple[float, float, float, float]:
    msg = rospy.wait_for_message(ODOM_TOPIC, Odometry, timeout=timeout_sec)
    pose = msg.pose.pose
    return (
        float(msg.header.stamp.to_sec()),
        float(pose.position.x),
        float(pose.position.y),
        yaw_from_quat(pose.orientation),
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def direction_label(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "+x" if dx >= 0.0 else "-x"
    return "+y" if dy >= 0.0 else "-y"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="publish a short /cmd_vel pulse")
    parser.add_argument("--linear-x", type=float, default=0.15)
    parser.add_argument("--angular-z", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=0.8)
    parser.add_argument("--cmd-rate-hz", type=float, default=20.0)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    args = parser.parse_args()

    rospy.init_node("cmd_vel_direction_diagnostic", anonymous=True, disable_signals=True)
    pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
    time.sleep(0.3)

    before = read_pose(args.timeout_sec)
    nonzero_count = 0
    if args.execute:
        cmd = Twist()
        cmd.linear.x = float(args.linear_x)
        cmd.angular.z = float(args.angular_z)
        start_sim = float(rospy.Time.now().to_sec())
        rate = rospy.Rate(args.cmd_rate_hz)
        while not rospy.is_shutdown() and float(rospy.Time.now().to_sec()) - start_sim < args.duration_sec:
            pub.publish(cmd)
            nonzero_count += 1
            rate.sleep()
    zero = Twist()
    for _ in range(10):
        pub.publish(zero)
        time.sleep(0.03)
    after = read_pose(args.timeout_sec)

    dx = after[1] - before[1]
    dy = after[2] - before[2]
    dyaw = math.atan2(math.sin(after[3] - before[3]), math.cos(after[3] - before[3]))
    displacement = math.hypot(dx, dy)
    summary = {
        "execute": bool(args.execute),
        "odom_topic": ODOM_TOPIC,
        "cmd_topic": CMD_TOPIC,
        "forbidden_sources_used": [],
        "used_gazebo_truth": False,
        "called_move_base": False,
        "sent_navigation_goal": False,
        "commanded_linear_x": float(args.linear_x) if args.execute else 0.0,
        "commanded_angular_z": float(args.angular_z) if args.execute else 0.0,
        "duration_sec_sim_time": float(args.duration_sec),
        "published_nonzero_count": nonzero_count,
        "before_stamp_x_y_yaw": list(before),
        "after_stamp_x_y_yaw": list(after),
        "sim_duration_observed_sec": after[0] - before[0],
        "dx_odom_m": dx,
        "dy_odom_m": dy,
        "displacement_m": displacement,
        "dyaw_rad": dyaw,
        "dyaw_deg": math.degrees(dyaw),
        "dominant_odom_direction": direction_label(dx, dy) if displacement > 1e-6 else "none",
    }
    write_json(OUT, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
