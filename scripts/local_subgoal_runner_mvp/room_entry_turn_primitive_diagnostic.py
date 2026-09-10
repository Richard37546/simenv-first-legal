#!/usr/bin/env python3
"""Standalone room-entry turn primitive diagnostic.

This script does not call the state machine or local runner. It only publishes
one fixed Twist primitive and measures the odometry response.
"""

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
OUT_DIR = ROOT / "debug" / "room_entry_turn_primitive_diagnostic"
ODOM_TOPIC = "/team/livox/icp_odom_gated"

MODES: Dict[str, Dict[str, Any]] = {
    "raw_inplace_turn": {"cmd_topic": "/cmd_vel_raw", "linear_x": 0.0, "angular_z": -0.45},
    "direct_inplace_turn": {"cmd_topic": "/cmd_vel", "linear_x": 0.0, "angular_z": -0.45},
    "raw_arc_turn": {"cmd_topic": "/cmd_vel_raw", "linear_x": 0.15, "angular_z": -0.45},
    "direct_arc_turn": {"cmd_topic": "/cmd_vel", "linear_x": 0.15, "angular_z": -0.45},
}


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_from_odom(msg: Odometry) -> Dict[str, float]:
    pose = msg.pose.pose
    return {
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "yaw": yaw_from_quat(pose.orientation),
    }


def read_pose(timeout_sec: float) -> Dict[str, float]:
    msg = rospy.wait_for_message(ODOM_TOPIC, Odometry, timeout=timeout_sec)
    return pose_from_odom(msg)


def make_twist(linear_x: float, angular_z: float) -> Twist:
    cmd = Twist()
    cmd.linear.x = float(linear_x)
    cmd.angular.z = float(angular_z)
    return cmd


def publish_zero(pub: rospy.Publisher, rate_hz: float, duration_sec: float) -> int:
    rate = rospy.Rate(rate_hz)
    zero = Twist()
    start_sim = rospy.Time.now()
    count = 0
    while not rospy.is_shutdown():
        elapsed = float((rospy.Time.now() - start_sim).to_sec())
        if elapsed >= float(duration_sec):
            break
        pub.publish(zero)
        count += 1
        rate.sleep()
    pub.publish(zero)
    return count + 1


def run_mode(mode: str, duration_sec: float, rate_hz: float, odom_timeout_sec: float) -> Dict[str, Any]:
    spec = MODES[mode]
    cmd_topic = str(spec["cmd_topic"])
    linear_x = float(spec["linear_x"])
    angular_z = float(spec["angular_z"])

    pub = rospy.Publisher(cmd_topic, Twist, queue_size=2)
    rospy.sleep(0.2)

    start_pose = read_pose(odom_timeout_sec)
    cmd = make_twist(linear_x, angular_z)
    rate = rospy.Rate(rate_hz)
    start_wall = time.monotonic()
    start_sim = rospy.Time.now()
    publish_count = 0
    sim_duration = 0.0

    while not rospy.is_shutdown():
        sim_duration = max(0.0, float((rospy.Time.now() - start_sim).to_sec()))
        if sim_duration >= float(duration_sec):
            break
        pub.publish(cmd)
        publish_count += 1
        rate.sleep()

    wall_duration = time.monotonic() - start_wall
    zero_publish_count = publish_zero(pub, rate_hz, 1.0)
    end_pose = read_pose(odom_timeout_sec)

    yaw_delta = normalize_angle(float(end_pose["yaw"]) - float(start_pose["yaw"]))
    translation_delta = math.hypot(float(end_pose["x"]) - float(start_pose["x"]), float(end_pose["y"]) - float(start_pose["y"]))

    return {
        "mode": mode,
        "cmd_topic": cmd_topic,
        "odom_topic": ODOM_TOPIC,
        "linear_x": linear_x,
        "angular_z": angular_z,
        "sim_duration_sec": sim_duration,
        "wall_duration_sec": wall_duration,
        "publish_count": publish_count,
        "zero_publish_count": zero_publish_count,
        "start_pose": start_pose,
        "end_pose": end_pose,
        "yaw_delta_rad": yaw_delta,
        "yaw_delta_deg": math.degrees(yaw_delta),
        "translation_delta_m": translation_delta,
        "average_publish_rate_sim_hz": float(publish_count) / sim_duration if sim_duration > 1e-6 else None,
        "average_publish_rate_wall_hz": float(publish_count) / wall_duration if wall_duration > 1e-6 else None,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=sorted(MODES.keys()))
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--odom-timeout-sec", type=float, default=5.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rospy.init_node("room_entry_turn_primitive_diagnostic", anonymous=True, disable_signals=True)
    result = run_mode(args.mode, args.duration_sec, args.rate_hz, args.odom_timeout_sec)
    out_path = OUT_DIR / f"latest_result_{args.mode}.json"
    write_json(out_path, result)
    print(
        "mode={mode}, yaw_delta_deg={yaw_delta_deg:.3f}, translation_delta_m={translation_delta_m:.3f}, "
        "sim_duration_sec={sim_duration_sec:.3f}, wall_duration_sec={wall_duration_sec:.3f}, publish_count={publish_count}".format(
            **result
        )
    )
    print(f"result_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
