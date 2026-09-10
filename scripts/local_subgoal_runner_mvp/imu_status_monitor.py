#!/usr/bin/env python3
"""Terminal IMU status monitor for manual robot runs.

This script is read-only. It subscribes to a sensor_msgs/Imu topic and prints a
low-rate status line suitable for watching yaw drift while the robot moves.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import rospy
from sensor_msgs.msg import Imu


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "debug" / "imu_status_monitor"


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def euler_from_quat(q: Any) -> tuple[float, float, float]:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class ImuMonitor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.latest: Optional[Imu] = None
        self.message_count = 0
        self.start_wall = time.monotonic()
        self.start_ros_sec: Optional[float] = None
        self.start_yaw: Optional[float] = None
        self.last_print = 0.0
        self.max_abs_yaw_delta = 0.0
        self.max_abs_angular_z = 0.0

    def cb(self, msg: Imu) -> None:
        self.latest = msg
        self.message_count += 1

    def snapshot(self) -> Optional[Dict[str, Any]]:
        if self.latest is None:
            return None
        msg = self.latest
        stamp_sec = float(msg.header.stamp.to_sec()) if msg.header.stamp else float(rospy.Time.now().to_sec())
        roll, pitch, yaw = euler_from_quat(msg.orientation)
        if self.start_yaw is None:
            self.start_yaw = yaw
            self.start_ros_sec = stamp_sec
        yaw_delta = normalize_angle(yaw - float(self.start_yaw))
        self.max_abs_yaw_delta = max(self.max_abs_yaw_delta, abs(yaw_delta))
        self.max_abs_angular_z = max(self.max_abs_angular_z, abs(float(msg.angular_velocity.z)))
        return {
            "topic": self.args.topic,
            "frame_id": msg.header.frame_id,
            "message_count": self.message_count,
            "stamp_sec": stamp_sec,
            "elapsed_wall_sec": time.monotonic() - self.start_wall,
            "elapsed_ros_sec": stamp_sec - self.start_ros_sec if self.start_ros_sec is not None else None,
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "yaw_delta_from_start_rad": yaw_delta,
            "yaw_delta_from_start_deg": math.degrees(yaw_delta),
            "angular_velocity_x": float(msg.angular_velocity.x),
            "angular_velocity_y": float(msg.angular_velocity.y),
            "angular_velocity_z": float(msg.angular_velocity.z),
            "linear_acceleration_x": float(msg.linear_acceleration.x),
            "linear_acceleration_y": float(msg.linear_acceleration.y),
            "linear_acceleration_z": float(msg.linear_acceleration.z),
            "max_abs_yaw_delta_deg": math.degrees(self.max_abs_yaw_delta),
            "max_abs_angular_velocity_z": self.max_abs_angular_z,
            "published_cmd_vel": False,
            "forbidden_sources_used": [],
        }

    def print_status(self, snap: Dict[str, Any]) -> None:
        print(
            "imu "
            f"topic={snap['topic']} "
            f"count={snap['message_count']} "
            f"yaw={snap['yaw_deg']:+7.2f}deg "
            f"dyaw={snap['yaw_delta_from_start_deg']:+7.2f}deg "
            f"roll={snap['roll_deg']:+6.2f}deg "
            f"pitch={snap['pitch_deg']:+6.2f}deg "
            f"wz={snap['angular_velocity_z']:+7.3f}rad/s "
            f"acc=({snap['linear_acceleration_x']:+6.2f},"
            f"{snap['linear_acceleration_y']:+6.2f},"
            f"{snap['linear_acceleration_z']:+6.2f}) "
            f"frame={snap['frame_id']}",
            flush=True,
        )


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/trunk_imu")
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 means run until Ctrl-C")
    parser.add_argument("--output-json", default=str(OUT_DIR / "imu_status_monitor_summary.json"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rospy.init_node("imu_status_monitor", anonymous=True, disable_signals=True)
    monitor = ImuMonitor(args)
    rospy.Subscriber(args.topic, Imu, monitor.cb, queue_size=200)
    period = 1.0 / max(0.1, args.print_hz)
    deadline = time.monotonic() + args.duration_sec if args.duration_sec > 0.0 else None
    rate = rospy.Rate(50.0)
    last_snapshot: Optional[Dict[str, Any]] = None
    try:
        while not rospy.is_shutdown():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            snap = monitor.snapshot()
            if snap is not None:
                last_snapshot = snap
                if now - monitor.last_print >= period:
                    monitor.print_status(snap)
                    monitor.last_print = now
            rate.sleep()
    except KeyboardInterrupt:
        pass
    if last_snapshot is None:
        last_snapshot = {
            "topic": args.topic,
            "message_count": 0,
            "published_cmd_vel": False,
            "forbidden_sources_used": [],
            "final_decision": "IMU_STATUS_MONITOR_NO_MESSAGES",
        }
    else:
        last_snapshot["final_decision"] = "IMU_STATUS_MONITOR_COMPLETE"
    write_json(Path(args.output_json), last_snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
