#!/usr/bin/env python3
"""Read-only preflight for the odometry required by navigation startup.

This utility intentionally subscribes once and never publishes, changes a
gate, or gives command authority to the audit bundle.  Its only purpose is to
keep the navigation runner from beginning target preparation before the gated
odometry stream has produced an actual usable message.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, Tuple

import rospy
from nav_msgs.msg import Odometry


DEFAULT_ODOM_TOPIC = "/team/livox/icp_odom_gated"
DEFAULT_TIMEOUT_SEC = 45.0


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def describe_usable_odom(message: Any, topic: str) -> Tuple[bool, Dict[str, Any]]:
    pose = message.pose.pose
    position = pose.position
    stamp = message.header.stamp
    payload: Dict[str, Any] = {
        "status": "NAVIGATION_ODOM_READY",
        "topic": topic,
        "message_stamp_sec": float(stamp.to_sec()),
        "position_xyz": [float(position.x), float(position.y), float(position.z)],
        "audit_only": True,
        "command_authority": False,
    }
    if not all(finite(value) for value in (position.x, position.y, position.z)):
        payload.update({"status": "NAVIGATION_ODOM_INVALID", "reason": "non_finite_position"})
        return False, payload
    if not finite(payload["message_stamp_sec"]) or payload["message_stamp_sec"] <= 0.0:
        payload.update({"status": "NAVIGATION_ODOM_INVALID", "reason": "missing_message_stamp"})
        return False, payload
    return True, payload


def wait_for_usable_odom(
    rospy_module: Any,
    odometry_type: Any,
    topic: str,
    timeout_sec: float,
) -> Tuple[bool, Dict[str, Any]]:
    try:
        if not rospy_module.core.is_initialized():
            rospy_module.init_node("navigation_gated_odom_preflight", anonymous=True, disable_signals=True)
        message = rospy_module.wait_for_message(topic, odometry_type, timeout=timeout_sec)
    except Exception as exc:
        return False, {
            "status": "NAVIGATION_ODOM_UNAVAILABLE",
            "topic": topic,
            "timeout_sec": float(timeout_sec),
            "reason": str(exc),
            "audit_only": True,
            "command_authority": False,
        }
    return describe_usable_odom(message, topic)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_sec <= 0.0:
        raise SystemExit("--timeout-sec must be positive")
    ready, payload = wait_for_usable_odom(rospy, Odometry, str(args.topic), float(args.timeout_sec))
    print(json.dumps(payload, sort_keys=True))
    return 0 if ready else 69


if __name__ == "__main__":
    raise SystemExit(main())
