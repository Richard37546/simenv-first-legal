#!/usr/bin/env python3
"""Passive current-interface metadata probe; never publishes motion or tracks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence

from danger_perception_node import ODOM_TOPIC, POINTS_TOPIC, RGB_INFO_TOPIC, RGB_TOPIC, atomic_write_json


def cadence(stamps: Sequence[float]) -> Dict[str, Optional[float]]:
    values = [float(value) for value in stamps if math.isfinite(float(value))]
    deltas = [b - a for a, b in zip(values, values[1:]) if b > a]
    return {
        "count": len(values),
        "median_period_sec": sorted(deltas)[len(deltas) // 2] if deltas else None,
        "approx_rate_hz": (1.0 / (sorted(deltas)[len(deltas) // 2])) if deltas and sorted(deltas)[len(deltas) // 2] > 0.0 else None,
    }


class Probe:
    def __init__(self, rospy: Any) -> None:
        from nav_msgs.msg import Odometry  # type: ignore
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2  # type: ignore

        self.rospy = rospy
        self.stamps: Dict[str, List[float]] = defaultdict(list)
        self.metadata: Dict[str, Any] = {}
        self.subscribers = [
            rospy.Subscriber(RGB_TOPIC, Image, self.rgb, queue_size=8),
            rospy.Subscriber(RGB_INFO_TOPIC, CameraInfo, self.camera_info, queue_size=3),
            rospy.Subscriber(POINTS_TOPIC, PointCloud2, self.points, queue_size=8),
            rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom, queue_size=12),
        ]

    def remember_stamp(self, key: str, msg: Any) -> None:
        stamp = float(msg.header.stamp.to_sec())
        if math.isfinite(stamp) and len(self.stamps[key]) < 200:
            self.stamps[key].append(stamp)

    def rgb(self, msg: Any) -> None:
        self.remember_stamp("rgb", msg)
        self.metadata["rgb"] = {
            "topic": RGB_TOPIC, "type": "sensor_msgs/Image", "frame_id": str(msg.header.frame_id),
            "width": int(msg.width), "height": int(msg.height), "encoding": str(msg.encoding), "step": int(msg.step),
        }

    def camera_info(self, msg: Any) -> None:
        self.remember_stamp("rgb_camera_info", msg)
        self.metadata["rgb_camera_info"] = {
            "topic": RGB_INFO_TOPIC, "type": "sensor_msgs/CameraInfo", "frame_id": str(msg.header.frame_id),
            "width": int(msg.width), "height": int(msg.height), "K": [float(value) for value in msg.K],
            "distortion_model": str(msg.distortion_model), "D": [float(value) for value in msg.D],
        }

    def points(self, msg: Any) -> None:
        self.remember_stamp("pointcloud", msg)
        self.metadata["pointcloud"] = {
            "topic": POINTS_TOPIC, "type": "sensor_msgs/PointCloud2", "frame_id": str(msg.header.frame_id),
            "width": int(msg.width), "height": int(msg.height), "point_step": int(msg.point_step), "row_step": int(msg.row_step),
            "is_dense": bool(msg.is_dense), "fields": [{"name": str(field.name), "offset": int(field.offset), "datatype": int(field.datatype), "count": int(field.count)} for field in msg.fields],
            "organized": bool(int(msg.height) > 1),
        }

    def odom(self, msg: Any) -> None:
        self.remember_stamp("odom", msg)
        self.metadata["odom"] = {
            "topic": ODOM_TOPIC, "type": "nav_msgs/Odometry", "frame_id": str(msg.header.frame_id), "child_frame_id": str(msg.child_frame_id),
        }

    @staticmethod
    def nearest_deltas(left: Sequence[float], right: Sequence[float]) -> List[float]:
        if not right:
            return []
        return [min(abs(value - other) for other in right) for value in left]

    def payload(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {"schema_version": 1, "metadata": self.metadata, "cadence": {key: cadence(value) for key, value in self.stamps.items()}}
        rgb = self.stamps.get("rgb", [])
        clouds = self.stamps.get("pointcloud", [])
        odom = self.stamps.get("odom", [])
        output["nearest_timestamp_deltas_sec"] = {
            "rgb_to_pointcloud": self.nearest_deltas(rgb, clouds)[:20],
            "pointcloud_to_odom": self.nearest_deltas(clouds, odom)[:20],
        }
        return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Passive RGB-D metadata/timestamp probe")
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--output", default="debug/rgbd_danger_perception/current_metadata.json")
    args = parser.parse_args(argv)
    import rospy  # type: ignore
    rospy.init_node("rgbd_danger_metadata_probe", anonymous=False)
    probe = Probe(rospy)
    deadline = time.monotonic() + max(0.0, float(args.duration_sec))
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        rospy.sleep(0.05)
    atomic_write_json(Path(args.output), probe.payload())
    print(json.dumps(probe.payload(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
