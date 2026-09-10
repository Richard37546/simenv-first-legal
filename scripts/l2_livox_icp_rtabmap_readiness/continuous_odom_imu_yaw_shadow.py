#!/usr/bin/env python3
"""Read-only online shadow for continuous ICP odom / IMU yaw agreement.

This node has audit authority only.  It never subscribes to, publishes to, or
otherwise influences a navigation or command topic.
"""

import json
import math
import os
import time
from pathlib import Path

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from continuous_yaw_consistency import ContinuousYawConsistency


def yaw_from_quat(quaternion):
    siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny_cosp, cosy_cosp)


def write_atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


class ContinuousOdomImuYawShadow:
    """Audit-only continuous relative-yaw observer."""

    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/team/livox/icp_odom_raw")
        self.imu_topic = rospy.get_param("~imu_topic", "/trunk_imu")
        self.status_topic = rospy.get_param(
            "~status_topic", "/audit/continuous_odom_imu_yaw_shadow_status"
        )
        self.output_dir = Path(rospy.get_param("~output_dir"))
        self.max_imu_age_sec = float(rospy.get_param("~max_imu_age_sec", 0.05))
        self.max_yaw_error_deg = float(rospy.get_param("~max_yaw_error_deg", 2.9))
        self.required_consecutive_samples = int(
            rospy.get_param("~required_consecutive_samples", 10)
        )
        self.monitor = ContinuousYawConsistency(
            math.radians(self.max_yaw_error_deg), self.required_consecutive_samples
        )
        self.latest_imu = None
        self.paired_samples = 0
        self.stale_or_unavailable_imu_samples = 0
        self.first_threshold_exceeded = None
        self.first_trigger = None
        self.max_error = None
        self.closed = False
        self.events_path = self.output_dir / "shadow_events.jsonl"
        self.result_path = self.output_dir / "shadow_result.json"
        self.ready_path = self.output_dir / "ready.json"
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10, latch=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_ready()
        self.append_event({"event_type": "shadow_ready", "wall_time_sec": time.time()})
        self.imu_subscriber = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=100)
        self.odom_subscriber = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=100)
        rospy.on_shutdown(self.close)

    def authority(self):
        return {
            "audit_authority": True,
            "production_authority": False,
            "odom_authority": False,
            "command_authority": False,
            "selection_authority": False,
            "completion_authority": False,
            "recoverability_authority": False,
            "fallback_authority": False,
        }

    def write_ready(self):
        payload = {
            "schema_version": "continuous_odom_imu_yaw_shadow_v1",
            "status": "READY",
            "pid": os.getpid(),
            "odom_topic": self.odom_topic,
            "imu_topic": self.imu_topic,
            "status_topic": self.status_topic,
            "output_dir": str(self.output_dir.resolve()),
            "max_imu_age_sec": self.max_imu_age_sec,
            "max_yaw_error_deg": self.max_yaw_error_deg,
            "required_consecutive_samples": self.required_consecutive_samples,
        }
        payload.update(self.authority())
        write_atomic_json(self.ready_path, payload)
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def append_event(self, payload):
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def current_result(self):
        payload = {
            "schema_version": "continuous_odom_imu_yaw_shadow_v1",
            "status": "CLOSED" if self.closed else "RUNNING",
            "pid": os.getpid(),
            "odom_topic": self.odom_topic,
            "imu_topic": self.imu_topic,
            "status_topic": self.status_topic,
            "output_dir": str(self.output_dir.resolve()),
            "max_imu_age_sec": self.max_imu_age_sec,
            "max_yaw_error_deg": self.max_yaw_error_deg,
            "required_consecutive_samples": self.required_consecutive_samples,
            "paired_samples": self.paired_samples,
            "stale_or_unavailable_imu_samples": self.stale_or_unavailable_imu_samples,
            "triggered": self.first_trigger is not None,
            "first_threshold_exceeded": self.first_threshold_exceeded,
            "first_trigger": self.first_trigger,
            "max_error": self.max_error,
            "shutdown_wall_time_sec": time.time() if self.closed else None,
        }
        payload.update(self.authority())
        return payload

    def publish_result(self):
        payload = self.current_result()
        write_atomic_json(self.result_path, payload)
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def imu_callback(self, message):
        quaternion = message.orientation
        norm = sum(float(value) ** 2 for value in (quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        if not math.isfinite(norm) or norm <= 1e-6:
            return
        self.latest_imu = (message.header.stamp.to_sec(), yaw_from_quat(quaternion))

    def odom_callback(self, message):
        odom_stamp = message.header.stamp.to_sec()
        if self.latest_imu is None:
            self.stale_or_unavailable_imu_samples += 1
            return
        imu_age_sec = abs(odom_stamp - self.latest_imu[0])
        if imu_age_sec > self.max_imu_age_sec:
            self.stale_or_unavailable_imu_samples += 1
            return
        observed = self.monitor.observe(
            yaw_from_quat(message.pose.pose.orientation), self.latest_imu[1]
        )
        self.paired_samples += 1
        sample = {
            "event_type": "paired_sample",
            "odom_stamp": odom_stamp,
            "imu_age_sec": imu_age_sec,
            "yaw_error_rad": observed["yaw_error_rad"],
            "yaw_error_deg": math.degrees(observed["yaw_error_rad"]),
            "consecutive_mismatch_samples": observed["consecutive_mismatch_samples"],
            "triggered": observed["triggered"],
        }
        self.append_event(sample)
        if self.max_error is None or sample["yaw_error_rad"] > self.max_error["yaw_error_rad"]:
            self.max_error = dict(sample)
        if self.first_threshold_exceeded is None and sample["yaw_error_deg"] > self.max_yaw_error_deg:
            self.first_threshold_exceeded = dict(sample)
            self.publish_result()
        if observed["triggered"] and self.first_trigger is None:
            self.first_trigger = dict(sample)
            self.publish_result()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.publish_result()


def main():
    rospy.init_node("continuous_odom_imu_yaw_shadow")
    ContinuousOdomImuYawShadow()
    rospy.spin()


if __name__ == "__main__":
    main()
