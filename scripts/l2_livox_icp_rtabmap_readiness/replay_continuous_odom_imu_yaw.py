#!/usr/bin/env python3
"""Read-only rosbag replay for continuous ICP-odom / IMU yaw agreement."""

import argparse
import json
import math
from pathlib import Path

import rosbag

from continuous_yaw_consistency import ContinuousYawConsistency


def yaw_from_quat(quaternion):
    siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny_cosp, cosy_cosp)


def run_replay(args):
    monitor = ContinuousYawConsistency(
        math.radians(args.max_yaw_error_deg),
        args.required_consecutive_samples,
    )
    latest_imu = None
    paired_samples = 0
    stale_imu_samples = 0
    first_exceeded = None
    first_trigger = None
    max_sample = None
    topics = [args.odom_topic, args.imu_topic]

    with rosbag.Bag(args.bag, "r") as bag:
        for topic, message, _ in bag.read_messages(topics=topics):
            if topic == args.imu_topic:
                latest_imu = (message.header.stamp.to_sec(), yaw_from_quat(message.orientation))
                continue
            if latest_imu is None:
                stale_imu_samples += 1
                continue
            odom_stamp = message.header.stamp.to_sec()
            imu_age_sec = abs(odom_stamp - latest_imu[0])
            if imu_age_sec > args.max_imu_age_sec:
                stale_imu_samples += 1
                continue
            result = monitor.observe(yaw_from_quat(message.pose.pose.orientation), latest_imu[1])
            paired_samples += 1
            sample = {
                "odom_stamp": odom_stamp,
                "imu_age_sec": imu_age_sec,
                "yaw_error_rad": result["yaw_error_rad"],
                "yaw_error_deg": math.degrees(result["yaw_error_rad"]),
                "consecutive_mismatch_samples": result["consecutive_mismatch_samples"],
            }
            if max_sample is None or sample["yaw_error_rad"] > max_sample["yaw_error_rad"]:
                max_sample = sample
            if (
                first_exceeded is None
                and sample["yaw_error_rad"] > math.radians(args.max_yaw_error_deg)
            ):
                first_exceeded = sample
            if result["triggered"] and first_trigger is None:
                first_trigger = sample

    return {
        "schema_version": "continuous_odom_imu_yaw_replay_v1",
        "bag": str(Path(args.bag).resolve()),
        "odom_topic": args.odom_topic,
        "imu_topic": args.imu_topic,
        "max_imu_age_sec": args.max_imu_age_sec,
        "max_yaw_error_deg": args.max_yaw_error_deg,
        "required_consecutive_samples": args.required_consecutive_samples,
        "paired_samples": paired_samples,
        "stale_or_unavailable_imu_samples": stale_imu_samples,
        "triggered": first_trigger is not None,
        "first_threshold_exceeded": first_exceeded,
        "first_trigger": first_trigger,
        "max_error": max_sample,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--odom-topic", default="/team/livox/icp_odom_raw")
    parser.add_argument("--imu-topic", default="/trunk_imu")
    parser.add_argument("--max-imu-age-sec", type=float, default=0.05)
    parser.add_argument("--max-yaw-error-deg", type=float, default=2.9)
    parser.add_argument("--required-consecutive-samples", type=int, default=10)
    args = parser.parse_args()
    result = run_replay(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
