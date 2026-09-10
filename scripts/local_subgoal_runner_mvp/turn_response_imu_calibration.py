#!/usr/bin/env python3
"""IMU-based turn response calibration for the local runner chain.

Default mode is read-only: it checks that IMU messages are available and writes
a dry-run summary. Use --execute explicitly to publish angular commands.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "debug" / "local_subgoal_runner_mvp"
SUMMARY_PATH = OUT_DIR / "turn_response_imu_calibration_summary.json"
REPORT_PATH = ROOT / "audit_reports" / "turn_response_imu_calibration_report.md"


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q: Any) -> float:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class ImuCollector:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.samples: List[Dict[str, float]] = []
        self.frame_id: Optional[str] = None

    def cb(self, msg: Imu) -> None:
        stamp = float(msg.header.stamp.to_sec()) if msg.header.stamp else float(rospy.Time.now().to_sec())
        yaw = yaw_from_quat(msg.orientation)
        self.frame_id = msg.header.frame_id
        self.samples.append(
            {
                "stamp_sec": stamp,
                "wall_sec": time.monotonic(),
                "yaw_rad": yaw,
                "angular_velocity_z": float(msg.angular_velocity.z),
            }
        )

    def latest(self) -> Optional[Dict[str, float]]:
        return self.samples[-1] if self.samples else None

    def wait_for_sample(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.samples:
                return True
            time.sleep(0.02)
        return bool(self.samples)


def parse_sweep_pairs(text: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        pairs.append((float(left), float(right)))
    return pairs


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_zero(pub: rospy.Publisher, count: int, rate_hz: float) -> None:
    rate = rospy.Rate(rate_hz)
    zero = Twist()
    for _ in range(max(1, count)):
        if rospy.is_shutdown():
            break
        pub.publish(zero)
        rate.sleep()


def run_turn_trial(
    *,
    collector: ImuCollector,
    pub: rospy.Publisher,
    angular_z: float,
    duration_sec: float,
    rate_hz: float,
    settle_sec: float,
) -> Dict[str, Any]:
    before_index = len(collector.samples)
    start = collector.latest()
    if start is None:
        raise RuntimeError("imu_sample_missing_before_trial")
    cmd = Twist()
    cmd.angular.z = float(angular_z)
    start_sim = float(rospy.Time.now().to_sec())
    start_wall = time.monotonic()
    count = 0
    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown() and float(rospy.Time.now().to_sec()) - start_sim < duration_sec:
        pub.publish(cmd)
        count += 1
        rate.sleep()
    publish_zero(pub, 6, rate_hz)
    if settle_sec > 0.0:
        time.sleep(settle_sec)
    after = collector.latest()
    if after is None:
        raise RuntimeError("imu_sample_missing_after_trial")
    trial_samples = collector.samples[before_index:]
    actual_yaw_delta = normalize_angle(after["yaw_rad"] - start["yaw_rad"])
    expected = float(angular_z) * float(duration_sec)
    abs_expected = abs(expected)
    abs_actual = abs(actual_yaw_delta)
    actual_sim_duration = max(0.0, after["stamp_sec"] - start["stamp_sec"])
    actual_wall_duration = max(0.0, time.monotonic() - start_wall)
    max_abs_wz = max((abs(s["angular_velocity_z"]) for s in trial_samples), default=0.0)
    mean_abs_wz = (
        sum(abs(s["angular_velocity_z"]) for s in trial_samples) / len(trial_samples)
        if trial_samples
        else 0.0
    )
    return {
        "angular_z_commanded_rad_s": float(angular_z),
        "duration_sec_commanded": float(duration_sec),
        "expected_yaw_delta_rad": expected,
        "actual_yaw_delta_rad": actual_yaw_delta,
        "actual_abs_yaw_delta_rad": abs_actual,
        "turn_efficiency_abs_actual_over_abs_expected": abs_actual / abs_expected if abs_expected > 1e-6 else None,
        "yaw_delta_error_rad": actual_yaw_delta - expected,
        "actual_sim_duration_sec": actual_sim_duration,
        "actual_wall_duration_sec": actual_wall_duration,
        "observed_realtime_factor": actual_sim_duration / actual_wall_duration if actual_wall_duration > 1e-6 else None,
        "effective_yaw_rate_rad_per_sim_sec": abs_actual / actual_sim_duration if actual_sim_duration > 1e-6 else None,
        "cmd_publish_count": count,
        "imu_sample_count": len(trial_samples),
        "max_abs_imu_angular_velocity_z_rad_s": max_abs_wz,
        "mean_abs_imu_angular_velocity_z_rad_s": mean_abs_wz,
        "start_imu_stamp_sec": start["stamp_sec"],
        "end_imu_stamp_sec": after["stamp_sec"],
    }


def write_report(summary: Dict[str, Any]) -> None:
    lines = [
        "# Turn Response IMU Calibration Report",
        "",
        f"- final_decision: `{summary.get('final_decision')}`",
        f"- execute: `{summary.get('execute')}`",
        f"- imu_topic: `{summary.get('imu_topic')}`",
        f"- cmd_topic: `{summary.get('cmd_topic')}`",
        f"- requested_trial_count: `{summary.get('requested_trial_count')}`",
        f"- completed_trial_count: `{summary.get('completed_trial_count')}`",
        f"- forbidden_sources_used: `{summary.get('forbidden_sources_used')}`",
        f"- used_gazebo_truth: `{summary.get('used_gazebo_truth')}`",
        f"- called_move_base: `{summary.get('called_move_base')}`",
        f"- sent_navigation_goal: `{summary.get('sent_navigation_goal')}`",
        "",
        "## Trials",
        "",
    ]
    for trial in summary.get("trials", []):
        lines.append(
            "- angular_z={angular_z_commanded_rad_s}, duration={duration_sec_commanded}, "
            "expected={expected_yaw_delta_rad}, actual={actual_yaw_delta_rad}, "
            "efficiency={turn_efficiency_abs_actual_over_abs_expected}".format(**trial)
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="publish angular commands; omitted means no motion")
    parser.add_argument("--imu-topic", default="/trunk_imu")
    parser.add_argument("--cmd-topic", default="/cmd_vel_raw")
    parser.add_argument("--sweep-pairs", default="0.10:1.0,0.15:1.0,0.20:1.0")
    parser.add_argument("--cmd-rate-hz", type=float, default=30.0)
    parser.add_argument("--input-timeout-sec", type=float, default=8.0)
    parser.add_argument("--settle-sec", type=float, default=0.4)
    parser.add_argument("--max-abs-angular-z", type=float, default=0.30)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rospy.init_node("turn_response_imu_calibration", anonymous=True, disable_signals=True)
    collector = ImuCollector(args.imu_topic)
    rospy.Subscriber(args.imu_topic, Imu, collector.cb, queue_size=200)
    pub = rospy.Publisher(args.cmd_topic, Twist, queue_size=10)
    trials_requested = parse_sweep_pairs(args.sweep_pairs)
    if not collector.wait_for_sample(args.input_timeout_sec):
        summary = {
            "final_decision": "TURN_RESPONSE_IMU_NO_IMU",
            "execute": bool(args.execute),
            "imu_topic": args.imu_topic,
            "cmd_topic": args.cmd_topic,
            "requested_trial_count": len(trials_requested),
            "completed_trial_count": 0,
            "trials": [],
            "forbidden_sources_used": [],
            "used_gazebo_truth": False,
            "called_move_base": False,
            "sent_navigation_goal": False,
        }
        write_json(SUMMARY_PATH, summary)
        write_report(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
    trials: List[Dict[str, Any]] = []
    final_decision = "TURN_RESPONSE_IMU_DRY_RUN_READY"
    if args.execute:
        for angular_z, duration_sec in trials_requested:
            if abs(angular_z) > args.max_abs_angular_z:
                final_decision = "TURN_RESPONSE_IMU_BLOCKED_BY_ANGULAR_LIMIT"
                break
            trials.append(
                run_turn_trial(
                    collector=collector,
                    pub=pub,
                    angular_z=angular_z,
                    duration_sec=duration_sec,
                    rate_hz=args.cmd_rate_hz,
                    settle_sec=args.settle_sec,
                )
            )
        else:
            final_decision = "TURN_RESPONSE_IMU_COMPLETE"
    summary = {
        "final_decision": final_decision,
        "execute": bool(args.execute),
        "imu_topic": args.imu_topic,
        "imu_frame_id": collector.frame_id,
        "cmd_topic": args.cmd_topic,
        "requested_trial_count": len(trials_requested),
        "completed_trial_count": len(trials),
        "dry_run_requires_execute_for_motion": not bool(args.execute),
        "trials": trials,
        "forbidden_sources_used": [],
        "used_gazebo_truth": False,
        "called_move_base": False,
        "sent_navigation_goal": False,
    }
    if args.execute:
        publish_zero(pub, 10, args.cmd_rate_hz)
    write_json(SUMMARY_PATH, summary)
    write_report(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if final_decision in {"TURN_RESPONSE_IMU_DRY_RUN_READY", "TURN_RESPONSE_IMU_COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
