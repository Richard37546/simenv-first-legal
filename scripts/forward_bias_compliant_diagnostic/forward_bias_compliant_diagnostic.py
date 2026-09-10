#!/usr/bin/env python3
"""Compliant forward-bias diagnostic.

Default mode is dry-run and does not publish /cmd_vel. Execute mode must be
requested explicitly and uses only /team/livox/icp_odom_gated for odometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "debug" / "forward_bias_compliant_diagnostic"
PLAN_PATH = OUT_DIR / "forward_bias_diagnostic_plan.json"
SUMMARY_PATH = OUT_DIR / "forward_bias_diagnostic_summary.json"
TRIALS_CSV_PATH = OUT_DIR / "forward_bias_trials.csv"
TIME_SERIES_CSV_PATH = OUT_DIR / "forward_bias_time_series.csv"

DEFAULT_ODOM_TOPIC = "/team/livox/icp_odom_gated"
CMD_TOPIC = "/cmd_vel"
EXPECTED_ODOM_FRAME = "team_livox_odom"
EXPECTED_ODOM_CHILD = "base"
FORBIDDEN_SOURCES = [
    "/Odometry_gazebo",
    "/state_from_gazebo",
    "/ground_truth/*",
    "/gazebo/model_states",
    "/gazebo/link_states",
    "danger_truth.json",
    "gazebo/get_model_state",
    "gazebo/set_model_state",
]


@dataclass
class PoseSample:
    stamp_sec: float
    wall_sec: float
    x: float
    y: float
    yaw: float
    frame_id: str
    child_frame_id: str


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_speeds(raw: str) -> List[float]:
    speeds = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0:
            raise argparse.ArgumentTypeError("speeds must be non-negative")
        speeds.append(value)
    if not speeds:
        raise argparse.ArgumentTypeError("at least one speed is required")
    return speeds


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "mode": "execute" if args.execute else "dry_run",
        "execute_requested": bool(args.execute),
        "odom_topic": args.odom_topic,
        "allowed_read_topics": [args.odom_topic, "/clock", CMD_TOPIC],
        "allowed_publish_topics": [CMD_TOPIC] if args.execute else [],
        "forbidden_sources_used": [],
        "used_gazebo_truth": False,
        "called_move_base": False,
        "sent_navigation_goal": False,
        "no_motion": {
            "included": True,
            "duration_sec": float(args.no_motion_sec),
            "linear_x": 0.0,
            "angular_z": 0.0,
        },
        "forward_trials": [
            {
                "trial_id": "speed_%0.2f_repeat_%d" % (speed, repeat_index),
                "speed_mps": speed,
                "duration_sec": float(args.duration_sec),
                "repeat_index": repeat_index,
                "commanded_linear_x": speed,
                "commanded_angular_z": 0.0,
            }
            for speed in args.speeds
            for repeat_index in range(1, int(args.repeats) + 1)
        ],
        "thresholds": {
            "yaw_drift_suspect_threshold_rad": float(args.yaw_drift_suspect_threshold_rad),
            "cross_track_suspect_threshold_m": float(args.cross_track_suspect_threshold_m),
            "no_motion_yaw_suspect_threshold_rad": float(args.no_motion_yaw_suspect_threshold_rad),
            "distance_efficiency_min": float(args.distance_efficiency_min),
            "time_series_decimation_sec": float(args.time_series_decimation_sec),
        },
        "output_files": {
            "plan": str(PLAN_PATH),
            "summary": str(SUMMARY_PATH),
            "trials_csv": str(TRIALS_CSV_PATH),
            "time_series_csv": str(TIME_SERIES_CSV_PATH),
        },
    }


def base_summary(args: argparse.Namespace, plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": plan["mode"],
        "execute_requested": bool(args.execute),
        "cmd_vel_published": False,
        "zero_cmd_vel_published_count": 0,
        "forbidden_sources_used": [],
        "used_gazebo_truth": False,
        "called_move_base": False,
        "sent_navigation_goal": False,
        "odom_topic": args.odom_topic,
        "speed_list": args.speeds,
        "duration_sec": float(args.duration_sec),
        "repeats": int(args.repeats),
        "no_motion_result": None,
        "trial_count": len(plan["forward_trials"]),
        "completed_trial_count": 0,
        "failed_trial_count": 0,
        "mean_yaw_delta_by_speed": {},
        "mean_abs_cross_track_by_speed": {},
        "drift_direction_consistency_by_speed": {},
        "time_series_full_logging_enabled": True,
        "time_series_sample_count_total": 0,
        "no_motion_time_series_sample_count": 0,
        "forward_time_series_sample_count_total": 0,
        "yaw_drift_monotonicity_by_trial": {},
        "max_yaw_delta_during_trial_by_trial": {},
        "final_yaw_delta_by_trial": {},
        "yaw_jump_suspected_by_trial": {},
        "speed_dependency_suspected": False,
        "odom_stability_suspected": False,
        "low_level_control_or_gait_bias_possible": False,
        "odom_estimation_yaw_drift_possible": False,
        "recommended_next_step": "Run --execute only in a cleared test area after reviewing the dry-run plan.",
        "diagnostic_plan_path": str(PLAN_PATH),
        "trials_csv_path": str(TRIALS_CSV_PATH),
        "time_series_csv_path": str(TIME_SERIES_CSV_PATH),
    }


def dry_run(args: argparse.Namespace) -> int:
    plan = build_plan(args)
    summary = base_summary(args, plan)
    summary.update(
        {
            "dry_run_checks": {
                "default_mode_is_dry_run": True,
                "execute_requires_explicit_flag": True,
                "odom_topic_allowed": args.odom_topic == DEFAULT_ODOM_TOPIC,
                "output_directory": str(OUT_DIR),
                "will_publish_cmd_vel": False,
            },
            "recommended_next_step": "Inspect the plan, then manually run --execute only when safe.",
        }
    )
    write_json(PLAN_PATH, plan)
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def init_rospy(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    import rospy  # type: ignore
    from geometry_msgs.msg import Twist  # type: ignore
    from nav_msgs.msg import Odometry  # type: ignore

    rospy.init_node("forward_bias_compliant_diagnostic", anonymous=True, disable_signals=True)
    return rospy, Twist, Odometry


def read_odom(rospy: Any, odom_type: Any, topic: str, timeout_sec: float) -> PoseSample:
    msg = rospy.wait_for_message(topic, odom_type, timeout=timeout_sec)
    return pose_sample_from_msg(msg)


def pose_sample_from_msg(msg: Any) -> PoseSample:
    frame_id = str(msg.header.frame_id)
    child_frame_id = str(msg.child_frame_id)
    if frame_id != EXPECTED_ODOM_FRAME or child_frame_id != EXPECTED_ODOM_CHILD:
        raise RuntimeError(
            "unexpected odom frame: header.frame_id=%r child_frame_id=%r" % (frame_id, child_frame_id)
        )
    pose = msg.pose.pose
    return PoseSample(
        stamp_sec=float(msg.header.stamp.to_sec()),
        wall_sec=time.monotonic(),
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=yaw_from_quat(pose.orientation),
        frame_id=frame_id,
        child_frame_id=child_frame_id,
    )


def validate_samples(samples: List[PoseSample]) -> None:
    for sample in samples:
        if sample.frame_id != EXPECTED_ODOM_FRAME or sample.child_frame_id != EXPECTED_ODOM_CHILD:
            raise RuntimeError(
                "unexpected odom frame in time series: header.frame_id=%r child_frame_id=%r"
                % (sample.frame_id, sample.child_frame_id)
            )


def should_keep_sample(samples: List[PoseSample], sample: PoseSample, decimation_sec: float) -> bool:
    if not samples:
        return True
    if decimation_sec <= 0.0:
        return True
    return sample.stamp_sec - samples[-1].stamp_sec >= decimation_sec


def zero_stop(pub: Any, twist_type: Any, rospy: Any, count: int, rate_hz: float) -> int:
    rate = rospy.Rate(rate_hz)
    msg = twist_type()
    published = 0
    for _ in range(int(count)):
        if rospy.is_shutdown():
            break
        pub.publish(msg)
        published += 1
        rate.sleep()
    return published


def publish_command_for_sim_duration(
    pub: Any,
    twist_type: Any,
    rospy: Any,
    odom_type: Any,
    odom_topic: str,
    linear_x: float,
    angular_z: float,
    duration_sec: float,
    rate_hz: float,
    time_series_decimation_sec: float,
) -> Tuple[int, float, float, float, float, List[PoseSample]]:
    samples: List[PoseSample] = []

    def cb(msg: Any) -> None:
        try:
            sample = pose_sample_from_msg(msg)
        except Exception:
            return
        if should_keep_sample(samples, sample, time_series_decimation_sec):
            samples.append(sample)

    sub = rospy.Subscriber(odom_topic, odom_type, cb, queue_size=200)
    cmd = twist_type()
    cmd.linear.x = float(linear_x)
    cmd.angular.z = float(angular_z)
    rate = rospy.Rate(rate_hz)
    sim_start = float(rospy.Time.now().to_sec())
    wall_start = time.monotonic()
    publish_count = 0
    try:
        while not rospy.is_shutdown():
            sim_now = float(rospy.Time.now().to_sec())
            if sim_now - sim_start >= float(duration_sec):
                break
            pub.publish(cmd)
            publish_count += 1
            rate.sleep()
        sim_end = float(rospy.Time.now().to_sec())
        wall_end = time.monotonic()
        validate_samples(samples)
        return publish_count, sim_start, sim_end, wall_start, wall_end, list(samples)
    finally:
        sub.unregister()


def collect_passive_samples(
    rospy: Any,
    odom_type: Any,
    topic: str,
    duration_sec: float,
    timeout_sec: float,
    time_series_decimation_sec: float,
) -> List[PoseSample]:
    samples: List[PoseSample] = []

    def cb(msg: Any) -> None:
        try:
            sample = pose_sample_from_msg(msg)
        except Exception:
            return
        if should_keep_sample(samples, sample, time_series_decimation_sec):
            samples.append(sample)

    sub = rospy.Subscriber(topic, odom_type, cb, queue_size=200)
    deadline = time.monotonic() + float(timeout_sec)
    try:
        while not rospy.is_shutdown() and not samples and time.monotonic() < deadline:
            time.sleep(0.02)
        if not samples:
            raise RuntimeError("timed out waiting for odom samples on %s" % topic)
        sim_start = float(rospy.Time.now().to_sec())
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown():
            if float(rospy.Time.now().to_sec()) - sim_start >= float(duration_sec):
                break
            rate.sleep()
        validate_samples(samples)
        return list(samples)
    finally:
        sub.unregister()


def drift_direction(cross_track: float) -> str:
    if cross_track > 0.0:
        return "left_of_start_heading"
    if cross_track < 0.0:
        return "right_of_start_heading"
    return "none"


def compute_trial_metrics(
    trial_id: str,
    speed: float,
    duration_sec: float,
    repeat_index: int,
    commanded_linear_x: float,
    commanded_angular_z: float,
    cmd_publish_count: int,
    cmd_publish_rate_hz: float,
    sim_start: float,
    sim_end: float,
    wall_start: float,
    wall_end: float,
    odom_topic: str,
    start: PoseSample,
    end: PoseSample,
    odom_message_count: int,
    thresholds: argparse.Namespace,
) -> Dict[str, Any]:
    dx = end.x - start.x
    dy = end.y - start.y
    yaw_delta = wrap(end.yaw - start.yaw)
    forward_unit = (math.cos(start.yaw), math.sin(start.yaw))
    left_unit = (-math.sin(start.yaw), math.cos(start.yaw))
    along_track = dx * forward_unit[0] + dy * forward_unit[1]
    cross_track = dx * left_unit[0] + dy * left_unit[1]
    displacement = math.hypot(dx, dy)
    expected_distance = float(speed) * float(duration_sec)
    distance_efficiency = displacement / expected_distance if expected_distance > 1e-9 else None
    yaw_per_meter = yaw_delta / displacement if displacement > 1e-9 else None
    actual_sim_duration = float(sim_end) - float(sim_start)
    yaw_per_sec = yaw_delta / actual_sim_duration if actual_sim_duration > 1e-9 else None
    forward_bias_suspected = (
        abs(yaw_delta) > float(thresholds.yaw_drift_suspect_threshold_rad)
        or abs(cross_track) > float(thresholds.cross_track_suspect_threshold_m)
    )
    return {
        "trial_id": trial_id,
        "speed_mps": float(speed),
        "duration_sec": float(duration_sec),
        "repeat_index": int(repeat_index),
        "commanded_linear_x": float(commanded_linear_x),
        "commanded_angular_z": float(commanded_angular_z),
        "cmd_publish_count": int(cmd_publish_count),
        "cmd_publish_rate_hz": float(cmd_publish_rate_hz),
        "sim_time_start_sec": float(sim_start),
        "sim_time_end_sec": float(sim_end),
        "actual_sim_duration_sec": actual_sim_duration,
        "wall_time_duration_sec": float(wall_end) - float(wall_start),
        "odom_topic": odom_topic,
        "odom_frame_id": start.frame_id,
        "odom_child_frame_id": start.child_frame_id,
        "odom_message_count": int(odom_message_count),
        "pose_start_x": start.x,
        "pose_start_y": start.y,
        "pose_start_yaw": start.yaw,
        "pose_end_x": end.x,
        "pose_end_y": end.y,
        "pose_end_yaw": end.yaw,
        "yaw_delta_rad": yaw_delta,
        "yaw_delta_deg": math.degrees(yaw_delta),
        "displacement_m": displacement,
        "expected_distance_m": expected_distance,
        "distance_efficiency_ratio": distance_efficiency,
        "along_track_progress_m": along_track,
        "cross_track_error_m": cross_track,
        "abs_cross_track_error_m": abs(cross_track),
        "drift_direction_relative_to_start_heading": drift_direction(cross_track),
        "yaw_drift_rate_rad_per_meter": yaw_per_meter,
        "yaw_drift_rate_deg_per_meter": math.degrees(yaw_per_meter) if yaw_per_meter is not None else None,
        "yaw_drift_rate_rad_per_sec": yaw_per_sec,
        "physical_bias_confirmed": False,
        "odom_measured_bias_confirmed": bool(forward_bias_suspected),
        "forward_bias_suspected": bool(forward_bias_suspected),
        "status": "PASS",
    }


def write_trials_csv(rows: List[Dict[str, Any]]) -> None:
    TRIALS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "speed_mps",
        "duration_sec",
        "repeat_index",
        "commanded_linear_x",
        "commanded_angular_z",
        "cmd_publish_count",
        "cmd_publish_rate_hz",
        "sim_time_start_sec",
        "sim_time_end_sec",
        "actual_sim_duration_sec",
        "wall_time_duration_sec",
        "odom_topic",
        "odom_frame_id",
        "odom_child_frame_id",
        "odom_message_count",
        "pose_start_x",
        "pose_start_y",
        "pose_start_yaw",
        "pose_end_x",
        "pose_end_y",
        "pose_end_yaw",
        "yaw_delta_rad",
        "yaw_delta_deg",
        "displacement_m",
        "expected_distance_m",
        "distance_efficiency_ratio",
        "along_track_progress_m",
        "cross_track_error_m",
        "abs_cross_track_error_m",
        "drift_direction_relative_to_start_heading",
        "yaw_drift_rate_rad_per_meter",
        "yaw_drift_rate_deg_per_meter",
        "yaw_drift_rate_rad_per_sec",
        "physical_bias_confirmed",
        "odom_measured_bias_confirmed",
        "forward_bias_suspected",
        "status",
        "failure_reason",
    ]
    with TRIALS_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_time_series_csv(rows: Iterable[Dict[str, Any]]) -> None:
    TIME_SERIES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "sample_index",
        "stamp_sec",
        "wall_sec",
        "elapsed_sim_sec",
        "x",
        "y",
        "yaw",
        "yaw_delta_from_start_rad",
        "yaw_delta_from_start_deg",
        "displacement_from_start_m",
        "along_track_progress_m",
        "cross_track_error_m",
        "frame_id",
        "child_frame_id",
    ]
    with TIME_SERIES_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_time_series_rows(trial_id: str, samples: List[PoseSample], include_track: bool) -> List[Dict[str, Any]]:
    if not samples:
        return []
    start = samples[0]
    forward_unit = (math.cos(start.yaw), math.sin(start.yaw))
    left_unit = (-math.sin(start.yaw), math.cos(start.yaw))
    rows: List[Dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        dx = sample.x - start.x
        dy = sample.y - start.y
        yaw_delta = wrap(sample.yaw - start.yaw)
        row: Dict[str, Any] = {
            "trial_id": trial_id,
            "sample_index": sample_index,
            "stamp_sec": sample.stamp_sec,
            "wall_sec": sample.wall_sec,
            "elapsed_sim_sec": sample.stamp_sec - start.stamp_sec,
            "x": sample.x,
            "y": sample.y,
            "yaw": sample.yaw,
            "yaw_delta_from_start_rad": yaw_delta,
            "yaw_delta_from_start_deg": math.degrees(yaw_delta),
            "displacement_from_start_m": math.hypot(dx, dy),
            "frame_id": sample.frame_id,
            "child_frame_id": sample.child_frame_id,
        }
        if include_track:
            row["along_track_progress_m"] = dx * forward_unit[0] + dy * forward_unit[1]
            row["cross_track_error_m"] = dx * left_unit[0] + dy * left_unit[1]
        else:
            row["along_track_progress_m"] = None
            row["cross_track_error_m"] = None
        rows.append(row)
    return rows


def time_series_diagnostics(samples: List[PoseSample]) -> Dict[str, Any]:
    if not samples:
        return {
            "monotonicity": "unknown",
            "max_yaw_delta": None,
            "final_yaw_delta": None,
            "yaw_jump_suspected": False,
        }
    start_yaw = samples[0].yaw
    yaw_deltas = [wrap(sample.yaw - start_yaw) for sample in samples]
    abs_yaw_deltas = [abs(value) for value in yaw_deltas]
    adjacent_jumps = [abs(yaw_deltas[i] - yaw_deltas[i - 1]) for i in range(1, len(yaw_deltas))]
    nondecreasing_abs = all(abs_yaw_deltas[i] <= abs_yaw_deltas[i + 1] + 1e-9 for i in range(len(abs_yaw_deltas) - 1))
    nonincreasing_abs = all(abs_yaw_deltas[i] + 1e-9 >= abs_yaw_deltas[i + 1] for i in range(len(abs_yaw_deltas) - 1))
    if nondecreasing_abs:
        monotonicity = "abs_yaw_delta_nondecreasing"
    elif nonincreasing_abs:
        monotonicity = "abs_yaw_delta_nonincreasing"
    else:
        monotonicity = "not_monotonic"
    return {
        "monotonicity": monotonicity,
        "max_yaw_delta": max(yaw_deltas, key=abs),
        "final_yaw_delta": yaw_deltas[-1],
        "yaw_jump_suspected": any(jump > 0.05 for jump in adjacent_jumps),
    }


def update_time_series_summary(
    summary: Dict[str, Any],
    trial_id: str,
    samples: List[PoseSample],
    is_no_motion: bool,
) -> None:
    diag = time_series_diagnostics(samples)
    sample_count = len(samples)
    summary["time_series_sample_count_total"] += sample_count
    if is_no_motion:
        summary["no_motion_time_series_sample_count"] = sample_count
    else:
        summary["forward_time_series_sample_count_total"] += sample_count
    summary["yaw_drift_monotonicity_by_trial"][trial_id] = diag["monotonicity"]
    summary["max_yaw_delta_during_trial_by_trial"][trial_id] = diag["max_yaw_delta"]
    summary["final_yaw_delta_by_trial"][trial_id] = diag["final_yaw_delta"]
    summary["yaw_jump_suspected_by_trial"][trial_id] = diag["yaw_jump_suspected"]


def summarize_trials(summary: Dict[str, Any], trials: List[Dict[str, Any]]) -> None:
    completed = [row for row in trials if row.get("status") == "PASS"]
    failed = [row for row in trials if row.get("status") != "PASS"]
    summary["completed_trial_count"] = len(completed)
    summary["failed_trial_count"] = len(failed)
    summary["trial_count"] = len(trials)
    mean_yaw: Dict[str, float] = {}
    mean_cross: Dict[str, float] = {}
    consistency: Dict[str, Any] = {}
    for speed in sorted({float(row["speed_mps"]) for row in completed}):
        key = "%0.2f" % speed
        rows = [row for row in completed if float(row["speed_mps"]) == speed]
        mean_yaw[key] = sum(float(row["yaw_delta_rad"]) for row in rows) / len(rows)
        mean_cross[key] = sum(float(row["abs_cross_track_error_m"]) for row in rows) / len(rows)
        directions = [str(row["drift_direction_relative_to_start_heading"]) for row in rows]
        most_common = max(set(directions), key=directions.count)
        consistency[key] = {
            "most_common_direction": most_common,
            "count": directions.count(most_common),
            "total": len(directions),
            "consistent": directions.count(most_common) == len(directions),
        }
    summary["mean_yaw_delta_by_speed"] = mean_yaw
    summary["mean_abs_cross_track_by_speed"] = mean_cross
    summary["drift_direction_consistency_by_speed"] = consistency
    summary["low_level_control_or_gait_bias_possible"] = any(
        bool(row.get("forward_bias_suspected")) for row in completed
    )
    summary["odom_estimation_yaw_drift_possible"] = bool(summary.get("odom_stability_suspected")) or any(
        bool(row.get("forward_bias_suspected")) for row in completed
    )
    if len(mean_cross) >= 2:
        ordered = [mean_cross[key] for key in sorted(mean_cross)]
        summary["speed_dependency_suspected"] = max(ordered) - min(ordered) > 0.05
    if failed:
        summary["recommended_next_step"] = "Review failed trials and odom frame/timeout before repeating execute."
    elif completed:
        summary["recommended_next_step"] = (
            "Compare odom-only drift with external visual observation before changing control compensation."
        )


def run_execute(args: argparse.Namespace) -> int:
    plan = build_plan(args)
    summary = base_summary(args, plan)
    write_json(PLAN_PATH, plan)
    rospy, twist_type, odom_type = init_rospy(args)
    pub = rospy.Publisher(CMD_TOPIC, twist_type, queue_size=1)
    trials: List[Dict[str, Any]] = []
    time_series_rows: List[Dict[str, Any]] = []
    cmd_vel_published = False
    zero_count = 0
    try:
        zero_count += zero_stop(pub, twist_type, rospy, 3, args.cmd_publish_rate_hz)
        cmd_vel_published = zero_count > 0
        no_motion_samples = collect_passive_samples(
            rospy,
            odom_type,
            args.odom_topic,
            args.no_motion_sec,
            args.odom_timeout_sec,
            args.time_series_decimation_sec,
        )
        time_series_rows.extend(build_time_series_rows("no_motion", no_motion_samples, include_track=False))
        update_time_series_summary(summary, "no_motion", no_motion_samples, is_no_motion=True)
        no_motion_start = no_motion_samples[0]
        no_motion_end = no_motion_samples[-1]
        no_motion_yaw_delta = wrap(no_motion_end.yaw - no_motion_start.yaw)
        no_motion_xy = math.hypot(no_motion_end.x - no_motion_start.x, no_motion_end.y - no_motion_start.y)
        odom_stability_suspected = abs(no_motion_yaw_delta) > float(args.no_motion_yaw_suspect_threshold_rad)
        summary["no_motion_result"] = {
            "yaw_start": no_motion_start.yaw,
            "yaw_end": no_motion_end.yaw,
            "yaw_delta_rad": no_motion_yaw_delta,
            "yaw_delta_deg": math.degrees(no_motion_yaw_delta),
            "xy_drift_m": no_motion_xy,
            "odom_message_count": len(no_motion_samples),
            "odom_yaw_stability_suspect": bool(odom_stability_suspected),
            "pass": not odom_stability_suspected,
        }
        summary["odom_stability_suspected"] = bool(odom_stability_suspected)

        trial_index = 0
        for speed in args.speeds:
            for repeat_index in range(1, int(args.repeats) + 1):
                trial_index += 1
                trial_id = "speed_%0.2f_repeat_%d" % (speed, repeat_index)
                try:
                    zero_count += zero_stop(pub, twist_type, rospy, 3, args.cmd_publish_rate_hz)
                    start = read_odom(rospy, odom_type, args.odom_topic, args.odom_timeout_sec)
                    (
                        publish_count,
                        sim_start,
                        sim_end,
                        wall_start,
                        wall_end,
                        command_samples,
                    ) = publish_command_for_sim_duration(
                        pub,
                        twist_type,
                        rospy,
                        odom_type,
                        args.odom_topic,
                        speed,
                        0.0,
                        args.duration_sec,
                        args.cmd_publish_rate_hz,
                        args.time_series_decimation_sec,
                    )
                    cmd_vel_published = cmd_vel_published or publish_count > 0
                    zero_count += zero_stop(pub, twist_type, rospy, 5, args.cmd_publish_rate_hz)
                    end = read_odom(rospy, odom_type, args.odom_topic, args.odom_timeout_sec)
                    trial_samples = [start] + command_samples + [end]
                    row = compute_trial_metrics(
                        trial_id,
                        speed,
                        args.duration_sec,
                        repeat_index,
                        speed,
                        0.0,
                        publish_count,
                        args.cmd_publish_rate_hz,
                        sim_start,
                        sim_end,
                        wall_start,
                        wall_end,
                        args.odom_topic,
                        start,
                        end,
                        len(trial_samples),
                        args,
                    )
                    trials.append(row)
                    time_series_rows.extend(build_time_series_rows(trial_id, trial_samples, include_track=True))
                    update_time_series_summary(summary, trial_id, trial_samples, is_no_motion=False)
                except Exception as exc:
                    zero_count += zero_stop(pub, twist_type, rospy, 5, args.cmd_publish_rate_hz)
                    trials.append(
                        {
                            "trial_id": trial_id,
                            "speed_mps": speed,
                            "duration_sec": args.duration_sec,
                            "repeat_index": repeat_index,
                            "commanded_linear_x": speed,
                            "commanded_angular_z": 0.0,
                            "cmd_publish_count": 0,
                            "cmd_publish_rate_hz": args.cmd_publish_rate_hz,
                            "odom_topic": args.odom_topic,
                            "physical_bias_confirmed": False,
                            "odom_measured_bias_confirmed": False,
                            "forward_bias_suspected": False,
                            "status": "FAIL",
                            "failure_reason": str(exc),
                        }
                    )
                    summarize_trials(summary, trials)
                    summary["cmd_vel_published"] = bool(cmd_vel_published)
                    summary["zero_cmd_vel_published_count"] = zero_count
                    write_trials_csv(trials)
                    write_time_series_csv(time_series_rows)
                    write_json(SUMMARY_PATH, summary)
                    return 1
    finally:
        if "pub" in locals() and "rospy" in locals():
            zero_count += zero_stop(pub, twist_type, rospy, 5, args.cmd_publish_rate_hz)

    summarize_trials(summary, trials)
    summary["cmd_vel_published"] = bool(cmd_vel_published)
    summary["zero_cmd_vel_published_count"] = zero_count
    write_trials_csv(trials)
    write_time_series_csv(time_series_rows)
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_trial_count"] == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Write diagnostic plan without publishing /cmd_vel.")
    mode.add_argument("--execute", action="store_true", help="Run trials and publish /cmd_vel.")
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--speeds", type=parse_speeds, default=parse_speeds("0.30,0.35,0.40"))
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-motion-sec", type=float, default=30.0)
    parser.add_argument("--cmd-publish-rate-hz", type=float, default=20.0)
    parser.add_argument("--odom-timeout-sec", type=float, default=3.0)
    parser.add_argument(
        "--time-series-decimation-sec",
        type=float,
        default=0.0,
        help="Minimum sim-time spacing between saved odom samples; 0 saves all samples.",
    )
    parser.add_argument("--yaw-drift-suspect-threshold-rad", type=float, default=0.08)
    parser.add_argument("--cross-track-suspect-threshold-m", type=float, default=0.15)
    parser.add_argument("--no-motion-yaw-suspect-threshold-rad", type=float, default=0.03)
    parser.add_argument("--distance-efficiency-min", type=float, default=0.70)
    args = parser.parse_args()
    if not args.execute:
        args.dry_run = True
    if args.odom_topic != DEFAULT_ODOM_TOPIC:
        raise SystemExit("Only %s is allowed for this diagnostic" % DEFAULT_ODOM_TOPIC)
    if args.duration_sec <= 0.0 or args.no_motion_sec <= 0.0:
        raise SystemExit("durations must be positive")
    if args.repeats <= 0:
        raise SystemExit("repeats must be positive")
    if args.time_series_decimation_sec < 0.0:
        raise SystemExit("time-series decimation must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    if args.execute:
        return run_execute(args)
    return dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
