#!/usr/bin/env python3
"""Restricted local subgoal runner MVP.

Default mode is dry-run. Execute mode is allowed only after live L3V safety,
gated odometry, target, and parameter checks pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "local_subgoal_runner_mvp"
REPORT = ROOT / "audit_reports" / "local_subgoal_runner_mvp_report.md"
TARGET_PATH = ROOT / "debug" / "short_horizon_target_selection" / "short_horizon_target_override.json"

TOPIC_ODOM = "/team/livox/icp_odom_gated"
TOPIC_GRID = "/team/local_traversability_grid"
TOPIC_STATUS = "/team/traversability_status"
TOPIC_CMD = "/cmd_vel"

FORBIDDEN_SOURCES: List[str] = []
ALIGN_TIERS = {
    1: {"angular_z": 0.60, "duration_sec": 5.00},
    2: {"angular_z": 0.80, "duration_sec": 5.00},
    3: {"angular_z": 1.00, "duration_sec": 5.00},
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"_read_error": str(exc)}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finite_xy(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(finite_number(v) for v in value)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid_bool:{value}")


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def init_rospy() -> Any:
    import rospy  # type: ignore

    if not rospy.core.is_initialized():
        rospy.init_node("local_subgoal_runner_mvp", anonymous=True, disable_signals=True)
    return rospy


def read_odom(rospy: Any, timeout: float = 3.0) -> Dict[str, Any]:
    from nav_msgs.msg import Odometry  # type: ignore

    try:
        msg = rospy.wait_for_message(TOPIC_ODOM, Odometry, timeout=timeout)
        pose = msg.pose.pose
        yaw = yaw_from_quat(pose.orientation)
        xy_yaw = [float(pose.position.x), float(pose.position.y), float(yaw)]
        finite = all(finite_number(v) for v in xy_yaw)
        return {
            "topic": TOPIC_ODOM,
            "message_received": True,
            "header_frame_id": msg.header.frame_id,
            "child_frame_id": msg.child_frame_id,
            "stamp_sec": float(msg.header.stamp.to_sec()),
            "pose_x_y_yaw": xy_yaw,
            "finite_pose": finite,
            "pass": bool(finite and msg.header.frame_id == "team_livox_odom" and msg.child_frame_id == "base"),
        }
    except Exception as exc:
        return {"topic": TOPIC_ODOM, "message_received": False, "pass": False, "error": str(exc)}


def grid_sample_is_valid(msg: Any) -> bool:
    if msg.header.frame_id != "base":
        return False
    if len(msg.data) == 0:
        return False
    return set(int(v) for v in msg.data).issubset({-1, 0, 100})


def grid_cell_ratios(msg: Any, x_min: float, x_max: float, y_min: float, y_max: float) -> Dict[str, Optional[float]]:
    width = int(msg.info.width)
    height = int(msg.info.height)
    resolution = float(msg.info.resolution)
    origin_y = float(msg.info.origin.position.y)
    cells: List[int] = []
    data = list(msg.data)
    for row in range(height):
        x = (row + 0.5) * resolution
        if x < x_min or x > x_max:
            continue
        for col in range(width):
            y = origin_y + (col + 0.5) * resolution
            if y_min <= y <= y_max:
                cells.append(int(data[row * width + col]))
    if not cells:
        return {
            "free_ratio": None,
            "occupied_ratio": None,
            "unknown_ratio": None,
            "cell_count": 0,
            "free_count": 0,
            "occupied_count": 0,
            "unknown_count": 0,
        }
    total = float(len(cells))
    free_count = cells.count(0)
    occupied_count = cells.count(100)
    unknown_count = cells.count(-1)
    return {
        "free_ratio": free_count / total,
        "occupied_ratio": occupied_count / total,
        "unknown_ratio": unknown_count / total,
        "cell_count": len(cells),
        "free_count": free_count,
        "occupied_count": occupied_count,
        "unknown_count": unknown_count,
    }


def grid_base_quality(latest: Any, grid_result: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    width = int(latest.info.width)
    height = int(latest.info.height)
    resolution = float(latest.info.resolution)
    data_len = len(latest.data)
    structure_pass = True
    if width != 60:
        structure_pass = False
        reasons.append(f"grid_width_not_60:{width}")
    if height != 60:
        structure_pass = False
        reasons.append(f"grid_height_not_60:{height}")
    if abs(resolution - 0.05) > 1e-6:
        structure_pass = False
        reasons.append(f"grid_resolution_not_0_05:{resolution}")
    if data_len != width * height:
        structure_pass = False
        reasons.append("grid_data_length_mismatch")
    front = grid_cell_ratios(latest, 0.0, 1.5, -0.6, 0.6)
    corridor = grid_cell_ratios(latest, 0.2, 1.2, -0.35, 0.35)
    front_sector_pass = bool(
        front["occupied_ratio"] is not None
        and front["occupied_ratio"] <= 0.10
        and front["unknown_ratio"] is not None
        and front["unknown_ratio"] <= 0.40
    )
    if not front_sector_pass:
        reasons.append("front_sector_not_clean")
    front_corridor_pass = bool(
        corridor["free_ratio"] is not None
        and corridor["free_ratio"] >= 0.70
        and corridor["occupied_ratio"] is not None
        and corridor["occupied_ratio"] <= 0.10
        and corridor["unknown_ratio"] is not None
        and corridor["unknown_ratio"] <= 0.30
    )
    if not front_corridor_pass:
        reasons.append("front_corridor_not_clean")
    return {
        "structure_pass": structure_pass,
        "grid_width": width,
        "grid_height": height,
        "grid_resolution_m": resolution,
        "grid_frame_id": latest.header.frame_id,
        "grid_free_count": list(latest.data).count(0),
        "grid_occupied_count": list(latest.data).count(100),
        "grid_unknown_count": list(latest.data).count(-1),
        "front_sector_pass": front_sector_pass,
        "front_corridor_pass": front_corridor_pass,
        "free_ratio": front["free_ratio"],
        "occupied_ratio": front["occupied_ratio"],
        "unknown_ratio": front["unknown_ratio"],
        "front_sector_free_count": front["free_count"],
        "front_sector_occupied_count": front["occupied_count"],
        "front_sector_unknown_count": front["unknown_count"],
        "front_corridor_free_ratio": corridor["free_ratio"],
        "front_corridor_occupied_ratio": corridor["occupied_ratio"],
        "front_corridor_unknown_ratio": corridor["unknown_ratio"],
        "front_sector_cell_count": front["cell_count"],
        "front_corridor_cell_count": corridor["cell_count"],
        "front_corridor_free_count": corridor["free_count"],
        "front_corridor_occupied_count": corridor["occupied_count"],
        "front_corridor_unknown_count": corridor["unknown_count"],
        "reasons": reasons,
    }


def collect_grid_samples(rospy: Any, timeout: float = 3.0, min_samples: int = 3, max_stamp_age_sec: float = 2.0) -> Dict[str, Any]:
    from nav_msgs.msg import OccupancyGrid  # type: ignore

    samples: List[Any] = []
    lock = threading.Lock()
    wall_start = time.monotonic()
    ros_start = float(rospy.Time.now().to_sec())

    def cb(msg: Any) -> None:
        with lock:
            samples.append(msg)

    sub = rospy.Subscriber(TOPIC_GRID, OccupancyGrid, cb, queue_size=10)
    try:
        while not rospy.is_shutdown() and time.monotonic() - wall_start < timeout:
            with lock:
                current = list(samples)
            valid = [msg for msg in current if grid_sample_is_valid(msg)]
            stamps = [float(msg.header.stamp.to_sec()) for msg in valid]
            latest_age = None
            if stamps:
                latest_age = max(0.0, float(rospy.Time.now().to_sec()) - stamps[-1])
            stamp_monotonic = len(stamps) >= 2 and all(stamps[i] <= stamps[i + 1] for i in range(len(stamps) - 1))
            stamp_changed = len(set(stamps)) > 1
            if len(current) >= min_samples and len(valid) >= min_samples and stamp_monotonic and stamp_changed and latest_age is not None and latest_age <= max_stamp_age_sec:
                break
            time.sleep(0.02)
        ros_end = float(rospy.Time.now().to_sec())
        with lock:
            final_samples = list(samples)
        return {
            "samples": final_samples,
            "collection_wall_duration_sec": time.monotonic() - wall_start,
            "collection_ros_start_sec": ros_start,
            "collection_ros_end_sec": ros_end,
            "message_count_total": len(final_samples),
            "sampler_mode": "subscriber_continuous",
        }
    finally:
        sub.unregister()


def audit_l3v_grid(rospy: Any, timeout: float = 3.0, max_stamp_age_sec: float = 2.0) -> Dict[str, Any]:
    try:
        collection = collect_grid_samples(rospy, timeout=timeout, min_samples=3, max_stamp_age_sec=max_stamp_age_sec)
        samples = collection["samples"]
    except Exception as exc:
        return {"topic": TOPIC_GRID, "sample_count": 0, "pass": False, "error": str(exc)}
    valid_samples = [sample for sample in samples if grid_sample_is_valid(sample)]
    stamps = [float(sample.header.stamp.to_sec()) for sample in valid_samples]
    seqs = [int(sample.header.seq) for sample in valid_samples]
    frame_ids = [sample.header.frame_id for sample in valid_samples]
    unique_values = sorted({int(v) for sample in valid_samples for v in sample.data})
    stamp_monotonic = len(stamps) >= 2 and all(stamps[i] <= stamps[i + 1] for i in range(len(stamps) - 1))
    seq_monotonic = len(seqs) >= 2 and all(seqs[i] <= seqs[i + 1] for i in range(len(seqs) - 1))
    stamp_changed = len(set(stamps)) > 1
    seq_changed = len(set(seqs)) > 1
    distinct_stamp_count = len(set(stamps))
    data_nonempty = bool(valid_samples) and all(len(sample.data) > 0 for sample in valid_samples)
    latest_grid_stamp_age_sec = None
    if stamps:
        latest_grid_stamp_age_sec = max(0.0, float(rospy.Time.now().to_sec()) - stamps[-1])
    freshness_pass = (
        len(samples) >= 3
        and len(valid_samples) >= 3
        and stamp_monotonic
        and stamp_changed
        and latest_grid_stamp_age_sec is not None
        and latest_grid_stamp_age_sec <= max_stamp_age_sec
    )
    schema_pass = bool(valid_samples) and set(unique_values).issubset({-1, 0, 100}) and all(frame == "base" for frame in frame_ids)
    base_quality: Dict[str, Any] = {"reasons": ["no_valid_grid_sample"]}
    if valid_samples:
        base_quality = grid_base_quality(valid_samples[-1], {})
    return {
        "topic": TOPIC_GRID,
        "sample_count": len(samples),
        "required_sample_count": 3,
        "valid_sample_count": len(valid_samples),
        "message_count_total": collection.get("message_count_total"),
        "distinct_stamp_count": distinct_stamp_count,
        "latest_grid_stamp_age_sec": latest_grid_stamp_age_sec,
        "max_grid_stamp_age_sec": max_stamp_age_sec,
        "collection_wall_duration_sec": collection.get("collection_wall_duration_sec"),
        "collection_ros_start_sec": collection.get("collection_ros_start_sec"),
        "collection_ros_end_sec": collection.get("collection_ros_end_sec"),
        "sampler_mode": collection.get("sampler_mode"),
        "weak_2_frame_evidence": bool(len(valid_samples) == 2 and stamp_monotonic and stamp_changed and seq_monotonic and seq_changed and schema_pass),
        "frame_ids": frame_ids,
        "stamp_values_sec": stamps,
        "seq_values": seqs,
        "unique_values": unique_values,
        "data_nonempty": data_nonempty,
        "stamp_monotonic": stamp_monotonic,
        "stamp_changed": stamp_changed,
        "seq_monotonic": seq_monotonic,
        "seq_changed": seq_changed,
        "grid_stamp_freshness_pass": bool(freshness_pass),
        "schema_pass": bool(schema_pass and data_nonempty),
        "base_quality": base_quality,
        "pass": bool(freshness_pass and schema_pass and data_nonempty),
    }


def evaluate_l3v_grid_quality(grid_result: Dict[str, Any], status_payload: Dict[str, Any]) -> Dict[str, Any]:
    base = grid_result.get("base_quality") or {}
    reasons = list(base.get("reasons", []))
    status = status_payload.get("local_traversability_status")
    input_freshness = status_payload.get("input_freshness") if isinstance(status_payload.get("input_freshness"), dict) else {}
    heartbeat = status_payload.get("heartbeat") if isinstance(status_payload.get("heartbeat"), dict) else {}
    grid_fresh = grid_result.get("pass") is True
    heartbeat_pass = bool(heartbeat.get("enabled") is True and finite_number(heartbeat.get("rate_hz")) and float(heartbeat.get("rate_hz")) >= 1.0)
    input_freshness_pass = input_freshness.get("all_required_inputs_fresh") is True
    if not grid_fresh:
        reasons.append("grid_freshness_not_pass")
    if not heartbeat_pass:
        reasons.append("heartbeat_not_confirmed")
    if not input_freshness_pass:
        reasons.append("input_freshness_not_pass")
    conflict_ratio = status_payload.get("odom_traversed_conflict_ratio")
    conflict_pass = finite_number(conflict_ratio) and float(conflict_ratio) <= 0.10
    if not conflict_pass:
        reasons.append("odom_traversed_conflict_ratio_high_or_missing")
    status_grid_consistency_pass = True
    if status == "FREE_SUPPORTED" and (
        base.get("front_corridor_occupied_ratio") is None
        or base.get("front_corridor_unknown_ratio") is None
        or float(base.get("front_corridor_occupied_ratio")) > 0.10
        or float(base.get("front_corridor_unknown_ratio")) > 0.30
    ):
        status_grid_consistency_pass = False
        reasons.append("free_status_but_grid_corridor_not_clean")
    if status not in {"FREE_SUPPORTED", "CONFLICT_NEEDS_CAUTION", "OBSTACLE_SUPPORTED", "UNKNOWN_INSUFFICIENT_EVIDENCE"}:
        status_grid_consistency_pass = False
        reasons.append("unrecognized_l3v_status")
    grid_quality_pass = bool(
        grid_fresh
        and base.get("structure_pass") is True
        and base.get("front_sector_pass") is True
        and base.get("front_corridor_pass") is True
        and status_grid_consistency_pass
        and heartbeat_pass
        and input_freshness_pass
        and conflict_pass
    )
    return {
        "grid_quality_pass": grid_quality_pass,
        "front_sector_pass": bool(base.get("front_sector_pass")),
        "front_corridor_pass": bool(base.get("front_corridor_pass")),
        "status_grid_consistency_pass": bool(status_grid_consistency_pass),
        "free_ratio": base.get("free_ratio"),
        "occupied_ratio": base.get("occupied_ratio"),
        "unknown_ratio": base.get("unknown_ratio"),
        "front_corridor_free_ratio": base.get("front_corridor_free_ratio"),
        "front_corridor_occupied_ratio": base.get("front_corridor_occupied_ratio"),
        "front_corridor_unknown_ratio": base.get("front_corridor_unknown_ratio"),
        "heartbeat_pass": heartbeat_pass,
        "input_freshness_pass": input_freshness_pass,
        "odom_traversed_conflict_ratio": conflict_ratio,
        "odom_traversed_cell_count": status_payload.get("odom_traversed_cell_count"),
        "odom_traversed_free_support_ratio": status_payload.get("odom_traversed_free_support_ratio"),
        "front_sector_cell_count": base.get("front_sector_cell_count"),
        "front_corridor_cell_count": base.get("front_corridor_cell_count"),
        "front_sector_free_count": base.get("front_sector_free_count"),
        "front_sector_occupied_count": base.get("front_sector_occupied_count"),
        "front_sector_unknown_count": base.get("front_sector_unknown_count"),
        "front_corridor_free_count": base.get("front_corridor_free_count"),
        "front_corridor_occupied_count": base.get("front_corridor_occupied_count"),
        "front_corridor_unknown_count": base.get("front_corridor_unknown_count"),
        "grid_width": base.get("grid_width"),
        "grid_height": base.get("grid_height"),
        "grid_resolution_m": base.get("grid_resolution_m"),
        "grid_frame_id": base.get("grid_frame_id"),
        "grid_free_count": base.get("grid_free_count"),
        "grid_occupied_count": base.get("grid_occupied_count"),
        "grid_unknown_count": base.get("grid_unknown_count"),
        "reasons": reasons,
    }


def read_status(rospy: Any, timeout: float = 3.0) -> Dict[str, Any]:
    from std_msgs.msg import String  # type: ignore

    try:
        msg = rospy.wait_for_message(TOPIC_STATUS, String, timeout=timeout)
        payload = json.loads(str(msg.data))
    except Exception as exc:
        return {"topic": TOPIC_STATUS, "message_received": False, "pass": False, "error": str(exc)}
    status_value = payload.get("local_traversability_status")
    boundary_checks = {
        "diagnostic_only_true": payload.get("diagnostic_only") is True,
        "safe_for_navigation_false": payload.get("safe_for_navigation") is False,
        "safe_for_frontier_false": payload.get("safe_for_frontier") is False,
        "autonomous_l4_allowed_false": payload.get("autonomous_l4_allowed") is False,
        "planner_ready_false": payload.get("planner_ready", False) is False,
        "published_cmd_vel_false": payload.get("published_cmd_vel", False) is False,
    }
    checks = {
        **boundary_checks,
        "local_traversability_status_recognized": status_value
        in {"FREE_SUPPORTED", "CONFLICT_NEEDS_CAUTION", "OBSTACLE_SUPPORTED", "UNKNOWN_INSUFFICIENT_EVIDENCE"},
    }
    return {
        "topic": TOPIC_STATUS,
        "message_received": True,
        "payload": payload,
        "status": status_value,
        "checks": checks,
        "boundary_pass": all(boundary_checks.values()),
        "pass": all(checks.values()),
    }


def classify_l3v_action_permission(
    status_payload: Dict[str, Any],
    grid_freshness_result: Dict[str, Any],
    grid_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = status_payload.get("local_traversability_status")
    grid_pass = grid_freshness_result.get("pass") is True
    quality = grid_quality or {}
    base = {
        "required_grid_sample_count": 3,
        "actual_grid_sample_count": grid_freshness_result.get("sample_count"),
        "valid_sample_count": grid_freshness_result.get("valid_sample_count"),
        "message_count_total": grid_freshness_result.get("message_count_total"),
        "distinct_stamp_count": grid_freshness_result.get("distinct_stamp_count"),
        "latest_grid_stamp_age_sec": grid_freshness_result.get("latest_grid_stamp_age_sec"),
        "sampler_mode": grid_freshness_result.get("sampler_mode"),
        "grid_freshness_pass": bool(grid_pass),
        "weak_2_frame_evidence": bool(grid_freshness_result.get("weak_2_frame_evidence")),
        "status": status,
        "allow_turn": False,
        "allow_forward": False,
        "must_stop": True,
        "reason": None,
        "severity": "blocked",
    }
    if not grid_pass:
        return {**base, "reason": "blocked_by_grid_freshness"}
    if status == "FREE_SUPPORTED":
        allow_forward = bool(quality.get("grid_quality_pass") and quality.get("front_corridor_pass"))
        reason = "free_supported" if allow_forward else "free_supported_turn_only_grid_quality_fail"
        return {
            **base,
            "allow_turn": True,
            "allow_forward": allow_forward,
            "must_stop": False,
            "reason": reason,
            "severity": "free" if allow_forward else "caution",
        }
    if status == "CONFLICT_NEEDS_CAUTION":
        return {
            **base,
            "allow_turn": True,
            "allow_forward": False,
            "must_stop": False,
            "reason": "conflict_caution_turn_only",
            "severity": "caution",
        }
    if status == "OBSTACLE_SUPPORTED":
        return {**base, "reason": "blocked_by_obstacle_supported"}
    if status == "UNKNOWN_INSUFFICIENT_EVIDENCE":
        return {**base, "reason": "blocked_by_unknown_insufficient_evidence"}
    return {**base, "reason": "blocked_by_unrecognized_l3v_status"}


def read_clock_diagnostic(rospy: Any, timeout: float = 1.0) -> Dict[str, Any]:
    try:
        use_sim_time = bool(rospy.get_param("/use_sim_time", False))
    except Exception:
        use_sim_time = False
    report: Dict[str, Any] = {
        "use_sim_time": use_sim_time,
        "clock_topic_checked": use_sim_time,
        "clock_message_received": None,
        "clock_stamp_sec": None,
    }
    if not use_sim_time:
        return report
    try:
        from rosgraph_msgs.msg import Clock  # type: ignore

        msg = rospy.wait_for_message("/clock", Clock, timeout=timeout)
        report["clock_message_received"] = True
        report["clock_stamp_sec"] = float(msg.clock.to_sec())
    except Exception as exc:
        report["clock_message_received"] = False
        report["error"] = str(exc)
    return report


def load_target() -> Dict[str, Any]:
    raw = read_json(TARGET_PATH)
    target_xy = raw.get("target_xy_team_livox_odom")
    target_frame = raw.get("target_frame")
    checks = {
        "file_exists": TARGET_PATH.exists(),
        "target_xy_valid": finite_xy(target_xy),
        "target_frame_team_livox_odom": target_frame == "team_livox_odom",
        "diagnostic_only_true": raw.get("diagnostic_only") is True,
        "safe_for_navigation_false": raw.get("safe_for_navigation") is False,
        "planner_ready_false": raw.get("planner_ready") is False,
        "send_to_navigation_false": raw.get("send_to_navigation") is False,
    }
    return {
        "path": str(TARGET_PATH),
        "raw": raw,
        "raw_target_x": float(target_xy[0]) if finite_xy(target_xy) else None,
        "raw_target_y": float(target_xy[1]) if finite_xy(target_xy) else None,
        "target_x": float(target_xy[0]) if finite_xy(target_xy) else None,
        "target_y": float(target_xy[1]) if finite_xy(target_xy) else None,
        "target_frame": target_frame,
        "checks": checks,
        "pass": all(checks.values()),
    }


def pose_tuple(odom: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    pose = odom.get("pose_x_y_yaw")
    if isinstance(pose, list) and len(pose) == 3 and all(finite_number(v) for v in pose):
        return float(pose[0]), float(pose[1]), float(pose[2])
    return None


def apply_local_goal_clamp(target: Dict[str, Any], odom: Dict[str, Any], radius_m: float) -> Dict[str, Any]:
    pose = pose_tuple(odom)
    if pose is None or not finite_number(target.get("raw_target_x")) or not finite_number(target.get("raw_target_y")):
        return target
    x, y, _ = pose
    raw_x = float(target["raw_target_x"])
    raw_y = float(target["raw_target_y"])
    dx = raw_x - x
    dy = raw_y - y
    raw_distance = math.hypot(dx, dy)
    clamped = raw_distance > float(radius_m)
    if clamped and raw_distance > 1e-9:
        scale = float(radius_m) / raw_distance
        effective_x = x + dx * scale
        effective_y = y + dy * scale
        effective_distance = float(radius_m)
    else:
        effective_x = raw_x
        effective_y = raw_y
        effective_distance = raw_distance
    updated = dict(target)
    updated.update(
        {
            "raw_target_distance_m": raw_distance,
            "effective_target_x": effective_x,
            "effective_target_y": effective_y,
            "effective_target_distance_m": effective_distance,
            "local_goal_radius_m": float(radius_m),
            "local_goal_clamped": bool(clamped),
            "target_x": effective_x,
            "target_y": effective_y,
        }
    )
    return updated


def compute_metrics(odom: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    pose = pose_tuple(odom)
    if pose is None or not finite_number(target.get("target_x")) or not finite_number(target.get("target_y")):
        return {"pass": False}
    x, y, yaw = pose
    dx = float(target["target_x"]) - x
    dy = float(target["target_y"]) - y
    distance = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    heading_error = normalize_angle(bearing - yaw)
    return {
        "pass": True,
        "distance_to_subgoal_m": distance,
        "target_bearing_rad": bearing,
        "heading_error_rad": heading_error,
    }


def evaluate_target_freshness(target: Dict[str, Any], initial_odom: Dict[str, Any]) -> Dict[str, Any]:
    raw = target.get("raw", {}) if isinstance(target.get("raw"), dict) else {}
    matched_pose = raw.get("matched_pose_x_y_yaw")
    current_pose = pose_tuple(initial_odom)
    target_matched_pose_available = (
        isinstance(matched_pose, list)
        and len(matched_pose) == 3
        and all(finite_number(v) for v in matched_pose)
    )
    result: Dict[str, Any] = {
        "target_freshness_check_enabled": True,
        "target_matched_pose_available": bool(target_matched_pose_available),
        "target_matched_pose_x": float(matched_pose[0]) if target_matched_pose_available else None,
        "target_matched_pose_y": float(matched_pose[1]) if target_matched_pose_available else None,
        "target_matched_pose_yaw": float(matched_pose[2]) if target_matched_pose_available else None,
        "current_initial_pose_x": current_pose[0] if current_pose else None,
        "current_initial_pose_y": current_pose[1] if current_pose else None,
        "current_initial_pose_yaw": current_pose[2] if current_pose else None,
        "target_pose_delta_xy_m": None,
        "target_pose_delta_yaw_rad": None,
        "target_age_ros_sec": None,
        "target_heading_error_at_generation_rad": raw.get("heading_error_rad_at_generation", raw.get("heading_error_rad")),
        "target_heading_error_at_execution_rad": None,
        "target_heading_error_delta_rad": None,
        "target_freshness_pass": False,
        "target_freshness_reason": None,
        "target_stale_pose_mismatch": False,
        "max_target_pose_delta_xy_m": 0.30,
        "max_target_pose_delta_yaw_rad": 0.20,
        "max_target_heading_error_delta_rad": 0.25,
        "max_target_age_ros_sec": 2.0,
    }
    if current_pose is None:
        result["target_freshness_reason"] = "current_initial_pose_unavailable"
        return result
    if not target_matched_pose_available:
        result["target_freshness_reason"] = "target_matched_pose_unavailable"
        result["target_stale_pose_mismatch"] = True
        return result

    matched_x, matched_y, matched_yaw = float(matched_pose[0]), float(matched_pose[1]), float(matched_pose[2])
    result["target_pose_delta_xy_m"] = math.hypot(current_pose[0] - matched_x, current_pose[1] - matched_y)
    result["target_pose_delta_yaw_rad"] = abs(normalize_angle(current_pose[2] - matched_yaw))

    matched_stamp = raw.get("matched_pose_stamp_sec")
    if not finite_number(matched_stamp):
        matched_stamp = raw.get("target_generated_ros_time_sec")
    if finite_number(matched_stamp) and finite_number(initial_odom.get("stamp_sec")):
        result["target_age_ros_sec"] = float(initial_odom["stamp_sec"]) - float(matched_stamp)

    execution_metrics = compute_metrics(initial_odom, target)
    if execution_metrics.get("pass"):
        result["target_heading_error_at_execution_rad"] = execution_metrics.get("heading_error_rad")
    if finite_number(result["target_heading_error_at_generation_rad"]) and finite_number(result["target_heading_error_at_execution_rad"]):
        result["target_heading_error_delta_rad"] = abs(
            normalize_angle(float(result["target_heading_error_at_execution_rad"]) - float(result["target_heading_error_at_generation_rad"]))
        )

    reasons: List[str] = []
    if finite_number(result["target_pose_delta_xy_m"]) and float(result["target_pose_delta_xy_m"]) > float(result["max_target_pose_delta_xy_m"]):
        reasons.append("target_pose_delta_xy_exceeds_threshold")
    if finite_number(result["target_pose_delta_yaw_rad"]) and float(result["target_pose_delta_yaw_rad"]) > float(result["max_target_pose_delta_yaw_rad"]):
        reasons.append("target_pose_delta_yaw_exceeds_threshold")
    if finite_number(result["target_heading_error_delta_rad"]) and float(result["target_heading_error_delta_rad"]) > float(result["max_target_heading_error_delta_rad"]):
        reasons.append("target_heading_error_delta_exceeds_threshold")
    if finite_number(result["target_age_ros_sec"]) and float(result["target_age_ros_sec"]) > float(result["max_target_age_ros_sec"]):
        reasons.append("target_age_ros_exceeds_threshold")

    result["target_stale_pose_mismatch"] = bool(reasons)
    result["target_freshness_pass"] = not reasons
    result["target_freshness_reason"] = "target_freshness_pass" if not reasons else ",".join(reasons)
    return result


def validate_params(args: argparse.Namespace) -> Dict[str, Any]:
    errors: List[str] = []
    if args.linear_x <= 0.0 or args.linear_x > 0.30:
        errors.append("linear_x_must_be_positive_and_not_exceed_0_30")
    if args.angular_z_max <= 0.0 or args.angular_z_max > 1.20:
        errors.append("angular_z_max_must_be_positive_and_not_exceed_1_20")
    if args.turn_slice_sec <= 0.0 or args.turn_slice_sec > 5.00:
        errors.append("turn_slice_sec_must_be_positive_and_not_exceed_5_00")
    if args.forward_slice_sec <= 0.0 or args.forward_slice_sec > 5.00:
        errors.append("forward_slice_sec_must_be_positive_and_not_exceed_5_00")
    if args.heading_threshold_rad <= 0.0:
        errors.append("heading_threshold_rad_must_be_positive")
    if args.max_steps <= 0 or args.max_steps > 120:
        errors.append("max_steps_must_be_positive_and_not_exceed_120")
    if args.max_runtime_sec <= 0.0 or args.max_runtime_sec > 300.0:
        errors.append("max_runtime_sec_must_be_positive_and_not_exceed_300")
    if args.no_progress_window <= 0:
        errors.append("no_progress_window_must_be_positive")
    if args.align_max_steps <= 0 or args.align_max_steps > 60:
        errors.append("align_max_steps_must_be_positive_and_not_exceed_60")
    if args.align_max_runtime_sec <= 0.0 or args.align_max_runtime_sec > 240.0:
        errors.append("align_max_runtime_sec_must_be_positive_and_not_exceed_240")
    if args.no_heading_progress_window <= 0:
        errors.append("no_heading_progress_window_must_be_positive")
    if args.min_heading_progress_rad <= 0.0:
        errors.append("min_heading_progress_rad_must_be_positive")
    if args.local_goal_radius_m <= 0.0:
        errors.append("local_goal_radius_m_must_be_positive")
    if args.motion_duration_timebase not in {"sim_time", "wall_time"}:
        errors.append("motion_duration_timebase_must_be_sim_time_or_wall_time")
    if args.grid_sample_timeout_sec <= 0.0:
        errors.append("grid_sample_timeout_sec_must_be_positive")
    if args.max_grid_stamp_age_sec <= 0.0:
        errors.append("max_grid_stamp_age_sec_must_be_positive")
    return {"pass": not errors, "errors": errors, "params": vars(args)}


def publish_zero(pub: Any, rospy: Any, count: int = 3) -> int:
    from geometry_msgs.msg import Twist  # type: ignore

    zero = Twist()
    published = 0
    for _ in range(count):
        pub.publish(zero)
        published += 1
        rospy.sleep(0.05)
    return published


def publish_slice(
    pub: Any,
    rospy: Any,
    linear_x: float,
    angular_z: float,
    duration_sec: float,
    publish_rate_hz: float = 20.0,
    motion_duration_timebase: str = "sim_time",
) -> Dict[str, Any]:
    from geometry_msgs.msg import Twist  # type: ignore

    cmd = Twist()
    cmd.linear.x = float(linear_x)
    cmd.angular.z = float(angular_z)
    nonzero = abs(cmd.linear.x) > 1e-9 or abs(cmd.angular.z) > 1e-9
    sleep_sec = 1.0 / max(float(publish_rate_hz), 1.0)
    wall_start = time.monotonic()
    ros_start = float(rospy.Time.now().to_sec())
    wall_timeout_sec = max(float(duration_sec) * 30.0, 60.0)
    count = 0
    history: List[Dict[str, Any]] = []
    warnings: List[str] = []
    while not rospy.is_shutdown():
        wall_elapsed = time.monotonic() - wall_start
        ros_now = float(rospy.Time.now().to_sec())
        sim_elapsed = max(0.0, ros_now - ros_start)
        elapsed = sim_elapsed if motion_duration_timebase == "sim_time" else wall_elapsed
        if elapsed >= float(duration_sec):
            break
        if motion_duration_timebase == "sim_time" and wall_elapsed >= wall_timeout_sec:
            warnings.append("sim_time_motion_slice_wall_timeout")
            break
        pub.publish(cmd)
        count += 1
        if len(history) < 3:
            history.append(
                {
                    "wall_t_sec": wall_elapsed,
                    "ros_t_sec": sim_elapsed,
                    "linear_x": cmd.linear.x,
                    "angular_z": cmd.angular.z,
                }
            )
        time.sleep(sleep_sec)
    wall_end = time.monotonic()
    ros_end = float(rospy.Time.now().to_sec())
    actual_wall = wall_end - wall_start
    actual_sim = max(0.0, ros_end - ros_start)
    observed_rtf = actual_sim / actual_wall if actual_wall > 1e-9 else None
    if count > 3:
        history.append({"wall_t_sec": actual_wall, "ros_t_sec": actual_sim, "linear_x": cmd.linear.x, "angular_z": cmd.angular.z})
    if count < int(float(duration_sec) * 10.0):
        warnings.append("low_cmd_publish_count")
    return {
        "nonzero_cmd_vel_published": nonzero and count > 0,
        "requested_duration_sec": float(duration_sec),
        "motion_duration_timebase": motion_duration_timebase,
        "ros_time_start_sec": ros_start,
        "ros_time_end_sec": ros_end,
        "actual_sim_duration_sec": actual_sim,
        "wall_time_start_sec": wall_start,
        "wall_time_end_sec": wall_end,
        "actual_wall_duration_sec": actual_wall,
        "observed_realtime_factor_during_slice": observed_rtf,
        "cmd_publish_count": count,
        "cmd_publish_rate_hz": publish_rate_hz,
        "actual_command_duration_sec": actual_wall,
        "commanded_twist_history": history,
        "warnings": warnings,
    }


def live_safety_check(rospy: Any, grid_timeout_sec: float = 3.0, max_grid_stamp_age_sec: float = 2.0) -> Dict[str, Any]:
    odom = read_odom(rospy, timeout=3.0)
    grid = audit_l3v_grid(rospy, timeout=grid_timeout_sec, max_stamp_age_sec=max_grid_stamp_age_sec)
    status = read_status(rospy, timeout=3.0)
    clock = read_clock_diagnostic(rospy, timeout=1.0)
    grid_quality = evaluate_l3v_grid_quality(grid, status.get("payload", {}))
    policy = classify_l3v_action_permission(status.get("payload", {}), grid, grid_quality)
    return {
        "odom": odom,
        "grid": grid,
        "status": status,
        "clock": clock,
        "grid_quality": grid_quality,
        "l3v_policy": policy,
        "pass": bool(odom.get("pass") and status.get("pass") and not policy.get("must_stop")),
    }


def empty_summary(mode: str) -> Dict[str, Any]:
    return {
        "stage": "LOCAL_SUBGOAL_RUNNER_MVP",
        "mode": mode,
        "final_decision": None,
        "initial_pose": {},
        "final_pose": {},
        "target": {},
        "initial_distance_to_subgoal_m": None,
        "final_distance_to_subgoal_m": None,
        "initial_distance_to_effective_goal_m": None,
        "final_distance_to_effective_goal_m": None,
        "distance_reduction_m": None,
        "distance_reduction_ratio": None,
        "initial_heading_error_rad": None,
        "final_heading_error_rad": None,
        "total_distance_delta_m": None,
        "any_forward_moved_toward_goal": False,
        "moved_toward_goal_step_count": 0,
        "motion_measurement": {
            "straight_line_displacement_m": None,
            "path_length_m": 0.0,
            "along_track_progress_m": None,
            "final_cross_track_error_m": None,
            "max_abs_cross_track_error_m": None,
            "net_distance_to_goal_reduction_m": None,
        },
        "final_pose_source": None,
        "last_accepted_pose_x": None,
        "last_accepted_pose_y": None,
        "last_accepted_pose_yaw": None,
        "last_observed_pose_x": None,
        "last_observed_pose_y": None,
        "last_observed_pose_yaw": None,
        "final_observed_straight_line_displacement_m": None,
        "final_observed_along_track_progress_m": None,
        "final_observed_cross_track_error_m": None,
        "final_observed_net_distance_to_goal_reduction_m": None,
        "final_accepted_straight_line_displacement_m": None,
        "final_accepted_along_track_progress_m": None,
        "final_accepted_cross_track_error_m": None,
        "final_accepted_net_distance_to_goal_reduction_m": None,
        "failure_step_index": None,
        "failure_reason": None,
        "failure_cross_track_error_m": None,
        "expected_forward_distance_m": None,
        "observed_step_displacement_m": None,
        "observed_path_length_m": None,
        "unexpected_xy_jump_excess_m": None,
        "xy_jump_margin_m": 0.5,
        "xy_jump_ratio_limit": 1.8,
        "xy_jump_policy_pass": None,
        "xy_jump_policy_reason": None,
        "whether_normal_commanded_forward_motion_classified_as_xy_jump": None,
        "cross_track_drift_m": None,
        "cross_track_drift_observed": False,
        "abs_cross_track_drift_m": None,
        "max_abs_cross_track_drift_m": None,
        "cross_track_drift_policy_source": None,
        "excessive_drift_policy_updated": True,
        "forward_bias_diagnostic_enabled": True,
        "forward_cmd_linear_x": None,
        "forward_cmd_angular_z": None,
        "forward_actual_sim_duration_sec": None,
        "forward_expected_distance_m": None,
        "forward_observed_displacement_m": None,
        "forward_distance_efficiency_ratio": None,
        "forward_yaw_before_rad": None,
        "forward_yaw_after_rad": None,
        "forward_yaw_delta_rad": None,
        "forward_yaw_drift_deg": None,
        "forward_cross_track_error_m": None,
        "forward_abs_cross_track_error_m": None,
        "forward_lateral_drift_direction": None,
        "forward_bias_suspected": False,
        "forward_bias_reason": None,
        "target_freshness_check_enabled": True,
        "target_matched_pose_available": None,
        "target_matched_pose_x": None,
        "target_matched_pose_y": None,
        "target_matched_pose_yaw": None,
        "current_initial_pose_x": None,
        "current_initial_pose_y": None,
        "current_initial_pose_yaw": None,
        "target_pose_delta_xy_m": None,
        "target_pose_delta_yaw_rad": None,
        "target_age_ros_sec": None,
        "target_heading_error_at_generation_rad": None,
        "target_heading_error_at_execution_rad": None,
        "target_heading_error_delta_rad": None,
        "target_stale_pose_mismatch": None,
        "target_freshness_pass": None,
        "target_freshness_reason": None,
        "max_target_pose_delta_xy_m": 0.30,
        "max_target_pose_delta_yaw_rad": 0.20,
        "max_target_heading_error_delta_rad": 0.25,
        "max_target_age_ros_sec": 2.0,
        "target_regenerated_by_wrapper": False,
        "target_regeneration_command": None,
        "target_regeneration_pass": None,
        "frame_contract_validation_rerun_by_wrapper": False,
        "frame_contract_validation_pass": None,
        "l3v_policy_initial_status": None,
        "l3v_policy_final_status": None,
        "l3v_policy_transition_observed": False,
        "grid_failure_diagnostic_enabled": True,
        "grid_failure_phase": None,
        "grid_failure_step_index": None,
        "grid_failure_l3v_status": None,
        "grid_failure_reason_list": [],
        "grid_failure_front_sector_pass": None,
        "grid_failure_front_corridor_pass": None,
        "grid_failure_grid_quality_pass": None,
        "grid_failure_occupied_ratio": None,
        "grid_failure_unknown_ratio": None,
        "grid_failure_free_ratio": None,
        "grid_failure_front_corridor_occupied_ratio": None,
        "grid_failure_front_corridor_unknown_ratio": None,
        "grid_failure_odom_traversed_conflict_ratio": None,
        "grid_failure_odom_traversed_cell_count": None,
        "grid_failure_odom_traversed_free_support_ratio": None,
        "grid_failure_pose_x_y_yaw": None,
        "grid_failure_heading_error_rad": None,
        "grid_failure_distance_to_subgoal_m": None,
        "initial_grid_occupied_ratio": None,
        "final_grid_occupied_ratio": None,
        "occupied_ratio_delta": None,
        "initial_front_corridor_occupied_ratio": None,
        "final_front_corridor_occupied_ratio": None,
        "front_corridor_occupied_ratio_delta": None,
        "initial_odom_traversed_conflict_ratio": None,
        "final_odom_traversed_conflict_ratio": None,
        "odom_traversed_conflict_ratio_delta": None,
        "grid_failure_likely_class": [],
        "grid_quality_failure_diagnostic_json_written": False,
        "motion_timing": {
            "motion_duration_timebase": None,
            "mean_observed_realtime_factor": None,
            "total_commanded_sim_duration_sec": 0.0,
            "total_commanded_wall_duration_sec": 0.0,
            "cmd_publish_count_total": 0,
            "slice_count": 0,
        },
        "step_count": 0,
        "turn_step_count": 0,
        "forward_step_count": 0,
        "zero_cmd_vel_published_count": 0,
        "nonzero_cmd_vel_published": False,
        "l3v_safety_fail_count": 0,
        "forbidden_sources_used": list(FORBIDDEN_SOURCES),
        "called_move_base": False,
        "sent_navigation_goal": False,
        "used_gazebo_truth": False,
        "safe_for_navigation": False,
        "planner_ready": False,
        "autonomous_l4_allowed": False,
        "steps": [],
        "warnings": [],
        "errors": [],
        "phase": "BLOCKED",
        "phase_history": [],
        "heading_progress": {
            "initial_abs_heading_error_rad": None,
            "final_abs_heading_error_rad": None,
            "heading_error_reduction_rad": None,
            "heading_error_reduction_ratio": None,
            "heading_progress_pass": False,
            "align_success": False,
            "align_progress_incomplete": False,
            "insufficient_yaw_response": False,
            "heading_diverged": False,
            "no_heading_progress_count": 0,
        },
        "align_profile": {
            "adaptive_enabled": True,
            "tier_history": [],
            "max_tier_reached": None,
            "align_max_steps": None,
            "align_max_runtime_sec": None,
            "min_heading_progress_rad": None,
        },
        "turn_response": {
            "total_expected_yaw_delta_rad": 0.0,
            "total_actual_yaw_delta_rad": 0.0,
            "mean_turn_efficiency": None,
            "min_turn_efficiency": None,
            "max_turn_efficiency": None,
            "low_response_step_count": 0,
            "cmd_publish_count_total": 0,
            "actual_command_duration_sec_total": 0.0,
            "heading_error_worsened_count": 0,
        },
        "distance_progress": {
            "initial_distance_to_subgoal_m": None,
            "final_distance_to_subgoal_m": None,
            "initial_distance_to_effective_goal_m": None,
            "final_distance_to_effective_goal_m": None,
            "distance_reduction_m": None,
            "distance_reduction_ratio": None,
            "distance_progress_pass": False,
            "no_distance_progress_count": 0,
        },
        "grid_quality": {
            "grid_quality_pass": None,
            "front_sector_pass": None,
            "front_corridor_pass": None,
            "status_grid_consistency_pass": None,
            "front_corridor_free_ratio": None,
            "front_corridor_occupied_ratio": None,
            "front_corridor_unknown_ratio": None,
            "reasons": [],
        },
        "motion_profile": {},
        "l3v_policy": {
            "required_grid_sample_count": 3,
            "actual_grid_sample_count": None,
            "valid_sample_count": None,
            "message_count_total": None,
            "distinct_stamp_count": None,
            "latest_grid_stamp_age_sec": None,
            "sampler_mode": None,
            "grid_freshness_pass": None,
            "weak_2_frame_evidence": False,
            "status": None,
            "allow_turn": None,
            "allow_forward": None,
            "must_stop": None,
            "reason": None,
            "severity": None,
        },
        "step_policy_counts": {
            "free_supported_steps": 0,
            "conflict_caution_steps": 0,
            "obstacle_blocked_steps": 0,
            "unknown_blocked_steps": 0,
            "turn_only_caution_steps": 0,
            "forward_allowed_steps": 0,
            "forward_blocked_by_caution_steps": 0,
            "blocked_steps": 0,
        },
    }


def decide_action(metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    heading_error = float(metrics["heading_error_rad"])
    if abs(heading_error) > args.heading_threshold_rad:
        if args.simple_motion_profile:
            return simple_turn_action(metrics, args)
        angular = math.copysign(min(args.angular_z_max, max(0.20, abs(heading_error) * 0.5)), heading_error)
        return {"action": "turn", "linear_x": 0.0, "angular_z": angular, "duration_sec": args.turn_slice_sec}
    return {"action": "forward", "linear_x": args.linear_x, "angular_z": 0.0, "duration_sec": args.forward_slice_sec}


def simple_turn_action(metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    heading_error = float(metrics["heading_error_rad"])
    angular = math.copysign(float(args.angular_z_max), heading_error if abs(heading_error) > 1e-9 else 1.0)
    return {"action": "turn", "linear_x": 0.0, "angular_z": angular, "duration_sec": args.turn_slice_sec}


def caution_turn_action(metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    heading_error = float(metrics["heading_error_rad"])
    angular = math.copysign(min(0.15, args.angular_z_max), heading_error if abs(heading_error) > 1e-6 else 1.0)
    return {"action": "turn", "linear_x": 0.0, "angular_z": angular, "duration_sec": min(args.turn_slice_sec, 0.50)}


def adaptive_turn_action(metrics: Dict[str, Any], tier: int) -> Dict[str, Any]:
    heading_error = float(metrics["heading_error_rad"])
    profile = ALIGN_TIERS.get(max(1, min(3, int(tier))), ALIGN_TIERS[3])
    return {
        "action": "turn",
        "linear_x": 0.0,
        "angular_z": math.copysign(profile["angular_z"], heading_error if abs(heading_error) > 1e-9 else 1.0),
        "duration_sec": profile["duration_sec"],
        "tier": max(1, min(3, int(tier))),
    }


def final_decision_for_l3v_block(policy: Dict[str, Any], execute: bool) -> str:
    if not execute:
        return "DRY_RUN_BLOCKED"
    if policy.get("reason") == "blocked_by_grid_freshness":
        return "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_L3V_FRESHNESS"
    if policy.get("reason") in {
        "blocked_by_obstacle_supported",
        "blocked_by_unknown_insufficient_evidence",
        "blocked_by_unrecognized_l3v_status",
    }:
        return "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_L3V_STATUS"
    return "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_L3V"


def update_policy_counts(summary: Dict[str, Any], policy: Dict[str, Any], planned_action: str) -> None:
    counts = summary["step_policy_counts"]
    status = policy.get("status")
    if status == "FREE_SUPPORTED":
        counts["free_supported_steps"] += 1
    elif status == "CONFLICT_NEEDS_CAUTION":
        counts["conflict_caution_steps"] += 1
        if policy.get("allow_turn") and not policy.get("allow_forward"):
            counts["turn_only_caution_steps"] += 1
        if planned_action == "forward" and not policy.get("allow_forward"):
            counts["forward_blocked_by_caution_steps"] += 1
    elif status == "OBSTACLE_SUPPORTED":
        counts["obstacle_blocked_steps"] += 1
    elif status == "UNKNOWN_INSUFFICIENT_EVIDENCE":
        counts["unknown_blocked_steps"] += 1
    if policy.get("allow_forward"):
        counts["forward_allowed_steps"] += 1
    if policy.get("must_stop"):
        counts["blocked_steps"] += 1


def update_progress(summary: Dict[str, Any]) -> None:
    initial = summary.get("initial_distance_to_subgoal_m")
    final = summary.get("final_distance_to_subgoal_m")
    if finite_number(initial) and finite_number(final):
        reduction = float(initial) - float(final)
        summary["distance_reduction_m"] = reduction
        summary["total_distance_delta_m"] = reduction
        summary["distance_reduction_ratio"] = reduction / float(initial) if float(initial) > 1e-9 else None
        summary["initial_distance_to_effective_goal_m"] = float(initial)
        summary["final_distance_to_effective_goal_m"] = float(final)
        summary["distance_progress"]["initial_distance_to_effective_goal_m"] = float(initial)
        summary["distance_progress"]["final_distance_to_effective_goal_m"] = float(final)


def compute_motion_measurement(
    start_pose: Optional[Tuple[float, float, float]],
    current_pose: Optional[Tuple[float, float, float]],
    target: Dict[str, Any],
    path_length_m: float,
    initial_distance_m: Any,
    final_distance_m: Any,
) -> Dict[str, Any]:
    result = {
        "straight_line_displacement_m": None,
        "path_length_m": path_length_m,
        "along_track_progress_m": None,
        "cross_track_error_m": None,
        "abs_cross_track_error_m": None,
        "final_cross_track_error_m": None,
        "max_abs_cross_track_error_m": None,
        "net_distance_to_goal_reduction_m": None,
    }
    if finite_number(initial_distance_m) and finite_number(final_distance_m):
        result["net_distance_to_goal_reduction_m"] = float(initial_distance_m) - float(final_distance_m)
    if start_pose is None or current_pose is None:
        return result
    if not finite_number(target.get("target_x")) or not finite_number(target.get("target_y")):
        return result
    sx, sy, _ = start_pose
    cx, cy, _ = current_pose
    gx = float(target["target_x"])
    gy = float(target["target_y"])
    vx = gx - sx
    vy = gy - sy
    goal_norm = math.hypot(vx, vy)
    dx = cx - sx
    dy = cy - sy
    result["straight_line_displacement_m"] = math.hypot(dx, dy)
    if goal_norm > 1e-9:
        ux = vx / goal_norm
        uy = vy / goal_norm
        along = dx * ux + dy * uy
        # Positive cross-track means current pose is left of the start->goal ray.
        cross = ux * dy - uy * dx
        result["along_track_progress_m"] = along
        result["cross_track_error_m"] = cross
        result["abs_cross_track_error_m"] = abs(cross)
        result["final_cross_track_error_m"] = cross
    return result


def pose_summary_fields(odom: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pose = pose_tuple(odom) if odom else None
    if pose is None:
        return {"x": None, "y": None, "yaw": None}
    return {"x": pose[0], "y": pose[1], "yaw": pose[2]}


def lateral_drift_direction(cross_track_error_m: Any) -> Optional[str]:
    if not finite_number(cross_track_error_m):
        return None
    cross = float(cross_track_error_m)
    if cross > 0.02:
        return "left_of_target_line"
    if cross < -0.02:
        return "right_of_target_line"
    return "near_centerline"


def evaluate_xy_jump_policy(
    action_name: str,
    observed_step_displacement_m: Any,
    expected_forward_distance_m: Any,
    max_xy_jump_m: float,
) -> Dict[str, Any]:
    result = {
        "xy_jump_policy_pass": True,
        "xy_jump_policy_reason": "not_evaluated",
        "unexpected_xy_jump_excess_m": None,
        "xy_jump_margin_m": 0.5,
        "xy_jump_ratio_limit": 1.8,
    }
    if not finite_number(observed_step_displacement_m):
        result["xy_jump_policy_pass"] = False
        result["xy_jump_policy_reason"] = "observed_step_displacement_unavailable"
        return result
    observed = float(observed_step_displacement_m)
    if action_name == "forward":
        if not finite_number(expected_forward_distance_m):
            result["xy_jump_policy_pass"] = False
            result["xy_jump_policy_reason"] = "expected_forward_distance_unavailable"
            return result
        expected = float(expected_forward_distance_m)
        result["unexpected_xy_jump_excess_m"] = observed - expected
        margin_limit = expected + float(result["xy_jump_margin_m"])
        ratio_limit = expected * float(result["xy_jump_ratio_limit"])
        if observed <= margin_limit or observed <= ratio_limit:
            result["xy_jump_policy_pass"] = True
            result["xy_jump_policy_reason"] = "observed_displacement_explained_by_forward_command"
        else:
            result["xy_jump_policy_pass"] = False
            result["xy_jump_policy_reason"] = "observed_displacement_exceeds_forward_command_envelope"
        return result
    result["unexpected_xy_jump_excess_m"] = observed
    if observed <= max_xy_jump_m:
        result["xy_jump_policy_pass"] = True
        result["xy_jump_policy_reason"] = "turn_step_within_max_xy_jump_m"
    else:
        result["xy_jump_policy_pass"] = False
        result["xy_jump_policy_reason"] = "turn_step_xy_jump_exceeded_max_xy_jump_m"
    return result


def number_delta(final_value: Any, initial_value: Any) -> Optional[float]:
    if finite_number(final_value) and finite_number(initial_value):
        return float(final_value) - float(initial_value)
    return None


def classify_grid_failure(summary: Dict[str, Any], initial_quality: Dict[str, Any], final_quality: Dict[str, Any]) -> List[str]:
    classes: List[str] = []
    front_corridor_delta = number_delta(
        final_quality.get("front_corridor_occupied_ratio"),
        initial_quality.get("front_corridor_occupied_ratio"),
    )
    occupied_delta = number_delta(final_quality.get("occupied_ratio"), initial_quality.get("occupied_ratio"))
    conflict_delta = number_delta(
        final_quality.get("odom_traversed_conflict_ratio"),
        initial_quality.get("odom_traversed_conflict_ratio"),
    )
    if summary.get("forward_bias_suspected") and finite_number(front_corridor_delta) and float(front_corridor_delta) > 0.02:
        classes.append("yaw_drift_shifted_front_corridor_possible")
    if (
        finite_number(initial_quality.get("odom_traversed_conflict_ratio"))
        and finite_number(final_quality.get("odom_traversed_conflict_ratio"))
        and float(initial_quality["odom_traversed_conflict_ratio"]) <= 0.05
        and float(final_quality["odom_traversed_conflict_ratio"]) >= 0.90
        and summary.get("any_forward_moved_toward_goal")
    ):
        classes.append("odom_traversed_conflict_policy_possible")
    elif finite_number(conflict_delta) and float(conflict_delta) > 0.20 and summary.get("any_forward_moved_toward_goal"):
        classes.append("odom_traversed_conflict_policy_possible")
    if finite_number(occupied_delta) and float(occupied_delta) > 0.05:
        classes.extend(["robot_self_or_footprint_contamination_possible", "sensor_projection_noise_possible"])
    if final_quality.get("front_sector_pass") is False or final_quality.get("front_corridor_pass") is False:
        classes.append("real_obstacle_or_wall_proximity_possible")
    if not classes:
        classes.append("unknown_requires_manual_review")
    return list(dict.fromkeys(classes))


def record_grid_failure_diagnostic(
    summary: Dict[str, Any],
    step_idx: int,
    phase: str,
    policy: Dict[str, Any],
    grid_quality: Dict[str, Any],
    initial_grid_quality: Dict[str, Any],
    current_odom: Dict[str, Any],
    metrics: Dict[str, Any],
    target: Dict[str, Any],
) -> None:
    pose = pose_tuple(current_odom)
    summary["grid_failure_diagnostic_enabled"] = True
    summary["grid_failure_phase"] = phase
    summary["grid_failure_step_index"] = step_idx
    summary["grid_failure_l3v_status"] = policy.get("status")
    summary["grid_failure_reason_list"] = list(grid_quality.get("reasons", []))
    summary["grid_failure_front_sector_pass"] = grid_quality.get("front_sector_pass")
    summary["grid_failure_front_corridor_pass"] = grid_quality.get("front_corridor_pass")
    summary["grid_failure_grid_quality_pass"] = grid_quality.get("grid_quality_pass")
    summary["grid_failure_occupied_ratio"] = grid_quality.get("occupied_ratio")
    summary["grid_failure_unknown_ratio"] = grid_quality.get("unknown_ratio")
    summary["grid_failure_free_ratio"] = grid_quality.get("free_ratio")
    summary["grid_failure_front_corridor_occupied_ratio"] = grid_quality.get("front_corridor_occupied_ratio")
    summary["grid_failure_front_corridor_unknown_ratio"] = grid_quality.get("front_corridor_unknown_ratio")
    summary["grid_failure_odom_traversed_conflict_ratio"] = grid_quality.get("odom_traversed_conflict_ratio")
    summary["grid_failure_odom_traversed_cell_count"] = grid_quality.get("odom_traversed_cell_count")
    summary["grid_failure_odom_traversed_free_support_ratio"] = grid_quality.get("odom_traversed_free_support_ratio")
    summary["grid_failure_pose_x_y_yaw"] = list(pose) if pose else None
    summary["grid_failure_heading_error_rad"] = metrics.get("heading_error_rad")
    summary["grid_failure_distance_to_subgoal_m"] = metrics.get("distance_to_subgoal_m")
    summary["initial_grid_occupied_ratio"] = initial_grid_quality.get("occupied_ratio")
    summary["final_grid_occupied_ratio"] = grid_quality.get("occupied_ratio")
    summary["occupied_ratio_delta"] = number_delta(grid_quality.get("occupied_ratio"), initial_grid_quality.get("occupied_ratio"))
    summary["initial_front_corridor_occupied_ratio"] = initial_grid_quality.get("front_corridor_occupied_ratio")
    summary["final_front_corridor_occupied_ratio"] = grid_quality.get("front_corridor_occupied_ratio")
    summary["front_corridor_occupied_ratio_delta"] = number_delta(
        grid_quality.get("front_corridor_occupied_ratio"),
        initial_grid_quality.get("front_corridor_occupied_ratio"),
    )
    summary["initial_odom_traversed_conflict_ratio"] = initial_grid_quality.get("odom_traversed_conflict_ratio")
    summary["final_odom_traversed_conflict_ratio"] = grid_quality.get("odom_traversed_conflict_ratio")
    summary["odom_traversed_conflict_ratio_delta"] = number_delta(
        grid_quality.get("odom_traversed_conflict_ratio"),
        initial_grid_quality.get("odom_traversed_conflict_ratio"),
    )
    summary["grid_failure_likely_class"] = classify_grid_failure(summary, initial_grid_quality, grid_quality)

    artifact = {
        "grid_dimensions": {"width": grid_quality.get("grid_width"), "height": grid_quality.get("grid_height")},
        "resolution": grid_quality.get("grid_resolution_m"),
        "frame_id": grid_quality.get("grid_frame_id"),
        "counts": {
            "free": grid_quality.get("grid_free_count"),
            "occupied": grid_quality.get("grid_occupied_count"),
            "unknown": grid_quality.get("grid_unknown_count"),
        },
        "front_sector": {
            "cell_count": grid_quality.get("front_sector_cell_count"),
            "free_count": grid_quality.get("front_sector_free_count"),
            "occupied_count": grid_quality.get("front_sector_occupied_count"),
            "unknown_count": grid_quality.get("front_sector_unknown_count"),
            "free_ratio": grid_quality.get("free_ratio"),
            "occupied_ratio": grid_quality.get("occupied_ratio"),
            "unknown_ratio": grid_quality.get("unknown_ratio"),
            "pass": grid_quality.get("front_sector_pass"),
        },
        "front_corridor": {
            "cell_count": grid_quality.get("front_corridor_cell_count"),
            "free_count": grid_quality.get("front_corridor_free_count"),
            "occupied_count": grid_quality.get("front_corridor_occupied_count"),
            "unknown_count": grid_quality.get("front_corridor_unknown_count"),
            "free_ratio": grid_quality.get("front_corridor_free_ratio"),
            "occupied_ratio": grid_quality.get("front_corridor_occupied_ratio"),
            "unknown_ratio": grid_quality.get("front_corridor_unknown_ratio"),
            "pass": grid_quality.get("front_corridor_pass"),
        },
        "odom_traversed": {
            "cell_count": grid_quality.get("odom_traversed_cell_count"),
            "conflict_ratio": grid_quality.get("odom_traversed_conflict_ratio"),
            "free_support_ratio": grid_quality.get("odom_traversed_free_support_ratio"),
        },
        "top_failure_reasons": list(grid_quality.get("reasons", [])),
        "current_pose": list(pose) if pose else None,
        "target": target,
        "forward_bias": {
            "forward_bias_suspected": summary.get("forward_bias_suspected"),
            "forward_bias_reason": summary.get("forward_bias_reason"),
            "forward_yaw_delta_rad": summary.get("forward_yaw_delta_rad"),
            "forward_yaw_drift_deg": summary.get("forward_yaw_drift_deg"),
            "forward_cross_track_error_m": summary.get("forward_cross_track_error_m"),
            "forward_lateral_drift_direction": summary.get("forward_lateral_drift_direction"),
        },
        "l3v_status": {
            "initial": summary.get("l3v_policy_initial_status"),
            "final": summary.get("l3v_policy_final_status"),
            "transition_observed": summary.get("l3v_policy_transition_observed"),
        },
        "likely_class": summary.get("grid_failure_likely_class"),
    }
    write_json(OUT / "grid_quality_failure_diagnostic.json", artifact)
    summary["grid_quality_failure_diagnostic_json_written"] = True


def write_report(summary: Dict[str, Any]) -> None:
    policy = summary.get("l3v_policy", {})
    gate = summary.get("initial_gate", {})
    grid = gate.get("grid", {})
    counts = summary.get("step_policy_counts", {})
    grid_quality = summary.get("grid_quality", {})
    heading_progress = summary.get("heading_progress", {})
    distance_progress = summary.get("distance_progress", {})
    align_profile = summary.get("align_profile", {})
    turn_response = summary.get("turn_response", {})
    motion_profile = summary.get("motion_profile", {})
    motion_measurement = summary.get("motion_measurement", {})
    motion_timing = summary.get("motion_timing", {})
    forward_steps = [step for step in summary.get("steps", []) if step.get("executed_action") == "forward"]
    last_forward_cmd = forward_steps[-1].get("cmd") if forward_steps else {}
    tier_history = align_profile.get("tier_history") or []
    tier_counts = {tier: tier_history.count(tier) for tier in [1, 2, 3]}
    lines = [
        "# Local Subgoal Runner MVP Report",
        "",
        f"- final_decision: `{summary.get('final_decision')}`",
        f"- phase: `{summary.get('phase')}`",
        f"- mode: `{summary.get('mode')}`",
        f"- execute: `{summary.get('mode') == 'execute'}`",
        f"- simple_motion_profile: `{motion_profile.get('simple_motion_profile')}`",
        f"- heading_threshold_rad: `{motion_profile.get('heading_threshold_rad')}`",
        f"- motion_duration_timebase: `{motion_timing.get('motion_duration_timebase', motion_profile.get('motion_duration_timebase'))}`",
        f"- turn_command: `{motion_profile.get('turn_command')}`",
        f"- forward_command: `{motion_profile.get('forward_command')}`",
        f"- raw_target_distance_m: `{summary.get('target', {}).get('raw_target_distance_m')}`",
        f"- effective_target_distance_m: `{summary.get('target', {}).get('effective_target_distance_m')}`",
        f"- local_goal_radius_m: `{summary.get('target', {}).get('local_goal_radius_m')}`",
        f"- local_goal_clamped: `{summary.get('target', {}).get('local_goal_clamped')}`",
        f"- initial_distance_to_subgoal_m: `{summary.get('initial_distance_to_subgoal_m')}`",
        f"- final_distance_to_subgoal_m: `{summary.get('final_distance_to_subgoal_m')}`",
        f"- initial_distance_to_effective_goal_m: `{summary.get('initial_distance_to_effective_goal_m')}`",
        f"- final_distance_to_effective_goal_m: `{summary.get('final_distance_to_effective_goal_m')}`",
        f"- total_distance_delta_m: `{summary.get('total_distance_delta_m')}`",
        f"- any_forward_moved_toward_goal: `{summary.get('any_forward_moved_toward_goal')}`",
        f"- moved_toward_goal_step_count: `{summary.get('moved_toward_goal_step_count')}`",
        f"- straight_line_displacement_m: `{motion_measurement.get('straight_line_displacement_m')}`",
        f"- path_length_m: `{motion_measurement.get('path_length_m')}`",
        f"- along_track_progress_m: `{motion_measurement.get('along_track_progress_m')}`",
        f"- final_cross_track_error_m: `{motion_measurement.get('final_cross_track_error_m')}`",
        f"- max_abs_cross_track_error_m: `{motion_measurement.get('max_abs_cross_track_error_m')}`",
        f"- net_distance_to_goal_reduction_m: `{motion_measurement.get('net_distance_to_goal_reduction_m')}`",
        f"- final_pose_source: `{summary.get('final_pose_source')}`",
        f"- last_accepted_pose_x: `{summary.get('last_accepted_pose_x')}`",
        f"- last_accepted_pose_y: `{summary.get('last_accepted_pose_y')}`",
        f"- last_accepted_pose_yaw: `{summary.get('last_accepted_pose_yaw')}`",
        f"- last_observed_pose_x: `{summary.get('last_observed_pose_x')}`",
        f"- last_observed_pose_y: `{summary.get('last_observed_pose_y')}`",
        f"- last_observed_pose_yaw: `{summary.get('last_observed_pose_yaw')}`",
        f"- final_observed_net_distance_to_goal_reduction_m: `{summary.get('final_observed_net_distance_to_goal_reduction_m')}`",
        f"- final_accepted_net_distance_to_goal_reduction_m: `{summary.get('final_accepted_net_distance_to_goal_reduction_m')}`",
        f"- failure_step_index: `{summary.get('failure_step_index')}`",
        f"- failure_reason: `{summary.get('failure_reason')}`",
        f"- failure_cross_track_error_m: `{summary.get('failure_cross_track_error_m')}`",
        f"- distance_reduction_ratio: `{summary.get('distance_reduction_ratio')}`",
        f"- initial_heading_error_rad: `{summary.get('initial_heading_error_rad')}`",
        f"- final_heading_error_rad: `{summary.get('final_heading_error_rad')}`",
        f"- heading_error_reduction_rad: `{heading_progress.get('heading_error_reduction_rad')}`",
        f"- heading_error_reduction_ratio: `{heading_progress.get('heading_error_reduction_ratio')}`",
        f"- heading_progress_pass: `{heading_progress.get('heading_progress_pass')}`",
        f"- align_success: `{heading_progress.get('align_success')}`",
        f"- align_progress_incomplete: `{heading_progress.get('align_progress_incomplete')}`",
        f"- insufficient_yaw_response: `{heading_progress.get('insufficient_yaw_response')}`",
        f"- heading_diverged: `{heading_progress.get('heading_diverged')}`",
        f"- distance_reduction_m: `{distance_progress.get('distance_reduction_m')}`",
        f"- distance_progress_pass: `{distance_progress.get('distance_progress_pass')}`",
        f"- step_count: `{summary.get('step_count')}`",
        f"- turn_step_count: `{summary.get('turn_step_count')}`",
        f"- forward_step_count: `{summary.get('forward_step_count')}`",
        f"- nonzero_cmd_vel_published: `{summary.get('nonzero_cmd_vel_published')}`",
        f"- zero_cmd_vel_published_count: `{summary.get('zero_cmd_vel_published_count')}`",
        f"- l3v_safety_fail_count: `{summary.get('l3v_safety_fail_count')}`",
        f"- entered_approach_phase: `{'APPROACH_PHASE' in summary.get('phase_history', [])}`",
        f"- forward_slice_executed: `{bool(forward_steps)}`",
        f"- forward_linear_x: `{last_forward_cmd.get('linear_x')}`",
        f"- forward_duration_sec: `{last_forward_cmd.get('duration_sec')}`",
        f"- linear_x: `{motion_profile.get('linear_x')}`",
        f"- forward_slice_sec: `{motion_profile.get('forward_slice_sec')}`",
        f"- motion_profile: `{motion_profile}`",
        f"- tier_counts: `{tier_counts}`",
        f"- max_tier_reached: `{align_profile.get('max_tier_reached')}`",
        f"- total_expected_yaw_delta_rad: `{turn_response.get('total_expected_yaw_delta_rad')}`",
        f"- total_actual_yaw_delta_rad: `{turn_response.get('total_actual_yaw_delta_rad')}`",
        f"- mean_turn_efficiency: `{turn_response.get('mean_turn_efficiency')}`",
        f"- min_turn_efficiency: `{turn_response.get('min_turn_efficiency')}`",
        f"- max_turn_efficiency: `{turn_response.get('max_turn_efficiency')}`",
        f"- cmd_publish_count_total: `{turn_response.get('cmd_publish_count_total')}`",
        f"- actual_command_duration_sec_total: `{turn_response.get('actual_command_duration_sec_total')}`",
        f"- mean_observed_realtime_factor: `{motion_timing.get('mean_observed_realtime_factor')}`",
        f"- total_commanded_sim_duration_sec: `{motion_timing.get('total_commanded_sim_duration_sec')}`",
        f"- total_commanded_wall_duration_sec: `{motion_timing.get('total_commanded_wall_duration_sec')}`",
        f"- motion_timing_cmd_publish_count_total: `{motion_timing.get('cmd_publish_count_total')}`",
        f"- heading_error_worsened_count: `{turn_response.get('heading_error_worsened_count')}`",
        "",
        "## L3V Grid Freshness",
        "",
        f"- required_sample_count: `{grid.get('required_sample_count', policy.get('required_grid_sample_count'))}`",
        f"- actual_sample_count: `{grid.get('sample_count', policy.get('actual_grid_sample_count'))}`",
        f"- valid_sample_count: `{grid.get('valid_sample_count', policy.get('valid_sample_count'))}`",
        f"- message_count_total: `{grid.get('message_count_total', policy.get('message_count_total'))}`",
        f"- distinct_stamp_count: `{grid.get('distinct_stamp_count', policy.get('distinct_stamp_count'))}`",
        f"- latest_grid_stamp_age_sec: `{grid.get('latest_grid_stamp_age_sec', policy.get('latest_grid_stamp_age_sec'))}`",
        f"- sampler_mode: `{grid.get('sampler_mode', policy.get('sampler_mode'))}`",
        f"- collection_wall_duration_sec: `{grid.get('collection_wall_duration_sec')}`",
        f"- collection_ros_start_sec: `{grid.get('collection_ros_start_sec')}`",
        f"- collection_ros_end_sec: `{grid.get('collection_ros_end_sec')}`",
        f"- stamp_values: `{grid.get('stamp_values_sec')}`",
        f"- seq_values: `{grid.get('seq_values')}`",
        f"- freshness_pass: `{grid.get('grid_stamp_freshness_pass', policy.get('grid_freshness_pass'))}`",
        f"- weak_2_frame_evidence: `{grid.get('weak_2_frame_evidence', policy.get('weak_2_frame_evidence'))}`",
        "",
        "## L3V Grid Quality",
        "",
        f"- grid_quality_pass: `{grid_quality.get('grid_quality_pass')}`",
        f"- front_sector_pass: `{grid_quality.get('front_sector_pass')}`",
        f"- front_corridor_pass: `{grid_quality.get('front_corridor_pass')}`",
        f"- status_grid_consistency_pass: `{grid_quality.get('status_grid_consistency_pass')}`",
        f"- front_corridor_free_ratio: `{grid_quality.get('front_corridor_free_ratio')}`",
        f"- front_corridor_occupied_ratio: `{grid_quality.get('front_corridor_occupied_ratio')}`",
        f"- front_corridor_unknown_ratio: `{grid_quality.get('front_corridor_unknown_ratio')}`",
        f"- grid_quality_reasons: `{grid_quality.get('reasons')}`",
        "",
        "## L3V Action-tier Policy",
        "",
        f"- status: `{policy.get('status')}`",
        f"- allow_turn: `{policy.get('allow_turn')}`",
        f"- allow_forward: `{policy.get('allow_forward')}`",
        f"- must_stop: `{policy.get('must_stop')}`",
        f"- reason: `{policy.get('reason')}`",
        f"- severity: `{policy.get('severity')}`",
        "",
        "## Action Counts",
        "",
        f"- turn steps: `{summary.get('turn_step_count')}`",
        f"- forward steps: `{summary.get('forward_step_count')}`",
        f"- caution turn-only steps: `{counts.get('turn_only_caution_steps')}`",
        f"- forward blocked by caution steps: `{counts.get('forward_blocked_by_caution_steps')}`",
        f"- blocked steps: `{counts.get('blocked_steps')}`",
        "",
        "## Boundary",
        "",
        "- Pose source: `/team/livox/icp_odom_gated`.",
        "- Safety sources: `/team/local_traversability_grid`, `/team/traversability_status`.",
        "- Target source: `debug/short_horizon_target_selection/short_horizon_target_override.json`.",
        f"- forbidden_sources_used: `{summary.get('forbidden_sources_used')}`",
        "- called_move_base=false",
        "- sent_navigation_goal=false",
        "- used_gazebo_truth=false",
        "- safe_for_navigation=false",
        "- planner_ready=false",
        "- autonomous_l4_allowed=false",
    ]
    if summary.get("errors"):
        lines += ["", "## Errors", ""]
        lines += [f"- `{err}`" for err in summary["errors"]]
    if summary.get("warnings"):
        lines += ["", "## Warnings", ""]
        lines += [f"- `{warning}`" for warning in summary["warnings"]]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    mode = "execute" if args.execute else "dry_run"
    summary = empty_summary(mode)
    params = validate_params(args)
    target = load_target()
    summary["target"] = target
    summary["target_regenerated_by_wrapper"] = os.environ.get("LSR_TARGET_REGENERATED_BY_WRAPPER") == "1"
    summary["target_regeneration_command"] = os.environ.get("LSR_TARGET_REGENERATION_COMMAND")
    if os.environ.get("LSR_TARGET_REGENERATION_PASS") is not None:
        summary["target_regeneration_pass"] = os.environ.get("LSR_TARGET_REGENERATION_PASS") == "1"
    summary["frame_contract_validation_rerun_by_wrapper"] = os.environ.get("LSR_FRAME_CONTRACT_VALIDATION_RERUN_BY_WRAPPER") == "1"
    if os.environ.get("LSR_FRAME_CONTRACT_VALIDATION_PASS") is not None:
        summary["frame_contract_validation_pass"] = os.environ.get("LSR_FRAME_CONTRACT_VALIDATION_PASS") == "1"
    summary["motion_profile"] = {
        "simple_motion_profile": args.simple_motion_profile,
        "heading_threshold_rad": args.heading_threshold_rad,
        "motion_duration_timebase": args.motion_duration_timebase,
        "angular_z_max": args.angular_z_max,
        "turn_slice_sec": args.turn_slice_sec,
        "linear_x": args.linear_x,
        "forward_slice_sec": args.forward_slice_sec,
        "local_goal_radius_m": args.local_goal_radius_m,
        "turn_command": {"linear_x": 0.0, "angular_z": args.angular_z_max, "duration_sec": args.turn_slice_sec},
        "forward_command": {"linear_x": args.linear_x, "angular_z": 0.0, "duration_sec": args.forward_slice_sec},
    }
    summary["motion_timing"]["motion_duration_timebase"] = args.motion_duration_timebase
    summary["align_profile"].update(
        {
            "align_max_steps": args.align_max_steps,
            "align_max_runtime_sec": args.align_max_runtime_sec,
            "min_heading_progress_rad": args.min_heading_progress_rad,
        }
    )

    if not params["pass"]:
        summary["errors"].extend(params["errors"])
        summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_PARAMS" if args.execute else "DRY_RUN_BLOCKED"
        return summary
    if not target["pass"]:
        summary["errors"].append("target_override_invalid_or_unavailable")
        summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_FAILED_BOUNDARY_VIOLATION" if args.execute else "DRY_RUN_BLOCKED"
        return summary

    rospy = init_rospy()
    initial_gate = live_safety_check(
        rospy,
        grid_timeout_sec=args.grid_sample_timeout_sec,
        max_grid_stamp_age_sec=args.max_grid_stamp_age_sec,
    )
    if not initial_gate["pass"]:
        policy = initial_gate.get("l3v_policy", {})
        summary["l3v_policy"] = policy
        summary["l3v_policy_initial_status"] = policy.get("status")
        summary["l3v_policy_final_status"] = policy.get("status")
        summary["grid_quality"] = initial_gate.get("grid_quality", summary["grid_quality"])
        summary["l3v_safety_fail_count"] += 1
        summary["errors"].append("initial_live_safety_gate_failed")
        summary["initial_gate"] = initial_gate
        if initial_gate["odom"].get("pass"):
            target = apply_local_goal_clamp(target, initial_gate["odom"], args.local_goal_radius_m)
            summary["target"] = target
            initial_metrics = compute_metrics(initial_gate["odom"], target)
            summary["initial_pose"] = initial_gate["odom"]
            summary["final_pose"] = initial_gate["odom"]
            if initial_metrics.get("pass"):
                summary["initial_distance_to_subgoal_m"] = initial_metrics["distance_to_subgoal_m"]
                summary["final_distance_to_subgoal_m"] = initial_metrics["distance_to_subgoal_m"]
                summary["initial_heading_error_rad"] = initial_metrics["heading_error_rad"]
                summary["final_heading_error_rad"] = initial_metrics["heading_error_rad"]
                summary["heading_progress"]["initial_abs_heading_error_rad"] = abs(float(initial_metrics["heading_error_rad"]))
                summary["heading_progress"]["final_abs_heading_error_rad"] = abs(float(initial_metrics["heading_error_rad"]))
                summary["heading_progress"]["heading_error_reduction_rad"] = 0.0
                summary["heading_progress"]["heading_error_reduction_ratio"] = 0.0
                summary["distance_progress"]["initial_distance_to_subgoal_m"] = initial_metrics["distance_to_subgoal_m"]
                summary["distance_progress"]["final_distance_to_subgoal_m"] = initial_metrics["distance_to_subgoal_m"]
                summary["distance_progress"]["distance_reduction_m"] = 0.0
                summary["distance_progress"]["distance_reduction_ratio"] = 0.0
                update_progress(summary)
        if args.execute and not initial_gate["odom"].get("pass"):
            summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_ODOM"
        else:
            summary["final_decision"] = final_decision_for_l3v_block(policy, args.execute)
        if args.execute:
            from geometry_msgs.msg import Twist  # type: ignore

            pub = rospy.Publisher(TOPIC_CMD, Twist, queue_size=1)
            rospy.sleep(0.2)
            update_policy_counts(summary, policy, "stop")
            summary["zero_cmd_vel_published_count"] += publish_zero(pub, rospy, count=3)
            summary["steps"].append(
                {
                    "step_index": 0,
                    "distance_to_subgoal_m": summary.get("initial_distance_to_subgoal_m"),
                    "heading_error_rad": summary.get("initial_heading_error_rad"),
                    "l3v_status": policy.get("status"),
                    "grid_freshness_pass": policy.get("grid_freshness_pass"),
                    "allow_turn": policy.get("allow_turn"),
                    "allow_forward": policy.get("allow_forward"),
                    "planned_action": "stop",
                    "executed_action": "zero_stop",
                    "cmd": {"linear_x": 0.0, "angular_z": 0.0, "duration_sec": 0.0},
                    "zero_cmd_vel_published_count": 3,
                    "reason": policy.get("reason"),
                }
            )
        return summary

    initial_odom = initial_gate["odom"]
    initial_policy = initial_gate.get("l3v_policy", summary["l3v_policy"])
    summary["l3v_policy_initial_status"] = initial_policy.get("status")
    summary["l3v_policy_final_status"] = initial_policy.get("status")
    target_freshness = evaluate_target_freshness(target, initial_odom)
    summary.update(target_freshness)
    if not target_freshness.get("target_freshness_pass"):
        summary["initial_gate"] = initial_gate
        summary["l3v_policy"] = initial_policy
        summary["grid_quality"] = initial_gate.get("grid_quality", summary["grid_quality"])
        summary["initial_pose"] = initial_odom
        summary["errors"].append("target_stale_pose_mismatch")
        summary["final_decision"] = (
            "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_TARGET_STALE_POSE_MISMATCH"
            if args.execute
            else "DRY_RUN_BLOCKED_BY_TARGET_STALE_POSE_MISMATCH"
        )
        return summary
    target = apply_local_goal_clamp(target, initial_odom, args.local_goal_radius_m)
    summary["target"] = target
    summary["initial_gate"] = initial_gate
    summary["l3v_policy"] = initial_gate.get("l3v_policy", summary["l3v_policy"])
    initial_metrics = compute_metrics(initial_odom, target)
    if not initial_metrics["pass"]:
        summary["errors"].append("initial_distance_or_heading_unavailable")
        summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_ODOM" if args.execute else "DRY_RUN_BLOCKED"
        return summary
    summary["initial_pose"] = initial_odom
    summary["initial_distance_to_subgoal_m"] = initial_metrics["distance_to_subgoal_m"]
    summary["initial_heading_error_rad"] = initial_metrics["heading_error_rad"]
    summary["heading_progress"]["initial_abs_heading_error_rad"] = abs(float(initial_metrics["heading_error_rad"]))
    summary["distance_progress"]["initial_distance_to_subgoal_m"] = initial_metrics["distance_to_subgoal_m"]
    summary["grid_quality"] = initial_gate.get("grid_quality", summary["grid_quality"])
    initial_grid_quality = dict(summary["grid_quality"])

    if not args.execute:
        policy = initial_gate.get("l3v_policy", {})
        summary["phase"] = "ALIGN_PHASE" if abs(float(initial_metrics["heading_error_rad"])) > args.heading_threshold_rad else "APPROACH_PHASE"
        summary["phase_history"].append(summary["phase"])
        planned = decide_action(initial_metrics, args)
        if planned["action"] == "forward" and not policy.get("allow_forward"):
            if policy.get("allow_turn") and policy.get("status") == "CONFLICT_NEEDS_CAUTION":
                planned = caution_turn_action(initial_metrics, args)
                planned["reason"] = "conflict_caution_turn_only_forward_blocked"
            else:
                planned = {"action": "stop", "linear_x": 0.0, "angular_z": 0.0, "duration_sec": 0.0, "reason": "forward_blocked_by_grid_quality"}
        summary["planned_first_action"] = planned
        summary["final_pose"] = initial_odom
        summary["final_distance_to_subgoal_m"] = initial_metrics["distance_to_subgoal_m"]
        summary["final_heading_error_rad"] = initial_metrics["heading_error_rad"]
        summary["heading_progress"]["final_abs_heading_error_rad"] = abs(float(initial_metrics["heading_error_rad"]))
        summary["heading_progress"]["heading_error_reduction_rad"] = 0.0
        summary["heading_progress"]["heading_error_reduction_ratio"] = 0.0
        summary["distance_progress"]["final_distance_to_subgoal_m"] = initial_metrics["distance_to_subgoal_m"]
        summary["distance_progress"]["distance_reduction_m"] = 0.0
        summary["distance_progress"]["distance_reduction_ratio"] = 0.0
        update_progress(summary)
        summary["final_decision"] = "DRY_RUN_PASS"
        return summary

    from geometry_msgs.msg import Twist  # type: ignore

    pub = rospy.Publisher(TOPIC_CMD, Twist, queue_size=1)
    rospy.sleep(0.2)
    current_odom = initial_odom
    last_accepted_odom = initial_odom
    last_observed_odom = initial_odom
    start_pose = pose_tuple(initial_odom)
    path_length_m = 0.0
    max_abs_cross_track_error_m = 0.0
    best_distance = float(initial_metrics["distance_to_subgoal_m"])
    no_heading_progress_count = 0
    no_distance_progress_count = 0
    align_tier = 1
    tier_low_progress_count = 0
    heading_worsened_count = 0
    align_start = time.monotonic()
    align_step_count = 0
    turn_efficiencies: List[float] = []
    observed_realtime_factors: List[float] = []
    start = time.monotonic()

    try:
        for step_idx in range(int(args.max_steps)):
            if time.monotonic() - start >= float(args.max_runtime_sec):
                summary["warnings"].append("max_runtime_reached")
                break
            gate = live_safety_check(
                rospy,
                grid_timeout_sec=args.grid_sample_timeout_sec,
                max_grid_stamp_age_sec=args.max_grid_stamp_age_sec,
            )
            policy = gate.get("l3v_policy", {})
            summary["l3v_policy"] = policy
            summary["l3v_policy_final_status"] = policy.get("status")
            summary["l3v_policy_transition_observed"] = bool(
                summary.get("l3v_policy_initial_status") != summary.get("l3v_policy_final_status")
            )
            grid_quality = gate.get("grid_quality", {})
            summary["grid_quality"] = grid_quality
            if not gate["odom"].get("pass") or not gate["status"].get("pass") or policy.get("must_stop"):
                summary["l3v_safety_fail_count"] += 1
                summary["steps"].append(
                    {
                        "step_index": step_idx,
                        "distance_to_subgoal_m": None,
                        "heading_error_rad": None,
                        "l3v_status": policy.get("status"),
                        "grid_freshness_pass": policy.get("grid_freshness_pass"),
                        "allow_turn": policy.get("allow_turn"),
                        "allow_forward": policy.get("allow_forward"),
                        "planned_action": "stop",
                        "executed_action": "blocked",
                        "cmd": {"linear_x": 0.0, "angular_z": 0.0, "duration_sec": 0.0},
                        "zero_cmd_vel_published_count": 0,
                        "reason": policy.get("reason"),
                        "gate": gate,
                    }
                )
                update_policy_counts(summary, policy, "stop")
                summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_ODOM" if not gate["odom"].get("pass") else final_decision_for_l3v_block(policy, True)
                break
            current_odom = gate["odom"]
            last_accepted_odom = current_odom
            last_observed_odom = current_odom
            metrics = compute_metrics(current_odom, target)
            if not metrics["pass"]:
                summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_ODOM"
                summary["steps"].append({"step_index": step_idx, "decision": "metrics_unavailable"})
                break
            if float(metrics["distance_to_subgoal_m"]) < float(args.success_distance_m):
                summary["phase"] = "DONE"
                summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_PASS"
                break
            phase = "ALIGN_PHASE" if abs(float(metrics["heading_error_rad"])) > float(args.heading_threshold_rad) else "APPROACH_PHASE"
            summary["phase"] = phase
            summary["phase_history"].append(phase)
            action = decide_action(metrics, args)
            if phase == "ALIGN_PHASE":
                action = simple_turn_action(metrics, args) if args.simple_motion_profile else adaptive_turn_action(metrics, align_tier)
            planned_action = action["action"]
            reason = policy.get("reason")
            if action["action"] == "turn":
                if not policy.get("allow_turn"):
                    summary["final_decision"] = final_decision_for_l3v_block(policy, True)
                    summary["steps"].append(
                        {
                            "step_index": step_idx,
                            "distance_to_subgoal_m": metrics.get("distance_to_subgoal_m"),
                            "heading_error_rad": metrics.get("heading_error_rad"),
                            "l3v_status": policy.get("status"),
                            "grid_freshness_pass": policy.get("grid_freshness_pass"),
                            "allow_turn": policy.get("allow_turn"),
                            "allow_forward": policy.get("allow_forward"),
                            "planned_action": "turn",
                            "executed_action": "blocked",
                            "cmd": {"linear_x": 0.0, "angular_z": 0.0, "duration_sec": 0.0},
                            "zero_cmd_vel_published_count": 0,
                            "reason": policy.get("reason"),
                        }
                    )
                    update_policy_counts(summary, policy, "turn")
                    break
                if policy.get("status") == "CONFLICT_NEEDS_CAUTION":
                    action = caution_turn_action(metrics, args)
                    reason = "conflict_caution_turn_only"
            elif action["action"] == "forward":
                if not grid_quality.get("grid_quality_pass") or not grid_quality.get("front_corridor_pass"):
                    zc = publish_zero(pub, rospy, count=3)
                    summary["zero_cmd_vel_published_count"] += zc
                    summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_STOPPED_GRID_QUALITY_FAIL"
                    record_grid_failure_diagnostic(
                        summary,
                        step_idx,
                        phase,
                        policy,
                        grid_quality,
                        initial_grid_quality,
                        current_odom,
                        metrics,
                        target,
                    )
                    summary["steps"].append(
                        {
                            "step_index": step_idx,
                            "phase": phase,
                            "distance_to_subgoal_m": metrics.get("distance_to_subgoal_m"),
                            "heading_error_rad": metrics.get("heading_error_rad"),
                            "l3v_status": policy.get("status"),
                            "grid_freshness_pass": policy.get("grid_freshness_pass"),
                            "grid_quality_pass": grid_quality.get("grid_quality_pass"),
                            "front_sector_pass": grid_quality.get("front_sector_pass"),
                            "front_corridor_pass": grid_quality.get("front_corridor_pass"),
                            "occupied_ratio": grid_quality.get("occupied_ratio"),
                            "unknown_ratio": grid_quality.get("unknown_ratio"),
                            "free_ratio": grid_quality.get("free_ratio"),
                            "front_corridor_occupied_ratio": grid_quality.get("front_corridor_occupied_ratio"),
                            "front_corridor_unknown_ratio": grid_quality.get("front_corridor_unknown_ratio"),
                            "odom_traversed_conflict_ratio": grid_quality.get("odom_traversed_conflict_ratio"),
                            "odom_traversed_cell_count": grid_quality.get("odom_traversed_cell_count"),
                            "odom_traversed_free_support_ratio": grid_quality.get("odom_traversed_free_support_ratio"),
                            "grid_quality_reasons": grid_quality.get("reasons"),
                            "grid_failure_likely_class": summary.get("grid_failure_likely_class"),
                            "allow_turn": policy.get("allow_turn"),
                            "allow_forward": policy.get("allow_forward"),
                            "planned_action": "forward",
                            "executed_action": "zero_stop",
                            "cmd": {"linear_x": 0.0, "angular_z": 0.0, "duration_sec": 0.0},
                            "zero_cmd_vel_published_count": zc,
                            "reason": "grid_quality_fail",
                        }
                    )
                    update_policy_counts(summary, policy, "forward")
                    break
                if not policy.get("allow_forward"):
                    if policy.get("allow_turn") and policy.get("status") == "CONFLICT_NEEDS_CAUTION" and abs(float(metrics["heading_error_rad"])) > 0.05:
                        action = caution_turn_action(metrics, args)
                        reason = "conflict_caution_turn_only_forward_blocked"
                    else:
                        summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_STOPPED_CONFLICT_TURN_ONLY" if policy.get("status") == "CONFLICT_NEEDS_CAUTION" else final_decision_for_l3v_block(policy, True)
                        summary["steps"].append(
                            {
                                "step_index": step_idx,
                                "distance_to_subgoal_m": metrics.get("distance_to_subgoal_m"),
                                "heading_error_rad": metrics.get("heading_error_rad"),
                                "l3v_status": policy.get("status"),
                                "grid_freshness_pass": policy.get("grid_freshness_pass"),
                                "allow_turn": policy.get("allow_turn"),
                                "allow_forward": policy.get("allow_forward"),
                                "planned_action": "forward",
                                "executed_action": "zero_stop",
                                "cmd": {"linear_x": 0.0, "angular_z": 0.0, "duration_sec": 0.0},
                                "zero_cmd_vel_published_count": publish_zero(pub, rospy, count=3),
                                "reason": "forward_blocked_by_caution" if policy.get("status") == "CONFLICT_NEEDS_CAUTION" else policy.get("reason"),
                            }
                        )
                        summary["zero_cmd_vel_published_count"] += summary["steps"][-1]["zero_cmd_vel_published_count"]
                        update_policy_counts(summary, policy, "forward")
                        break
            update_policy_counts(summary, policy, planned_action)
            before_pose = pose_tuple(current_odom)
            cmd_result = publish_slice(
                pub,
                rospy,
                action["linear_x"],
                action["angular_z"],
                action["duration_sec"],
                motion_duration_timebase=args.motion_duration_timebase,
            )
            summary["nonzero_cmd_vel_published"] = summary["nonzero_cmd_vel_published"] or bool(cmd_result.get("nonzero_cmd_vel_published"))
            summary["turn_response"]["cmd_publish_count_total"] += int(cmd_result.get("cmd_publish_count", 0))
            summary["turn_response"]["actual_command_duration_sec_total"] += float(cmd_result.get("actual_command_duration_sec", 0.0))
            summary["motion_timing"]["total_commanded_sim_duration_sec"] += float(cmd_result.get("actual_sim_duration_sec") or 0.0)
            summary["motion_timing"]["total_commanded_wall_duration_sec"] += float(cmd_result.get("actual_wall_duration_sec") or 0.0)
            summary["motion_timing"]["cmd_publish_count_total"] += int(cmd_result.get("cmd_publish_count", 0))
            summary["motion_timing"]["slice_count"] += 1
            if finite_number(cmd_result.get("observed_realtime_factor_during_slice")):
                observed_realtime_factors.append(float(cmd_result["observed_realtime_factor_during_slice"]))
            summary["warnings"].extend(cmd_result.get("warnings", []))
            step_zero_count = publish_zero(pub, rospy, count=3)
            summary["zero_cmd_vel_published_count"] += step_zero_count
            after_odom = read_odom(rospy, timeout=3.0)
            last_observed_odom = after_odom
            after_metrics = compute_metrics(after_odom, target)
            after_pose = pose_tuple(after_odom)
            xy_jump = None
            if before_pose is not None and after_pose is not None:
                xy_jump = math.hypot(after_pose[0] - before_pose[0], after_pose[1] - before_pose[1])
                path_length_m += xy_jump
            yaw_delta = None
            if before_pose is not None and after_pose is not None:
                yaw_delta = normalize_angle(after_pose[2] - before_pose[2])
            expected_yaw_delta = abs(float(action["angular_z"])) * float(action["duration_sec"])
            turn_efficiency = abs(float(yaw_delta)) / expected_yaw_delta if yaw_delta is not None and expected_yaw_delta > 1e-9 else None
            distance_delta = None
            moved_toward_goal = None
            heading_error_delta = None
            heading_improved = None
            if after_metrics.get("pass"):
                distance_delta = float(metrics["distance_to_subgoal_m"]) - float(after_metrics["distance_to_subgoal_m"])
                moved_toward_goal = distance_delta > 0.0
                heading_error_delta = abs(float(metrics["heading_error_rad"])) - abs(float(after_metrics["heading_error_rad"]))
                heading_improved = heading_error_delta > 0.0
                if action["action"] == "forward" and moved_toward_goal:
                    summary["any_forward_moved_toward_goal"] = True
                    summary["moved_toward_goal_step_count"] += 1
            step_measurement = compute_motion_measurement(
                start_pose,
                after_pose,
                target,
                path_length_m,
                summary.get("initial_distance_to_subgoal_m"),
                after_metrics.get("distance_to_subgoal_m"),
            )
            if step_measurement.get("abs_cross_track_error_m") is not None:
                max_abs_cross_track_error_m = max(max_abs_cross_track_error_m, float(step_measurement["abs_cross_track_error_m"]))
            step_measurement["max_abs_cross_track_error_m"] = max_abs_cross_track_error_m
            summary["motion_measurement"].update(step_measurement)
            actual_sim_duration = cmd_result.get("actual_sim_duration_sec")
            if not finite_number(actual_sim_duration):
                actual_sim_duration = action.get("duration_sec")
            commanded_linear_x_abs = abs(float(action["linear_x"]))
            commanded_angular_z_abs = abs(float(action["angular_z"]))
            expected_forward_distance_m = (
                commanded_linear_x_abs * float(actual_sim_duration)
                if action["action"] == "forward" and finite_number(actual_sim_duration)
                else None
            )
            observed_step_displacement_m = xy_jump
            observed_path_length_m = xy_jump
            xy_jump_policy = evaluate_xy_jump_policy(
                action["action"],
                observed_step_displacement_m,
                expected_forward_distance_m,
                float(args.max_xy_jump_m),
            )
            cross_track_drift_m = step_measurement.get("cross_track_error_m")
            abs_cross_track_drift_m = abs(float(cross_track_drift_m)) if finite_number(cross_track_drift_m) else None
            forward_distance_efficiency_ratio = (
                float(observed_step_displacement_m) / max(float(expected_forward_distance_m), 1e-6)
                if action["action"] == "forward" and finite_number(observed_step_displacement_m) and finite_number(expected_forward_distance_m)
                else None
            )
            forward_yaw_drift_deg = math.degrees(float(yaw_delta)) if action["action"] == "forward" and yaw_delta is not None else None
            forward_lateral_direction = lateral_drift_direction(cross_track_drift_m) if action["action"] == "forward" else None
            forward_bias_reasons: List[str] = []
            if action["action"] == "forward":
                if yaw_delta is not None and abs(float(yaw_delta)) > 0.08:
                    forward_bias_reasons.append("forward_yaw_delta_abs_gt_0.08_rad")
                if finite_number(cross_track_drift_m) and abs(float(cross_track_drift_m)) > 0.15:
                    forward_bias_reasons.append("forward_cross_track_error_abs_gt_0.15_m")
            forward_bias_suspected = bool(forward_bias_reasons)
            if action["action"] == "turn":
                summary["turn_response"]["total_expected_yaw_delta_rad"] += expected_yaw_delta
                if yaw_delta is not None:
                    summary["turn_response"]["total_actual_yaw_delta_rad"] += abs(float(yaw_delta))
                if turn_efficiency is not None:
                    turn_efficiencies.append(turn_efficiency)
                    if turn_efficiency < 0.10:
                        summary["turn_response"]["low_response_step_count"] += 1
            if action["action"] == "turn":
                summary["turn_step_count"] += 1
            else:
                summary["forward_step_count"] += 1
            summary["step_count"] += 1
            step = {
                "step_index": step_idx,
                "phase": phase,
                "distance_to_subgoal_m": metrics.get("distance_to_subgoal_m"),
                "heading_error_rad": metrics.get("heading_error_rad"),
                "prev_heading_error_rad": metrics.get("heading_error_rad"),
                "new_heading_error_rad": after_metrics.get("heading_error_rad"),
                "heading_error_reduction_rad": (
                    abs(float(metrics["heading_error_rad"])) - abs(float(after_metrics["heading_error_rad"]))
                    if after_metrics.get("pass") else None
                ),
                "prev_distance_to_effective_goal_m": metrics.get("distance_to_subgoal_m"),
                "new_distance_to_effective_goal_m": after_metrics.get("distance_to_subgoal_m"),
                "distance_delta_m": distance_delta,
                "moved_toward_goal": moved_toward_goal,
                "heading_error_delta_rad": heading_error_delta,
                "heading_improved": heading_improved,
                "straight_line_displacement_m": step_measurement.get("straight_line_displacement_m"),
                "path_length_m": step_measurement.get("path_length_m"),
                "along_track_progress_m": step_measurement.get("along_track_progress_m"),
                "cross_track_error_m": step_measurement.get("cross_track_error_m"),
                "abs_cross_track_error_m": step_measurement.get("abs_cross_track_error_m"),
                "commanded_linear_x_abs": commanded_linear_x_abs,
                "commanded_angular_z_abs": commanded_angular_z_abs,
                "expected_forward_distance_m": expected_forward_distance_m,
                "observed_step_displacement_m": observed_step_displacement_m,
                "observed_path_length_m": observed_path_length_m,
                "unexpected_xy_jump_excess_m": xy_jump_policy.get("unexpected_xy_jump_excess_m"),
                "whether_normal_commanded_forward_motion_classified_as_xy_jump": (
                    action["action"] == "forward" and not bool(xy_jump_policy.get("xy_jump_policy_pass"))
                ),
                "xy_jump_policy_pass": xy_jump_policy.get("xy_jump_policy_pass"),
                "xy_jump_policy_reason": xy_jump_policy.get("xy_jump_policy_reason"),
                "xy_jump_margin_m": xy_jump_policy.get("xy_jump_margin_m"),
                "xy_jump_ratio_limit": xy_jump_policy.get("xy_jump_ratio_limit"),
                "cross_track_drift_m": cross_track_drift_m,
                "abs_cross_track_drift_m": abs_cross_track_drift_m,
                "max_abs_cross_track_drift_m": max_abs_cross_track_error_m,
                "cross_track_drift_policy_source": "diagnostic_only_no_new_failure_threshold",
                "forward_bias_diagnostic_enabled": action["action"] == "forward",
                "forward_cmd_linear_x": action["linear_x"] if action["action"] == "forward" else None,
                "forward_cmd_angular_z": action["angular_z"] if action["action"] == "forward" else None,
                "forward_actual_sim_duration_sec": actual_sim_duration if action["action"] == "forward" else None,
                "forward_expected_distance_m": expected_forward_distance_m,
                "forward_observed_displacement_m": observed_step_displacement_m if action["action"] == "forward" else None,
                "forward_distance_efficiency_ratio": forward_distance_efficiency_ratio,
                "forward_yaw_before_rad": before_pose[2] if action["action"] == "forward" and before_pose else None,
                "forward_yaw_after_rad": after_pose[2] if action["action"] == "forward" and after_pose else None,
                "forward_yaw_delta_rad": yaw_delta if action["action"] == "forward" else None,
                "forward_yaw_drift_deg": forward_yaw_drift_deg,
                "forward_cross_track_error_m": cross_track_drift_m if action["action"] == "forward" else None,
                "forward_abs_cross_track_error_m": abs_cross_track_drift_m if action["action"] == "forward" else None,
                "forward_lateral_drift_direction": forward_lateral_direction,
                "forward_bias_suspected": forward_bias_suspected if action["action"] == "forward" else False,
                "forward_bias_reason": ",".join(forward_bias_reasons) if forward_bias_reasons else None,
                "distance_to_effective_goal_m": after_metrics.get("distance_to_subgoal_m"),
                "prev_distance_m": metrics.get("distance_to_subgoal_m"),
                "new_distance_m": after_metrics.get("distance_to_subgoal_m"),
                "distance_reduction_m": (
                    float(metrics["distance_to_subgoal_m"]) - float(after_metrics["distance_to_subgoal_m"])
                    if after_metrics.get("pass") else None
                ),
                "l3v_status": policy.get("status"),
                "grid_freshness_pass": policy.get("grid_freshness_pass"),
                "grid_quality_pass": grid_quality.get("grid_quality_pass"),
                "front_corridor_pass": grid_quality.get("front_corridor_pass"),
                "allow_turn": policy.get("allow_turn"),
                "allow_forward": policy.get("allow_forward"),
                "planned_action": planned_action,
                "executed_action": action["action"],
                "cmd": {
                    "linear_x": action["linear_x"],
                    "angular_z": action["angular_z"],
                    "duration_sec": action["duration_sec"],
                },
                "tier": action.get("tier"),
                "prev_pose": {"x": before_pose[0], "y": before_pose[1], "yaw": before_pose[2]} if before_pose else {},
                "new_pose": {"x": after_pose[0], "y": after_pose[1], "yaw": after_pose[2]} if after_pose else {},
                "yaw_delta_rad": yaw_delta,
                "expected_yaw_delta_rad": expected_yaw_delta,
                "turn_efficiency": turn_efficiency,
                "cmd_publish_count": cmd_result.get("cmd_publish_count"),
                "cmd_publish_rate_hz": cmd_result.get("cmd_publish_rate_hz"),
                "motion_duration_timebase": cmd_result.get("motion_duration_timebase"),
                "requested_duration_sec": cmd_result.get("requested_duration_sec"),
                "ros_time_start_sec": cmd_result.get("ros_time_start_sec"),
                "ros_time_end_sec": cmd_result.get("ros_time_end_sec"),
                "actual_sim_duration_sec": cmd_result.get("actual_sim_duration_sec"),
                "wall_time_start_sec": cmd_result.get("wall_time_start_sec"),
                "wall_time_end_sec": cmd_result.get("wall_time_end_sec"),
                "actual_wall_duration_sec": cmd_result.get("actual_wall_duration_sec"),
                "observed_realtime_factor_during_slice": cmd_result.get("observed_realtime_factor_during_slice"),
                "actual_command_duration_sec": cmd_result.get("actual_command_duration_sec"),
                "commanded_twist_history": cmd_result.get("commanded_twist_history"),
                "zero_cmd_vel_published_count": step_zero_count,
                "reason": reason,
                "action": action,
                "distance_before_m": metrics.get("distance_to_subgoal_m"),
                "heading_error_before_rad": metrics.get("heading_error_rad"),
                "odom_after": after_odom,
                "distance_after_m": after_metrics.get("distance_to_subgoal_m"),
                "heading_error_after_rad": after_metrics.get("heading_error_rad"),
                "xy_jump_m": xy_jump,
            }
            summary["steps"].append(step)
            summary["observed_step_displacement_m"] = observed_step_displacement_m
            summary["observed_path_length_m"] = observed_path_length_m
            summary["unexpected_xy_jump_excess_m"] = xy_jump_policy.get("unexpected_xy_jump_excess_m")
            summary["xy_jump_margin_m"] = xy_jump_policy.get("xy_jump_margin_m")
            summary["xy_jump_ratio_limit"] = xy_jump_policy.get("xy_jump_ratio_limit")
            summary["xy_jump_policy_pass"] = xy_jump_policy.get("xy_jump_policy_pass")
            summary["xy_jump_policy_reason"] = xy_jump_policy.get("xy_jump_policy_reason")
            summary["whether_normal_commanded_forward_motion_classified_as_xy_jump"] = (
                action["action"] == "forward" and not bool(xy_jump_policy.get("xy_jump_policy_pass"))
            )
            summary["cross_track_drift_m"] = cross_track_drift_m
            summary["cross_track_drift_observed"] = finite_number(cross_track_drift_m)
            summary["abs_cross_track_drift_m"] = abs_cross_track_drift_m
            summary["max_abs_cross_track_drift_m"] = max_abs_cross_track_error_m
            summary["cross_track_drift_policy_source"] = "diagnostic_only_no_new_failure_threshold"
            if action["action"] == "forward":
                summary["expected_forward_distance_m"] = expected_forward_distance_m
                summary["forward_cmd_linear_x"] = action["linear_x"]
                summary["forward_cmd_angular_z"] = action["angular_z"]
                summary["forward_actual_sim_duration_sec"] = actual_sim_duration
                summary["forward_expected_distance_m"] = expected_forward_distance_m
                summary["forward_observed_displacement_m"] = observed_step_displacement_m
                summary["forward_distance_efficiency_ratio"] = forward_distance_efficiency_ratio
                summary["forward_yaw_before_rad"] = before_pose[2] if before_pose else None
                summary["forward_yaw_after_rad"] = after_pose[2] if after_pose else None
                summary["forward_yaw_delta_rad"] = yaw_delta
                summary["forward_yaw_drift_deg"] = forward_yaw_drift_deg
                summary["forward_cross_track_error_m"] = cross_track_drift_m
                summary["forward_abs_cross_track_error_m"] = abs_cross_track_drift_m
                summary["forward_lateral_drift_direction"] = forward_lateral_direction
                summary["forward_bias_suspected"] = forward_bias_suspected
                summary["forward_bias_reason"] = ",".join(forward_bias_reasons) if forward_bias_reasons else None
            if not xy_jump_policy.get("xy_jump_policy_pass"):
                summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_FAILED_EXCESSIVE_DRIFT"
                summary["failure_step_index"] = step_idx
                summary["failure_reason"] = xy_jump_policy.get("xy_jump_policy_reason")
                summary["failure_cross_track_error_m"] = step_measurement.get("cross_track_error_m")
                break
            if not after_odom.get("pass") or not after_metrics.get("pass"):
                summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_ODOM"
                break
            current_odom = after_odom
            last_accepted_odom = after_odom
            distance_after = float(after_metrics["distance_to_subgoal_m"])
            if distance_after < float(args.success_distance_m):
                summary["phase"] = "DONE"
                summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_PASS"
                break
            if action["action"] == "turn":
                heading_reduction = abs(float(metrics["heading_error_rad"])) - abs(float(after_metrics["heading_error_rad"]))
                heading_worsened = heading_reduction < 0.0
                step["heading_error_worsened"] = heading_worsened
                if heading_worsened:
                    heading_worsened_count += 1
                    summary["turn_response"]["heading_error_worsened_count"] = heading_worsened_count
                else:
                    heading_worsened_count = 0
                if heading_worsened_count >= 2:
                    summary["heading_progress"]["heading_diverged"] = True
                    summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_STOPPED_HEADING_DIVERGED"
                    break
                align_step_count += 1
                summary["align_profile"]["tier_history"].append(align_tier)
                summary["align_profile"]["max_tier_reached"] = max(summary["align_profile"]["tier_history"])
                if heading_reduction > float(args.min_heading_progress_rad):
                    no_heading_progress_count = 0
                    tier_low_progress_count = 0
                    summary["heading_progress"]["heading_progress_pass"] = True
                else:
                    no_heading_progress_count += 1
                    tier_low_progress_count += 1
                summary["heading_progress"]["no_heading_progress_count"] = no_heading_progress_count
                if abs(float(after_metrics["heading_error_rad"])) <= float(args.heading_threshold_rad):
                    summary["phase"] = "APPROACH_PHASE"
                    summary["heading_progress"]["align_success"] = True
                elif args.simple_motion_profile:
                    pass
                elif tier_low_progress_count >= 2 and align_tier < 3:
                    align_tier += 1
                    tier_low_progress_count = 0
                elif align_tier >= 3 and no_heading_progress_count >= int(args.no_heading_progress_window):
                    summary["heading_progress"]["insufficient_yaw_response"] = True
                    summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_STOPPED_INSUFFICIENT_YAW_RESPONSE"
                    break
                elif align_step_count >= int(args.align_max_steps) or time.monotonic() - align_start >= float(args.align_max_runtime_sec):
                    total_reduction = abs(float(summary["initial_heading_error_rad"])) - abs(float(after_metrics["heading_error_rad"]))
                    ratio = total_reduction / abs(float(summary["initial_heading_error_rad"])) if abs(float(summary["initial_heading_error_rad"])) > 1e-9 else 0.0
                    if total_reduction >= 0.20 or ratio >= 0.20:
                        summary["heading_progress"]["align_progress_incomplete"] = True
                        summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_ALIGN_PROGRESS_INCOMPLETE"
                    elif total_reduction < 0.10 and align_tier >= 3:
                        summary["heading_progress"]["insufficient_yaw_response"] = True
                        summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_STOPPED_INSUFFICIENT_YAW_RESPONSE"
                    else:
                        summary["heading_progress"]["align_progress_incomplete"] = True
                        summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_ALIGN_PROGRESS_INCOMPLETE"
                    break
            else:
                distance_reduction = float(metrics["distance_to_subgoal_m"]) - distance_after
                if distance_reduction > 0.0:
                    best_distance = distance_after
                    no_distance_progress_count = 0
                    if distance_reduction >= 0.03:
                        summary["distance_progress"]["distance_progress_pass"] = True
                else:
                    no_distance_progress_count += 1
                summary["distance_progress"]["no_distance_progress_count"] = no_distance_progress_count
                if not args.simple_motion_profile and no_distance_progress_count >= int(args.no_progress_window):
                    summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_STOPPED_NO_DISTANCE_PROGRESS"
                    break
    finally:
        summary["zero_cmd_vel_published_count"] += publish_zero(pub, rospy, count=3)

    final_pose_odom = current_odom
    summary["final_pose_source"] = "last_accepted_odom"
    if (
        summary.get("final_decision") == "EXECUTE_LOCAL_SUBGOAL_FAILED_EXCESSIVE_DRIFT"
        and last_observed_odom
        and last_observed_odom.get("pass")
    ):
        final_pose_odom = last_observed_odom
        summary["final_pose_source"] = "last_observed_odom_after_excessive_drift"

    accepted_pose_fields = pose_summary_fields(last_accepted_odom)
    observed_pose_fields = pose_summary_fields(last_observed_odom)
    summary["last_accepted_pose_x"] = accepted_pose_fields["x"]
    summary["last_accepted_pose_y"] = accepted_pose_fields["y"]
    summary["last_accepted_pose_yaw"] = accepted_pose_fields["yaw"]
    summary["last_observed_pose_x"] = observed_pose_fields["x"]
    summary["last_observed_pose_y"] = observed_pose_fields["y"]
    summary["last_observed_pose_yaw"] = observed_pose_fields["yaw"]

    final_metrics = compute_metrics(final_pose_odom, target)
    accepted_metrics = compute_metrics(last_accepted_odom, target)
    observed_metrics = compute_metrics(last_observed_odom, target)
    summary["final_pose"] = final_pose_odom
    if final_metrics.get("pass"):
        summary["final_distance_to_subgoal_m"] = final_metrics["distance_to_subgoal_m"]
        summary["final_heading_error_rad"] = final_metrics["heading_error_rad"]
    update_progress(summary)
    accepted_measurement = compute_motion_measurement(
        start_pose,
        pose_tuple(last_accepted_odom),
        target,
        path_length_m,
        summary.get("initial_distance_to_subgoal_m"),
        accepted_metrics.get("distance_to_subgoal_m"),
    )
    observed_measurement = compute_motion_measurement(
        start_pose,
        pose_tuple(last_observed_odom),
        target,
        path_length_m,
        summary.get("initial_distance_to_subgoal_m"),
        observed_metrics.get("distance_to_subgoal_m"),
    )
    summary["final_accepted_straight_line_displacement_m"] = accepted_measurement.get("straight_line_displacement_m")
    summary["final_accepted_along_track_progress_m"] = accepted_measurement.get("along_track_progress_m")
    summary["final_accepted_cross_track_error_m"] = accepted_measurement.get("cross_track_error_m")
    summary["final_accepted_net_distance_to_goal_reduction_m"] = accepted_measurement.get("net_distance_to_goal_reduction_m")
    summary["final_observed_straight_line_displacement_m"] = observed_measurement.get("straight_line_displacement_m")
    summary["final_observed_along_track_progress_m"] = observed_measurement.get("along_track_progress_m")
    summary["final_observed_cross_track_error_m"] = observed_measurement.get("cross_track_error_m")
    summary["final_observed_net_distance_to_goal_reduction_m"] = observed_measurement.get("net_distance_to_goal_reduction_m")
    final_measurement = compute_motion_measurement(
        start_pose,
        pose_tuple(final_pose_odom),
        target,
        path_length_m,
        summary.get("initial_distance_to_subgoal_m"),
        summary.get("final_distance_to_subgoal_m"),
    )
    if summary["motion_measurement"].get("max_abs_cross_track_error_m") is not None:
        final_measurement["max_abs_cross_track_error_m"] = summary["motion_measurement"]["max_abs_cross_track_error_m"]
    elif final_measurement.get("abs_cross_track_error_m") is not None:
        final_measurement["max_abs_cross_track_error_m"] = final_measurement["abs_cross_track_error_m"]
    summary["motion_measurement"].update(final_measurement)
    if turn_efficiencies:
        summary["turn_response"]["mean_turn_efficiency"] = sum(turn_efficiencies) / len(turn_efficiencies)
        summary["turn_response"]["min_turn_efficiency"] = min(turn_efficiencies)
        summary["turn_response"]["max_turn_efficiency"] = max(turn_efficiencies)
    if observed_realtime_factors:
        summary["motion_timing"]["mean_observed_realtime_factor"] = sum(observed_realtime_factors) / len(observed_realtime_factors)
    if finite_number(summary.get("initial_heading_error_rad")) and finite_number(summary.get("final_heading_error_rad")):
        initial_abs_heading = abs(float(summary["initial_heading_error_rad"]))
        final_abs_heading = abs(float(summary["final_heading_error_rad"]))
        heading_reduction = initial_abs_heading - final_abs_heading
        summary["heading_progress"]["initial_abs_heading_error_rad"] = initial_abs_heading
        summary["heading_progress"]["final_abs_heading_error_rad"] = final_abs_heading
        summary["heading_progress"]["heading_error_reduction_rad"] = heading_reduction
        summary["heading_progress"]["heading_error_reduction_ratio"] = heading_reduction / initial_abs_heading if initial_abs_heading > 1e-9 else None
        summary["heading_progress"]["heading_progress_pass"] = bool(heading_reduction > 0.03)
    summary["distance_progress"]["initial_distance_to_subgoal_m"] = summary.get("initial_distance_to_subgoal_m")
    summary["distance_progress"]["final_distance_to_subgoal_m"] = summary.get("final_distance_to_subgoal_m")
    summary["distance_progress"]["initial_distance_to_effective_goal_m"] = summary.get("initial_distance_to_effective_goal_m")
    summary["distance_progress"]["final_distance_to_effective_goal_m"] = summary.get("final_distance_to_effective_goal_m")
    summary["distance_progress"]["distance_reduction_m"] = summary.get("distance_reduction_m")
    summary["distance_progress"]["distance_reduction_ratio"] = summary.get("distance_reduction_ratio")
    summary["distance_progress"]["distance_progress_pass"] = bool(
        summary["forward_step_count"] > 0
        and finite_number(summary.get("distance_reduction_m"))
        and float(summary["distance_reduction_m"]) > 0.03
    )

    if summary["final_decision"] is None:
        if finite_number(summary.get("final_distance_to_subgoal_m")) and float(summary["final_distance_to_subgoal_m"]) <= float(args.success_distance_m):
            summary["phase"] = "DONE"
            summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_PASS"
        elif summary.get("any_forward_moved_toward_goal") and summary["forward_step_count"] > 0:
            summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_GOAL_DIRECTED_PROGRESS"
        elif summary.get("l3v_policy", {}).get("status") == "CONFLICT_NEEDS_CAUTION":
            summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_STOPPED_CONFLICT_TURN_ONLY"
        elif summary["forward_step_count"] > 0:
            summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_APPROACH_EXECUTED_NO_CLEAR_PROGRESS"
        else:
            total_reduction = summary["heading_progress"].get("heading_error_reduction_rad")
            ratio_h = summary["heading_progress"].get("heading_error_reduction_ratio")
            if finite_number(total_reduction) and (float(total_reduction) >= 0.20 or (finite_number(ratio_h) and float(ratio_h) >= 0.20)):
                summary["heading_progress"]["align_progress_incomplete"] = True
                summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_ALIGN_PROGRESS_INCOMPLETE"
            elif finite_number(total_reduction) and float(total_reduction) < 0.10 and summary["align_profile"].get("max_tier_reached") == 3:
                summary["heading_progress"]["insufficient_yaw_response"] = True
                summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_STOPPED_INSUFFICIENT_YAW_RESPONSE"
            else:
                summary["heading_progress"]["align_progress_incomplete"] = True
                summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_ALIGN_PROGRESS_INCOMPLETE"
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read live inputs and plan the first slice without nonzero /cmd_vel.")
    mode.add_argument("--execute", action="store_true", help="Publish restricted nonzero /cmd_vel slices if all gates pass.")
    parser.add_argument("--simple-motion-profile", type=parse_bool, default=False)
    parser.add_argument("--heading-threshold-rad", type=float, default=0.35)
    parser.add_argument("--align-max-steps", type=int, default=30)
    parser.add_argument("--align-max-runtime-sec", type=float, default=45.0)
    parser.add_argument("--no-heading-progress-window", type=int, default=6)
    parser.add_argument("--min-heading-progress-rad", type=float, default=0.03)
    parser.add_argument("--success-distance-m", type=float, default=0.70)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-runtime-sec", type=float, default=45.0)
    parser.add_argument("--angular-z-max", type=float, default=0.30)
    parser.add_argument("--turn-slice-sec", type=float, default=1.00)
    parser.add_argument("--linear-x", type=float, default=0.30)
    parser.add_argument("--forward-slice-sec", type=float, default=0.50)
    parser.add_argument("--motion-duration-timebase", choices=["sim_time", "wall_time"], default="sim_time")
    parser.add_argument("--max-xy-jump-m", type=float, default=0.20)
    parser.add_argument("--min-progress-ratio", type=float, default=0.20)
    parser.add_argument("--no-progress-window", type=int, default=5)
    parser.add_argument("--grid-sample-timeout-sec", type=float, default=3.0)
    parser.add_argument("--max-grid-stamp-age-sec", type=float, default=2.0)
    parser.add_argument("--local-goal-radius-m", type=float, default=3.00)
    args = parser.parse_args()
    if not args.execute:
        args.dry_run = True
    return args


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        summary = run(args)
    except Exception as exc:
        summary = empty_summary("execute" if args.execute else "dry_run")
        summary["final_decision"] = "EXECUTE_LOCAL_SUBGOAL_BLOCKED_BY_ODOM" if args.execute else "DRY_RUN_BLOCKED"
        summary["errors"].append(f"unhandled_exception: {exc}")
    write_json(OUT / "local_subgoal_runner_summary.json", summary)
    write_report(summary)
    print(json.dumps({"final_decision": summary["final_decision"], "summary": str(OUT / "local_subgoal_runner_summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
