#!/usr/bin/env python3
"""Diagnose local-goal geometry and optional in-place turn response."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = ROOT / "debug" / "short_horizon_target_selection" / "short_horizon_target_override.json"
OUT = ROOT / "debug" / "local_subgoal_runner_mvp" / "local_goal_and_turn_diagnostic_summary.json"
REPORT = ROOT / "audit_reports" / "local_goal_and_turn_diagnostic_report.md"

TOPIC_ODOM = "/team/livox/icp_odom_gated"
TOPIC_CMD = "/cmd_vel"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid_bool:{value}")


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def pose_tuple(odom: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    pose = odom.get("pose_x_y_yaw")
    if isinstance(pose, list) and len(pose) == 3 and all(finite_number(v) for v in pose):
        return float(pose[0]), float(pose[1]), float(pose[2])
    return None


def read_odom(rospy: Any, timeout: float = 5.0) -> Dict[str, Any]:
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


def load_target() -> Dict[str, Any]:
    try:
        raw = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"pass": False, "path": str(TARGET_PATH), "error": str(exc)}
    xy = raw.get("target_xy_team_livox_odom")
    valid_xy = isinstance(xy, list) and len(xy) == 2 and all(finite_number(v) for v in xy)
    checks = {
        "file_exists": TARGET_PATH.exists(),
        "target_xy_valid": valid_xy,
        "target_frame_team_livox_odom": raw.get("target_frame") == "team_livox_odom",
        "diagnostic_only_true": raw.get("diagnostic_only") is True,
        "safe_for_navigation_false": raw.get("safe_for_navigation") is False,
        "planner_ready_false": raw.get("planner_ready") is False,
        "send_to_navigation_false": raw.get("send_to_navigation") is False,
    }
    return {
        "pass": all(checks.values()),
        "path": str(TARGET_PATH),
        "raw": raw,
        "checks": checks,
        "raw_target_x": float(xy[0]) if valid_xy else None,
        "raw_target_y": float(xy[1]) if valid_xy else None,
        "target_frame": raw.get("target_frame"),
    }


def compute_geometry(odom: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    pose = pose_tuple(odom)
    if pose is None or not finite_number(target.get("raw_target_x")) or not finite_number(target.get("raw_target_y")):
        return {"pass": False}
    x, y, yaw = pose
    tx = float(target["raw_target_x"])
    ty = float(target["raw_target_y"])
    dx = tx - x
    dy = ty - y
    distance = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    heading_error = normalize_angle(bearing - yaw)
    return {
        "pass": True,
        "current_pose_x": x,
        "current_pose_y": y,
        "current_yaw": yaw,
        "raw_target_x": tx,
        "raw_target_y": ty,
        "target_distance_m": distance,
        "target_bearing_global_rad": bearing,
        "heading_error_rad": heading_error,
        "heading_error_deg": math.degrees(heading_error),
        "target_vector_x": dx,
        "target_vector_y": dy,
        "target_is_large_turn": abs(heading_error) > 0.75,
        "target_is_sideways": abs(heading_error) > 1.0,
    }


def publish_zero(pub: Any, count: int = 3) -> int:
    from geometry_msgs.msg import Twist  # type: ignore

    msg = Twist()
    sent = 0
    for _ in range(count):
        pub.publish(msg)
        sent += 1
        time.sleep(0.05)
    return sent


def publish_turn_slice(pub: Any, rospy: Any, angular_z: float, duration_sec: float, timebase: str, rate_hz: float) -> Dict[str, Any]:
    from geometry_msgs.msg import Twist  # type: ignore

    cmd = Twist()
    cmd.angular.z = float(angular_z)
    wall_start = time.monotonic()
    ros_start = float(rospy.Time.now().to_sec())
    wall_timeout_sec = max(float(duration_sec) * 30.0, 60.0)
    sleep_sec = 1.0 / max(float(rate_hz), 1.0)
    count = 0
    warnings = []
    while not rospy.is_shutdown():
        wall_elapsed = time.monotonic() - wall_start
        ros_now = float(rospy.Time.now().to_sec())
        sim_elapsed = max(0.0, ros_now - ros_start)
        elapsed = sim_elapsed if timebase == "sim_time" else wall_elapsed
        if elapsed >= float(duration_sec):
            break
        if timebase == "sim_time" and wall_elapsed >= wall_timeout_sec:
            warnings.append("sim_time_motion_slice_wall_timeout")
            break
        pub.publish(cmd)
        count += 1
        time.sleep(sleep_sec)
    wall_end = time.monotonic()
    ros_end = float(rospy.Time.now().to_sec())
    actual_wall = wall_end - wall_start
    actual_sim = max(0.0, ros_end - ros_start)
    return {
        "motion_duration_timebase": timebase,
        "requested_duration_sec": float(duration_sec),
        "ros_time_start_sec": ros_start,
        "ros_time_end_sec": ros_end,
        "actual_sim_duration_sec": actual_sim,
        "wall_time_start_sec": wall_start,
        "wall_time_end_sec": wall_end,
        "actual_wall_duration_sec": actual_wall,
        "observed_realtime_factor": actual_sim / actual_wall if actual_wall > 1e-9 else None,
        "cmd_publish_count": count,
        "warnings": warnings,
    }


def turn_response(before: Dict[str, Any], after: Dict[str, Any], target: Dict[str, Any], angular_z: float, duration_sec: float) -> Dict[str, Any]:
    before_pose = pose_tuple(before)
    after_pose = pose_tuple(after)
    before_geo = compute_geometry(before, target)
    after_geo = compute_geometry(after, target)
    if before_pose is None or after_pose is None or not before_geo.get("pass") or not after_geo.get("pass"):
        return {"pass": False}
    yaw_delta = normalize_angle(after_pose[2] - before_pose[2])
    xy_drift = math.hypot(after_pose[0] - before_pose[0], after_pose[1] - before_pose[1])
    expected = abs(float(angular_z)) * float(duration_sec)
    sx, sy, _ = before_pose
    gx = float(target["raw_target_x"])
    gy = float(target["raw_target_y"])
    vx = gx - sx
    vy = gy - sy
    norm = math.hypot(vx, vy)
    along = None
    cross = None
    if norm > 1e-9:
        ux = vx / norm
        uy = vy / norm
        dx = after_pose[0] - sx
        dy = after_pose[1] - sy
        along = dx * ux + dy * uy
        cross = ux * dy - uy * dx
    reduction = abs(float(before_geo["heading_error_rad"])) - abs(float(after_geo["heading_error_rad"]))
    return {
        "pass": True,
        "yaw_before": before_pose[2],
        "yaw_after": after_pose[2],
        "yaw_delta_rad": yaw_delta,
        "xy_drift_m": xy_drift,
        "heading_error_before": before_geo["heading_error_rad"],
        "heading_error_after": after_geo["heading_error_rad"],
        "heading_error_reduction_rad": reduction,
        "expected_yaw_delta_rad": expected,
        "turn_efficiency": abs(yaw_delta) / expected if expected > 1e-9 else None,
        "along_track_progress_m": along,
        "cross_track_error_m": cross,
    }


def write_outputs(summary: Dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    geometry = summary.get("target_geometry", {})
    response = summary.get("turn_response", {})
    timing = summary.get("turn_timing", {})
    lines = [
        "# Local Goal And Turn Diagnostic Report",
        "",
        f"- execute_turn: `{summary.get('execute_turn')}`",
        f"- current_pose_x: `{geometry.get('current_pose_x')}`",
        f"- current_pose_y: `{geometry.get('current_pose_y')}`",
        f"- current_yaw: `{geometry.get('current_yaw')}`",
        f"- raw_target_x: `{geometry.get('raw_target_x')}`",
        f"- raw_target_y: `{geometry.get('raw_target_y')}`",
        f"- target_distance_m: `{geometry.get('target_distance_m')}`",
        f"- target_bearing_global_rad: `{geometry.get('target_bearing_global_rad')}`",
        f"- heading_error_rad: `{geometry.get('heading_error_rad')}`",
        f"- heading_error_deg: `{geometry.get('heading_error_deg')}`",
        f"- target_vector_x: `{geometry.get('target_vector_x')}`",
        f"- target_vector_y: `{geometry.get('target_vector_y')}`",
        f"- target_is_large_turn: `{geometry.get('target_is_large_turn')}`",
        f"- target_is_sideways: `{geometry.get('target_is_sideways')}`",
        "",
        "## Turn Response",
        "",
        f"- yaw_before: `{response.get('yaw_before')}`",
        f"- yaw_after: `{response.get('yaw_after')}`",
        f"- yaw_delta_rad: `{response.get('yaw_delta_rad')}`",
        f"- xy_drift_m: `{response.get('xy_drift_m')}`",
        f"- heading_error_before: `{response.get('heading_error_before')}`",
        f"- heading_error_after: `{response.get('heading_error_after')}`",
        f"- heading_error_reduction_rad: `{response.get('heading_error_reduction_rad')}`",
        f"- expected_yaw_delta_rad: `{response.get('expected_yaw_delta_rad')}`",
        f"- turn_efficiency: `{response.get('turn_efficiency')}`",
        f"- along_track_progress_m: `{response.get('along_track_progress_m')}`",
        f"- cross_track_error_m: `{response.get('cross_track_error_m')}`",
        f"- actual_sim_duration_sec: `{timing.get('actual_sim_duration_sec')}`",
        f"- actual_wall_duration_sec: `{timing.get('actual_wall_duration_sec')}`",
        f"- observed_realtime_factor: `{timing.get('observed_realtime_factor')}`",
        "",
        "## Boundary",
        "",
        f"- forbidden_sources_used: `{summary.get('forbidden_sources_used')}`",
        f"- called_move_base: `{summary.get('called_move_base')}`",
        f"- sent_navigation_goal: `{summary.get('sent_navigation_goal')}`",
    ]
    if summary.get("errors"):
        lines += ["", "## Errors", ""]
        lines += [f"- `{err}`" for err in summary["errors"]]
    if summary.get("warnings"):
        lines += ["", "## Warnings", ""]
        lines += [f"- `{warning}`" for warning in summary["warnings"]]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-turn", type=parse_bool, default=False)
    parser.add_argument("--turn-angular-z", type=float, default=0.3)
    parser.add_argument("--turn-duration-sec", type=float, default=5.0)
    parser.add_argument("--motion-duration-timebase", choices=["sim_time", "wall_time"], default="sim_time")
    parser.add_argument("--publish-rate-hz", type=float, default=20.0)
    parser.add_argument("--grid-sample-timeout-sec", type=float, default=8.0)
    args = parser.parse_args()

    summary: Dict[str, Any] = {
        "execute_turn": bool(args.execute_turn),
        "turn_angular_z": float(args.turn_angular_z),
        "turn_duration_sec": float(args.turn_duration_sec),
        "motion_duration_timebase": args.motion_duration_timebase,
        "publish_rate_hz": float(args.publish_rate_hz),
        "grid_sample_timeout_sec": float(args.grid_sample_timeout_sec),
        "forbidden_sources_used": [],
        "called_move_base": False,
        "sent_navigation_goal": False,
        "errors": [],
        "warnings": [],
    }

    try:
        import rospy  # type: ignore

        rospy.init_node("local_goal_and_turn_diagnostic", anonymous=True, disable_signals=True)
        odom_before = read_odom(rospy)
        target = load_target()
        summary["odom_before"] = odom_before
        summary["target"] = target
        if not odom_before.get("pass"):
            summary["errors"].append("compliant_odom_unavailable")
        if not target.get("pass"):
            summary["errors"].append("target_unavailable_or_boundary_invalid")
        geometry = compute_geometry(odom_before, target)
        summary["target_geometry"] = geometry
        if args.execute_turn and geometry.get("pass") and odom_before.get("pass") and target.get("pass"):
            from geometry_msgs.msg import Twist  # noqa: F401  # type: ignore

            pub = rospy.Publisher(TOPIC_CMD, Twist, queue_size=1)
            time.sleep(0.2)
            sign = 1.0 if float(geometry["heading_error_rad"]) >= 0.0 else -1.0
            angular_z = sign * abs(float(args.turn_angular_z))
            timing = publish_turn_slice(pub, rospy, angular_z, args.turn_duration_sec, args.motion_duration_timebase, args.publish_rate_hz)
            summary["turn_timing"] = timing
            summary["warnings"].extend(timing.get("warnings", []))
            summary["zero_cmd_vel_published_count"] = publish_zero(pub, count=3)
            odom_after = read_odom(rospy)
            summary["odom_after"] = odom_after
            summary["turn_response"] = turn_response(odom_before, odom_after, target, angular_z, args.turn_duration_sec)
        else:
            summary["turn_timing"] = {}
            summary["turn_response"] = {}
    except Exception as exc:
        summary["errors"].append(str(exc))

    write_outputs(summary)
    print(json.dumps({"execute_turn": summary["execute_turn"], "summary": str(OUT), "report": str(REPORT)}))
    return 0 if not summary.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
