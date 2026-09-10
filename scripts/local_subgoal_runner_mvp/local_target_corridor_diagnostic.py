#!/usr/bin/env python3
"""Read-only diagnostic for target/corridor heading alignment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = ROOT / "debug" / "short_horizon_target_selection" / "short_horizon_target_override.json"
OUT = ROOT / "debug" / "local_subgoal_runner_mvp" / "local_target_corridor_diagnostic_summary.json"
REPORT = ROOT / "audit_reports" / "local_target_corridor_diagnostic_report.md"
TOPIC_ODOM = "/team/livox/icp_odom_gated"


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


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
        "target_x": float(xy[0]) if valid_xy else None,
        "target_y": float(xy[1]) if valid_xy else None,
    }


def pose_tuple(odom: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    pose = odom.get("pose_x_y_yaw")
    if isinstance(pose, list) and len(pose) == 3 and all(finite_number(v) for v in pose):
        return float(pose[0]), float(pose[1]), float(pose[2])
    return None


def classify_issue(robot_aligned: Optional[bool], target_aligned: Optional[bool], corridor_known: bool) -> str:
    if not corridor_known:
        return "unknown_corridor_heading"
    if target_aligned is False and robot_aligned is False:
        return "both_target_and_robot_suspect"
    if target_aligned is False:
        return "target_selection_suspect"
    if robot_aligned is False:
        return "robot_orientation_suspect"
    return "target_and_robot_aligned"


def compute(odom: Dict[str, Any], target: Dict[str, Any], corridor_heading_rad: Optional[float]) -> Dict[str, Any]:
    pose = pose_tuple(odom)
    if pose is None or not finite_number(target.get("target_x")) or not finite_number(target.get("target_y")):
        return {"pass": False}
    x, y, yaw = pose
    tx = float(target["target_x"])
    ty = float(target["target_y"])
    vx = tx - x
    vy = ty - y
    distance = math.hypot(vx, vy)
    bearing = math.atan2(vy, vx)
    heading_error = normalize_angle(bearing - yaw)
    result: Dict[str, Any] = {
        "pass": True,
        "current_pose_x": x,
        "current_pose_y": y,
        "current_yaw_rad": yaw,
        "current_yaw_deg": math.degrees(yaw),
        "target_x": tx,
        "target_y": ty,
        "target_vector_x": vx,
        "target_vector_y": vy,
        "target_distance_m": distance,
        "target_bearing_global_rad": bearing,
        "target_bearing_global_deg": math.degrees(bearing),
        "heading_error_rad": heading_error,
        "heading_error_deg": math.degrees(heading_error),
        "corridor_alignment": "unknown",
        "target_aligned_with_corridor": None,
        "robot_aligned_with_corridor": None,
        "likely_issue": "unknown_corridor_heading",
        "note": "Provide --corridor-heading-rad to evaluate corridor alignment.",
    }
    if corridor_heading_rad is not None:
        robot_err = normalize_angle(yaw - corridor_heading_rad)
        target_err = normalize_angle(bearing - corridor_heading_rad)
        target_aligned = abs(target_err) <= 0.35
        robot_aligned = abs(robot_err) <= 0.35
        result.update(
            {
                "corridor_alignment": "evaluated",
                "corridor_heading_rad": corridor_heading_rad,
                "corridor_heading_deg": math.degrees(corridor_heading_rad),
                "robot_to_corridor_heading_error_rad": robot_err,
                "robot_to_corridor_heading_error_deg": math.degrees(robot_err),
                "target_to_corridor_heading_error_rad": target_err,
                "target_to_corridor_heading_error_deg": math.degrees(target_err),
                "target_aligned_with_corridor": target_aligned,
                "robot_aligned_with_corridor": robot_aligned,
                "likely_issue": classify_issue(robot_aligned, target_aligned, True),
                "note": None,
            }
        )
    return result


def write_outputs(summary: Dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    d = summary.get("diagnostic", {})
    lines = [
        "# Local Target Corridor Diagnostic Report",
        "",
        f"- current_yaw_rad: `{d.get('current_yaw_rad')}`",
        f"- current_yaw_deg: `{d.get('current_yaw_deg')}`",
        f"- target_bearing_global_rad: `{d.get('target_bearing_global_rad')}`",
        f"- target_bearing_global_deg: `{d.get('target_bearing_global_deg')}`",
        f"- heading_error_rad: `{d.get('heading_error_rad')}`",
        f"- heading_error_deg: `{d.get('heading_error_deg')}`",
        f"- corridor_heading_rad: `{d.get('corridor_heading_rad')}`",
        f"- corridor_heading_deg: `{d.get('corridor_heading_deg')}`",
        f"- robot_to_corridor_heading_error_deg: `{d.get('robot_to_corridor_heading_error_deg')}`",
        f"- target_to_corridor_heading_error_deg: `{d.get('target_to_corridor_heading_error_deg')}`",
        f"- target_aligned_with_corridor: `{d.get('target_aligned_with_corridor')}`",
        f"- robot_aligned_with_corridor: `{d.get('robot_aligned_with_corridor')}`",
        f"- likely_issue: `{d.get('likely_issue')}`",
        f"- note: `{d.get('note')}`",
        "",
        "## Interpretation",
        "",
        f"- target_sideways_relative_to_robot: `{abs(float(d.get('heading_error_rad') or 0.0)) > 1.0 if d.get('pass') else None}`",
        f"- corridor_alignment: `{d.get('corridor_alignment')}`",
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
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corridor-heading-rad", type=float, default=None)
    args = parser.parse_args()
    summary: Dict[str, Any] = {
        "forbidden_sources_used": [],
        "called_move_base": False,
        "sent_navigation_goal": False,
        "errors": [],
    }
    try:
        import rospy  # type: ignore

        rospy.init_node("local_target_corridor_diagnostic", anonymous=True, disable_signals=True)
        odom = read_odom(rospy)
        target = load_target()
        summary["odom"] = odom
        summary["target"] = target
        if not odom.get("pass"):
            summary["errors"].append("compliant_odom_unavailable")
        if not target.get("pass"):
            summary["errors"].append("target_unavailable_or_boundary_invalid")
        summary["diagnostic"] = compute(odom, target, args.corridor_heading_rad)
    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["diagnostic"] = {"pass": False}
    write_outputs(summary)
    print(json.dumps({"summary": str(OUT), "report": str(REPORT), "likely_issue": summary.get("diagnostic", {}).get("likely_issue")}))
    return 0 if not summary.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
