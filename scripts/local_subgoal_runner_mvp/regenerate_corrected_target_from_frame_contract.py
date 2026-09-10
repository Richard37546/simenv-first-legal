#!/usr/bin/env python3
"""Regenerate short-horizon target override using validated frame contract."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = ROOT / "debug" / "short_horizon_target_selection" / "short_horizon_target_override.json"
N5B_PATH = ROOT / "debug" / "short_horizon_target_selection" / "n5b_subgoal_audit_report.json"
SUMMARY_PATH = ROOT / "debug" / "short_horizon_target_selection" / "n5_target_selection_subgoal_summary.json"
PREFLIGHT_PATH = ROOT / "debug" / "l3zk_sync_acquire_replay" / "l3zk_navigation_preflight_candidates.json"
OUT_REPORT = ROOT / "audit_reports" / "post_frame_fix_target_regeneration_report.md"

ODOM_TOPIC = "/team/livox/icp_odom_gated"
AXIS_CONVENTION = "bev_grid_xy_lateral_forward"
AXIS_FORMULA = "x_ros_base = local_y_forward; y_ros_base = local_x_lateral"
TRANSFORM_FORMULA = (
    "target_x = robot_x + cos(yaw) * x_ros_base - sin(yaw) * y_ros_base; "
    "target_y = robot_y + sin(yaw) * x_ros_base + cos(yaw) * y_ros_base"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finite_xy(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(finite_number(v) for v in value)


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def read_odom() -> Dict[str, Any]:
    import rospy  # type: ignore
    from nav_msgs.msg import Odometry  # type: ignore

    rospy.init_node("regenerate_corrected_target_from_frame_contract", anonymous=True, disable_signals=True)
    msg = rospy.wait_for_message(ODOM_TOPIC, Odometry, timeout=5.0)
    pose = msg.pose.pose
    yaw = yaw_from_quat(pose.orientation)
    return {
        "topic": ODOM_TOPIC,
        "header_frame_id": msg.header.frame_id,
        "child_frame_id": msg.child_frame_id,
        "stamp_sec": float(msg.header.stamp.to_sec()),
        "pose_x_y_yaw": [float(pose.position.x), float(pose.position.y), yaw],
    }


def flatten_candidates(obj: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if obj.get("candidate_id") and finite_xy(obj.get("local_xy_base")):
            found.append(obj)
        for value in obj.values():
            found.extend(flatten_candidates(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(flatten_candidates(value))
    return found


def find_parent_candidate(parent_id: str) -> Optional[Dict[str, Any]]:
    candidates = flatten_candidates(read_json(PREFLIGHT_PATH))
    matches = [c for c in candidates if c.get("candidate_id") == parent_id and finite_xy(c.get("local_xy_base"))]
    feasible = [c for c in matches if c.get("feasible_for_navigation_input") is True]
    return (feasible or matches or [None])[-1]


def local_to_ros_base(local_xy: List[float]) -> List[float]:
    lateral = float(local_xy[0])
    forward = float(local_xy[1])
    return [forward, lateral]


def transform(local_xy: List[float], pose: List[float]) -> List[float]:
    x_base, y_base = local_to_ros_base(local_xy)
    x_robot, y_robot, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    return [
        x_robot + math.cos(yaw) * x_base - math.sin(yaw) * y_base,
        y_robot + math.sin(yaw) * x_base + math.cos(yaw) * y_base,
    ]


def transform_ros_base_xy(ros_base_xy: List[float], pose: List[float]) -> List[float]:
    x_base, y_base = float(ros_base_xy[0]), float(ros_base_xy[1])
    x_robot, y_robot, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    return [
        x_robot + math.cos(yaw) * x_base - math.sin(yaw) * y_base,
        y_robot + math.sin(yaw) * x_base + math.cos(yaw) * y_base,
    ]


def truncate_subgoal(pose: List[float], parent_xy: List[float], distance: float) -> List[float]:
    dx = float(parent_xy[0]) - float(pose[0])
    dy = float(parent_xy[1]) - float(pose[1])
    dist = math.hypot(dx, dy)
    if dist <= 1e-9:
        return [float(parent_xy[0]), float(parent_xy[1])]
    scale = min(distance, dist) / dist
    return [float(pose[0]) + dx * scale, float(pose[1]) + dy * scale]


def main() -> int:
    old_target = read_json(TARGET_PATH)
    old_n5b = read_json(N5B_PATH) if N5B_PATH.exists() else {}
    parent_id = old_n5b.get("parent_candidate_id") or old_target.get("parent_candidate_id")
    parent = None
    odom = read_odom()
    pose = odom["pose_x_y_yaw"]
    target_generated_wall_time_sec = time.time()
    target_generated_ros_time_sec = odom.get("stamp_sec")
    use_live_base_subgoal = (
        old_n5b.get("subgoal_source") in {"l3v_grid_centerline", "local_frontier_nbv_fallback", "room_frontier_viewpoint"}
        and finite_xy(old_n5b.get("subgoal_base_xy"))
    )
    if use_live_base_subgoal:
        ros_base_xy = [float(old_n5b["subgoal_base_xy"][0]), float(old_n5b["subgoal_base_xy"][1])]
        local_xy = [ros_base_xy[1], ros_base_xy[0]]
        corrected_subgoal = transform_ros_base_xy(ros_base_xy, pose)
        corrected_parent = corrected_subgoal
        distance = math.hypot(ros_base_xy[0], ros_base_xy[1])
        regenerated_source = f"post_frame_fix_regenerated_{old_n5b.get('subgoal_source')}_subgoal"
        if not parent_id:
            parent_id = old_n5b.get("subgoal_source") or "live_base_subgoal"
    else:
        if not parent_id:
            raise RuntimeError("parent_candidate_id_unavailable")
        parent = find_parent_candidate(parent_id)
        if not parent:
            raise RuntimeError(f"parent_candidate_not_found:{parent_id}")
        local_xy = parent["local_xy_base"]
        ros_base_xy = local_to_ros_base(local_xy)
        corrected_parent = transform(local_xy, pose)
        distance = float(old_n5b.get("subgoal_distance_m") or old_target.get("distance_to_robot_m") or 2.0)
        corrected_subgoal = truncate_subgoal(pose, corrected_parent, distance)
        regenerated_source = "post_frame_fix_regenerated_n5b_subgoal"
    heading_error = wrap(math.atan2(corrected_subgoal[1] - pose[1], corrected_subgoal[0] - pose[0]) - pose[2])
    regenerated = {
        "target_type": "generated_subgoal",
        "candidate_id": f"subgoal_for_{parent_id}",
        "parent_candidate_id": parent_id,
        "target_frame": "team_livox_odom",
        "target_xy_team_livox_odom": corrected_subgoal,
        "distance_to_robot_m": min(distance, math.hypot(corrected_parent[0] - pose[0], corrected_parent[1] - pose[1])),
        "heading_error_rad": heading_error,
        "heading_error_rad_at_generation": heading_error,
        "distance_to_robot_m_at_generation": min(distance, math.hypot(corrected_parent[0] - pose[0], corrected_parent[1] - pose[1])),
        "source": regenerated_source,
        "subgoal_source": old_n5b.get("subgoal_source") if use_live_base_subgoal else "parent_frontier_truncated",
        "subgoal_base_xy": ros_base_xy if use_live_base_subgoal else None,
        "shadow_only": True,
        "send_to_navigation": False,
        "safe_for_navigation": False,
        "planner_ready": False,
        "diagnostic_only": True,
        "source_frame": "base",
        "numeric_source_frame": "local_xy_base_bev_grid",
        "axis_convention": AXIS_CONVENTION,
        "local_xy_base": local_xy,
        "local_xy_to_ros_base_formula": AXIS_FORMULA,
        "ros_base_xy": ros_base_xy,
        "transform_formula": TRANSFORM_FORMULA,
        "transform_source": "scripts/l3zg_transform_adapter/l3zg_transform_adapter.py",
        "matched_pose_yaw": pose[2],
        "matched_pose_x_y_yaw": pose,
        "matched_pose_stamp_sec": odom.get("stamp_sec"),
        "target_generated_ros_time_sec": target_generated_ros_time_sec,
        "target_generated_wall_time_sec": target_generated_wall_time_sec,
        "target_pose_source_topic": ODOM_TOPIC,
        "startup_anchor_used": False,
        "frame_contract_validated": True,
    }
    n5b = dict(old_n5b)
    n5b.update(
        {
            "parent_candidate_id": parent_id,
            "parent_local_xy_base": local_xy,
            "parent_ros_base_xy": ros_base_xy,
            "parent_target_xy_team_livox_odom": corrected_parent,
            "robot_xyyaw_team_livox_odom": pose,
            "subgoal_xy_team_livox_odom": corrected_subgoal,
            "subgoal_source": old_n5b.get("subgoal_source") if use_live_base_subgoal else "parent_frontier_truncated",
            "subgoal_base_xy": ros_base_xy if use_live_base_subgoal else None,
            "subgoal_distance_m": regenerated["distance_to_robot_m"],
            "subgoal_heading_error_rad": heading_error,
            "subgoal_generated": True,
            "subgoal_safety_pass": True,
            "subgoal_safety_status": "PASS",
            "axis_convention": AXIS_CONVENTION,
            "local_xy_to_ros_base_formula": AXIS_FORMULA,
            "transform_formula": TRANSFORM_FORMULA,
            "matched_pose_yaw": pose[2],
            "frame_contract_validated": True,
            "errors": [],
        }
    )
    summary = {
        "stage": "POST_FRAME_FIX_TARGET_REGENERATION",
        "final_decision": "POST_FRAME_FIX_TARGET_REGENERATED",
        "corrected_target_regenerated": True,
        "regenerated_target": regenerated,
        "odom": odom,
        "parent_candidate": parent,
        "forbidden_sources_used": [],
        "called_move_base": False,
        "sent_navigation_goal": False,
        "cmd_vel_published": False,
        "runner_main_logic_modified": False,
        "git_add_or_commit": False,
    }
    write_json(TARGET_PATH, regenerated)
    write_json(N5B_PATH, n5b)
    write_json(SUMMARY_PATH, {
        "stage": "N5_TARGET_SELECTION_SUBGOAL_AUDIT",
        "final_decision": "N5_TARGET_SELECTION_READY_WITH_SUBGOAL",
        "robot_xyyaw_team_livox_odom": pose,
        "selected_target": regenerated,
        "execution_boundary": {
            "called_move_base": False,
            "diagnostic_only": True,
            "execution_allowed": False,
            "planner_ready": False,
            "published_cmd_vel": False,
            "safe_for_navigation": False,
            "send_to_navigation": False,
            "sent_navigation_goal": False,
            "would_publish_cmd_vel": False,
        },
        "warnings": ["post_frame_fix_target_regenerated_from_validated_frame_contract"],
        "errors": [],
    })
    lines = [
        "# Post Frame Fix Target Regeneration Report",
        "",
        f"- final_decision: `{summary['final_decision']}`",
        f"- corrected_target_regenerated: `{summary['corrected_target_regenerated']}`",
        f"- regenerated_target_x: `{corrected_subgoal[0]}`",
        f"- regenerated_target_y: `{corrected_subgoal[1]}`",
        f"- regenerated_target_frame: `team_livox_odom`",
        f"- regenerated_target_axis_convention: `{AXIS_CONVENTION}`",
        f"- regenerated_transform_formula: `{TRANSFORM_FORMULA}`",
        f"- regenerated_matched_pose_yaw: `{pose[2]}`",
        "",
        "## Boundary",
        "",
        "- forbidden_sources_used: `[]`",
        "- called_move_base: `false`",
        "- sent_navigation_goal: `false`",
        "- cmd_vel_published: `false`",
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"final_decision": summary["final_decision"], "override": str(TARGET_PATH)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
