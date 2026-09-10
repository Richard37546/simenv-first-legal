#!/usr/bin/env python3
"""Validate coordinate frame contract for local subgoal target generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "debug" / "local_subgoal_runner_mvp" / "frame_contract_validation_summary.json"
REPORT_PATH = ROOT / "audit_reports" / "frame_contract_validation_report.md"
MANIFEST_PATH = ROOT / "generated_building" / "scene_manifest.json"
TARGET_PATH = ROOT / "debug" / "short_horizon_target_selection" / "short_horizon_target_override.json"
N5B_PATH = ROOT / "debug" / "short_horizon_target_selection" / "n5b_subgoal_audit_report.json"
PREFLIGHT_PATH = ROOT / "debug" / "l3zk_sync_acquire_replay" / "l3zk_navigation_preflight_candidates.json"

ODOM_TOPIC = "/team/livox/icp_odom_gated"
OFFICIAL_CORRIDOR_HEADING_WORLD = math.pi / 2.0


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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

    rospy.init_node("frame_contract_validation", anonymous=True, disable_signals=True)
    msg = rospy.wait_for_message(ODOM_TOPIC, Odometry, timeout=5.0)
    pose = msg.pose.pose
    yaw = yaw_from_quat(pose.orientation)
    return {
        "topic": ODOM_TOPIC,
        "header_frame_id": msg.header.frame_id,
        "child_frame_id": msg.child_frame_id,
        "stamp_sec": float(msg.header.stamp.to_sec()),
        "pose_x_y_yaw": [float(pose.position.x), float(pose.position.y), yaw],
        "current_pose_x": float(pose.position.x),
        "current_pose_y": float(pose.position.y),
        "current_yaw_rad": yaw,
        "current_yaw_deg": math.degrees(yaw),
    }


def ros_base_transform(local_xy: List[float], pose: List[float]) -> List[float]:
    x_base, y_base = float(local_xy[0]), float(local_xy[1])
    x_robot, y_robot, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    return [
        x_robot + math.cos(yaw) * x_base - math.sin(yaw) * y_base,
        y_robot + math.sin(yaw) * x_base + math.cos(yaw) * y_base,
    ]


def bev_lateral_forward_to_ros_base(local_xy: List[float]) -> List[float]:
    lateral = float(local_xy[0])
    forward = float(local_xy[1])
    return [forward, lateral]


def fixed_transform(local_xy: List[float], pose: List[float]) -> List[float]:
    return ros_base_transform(bev_lateral_forward_to_ros_base(local_xy), pose)


def bearing_metrics(pose: List[float], target_xy: List[float], corridor_odom: float) -> Dict[str, Any]:
    dx = float(target_xy[0]) - float(pose[0])
    dy = float(target_xy[1]) - float(pose[1])
    bearing = math.atan2(dy, dx)
    heading_error = wrap(bearing - float(pose[2]))
    target_to_corridor = wrap(bearing - corridor_odom)
    robot_to_corridor = wrap(float(pose[2]) - corridor_odom)
    return {
        "target_vector_x": dx,
        "target_vector_y": dy,
        "target_bearing_odom_rad": bearing,
        "target_bearing_odom_deg": math.degrees(bearing),
        "heading_error_pose_target_rad": heading_error,
        "heading_error_pose_target_deg": math.degrees(heading_error),
        "target_to_corridor_odom_error_rad": target_to_corridor,
        "target_to_corridor_odom_error_deg": math.degrees(target_to_corridor),
        "robot_to_corridor_odom_error_rad": robot_to_corridor,
        "robot_to_corridor_odom_error_deg": math.degrees(robot_to_corridor),
    }


def truncate_subgoal(pose: List[float], parent_xy: List[float], distance: float = 2.0) -> List[float]:
    dx = float(parent_xy[0]) - float(pose[0])
    dy = float(parent_xy[1]) - float(pose[1])
    dist = math.hypot(dx, dy)
    if dist <= 1e-9:
        return [float(parent_xy[0]), float(parent_xy[1])]
    scale = min(distance, dist) / dist
    return [float(pose[0]) + dx * scale, float(pose[1]) + dy * scale]


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


def find_parent_candidate(parent_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not parent_id or not PREFLIGHT_PATH.exists():
        return None
    candidates = flatten_candidates(read_json(PREFLIGHT_PATH))
    matches = [c for c in candidates if c.get("candidate_id") == parent_id and finite_xy(c.get("local_xy_base"))]
    feasible = [c for c in matches if c.get("feasible_for_navigation_input") is True]
    return (feasible or matches or [None])[-1]


def synthetic_result(name: str, local_xy: List[float], pose: List[float], corridor_odom: float, use_fixed: bool) -> Dict[str, Any]:
    target = fixed_transform(local_xy, pose) if use_fixed else ros_base_transform(local_xy, pose)
    result = {
        "name": name,
        "local_xy": local_xy,
        "transform_mode": "fixed_bev_lateral_forward_to_ros_base" if use_fixed else "legacy_ros_base_direct",
        "transformed_target_x": target[0],
        "transformed_target_y": target[1],
    }
    result.update(bearing_metrics(pose, target, corridor_odom))
    return result


def write_outputs(summary: Dict[str, Any]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Frame Contract Validation Report",
        "",
        f"- final_decision: `{summary.get('final_decision')}`",
        f"- frame_contract_validated: `{summary.get('frame_contract_validated')}`",
        f"- startup_anchor_quality: `{summary.get('startup_anchor_quality')}`",
        f"- official_corridor_heading_in_team_livox_odom_deg: `{summary.get('official_corridor_heading_in_team_livox_odom_deg')}`",
        "",
        "## Current Target",
        "",
        f"- target_frame: `{summary.get('target_declared_frame')}`",
        f"- target_bearing_odom_deg_before: `{summary.get('target_bearing_odom_deg_before')}`",
        f"- target_to_corridor_odom_error_deg_before: `{summary.get('target_to_corridor_odom_error_deg_before')}`",
        f"- target_frame_label_consistent_before: `{summary.get('target_frame_label_consistent_before')}`",
        "",
        "## Corrected Candidate Projection",
        "",
        f"- parent_candidate_id: `{summary.get('parent_candidate_id')}`",
        f"- parent_local_xy_base: `{summary.get('parent_local_xy_base')}`",
        f"- corrected_parent_target_xy: `{summary.get('corrected_parent_target_xy')}`",
        f"- corrected_subgoal_xy: `{summary.get('corrected_subgoal_xy')}`",
        f"- target_bearing_odom_deg_after: `{summary.get('target_bearing_odom_deg_after')}`",
        f"- target_to_corridor_odom_error_deg_after: `{summary.get('target_to_corridor_odom_error_deg_after')}`",
        f"- target_frame_label_consistent_after: `{summary.get('target_frame_label_consistent_after')}`",
        "",
        "## Synthetic Invariants",
        "",
    ]
    for item in summary.get("synthetic_invariants", []):
        lines += [
            f"### {item.get('name')} / {item.get('transform_mode')}",
            "",
            f"- transformed_target_x: `{item.get('transformed_target_x')}`",
            f"- transformed_target_y: `{item.get('transformed_target_y')}`",
            f"- target_bearing_odom_deg: `{item.get('target_bearing_odom_deg')}`",
            f"- target_to_corridor_odom_error_deg: `{item.get('target_to_corridor_odom_error_deg')}`",
            "",
        ]
    lines += [
        "## Boundary",
        "",
        f"- forbidden_sources_used: `{summary.get('forbidden_sources_used')}`",
        f"- called_move_base: `{summary.get('called_move_base')}`",
        f"- sent_navigation_goal: `{summary.get('sent_navigation_goal')}`",
        f"- cmd_vel_published: `{summary.get('cmd_vel_published')}`",
        f"- runner_main_logic_modified: `{summary.get('runner_main_logic_modified')}`",
        f"- git_add_or_commit: `{summary.get('git_add_or_commit')}`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    summary: Dict[str, Any] = {
        "forbidden_sources_used": [],
        "called_move_base": False,
        "sent_navigation_goal": False,
        "cmd_vel_published": False,
        "runner_main_logic_modified": False,
        "git_add_or_commit": False,
    }
    manifest = read_json(MANIFEST_PATH)
    target = read_json(TARGET_PATH)
    n5b = read_json(N5B_PATH) if N5B_PATH.exists() else {}
    odom = read_odom()
    pose = odom["pose_x_y_yaw"]
    world_start_yaw = float(manifest["robot_start"]["yaw"])
    world_to_odom_yaw_offset = wrap(float(pose[2]) - world_start_yaw)
    corridor_odom = wrap(OFFICIAL_CORRIDOR_HEADING_WORLD + world_to_odom_yaw_offset)

    summary.update(
        {
            "startup_anchor_quality": "approximate_current_start_sample",
            "world_start_x": float(manifest["robot_start"]["x"]),
            "world_start_y": float(manifest["robot_start"]["y"]),
            "world_start_yaw": world_start_yaw,
            "odom": odom,
            "odom_start_x": pose[0],
            "odom_start_y": pose[1],
            "odom_start_yaw": pose[2],
            "world_to_odom_yaw_offset": world_to_odom_yaw_offset,
            "world_to_odom_yaw_offset_deg": math.degrees(world_to_odom_yaw_offset),
            "official_corridor_heading_in_team_livox_odom_rad": corridor_odom,
            "official_corridor_heading_in_team_livox_odom_deg": math.degrees(corridor_odom),
            "target_declared_frame": target.get("target_frame"),
        }
    )

    current_target = target.get("target_xy_team_livox_odom")
    before = bearing_metrics(pose, current_target, corridor_odom) if finite_xy(current_target) else {}
    summary["current_target_xy"] = current_target
    summary["target_bearing_odom_deg_before"] = before.get("target_bearing_odom_deg")
    summary["target_to_corridor_odom_error_deg_before"] = before.get("target_to_corridor_odom_error_deg")
    summary["target_frame_label_consistent_before"] = bool(
        target.get("target_frame") == "team_livox_odom"
        and before.get("target_to_corridor_odom_error_deg") is not None
        and abs(float(before["target_to_corridor_odom_error_deg"])) <= 20.0
    )

    summary["synthetic_invariants"] = [
        synthetic_result("ros_base_forward", [2.0, 0.0], pose, corridor_odom, use_fixed=False),
        synthetic_result("bev_y_forward_legacy", [0.0, 2.0], pose, corridor_odom, use_fixed=False),
        synthetic_result("bev_y_forward_fixed", [0.0, 2.0], pose, corridor_odom, use_fixed=True),
    ]

    parent_id = n5b.get("parent_candidate_id") or target.get("parent_candidate_id")
    parent = find_parent_candidate(parent_id)
    summary["parent_candidate_id"] = parent_id
    summary["parent_local_xy_base"] = parent.get("local_xy_base") if parent else None
    if parent and finite_xy(parent.get("local_xy_base")):
        corrected_parent = fixed_transform(parent["local_xy_base"], pose)
        corrected_subgoal = truncate_subgoal(pose, corrected_parent, float(n5b.get("subgoal_distance_m") or 2.0))
        after = bearing_metrics(pose, corrected_subgoal, corridor_odom)
        summary.update(
            {
                "corrected_parent_target_xy": corrected_parent,
                "corrected_subgoal_xy": corrected_subgoal,
                "target_bearing_odom_deg_after": after.get("target_bearing_odom_deg"),
                "target_to_corridor_odom_error_deg_after": after.get("target_to_corridor_odom_error_deg"),
                "heading_error_pose_target_deg_after": after.get("heading_error_pose_target_deg"),
                "target_frame_label_consistent_after": abs(float(after["target_to_corridor_odom_error_deg"])) <= 20.0,
            }
        )
    else:
        summary.update(
            {
                "corrected_parent_target_xy": None,
                "corrected_subgoal_xy": None,
                "target_bearing_odom_deg_after": None,
                "target_to_corridor_odom_error_deg_after": None,
                "target_frame_label_consistent_after": False,
            }
        )

    summary["invariant_1_corridor_near_odom_zero"] = abs(math.degrees(corridor_odom)) <= 20.0
    summary["invariant_3_current_real_target_aligned"] = bool(summary["target_frame_label_consistent_before"])
    summary["invariant_4_n5b_same_frame_interpolation"] = bool(
        n5b.get("subgoal_generated") is True
        and finite_xy(n5b.get("parent_target_xy_team_livox_odom"))
        and finite_xy(n5b.get("subgoal_xy_team_livox_odom"))
    )
    summary["invariant_5_runner_same_frame_heading"] = target.get("target_frame") == odom.get("header_frame_id")
    summary["invariant_6_odom_forward_maps_world_y"] = summary["invariant_1_corridor_near_odom_zero"]
    summary["frame_contract_validated"] = bool(
        summary["invariant_1_corridor_near_odom_zero"]
        and summary["target_frame_label_consistent_after"]
        and summary["invariant_4_n5b_same_frame_interpolation"]
        and summary["invariant_5_runner_same_frame_heading"]
        and summary["invariant_6_odom_forward_maps_world_y"]
    )
    summary["final_decision"] = "FRAME_CONTRACT_VALIDATED" if summary["frame_contract_validated"] else "FRAME_CONTRACT_VALIDATION_FAILED"
    write_outputs(summary)
    return 0 if summary["frame_contract_validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
