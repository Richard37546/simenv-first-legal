#!/usr/bin/env python3
"""Read-only audit for the locked doorway entry target.

This script does not publish cmd_vel, does not call move_base, and does not use
Gazebo truth. It checks whether the currently locked room-entry target is
geometrically reachable in the live local traversability grid.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry


ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = ROOT / "debug" / "short_horizon_target_selection" / "short_horizon_target_override.json"
DOORWAY_PATH = ROOT / "debug" / "doorway_candidate_detector" / "latest_doorway_candidate.json"
RUNNER_SUMMARY_PATH = ROOT / "debug" / "block_astar_dwa_mature" / "block_astar_dwa_mature_summary.json"
SUMMARY_PATH = ROOT / "debug" / "door_entry_target_audit" / "door_entry_target_audit_summary.json"
REPORT_PATH = ROOT / "audit_reports" / "door_entry_target_audit_report.md"

ODOM_TOPIC = "/team/livox/icp_odom_gated"
GRID_TOPIC = "/team/local_traversability_grid"


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc), "_path": str(path)}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_tuple(msg: Odometry) -> Tuple[float, float, float]:
    pose = msg.pose.pose
    return float(pose.position.x), float(pose.position.y), yaw_from_quat(pose.orientation)


def target_base_xy(target_xy: Sequence[float], pose: Tuple[float, float, float]) -> Tuple[float, float]:
    dx = float(target_xy[0]) - pose[0]
    dy = float(target_xy[1]) - pose[1]
    yaw = pose[2]
    x_base = math.cos(yaw) * dx + math.sin(yaw) * dy
    y_base = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return x_base, y_base


def grid_array(msg: OccupancyGrid) -> np.ndarray:
    return np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))


def local_xy_to_cell(x: float, y: float, msg: OccupancyGrid) -> Tuple[int, int]:
    res = float(msg.info.resolution)
    ox = float(msg.info.origin.position.x)
    oy = float(msg.info.origin.position.y)
    row = int(math.floor((x - ox) / res))
    col = int(math.floor((y - oy) / res))
    return row, col


def cell_window_counts(grid: np.ndarray, cell: Tuple[int, int], radius_cells: int = 2) -> Dict[str, Any]:
    row, col = cell
    r0 = max(0, row - radius_cells)
    r1 = min(grid.shape[0], row + radius_cells + 1)
    c0 = max(0, col - radius_cells)
    c1 = min(grid.shape[1], col + radius_cells + 1)
    if row < 0 or col < 0 or row >= grid.shape[0] or col >= grid.shape[1]:
        return {"in_grid": False, "cell": [row, col]}
    window = grid[r0:r1, c0:c1]
    count = int(window.size)
    free = int((window == 0).sum())
    occupied = int((window == 100).sum())
    unknown = int((window == -1).sum())
    return {
        "in_grid": True,
        "cell": [row, col],
        "cell_count": count,
        "free_count": free,
        "occupied_count": occupied,
        "unknown_count": unknown,
        "free_ratio": float(free / count) if count else None,
        "blocked_ratio": float((occupied + unknown) / count) if count else None,
        "raw_cell_value": int(grid[row, col]),
    }


def inflate_blocked(grid: np.ndarray, robot_radius_m: float, resolution_m: float, treat_unknown_as_blocked: bool) -> np.ndarray:
    blocked = grid == 100
    if treat_unknown_as_blocked:
        blocked = blocked | (grid == -1)
    inflated = blocked.copy()
    radius = max(0, int(math.ceil(robot_radius_m / resolution_m)))
    rows, cols = np.where(blocked)
    for row, col in zip(rows.tolist(), cols.tolist()):
        r0 = max(0, row - radius)
        r1 = min(grid.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(grid.shape[1], col + radius + 1)
        inflated[r0:r1, c0:c1] = True
    return inflated


def sample_line(
    grid: np.ndarray,
    msg: OccupancyGrid,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    inflated: np.ndarray,
    sample_step_m: float,
) -> Dict[str, Any]:
    dist = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    steps = max(1, int(math.ceil(dist / max(sample_step_m, 1e-3))))
    samples: List[Dict[str, Any]] = []
    blocked_count = 0
    out_of_grid_count = 0
    for idx in range(steps + 1):
        t = idx / steps
        x = start_xy[0] + (end_xy[0] - start_xy[0]) * t
        y = start_xy[1] + (end_xy[1] - start_xy[1]) * t
        row, col = local_xy_to_cell(x, y, msg)
        in_grid = 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
        value = int(grid[row, col]) if in_grid else None
        inflated_blocked = bool(in_grid and inflated[row, col])
        if not in_grid:
            out_of_grid_count += 1
        if inflated_blocked or not in_grid:
            blocked_count += 1
        samples.append(
            {
                "index": idx,
                "x_base": float(x),
                "y_base": float(y),
                "cell": [row, col],
                "in_grid": in_grid,
                "value": value,
                "inflated_blocked": inflated_blocked,
            }
        )
    return {
        "distance_m": float(dist),
        "sample_count": len(samples),
        "blocked_or_out_of_grid_count": int(blocked_count),
        "out_of_grid_count": int(out_of_grid_count),
        "pass": blocked_count == 0,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-timeout-sec", type=float, default=5.0)
    parser.add_argument("--robot-radius-m", type=float, default=0.05)
    parser.add_argument("--treat-unknown-as-blocked", action="store_true", default=True)
    parser.add_argument("--line-sample-step-m", type=float, default=0.05)
    parser.add_argument("--min-target-x-base-m", type=float, default=0.15)
    args = parser.parse_args()

    rospy.init_node("door_entry_target_audit", anonymous=True, disable_signals=True)
    start_wall = time.monotonic()
    target = read_json(TARGET_PATH)
    doorway = read_json(DOORWAY_PATH)
    runner = read_json(RUNNER_SUMMARY_PATH)
    odom = rospy.wait_for_message(ODOM_TOPIC, Odometry, timeout=args.input_timeout_sec)
    grid_msg = rospy.wait_for_message(GRID_TOPIC, OccupancyGrid, timeout=args.input_timeout_sec)
    pose = pose_tuple(odom)
    grid = grid_array(grid_msg)
    resolution = float(grid_msg.info.resolution)
    inflated = inflate_blocked(grid, args.robot_radius_m, resolution, args.treat_unknown_as_blocked)

    target_xy = target.get("target_xy_team_livox_odom")
    target_valid = isinstance(target_xy, list) and len(target_xy) == 2 and all(finite_number(v) for v in target_xy)
    base_xy: Optional[Tuple[float, float]] = target_base_xy(target_xy, pose) if target_valid else None
    target_cell_counts: Optional[Dict[str, Any]] = None
    line_check: Optional[Dict[str, Any]] = None
    target_in_front = False
    target_blocked_by_grid = None
    if base_xy is not None:
        target_in_front = base_xy[0] >= args.min_target_x_base_m
        target_cell = local_xy_to_cell(base_xy[0], base_xy[1], grid_msg)
        target_cell_counts = cell_window_counts(grid, target_cell, radius_cells=2)
        if target_cell_counts.get("in_grid"):
            row, col = target_cell
            target_blocked_by_grid = bool(inflated[row, col])
        line_check = sample_line(grid, grid_msg, (0.0, 0.0), base_xy, inflated, args.line_sample_step_m)

    runner_failure = runner.get("final_decision")
    runner_failure_diagnostic = runner.get("failure_diagnostic") if isinstance(runner.get("failure_diagnostic"), dict) else {}
    likely_issue = "unknown"
    if not target_valid:
        likely_issue = "target_missing_or_invalid"
    elif base_xy is not None and not target_in_front:
        likely_issue = "target_not_in_front_of_robot"
    elif target_blocked_by_grid is True:
        likely_issue = "doorway_commit_target_blocked_in_local_grid"
    elif line_check and not line_check.get("pass"):
        likely_issue = "doorway_commit_path_blocked_or_out_of_grid"
    elif runner_failure == "BLOCK_ASTAR_DWA_BLOCKED_NO_PATH":
        likely_issue = "astar_no_path_despite_point_line_check"
    else:
        likely_issue = "target_geometry_appears_passable_check_control_layer"

    summary = {
        "final_decision": "DOOR_ENTRY_TARGET_AUDIT_COMPLETE",
        "target_source": target.get("source"),
        "target_subgoal_source": target.get("subgoal_source"),
        "door_side": target.get("door_side"),
        "doorway_commit_depth_m": target.get("doorway_commit_depth_m"),
        "doorway_commit_yaw_rad": target.get("doorway_commit_yaw_rad"),
        "doorway_entry_xy_odom": target.get("doorway_entry_xy_odom"),
        "target_xy_team_livox_odom": target_xy,
        "robot_pose_x_y_yaw": list(pose),
        "target_base_xy": list(base_xy) if base_xy is not None else None,
        "target_distance_base_m": float(math.hypot(base_xy[0], base_xy[1])) if base_xy else None,
        "target_heading_base_rad": float(math.atan2(base_xy[1], max(base_xy[0], 1e-6))) if base_xy else None,
        "target_in_front": bool(target_in_front),
        "target_cell_counts": target_cell_counts,
        "target_blocked_by_inflated_grid": target_blocked_by_grid,
        "line_of_travel_check": line_check,
        "runner_final_decision": runner_failure,
        "runner_goal_block": runner_failure_diagnostic.get("goal_block"),
        "runner_goal_window": runner_failure_diagnostic.get("goal_window"),
        "doorway_latest_final_decision": doorway.get("final_decision"),
        "doorway_latest_stage_gate_active": doorway.get("stage_gate_active"),
        "grid_frame_id": grid_msg.header.frame_id,
        "grid_stamp_sec": float(grid_msg.header.stamp.to_sec()),
        "grid_value_counts": {
            "free": int((grid == 0).sum()),
            "occupied": int((grid == 100).sum()),
            "unknown": int((grid == -1).sum()),
        },
        "likely_issue": likely_issue,
        "forbidden_sources_used": [],
        "used_gazebo_truth": False,
        "called_move_base": False,
        "sent_navigation_goal": False,
        "cmd_vel_published": False,
        "wall_duration_sec": float(time.monotonic() - start_wall),
    }
    write_json(SUMMARY_PATH, summary)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Door Entry Target Audit",
        "",
        f"- final_decision: `{summary['final_decision']}`",
        f"- likely_issue: `{summary['likely_issue']}`",
        f"- target_source: `{summary.get('target_source')}`",
        f"- door_side: `{summary.get('door_side')}`",
        f"- doorway_commit_depth_m: `{summary.get('doorway_commit_depth_m')}`",
        f"- target_base_xy: `{summary.get('target_base_xy')}`",
        f"- target_in_front: `{summary.get('target_in_front')}`",
        f"- target_blocked_by_inflated_grid: `{summary.get('target_blocked_by_inflated_grid')}`",
        f"- line_of_travel_pass: `{(summary.get('line_of_travel_check') or {}).get('pass')}`",
        f"- runner_final_decision: `{summary.get('runner_final_decision')}`",
        f"- forbidden_sources_used: `{summary.get('forbidden_sources_used')}`",
        f"- cmd_vel_published: `{summary.get('cmd_vel_published')}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"final_decision": summary["final_decision"], "likely_issue": likely_issue, "summary": str(SUMMARY_PATH)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
