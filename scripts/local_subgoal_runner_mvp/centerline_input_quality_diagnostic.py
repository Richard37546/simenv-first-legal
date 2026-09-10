#!/usr/bin/env python3
"""Read-only diagnostic for corridor-center input observability.

The script samples the same compliant runtime inputs used by the local runner
and answers a narrow question: when the robot should follow a corridor center,
do the available inputs expose enough lateral geometry to support that control?

It never publishes /cmd_vel and never reads Gazebo truth sources.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = ROOT / "debug" / "short_horizon_target_selection" / "short_horizon_target_override.json"
OUT_DIR = ROOT / "debug" / "centerline_input_quality"
REPORT_PATH = ROOT / "audit_reports" / "centerline_input_quality_diagnostic_report.md"

TOPIC_ODOM = "/team/livox/icp_odom_gated"
TOPIC_GRID = "/team/local_traversability_grid"
TOPIC_STATUS = "/team/traversability_status"


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def pose_tuple(msg: Odometry) -> Tuple[float, float, float]:
    pose = msg.pose.pose
    return float(pose.position.x), float(pose.position.y), yaw_from_quat(pose.orientation)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"_read_error": str(exc)}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def target_to_base(target_xy: Sequence[float], pose: Tuple[float, float, float]) -> Tuple[float, float]:
    dx = float(target_xy[0]) - pose[0]
    dy = float(target_xy[1]) - pose[1]
    c = math.cos(pose[2])
    s = math.sin(pose[2])
    return c * dx + s * dy, -s * dx + c * dy


def grid_array(msg: OccupancyGrid) -> np.ndarray:
    return np.array(msg.data, dtype=np.int16).reshape((int(msg.info.height), int(msg.info.width)))


def cell_to_local_xy(cell: Tuple[int, int], msg: OccupancyGrid) -> Tuple[float, float]:
    res = float(msg.info.resolution)
    ox = float(msg.info.origin.position.x)
    oy = float(msg.info.origin.position.y)
    row, col = cell
    return ox + (row + 0.5) * res, oy + (col + 0.5) * res


def local_rect_counts(
    grid_msg: OccupancyGrid,
    grid: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Dict[str, Any]:
    values: List[int] = []
    h, w = grid.shape
    for row in range(h):
        for col in range(w):
            x, y = cell_to_local_xy((row, col), grid_msg)
            if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                values.append(int(grid[row, col]))
    total = len(values)
    free = sum(1 for v in values if v == 0)
    occ = sum(1 for v in values if v == 100)
    unk = sum(1 for v in values if v == -1)
    return {
        "x_range_m": list(x_range),
        "y_range_m": list(y_range),
        "cell_count": total,
        "free_count": int(free),
        "occupied_count": int(occ),
        "unknown_count": int(unk),
        "free_ratio": float(free / total) if total else None,
        "occupied_ratio": float(occ / total) if total else None,
        "unknown_ratio": float(unk / total) if total else None,
    }


def estimate_grid_corridor_center(grid_msg: OccupancyGrid, grid: np.ndarray, args: argparse.Namespace) -> Dict[str, Any]:
    left_wall_y: List[float] = []
    right_wall_y: List[float] = []
    h, w = grid.shape
    for row in range(h):
        for col in range(w):
            if int(grid[row, col]) != 100:
                continue
            x, y = cell_to_local_xy((row, col), grid_msg)
            if not (args.center_x_min_m <= x <= args.center_x_max_m):
                continue
            if y >= args.wall_min_abs_y_m:
                left_wall_y.append(float(y))
            elif y <= -args.wall_min_abs_y_m:
                right_wall_y.append(float(y))

    result: Dict[str, Any] = {
        "x_range_m": [args.center_x_min_m, args.center_x_max_m],
        "wall_min_abs_y_m": args.wall_min_abs_y_m,
        "left_wall_cell_count": len(left_wall_y),
        "right_wall_cell_count": len(right_wall_y),
        "left_wall_inner_y_m": None,
        "right_wall_inner_y_m": None,
        "estimated_center_y_m": None,
        "estimated_width_m": None,
        "bilateral_wall_support": False,
        "centerline_observable": False,
        "reason": None,
    }
    if len(left_wall_y) < args.min_wall_count or len(right_wall_y) < args.min_wall_count:
        result["reason"] = "insufficient_bilateral_wall_support"
        return result

    left_inner = min(left_wall_y)
    right_inner = max(right_wall_y)
    width = left_inner - right_inner
    center_y = 0.5 * (left_inner + right_inner)
    result.update(
        {
            "left_wall_inner_y_m": float(left_inner),
            "right_wall_inner_y_m": float(right_inner),
            "estimated_center_y_m": float(center_y),
            "estimated_width_m": float(width),
            "bilateral_wall_support": True,
        }
    )
    if width < args.min_corridor_width_m or width > args.max_corridor_width_m:
        result["reason"] = "estimated_width_out_of_range"
        return result
    if abs(center_y) > args.max_center_abs_y_m:
        result["reason"] = "estimated_center_out_of_range"
        return result
    result["centerline_observable"] = True
    result["reason"] = "bilateral_wall_center_observable"
    return result


class Sampler:
    def __init__(self) -> None:
        self.odom_samples: List[Odometry] = []
        self.grid_samples: List[OccupancyGrid] = []
        self.status_samples: List[Dict[str, Any]] = []

    def odom_cb(self, msg: Odometry) -> None:
        self.odom_samples.append(msg)

    def grid_cb(self, msg: OccupancyGrid) -> None:
        self.grid_samples.append(msg)

    def status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            payload = {"_parse_error": str(exc)}
        self.status_samples.append(payload)


def summarize(args: argparse.Namespace, sampler: Sampler) -> Dict[str, Any]:
    target = read_json(TARGET_PATH)
    target_xy = target.get("target_xy_team_livox_odom")
    latest_odom = sampler.odom_samples[-1] if sampler.odom_samples else None
    latest_grid = sampler.grid_samples[-1] if sampler.grid_samples else None
    latest_status = sampler.status_samples[-1] if sampler.status_samples else {}

    pose = pose_tuple(latest_odom) if latest_odom else None
    target_base_xy = None
    heading_error_rad = None
    if (
        pose is not None
        and isinstance(target_xy, list)
        and len(target_xy) == 2
        and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in target_xy)
    ):
        target_base_xy = target_to_base(target_xy, pose)
        heading_error_rad = math.atan2(target_base_xy[1], max(target_base_xy[0], 1e-6))

    grid_summary: Dict[str, Any] = {"available": latest_grid is not None}
    center_estimate: Dict[str, Any] = {"centerline_observable": False, "reason": "grid_unavailable"}
    if latest_grid is not None:
        grid = grid_array(latest_grid)
        center_estimate = estimate_grid_corridor_center(latest_grid, grid, args)
        grid_summary = {
            "available": True,
            "frame_id": latest_grid.header.frame_id,
            "stamp_sec": float(latest_grid.header.stamp.to_sec()),
            "width": int(latest_grid.info.width),
            "height": int(latest_grid.info.height),
            "resolution_m": float(latest_grid.info.resolution),
            "unique_values": sorted(int(v) for v in np.unique(grid).tolist()),
            "value_counts": {
                "free": int((grid == 0).sum()),
                "occupied": int((grid == 100).sum()),
                "unknown": int((grid == -1).sum()),
                "other": int(((grid != 0) & (grid != 100) & (grid != -1)).sum()),
            },
            "front_center_counts": local_rect_counts(latest_grid, grid, (0.35, 1.35), (-0.30, 0.30)),
            "door_width_counts": local_rect_counts(latest_grid, grid, (0.60, 1.60), (-0.75, 0.75)),
            "left_wall_band_counts": local_rect_counts(
                latest_grid, grid, (args.center_x_min_m, args.center_x_max_m), (args.wall_min_abs_y_m, 1.50)
            ),
            "right_wall_band_counts": local_rect_counts(
                latest_grid, grid, (args.center_x_min_m, args.center_x_max_m), (-1.50, -args.wall_min_abs_y_m)
            ),
        }

    target_lateral_abs = abs(target_base_xy[1]) if target_base_xy is not None else None
    target_says_centered = target_lateral_abs is not None and target_lateral_abs <= args.target_centered_abs_y_m
    input_quality_class = "unknown"
    if latest_grid is None or latest_odom is None:
        input_quality_class = "missing_required_inputs"
    elif center_estimate.get("centerline_observable"):
        input_quality_class = "centerline_observable_from_grid"
    elif target_says_centered:
        input_quality_class = "target_centered_but_grid_centerline_unobservable"
    else:
        input_quality_class = "grid_centerline_unobservable"

    return {
        "final_decision": "CENTERLINE_INPUT_QUALITY_DIAGNOSTIC_COMPLETE",
        "diagnostic_only": True,
        "published_cmd_vel": False,
        "forbidden_sources_used": [],
        "called_move_base": False,
        "sent_navigation_goal": False,
        "topics_read": [TOPIC_ODOM, TOPIC_GRID, TOPIC_STATUS],
        "sample_duration_sec": args.sample_sec,
        "odom_message_count": len(sampler.odom_samples),
        "grid_message_count": len(sampler.grid_samples),
        "status_message_count": len(sampler.status_samples),
        "target_source": target.get("source"),
        "target_subgoal_source": target.get("subgoal_source"),
        "target_xy_team_livox_odom": target_xy,
        "current_pose_x_y_yaw": list(pose) if pose is not None else None,
        "target_base_xy": list(target_base_xy) if target_base_xy is not None else None,
        "target_lateral_abs_y_m": target_lateral_abs,
        "target_says_centered": target_says_centered,
        "heading_error_rad": heading_error_rad,
        "heading_error_deg": math.degrees(heading_error_rad) if heading_error_rad is not None else None,
        "status_payload": latest_status,
        "grid_summary": grid_summary,
        "grid_corridor_center_estimate": center_estimate,
        "input_quality_classification": input_quality_class,
        "interpretation": interpret(input_quality_class),
    }


def interpret(input_quality_class: str) -> str:
    if input_quality_class == "centerline_observable_from_grid":
        return "Grid contains enough bilateral wall evidence; controller should be able to use grid-derived lateral error."
    if input_quality_class == "target_centered_but_grid_centerline_unobservable":
        return "Target line appears centered, but grid cannot independently observe corridor center; controller may believe it is centered while the robot is visually drifting."
    if input_quality_class == "grid_centerline_unobservable":
        return "Grid does not expose stable corridor-center geometry; tune or augment perception before further controller tuning."
    if input_quality_class == "missing_required_inputs":
        return "Required odom/grid/status input is missing; runtime chain is not ready for centerline control."
    return "Insufficient evidence for a reliable classification."


def write_report(summary: Dict[str, Any]) -> None:
    center = summary.get("grid_corridor_center_estimate", {})
    lines = [
        "# Centerline Input Quality Diagnostic",
        "",
        f"- final_decision: `{summary.get('final_decision')}`",
        f"- input_quality_classification: `{summary.get('input_quality_classification')}`",
        f"- interpretation: {summary.get('interpretation')}",
        f"- odom_message_count: `{summary.get('odom_message_count')}`",
        f"- grid_message_count: `{summary.get('grid_message_count')}`",
        f"- status_message_count: `{summary.get('status_message_count')}`",
        f"- target_base_xy: `{summary.get('target_base_xy')}`",
        f"- target_says_centered: `{summary.get('target_says_centered')}`",
        f"- centerline_observable: `{center.get('centerline_observable')}`",
        f"- center_reason: `{center.get('reason')}`",
        f"- left_wall_cell_count: `{center.get('left_wall_cell_count')}`",
        f"- right_wall_cell_count: `{center.get('right_wall_cell_count')}`",
        f"- estimated_center_y_m: `{center.get('estimated_center_y_m')}`",
        f"- estimated_width_m: `{center.get('estimated_width_m')}`",
        "",
        "## Boundary",
        "",
        "- published_cmd_vel: `false`",
        "- called_move_base: `false`",
        "- sent_navigation_goal: `false`",
        "- forbidden_sources_used: `[]`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sec", type=float, default=3.0)
    parser.add_argument("--center-x-min-m", type=float, default=0.80)
    parser.add_argument("--center-x-max-m", type=float, default=2.40)
    parser.add_argument("--wall-min-abs-y-m", type=float, default=0.45)
    parser.add_argument("--min-wall-count", type=int, default=6)
    parser.add_argument("--min-corridor-width-m", type=float, default=0.90)
    parser.add_argument("--max-corridor-width-m", type=float, default=2.80)
    parser.add_argument("--max-center-abs-y-m", type=float, default=0.45)
    parser.add_argument("--target-centered-abs-y-m", type=float, default=0.20)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rospy.init_node("centerline_input_quality_diagnostic", anonymous=True, disable_signals=True)
    sampler = Sampler()
    rospy.Subscriber(TOPIC_ODOM, Odometry, sampler.odom_cb, queue_size=100)
    rospy.Subscriber(TOPIC_GRID, OccupancyGrid, sampler.grid_cb, queue_size=20)
    rospy.Subscriber(TOPIC_STATUS, String, sampler.status_cb, queue_size=20)

    deadline = time.monotonic() + max(0.1, args.sample_sec)
    rate = rospy.Rate(20.0)
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        rate.sleep()

    summary = summarize(args, sampler)
    out_path = OUT_DIR / "centerline_input_quality_summary.json"
    write_json(out_path, summary)
    write_report(summary)
    print(json.dumps({"final_decision": summary["final_decision"], "summary": str(out_path), "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
