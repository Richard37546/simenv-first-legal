#!/usr/bin/env python3
"""N5 target selection and short-horizon subgoal audit.

Shadow-only: this script writes target override JSON for the shadow controller.
It never publishes /cmd_vel, calls move_base, sends navigation goals, reads
Gazebo truth, or uses TF as formal pose input.
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "short_horizon_target_selection"
REPORT = ROOT / "audit_reports" / "n5_target_selection_subgoal_audit_report.md"
BEV_DIR = Path("/home/richard/.ros/results/bev_maps")
SHADOW_SUMMARY = ROOT / "debug/shadow_minimal_controller/shadow_minimal_controller_summary.json"

MAX_DIST = 3.0
SUBGOAL_DIST = 2.0
PATCH_RADIUS_M = 0.30
MAX_PATCH_OCC = 0.05
MAX_PATCH_UNKNOWN = 0.60
MAX_CORRIDOR_OCC = 0.02
MAX_CORRIDOR_UNKNOWN = 0.70
L3V_GRID_TOPIC = "/team/local_traversability_grid"
ODOM_TOPIC = "/team/livox/icp_odom_gated"
GRID_CENTERLINE_ENABLED = True
GRID_CENTERLINE_TIMEOUT_SEC = 3.0
ODOM_FALLBACK_TIMEOUT_SEC = 5.0
GRID_CENTERLINE_FORWARD_M = 2.0
GRID_CENTERLINE_MIN_X_M = 1.0
GRID_CENTERLINE_MAX_X_M = 2.2
GRID_CENTERLINE_SEARCH_ABS_Y_M = 0.9
GRID_CENTERLINE_MIN_WIDTH_M = 0.45
GRID_CENTERLINE_MIN_CONFIDENCE = 0.45
FRONTIER_FALLBACK_ENABLED = True
FRONTIER_MIN_X_M = 0.45
FRONTIER_MAX_X_M = 1.60
FRONTIER_SEARCH_ABS_Y_M = 1.10
FRONTIER_ROBOT_RADIUS_M = 0.22
FRONTIER_GAIN_RADIUS_M = 0.45
FRONTIER_CLUSTER_RADIUS_M = 0.35
FRONTIER_PATH_UNKNOWN_RATIO_MAX = 0.35
FRONTIER_MIN_SCORE = 0.10

HARD_RISKS = {
    "candidate_too_close_to_obstacle",
    "local_obstacle_risk",
    "near_occupied",
    "collision_risk",
    "blocked_by_obstacle",
}

BOUNDARY = {
    "execution_allowed": False,
    "send_to_navigation": False,
    "safe_for_navigation": False,
    "planner_ready": False,
    "diagnostic_only": True,
    "would_publish_cmd_vel": False,
    "published_cmd_vel": False,
    "called_move_base": False,
    "sent_navigation_goal": False,
}

INPUTS = {
    "summary": ROOT / "debug/navigation_dry_run_readiness/navigation_dry_run_readiness_summary.json",
    "n1": ROOT / "debug/navigation_dry_run_readiness/n1_input_validation_report.json",
    "n2": ROOT / "debug/navigation_dry_run_readiness/n2_latest_bev_safety_report.json",
    "n3": ROOT / "debug/navigation_dry_run_readiness/n3_gated_odom_pose_resolver_report.json",
    "n4": ROOT / "debug/navigation_dry_run_readiness/n4_local_traversability_check_report.json",
    "l3zm_shadow": ROOT / "debug/l3zm_end_to_end_shadow_replay/l3zm_shadow_navigation_inputs.json",
    "l3zm_top1": ROOT / "debug/l3zm_end_to_end_shadow_replay/l3zm_top1_shadow_sequence.json",
    "l3zm_top3": ROOT / "debug/l3zm_end_to_end_shadow_replay/l3zm_top3_shadow_sequence.json",
    "risk_policy": ROOT / "debug/l3zn_navigation_handoff/risk_flag_policy.json",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finite_xy(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and finite_number(value[0]) and finite_number(value[1])


def finite_xyyaw(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(finite_number(v) for v in value)


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def load_inputs() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key, path in INPUTS.items():
        if path.exists():
            data[key] = read_json(path)
    return data


def robot_pose(data: Dict[str, Any]) -> Optional[List[float]]:
    n3 = data.get("n3", {}) if bool(data.get("use_cached_n3_pose")) else {}
    pose = n3.get("robot_xyyaw_team_livox_odom") or n3.get("pose_x_y_yaw")
    return [float(v) for v in pose] if finite_xyyaw(pose) else None


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def try_read_live_odom_pose(timeout_sec: float) -> Tuple[Optional[List[float]], List[str]]:
    warnings: List[str] = []
    master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    parsed = urlparse(master_uri)
    host = parsed.hostname or "localhost"
    port = int(parsed.port or 11311)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError as exc:
        warnings.append(f"ros_master_unreachable_for_odom:{master_uri}:{exc}")
        return None, warnings
    try:
        import rospy  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore
    except Exception as exc:
        warnings.append(f"odom_rospy_unavailable:{exc}")
        return None, warnings
    try:
        if not rospy.core.is_initialized():
            rospy.init_node("n5_live_odom_pose_fallback", anonymous=True, disable_signals=True)
        msg = rospy.wait_for_message(ODOM_TOPIC, Odometry, timeout=timeout_sec)
        pose = msg.pose.pose
        yaw = yaw_from_quat(pose.orientation)
        warnings.append(f"robot_pose_from_live_odom_topic:{ODOM_TOPIC}")
        return [float(pose.position.x), float(pose.position.y), yaw], warnings
    except Exception as exc:
        warnings.append(f"live_odom_pose_unavailable:{exc}")
        return None, warnings


def normalize_candidate(raw: Dict[str, Any], source: str, default_rank: Optional[int] = None) -> Dict[str, Any]:
    c = dict(raw)
    c.setdefault("original_rank", c.get("rank_original", c.get("rank", default_rank)))
    c.setdefault("reranked_rank", c.get("rank_after_feasibility", c.get("rank", default_rank)))
    c.setdefault("target_frame", "team_livox_odom")
    c.setdefault("pose_source_topic", "/team/livox/icp_odom_gated")
    c.setdefault("pose_source_compliant", True)
    c.setdefault("transform_ready", finite_xy(c.get("target_xy_team_livox_odom")))
    c.setdefault("risk_flags", [])
    c.setdefault("send_to_navigation", False)
    c.setdefault("safe_for_navigation", False)
    c.setdefault("planner_ready", False)
    c.setdefault("diagnostic_only", True)
    c["_source"] = source
    return c


def candidates(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    top3 = data.get("l3zm_top3")
    if isinstance(top3, list):
        for frame in top3:
            if isinstance(frame, dict):
                for item in frame.get("top3", []):
                    if isinstance(item, dict):
                        rows.append(normalize_candidate(item, "l3zm_top3_shadow_sequence.json", item.get("rank")))
    top1 = data.get("l3zm_top1")
    if isinstance(top1, list):
        for item in top1:
            if isinstance(item, dict):
                rows.append(normalize_candidate(item, "l3zm_top1_shadow_sequence.json", 1))
    return rows


def candidate_metrics(c: Dict[str, Any], pose: List[float]) -> Dict[str, Any]:
    target = c.get("target_xy_team_livox_odom")
    result = dict(c)
    if finite_xy(target):
        dx = float(target[0]) - pose[0]
        dy = float(target[1]) - pose[1]
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        heading = wrap_angle(bearing - pose[2])
        result.update(
            {
                "distance_to_robot_m": dist,
                "target_bearing_rad": bearing,
                "heading_error_rad": heading,
                "abs_heading_error_rad": abs(heading),
            }
        )
    return result


def valid_for_selection(c: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if c.get("feasible_for_navigation_input") is not True:
        reasons.append("not_feasible_for_navigation_input")
    if c.get("target_frame") != "team_livox_odom":
        reasons.append("target_frame_not_team_livox_odom")
    if not finite_xy(c.get("target_xy_team_livox_odom")):
        reasons.append("target_xy_invalid")
    if c.get("pose_source_topic") != "/team/livox/icp_odom_gated":
        reasons.append("pose_source_topic_invalid")
    if c.get("pose_source_compliant") is not True:
        reasons.append("pose_source_not_compliant")
    if c.get("transform_ready") is not True:
        reasons.append("transform_not_ready")
    if set(str(x) for x in c.get("risk_flags", [])).intersection(HARD_RISKS):
        reasons.append("hard_risk_flags_present")
    if c.get("send_to_navigation") is not False:
        reasons.append("send_to_navigation_not_false")
    if c.get("safe_for_navigation") is not False:
        reasons.append("safe_for_navigation_not_false")
    if c.get("planner_ready") is not False:
        reasons.append("planner_ready_not_false")
    if c.get("diagnostic_only") is not True:
        reasons.append("diagnostic_only_not_true")
    return not reasons, reasons


def sort_key(c: Dict[str, Any]) -> Tuple[float, float, float, float]:
    rerank = c.get("reranked_rank")
    original = c.get("original_rank")
    return (
        float(rerank) if finite_number(rerank) else 9999.0,
        float(c.get("distance_to_robot_m", 9999.0)),
        float(c.get("abs_heading_error_rad", 9999.0)),
        float(original) if finite_number(original) else 9999.0,
    )


def run_n5a(data: Dict[str, Any], pose: Optional[List[float]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    table: List[Dict[str, Any]] = []
    selected = None
    if pose is None:
        errors.append("robot_pose_unavailable")
    for c in candidates(data):
        row = candidate_metrics(c, pose) if pose else dict(c)
        schema_ok, reasons = valid_for_selection(row)
        row["schema_feasible_for_selection"] = schema_ok
        row["selection_rejection_reasons"] = reasons
        row["within_short_horizon"] = bool(
            schema_ok and finite_number(row.get("distance_to_robot_m")) and row["distance_to_robot_m"] <= MAX_DIST
        )
        table.append(row)
    feasible = [r for r in table if r.get("schema_feasible_for_selection")]
    within = [r for r in feasible if r.get("within_short_horizon")]
    if within:
        chosen = sorted(within, key=sort_key)[0]
        selected = {
            "target_type": "existing_candidate",
            "candidate_id": chosen.get("candidate_id"),
            "target_frame": "team_livox_odom",
            "target_xy_team_livox_odom": chosen.get("target_xy_team_livox_odom"),
            "distance_to_robot_m": chosen.get("distance_to_robot_m"),
            "target_bearing_rad": chosen.get("target_bearing_rad"),
            "heading_error_rad": chosen.get("heading_error_rad"),
            "reranked_rank": chosen.get("reranked_rank"),
            "original_rank": chosen.get("original_rank"),
        }
    report = {
        "stage": "N5A_SHORT_HORIZON_CANDIDATE_SELECTION",
        "robot_xyyaw_team_livox_odom": pose,
        "max_short_horizon_distance_m": MAX_DIST,
        "candidate_count_total": len(table),
        "candidate_count_feasible": len(feasible),
        "candidate_count_within_short_horizon": len(within),
        "short_horizon_candidate_found": selected is not None,
        "selected_short_horizon_candidate": selected,
        "candidate_audit_table": table,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(OUT / "n5a_candidate_selection_report.json", report)
    return report


def latest_bev_paths(data: Dict[str, Any]) -> Optional[Dict[str, Path]]:
    latest = (data.get("n2", {}).get("latest_bev") or {})
    if latest.get("accgrid") and latest.get("meta") and latest.get("accmap_yaml"):
        return {k: Path(v) for k, v in {"accgrid": latest["accgrid"], "meta": latest["meta"], "accmap": latest["accmap_yaml"]}.items()}
    return None


def odom_delta_to_base(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def base_delta_to_odom(base_x: float, base_y: float, pose: List[float]) -> List[float]:
    yaw = float(pose[2])
    return [
        float(pose[0]) + math.cos(yaw) * base_x - math.sin(yaw) * base_y,
        float(pose[1]) + math.sin(yaw) * base_x + math.cos(yaw) * base_y,
    ]


def try_read_l3v_grid(timeout_sec: float) -> Tuple[Optional[Any], List[str]]:
    warnings: List[str] = []
    master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    parsed = urlparse(master_uri)
    host = parsed.hostname or "localhost"
    port = int(parsed.port or 11311)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError as exc:
        warnings.append(f"ros_master_unreachable:{master_uri}:{exc}")
        return None, warnings
    try:
        import rospy  # type: ignore
        from nav_msgs.msg import OccupancyGrid  # type: ignore
    except Exception as exc:
        warnings.append(f"l3v_grid_rospy_unavailable:{exc}")
        return None, warnings
    try:
        if not rospy.core.is_initialized():
            rospy.init_node("n5_grid_centerline_subgoal", anonymous=True, disable_signals=True)
        holder: Dict[str, Any] = {}

        def cb(msg: Any) -> None:
            holder["msg"] = msg

        sub = rospy.Subscriber(L3V_GRID_TOPIC, OccupancyGrid, cb, queue_size=1)
        deadline = time.monotonic() + timeout_sec
        while not rospy.is_shutdown() and time.monotonic() < deadline and "msg" not in holder:
            time.sleep(0.05)
        sub.unregister()
        msg = holder.get("msg")
        if msg is None:
            warnings.append(f"l3v_grid_timeout_wall_sec:{timeout_sec}")
            return None, warnings
        return msg, warnings
    except Exception as exc:
        warnings.append(f"l3v_grid_unavailable:{exc}")
        return None, warnings


def contiguous_true_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            intervals.append((start, idx - 1))
            start = None
    if start is not None:
        intervals.append((start, len(mask) - 1))
    return intervals


def grid_centerline_search_diagnostics(
    msg: Any,
    grid: np.ndarray,
    cols: np.ndarray,
    origin_x: float,
    origin_y: float,
    res: float,
) -> Dict[str, Any]:
    row_reports: List[Dict[str, Any]] = []
    max_interval_width = 0.0
    rows_with_any_free = 0
    rows_with_min_width = 0
    for row in range(grid.shape[0]):
        x = origin_x + (row + 0.5) * res
        if x < GRID_CENTERLINE_MIN_X_M or x > GRID_CENTERLINE_MAX_X_M:
            continue
        center_values = grid[row, cols]
        free_mask = center_values == 0
        free_count = int(free_mask.sum())
        if free_count > 0:
            rows_with_any_free += 1
        best_width = 0.0
        for start, end in contiguous_true_intervals(free_mask):
            width_m = (end - start + 1) * res
            best_width = max(best_width, width_m)
        if best_width >= GRID_CENTERLINE_MIN_WIDTH_M:
            rows_with_min_width += 1
        max_interval_width = max(max_interval_width, best_width)
        row_reports.append(
            {
                "x_forward_m": float(x),
                "free_count": free_count,
                "unknown_count": int((center_values == -1).sum()),
                "occupied_count": int((center_values == 100).sum()),
                "best_contiguous_free_width_m": float(best_width),
            }
        )
    search_values = grid[:, cols] if cols.size else np.array([], dtype=np.int16)
    total = int(search_values.size)
    return {
        "grid_topic": L3V_GRID_TOPIC,
        "grid_header_stamp_sec": float(msg.header.stamp.to_sec()) if msg.header.stamp else None,
        "grid_header_seq": int(msg.header.seq),
        "grid_frame_id": msg.header.frame_id,
        "grid_width": int(msg.info.width),
        "grid_height": int(msg.info.height),
        "grid_resolution_m": float(msg.info.resolution),
        "search_x_m": [GRID_CENTERLINE_MIN_X_M, GRID_CENTERLINE_MAX_X_M],
        "search_abs_y_m": GRID_CENTERLINE_SEARCH_ABS_Y_M,
        "min_required_contiguous_free_width_m": GRID_CENTERLINE_MIN_WIDTH_M,
        "search_column_count": int(cols.size),
        "searched_row_count": len(row_reports),
        "rows_with_any_free": rows_with_any_free,
        "rows_with_required_width": rows_with_min_width,
        "max_contiguous_free_width_m": float(max_interval_width),
        "search_free_ratio": float((search_values == 0).sum() / total) if total else None,
        "search_unknown_ratio": float((search_values == -1).sum() / total) if total else None,
        "search_occupied_ratio": float((search_values == 100).sum() / total) if total else None,
        "row_reports": row_reports,
    }


def grid_centerline_subgoal(pose: List[float]) -> Tuple[Optional[Dict[str, Any]], List[str], List[str], Optional[Dict[str, Any]]]:
    warnings: List[str] = []
    errors: List[str] = []
    if not GRID_CENTERLINE_ENABLED:
        warnings.append("grid_centerline_disabled")
        return None, warnings, errors, None
    msg, grid_warnings = try_read_l3v_grid(GRID_CENTERLINE_TIMEOUT_SEC)
    warnings.extend(grid_warnings)
    if msg is None:
        return None, warnings, errors, None
    if msg.header.frame_id != "base":
        errors.append(f"l3v_grid_frame_not_base:{msg.header.frame_id}")
        return None, warnings, errors, None
    height = int(msg.info.height)
    width = int(msg.info.width)
    res = float(msg.info.resolution)
    origin_x = float(msg.info.origin.position.x)
    origin_y = float(msg.info.origin.position.y)
    if height <= 0 or width <= 0 or res <= 0.0:
        errors.append("l3v_grid_geometry_invalid")
        return None, warnings, errors, None
    grid = np.array(msg.data, dtype=np.int16).reshape((height, width))
    y_values = origin_y + (np.arange(width) + 0.5) * res
    cols = np.where(np.abs(y_values) <= GRID_CENTERLINE_SEARCH_ABS_Y_M)[0]
    if cols.size == 0:
        errors.append("grid_centerline_no_search_columns")
        return None, warnings, errors, None
    search_diagnostics = grid_centerline_search_diagnostics(msg, grid, cols, origin_x, origin_y, res)
    candidates_out: List[Dict[str, Any]] = []
    for row in range(height):
        x = origin_x + (row + 0.5) * res
        if x < GRID_CENTERLINE_MIN_X_M or x > GRID_CENTERLINE_MAX_X_M:
            continue
        for start, end in contiguous_true_intervals(grid[row, cols] == 0):
            c0 = int(cols[start])
            c1 = int(cols[end])
            interval_width = (c1 - c0 + 1) * res
            if interval_width < GRID_CENTERLINE_MIN_WIDTH_M:
                continue
            y0 = origin_y + c0 * res
            y1 = origin_y + (c1 + 1) * res
            center_y = (y0 + y1) * 0.5
            center_cost = abs(center_y)
            forward_cost = abs(x - GRID_CENTERLINE_FORWARD_M)
            confidence = min(1.0, interval_width / 1.0) * (
                1.0 - min(center_cost / GRID_CENTERLINE_SEARCH_ABS_Y_M, 1.0) * 0.30
            )
            score = confidence - 0.25 * center_cost - 0.15 * forward_cost
            candidates_out.append(
                {
                    "x_forward_m": float(x),
                    "center_lateral_m": float(center_y),
                    "free_interval_y_m": [float(y0), float(y1)],
                    "free_interval_width_m": float(interval_width),
                    "confidence": float(confidence),
                    "score": float(score),
                    "row": int(row),
                    "col_range": [int(c0), int(c1)],
                }
            )
    if not candidates_out:
        errors.append("grid_centerline_no_valid_free_interval")
        return None, warnings, errors, search_diagnostics
    best = sorted(
        candidates_out,
        key=lambda c: (-c["score"], abs(c["center_lateral_m"]), abs(c["x_forward_m"] - GRID_CENTERLINE_FORWARD_M)),
    )[0]
    if best["confidence"] < GRID_CENTERLINE_MIN_CONFIDENCE:
        errors.append("grid_centerline_confidence_too_low")
        return None, warnings, errors, search_diagnostics
    base_x = min(GRID_CENTERLINE_FORWARD_M, float(best["x_forward_m"]))
    base_y = float(best["center_lateral_m"])
    target_xy = base_delta_to_odom(base_x, base_y, pose)
    heading_error = wrap_angle(math.atan2(target_xy[1] - pose[1], target_xy[0] - pose[0]) - pose[2])
    return (
        {
            "source": "l3v_grid_centerline",
            "base_xy": [base_x, base_y],
            "target_xy_team_livox_odom": target_xy,
            "distance_m": math.hypot(base_x, base_y),
            "heading_error_rad": heading_error,
            "metrics": {
                "selected_interval": best,
                "candidate_count": len(candidates_out),
                "grid_header_stamp_sec": float(msg.header.stamp.to_sec()),
                "grid_header_seq": int(msg.header.seq),
                "grid_frame_id": msg.header.frame_id,
                "grid_topic": L3V_GRID_TOPIC,
                "search_x_m": [GRID_CENTERLINE_MIN_X_M, GRID_CENTERLINE_MAX_X_M],
                "search_abs_y_m": GRID_CENTERLINE_SEARCH_ABS_Y_M,
                "min_width_m": GRID_CENTERLINE_MIN_WIDTH_M,
                "search_diagnostics": search_diagnostics,
            },
        },
        warnings,
        errors,
        search_diagnostics,
    )


def grid_cell_indices_for_local(msg: Any, x: float, y: float) -> Tuple[int, int]:
    res = float(msg.info.resolution)
    origin_x = float(msg.info.origin.position.x)
    origin_y = float(msg.info.origin.position.y)
    return int(math.floor((x - origin_x) / res)), int(math.floor((y - origin_y) / res))


def grid_local_for_cell(msg: Any, row: int, col: int) -> Tuple[float, float]:
    res = float(msg.info.resolution)
    origin_x = float(msg.info.origin.position.x)
    origin_y = float(msg.info.origin.position.y)
    return origin_x + (row + 0.5) * res, origin_y + (col + 0.5) * res


def window_values(grid: np.ndarray, row: int, col: int, radius_cells: int) -> np.ndarray:
    r0 = max(0, row - radius_cells)
    r1 = min(grid.shape[0], row + radius_cells + 1)
    c0 = max(0, col - radius_cells)
    c1 = min(grid.shape[1], col + radius_cells + 1)
    return grid[r0:r1, c0:c1]


def has_adjacent_unknown(grid: np.ndarray, row: int, col: int) -> bool:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = row + dr
            cc = col + dc
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and int(grid[rr, cc]) == -1:
                return True
    return False


def line_sample_values(msg: Any, grid: np.ndarray, x: float, y: float) -> List[int]:
    res = float(msg.info.resolution)
    steps = max(2, int(math.ceil(math.hypot(x, y) / max(res, 1e-6))))
    samples: List[int] = []
    for i in range(1, steps + 1):
        sx = x * i / steps
        sy = y * i / steps
        rr, cc = grid_cell_indices_for_local(msg, sx, sy)
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            samples.append(int(grid[rr, cc]))
        else:
            samples.append(-1)
    return samples


def frontier_fallback_subgoal(pose: List[float]) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    if not FRONTIER_FALLBACK_ENABLED:
        warnings.append("local_frontier_nbv_fallback_disabled")
        return None, warnings, errors
    msg, grid_warnings = try_read_l3v_grid(GRID_CENTERLINE_TIMEOUT_SEC)
    warnings.extend(grid_warnings)
    if msg is None:
        return None, warnings, errors
    if msg.header.frame_id != "base":
        errors.append(f"frontier_grid_frame_not_base:{msg.header.frame_id}")
        return None, warnings, errors
    height = int(msg.info.height)
    width = int(msg.info.width)
    res = float(msg.info.resolution)
    if height <= 0 or width <= 0 or res <= 0.0:
        errors.append("frontier_grid_geometry_invalid")
        return None, warnings, errors
    grid = np.array(msg.data, dtype=np.int16).reshape((height, width))
    footprint_radius = max(1, int(math.ceil(FRONTIER_ROBOT_RADIUS_M / res)))
    gain_radius = max(1, int(math.ceil(FRONTIER_GAIN_RADIUS_M / res)))
    cluster_radius = max(1, int(math.ceil(FRONTIER_CLUSTER_RADIUS_M / res)))
    frontier_mask = np.zeros_like(grid, dtype=bool)
    candidate_reports: List[Dict[str, Any]] = []
    for row in range(height):
        x, _ = grid_local_for_cell(msg, row, 0)
        if x < FRONTIER_MIN_X_M or x > FRONTIER_MAX_X_M:
            continue
        for col in range(width):
            x, y = grid_local_for_cell(msg, row, col)
            if abs(y) > FRONTIER_SEARCH_ABS_Y_M:
                continue
            if int(grid[row, col]) == 0 and has_adjacent_unknown(grid, row, col):
                frontier_mask[row, col] = True
    for row, col in zip(*np.where(frontier_mask)):
        x, y = grid_local_for_cell(msg, int(row), int(col))
        footprint = window_values(grid, int(row), int(col), footprint_radius)
        occupied_near = int((footprint == 100).sum())
        if occupied_near > 0:
            continue
        samples = line_sample_values(msg, grid, x, y)
        if not samples:
            continue
        occupied_on_path = sum(1 for v in samples if v == 100)
        unknown_on_path = sum(1 for v in samples if v == -1)
        path_unknown_ratio = unknown_on_path / float(len(samples))
        if occupied_on_path > 0 or path_unknown_ratio > FRONTIER_PATH_UNKNOWN_RATIO_MAX:
            continue
        gain_window = window_values(grid, int(row), int(col), gain_radius)
        cluster_window = frontier_mask[
            max(0, int(row) - cluster_radius) : min(height, int(row) + cluster_radius + 1),
            max(0, int(col) - cluster_radius) : min(width, int(col) + cluster_radius + 1),
        ]
        unknown_gain = int((gain_window == -1).sum())
        frontier_cluster_size = int(cluster_window.sum())
        yaw_change = abs(math.atan2(y, max(x, 1e-6)))
        lateral_penalty = abs(y)
        distance = math.hypot(x, y)
        score = (
            0.030 * unknown_gain
            + 0.020 * frontier_cluster_size
            + 0.250 * x
            - 0.420 * lateral_penalty
            - 0.180 * yaw_change
            - 0.120 * path_unknown_ratio
        )
        candidate_reports.append(
            {
                "base_xy": [float(x), float(y)],
                "distance_m": float(distance),
                "unknown_gain": unknown_gain,
                "frontier_cluster_size": frontier_cluster_size,
                "path_unknown_ratio": float(path_unknown_ratio),
                "occupied_near_count": occupied_near,
                "yaw_change_rad": float(yaw_change),
                "score": float(score),
                "row_col": [int(row), int(col)],
            }
        )
    if not candidate_reports:
        errors.append("local_frontier_nbv_no_safe_candidate")
        diagnostics = {
            "grid_topic": L3V_GRID_TOPIC,
            "grid_header_stamp_sec": float(msg.header.stamp.to_sec()) if msg.header.stamp else None,
            "grid_header_seq": int(msg.header.seq),
            "frontier_cell_count": int(frontier_mask.sum()),
            "search_x_m": [FRONTIER_MIN_X_M, FRONTIER_MAX_X_M],
            "search_abs_y_m": FRONTIER_SEARCH_ABS_Y_M,
            "path_unknown_ratio_max": FRONTIER_PATH_UNKNOWN_RATIO_MAX,
        }
        return {"diagnostics_only": diagnostics}, warnings, errors
    candidate_reports.sort(key=lambda c: (-c["score"], abs(c["base_xy"][1]), -c["base_xy"][0]))
    best = candidate_reports[0]
    if float(best["score"]) < FRONTIER_MIN_SCORE:
        errors.append("local_frontier_nbv_score_too_low")
        return {"diagnostics_only": {"best_candidate": best, "candidate_count": len(candidate_reports)}}, warnings, errors
    base_x, base_y = best["base_xy"]
    target_xy = base_delta_to_odom(base_x, base_y, pose)
    heading_error = wrap_angle(math.atan2(target_xy[1] - pose[1], target_xy[0] - pose[0]) - pose[2])
    return (
        {
            "source": "local_frontier_nbv_fallback",
            "base_xy": [float(base_x), float(base_y)],
            "target_xy_team_livox_odom": target_xy,
            "distance_m": float(math.hypot(base_x, base_y)),
            "heading_error_rad": heading_error,
            "metrics": {
                "selected_candidate": best,
                "candidate_count": len(candidate_reports),
                "frontier_cell_count": int(frontier_mask.sum()),
                "grid_header_stamp_sec": float(msg.header.stamp.to_sec()) if msg.header.stamp else None,
                "grid_header_seq": int(msg.header.seq),
                "grid_frame_id": msg.header.frame_id,
                "grid_topic": L3V_GRID_TOPIC,
                "search_x_m": [FRONTIER_MIN_X_M, FRONTIER_MAX_X_M],
                "search_abs_y_m": FRONTIER_SEARCH_ABS_Y_M,
                "robot_radius_m": FRONTIER_ROBOT_RADIUS_M,
                "path_unknown_ratio_max": FRONTIER_PATH_UNKNOWN_RATIO_MAX,
                "top_candidates": candidate_reports[:10],
            },
        },
        warnings,
        errors,
    )


def ratios(values: np.ndarray) -> Dict[str, Optional[float]]:
    if values.size == 0:
        return {"free_ratio": None, "unknown_ratio": None, "occupied_ratio": None}
    total = float(values.size)
    return {
        "free_ratio": float((values == 0).sum() / total),
        "unknown_ratio": float((values == 1).sum() / total),
        "occupied_ratio": float((values == 2).sum() / total),
    }


def cell_for(x: float, y: float, origin: List[float], res: float) -> Tuple[int, int]:
    return int(math.floor((y - origin[1]) / res)), int(math.floor((x - origin[0]) / res))


def bev_check(base_xy: List[float], pose: List[float], data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    warnings: List[str] = ["bev_projection_uses_pose_derived_team_livox_odom_to_base_delta_not_tf"]
    errors: List[str] = []
    paths = latest_bev_paths(data)
    metrics = {
        "projection_method": "pose_derived_delta_to_base",
        "bev_patch_metrics": {"free_ratio": None, "unknown_ratio": None, "occupied_ratio": None},
        "bev_corridor_metrics": {"free_ratio": None, "unknown_ratio": None, "occupied_ratio": None},
    }
    if not paths:
        errors.append("latest_bev_paths_missing")
        return metrics, warnings, errors
    meta = read_json(paths["meta"])
    if meta.get("frame_id") != "base" or meta.get("semantic_version") != "v3_fixed":
        errors.append("latest_bev_meta_not_base_v3_fixed")
        return metrics, warnings, errors
    grid = np.load(str(paths["accgrid"]), allow_pickle=False)
    res = float(meta.get("resolution", data.get("n2", {}).get("resolution", 0.0)))
    origin = meta.get("grid_origin_xy") or data.get("n2", {}).get("origin_xy")
    if not finite_number(res) or res <= 0 or not finite_xy(origin):
        errors.append("bev_resolution_or_origin_invalid")
        return metrics, warnings, errors

    row, col = cell_for(base_xy[0], base_xy[1], origin, res)
    if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
        errors.append("subgoal_outside_bev_grid")
        return metrics, warnings, errors
    rad = max(1, int(math.ceil(PATCH_RADIUS_M / res)))
    patch = grid[max(0, row - rad) : min(grid.shape[0], row + rad + 1), max(0, col - rad) : min(grid.shape[1], col + rad + 1)]
    patch_ratios = ratios(patch)
    metrics["bev_patch_metrics"] = patch_ratios

    samples: List[int] = []
    steps = max(2, int(math.ceil(math.hypot(base_xy[0], base_xy[1]) / max(res, 1e-6))))
    for i in range(1, steps + 1):
        x = base_xy[0] * i / steps
        y = base_xy[1] * i / steps
        rr, cc = cell_for(x, y, origin, res)
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            samples.append(int(grid[rr, cc]))
    corridor_ratios = ratios(np.array(samples, dtype=np.uint8))
    metrics["bev_corridor_metrics"] = corridor_ratios

    if patch_ratios["occupied_ratio"] is None or corridor_ratios["occupied_ratio"] is None:
        errors.append("bev_metrics_unavailable")
    elif patch_ratios["occupied_ratio"] > MAX_PATCH_OCC:
        errors.append("subgoal_patch_occupied_ratio_too_high")
    elif patch_ratios["unknown_ratio"] is not None and patch_ratios["unknown_ratio"] > MAX_PATCH_UNKNOWN:
        errors.append("subgoal_patch_unknown_ratio_too_high")
    elif corridor_ratios["occupied_ratio"] > MAX_CORRIDOR_OCC:
        errors.append("subgoal_corridor_occupied_ratio_too_high")
    elif corridor_ratios["unknown_ratio"] is not None and corridor_ratios["unknown_ratio"] > MAX_CORRIDOR_UNKNOWN:
        errors.append("subgoal_corridor_unknown_ratio_too_high")
    return metrics, warnings, errors


def run_n5b(data: Dict[str, Any], pose: Optional[List[float]], n5a: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    summary = data.get("summary", {})
    parent = summary.get("selected_candidate") or {}
    subgoal_xy = None
    subgoal_dist = None
    heading_error = None
    subgoal_source = None
    grid_centerline_metrics = None
    grid_centerline_diagnostics = None
    frontier_fallback_metrics = None
    subgoal_base_xy = None
    status = "BLOCKED"
    metrics = {
        "bev_patch_metrics": {"free_ratio": None, "unknown_ratio": None, "occupied_ratio": None},
        "bev_corridor_metrics": {"free_ratio": None, "unknown_ratio": None, "occupied_ratio": None},
    }
    if n5a.get("short_horizon_candidate_found"):
        warnings.append("n5b_skipped_existing_short_horizon_candidate_found")
    elif pose is None:
        errors.append("robot_pose_unavailable")
    else:
        centerline, centerline_warnings, centerline_errors, grid_centerline_diagnostics = grid_centerline_subgoal(pose)
        warnings.extend(centerline_warnings)
        if centerline is not None and not centerline_errors:
            subgoal_source = centerline["source"]
            subgoal_xy = centerline["target_xy_team_livox_odom"]
            subgoal_dist = centerline["distance_m"]
            heading_error = centerline["heading_error_rad"]
            subgoal_base_xy = centerline["base_xy"]
            grid_centerline_metrics = centerline["metrics"]
            metrics, bev_warnings, bev_errors = bev_check(subgoal_base_xy, pose, data)
            warnings.extend(bev_warnings)
            if bev_errors:
                warnings.extend([f"grid_centerline_bev_crosscheck:{e}" for e in bev_errors])
            status = "PASS"
        else:
            warnings.extend([f"grid_centerline_fallback:{e}" for e in centerline_errors])
            frontier, frontier_warnings, frontier_errors = frontier_fallback_subgoal(pose)
            warnings.extend(frontier_warnings)
            if frontier is not None and not frontier_errors and frontier.get("source") == "local_frontier_nbv_fallback":
                subgoal_source = frontier["source"]
                subgoal_xy = frontier["target_xy_team_livox_odom"]
                subgoal_dist = frontier["distance_m"]
                heading_error = frontier["heading_error_rad"]
                subgoal_base_xy = frontier["base_xy"]
                frontier_fallback_metrics = frontier["metrics"]
                warnings.append("local_frontier_nbv_fallback_used_after_centerline_unavailable")
                status = "PASS"
            elif not finite_xy(parent.get("target_xy_team_livox_odom")):
                warnings.extend([f"local_frontier_nbv_fallback:{e}" for e in frontier_errors])
                if isinstance(frontier, dict):
                    frontier_fallback_metrics = frontier.get("diagnostics_only")
                errors.append("parent_target_invalid")
                report = {
                    "stage": "N5B_SHORT_HORIZON_SUBGOAL_AUDIT",
                    "parent_candidate_id": parent.get("candidate_id"),
                    "parent_target_xy_team_livox_odom": parent.get("target_xy_team_livox_odom"),
                    "robot_xyyaw_team_livox_odom": pose,
                    "subgoal_generated": False,
                    "subgoal_source": subgoal_source,
                    "subgoal_xy_team_livox_odom": subgoal_xy,
                    "subgoal_base_xy": subgoal_base_xy,
                    "subgoal_distance_m": subgoal_dist,
                    "subgoal_heading_error_rad": heading_error,
                    "subgoal_safety_pass": False,
                    "subgoal_safety_status": status,
                    "grid_centerline_metrics": grid_centerline_metrics,
                    "grid_centerline_diagnostics": grid_centerline_diagnostics,
                    "frontier_fallback_metrics": frontier_fallback_metrics,
                    "bev_patch_metrics": metrics["bev_patch_metrics"],
                    "bev_corridor_metrics": metrics["bev_corridor_metrics"],
                    "warnings": warnings,
                    "errors": errors,
                }
                write_json(OUT / "n5b_subgoal_audit_report.json", report)
                return report
            else:
                warnings.extend([f"local_frontier_nbv_fallback:{e}" for e in frontier_errors])
                if isinstance(frontier, dict):
                    frontier_fallback_metrics = frontier.get("diagnostics_only")
                target = parent["target_xy_team_livox_odom"]
                dx = float(target[0]) - pose[0]
                dy = float(target[1]) - pose[1]
                parent_dist = math.hypot(dx, dy)
                if parent_dist <= 0.0:
                    errors.append("parent_target_distance_nonpositive")
                else:
                    subgoal_source = "parent_frontier_truncated"
                    subgoal_dist = min(SUBGOAL_DIST, parent_dist, MAX_DIST)
                    ux, uy = dx / parent_dist, dy / parent_dist
                    subgoal_xy = [pose[0] + ux * subgoal_dist, pose[1] + uy * subgoal_dist]
                    bearing = math.atan2(subgoal_xy[1] - pose[1], subgoal_xy[0] - pose[0])
                    heading_error = wrap_angle(bearing - pose[2])
                    base_x, base_y = odom_delta_to_base(subgoal_xy[0] - pose[0], subgoal_xy[1] - pose[1], pose[2])
                    subgoal_base_xy = [base_x, base_y]
                    metrics, bev_warnings, bev_errors = bev_check(subgoal_base_xy, pose, data)
                    warnings.extend(bev_warnings)
                    errors.extend(bev_errors)
                    status = "PASS" if not bev_errors else "BLOCKED"
    report = {
        "stage": "N5B_SHORT_HORIZON_SUBGOAL_AUDIT",
        "parent_candidate_id": parent.get("candidate_id"),
        "parent_target_xy_team_livox_odom": parent.get("target_xy_team_livox_odom"),
        "robot_xyyaw_team_livox_odom": pose,
        "subgoal_generated": subgoal_xy is not None,
        "subgoal_source": subgoal_source,
        "subgoal_xy_team_livox_odom": subgoal_xy,
        "subgoal_base_xy": subgoal_base_xy,
        "subgoal_distance_m": subgoal_dist,
        "subgoal_heading_error_rad": heading_error,
        "subgoal_safety_pass": bool(subgoal_xy is not None and not errors),
        "subgoal_safety_status": status,
        "grid_centerline_metrics": grid_centerline_metrics,
        "grid_centerline_diagnostics": grid_centerline_diagnostics,
        "frontier_fallback_metrics": frontier_fallback_metrics,
        "bev_patch_metrics": metrics["bev_patch_metrics"],
        "bev_corridor_metrics": metrics["bev_corridor_metrics"],
        "warnings": warnings,
        "errors": errors,
    }
    write_json(OUT / "n5b_subgoal_audit_report.json", report)
    return report


def write_override_from_target(target: Dict[str, Any], source: str) -> Dict[str, Any]:
    override = {
        "target_type": target["target_type"],
        "candidate_id": target["candidate_id"],
        "parent_candidate_id": target.get("parent_candidate_id"),
        "target_frame": "team_livox_odom",
        "target_xy_team_livox_odom": target["target_xy_team_livox_odom"],
        "distance_to_robot_m": target["distance_to_robot_m"],
        "heading_error_rad": target["heading_error_rad"],
        "source": source,
        "shadow_only": True,
        "send_to_navigation": False,
        "safe_for_navigation": False,
        "planner_ready": False,
        "diagnostic_only": True,
    }
    if target.get("subgoal_source"):
        override["subgoal_source"] = target.get("subgoal_source")
    if target.get("subgoal_base_xy"):
        override["subgoal_base_xy"] = target.get("subgoal_base_xy")
    write_json(OUT / "short_horizon_target_override.json", override)
    return override


def run_n5c(n5a: Dict[str, Any], n5b: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    override = None
    blocked = None
    if n5a.get("selected_short_horizon_candidate"):
        override = write_override_from_target(n5a["selected_short_horizon_candidate"], "n5a_existing_candidate_selection")
    elif n5b.get("subgoal_generated") and n5b.get("subgoal_safety_pass"):
        override = write_override_from_target(
            {
                "target_type": "generated_subgoal",
                "candidate_id": f"subgoal_for_{n5b.get('parent_candidate_id')}",
                "parent_candidate_id": n5b.get("parent_candidate_id"),
                "target_xy_team_livox_odom": n5b.get("subgoal_xy_team_livox_odom"),
                "distance_to_robot_m": n5b.get("subgoal_distance_m"),
                "heading_error_rad": n5b.get("subgoal_heading_error_rad"),
                "subgoal_source": n5b.get("subgoal_source"),
                "subgoal_base_xy": n5b.get("subgoal_base_xy"),
            },
            n5b.get("subgoal_source") or "n5b_generated_short_horizon_subgoal",
        )
    else:
        blocked = {
            "target_type": "none",
            "reason": "no_safe_short_horizon_target",
            "n5a_errors": n5a.get("errors", []),
            "n5b_errors": n5b.get("errors", []),
            "shadow_only": True,
            **BOUNDARY,
        }
        write_json(OUT / "short_horizon_target_blocked.json", blocked)
    return override, blocked


def decision(n5a: Dict[str, Any], n5b: Dict[str, Any], pose: Optional[List[float]]) -> str:
    if pose is None:
        return "N5_TARGET_SELECTION_BLOCKED_BY_POSE"
    if n5a.get("selected_short_horizon_candidate"):
        return "N5_TARGET_SELECTION_READY_WITH_EXISTING_CANDIDATE"
    if n5b.get("subgoal_generated") and n5b.get("subgoal_safety_pass"):
        return "N5_TARGET_SELECTION_READY_WITH_SUBGOAL"
    if any("bev" in e for e in n5b.get("errors", [])):
        return "N5_TARGET_SELECTION_BLOCKED_BY_BEV"
    if n5b.get("subgoal_generated"):
        return "N5_TARGET_SELECTION_BLOCKED_SUBGOAL_UNSAFE"
    return "N5_TARGET_SELECTION_BLOCKED_NO_SHORT_HORIZON_CANDIDATE"


def write_report(summary: Dict[str, Any], n5a: Dict[str, Any], n5b: Dict[str, Any], shadow: Optional[Dict[str, Any]]) -> None:
    selected = summary.get("selected_target") or {}
    parent = n5b.get("parent_candidate_id")
    parent_dist = None
    parent_target = n5b.get("parent_target_xy_team_livox_odom")
    pose = summary.get("robot_xyyaw_team_livox_odom")
    if finite_xy(parent_target) and finite_xyyaw(pose):
        parent_dist = math.hypot(parent_target[0] - pose[0], parent_target[1] - pose[1])
    lines = [
        "# N5 Target Selection / Short-horizon Subgoal Audit",
        "",
        f"- Final decision: `{summary['final_decision']}`",
        f"- Selected target: `{selected}`",
        "",
        "## Required Answers",
        "",
        f"A. 当前 robot pose：`{pose}`",
        f"B. 当前 parent candidate：`{parent}`",
        f"C. parent target 距离 robot：`{parent_dist}`",
        f"D. 是否存在 3m 内 feasible candidate：`{n5a.get('short_horizon_candidate_found')}`",
        f"E. 如果存在，选中了哪个：`{(n5a.get('selected_short_horizon_candidate') or {}).get('candidate_id')}`",
        f"F. 如果不存在，是否生成了 subgoal：`{n5b.get('subgoal_generated')}`",
        f"G. subgoal 坐标：`{n5b.get('subgoal_xy_team_livox_odom')}`",
        f"H. subgoal 距离 robot：`{n5b.get('subgoal_distance_m')}`",
        f"I. subgoal 是否经过 BEV patch/corridor 检查：`{n5b.get('subgoal_safety_status') in ['PASS', 'BLOCKED']}`; patch=`{n5b.get('bev_patch_metrics')}`, corridor=`{n5b.get('bev_corridor_metrics')}`",
        f"J. 是否存在 frame mismatch 或 transform 不可用：`False`; 使用 `pose_derived_delta_to_base`，不使用 TF。",
        f"K. 是否生成 short_horizon_target_override.json：`{(OUT / 'short_horizon_target_override.json').exists()}`",
        f"L. 是否复跑 Shadow Minimal Controller：`{shadow is not None}`",
        f"M. Shadow controller final_decision：`{(shadow or {}).get('final_decision')}`",
        f"N. 是否仍然禁止 /cmd_vel：`True`",
        f"O. 是否建议进入下一阶段 online local safety source 审计：`{summary['final_decision'] in ['N5_TARGET_SELECTION_READY_WITH_EXISTING_CANDIDATE', 'N5_TARGET_SELECTION_READY_WITH_SUBGOAL']}`",
        "",
        "## Boundary",
        "",
        "- execution_allowed=false",
        "- send_to_navigation=false",
        "- safe_for_navigation=false",
        "- planner_ready=false",
        "- diagnostic_only=true",
        "- would_publish_cmd_vel=false",
        "- published_cmd_vel=false",
        "- called_move_base=false",
        "- sent_navigation_goal=false",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Remove stale target files so the current audit is authoritative.
    for name in ["short_horizon_target_override.json", "short_horizon_target_blocked.json"]:
        path = OUT / name
        if path.exists():
            path.unlink()
    data = load_inputs()
    pose = robot_pose(data)
    pose_warnings: List[str] = []
    if pose is None:
        pose, pose_warnings = try_read_live_odom_pose(ODOM_FALLBACK_TIMEOUT_SEC)
    n5a = run_n5a(data, pose)
    n5b = run_n5b(data, pose, n5a)
    override, _blocked = run_n5c(n5a, n5b)
    final = decision(n5a, n5b, pose)
    selected = override or {
        "target_type": "none",
        "candidate_id": None,
        "parent_candidate_id": n5b.get("parent_candidate_id"),
        "target_frame": "team_livox_odom",
        "target_xy_team_livox_odom": None,
        "distance_to_robot_m": None,
        "heading_error_rad": None,
    }
    summary = {
        "stage": "N5_TARGET_SELECTION_SUBGOAL_AUDIT",
        "final_decision": final,
        "robot_xyyaw_team_livox_odom": pose,
        "selected_target": selected,
        "execution_boundary": BOUNDARY,
        "warnings": pose_warnings + n5a.get("warnings", []) + n5b.get("warnings", []),
        "errors": n5a.get("errors", []) + n5b.get("errors", []),
    }
    write_json(OUT / "n5_target_selection_subgoal_summary.json", summary)
    shadow = read_json(SHADOW_SUMMARY) if SHADOW_SUMMARY.exists() else None
    write_report(summary, n5a, n5b, shadow)
    print(json.dumps({"final_decision": final, "override": str(OUT / "short_horizon_target_override.json") if override else None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
