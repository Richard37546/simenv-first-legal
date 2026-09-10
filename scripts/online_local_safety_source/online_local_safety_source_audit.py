#!/usr/bin/env python3
"""Online local safety source audit.

Read-only ROS audit. It checks local traversability topics first, then a
/bev/occupancy_grid front-sector fallback. It never publishes /cmd_vel, calls
move_base, sends navigation goals, reads Gazebo truth, or uses TF.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "online_local_safety_source"
REPORT = ROOT / "audit_reports" / "online_local_safety_source_audit_report.md"
SHADOW_SUMMARY = ROOT / "debug/shadow_minimal_controller/shadow_minimal_controller_summary.json"

TOPIC_ODOM = "/team/livox/icp_odom_gated"
TOPIC_BEV = "/bev/occupancy_grid"
TOPIC_LOCAL_GRID = "/team/local_traversability_grid"
TOPIC_TRAV_STATUS = "/team/traversability_status"
TOPIC_FRONT_AUX = "/team/front_auxiliary_evidence"
LOCAL_GRID_VALUES = {-1, 0, 100}
OPTIONAL_TOPICS = [
    "/team/local_traversability_evidence",
    "/team/front_auxiliary_evidence",
    "/team/traversability_debug",
]

FRONT = {
    "forward_min_m": 0.20,
    "forward_max_m": 1.50,
    "lateral_half_width_m": 0.40,
    "max_occupied_ratio": 0.02,
    "max_unknown_ratio": 0.40,
}
CORRIDOR = {
    "length_m": 1.50,
    "half_width_m": 0.25,
    "max_occupied_ratio": 0.02,
    "max_unknown_ratio": 0.50,
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


def init_rospy():
    import rospy  # type: ignore

    if not rospy.core.is_initialized():
        rospy.init_node("online_local_safety_source_audit", anonymous=True, disable_signals=True)
    return rospy


def topic_map(rospy) -> Dict[str, str]:
    return {name: typ for name, typ in rospy.get_published_topics(namespace="/")}


def msg_to_text(msg: Any) -> str:
    if hasattr(msg, "data"):
        return str(msg.data)
    return str(msg)


def occupancy_semantics(values: List[int]) -> Tuple[str, Dict[int, int], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    unique = set(values)
    if unique.issubset({0, 1, 2}):
        return "v3_fixed_0_free_1_unknown_2_occupied", {0: 0, 1: 1, 2: 2}, warnings, errors
    if unique.issubset({-1, 0, 100}):
        warnings.append("ros_occupancy_grid_semantics_converted_minus1_unknown_0_free_100_occupied")
        return "ros_occupancy_grid_minus1_unknown_0_free_100_occupied", {-1: 1, 0: 0, 100: 2}, warnings, errors
    errors.append(f"unknown_occupancy_grid_values: {sorted(unique)}")
    return "unknown", {}, warnings, errors


def cell_to_xy(info: Any, row: int, col: int) -> Tuple[float, float]:
    res = float(info.resolution)
    ox = float(info.origin.position.x)
    oy = float(info.origin.position.y)
    return ox + (col + 0.5) * res, oy + (row + 0.5) * res


def ratios(cells: List[int]) -> Dict[str, Optional[float]]:
    if not cells:
        return {"free_ratio": None, "unknown_ratio": None, "occupied_ratio": None}
    total = float(len(cells))
    return {
        "free_ratio": cells.count(0) / total,
        "unknown_ratio": cells.count(1) / total,
        "occupied_ratio": cells.count(2) / total,
    }


def occupancy_grid_cells(msg: Any, mapping: Dict[int, int], predicate) -> List[int]:
    width = int(msg.info.width)
    height = int(msg.info.height)
    out: List[int] = []
    data = list(msg.data)
    for row in range(height):
        for col in range(width):
            x, y = cell_to_xy(msg.info, row, col)
            if predicate(x, y):
                raw = int(data[row * width + col])
                if raw in mapping:
                    out.append(mapping[raw])
    return out


def grid_dimensions_valid(msg: Any) -> bool:
    return (
        int(msg.info.width) > 0
        and int(msg.info.height) > 0
        and finite_number(float(msg.info.resolution))
        and float(msg.info.resolution) > 0.0
        and len(msg.data) == int(msg.info.width) * int(msg.info.height)
    )


def parse_json_payload(payload: Optional[str]) -> Dict[str, Any]:
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def collect_grid_samples(rospy, topic: str, msg_type: Any, timeout: float = 4.0, min_samples: int = 3) -> List[Any]:
    samples: List[Any] = []
    lock = threading.Lock()

    def cb(msg: Any) -> None:
        with lock:
            samples.append(msg)

    sub = rospy.Subscriber(topic, msg_type, cb, queue_size=10)
    deadline = time.monotonic() + float(timeout)
    try:
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with lock:
                if len(samples) >= min_samples:
                    return list(samples[:min_samples])
            rospy.sleep(0.05)
        with lock:
            return list(samples)
    finally:
        sub.unregister()


def audit_grid_stamp_freshness(samples: List[Any]) -> Dict[str, Any]:
    stamp_values = [float(sample.header.stamp.to_sec()) for sample in samples]
    seq_values = [int(sample.header.seq) for sample in samples]
    frame_ids = [sample.header.frame_id for sample in samples]
    unique_values = sorted({int(v) for sample in samples for v in sample.data})
    stamp_monotonic = len(stamp_values) >= 2 and all(stamp_values[i] <= stamp_values[i + 1] for i in range(len(stamp_values) - 1))
    seq_monotonic = len(seq_values) >= 2 and all(seq_values[i] <= seq_values[i + 1] for i in range(len(seq_values) - 1))
    stamp_changed = len(set(stamp_values)) > 1
    seq_changed = len(set(seq_values)) > 1
    schema_pass = bool(samples) and set(unique_values).issubset(LOCAL_GRID_VALUES)
    return {
        "sample_count": len(samples),
        "required_sample_count": 3,
        "frame_ids": frame_ids,
        "frame_id_pass": bool(samples) and all(frame == "base" for frame in frame_ids),
        "stamp_values_sec": stamp_values,
        "stamp_monotonic": stamp_monotonic,
        "stamp_changed": stamp_changed,
        "seq_values": seq_values,
        "seq_monotonic": seq_monotonic,
        "seq_changed": seq_changed,
        "unique_values": unique_values,
        "schema_pass": schema_pass,
        "grid_stamp_freshness_pass": bool(len(samples) >= 3 and stamp_monotonic and stamp_changed and seq_monotonic and seq_changed),
    }


def audit_local_traversability(rospy, topics: Dict[str, str]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    status_payload = None
    status_json: Dict[str, Any] = {}
    status_value = None
    grid_report: Dict[str, Any] = {
        "topic_exists": TOPIC_LOCAL_GRID in topics,
        "topic_type": topics.get(TOPIC_LOCAL_GRID),
        "message_received": False,
    }

    if TOPIC_TRAV_STATUS in topics:
        try:
            from std_msgs.msg import String  # type: ignore

            status_msg = rospy.wait_for_message(TOPIC_TRAV_STATUS, String, timeout=1.0)
            status_payload = msg_to_text(status_msg)
            status_json = parse_json_payload(status_payload)
            for token in [
                "FREE_SUPPORTED",
                "OBSTACLE_SUPPORTED",
                "UNKNOWN_INSUFFICIENT_EVIDENCE",
                "CONFLICT_NEEDS_CAUTION",
            ]:
                if token in status_payload:
                    status_value = token
                    break
        except Exception as exc:
            warnings.append(f"traversability_status_read_failed: {exc}")

    local_pass = False
    freshness = {
        "sample_count": 0,
        "required_sample_count": 3,
        "grid_stamp_freshness_pass": False,
        "stamp_monotonic": False,
    }
    if TOPIC_LOCAL_GRID in topics and topics.get(TOPIC_LOCAL_GRID) == "nav_msgs/OccupancyGrid":
        try:
            from nav_msgs.msg import OccupancyGrid  # type: ignore

            samples = collect_grid_samples(rospy, TOPIC_LOCAL_GRID, OccupancyGrid, timeout=20.0, min_samples=3)
            if not samples:
                raise RuntimeError("no local traversability grid samples received")
            freshness = audit_grid_stamp_freshness(samples)
            msg = samples[-1]
            values = sorted(set(int(v) for v in msg.data))
            semantics, mapping, sem_warnings, sem_errors = occupancy_semantics(values)
            warnings.extend(sem_warnings)
            errors.extend(sem_errors)
            grid_report.update(
                {
                    "message_received": True,
                    "frame_id": msg.header.frame_id,
                    "width": int(msg.info.width),
                    "height": int(msg.info.height),
                    "resolution": float(msg.info.resolution),
                    "data_nonempty": len(msg.data) > 0,
                    "unique_values": values,
                    "grid_semantics": semantics,
                    "stamp_freshness": freshness,
                }
            )
            if msg.header.frame_id != "base":
                errors.append(f"local_traversability_grid_frame_not_base: {msg.header.frame_id}")
            if not grid_dimensions_valid(msg):
                errors.append("local_traversability_grid_dimensions_invalid")
            if not freshness.get("schema_pass"):
                errors.append(f"local_traversability_grid_schema_invalid: {freshness.get('unique_values')}")
            if not freshness.get("grid_stamp_freshness_pass"):
                errors.append("local_traversability_grid_stamp_stale_or_not_monotonic")
            if not errors:
                cells = occupancy_grid_cells(
                    msg,
                    mapping,
                    lambda x, y: 0.0 <= x <= 3.0 and -0.5 <= y <= 0.5,
                )
                sector = ratios(cells)
                grid_report["front_sector"] = sector
                local_pass = (
                    sector["occupied_ratio"] is not None
                    and sector["occupied_ratio"] <= 0.02
                    and sector["unknown_ratio"] is not None
                    and sector["unknown_ratio"] <= 0.40
                )
        except Exception as exc:
            warnings.append(f"local_traversability_grid_unavailable_or_timeout: {exc}")
    elif TOPIC_LOCAL_GRID in topics:
        errors.append(f"local_traversability_grid_wrong_type: {topics.get(TOPIC_LOCAL_GRID)}")
    else:
        warnings.append("local_traversability_grid_topic_not_published")

    if status_value == "OBSTACLE_SUPPORTED":
        local_pass = False
        errors.append("traversability_status_obstacle_supported")
    elif status_value in {"CONFLICT_NEEDS_CAUTION", "UNKNOWN_INSUFFICIENT_EVIDENCE"}:
        local_pass = False
        warnings.append(f"traversability_status_not_clear_free: {status_value}")
    if status_json:
        if status_json.get("local_traversability_status") != "FREE_SUPPORTED":
            local_pass = False
            errors.append(f"traversability_status_not_free_supported: {status_json.get('local_traversability_status')}")
        for flag in ["safe_for_navigation", "safe_for_frontier", "autonomous_l4_allowed", "published_cmd_vel"]:
            if status_json.get(flag) is not False:
                local_pass = False
                errors.append(f"traversability_status_boundary_flag_not_false: {flag}={status_json.get(flag)}")
        if status_json.get("diagnostic_only") is not True:
            local_pass = False
            errors.append(f"traversability_status_diagnostic_only_not_true: {status_json.get('diagnostic_only')}")

    report = {
        "stage": "ONLINE_LOCAL_TRAVERSABILITY_TOPIC_AUDIT",
        "topics": {
            TOPIC_LOCAL_GRID: topics.get(TOPIC_LOCAL_GRID),
            TOPIC_TRAV_STATUS: topics.get(TOPIC_TRAV_STATUS),
            TOPIC_FRONT_AUX: topics.get(TOPIC_FRONT_AUX),
            **{topic: topics.get(topic) for topic in OPTIONAL_TOPICS},
        },
        "local_traversability_online_available": bool(local_pass),
        "online_local_safety_source": "local_traversability_grid" if local_pass else None,
        "grid_stamp_freshness_pass": bool(freshness.get("grid_stamp_freshness_pass")),
        "grid_stamp_monotonic": bool(freshness.get("stamp_monotonic")),
        "grid": grid_report,
        "traversability_status_payload": status_payload,
        "traversability_status_json": status_json,
        "traversability_status": status_value,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(OUT / "local_traversability_online_audit.json", report)
    return report


def subgoal_direction() -> Optional[Tuple[float, float, float]]:
    override_path = ROOT / "debug/short_horizon_target_selection/short_horizon_target_override.json"
    n3_path = ROOT / "debug/navigation_dry_run_readiness/n3_gated_odom_pose_resolver_report.json"
    if not override_path.exists() or not n3_path.exists():
        return None
    override = read_json(override_path)
    n3 = read_json(n3_path)
    pose = n3.get("robot_xyyaw_team_livox_odom") or n3.get("pose_x_y_yaw")
    target = override.get("target_xy_team_livox_odom")
    if not finite_xyyaw(pose) or not finite_xy(target):
        return None
    dx = float(target[0]) - float(pose[0])
    dy = float(target[1]) - float(pose[1])
    yaw = float(pose[2])
    c = math.cos(yaw)
    s = math.sin(yaw)
    bx = c * dx + s * dy
    by = -s * dx + c * dy
    dist = math.hypot(bx, by)
    return bx, by, dist


def audit_bev_fallback(rospy, topics: Dict[str, str]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    report: Dict[str, Any] = {
        "stage": "BEV_FRONT_SECTOR_FALLBACK_SAFETY_CHECKER",
        "bev_topic_available": TOPIC_BEV in topics,
        "bev_frame_id": None,
        "grid_semantics": None,
        "front_sector": {"free_ratio": None, "unknown_ratio": None, "occupied_ratio": None, "pass": False},
        "subgoal_corridor": {"free_ratio": None, "unknown_ratio": None, "occupied_ratio": None, "pass": False},
        "online_local_safety_pass": False,
        "must_stop": True,
        "warnings": warnings,
        "errors": errors,
    }
    if TOPIC_BEV not in topics:
        errors.append("bev_occupancy_grid_topic_not_published")
        write_json(OUT / "bev_front_sector_fallback_report.json", report)
        return report
    if topics.get(TOPIC_BEV) != "nav_msgs/OccupancyGrid":
        errors.append(f"bev_occupancy_grid_wrong_type: {topics.get(TOPIC_BEV)}")
        write_json(OUT / "bev_front_sector_fallback_report.json", report)
        return report

    try:
        from nav_msgs.msg import OccupancyGrid  # type: ignore

        msg = rospy.wait_for_message(TOPIC_BEV, OccupancyGrid, timeout=3.0)
    except Exception as exc:
        errors.append(f"bev_occupancy_grid_unavailable_or_timeout: {exc}")
        write_json(OUT / "bev_front_sector_fallback_report.json", report)
        return report

    values = sorted(set(int(v) for v in msg.data))
    semantics, mapping, sem_warnings, sem_errors = occupancy_semantics(values)
    warnings.extend(sem_warnings)
    errors.extend(sem_errors)
    report.update(
        {
            "bev_frame_id": msg.header.frame_id,
            "width": int(msg.info.width),
            "height": int(msg.info.height),
            "resolution": float(msg.info.resolution),
            "unique_values": values,
            "grid_semantics": semantics,
        }
    )
    if msg.header.frame_id != "base":
        errors.append(f"frame_mismatch_or_transform_unavailable: {msg.header.frame_id}")
    if not grid_dimensions_valid(msg):
        errors.append("bev_grid_dimensions_invalid")
    if errors:
        write_json(OUT / "bev_front_sector_fallback_report.json", report)
        return report

    front_cells = occupancy_grid_cells(
        msg,
        mapping,
        lambda x, y: FRONT["forward_min_m"] <= x <= FRONT["forward_max_m"]
        and -FRONT["lateral_half_width_m"] <= y <= FRONT["lateral_half_width_m"],
    )
    front = ratios(front_cells)
    front_pass = (
        front["occupied_ratio"] is not None
        and front["occupied_ratio"] <= FRONT["max_occupied_ratio"]
        and front["unknown_ratio"] is not None
        and front["unknown_ratio"] <= FRONT["max_unknown_ratio"]
    )
    report["front_sector"] = {**front, "pass": front_pass}

    direction = subgoal_direction()
    if direction is None:
        errors.append("subgoal_direction_unavailable")
        write_json(OUT / "bev_front_sector_fallback_report.json", report)
        return report
    bx, by, dist = direction
    length = min(CORRIDOR["length_m"], dist)
    if dist <= 0.0:
        errors.append("subgoal_distance_nonpositive")
        write_json(OUT / "bev_front_sector_fallback_report.json", report)
        return report
    ux, uy = bx / dist, by / dist
    corridor_cells = occupancy_grid_cells(
        msg,
        mapping,
        lambda x, y: 0.0 <= (x * ux + y * uy) <= length
        and abs(-x * uy + y * ux) <= CORRIDOR["half_width_m"],
    )
    corridor = ratios(corridor_cells)
    corridor_pass = (
        corridor["occupied_ratio"] is not None
        and corridor["occupied_ratio"] <= CORRIDOR["max_occupied_ratio"]
        and corridor["unknown_ratio"] is not None
        and corridor["unknown_ratio"] <= CORRIDOR["max_unknown_ratio"]
    )
    report["subgoal_corridor"] = {**corridor, "pass": corridor_pass}
    report["corridor_length_m"] = length
    report["corridor_half_width_m"] = CORRIDOR["half_width_m"]
    report["online_local_safety_pass"] = bool(front_pass and corridor_pass)
    report["must_stop"] = not report["online_local_safety_pass"]
    write_json(OUT / "bev_front_sector_fallback_report.json", report)
    return report


def decision(local: Dict[str, Any], bev: Dict[str, Any]) -> str:
    if local.get("local_traversability_online_available"):
        return "ONLINE_LOCAL_SAFETY_READY_WITH_LOCAL_TRAVERSABILITY"
    if (local.get("grid") or {}).get("message_received") and not local.get("grid_stamp_freshness_pass"):
        return "ONLINE_LOCAL_SAFETY_BLOCKED_BY_STALE_LOCAL_GRID"
    if any("frame_mismatch" in e for e in bev.get("errors", [])):
        return "ONLINE_LOCAL_SAFETY_BLOCKED_BY_FRAME"
    if any("pose" in e for e in bev.get("errors", [])):
        return "ONLINE_LOCAL_SAFETY_BLOCKED_BY_POSE"
    if bev.get("online_local_safety_pass") and bev.get("must_stop") is False:
        return "ONLINE_LOCAL_SAFETY_READY_WITH_BEV_FALLBACK"
    if bev.get("bev_topic_available"):
        return "ONLINE_LOCAL_SAFETY_BLOCKED_BY_BEV_FALLBACK"
    return "ONLINE_LOCAL_SAFETY_BLOCKED_NO_ONLINE_SOURCE"


def build_summary(local: Dict[str, Any], bev: Dict[str, Any]) -> Dict[str, Any]:
    final = decision(local, bev)
    if final == "ONLINE_LOCAL_SAFETY_READY_WITH_LOCAL_TRAVERSABILITY":
        selected = "local_traversability_grid"
        safety_pass = True
        must_stop = False
        front = (local.get("grid") or {}).get("front_sector", {})
        corridor = {"free_ratio": None, "unknown_ratio": None, "occupied_ratio": None, "pass": None}
    elif final == "ONLINE_LOCAL_SAFETY_READY_WITH_BEV_FALLBACK":
        selected = "bev_front_sector_fallback"
        safety_pass = True
        must_stop = False
        front = bev.get("front_sector", {})
        corridor = bev.get("subgoal_corridor", {})
    else:
        selected = "none"
        safety_pass = False
        must_stop = True
        front = bev.get("front_sector", {})
        corridor = bev.get("subgoal_corridor", {})
    return {
        "stage": "ONLINE_LOCAL_SAFETY_SOURCE_AUDIT",
        "final_decision": final,
        "selected_online_local_safety_source": selected,
        "online_local_safety_pass": safety_pass,
        "must_stop": must_stop,
        "grid_stamp_freshness_pass": bool(local.get("grid_stamp_freshness_pass")),
        "grid_stamp_monotonic": bool(local.get("grid_stamp_monotonic")),
        "front_sector": front,
        "subgoal_corridor": corridor,
        "execution_boundary": BOUNDARY,
        "warnings": local.get("warnings", []) + bev.get("warnings", []),
        "errors": local.get("errors", []) + bev.get("errors", []),
    }


def write_report(local: Dict[str, Any], bev: Dict[str, Any], summary: Dict[str, Any]) -> None:
    shadow = read_json(SHADOW_SUMMARY) if SHADOW_SUMMARY.exists() else {}
    lines = [
        "# Online Local Safety Source Audit",
        "",
        f"- Final decision: `{summary['final_decision']}`",
        f"- Selected source: `{summary['selected_online_local_safety_source']}`",
        f"- online_local_safety_pass: `{summary['online_local_safety_pass']}`",
        f"- must_stop: `{summary['must_stop']}`",
        f"- grid_stamp_freshness_pass: `{summary.get('grid_stamp_freshness_pass')}`",
        f"- grid_stamp_monotonic: `{summary.get('grid_stamp_monotonic')}`",
        "",
        "## Required Answers",
        "",
        f"A. /team/local_traversability_grid 是否存在：`{local.get('topics', {}).get(TOPIC_LOCAL_GRID) is not None}`",
        f"B. /team/traversability_status 是否存在：`{local.get('topics', {}).get(TOPIC_TRAV_STATUS) is not None}`",
        f"C. 是否能读取在线 local traversability：`{local.get('local_traversability_online_available')}`",
        f"D. 如果不能，是否启用了 /bev/occupancy_grid front-sector fallback：`{not local.get('local_traversability_online_available')}`",
        f"E. /bev/occupancy_grid frame_id：`{bev.get('bev_frame_id')}`",
        f"F. /bev/occupancy_grid data 语义：`{bev.get('grid_semantics')}`",
        f"G. front sector free/unknown/occupied：`{summary.get('front_sector')}`",
        f"H. subgoal corridor free/unknown/occupied：`{summary.get('subgoal_corridor')}`",
        f"I. online_local_safety_pass 是否为 true：`{summary.get('online_local_safety_pass')}`",
        f"J. must_stop 是否为 false：`{summary.get('must_stop') is False}`",
        f"K. Shadow Minimal Controller 是否读取了 online local safety report：`{shadow.get('online_local_safety_source') is not None}`",
        f"L. Safety Gate 是否根据 online safety report 修改 shadow_cmd_vel：`{shadow.get('online_local_safety_forced_zero')}`",
        f"M. 是否仍然没有发布 /cmd_vel：`{shadow.get('published_cmd_vel') is False}`",
        f"N. 是否建议进入 Manual Low-speed Navigation Smoke Test 前置审计：`{summary['final_decision'] in ['ONLINE_LOCAL_SAFETY_READY_WITH_LOCAL_TRAVERSABILITY', 'ONLINE_LOCAL_SAFETY_READY_WITH_BEV_FALLBACK']}`",
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
    rospy = init_rospy()
    topics = topic_map(rospy)
    local = audit_local_traversability(rospy, topics)
    bev = audit_bev_fallback(rospy, topics)
    summary = build_summary(local, bev)
    write_json(OUT / "online_local_safety_source_summary.json", summary)
    write_report(local, bev, summary)
    print(json.dumps({"final_decision": summary["final_decision"], "summary": str(OUT / "online_local_safety_source_summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
