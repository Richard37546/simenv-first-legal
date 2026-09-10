#!/usr/bin/env python3
"""Manual low-speed turn-in-place smoke pre-audit.

Default mode is dry-run only. It reads odometry and safety topics, writes JSON
reports, and never publishes nonzero /cmd_vel unless --execute is explicit.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "manual_low_speed_turn_smoke"
REPORT = ROOT / "audit_reports" / "manual_low_speed_turn_smoke_report.md"
ONLINE_SAFETY_SUMMARY = ROOT / "debug" / "online_local_safety_source" / "online_local_safety_source_summary.json"
SHADOW_SUMMARY = ROOT / "debug" / "shadow_minimal_controller" / "shadow_minimal_controller_summary.json"

TOPIC_ODOM = "/team/livox/icp_odom_gated"
TOPIC_GRID = "/team/local_traversability_grid"
TOPIC_STATUS = "/team/traversability_status"
TOPIC_CMD = "/cmd_vel"

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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": str(exc)}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def init_rospy():
    import rospy  # type: ignore

    if not rospy.core.is_initialized():
        rospy.init_node("manual_low_speed_turn_smoke", anonymous=True, disable_signals=True)
    return rospy


def read_odom(rospy, timeout: float = 3.0) -> Dict[str, Any]:
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
            "pose_x_y_yaw": xy_yaw,
            "finite_pose": finite,
            "pass": bool(finite and msg.header.frame_id == "team_livox_odom" and msg.child_frame_id == "base"),
        }
    except Exception as exc:
        return {"topic": TOPIC_ODOM, "message_received": False, "pass": False, "error": str(exc)}


def collect_grid_samples(rospy, timeout: float = 20.0, min_samples: int = 3) -> List[Any]:
    from nav_msgs.msg import OccupancyGrid  # type: ignore

    samples: List[Any] = []
    lock = threading.Lock()

    def cb(msg: Any) -> None:
        with lock:
            samples.append(msg)

    sub = rospy.Subscriber(TOPIC_GRID, OccupancyGrid, cb, queue_size=10)
    deadline = time.monotonic() + timeout
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


def audit_l3v_grid(rospy) -> Dict[str, Any]:
    try:
        samples = collect_grid_samples(rospy)
    except Exception as exc:
        return {"topic": TOPIC_GRID, "sample_count": 0, "pass": False, "error": str(exc)}
    stamps = [float(sample.header.stamp.to_sec()) for sample in samples]
    seqs = [int(sample.header.seq) for sample in samples]
    frame_ids = [sample.header.frame_id for sample in samples]
    unique_values = sorted({int(v) for sample in samples for v in sample.data})
    stamp_monotonic = len(stamps) >= 2 and all(stamps[i] <= stamps[i + 1] for i in range(len(stamps) - 1))
    seq_monotonic = len(seqs) >= 2 and all(seqs[i] <= seqs[i + 1] for i in range(len(seqs) - 1))
    stamp_fresh = len(samples) >= 3 and stamp_monotonic and len(set(stamps)) > 1 and seq_monotonic and len(set(seqs)) > 1
    schema_pass = bool(samples) and set(unique_values).issubset({-1, 0, 100}) and all(frame == "base" for frame in frame_ids)
    return {
        "topic": TOPIC_GRID,
        "sample_count": len(samples),
        "frame_ids": frame_ids,
        "stamp_values_sec": stamps,
        "seq_values": seqs,
        "unique_values": unique_values,
        "stamp_monotonic": stamp_monotonic,
        "grid_stamp_freshness_pass": bool(stamp_fresh),
        "schema_pass": bool(schema_pass),
        "pass": bool(stamp_fresh and schema_pass),
    }


def read_status(rospy) -> Dict[str, Any]:
    from std_msgs.msg import String  # type: ignore

    try:
        msg = rospy.wait_for_message(TOPIC_STATUS, String, timeout=2.0)
        payload = json.loads(str(msg.data))
    except Exception as exc:
        return {"topic": TOPIC_STATUS, "message_received": False, "pass": False, "error": str(exc)}
    pass_flags = (
        payload.get("local_traversability_status") == "FREE_SUPPORTED"
        and payload.get("diagnostic_only") is True
        and payload.get("safe_for_navigation") is False
        and payload.get("autonomous_l4_allowed") is False
        and payload.get("published_cmd_vel") is False
    )
    return {"topic": TOPIC_STATUS, "message_received": True, "payload": payload, "pass": bool(pass_flags)}


def validate_params(linear_x: float, angular_z: float, duration_sec: float) -> Dict[str, Any]:
    errors: List[str] = []
    if abs(linear_x) > 1e-9:
        errors.append("linear_x_must_be_zero")
    if abs(angular_z) > 0.15:
        errors.append("angular_z_exceeds_0_15_rad_s")
    if duration_sec <= 0.0 or duration_sec > 0.50:
        errors.append("duration_sec_out_of_range")
    return {
        "linear_x": linear_x,
        "angular_z": angular_z,
        "duration_sec": duration_sec,
        "pass": not errors,
        "errors": errors,
    }


def safety_from_files() -> Dict[str, Any]:
    online = read_json(ONLINE_SAFETY_SUMMARY)
    shadow = read_json(SHADOW_SUMMARY)
    online_pass = bool(
        online.get("online_local_safety_pass") is True
        and online.get("must_stop") is False
        and online.get("grid_stamp_freshness_pass") is True
        and online.get("selected_online_local_safety_source") == "local_traversability_grid"
    )
    shadow_boundary_pass = bool(
        not shadow
        or (
            shadow.get("would_publish_cmd_vel") is False
            and shadow.get("published_cmd_vel") is False
            and shadow.get("called_move_base") is False
            and shadow.get("sent_navigation_goal") is False
        )
    )
    return {
        "online_safety_summary_path": str(ONLINE_SAFETY_SUMMARY),
        "shadow_summary_path": str(SHADOW_SUMMARY),
        "online_local_safety_pass": online_pass,
        "shadow_boundary_pass": shadow_boundary_pass,
        "online_final_decision": online.get("final_decision"),
        "shadow_final_decision": shadow.get("final_decision"),
        "pass": bool(online_pass and shadow_boundary_pass),
    }


def publish_zero(pub, count: int = 3) -> None:
    from geometry_msgs.msg import Twist  # type: ignore
    import rospy  # type: ignore

    zero = Twist()
    for _ in range(count):
        pub.publish(zero)
        rospy.sleep(0.05)


def execute_turn(rospy, args, odom_before: Dict[str, Any]) -> Dict[str, Any]:
    from geometry_msgs.msg import Twist  # type: ignore

    pub = rospy.Publisher(TOPIC_CMD, Twist, queue_size=1)
    rospy.sleep(0.2)
    cmd = Twist()
    cmd.linear.x = float(args.linear_x)
    cmd.angular.z = float(args.angular_z)
    nonzero_published = False
    try:
        start = time.monotonic()
        rate = rospy.Rate(20)
        while time.monotonic() - start < float(args.duration_sec):
            pub.publish(cmd)
            nonzero_published = True
            rate.sleep()
    finally:
        publish_zero(pub, count=3)
    odom_after = read_odom(rospy, timeout=3.0)
    yaw_before = (odom_before.get("pose_x_y_yaw") or [None, None, None])[2]
    yaw_after = (odom_after.get("pose_x_y_yaw") or [None, None, None])[2]
    yaw_delta = None
    drift = None
    if finite_number(yaw_before) and finite_number(yaw_after):
        yaw_delta = math.atan2(math.sin(float(yaw_after) - float(yaw_before)), math.cos(float(yaw_after) - float(yaw_before)))
    if odom_before.get("pose_x_y_yaw") and odom_after.get("pose_x_y_yaw"):
        drift = math.hypot(
            float(odom_after["pose_x_y_yaw"][0]) - float(odom_before["pose_x_y_yaw"][0]),
            float(odom_after["pose_x_y_yaw"][1]) - float(odom_before["pose_x_y_yaw"][1]),
        )
    return {
        "nonzero_cmd_vel_published": nonzero_published,
        "zero_cmd_vel_published_count": 3,
        "odom_after": odom_after,
        "yaw_delta_rad": yaw_delta,
        "xy_drift_m": drift,
    }


def write_report(summary: Dict[str, Any]) -> None:
    lines = [
        "# Manual Low-speed Turn-in-place Smoke Pre-audit",
        "",
        f"- final_decision: `{summary.get('final_decision')}`",
        f"- mode: `{summary.get('mode')}`",
        f"- l3v_stamp_freshness_pass: `{summary.get('l3v_grid', {}).get('grid_stamp_freshness_pass')}`",
        f"- online_local_safety_pass: `{summary.get('file_safety', {}).get('online_local_safety_pass')}`",
        f"- odom_pose_x_y_yaw: `{summary.get('odom_before', {}).get('pose_x_y_yaw')}`",
        f"- nonzero_cmd_vel_published: `{summary.get('nonzero_cmd_vel_published')}`",
        "",
        "## Boundary",
        "",
        "- dry-run default publishes no nonzero /cmd_vel.",
        "- execute mode requires explicit `--execute` and all gates passing.",
        "- called_move_base=false",
        "- sent_navigation_goal=false",
        "- Gazebo truth sources used=false",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--linear-x", type=float, default=0.0)
    parser.add_argument("--angular-z", type=float, default=0.10)
    parser.add_argument("--duration-sec", type=float, default=0.30)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rospy = init_rospy()
    params = validate_params(args.linear_x, args.angular_z, args.duration_sec)
    odom = read_odom(rospy)
    l3v = audit_l3v_grid(rospy)
    status = read_status(rospy)
    file_safety = safety_from_files()
    gates_pass = bool(params["pass"] and odom["pass"] and l3v["pass"] and status["pass"] and file_safety["pass"])

    execution_result: Dict[str, Any] = {}
    final = "DRY_RUN_PREAUDIT_PASS" if gates_pass else "DRY_RUN_PREAUDIT_BLOCKED"
    boundary = dict(BOUNDARY)
    nonzero_published = False
    zero_published_count = 0

    if args.execute:
        boundary["would_publish_cmd_vel"] = True
        if not params["pass"]:
            final = "EXECUTE_TURN_SMOKE_BLOCKED_BY_PARAMS"
        elif not l3v["pass"]:
            final = "EXECUTE_TURN_SMOKE_BLOCKED_BY_L3V_STAMP"
        elif not file_safety["pass"] or not status["pass"]:
            final = "EXECUTE_TURN_SMOKE_BLOCKED_BY_ONLINE_SAFETY"
        elif not odom["pass"]:
            final = "EXECUTE_TURN_SMOKE_BLOCKED_BY_ODOM"
        elif not gates_pass:
            final = "EXECUTE_TURN_SMOKE_BLOCKED_BY_BOUNDARY"
        else:
            execution_result = execute_turn(rospy, args, odom)
            nonzero_published = bool(execution_result.get("nonzero_cmd_vel_published"))
            zero_published_count = int(execution_result.get("zero_cmd_vel_published_count", 0))
            boundary["published_cmd_vel"] = nonzero_published
            yaw_delta = execution_result.get("yaw_delta_rad")
            drift = execution_result.get("xy_drift_m")
            if not finite_number(yaw_delta) or abs(float(yaw_delta)) < 0.005:
                final = "EXECUTE_TURN_SMOKE_FAILED_NO_YAW_RESPONSE"
            elif finite_number(drift) and float(drift) > 0.10:
                final = "EXECUTE_TURN_SMOKE_FAILED_EXCESSIVE_DRIFT"
            else:
                final = "EXECUTE_TURN_SMOKE_PASS_WITH_WARNINGS" if status.get("warnings") else "EXECUTE_TURN_SMOKE_PASS"

    summary = {
        "stage": "MANUAL_LOW_SPEED_TURN_IN_PLACE_SMOKE",
        "mode": "execute" if args.execute else "dry_run_preaudit",
        "final_decision": final,
        "params": params,
        "odom_before": odom,
        "l3v_grid": l3v,
        "traversability_status": status,
        "file_safety": file_safety,
        "execution_result": execution_result,
        "gates_pass": gates_pass,
        "execution_boundary": boundary,
        "nonzero_cmd_vel_published": nonzero_published,
        "zero_cmd_vel_published_count": zero_published_count,
        "called_move_base": False,
        "sent_navigation_goal": False,
        "forbidden_sources_used": [],
    }
    write_json(OUT / "manual_low_speed_turn_smoke_summary.json", summary)
    write_report(summary)
    print(json.dumps({"final_decision": final, "summary": str(OUT / "manual_low_speed_turn_smoke_summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
