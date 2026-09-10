#!/usr/bin/env python3
"""Turn-only response sweep diagnostic for local subgoal alignment tuning."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PREFIX = "turn_response_sweep"

TOPIC_ODOM = "/team/livox/icp_odom_gated"
TOPIC_CMD = "/cmd_vel"


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def parse_angular_values(raw: str) -> List[float]:
    values: List[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        value = float(text)
        if value <= 0.0:
            raise argparse.ArgumentTypeError("angular-z-values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("angular-z-values must not be empty")
    return values


def parse_sweep_pairs(raw: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        if ":" not in text:
            raise argparse.ArgumentTypeError("sweep-pairs entries must use angular_z:duration_sec")
        angular_text, duration_text = text.split(":", 1)
        angular_z = float(angular_text.strip())
        duration_sec = float(duration_text.strip())
        if angular_z <= 0.0 or duration_sec <= 0.0:
            raise argparse.ArgumentTypeError("sweep-pairs angular_z and duration_sec must be positive")
        pairs.append((angular_z, duration_sec))
    if not pairs:
        raise argparse.ArgumentTypeError("sweep-pairs must not be empty")
    return pairs


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


def publish_zero(pub: Any, count: int) -> int:
    from geometry_msgs.msg import Twist  # type: ignore

    msg = Twist()
    sent = 0
    for _ in range(max(0, int(count))):
        pub.publish(msg)
        sent += 1
        time.sleep(0.05)
    return sent


def publish_turn_slice(
    pub: Any,
    rospy: Any,
    angular_z: float,
    duration_sec: float,
    timebase: str,
    rate_hz: float,
    wall_timeout_multiplier: float,
) -> Dict[str, Any]:
    from geometry_msgs.msg import Twist  # type: ignore

    cmd = Twist()
    cmd.linear.x = 0.0
    cmd.angular.z = float(angular_z)

    wall_start = time.monotonic()
    ros_start = float(rospy.Time.now().to_sec())
    wall_timeout_sec = max(float(duration_sec) * float(wall_timeout_multiplier), 10.0)
    sleep_sec = 1.0 / max(float(rate_hz), 1.0)
    count = 0
    warnings: List[str] = []

    while not rospy.is_shutdown():
        wall_elapsed = time.monotonic() - wall_start
        ros_now = float(rospy.Time.now().to_sec())
        sim_elapsed = max(0.0, ros_now - ros_start)
        elapsed = sim_elapsed if timebase == "sim_time" else wall_elapsed
        if elapsed >= float(duration_sec):
            break
        if wall_elapsed >= wall_timeout_sec:
            warnings.append(f"{timebase}_motion_slice_wall_timeout")
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


def compute_turn_response(
    before: Dict[str, Any],
    after: Dict[str, Any],
    angular_z: float,
    duration_sec: float,
    actual_sim_duration_sec: Optional[float],
) -> Dict[str, Any]:
    before_pose = pose_tuple(before)
    after_pose = pose_tuple(after)
    if before_pose is None or after_pose is None:
        return {"pass": False, "error": "missing_valid_pose"}

    yaw_delta = normalize_angle(after_pose[2] - before_pose[2])
    abs_yaw_delta = abs(yaw_delta)
    xy_drift = math.hypot(after_pose[0] - before_pose[0], after_pose[1] - before_pose[1])
    expected = abs(float(angular_z)) * float(duration_sec)
    effective_yaw_rate = None
    if finite_number(actual_sim_duration_sec) and float(actual_sim_duration_sec) > 1e-9:
        effective_yaw_rate = abs_yaw_delta / float(actual_sim_duration_sec)
    return {
        "pass": True,
        "yaw_before_rad": before_pose[2],
        "yaw_after_rad": after_pose[2],
        "yaw_delta_rad": yaw_delta,
        "abs_yaw_delta_rad": abs_yaw_delta,
        "expected_yaw_delta_rad": expected,
        "effective_yaw_rate_rad_per_sim_sec": effective_yaw_rate,
        "turn_efficiency": abs_yaw_delta / expected if expected > 1e-9 else None,
        "xy_drift_m": xy_drift,
        "drift_per_yaw_rad": xy_drift / abs_yaw_delta if abs_yaw_delta > 1e-9 else None,
    }


def choose_preferred(results: List[Dict[str, Any]]) -> Tuple[Optional[float], str]:
    candidates = [
        r for r in results
        if r.get("pass") and finite_number(r.get("turn_efficiency")) and float(r["turn_efficiency"]) > 0.0
    ]
    if not candidates:
        return None, "No successful turn response samples were available."
    candidates.sort(
        key=lambda r: (
            float(r.get("turn_efficiency") or 0.0),
            -float(r.get("drift_per_yaw_rad") or 1e9),
            -float(r.get("angular_z") or 0.0),
        ),
        reverse=True,
    )
    best = candidates[0]
    return float(best["angular_z"]), (
        "Selected the tested angular_z with the highest measured turn_efficiency, "
        "using lower drift_per_yaw_rad as the secondary preference."
    )


def choose_for_band(
    results: List[Dict[str, Any]],
    yaw_min: float,
    yaw_max: float,
    drift_max: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    successful = [r for r in results if r.get("pass")]
    bounded = [
        r for r in successful
        if finite_number(r.get("abs_yaw_delta_rad"))
        and yaw_min <= float(r["abs_yaw_delta_rad"]) <= yaw_max
        and finite_number(r.get("xy_drift_m"))
        and float(r["xy_drift_m"]) < drift_max
    ]
    if bounded:
        bounded.sort(
            key=lambda r: (
                float(r.get("turn_efficiency") or 0.0),
                -float(r.get("drift_per_yaw_rad") or 1e9),
            ),
            reverse=True,
        )
        return bounded[0], "matched_yaw_window_and_drift_constraint"

    fallback = [
        r for r in successful
        if finite_number(r.get("drift_per_yaw_rad"))
    ]
    if not fallback:
        return None, "no_successful_candidate"
    fallback.sort(key=lambda r: float(r.get("drift_per_yaw_rad") or 1e9))
    return fallback[0], "fallback_lowest_drift_per_yaw_rad"


def compact_choice(choice: Optional[Dict[str, Any]], reason: str) -> Dict[str, Any]:
    if choice is None:
        return {"choice": None, "selection_reason": reason}
    return {
        "angular_z": choice.get("angular_z"),
        "duration_sec": choice.get("duration_sec"),
        "yaw_delta_rad": choice.get("yaw_delta_rad"),
        "abs_yaw_delta_rad": choice.get("abs_yaw_delta_rad"),
        "turn_efficiency": choice.get("turn_efficiency"),
        "xy_drift_m": choice.get("xy_drift_m"),
        "drift_per_yaw_rad": choice.get("drift_per_yaw_rad"),
        "selection_reason": reason,
    }


def build_adaptive_policy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    coarse, coarse_reason = choose_for_band(results, 0.35, 0.8, 0.15)
    medium, medium_reason = choose_for_band(results, 0.25, 0.6, 0.12)
    fine, fine_reason = choose_for_band(results, 0.1, 0.35, 0.08)
    policy = {
        "coarse_align_choice": compact_choice(coarse, coarse_reason),
        "medium_align_choice": compact_choice(medium, medium_reason),
        "fine_align_choice": compact_choice(fine, fine_reason),
        "policy": {
            "if_abs_heading_error_rad_gt_1_0": compact_choice(coarse, coarse_reason),
            "elif_abs_heading_error_rad_gt_0_6": compact_choice(medium, medium_reason),
            "elif_abs_heading_error_rad_gt_0_35": compact_choice(fine, fine_reason),
            "else": {"enter_approach_phase": True},
        },
    }
    fallback_reasons = [
        item["selection_reason"]
        for item in [policy["coarse_align_choice"], policy["medium_align_choice"], policy["fine_align_choice"]]
        if item.get("selection_reason") == "fallback_lowest_drift_per_yaw_rad"
    ]
    if fallback_reasons:
        policy["recommendation_reason"] = (
            "At least one ALIGN band did not satisfy the requested yaw/drift window; "
            "that band falls back to the lowest drift_per_yaw_rad candidate, so validate carefully before applying."
        )
    else:
        policy["recommendation_reason"] = (
            "Each ALIGN band selected a measured compact turn slice within its target yaw window and drift constraint."
        )
    return policy


def output_paths(output_prefix: str) -> Tuple[Path, Path]:
    safe_prefix = output_prefix.strip() or DEFAULT_OUTPUT_PREFIX
    return (
        ROOT / "debug" / "local_subgoal_runner_mvp" / f"{safe_prefix}_summary.json",
        ROOT / "audit_reports" / f"{safe_prefix}_report.md",
    )


def write_outputs(summary: Dict[str, Any], out_path: Path, report_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Turn Response Sweep Diagnostic Report",
        "",
        f"- final_decision: `{summary.get('final_decision')}`",
        f"- motion_duration_timebase: `{summary.get('motion_duration_timebase')}`",
        f"- turn_duration_sec: `{summary.get('turn_duration_sec')}`",
        f"- total_test_count_requested: `{summary.get('total_test_count_requested')}`",
        f"- total_test_count_completed: `{summary.get('total_test_count_completed')}`",
        f"- early_stop_reason: `{summary.get('early_stop_reason')}`",
        f"- preferred_align_angular_z: `{summary.get('preferred_align_angular_z')}`",
        f"- recommendation_reason: `{summary.get('recommendation_reason')}`",
        f"- cumulative_displacement_from_sweep_start_m: `{summary.get('cumulative_displacement_from_sweep_start_m')}`",
        "",
        "## Sweep Results",
        "",
    ]
    for result in summary.get("results", []):
        lines.extend(
            [
                f"### angular_z {result.get('angular_z')} duration {result.get('duration_sec')}",
                "",
                f"- expected_yaw_delta_rad: `{result.get('expected_yaw_delta_rad')}`",
                f"- yaw_delta_rad: `{result.get('yaw_delta_rad')}`",
                f"- abs_yaw_delta_rad: `{result.get('abs_yaw_delta_rad')}`",
                f"- effective_yaw_rate_rad_per_sim_sec: `{result.get('effective_yaw_rate_rad_per_sim_sec')}`",
                f"- turn_efficiency: `{result.get('turn_efficiency')}`",
                f"- xy_drift_m: `{result.get('xy_drift_m')}`",
                f"- drift_per_yaw_rad: `{result.get('drift_per_yaw_rad')}`",
                f"- actual_sim_duration_sec: `{result.get('actual_sim_duration_sec')}`",
                f"- actual_wall_duration_sec: `{result.get('actual_wall_duration_sec')}`",
                f"- observed_realtime_factor: `{result.get('observed_realtime_factor')}`",
                f"- cmd_publish_count: `{result.get('cmd_publish_count')}`",
                f"- zero_stop_count: `{result.get('zero_stop_count')}`",
                f"- warnings: `{result.get('warnings')}`",
                "",
            ]
        )

    policy = summary.get("recommended_adaptive_align_policy")
    if policy:
        lines.extend(
            [
                "## Adaptive ALIGN Policy Proposal",
                "",
                f"- coarse_align_choice: `{policy.get('coarse_align_choice')}`",
                f"- medium_align_choice: `{policy.get('medium_align_choice')}`",
                f"- fine_align_choice: `{policy.get('fine_align_choice')}`",
                f"- policy: `{policy.get('policy')}`",
                f"- recommendation_reason: `{policy.get('recommendation_reason')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Boundary",
            "",
            f"- forbidden_sources_used: `{summary.get('forbidden_sources_used')}`",
            f"- called_move_base: `{summary.get('called_move_base')}`",
            f"- sent_navigation_goal: `{summary.get('sent_navigation_goal')}`",
            f"- runner_main_logic_modified: `{summary.get('runner_main_logic_modified')}`",
            f"- git_add_or_commit: `{summary.get('git_add_or_commit')}`",
        ]
    )
    if summary.get("errors"):
        lines += ["", "## Errors", ""]
        lines += [f"- `{err}`" for err in summary["errors"]]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angular-z-values", type=parse_angular_values, default=parse_angular_values("0.3,0.5,0.8"))
    parser.add_argument("--sweep-pairs", type=parse_sweep_pairs, default=None)
    parser.add_argument("--turn-duration-sec", type=float, default=3.0)
    parser.add_argument("--motion-duration-timebase", choices=["sim_time", "wall_time"], default="sim_time")
    parser.add_argument("--publish-rate-hz", type=float, default=20.0)
    parser.add_argument("--wall-timeout-multiplier", type=float, default=30.0)
    parser.add_argument("--zero-stop-count", type=int, default=3)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()
    out_path, report_path = output_paths(args.output_prefix)
    sweep_pairs = args.sweep_pairs or [(angular_z, float(args.turn_duration_sec)) for angular_z in args.angular_z_values]

    summary: Dict[str, Any] = {
        "final_decision": "TURN_RESPONSE_SWEEP_NOT_RUN",
        "motion_duration_timebase": args.motion_duration_timebase,
        "turn_duration_sec": float(args.turn_duration_sec) if args.sweep_pairs is None else None,
        "angular_z_values": list(args.angular_z_values),
        "sweep_pairs": [{"angular_z": a, "duration_sec": d} for a, d in sweep_pairs],
        "output_prefix": args.output_prefix,
        "publish_rate_hz": float(args.publish_rate_hz),
        "wall_timeout_multiplier": float(args.wall_timeout_multiplier),
        "zero_stop_count": int(args.zero_stop_count),
        "total_test_count_requested": len(sweep_pairs),
        "total_test_count_completed": 0,
        "early_stop_reason": None,
        "cumulative_displacement_from_sweep_start_m": None,
        "results": [],
        "preferred_align_angular_z": None,
        "recommended_adaptive_align_policy": None,
        "recommendation_reason": None,
        "forbidden_sources_used": [],
        "called_move_base": False,
        "sent_navigation_goal": False,
        "runner_main_logic_modified": False,
        "git_add_or_commit": False,
        "errors": [],
    }

    try:
        import rospy  # type: ignore
        from geometry_msgs.msg import Twist  # type: ignore  # noqa: F401

        rospy.init_node("turn_response_sweep_diagnostic", anonymous=True, disable_signals=True)
        pub = rospy.Publisher(TOPIC_CMD, Twist, queue_size=1)
        time.sleep(0.5)
        sweep_start_pose: Optional[Tuple[float, float, float]] = None

        for angular_z, duration_sec in sweep_pairs:
            before = read_odom(rospy)
            before_pose = pose_tuple(before)
            if sweep_start_pose is None and before_pose is not None:
                sweep_start_pose = before_pose
            result: Dict[str, Any] = {
                "angular_z": float(angular_z),
                "duration_sec": float(duration_sec),
                "odom_before": before,
                "pass": False,
                "warnings": [],
            }
            if not before.get("pass"):
                result["error"] = "compliant_odom_before_unavailable"
                summary["results"].append(result)
                summary["errors"].append(f"angular_z_{angular_z}:compliant_odom_before_unavailable")
                publish_zero(pub, args.zero_stop_count)
                continue

            timing = publish_turn_slice(
                pub=pub,
                rospy=rospy,
                angular_z=float(angular_z),
                duration_sec=float(duration_sec),
                timebase=args.motion_duration_timebase,
                rate_hz=float(args.publish_rate_hz),
                wall_timeout_multiplier=float(args.wall_timeout_multiplier),
            )
            zero_count = publish_zero(pub, args.zero_stop_count)
            time.sleep(0.2)
            after = read_odom(rospy)
            response = compute_turn_response(
                before,
                after,
                float(angular_z),
                float(duration_sec),
                timing.get("actual_sim_duration_sec"),
            )
            result.update(response)
            result.update(
                {
                    "odom_after": after,
                    "actual_sim_duration_sec": timing.get("actual_sim_duration_sec"),
                    "actual_wall_duration_sec": timing.get("actual_wall_duration_sec"),
                    "observed_realtime_factor": timing.get("observed_realtime_factor"),
                    "cmd_publish_count": timing.get("cmd_publish_count"),
                    "zero_stop_count": zero_count,
                    "warnings": timing.get("warnings", []),
                }
            )
            if not after.get("pass"):
                result["pass"] = False
                result["error"] = "compliant_odom_after_unavailable"
                summary["errors"].append(f"angular_z_{angular_z}:compliant_odom_after_unavailable")
            if timing.get("warnings"):
                summary["errors"].extend([f"angular_z_{angular_z}:{w}" for w in timing["warnings"]])
            summary["results"].append(result)
            summary["total_test_count_completed"] = len(summary["results"])

            after_pose = pose_tuple(after)
            if sweep_start_pose is not None and after_pose is not None:
                displacement = math.hypot(after_pose[0] - sweep_start_pose[0], after_pose[1] - sweep_start_pose[1])
                result["cumulative_displacement_from_sweep_start_m"] = displacement
                summary["cumulative_displacement_from_sweep_start_m"] = displacement
                if displacement > 0.8:
                    summary["early_stop_reason"] = "cumulative_displacement_from_sweep_start_m_gt_0_8"
                    summary["final_decision"] = "TURN_RESPONSE_COMPACT_EARLY_STOP" if args.sweep_pairs else "TURN_RESPONSE_SWEEP_EARLY_STOP"
                    break

        preferred, reason = choose_preferred(summary["results"])
        summary["preferred_align_angular_z"] = preferred
        policy = build_adaptive_policy(summary["results"])
        summary["recommended_adaptive_align_policy"] = policy
        summary["recommendation_reason"] = policy.get("recommendation_reason") or reason
        if summary["final_decision"].endswith("EARLY_STOP"):
            pass
        elif any(r.get("pass") for r in summary["results"]):
            if args.sweep_pairs:
                summary["final_decision"] = "TURN_RESPONSE_COMPACT_COMPLETE" if not summary["errors"] else "TURN_RESPONSE_COMPACT_COMPLETED_WITH_WARNINGS"
            else:
                summary["final_decision"] = "TURN_RESPONSE_SWEEP_COMPLETE" if not summary["errors"] else "TURN_RESPONSE_SWEEP_COMPLETED_WITH_WARNINGS"
        else:
            summary["final_decision"] = "TURN_RESPONSE_SWEEP_FAILED"
    except Exception as exc:
        summary["final_decision"] = "TURN_RESPONSE_SWEEP_FAILED"
        summary["errors"].append(str(exc))
    finally:
        write_outputs(summary, out_path, report_path)

    success_decisions = {
        "TURN_RESPONSE_SWEEP_COMPLETE",
        "TURN_RESPONSE_SWEEP_COMPLETED_WITH_WARNINGS",
        "TURN_RESPONSE_SWEEP_EARLY_STOP",
        "TURN_RESPONSE_COMPACT_COMPLETE",
        "TURN_RESPONSE_COMPACT_COMPLETED_WITH_WARNINGS",
        "TURN_RESPONSE_COMPACT_EARLY_STOP",
    }
    return 0 if summary["final_decision"] in success_decisions else 1


if __name__ == "__main__":
    raise SystemExit(main())
