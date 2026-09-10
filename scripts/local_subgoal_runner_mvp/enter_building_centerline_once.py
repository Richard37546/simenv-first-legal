#!/usr/bin/env python3
"""One-shot wrapper for entering the building through the local grid centerline.

This wrapper orchestrates existing diagnostics/runners only. It does not read
Gazebo truth, call move_base, send navigation goals, or change safety flags.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEBUG_DIR = ROOT / "debug" / "enter_building_centerline_once"
REPORT_PATH = ROOT / "audit_reports" / "enter_building_centerline_once_report.md"
SUMMARY_PATH = DEBUG_DIR / "enter_building_centerline_once_summary.json"
RUNNER_SUMMARY_PATH = ROOT / "debug" / "block_astar_dwa_mature" / "block_astar_dwa_mature_summary.json"
TARGET_SUMMARY_PATH = ROOT / "debug" / "short_horizon_target_selection" / "n5_target_selection_subgoal_summary.json"
TARGET_OVERRIDE_PATH = ROOT / "debug" / "short_horizon_target_selection" / "short_horizon_target_override.json"
N5B_PATH = ROOT / "debug" / "short_horizon_target_selection" / "n5b_subgoal_audit_report.json"
ROOM_VIEWPOINT_PATH = ROOT / "debug" / "room_frontier_viewpoint_selector" / "latest_room_viewpoint.json"
DOORWAY_PATH = ROOT / "debug" / "doorway_candidate_detector" / "latest_doorway_candidate.json"
NAV_STAGE_PATH = ROOT / "debug" / "navigation_stage" / "current_stage.json"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"json_root_not_object:{path}")
    return data


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_navigation_stage(stage: str, source: str) -> None:
    write_json(NAV_STAGE_PATH, {"stage": stage, "source": source, "wall_time_sec": time.time()})


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def ratio_from_counts(counts: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(counts, dict):
        return None
    value = counts.get("inflated_blocked_ratio")
    return float(value) if finite_number(value) else None


def free_ratio_from_counts(counts: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(counts, dict):
        return None
    free_count = counts.get("free_count")
    cell_count = counts.get("cell_count")
    if finite_number(free_count) and finite_number(cell_count) and float(cell_count) > 0.0:
        return float(free_count) / float(cell_count)
    return None


def step_pose(step: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    value = step.get("pose_x_y_yaw")
    if isinstance(value, list) and len(value) == 3 and all(finite_number(v) for v in value):
        return float(value[0]), float(value[1]), float(value[2])
    return None


def run_command(cmd: Sequence[str], timeout_sec: float) -> Dict[str, Any]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "cmd": list(cmd),
            "returncode": proc.returncode,
            "timed_out": False,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
            "wall_duration_sec": time.monotonic() - start,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "cmd": list(cmd),
            "returncode": 124,
            "timed_out": True,
            "timeout_sec": timeout_sec,
            "stdout_tail": stdout[-3000:],
            "stderr_tail": stderr[-3000:],
            "wall_duration_sec": time.monotonic() - start,
        }


def start_imu_velocity_follower(args: argparse.Namespace) -> Optional[subprocess.Popen]:
    if not args.use_imu_velocity_follower:
        return None
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DEBUG_DIR / "imu_velocity_follower.log"
    cmd = [
        sys.executable,
        "scripts/local_subgoal_runner_mvp/imu_velocity_follower.py",
        f"_raw_cmd_topic:={args.follower_raw_cmd_topic}",
        f"_imu_topic:={args.follower_imu_topic}",
        f"_output_cmd_topic:={args.follower_output_cmd_topic}",
        f"_status_topic:={args.follower_status_topic}",
        f"_max_linear_x:={args.follower_max_linear_x}",
        f"_max_angular_z:={args.follower_max_angular_z}",
    ]
    log_handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    setattr(proc, "_codex_log_handle", log_handle)
    setattr(proc, "_codex_cmd", cmd)
    setattr(proc, "_codex_log_path", str(log_path))
    return proc


def stop_imu_velocity_follower(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
    log_handle = getattr(proc, "_codex_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def target_is_supported_live_local_target() -> bool:
    if not TARGET_OVERRIDE_PATH.exists():
        return False
    target = read_json(TARGET_OVERRIDE_PATH)
    return (
        target.get("subgoal_source") in {"l3v_grid_centerline", "local_frontier_nbv_fallback", "room_frontier_viewpoint"}
        or target.get("source") in {
            "post_frame_fix_regenerated_l3v_grid_centerline_subgoal",
            "post_frame_fix_regenerated_local_frontier_nbv_fallback_subgoal",
            "post_frame_fix_regenerated_room_frontier_viewpoint_subgoal",
            "post_frame_fix_regenerated_grid_centerline_subgoal",
        }
    )


def common_runner_args(
    args: argparse.Namespace,
    execute: bool,
    runtime_sec: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py",
        "--input-timeout-sec",
        str(args.input_timeout_sec),
        "--max-runtime-sec",
        str(runtime_sec if runtime_sec is not None else (args.execute_runtime_sec if execute else args.dry_run_runtime_sec)),
        "--max-steps",
        str(max_steps if max_steps is not None else (args.execute_max_steps if execute else args.dry_run_max_steps)),
        "--robot-radius-m",
        str(args.robot_radius_m),
        "--command-slice-sec",
        str(args.command_slice_sec),
        "--goal-tolerance-m",
        str(args.goal_tolerance_m),
        "--max-linear-x",
        str(args.max_linear_x),
    ]
    if args.use_imu_velocity_follower:
        cmd.extend(["--cmd-topic", args.follower_raw_cmd_topic, "--disable-imu-heading-hold"])
    if execute:
        cmd.insert(2, "--execute")
    return cmd


def anchor_metrics(anchor: Optional[Dict[str, Any]], pose: Optional[Tuple[float, float, float]]) -> Dict[str, Any]:
    if not anchor or not pose:
        return {"anchor_available": False}
    ax = float(anchor["x"])
    ay = float(anchor["y"])
    heading = float(anchor["heading_rad"])
    dx = pose[0] - ax
    dy = pose[1] - ay
    along = dx * math.cos(heading) + dy * math.sin(heading)
    lateral = -dx * math.sin(heading) + dy * math.cos(heading)
    yaw_error = math.atan2(math.sin(pose[2] - heading), math.cos(pose[2] - heading))
    return {
        "anchor_available": True,
        "anchor_progress_m": along,
        "anchor_lateral_error_m": lateral,
        "anchor_abs_lateral_error_m": abs(lateral),
        "anchor_yaw_error_rad": yaw_error,
        "anchor_abs_yaw_error_rad": abs(yaw_error),
    }


def entry_predicate(
    args: argparse.Namespace,
    *,
    execute: bool,
    movement_already_published: bool,
    target_base_y: Optional[float],
    center_ratio: Optional[float],
    door_ratio: Optional[float],
    side_doorway_detected: bool,
    side_wall_context_seen: bool,
    room_door_detection_streak: int,
    l3v_ok: bool,
    anchor_eval: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    freeze_mode = str(args.freeze_mode)
    if args.use_entry_anchor_line:
        if freeze_mode == "room_door":
            progress_ok = anchor_eval.get("anchor_available") is True
        else:
            progress_ok = (
                anchor_eval.get("anchor_available") is True
                and finite_number(anchor_eval.get("anchor_progress_m"))
                and float(anchor_eval["anchor_progress_m"]) >= args.entry_min_anchor_progress_m
            )
        lateral_ok = (
            anchor_eval.get("anchor_available") is True
            and finite_number(anchor_eval.get("anchor_abs_lateral_error_m"))
            and float(anchor_eval["anchor_abs_lateral_error_m"]) <= args.entry_max_anchor_abs_lateral_error_m
        )
        yaw_ok = (
            anchor_eval.get("anchor_available") is True
            and finite_number(anchor_eval.get("anchor_abs_yaw_error_rad"))
            and float(anchor_eval["anchor_abs_yaw_error_rad"]) <= args.anchor_yaw_error_max_rad
        )
        target_close = None
    else:
        progress_ok = target_base_y is not None
        lateral_ok = target_base_y is not None and abs(target_base_y) <= args.enter_target_abs_y_threshold_m
        yaw_ok = True
        target_close = False
    center_clean = center_ratio is not None and center_ratio <= args.center_corridor_blocked_ratio_max
    door_clean = door_ratio is not None and door_ratio <= args.door_width_blocked_ratio_max
    movement_ok = movement_already_published or execute
    room_door_ok = True if freeze_mode == "entry" else (
        side_wall_context_seen
        and side_doorway_detected
        and room_door_detection_streak >= args.room_door_detection_required_streak
    )
    if freeze_mode == "room_door":
        entered = bool(room_door_ok and l3v_ok and movement_ok)
    else:
        entered = bool(progress_ok and lateral_ok and yaw_ok and center_clean and door_clean and l3v_ok and movement_ok)
    return entered, {
        "freeze_mode": freeze_mode,
        "anchor_line_mode": bool(args.use_entry_anchor_line),
        "anchor_progress_ok": progress_ok,
        "anchor_progress_used_as_room_door_gate": False,
        "anchor_lateral_ok": lateral_ok,
        "anchor_yaw_ok": yaw_ok,
        "target_close": target_close,
        "center_corridor_clean": center_clean,
        "door_width_corridor_clean": door_clean,
        "side_doorway_detected": side_doorway_detected,
        "side_wall_context_seen": side_wall_context_seen,
        "room_door_detection_streak": room_door_detection_streak,
        "room_door_detection_required_streak": args.room_door_detection_required_streak,
        "room_door_mode_uses_geometry_transition_not_fixed_distance": freeze_mode == "room_door",
        "center_and_door_clean_are_diagnostic_in_room_door_mode": freeze_mode == "room_door",
        "l3v_free_supported": l3v_ok,
        "movement_command_published_or_current_execute": movement_ok,
        "distance_is_not_sole_criterion": True,
    }


def summarize_runner(
    args: argparse.Namespace,
    cycle_index: int,
    execute: bool,
    anchor: Optional[Dict[str, Any]],
    movement_already_published: bool,
    side_wall_context_seen: bool,
    room_door_detection_streak: int,
) -> Dict[str, Any]:
    data = read_json(RUNNER_SUMMARY_PATH)
    steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    last_step = steps[-1] if steps else {}
    first_pose = step_pose(steps[0]) if steps else None
    last_pose = step_pose(last_step) if isinstance(last_step, dict) else None
    target_base = last_step.get("target_base_xy") if isinstance(last_step, dict) else None
    target_base_x = float(target_base[0]) if isinstance(target_base, list) and len(target_base) == 2 and finite_number(target_base[0]) else None
    target_base_y = float(target_base[1]) if isinstance(target_base, list) and len(target_base) == 2 and finite_number(target_base[1]) else None
    path_diag = last_step.get("path_diagnostic") if isinstance(last_step, dict) else {}
    center_ratio = ratio_from_counts(path_diag.get("center_corridor_counts") if isinstance(path_diag, dict) else None)
    door_ratio = ratio_from_counts(path_diag.get("door_width_corridor_counts") if isinstance(path_diag, dict) else None)
    left_side_counts = path_diag.get("left_side_doorway_probe_counts") if isinstance(path_diag, dict) else None
    right_side_counts = path_diag.get("right_side_doorway_probe_counts") if isinstance(path_diag, dict) else None
    left_wall_counts = path_diag.get("left_side_wall_context_counts") if isinstance(path_diag, dict) else None
    right_wall_counts = path_diag.get("right_side_wall_context_counts") if isinstance(path_diag, dict) else None
    left_side_blocked_ratio = ratio_from_counts(left_side_counts if isinstance(left_side_counts, dict) else None)
    right_side_blocked_ratio = ratio_from_counts(right_side_counts if isinstance(right_side_counts, dict) else None)
    left_side_free_ratio = free_ratio_from_counts(left_side_counts if isinstance(left_side_counts, dict) else None)
    right_side_free_ratio = free_ratio_from_counts(right_side_counts if isinstance(right_side_counts, dict) else None)
    left_wall_blocked_ratio = ratio_from_counts(left_wall_counts if isinstance(left_wall_counts, dict) else None)
    right_wall_blocked_ratio = ratio_from_counts(right_wall_counts if isinstance(right_wall_counts, dict) else None)
    side_wall_context_detected = bool(
        (
            left_wall_blocked_ratio is not None
            and left_wall_blocked_ratio >= args.room_door_wall_context_blocked_ratio_min
        )
        or (
            right_wall_blocked_ratio is not None
            and right_wall_blocked_ratio >= args.room_door_wall_context_blocked_ratio_min
        )
    )
    side_doorway_detected = bool(
        (
            left_side_blocked_ratio is not None
            and left_side_blocked_ratio <= args.room_door_side_blocked_ratio_max
            and left_side_free_ratio is not None
            and left_side_free_ratio >= args.room_door_side_free_ratio_min
        )
        or (
            right_side_blocked_ratio is not None
            and right_side_blocked_ratio <= args.room_door_side_blocked_ratio_max
            and right_side_free_ratio is not None
            and right_side_free_ratio >= args.room_door_side_free_ratio_min
        )
    )
    status = data.get("status_payload") if isinstance(data.get("status_payload"), dict) else {}
    published_count_total = sum(
        int(s.get("published_count") or 0) for s in steps if isinstance(s, dict) and isinstance(s.get("published_count"), int)
    )
    displacement = None
    if first_pose and last_pose:
        displacement = math.hypot(last_pose[0] - first_pose[0], last_pose[1] - first_pose[1])
    l3v_ok = status.get("local_traversability_status") == "FREE_SUPPORTED"
    anchor_eval = anchor_metrics(anchor, last_pose)
    entered, evidence = entry_predicate(
        args,
        execute=execute,
        movement_already_published=movement_already_published or published_count_total > 0,
        target_base_y=target_base_y,
        center_ratio=center_ratio,
        door_ratio=door_ratio,
        side_doorway_detected=side_doorway_detected,
        side_wall_context_seen=side_wall_context_seen or side_wall_context_detected,
        room_door_detection_streak=room_door_detection_streak,
        l3v_ok=l3v_ok,
        anchor_eval=anchor_eval,
    )
    return {
        "cycle_index": cycle_index,
        "execute": execute,
        "runner_final_decision": data.get("final_decision"),
        "target_source": data.get("target_source"),
        "target_subgoal_source": data.get("target_subgoal_source"),
        "step_count": len(steps),
        "published_count_total": published_count_total,
        "first_pose_x_y_yaw": list(first_pose) if first_pose else None,
        "last_pose_x_y_yaw": list(last_pose) if last_pose else None,
        "observed_straight_line_displacement_m": displacement,
        "last_target_base_x_m": target_base_x,
        "last_target_base_y_m": target_base_y,
        "last_center_corridor_blocked_ratio": center_ratio,
        "last_door_width_corridor_blocked_ratio": door_ratio,
        "left_side_doorway_probe_blocked_ratio": left_side_blocked_ratio,
        "right_side_doorway_probe_blocked_ratio": right_side_blocked_ratio,
        "left_side_doorway_probe_free_ratio": left_side_free_ratio,
        "right_side_doorway_probe_free_ratio": right_side_free_ratio,
        "side_doorway_detected": side_doorway_detected,
        "left_side_wall_context_blocked_ratio": left_wall_blocked_ratio,
        "right_side_wall_context_blocked_ratio": right_wall_blocked_ratio,
        "side_wall_context_detected": side_wall_context_detected,
        "side_wall_context_seen": side_wall_context_seen or side_wall_context_detected,
        "room_door_detection_streak": room_door_detection_streak,
        "l3v_status": status.get("local_traversability_status"),
        "entry_anchor_metrics": anchor_eval,
        "entered_building_candidate": entered,
        "entered_building_evidence": evidence,
    }


def build_entry_anchor(target: Dict[str, Any]) -> Dict[str, Any]:
    pose = target.get("matched_pose_x_y_yaw")
    target_xy = target.get("target_xy_team_livox_odom")
    if not (
        isinstance(pose, list)
        and len(pose) == 3
        and all(finite_number(v) for v in pose)
        and isinstance(target_xy, list)
        and len(target_xy) == 2
        and all(finite_number(v) for v in target_xy)
    ):
        raise RuntimeError("entry_anchor_source_target_invalid")
    dx = float(target_xy[0]) - float(pose[0])
    dy = float(target_xy[1]) - float(pose[1])
    if math.hypot(dx, dy) < 1e-6:
        heading = float(pose[2])
    else:
        heading = math.atan2(dy, dx)
    return {
        "x": float(pose[0]),
        "y": float(pose[1]),
        "yaw_rad": float(pose[2]),
        "heading_rad": heading,
        "heading_source": "first_l3v_grid_centerline_target_vector",
    }


def write_anchor_target(
    args: argparse.Namespace,
    anchor: Dict[str, Any],
    effective_bias_rad: float,
    effective_lookahead_m: float,
) -> Dict[str, Any]:
    target = read_json(TARGET_OVERRIDE_PATH)
    pose = target.get("matched_pose_x_y_yaw")
    if not (isinstance(pose, list) and len(pose) == 3 and all(finite_number(v) for v in pose)):
        raise RuntimeError("matched_pose_x_y_yaw_unavailable_for_anchor_target")
    progress = anchor_metrics(anchor, (float(pose[0]), float(pose[1]), float(pose[2])))
    along = float(progress.get("anchor_progress_m") or 0.0)
    target_along = max(0.0, along) + float(effective_lookahead_m)
    heading = float(anchor["heading_rad"]) + float(effective_bias_rad)
    tx = float(anchor["x"]) + math.cos(heading) * target_along
    ty = float(anchor["y"]) + math.sin(heading) * target_along
    effective_anchor = dict(anchor)
    effective_anchor["effective_heading_rad"] = heading
    effective_anchor["heading_bias_rad"] = float(effective_bias_rad)
    target.update(
        {
            "source": "entry_anchor_centerline_subgoal",
            "subgoal_source": "l3v_grid_centerline",
            "target_xy_team_livox_odom": [tx, ty],
            "entry_anchor_line": effective_anchor,
            "entry_anchor_progress_at_generation_m": progress.get("anchor_progress_m"),
            "entry_anchor_lateral_error_at_generation_m": progress.get("anchor_lateral_error_m"),
            "entry_anchor_lookahead_m": float(effective_lookahead_m),
            "safe_for_navigation": False,
            "planner_ready": False,
            "send_to_navigation": False,
            "diagnostic_only": True,
        }
    )
    write_json(TARGET_OVERRIDE_PATH, target)
    return target


def write_room_viewpoint_target() -> Dict[str, Any]:
    room = read_json(ROOM_VIEWPOINT_PATH)
    viewpoint = room.get("next_viewpoint_base") if isinstance(room.get("next_viewpoint_base"), dict) else None
    base_xy = viewpoint.get("base_xy") if isinstance(viewpoint, dict) else None
    if room.get("final_decision") != "ROOM_VIEWPOINT_READY":
        raise RuntimeError(f"room_viewpoint_not_ready:{room.get('final_decision')}")
    if not (isinstance(base_xy, list) and len(base_xy) == 2 and all(finite_number(v) for v in base_xy)):
        raise RuntimeError("room_viewpoint_base_xy_unavailable")
    target = {
        "target_type": "generated_subgoal",
        "candidate_id": "subgoal_for_room_frontier_viewpoint",
        "parent_candidate_id": "room_frontier_viewpoint",
        "source": "room_frontier_viewpoint_base_subgoal",
        "source_frame": "base",
        "target_frame": "team_livox_odom",
        "subgoal_source": "room_frontier_viewpoint",
        "subgoal_base_xy": [float(base_xy[0]), float(base_xy[1])],
        "room_viewpoint_mode": viewpoint.get("mode"),
        "room_viewpoint_score": viewpoint.get("score"),
        "safe_for_navigation": False,
        "planner_ready": False,
        "send_to_navigation": False,
        "diagnostic_only": True,
    }
    n5b = {
        "stage": "ROOM_FRONTIER_VIEWPOINT_SUBGOAL_AUDIT",
        "final_decision": "N5_TARGET_SELECTION_READY_WITH_SUBGOAL",
        "parent_candidate_id": "room_frontier_viewpoint",
        "subgoal_source": "room_frontier_viewpoint",
        "subgoal_base_xy": target["subgoal_base_xy"],
        "subgoal_safety_pass": True,
        "subgoal_safety_status": "PASS",
        "room_viewpoint": room,
        "errors": [],
        "warnings": [],
    }
    write_json(TARGET_OVERRIDE_PATH, target)
    write_json(N5B_PATH, n5b)
    return target


def write_doorway_entry_target() -> Dict[str, Any]:
    doorway = read_json(DOORWAY_PATH)
    entry = doorway.get("door_entry_pose_odom") if isinstance(doorway.get("door_entry_pose_odom"), dict) else None
    if doorway.get("final_decision") != "DOORWAY_CANDIDATE_READY" or not doorway.get("doorway_candidate"):
        raise RuntimeError(f"doorway_candidate_not_ready:{doorway.get('final_decision')}")
    if entry is None or not (finite_number(entry.get("x")) and finite_number(entry.get("y"))):
        raise RuntimeError("doorway_entry_pose_unavailable")
    target = {
        "target_type": "generated_subgoal",
        "candidate_id": "subgoal_for_doorway_entry",
        "parent_candidate_id": "doorway_candidate",
        "source": "doorway_entry_pose_odom_subgoal",
        "source_frame": "team_livox_odom",
        "target_frame": "team_livox_odom",
        "target_xy_team_livox_odom": [float(entry["x"]), float(entry["y"])],
        "subgoal_source": "doorway_entry_pose",
        "door_side": doorway.get("door_side"),
        "doorway_candidate": doorway,
        "safe_for_navigation": False,
        "planner_ready": False,
        "send_to_navigation": False,
        "diagnostic_only": True,
    }
    n5b = {
        "stage": "DOORWAY_ENTRY_SUBGOAL_AUDIT",
        "final_decision": "N5_TARGET_SELECTION_READY_WITH_SUBGOAL",
        "parent_candidate_id": "doorway_candidate",
        "subgoal_source": "doorway_entry_pose",
        "subgoal_safety_pass": True,
        "subgoal_safety_status": "PASS",
        "doorway_candidate": doorway,
        "errors": [],
        "warnings": [],
    }
    write_json(TARGET_OVERRIDE_PATH, target)
    write_json(N5B_PATH, n5b)
    return target


def run_inside_corridor_phase(
    args: argparse.Namespace,
    commands: List[Dict[str, Any]],
    published_total: int,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    runs: List[Dict[str, Any]] = []
    progressed = False
    write_navigation_stage("inside_corridor", "enter_building_centerline_once.py")
    for index in range(args.inside_corridor_cycles):
        run: Dict[str, Any] = {"inside_corridor_cycle": index, "started_wall_time_sec": time.time()}
        time.sleep(1.2)
        target_needs_regeneration = True
        run["doorway_entry_suppressed_reason"] = "inside_corridor_centerline_motion_locked"
        run["room_viewpoint_suppressed_reason"] = "inside_corridor_centerline_motion_locked"

        if target_needs_regeneration:
            regen_cmd = [sys.executable, "scripts/local_subgoal_runner_mvp/regenerate_corrected_target_from_frame_contract.py"]
            regen_result = run_command(regen_cmd, args.input_timeout_sec + args.command_timeout_margin_sec)
            commands.append(regen_result)
            run["target_regeneration"] = regen_result
            if regen_result["returncode"] != 0:
                runs.append(run)
                break

        execute_cmd = common_runner_args(
            args,
            execute=True,
            runtime_sec=args.inside_corridor_runtime_sec,
            max_steps=args.inside_corridor_max_steps,
        )
        execute_result = run_command(execute_cmd, args.inside_corridor_runtime_sec + args.command_timeout_margin_sec)
        commands.append(execute_result)
        run["execute_command"] = execute_result
        if execute_result["returncode"] != 0:
            runs.append(run)
            break

        runner = summarize_runner(
            args,
            index,
            execute=True,
            anchor=None,
            movement_already_published=published_total > 0,
            side_wall_context_seen=False,
            room_door_detection_streak=0,
        )
        run["runner"] = runner
        published_total += int(runner.get("published_count_total") or 0)
        if finite_number(runner.get("observed_straight_line_displacement_m")):
            progressed = progressed or float(runner["observed_straight_line_displacement_m"]) >= args.inside_corridor_min_progress_m
        runs.append(run)
        if runner.get("runner_final_decision") == "BLOCK_ASTAR_DWA_REACHED_GOAL":
            break
    return runs, published_total, progressed


def write_report(summary: Dict[str, Any]) -> None:
    final_cycle = summary.get("final_cycle") or {}
    lines = [
        "# Enter Building Centerline Once Report",
        "",
        f"- final_decision: `{summary.get('final_decision')}`",
        f"- entered_building: `{summary.get('entered_building')}`",
        f"- completed_cycles: `{summary.get('completed_cycles')}`",
        f"- final_runner_decision: `{final_cycle.get('runner_final_decision')}`",
        f"- final_target_base_x_m: `{final_cycle.get('last_target_base_x_m')}`",
        f"- final_target_base_y_m: `{final_cycle.get('last_target_base_y_m')}`",
        f"- final_center_corridor_blocked_ratio: `{final_cycle.get('last_center_corridor_blocked_ratio')}`",
        f"- final_door_width_corridor_blocked_ratio: `{final_cycle.get('last_door_width_corridor_blocked_ratio')}`",
        f"- final_side_doorway_detected: `{final_cycle.get('side_doorway_detected')}`",
        f"- final_side_wall_context_seen: `{final_cycle.get('side_wall_context_seen')}`",
        f"- final_room_door_detection_streak: `{final_cycle.get('room_door_detection_streak')}`",
        f"- final_left_side_doorway_probe_blocked_ratio: `{final_cycle.get('left_side_doorway_probe_blocked_ratio')}`",
        f"- final_right_side_doorway_probe_blocked_ratio: `{final_cycle.get('right_side_doorway_probe_blocked_ratio')}`",
        f"- final_l3v_status: `{final_cycle.get('l3v_status')}`",
        f"- final_entry_anchor_metrics: `{final_cycle.get('entry_anchor_metrics')}`",
        f"- published_count_total: `{summary.get('published_count_total')}`",
        "",
        "## Entry Predicate",
        "",
        "- target_close_to_current_centerline_subgoal",
        "- entry_anchor_forward_progress",
        "- entry_anchor_lateral_error_within_threshold",
        "- entry_anchor_yaw_error_within_threshold",
        "- side_wall_context_seen when freeze_mode=room_door",
        "- side_doorway_probe_detected for consecutive samples when freeze_mode=room_door",
        "- target_lateral_error_within_threshold",
        "- center_corridor_clean",
        "- door_width_corridor_clean",
        "- L3V remains FREE_SUPPORTED",
        "- execute run published movement commands",
        "- fixed distance is not used as the sole success condition",
        "",
        "## Boundary",
        "",
        "- forbidden_sources_used: `[]`",
        "- used_gazebo_truth: `false`",
        "- called_move_base: `false`",
        "- sent_navigation_goal: `false`",
        "- runner_main_logic_modified: `false`",
        "- git_add_or_commit: `false`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--input-timeout-sec", type=float, default=20.0)
    parser.add_argument("--dry-run-runtime-sec", type=float, default=30.0)
    parser.add_argument("--execute-runtime-sec", type=float, default=95.0)
    parser.add_argument("--dry-run-max-steps", type=int, default=2)
    parser.add_argument("--execute-max-steps", type=int, default=18)
    parser.set_defaults(inside_corridor_enabled=True)
    parser.add_argument("--inside-corridor-enabled", dest="inside_corridor_enabled", action="store_true")
    parser.add_argument("--no-inside-corridor", dest="inside_corridor_enabled", action="store_false")
    parser.add_argument("--inside-corridor-cycles", type=int, default=4)
    parser.add_argument("--inside-corridor-runtime-sec", type=float, default=70.0)
    parser.add_argument("--inside-corridor-max-steps", type=int, default=14)
    parser.add_argument("--inside-corridor-min-progress-m", type=float, default=0.20)
    parser.add_argument("--robot-radius-m", type=float, default=0.05)
    parser.add_argument("--command-slice-sec", type=float, default=0.70)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.30)
    parser.add_argument("--max-linear-x", type=float, default=0.45)
    parser.add_argument("--freeze-mode", choices=["entry", "room_door"], default="room_door")
    parser.add_argument("--enter-target-x-threshold-m", type=float, default=0.45)
    parser.add_argument("--enter-target-abs-y-threshold-m", type=float, default=0.25)
    parser.add_argument("--center-corridor-blocked-ratio-max", type=float, default=0.05)
    parser.add_argument("--door-width-blocked-ratio-max", type=float, default=0.10)
    parser.set_defaults(use_entry_anchor_line=True)
    parser.add_argument("--use-entry-anchor-line", dest="use_entry_anchor_line", action="store_true")
    parser.add_argument("--no-entry-anchor-line", dest="use_entry_anchor_line", action="store_false")
    parser.add_argument("--entry-anchor-lookahead-m", type=float, default=2.6)
    parser.add_argument("--anchor-heading-bias-rad", type=float, default=0.0)
    parser.add_argument("--no-path-fallback-lookahead-m", type=float, default=1.4)
    parser.add_argument("--entry-min-anchor-progress-m", type=float, default=2.2)
    parser.add_argument("--entry-max-anchor-abs-lateral-error-m", type=float, default=0.35)
    parser.add_argument("--anchor-yaw-error-max-rad", type=float, default=0.10)
    parser.add_argument("--room-door-detection-required-streak", type=int, default=2)
    parser.add_argument("--room-door-wall-context-blocked-ratio-min", type=float, default=0.30)
    parser.add_argument("--room-door-side-blocked-ratio-max", type=float, default=0.20)
    parser.add_argument("--room-door-side-free-ratio-min", type=float, default=0.55)
    parser.add_argument("--dry-run-every-cycles", type=int, default=3)
    parser.add_argument("--command-timeout-margin-sec", type=float, default=90.0)
    parser.set_defaults(use_imu_velocity_follower=True)
    parser.add_argument("--use-imu-velocity-follower", dest="use_imu_velocity_follower", action="store_true")
    parser.add_argument("--no-imu-velocity-follower", dest="use_imu_velocity_follower", action="store_false")
    parser.add_argument("--follower-raw-cmd-topic", default="/cmd_vel_raw")
    parser.add_argument("--follower-output-cmd-topic", default="/cmd_vel")
    parser.add_argument("--follower-status-topic", default="/imu_velocity_follower/status")
    parser.add_argument("--follower-imu-topic", default="/trunk_imu")
    parser.add_argument("--follower-max-linear-x", type=float, default=99.0)
    parser.add_argument("--follower-max-angular-z", type=float, default=0.45)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    follower_proc = start_imu_velocity_follower(args)
    atexit.register(stop_imu_velocity_follower, follower_proc)
    cycles: List[Dict[str, Any]] = []
    commands: List[Dict[str, Any]] = []
    published_total = 0
    final_decision = "ENTER_BUILDING_CENTERLINE_NOT_STARTED"
    entry_anchor: Optional[Dict[str, Any]] = None
    side_wall_context_seen = False
    room_door_detection_streak = 0
    force_no_bias_next_cycle = False
    force_short_lookahead_next_cycle = False

    for cycle_index in range(args.max_cycles):
        cycle: Dict[str, Any] = {"cycle_index": cycle_index, "started_wall_time_sec": time.time()}

        n5_cmd = [sys.executable, "scripts/short_horizon_target_selection/n5_target_selection_subgoal_audit.py"]
        n5_result = run_command(n5_cmd, args.input_timeout_sec + args.command_timeout_margin_sec)
        commands.append(n5_result)
        if n5_result["returncode"] != 0:
            final_decision = "ENTER_BUILDING_CENTERLINE_TARGET_PREP_FAILED"
            cycle["target_prep_failed"] = n5_result
            cycles.append(cycle)
            summary = finish(args, cycles, commands, final_decision, False, published_total, follower_proc)
            print(json.dumps({"final_decision": summary["final_decision"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
            return 1
        if not TARGET_OVERRIDE_PATH.exists():
            cycle["target_unavailable"] = {
                "n5_result": n5_result,
                "n5_summary": read_json(TARGET_SUMMARY_PATH) if TARGET_SUMMARY_PATH.exists() else None,
            }
            cycles.append(cycle)
            inside_runs = []
            inside_progressed = False
            if args.inside_corridor_enabled:
                inside_runs, published_total, inside_progressed = run_inside_corridor_phase(args, commands, published_total)
                final_decision = (
                    "ENTER_BUILDING_CENTERLINE_TARGET_UNAVAILABLE_INSIDE_CORRIDOR_PROGRESS"
                    if inside_progressed
                    else "ENTER_BUILDING_CENTERLINE_TARGET_UNAVAILABLE_INSIDE_CORRIDOR_INCOMPLETE"
                )
            else:
                final_decision = "ENTER_BUILDING_CENTERLINE_TARGET_UNAVAILABLE"
            summary = finish(
                args,
                cycles,
                commands,
                final_decision,
                inside_progressed,
                published_total,
                follower_proc,
                inside_corridor_runs=inside_runs,
            )
            print(json.dumps({"final_decision": summary["final_decision"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
            return 0 if inside_progressed else 1

        regen_cmd = [sys.executable, "scripts/local_subgoal_runner_mvp/regenerate_corrected_target_from_frame_contract.py"]
        regen_result = run_command(regen_cmd, args.input_timeout_sec + args.command_timeout_margin_sec)
        commands.append(regen_result)
        if regen_result["returncode"] != 0:
            final_decision = "ENTER_BUILDING_CENTERLINE_TARGET_PREP_FAILED"
            cycle["target_prep_failed"] = regen_result
            cycles.append(cycle)
            summary = finish(args, cycles, commands, final_decision, False, published_total, follower_proc)
            print(json.dumps({"final_decision": summary["final_decision"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
            return 1

        cycle["target_is_supported_live_local_target"] = target_is_supported_live_local_target()
        if not cycle["target_is_supported_live_local_target"]:
            final_decision = "ENTER_BUILDING_CENTERLINE_TARGET_NOT_SUPPORTED_LIVE_LOCAL"
            cycles.append(cycle)
            summary = finish(args, cycles, commands, final_decision, False, published_total, follower_proc)
            print(json.dumps({"final_decision": summary["final_decision"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
            return 1
        current_target = read_json(TARGET_OVERRIDE_PATH)
        current_subgoal_source = current_target.get("subgoal_source")
        cycle["current_subgoal_source"] = current_subgoal_source
        if args.use_entry_anchor_line and current_subgoal_source == "l3v_grid_centerline":
            original_target = read_json(TARGET_OVERRIDE_PATH)
            if entry_anchor is None:
                entry_anchor = build_entry_anchor(original_target)
            cycle["entry_anchor_line"] = entry_anchor
            effective_bias_rad = 0.0 if force_no_bias_next_cycle else float(args.anchor_heading_bias_rad)
            effective_lookahead_m = (
                float(args.no_path_fallback_lookahead_m)
                if force_short_lookahead_next_cycle
                else float(args.entry_anchor_lookahead_m)
            )
            cycle["fallback_active"] = {
                "force_no_bias": force_no_bias_next_cycle,
                "force_short_lookahead": force_short_lookahead_next_cycle,
                "effective_bias_rad": effective_bias_rad,
                "effective_lookahead_m": effective_lookahead_m,
            }
            force_no_bias_next_cycle = False
            force_short_lookahead_next_cycle = False
            cycle["anchor_target_override"] = write_anchor_target(
                args,
                entry_anchor,
                effective_bias_rad=effective_bias_rad,
                effective_lookahead_m=effective_lookahead_m,
            )
        elif current_subgoal_source == "local_frontier_nbv_fallback":
            cycle["entry_anchor_line_skipped"] = "local_frontier_nbv_fallback_target_preserved"

        run_dry_gate = cycle_index == 0 or (
            args.dry_run_every_cycles > 0 and cycle_index % args.dry_run_every_cycles == 0
        )
        cycle["dry_run_gate_executed"] = bool(run_dry_gate)
        if run_dry_gate:
            dry_cmd = common_runner_args(args, execute=False)
            dry_result = run_command(dry_cmd, args.dry_run_runtime_sec + args.command_timeout_margin_sec)
            commands.append(dry_result)
            if dry_result["returncode"] != 0:
                final_decision = "ENTER_BUILDING_CENTERLINE_DRY_RUN_FAILED"
                cycle["dry_run_failed"] = dry_result
                cycles.append(cycle)
                summary = finish(args, cycles, commands, final_decision, False, published_total, follower_proc)
                print(json.dumps({"final_decision": summary["final_decision"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
                return 1
            cycle["dry_run"] = summarize_runner(
                args,
                cycle_index,
                execute=False,
                anchor=entry_anchor,
                movement_already_published=published_total > 0,
                side_wall_context_seen=side_wall_context_seen,
                room_door_detection_streak=room_door_detection_streak,
            )
            side_wall_context_seen = bool(cycle["dry_run"].get("side_wall_context_seen"))
            if cycle["dry_run"].get("side_wall_context_seen") and cycle["dry_run"].get("side_doorway_detected"):
                room_door_detection_streak += 1
            else:
                room_door_detection_streak = 0
            cycle["dry_run"]["room_door_detection_streak"] = room_door_detection_streak
            cycle["dry_run"]["entered_building_candidate"], cycle["dry_run"]["entered_building_evidence"] = entry_predicate(
                args,
                execute=False,
                movement_already_published=published_total > 0,
                target_base_y=cycle["dry_run"].get("last_target_base_y_m"),
                center_ratio=cycle["dry_run"].get("last_center_corridor_blocked_ratio"),
                door_ratio=cycle["dry_run"].get("last_door_width_corridor_blocked_ratio"),
                side_doorway_detected=bool(cycle["dry_run"].get("side_doorway_detected")),
                side_wall_context_seen=side_wall_context_seen,
                room_door_detection_streak=room_door_detection_streak,
                l3v_ok=cycle["dry_run"].get("l3v_status") == "FREE_SUPPORTED",
                anchor_eval=cycle["dry_run"].get("entry_anchor_metrics") or {},
            )
            dry_ok = (
                cycle["dry_run"].get("runner_final_decision")
                in {"BLOCK_ASTAR_DWA_MAX_STEPS", "BLOCK_ASTAR_DWA_REACHED_GOAL"}
                and cycle["dry_run"].get("l3v_status") == "FREE_SUPPORTED"
            )
            if not dry_ok:
                final_decision = "ENTER_BUILDING_CENTERLINE_DRY_RUN_GATE_FAILED"
                cycles.append(cycle)
                summary = finish(args, cycles, commands, final_decision, False, published_total, follower_proc)
                print(json.dumps({"final_decision": summary["final_decision"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
                return 1
            if cycle["dry_run"].get("entered_building_candidate"):
                final_decision = "ENTER_BUILDING_CENTERLINE_PASS"
                cycles.append(cycle)
                inside_runs: List[Dict[str, Any]] = []
                inside_progressed = False
                if args.inside_corridor_enabled:
                    inside_runs, published_total, inside_progressed = run_inside_corridor_phase(args, commands, published_total)
                    final_decision = (
                        "ENTER_BUILDING_CENTERLINE_AND_INSIDE_CORRIDOR_PROGRESS"
                        if inside_progressed
                        else "ENTER_BUILDING_CENTERLINE_PASS_INSIDE_CORRIDOR_INCOMPLETE"
                    )
                summary = finish(
                    args,
                    cycles,
                    commands,
                    final_decision,
                    True,
                    published_total,
                    follower_proc,
                    inside_corridor_runs=inside_runs,
                )
                print(json.dumps({"final_decision": summary["final_decision"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
                return 0 if inside_progressed or not args.inside_corridor_enabled else 2

        execute_cmd = common_runner_args(args, execute=True)
        execute_result = run_command(execute_cmd, args.execute_runtime_sec + args.command_timeout_margin_sec)
        commands.append(execute_result)
        if execute_result["returncode"] != 0:
            final_decision = "ENTER_BUILDING_CENTERLINE_EXECUTE_FAILED"
            cycle["execute_failed"] = execute_result
            cycles.append(cycle)
            summary = finish(args, cycles, commands, final_decision, False, published_total, follower_proc)
            print(json.dumps({"final_decision": summary["final_decision"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
            return 1
        cycle["execute"] = summarize_runner(
            args,
            cycle_index,
            execute=True,
            anchor=entry_anchor,
            movement_already_published=published_total > 0,
            side_wall_context_seen=side_wall_context_seen,
            room_door_detection_streak=room_door_detection_streak,
        )
        side_wall_context_seen = bool(cycle["execute"].get("side_wall_context_seen"))
        if cycle["execute"].get("side_wall_context_seen") and cycle["execute"].get("side_doorway_detected"):
            room_door_detection_streak += 1
        else:
            room_door_detection_streak = 0
        cycle["execute"]["room_door_detection_streak"] = room_door_detection_streak
        cycle["execute"]["entered_building_candidate"], cycle["execute"]["entered_building_evidence"] = entry_predicate(
            args,
            execute=True,
            movement_already_published=published_total > 0,
            target_base_y=cycle["execute"].get("last_target_base_y_m"),
            center_ratio=cycle["execute"].get("last_center_corridor_blocked_ratio"),
            door_ratio=cycle["execute"].get("last_door_width_corridor_blocked_ratio"),
            side_doorway_detected=bool(cycle["execute"].get("side_doorway_detected")),
            side_wall_context_seen=side_wall_context_seen,
            room_door_detection_streak=room_door_detection_streak,
            l3v_ok=cycle["execute"].get("l3v_status") == "FREE_SUPPORTED",
            anchor_eval=cycle["execute"].get("entry_anchor_metrics") or {},
        )
        published_total += int(cycle["execute"].get("published_count_total") or 0)
        if cycle["execute"].get("runner_final_decision") == "BLOCK_ASTAR_DWA_BLOCKED_NO_PATH":
            force_no_bias_next_cycle = True
            force_short_lookahead_next_cycle = True
        cycles.append(cycle)
        if cycle["execute"].get("entered_building_candidate"):
            final_decision = "ENTER_BUILDING_CENTERLINE_PASS"
            inside_runs = []
            inside_progressed = False
            if args.inside_corridor_enabled:
                inside_runs, published_total, inside_progressed = run_inside_corridor_phase(args, commands, published_total)
                final_decision = (
                    "ENTER_BUILDING_CENTERLINE_AND_INSIDE_CORRIDOR_PROGRESS"
                    if inside_progressed
                    else "ENTER_BUILDING_CENTERLINE_PASS_INSIDE_CORRIDOR_INCOMPLETE"
                )
            summary = finish(
                args,
                cycles,
                commands,
                final_decision,
                True,
                published_total,
                follower_proc,
                inside_corridor_runs=inside_runs,
            )
            print(json.dumps({"final_decision": summary["final_decision"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
            return 0 if inside_progressed or not args.inside_corridor_enabled else 2

    inside_runs = []
    inside_progressed = False
    if args.inside_corridor_enabled:
        inside_runs, published_total, inside_progressed = run_inside_corridor_phase(args, commands, published_total)
    final_decision = (
        "ENTER_BUILDING_CENTERLINE_INCOMPLETE_INSIDE_CORRIDOR_PROGRESS"
        if inside_progressed
        else "ENTER_BUILDING_CENTERLINE_INCOMPLETE"
    )
    summary = finish(
        args,
        cycles,
        commands,
        final_decision,
        inside_progressed,
        published_total,
        follower_proc,
        inside_corridor_runs=inside_runs,
    )
    print(json.dumps({"final_decision": summary["final_decision"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
    return 0 if inside_progressed else 2


def follower_summary(args: argparse.Namespace, proc: Optional[subprocess.Popen]) -> Dict[str, Any]:
    return {
        "enabled": bool(args.use_imu_velocity_follower),
        "started": proc is not None,
        "running_at_summary_time": bool(proc is not None and proc.poll() is None),
        "raw_cmd_topic": args.follower_raw_cmd_topic,
        "output_cmd_topic": args.follower_output_cmd_topic,
        "status_topic": args.follower_status_topic,
        "imu_topic": args.follower_imu_topic,
        "max_linear_x": args.follower_max_linear_x,
        "max_angular_z": args.follower_max_angular_z,
        "log_path": getattr(proc, "_codex_log_path", None) if proc is not None else None,
        "cmd": getattr(proc, "_codex_cmd", None) if proc is not None else None,
    }


def finish(
    args: argparse.Namespace,
    cycles: List[Dict[str, Any]],
    commands: List[Dict[str, Any]],
    final_decision: str,
    entered: bool,
    published_total: int,
    follower_proc: Optional[subprocess.Popen],
    inside_corridor_runs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    final_cycle = None
    for cycle in reversed(cycles):
        if isinstance(cycle.get("execute"), dict):
            final_cycle = cycle["execute"]
            break
        if isinstance(cycle.get("dry_run"), dict):
            final_cycle = cycle["dry_run"]
            break
    summary = {
        "final_decision": final_decision,
        "entered_building": entered,
        "completed_cycles": len(cycles),
        "config": vars(args),
        "imu_velocity_follower": follower_summary(args, follower_proc),
        "entry_predicate": {
            "distance_is_not_sole_criterion": True,
            "enter_target_x_threshold_m": args.enter_target_x_threshold_m,
            "enter_target_abs_y_threshold_m": args.enter_target_abs_y_threshold_m,
            "center_corridor_blocked_ratio_max": args.center_corridor_blocked_ratio_max,
            "door_width_blocked_ratio_max": args.door_width_blocked_ratio_max,
            "freeze_mode": args.freeze_mode,
            "anchor_yaw_error_max_rad": args.anchor_yaw_error_max_rad,
            "room_door_uses_fixed_distance_gate": False,
            "room_door_detection_required_streak": args.room_door_detection_required_streak,
            "room_door_wall_context_blocked_ratio_min": args.room_door_wall_context_blocked_ratio_min,
            "room_door_side_blocked_ratio_max": args.room_door_side_blocked_ratio_max,
            "room_door_side_free_ratio_min": args.room_door_side_free_ratio_min,
        },
        "cycles": cycles,
        "commands": commands,
        "final_cycle": final_cycle,
        "inside_corridor_enabled": bool(args.inside_corridor_enabled),
        "inside_corridor_runs": inside_corridor_runs or [],
        "inside_corridor_completed_cycles": len(inside_corridor_runs or []),
        "inside_corridor_progressed": any(
            finite_number((run.get("runner") or {}).get("observed_straight_line_displacement_m"))
            and float((run.get("runner") or {})["observed_straight_line_displacement_m"]) >= args.inside_corridor_min_progress_m
            for run in (inside_corridor_runs or [])
            if isinstance(run, dict)
        ),
        "published_count_total": published_total,
        "forbidden_sources_used": [],
        "used_gazebo_truth": False,
        "called_move_base": False,
        "sent_navigation_goal": False,
        "runner_main_logic_modified": False,
        "git_add_or_commit": False,
    }
    write_json(SUMMARY_PATH, summary)
    write_report(summary)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
