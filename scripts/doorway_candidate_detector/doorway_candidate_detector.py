#!/usr/bin/env python3
"""Geometry-first, vision-assisted doorway candidate detector.

This node is read-only. It uses the L3V local traversability grid as the hard
gate and the optional vision semantics JSON as an advisory signal.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "doorway_candidate_detector"
LATEST_PATH = OUT / "latest_doorway_candidate.json"
VISION_PATH = ROOT / "debug" / "vision_scene_semantics" / "latest_project_scene_analysis.json"
STAGE_PATH = ROOT / "debug" / "navigation_stage" / "current_stage.json"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def pose_tuple(msg: Odometry) -> Tuple[float, float, float]:
    pose = msg.pose.pose
    return float(pose.position.x), float(pose.position.y), yaw_from_quat(pose.orientation)


def grid_array(msg: OccupancyGrid) -> np.ndarray:
    return np.array(msg.data, dtype=np.int16).reshape((int(msg.info.height), int(msg.info.width)))


def rect_counts(msg: OccupancyGrid, grid: np.ndarray, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Dict[str, Any]:
    res = float(msg.info.resolution)
    ox = float(msg.info.origin.position.x)
    oy = float(msg.info.origin.position.y)
    r0 = max(0, int(math.floor((min(x_range) - ox) / res)))
    r1 = min(grid.shape[0], int(math.ceil((max(x_range) - ox) / res)))
    c0 = max(0, int(math.floor((min(y_range) - oy) / res)))
    c1 = min(grid.shape[1], int(math.ceil((max(y_range) - oy) / res)))
    window = grid[r0:r1, c0:c1]
    cell_count = int(window.size)
    free_count = int((window == 0).sum()) if cell_count else 0
    occupied_count = int((window == 100).sum()) if cell_count else 0
    unknown_count = int((window == -1).sum()) if cell_count else 0
    blocked_count = occupied_count + unknown_count
    return {
        "x_range_m": [float(x_range[0]), float(x_range[1])],
        "y_range_m": [float(y_range[0]), float(y_range[1])],
        "cell_count": cell_count,
        "free_count": free_count,
        "occupied_count": occupied_count,
        "unknown_count": unknown_count,
        "blocked_count": blocked_count,
        "free_ratio": float(free_count / cell_count) if cell_count else None,
        "blocked_ratio": float(blocked_count / cell_count) if cell_count else None,
    }


def side_opening_profile(
    msg: OccupancyGrid,
    grid: np.ndarray,
    side: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    bin_width_m: float,
    open_free_ratio_min: float,
    open_blocked_ratio_max: float,
    wall_blocked_ratio_min: float,
    opening_min_width_m: float,
    opening_max_width_m: float,
) -> Dict[str, Any]:
    bins: List[Dict[str, Any]] = []
    x = float(x_range[0])
    x_end = float(x_range[1])
    width = max(0.05, float(bin_width_m))
    while x < x_end - 1e-6:
        nx = min(x + width, x_end)
        counts = rect_counts(msg, grid, (x, nx), y_range)
        free_ratio = float(counts.get("free_ratio") or 0.0)
        blocked_ratio = float(counts.get("blocked_ratio") or 0.0)
        open_like = free_ratio >= open_free_ratio_min and blocked_ratio <= open_blocked_ratio_max
        wall_like = blocked_ratio >= wall_blocked_ratio_min
        bins.append(
            {
                "index": len(bins),
                "x_range_m": [float(x), float(nx)],
                "x_center_m": float(0.5 * (x + nx)),
                "free_ratio": counts.get("free_ratio"),
                "blocked_ratio": counts.get("blocked_ratio"),
                "occupied_count": counts.get("occupied_count"),
                "unknown_count": counts.get("unknown_count"),
                "cell_count": counts.get("cell_count"),
                "classification": "open" if open_like else ("wall_or_unknown" if wall_like else "mixed"),
            }
        )
        x = nx

    segments: List[Dict[str, Any]] = []
    start_idx: Optional[int] = None
    for idx, item in enumerate(bins + [{"classification": "sentinel"}]):
        if item.get("classification") == "open":
            if start_idx is None:
                start_idx = idx
            continue
        if start_idx is not None:
            end_idx = idx - 1
            start_x = float(bins[start_idx]["x_range_m"][0])
            end_x = float(bins[end_idx]["x_range_m"][1])
            before = bins[start_idx - 1] if start_idx > 0 else None
            after = bins[end_idx + 1] if end_idx + 1 < len(bins) else None
            before_wall = bool(before and before.get("classification") == "wall_or_unknown")
            after_wall = bool(after and after.get("classification") == "wall_or_unknown")
            segment_width = end_x - start_x
            center_x = 0.5 * (start_x + end_x)
            width_plausible = opening_min_width_m <= segment_width <= opening_max_width_m
            center_estimated = bool(width_plausible and before_wall and after_wall)
            partial = bool(width_plausible and not center_estimated)
            segments.append(
                {
                    "start_x_m": start_x,
                    "end_x_m": end_x,
                    "center_x_m": center_x,
                    "width_m": segment_width,
                    "start_bin_index": start_idx,
                    "end_bin_index": end_idx,
                    "before_wall_or_unknown": before_wall,
                    "after_wall_or_unknown": after_wall,
                    "width_plausible": width_plausible,
                    "opening_center_estimated": center_estimated,
                    "partial_opening_observed": partial,
                    "opening_center_base_xy": [center_x, 0.5 * (y_range[0] + y_range[1])],
                }
            )
            start_idx = None

    def segment_score(seg: Dict[str, Any]) -> Tuple[int, float, float]:
        complete = 1 if seg.get("opening_center_estimated") else 0
        plausible = 1 if seg.get("width_plausible") else 0
        return complete + plausible, -abs(float(seg.get("width_m") or 0.0) - 0.9), -float(seg.get("start_x_m") or 0.0)

    selected = max(segments, key=segment_score) if segments else None
    return {
        "side": side,
        "x_range_m": [float(x_range[0]), float(x_range[1])],
        "y_range_m": [float(y_range[0]), float(y_range[1])],
        "bin_width_m": width,
        "open_free_ratio_min": float(open_free_ratio_min),
        "open_blocked_ratio_max": float(open_blocked_ratio_max),
        "wall_blocked_ratio_min": float(wall_blocked_ratio_min),
        "opening_min_width_m": float(opening_min_width_m),
        "opening_max_width_m": float(opening_max_width_m),
        "bins": bins,
        "open_segments": segments,
        "selected_opening": selected,
        "opening_segment_count": len(segments),
        "opening_center_estimated": bool(selected and selected.get("opening_center_estimated")),
        "partial_opening_observed": bool(selected and selected.get("partial_opening_observed")),
    }


def vision_score(vision: Dict[str, Any], max_age_sec: float) -> Dict[str, Any]:
    now = time.time()
    age = None
    if isinstance(vision.get("analysis_wall_time_sec"), (int, float)):
        age = now - float(vision["analysis_wall_time_sec"])
    semantics = vision.get("semantics") if isinstance(vision.get("semantics"), dict) else {}
    scene = str(semantics.get("primary_scene") or "unknown")
    room = semantics.get("room_opening") if isinstance(semantics.get("room_opening"), dict) else {}
    door = semantics.get("door") if isinstance(semantics.get("door"), dict) else {}
    passability = str(semantics.get("passability") or "unknown")
    base_conf = float(semantics.get("confidence") or 0.0) if isinstance(semantics.get("confidence"), (int, float)) else 0.0
    fresh = age is not None and age <= max_age_sec
    positive = scene in ("room_opening", "normal_door", "room", "corridor") or bool(room.get("visible")) or bool(door.get("visible"))
    passable = passability in ("passable", "narrow", "unknown")
    score = base_conf if fresh and positive and passable else 0.0
    return {
        "vision_available": bool(vision),
        "vision_fresh": fresh,
        "vision_age_sec": age,
        "vision_primary_scene": scene,
        "vision_passability": passability,
        "vision_room_opening_visible": bool(room.get("visible")),
        "vision_door_visible": bool(door.get("visible")),
        "vision_confidence": base_conf,
        "vision_doorway_support_score": score,
    }


def current_stage(path: Path) -> Dict[str, Any]:
    data = read_json(path)
    stage = str(data.get("stage") or "unknown")
    payload = {"stage": stage, "source": str(path), "available": bool(data)}
    for key in ("room_zone_reached", "anchor_progress_m", "room_zone_start_progress_m"):
        if key in data:
            payload[key] = data[key]
    return payload


class DoorwayCandidateDetector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.latest_grid: Optional[OccupancyGrid] = None
        self.latest_odom: Optional[Odometry] = None
        self.streak = {"left": 0, "right": 0}
        self.soft_streak = {"left": 0, "right": 0}
        self.history: List[Dict[str, Any]] = []
        self.pub = rospy.Publisher(args.output_topic, String, queue_size=2)
        rospy.Subscriber(args.grid_topic, OccupancyGrid, self.grid_cb, queue_size=1)
        rospy.Subscriber(args.odom_topic, Odometry, self.odom_cb, queue_size=5)
        threading.Thread(target=self.wall_loop, daemon=True).start()

    def grid_cb(self, msg: OccupancyGrid) -> None:
        self.latest_grid = msg

    def odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def wall_loop(self) -> None:
        while not rospy.is_shutdown():
            self.timer_cb(None)
            time.sleep(max(0.05, float(self.args.publish_interval_sec)))

    def side_candidate(self, side: str, side_counts: Dict[str, Any], wall_counts: Dict[str, Any]) -> Dict[str, Any]:
        side_free = float(side_counts.get("free_ratio") or 0.0)
        side_blocked = float(side_counts.get("blocked_ratio") or 1.0)
        wall_blocked = float(wall_counts.get("blocked_ratio") or 0.0)
        wall_context_pass = wall_blocked >= self.args.wall_context_blocked_ratio_min
        geometry_pass = (
            side_free >= self.args.side_free_ratio_min
            and side_blocked <= self.args.side_blocked_ratio_max
        )
        soft_geometry_pass = (
            side_free >= self.args.room_zone_soft_side_free_ratio_min
            and side_blocked <= self.args.room_zone_soft_side_blocked_ratio_max
            and wall_context_pass
        )
        self.streak[side] = self.streak[side] + 1 if geometry_pass else 0
        self.soft_streak[side] = self.soft_streak[side] + 1 if soft_geometry_pass else 0
        geometry_conf = min(1.0, 0.45 * side_free + 0.35 * wall_blocked + 0.20 * (1.0 - side_blocked))
        return {
            "side": side,
            "geometry_pass": geometry_pass,
            "soft_geometry_pass": soft_geometry_pass,
            "wall_context_pass": wall_context_pass,
            "wall_context_required_for_confirmation": False,
            "geometry_confidence": geometry_conf,
            "stable_frame_count": self.streak[side],
            "soft_stable_frame_count": self.soft_streak[side],
            "side_probe_counts": side_counts,
            "wall_context_counts": wall_counts,
        }

    def entry_pose(
        self,
        pose: Optional[Tuple[float, float, float]],
        side: str,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if pose is None:
            return None
        selected = profile.get("selected_opening") if isinstance(profile, dict) else None
        center_base = selected.get("opening_center_base_xy") if isinstance(selected, dict) else None
        if (
            isinstance(selected, dict)
            and selected.get("opening_center_estimated")
            and isinstance(center_base, list)
            and len(center_base) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in center_base)
        ):
            x_local = float(center_base[0])
            y_local = float(center_base[1])
            source = "local_side_opening_profile_center"
        else:
            x_local = self.args.entry_x_m
            y_local = self.args.entry_abs_y_m if side == "left" else -self.args.entry_abs_y_m
            source = "local_side_opening_geometry_fallback"
        yaw_local = math.pi / 2.0 if side == "left" else -math.pi / 2.0
        c = math.cos(pose[2])
        s = math.sin(pose[2])
        x = pose[0] + c * x_local - s * y_local
        y = pose[1] + s * x_local + c * y_local
        yaw = math.atan2(math.sin(pose[2] + yaw_local), math.cos(pose[2] + yaw_local))
        return {
            "x": x,
            "y": y,
            "yaw": yaw,
            "source": source,
            "base_xy": [x_local, y_local],
            "opening_center_estimated": bool(source == "local_side_opening_profile_center"),
        }

    def update_history(self, stage: Dict[str, Any], left: Dict[str, Any], right: Dict[str, Any], best: Dict[str, Any]) -> List[Dict[str, Any]]:
        item = {
            "wall_time_sec": time.time(),
            "stage": stage.get("stage"),
            "room_zone_reached": bool(stage.get("room_zone_reached")),
            "anchor_progress_m": stage.get("anchor_progress_m"),
            "best_side": best.get("side"),
            "best_geometry_confidence": best.get("geometry_confidence"),
            "left_free_ratio": (left.get("side_probe_counts") or {}).get("free_ratio"),
            "left_blocked_ratio": (left.get("side_probe_counts") or {}).get("blocked_ratio"),
            "left_geometry_pass": left.get("geometry_pass"),
            "left_soft_geometry_pass": left.get("soft_geometry_pass"),
            "left_soft_stable_frame_count": left.get("soft_stable_frame_count"),
            "right_free_ratio": (right.get("side_probe_counts") or {}).get("free_ratio"),
            "right_blocked_ratio": (right.get("side_probe_counts") or {}).get("blocked_ratio"),
            "right_geometry_pass": right.get("geometry_pass"),
            "right_soft_geometry_pass": right.get("soft_geometry_pass"),
            "right_soft_stable_frame_count": right.get("soft_stable_frame_count"),
        }
        self.history.append(item)
        max_len = max(1, int(self.args.history_max_samples))
        if len(self.history) > max_len:
            self.history = self.history[-max_len:]
        return list(self.history)

    def timer_cb(self, _event: Any) -> None:
        grid_msg = self.latest_grid
        stage = current_stage(Path(self.args.stage_json_path))
        active_stages = {s.strip() for s in self.args.active_stages.split(",") if s.strip()}
        stage_active = bool(self.args.ignore_stage_gate or stage["stage"] in active_stages)
        if grid_msg is None:
            payload = {
                "final_decision": "DOORWAY_CANDIDATE_WAITING_FOR_GRID",
                "navigation_stage": stage,
                "stage_gate_active": stage_active,
                "forbidden_sources_used": [],
            }
            write_json(LATEST_PATH, payload)
            self.pub.publish(String(data=json.dumps(payload, sort_keys=True)))
            return
        grid = grid_array(grid_msg)
        pose = pose_tuple(self.latest_odom) if self.latest_odom is not None else None
        center = rect_counts(grid_msg, grid, (0.35, 1.35), (-0.30, 0.30))
        left_side = rect_counts(grid_msg, grid, (0.60, 2.20), (0.95, 1.45))
        right_side = rect_counts(grid_msg, grid, (0.60, 2.20), (-1.45, -0.95))
        left_wall = rect_counts(grid_msg, grid, (0.60, 2.60), (0.90, 1.50))
        right_wall = rect_counts(grid_msg, grid, (0.60, 2.60), (-1.50, -0.90))
        left_profile = side_opening_profile(
            grid_msg,
            grid,
            "left",
            (self.args.profile_x_min_m, self.args.profile_x_max_m),
            (self.args.profile_left_y_min_m, self.args.profile_left_y_max_m),
            self.args.profile_bin_width_m,
            self.args.profile_open_free_ratio_min,
            self.args.profile_open_blocked_ratio_max,
            self.args.profile_wall_blocked_ratio_min,
            self.args.profile_opening_min_width_m,
            self.args.profile_opening_max_width_m,
        )
        right_profile = side_opening_profile(
            grid_msg,
            grid,
            "right",
            (self.args.profile_x_min_m, self.args.profile_x_max_m),
            (self.args.profile_right_y_min_m, self.args.profile_right_y_max_m),
            self.args.profile_bin_width_m,
            self.args.profile_open_free_ratio_min,
            self.args.profile_open_blocked_ratio_max,
            self.args.profile_wall_blocked_ratio_min,
            self.args.profile_opening_min_width_m,
            self.args.profile_opening_max_width_m,
        )
        left = self.side_candidate("left", left_side, left_wall)
        right = self.side_candidate("right", right_side, right_wall)
        vision = vision_score(read_json(Path(self.args.vision_json_path)), self.args.vision_max_age_sec)
        candidates = [left, right]
        candidates.sort(key=lambda c: (c["stable_frame_count"], c["geometry_confidence"]), reverse=True)
        best = candidates[0]
        soft_candidates = sorted(candidates, key=lambda c: (c["soft_stable_frame_count"], c["geometry_confidence"]), reverse=True)
        soft_best = soft_candidates[0]
        history = self.update_history(stage, left, right, best)
        room_zone_reached = bool(stage.get("room_zone_reached"))
        fused_conf = 0.75 * float(best["geometry_confidence"]) + 0.25 * float(vision["vision_doorway_support_score"])
        hard_doorway = bool(stage_active and best["geometry_pass"] and best["stable_frame_count"] >= self.args.required_stable_frames)
        soft_doorway = bool(
            stage_active
            and room_zone_reached
            and soft_best["soft_geometry_pass"]
            and soft_best["soft_stable_frame_count"] >= self.args.room_zone_soft_required_stable_frames
            and float(soft_best["geometry_confidence"]) >= self.args.room_zone_soft_confidence_min
        )
        doorway = bool(hard_doorway or soft_doorway)
        decision_side = best["side"] if hard_doorway else soft_best["side"]
        if not stage_active:
            final_decision = "DOORWAY_CANDIDATE_INACTIVE_STAGE"
        elif doorway:
            final_decision = "DOORWAY_CANDIDATE_READY"
        else:
            final_decision = "DOORWAY_CANDIDATE_NOT_CONFIRMED"
        payload = {
            "final_decision": final_decision,
            "doorway_candidate": doorway,
            "door_side": decision_side if doorway else None,
            "door_entry_pose_odom": self.entry_pose(
                pose,
                decision_side,
                left_profile if decision_side == "left" else right_profile,
            )
            if doorway
            else None,
            "opening_width_m": None,
            "fused_confidence": fused_conf,
            "decision_mode": "hard_geometry" if hard_doorway else ("room_zone_soft_history" if soft_doorway else "not_confirmed"),
            "room_zone_reached": room_zone_reached,
            "soft_best_side": soft_best["side"],
            "soft_best_confidence": soft_best["geometry_confidence"],
            "center_corridor_counts": center,
            "left_candidate": left,
            "right_candidate": right,
            "doorway_geometry_profile": {
                "left": left_profile,
                "right": right_profile,
                "selected_by_candidate_side": left_profile if decision_side == "left" else right_profile,
                "diagnostic_only": True,
                "control_logic_changed": False,
            },
            "doorway_candidate_history": history,
            "vision_support": vision,
            "navigation_stage": stage,
            "stage_gate_active": stage_active,
            "active_stages": sorted(active_stages),
            "grid_stamp_sec": float(grid_msg.header.stamp.to_sec()) if grid_msg.header.stamp else None,
            "odom_pose_x_y_yaw": list(pose) if pose else None,
            "output_topic": self.args.output_topic,
            "forbidden_sources_used": [],
            "called_move_base": False,
            "sent_navigation_goal": False,
            "cmd_vel_published": False,
        }
        write_json(LATEST_PATH, payload)
        self.pub.publish(String(data=json.dumps(payload, sort_keys=True, ensure_ascii=False)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-topic", default="/team/local_traversability_grid")
    parser.add_argument("--odom-topic", default="/team/livox/icp_odom_gated")
    parser.add_argument("--output-topic", default="/team/doorway_candidate")
    parser.add_argument("--vision-json-path", default=str(VISION_PATH))
    parser.add_argument("--stage-json-path", default=str(STAGE_PATH))
    parser.add_argument("--active-stages", default="inside_corridor,doorway_verify,room_entry,room_search")
    parser.add_argument("--ignore-stage-gate", action="store_true")
    parser.add_argument("--vision-max-age-sec", type=float, default=8.0)
    parser.add_argument("--publish-interval-sec", type=float, default=0.5)
    parser.add_argument("--required-stable-frames", type=int, default=3)
    parser.add_argument("--side-free-ratio-min", type=float, default=0.55)
    parser.add_argument("--side-blocked-ratio-max", type=float, default=0.20)
    parser.add_argument("--wall-context-blocked-ratio-min", type=float, default=0.30)
    parser.add_argument("--room-zone-soft-required-stable-frames", type=int, default=2)
    parser.add_argument("--room-zone-soft-side-free-ratio-min", type=float, default=0.35)
    parser.add_argument("--room-zone-soft-side-blocked-ratio-max", type=float, default=0.75)
    parser.add_argument("--room-zone-soft-confidence-min", type=float, default=0.38)
    parser.add_argument("--history-max-samples", type=int, default=24)
    parser.add_argument("--entry-x-m", type=float, default=1.20)
    parser.add_argument("--entry-abs-y-m", type=float, default=0.85)
    parser.add_argument("--profile-x-min-m", type=float, default=0.30)
    parser.add_argument("--profile-x-max-m", type=float, default=3.00)
    parser.add_argument("--profile-bin-width-m", type=float, default=0.15)
    parser.add_argument("--profile-left-y-min-m", type=float, default=0.75)
    parser.add_argument("--profile-left-y-max-m", type=float, default=1.50)
    parser.add_argument("--profile-right-y-min-m", type=float, default=-1.50)
    parser.add_argument("--profile-right-y-max-m", type=float, default=-0.75)
    parser.add_argument("--profile-open-free-ratio-min", type=float, default=0.45)
    parser.add_argument("--profile-open-blocked-ratio-max", type=float, default=0.55)
    parser.add_argument("--profile-wall-blocked-ratio-min", type=float, default=0.65)
    parser.add_argument("--profile-opening-min-width-m", type=float, default=0.45)
    parser.add_argument("--profile-opening-max-width-m", type=float, default=1.50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args(rospy.myargv()[1:])
    rospy.init_node("doorway_candidate_detector", anonymous=False)
    DoorwayCandidateDetector(args)
    rospy.spin()


if __name__ == "__main__":
    main()
