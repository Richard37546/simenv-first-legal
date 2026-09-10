#!/usr/bin/env python3
"""Read-only room frontier viewpoint selector.

This is an MVP target selector for the post-doorway phase. It does not move the
robot; it only reports a candidate viewpoint in the local base frame.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "room_frontier_viewpoint_selector"
LATEST_PATH = OUT / "latest_room_viewpoint.json"
VISION_PATH = ROOT / "debug" / "vision_scene_semantics" / "latest_project_scene_analysis.json"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def grid_array(msg: OccupancyGrid) -> np.ndarray:
    return np.array(msg.data, dtype=np.int16).reshape((int(msg.info.height), int(msg.info.width)))


def cell_to_local_xy(msg: OccupancyGrid, cell: Tuple[int, int]) -> Tuple[float, float]:
    res = float(msg.info.resolution)
    ox = float(msg.info.origin.position.x)
    oy = float(msg.info.origin.position.y)
    return ox + (cell[0] + 0.5) * res, oy + (cell[1] + 0.5) * res


def local_xy_to_cell(msg: OccupancyGrid, x: float, y: float) -> Tuple[int, int]:
    res = float(msg.info.resolution)
    ox = float(msg.info.origin.position.x)
    oy = float(msg.info.origin.position.y)
    return int(math.floor((x - ox) / res)), int(math.floor((y - oy) / res))


def semantics_summary(path: Path, max_age_sec: float) -> Dict[str, Any]:
    data = read_json(path)
    semantics = data.get("semantics") if isinstance(data.get("semantics"), dict) else {}
    danger = semantics.get("danger_source") if isinstance(semantics.get("danger_source"), dict) else {}
    age = None
    if isinstance(data.get("analysis_wall_time_sec"), (int, float)):
        age = time.time() - float(data["analysis_wall_time_sec"])
    return {
        "vision_available": bool(data),
        "vision_fresh": age is not None and age <= max_age_sec,
        "vision_age_sec": age,
        "vision_primary_scene": str(semantics.get("primary_scene") or "unknown"),
        "danger_source_visible": bool(danger.get("visible")),
        "vision_confidence": float(semantics.get("confidence") or 0.0) if isinstance(semantics.get("confidence"), (int, float)) else 0.0,
    }


class RoomFrontierViewpointSelector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.latest_grid: Optional[OccupancyGrid] = None
        self.pub = rospy.Publisher(args.output_topic, String, queue_size=2)
        rospy.Subscriber(args.grid_topic, OccupancyGrid, self.grid_cb, queue_size=1)
        threading.Thread(target=self.wall_loop, daemon=True).start()

    def grid_cb(self, msg: OccupancyGrid) -> None:
        self.latest_grid = msg

    def wall_loop(self) -> None:
        while not rospy.is_shutdown():
            self.timer_cb(None)
            time.sleep(max(0.05, float(self.args.publish_interval_sec)))

    def frontier_mask(self, grid: np.ndarray) -> np.ndarray:
        free = grid == 0
        unknown = grid == -1
        frontier = np.zeros_like(free, dtype=bool)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.zeros_like(unknown, dtype=bool)
            if dr == 1:
                shifted[1:, :] = unknown[:-1, :]
            elif dr == -1:
                shifted[:-1, :] = unknown[1:, :]
            elif dc == 1:
                shifted[:, 1:] = unknown[:, :-1]
            elif dc == -1:
                shifted[:, :-1] = unknown[:, 1:]
            frontier |= free & shifted
        return frontier

    def select_viewpoint(self, msg: OccupancyGrid, grid: np.ndarray) -> Optional[Dict[str, Any]]:
        frontier = self.frontier_mask(grid)
        rows, cols = np.where(frontier)
        scored = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            x, y = cell_to_local_xy(msg, (r, c))
            if not (self.args.x_min_m <= x <= self.args.x_max_m):
                continue
            if abs(y) > self.args.y_abs_max_m:
                continue
            score = x - self.args.lateral_penalty * abs(y)
            scored.append((score, r, c, x, y))
        if not scored:
            fallback = local_xy_to_cell(msg, self.args.fallback_x_m, 0.0)
            r = max(0, min(grid.shape[0] - 1, fallback[0]))
            c = max(0, min(grid.shape[1] - 1, fallback[1]))
            if grid[r, c] == 0:
                x, y = cell_to_local_xy(msg, (r, c))
                return {"mode": "fallback_forward_free", "base_xy": [x, y], "yaw_rad": 0.0, "score": 0.0}
            return None
        scored.sort(reverse=True)
        score, _r, _c, x, y = scored[0]
        return {"mode": "frontier", "base_xy": [x, y], "yaw_rad": math.atan2(y, max(x, 1e-6)), "score": score}

    def timer_cb(self, _event: Any) -> None:
        msg = self.latest_grid
        if msg is None:
            payload = {"final_decision": "ROOM_VIEWPOINT_WAITING_FOR_GRID", "forbidden_sources_used": []}
            write_json(LATEST_PATH, payload)
            self.pub.publish(String(data=json.dumps(payload, sort_keys=True)))
            return
        grid = grid_array(msg)
        viewpoint = self.select_viewpoint(msg, grid)
        vision = semantics_summary(Path(self.args.vision_json_path), self.args.vision_max_age_sec)
        payload = {
            "final_decision": "ROOM_VIEWPOINT_READY" if viewpoint else "ROOM_VIEWPOINT_NOT_AVAILABLE",
            "next_viewpoint_base": viewpoint,
            "frontier_cell_count": int(self.frontier_mask(grid).sum()),
            "vision_support": vision,
            "room_search_state": "DANGER_VISIBLE" if vision["danger_source_visible"] else "SEARCHING",
            "grid_stamp_sec": float(msg.header.stamp.to_sec()) if msg.header.stamp else None,
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
    parser.add_argument("--output-topic", default="/team/room_frontier_viewpoint")
    parser.add_argument("--vision-json-path", default=str(VISION_PATH))
    parser.add_argument("--vision-max-age-sec", type=float, default=8.0)
    parser.add_argument("--publish-interval-sec", type=float, default=0.5)
    parser.add_argument("--x-min-m", type=float, default=0.40)
    parser.add_argument("--x-max-m", type=float, default=2.60)
    parser.add_argument("--y-abs-max-m", type=float, default=1.20)
    parser.add_argument("--lateral-penalty", type=float, default=0.35)
    parser.add_argument("--fallback-x-m", type=float, default=1.40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args(rospy.myargv()[1:])
    rospy.init_node("room_frontier_viewpoint_selector", anonymous=False)
    RoomFrontierViewpointSelector(args)
    rospy.spin()


if __name__ == "__main__":
    main()
