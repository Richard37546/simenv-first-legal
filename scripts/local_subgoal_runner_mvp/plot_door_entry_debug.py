#!/usr/bin/env python3
"""Offline door-entry debug plot from saved JSON logs.

This script is diagnostic-only. It reads existing debug JSON files and writes a
PNG; it does not publish ROS topics or affect navigation decisions.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle


ROOT = Path(__file__).resolve().parents[2]
DOORWAY_PATH = ROOT / "debug/doorway_candidate_detector/latest_doorway_candidate.json"
TARGET_PATH = ROOT / "debug/short_horizon_target_selection/short_horizon_target_override.json"
RUNNER_PATH = ROOT / "debug/block_astar_dwa_mature/block_astar_dwa_mature_summary.json"
OUT_PATH = ROOT / "audit_logs/door_entry_debug.png"


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def target_to_base(target_xy: Sequence[float], pose: Sequence[float]) -> Tuple[float, float]:
    dx = float(target_xy[0]) - float(pose[0])
    dy = float(target_xy[1]) - float(pose[1])
    yaw = float(pose[2])
    return (
        math.cos(yaw) * dx + math.sin(yaw) * dy,
        -math.sin(yaw) * dx + math.cos(yaw) * dy,
    )


def selected_opening(target: Dict[str, Any], doorway: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    trigger = target.get("room_side_gap_trigger") if isinstance(target.get("room_side_gap_trigger"), dict) else {}
    selected = trigger.get("selected_opening") if isinstance(trigger.get("selected_opening"), dict) else {}
    side = trigger.get("door_side")
    if selected and side in {"left", "right"}:
        return str(side), selected

    profile = doorway.get("doorway_geometry_profile") if isinstance(doorway.get("doorway_geometry_profile"), dict) else {}
    for candidate_side in ("left", "right"):
        side_profile = profile.get(candidate_side) if isinstance(profile.get(candidate_side), dict) else {}
        selected = side_profile.get("selected_opening") if isinstance(side_profile.get("selected_opening"), dict) else {}
        if selected:
            return candidate_side, selected
    return "", {}


def iter_bins(doorway: Dict[str, Any], side: str) -> Iterable[Dict[str, Any]]:
    profile = doorway.get("doorway_geometry_profile") if isinstance(doorway.get("doorway_geometry_profile"), dict) else {}
    side_profile = profile.get(side) if isinstance(profile.get(side), dict) else {}
    bins = side_profile.get("bins")
    return bins if isinstance(bins, list) else []


def draw_profile_bins(ax: Any, doorway: Dict[str, Any], side: str) -> None:
    y0, y1 = (0.95, 1.45) if side == "left" else (-1.45, -0.95)
    for item in iter_bins(doorway, side):
        xr = item.get("x_range_m")
        if not (isinstance(xr, list) and len(xr) == 2 and finite(xr[0]) and finite(xr[1])):
            continue
        cls = item.get("classification")
        color = {
            "open": "#d7f5d7",
            "wall_or_unknown": "#555555",
            "mixed": "#cfcfcf",
        }.get(cls, "#eeeeee")
        ax.add_patch(
            Rectangle(
                (float(xr[0]), y0),
                float(xr[1]) - float(xr[0]),
                y1 - y0,
                facecolor=color,
                edgecolor="#888888",
                linewidth=0.4,
                alpha=0.95,
            )
        )


def draw_segment(ax: Any, side: str, seg: Dict[str, Any]) -> None:
    if not all(finite(seg.get(k)) for k in ("start_x_m", "end_x_m", "center_x_m")):
        return
    y = 1.15 if side == "left" else -1.15
    start = float(seg["start_x_m"])
    end = float(seg["end_x_m"])
    center = float(seg["center_x_m"])
    ax.plot([start, end], [y, y], color="#00a651", linewidth=5, solid_capstyle="butt", label="opening start/end")
    ax.scatter([center], [y], color="#006b2e", s=70, zorder=5, label="opening center")
    ax.text(start, y + (0.12 if side == "left" else -0.18), f"start {start:.2f}", fontsize=8)
    ax.text(end, y + (0.12 if side == "left" else -0.18), f"end {end:.2f}", fontsize=8, ha="right")
    ax.text(center, y + (0.28 if side == "left" else -0.34), f"center {center:.2f}", fontsize=8, ha="center")


def main() -> int:
    doorway = read_json(DOORWAY_PATH)
    target = read_json(TARGET_PATH)
    runner = read_json(RUNNER_PATH)
    side, opening = selected_opening(target, doorway)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title("Door Entry Debug (base frame, profile-derived local projection)")
    ax.set_xlabel("x forward in robot base frame (m)")
    ax.set_ylabel("y left in robot base frame (m)")
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    ax.add_patch(Rectangle((0.0, -0.30), 3.0, 0.60, facecolor="#f8f8f8", edgecolor="#bbbbbb", alpha=0.7, label="corridor center"))
    draw_profile_bins(ax, doorway, "left")
    draw_profile_bins(ax, doorway, "right")

    ax.add_patch(Rectangle((0.0, -1.6), 0.6, 3.2, facecolor="#4da3ff", alpha=0.10, edgecolor="#4da3ff", linewidth=1.5, label="near field x=[0,0.6]"))
    ax.add_patch(Rectangle((-0.08, -0.18), 0.16, 0.36, facecolor="#1f77b4", edgecolor="#0b315c", zorder=6))
    ax.add_patch(FancyArrow(0, 0, 0.35, 0, width=0.03, head_width=0.12, head_length=0.12, color="#1f77b4", zorder=7))
    ax.text(0.02, 0.22, "robot base\nx forward", fontsize=9, color="#0b315c")

    if side and opening:
        draw_segment(ax, side, opening)
        ax.text(2.05, 1.50 if side == "left" else -1.50, f"door_side={side}", fontsize=10, weight="bold")

    subgoal = target.get("subgoal_base_xy")
    if isinstance(subgoal, list) and len(subgoal) == 2 and all(finite(v) for v in subgoal):
        ax.scatter([float(subgoal[0])], [float(subgoal[1])], marker="*", s=180, color="#ff7f0e", label="subgoal_base_xy", zorder=8)
        ax.plot([0, float(subgoal[0])], [0, float(subgoal[1])], color="#ff7f0e", linestyle=":", linewidth=1.5)
        ax.text(float(subgoal[0]) + 0.04, float(subgoal[1]), f"subgoal {float(subgoal[0]):.2f},{float(subgoal[1]):.2f}", fontsize=9, color="#a64b00")

    steps = runner.get("steps") if isinstance(runner.get("steps"), list) else []
    colors = ["#d62728", "#9467bd", "#8c564b"]
    for idx, step in enumerate(steps[:3]):
        tb = step.get("target_base_xy") if isinstance(step, dict) else None
        if isinstance(tb, list) and len(tb) == 2 and all(finite(v) for v in tb):
            color = colors[idx % len(colors)]
            ax.scatter([float(tb[0])], [float(tb[1])], marker="x", s=120, color=color, label=f"runner target_base_xy step{idx}", zorder=9)
            ax.text(float(tb[0]) + 0.04, float(tb[1]) - 0.08, f"runner{idx} {float(tb[0]):.2f},{float(tb[1]):.2f}", fontsize=9, color=color)

    info_lines = [
        f"source={target.get('source')}",
        f"final_decision={runner.get('final_decision')}",
    ]
    trigger = target.get("room_side_gap_trigger") if isinstance(target.get("room_side_gap_trigger"), dict) else {}
    if trigger:
        info_lines.append(f"trigger={trigger.get('trigger_source')} center_x={trigger.get('center_x_base_m')} width={trigger.get('opening_width_m')}")
    turn = target.get("room_side_gap_turn_result") if isinstance(target.get("room_side_gap_turn_result"), dict) else {}
    if turn:
        info_lines.append(f"turn yaw: {turn.get('initial_yaw_rad')} -> {turn.get('final_yaw_rad')} abs_delta={turn.get('actual_abs_yaw_delta_rad')}")
    ax.text(-0.45, -1.62, "\n".join(info_lines), fontsize=8, va="bottom", family="monospace")

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, label in zip(handles, labels):
        unique.setdefault(label, h)
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(OUT_PATH), dpi=180)
    print(str(OUT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
