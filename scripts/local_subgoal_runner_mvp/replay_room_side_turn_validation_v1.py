#!/usr/bin/env python3
"""Offline run_0039 replay; reads archived observation JSON only and publishes nothing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from room_side_turn_validation_v1 import (
    committed_room_side_gap_trigger,
    select_follow_corridor_opening_action,
    select_room_side_gap_candidate,
    update_committed_room_side_gap,
    update_room_side_gap_observation_from_progress,
)


ROOT = Path(__file__).resolve().parents[2]


def finite(value: Any) -> bool:
    return isinstance(value, (int, float))


def replay(archive: Path) -> Dict[str, Any]:
    summary = json.loads((archive / "state_machine_navigation_summary.json").read_text(encoding="utf-8"))
    parameters = dict(summary.get("config") or {})
    current: Optional[Dict[str, Any]] = None
    committed: Optional[Dict[str, Any]] = None
    rows = []
    first_trigger = None
    now_wall = 1000.0
    for item in summary.get("state_trace", []):
        if item.get("state") != "FOLLOW_CORRIDOR":
            continue
        for phase in ("before", "after"):
            doorway = item.get(f"doorway_{phase}")
            if not isinstance(doorway, dict):
                continue
            room_zone = bool(item.get(f"room_zone_reached_{phase}"))
            metrics = item.get(f"corridor_anchor_metrics_{phase}")
            progress = metrics.get("anchor_progress_m") if isinstance(metrics, dict) else None
            candidate = select_room_side_gap_candidate(doorway, parameters)
            if room_zone and finite(progress):
                current, status = update_room_side_gap_observation_from_progress(parameters, current, candidate, float(progress))
                committed = update_committed_room_side_gap(parameters, committed, status, float(progress), now_wall)
                committed_status = committed_room_side_gap_trigger(parameters, committed, float(progress), now_wall)
                trigger = committed_status if committed_status.get("trigger_ready") else status
            else:
                status = {"available": False, "reason": "room_zone_or_progress_unavailable", "trigger_ready": False}
                committed_status = {"trigger_ready": False, "reason": "not_evaluated"}
                trigger = status
            selected = doorway.get("doorway_geometry_profile") if isinstance(doorway.get("doorway_geometry_profile"), dict) else {}
            fully_bounded = False
            for side in ("left", "right"):
                opening = (selected.get(side) or {}).get("selected_opening") if isinstance(selected.get(side), dict) else None
                fully_bounded = fully_bounded or bool(isinstance(opening, dict) and opening.get("opening_center_estimated"))
            decision = select_follow_corridor_opening_action(
                doorway_control_enabled=bool(parameters.get("enable_doorway_control", True)),
                fully_bounded_doorway_ready=bool(doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY" and fully_bounded),
                room_side_gap_enabled=bool(parameters.get("enable_room_side_gap_trigger", True)),
                room_side_gap_trigger_ready=bool(trigger.get("trigger_ready")),
                forced_room_entry_enabled=bool(parameters.get("enable_forced_room_entry_mvp", False)),
            )
            row = {
                "iteration": item.get("iteration"),
                "phase": phase,
                "room_zone_reached": room_zone,
                "anchor_progress_m": progress,
                "candidate": candidate,
                "observation": status,
                "committed_trigger": committed_status,
                "transition_decision": decision,
            }
            rows.append(row)
            if first_trigger is None and decision.get("action") == "ROOM_SIDE_TURN":
                first_trigger = {
                    "iteration": item.get("iteration"),
                    "phase": phase,
                    "candidate_side": trigger.get("door_side") or trigger.get("side"),
                    "candidate_center_x_m": trigger.get("center_x_base_m"),
                    "candidate_width_m": trigger.get("opening_width_m") or trigger.get("width_m"),
                    "trigger_source": trigger.get("trigger_source") or trigger.get("reason"),
                    "forced_mode_suppresses_gap": decision.get("forced_mode_suppresses_gap"),
                }
            now_wall += 1.0
    expected = bool(
        first_trigger
        and first_trigger.get("iteration") == 5
        and first_trigger.get("phase") == "after"
        and first_trigger.get("candidate_side") == "right"
    )
    return {
        "schema_version": 1,
        "archive": str(archive),
        "offline_only": True,
        "ros_initialized": False,
        "ros_commands_published": False,
        "forbidden_truth_read": False,
        "actual_run_parameters": parameters,
        "original_forced_mode_enabled": bool(parameters.get("enable_forced_room_entry_mvp")),
        "first_trigger": first_trigger,
        "expected_iteration_5_after_right_trigger": expected,
        "forced_guard_no_longer_suppresses": bool(first_trigger and first_trigger.get("forced_mode_suppresses_gap") is False),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive
    if archive is None:
        choices = sorted((ROOT / "debug" / "state_machine_navigation" / "run_archives").glob("run_0039_*"))
        archive = choices[-1]
    result = replay(archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "first_trigger": result["first_trigger"], "expected": result["expected_iteration_5_after_right_trigger"]}, sort_keys=True))
    return 0 if result["expected_iteration_5_after_right_trigger"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
