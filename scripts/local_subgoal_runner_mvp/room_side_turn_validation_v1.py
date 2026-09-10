#!/usr/bin/env python3
"""Pure decision and feedback-control core for room-side-turn validation V1.

The module deliberately has no ROS imports.  Online adapters live in
``navigation_state_machine.py`` while this file remains replay/test friendly.
It never creates an entry/commit target and never selects ENTER_ROOM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _param(parameters: Dict[str, Any], name: str, default: Any) -> Any:
    return parameters.get(name, default)


def select_follow_corridor_opening_action(
    *,
    doorway_control_enabled: bool,
    fully_bounded_doorway_ready: bool,
    room_side_gap_enabled: bool,
    room_side_gap_trigger_ready: bool,
    forced_room_entry_enabled: bool,
) -> Dict[str, Any]:
    """Apply the explicit V1 priority without letting forced mode suppress gaps."""
    common = {
        "forced_room_entry_enabled": bool(forced_room_entry_enabled),
        "forced_mode_suppresses_gap": False,
        "fully_bounded_doorway_ready": bool(fully_bounded_doorway_ready),
        "room_side_gap_trigger_ready": bool(room_side_gap_trigger_ready),
    }
    if doorway_control_enabled and fully_bounded_doorway_ready:
        return {
            **common,
            "action": "DOORWAY_VERIFY",
            "reason": "fully_bounded_doorway_ready_has_priority",
        }
    if room_side_gap_enabled and room_side_gap_trigger_ready:
        return {
            **common,
            "action": "ROOM_SIDE_TURN",
            "reason": (
                "room_side_gap_ready_forced_mode_not_suppressing"
                if forced_room_entry_enabled
                else "room_side_gap_ready"
            ),
        }
    if not room_side_gap_enabled:
        reason = "room_side_gap_disabled"
    elif not room_side_gap_trigger_ready:
        reason = "room_side_gap_unavailable_or_not_stable"
    else:
        reason = "no_actionable_opening"
    return {**common, "action": "FOLLOW_CORRIDOR", "reason": reason}


def select_room_side_gap_candidate(doorway: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Pure form of the existing profile candidate selector, with rejection evidence."""
    profile_root = doorway.get("doorway_geometry_profile") if isinstance(doorway.get("doorway_geometry_profile"), dict) else {}
    candidates = []
    evaluations = []
    for side in ("left", "right"):
        side_profile = profile_root.get(side)
        if not isinstance(side_profile, dict):
            continue
        side_candidate = doorway.get(f"{side}_candidate") if isinstance(doorway.get(f"{side}_candidate"), dict) else {}
        geometry_support = bool(side_candidate.get("geometry_pass") or side_candidate.get("soft_geometry_pass"))
        segments = side_profile.get("open_segments")
        if not isinstance(segments, list):
            selected = side_profile.get("selected_opening")
            segments = [selected] if isinstance(selected, dict) else []
        for segment_index, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
            center_x = seg.get("center_x_m")
            width = seg.get("width_m")
            if not (finite_number(center_x) and finite_number(width)):
                evaluations.append({"side": side, "segment_index": segment_index, "first_reject_condition": "nonfinite_center_or_width"})
                continue
            center_x = float(center_x)
            width = float(width)
            wall_support_count = int(bool(seg.get("before_wall_or_unknown"))) + int(bool(seg.get("after_wall_or_unknown")))
            checks = [
                ("center_out_of_range", float(_param(parameters, "room_side_gap_min_center_x_m", 0.35)) <= center_x <= float(_param(parameters, "room_side_gap_max_center_x_m", 1.35))),
                ("width_out_of_range", float(_param(parameters, "room_side_gap_min_width_m", 0.75)) <= width <= float(_param(parameters, "room_side_gap_max_width_m", 2.0))),
                ("profile_not_plausible", not bool(_param(parameters, "room_side_gap_require_profile_plausible", True)) or bool(seg.get("width_plausible"))),
                ("profile_door_signal_missing", not bool(_param(parameters, "room_side_gap_require_profile_door_signal", True)) or bool(seg.get("partial_opening_observed")) or bool(seg.get("opening_center_estimated"))),
                ("wall_or_geometry_support_missing", wall_support_count >= int(_param(parameters, "room_side_gap_min_wall_support_count", 1)) or geometry_support),
            ]
            first_reject = next((name for name, passed in checks if not passed), None)
            evaluation = {
                "side": side,
                "segment_index": segment_index,
                "center_x_m": center_x,
                "width_m": width,
                "start_x_m": seg.get("start_x_m"),
                "end_x_m": seg.get("end_x_m"),
                "wall_support_count": wall_support_count,
                "geometry_support": geometry_support,
                "first_reject_condition": first_reject,
                "selected_opening": seg,
            }
            evaluations.append(evaluation)
            if first_reject is not None:
                continue
            score = (
                3.0 * wall_support_count
                + (2.0 if geometry_support else 0.0)
                - abs(center_x - float(_param(parameters, "room_side_gap_preferred_center_x_m", 0.85)))
                - 0.25 * abs(width - float(_param(parameters, "room_side_gap_preferred_width_m", 1.0)))
            )
            candidates.append(
                {
                    "available": True,
                    "door_side": side,
                    "center_x_base_m": center_x,
                    "opening_width_m": width,
                    "start_x_m": float(seg["start_x_m"]) if finite_number(seg.get("start_x_m")) else None,
                    "end_x_m": float(seg["end_x_m"]) if finite_number(seg.get("end_x_m")) else None,
                    "width_plausible": bool(seg.get("width_plausible")),
                    "wall_support_count": wall_support_count,
                    "geometry_support": geometry_support,
                    "selected_opening": seg,
                    "score": score,
                    "source": "room_side_gap_profile",
                    "doorway_final_decision": doorway.get("final_decision"),
                    "doorway_decision_mode": doorway.get("decision_mode"),
                }
            )
    if not candidates:
        return {"available": False, "reason": "no_room_side_gap_candidate", "evaluations": evaluations}
    preferred_side = doorway.get("door_side")
    if doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY" and preferred_side in {"left", "right"}:
        preferred = [item for item in candidates if item.get("door_side") == preferred_side]
        if preferred:
            candidates = preferred
        else:
            return {
                "available": False,
                "reason": "room_side_gap_candidate_side_mismatch",
                "preferred_door_side": preferred_side,
                "candidate_sides": sorted({str(item.get("door_side")) for item in candidates}),
                "evaluations": evaluations,
            }
    candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    best = dict(candidates[0])
    best.update({"reason": "room_side_gap_candidate", "evaluations": evaluations})
    return best


def update_room_side_gap_observation_from_progress(
    parameters: Dict[str, Any],
    current: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
    current_progress: Optional[float],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not candidate.get("available"):
        status = {"available": False, "reason": candidate.get("reason", "room_side_gap_unavailable")}
        if isinstance(current, dict) and finite_number(current.get("alignment_target_anchor_progress_m")):
            preferred_side = candidate.get("preferred_door_side")
            if candidate.get("reason") == "room_side_gap_candidate_side_mismatch" and preferred_side in {"left", "right"} and current.get("door_side") != preferred_side:
                status.update({"door_side": current.get("door_side"), "preferred_door_side": preferred_side, "alignment_cleared": True, "alignment_clear_reason": "room_side_gap_preferred_side_changed"})
                return None, status
            target = float(current["alignment_target_anchor_progress_m"])
            error = target - float(current_progress) if finite_number(current_progress) else None
            missed = int(current.get("missed_observation_count") or 0) + 1
            current["missed_observation_count"] = missed
            status.update({"door_side": current.get("door_side"), "center_x_base_m": current.get("center_x_base_m"), "opening_width_m": current.get("opening_width_m"), "selected_opening": current.get("selected_opening"), "stable_observation_count": int(current.get("stable_observation_count") or 0), "alignment_locked": True, "anchor_progress_m": current_progress, "alignment_target_anchor_progress_m": target, "alignment_longitudinal_error_m": error, "missed_observation_count": missed, "trigger_ready": False, "trigger_suppressed_reason": "room_side_gap_not_visible_current_frame"})
            if missed > int(_param(parameters, "room_side_gap_max_missed_observations", 0)):
                status.update({"alignment_cleared": True, "alignment_clear_reason": "room_side_gap_lost"})
                return None, status
            return current, status
        return current, status
    previous_center = current.get("center_x_base_m") if isinstance(current, dict) else None
    previous_width = current.get("opening_width_m") if isinstance(current, dict) else None
    same_side = isinstance(current, dict) and current.get("door_side") == candidate.get("door_side")
    already_locked = bool(same_side and finite_number(current.get("alignment_target_anchor_progress_m")))
    center_delta = abs(float(candidate["center_x_base_m"]) - float(previous_center)) if finite_number(previous_center) else None
    width_delta = abs(float(candidate["opening_width_m"]) - float(previous_width)) if finite_number(previous_width) else None
    stable = bool(same_side and center_delta is not None and width_delta is not None and center_delta <= float(_param(parameters, "room_side_gap_center_stability_tolerance_m", 0.65)) and width_delta <= float(_param(parameters, "room_side_gap_width_stability_tolerance_m", 1.2)))
    stable_count = int(current.get("stable_observation_count") or 0) + 1 if stable and isinstance(current, dict) else 1
    observation = dict(candidate)
    alignment_target = float(current["alignment_target_anchor_progress_m"]) if already_locked else (float(current_progress) + float(candidate["center_x_base_m"]) if stable_count >= int(_param(parameters, "room_side_gap_required_stable_observations", 2)) and finite_number(current_progress) else None)
    error = float(alignment_target) - float(current_progress) if finite_number(alignment_target) and finite_number(current_progress) else None
    observation.update({"stable_observation_count": stable_count, "center_delta_m": center_delta, "width_delta_m": width_delta, "missed_observation_count": 0, "anchor_progress_m": current_progress, "alignment_locked": finite_number(alignment_target), "alignment_target_anchor_progress_m": alignment_target, "alignment_longitudinal_error_m": error, "trigger_ready": bool(finite_number(error) and abs(float(error)) <= float(_param(parameters, "room_side_gap_turn_alignment_tolerance_m", 0.25)))})
    return observation, observation


def commit_room_side_gap_if_ready(parameters: Dict[str, Any], status: Dict[str, Any], now_wall: float) -> Optional[Dict[str, Any]]:
    if not status.get("available") or int(status.get("stable_observation_count") or 0) < int(_param(parameters, "room_side_gap_required_stable_observations", 2)):
        return None
    selected = status.get("selected_opening") if isinstance(status.get("selected_opening"), dict) else {}
    start_x = status.get("start_x_m", selected.get("start_x_m")); end_x = status.get("end_x_m", selected.get("end_x_m")); width = status.get("opening_width_m", selected.get("width_m")); target = status.get("alignment_target_anchor_progress_m"); seen = status.get("anchor_progress_m")
    if not all(finite_number(x) for x in (start_x, end_x, width, target, seen)):
        return None
    if not bool(status.get("width_plausible", selected.get("width_plausible"))) or float(start_x) > 0.6 or float(end_x) < 1.0:
        return None
    if not (bool(status.get("geometry_support")) or status.get("doorway_final_decision") == "DOORWAY_CANDIDATE_READY"):
        return None
    side = status.get("door_side")
    if side not in {"left", "right"}:
        return None
    return {"source": "committed_room_side_gap", "side": side, "door_side": side, "target_progress_m": float(target), "alignment_target_anchor_progress_m": float(target), "last_seen_progress_m": float(seen), "center_x_base_m": status.get("center_x_base_m"), "start_x_m": float(start_x), "end_x_m": float(end_x), "width_m": float(width), "opening_width_m": float(width), "stable_count": int(status.get("stable_observation_count") or 0), "created_wall_time_sec": float(now_wall), "selected_opening": selected}


def update_committed_room_side_gap(parameters: Dict[str, Any], committed: Optional[Dict[str, Any]], status: Dict[str, Any], current_progress: Optional[float], now_wall: float) -> Optional[Dict[str, Any]]:
    if isinstance(committed, dict) and finite_number(current_progress) and finite_number(committed.get("target_progress_m")) and float(current_progress) - float(committed["target_progress_m"]) > 0.45:
        committed = None
    if isinstance(committed, dict) and status.get("available") and status.get("door_side") in {"left", "right"} and committed.get("side") in {"left", "right"} and status.get("door_side") != committed.get("side") and int(status.get("stable_observation_count") or 0) >= 2:
        committed = None
    new_commit = commit_room_side_gap_if_ready(parameters, status, now_wall)
    if new_commit is not None and not isinstance(committed, dict):
        committed = new_commit
    elif isinstance(committed, dict) and status.get("available") and status.get("door_side") == committed.get("side"):
        if finite_number(status.get("anchor_progress_m")):
            committed["last_seen_progress_m"] = float(status["anchor_progress_m"])
        committed["stable_count"] = max(int(committed.get("stable_count") or 0), int(status.get("stable_observation_count") or 0))
    return committed


def committed_room_side_gap_trigger(parameters: Dict[str, Any], committed: Optional[Dict[str, Any]], current_progress: Optional[float], now_wall: float) -> Dict[str, Any]:
    if not isinstance(committed, dict):
        return {"trigger_ready": False, "reason": "no_committed_room_side_gap"}
    if not (finite_number(current_progress) and finite_number(committed.get("target_progress_m"))):
        return {"trigger_ready": False, "reason": "committed_room_side_gap_progress_unavailable"}
    error = float(committed["target_progress_m"]) - float(current_progress)
    overshoot = -error
    status = dict(committed)
    status.update({"trigger_ready": False, "trigger_source": "committed_room_side_gap", "anchor_progress_m": float(current_progress), "alignment_longitudinal_error_m": error, "overshoot_m": overshoot, "age_wall_time_sec": float(now_wall) - float(committed.get("created_wall_time_sec") or now_wall)})
    if overshoot > 0.45:
        status["reason"] = "committed_room_side_gap_overshot"
    elif abs(error) <= float(_param(parameters, "room_side_gap_turn_alignment_tolerance_m", 0.25)):
        status.update({"trigger_ready": True, "trigger_source": "committed_room_side_gap_alignment_reached", "reason": "committed_room_side_gap_alignment_reached"})
    else:
        status["reason"] = "committed_room_side_gap_alignment_pending"
    return status


def classify_stream_age(age_wall_sec: Optional[float], fresh_sec: float, lost_sec: float) -> str:
    if age_wall_sec is None or not finite_number(age_wall_sec) or float(age_wall_sec) >= float(lost_sec):
        return "LOST"
    if float(age_wall_sec) > float(fresh_sec):
        return "STALE"
    return "FRESH"


@dataclass
class FeedbackYawTurnCore:
    side: str
    target_yaw_rad: float
    min_yaw_rad: float
    angular_z: float
    watchdog_sec: float
    no_response_sample_limit: int
    yaw_response_epsilon_rad: float
    previous_yaw: Optional[float] = None
    accumulated_yaw_rad: float = 0.0
    no_response_samples: int = 0
    stopped: bool = False
    stop_reason: Optional[str] = None

    @property
    def sign(self) -> float:
        return 1.0 if self.side == "left" else -1.0

    def update(self, *, odom_state: str, yaw_rad: Optional[float], wall_elapsed_sec: float) -> Dict[str, Any]:
        if self.stopped:
            return self._result(0.0, 0.0)
        if wall_elapsed_sec >= self.watchdog_sec:
            self.stopped = True; self.stop_reason = "wall_watchdog"
            return self._result(0.0, 0.0)
        if odom_state == "LOST":
            self.stopped = True; self.stop_reason = "odom_lost"
            return self._result(0.0, 0.0)
        if odom_state != "FRESH" or not finite_number(yaw_rad):
            return self._result(0.0, 0.0, waiting_reason="odom_stale_wait_zero")
        yaw = float(yaw_rad)
        if self.previous_yaw is None:
            self.previous_yaw = yaw
            return self._result(0.0, self.sign * abs(self.angular_z), waiting_reason="initial_yaw_accepted")
        increment = normalize_angle(yaw - self.previous_yaw)
        self.previous_yaw = yaw
        self.accumulated_yaw_rad += increment
        directed_increment = self.sign * increment
        if directed_increment < self.yaw_response_epsilon_rad:
            self.no_response_samples += 1
        else:
            self.no_response_samples = 0
        directed_yaw = self.sign * self.accumulated_yaw_rad
        if directed_yaw >= self.target_yaw_rad:
            self.stopped = True; self.stop_reason = "target_yaw_reached"
            return self._result(0.0, 0.0)
        if self.no_response_samples >= self.no_response_sample_limit:
            self.stopped = True; self.stop_reason = "yaw_no_response"
            return self._result(0.0, 0.0)
        return self._result(0.0, self.sign * abs(self.angular_z))

    def _result(self, linear_x: float, angular_z: float, waiting_reason: Optional[str] = None) -> Dict[str, Any]:
        directed = self.sign * self.accumulated_yaw_rad
        return {"linear_x": float(linear_x), "angular_z": float(angular_z), "accumulated_yaw_rad": float(self.accumulated_yaw_rad), "directed_yaw_rad": float(directed), "target_reached": bool(directed >= self.target_yaw_rad), "minimum_yaw_sufficient": bool(directed >= self.min_yaw_rad), "no_response_samples": int(self.no_response_samples), "stopped": bool(self.stopped), "stop_reason": self.stop_reason, "waiting_reason": waiting_reason}


def post_turn_transition(*, turn_stop_reason: str, minimum_yaw_sufficient: bool, target_reached: bool, fresh_grid_count: int, required_fresh_grids: int, grid_safe_for_navigation: bool, fresh_side_observation: bool) -> Dict[str, Any]:
    """Choose only DOORWAY_VERIFY, ROOM_SIDE_TURN hold, FOLLOW, or FAILED."""
    if turn_stop_reason in {"odom_lost", "yaw_no_response"}:
        return {"next_state": "FAILED", "reason": f"room_side_turn_failed_hold:{turn_stop_reason}", "entry_target_allowed": False}
    if not minimum_yaw_sufficient or not target_reached:
        return {"next_state": "FOLLOW_CORRIDOR", "reason": f"room_side_turn_insufficient:{turn_stop_reason}", "entry_target_allowed": False}
    if fresh_grid_count < required_fresh_grids:
        return {"next_state": "FOLLOW_CORRIDOR", "reason": "post_turn_fresh_grid_requirement_not_met", "entry_target_allowed": False}
    if not grid_safe_for_navigation:
        return {"next_state": "FOLLOW_CORRIDOR", "reason": "post_turn_grid_not_safe_for_navigation", "entry_target_allowed": False}
    if not fresh_side_observation:
        return {"next_state": "FOLLOW_CORRIDOR", "reason": "post_turn_fresh_side_observation_unavailable", "entry_target_allowed": False}
    return {"next_state": "DOORWAY_VERIFY", "reason": "room_side_turn_target_reached_fresh_observation_ready", "entry_target_allowed": False}
