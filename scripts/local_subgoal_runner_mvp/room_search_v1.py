#!/usr/bin/env python3
"""ROOM_SEARCH V2-A: a small NBV-lite decision core.

This module has no ROS publisher, planner, or collision checker.  The caller
supplies only cells that the existing Grid/Block-A*/DWA chain has admitted.
It keeps a coarse *observation* memory and actual-pose breadcrumbs; neither is
a navigation map or a second costmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


COARSE_OBSERVATION_RESOLUTION_M = 0.5
PREFERRED_MIN_RADIUS_M = 0.8
PREFERRED_MAX_RADIUS_M = 1.2
LOCAL_MAX_RADIUS_M = 1.5
MAX_CANDIDATES = 5
RAW_SECTOR_REPRESENTATIVE_CAP = 12
ANTI_INFINITE_DECISION_GUARD = 24
DOOR_NEAR_SOFT_DEPTH_M = 1.70
DOOR_NEAR_UTILITY_FACTOR = 0.70
# Engineering defaults, not competition rules or an empirically optimal sweep.
LOW_GAIN_RATIO = 0.20
LOW_GAIN_PATIENCE = 2
REPOSITION_FAILURE_DECISIONS = {
    "BLOCK_ASTAR_DWA_BLOCKED_NO_PATH",
    "BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD",
    "BLOCK_ASTAR_DWA_BLOCKED_UNSTABLE_PATH",
    "BLOCK_ASTAR_DWA_BLOCKED_BY_TARGET",
    "BLOCK_ASTAR_DWA_MAX_STEPS",
}


def _unit(vector: Sequence[float]) -> Tuple[float, float]:
    norm = math.hypot(float(vector[0]), float(vector[1]))
    if norm <= 0.0:
        raise ValueError("room_search_vector_zero_length")
    return float(vector[0]) / norm, float(vector[1]) / norm


def _coarse(point: Sequence[float]) -> Tuple[int, int]:
    return (
        int(math.floor(float(point[0]) / COARSE_OBSERVATION_RESOLUTION_M)),
        int(math.floor(float(point[1]) / COARSE_OBSERVATION_RESOLUTION_M)),
    )


MISSION_ACTION_CELL_FRAME = "portal_room_local"
MISSION_ACTION_CELL_INDEXING = "floor(room_x/resolution),floor(room_y/resolution)"
MISSION_ACTION_TRANSLATE_AND_OBSERVE = "TRANSLATE_AND_OBSERVE"
MISSION_ACTION_INTENT_GENERIC = "GENERIC_PREDICTED_NEW_REGION"
MISSION_ACTION_INTENT_OCCLUSION = "OCCLUSION_REVEAL_REGION"
DANGER_REOBSERVE_MISSION_INTENT = "DANGER_REOBSERVE"
DANGER_REOBSERVE_EXECUTION_EXISTING_VIEWPOINT = "EXISTING_CANDIDATE_VIEWPOINT"
DANGER_REOBSERVE_TERMINAL_CONFIRMED = "CONFIRMED"
DANGER_REOBSERVE_TERMINAL_UNRESOLVED = "REOBSERVED_UNRESOLVED"
DANGER_REOBSERVE_TERMINAL_NOT_OBTAINED = "OBSERVATION_NOT_OBTAINED"
DANGER_REOBSERVE_TERMINAL_UNKNOWN = "UNKNOWN"
MISSION_ACTION_FIELD_ROLES = {
    "action_id": "ACTION_IDENTITY",
    "candidate_id": "ACTION_IDENTITY",
    "action_type": "ACTION_IDENTITY",
    "target_odom_xy": "ACTION_IDENTITY",
    "observation_intent_type": "ACTION_IDENTITY",
    "intended_observation_cell_ids": "COMPLETION_AUTHORITY_INPUT",
    "cell_frame": "COMPLETION_AUTHORITY_INPUT",
    "cell_resolution_m": "COMPLETION_AUTHORITY_INPUT",
    "cell_indexing": "COMPLETION_AUTHORITY_INPUT",
    "predicted_view_heading_base_rad": "EXECUTION_INPUT",
    "aim_yaw_odom_rad": "EXECUTION_INPUT",
    "predicted_new_count": "TELEMETRY_ONLY",
    "predicted_visible_count": "TELEMETRY_ONLY",
    "predicted_occlusion_reveal_count": "TELEMETRY_ONLY",
    "nbv_value": "TELEMETRY_ONLY",
    "cheap_rank": "TELEMETRY_ONLY",
    "sector": "TELEMETRY_ONLY",
}


def canonical_observation_cell_ids(points: Iterable[Sequence[float]]) -> Tuple[Tuple[int, int], ...]:
    """Return the one canonical ROOM_SEARCH observation-cell representation.

    This is deliberately only a portal-room-local coarse representation.  It
    neither queries a Grid nor evaluates visibility; callers must supply points
    from the existing authoritative visibility helper.
    """
    cells = set()
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            cells.add(_coarse(point))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(cells))


def _canonical_cell_id_pairs(cells: Iterable[Sequence[int]]) -> Tuple[Tuple[int, int], ...]:
    canonical = set()
    for cell in cells:
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            continue
        x, y = cell
        if isinstance(x, bool) or isinstance(y, bool):
            continue
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        canonical.add((int(x), int(y)))
    return tuple(sorted(canonical))


def _finite_optional(value: object) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _normalize_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


@dataclass(frozen=True)
class MissionActionSpec:
    """Immutable ROOM_SEARCH action identity with no motion authority."""

    action_id: str
    candidate_id: str
    action_type: str
    target_odom_xy: Tuple[float, float]
    observation_intent_type: str
    intended_observation_cell_ids: Tuple[Tuple[int, int], ...]
    cell_frame: str
    cell_resolution_m: float
    cell_indexing: str
    predicted_view_heading_base_rad: Optional[float]
    aim_yaw_odom_rad: Optional[float]
    predicted_new_count: Optional[int]
    predicted_visible_count: Optional[int]
    predicted_occlusion_reveal_count: Optional[int]
    nbv_value: Optional[float]
    cheap_rank: Optional[int]
    sector: Optional[int]

    @property
    def intent_valid(self) -> bool:
        return (
            self.action_type == MISSION_ACTION_TRANSLATE_AND_OBSERVE
            and self.observation_intent_type in {MISSION_ACTION_INTENT_GENERIC, MISSION_ACTION_INTENT_OCCLUSION}
            and bool(self.intended_observation_cell_ids)
            and self.cell_frame == MISSION_ACTION_CELL_FRAME
            and self.cell_resolution_m == COARSE_OBSERVATION_RESOLUTION_M
            and self.cell_indexing == MISSION_ACTION_CELL_INDEXING
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "action_id": self.action_id,
            "candidate_id": self.candidate_id,
            "action_type": self.action_type,
            "target_odom_xy": list(self.target_odom_xy),
            "observation_intent_type": self.observation_intent_type,
            "intended_observation_cell_ids": [list(cell) for cell in self.intended_observation_cell_ids],
            "cell_frame": self.cell_frame,
            "cell_resolution_m": self.cell_resolution_m,
            "cell_indexing": self.cell_indexing,
            "predicted_view_heading_base_rad": self.predicted_view_heading_base_rad,
            "aim_yaw_odom_rad": self.aim_yaw_odom_rad,
            "predicted_new_count": self.predicted_new_count,
            "predicted_visible_count": self.predicted_visible_count,
            "predicted_occlusion_reveal_count": self.predicted_occlusion_reveal_count,
            "nbv_value": self.nbv_value,
            "cheap_rank": self.cheap_rank,
            "sector": self.sector,
            "intent_valid": self.intent_valid,
            "field_roles": dict(MISSION_ACTION_FIELD_ROLES),
        }


@dataclass(frozen=True)
class DangerReobserveMissionActionSpec:
    """Frozen, result-agnostic request to observe one existing tentative H1.

    This deliberately does not reuse coverage-cell intent semantics and has no
    ranking, preflight, command, detector, or world-coordinate authority.
    ``hypothesis_id`` is an in-process association-episode identity only.
    """

    action_id: str
    run_id: str
    candidate_id: str
    mission_intent: str
    hypothesis_id: str
    frozen_last_observed_stamp_sec: float
    frozen_position_team_livox_odom_xyz: Tuple[float, float, float]
    target_odom_xy: Tuple[float, float]
    aim_yaw_odom_rad: Optional[float]
    execution_mode: str
    viewpoint_abs_bearing_rad: float
    viewpoint_distance_m: float
    dispatch_time_sec: float

    @property
    def valid(self) -> bool:
        return (
            self.mission_intent == DANGER_REOBSERVE_MISSION_INTENT
            and bool(self.run_id) and bool(self.candidate_id) and bool(self.hypothesis_id)
            and _finite_optional(self.frozen_last_observed_stamp_sec) is not None
            and len(self.frozen_position_team_livox_odom_xyz) == 3
            and all(_finite_optional(value) is not None for value in self.frozen_position_team_livox_odom_xyz)
            and len(self.target_odom_xy) == 2 and all(_finite_optional(value) is not None for value in self.target_odom_xy)
            and self.execution_mode == DANGER_REOBSERVE_EXECUTION_EXISTING_VIEWPOINT
            and _finite_optional(self.viewpoint_abs_bearing_rad) is not None
            and _finite_optional(self.viewpoint_distance_m) is not None
            and _finite_optional(self.dispatch_time_sec) is not None
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "action_id": self.action_id,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "mission_intent": self.mission_intent,
            "hypothesis_id": self.hypothesis_id,
            "frozen_last_observed_stamp_sec": self.frozen_last_observed_stamp_sec,
            "frozen_position_team_livox_odom_xyz": list(self.frozen_position_team_livox_odom_xyz),
            "target_odom_xy": list(self.target_odom_xy),
            "aim_yaw_odom_rad": self.aim_yaw_odom_rad,
            "execution_mode": self.execution_mode,
            "viewpoint_abs_bearing_rad": self.viewpoint_abs_bearing_rad,
            "viewpoint_distance_m": self.viewpoint_distance_m,
            "dispatch_time_sec": self.dispatch_time_sec,
            "valid": self.valid,
            "authority": "FORMAL_MISSION_SEMANTICS_ONLY",
        }


def freeze_danger_reobserve_mission_action_spec(
    candidate: Dict[str, object], *, action_id: str, run_id: str, dispatch_time_sec: float,
) -> Optional[DangerReobserveMissionActionSpec]:
    """Freeze one already-produced H1 opportunity; never create new evidence."""
    if str(candidate.get("target_priority_class") or "") != "DANGER_REOBSERVE":
        return None
    opportunities = candidate.get("danger_reobserve_opportunities")
    if not isinstance(opportunities, (list, tuple)) or not opportunities:
        return None
    row = opportunities[0] if isinstance(opportunities[0], dict) else {}
    hypothesis_id = str(row.get("hypothesis_id") or "")
    xyz = row.get("position_xyz_m")
    target = candidate.get("target_xy_team_livox_odom")
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3 or not isinstance(target, (list, tuple)) or len(target) != 2:
        return None
    numbers = [_finite_optional(value) for value in (*xyz, *target, row.get("last_observed_stamp_sec"), row.get("abs_bearing_rad"), row.get("distance_m"), dispatch_time_sec)]
    if not hypothesis_id or any(value is None for value in numbers):
        return None
    heading = _finite_optional(candidate.get("heading_change_rad"))
    spec = DangerReobserveMissionActionSpec(
        action_id=str(action_id), run_id=str(run_id),
        candidate_id=str(candidate.get("_room_search_audit_candidate_id") or ""),
        mission_intent=DANGER_REOBSERVE_MISSION_INTENT, hypothesis_id=hypothesis_id,
        frozen_last_observed_stamp_sec=float(numbers[5]),
        frozen_position_team_livox_odom_xyz=(float(numbers[0]), float(numbers[1]), float(numbers[2])),
        target_odom_xy=(float(numbers[3]), float(numbers[4])), aim_yaw_odom_rad=heading,
        execution_mode=DANGER_REOBSERVE_EXECUTION_EXISTING_VIEWPOINT,
        viewpoint_abs_bearing_rad=float(numbers[6]), viewpoint_distance_m=float(numbers[7]),
        dispatch_time_sec=float(numbers[8]),
    )
    return spec if spec.valid else None


def evaluate_danger_reobserve_terminal(
    spec: DangerReobserveMissionActionSpec,
    *, execution_viewpoint_obtained: bool, terminal_time_sec: object,
    confirmed_tracks: Optional[Iterable[Dict[str, object]]],
    tentative_hypotheses: Optional[Iterable[Dict[str, object]]],
) -> Dict[str, object]:
    """Classify terminal sidecar evidence without mutating perception or SEEN."""
    terminal = _finite_optional(terminal_time_sec)
    base = {
        "action_id": spec.action_id, "hypothesis_id": spec.hypothesis_id,
        "dispatch_time_sec": spec.dispatch_time_sec, "terminal_time_sec": terminal,
        "frozen_last_observed_stamp_sec": spec.frozen_last_observed_stamp_sec,
        "danger_observation_time_sec": None,
    }
    if not spec.valid or terminal is None or terminal < spec.dispatch_time_sec:
        return {**base, "terminal_outcome": DANGER_REOBSERVE_TERMINAL_UNKNOWN, "reason": "ACTION_OR_TERMINAL_TIME_INVALID"}
    if not execution_viewpoint_obtained:
        return {**base, "terminal_outcome": DANGER_REOBSERVE_TERMINAL_NOT_OBTAINED, "reason": "EXISTING_EXECUTION_VIEWPOINT_NOT_OBTAINED"}
    if confirmed_tracks is None or tentative_hypotheses is None:
        return {**base, "terminal_outcome": DANGER_REOBSERVE_TERMINAL_UNKNOWN, "reason": "TERMINAL_DANGER_SNAPSHOT_MISSING"}

    def fresh(row: Dict[str, object], identity_key: str) -> Optional[float]:
        if str(row.get(identity_key) or "") != spec.hypothesis_id:
            return None
        stamp = _finite_optional(row.get("last_observed_stamp_sec"))
        if stamp is None or stamp <= spec.frozen_last_observed_stamp_sec:
            return None
        if stamp < spec.dispatch_time_sec or stamp > terminal:
            return None
        return stamp

    for row in confirmed_tracks:
        if isinstance(row, dict):
            stamp = fresh(row, "track_id")
            if stamp is not None:
                return {**base, "terminal_outcome": DANGER_REOBSERVE_TERMINAL_CONFIRMED,
                        "reason": "SAME_HYPOTHESIS_FRESH_CONFIRMED_TRACK", "danger_observation_time_sec": stamp}
    for row in tentative_hypotheses:
        if isinstance(row, dict):
            stamp = fresh(row, "hypothesis_id")
            if stamp is not None:
                return {**base, "terminal_outcome": DANGER_REOBSERVE_TERMINAL_UNRESOLVED,
                        "reason": "SAME_HYPOTHESIS_FRESH_TENTATIVE_TRACK", "danger_observation_time_sec": stamp}
    return {**base, "terminal_outcome": DANGER_REOBSERVE_TERMINAL_UNKNOWN,
            "reason": "NO_SAME_HYPOTHESIS_FRESH_TERMINAL_EVIDENCE"}


class DangerReobserveEpisodeGuard:
    """One formal H1 action per ROOM_SEARCH run; never affects candidate ranking."""
    def __init__(self, run_id: str) -> None:
        self.run_id = str(run_id)
        self._closed = set()

    def admission_allowed(self, spec: Optional[DangerReobserveMissionActionSpec]) -> bool:
        return bool(spec is not None and spec.valid and spec.run_id == self.run_id and (self.run_id, spec.hypothesis_id) not in self._closed)

    def close(self, spec: DangerReobserveMissionActionSpec, terminal: Dict[str, object]) -> None:
        self._closed.add((self.run_id, spec.hypothesis_id))

    def suppression_reason(self, spec: Optional[DangerReobserveMissionActionSpec]) -> Optional[str]:
        if spec is not None and (self.run_id, spec.hypothesis_id) in self._closed:
            return "SAME_RUN_HYPOTHESIS_EPISODE_ALREADY_TERMINAL"
        return None


def freeze_mission_action_spec(
    candidate: Dict[str, object], decision_pose_xy_yaw: Sequence[float], action_id: str,
    seen_cell_ids: Iterable[Sequence[int]],
) -> MissionActionSpec:
    """Freeze only selected-candidate observation facts; never mutate candidate/SEEN."""
    candidate_id = str(candidate.get("_room_search_audit_candidate_id") or "")
    target = candidate.get("target_xy_team_livox_odom")
    target_xy = (0.0, 0.0)
    if isinstance(target, (list, tuple)) and len(target) == 2:
        x, y = _finite_optional(target[0]), _finite_optional(target[1])
        if x is not None and y is not None:
            target_xy = (x, y)
    intent_type, intended = candidate_observation_intent_cell_ids(candidate, seen_cell_ids)
    heading = _finite_optional(candidate.get("heading_change_rad"))
    aim = None
    if heading is not None and isinstance(decision_pose_xy_yaw, (list, tuple)) and len(decision_pose_xy_yaw) == 3:
        decision_yaw = _finite_optional(decision_pose_xy_yaw[2])
        if decision_yaw is not None:
            aim = _normalize_angle(decision_yaw + heading)

    def count(name: str) -> Optional[int]:
        value = candidate.get(name)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    return MissionActionSpec(
        action_id=str(action_id), candidate_id=candidate_id, action_type=MISSION_ACTION_TRANSLATE_AND_OBSERVE,
        target_odom_xy=target_xy, observation_intent_type=intent_type, intended_observation_cell_ids=intended,
        cell_frame=MISSION_ACTION_CELL_FRAME, cell_resolution_m=COARSE_OBSERVATION_RESOLUTION_M,
        cell_indexing=MISSION_ACTION_CELL_INDEXING, predicted_view_heading_base_rad=heading,
        aim_yaw_odom_rad=aim, predicted_new_count=count("new_observable_cells"),
        predicted_visible_count=len(canonical_observation_cell_ids(candidate.get("visible_room_points") or [])),
        predicted_occlusion_reveal_count=len(canonical_observation_cell_ids(candidate.get("occlusion_reveal_room_points") or [])),
        nbv_value=_finite_optional(candidate.get("nbv_value")),
        cheap_rank=count("cheap_rank"), sector=count("sector"),
    )


def candidate_observation_intent_cell_ids(
    candidate: Dict[str, object], seen_cell_ids: Iterable[Sequence[int]],
) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    """Expose current candidate intent facts without creating a MissionAction."""
    candidate_type = str(candidate.get("target_priority_class") or "")
    seen = set(_canonical_cell_id_pairs(seen_cell_ids))
    if candidate_type == "GENERIC_COVERAGE":
        return (
            MISSION_ACTION_INTENT_GENERIC,
            tuple(sorted(set(canonical_observation_cell_ids(candidate.get("visible_room_points") or [])) - seen)),
        )
    if candidate_type == "OCCLUSION":
        return (
            MISSION_ACTION_INTENT_OCCLUSION,
            canonical_observation_cell_ids(candidate.get("occlusion_reveal_room_points") or []),
        )
    return "UNSUPPORTED_CANDIDATE_INTENT", ()


@dataclass(frozen=True)
class MissionActionObservationResult:
    """Pure observation-completion result; it has no SEEN or control authority."""

    observation_intent_status: str
    mission_action_observation_outcome: str
    actual_visible_cell_ids: Tuple[Tuple[int, int], ...]
    intended_visible_intersection: Tuple[Tuple[int, int], ...]
    actual_new_observation_cells: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "observation_intent_status": self.observation_intent_status,
            "mission_action_observation_outcome": self.mission_action_observation_outcome,
            "actual_visible_cell_ids": [list(cell) for cell in self.actual_visible_cell_ids],
            "intended_visible_intersection": [list(cell) for cell in self.intended_visible_intersection],
            "actual_new_observation_cells": self.actual_new_observation_cells,
        }


@dataclass(frozen=True)
class FailedObservationOpportunityIdentity:
    """Mission-local observation claim plus its existing coarse viewpoint context.

    This is deliberately not a SEEN record and not a navigation obstacle.  It
    means only that this exact claim did not produce observation evidence from
    this local viewpoint neighbourhood during the current ROOM_SEARCH object.
    """

    observation_intent_type: str
    intended_observation_cell_ids: Tuple[Tuple[int, int], ...]
    target_coarse_cell: Tuple[int, int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "observation_intent_type": self.observation_intent_type,
            "intended_observation_cell_ids": [list(cell) for cell in self.intended_observation_cell_ids],
            "target_coarse_cell": list(self.target_coarse_cell),
            "cell_frame": MISSION_ACTION_CELL_FRAME,
            "cell_resolution_m": COARSE_OBSERVATION_RESOLUTION_M,
            "cell_indexing": MISSION_ACTION_CELL_INDEXING,
        }


def evaluate_mission_action_observation(
    spec: MissionActionSpec,
    *,
    terminal_evidence_complete: bool,
    actual_visible_cell_ids: Iterable[Sequence[int]],
    actual_new_observation_cells: object,
) -> MissionActionObservationResult:
    """Classify frozen intent against factual terminal visibility without side effects."""
    actual = _canonical_cell_id_pairs(actual_visible_cell_ids)
    actual_new = (
        int(actual_new_observation_cells)
        if isinstance(actual_new_observation_cells, int)
        and not isinstance(actual_new_observation_cells, bool)
        and actual_new_observation_cells >= 0
        else None
    )
    if not spec.intent_valid or not bool(terminal_evidence_complete) or actual_new is None:
        return MissionActionObservationResult(
            "OBSERVATION_INTENT_UNKNOWN", "OBSERVATION_INTENT_UNKNOWN", actual, (), actual_new,
        )
    intersection = tuple(sorted(set(actual) & set(spec.intended_observation_cell_ids)))
    if not intersection:
        return MissionActionObservationResult(
            "OBSERVATION_INTENT_UNSATISFIED", "OBSERVATION_INTENT_UNSATISFIED", actual, intersection, actual_new,
        )
    outcome = (
        "OBSERVATION_INTENT_SATISFIED_WITH_NEW_INFORMATION"
        if actual_new > 0 else "ACTION_COMPLETE_NO_NEW_INFORMATION"
    )
    return MissionActionObservationResult(
        "OBSERVATION_INTENT_SATISFIED", outcome, actual, intersection, actual_new,
    )


@dataclass(frozen=True)
class PortalAnchor:
    """Frozen Portal frame plus the actual, current pose saved at entry."""

    center_xy: Tuple[float, float]
    inward_normal: Tuple[float, float]
    tangent: Tuple[float, float]
    width_m: float
    door_return_anchor_xy_yaw: Tuple[float, float, float]
    door_return_anchor_stamp_sec: Optional[float]

    @classmethod
    def from_frozen_target(
        cls, target: Dict[str, object], actual_entry_pose: Sequence[float], entry_stamp_sec: Optional[float]
    ) -> "PortalAnchor":
        frozen = target.get("frozen_geometry") if isinstance(target, dict) else None
        if not isinstance(frozen, dict):
            raise ValueError("room_search_frozen_geometry_missing")
        center, normal = frozen.get("portal_center_odom"), frozen.get("portal_normal_odom")
        width = target.get("portal_width_m") or frozen.get("portal_width_m")
        if not (isinstance(center, (list, tuple)) and isinstance(normal, (list, tuple)) and len(actual_entry_pose) == 3):
            raise ValueError("room_search_portal_geometry_invalid")
        if not isinstance(width, (int, float)) or float(width) <= 0.0:
            raise ValueError("room_search_portal_width_invalid")
        normal_xy = _unit(normal)
        return cls(
            center_xy=(float(center[0]), float(center[1])),
            inward_normal=normal_xy,
            tangent=(-normal_xy[1], normal_xy[0]),
            width_m=float(width),
            door_return_anchor_xy_yaw=(
                float(actual_entry_pose[0]), float(actual_entry_pose[1]), float(actual_entry_pose[2]),
            ),
            door_return_anchor_stamp_sec=entry_stamp_sec,
        )

    def local_xy(self, pose_xy: Sequence[float]) -> Tuple[float, float]:
        dx, dy = float(pose_xy[0]) - self.center_xy[0], float(pose_xy[1]) - self.center_xy[1]
        return dx * self.inward_normal[0] + dy * self.inward_normal[1], dx * self.tangent[0] + dy * self.tangent[1]

    def odom_xy(self, room_xy: Sequence[float]) -> Tuple[float, float]:
        return (
            self.center_xy[0] + float(room_xy[0]) * self.inward_normal[0] + float(room_xy[1]) * self.tangent[0],
            self.center_xy[1] + float(room_xy[0]) * self.inward_normal[1] + float(room_xy[1]) * self.tangent[1],
        )


@dataclass
class ObservationMemory:
    """Coarse SEEN/UNSEEN memory; it deliberately contains no obstacles."""

    seen: Set[Tuple[int, int]] = field(default_factory=set)

    def new_count(self, visible_room_points: Iterable[Sequence[float]]) -> int:
        return len({_coarse(point) for point in visible_room_points} - self.seen)

    def update(self, visible_room_points: Iterable[Sequence[float]]) -> int:
        cells = {_coarse(point) for point in visible_room_points}
        new = cells - self.seen
        self.seen.update(cells)
        return len(new)

    def unseen(self, point: Sequence[float]) -> bool:
        return _coarse(point) not in self.seen


def occlusion_reveal_room_points(
    observation: ObservationMemory,
    candidate_visible_room_points: Iterable[Sequence[float]],
    current_los_clear: Callable[[Sequence[float]], bool],
) -> List[Tuple[float, float]]:
    """Return unseen cells visible from a candidate but occluded from now.

    Grid ray casting belongs to the caller because only it owns the current
    formal Grid.  This pure decision helper deliberately owns no map and
    cannot make an unsafe viewpoint executable.
    """
    revealed: List[Tuple[float, float]] = []
    coarse_seen: Set[Tuple[int, int]] = set()
    for point in candidate_visible_room_points:
        normalized = (float(point[0]), float(point[1]))
        cell = _coarse(normalized)
        if cell in observation.seen or cell in coarse_seen:
            continue
        if not bool(current_los_clear(normalized)):
            revealed.append(normalized)
            coarse_seen.add(cell)
    return revealed


def strategic_reposition_needed(runner_final_decision: Optional[object]) -> bool:
    """Keep reposition limited to explicit existing execution failures."""
    return str(runner_final_decision or "") in REPOSITION_FAILURE_DECISIONS


@dataclass
class RoomSearchV2:
    anchor: PortalAnchor
    observation: ObservationMemory = field(default_factory=ObservationMemory)
    breadcrumbs: List[Tuple[float, float, float]] = field(default_factory=list)
    safe_actual_poses: List[Tuple[float, float, float]] = field(default_factory=list)
    rejected_context_points: List[Tuple[float, float]] = field(default_factory=list)
    cooldown_sector: Optional[int] = None
    gain_history: List[Dict[str, object]] = field(default_factory=list)
    peak_best_nbv_value: Optional[float] = None
    low_gain_streak: int = 0
    marginal_epoch_reset_execution_ids: Set[str] = field(default_factory=set)
    failed_observation_opportunities: Dict[FailedObservationOpportunityIdentity, Dict[str, object]] = field(default_factory=dict)

    def room_xy(self, odom_xy: Sequence[float]) -> Tuple[float, float]:
        return self.anchor.local_xy(odom_xy)

    def _near_rejected(self, point: Sequence[float]) -> bool:
        return any(math.hypot(float(point[0]) - x, float(point[1]) - y) <= COARSE_OBSERVATION_RESOLUTION_M for x, y in self.rejected_context_points)

    @staticmethod
    def _failed_opportunity_identity(
        observation_intent_type: object,
        intended_cells: Iterable[Sequence[int]],
        target_xy: object,
    ) -> Optional[FailedObservationOpportunityIdentity]:
        cells = _canonical_cell_id_pairs(intended_cells)
        if (
            str(observation_intent_type) not in {MISSION_ACTION_INTENT_GENERIC, MISSION_ACTION_INTENT_OCCLUSION}
            or not cells
            or not isinstance(target_xy, (list, tuple))
            or len(target_xy) != 2
        ):
            return None
        x, y = _finite_optional(target_xy[0]), _finite_optional(target_xy[1])
        if x is None or y is None:
            return None
        return FailedObservationOpportunityIdentity(str(observation_intent_type), cells, _coarse((x, y)))

    def _failed_opportunity_suppression(
        self, identity: Optional[FailedObservationOpportunityIdentity],
    ) -> Optional[Dict[str, object]]:
        if identity is None:
            return None
        for failed_identity, provenance in self.failed_observation_opportunities.items():
            # Reuse the existing coarse-cell plus Moore-neighbourhood local
            # context convention.  This blocks tiny retargeting around one
            # failed viewpoint, while a materially different viewpoint can
            # still claim the same unseen cells.
            if (
                failed_identity.observation_intent_type == identity.observation_intent_type
                and failed_identity.intended_observation_cell_ids == identity.intended_observation_cell_ids
                and max(
                    abs(failed_identity.target_coarse_cell[0] - identity.target_coarse_cell[0]),
                    abs(failed_identity.target_coarse_cell[1] - identity.target_coarse_cell[1]),
                ) <= 1
            ):
                return dict(provenance)
        return None

    def failed_observation_opportunity_suppression_for_spec(
        self, spec: MissionActionSpec,
    ) -> Optional[Dict[str, object]]:
        """Return mission-local suppression provenance; never mutates SEEN/control."""
        if not spec.intent_valid:
            return None
        identity = self._failed_opportunity_identity(
            spec.observation_intent_type, spec.intended_observation_cell_ids, spec.target_odom_xy,
        )
        return self._failed_opportunity_suppression(identity)

    def record_failed_observation_opportunity(
        self,
        spec: MissionActionSpec,
        observation: MissionActionObservationResult,
        *,
        decision_id: object,
    ) -> Dict[str, object]:
        """Record one truthful local zero-observation failure for next-candidate eligibility.

        Unknown terminal evidence, a fulfilled intended observation, and any
        positive actual gain intentionally leave this state untouched.
        """
        actual_new = observation.actual_new_observation_cells
        if not spec.intent_valid:
            return {"recorded": False, "reason": "MISSION_ACTION_INTENT_INVALID"}
        if observation.observation_intent_status == "OBSERVATION_INTENT_UNKNOWN" or actual_new is None:
            return {"recorded": False, "reason": "TERMINAL_OBSERVATION_EVIDENCE_UNAVAILABLE"}
        if actual_new != 0:
            return {"recorded": False, "reason": "ACTUAL_OBSERVATION_PROGRESS"}
        if observation.observation_intent_status != "OBSERVATION_INTENT_UNSATISFIED":
            return {"recorded": False, "reason": "OBSERVATION_INTENT_NOT_FAILED"}
        identity = self._failed_opportunity_identity(
            spec.observation_intent_type, spec.intended_observation_cell_ids, spec.target_odom_xy,
        )
        if identity is None:
            return {"recorded": False, "reason": "FAILED_OPPORTUNITY_IDENTITY_UNAVAILABLE"}
        provenance = {
            "recorded": True,
            "reason": "OBSERVATION_OPPORTUNITY_ZERO_ACTUAL_GAIN",
            "decision_id": int(decision_id) if isinstance(decision_id, int) and not isinstance(decision_id, bool) else None,
            "candidate_id": spec.candidate_id,
            "failed_opportunity_identity": identity.to_dict(),
            "intended_observation_cell_ids": [list(cell) for cell in spec.intended_observation_cell_ids],
            "actual_visible_cell_ids": [list(cell) for cell in observation.actual_visible_cell_ids],
            "actual_new_observation_cells": actual_new,
        }
        self.failed_observation_opportunities.setdefault(identity, provenance)
        return dict(self.failed_observation_opportunities[identity])

    def _with_failed_observation_opportunity_status(self, candidate: Dict[str, object]) -> Dict[str, object]:
        row = dict(candidate)
        intent_type, intended = candidate_observation_intent_cell_ids(row, self.observation.seen)
        identity = self._failed_opportunity_identity(intent_type, intended, row.get("target_xy_team_livox_odom"))
        suppression = self._failed_opportunity_suppression(identity)
        row["failed_observation_opportunity_suppressed"] = bool(suppression is not None)
        row["failed_observation_opportunity_reason"] = (
            "OBSERVATION_OPPORTUNITY_ZERO_ACTUAL_GAIN" if suppression is not None else None
        )
        row["failed_observation_opportunity_identity"] = None if identity is None else identity.to_dict()
        return row

    def _targets_recent_trajectory_coarse_location(self, target_xy: object) -> bool:
        """Return whether a target is in the existing route's local coarse neighborhood.

        Breadcrumbs are room-local, measured terminal poses already recorded by
        the caller after each ROOM_SEARCH execution attempt.  The one-cell
        Moore neighborhood avoids a false distinction caused only by a target
        and a prior pose landing on opposite sides of the same 0.5 m coarse
        cell boundary.  It is a binary ranking feature, never an admission or
        collision rule.
        """
        if not isinstance(target_xy, (list, tuple)) or len(target_xy) < 2:
            return False
        target_cell = _coarse(target_xy)
        for breadcrumb in self.breadcrumbs:
            breadcrumb_cell = _coarse(breadcrumb)
            if max(abs(target_cell[0] - breadcrumb_cell[0]), abs(target_cell[1] - breadcrumb_cell[1])) <= 1:
                return True
        return False

    def _with_cheap_task_values(self, candidate: Dict[str, object]) -> Dict[str, object]:
        """Attach the existing cheap ROOM_SEARCH value without motion authority."""
        row = dict(candidate)
        reveal = self.observation.new_count(row.get("occlusion_reveal_room_points") or [])
        coverage = self.observation.new_count(row.get("visible_room_points") or [])
        room_x = float((row.get("room_target_xy") or [0.0])[0])
        target_priority_class = "DANGER_REOBSERVE" if bool(row.get("danger_reobserve_supported")) else ("OCCLUSION" if reveal > 0 else "GENERIC_COVERAGE")
        trajectory_revisit = self._targets_recent_trajectory_coarse_location(row.get("target_xy_team_livox_odom"))
        row.update({
            "occlusion_reveal_cells": reveal,
            "new_observable_cells": coverage,
            "target_priority_class": target_priority_class,
            "trajectory_revisit_coarse_location": trajectory_revisit,
            "generic_trajectory_revisit_preference_applied": bool(
                trajectory_revisit and target_priority_class == "GENERIC_COVERAGE"
            ),
            "cheap_geometric_distance_m": float(row.get("candidate_radius_m") or math.hypot(*(row.get("base_xy") or [0.0, 0.0]))),
            "heading_change_rad": float(row.get("heading_change_rad") or math.atan2(float((row.get("base_xy") or [0.0, 0.0])[1]), float((row.get("base_xy") or [0.0, 0.0])[0]))),
            "door_keepout_soft_factor": DOOR_NEAR_UTILITY_FACTOR if room_x < DOOR_NEAR_SOFT_DEPTH_M else 1.0,
        })
        return row

    @staticmethod
    def _within_sector_key(row: Dict[str, object]) -> Tuple[float, ...]:
        """Keep danger priority; rank normal sector representatives by coverage.

        ``occlusion_reveal_cells`` remains candidate provenance and a
        deterministic equal-coverage tie-break.  It is not allowed to replace
        a normal candidate with strictly greater remaining unobserved coverage.
        """
        radius = float(row.get("candidate_radius_m") or math.hypot(*(row.get("base_xy") or [0.0, 0.0])))
        preferred_error = 0.0 if PREFERRED_MIN_RADIUS_M <= radius <= PREFERRED_MAX_RADIUS_M else min(
            abs(radius - PREFERRED_MIN_RADIUS_M), abs(radius - PREFERRED_MAX_RADIUS_M)
        )
        return (
            0.0 if bool(row.get("danger_reobserve_supported")) else 1.0,
            -float(row.get("new_observable_cells") or 0),
            -float(row.get("occlusion_reveal_cells") or 0),
            preferred_error,
            -radius,
        )

    @staticmethod
    def _within_sector_discard_reason(loser: Dict[str, object], winner: Dict[str, object]) -> str:
        """Name the first existing representative-order difference for audit only."""
        loser_occlusion = float(loser.get("occlusion_reveal_cells") or 0)
        winner_occlusion = float(winner.get("occlusion_reveal_cells") or 0)
        if loser_occlusion < winner_occlusion:
            return "LOWER_OCCLUSION"
        loser_coverage = float(loser.get("new_observable_cells") or 0)
        winner_coverage = float(winner.get("new_observable_cells") or 0)
        if loser_coverage < winner_coverage:
            return "LOWER_NEW_OBSERVABLE"
        loser_radius = float(loser.get("candidate_radius_m") or math.hypot(*(loser.get("base_xy") or [0.0, 0.0])))
        winner_radius = float(winner.get("candidate_radius_m") or math.hypot(*(winner.get("base_xy") or [0.0, 0.0])))
        loser_error = 0.0 if PREFERRED_MIN_RADIUS_M <= loser_radius <= PREFERRED_MAX_RADIUS_M else min(
            abs(loser_radius - PREFERRED_MIN_RADIUS_M), abs(loser_radius - PREFERRED_MAX_RADIUS_M)
        )
        winner_error = 0.0 if PREFERRED_MIN_RADIUS_M <= winner_radius <= PREFERRED_MAX_RADIUS_M else min(
            abs(winner_radius - PREFERRED_MIN_RADIUS_M), abs(winner_radius - PREFERRED_MAX_RADIUS_M)
        )
        if loser_error > winner_error:
            return "RADIAL_PREFERENCE"
        if loser_radius < winner_radius:
            return "RADIUS_TIEBREAK"
        return "OTHER_EXISTING_ORDER"

    @staticmethod
    def _candidate_audit_summary(row: Dict[str, object]) -> Dict[str, object]:
        """Return compact existing candidate fields; never visibility/path arrays."""
        return {
            "candidate_id": row.get("_room_search_audit_candidate_id"),
            "target_room_xy": list(row.get("room_target_xy") or []),
            "target_odom_xy": list(row.get("target_xy_team_livox_odom") or []),
            "sector": row.get("sector"),
            "candidate_radius_m": row.get("candidate_radius_m"),
            "preferred_radial_band": row.get("preferred_radial_band"),
            "cheap_geometric_distance_m": row.get("cheap_geometric_distance_m"),
            "heading_change_rad": row.get("heading_change_rad"),
            "door_keepout_soft_factor": row.get("door_keepout_soft_factor"),
            "action_class": row.get("target_priority_class"),
            "occlusion_reveal_cells": row.get("occlusion_reveal_cells"),
            "new_observable_cells": row.get("new_observable_cells"),
            "trajectory_revisit_coarse_location": row.get("trajectory_revisit_coarse_location"),
            "generic_trajectory_revisit_preference_applied": row.get("generic_trajectory_revisit_preference_applied"),
            "failed_observation_opportunity_suppressed": row.get("failed_observation_opportunity_suppressed"),
            "failed_observation_opportunity_reason": row.get("failed_observation_opportunity_reason"),
        }

    def candidates_from_planning_free_base(
        self, current_pose_xy_yaw: Sequence[float], planning_free_base_points: Iterable[Sequence[float]],
        task_enricher: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
        audit: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, object]]:
        """Represent spatially distinct *current* free Grid directions.

        There is no left/right order: an angular sector exists only when the
        supplied existing planning mask contains a local free point in it.
        """
        x, y, yaw = (float(value) for value in current_pose_xy_yaw)
        by_sector: Dict[int, Dict[str, object]] = {}
        raw_rows: List[Dict[str, object]] = []
        for base in planning_free_base_points:
            bx, by = float(base[0]), float(base[1])
            radius = math.hypot(bx, by)
            if radius > LOCAL_MAX_RADIUS_M or radius < 0.25:
                continue
            sector = int(math.floor((math.atan2(by, bx) + math.pi) / (math.pi / 6.0))) % 12
            odom = (x + math.cos(yaw) * bx - math.sin(yaw) * by, y + math.sin(yaw) * bx + math.cos(yaw) * by)
            row: Dict[str, object] = {
                "sector": sector,
                "base_xy": [bx, by],
                "target_xy_team_livox_odom": [odom[0], odom[1]],
                "room_target_xy": list(self.room_xy(odom)),
                "candidate_radius_m": radius,
                "preferred_radial_band": bool(PREFERRED_MIN_RADIUS_M <= radius <= PREFERRED_MAX_RADIUS_M),
            }
            if task_enricher is not None:
                row = self._with_cheap_task_values(task_enricher(dict(row)))
                row = self._with_failed_observation_opportunity_status(row)
            if audit is not None:
                row["_room_search_audit_candidate_id"] = f"raw-{len(raw_rows):04d}"
                raw_rows.append(row)
            # Eligibility is decided before sector compression.  Otherwise a
            # suppressed claim could occupy a sector and hide a different,
            # still-valid observation opportunity in that same sector.
            if bool(row.get("failed_observation_opportunity_suppressed")):
                continue
            existing = by_sector.get(sector)
            if existing is None or self._within_sector_key(row) < self._within_sector_key(existing):
                by_sector[sector] = row
        skipped_sector = self.cooldown_sector
        self.cooldown_sector = None
        candidates: List[Dict[str, object]] = []
        for sector, row in by_sector.items():
            if skipped_sector is not None and sector == skipped_sector:
                continue
            if self._near_rejected(row["target_xy_team_livox_odom"]):
                continue
            candidates.append(dict(row))
        if audit is not None:
            audit["raw_candidate_count_before_sector_compression"] = len(raw_rows)
            audit["raw_candidates"] = []
            for row in raw_rows:
                winner = by_sector.get(int(row["sector"]))
                survived = winner is row
                item = self._candidate_audit_summary(row)
                item["survived_as_sector_representative"] = survived
                item["representative_winner_candidate_id"] = (
                    winner.get("_room_search_audit_candidate_id") if winner is not None and not survived else None
                )
                item["representative_comparison_reason"] = (
                    None if survived or winner is None else self._within_sector_discard_reason(row, winner)
                )
                item["included_in_global_ranking"] = bool(
                    survived and any(candidate.get("_room_search_audit_candidate_id") == row.get("_room_search_audit_candidate_id") for candidate in candidates)
                )
                audit["raw_candidates"].append(item)
            audit["sector_representative_count"] = len(by_sector)
            audit["ranked_representative_count"] = len(candidates)
        # Keep one representative per current free angular sector for task
        # scoring.  The five-candidate cap belongs *after* value ranking, not
        # before it, so a useful occlusion sector cannot disappear by index.
        return sorted(
            candidates,
            key=lambda row: (not bool(row["preferred_radial_band"]), float(row["candidate_radius_m"]), int(row["sector"])),
        )[:RAW_SECTOR_REPRESENTATIVE_CAP]

    def cheap_rank_candidates(self, candidates: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        """Rank danger, then normal candidates by remaining coverage before preflight.

        The caller supplies current-Grid visibility and occlusion-reveal point
        sets.  This function only applies the fixed hierarchy; it never calls a
        planner or treats a candidate as executable.
        """
        ranked: List[Dict[str, object]] = []
        for candidate in candidates:
            ranked.append(self._with_cheap_task_values(candidate))

        def priority_key(row: Dict[str, object]) -> Tuple[float, ...]:
            danger_reobserve = bool(row.get("danger_reobserve_supported"))
            occlusion = int(row.get("occlusion_reveal_cells") or 0)
            coverage = int(row.get("new_observable_cells") or 0)
            generic_trajectory_revisit = bool(row.get("generic_trajectory_revisit_preference_applied"))
            distance = float(row.get("cheap_geometric_distance_m") or float("inf"))
            heading = abs(float(row.get("heading_change_rad") or 0.0))
            danger_bearing = abs(float(row.get("danger_reobserve_abs_bearing_rad") or float("inf")))
            return (
                0.0 if danger_reobserve else 1.0,
                danger_bearing if danger_reobserve else 0.0,
                -float(coverage),
                -float(occlusion),
                # Only equal-value GENERIC candidates use the soft physical
                # revisit preference.  Danger keeps its separate task
                # priority, and normal coverage remains ahead of revisit.
                1.0 if generic_trajectory_revisit else 0.0,
                distance,
                heading,
                -float(row.get("door_keepout_soft_factor") or 1.0),
                float(row.get("sector") or 0),
            )

        return sorted(ranked, key=priority_key)

    @staticmethod
    def admit_ranked_candidates(
        candidates: Iterable[Dict[str, object]], preflight: Callable[[Dict[str, object]], Dict[str, object]],
        audit_observer: Optional[Callable[[int, Dict[str, object], Dict[str, object]], None]] = None,
    ) -> Tuple[Optional[Dict[str, object]], int]:
        """Try ranked candidates in order and stop at the first formal admission."""
        attempts = 0
        for rank, candidate in enumerate(candidates, 1):
            attempts += 1
            result = preflight(dict(candidate))
            if audit_observer is not None:
                audit_observer(rank, dict(candidate), dict(result))
            if not bool(result.get("legal")):
                continue
            selected = dict(candidate)
            selected.update(result)
            selected["cheap_rank"] = rank
            return selected, attempts
        return None, attempts

    @staticmethod
    def admit_with_terminal_expansion(
        candidates: Iterable[Dict[str, object]], preflight: Callable[[Dict[str, object]], Dict[str, object]], normal_cap: int = MAX_CANDIDATES,
        audit_observer: Optional[Callable[[int, Dict[str, object], Dict[str, object]], None]] = None,
    ) -> Tuple[Optional[Dict[str, object]], int, int]:
        """Keep the five-candidate fast path, then exhaust the bounded sector set.

        The caller still owns execution.  This helper only controls the number
        and order of existing one-step formal preflights.
        """
        ranked = [dict(candidate) for candidate in candidates]
        selected, normal_attempts = RoomSearchV2.admit_ranked_candidates(
            ranked[:normal_cap], preflight, audit_observer=audit_observer,
        )
        if selected is not None:
            return selected, normal_attempts, 0
        expansion_observer = None
        if audit_observer is not None:
            expansion_observer = lambda rank, candidate, result: audit_observer(normal_cap + rank, candidate, result)
        selected, expansion_attempts = RoomSearchV2.admit_ranked_candidates(
            ranked[normal_cap:], preflight, audit_observer=expansion_observer,
        )
        if selected is not None:
            selected["cheap_rank"] = normal_cap + int(selected.get("cheap_rank") or 0)
        return selected, normal_attempts + expansion_attempts, expansion_attempts

    def score_candidates(self, candidates: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        scored: List[Dict[str, object]] = []
        for candidate in candidates:
            visible = candidate.get("visible_room_points") or []
            path_length = float(candidate.get("path_length_m") or 0.0)
            raw_gain = self.observation.new_count(visible)
            room_x = float((candidate.get("room_target_xy") or [0.0])[0])
            soft_factor = DOOR_NEAR_UTILITY_FACTOR if room_x < DOOR_NEAR_SOFT_DEPTH_M else 1.0
            utility = (raw_gain * soft_factor) / max(path_length, 1e-6)
            row = dict(candidate)
            row.update({
                "new_observable_cells": raw_gain,
                "door_keepout_soft_factor": soft_factor,
                "nbv_value": utility,
            })
            scored.append(row)
        return sorted(scored, key=lambda row: (-float(row["nbv_value"]), float(row["path_length_m"]), abs(float(row.get("heading_change_rad") or 0.0))))

    def select_best(self, candidates: Iterable[Dict[str, object]]) -> Optional[Dict[str, object]]:
        scored = self.score_candidates(candidates)
        if not scored:
            return None
        return scored[0]

    @staticmethod
    def termination_baseline_eligibility(candidate: Dict[str, object]) -> Dict[str, object]:
        """Classify whether a candidate's NBV can update completion evidence."""
        raw_path_length = candidate.get("path_length_m")
        if isinstance(raw_path_length, bool):
            return {
                "termination_baseline_eligible": False,
                "termination_baseline_ineligibility_reason": "NONFINITE_PATH",
                "path_length_m": raw_path_length,
            }
        try:
            path_length = float(raw_path_length)
        except (TypeError, ValueError):
            return {
                "termination_baseline_eligible": False,
                "termination_baseline_ineligibility_reason": "MISSING_PATH",
                "path_length_m": raw_path_length,
            }
        if not math.isfinite(path_length):
            return {
                "termination_baseline_eligible": False,
                "termination_baseline_ineligibility_reason": "NONFINITE_PATH",
                "path_length_m": raw_path_length,
            }
        if path_length == 0.0:
            return {
                "termination_baseline_eligible": False,
                "termination_baseline_ineligibility_reason": "ZERO_PATH",
                "path_length_m": path_length,
            }
        if path_length < 0.0:
            return {
                "termination_baseline_eligible": False,
                "termination_baseline_ineligibility_reason": "NEGATIVE_PATH",
                "path_length_m": path_length,
            }
        return {
            "termination_baseline_eligible": True,
            "termination_baseline_ineligibility_reason": None,
            "path_length_m": path_length,
        }

    def reset_marginal_completion_epoch(
        self,
        *,
        decision_index: int,
        run_id: str,
        actual_new_observation_cells: object,
    ) -> Dict[str, object]:
        """Close one marginal epoch after a truthful substantive observation."""
        actual_new = (
            int(actual_new_observation_cells)
            if isinstance(actual_new_observation_cells, int)
            and not isinstance(actual_new_observation_cells, bool)
            else 0
        )
        actual_new = max(0, actual_new)
        execution_id = "%s:room_search_decision:%d" % (str(run_id or "UNBOUND_RUN"), int(decision_index))
        record: Dict[str, object] = {
            "event": "MARGINAL_COMPLETION_EPOCH_RESET",
            "reason": "ACTUAL_OBSERVATION_PROGRESS",
            "run_id": str(run_id or ""),
            "decision_index": int(decision_index),
            "execution_id": execution_id,
            "actual_new_observation_cells": actual_new,
            "peak_best_nbv_value_before": self.peak_best_nbv_value,
            "low_gain_streak_before": self.low_gain_streak,
            "peak_best_nbv_value_after": self.peak_best_nbv_value,
            "low_gain_streak_after": self.low_gain_streak,
            "reset_applied": False,
        }
        if actual_new <= 0:
            record["reason"] = "NO_SUBSTANTIVE_PROGRESS"
            return record
        if execution_id in self.marginal_epoch_reset_execution_ids:
            record["reason"] = "DUPLICATE_EXECUTION"
            return record
        self.marginal_epoch_reset_execution_ids.add(execution_id)
        self.peak_best_nbv_value = None
        self.low_gain_streak = 0
        record.update({
            "peak_best_nbv_value_after": None,
            "low_gain_streak_after": 0,
            "reset_applied": True,
        })
        self.gain_history.append(record)
        return record

    def evaluate_marginal_value(self, candidate: Dict[str, object], decision_index: int) -> Dict[str, object]:
        """Record the session-local diminishing-return decision for one NBV.

        This observes the already ranked and planner-admitted candidate only.
        It does not inspect runner failures or alter candidate ranking.
        """
        current_value = float(candidate.get("nbv_value") or 0.0)
        baseline = self.termination_baseline_eligibility(candidate)
        record: Dict[str, object] = {
            "decision_index": int(decision_index),
            "best_candidate_nbv_value": current_value,
            "peak_best_nbv_value": self.peak_best_nbv_value,
            "relative_gain": None,
            "low_gain_streak": self.low_gain_streak,
            "completion_reason": None,
            **baseline,
        }
        if not bool(baseline["termination_baseline_eligible"]):
            self.gain_history.append(record)
            return record
        if str(candidate.get("target_priority_class") or "") == "OCCLUSION":
            # Generic coverage diminishing-return evidence is not evidence
            # against a presently hidden region.  Keep its exact path-cost
            # value for diagnostics, but restart the generic-only streak.
            self.low_gain_streak = 0
            record.update({
                "low_gain_streak": 0,
                "generic_low_gain_bypassed": True,
            })
            self.gain_history.append(record)
            return record
        if current_value <= 0.0:
            record["completion_reason"] = "NON_POSITIVE_GAIN"
            self.gain_history.append(record)
            return record

        if self.peak_best_nbv_value is None:
            self.peak_best_nbv_value = current_value
        else:
            self.peak_best_nbv_value = max(self.peak_best_nbv_value, current_value)
        relative_gain = current_value / self.peak_best_nbv_value
        if relative_gain <= LOW_GAIN_RATIO:
            self.low_gain_streak += 1
        else:
            self.low_gain_streak = 0

        record.update({
            "peak_best_nbv_value": self.peak_best_nbv_value,
            "relative_gain": relative_gain,
            "low_gain_streak": self.low_gain_streak,
        })
        if self.low_gain_streak >= LOW_GAIN_PATIENCE:
            record["completion_reason"] = "DIMINISHING_RETURN"
        self.gain_history.append(record)
        return record

    def update_actual_view(self, actual_pose_xy_yaw: Sequence[float], visible_room_points: Iterable[Sequence[float]]) -> int:
        del actual_pose_xy_yaw  # The caller computed visibility from this actual pose.
        return self.observation.update(visible_room_points)

    def record_actual_breadcrumb(self, actual_pose_xy_yaw: Sequence[float], spacing_m: float) -> bool:
        point = tuple(float(value) for value in actual_pose_xy_yaw)
        if len(point) != 3:
            raise ValueError("room_search_actual_breadcrumb_requires_xy_yaw")
        if self.breadcrumbs and math.hypot(point[0] - self.breadcrumbs[-1][0], point[1] - self.breadcrumbs[-1][1]) <= float(spacing_m):
            return False
        self.breadcrumbs.append(point)
        return True

    def record_safe_actual_pose(self, actual_pose_xy_yaw: Sequence[float], spacing_m: float) -> bool:
        """Store an actual pose for bounded search-time reposition only."""
        point = tuple(float(value) for value in actual_pose_xy_yaw)
        if len(point) != 3:
            raise ValueError("room_search_safe_actual_pose_requires_xy_yaw")
        if self.safe_actual_poses and math.hypot(point[0] - self.safe_actual_poses[-1][0], point[1] - self.safe_actual_poses[-1][1]) <= float(spacing_m):
            return False
        self.safe_actual_poses.append(point)
        return True

    def reposition_anchor_candidates(
        self, current_pose_xy_yaw: Sequence[float], minimum_distance_m: float, maximum_attempts: int = 2
    ) -> List[Dict[str, object]]:
        """Return only recent, distinct *actual* poses; caller still preflights them."""
        current = tuple(float(value) for value in current_pose_xy_yaw)
        anchors: List[Dict[str, object]] = []
        for index in range(len(self.safe_actual_poses) - 1, -1, -1):
            pose = self.safe_actual_poses[index]
            distance = math.hypot(pose[0] - current[0], pose[1] - current[1])
            if distance <= float(minimum_distance_m):
                continue
            anchors.append({
                "kind": "STRATEGIC_REPOSITION_ANCHOR",
                "anchor_index": index,
                "actual_pose_xy_yaw": list(pose),
                "target_xy_team_livox_odom": [pose[0], pose[1]],
                "anchor_distance_m": distance,
            })
            if len(anchors) >= max(0, int(maximum_attempts)):
                break
        return anchors

    def arm_one_decision_sector_cooldown(self, sector: Optional[object]) -> None:
        self.cooldown_sector = int(sector) if isinstance(sector, (int, float)) else None

    def reject_context_target(self, target_xy: Sequence[float]) -> None:
        point = (float(target_xy[0]), float(target_xy[1]))
        if not self._near_rejected(point):
            self.rejected_context_points.append(point)

    def door_return_target(self) -> Dict[str, object]:
        x, y, _yaw = self.anchor.door_return_anchor_xy_yaw
        return {"kind": "DOOR_RETURN_ANCHOR", "target_xy_team_livox_odom": [x, y]}

    def at_door_return_anchor(self, actual_pose_xy_yaw: Sequence[float], goal_tolerance_m: float) -> bool:
        x, y, _yaw = self.anchor.door_return_anchor_xy_yaw
        return math.hypot(float(actual_pose_xy_yaw[0]) - x, float(actual_pose_xy_yaw[1]) - y) <= float(goal_tolerance_m)

    def choose_return_target(
        self, door_option: Dict[str, object], breadcrumb_options: Iterable[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        """Prefer a presently executable door target, then useful safe anchors.

        ``legal`` is supplied by the existing no-execute runner.  This helper
        only chooses between results; it neither plans nor replays an old path.
        """
        if bool(door_option.get("legal")):
            selected = dict(door_option)
            selected["selected_target_type"] = "DOOR_DIRECT"
            return selected
        eligible = [dict(option) for option in breadcrumb_options if bool(option.get("legal"))]
        if not eligible:
            return None
        selected = min(
            eligible,
            key=lambda option: (
                float(option.get("target_distance_to_door") or float("inf")),
                float(option.get("path_length_m") or float("inf")),
                int(option.get("breadcrumb_index") or 0),
            ),
        )
        selected["selected_target_type"] = "BREADCRUMB_FALLBACK"
        return selected

    @staticmethod
    def choose_safe_return_transition(
        current_xy: Sequence[float], intended_target_xy: Sequence[float], candidates: Iterable[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        """Choose a formally admitted local transition with return progress.

        Distance reduction remains preferred.  If the target is lateral or
        behind, a forward local move may instead make its relative bearing
        strictly smaller; that is useful reorientation, not a blind reverse.
        """
        before = math.hypot(float(intended_target_xy[0]) - float(current_xy[0]), float(intended_target_xy[1]) - float(current_xy[1]))
        current_yaw = float(current_xy[2]) if len(current_xy) >= 3 else 0.0
        target_bearing = abs(math.atan2(float(intended_target_xy[1]) - float(current_xy[1]), float(intended_target_xy[0]) - float(current_xy[0])) - current_yaw)
        target_bearing = abs((target_bearing + math.pi) % (2.0 * math.pi) - math.pi)
        improving: List[Tuple[float, float, Dict[str, object]]] = []
        for candidate in candidates:
            if not bool(candidate.get("legal")):
                continue
            candidate_xy = candidate.get("target_xy_team_livox_odom")
            if not isinstance(candidate_xy, (list, tuple)) or len(candidate_xy) < 2:
                continue
            after = math.hypot(float(intended_target_xy[0]) - float(candidate_xy[0]), float(intended_target_xy[1]) - float(candidate_xy[1]))
            transition_heading = math.atan2(float(candidate_xy[1]) - float(current_xy[1]), float(candidate_xy[0]) - float(current_xy[0]))
            after_bearing = abs(math.atan2(float(intended_target_xy[1]) - float(candidate_xy[1]), float(intended_target_xy[0]) - float(candidate_xy[0])) - transition_heading)
            after_bearing = abs((after_bearing + math.pi) % (2.0 * math.pi) - math.pi)
            distance_gain, bearing_gain = before - after, target_bearing - after_bearing
            if distance_gain > 0.05 or (target_bearing >= math.pi / 2.0 and bearing_gain > 1e-6):
                row = dict(candidate)
                row.update({"return_distance_gain_m": distance_gain, "return_bearing_gain_rad": bearing_gain})
                improving.append((distance_gain, bearing_gain, row))
        return max(improving, key=lambda item: (item[0] > 0.05, item[0], item[1]))[2] if improving else None
