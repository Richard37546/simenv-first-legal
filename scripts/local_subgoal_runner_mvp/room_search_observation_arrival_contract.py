#!/usr/bin/env python3
"""Pure, offline-only ROOM_SEARCH observation-arrival shadow evaluator.

This module deliberately has no ROS, filesystem, planner, runner, command, or
state-machine dependency.  It classifies evidence that was already acquired;
it cannot change navigation, SEEN, completion, recovery, or control authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple


CellId = Tuple[int, int]

NAVIGATION_NOT_REACHED = "NAVIGATION_NOT_REACHED"
OBSERVATION_VALID_REACHED_CANDIDATE = "OBSERVATION_VALID_REACHED_CANDIDATE"
OBSERVATION_VALUE_PARTIAL = "OBSERVATION_VALUE_PARTIAL"
OBSERVATION_VALUE_COLLAPSED = "OBSERVATION_VALUE_COLLAPSED"
OBSERVATION_VALIDITY_UNKNOWN = "OBSERVATION_VALIDITY_UNKNOWN"

LOW_OPPORTUNITY_BOUNDARY_NOT_CALIBRATED = "BOUNDARY_NOT_CALIBRATED"


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("%s_must_be_nonnegative_int" % name)
    return int(value)


def canonical_cell_ids(cells: Iterable[Sequence[int]]) -> Tuple[CellId, ...]:
    """Return sorted, unique room-local coarse cell identities without mutation."""
    normalized = set()
    for cell in cells:
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            raise ValueError("cell_id_must_be_pair")
        x, y = cell[0], cell[1]
        if (not isinstance(x, int) or isinstance(x, bool)
                or not isinstance(y, int) or isinstance(y, bool)):
            raise ValueError("cell_id_values_must_be_int")
        normalized.add((int(x), int(y)))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ObservationArrivalShadowResult:
    """Immutable evidence-only result; ``authority_enabled`` is always false."""

    candidate_id: str
    navigation_reached: bool
    observation_state: str
    predicted_new_count: int
    actual_new_count: int
    predicted_new_cell_ids: Tuple[CellId, ...]
    actual_new_cell_ids: Tuple[CellId, ...]
    retained_predicted_cell_ids: Tuple[CellId, ...]
    unexpected_useful_actual_cell_ids: Tuple[CellId, ...]
    terminal_evidence_complete: bool
    terminal_evidence_qualification: Optional[str]
    low_opportunity_candidate: Optional[bool]
    low_opportunity_reason: str
    terminal_pose_xy_yaw: Optional[Tuple[float, float, float]]
    candidate_observation_yaw_rad: Optional[float]
    post_arrival_viability: Optional[str]
    authority_enabled: bool = False

    def to_dict(self) -> dict:
        """Return deterministic JSON-ready telemetry; no side effects."""
        return {
            "candidate_id": self.candidate_id,
            "navigation_reached": self.navigation_reached,
            "observation_state": self.observation_state,
            "predicted_new_count": self.predicted_new_count,
            "actual_new_count": self.actual_new_count,
            "predicted_new_cell_ids": [list(cell) for cell in self.predicted_new_cell_ids],
            "actual_new_cell_ids": [list(cell) for cell in self.actual_new_cell_ids],
            "retained_predicted_cell_ids": [list(cell) for cell in self.retained_predicted_cell_ids],
            "unexpected_useful_actual_cell_ids": [list(cell) for cell in self.unexpected_useful_actual_cell_ids],
            "terminal_evidence_complete": self.terminal_evidence_complete,
            "terminal_evidence_qualification": self.terminal_evidence_qualification,
            "low_opportunity_candidate": self.low_opportunity_candidate,
            "low_opportunity_reason": self.low_opportunity_reason,
            "terminal_pose_xy_yaw": None if self.terminal_pose_xy_yaw is None else list(self.terminal_pose_xy_yaw),
            "candidate_observation_yaw_rad": self.candidate_observation_yaw_rad,
            "post_arrival_viability": self.post_arrival_viability,
            "authority_enabled": False,
        }


def _optional_pose(value: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("terminal_pose_xy_yaw_must_be_triplet_or_none")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        raise ValueError("terminal_pose_xy_yaw_must_be_numeric")


def evaluate_observation_arrival_shadow(
    *,
    navigation_reached: bool,
    candidate_id: str,
    predicted_new_cell_ids: Iterable[Sequence[int]],
    predicted_new_count: int,
    actual_new_cell_ids: Iterable[Sequence[int]],
    actual_new_count: int,
    terminal_evidence_complete: bool,
    terminal_evidence_qualification: Optional[str] = None,
    terminal_pose_xy_yaw: Optional[Sequence[float]] = None,
    candidate_observation_yaw_rad: Optional[float] = None,
    post_arrival_viability: Optional[str] = None,
) -> ObservationArrivalShadowResult:
    """Classify immutable observation evidence using the approved C0 V0 rules.

    ``post_arrival_viability`` is retained as independent telemetry and never
    participates in the state decision.  No low-opportunity threshold exists
    in C0, so that field remains ``None`` with a calibrated-boundary reason.
    """
    if not isinstance(navigation_reached, bool):
        raise ValueError("navigation_reached_must_be_bool")
    if not isinstance(terminal_evidence_complete, bool):
        raise ValueError("terminal_evidence_complete_must_be_bool")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id_must_be_nonempty_str")
    predicted = canonical_cell_ids(predicted_new_cell_ids)
    actual = canonical_cell_ids(actual_new_cell_ids)
    predicted_count = _nonnegative_int(predicted_new_count, "predicted_new_count")
    actual_count = _nonnegative_int(actual_new_count, "actual_new_count")
    if predicted_count != len(predicted):
        raise ValueError("predicted_new_count_does_not_match_cell_ids")
    if actual_count != len(actual):
        raise ValueError("actual_new_count_does_not_match_cell_ids")

    predicted_set, actual_set = set(predicted), set(actual)
    retained = tuple(sorted(predicted_set & actual_set))
    unexpected = tuple(sorted(actual_set - predicted_set))

    if not navigation_reached:
        state = NAVIGATION_NOT_REACHED
    elif not terminal_evidence_complete:
        state = OBSERVATION_VALIDITY_UNKNOWN
    elif predicted_count > 0 and actual_count == 0:
        state = OBSERVATION_VALUE_COLLAPSED
    elif predicted_count > 0 and actual_count >= predicted_count:
        state = OBSERVATION_VALID_REACHED_CANDIDATE
    elif predicted_count > 0 and 0 < actual_count < predicted_count:
        state = OBSERVATION_VALUE_PARTIAL
    else:
        state = OBSERVATION_VALIDITY_UNKNOWN

    yaw = None if candidate_observation_yaw_rad is None else float(candidate_observation_yaw_rad)
    return ObservationArrivalShadowResult(
        candidate_id=candidate_id,
        navigation_reached=navigation_reached,
        observation_state=state,
        predicted_new_count=predicted_count,
        actual_new_count=actual_count,
        predicted_new_cell_ids=predicted,
        actual_new_cell_ids=actual,
        retained_predicted_cell_ids=retained,
        unexpected_useful_actual_cell_ids=unexpected,
        terminal_evidence_complete=terminal_evidence_complete,
        terminal_evidence_qualification=(
            str(terminal_evidence_qualification)
            if terminal_evidence_qualification is not None else None
        ),
        low_opportunity_candidate=None,
        low_opportunity_reason=LOW_OPPORTUNITY_BOUNDARY_NOT_CALIBRATED,
        terminal_pose_xy_yaw=_optional_pose(terminal_pose_xy_yaw),
        candidate_observation_yaw_rad=yaw,
        post_arrival_viability=(str(post_arrival_viability) if post_arrival_viability is not None else None),
        authority_enabled=False,
    )
