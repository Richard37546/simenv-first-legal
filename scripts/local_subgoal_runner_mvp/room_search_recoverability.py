#!/usr/bin/env python3
"""Pure, shadow-only Recoverability R1 evidence contracts.

This module intentionally does not publish, plan, sample DWA, or mutate a
ROOM_SEARCH state machine.  It only classifies already-frozen terminal and
retreat evidence.  In particular, the current ROOM_SEARCH dry preflight has
no complete candidate action-terminal prediction; ``prepare_*`` preserves
that fact as ``TERMINAL_STATE_UNKNOWN`` instead of treating a one-step DWA
slice as the terminal state of a multi-step candidate execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RECOVERABLE = "RECOVERABLE"
NON_RECOVERABLE = "NON_RECOVERABLE"
UNKNOWN = "UNKNOWN"
TERMINAL_STATE_UNKNOWN = "TERMINAL_STATE_UNKNOWN"
MAX_RECOVERABILITY_PREDECESSORS = 3


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _pose_or_none(value: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_finite(item) for item in value):
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _canonical_epoch(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "grid_header_stamp_sec": value.get("grid_header_stamp_sec"),
        "grid_content_stamp": value.get("grid_content_stamp"),
        "content_generation_id": value.get("content_generation_id"),
        "grid_content_hash": value.get("grid_content_hash"),
    }


def recoverability_epoch_identity(planning_context: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose only the existing planning epoch identity for evidence binding."""
    return _canonical_epoch(planning_context)


@dataclass(frozen=True)
class PredictedTerminalState:
    candidate_id: str
    decision_id: int
    pose_xy_yaw: Optional[Tuple[float, float, float]]
    completion_semantics: str
    prediction_horizon_sec: Optional[float]
    source: str
    source_evidence: Dict[str, Any]
    status: str
    reason: str
    epoch_identity: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        pose = list(self.pose_xy_yaw) if self.pose_xy_yaw is not None else [None, None, None]
        return {
            "candidate_id": self.candidate_id,
            "decision_id": self.decision_id,
            "x": pose[0], "y": pose[1], "yaw": pose[2],
            "completion_semantics": self.completion_semantics,
            "prediction_horizon_sec": self.prediction_horizon_sec,
            "source": self.source,
            "source_evidence": dict(self.source_evidence),
            "status": self.status,
            "reason": self.reason,
            "epoch_identity": dict(self.epoch_identity),
        }


@dataclass(frozen=True)
class RecoverabilityCertificate:
    state_id: str
    pose_xy_yaw: Optional[Tuple[float, float, float]]
    status: str
    predecessor_state_id: Optional[str]
    epoch_identity: Dict[str, Any]
    retreat_transition_type: Optional[str]
    fresh: bool
    reason: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        pose = list(self.pose_xy_yaw) if self.pose_xy_yaw is not None else [None, None, None]
        return {
            "state_id": self.state_id,
            "x": pose[0], "y": pose[1], "yaw": pose[2],
            "status": self.status,
            "certified_predecessor_identity": self.predecessor_state_id,
            "evidence_epoch": dict(self.epoch_identity),
            "retreat_transition_type": self.retreat_transition_type,
            "fresh": self.fresh,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


def make_door_anchor_root_certificate(
    anchor: Any,
    epoch_identity: Mapping[str, Any],
) -> RecoverabilityCertificate:
    """Create R0 from the existing frozen DoorAnchor, never from breadcrumbs."""
    pose = _pose_or_none(getattr(anchor, "door_return_anchor_xy_yaw", None))
    epoch = _canonical_epoch(epoch_identity)
    if pose is None:
        return RecoverabilityCertificate(
            state_id="R0_DOOR_ANCHOR", pose_xy_yaw=None, status=UNKNOWN,
            predecessor_state_id=None, epoch_identity=epoch, retreat_transition_type=None,
            fresh=False, reason="DOOR_ANCHOR_POSE_UNAVAILABLE", evidence={},
        )
    return RecoverabilityCertificate(
        state_id="R0_DOOR_ANCHOR", pose_xy_yaw=pose, status=RECOVERABLE,
        predecessor_state_id=None, epoch_identity=epoch, retreat_transition_type="ROOT",
        fresh=True, reason="FROZEN_DOOR_ANCHOR_ROOT", evidence={
            "door_return_anchor_stamp_sec": getattr(anchor, "door_return_anchor_stamp_sec", None),
            "recency_index": 0,
        },
    )


def prepare_candidate_terminal_state(
    candidate: Mapping[str, Any],
    decision_id: int,
    epoch_identity: Mapping[str, Any],
) -> PredictedTerminalState:
    """Use an explicit full-action terminal source when one is supplied.

    The current runner/preflight contract does *not* supply this field.  Its
    ``last_dwa`` slice is intentionally not read here: it represents one
    command slice, whereas a ROOM_SEARCH candidate may execute the complete
    ``runner_max_steps`` contract.
    """
    candidate_id = str(candidate.get("_room_search_audit_candidate_id") or candidate.get("candidate_id") or "unknown")
    epoch = _canonical_epoch(epoch_identity)
    runner = candidate.get("runner")
    runner = runner if isinstance(runner, Mapping) else {}
    supplied = runner.get("recoverability_candidate_terminal_prediction")
    supplied = supplied if isinstance(supplied, Mapping) else None
    pose = _pose_or_none(supplied.get("pose_xy_yaw") if supplied is not None else None)
    completion = str(supplied.get("completion_semantics") or "") if supplied is not None else ""
    if (
        supplied is not None
        and pose is not None
        and completion == "FULL_CANDIDATE_ACTION_COMPLETION"
        and str(supplied.get("candidate_id") or candidate_id) == candidate_id
    ):
        return PredictedTerminalState(
            candidate_id=candidate_id, decision_id=int(decision_id), pose_xy_yaw=pose,
            completion_semantics=completion,
            prediction_horizon_sec=(float(supplied["prediction_horizon_sec"]) if _finite(supplied.get("prediction_horizon_sec")) else None),
            source=str(supplied.get("source") or "RUNNER_FULL_ACTION_TERMINAL"),
            source_evidence=dict(supplied), status="TERMINAL_STATE_AVAILABLE",
            reason="EXPLICIT_FULL_CANDIDATE_ACTION_TERMINAL", epoch_identity=epoch,
        )
    return PredictedTerminalState(
        candidate_id=candidate_id, decision_id=int(decision_id), pose_xy_yaw=None,
        completion_semantics="FULL_CANDIDATE_ACTION_COMPLETION_REQUIRED",
        prediction_horizon_sec=None,
        source="CURRENT_ROOM_SEARCH_PREFLIGHT_HAS_NO_FULL_ACTION_TERMINAL",
        source_evidence={
            "runner_final_decision": runner.get("runner_final_decision"),
            "runner_max_steps_semantics": "FORMAL_EXECUTION_MAY_USE_RUNNER_MAX_STEPS",
            "last_dwa_present": isinstance(runner.get("last_dwa"), Mapping),
            "last_dwa_not_used_as_terminal": True,
        },
        status=TERMINAL_STATE_UNKNOWN,
        reason="TERMINAL_STATE_UNKNOWN_NO_COMPLETE_ACTION_PREDICTION",
        epoch_identity=epoch,
    )


def _epoch_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _canonical_epoch(left) == _canonical_epoch(right) and any(
        value is not None for value in _canonical_epoch(left).values()
    )


def _transition_result(value: Any) -> Tuple[str, str]:
    """Classify supplied, pure transition evidence without executing anything."""
    if not isinstance(value, Mapping):
        return UNKNOWN, "RETREAT_TRANSITION_EVIDENCE_MISSING"
    if not bool(value.get("complete")):
        return UNKNOWN, str(value.get("reason") or "RETREAT_TRANSITION_INCOMPLETE")
    if bool(value.get("executable")):
        return RECOVERABLE, str(value.get("reason") or "RETREAT_TRANSITION_EXECUTABLE")
    return NON_RECOVERABLE, str(value.get("reason") or "RETREAT_TRANSITION_NOT_EXECUTABLE")


def evaluate_retreat(
    terminal: PredictedTerminalState,
    predecessor: RecoverabilityCertificate,
    frozen_evidence: Mapping[str, Any],
) -> RecoverabilityCertificate:
    """Evaluate only direct or one-orientation-plus-translation retreat proof.

    ``frozen_evidence`` is expected to contain already-produced pure evidence:
    ``direct_translation`` and ``orientation_then_translation``.  The
    evaluator neither asks ROS for data nor invokes a runner.
    """
    epoch = _canonical_epoch(terminal.epoch_identity)
    if terminal.status != "TERMINAL_STATE_AVAILABLE" or terminal.pose_xy_yaw is None:
        return RecoverabilityCertificate(
            state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
            status=UNKNOWN, predecessor_state_id=predecessor.state_id, epoch_identity=epoch,
            retreat_transition_type=None, fresh=False,
            reason=terminal.reason, evidence={"terminal": terminal.to_dict()},
        )
    if predecessor.status != RECOVERABLE or not predecessor.fresh or not _epoch_equal(epoch, predecessor.epoch_identity):
        return RecoverabilityCertificate(
            state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
            status=UNKNOWN, predecessor_state_id=predecessor.state_id, epoch_identity=epoch,
            retreat_transition_type=None, fresh=False,
            reason="PREDECESSOR_CERTIFICATE_STALE_OR_UNCERTIFIED", evidence={"predecessor": predecessor.to_dict()},
        )
    direct_status, direct_reason = _transition_result(frozen_evidence.get("direct_translation"))
    if direct_status == RECOVERABLE:
        return RecoverabilityCertificate(
            state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
            status=RECOVERABLE, predecessor_state_id=predecessor.state_id, epoch_identity=epoch,
            retreat_transition_type="DIRECT_TRANSLATION", fresh=True, reason=direct_reason,
            evidence={"direct_translation": dict(frozen_evidence.get("direct_translation") or {})},
        )
    orientation_status, orientation_reason = _transition_result(frozen_evidence.get("orientation_then_translation"))
    if orientation_status == RECOVERABLE:
        return RecoverabilityCertificate(
            state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
            status=RECOVERABLE, predecessor_state_id=predecessor.state_id, epoch_identity=epoch,
            retreat_transition_type="ORIENTATION_THEN_TRANSLATION", fresh=True, reason=orientation_reason,
            evidence={
                "direct_translation": dict(frozen_evidence.get("direct_translation") or {}),
                "orientation_then_translation": dict(frozen_evidence.get("orientation_then_translation") or {}),
            },
        )
    if direct_status == UNKNOWN or orientation_status == UNKNOWN:
        return RecoverabilityCertificate(
            state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
            status=UNKNOWN, predecessor_state_id=predecessor.state_id, epoch_identity=epoch,
            retreat_transition_type=None, fresh=False,
            reason="%s|%s" % (direct_reason, orientation_reason), evidence={
                "direct_translation": dict(frozen_evidence.get("direct_translation") or {}),
                "orientation_then_translation": dict(frozen_evidence.get("orientation_then_translation") or {}),
            },
        )
    return RecoverabilityCertificate(
        state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
        status=NON_RECOVERABLE, predecessor_state_id=predecessor.state_id, epoch_identity=epoch,
        retreat_transition_type=None, fresh=True,
        reason="DIRECT_AND_ONE_ORIENTATION_RETREATS_NOT_EXECUTABLE", evidence={
            "direct_translation": dict(frozen_evidence.get("direct_translation") or {}),
            "orientation_then_translation": dict(frozen_evidence.get("orientation_then_translation") or {}),
        },
    )


def evaluate_predecessor_set(
    terminal: PredictedTerminalState,
    predecessors: Iterable[RecoverabilityCertificate],
    evidence_by_predecessor: Mapping[str, Mapping[str, Any]],
    max_predecessors: int = MAX_RECOVERABILITY_PREDECESSORS,
) -> RecoverabilityCertificate:
    """Use bounded newest-first predecessor evidence without first-only failure."""
    if terminal.status != "TERMINAL_STATE_AVAILABLE" or terminal.pose_xy_yaw is None:
        return RecoverabilityCertificate(
            state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
            status=UNKNOWN, predecessor_state_id=None, epoch_identity=dict(terminal.epoch_identity),
            retreat_transition_type=None, fresh=False, reason=terminal.reason,
            evidence={"terminal": terminal.to_dict()},
        )
    def predecessor_order(item: RecoverabilityCertificate) -> Tuple[int, str]:
        raw_index = item.evidence.get("recency_index") if isinstance(item.evidence, Mapping) else None
        index = int(raw_index) if isinstance(raw_index, int) and not isinstance(raw_index, bool) else -1
        return (-index, str(item.state_id))

    ordered: List[RecoverabilityCertificate] = sorted(list(predecessors), key=predecessor_order)
    cap = max(1, int(max_predecessors))
    considered = ordered[:cap]
    truncated = len(ordered) > len(considered)
    results: List[RecoverabilityCertificate] = []
    for predecessor in considered:
        result = evaluate_retreat(terminal, predecessor, evidence_by_predecessor.get(predecessor.state_id, {}))
        results.append(result)
        if result.status == RECOVERABLE:
            return RecoverabilityCertificate(
                **{**result.__dict__, "evidence": {
                    **result.evidence,
                    "predecessors_considered": [item.state_id for item in considered],
                    "predecessor_order": "RECENCY_INDEX_DESCENDING_THEN_STATE_ID_ASCENDING",
                    "cap_truncated": truncated,
                }}
            )
    if truncated:
        return RecoverabilityCertificate(
            state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
            status=UNKNOWN, predecessor_state_id=None, epoch_identity=dict(terminal.epoch_identity),
            retreat_transition_type=None, fresh=False, reason="PREDECESSOR_CAP_TRUNCATED",
            evidence={"predecessors_considered": [item.state_id for item in considered], "predecessor_cap": cap},
        )
    if any(item.status == UNKNOWN for item in results):
        return RecoverabilityCertificate(
            state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
            status=UNKNOWN, predecessor_state_id=None, epoch_identity=dict(terminal.epoch_identity),
            retreat_transition_type=None, fresh=False, reason="PREDECESSOR_RETREAT_EVIDENCE_INCOMPLETE",
            evidence={"predecessor_results": [item.to_dict() for item in results]},
        )
    if not results:
        return RecoverabilityCertificate(
            state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
            status=UNKNOWN, predecessor_state_id=None, epoch_identity=dict(terminal.epoch_identity),
            retreat_transition_type=None, fresh=False, reason="NO_CERTIFIED_PREDECESSOR_AVAILABLE", evidence={},
        )
    return RecoverabilityCertificate(
        state_id="R1:%s" % terminal.candidate_id, pose_xy_yaw=terminal.pose_xy_yaw,
        status=NON_RECOVERABLE, predecessor_state_id=None, epoch_identity=dict(terminal.epoch_identity),
        retreat_transition_type=None, fresh=True, reason="ALL_COMPLETE_PREDECESSOR_RETREATS_NON_RECOVERABLE",
        evidence={"predecessor_results": [item.to_dict() for item in results]},
    )


def terminal_accuracy_telemetry(
    terminal: PredictedTerminalState,
    actual_pose_xy_yaw: Sequence[float],
    outcome: str,
) -> Dict[str, Any]:
    """Bind future S1-prediction accuracy evidence after a fresh actual pose."""
    actual = _pose_or_none(actual_pose_xy_yaw)
    prediction = terminal.pose_xy_yaw
    position_error = None
    yaw_error = None
    if prediction is not None and actual is not None:
        position_error = math.hypot(prediction[0] - actual[0], prediction[1] - actual[1])
        yaw_error = math.atan2(math.sin(prediction[2] - actual[2]), math.cos(prediction[2] - actual[2]))
    return {
        "candidate_id": terminal.candidate_id,
        "decision_id": terminal.decision_id,
        "prediction": terminal.to_dict(),
        "actual_x": actual[0] if actual is not None else None,
        "actual_y": actual[1] if actual is not None else None,
        "actual_yaw": actual[2] if actual is not None else None,
        "position_error_m": position_error,
        "yaw_error_rad": yaw_error,
        "completion_outcome": str(outcome),
        "physical_accuracy_validated": False,
    }
