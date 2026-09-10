#!/usr/bin/env python3
"""Pure, authority-free ROOM_SEARCH Formal Mission Comparison V1.

This module intentionally has no ROS, runner, file-write, or production
selection import.  It consumes one already frozen same-epoch cohort and emits
an explainable shadow winner only.  The live first-legal path remains outside
this module.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


SCHEMA_VERSION = "formal_mission_comparison_v1"
MISSION_INTENT_EXPLORATION = "EXPLORATION"
MISSION_INTENT_DANGER_REOBSERVE = "DANGER_REOBSERVE"
CONTINUATION_VIABLE = "CONTINUATION_VIABLE"
CONTINUATION_NON_VIABLE = "CONTINUATION_NON_VIABLE"
CONTINUATION_UNKNOWN = "CONTINUATION_UNKNOWN"


def _finite(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cells(value: object) -> Set[Tuple[int, int]]:
    result: Set[Tuple[int, int]] = set()
    if not isinstance(value, (list, tuple)):
        return result
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in item):
            result.add((int(item[0]), int(item[1])))
    return result


def _mapping(value: object) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _intent(row: Mapping[str, Any]) -> str:
    intent = str(row.get("mission_intent") or "")
    if intent in {MISSION_INTENT_EXPLORATION, MISSION_INTENT_DANGER_REOBSERVE}:
        return intent
    danger = _mapping(row.get("danger_reobserve"))
    return MISSION_INTENT_DANGER_REOBSERVE if danger else MISSION_INTENT_EXPLORATION


def _continuation(row: Mapping[str, Any]) -> Tuple[str, bool]:
    evidence = _mapping(row.get("continuation"))
    status = str(evidence.get("status") or CONTINUATION_UNKNOWN)
    if status not in {CONTINUATION_VIABLE, CONTINUATION_NON_VIABLE, CONTINUATION_UNKNOWN}:
        status = CONTINUATION_UNKNOWN
    # A NON_VIABLE result may eliminate an action only when the evaluator says
    # it exhausted that candidate's own productive-motion evidence.
    complete_nonviable = bool(evidence.get("complete")) and status == CONTINUATION_NON_VIABLE
    return status, complete_nonviable


def _action_valid(row: Mapping[str, Any], intent: str) -> Tuple[bool, Optional[str]]:
    if intent == MISSION_INTENT_DANGER_REOBSERVE:
        danger = _mapping(row.get("danger_reobserve"))
        if not bool(danger.get("available")):
            return False, "DANGER_EVIDENCE_UNAVAILABLE"
        if not str(danger.get("hypothesis_id") or ""):
            return False, "DANGER_HYPOTHESIS_ID_MISSING"
        if bool(danger.get("repeat_guard_closed")):
            return False, "DANGER_EPISODE_ALREADY_CLOSED"
        return True, None
    action = _mapping(row.get("exploration_action"))
    cells = _cells(action.get("intended_observation_cell_ids"))
    if not bool(action.get("intent_valid")):
        return False, "EXPLORATION_ACTION_INVALID"
    if not cells:
        return False, "EXPLORATION_INTENDED_CELLS_EMPTY"
    return True, None


def _preflight_legal(row: Mapping[str, Any]) -> bool:
    return bool(_mapping(row.get("formal_preflight")).get("legal"))


def _candidate_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    candidate_id = str(row.get("candidate_id") or "")
    intent = _intent(row)
    legal = _preflight_legal(row)
    action_valid, action_reason = _action_valid(row, intent)
    continuation, complete_nonviable = _continuation(row)
    reasons: List[str] = []
    if not candidate_id:
        reasons.append("CANDIDATE_ID_MISSING")
    if not legal:
        reasons.append("FORMAL_PREFLIGHT_NOT_LEGAL")
    if not action_valid:
        reasons.append(action_reason or "ACTION_SEMANTICS_INVALID")
    if complete_nonviable:
        reasons.append("CONTINUATION_NON_VIABLE_COMPLETE")
    retained = bool(candidate_id and legal and action_valid and not complete_nonviable)
    exploration = _mapping(row.get("exploration_action"))
    return {
        "candidate_id": candidate_id,
        "mission_intent": intent,
        "formal_executable": legal,
        "action_valid": action_valid,
        "hard_admission": "RETAINED" if retained else "EXCLUDED",
        "hard_admission_reasons": reasons,
        "intended_observation_cell_ids": [list(cell) for cell in sorted(_cells(exploration.get("intended_observation_cell_ids")))],
        "danger_reobserve": copy.deepcopy(_mapping(row.get("danger_reobserve"))),
        "continuation": {
            "status": continuation,
            "complete_nonviable": complete_nonviable,
            "provenance": copy.deepcopy(_mapping(row.get("continuation")).get("provenance") or {}),
        },
        "formal_path_cost_m": _finite(_mapping(row.get("formal_preflight")).get("path_cost_m")),
        "geometric_distance_m": _finite(_mapping(row.get("geometry")).get("geometric_distance_m")),
        "heading_change_rad": _finite(_mapping(row.get("geometry")).get("heading_change_rad")),
        "door_factor": _finite(_mapping(row.get("geometry")).get("door_factor")),
        "uncertainty": list(row.get("uncertainty") or []),
        "provenance": copy.deepcopy(_mapping(row.get("provenance"))),
        # These are deliberately carried but never read by the comparator.
        "telemetry_only": {
            "cheap_rank": row.get("cheap_rank"),
            "nbv_value": row.get("nbv_value"),
            "r1_recoverability": copy.deepcopy(row.get("r1_recoverability")),
            "predicted_new_count": row.get("predicted_new_count"),
            "predicted_occlusion_reveal_count": row.get("predicted_occlusion_reveal_count"),
        },
        "_source": copy.deepcopy(dict(row)),
    }


def build_formal_mission_comparison_set(
    epoch_id: str,
    candidates: Iterable[Mapping[str, Any]],
    *, cohort_complete: bool, global_invalidation: Optional[str] = None,
) -> Dict[str, Any]:
    """Build every record before comparison; never stop at a first legal row."""
    records = [_candidate_record(row) for row in candidates]
    ids = [row["candidate_id"] for row in records]
    duplicate_ids = sorted({candidate_id for candidate_id in ids if candidate_id and ids.count(candidate_id) > 1})
    complete = bool(cohort_complete) and not global_invalidation and not duplicate_ids
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch_id": str(epoch_id),
        "comparison_completeness": "COMPLETE" if complete else "INCOMPLETE",
        "global_invalidation": global_invalidation,
        "duplicate_candidate_ids": duplicate_ids,
        "all_candidates": records,
        "legal_executable_candidate_ids": [row["candidate_id"] for row in records if row["formal_executable"]],
        "mission_admissible_candidate_ids": [row["candidate_id"] for row in records if row["hard_admission"] == "RETAINED"],
        "telemetry_authority": "NONE",
        "selection_authority": "SHADOW_ONLY",
        "commands_published": False,
        "selection_performed": False,
    }


def _narrow_intent(rows: List[Dict[str, Any]], trace: List[str]) -> List[Dict[str, Any]]:
    dangers = [row for row in rows if row["mission_intent"] == MISSION_INTENT_DANGER_REOBSERVE]
    if dangers and len(dangers) < len(rows):
        trace.append("DANGER_PRIORITY")
        return dangers
    return rows


def _narrow_continuation(rows: List[Dict[str, Any]], trace: List[str]) -> List[Dict[str, Any]]:
    viable = [row for row in rows if row["continuation"]["status"] == CONTINUATION_VIABLE]
    if viable and len(viable) < len(rows):
        trace.append("CONTINUATION_VIABLE")
        return viable
    return rows


def _narrow_strict_superset(rows: List[Dict[str, Any]], trace: List[str]) -> List[Dict[str, Any]]:
    if not rows or any(row["mission_intent"] != MISSION_INTENT_EXPLORATION for row in rows):
        return rows
    cell_sets = {row["candidate_id"]: set(map(tuple, row["intended_observation_cell_ids"])) for row in rows}
    survivors = [
        row for row in rows
        if not any(cell_sets[other["candidate_id"]] > cell_sets[row["candidate_id"]] for other in rows if other is not row)
    ]
    if len(survivors) < len(rows):
        trace.append("STRICT_CELL_SUPERSET")
    return survivors


def _narrow_numeric(
    rows: List[Dict[str, Any]], field: str, *, highest: bool, reason: str, trace: List[str],
) -> List[Dict[str, Any]]:
    values = [_finite(row.get(field)) for row in rows]
    # Missing cost evidence is UNKNOWN, never silently treated as cheap.
    if not rows or any(value is None for value in values):
        return rows
    chosen = max(values) if highest else min(values)
    survivors = [row for row, value in zip(rows, values) if value == chosen]
    if len(survivors) < len(rows):
        trace.append(reason)
    return survivors


def select_v1_shadow_winner(comparison: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the approved V1 lexicographic rule to a complete cohort only."""
    if str(comparison.get("comparison_completeness")) != "COMPLETE":
        return {
            "status": "NO_SHADOW_WINNER", "reason": "FORMAL_COMPARISON_SET_INCOMPLETE",
            "winner_candidate_id": None, "win_reason": None, "win_reason_trace": [],
            "selection_authority": "SHADOW_ONLY", "commands_published": False,
        }
    rows = [dict(row) for row in comparison.get("all_candidates") or [] if row.get("hard_admission") == "RETAINED"]
    if not rows:
        return {
            "status": "NO_SHADOW_WINNER", "reason": "NO_MISSION_ADMISSIBLE_CANDIDATE",
            "winner_candidate_id": None, "win_reason": None, "win_reason_trace": [],
            "selection_authority": "SHADOW_ONLY", "commands_published": False,
        }
    trace: List[str] = []
    rows = _narrow_intent(rows, trace)
    rows = _narrow_continuation(rows, trace)
    rows = _narrow_strict_superset(rows, trace)
    rows = _narrow_numeric(rows, "formal_path_cost_m", highest=False, reason="LOWER_FORMAL_PATH_COST", trace=trace)
    rows = _narrow_numeric(rows, "geometric_distance_m", highest=False, reason="LOWER_GEOMETRIC_DISTANCE", trace=trace)
    # Heading is compared by absolute magnitude; keep the source value intact
    # in the record for audit provenance.
    for row in rows:
        heading = _finite(row.get("heading_change_rad"))
        row["_absolute_heading_change_rad"] = None if heading is None else abs(heading)
    rows = _narrow_numeric(rows, "_absolute_heading_change_rad", highest=False, reason="LOWER_ABS_HEADING_CHANGE", trace=trace)
    rows = _narrow_numeric(rows, "door_factor", highest=True, reason="HIGHER_DOOR_FACTOR", trace=trace)
    if len(rows) > 1:
        rows = sorted(rows, key=lambda row: str(row["candidate_id"]))
        trace.append("STABLE_CANDIDATE_ID")
    winner = rows[0]
    return {
        "status": "SHADOW_WINNER_AVAILABLE",
        "reason": "FORMAL_MISSION_COMPARISON_V1",
        "winner_candidate_id": winner["candidate_id"],
        "win_reason": trace[-1] if trace else "SOLE_MISSION_ADMISSIBLE_CANDIDATE",
        "win_reason_trace": trace or ["SOLE_MISSION_ADMISSIBLE_CANDIDATE"],
        "winner_mission_intent": winner["mission_intent"],
        "winner_continuation_status": winner["continuation"]["status"],
        "selection_authority": "SHADOW_ONLY",
        "commands_published": False,
    }


def evaluate_formal_mission_comparison_v1(
    epoch_id: str,
    candidates: Iterable[Mapping[str, Any]],
    *, cohort_complete: bool, global_invalidation: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience pure entry point used by tests and offline replay tools."""
    comparison = build_formal_mission_comparison_set(
        epoch_id, candidates, cohort_complete=cohort_complete, global_invalidation=global_invalidation,
    )
    return {
        "comparison": comparison,
        "shadow_selection": select_v1_shadow_winner(comparison),
        "production_first_legal_unchanged": True,
        "command_authority": False,
    }


def materialize_stage_b_shadow_records(
    stage_b_result: Mapping[str, Any], frozen_candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Adapt existing Stage-B output to V1 inputs without re-running a runner.

    This is deliberately conservative.  A Stage-B result that contradicts its
    captured production admission is not reinterpreted as a legal cohort: it
    becomes incomplete, because the frozen preflight context is not comparable
    to the real selection-time context.
    """
    evaluation = _mapping(stage_b_result.get("evaluation"))
    legal_set = _mapping(evaluation.get("legal_comparison_set"))
    continuation_rows = _mapping(_mapping(evaluation.get("candidate_continuation_evidence")).get("records_by_candidate_id"))
    results = {
        str(item.get("candidate_id") or ""): _mapping(item)
        for item in legal_set.get("all_preflight_results") or []
        if isinstance(item, Mapping)
    }
    events = [item for item in stage_b_result.get("production_events") or [] if isinstance(item, Mapping)]
    admission = next((item for item in events if item.get("event") == "PRODUCTION_ADMISSION"), {})
    production_winner = str(admission.get("selected_candidate_id") or "") or None
    production_result = results.get(production_winner or "")
    context_mismatch = bool(
        production_winner
        and (not production_result or str(production_result.get("legality")) != "LEGAL_NOW")
    )
    rows: List[Dict[str, Any]] = []
    for candidate in frozen_candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        preflight = results.get(candidate_id, {})
        opportunity = _mapping(candidate.get("opportunity"))
        danger = _mapping(candidate.get("danger"))
        opportunities = danger.get("opportunities") or []
        h1 = opportunities[0] if isinstance(opportunities, list) and opportunities and isinstance(opportunities[0], Mapping) else {}
        candidate_type = str(candidate.get("candidate_type") or candidate.get("target_priority_class") or "")
        is_danger = candidate_type == MISSION_INTENT_DANGER_REOBSERVE
        if candidate_type == "OCCLUSION":
            intended = opportunity.get("predicted_occlusion_reveal_cell_ids") or []
        else:
            intended = opportunity.get("predicted_new_cell_ids") or []
        rank_fields = _mapping(candidate.get("cheap_rank_components"))
        continuation = _mapping(continuation_rows.get(candidate_id))
        if not continuation:
            continuation = {
                "status": CONTINUATION_UNKNOWN,
                "complete": False,
                "reason": "PREFLIGHT_INCOMPLETE" if str(preflight.get("legality")) != "LEGAL_NOW" else "MISSING_FUTURE_STATE_CONTEXT",
                "provenance": "STAGE_B_CANDIDATE_CONTINUATION_RECORD_MISSING",
            }
        rows.append({
            "candidate_id": candidate_id,
            "mission_intent": MISSION_INTENT_DANGER_REOBSERVE if is_danger else MISSION_INTENT_EXPLORATION,
            "formal_preflight": {
                "legal": str(preflight.get("legality")) == "LEGAL_NOW",
                "path_cost_m": preflight.get("path_length_m"),
                "reason": preflight.get("failure_reason"),
            },
            "exploration_action": {
                "intent_valid": bool(intended) and not is_danger,
                "intended_observation_cell_ids": intended,
            },
            "danger_reobserve": {
                # Old Stage-B snapshots did not carry all G2 frozen fields;
                # do not promote an ID-only opportunity into a valid action.
                "available": bool(is_danger and h1.get("hypothesis_id") and h1.get("last_observed_stamp_sec") is not None and h1.get("position_xyz_m") is not None),
                "hypothesis_id": h1.get("hypothesis_id"),
                "repeat_guard_closed": False,
            },
            "continuation": copy.deepcopy(continuation),
            "geometry": {
                "geometric_distance_m": candidate.get("distance_m"),
                "heading_change_rad": candidate.get("heading_rad"),
                "door_factor": candidate.get("door_keepout_soft_factor"),
            },
            "cheap_rank": candidate.get("rank"),
            "nbv_value": None,
            "predicted_new_count": opportunity.get("predicted_new_count"),
            "predicted_occlusion_reveal_count": opportunity.get("predicted_occlusion_reveal_count"),
            "uncertainty": (
                (["STAGE_B_PREFLIGHT_CONTEXT_MISMATCH"] if context_mismatch else [])
                + ([str(continuation.get("reason"))] if str(continuation.get("status")) == CONTINUATION_UNKNOWN else [])
            ),
            "provenance": {
                "stage_b_epoch_id": _mapping(evaluation.get("room_search_decision_epoch")).get("epoch_id"),
                "stage_b_preflight": copy.deepcopy(preflight),
                "cheap_rank_components": copy.deepcopy(rank_fields),
            },
        })
    cohort_complete = bool(
        evaluation.get("status") == "STAGE_B_SHADOW_COMPLETE"
        and legal_set.get("comparison_completeness") == "COMPLETE"
        and not legal_set.get("decision_global_invalidation")
        and not context_mismatch
    )
    return {
        "epoch_id": str(_mapping(evaluation.get("room_search_decision_epoch")).get("epoch_id") or stage_b_result.get("epoch_id") or ""),
        "candidates": rows,
        "cohort_complete": cohort_complete,
        "global_invalidation": legal_set.get("decision_global_invalidation"),
        "production_winner_candidate_id": production_winner,
        "context_mismatch": context_mismatch,
        "context_mismatch_reason": (
            "STAGE_B_PREFLIGHT_DOES_NOT_REPRODUCE_CAPTURED_PRODUCTION_ADMISSION"
            if context_mismatch else None
        ),
    }


def replay_stage_b_shadow_result(
    stage_b_result: Mapping[str, Any], frozen_candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Run V1 over an existing Stage-B result, retaining all uncertainty."""
    materialized = materialize_stage_b_shadow_records(stage_b_result, frozen_candidates)
    evaluation = evaluate_formal_mission_comparison_v1(
        materialized["epoch_id"], materialized["candidates"],
        cohort_complete=bool(materialized["cohort_complete"]),
        global_invalidation=materialized["global_invalidation"],
    )
    winner = evaluation["shadow_selection"].get("winner_candidate_id")
    return {
        **evaluation,
        "production_winner_candidate_id": materialized["production_winner_candidate_id"],
        "same_as_production": (winner == materialized["production_winner_candidate_id"] if winner else None),
        "replay_context_mismatch": materialized["context_mismatch"],
        "replay_context_mismatch_reason": materialized["context_mismatch_reason"],
    }
