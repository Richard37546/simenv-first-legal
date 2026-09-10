#!/usr/bin/env python3
"""Offline-only ROOM_SEARCH Stage-A multi-candidate comparison contract.

This module has no ROS publisher and no production import site.  It consumes a
validated frozen-decision bundle, reuses ``frozen_decision_audit`` for the
existing runner's supplied-snapshot dry preflight, and emits advisory evidence.
It deliberately has no winner selector or execution authority.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import frozen_decision_audit
import formal_mission_comparison_v1


SCHEMA_VERSION = "room_search_stage_a_contract_v1"
STAGE_B_SCHEMA_VERSION = "room_search_stage_b_shadow_v1"
CONTINUATION_EVIDENCE_SCHEMA_VERSION = "candidate_continuation_shadow_v1"
OPPORTUNITY_NONEMPTY = "OPPORTUNITY_NONEMPTY"
OPPORTUNITY_EMPTY = "OPPORTUNITY_EMPTY"
OPPORTUNITY_UNKNOWN = "OPPORTUNITY_UNKNOWN"
GLOBAL_INVALID_REASONS = {
    "GRID_STATUS_CONTRACT_INVALID",
    "DECISION_GLOBAL_L3V_STATUS",
    "START_OUT_OF_GRID",
    "START_FOOTPRINT_CLEARANCE_INVALID",
}


def _continuation_unknown(reason: str, provenance: str) -> Dict[str, Any]:
    """Return an explicit non-authoritative C0 UNKNOWN record.

    The taxonomy lives at this adapter boundary because the frozen runner keeps
    its native detailed terminal reason.  No UNKNOWN is silently collapsed
    into a generic missing-record result.
    """
    return {
        "status": "CONTINUATION_UNKNOWN",
        "complete": False,
        "reason": str(reason),
        "provenance": str(provenance),
        "commands_published": False,
        "selection_authority": False,
    }


def _continuation_unknown_reason(native_reason: Any) -> str:
    """Map existing C0/preflight facts to stable, explainable UNKNOWN classes."""
    reason = str(native_reason or "UNKNOWN_REASON_UNAVAILABLE")
    if reason in {"FROZEN_GRID_STATUS_UNQUALIFIED", "DECISION_GLOBAL_L3V_STATUS"}:
        return "GRID_STATUS_UNQUALIFIED"
    if reason in {"CURRENT_PRODUCTIVE_SET_EMPTY", "S1_NEXT_PRODUCTIVE_EMPTY"}:
        return "INSUFFICIENT_LOCAL_MOTION_EVIDENCE"
    if "EPOCH" in reason or "IDENTITY" in reason or "STALE" in reason:
        return "STALE_OR_EPOCH_MISMATCH"
    if "PHASE2" in reason or "APPLICABLE" in reason or "LOCAL_CONTROL_MODE" in reason:
        return "EVALUATOR_NOT_APPLICABLE"
    if "PREFLIGHT" in reason:
        return "PREFLIGHT_INCOMPLETE"
    if "TARGET" in reason or "ACTION" in reason:
        return "UNSUPPORTED_ACTION_TYPE"
    if "CONTEXT" in reason or "S1_" in reason:
        return "MISSING_FUTURE_STATE_CONTEXT"
    return f"OTHER:{reason}"


def _shadow_continuation_args(runner_module: Any, runner_args: Any) -> Any:
    """Clone only the frozen evaluator profile; never alter live runner args."""
    args = copy.deepcopy(runner_args)
    offline_mode = getattr(runner_module, "ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_OFFLINE_FROZEN", None)
    if offline_mode is None:
        raise ValueError("EVALUATOR_NOT_APPLICABLE:PHASE2_OFFLINE_PROFILE_MISSING")
    args.execute = False
    args.room_local_phase2_productive_admission = offline_mode
    args.continuation_viability_shadow = True
    # This evidence path must remain unable to grant production local authority.
    args.continuation_local_selection_authority = getattr(
        runner_module, "CONTINUATION_LOCAL_SELECTION_AUTHORITY_DISABLED", "disabled",
    )
    return args


def candidate_level_continuation_evidence(
    *,
    runner_module: Any,
    runner_args: Any,
    grid_msg: Any,
    status: Mapping[str, Any],
    pose_odom_xy_yaw: Sequence[float],
    epoch_id: str,
    candidates: Sequence[Mapping[str, Any]],
    formal_legal_candidate_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Produce one pure C0 record per formally legal candidate in one epoch.

    The original supplied-snapshot preflight remains the sole source of formal
    legality.  The cloned Phase-2 profile exists only to expose the runner's
    already-existing productive-motion evidence to C0.
    """
    legal_ids = {str(candidate_id) for candidate_id in formal_legal_candidate_ids}
    evidence: Dict[str, Dict[str, Any]] = {}
    if not hasattr(runner_module, "FrozenRoomLocalEpoch") or not hasattr(
        runner_module, "evaluate_frozen_room_local_candidate"
    ) or not hasattr(runner_module, "evaluate_frozen_room_local_continuation_cohort"):
        for candidate_id in legal_ids:
            evidence[candidate_id] = _continuation_unknown(
                "EVALUATOR_NOT_APPLICABLE", "C0_FROZEN_CANDIDATE_EVALUATOR_UNAVAILABLE",
            )
        return evidence
    if str(status.get("local_traversability_status") or "") != "FREE_SUPPORTED":
        for candidate_id in legal_ids:
            evidence[candidate_id] = _continuation_unknown(
                "GRID_STATUS_UNQUALIFIED", "C0_SKIPPED_FROZEN_GRID_STATUS",
            )
        return evidence
    try:
        shadow_args = _shadow_continuation_args(runner_module, runner_args)
        frozen_epoch = runner_module.FrozenRoomLocalEpoch.from_live_inputs(
            epoch_id=str(epoch_id), pose_odom_xy_yaw=pose_odom_xy_yaw,
            grid_msg=grid_msg, status_payload=copy.deepcopy(dict(status)), args=shadow_args,
            previous_cmd=(0.0, 0.0), wall_heading_prior=None,
        )
    except Exception as exc:
        reason = _continuation_unknown_reason(str(exc))
        for candidate_id in legal_ids:
            evidence[candidate_id] = _continuation_unknown(reason, f"C0_EPOCH_BUILD_FAILED:{type(exc).__name__}")
        return evidence
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id not in legal_ids:
            continue
        try:
            frozen_candidate = runner_module.FrozenRoomLocalCandidate.from_mapping(copy.deepcopy(dict(candidate)))
            local = runner_module.evaluate_frozen_room_local_candidate(frozen_epoch, frozen_candidate)
            motions = local.get("motion_candidates") or []
            cohort = runner_module.evaluate_frozen_room_local_continuation_cohort(
                frozen_epoch, frozen_candidate, motions,
            )
            native_status = str(cohort.get("continuation_status") or "CONTINUATION_UNKNOWN")
            native_reason = str(cohort.get("reason") or "UNKNOWN_REASON_UNAVAILABLE")
            if native_status in {"CONTINUATION_VIABLE", "CONTINUATION_NON_VIABLE"}:
                evidence[candidate_id] = {
                    "status": native_status,
                    "complete": True,
                    "reason": native_reason,
                    "provenance": "C0_FROZEN_CANDIDATE_CONTINUATION",
                    "candidate_id": candidate_id,
                    "epoch_id": str(epoch_id),
                    "motion_candidate_count": len(motions),
                    "continuation": copy.deepcopy(cohort),
                    "commands_published": False,
                    "selection_authority": False,
                }
            else:
                row = _continuation_unknown(
                    _continuation_unknown_reason(native_reason),
                    f"C0_FROZEN_CANDIDATE_CONTINUATION:{native_reason}",
                )
                row.update({
                    "candidate_id": candidate_id, "epoch_id": str(epoch_id),
                    "motion_candidate_count": len(motions), "continuation": copy.deepcopy(cohort),
                })
                evidence[candidate_id] = row
        except ValueError as exc:
            evidence[candidate_id] = _continuation_unknown(
                _continuation_unknown_reason(str(exc)), f"C0_CANDIDATE_EVALUATION_FAILED:{exc}",
            )
        except Exception as exc:
            evidence[candidate_id] = _continuation_unknown(
                f"OTHER:{type(exc).__name__}", "C0_CANDIDATE_EVALUATION_EXCEPTION",
            )
    for candidate_id in legal_ids:
        evidence.setdefault(candidate_id, _continuation_unknown("PREFLIGHT_INCOMPLETE", "C0_LEGAL_CANDIDATE_MISSING"))
    return evidence


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_cells(cells: Iterable[Sequence[int]]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted({(int(cell[0]), int(cell[1])) for cell in cells}))


def _danger_identities(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    danger = candidate.get("danger")
    if not isinstance(danger, Mapping):
        return ()
    identities: List[str] = []
    for opportunity in danger.get("opportunities") or []:
        if isinstance(opportunity, Mapping):
            # ``hypothesis_id`` is the producer's real, short-window H1
            # identity.  Never substitute a JSON hash when this field exists.
            identity = opportunity.get("hypothesis_id") or opportunity.get("danger_id") or opportunity.get("id") or opportunity.get("identity")
            identities.append(str(identity) if identity is not None else _canonical_json_hash(opportunity))
        else:
            identities.append(str(opportunity))
    return tuple(sorted(set(identities)))


@dataclass(frozen=True)
class CandidateOpportunityClaim:
    candidate_id: str
    rank: int
    visible_cell_ids: Tuple[Tuple[int, int], ...]
    new_cell_ids: Tuple[Tuple[int, int], ...]
    occlusion_reveal_cell_ids: Tuple[Tuple[int, int], ...]
    danger_task_identities: Tuple[str, ...]
    status: str
    mission_value_semantics: str
    grid_hash: str
    seen_hash: str
    provenance: str

    @classmethod
    def from_frozen_candidate(
        cls,
        candidate: Mapping[str, Any],
        grid_hash: str,
        seen_hash: str,
    ) -> "CandidateOpportunityClaim":
        opportunity = candidate.get("opportunity")
        candidate_id = str(candidate.get("candidate_id") or "")
        rank = int(candidate.get("rank") or 0)
        if not isinstance(opportunity, Mapping):
            return cls(
                candidate_id, rank, (), (), (), _danger_identities(candidate),
                OPPORTUNITY_UNKNOWN, "MISSION_VALUE_UNKNOWN", grid_hash, seen_hash,
                "frozen candidates.json:opportunity missing",
            )
        visible = _canonical_cells(opportunity.get("predicted_visible_cell_ids") or [])
        new = _canonical_cells(opportunity.get("predicted_new_cell_ids") or [])
        occlusion = _canonical_cells(opportunity.get("predicted_occlusion_reveal_cell_ids") or [])
        danger = _danger_identities(candidate)
        nonempty = bool(visible or new or occlusion or danger)
        return cls(
            candidate_id=candidate_id,
            rank=rank,
            visible_cell_ids=visible,
            new_cell_ids=new,
            occlusion_reveal_cell_ids=occlusion,
            danger_task_identities=danger,
            status=OPPORTUNITY_NONEMPTY if nonempty else OPPORTUNITY_EMPTY,
            mission_value_semantics=(
                "DIRECT_OBSERVATION_OR_TASK_VALUE_EVIDENCED"
                if nonempty else "REPOSITION_OR_UNKNOWN_MISSION_VALUE"
            ),
            grid_hash=grid_hash,
            seen_hash=seen_hash,
            provenance="frozen candidates.json exact coarse-cell claims",
        )

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        for key in ("visible_cell_ids", "new_cell_ids", "occlusion_reveal_cell_ids"):
            row[key] = [list(cell) for cell in getattr(self, key)]
        row["danger_task_identities"] = list(self.danger_task_identities)
        return row


@dataclass(frozen=True)
class RoomSearchDecisionEpoch:
    epoch_id: str
    run_id: str
    decision_id: int
    robot_pose_odom_xy_yaw: Tuple[float, float, float]
    grid_generation: int
    grid_stamp_sec: float
    grid_hash: str
    grid_status: str
    seen_hash: str
    seen_cells: Tuple[Tuple[int, int], ...]
    portal_context_hash: str
    parameters_sha256: str
    source_state_sha256: str
    ranked_candidate_ids: Tuple[str, ...]
    opportunity_claims: Tuple[CandidateOpportunityClaim, ...]

    @classmethod
    def from_bundle(cls, bundle_path: Path) -> "RoomSearchDecisionEpoch":
        bundle = Path(bundle_path).resolve()
        decision = _read_json(bundle / "decision.json")
        grid = _read_json(bundle / "grid.json")["planning_identity"]
        seen = _read_json(bundle / "seen.json")
        source = _read_json(bundle / "source_manifest.json")
        candidates = _read_json(bundle / "candidates.json")["ranked_candidates"]
        parameters_sha256 = _sha256(bundle / "parameters.json")
        claims = tuple(
            CandidateOpportunityClaim.from_frozen_candidate(
                candidate, str(grid["grid_content_hash"]), str(seen["canonical_hash"]),
            )
            for candidate in candidates
        )
        identity_payload = {
            "run_id": decision["run_id"],
            "decision_id": decision["decision_id"],
            "pose": decision["robot_pose_odom_xy_yaw"],
            "grid": grid,
            "seen_hash": seen["canonical_hash"],
            "portal_context": decision.get("portal_context"),
            "parameters_sha256": parameters_sha256,
            "source_state_sha256": source["relevant_source_state_sha256"],
            "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
            "opportunity_claims": [claim.to_dict() for claim in claims],
        }
        return cls(
            epoch_id=str(decision.get("epoch_id") or _canonical_json_hash(identity_payload)),
            run_id=str(decision["run_id"]),
            decision_id=int(decision["decision_id"]),
            robot_pose_odom_xy_yaw=tuple(float(value) for value in decision["robot_pose_odom_xy_yaw"]),
            grid_generation=int(grid["content_generation_id"]),
            grid_stamp_sec=float(grid["grid_content_stamp"]),
            grid_hash=str(grid["grid_content_hash"]),
            grid_status=str(grid["local_traversability_status"]),
            seen_hash=str(seen["canonical_hash"]),
            seen_cells=_canonical_cells(seen["cells"]),
            portal_context_hash=_canonical_json_hash(decision.get("portal_context")),
            parameters_sha256=parameters_sha256,
            source_state_sha256=str(source["relevant_source_state_sha256"]),
            ranked_candidate_ids=tuple(str(candidate["candidate_id"]) for candidate in candidates),
            opportunity_claims=claims,
        )

    def identity(self) -> Dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "grid_generation": self.grid_generation,
            "grid_stamp_sec": self.grid_stamp_sec,
            "grid_hash": self.grid_hash,
            "grid_status": self.grid_status,
            "parameters_sha256": self.parameters_sha256,
            "source_state_sha256": self.source_state_sha256,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.identity(),
            "run_id": self.run_id,
            "decision_id": self.decision_id,
            "robot_pose_odom_xy_yaw": list(self.robot_pose_odom_xy_yaw),
            "seen_hash": self.seen_hash,
            "seen_cells": [list(cell) for cell in self.seen_cells],
            "portal_context_hash": self.portal_context_hash,
            "ranked_candidate_ids": list(self.ranked_candidate_ids),
            "opportunity_claims": [claim.to_dict() for claim in self.opportunity_claims],
        }


@dataclass(frozen=True)
class CandidatePreflightResult:
    rank: int
    candidate_id: str
    legality: str
    astar_result: str
    path_cell_count: Optional[int]
    path_length_m: Optional[float]
    dwa_safe_moving: Optional[int]
    dwa_admitted_count: Optional[int]
    rejection_reason: Optional[str]
    failure_reason: Optional[str]
    epoch_id: str
    grid_generation: int
    grid_hash: str
    source_state_sha256: str
    commands_published: bool

    @classmethod
    def from_raw(
        cls,
        raw: Mapping[str, Any],
        candidate: Mapping[str, Any],
        epoch: RoomSearchDecisionEpoch,
    ) -> "CandidatePreflightResult":
        return cls(
            rank=int(raw.get("rank") or candidate.get("rank") or 0),
            candidate_id=str(raw.get("candidate_id") or candidate.get("candidate_id") or ""),
            legality=str(raw.get("legality") or "UNKNOWN"),
            astar_result=str(raw.get("astar_result") or "UNKNOWN"),
            path_cell_count=(None if raw.get("path_cell_count") is None else int(raw["path_cell_count"])),
            path_length_m=(None if raw.get("path_length_m") is None else float(raw["path_length_m"])),
            dwa_safe_moving=(None if raw.get("dwa_safe_moving") is None else int(raw["dwa_safe_moving"])),
            dwa_admitted_count=(None if raw.get("dwa_admitted_count") is None else int(raw["dwa_admitted_count"])),
            rejection_reason=(None if raw.get("candidate_specific_rejection") is None else str(raw["candidate_specific_rejection"])),
            failure_reason=(None if raw.get("failure_reason") is None else str(raw["failure_reason"])),
            epoch_id=epoch.epoch_id,
            grid_generation=epoch.grid_generation,
            grid_hash=epoch.grid_hash,
            source_state_sha256=epoch.source_state_sha256,
            commands_published=bool(raw.get("commands_published", False)),
        )


@dataclass(frozen=True)
class LegalComparisonSet:
    epoch_id: str
    all_preflight_results: Tuple[CandidatePreflightResult, ...]
    legal_executable_candidate_ids: Tuple[str, ...]
    observation_valued_candidate_ids: Tuple[str, ...]
    empty_opportunity_legal_candidate_ids: Tuple[str, ...]
    comparison_completeness: str
    recoverability_status: str
    decision_global_invalidation: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "all_preflight_results": [asdict(result) for result in self.all_preflight_results],
            "LEGAL_EXECUTABLE_SET": list(self.legal_executable_candidate_ids),
            "observation_valued_comparison_candidates": list(self.observation_valued_candidate_ids),
            "empty_opportunity_legal_candidates": list(self.empty_opportunity_legal_candidate_ids),
            "comparison_completeness": self.comparison_completeness,
            "recoverability_status": self.recoverability_status,
            "decision_global_invalidation": self.decision_global_invalidation,
        }


def _is_global_invalid(result: CandidatePreflightResult) -> bool:
    return result.failure_reason in GLOBAL_INVALID_REASONS


def collect_same_epoch_preflights(
    epoch: RoomSearchDecisionEpoch,
    candidates: Iterable[Mapping[str, Any]],
    supplied_snapshot_preflight: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> Tuple[CandidatePreflightResult, ...]:
    """Collect every candidate-specific result; stop only on global invalidity."""
    results: List[CandidatePreflightResult] = []
    for candidate in candidates:
        raw = supplied_snapshot_preflight(copy.deepcopy(dict(candidate)))
        result = CandidatePreflightResult.from_raw(raw, candidate, epoch)
        if result.commands_published:
            raise ValueError("stage_a_dry_preflight_published_command")
        if result.candidate_id != str(candidate.get("candidate_id") or ""):
            raise ValueError("stage_a_preflight_candidate_identity_mismatch")
        supplied_epoch = raw.get("epoch_id")
        if supplied_epoch is not None and str(supplied_epoch) != epoch.epoch_id:
            raise ValueError("stage_a_preflight_epoch_identity_mismatch")
        supplied_grid = raw.get("grid_hash")
        if supplied_grid is not None and str(supplied_grid) != epoch.grid_hash:
            raise ValueError("stage_a_preflight_grid_identity_mismatch")
        results.append(result)
        if _is_global_invalid(result):
            break
    return tuple(results)


def exact_opportunity_relation(
    left: CandidateOpportunityClaim,
    right: CandidateOpportunityClaim,
) -> Dict[str, Any]:
    a, b = set(left.new_cell_ids), set(right.new_cell_ids)
    intersection, union = a & b, a | b
    if left.status == OPPORTUNITY_UNKNOWN or right.status == OPPORTUNITY_UNKNOWN:
        relation = "UNKNOWN"
    elif a == b:
        relation = "IDENTICAL" if a else "EMPTY"
    elif a > b:
        relation = "STRICT_SUPERSET"
    elif a < b:
        relation = "STRICT_SUBSET"
    elif not intersection:
        relation = "DISJOINT"
    else:
        relation = "PARTIAL_OVERLAP_WITH_UNIQUE_CELLS"
    return {
        "a_candidate_id": left.candidate_id,
        "b_candidate_id": right.candidate_id,
        "relation": relation,
        "a_size": len(a),
        "b_size": len(b),
        "intersection_size": len(intersection),
        "union_size": len(union),
        "intersection_cells": [list(cell) for cell in sorted(intersection)],
        "a_only_cells": [list(cell) for cell in sorted(a - b)],
        "b_only_cells": [list(cell) for cell in sorted(b - a)],
        "exact_equality": a == b,
    }


def opportunity_claims_exactly_identical(
    left: CandidateOpportunityClaim,
    right: CandidateOpportunityClaim,
) -> bool:
    """Compare the complete mission claim while allowing Grid identity to drift."""
    return (
        left.candidate_id == right.candidate_id
        and left.visible_cell_ids == right.visible_cell_ids
        and left.new_cell_ids == right.new_cell_ids
        and left.occlusion_reveal_cell_ids == right.occlusion_reveal_cell_ids
        and left.danger_task_identities == right.danger_task_identities
        and left.status == right.status
    )


def build_legal_comparison_set(
    epoch: RoomSearchDecisionEpoch,
    preflights: Sequence[CandidatePreflightResult],
) -> LegalComparisonSet:
    claims = {claim.candidate_id: claim for claim in epoch.opportunity_claims}
    global_invalid = next((result.failure_reason for result in preflights if _is_global_invalid(result)), None)
    legal = tuple(result.candidate_id for result in preflights if result.legality == "LEGAL_NOW")
    observation = tuple(
        candidate_id for candidate_id in legal
        if claims[candidate_id].status == OPPORTUNITY_NONEMPTY
    )
    empty = tuple(
        candidate_id for candidate_id in legal
        if claims[candidate_id].status == OPPORTUNITY_EMPTY
    )
    complete = (
        global_invalid is None
        and len(preflights) == len(epoch.ranked_candidate_ids)
        and all(result.legality in {"LEGAL_NOW", "ILLEGAL"} for result in preflights)
    )
    return LegalComparisonSet(
        epoch_id=epoch.epoch_id,
        all_preflight_results=tuple(preflights),
        legal_executable_candidate_ids=legal,
        observation_valued_candidate_ids=observation,
        empty_opportunity_legal_candidate_ids=empty,
        comparison_completeness="COMPLETE" if complete else "INCOMPLETE",
        recoverability_status="RECOVERABILITY_UNKNOWN",
        decision_global_invalidation=global_invalid,
    )


def validate_commit_context(
    epoch_identity: Mapping[str, Any],
    commit_identity: Mapping[str, Any],
    selected_epoch_claim: CandidateOpportunityClaim,
    selected_commit_claim: CandidateOpportunityClaim,
    selected_latest_legality: str,
    commit_global_status: str = "QUALIFIED",
) -> Dict[str, Any]:
    """Conservative Stage-A S_epoch/S_commit policy; never chooses a fallback."""
    same_identity = dict(epoch_identity) == dict(commit_identity)
    new_cell_relation = exact_opportunity_relation(selected_epoch_claim, selected_commit_claim)["relation"]
    claims_identical = opportunity_claims_exactly_identical(selected_epoch_claim, selected_commit_claim)
    base = {
        "same_identity": same_identity,
        "selected_latest_legality": str(selected_latest_legality),
        "selected_opportunity_relation": "IDENTICAL" if claims_identical else "CHANGED",
        "selected_new_cell_relation": new_cell_relation,
        "silent_demotion_performed": False,
        "fallback_candidate_id": None,
        "authority": "OFFLINE_CONTRACT_TEST_ONLY",
    }
    if commit_global_status != "QUALIFIED":
        return {**base, "result": "DISCARD_EPOCH_AND_REGENERATE", "completeness": "INCOMPLETE_GLOBAL_INVALID"}
    if selected_latest_legality != "LEGAL_NOW":
        return {**base, "result": "DISCARD_EPOCH_AND_REGENERATE", "completeness": "COMPLETE_REJECTION"}
    if same_identity and claims_identical:
        return {**base, "result": "CONTEXT_CONSISTENT", "completeness": "COMPLETE"}
    if not claims_identical:
        return {**base, "result": "DISCARD_EPOCH_AND_REGENERATE", "completeness": "COMPLETE_REJECTION"}
    return {**base, "result": "COMMIT_CONTEXT_VALID", "completeness": "COMPLETE"}


def _load_frozen_room_search(bundle: Path) -> Tuple[Any, Path]:
    path = bundle / "sources/scripts/local_subgoal_runner_mvp/room_search_v1.py"
    name = f"frozen_room_search_stage_a_{_sha256(path)[:16]}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("frozen_room_search_spec_unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module, path


def _cell_center_points(claim: CandidateOpportunityClaim, resolution_m: float) -> List[Tuple[float, float]]:
    return [((cell[0] + 0.5) * resolution_m, (cell[1] + 0.5) * resolution_m) for cell in claim.visible_cell_ids]


def current_nbv_multi_candidate_evidence(
    bundle: Path,
    epoch: RoomSearchDecisionEpoch,
    candidates: Sequence[Mapping[str, Any]],
    comparison: LegalComparisonSet,
) -> Dict[str, Any]:
    """Call the exact frozen ``score_candidates`` once; never call ``select_best``."""
    module, source_path = _load_frozen_room_search(bundle)
    decision = _read_json(bundle / "decision.json")
    seen_doc = _read_json(bundle / "seen.json")
    portal = decision["portal_context"]
    anchor = module.PortalAnchor(
        center_xy=tuple(float(value) for value in portal["center_xy"]),
        inward_normal=tuple(float(value) for value in portal["inward_normal"]),
        tangent=tuple(float(value) for value in portal["tangent"]),
        width_m=float(portal["width_m"]),
        door_return_anchor_xy_yaw=tuple(float(value) for value in portal["door_return_anchor_xy_yaw"]),
        door_return_anchor_stamp_sec=portal.get("door_return_anchor_stamp_sec"),
    )
    search = module.RoomSearchV2(
        anchor=anchor,
        observation=module.ObservationMemory(seen=set(epoch.seen_cells)),
    )
    claims = {claim.candidate_id: claim for claim in epoch.opportunity_claims}
    candidate_rows = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    preflights = {result.candidate_id: result for result in comparison.all_preflight_results}
    inputs: List[Dict[str, Any]] = []
    for candidate_id in comparison.observation_valued_candidate_ids:
        candidate = candidate_rows[candidate_id]
        claim = claims[candidate_id]
        preflight = preflights[candidate_id]
        inputs.append({
            "candidate_id": candidate_id,
            "cheap_rank": int(candidate["rank"]),
            "room_target_xy": list(candidate["target_room_xy"]),
            "visible_room_points": _cell_center_points(claim, float(seen_doc["coarse_resolution_m"])),
            "path_length_m": preflight.path_length_m,
            "heading_change_rad": float((candidate.get("cheap_rank_components") or {}).get("heading_change_rad") or 0.0),
        })
    state_before = {
        "seen": sorted(search.observation.seen),
        "gain_history": copy.deepcopy(search.gain_history),
        "peak_best_nbv_value": search.peak_best_nbv_value,
        "low_gain_streak": search.low_gain_streak,
    }
    scored = search.score_candidates(inputs)
    state_after = {
        "seen": sorted(search.observation.seen),
        "gain_history": copy.deepcopy(search.gain_history),
        "peak_best_nbv_value": search.peak_best_nbv_value,
        "low_gain_streak": search.low_gain_streak,
    }
    if state_before != state_after:
        raise ValueError("stage_a_nbv_scoring_mutated_room_search_state")
    evidence = []
    for row in sorted(scored, key=lambda item: int(item["cheap_rank"])):
        evidence.append({
            "candidate_id": row["candidate_id"],
            "cheap_rank": int(row["cheap_rank"]),
            "new_observable_cells": int(row["new_observable_cells"]),
            "path_length_m": float(row["path_length_m"]),
            "door_keepout_soft_factor": float(row["door_keepout_soft_factor"]),
            "nbv_value": float(row["nbv_value"]),
        })
    return {
        "authority": "ADVISORY_MULTI_CANDIDATE_EVIDENCE",
        "formula_implementation": "RoomSearchV2.score_candidates",
        "formula_source_bundle_path": str(source_path),
        "formula_source_sha256": _sha256(source_path),
        "visible_point_reconstruction": "CANONICAL_CELL_CENTER_EQUIVALENT_FOR_COARSE_NEW_COUNT",
        "selection_performed": False,
        "select_best_called": False,
        "state_mutated": False,
        "candidate_evidence": evidence,
    }


def run_frozen_stage_a(
    bundle_path: Path,
    *,
    historical_provenance_replay: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build advisory Stage-A evidence from a validated frozen epoch.

    A supplied replay is accepted only for an exact historical source replay:
    it must identify this bundle, run and source-state hash.  This keeps an
    immutable historical bundle distinct from current-source certification.
    """
    bundle = Path(bundle_path).resolve()
    validation = frozen_decision_audit.validate_snapshot(bundle)
    if validation.get("status") != "SNAPSHOT_VALID":
        return {"status": "NOT_READY", "reason": "SNAPSHOT_INVALID", "validation": validation}
    epoch = RoomSearchDecisionEpoch.from_bundle(bundle)
    candidates = _read_json(bundle / "candidates.json")["ranked_candidates"]
    replay = (
        dict(historical_provenance_replay)
        if historical_provenance_replay is not None
        else frozen_decision_audit.replay_snapshot(bundle)
    )
    if replay.get("status") != "DRY_PREFLIGHT_COMPLETE":
        return {"status": "NOT_READY", "reason": "SUPPLIED_SNAPSHOT_PREFLIGHT_UNAVAILABLE", "replay": replay}
    if historical_provenance_replay is not None and (
        Path(str(replay.get("bundle") or "")).resolve() != bundle
        or str(replay.get("run_id")) != epoch.run_id
        or str(replay.get("source_state_sha256")) != epoch.source_state_sha256
        or replay.get("commands_published") is not False
        or replay.get("selection_performed") is not False
    ):
        return {
            "status": "NOT_READY",
            "reason": "HISTORICAL_PROVENANCE_REPLAY_IDENTITY_MISMATCH",
            "replay": replay,
        }
    replay_by_id = {str(result["candidate_id"]): result for result in replay["results"]}

    def supplied(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return replay_by_id[str(candidate["candidate_id"])]

    preflights = collect_same_epoch_preflights(epoch, candidates, supplied)
    comparison = build_legal_comparison_set(epoch, preflights)
    claims = {claim.candidate_id: claim for claim in epoch.opportunity_claims}
    relations = []
    legal_ids = comparison.legal_executable_candidate_ids
    for index, left_id in enumerate(legal_ids):
        for right_id in legal_ids[index + 1:]:
            relations.append(exact_opportunity_relation(claims[left_id], claims[right_id]))
    nbv = current_nbv_multi_candidate_evidence(bundle, epoch, candidates, comparison)
    decision = _read_json(bundle / "decision.json")
    return {
        "status": "STAGE_A_OFFLINE_COMPARISON_COMPLETE" if comparison.comparison_completeness == "COMPLETE" else "NOT_READY",
        "schema_version": SCHEMA_VERSION,
        "bundle": str(bundle),
        "production_authority": False,
        "commands_published": False,
        "selection_performed": False,
        "winner_selector_implemented": False,
        "room_search_decision_epoch": epoch.to_dict(),
        "legal_comparison_set": comparison.to_dict(),
        "opportunity_exact_set_relations": relations,
        "cheap_rank": {
            "authority": "EVALUATION_SCHEDULING_ORDER + PROVENANCE",
            "ordered_candidate_ids": list(epoch.ranked_candidate_ids),
        },
        "nbv": nbv,
        "first_legal_baseline": {
            "authority": "FIRST_LEGAL_BASELINE_CANDIDATE",
            "candidate_id": decision["selected_candidate_id"],
            "rank": decision["selected_rank"],
            "production_selected_target_changed": False,
        },
        "comparison_semantics": {
            "empty_opportunity_legal_candidates_retained": True,
            "empty_opportunity_semantics": "REPOSITION_OR_UNKNOWN_MISSION_VALUE",
            "recoverability": "RECOVERABILITY_UNKNOWN",
            "final_mission_preference": "MISSION_PREFERENCE_INDETERMINATE",
        },
        "supplied_snapshot_equivalence": {
            "implementation": "frozen_decision_audit.replay_snapshot",
            "existing_runner_reused": True,
            "second_astar_or_dwa_implementation": False,
            "replay_grid_identity": replay["grid_identity"],
            "replay_source_state_sha256": replay["source_state_sha256"],
            "semantic_limit": replay["semantic_limit"],
        },
    }


def validate_stage_b_snapshot(bundle_path: Path) -> Dict[str, Any]:
    """Validate a pre-selection Stage-B snapshot without requiring a winner."""
    bundle = Path(bundle_path).resolve()
    required = {
        "manifest.json", "decision.json", "grid.json", "status.json", "seen.json",
        "candidates.json", "parameters.json", "source_manifest.json", "hashes.sha256",
    }
    errors: List[str] = []
    for name in sorted(required):
        if not (bundle / name).is_file():
            errors.append(f"required_file_missing:{name}")
    if errors:
        return {"status": "SNAPSHOT_INVALID", "bundle": str(bundle), "errors": errors}
    try:
        manifest = _read_json(bundle / "manifest.json")
        decision = _read_json(bundle / "decision.json")
        grid_doc = _read_json(bundle / "grid.json")
        status = _read_json(bundle / "status.json")
        seen = _read_json(bundle / "seen.json")
        candidates = _read_json(bundle / "candidates.json")
        parameters = _read_json(bundle / "parameters.json")
        source_manifest = _read_json(bundle / "source_manifest.json")
    except Exception as exc:
        return {
            "status": "SNAPSHOT_INVALID", "bundle": str(bundle),
            "errors": [f"json_read_failed:{type(exc).__name__}:{exc}"],
        }

    for name, payload in (
        ("manifest", manifest), ("decision", decision), ("grid", grid_doc),
        ("seen", seen), ("candidates", candidates), ("parameters", parameters),
        ("source_manifest", source_manifest),
    ):
        if payload.get("schema_version") != STAGE_B_SCHEMA_VERSION:
            errors.append(f"schema_version_mismatch:{name}")
    if manifest.get("bundle_type") != "ROOM_SEARCH_STAGE_B_PRESELECTION_EPOCH":
        errors.append("bundle_type_invalid")
    if not str(bundle.name).endswith(".ready"):
        errors.append("ready_suffix_missing")
    if decision.get("selected_candidate_id") is not None or decision.get("selected_rank") is not None:
        errors.append("preselection_snapshot_contains_selection")
    if decision.get("preselection_snapshot") is not True:
        errors.append("preselection_marker_missing")
    for key in (
        "production_authority", "selection_authority", "command_authority",
        "completion_authority", "recoverability_authority", "fallback_authority",
    ):
        if manifest.get(key) is not False:
            errors.append(f"authority_not_false:{key}")
    if parameters.get("execute_flag_present") is not False:
        errors.append("dry_preflight_template_contains_execute")

    expected_hashes: Dict[str, str] = {}
    try:
        for line in (bundle / "hashes.sha256").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                errors.append("hash_manifest_line_invalid")
                continue
            expected_hashes[parts[1]] = parts[0]
    except OSError as exc:
        errors.append(f"hash_manifest_read_failed:{type(exc).__name__}")
    for relative, expected in expected_hashes.items():
        path = (bundle / relative).resolve()
        try:
            path.relative_to(bundle)
        except ValueError:
            errors.append(f"hash_path_escapes_bundle:{relative}")
            continue
        if not path.is_file():
            errors.append(f"hashed_file_missing:{relative}")
        elif _sha256(path) != expected:
            errors.append(f"file_hash_mismatch:{relative}")
    actual_hashed = {
        str(path.relative_to(bundle)) for path in bundle.rglob("*")
        if path.is_file() and path.name not in {"hashes.sha256", "production_events.jsonl"}
    }
    expected_core = {name for name in expected_hashes if name != "production_events.jsonl"}
    if expected_core != actual_hashed:
        for relative in sorted(actual_hashed - expected_core):
            errors.append(f"file_missing_from_hash_manifest:{relative}")
        for relative in sorted(expected_core - actual_hashed):
            errors.append(f"hash_manifest_references_missing_file:{relative}")

    grid = grid_doc.get("occupancy_grid") or {}
    info = grid.get("info") or {}
    data = grid.get("data")
    if not (
        isinstance(data, list) and isinstance(info.get("width"), int)
        and isinstance(info.get("height"), int)
        and len(data) == int(info["width"]) * int(info["height"])
    ):
        errors.append("grid_payload_shape_invalid")
    else:
        try:
            computed_grid_hash = frozen_decision_audit._grid_content_hash(grid, status)
            identity = grid_doc.get("planning_identity") or {}
            if status.get("grid_content_hash") != computed_grid_hash:
                errors.append("grid_status_content_hash_mismatch")
            if identity.get("grid_content_hash") != computed_grid_hash:
                errors.append("planning_identity_grid_hash_mismatch")
            if manifest.get("grid_content_hash") != computed_grid_hash:
                errors.append("manifest_grid_content_hash_mismatch")
            if int(identity.get("content_generation_id")) != int(status.get("content_generation_id")):
                errors.append("grid_generation_mismatch")
            if float(grid["header"]["stamp_sec"]) != float(status.get("grid_content_stamp")):
                errors.append("grid_status_stamp_mismatch")
        except Exception as exc:
            errors.append(f"grid_identity_validation_failed:{type(exc).__name__}")

    canonical_seen = _canonical_cells(seen.get("cells") or [])
    if list(canonical_seen) != [tuple(cell) for cell in seen.get("cells") or []]:
        errors.append("seen_cells_not_canonical")
    if len(canonical_seen) != seen.get("count"):
        errors.append("seen_count_mismatch")
    if frozen_decision_audit._seen_hash(canonical_seen) != seen.get("canonical_hash"):
        errors.append("seen_hash_mismatch")
    ranked = candidates.get("ranked_candidates") or []
    if len(ranked) != candidates.get("ranked_candidate_count") or len(ranked) != manifest.get("ranked_candidate_count"):
        errors.append("ranked_candidate_count_mismatch")
    if len(ranked) <= 1:
        errors.append("ranked_candidate_count_not_multi")
    ids = [row.get("candidate_id") for row in ranked]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        errors.append("candidate_ids_invalid_or_nonunique")
    if [row.get("rank") for row in ranked] != list(range(1, len(ranked) + 1)):
        errors.append("candidate_rank_order_invalid")

    source_rows = source_manifest.get("source_files") or []
    if source_manifest.get("missing_source_files"):
        errors.append("source_manifest_has_missing_files")
    state_rows = []
    for row in source_rows:
        path = bundle / str(row.get("bundle_relative_path"))
        if not path.is_file():
            errors.append(f"source_copy_missing:{row.get('repo_relative_path')}")
            continue
        actual = _sha256(path)
        if actual != row.get("sha256"):
            errors.append(f"source_hash_mismatch:{row.get('repo_relative_path')}")
        state_rows.append([row.get("repo_relative_path"), row.get("sha256"), row.get("size_bytes")])
    if _canonical_json_hash(state_rows) != source_manifest.get("relevant_source_state_sha256"):
        errors.append("source_state_hash_mismatch")
    if decision.get("epoch_id") != manifest.get("epoch_id"):
        errors.append("epoch_identity_mismatch")
    if decision.get("run_id") != manifest.get("run_id") or decision.get("decision_id") != manifest.get("decision_id"):
        errors.append("decision_manifest_identity_mismatch")
    return {
        "status": "SNAPSHOT_VALID" if not errors else "SNAPSHOT_INVALID",
        "bundle": str(bundle), "run_id": manifest.get("run_id"),
        "decision_id": manifest.get("decision_id"), "epoch_id": manifest.get("epoch_id"),
        "ranked_candidate_count": len(ranked), "errors": errors,
    }


def run_stage_b_shadow_bundle(
    bundle_path: Path,
    production_telemetry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate all frozen candidates offline; never select or execute one."""
    bundle = Path(bundle_path).resolve()
    validation = validate_stage_b_snapshot(bundle)
    if validation.get("status") != "SNAPSHOT_VALID":
        return {"status": "NOT_READY", "reason": "SNAPSHOT_INVALID", "validation": validation}
    epoch = RoomSearchDecisionEpoch.from_bundle(bundle)
    candidates = _read_json(bundle / "candidates.json")["ranked_candidates"]
    parameters = _read_json(bundle / "parameters.json")
    grid_doc = _read_json(bundle / "grid.json")
    status = _read_json(bundle / "status.json")
    decision = _read_json(bundle / "decision.json")
    try:
        runner_module = frozen_decision_audit._load_frozen_runner(bundle)
        runner_args = frozen_decision_audit._runner_args(
            runner_module, parameters["production_dry_preflight_command_template"],
        )
        grid_msg = frozen_decision_audit._grid_message_from_payload(grid_doc["occupancy_grid"])
    except Exception as exc:
        return {
            "status": "NOT_READY", "reason": "FROZEN_EXECUTION_SOURCE_LOAD_FAILED",
            "exception": f"{type(exc).__name__}:{exc}", "validation": validation,
        }
    raw_results: List[Dict[str, Any]] = []
    candidate_timing: List[Dict[str, Any]] = []
    for candidate in candidates:
        started_ns = time.perf_counter_ns()
        raw = frozen_decision_audit._offline_candidate_preflight(
            runner_module, runner_args, grid_msg, status,
            decision["robot_pose_odom_xy_yaw"], copy.deepcopy(candidate),
        )
        elapsed_ns = time.perf_counter_ns() - started_ns
        raw.update({"epoch_id": epoch.epoch_id, "grid_hash": epoch.grid_hash})
        raw_results.append(raw)
        candidate_timing.append({
            "candidate_id": candidate.get("candidate_id"), "rank": candidate.get("rank"),
            "preflight_elapsed_ns": elapsed_ns,
        })
    by_id = {str(row["candidate_id"]): row for row in raw_results}
    preflights = collect_same_epoch_preflights(epoch, candidates, lambda row: by_id[str(row["candidate_id"])])
    comparison = build_legal_comparison_set(epoch, preflights)
    continuation_started_ns = time.perf_counter_ns()
    continuation_by_candidate = candidate_level_continuation_evidence(
        runner_module=runner_module,
        runner_args=runner_args,
        grid_msg=grid_msg,
        status=status,
        pose_odom_xy_yaw=decision["robot_pose_odom_xy_yaw"],
        epoch_id=epoch.epoch_id,
        candidates=candidates,
        formal_legal_candidate_ids=comparison.legal_executable_candidate_ids,
    )
    continuation_elapsed_ns = time.perf_counter_ns() - continuation_started_ns
    claims = {claim.candidate_id: claim for claim in epoch.opportunity_claims}
    relations_started_ns = time.perf_counter_ns()
    relations = [
        exact_opportunity_relation(claims[left], claims[right])
        for index, left in enumerate(comparison.legal_executable_candidate_ids)
        for right in comparison.legal_executable_candidate_ids[index + 1:]
    ]
    relations_elapsed_ns = time.perf_counter_ns() - relations_started_ns
    nbv_started_ns = time.perf_counter_ns()
    nbv = current_nbv_multi_candidate_evidence(bundle, epoch, candidates, comparison)
    nbv_elapsed_ns = time.perf_counter_ns() - nbv_started_ns
    epoch_unqualified_conflict = bool(
        comparison.decision_global_invalidation == "DECISION_GLOBAL_L3V_STATUS"
        and epoch.grid_status == "CONFLICT_NEEDS_CAUTION"
    )
    stage_b_evaluation = {
        "status": (
            "STAGE_B_SHADOW_COMPLETE" if comparison.comparison_completeness == "COMPLETE"
            else "DECISION_EPOCH_UNQUALIFIED" if epoch_unqualified_conflict
            else "NOT_READY"
        ),
        "reason": (
            "DECISION_EPOCH_UNQUALIFIED:CONFLICT_NEEDS_CAUTION"
            if epoch_unqualified_conflict else None
        ),
        "schema_version": STAGE_B_SCHEMA_VERSION,
        "bundle": str(bundle),
        "production_authority": False,
        "selection_authority": False,
        "command_authority": False,
        "completion_authority": False,
        "recoverability_authority": False,
        "fallback_authority": False,
        "commands_published": False,
        "selection_performed": False,
        "winner_selector_implemented": False,
        "optional_production_telemetry": copy.deepcopy(dict(production_telemetry or {})),
        "production_telemetry_used_for_comparison": False,
        "room_search_decision_epoch": epoch.to_dict(),
        "legal_comparison_set": comparison.to_dict(),
        "candidate_continuation_evidence": {
            "schema_version": CONTINUATION_EVIDENCE_SCHEMA_VERSION,
            "epoch_id": epoch.epoch_id,
            "formal_legal_candidate_ids": list(comparison.legal_executable_candidate_ids),
            "records_by_candidate_id": continuation_by_candidate,
            "authority": "SHADOW_ONLY",
            "commands_published": False,
            "selection_authority": False,
        },
        "opportunity_exact_set_relations": relations,
        "cheap_rank": {
            "authority": "EVALUATION_SCHEDULING_ORDER + PROVENANCE",
            "ordered_candidate_ids": list(epoch.ranked_candidate_ids),
        },
        "nbv": nbv,
        "timing": {
            "candidate_preflights": candidate_timing,
            "candidate_continuation_elapsed_ns": continuation_elapsed_ns,
            "opportunity_relations_elapsed_ns": relations_elapsed_ns,
            "multi_candidate_nbv_elapsed_ns": nbv_elapsed_ns,
        },
        "same_epoch_constraints": {
            "epoch_id": epoch.epoch_id,
            "grid_generation": epoch.grid_generation,
            "grid_hash": epoch.grid_hash,
            "mission_state_staleness_checked_by_sidecar": True,
        },
        "comparison_semantics": {
            "empty_opportunity_legal_candidates_retained": True,
            "recoverability": "RECOVERABILITY_UNKNOWN",
            "final_mission_preference": "MISSION_PREFERENCE_INDETERMINATE",
            "winner": None,
        },
    }
    # V1 consumes exactly the already-frozen, same-epoch evidence.  Its
    # winner remains advisory: neither this function nor its caller receives
    # a command, selection, or completion authority from it.
    formal_input = formal_mission_comparison_v1.materialize_stage_b_shadow_records(
        {"evaluation": stage_b_evaluation}, candidates,
    )
    formal_v1 = formal_mission_comparison_v1.evaluate_formal_mission_comparison_v1(
        formal_input["epoch_id"], formal_input["candidates"],
        cohort_complete=bool(formal_input["cohort_complete"]),
        global_invalidation=formal_input["global_invalidation"],
    )
    stage_b_evaluation["formal_mission_comparison_v1"] = formal_v1
    stage_b_evaluation["formal_mission_comparison_v1"]["materialization"] = {
        "context_mismatch": formal_input["context_mismatch"],
        "context_mismatch_reason": formal_input["context_mismatch_reason"],
    }
    return stage_b_evaluation


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage-b", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage_b_shadow_bundle(args.bundle) if args.stage_b else run_frozen_stage_a(args.bundle)
    if args.output is not None:
        _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("status") in {
        "STAGE_A_OFFLINE_COMPARISON_COMPLETE", "STAGE_B_SHADOW_COMPLETE", "DECISION_EPOCH_UNQUALIFIED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(_main())
