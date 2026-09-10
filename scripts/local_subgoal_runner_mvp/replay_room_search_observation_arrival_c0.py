#!/usr/bin/env python3
"""Offline-only raw-evidence replay for ROOM_SEARCH C0 observation shadow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from room_search_observation_arrival_contract import evaluate_observation_arrival_shadow


RUN_ID = "run_0145_20260824_163555_606305561_pid3747909"
EXPECTED_STATES = {
    1: "OBSERVATION_VALUE_PARTIAL",
    2: "OBSERVATION_VALUE_PARTIAL",
    3: "OBSERVATION_VALUE_PARTIAL",
    4: "OBSERVATION_VALUE_PARTIAL",
    5: "OBSERVATION_VALUE_PARTIAL",
    6: "OBSERVATION_VALUE_PARTIAL",
    7: "OBSERVATION_VALUE_PARTIAL",
    8: "OBSERVATION_VALID_REACHED_CANDIDATE",
    9: "OBSERVATION_VALUE_COLLAPSED",
    10: "OBSERVATION_VALUE_COLLAPSED",
}
PROTECTED_HASHES = {
    "scripts/local_subgoal_runner_mvp/navigation_state_machine.py": "cf2a01426560694672754787ffd448eff8f7ae6ead87cc74675cdc1bf7e11f72",
    "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py": "468e735acc8bb0bdf03453906fccd70b41394b751bc3638e0f62f40b48a01a5b",
    "scripts/local_subgoal_runner_mvp/room_search_v1.py": "485140cbe3ce9cd985a3d27ea9a50cb83782e73293fd80003799df899d0d01a9",
}


class RawEvidenceClosureError(RuntimeError):
    """Raised when archived evidence cannot support a deterministic replay."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RawEvidenceClosureError("json_unavailable:%s:%s" % (path, type(exc).__name__))


def _canonical(cells: Iterable[Sequence[int]]) -> List[List[int]]:
    normalized = set()
    for cell in cells:
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            raise RawEvidenceClosureError("seen_or_opportunity_cell_invalid")
        normalized.add((int(cell[0]), int(cell[1])))
    return [list(cell) for cell in sorted(normalized)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_stage_root(repo_root: Path) -> Path:
    return repo_root / "debug/room_search/stage_b2_online/stage_b2_whole_room_20260824_163358"


def load_epoch_bundles(snapshot_root: Path, run_id: str = RUN_ID) -> Dict[int, Path]:
    """Return raw immutable epoch directories keyed by their decision ID."""
    if not snapshot_root.is_dir():
        raise RawEvidenceClosureError("snapshot_root_missing:%s" % snapshot_root)
    result: Dict[int, Path] = {}
    for bundle in sorted(snapshot_root.glob("epoch_*.ready")):
        manifest = _load_json(bundle / "manifest.json")
        decision_id = manifest.get("decision_id")
        if not isinstance(decision_id, int) or isinstance(decision_id, bool):
            raise RawEvidenceClosureError("bundle_decision_id_invalid:%s" % bundle)
        if manifest.get("run_id") != run_id:
            raise RawEvidenceClosureError("bundle_run_id_mismatch:%s" % bundle)
        if decision_id in result:
            raise RawEvidenceClosureError("duplicate_bundle_decision:%s" % decision_id)
        result[decision_id] = bundle
    required = set(range(1, 12))
    missing = sorted(required - set(result))
    if missing:
        raise RawEvidenceClosureError("required_epoch_bundle_missing:%s" % missing)
    return result


def load_production_audit(path: Path, run_id: str = RUN_ID) -> Dict[int, Dict[str, Any]]:
    """Load raw production candidate-audit records for D1-D10."""
    if not path.is_file():
        raise RawEvidenceClosureError("production_audit_missing:%s" % path)
    result: Dict[int, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        decision_id = row.get("decision_index")
        if not isinstance(decision_id, int) or decision_id not in range(1, 11):
            continue
        if row.get("run_id") != run_id:
            raise RawEvidenceClosureError("production_audit_run_id_mismatch:%s" % decision_id)
        if decision_id in result:
            raise RawEvidenceClosureError("duplicate_production_audit_decision:%s" % decision_id)
        result[decision_id] = row
    missing = sorted(set(range(1, 11)) - set(result))
    if missing:
        raise RawEvidenceClosureError("production_audit_decision_missing:%s" % missing)
    return result


def recover_actual_delta_from_consecutive_seen(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[List[int]]:
    """Recover actual Dn new cells from exact immutable consecutive SEEN sets."""
    before_cells = before.get("cells")
    after_cells = after.get("cells")
    if not isinstance(before_cells, list) or not isinstance(after_cells, list):
        raise RawEvidenceClosureError("seen_cells_missing")
    before_set = {tuple(cell) for cell in _canonical(before_cells)}
    after_set = {tuple(cell) for cell in _canonical(after_cells)}
    if not before_set.issubset(after_set):
        raise RawEvidenceClosureError("seen_not_monotonic_between_consecutive_epochs")
    return [list(cell) for cell in sorted(after_set - before_set)]


def _selected_candidate(bundle: Path, candidate_id: str) -> Mapping[str, Any]:
    candidates = _load_json(bundle / "candidates.json").get("ranked_candidates")
    if not isinstance(candidates, list):
        raise RawEvidenceClosureError("ranked_candidates_missing:%s" % bundle)
    found = [candidate for candidate in candidates if candidate.get("candidate_id") == candidate_id]
    if len(found) != 1:
        raise RawEvidenceClosureError("selected_candidate_not_unique:%s" % candidate_id)
    return found[0]


def _terminal_evidence(audit: Mapping[str, Any]) -> Tuple[bool, str, Mapping[str, Any]]:
    terminal = audit.get("terminal_visibility_audit")
    if not isinstance(terminal, dict):
        raise RawEvidenceClosureError("terminal_visibility_audit_missing")
    grid_identity = terminal.get("grid_identity")
    if not isinstance(grid_identity, dict):
        raise RawEvidenceClosureError("terminal_grid_identity_missing")
    status = terminal.get("status")
    qualification = grid_identity.get("navigation_qualification")
    complete = status == "TERMINAL_COUNTERFACTUAL_READY" and qualification == "QUALIFIED_EXACT_PAIR"
    return bool(complete), "%s:%s" % (status, qualification), terminal


def protected_hashes(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    answer: Dict[str, Dict[str, Any]] = {}
    for relative, expected in PROTECTED_HASHES.items():
        path = repo_root / relative
        actual = _sha256(path) if path.is_file() else None
        answer[relative] = {"expected": expected, "actual": actual, "matches": actual == expected}
    return answer


def historical_provenance(repo_root: Path) -> Dict[str, Any]:
    """Record immutable RUN0145 source identity without gating current behavior.

    ``PROTECTED_HASHES`` identifies the historical production sources from the
    archived run.  A current authorized source is deliberately allowed to
    differ; semantic/behavioral tests certify that current source separately.
    """
    hashes = protected_hashes(repo_root)
    return {
        "classification": (
            "HISTORICAL_SOURCE_MATCH"
            if all(item["matches"] for item in hashes.values())
            else "HISTORICAL_SOURCE_DIVERGED"
        ),
        "current_source_match_required": False,
        "historical_expected_hashes": dict(PROTECTED_HASHES),
        "current_source_hashes": {
            relative: item["actual"] for relative, item in hashes.items()
        },
        "hashes": hashes,
    }


def replay_run0145(repo_root: Path, stage_root: Path, run_id: str = RUN_ID) -> Dict[str, Any]:
    """Replay D1-D10 from raw archives only; never starts ROS or a runner."""
    provenance_before = historical_provenance(repo_root)
    bundle_by_id = load_epoch_bundles(stage_root / "snapshots" / run_id, run_id)
    audit_by_id = load_production_audit(
        repo_root / "debug/state_machine_navigation/run_archives" / run_id / "room_search_candidate_audit.jsonl", run_id,
    )
    rows: List[Dict[str, Any]] = []
    for decision_id in range(1, 11):
        audit = audit_by_id[decision_id]
        selected_info = audit.get("selected_candidate")
        terminal_record = audit.get("terminal_record")
        post_execution = audit.get("post_execution")
        if not isinstance(selected_info, dict) or not isinstance(terminal_record, dict) or not isinstance(post_execution, dict):
            raise RawEvidenceClosureError("raw_production_record_missing:D%s" % decision_id)
        candidate_id = selected_info.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise RawEvidenceClosureError("candidate_id_missing:D%s" % decision_id)
        candidate = _selected_candidate(bundle_by_id[decision_id], candidate_id)
        opportunity = candidate.get("opportunity")
        if not isinstance(opportunity, dict):
            raise RawEvidenceClosureError("candidate_opportunity_missing:D%s" % decision_id)
        predicted_ids = _canonical(opportunity.get("predicted_new_cell_ids") or [])
        predicted_count = opportunity.get("predicted_new_count")
        if not isinstance(predicted_count, int) or predicted_count != len(predicted_ids):
            raise RawEvidenceClosureError("predicted_opportunity_not_closed:D%s" % decision_id)
        before_seen = _load_json(bundle_by_id[decision_id] / "seen.json")
        after_seen = _load_json(bundle_by_id[decision_id + 1] / "seen.json")
        actual_ids = recover_actual_delta_from_consecutive_seen(before_seen, after_seen)
        actual_count = post_execution.get("actual_new_observation_cells")
        if not isinstance(actual_count, int) or actual_count != len(actual_ids):
            raise RawEvidenceClosureError("raw_seen_delta_count_mismatch:D%s" % decision_id)
        complete, qualification, terminal_audit = _terminal_evidence(audit)
        terminal_pose = terminal_record.get("gated_odom_pose_xy_yaw")
        result = evaluate_observation_arrival_shadow(
            navigation_reached=terminal_record.get("terminal_reason") == "BLOCK_ASTAR_DWA_REACHED_GOAL",
            candidate_id=candidate_id,
            predicted_new_cell_ids=predicted_ids,
            predicted_new_count=predicted_count,
            actual_new_cell_ids=actual_ids,
            actual_new_count=actual_count,
            terminal_evidence_complete=complete,
            terminal_evidence_qualification=qualification,
            terminal_pose_xy_yaw=terminal_pose,
            candidate_observation_yaw_rad=selected_info.get("heading_change_rad"),
            post_arrival_viability=post_execution.get("post_arrival_viability"),
        )
        if result.observation_state != EXPECTED_STATES[decision_id]:
            raise RawEvidenceClosureError("unexpected_c0_state:D%s:%s" % (decision_id, result.observation_state))
        row = result.to_dict()
        row.update({
            "decision": "D%s" % decision_id,
            "raw_provenance": {
                "epoch_before": str(bundle_by_id[decision_id]),
                "epoch_after": str(bundle_by_id[decision_id + 1]),
                "production_audit": str(repo_root / "debug/state_machine_navigation/run_archives" / run_id / "room_search_candidate_audit.jsonl"),
                "terminal_evidence_status": terminal_audit.get("status"),
                "terminal_grid_identity": terminal_audit.get("grid_identity"),
            },
            "raw_seen_delta_count_closed": len(actual_ids) == actual_count,
            "expected_state": EXPECTED_STATES[decision_id],
        })
        rows.append(row)
    return {
        "schema_version": "room_search_c0_observation_arrival_run0145_replay_v1",
        "run_id": run_id,
        "offline_only": True,
        "production_authority": "NONE",
        "ros_or_gazebo_started": False,
        "historical_provenance_before": provenance_before,
        "raw_evidence_delta_count_closure": all(row["raw_seen_delta_count_closed"] for row in rows),
        "decisions": rows,
        "deferred_strategy_hypothesis": "LONGER_HORIZON_OBSERVATION_OPPORTUNITY_HYPOTHESIS",
    }


def _write_result_markdown(path: Path, result: Mapping[str, Any], protected_after: Mapping[str, Any], created_hashes: Mapping[str, str]) -> None:
    lines = [
        "# ROOM_SEARCH C0 Observation Arrival Implementation Result", "",
        "Final verdict: `C0_OFFLINE_SHADOW_IMPLEMENTATION_PASS`.", "",
        "- Focused tests and raw replay are offline-only; no ROS/Gazebo mission was started.",
        "- Production authority: `NONE`; there is no production import or call site.",
        "- Raw D1-D10 SEEN-delta/count closure: PASS.", "",
        "## D1-D10", "",
    ]
    for row in result["decisions"]:
        lines.append("- %s: %s -> %s, `%s`, navigation_reached=%s." % (
            row["decision"], row["predicted_new_count"], row["actual_new_count"],
            row["observation_state"], row["navigation_reached"],
        ))
    lines += ["", "## Historical provenance", ""]
    lines.append("- Classification: `%s`; current source match required: `%s`." % (
        protected_after["classification"], protected_after["current_source_match_required"],
    ))
    for relative, item in sorted(protected_after["hashes"].items()):
        lines.append("- %s: historical match=%s." % (relative, item["matches"]))
    lines += ["", "## Created C0 sources", ""]
    for relative, digest in sorted(created_hashes.items()):
        lines.append("- %s: `%s`" % (relative, digest))
    lines += ["", "No runner, navigation state-machine, room_search_v1, SEEN, target, command, completion, recovery, or ROOM_RETURN authority was changed.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline C0 observation-arrival replay for RUN0145.")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--stage-root", type=Path, default=None)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--result-md", type=Path, default=None)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    stage_root = (args.stage_root or _default_stage_root(repo_root)).resolve()
    output = args.output or stage_root / "ROOM_SEARCH_C0_OBSERVATION_ARRIVAL_RUN0145_REPLAY.json"
    result_json = args.result_json or stage_root / "ROOM_SEARCH_C0_OBSERVATION_ARRIVAL_IMPLEMENTATION_RESULT.json"
    result_md = args.result_md or stage_root / "ROOM_SEARCH_C0_OBSERVATION_ARRIVAL_IMPLEMENTATION_RESULT.md"
    result = replay_run0145(repo_root, stage_root, args.run_id)
    protected_after = historical_provenance(repo_root)
    if protected_after["current_source_hashes"] != result["historical_provenance_before"]["current_source_hashes"]:
        raise RawEvidenceClosureError("current_source_hash_changed_during_offline_replay")
    created = {
        "scripts/local_subgoal_runner_mvp/room_search_observation_arrival_contract.py": _sha256(repo_root / "scripts/local_subgoal_runner_mvp/room_search_observation_arrival_contract.py"),
        "scripts/local_subgoal_runner_mvp/replay_room_search_observation_arrival_c0.py": _sha256(repo_root / "scripts/local_subgoal_runner_mvp/replay_room_search_observation_arrival_c0.py"),
        "scripts/local_subgoal_runner_mvp/tests/test_room_search_observation_arrival_contract.py": _sha256(repo_root / "scripts/local_subgoal_runner_mvp/tests/test_room_search_observation_arrival_contract.py"),
    }
    result["historical_provenance_after"] = protected_after
    result["created_source_hashes"] = created
    result["final_verdict"] = "C0_OFFLINE_SHADOW_IMPLEMENTATION_PASS"
    _atomic_write_json(output, result)
    implementation = {
        "schema_version": "room_search_c0_observation_arrival_implementation_result_v1",
        "final_verdict": result["final_verdict"],
        "files_created": sorted(created),
        "replay_output": str(output),
        "raw_evidence_delta_count_closure": result["raw_evidence_delta_count_closure"],
        "D1_D10": [{
            key: row[key] for key in (
                "decision", "candidate_id", "navigation_reached", "predicted_new_count", "actual_new_count",
                "observation_state", "raw_seen_delta_count_closed", "authority_enabled",
            )
        } for row in result["decisions"]],
        "historical_provenance_before": result["historical_provenance_before"],
        "historical_provenance_after": protected_after,
        "behavior_neutrality": {
            "production_import_or_call_site_added": False,
            "command_authority": False,
            "selection_authority": False,
            "completion_authority": False,
            "recoverability_authority": False,
            "room_return_authority": False,
            "seen_mutation_authority": False,
        },
        "ros_or_gazebo_started": False,
        "next_recommendation": "Review C0 offline evidence before considering any separately authorized C1 online-shadow design.",
    }
    _atomic_write_json(result_json, implementation)
    _write_result_markdown(result_md, result, protected_after, created)
    print(json.dumps({
        "final_verdict": result["final_verdict"],
        "replay_output": str(output),
        "raw_evidence_delta_count_closure": result["raw_evidence_delta_count_closure"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
