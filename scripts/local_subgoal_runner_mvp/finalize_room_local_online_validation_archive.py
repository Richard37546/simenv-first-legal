#!/usr/bin/env python3
"""Finalize the validation archive without claiming missing evidence exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict


SOURCE_FILES = (
    "scripts/local_subgoal_runner_mvp/navigation_state_machine.py",
    "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py",
    "scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh",
    "scripts/local_subgoal_runner_mvp/room_local_online_validation_observer.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo), check=True, text=True, stdout=subprocess.PIPE).stdout


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--exit-status", required=True, type=int)
    parser.add_argument("--launch-command", required=True)
    parser.add_argument("--validation-enabled", choices=("true", "false"), required=True)
    args = parser.parse_args()
    repo, archive = Path(args.repo_root).resolve(), Path(args.archive_dir).resolve()
    if not archive.is_dir():
        raise SystemExit("archive directory missing: %s" % archive)
    summary = read_json(archive / "state_machine_navigation_summary.json")
    runner = read_json(archive / "block_astar_dwa_mature_summary.json")
    telemetry = archive / "telemetry" / "command_odom_correlation.json"
    correlation = read_json(telemetry)
    result = str(summary.get("final_decision") or runner.get("final_decision") or "UNKNOWN")
    runner_control_reason = str(runner.get("validation_abort_reason") or "") or None
    if runner_control_reason is None and str(runner.get("final_decision") or "") == "ROOM_LOCAL_ONLINE_VALIDATION_ABORT":
        runner_control_reason = "ROOM_LOCAL_ONLINE_VALIDATION_ABORT"
    observer_control_reason = str(correlation.get("control_failure_reason") or "") or None
    control_failure_reason = runner_control_reason or observer_control_reason
    artifacts = {
        "state_machine_summary": archive / "state_machine_navigation_summary.json",
        "runner_summary": archive / "block_astar_dwa_mature_summary.json",
        "observer_log": archive / "room_local_validation_observer.log",
        "observer_telemetry": telemetry,
        "terminal_log": archive / "terminal_output.log",
    }
    artifact_status = {name: {"path": str(path.relative_to(archive)), "status": "PRESENT" if path.is_file() else "MISSING"} for name, path in artifacts.items()}
    critical_missing = [name for name in ("state_machine_summary", "runner_summary", "observer_telemetry") if artifact_status[name]["status"] != "PRESENT"]
    control_contract_status = "CONTROL_CONTRACT_FAILURE" if control_failure_reason else "CONTROL_CONTRACT_PASS"
    if critical_missing:
        online_evidence_status = "ONLINE_EVIDENCE_INSUFFICIENT"
    else:
        observed_status = str(correlation.get("online_evidence_status") or "")
        online_evidence_status = (
            observed_status
            if observed_status in {"ONLINE_EVIDENCE_COMPLETE", "ONLINE_EVIDENCE_INSUFFICIENT"}
            else "ONLINE_EVIDENCE_INSUFFICIENT"
        )
    evidence_insufficient_reason = None
    updates = correlation.get("evidence_status_updates") if isinstance(correlation.get("evidence_status_updates"), list) else []
    if updates:
        evidence_insufficient_reason = str((updates[0] if isinstance(updates[0], dict) else {}).get("reason") or "") or None
    if critical_missing and evidence_insufficient_reason is None:
        evidence_insufficient_reason = "MISSING_ARCHIVE_ARTIFACT:" + ",".join(critical_missing)
    source_identity = {
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "head": git(repo, "rev-parse", "HEAD").strip(),
        "git_status_short": git(repo, "status", "--short"),
        "files": {relative: sha256(repo / relative) for relative in SOURCE_FILES},
    }
    resolved_config = archive / "resolved_parameters.json"
    source_identity["relevant_config"] = {
        "path": "resolved_parameters.json",
        "status": "PRESENT" if resolved_config.is_file() else "MISSING",
        "sha256": sha256(resolved_config) if resolved_config.is_file() else None,
    }
    manifest = {
        "schema_version": "room_local_online_validation_archive_v1",
        "finalized": True,
        "finalized_wall_time_sec": time.time(),
        "run_id": args.run_id,
        "archive_dir": str(archive),
        "launch_command": args.launch_command,
        "activation_flags": {"room_local_guarded_online_validation": args.validation_enabled == "true"},
        "exit_status": args.exit_status,
        "result": result,
        "control_contract_status": control_contract_status,
        "control_failure_reason": control_failure_reason,
        "online_evidence_status": online_evidence_status,
        "evidence_insufficient_reason": evidence_insufficient_reason,
        "violation_status": "VIOLATION" if control_failure_reason else "NONE",
        "abort_reason": control_failure_reason,
        "artifacts": artifact_status,
        "critical_evidence_missing": critical_missing,
        "readiness": "PASS" if control_contract_status == "CONTROL_CONTRACT_PASS" and online_evidence_status == "ONLINE_EVIDENCE_COMPLETE" else "FAIL",
        "source_identity": source_identity,
    }
    (archive / "room_local_online_validation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
