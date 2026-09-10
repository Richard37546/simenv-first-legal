#!/usr/bin/env python3
"""Behavior-neutral ROOM_SEARCH Stage-B snapshot/event capture.

Production may only call this module to freeze already-produced values and to
submit scalar audit events.  It has no planner, publisher, target writer, or
shadow-result reader.  All filesystem I/O runs on one bounded daemon writer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from frozen_decision_audit import (
    _candidate_evidence,
    _canonical_cells,
    _canonical_json_hash,
    _grid_payload,
    _json_safe,
    _seen_hash,
    _sha256_bytes,
    _sha256_path,
    _write_json,
)


SCHEMA_VERSION = "room_search_stage_b_shadow_v1"
SHADOW_FLAG = "ROOM_SEARCH_STAGE_B_SHADOW"
SHADOW_DIR_ENV = "ROOM_SEARCH_STAGE_B_SHADOW_DIR"
SHADOW_MODE_ENV = "ROOM_SEARCH_STAGE_B_MODE"
FROZEN_CAPTURE_FLAG = "ROOM_SEARCH_FROZEN_DECISION_CAPTURE"
MODE_OFF = "OFF"
MODE_ONE_SHOT = "ONE_SHOT"
MODE_MULTI_DECISION = "MULTI_DECISION"
SUPPORTED_MODES = {MODE_OFF, MODE_ONE_SHOT, MODE_MULTI_DECISION}
SNAPSHOT_JOB_CAPACITY = 4
EVENT_QUEUE_CAPACITY = 64
WRITER_LIFETIME_SEC = 120.0
RELEVANT_SOURCE_PATHS = (
    "scripts/local_subgoal_runner_mvp/navigation_state_machine.py",
    "scripts/local_subgoal_runner_mvp/room_search_v1.py",
    "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py",
    "scripts/local_subgoal_runner_mvp/local_grid_contract.py",
    "scripts/local_subgoal_runner_mvp/inflation_geometry.py",
    "scripts/local_subgoal_runner_mvp/frozen_decision_audit.py",
    "scripts/local_subgoal_runner_mvp/corridor_axis_evidence.py",
    "scripts/local_subgoal_runner_mvp/formal_mission_comparison_v1.py",
    "scripts/local_subgoal_runner_mvp/room_search_stage_a_contract.py",
    "scripts/local_subgoal_runner_mvp/room_search_stage_b_shadow_capture.py",
    "scripts/local_subgoal_runner_mvp/room_search_stage_b_shadow_sidecar.py",
)
REQUIRED_SNAPSHOT_FILES = (
    "manifest.json",
    "decision.json",
    "grid.json",
    "status.json",
    "seen.json",
    "candidates.json",
    "parameters.json",
    "source_manifest.json",
    "hashes.sha256",
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _event_metadata(status: str, **extra: Any) -> Dict[str, Any]:
    allowed = {
        "status": str(status),
        "wall_monotonic_ns": time.perf_counter_ns(),
    }
    allowed.update({key: _json_safe(value) for key, value in extra.items()})
    return allowed


class RoomSearchStageBShadowCapture:
    """Default-OFF bounded audit handoff; never reads shadow output or controls production."""

    def __init__(
        self,
        repo_root: Path,
        enabled: bool,
        output_root: Path,
        conflict: bool = False,
        mode: str = MODE_ONE_SHOT,
        snapshot_job_capacity: int = SNAPSHOT_JOB_CAPACITY,
        event_queue_capacity: int = EVENT_QUEUE_CAPACITY,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.mode = str(mode or MODE_ONE_SHOT).strip().upper()
        self.mode_valid = self.mode in SUPPORTED_MODES
        self.enabled = bool(enabled) and not bool(conflict) and self.mode_valid and self.mode != MODE_OFF
        self.requested = bool(enabled)
        self.conflict = bool(conflict)
        self.output_root = Path(output_root)
        self._job_queue_capacity = max(1, int(snapshot_job_capacity))
        self._event_queue_capacity = max(1, int(event_queue_capacity))
        self._job_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self._job_queue_capacity)
        self._event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self._event_queue_capacity)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        # Legacy fields remain for ONE_SHOT callers/tests; event routing never uses them.
        self._submitted = False
        self._epoch_id: Optional[str] = None
        self._run_id: Optional[str] = None
        self._decision_id: Optional[int] = None
        self._ready_path: Optional[Path] = None
        self._writer_error: Optional[str] = None
        self._event_drop_count = 0
        self._capture_drop_count = 0
        self._max_job_queue_depth = 0
        self._epochs: Dict[int, Dict[str, Any]] = {}
        self._ready_paths: Dict[int, Path] = {}
        self._pending_events: Dict[int, List[Dict[str, Any]]] = {}
        self._run_event_path: Optional[Path] = None
        self._production_finished = False
        self._source_bytes: Dict[str, bytes] = {}
        self._missing_sources: List[str] = []
        if self.enabled:
            for relative in RELEVANT_SOURCE_PATHS:
                path = self.repo_root / relative
                try:
                    if not path.is_file():
                        raise FileNotFoundError(str(path))
                    self._source_bytes[relative] = path.read_bytes()
                except OSError:
                    self._missing_sources.append(relative)

    @classmethod
    def from_environment(cls, repo_root: Path) -> "RoomSearchStageBShadowCapture":
        root = Path(repo_root).resolve()
        requested = _truthy(os.environ.get(SHADOW_FLAG, ""))
        conflict = requested and _truthy(os.environ.get(FROZEN_CAPTURE_FLAG, ""))
        configured = os.environ.get(SHADOW_DIR_ENV, "").strip()
        archive = os.environ.get("STATE_MACHINE_RUN_ARCHIVE_DIR", "").strip()
        output = Path(configured) if configured else (
            Path(archive) / "room_search_stage_b_shadow"
            if archive else root / "debug" / "room_search" / "stage_b_shadow"
        )
        mode = os.environ.get(SHADOW_MODE_ENV, MODE_ONE_SHOT).strip().upper() or MODE_ONE_SHOT
        return cls(root, requested, output, conflict=conflict, mode=mode)

    @property
    def writer_error(self) -> Optional[str]:
        return self._writer_error

    @property
    def ready_path(self) -> Optional[Path]:
        return self._ready_path

    @property
    def event_drop_count(self) -> int:
        return int(self._event_drop_count)

    @property
    def ready_paths(self) -> Dict[int, Path]:
        return dict(self._ready_paths)

    def _source_manifest(self) -> Dict[str, Any]:
        rows = [
            {
                "repo_relative_path": relative,
                "bundle_relative_path": f"sources/{relative}",
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
            }
            for relative, payload in sorted(self._source_bytes.items())
        ]
        state_rows = [[row["repo_relative_path"], row["sha256"], row["size_bytes"]] for row in rows]
        return {
            "schema_version": SCHEMA_VERSION,
            "source_files": rows,
            "missing_source_files": list(self._missing_sources),
            "relevant_source_state_sha256": _canonical_json_hash(state_rows),
            "source_recovery_mode": "EXACT_BUNDLE_SOURCE_COPIES",
            "python_executable": sys.executable,
        }

    def try_capture_epoch(
        self,
        *,
        run_id: str,
        decision_id: int,
        ros_timestamp_sec: float,
        pose_odom_xy_yaw: Sequence[float],
        pose_room_xy: Sequence[float],
        portal_context: Mapping[str, Any],
        planning_context: Mapping[str, Any],
        seen_cells: Iterable[Sequence[int]],
        ranked_candidates: Sequence[Mapping[str, Any]],
        parameters: Mapping[str, Any],
        runner_command_template: Sequence[str],
        breadcrumbs: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        start_ns = time.perf_counter_ns()
        if not self.requested:
            return _event_metadata("SHADOW_DISABLED", capture_duration_ns=time.perf_counter_ns() - start_ns)
        if self.conflict:
            return _event_metadata(
                "SHADOW_SKIPPED", reason="CONFLICTING_FROZEN_CAPTURE_FLAG",
                capture_duration_ns=time.perf_counter_ns() - start_ns,
            )
        if not self.mode_valid:
            return _event_metadata("SHADOW_SKIPPED", reason="INVALID_CAPTURE_MODE", mode=self.mode)
        if self.mode == MODE_OFF:
            return _event_metadata("SHADOW_DISABLED", mode=self.mode)
        if not self.enabled:
            return _event_metadata("SHADOW_SKIPPED", reason="CAPTURE_NOT_ENABLED", mode=self.mode)
        if not (
            planning_context.get("matched_pair_found")
            and planning_context.get("qualified")
            and planning_context.get("grid_msg") is not None
            and isinstance(planning_context.get("status_payload"), dict)
        ):
            return _event_metadata("SHADOW_SKIPPED", reason="EXACT_GRID_STATUS_NOT_QUALIFIED")
        if len(ranked_candidates) <= 1:
            return _event_metadata("SHADOW_SKIPPED", reason="RANKED_REPRESENTATIVES_LE_1")
        with self._lock:
            if self.mode == MODE_ONE_SHOT and self._submitted:
                self._submit_event_for_decision(
                    int(self._decision_id or decision_id),
                    {"event": "MISSION_DECISION_ADVANCED", "observed_decision_id": int(decision_id)},
                )
                return _event_metadata(
                    "SHADOW_SKIPPED", reason="ONE_SHOT_ALREADY_SUBMITTED", epoch_id=self._epoch_id,
                    decision_id=int(decision_id), mode=self.mode,
                )
            if int(decision_id) in self._epochs:
                return _event_metadata(
                    "SHADOW_SKIPPED", reason="EPOCH_ALREADY_CAPTURED_DECISION", decision_id=int(decision_id),
                    epoch_id=self._epochs[int(decision_id)]["epoch_id"], mode=self.mode,
                )

        try:
            canonical_seen = _canonical_cells(seen_cells)
            candidate_evidence = [
                _candidate_evidence(dict(candidate), rank, canonical_seen)
                for rank, candidate in enumerate(ranked_candidates, 1)
            ]
            source_manifest = self._source_manifest()
            parameters_payload = {
                "schema_version": SCHEMA_VERSION,
                "navigation_arguments": copy.deepcopy(dict(parameters)),
                "production_dry_preflight_command_template": list(runner_command_template),
                "execute_flag_present": "--execute" in runner_command_template,
            }
            planning_identity = {
                "matched_pair_found": True,
                "qualified": True,
                "qualification_errors": list(planning_context.get("qualification_errors") or []),
                "grid_header_stamp_sec": planning_context.get("grid_header_stamp_sec"),
                "grid_content_stamp": planning_context.get("grid_content_stamp"),
                "content_generation_id": planning_context.get("content_generation_id"),
                "grid_content_hash": planning_context.get("grid_content_hash"),
                "local_traversability_status": planning_context.get("local_traversability_status"),
                "producer_instance_id": planning_context.get("status_payload", {}).get("producer_instance_id"),
                "exact_pairing_provenance": "ExactGridStatusPairCache.matching_pair + qualified_for_navigation",
                "formal_epoch_eligibility": (
                    "ELIGIBLE" if planning_context.get("local_traversability_status") == "FREE_SUPPORTED"
                    else "UNQUALIFIED:" + str(planning_context.get("local_traversability_status") or "UNKNOWN")
                ),
            }
            seen_payload = {
                "schema_version": SCHEMA_VERSION,
                "cell_identity_format": "room_local_floor_xy_index",
                "frame": "portal_room_local",
                "coarse_resolution_m": 0.5,
                "origin_xy": [0.0, 0.0],
                "indexing": "floor(room_x/resolution),floor(room_y/resolution)",
                "cells": canonical_seen,
                "count": len(canonical_seen),
                "canonical_hash": _seen_hash(canonical_seen),
            }
            identity = {
                "run_id": str(run_id or f"pid-{os.getpid()}"),
                "decision_id": int(decision_id),
                "pose": [float(value) for value in pose_odom_xy_yaw],
                "grid": planning_identity,
                "seen_hash": seen_payload["canonical_hash"],
                "portal_context": copy.deepcopy(dict(portal_context)),
                "parameters_identity": _canonical_json_hash(parameters_payload),
                "source_state_sha256": source_manifest["relevant_source_state_sha256"],
                "candidates": candidate_evidence,
            }
            epoch_id = _canonical_json_hash(identity)
            frozen = {
                "schema_version": SCHEMA_VERSION,
                "run_id": identity["run_id"],
                "decision_id": int(decision_id),
                "epoch_id": epoch_id,
                "ros_timestamp_sec": float(ros_timestamp_sec),
                "wall_timestamp_sec": time.time(),
                "capture_monotonic_ns": start_ns,
                "pose_odom_xy_yaw": identity["pose"],
                "pose_room_xy": [float(value) for value in pose_room_xy],
                "portal_context": copy.deepcopy(dict(portal_context)),
                "grid": _grid_payload(planning_context["grid_msg"]),
                "status": copy.deepcopy(planning_context["status_payload"]),
                "planning_identity": planning_identity,
                "conflict_epoch_audit": (
                    {
                        "reason": "DECISION_EPOCH_UNQUALIFIED:CONFLICT_NEEDS_CAUTION",
                        "run_id": identity["run_id"],
                        "decision_id": int(decision_id),
                        "epoch_id": epoch_id,
                        "l3v_conflict_provenance": copy.deepcopy(
                            planning_context["status_payload"].get("conflict_provenance")
                        ),
                    }
                    if planning_context.get("local_traversability_status") == "CONFLICT_NEEDS_CAUTION" else None
                ),
                "seen": seen_payload,
                "candidates": candidate_evidence,
                "parameters": parameters_payload,
                "breadcrumbs": copy.deepcopy(list(breadcrumbs)),
                "source_manifest": source_manifest,
                "source_bytes": dict(self._source_bytes),
            }
            try:
                self._job_queue.put_nowait(frozen)
            except queue.Full:
                self._capture_drop_count += 1
                self._submit_run_event({
                    "event": "EPOCH_CAPTURE_DROPPED_BACKLOG", "run_id": identity["run_id"],
                    "decision_id": int(decision_id), "reason": "SNAPSHOT_JOB_QUEUE_FULL",
                    "job_queue_depth": self._job_queue.qsize(), "job_queue_capacity": self._job_queue_capacity,
                })
                return _event_metadata(
                    "SHADOW_SKIPPED", reason="SNAPSHOT_JOB_QUEUE_FULL", decision_id=int(decision_id),
                    job_queue_depth=self._job_queue.qsize(), job_queue_capacity=self._job_queue_capacity,
                    capture_duration_ns=time.perf_counter_ns() - start_ns,
                )
            with self._lock:
                self._submitted = True
                self._epoch_id = epoch_id
                self._run_id = identity["run_id"]
                self._decision_id = int(decision_id)
                self._epochs[int(decision_id)] = {
                    "run_id": identity["run_id"], "decision_id": int(decision_id), "epoch_id": epoch_id,
                    "capture_monotonic_ns": start_ns,
                }
                self._max_job_queue_depth = max(self._max_job_queue_depth, self._job_queue.qsize())
                self._run_event_path = self.output_root / identity["run_id"] / "stage_b2_run_events.jsonl"
                if self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(
                        target=self._writer_main, name="room-search-stage-b-shadow-writer", daemon=True,
                    )
                    self._thread.start()
            duration = time.perf_counter_ns() - start_ns
            self._submit_event_for_decision(int(decision_id), {
                "event": "EPOCH_CAPTURE_SUBMITTED",
                "capture_duration_ns": duration,
                "ros_timestamp_sec": float(ros_timestamp_sec),
                "job_queue_depth": self._job_queue.qsize(),
                "job_queue_capacity": self._job_queue_capacity,
            })
            return _event_metadata(
                "SHADOW_CAPTURE_SUBMITTED", epoch_id=epoch_id, decision_id=int(decision_id),
                mode=self.mode, capture_duration_ns=duration, job_queue_capacity=self._job_queue_capacity,
                event_queue_capacity=self._event_queue_capacity,
            )
        except Exception as exc:
            self._writer_error = f"CAPTURE_EXCEPTION:{type(exc).__name__}:{exc}"
            return _event_metadata(
                "SHADOW_FAILED_ISOLATED", reason="CAPTURE_EXCEPTION",
                exception_type=type(exc).__name__, capture_duration_ns=time.perf_counter_ns() - start_ns,
            )

    def _submit_run_event(self, payload: Mapping[str, Any]) -> None:
        """Best-effort bounded record for an event with no epoch bundle."""
        try:
            self._event_queue.put_nowait({"_run_event": True, **copy.deepcopy(dict(payload))})
        except queue.Full:
            self._event_drop_count += 1

    def _submit_event_for_decision(self, decision_id: Optional[int], payload: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.enabled or decision_id is None:
            return _event_metadata("SHADOW_SKIPPED", reason="NO_CAPTURE_FOR_THIS_DECISION", decision_id=decision_id)
        with self._lock:
            epoch = self._epochs.get(int(decision_id))
        if epoch is None:
            return _event_metadata("SHADOW_SKIPPED", reason="NO_CAPTURE_FOR_THIS_DECISION", decision_id=int(decision_id))
        event = {
            "schema_version": SCHEMA_VERSION,
            "run_id": epoch["run_id"],
            "decision_id": int(decision_id),
            "epoch_id": epoch["epoch_id"],
            "wall_monotonic_ns": time.perf_counter_ns(),
            **copy.deepcopy(dict(payload)),
        }
        try:
            self._event_queue.put_nowait(event)
            return _event_metadata("SHADOW_EVENT_SUBMITTED", epoch_id=epoch["epoch_id"], event=event.get("event"))
        except queue.Full:
            self._event_drop_count += 1
            return _event_metadata(
                "SHADOW_INCOMPLETE", reason="EVENT_QUEUE_FULL", epoch_id=epoch["epoch_id"],
                event_drop_count=self._event_drop_count,
            )

    def _decision_from_scalars(self, audit_scalars: Dict[str, Any]) -> Optional[int]:
        decision_id = audit_scalars.pop("decision_id", None)
        return int(decision_id) if decision_id is not None else None

    def record_production_admission(self, **audit_scalars: Any) -> Dict[str, Any]:
        return self._submit_event_for_decision(self._decision_from_scalars(audit_scalars), {"event": "PRODUCTION_ADMISSION", **audit_scalars})

    def record_production_nbv(self, **audit_scalars: Any) -> Dict[str, Any]:
        return self._submit_event_for_decision(self._decision_from_scalars(audit_scalars), {"event": "PRODUCTION_SINGLETON_NBV", **audit_scalars})

    def record_execution_handoff(self, **audit_scalars: Any) -> Dict[str, Any]:
        return self._submit_event_for_decision(self._decision_from_scalars(audit_scalars), {"event": "PRODUCTION_EXECUTION_HANDOFF", **audit_scalars})

    def record_production_completion(self, **audit_scalars: Any) -> Dict[str, Any]:
        return self._submit_event_for_decision(self._decision_from_scalars(audit_scalars), {"event": "PRODUCTION_COMPLETION", **audit_scalars})

    def record_execution_outcome(self, **audit_scalars: Any) -> Dict[str, Any]:
        return self._submit_event_for_decision(self._decision_from_scalars(audit_scalars), {"event": "PRODUCTION_EXECUTION_OUTCOME", **audit_scalars})

    def note_production_finish(self, **audit_scalars: Any) -> Dict[str, Any]:
        decision_id = self._decision_from_scalars(audit_scalars)
        self._production_finished = True
        result = self._submit_event_for_decision(decision_id, {"event": "PRODUCTION_FINISH", **audit_scalars})
        self._submit_run_event({"event": "PRODUCTION_FINISH", "run_id": self._run_id, "decision_id": decision_id})
        return result

    def wait_for_writer(self, timeout_sec: float = 5.0) -> Optional[Path]:
        """Offline-test helper; production must never call or wait on this."""
        thread = self._thread
        if thread is not None:
            deadline = time.monotonic() + max(0.0, float(timeout_sec))
            while self._ready_path is None and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            if self._ready_path is not None and thread.is_alive():
                thread.join(max(0.0, deadline - time.monotonic()))
        return self._ready_path

    def _writer_main(self) -> None:
        try:
            deadline = time.monotonic() + WRITER_LIFETIME_SEC
            while time.monotonic() < deadline:
                try:
                    frozen = self._job_queue.get(timeout=0.05)
                except queue.Empty:
                    frozen = None
                if frozen is not None:
                    ready = self._write_ready_bundle(frozen)
                    decision_id = int(frozen["decision_id"])
                    with self._lock:
                        self._ready_paths[decision_id] = ready
                        if self._ready_path is None:
                            self._ready_path = ready
                        pending = self._pending_events.pop(decision_id, [])
                    for event in pending:
                        self._append_event(ready, event)
                drained = False
                while True:
                    try:
                        event = self._event_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    if event.get("_run_event"):
                        self._append_run_event(event)
                        continue
                    decision_id = event.get("decision_id")
                    with self._lock:
                        ready = self._ready_paths.get(int(decision_id)) if decision_id is not None else None
                        if ready is None and decision_id is not None:
                            self._pending_events.setdefault(int(decision_id), []).append(event)
                    if ready is not None:
                        self._append_event(ready, event)
                if self._production_finished and self._job_queue.empty() and self._event_queue.empty() and not self._pending_events:
                    break
        except Exception as exc:
            self._writer_error = f"WRITER_EXCEPTION:{type(exc).__name__}:{exc}"
        finally:
            self._write_capture_counters()

    def _write_capture_counters(self) -> None:
        run_id = self._run_id
        if not run_id:
            return
        payload = {
            "schema_version": SCHEMA_VERSION,
            "authority": "AUDIT_ONLY",
            "total_epochs_seen": len(self._epochs),
            "captures_attempted": len(self._epochs) + self._capture_drop_count,
            "captures_succeeded": len(self._ready_paths),
            "captures_dropped": self._capture_drop_count,
            "maximum_backlog_depth": self._max_job_queue_depth,
            "event_drop_count": self._event_drop_count,
            "writer_error": self._writer_error,
        }
        try:
            _write_json(self.output_root / str(run_id) / "stage_b2_capture_counters.json", payload)
        except Exception as exc:
            self._writer_error = f"COUNTER_WRITE_EXCEPTION:{type(exc).__name__}:{exc}"

    def _append_run_event(self, event: Mapping[str, Any]) -> None:
        path = self._run_event_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(dict(event)), ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _append_event(ready: Path, event: Mapping[str, Any]) -> None:
        with (ready / "production_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(dict(event)), ensure_ascii=False, sort_keys=True) + "\n")

    def _write_ready_bundle(self, frozen: Mapping[str, Any]) -> Path:
        parent = self.output_root / str(frozen["run_id"])
        parent.mkdir(parents=True, exist_ok=True)
        name = f"epoch_{int(frozen['decision_id']):04d}_{str(frozen['epoch_id'])[:16]}"
        destination = parent / f"{name}.ready"
        temporary = Path(tempfile.mkdtemp(prefix=f"{name}.tmp-", dir=str(parent)))
        try:
            for relative, payload in sorted(frozen["source_bytes"].items()):
                output = temporary / "sources" / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(payload)
            decision = {
                "schema_version": SCHEMA_VERSION,
                "run_id": frozen["run_id"],
                "decision_id": frozen["decision_id"],
                "epoch_id": frozen["epoch_id"],
                "ros_timestamp_sec": frozen["ros_timestamp_sec"],
                "wall_timestamp_sec": frozen["wall_timestamp_sec"],
                "capture_monotonic_ns": frozen["capture_monotonic_ns"],
                "robot_pose_odom_xy_yaw": frozen["pose_odom_xy_yaw"],
                "robot_pose_room_xy": frozen["pose_room_xy"],
                "room_frame": "portal_room_local",
                "portal_context": frozen["portal_context"],
                "selected_candidate_id": None,
                "selected_rank": None,
                "production_authority": False,
                "preselection_snapshot": True,
                "conflict_epoch_audit": frozen.get("conflict_epoch_audit"),
            }
            candidates = {
                "schema_version": SCHEMA_VERSION,
                "ranked_candidate_count": len(frozen["candidates"]),
                "ranked_candidates": frozen["candidates"],
                "revisit_provenance": {
                    "coarse_resolution_m": 0.5,
                    "breadcrumbs_room_xy_yaw": frozen["breadcrumbs"],
                },
            }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "bundle_type": "ROOM_SEARCH_STAGE_B_PRESELECTION_EPOCH",
                "run_id": frozen["run_id"],
                "decision_id": frozen["decision_id"],
                "epoch_id": frozen["epoch_id"],
                "ranked_candidate_count": len(frozen["candidates"]),
                "grid_content_hash": frozen["planning_identity"]["grid_content_hash"],
                "grid_content_generation_id": frozen["planning_identity"]["content_generation_id"],
                "seen_canonical_hash": frozen["seen"]["canonical_hash"],
                "required_files": list(REQUIRED_SNAPSHOT_FILES),
                "capture_mode": self.mode,
                "one_shot": self.mode == MODE_ONE_SHOT,
                "capture_queue_capacity": self._job_queue_capacity,
                "event_queue_capacity": self._event_queue_capacity,
                "production_authority": False,
                "selection_authority": False,
                "command_authority": False,
                "completion_authority": False,
                "recoverability_authority": False,
                "fallback_authority": False,
            }
            _write_json(temporary / "decision.json", decision)
            _write_json(temporary / "grid.json", {
                "schema_version": SCHEMA_VERSION,
                "planning_identity": frozen["planning_identity"],
                "occupancy_grid": frozen["grid"],
            })
            _write_json(temporary / "status.json", frozen["status"])
            _write_json(temporary / "seen.json", frozen["seen"])
            _write_json(temporary / "candidates.json", candidates)
            _write_json(temporary / "parameters.json", frozen["parameters"])
            _write_json(temporary / "source_manifest.json", frozen["source_manifest"])
            _write_json(temporary / "manifest.json", manifest)
            rows = []
            for path in sorted(path for path in temporary.rglob("*") if path.is_file()):
                relative = str(path.relative_to(temporary))
                if relative != "hashes.sha256":
                    rows.append(f"{_sha256_path(path)}  {relative}")
            (temporary / "hashes.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
            if destination.exists():
                raise FileExistsError(str(destination))
            os.replace(str(temporary), str(destination))
            return destination
        except Exception:
            shutil.rmtree(str(temporary), ignore_errors=True)
            raise
