#!/usr/bin/env python3
"""Audit-only ROOM_SEARCH frozen-decision capture, validation, and replay.

The online hook copies already-existing decision state only.  It never calls a
planner, mutates ROOM_SEARCH state, publishes a command, or participates in
selection.  Offline replay loads the exact source copies stored in a bundle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = "room_search_frozen_decision_v1"
CAPTURE_FLAG = "ROOM_SEARCH_FROZEN_DECISION_CAPTURE"
CAPTURE_DIR_ENV = "ROOM_SEARCH_FROZEN_DECISION_CAPTURE_DIR"
COARSE_RESOLUTION_M = 0.5
REQUIRED_FILES = (
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
RELEVANT_SOURCE_PATHS = (
    "scripts/local_subgoal_runner_mvp/navigation_state_machine.py",
    "scripts/local_subgoal_runner_mvp/room_search_v1.py",
    "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py",
    "scripts/local_subgoal_runner_mvp/local_grid_contract.py",
    "scripts/local_subgoal_runner_mvp/inflation_geometry.py",
    "scripts/local_subgoal_runner_mvp/frozen_decision_audit.py",
    "scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh",
    "scripts/local_subgoal_runner_mvp/start_runtime_stack_tmux.sh",
)

_CAPTURE_LOCK = threading.Lock()
_CAPTURED_SESSION_KEYS = set()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _coarse_cell(point: Sequence[float], resolution_m: float = COARSE_RESOLUTION_M) -> Tuple[int, int]:
    return (
        int(math.floor(float(point[0]) / float(resolution_m))),
        int(math.floor(float(point[1]) / float(resolution_m))),
    )


def _canonical_cells(cells: Iterable[Sequence[int]]) -> List[List[int]]:
    return [[int(x), int(y)] for x, y in sorted({(int(cell[0]), int(cell[1])) for cell in cells})]


def _cells_from_points(points: Iterable[Sequence[float]]) -> List[List[int]]:
    return _canonical_cells(_coarse_cell(point) for point in points)


def _seen_hash(cells: Iterable[Sequence[int]]) -> str:
    canonical = _canonical_cells(cells)
    encoded = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(encoded)


def _grid_payload(grid_msg: Any) -> Dict[str, Any]:
    header = getattr(grid_msg, "header", None)
    info = getattr(grid_msg, "info", None)
    origin = getattr(info, "origin", None)
    position = getattr(origin, "position", None)
    orientation = getattr(origin, "orientation", None)
    stamp = getattr(header, "stamp", None)
    stamp_sec = float(stamp.to_sec()) if hasattr(stamp, "to_sec") else float(stamp)
    return {
        "header": {
            "frame_id": str(getattr(header, "frame_id", "")),
            "stamp_sec": stamp_sec,
            "seq": getattr(header, "seq", None),
        },
        "info": {
            "resolution": float(getattr(info, "resolution")),
            "width": int(getattr(info, "width")),
            "height": int(getattr(info, "height")),
            "origin": {
                "position": {
                    "x": float(getattr(position, "x")),
                    "y": float(getattr(position, "y")),
                    "z": float(getattr(position, "z", 0.0)),
                },
                "orientation": {
                    "x": float(getattr(orientation, "x", 0.0)),
                    "y": float(getattr(orientation, "y", 0.0)),
                    "z": float(getattr(orientation, "z", 0.0)),
                    "w": float(getattr(orientation, "w", 1.0)),
                },
            },
        },
        "data": [int(value) for value in getattr(grid_msg, "data", [])],
    }


def _candidate_evidence(candidate: Dict[str, Any], rank: int, seen_cells: Iterable[Sequence[int]]) -> Dict[str, Any]:
    seen = {(int(cell[0]), int(cell[1])) for cell in seen_cells}
    visible = _cells_from_points(candidate.get("visible_room_points") or [])
    visible_tuples = {(cell[0], cell[1]) for cell in visible}
    new_cells = _canonical_cells(visible_tuples - seen)
    occlusion = _cells_from_points(candidate.get("occlusion_reveal_room_points") or [])
    return {
        "candidate_id": candidate.get("_room_search_audit_candidate_id"),
        "rank": int(rank),
        "sector": candidate.get("sector"),
        "candidate_type": candidate.get("target_priority_class"),
        "target_odom_xy": list(candidate.get("target_xy_team_livox_odom") or []),
        "target_room_xy": list(candidate.get("room_target_xy") or []),
        "candidate_radius_m": candidate.get("candidate_radius_m"),
        "distance_m": candidate.get("cheap_geometric_distance_m"),
        "heading_rad": candidate.get("heading_change_rad"),
        "preferred_radial_band": candidate.get("preferred_radial_band"),
        "revisit": {
            "trajectory_revisit_coarse_location": candidate.get("trajectory_revisit_coarse_location"),
            "generic_trajectory_revisit_preference_applied": candidate.get(
                "generic_trajectory_revisit_preference_applied"
            ),
        },
        "danger": {
            "reobserve_supported": candidate.get("danger_reobserve_supported"),
            "reobserve_abs_bearing_rad": candidate.get("danger_reobserve_abs_bearing_rad"),
            "opportunities": copy.deepcopy(candidate.get("danger_reobserve_opportunities") or []),
        },
        "door_keepout_soft_factor": candidate.get("door_keepout_soft_factor"),
        "cheap_rank_components": {
            "action_class": candidate.get("target_priority_class"),
            "danger_reobserve_supported": candidate.get("danger_reobserve_supported"),
            "danger_reobserve_abs_bearing_rad": candidate.get("danger_reobserve_abs_bearing_rad"),
            "new_observable_cells": candidate.get("new_observable_cells"),
            "occlusion_reveal_cells": candidate.get("occlusion_reveal_cells"),
            "generic_trajectory_revisit_preference_applied": candidate.get(
                "generic_trajectory_revisit_preference_applied"
            ),
            "cheap_geometric_distance_m": candidate.get("cheap_geometric_distance_m"),
            "heading_change_rad": candidate.get("heading_change_rad"),
            "door_keepout_soft_factor": candidate.get("door_keepout_soft_factor"),
            "sector": candidate.get("sector"),
        },
        "opportunity": {
            "cell_identity_format": "room_local_floor_xy_index",
            "frame": "portal_room_local",
            "coarse_resolution_m": COARSE_RESOLUTION_M,
            "origin_xy": [0.0, 0.0],
            "indexing": "floor(room_x/resolution),floor(room_y/resolution)",
            "predicted_visible_cell_ids": visible,
            "predicted_new_cell_ids": new_cells,
            "predicted_occlusion_reveal_cell_ids": occlusion,
            "predicted_visible_count": len(visible),
            "predicted_new_count": len(new_cells),
            "predicted_occlusion_reveal_count": len(occlusion),
        },
    }


def _git_value(repo_root: Path, args: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=10,
        )
        return result.stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


class FrozenDecisionCapture:
    """One-shot, opt-in capture whose writer has no decision authority."""

    def __init__(self, repo_root: Path, enabled: bool, output_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.enabled = bool(enabled)
        self.output_root = Path(output_root)
        self._thread: Optional[threading.Thread] = None
        self._bundle_path: Optional[Path] = None
        self._error: Optional[str] = None
        self._source_bytes: Dict[str, bytes] = {}
        self._missing_sources: List[str] = []
        if self.enabled:
            # Bind the loaded runtime's source files before entering the
            # decision loop, not on the decision-critical capture path.
            for relative in RELEVANT_SOURCE_PATHS:
                path = self.repo_root / relative
                try:
                    if not path.is_file():
                        raise FileNotFoundError(str(path))
                    self._source_bytes[relative] = path.read_bytes()
                except OSError:
                    # Audit incompleteness must fail snapshot validation, not
                    # interrupt production ROOM_SEARCH authority.
                    self._missing_sources.append(relative)

    @classmethod
    def from_environment(cls, repo_root: Path) -> "FrozenDecisionCapture":
        root = Path(repo_root).resolve()
        enabled = _truthy(os.environ.get(CAPTURE_FLAG, ""))
        configured = os.environ.get(CAPTURE_DIR_ENV, "").strip()
        if configured:
            output = Path(configured)
        else:
            archive = os.environ.get("STATE_MACHINE_RUN_ARCHIVE_DIR", "").strip()
            output = (
                Path(archive) / "frozen_decisions"
                if archive else root / "debug" / "room_search" / "frozen_decisions"
            )
        return cls(root, enabled, output)

    @property
    def bundle_path(self) -> Optional[Path]:
        return self._bundle_path

    @property
    def error(self) -> Optional[str]:
        return self._error

    def wait(self, timeout_sec: Optional[float] = None) -> Optional[Path]:
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)
        return self._bundle_path

    def capture_once(
        self,
        *,
        run_id: str,
        decision_id: int,
        ros_timestamp_sec: float,
        pose_odom_xy_yaw: Sequence[float],
        pose_room_xy: Sequence[float],
        portal_context: Dict[str, Any],
        planning_context: Dict[str, Any],
        seen_cells: Iterable[Sequence[int]],
        ranked_candidates: Sequence[Dict[str, Any]],
        selected_candidate: Dict[str, Any],
        parameters: Dict[str, Any],
        runner_command_template: Sequence[str],
        breadcrumbs: Sequence[Sequence[float]],
        capture_trigger_reason: str,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"status": "CAPTURE_DISABLED"}
        if not (
            planning_context.get("matched_pair_found")
            and planning_context.get("qualified")
            and planning_context.get("grid_msg") is not None
            and isinstance(planning_context.get("status_payload"), dict)
        ):
            return {"status": "CAPTURE_TRIGGER_NOT_MET", "reason": "EXACT_GRID_STATUS_NOT_QUALIFIED"}
        if len(ranked_candidates) <= 1:
            return {"status": "CAPTURE_TRIGGER_NOT_MET", "reason": "RANKED_REPRESENTATIVES_LE_1"}
        selected_id = selected_candidate.get("_room_search_audit_candidate_id")
        selected_rank = selected_candidate.get("cheap_rank")
        if not selected_id or not isinstance(selected_rank, int):
            return {"status": "CAPTURE_TRIGGER_NOT_MET", "reason": "NORMAL_SELECTED_CANDIDATE_UNAVAILABLE"}

        session_key = str(run_id or f"pid-{os.getpid()}")
        with _CAPTURE_LOCK:
            if session_key in _CAPTURED_SESSION_KEYS:
                return {"status": "CAPTURE_ALREADY_COMPLETED_FOR_RUN"}
            _CAPTURED_SESSION_KEYS.add(session_key)

        canonical_seen = _canonical_cells(seen_cells)
        frozen = {
            "run_id": session_key,
            "decision_id": int(decision_id),
            "ros_timestamp_sec": float(ros_timestamp_sec),
            "wall_timestamp_sec": time.time(),
            "pose_odom_xy_yaw": [float(value) for value in pose_odom_xy_yaw],
            "pose_room_xy": [float(value) for value in pose_room_xy],
            "portal_context": copy.deepcopy(portal_context),
            "grid": _grid_payload(planning_context["grid_msg"]),
            "status": copy.deepcopy(planning_context["status_payload"]),
            "planning_identity": {
                "matched_pair_found": bool(planning_context.get("matched_pair_found")),
                "qualified": bool(planning_context.get("qualified")),
                "qualification_errors": list(planning_context.get("qualification_errors") or []),
                "grid_header_stamp_sec": planning_context.get("grid_header_stamp_sec"),
                "grid_content_stamp": planning_context.get("grid_content_stamp"),
                "content_generation_id": planning_context.get("content_generation_id"),
                "grid_content_hash": planning_context.get("grid_content_hash"),
                "local_traversability_status": planning_context.get("local_traversability_status"),
                "exact_pairing_provenance": "ExactGridStatusPairCache.matching_pair + qualified_for_navigation",
            },
            "seen": {
                "schema_version": SCHEMA_VERSION,
                "cell_identity_format": "room_local_floor_xy_index",
                "frame": "portal_room_local",
                "coarse_resolution_m": COARSE_RESOLUTION_M,
                "origin_xy": [0.0, 0.0],
                "indexing": "floor(room_x/resolution),floor(room_y/resolution)",
                "cells": canonical_seen,
                "count": len(canonical_seen),
                "canonical_hash": _seen_hash(canonical_seen),
            },
            "candidates": [
                _candidate_evidence(candidate, rank, canonical_seen)
                for rank, candidate in enumerate(ranked_candidates, 1)
            ],
            "selected_candidate_id": selected_id,
            "selected_rank": int(selected_rank),
            "selected_production_preflight": {
                "legal": bool(selected_candidate.get("legal")),
                "path_length_m": selected_candidate.get("path_length_m"),
                "runner_final_decision": (
                    selected_candidate.get("runner", {}).get("runner_final_decision")
                    if isinstance(selected_candidate.get("runner"), dict) else None
                ),
                "astar_path_exists": (
                    selected_candidate.get("runner", {}).get("last_dwa", {}).get("p_through_astar_path_exists")
                    if isinstance(selected_candidate.get("runner"), dict)
                    and isinstance(selected_candidate.get("runner", {}).get("last_dwa"), dict) else None
                ),
                "dwa_safe_moving_candidate_count": (
                    selected_candidate.get("runner", {}).get("last_dwa", {}).get("safe_moving_candidate_count")
                    if isinstance(selected_candidate.get("runner"), dict)
                    and isinstance(selected_candidate.get("runner", {}).get("last_dwa"), dict) else None
                ),
                "dwa_total_candidate_count": (
                    selected_candidate.get("runner", {}).get("last_dwa", {}).get("sample_count")
                    if isinstance(selected_candidate.get("runner"), dict)
                    and isinstance(selected_candidate.get("runner", {}).get("last_dwa"), dict) else None
                ),
                "grid_identity": {
                    key: selected_candidate.get("runner", {}).get("status_payload", {}).get(key)
                    for key in (
                        "grid_content_stamp", "content_generation_id", "grid_content_hash",
                        "local_traversability_status", "safe_for_navigation",
                    )
                } if isinstance(selected_candidate.get("runner"), dict)
                and isinstance(selected_candidate.get("runner", {}).get("status_payload"), dict) else {},
            },
            "parameters": copy.deepcopy(parameters),
            "runner_command_template": list(runner_command_template),
            "breadcrumbs": copy.deepcopy(list(breadcrumbs)),
            "capture_trigger_reason": str(capture_trigger_reason),
            "source_bytes": dict(self._source_bytes),
            "missing_sources": list(self._missing_sources),
        }
        self._thread = threading.Thread(
            target=self._write_frozen_bundle,
            args=(frozen,),
            name="room-search-frozen-decision-writer",
            daemon=False,
        )
        self._thread.start()
        return {
            "status": "CAPTURE_FROZEN_IN_MEMORY",
            "run_id": session_key,
            "decision_id": int(decision_id),
            "selected_candidate_id": selected_id,
            "selected_rank": int(selected_rank),
        }

    def _write_frozen_bundle(self, frozen: Dict[str, Any]) -> None:
        run_id = str(frozen["run_id"])
        decision_id = int(frozen["decision_id"])
        bundle_name = f"frozen_decision_{decision_id:04d}"
        parent = self.output_root / run_id
        parent.mkdir(parents=True, exist_ok=True)
        destination = parent / bundle_name
        temporary = Path(tempfile.mkdtemp(prefix=bundle_name + ".tmp-", dir=str(parent)))
        try:
            sources_dir = temporary / "sources"
            source_rows = []
            for relative, payload in sorted(frozen["source_bytes"].items()):
                output = sources_dir / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(payload)
                source_rows.append({
                    "repo_relative_path": relative,
                    "bundle_relative_path": str(output.relative_to(temporary)),
                    "sha256": _sha256_bytes(payload),
                    "size_bytes": len(payload),
                })

            source_state_rows = [
                [row["repo_relative_path"], row["sha256"], row["size_bytes"]]
                for row in source_rows
            ]
            global_git_status = _git_value(self.repo_root, ["status", "--porcelain=v1"])
            relevant_git_status = _git_value(
                self.repo_root,
                ["status", "--porcelain=v1", "--", *RELEVANT_SOURCE_PATHS],
            )
            dirty_diff = b""
            dirty_diff_available = False
            try:
                result = subprocess.run(
                    ["git", "diff", "--binary", "--", *RELEVANT_SOURCE_PATHS],
                    cwd=str(self.repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True, timeout=15,
                )
                dirty_diff = result.stdout
                dirty_diff_available = True
            except Exception:
                pass
            source_manifest = {
                "schema_version": SCHEMA_VERSION,
                "git_branch": _git_value(self.repo_root, ["branch", "--show-current"]),
                "git_head": _git_value(self.repo_root, ["rev-parse", "HEAD"]),
                "git_dirty": bool(global_git_status),
                "git_status_entry_count": len(global_git_status.splitlines()) if global_git_status is not None else None,
                "git_status_porcelain_sha256": (
                    _sha256_bytes(global_git_status.encode("utf-8")) if global_git_status is not None else None
                ),
                "relevant_git_status_porcelain": relevant_git_status,
                "relevant_dirty_diff_sha256": _sha256_bytes(dirty_diff),
                "relevant_dirty_diff_available": dirty_diff_available,
                "relevant_source_state_sha256": _canonical_json_hash(source_state_rows),
                "source_files": source_rows,
                "missing_source_files": list(frozen["missing_sources"]),
                "source_recovery_mode": "EXACT_BUNDLE_SOURCE_COPIES",
                "runtime": {
                    "python_executable": sys.executable,
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "ros_distro": os.environ.get("ROS_DISTRO"),
                    "ros_version": os.environ.get("ROS_VERSION"),
                },
            }

            parameters_payload = {
                "schema_version": SCHEMA_VERSION,
                "navigation_arguments": frozen["parameters"],
                "production_dry_preflight_command_template": frozen["runner_command_template"],
                "execute_flag_present": "--execute" in frozen["runner_command_template"],
            }
            decision_payload = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "decision_id": decision_id,
                "ros_timestamp_sec": frozen["ros_timestamp_sec"],
                "wall_timestamp_sec": frozen["wall_timestamp_sec"],
                "robot_pose_odom_xy_yaw": frozen["pose_odom_xy_yaw"],
                "robot_pose_room_xy": frozen["pose_room_xy"],
                "room_frame": "portal_room_local",
                "portal_context": frozen["portal_context"],
                "selected_candidate_id": frozen["selected_candidate_id"],
                "selected_rank": frozen["selected_rank"],
                "selected_production_preflight": frozen["selected_production_preflight"],
                "ranked_candidate_count": len(frozen["candidates"]),
                "capture_trigger_reason": frozen["capture_trigger_reason"],
                "capture_write_mode": "ONE_SHOT_BACKGROUND_WRITE_AFTER_IN_MEMORY_FREEZE",
                "production_authority": False,
            }
            candidates_payload = {
                "schema_version": SCHEMA_VERSION,
                "ranked_candidate_count": len(frozen["candidates"]),
                "ranked_candidates": frozen["candidates"],
                "revisit_provenance": {
                    "coarse_resolution_m": COARSE_RESOLUTION_M,
                    "breadcrumbs_room_xy_yaw": frozen["breadcrumbs"],
                },
            }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "bundle_type": "ROOM_SEARCH_FROZEN_DECISION",
                "run_id": run_id,
                "decision_id": decision_id,
                "selected_candidate_id": frozen["selected_candidate_id"],
                "selected_rank": frozen["selected_rank"],
                "ranked_candidate_count": len(frozen["candidates"]),
                "grid_content_hash": frozen["planning_identity"]["grid_content_hash"],
                "grid_content_generation_id": frozen["planning_identity"]["content_generation_id"],
                "grid_content_stamp": frozen["planning_identity"]["grid_content_stamp"],
                "seen_canonical_hash": frozen["seen"]["canonical_hash"],
                "parameters_sha256": None,
                "source_state_sha256": source_manifest["relevant_source_state_sha256"],
                "required_files": list(REQUIRED_FILES),
                "online_capture_performed_preflight_for_lower_ranks": False,
                "online_capture_published_commands": False,
                "audit_has_production_authority": False,
            }

            _write_json(temporary / "decision.json", decision_payload)
            _write_json(temporary / "grid.json", {
                "schema_version": SCHEMA_VERSION,
                "planning_identity": frozen["planning_identity"],
                "occupancy_grid": frozen["grid"],
            })
            _write_json(temporary / "status.json", frozen["status"])
            _write_json(temporary / "seen.json", frozen["seen"])
            _write_json(temporary / "candidates.json", candidates_payload)
            _write_json(temporary / "parameters.json", parameters_payload)
            _write_json(temporary / "source_manifest.json", source_manifest)
            manifest["parameters_sha256"] = _sha256_path(temporary / "parameters.json")
            _write_json(temporary / "manifest.json", manifest)

            hash_rows = []
            for path in sorted(path for path in temporary.rglob("*") if path.is_file()):
                relative = str(path.relative_to(temporary))
                if relative == "hashes.sha256":
                    continue
                hash_rows.append(f"{_sha256_path(path)}  {relative}")
            (temporary / "hashes.sha256").write_text("\n".join(hash_rows) + "\n", encoding="utf-8")
            if destination.exists():
                raise FileExistsError(str(destination))
            os.replace(str(temporary), str(destination))
            self._bundle_path = destination
        except Exception as exc:
            self._error = f"{type(exc).__name__}:{exc}"
            shutil.rmtree(str(temporary), ignore_errors=True)


class _Stamp:
    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)

    def to_sec(self) -> float:
        return self.seconds


def _grid_message_from_payload(payload: Dict[str, Any]) -> Any:
    header = payload["header"]
    info = payload["info"]
    origin = info["origin"]
    position = origin["position"]
    orientation = origin["orientation"]
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=header["frame_id"], stamp=_Stamp(header["stamp_sec"]), seq=header.get("seq")
        ),
        info=SimpleNamespace(
            resolution=float(info["resolution"]), width=int(info["width"]), height=int(info["height"]),
            origin=SimpleNamespace(
                position=SimpleNamespace(**position), orientation=SimpleNamespace(**orientation)
            ),
        ),
        data=list(payload["data"]),
    )


def _wire_float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _grid_content_hash(grid_payload: Dict[str, Any], status: Dict[str, Any]) -> str:
    info = grid_payload["info"]
    canonical = {
        "producer_instance_id": status["producer_instance_id"],
        "content_generation_id": status["content_generation_id"],
        "grid_content_stamp": status["grid_content_stamp"],
        "frame_id": grid_payload["header"]["frame_id"],
        "origin": [info["origin"]["position"]["x"], info["origin"]["position"]["y"]],
        "resolution": _wire_float32(info["resolution"]),
        "width": info["width"],
        "height": info["height"],
        "data": list(grid_payload["data"]),
    }
    return _canonical_json_hash(canonical)


def validate_snapshot(bundle_path: Path) -> Dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    errors: List[str] = []
    for name in REQUIRED_FILES:
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
        source_manifest = _read_json(bundle / "source_manifest.json")
        parameters = _read_json(bundle / "parameters.json")
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
        if payload.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"schema_version_mismatch:{name}")

    expected_hashes: Dict[str, str] = {}
    for line in (bundle / "hashes.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            errors.append("hash_manifest_line_invalid")
            continue
        expected_hashes[parts[1]] = parts[0]
    for relative, expected in expected_hashes.items():
        path = (bundle / relative).resolve()
        try:
            path.relative_to(bundle)
        except ValueError:
            errors.append(f"hash_path_escapes_bundle:{relative}")
            continue
        if not path.is_file():
            errors.append(f"hashed_file_missing:{relative}")
        elif _sha256_path(path) != expected:
            errors.append(f"file_hash_mismatch:{relative}")
    actual_hashed_files = {
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if (
            path.is_file()
            and path.name != "hashes.sha256"
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    }
    if set(expected_hashes) != actual_hashed_files:
        for relative in sorted(actual_hashed_files - set(expected_hashes)):
            errors.append(f"file_missing_from_hash_manifest:{relative}")
        for relative in sorted(set(expected_hashes) - actual_hashed_files):
            errors.append(f"hash_manifest_references_missing_file:{relative}")

    grid = grid_doc.get("occupancy_grid") or {}
    info = grid.get("info") or {}
    data = grid.get("data")
    if not (
        isinstance(data, list)
        and isinstance(info.get("width"), int)
        and isinstance(info.get("height"), int)
        and len(data) == int(info["width"]) * int(info["height"])
    ):
        errors.append("grid_payload_shape_invalid")
    else:
        try:
            computed_grid_hash = _grid_content_hash(grid, status)
            if status.get("grid_content_hash") != computed_grid_hash:
                errors.append("grid_status_content_hash_mismatch")
            if manifest.get("grid_content_hash") != computed_grid_hash:
                errors.append("manifest_grid_content_hash_mismatch")
            if float(grid["header"]["stamp_sec"]) != float(status.get("grid_content_stamp")):
                errors.append("grid_status_stamp_mismatch")
        except Exception as exc:
            errors.append(f"grid_hash_validation_failed:{type(exc).__name__}")

    canonical_seen = _canonical_cells(seen.get("cells") or [])
    if canonical_seen != seen.get("cells"):
        errors.append("seen_cells_not_canonical")
    if len(canonical_seen) != seen.get("count"):
        errors.append("seen_count_mismatch")
    if _seen_hash(canonical_seen) != seen.get("canonical_hash"):
        errors.append("seen_hash_mismatch")
    if manifest.get("seen_canonical_hash") != seen.get("canonical_hash"):
        errors.append("manifest_seen_hash_mismatch")

    ranked = candidates.get("ranked_candidates") or []
    if len(ranked) != candidates.get("ranked_candidate_count"):
        errors.append("ranked_candidate_count_mismatch")
    if len(ranked) != manifest.get("ranked_candidate_count"):
        errors.append("manifest_ranked_candidate_count_mismatch")
    ids = [candidate.get("candidate_id") for candidate in ranked]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        errors.append("candidate_ids_invalid_or_nonunique")
    if [candidate.get("rank") for candidate in ranked] != list(range(1, len(ranked) + 1)):
        errors.append("candidate_rank_order_invalid")
    for candidate in ranked:
        opportunity = candidate.get("opportunity") or {}
        for prefix in ("visible", "new", "occlusion_reveal"):
            cells = opportunity.get(f"predicted_{prefix}_cell_ids") or []
            if _canonical_cells(cells) != cells:
                errors.append(f"candidate_cell_set_not_canonical:{candidate.get('candidate_id')}:{prefix}")
            if len(cells) != opportunity.get(f"predicted_{prefix}_count"):
                errors.append(f"candidate_cell_count_mismatch:{candidate.get('candidate_id')}:{prefix}")
        components = candidate.get("cheap_rank_components") or {}
        if components.get("new_observable_cells") != opportunity.get("predicted_new_count"):
            errors.append(f"candidate_new_count_proxy_mismatch:{candidate.get('candidate_id')}")
        if components.get("occlusion_reveal_cells") != opportunity.get("predicted_occlusion_reveal_count"):
            errors.append(f"candidate_occlusion_count_proxy_mismatch:{candidate.get('candidate_id')}")

    source_rows = source_manifest.get("source_files") or []
    if source_manifest.get("missing_source_files"):
        errors.append("source_manifest_has_missing_files")
    state_rows = []
    for row in source_rows:
        path = bundle / str(row.get("bundle_relative_path"))
        if not path.is_file():
            errors.append(f"source_copy_missing:{row.get('repo_relative_path')}")
            continue
        actual = _sha256_path(path)
        if actual != row.get("sha256"):
            errors.append(f"source_hash_mismatch:{row.get('repo_relative_path')}")
        state_rows.append([row.get("repo_relative_path"), row.get("sha256"), row.get("size_bytes")])
    if _canonical_json_hash(state_rows) != source_manifest.get("relevant_source_state_sha256"):
        errors.append("source_state_hash_mismatch")
    if manifest.get("source_state_sha256") != source_manifest.get("relevant_source_state_sha256"):
        errors.append("manifest_source_state_hash_mismatch")
    if _sha256_path(bundle / "parameters.json") != manifest.get("parameters_sha256"):
        errors.append("parameters_hash_mismatch")
    if parameters.get("execute_flag_present") is not False:
        errors.append("dry_preflight_template_contains_execute")
    if decision.get("run_id") != manifest.get("run_id") or decision.get("decision_id") != manifest.get("decision_id"):
        errors.append("decision_manifest_identity_mismatch")
    selected_rank = decision.get("selected_rank")
    if decision.get("selected_candidate_id") not in ids:
        errors.append("selected_candidate_not_ranked")
    elif not isinstance(selected_rank, int) or isinstance(selected_rank, bool) or not 1 <= selected_rank <= len(ranked):
        errors.append("selected_rank_invalid")
    elif ranked[selected_rank - 1].get("candidate_id") != decision.get("selected_candidate_id"):
        errors.append("selected_candidate_rank_mismatch")

    return {
        "status": "SNAPSHOT_VALID" if not errors else "SNAPSHOT_INVALID",
        "bundle": str(bundle),
        "run_id": manifest.get("run_id"),
        "decision_id": manifest.get("decision_id"),
        "ranked_candidate_count": len(ranked),
        "errors": errors,
    }


def _source_row(source_manifest: Dict[str, Any], relative: str) -> Optional[Dict[str, Any]]:
    for row in source_manifest.get("source_files") or []:
        if row.get("repo_relative_path") == relative:
            return row
    return None


def _load_frozen_runner(bundle: Path) -> Any:
    source_root = bundle / "sources"
    runner_path = source_root / "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py"
    module_dir = str(runner_path.parent)
    sys.path.insert(0, module_dir)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("frozen_block_astar_dwa_mature_runner", str(runner_path))
        if spec is None or spec.loader is None:
            raise ImportError("frozen_runner_spec_unavailable")
        module = importlib.util.module_from_spec(spec)
        # Python 3.8 dataclasses resolve postponed annotations through
        # sys.modules during class creation.  Register the isolated frozen
        # module first; this changes neither source selection nor evaluation.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        return module
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if sys.path and sys.path[0] == module_dir:
            sys.path.pop(0)


def _runner_args(module: Any, command: Sequence[str]) -> Any:
    tokens = list(command)
    if len(tokens) < 2:
        raise ValueError("runner_command_template_invalid")
    options = tokens[2:]
    if "--execute" in options:
        raise ValueError("offline_replay_execute_flag_forbidden")
    args = module.build_arg_parser().parse_args(options)
    args.execute = False
    return args


def _offline_candidate_preflight(
    runner_module: Any,
    runner_args: Any,
    grid_msg: Any,
    status: Dict[str, Any],
    pose: Sequence[float],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    # Current production exposes the authoritative pure core.  Prefer it over
    # the historical compatibility body below so new evaluator results use the
    # same A*/Grid/DWA/Phase-2 methods as the production runner.  The fallback
    # remains only to read older archived source bundles that predate this seam.
    if hasattr(runner_module, "FrozenRoomLocalEpoch") and hasattr(runner_module, "evaluate_frozen_room_local_candidate"):
        try:
            frozen_epoch = runner_module.FrozenRoomLocalEpoch.from_live_inputs(
                epoch_id="offline_frozen_candidate_preflight",
                pose_odom_xy_yaw=pose,
                grid_msg=grid_msg,
                status_payload=status,
                args=runner_args,
                previous_cmd=(0.0, 0.0),
                wall_heading_prior=None,
            )
            frozen_candidate = runner_module.FrozenRoomLocalCandidate.from_mapping(candidate)
            pure = runner_module.evaluate_frozen_room_local_candidate(frozen_epoch, frozen_candidate)
            legal = pure.get("formal_status") == "FORMALLY_EXECUTABLE"
            terminal = str(pure.get("terminal_reason") or "")
            return {
                "rank": candidate.get("rank"), "candidate_id": candidate.get("candidate_id"),
                "legality": "LEGAL_NOW" if legal else "ILLEGAL",
                "astar_result": "PATH" if pure.get("path_exists") else "NO_PATH",
                "candidate_specific_rejection": None if legal else terminal,
                "dwa_safe_moving": pure.get("safe_moving_candidate_count"),
                "dwa_admitted_count": len(pure.get("motion_candidates") or []),
                "path_cell_count": pure.get("path_cell_count"), "path_length_m": pure.get("path_length_m"),
                "failure_reason": None if legal else terminal,
                "commands_published": False,
                "pure_evaluator_record": pure,
            }
        except Exception as exc:
            return {
                "rank": candidate.get("rank"), "candidate_id": candidate.get("candidate_id"),
                "legality": "UNKNOWN", "astar_result": "UNKNOWN", "candidate_specific_rejection": None,
                "dwa_safe_moving": None, "dwa_admitted_count": None, "path_cell_count": None,
                "path_length_m": None, "failure_reason": f"PURE_EVALUATOR_EXCEPTION:{type(exc).__name__}:{exc}",
                "commands_published": False,
            }
    candidate_id = candidate.get("candidate_id")
    rank = candidate.get("rank")
    result: Dict[str, Any] = {
        "rank": rank,
        "candidate_id": candidate_id,
        "legality": "UNKNOWN",
        "astar_result": "UNKNOWN",
        "candidate_specific_rejection": None,
        "dwa_safe_moving": None,
        "dwa_admitted_count": None,
        "path_cell_count": None,
        "path_length_m": None,
        "failure_reason": None,
        "commands_published": False,
    }
    try:
        target_xy = candidate.get("target_odom_xy")
        if not (isinstance(target_xy, list) and len(target_xy) == 2):
            result.update({"legality": "INVALID", "failure_reason": "TARGET_INVALID"})
            return result
        runner = object.__new__(runner_module.BlockAStarDwaRunner)
        runner.args = copy.copy(runner_args)
        runner.args.execute = False
        runner.pub = None
        runner.prev_cmd = (0.0, 0.0)
        runner.path_stability_hold_count = 0
        runner.no_path_hold_count = 0
        runner.unilateral_clearance_safety_side = None
        runner.unilateral_clearance_missing_count = 0
        runner.latest_wall_heading_prior = None
        runner.wall_heading_point_buffer = []
        runner.latest_imu = None
        runner.imu_heading_anchor_yaw = None
        runner.imu_heading_anchor_wall_time = None
        runner.imu_heading_hold_active_last = False

        qualification_errors = runner.grid_qualification_errors(grid_msg, status)
        if qualification_errors:
            result.update({
                "legality": "INVALID",
                "failure_reason": "GRID_STATUS_CONTRACT_INVALID",
                "qualification_errors": qualification_errors,
            })
            return result
        if status.get("local_traversability_status") != "FREE_SUPPORTED":
            result.update({"legality": "INVALID", "failure_reason": "DECISION_GLOBAL_L3V_STATUS"})
            return result

        pose_tuple = (float(pose[0]), float(pose[1]), float(pose[2]))
        tx_base, ty_base = runner_module.target_to_base(target_xy, pose_tuple)
        distance = math.hypot(tx_base, ty_base)
        if distance <= float(runner.args.goal_tolerance_m):
            result.update({
                "legality": "ILLEGAL",
                "candidate_specific_rejection": "WITHIN_GOAL_TOLERANCE_WITHOUT_ASTAR_DWA_ADMISSION",
                "failure_reason": "CURRENT_PREFLIGHT_REQUIRES_ASTAR_AND_SAFE_MOVING_EVIDENCE",
            })
            return result
        target = {"source": "ROOM_SEARCH_V2", "target_xy_team_livox_odom": list(target_xy)}
        raw_grid = runner.grid_array(grid_msg)
        grid, _preprocess = runner.apply_centerline_thin_barrier_clearing(
            grid_msg, raw_grid, (tx_base, ty_base), False, status,
        )
        occupied_inflated = runner.occupied_inflated_mask(grid, float(grid_msg.info.resolution))
        blocked = runner.inflate_obstacles(grid, float(grid_msg.info.resolution))
        start_cell = runner.local_xy_to_cell(0.0, 0.0, grid_msg)
        if start_cell is None:
            result.update({"legality": "INVALID", "failure_reason": "START_OUT_OF_GRID"})
            return result
        planning_blocked, start_clearance = runner.apply_start_footprint_clearance(
            grid_msg, raw_grid, blocked, occupied_inflated, qualification_passed=True,
        )
        if start_clearance.get("reason") in {"GRID_OR_MASK_INVALID", "START_OUT_OF_GRID", "INVALID_RADIUS"}:
            result.update({
                "legality": "INVALID", "failure_reason": "START_FOOTPRINT_CLEARANCE_INVALID",
                "start_footprint_clearance": start_clearance,
            })
            return result
        (tx_base, ty_base), _shape = runner.unilateral_clearance_target_shape(
            grid_msg, target, grid, planning_blocked, (tx_base, ty_base),
        )
        distance = math.hypot(tx_base, ty_base)
        goal_x = max(0.0, min(tx_base, runner.args.max_goal_x_m))
        goal_y = max(-runner.args.max_goal_abs_y_m, min(ty_base, runner.args.max_goal_abs_y_m))
        goal_cell = runner.local_xy_to_cell(goal_x, goal_y, grid_msg)
        if goal_cell is None:
            result.update({
                "legality": "ILLEGAL", "astar_result": "NO_PATH",
                "candidate_specific_rejection": "TARGET_OUT_OF_GRID",
                "failure_reason": "BLOCK_ASTAR_DWA_GRID_TARGET_OUT_OF_BOUNDS",
            })
            return result
        raw_path = runner.block_astar(planning_blocked, start_cell, goal_cell, grid_msg, None)
        planner_path_source = "coarse_block_astar"
        if len(raw_path) < 2:
            fine_path = runner.fine_grid_astar_fallback(planning_blocked, start_cell, goal_cell)
            if fine_path:
                raw_path = fine_path
                planner_path_source = "fine_grid_fallback"
        path = runner.smooth_path(raw_path)
        result.update({
            "astar_result": "PATH" if len(path) >= 2 else "NO_PATH",
            "planner_path_source": planner_path_source,
            "path_cell_count": len(path),
            "path_length_m": max(0.05, len(path) * float(grid_msg.info.resolution)) if path else None,
        })
        if len(path) < 2:
            result.update({
                "legality": "ILLEGAL",
                "candidate_specific_rejection": "NO_PATH",
                "failure_reason": "BLOCK_ASTAR_DWA_BLOCKED_NO_PATH",
            })
            return result
        waypoint_x, waypoint_y, look_index = runner.select_lookahead_waypoint(path, grid_msg)
        _v, _w, dwa = runner.choose_dwa(
            grid_msg,
            planning_blocked,
            (waypoint_x, waypoint_y),
            (tx_base, ty_base),
            distance,
            {"enabled": False, "active": False, "reason": "offline_legality_only"},
            room_search_safe_moving_eligibility=True,
            pose_odom=pose_tuple,
            target_in_front=(tx_base > 0.0),
            astar_path_exists=True,
        )
        safe_count = int(dwa.get("safe_moving_candidate_count") or 0)
        admitted_count = int(dwa.get("sample_count") or 0)
        result.update({
            "dwa_safe_moving": safe_count,
            "dwa_admitted_count": admitted_count,
            "lookahead_path_index": int(look_index),
            "lookahead_xy": [float(waypoint_x), float(waypoint_y)],
        })
        if bool(dwa.get("blocked")) or safe_count <= 0:
            result.update({
                "legality": "ILLEGAL",
                "candidate_specific_rejection": "DWA_NO_SAFE_MOVING_COMMAND",
                "failure_reason": "BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD",
            })
            return result
        result.update({"legality": "LEGAL_NOW", "failure_reason": None})
        return result
    except Exception as exc:
        result.update({"legality": "UNKNOWN", "failure_reason": f"OFFLINE_REPLAY_EXCEPTION:{type(exc).__name__}:{exc}"})
        return result


def replay_snapshot(bundle_path: Path, runner_module: Optional[Any] = None) -> Dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    validation = validate_snapshot(bundle)
    if validation["status"] != "SNAPSHOT_VALID":
        return {"status": "SNAPSHOT_INVALID", "validation": validation, "results": []}
    source_manifest = _read_json(bundle / "source_manifest.json")
    audit_relative = "scripts/local_subgoal_runner_mvp/frozen_decision_audit.py"
    audit_row = _source_row(source_manifest, audit_relative)
    if audit_row is None:
        return {"status": "SOURCE_IDENTITY_MISMATCH", "reason": "FROZEN_AUDIT_SOURCE_MISSING", "results": []}
    current_audit_hash = _sha256_path(Path(__file__).resolve())
    if current_audit_hash != audit_row.get("sha256") and runner_module is None:
        frozen_script = bundle / str(audit_row["bundle_relative_path"])
        return {
            "status": "SOURCE_IDENTITY_MISMATCH",
            "reason": "CURRENT_AUDIT_TOOL_DIFFERS_FROM_FROZEN_COPY",
            "frozen_replay_command": [sys.executable, str(frozen_script), "replay", str(bundle)],
            "results": [],
        }
    parameters = _read_json(bundle / "parameters.json")
    grid_doc = _read_json(bundle / "grid.json")
    status = _read_json(bundle / "status.json")
    decision = _read_json(bundle / "decision.json")
    candidates = _read_json(bundle / "candidates.json")["ranked_candidates"]
    try:
        module = runner_module if runner_module is not None else _load_frozen_runner(bundle)
        args = _runner_args(module, parameters["production_dry_preflight_command_template"])
        grid_msg = _grid_message_from_payload(grid_doc["occupancy_grid"])
    except Exception as exc:
        return {
            "status": "SOURCE_IDENTITY_MISMATCH",
            "reason": f"FROZEN_EXECUTION_SOURCE_LOAD_FAILED:{type(exc).__name__}:{exc}",
            "results": [],
        }
    results = [
        _offline_candidate_preflight(
            module, args, grid_msg, status, decision["robot_pose_odom_xy_yaw"], candidate,
        )
        for candidate in candidates
    ]
    return {
        "status": "DRY_PREFLIGHT_COMPLETE",
        "schema_version": SCHEMA_VERSION,
        "bundle": str(bundle),
        "run_id": decision["run_id"],
        "decision_id": decision["decision_id"],
        "grid_identity": grid_doc["planning_identity"],
        "source_state_sha256": source_manifest["relevant_source_state_sha256"],
        "parameters_sha256": _sha256_path(bundle / "parameters.json"),
        "commands_published": False,
        "selection_performed": False,
        "semantic_limit": "CURRENT_EXECUTABILITY_ONLY",
        "results": results,
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("bundle", type=Path)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("bundle", type=Path)
    replay_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        result = validate_snapshot(args.bundle)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "SNAPSHOT_VALID" else 2
    result = replay_snapshot(args.bundle)
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "DRY_PREFLIGHT_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
