#!/usr/bin/env python3
"""Authority-free ROOM_SEARCH locomotion-compatibility shadow capture.

This module intentionally has no ROS imports, publishers, planners, or command
writers.  It consumes values that production has already computed and moves
them to a bounded background writer only when explicitly enabled.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


FLAG = "ROOM_SEARCH_HIGH_LEVEL_LOCOMOTION_COMPATIBILITY_SHADOW_V0"
CANDIDATES = "HIGH_LEVEL_LOCOMOTION_COMPATIBILITY_CANDIDATES.jsonl"
OUTCOMES = "HIGH_LEVEL_LOCOMOTION_COMPATIBILITY_SELECTED_OUTCOMES.json"
SCENE = "ROOM_SEARCH_SCENE_IDENTITY.json"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


class HighLevelLocomotionCompatibilityShadow:
    """A best-effort, loss-tolerant evidence sink with zero control authority."""

    def __init__(self, enabled: bool, directory: Optional[Path], root: Path) -> None:
        self.enabled = bool(enabled and directory is not None)
        self.directory = directory
        self.root = root
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=128)
        self.outcomes = []
        self._lock = threading.Lock()
        self.dropped_events = 0
        self.writer: Optional[threading.Thread] = None
        if self.enabled:
            assert self.directory is not None
            self.directory.mkdir(parents=True, exist_ok=True)
            self.writer = threading.Thread(target=self._writer, name="room-search-hl-shadow", daemon=True)
            self.writer.start()
            self._submit("scene", {})

    @classmethod
    def from_environment(cls, root: Path) -> "HighLevelLocomotionCompatibilityShadow":
        enabled = os.environ.get(FLAG, "").strip().lower() in {"1", "true", "yes", "on"}
        archive = os.environ.get("STATE_MACHINE_RUN_ARCHIVE_DIR", "").strip()
        return cls(enabled, Path(archive) if archive else None, root)

    def _scene_identity(self) -> Dict[str, Any]:
        assert self.directory is not None
        generated = self.root / "generated_building"
        names = (
            "scene_manifest.json", "layout_metadata.json", "building_config.json",
            "competition_scene.world", "world.sdf",
        )
        files = []
        for name in names:
            path = generated / name
            files.append({"path": str(path.relative_to(self.root)), "exists": path.is_file(), "sha256": _sha256(path)})
        return {
            "schema_version": "room_search_scene_identity_v0",
            "authority_enabled": False,
            "run_id": os.environ.get("STATE_MACHINE_RUN_ID", ""),
            "scene_seed": None,
            "scene_seed_status": "NOT_RETAINED_BY_EXISTING_INTERFACE",
            "identity_classification": "PARTIAL_NOT_REPLAY_CLOSED",
            "world_files": files,
            "launch_config": os.environ.get("STATE_MACHINE_LAUNCH_CONFIG", None),
            "relevant_environment": {
                FLAG: os.environ.get(FLAG),
                "STATE_MACHINE_RUN_ID": os.environ.get("STATE_MACHINE_RUN_ID"),
                "STATE_MACHINE_RUN_ARCHIVE_DIR": os.environ.get("STATE_MACHINE_RUN_ARCHIVE_DIR"),
                "RUN_RANDOM_SEED": os.environ.get("RUN_RANDOM_SEED"),
            },
            "note": "No Gazebo truth, object pose, or random seed is fabricated by the shadow.",
        }

    def _writer(self) -> None:
        assert self.directory is not None
        candidate_path = self.directory / CANDIDATES
        while True:
            event = self.queue.get()
            if event.get("kind") == "STOP":
                return
            try:
                if event.get("kind") == "candidate":
                    with candidate_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event["payload"], ensure_ascii=False, sort_keys=True) + "\n")
                elif event.get("kind") == "scene":
                    (self.directory / SCENE).write_text(
                        json.dumps(self._scene_identity(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                elif event.get("kind") == "outcome":
                    with self._lock:
                        self.outcomes.append(event["payload"])
                        snapshot = list(self.outcomes)
                    (self.directory / OUTCOMES).write_text(
                        json.dumps({"schema_version": "high_level_locomotion_selected_outcomes_v0", "authority_enabled": False,
                                    "outcomes": snapshot, "dropped_events": self.dropped_events}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            except Exception:
                # Evidence loss is isolated from production control.
                pass

    def _submit(self, kind: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.queue.put_nowait({"kind": kind, "payload": payload})
        except queue.Full:
            self.dropped_events += 1

    def record_preflight(
        self, *, decision_id: int, rank: int, candidate: Dict[str, Any], preflight: Dict[str, Any],
        pose_xy_yaw: Sequence[float], planning_context: Dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        target = candidate.get("target_xy_team_livox_odom") or candidate.get("base_xy")
        target = target if isinstance(target, (list, tuple)) and len(target) >= 2 else (None, None)
        dx = float(target[0]) - float(pose_xy_yaw[0]) if _finite(target[0]) else None
        dy = float(target[1]) - float(pose_xy_yaw[1]) if _finite(target[1]) else None
        world_bearing = math.atan2(dy, dx) if dx is not None and dy is not None else None
        relative_bearing = _angle(world_bearing - float(pose_xy_yaw[2])) if world_bearing is not None else None
        runner = preflight.get("runner") if isinstance(preflight.get("runner"), dict) else {}
        dwa = runner.get("last_dwa") if isinstance(runner.get("last_dwa"), dict) else {}
        self._submit("candidate", {
            "schema_version": "high_level_locomotion_compatibility_candidate_v0",
            "authority_enabled": False, "run_id": os.environ.get("STATE_MACHINE_RUN_ID", ""),
            "decision_id": int(decision_id), "candidate_id": candidate.get("_room_search_audit_candidate_id"), "rank": int(rank),
            "robot_pose_odom_xy_yaw": [float(v) for v in pose_xy_yaw],
            "candidate_target_xy_team_livox_odom": [float(target[0]), float(target[1])] if _finite(target[0]) else None,
            "candidate_distance_m": math.hypot(dx, dy) if dx is not None and dy is not None else None,
            "candidate_world_bearing_rad": world_bearing, "candidate_relative_bearing_rad": relative_bearing,
            "heading_change_rad": candidate.get("heading_change_rad"), "sector": candidate.get("sector"),
            "observation_yaw_rad": candidate.get("observation_yaw_rad"),
            "grid_identity": {key: planning_context.get(key) for key in ("grid_header_stamp_sec", "grid_content_stamp", "content_generation_id", "grid_content_hash")},
            "formal_preflight": {"legal": bool(preflight.get("legal")), "runner_final_decision": runner.get("runner_final_decision"),
                                 "astar_path_exists": dwa.get("p_through_astar_path_exists"),
                                 "safe_moving_candidate_count": dwa.get("safe_moving_candidate_count"),
                                 "total_dwa_sample_count": dwa.get("sample_count"),
                                 "same_evaluation_shadow": dwa.get("high_level_locomotion_shadow", "DWA_EVIDENCE_UNAVAILABLE")},
            "descriptive_flags": {"LARGE_INITIAL_HEADING_BURDEN": relative_bearing,
                                    "COMPATIBILITY_EVIDENCE_INCOMPLETE": not isinstance(dwa.get("high_level_locomotion_shadow"), dict)},
        })

    def record_outcome(self, *, decision_id: int, candidate: Dict[str, Any], start_pose: Sequence[float], terminal_pose: Sequence[float], runner: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        target = candidate.get("target_xy_team_livox_odom") or (None, None)
        before = math.hypot(float(target[0]) - float(start_pose[0]), float(target[1]) - float(start_pose[1])) if _finite(target[0]) else None
        after = math.hypot(float(target[0]) - float(terminal_pose[0]), float(target[1]) - float(terminal_pose[1])) if _finite(target[0]) else None
        steps = runner.get("steps") if isinstance(runner.get("steps"), list) else []
        command_modes = [{"linear_x": step.get("cmd_linear_x"), "angular_z": step.get("cmd_angular_z")} for step in steps if isinstance(step, dict)]
        self._submit("outcome", {
            "authority_enabled": False, "run_id": os.environ.get("STATE_MACHINE_RUN_ID", ""), "decision_id": int(decision_id),
            "candidate_id": candidate.get("_room_search_audit_candidate_id"), "rank": candidate.get("cheap_rank"),
            "runner_final_decision": runner.get("runner_final_decision"), "target_distance_start_m": before, "target_distance_terminal_m": after,
            "actual_xy_delta_m": [float(terminal_pose[0]) - float(start_pose[0]), float(terminal_pose[1]) - float(start_pose[1])],
            "actual_yaw_delta_rad": _angle(float(terminal_pose[2]) - float(start_pose[2])),
            "command_mode_summary_existing_steps": command_modes,
        })
