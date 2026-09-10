#!/usr/bin/env python3
"""Frozen P2K-G8 METHOD_3 + METHOD_4 wrapper for audit-only Portal Shadow.

Portal geometry is delegated to the frozen G8 reference.  This module neither
loads world/Truth data nor has ROS dependencies or control semantics.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
G8_REFERENCE = ROOT / "audit_tools/p2kg8_offline_portal_recognition.py"
G8_FREEZE = ROOT / "debug/odom_accuracy_audit_v1/p2kg8_offline_portal_046/development_method_freeze.json"
FORBIDDEN_INPUT_TERMS = {
    "world_door_center", "world_door_width", "room_side", "doorway_timestamp",
    "truth_pose", "collision_model", "model_states", "entry_pose", "target", "cmd_vel",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_module():
    spec = importlib.util.spec_from_file_location("p2kg8_frozen_portal_method", G8_REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("G8_FROZEN_METHOD_REPRODUCTION_FAILED: import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenPortalMethod:
    """Serial METHOD_4 tracker with byte-for-byte G8 transition logic."""

    def __init__(self) -> None:
        self.g8 = _frozen_module()
        self.freeze = json.loads(G8_FREEZE.read_text(encoding="utf-8"))
        if sha256(G8_REFERENCE) != self.freeze["reference_sha256"]:
            raise RuntimeError("G8_FROZEN_METHOD_REPRODUCTION_FAILED: source hash")
        self.tracks = {
            "left": {"first": None, "last": None, "id": 0},
            "right": {"first": None, "last": None, "id": 0},
        }
        self.last_source_stamp: float | None = None
        self.track_version = 0

    @property
    def method_sha256(self) -> str:
        return sha256(G8_REFERENCE)

    def reset(self) -> None:
        for side in self.tracks:
            self.tracks[side] = {"first": None, "last": None, "id": 0}
        self.last_source_stamp = None
        self.track_version = 0

    def process(self, rays, source_stamp: float, provenance: str = "l1s_filtered_cloud") -> list[dict[str, Any]]:
        if self.last_source_stamp is not None and source_stamp < self.last_source_stamp:
            return [self._out_of_order(source_stamp, provenance, side, len(rays)) for side in ("left", "right")]
        frame = {"source_stamp": source_stamp, "frame_id": "base", "rays": rays, "source": provenance}
        self.g8.assert_truth_disabled(frame)
        rows = [self.g8.candidate(frame, side, 3) for side in ("left", "right")]
        for row in rows:
            # This is intentionally the same transition body as G8.method4_track.
            state = self.tracks[row["side"]]
            stamp = row["source_stamp"]
            if row["observation_state"] == "confirmed":
                if state["last"] is None or stamp - state["last"] > 0.20:
                    state["id"] += 1
                    state["first"] = stamp
                state["last"] = stamp
                duration = stamp - state["first"]
                row["track_id"] = f"{row['side']}-{state['id']}"
                row["first_seen_stamp"] = state["first"]
                row["stable_duration"] = duration
                if duration < self.g8.TEMPORAL_STABLE_SEC:
                    row["candidate_available"] = False
                    row["observation_state"] = "partial"
                    row["rejection_reason"] = "TEMPORAL_STABILITY_NOT_YET_MET"
                else:
                    row["temporal_support"] = True
            else:
                row["track_id"] = None
                row["first_seen_stamp"] = None
                row["stable_duration"] = 0.0
            row["portal_track_version"] = self.track_version
        self.last_source_stamp = source_stamp
        self.track_version += 1
        self._assert_contract(rows)
        return rows

    @staticmethod
    def _out_of_order(stamp: float, provenance: str, side: str, ray_count: int) -> dict[str, Any]:
        return {
            "source_stamp": stamp, "evidence_stamp": stamp, "frame_id": "base",
            "method_version": "p2kg8_method_3", "side": side,
            "candidate_available": False, "portal_center_base": None, "portal_width": None,
            "portal_normal_base": None, "left_boundary": None, "right_boundary": None,
            "wall_gap_support": {}, "free_space_support": {}, "traversability_support": False,
            "temporal_support": False, "observation_state": "unknown",
            "rejection_reason": "OUT_OF_ORDER_SOURCE_STAMP", "source_ray_count": int(ray_count),
            "evidence_provenance": provenance, "input_quality_flags": ["OUT_OF_ORDER_SOURCE_STAMP"],
            "track_id": None, "first_seen_stamp": None, "stable_duration": 0.0,
        }

    @staticmethod
    def _assert_contract(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["candidate_available"] and row["observation_state"] != "confirmed":
                raise RuntimeError("EVIDENCE_INVALID: candidate/confirmed invariant")
            if FORBIDDEN_INPUT_TERMS & set(row):
                raise RuntimeError("TRUTH_LEAKAGE_VIOLATION")
