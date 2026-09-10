"""Shared validator and serializer for the frozen P2KG11 Portal frame contract."""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List


PORTAL_FRAME_CONTRACT = "p2kg11_portal_frame_v1"
# Frozen p2kg11_portal_frame_v1 side fields.  Presence is required even when a
# field's payload value is explicitly null under the contract.
PORTAL_FRAME_SIDE_FIELDS = frozenset({
    "candidate_available", "evidence_provenance", "evidence_stamp",
    "first_seen_stamp", "frame_id", "free_space_support", "input_quality_flags",
    "left_boundary", "method_version", "observation_state", "portal_center_base",
    "portal_normal_base", "portal_track_version", "portal_width", "rejection_reason",
    "right_boundary", "side", "source_ray_count", "source_stamp", "stable_duration",
    "temporal_support", "track_id", "traversability_support", "wall_gap_support",
})
PORTAL_OBSERVATION_STATES = {"confirmed", "partial", "rejected"}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_portal_frame_v1(payload: Any) -> List[str]:
    """Return contract violations; valid payloads return an empty list."""
    if not isinstance(payload, dict):
        return ["payload"]
    errors: List[str] = []
    if payload.get("contract_version") != PORTAL_FRAME_CONTRACT:
        errors.append("contract_version")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"]:
        errors.append("run_id")
    if not isinstance(payload.get("frame_sequence"), int) or isinstance(payload.get("frame_sequence"), bool) or payload["frame_sequence"] < 1:
        errors.append("frame_sequence")
    if not _finite_number(payload.get("source_stamp")):
        errors.append("source_stamp")
    if not isinstance(payload.get("frame_id"), str) or not payload["frame_id"]:
        errors.append("frame_id")
    for side in ("left", "right"):
        row = payload.get(side)
        if not isinstance(row, dict):
            errors.append(side)
            continue
        missing = PORTAL_FRAME_SIDE_FIELDS - set(row)
        if missing:
            errors.append(f"{side}.missing.{','.join(sorted(missing))}")
        if row.get("side") != side:
            errors.append(f"{side}.side")
        if row.get("source_stamp") != payload.get("source_stamp"):
            errors.append(f"{side}.source_stamp")
        if row.get("frame_id") != payload.get("frame_id"):
            errors.append(f"{side}.frame_id")
        if row.get("observation_state") not in PORTAL_OBSERVATION_STATES:
            errors.append(f"{side}.observation_state")
        if not isinstance(row.get("candidate_available"), bool):
            errors.append(f"{side}.candidate_available")
    return errors


def portal_frame_payload_v1(
    run_id: str,
    frame_sequence: int,
    source_stamp: float,
    frame_id: str,
    outputs: Iterable[Dict[str, Any]],
) -> str:
    """Build the atomic frame payload and reject any incomplete side contract."""
    rows = list(outputs)
    by_side = {row.get("side"): row for row in rows}
    payload = {
        "contract_version": PORTAL_FRAME_CONTRACT,
        "run_id": run_id,
        "frame_sequence": frame_sequence,
        "source_stamp": source_stamp,
        "frame_id": frame_id,
        "left": by_side.get("left"),
        "right": by_side.get("right"),
    }
    if len(rows) != 2 or set(by_side) != {"left", "right"}:
        raise ValueError("p2kg11_portal_frame_v1 requires exactly one left and one right result")
    errors = validate_portal_frame_v1(payload)
    if errors:
        raise ValueError("p2kg11_portal_frame_v1 invalid: " + ",".join(errors))
    return json.dumps(payload, sort_keys=True)
