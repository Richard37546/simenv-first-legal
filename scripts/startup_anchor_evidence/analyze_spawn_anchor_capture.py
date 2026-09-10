#!/usr/bin/env python3
"""Offline-only evaluation of the fixed spawn-to-odom anchor candidate.

The candidate path consumes only explicit spawn metadata and gated odometry.
Gazebo model state is loaded afterwards, exclusively to score the already
constructed estimates.  This file has no ROS imports or publishers.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _normalise(q: Sequence[float]) -> List[float]:
    length = math.sqrt(sum(float(value) ** 2 for value in q))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("invalid_quaternion")
    return [float(value) / length for value in q]


def _multiply(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]


def _inverse(q: Sequence[float]) -> List[float]:
    return [-q[0], -q[1], -q[2], q[3]]


def _rotate(q: Sequence[float], point: Sequence[float]) -> List[float]:
    return _multiply(_multiply(q, [point[0], point[1], point[2], 0.0]), _inverse(q))[:3]


def _yaw_quaternion(yaw_rad: float) -> List[float]:
    return [0.0, 0.0, math.sin(float(yaw_rad) / 2.0), math.cos(float(yaw_rad) / 2.0)]


def _pose_from_row(row: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    pose = row["pose"]
    position = [float(value) for value in pose["position_xyz"]]
    quaternion = _normalise([float(value) for value in pose["quaternion_xyzw"]])
    if len(position) != 3 or not _finite(position):
        raise ValueError("invalid_position")
    return position, quaternion


def candidate_transform(spawn_xyz_yaw: Sequence[float], anchor_odom_row: Dict[str, Any]) -> Dict[str, List[float]]:
    """Construct the fixed candidate using spawn metadata and odom only."""
    if len(spawn_xyz_yaw) != 4 or not _finite(spawn_xyz_yaw):
        raise ValueError("invalid_explicit_spawn")
    odom_position, odom_quaternion = _pose_from_row(anchor_odom_row)
    world_base_quaternion = _yaw_quaternion(float(spawn_xyz_yaw[3]))
    world_odom_quaternion = _normalise(_multiply(world_base_quaternion, _inverse(odom_quaternion)))
    rotated_odom_origin = _rotate(world_odom_quaternion, odom_position)
    return {
        "translation_xyz": [float(spawn_xyz_yaw[index]) - rotated_odom_origin[index] for index in range(3)],
        "quaternion_xyzw": world_odom_quaternion,
    }


def estimate_world_pose(candidate: Dict[str, Sequence[float]], odom_row: Dict[str, Any]) -> Dict[str, List[float]]:
    """Apply an already frozen candidate; this path never reads oracle data."""
    position, quaternion = _pose_from_row(odom_row)
    estimated_position = [
        float(candidate["translation_xyz"][index]) + _rotate(candidate["quaternion_xyzw"], position)[index]
        for index in range(3)
    ]
    estimated_quaternion = _normalise(_multiply(candidate["quaternion_xyzw"], quaternion))
    return {"position_xyz": estimated_position, "quaternion_xyzw": estimated_quaternion}


def _rows(path: pathlib.Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _continuity_valid(rows: Iterable[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    events = []
    for row in rows:
        if row.get("event_type") != "EXTERNAL_AUDIT_EVENT" or row.get("source") != "ODOM_EPOCH_AUDIT":
            continue
        try:
            payload = json.loads(row["payload"])
        except (KeyError, TypeError, ValueError):
            continue
        events.append(payload)
    accepted = [event for event in events if event.get("event_type") in {"FIRST_ACCEPT", "GATED_ACCEPT"}]
    identities = {str(event.get("continuity_id", "")) for event in accepted}
    valid = bool(accepted) and len(identities) == 1 and all(event.get("continuity_state") == "CONTINUOUS" for event in accepted)
    return valid, accepted


def _nearest_oracle(rows: List[Dict[str, Any]], times: List[float], stamp: float) -> Dict[str, Any]:
    index = bisect.bisect_left(times, stamp)
    choices = rows[max(0, index - 1):min(len(rows), index + 1)]
    if not choices:
        raise ValueError("oracle_epoch_empty")
    return min(choices, key=lambda row: abs(float(row["sim_time_at_callback_sec"]) - stamp))


def _percentile(values: List[float], fraction: float) -> float:
    return sorted(values)[int(float(fraction) * (len(values) - 1))]


def analyze(archive_dir: pathlib.Path, anchor_kind: str = "gated") -> Dict[str, Any]:
    provenance = json.loads((archive_dir / "provenance.json").read_text(encoding="utf-8"))
    spawn = provenance.get("effective_spawn_provenance", {})
    if spawn.get("status") != "RECORDED_EXPLICIT_EFFECTIVE_INPUT":
        raise ValueError("effective_spawn_provenance_unavailable")
    values = spawn.get("values", {})
    spawn_xyz_yaw = [float(values[key]) for key in ("ROBOT_X", "ROBOT_Y", "ROBOT_Z", "ROBOT_YAW")]
    production = _rows(archive_dir / "production_evidence.jsonl")
    if anchor_kind not in {"gated", "raw"}:
        raise ValueError("invalid_anchor_kind")
    odom_event = "GATED_ODOM" if anchor_kind == "gated" else "RAW_ICP_ODOM"
    odom = [
        row for row in production
        if row.get("event_type") == odom_event
        and row.get("header_frame_id") == "team_livox_odom"
        and row.get("child_frame_id") == "base"
    ]
    if not odom:
        raise ValueError("no_valid_%s_odom" % anchor_kind)
    continuity_valid, continuity_events = _continuity_valid(production)
    if not continuity_valid:
        raise ValueError("continuity_not_valid")

    # Candidate construction and every estimated pose are complete before the
    # TEST_ONLY oracle file is opened below.
    anchor = odom[0]
    candidate = candidate_transform(spawn_xyz_yaw, anchor)
    estimated = [
        {"stamp": float(row["message_stamp_sec"]), "epoch": row.get("sim_time_epoch_id"), "pose": estimate_world_pose(candidate, row)}
        for row in odom
    ]

    oracle_rows = [
        row for row in _rows(archive_dir / "test_only_gazebo_oracle.jsonl")
        if row.get("provenance") == "TEST_ONLY_ORACLE"
        and row.get("event_type") == "GAZEBO_MODEL_STATE"
        and row.get("model_identity") == "a1_gazebo"
    ]
    oracle_by_epoch: Dict[str, List[Dict[str, Any]]] = {}
    for row in oracle_rows:
        oracle_by_epoch.setdefault(str(row.get("sim_time_epoch_id")), []).append(row)
    for rows in oracle_by_epoch.values():
        rows.sort(key=lambda row: float(row["sim_time_at_callback_sec"]))
    oracle_times = {epoch: [float(row["sim_time_at_callback_sec"]) for row in rows] for epoch, rows in oracle_by_epoch.items()}

    samples = []
    travelled_distance = 0.0
    previous_odom_position = None
    for odom_row, estimate in zip(odom, estimated):
        epoch = str(estimate["epoch"])
        if epoch not in oracle_by_epoch:
            raise ValueError("oracle_epoch_missing")
        oracle = _nearest_oracle(oracle_by_epoch[epoch], oracle_times[epoch], estimate["stamp"])
        oracle_position = [float(value) for value in oracle["position_xyz"]]
        estimate_position = estimate["pose"]["position_xyz"]
        error = math.sqrt(sum((estimate_position[index] - oracle_position[index]) ** 2 for index in range(3)))
        odom_position, _unused = _pose_from_row(odom_row)
        if previous_odom_position is not None:
            travelled_distance += math.sqrt(sum((odom_position[index] - previous_odom_position[index]) ** 2 for index in range(3)))
        previous_odom_position = odom_position
        samples.append({
            "stamp_sec": estimate["stamp"],
            "elapsed_sec": estimate["stamp"] - float(anchor["message_stamp_sec"]),
            "travelled_distance_m": travelled_distance,
            "error_3d_m": error,
            "oracle_pair_gap_sec": abs(float(oracle["sim_time_at_callback_sec"]) - estimate["stamp"]),
        })
    errors = [sample["error_3d_m"] for sample in samples]
    return {
        "schema_version": "spawn_anchor_offline_validation_v1",
        "run_id": provenance.get("run_id"),
        "candidate_input": {
            "effective_spawn": spawn,
            "anchor": {
                "message_stamp_sec": anchor["message_stamp_sec"],
                "frame_id": anchor["header_frame_id"],
                "child_frame_id": anchor["child_frame_id"],
                "sim_time_epoch_id": anchor.get("sim_time_epoch_id"),
                "pose": anchor["pose"],
            },
            "continuity_id": continuity_events[0]["continuity_id"],
        },
        "candidate_transform_construction": "spawn_metadata_plus_first_%s_odom_only" % anchor_kind,
        "oracle_role": "TEST_ONLY_ORACLE_AFTER_ESTIMATE_CONSTRUCTION",
        "statistics": {
            "sample_count": len(samples),
            "anchor_error_3d_m": errors[0],
            "median_error_3d_m": _percentile(errors, 0.5),
            "p95_error_3d_m": _percentile(errors, 0.95),
            "max_error_3d_m": max(errors),
            "max_oracle_pair_gap_sec": max(sample["oracle_pair_gap_sec"] for sample in samples),
            "total_odom_travelled_distance_m": samples[-1]["travelled_distance_m"],
        },
        "error_vs_time_distance": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor-kind", choices=("raw", "gated"), default="gated")
    args = parser.parse_args()
    result = analyze(pathlib.Path(args.archive_dir), args.anchor_kind)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
