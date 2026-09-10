#!/usr/bin/env python3
"""Result-only world-coordinate writer for confirmed danger tracks.

This sidecar deliberately has no navigation, selection, command, grid, or
Gazebo input.  It freezes ``T_world_odom`` from explicit effective spawn
metadata and the first raw ICP odometry sample.  A result is emitted only
after the existing odometry gate reports one continuous run-scoped epoch.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RAW_ODOM_TOPIC = "/team/livox/icp_odom_raw"
GATE_STATUS_TOPIC = "/team/livox/icp_odom_gate_status"
TRACK_TOPIC = "/team/danger_tracks"
ODOM_FRAME = "team_livox_odom"
BASE_FRAME = "base"


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _unit(q: Sequence[float]) -> List[float]:
    if len(q) != 4 or not _finite(q):
        raise ValueError("invalid_quaternion")
    norm = math.sqrt(sum(float(value) ** 2 for value in q))
    if norm <= 1e-12:
        raise ValueError("invalid_quaternion")
    return [float(value) / norm for value in q]


def _mul(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]


def _inverse(q: Sequence[float]) -> List[float]:
    q = _unit(q)
    return [-q[0], -q[1], -q[2], q[3]]


def _rotate(q: Sequence[float], point: Sequence[float]) -> List[float]:
    return _mul(_mul(_unit(q), [float(point[0]), float(point[1]), float(point[2]), 0.0]), _inverse(q))[:3]


def _yaw_quaternion(yaw_rad: float) -> List[float]:
    return [0.0, 0.0, math.sin(float(yaw_rad) / 2.0), math.cos(float(yaw_rad) / 2.0)]


@dataclass(frozen=True)
class Pose:
    position_xyz: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    stamp_sec: float


@dataclass(frozen=True)
class EffectiveSpawn:
    position_xyz: Tuple[float, float, float]
    yaw_rad: float
    provenance: str


def transform_from_spawn_and_first_raw(spawn: EffectiveSpawn, first_raw: Pose) -> Dict[str, List[float]]:
    """T_world_odom = T_world_base_spawn * inverse(T_odom_base(first_raw))."""
    if not _finite((*spawn.position_xyz, spawn.yaw_rad, *first_raw.position_xyz, *first_raw.quaternion_xyzw)):
        raise ValueError("nonfinite_anchor_input")
    world_base_q = _yaw_quaternion(spawn.yaw_rad)
    world_odom_q = _unit(_mul(world_base_q, _inverse(first_raw.quaternion_xyzw)))
    rotated_odom_origin = _rotate(world_odom_q, first_raw.position_xyz)
    return {
        "translation_xyz": [float(spawn.position_xyz[i]) - rotated_odom_origin[i] for i in range(3)],
        "quaternion_xyzw": world_odom_q,
    }


def apply_transform(transform: Dict[str, Sequence[float]], odom_xyz: Sequence[float]) -> List[float]:
    if len(odom_xyz) != 3 or not _finite(odom_xyz):
        raise ValueError("invalid_odom_xyz")
    translation, quaternion = transform["translation_xyz"], transform["quaternion_xyzw"]
    if len(translation) != 3:
        raise ValueError("invalid_transform")
    rotated = _rotate(quaternion, odom_xyz)
    return [float(translation[i]) + rotated[i] for i in range(3)]


def explicit_spawn_from_environment(environment: Optional[Dict[str, str]] = None) -> EffectiveSpawn:
    environment = os.environ if environment is None else environment
    required = ("ROBOT_X", "ROBOT_Y", "ROBOT_Z", "ROBOT_YAW")
    missing = [key for key in required if not str(environment.get(key, "")).strip()]
    source = str(environment.get("STARTUP_ANCHOR_SPAWN_SOURCE", "")).strip()
    if missing or source != "EXPLICIT_AUDIT_WRAPPER_ENV":
        raise ValueError("explicit_effective_spawn_provenance_required")
    values = [float(environment[key]) for key in required]
    if not _finite(values):
        raise ValueError("invalid_effective_spawn")
    return EffectiveSpawn(tuple(values[:3]), values[3], source)


class ResultCoordinateAuthority:
    """Small state holder with a deliberately fail-closed result boundary."""
    def __init__(self, run_id: str, spawn: EffectiveSpawn) -> None:
        if not run_id or not all(char.isalnum() or char in "_.-" for char in run_id):
            raise ValueError("valid_run_id_required")
        self.run_id = run_id
        self.spawn = spawn
        self.first_raw: Optional[Pose] = None
        self.latest_raw_stamp_sec: Optional[float] = None
        self.transform: Optional[Dict[str, List[float]]] = None
        self.continuity_id: Optional[str] = None
        self.continuity_valid = False
        self.invalid_reason: Optional[str] = None
        self.last_gate_continuity_id: Optional[str] = None
        self.last_gate_rejection_reason: Optional[str] = None

    def observe_first_raw(self, pose: Pose) -> None:
        if self.invalid_reason is not None:
            return
        if not _finite((*pose.position_xyz, *pose.quaternion_xyzw, pose.stamp_sec)):
            if self.first_raw is None:
                self.invalid_reason = "invalid_first_raw"
            return
        if self.first_raw is None:
            self.first_raw = pose
            self.transform = transform_from_spawn_and_first_raw(self.spawn, pose)
        self.latest_raw_stamp_sec = pose.stamp_sec

    def raw_elapsed_duration_sec(self) -> float:
        """Return a duration only when both endpoints share raw-odom time."""
        if self.first_raw is None or self.latest_raw_stamp_sec is None:
            raise RuntimeError("raw_odom_duration_authority_unavailable")
        elapsed = float(self.latest_raw_stamp_sec) - float(self.first_raw.stamp_sec)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError("raw_odom_duration_invalid")
        return elapsed

    def observe_gate_status(self, status: Dict[str, Any]) -> None:
        if self.invalid_reason is not None:
            return
        continuity_id = str(status.get("continuity_id", ""))
        state = str(status.get("continuity_state", ""))
        localization = str(status.get("localization_authority", ""))
        self.last_gate_continuity_id = continuity_id or None
        expected_prefix = self.run_id + ":"
        if state in {"REBASE_REQUIRED", "INVALID"} or localization != "CONTINUOUS":
            self.continuity_valid = False
            self.invalid_reason = "continuity_invalid:" + (state or localization or "unknown")
            self.last_gate_rejection_reason = self.invalid_reason
            return
        if state != "CONTINUOUS" or not continuity_id.startswith(expected_prefix):
            self.continuity_valid = False
            self.last_gate_rejection_reason = (
                "continuity_run_id_mismatch" if continuity_id else "continuity_status_incomplete"
            )
            return
        if self.continuity_id is not None and continuity_id != self.continuity_id:
            self.continuity_valid = False
            self.invalid_reason = "continuity_identity_changed"
            self.last_gate_rejection_reason = self.invalid_reason
            return
        self.continuity_id = continuity_id
        self.continuity_valid = True
        self.last_gate_rejection_reason = None

    def ready(self) -> bool:
        return self.transform is not None and self.first_raw is not None and self.continuity_valid and self.invalid_reason is None

    def provenance(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "authority": "RESULT_COORDINATE_ONLY",
            "anchor_construction": "effective_spawn_world_pose_plus_inverse_first_raw_team_livox_odom",
            "effective_spawn": {
                "position_xyz": list(self.spawn.position_xyz),
                "yaw_rad": self.spawn.yaw_rad,
                "provenance": self.spawn.provenance,
            },
            "first_raw": None if self.first_raw is None else {
                "stamp_sec": self.first_raw.stamp_sec,
                "frame_id": ODOM_FRAME,
                "child_frame_id": BASE_FRAME,
                "position_xyz": list(self.first_raw.position_xyz),
                "quaternion_xyzw": list(self.first_raw.quaternion_xyzw),
            },
            "latest_raw_stamp_sec": self.latest_raw_stamp_sec,
            "continuity_id": self.continuity_id,
            "continuity_valid": self.continuity_valid,
            "invalid_reason": self.invalid_reason,
            "last_gate_continuity_id": self.last_gate_continuity_id,
            "last_gate_rejection_reason": self.last_gate_rejection_reason,
            "transform_world_odom": self.transform,
            "gazebo_input_used": False,
        }

    def pending_result_document(self) -> Dict[str, Any]:
        """Replace any prior-run result with this run's not-yet-ready document."""
        provenance = self.provenance()
        provenance["result_state"] = "PENDING_CURRENT_RUN_COORDINATE_AUTHORITY"
        return {
            "exploration_time": 0.0,
            "detected_danger_sources": [],
            "coordinate_provenance": provenance,
        }

    def result_document(self, confirmed_tracks: Sequence[Dict[str, Any]], exploration_time: float) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("world_result_authority_not_ready")
        positions = []
        for track in confirmed_tracks:
            xyz = track.get("position_xyz_m")
            positions.append({"position": apply_transform(self.transform, xyz)})
        return {
            "exploration_time": max(0.0, float(exploration_time)),
            "detected_danger_sources": positions,
            "coordinate_provenance": self.provenance(),
        }


def atomic_write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _pose_from_odom(message: Any) -> Pose:
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    return Pose((float(position.x), float(position.y), float(position.z)),
                (float(orientation.x), float(orientation.y), float(orientation.z), float(orientation.w)),
                float(message.header.stamp.to_sec()))


class RosWorldResultCoordinateNode:
    def __init__(self, rospy: Any, Odometry: Any, String: Any) -> None:
        run_id = str(os.environ.get("WORLD_RESULT_RUN_ID") or os.environ.get("STARTUP_ANCHOR_RUN_ID", "")).strip()
        configured_run_id = str(os.environ.get("STARTUP_ANCHOR_RUN_ID", "")).strip()
        if not run_id or configured_run_id != run_id:
            raise RuntimeError("run_identity_must_match_startup_anchor_run_id")
        self.rospy = rospy
        self.authority = ResultCoordinateAuthority(run_id, explicit_spawn_from_environment())
        self.result_path = pathlib.Path(rospy.get_param("~result_path", "results/detected_danger.json"))
        self.status_path = pathlib.Path(rospy.get_param("~status_path", "debug/rgbd_danger_perception/world_result_coordinate_status.json"))
        self.tracks: List[Dict[str, Any]] = []
        rospy.Subscriber(RAW_ODOM_TOPIC, Odometry, self.raw_cb, queue_size=1)
        rospy.Subscriber(GATE_STATUS_TOPIC, String, self.gate_cb, queue_size=10)
        rospy.Subscriber(TRACK_TOPIC, String, self.track_cb, queue_size=10)
        rospy.on_shutdown(self.persist_status)
        self.persist_pending_result()
        self.persist_status()

    def persist_status(self) -> None:
        try:
            atomic_write_json(self.status_path, self.authority.provenance())
        except Exception as error:
            self.rospy.logerr("world result status write failed: %s", error)

    def persist_pending_result(self) -> None:
        try:
            atomic_write_json(self.result_path, self.authority.pending_result_document())
        except OSError as error:
            self.rospy.logerr("world result pending write failed: %s", error)

    def persist_result_if_ready(self) -> None:
        if not self.authority.ready():
            return
        try:
            elapsed = self.authority.raw_elapsed_duration_sec()
            atomic_write_json(self.result_path, self.authority.result_document(self.tracks, elapsed))
        except (ValueError, RuntimeError, OSError) as error:
            self.rospy.logerr("world result write rejected: %s", error)

    def raw_cb(self, message: Any) -> None:
        self.authority.observe_first_raw(_pose_from_odom(message))
        self.persist_status()
        self.persist_result_if_ready()

    def gate_cb(self, message: Any) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(status, dict):
            self.authority.observe_gate_status(status)
            self.persist_status()
            self.persist_result_if_ready()

    def track_cb(self, message: Any) -> None:
        try:
            snapshot = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if snapshot.get("schema_version") != 1 or snapshot.get("frame_id") != ODOM_FRAME:
            return
        tracks = snapshot.get("tracks")
        if not isinstance(tracks, list):
            return
        self.tracks = [row for row in tracks if isinstance(row, dict) and row.get("state") == "CONFIRMED"]
        self.persist_result_if_ready()


def main() -> int:
    import rospy
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    rospy.init_node("world_result_coordinate", anonymous=False)
    RosWorldResultCoordinateNode(rospy, Odometry, String)
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
