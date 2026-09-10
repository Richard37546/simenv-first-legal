#!/usr/bin/env python3
"""Authority-free startup-anchor evidence recorder.

This process is deliberately an archive producer only.  It has no publishers
and its TEST_ONLY_ORACLE rows are written to a separate file so no navigation,
localization, TF, control, or result path can consume Gazebo truth at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Optional


ROOT = pathlib.Path("/home/richard/simenv_official_clean")
SCHEMA_VERSION = "startup_anchor_evidence_v1"
SPAWN_KEYS = ("ROBOT_X", "ROBOT_Y", "ROBOT_Z", "ROBOT_YAW")


def _json_default(value: Any) -> str:
    return str(value)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_spawn_provenance(environment: Dict[str, str]) -> Dict[str, Any]:
    """Record only explicitly supplied launch values; never synthesize defaults.

    This is audit metadata.  The recorder is deliberately not allowed to infer
    a spawn pose from Gazebo, scene files, or its own fallback values.
    """
    source = str(environment.get("STARTUP_ANCHOR_SPAWN_SOURCE", ""))
    values = {key: str(environment.get(key, "")) for key in SPAWN_KEYS}
    complete = all(values.values())
    explicit = source == "EXPLICIT_AUDIT_WRAPPER_ENV"
    return {
        "status": "RECORDED_EXPLICIT_EFFECTIVE_INPUT" if complete and explicit else "UNAVAILABLE",
        "source": source or "UNSPECIFIED",
        "values": values,
        "recording_rule": "environment values only; no defaulting or Gazebo-derived correction",
    }


class SimTimeTracker:
    """Keep `/clock` time distinct from message header and host wall time."""

    def __init__(self, run_id: str) -> None:
        self.run_id = str(run_id)
        self.latest_sim_time_sec: Optional[float] = None
        self.epoch_index = 1

    @property
    def epoch_id(self) -> str:
        return "%s:sim-%04d" % (self.run_id, self.epoch_index)

    def callback_fields(self) -> Dict[str, Any]:
        if self.latest_sim_time_sec is None:
            return {
                "sim_time_at_callback_sec": None,
                "sim_time_status": "UNAVAILABLE_NO_CLOCK",
                "sim_time_epoch_id": None,
            }
        return {
            "sim_time_at_callback_sec": self.latest_sim_time_sec,
            "sim_time_status": "AVAILABLE",
            "sim_time_epoch_id": self.epoch_id,
        }

    def observe_clock(self, value: float) -> Dict[str, Any]:
        sim_time = float(value)
        if not math.isfinite(sim_time):
            return {"accepted": False, "reason": "NONFINITE_CLOCK", **self.callback_fields()}
        previous = self.latest_sim_time_sec
        regressed = previous is not None and sim_time < previous
        if regressed:
            self.epoch_index += 1
        self.latest_sim_time_sec = sim_time
        return {
            "accepted": True,
            "previous_sim_time_sec": previous,
            "clock_regressed": regressed,
            "sim_time_sec": sim_time,
            **self.callback_fields(),
        }


def nearest_oracle_samples(rows: Iterable[Dict[str, Any]], sim_time_sec: float, epoch_id: str) -> Dict[str, Optional[Dict[str, Any]]]:
    """Return bracketing TEST_ONLY oracle rows for one simulation-time epoch."""
    earlier = None
    later = None
    for row in rows:
        if row.get("provenance") != "TEST_ONLY_ORACLE" or row.get("sim_time_epoch_id") != epoch_id:
            continue
        value = row.get("sim_time_at_callback_sec")
        if not isinstance(value, (int, float)):
            continue
        if value <= sim_time_sec and (earlier is None or value > earlier["sim_time_at_callback_sec"]):
            earlier = row
        if value >= sim_time_sec and (later is None or value < later["sim_time_at_callback_sec"]):
            later = row
    return {"earlier_or_equal": earlier, "later_or_equal": later}


class ArchiveWriter:
    """Run-scoped append-only archive writer, independent of rospy."""

    def __init__(self, run_id: str, archive_dir: pathlib.Path) -> None:
        self.run_id = str(run_id)
        self.archive_dir = pathlib.Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = 0
        self.events_path = self.archive_dir / "production_evidence.jsonl"
        self.oracle_path = self.archive_dir / "test_only_gazebo_oracle.jsonl"
        self.ready_path = self.archive_dir / "recorder.ready.json"
        self.final_path = self.archive_dir / "recorder.final.json"
        existing = self.archive_dir / "run_identity.json"
        if existing.exists():
            previous = json.loads(existing.read_text(encoding="utf-8"))
            if str(previous.get("run_id")) != self.run_id:
                raise RuntimeError("archive_run_id_mismatch")
        else:
            existing.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "run_id": self.run_id}, indent=2) + "\n", encoding="utf-8")

    def _append(self, path: pathlib.Path, row: Dict[str, Any]) -> Dict[str, Any]:
        self.sequence += 1
        stamped = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "event_sequence": self.sequence,
            "wall_time_unix_sec": time.time(),
            **row,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped, sort_keys=True, default=_json_default) + "\n")
        return stamped

    def production_event(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        return self._append(self.events_path, {"event_type": str(event_type), "provenance": "PRODUCTION_OBSERVATION", **fields})

    def oracle_event(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        return self._append(self.oracle_path, {"event_type": str(event_type), "provenance": "TEST_ONLY_ORACLE", **fields})

    def write_provenance(self) -> None:
        relevant = [
            ROOT / "auto.sh",
            ROOT / "scripts/local_subgoal_runner_mvp/start_runtime_stack_tmux.sh",
            ROOT / "scripts/slam_bev_runtime/run_slam_bev_runtime.sh",
            ROOT / "scripts/l2_livox_icp_rtabmap_readiness/l2_livox_odom_gate.py",
            ROOT / "scripts/startup_anchor_evidence/startup_anchor_audit_recorder.py",
        ]
        def git(*args: str) -> str:
            try:
                return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
            except Exception:
                return "UNAVAILABLE"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "source_head": git("rev-parse", "HEAD"),
            "dirty_status": git("status", "--short"),
            "relevant_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in relevant if path.is_file()},
            "launch_config": {
                key: os.environ.get(key, "")
                for key in ("SEED", "GUI", "PAUSED", "ROBOT_X", "ROBOT_Y", "ROBOT_Z", "ROBOT_YAW", "UNITREE_CTRL_DT", "UNITREE_RL_POLICY", "BUILDING_WORLD_FILE")
            },
            "effective_spawn_provenance": effective_spawn_provenance(os.environ),
            "oracle_isolation": "TEST_ONLY_ORACLE; not published or consumed by production authority",
        }
        (self.archive_dir / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def mark_ready(self, ready_file: pathlib.Path, **timing: Any) -> None:
        row = self.production_event("AUDIT_RECORDER_READY", **timing)
        payload = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, "status": "READY", "event_sequence": row["event_sequence"]}
        self.ready_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        pathlib.Path(ready_file).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def finalize(self, reason: str, **timing: Any) -> None:
        row = self.production_event("MISSION_STOP_MARKER", reason=str(reason), **timing)
        self.final_path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "run_id": self.run_id, "status": "FINALIZED", "event_sequence": row["event_sequence"], "reason": str(reason)}, indent=2) + "\n", encoding="utf-8")

    def launcher_marker(self, event_type: str, **fields: Any) -> None:
        row = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, "provenance": "PRODUCTION_LAUNCH_METADATA", "event_type": str(event_type), "wall_time_unix_sec": time.time(), **fields}
        with (self.archive_dir / "launcher_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _stamp(msg: Any) -> float:
    return float(msg.header.stamp.to_sec())


def _pose(msg: Any) -> Dict[str, Any]:
    value = msg.pose.pose
    return {
        "position_xyz": [value.position.x, value.position.y, value.position.z],
        "quaternion_xyzw": [value.orientation.x, value.orientation.y, value.orientation.z, value.orientation.w],
    }


def odom_evidence_row(msg: Any, sim_time: SimTimeTracker) -> Dict[str, Any]:
    """Preserve header stamp and callback-time simulation state separately."""
    stamp = _stamp(msg)
    return {
        "message_stamp_sec": stamp,
        "header_stamp_sec": stamp,
        "header_frame_id": msg.header.frame_id,
        "child_frame_id": msg.child_frame_id,
        "pose": _pose(msg),
        **sim_time.callback_fields(),
    }


def run_ros(args: argparse.Namespace) -> int:
    import rospy
    from gazebo_msgs.msg import ModelStates
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import Bool, String

    writer = ArchiveWriter(args.run_id, pathlib.Path(args.archive_dir))
    writer.write_provenance()
    sim_time = SimTimeTracker(args.run_id)
    seen = set()

    def first(event_type: str, **fields: Any) -> None:
        if event_type not in seen:
            seen.add(event_type)
            writer.production_event(event_type, **fields)

    def clock_cb(msg: Clock) -> None:
        update = sim_time.observe_clock(float(msg.clock.to_sec()))
        if not update["accepted"]:
            writer.oracle_event("CLOCK_SAMPLE_REJECTED", **update)
            return
        writer.oracle_event("CLOCK_SAMPLE", **update)
        if update["clock_regressed"]:
            writer.oracle_event("SIM_TIME_DISCONTINUITY", **update)
        first("FIRST_CLOCK", **update)

    def raw_odom_cb(msg: Odometry) -> None:
        row = odom_evidence_row(msg, sim_time)
        first("FIRST_RAW_ICP_ODOM", **row)
        writer.production_event("RAW_ICP_ODOM", **row)

    def gated_odom_cb(msg: Odometry) -> None:
        row = odom_evidence_row(msg, sim_time)
        first("FIRST_GATED_ODOM", **row)
        writer.production_event("GATED_ODOM", **row)

    def twist_cb(label: str):
        def callback(msg: Twist) -> None:
            row = {"linear_xyz": [msg.linear.x, msg.linear.y, msg.linear.z], "angular_xyz": [msg.angular.x, msg.angular.y, msg.angular.z], **sim_time.callback_fields()}
            first("FIRST_" + label, **row)
            if any(abs(value) > 1e-9 for value in row["linear_xyz"] + row["angular_xyz"]):
                first("FIRST_NONZERO_" + label, **row)
                first("MISSION_START_MARKER", source=label, **sim_time.callback_fields())
            writer.production_event(label, **row)
        return callback

    def world_cb(msg: ModelStates) -> None:
        if "a1_gazebo" not in msg.name:
            return
        pose = msg.pose[msg.name.index("a1_gazebo")]
        row = {"model_identity": "a1_gazebo", "model_name": "a1_gazebo", "position_xyz": [pose.position.x, pose.position.y, pose.position.z], "quaternion_xyzw": [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], **sim_time.callback_fields()}
        if "FIRST_GAZEBO_MODEL_STATE" not in seen:
            seen.add("FIRST_GAZEBO_MODEL_STATE")
            writer.oracle_event("FIRST_GAZEBO_MODEL_STATE", **row)
        writer.oracle_event("GAZEBO_MODEL_STATE", **row)

    def text_event_cb(source: str):
        def callback(msg: String) -> None:
            writer.production_event("EXTERNAL_AUDIT_EVENT", source=source, payload=msg.data, **sim_time.callback_fields())
        return callback

    def bool_event_cb(source: str):
        def callback(msg: Bool) -> None:
            writer.production_event("READINESS_STATUS", source=source, value=bool(msg.data), **sim_time.callback_fields())
            if bool(msg.data):
                first("FIRST_READY_" + source, **sim_time.callback_fields())
        return callback

    rospy.init_node("startup_anchor_audit_recorder", anonymous=True, disable_signals=True)
    rospy.Subscriber("/clock", Clock, clock_cb, queue_size=1000)
    rospy.Subscriber("/gazebo/model_states", ModelStates, world_cb, queue_size=1000)
    rospy.Subscriber("/team/livox/icp_odom_raw", Odometry, raw_odom_cb, queue_size=1000)
    rospy.Subscriber("/team/livox/icp_odom_gated", Odometry, gated_odom_cb, queue_size=1000)
    rospy.Subscriber("/cmd_vel_raw", Twist, twist_cb("CMD_VEL_RAW"), queue_size=1000)
    rospy.Subscriber("/cmd_vel", Twist, twist_cb("CMD_VEL"), queue_size=1000)
    rospy.Subscriber("/audit/startup_anchor/odom_epoch_event", String, text_event_cb("ODOM_EPOCH_AUDIT"), queue_size=1000)
    rospy.Subscriber("/audit/room_local_validation/event", String, text_event_cb("MISSION_EVENT"), queue_size=1000)
    rospy.Subscriber("/unitree/rl_mode_ready", Bool, bool_event_cb("STAIR_RL"), queue_size=1000)
    rospy.Subscriber("/imu_velocity_follower/status", String, text_event_cb("IMU_VELOCITY_FOLLOWER"), queue_size=1000)
    writer.mark_ready(pathlib.Path(args.ready_file), **sim_time.callback_fields())
    stop_reason = {"value": "ROS_SHUTDOWN"}
    def stop(signum: int, _frame: Any) -> None:
        stop_reason["value"] = "SIGNAL_%d" % int(signum)
        rospy.signal_shutdown(stop_reason["value"])
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        rate.sleep()
    writer.finalize(stop_reason["value"], **sim_time.callback_fields())
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--launcher-marker", default="")
    parser.add_argument("--paused", default="")
    args = parser.parse_args(argv)
    if not args.run_id or "/" in args.run_id:
        parser.error("invalid --run-id")
    if args.launcher_marker:
        ArchiveWriter(args.run_id, pathlib.Path(args.archive_dir)).launcher_marker(args.launcher_marker, paused=str(args.paused))
        return 0
    return run_ros(args)


if __name__ == "__main__":
    sys.exit(main())
