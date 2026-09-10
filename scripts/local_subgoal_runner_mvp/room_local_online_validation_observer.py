#!/usr/bin/env python3
"""Validation-only, authority-free command/odometry correlation sidecar."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


CRITICAL_STREAMS = ("decision", "command_slice", "cmd_raw", "cmd_final", "odom")
EPS = 1e-9
EVIDENCE_STATUS_TOPIC = "/audit/room_local_validation/evidence_status"


@dataclass
class SliceRecord:
    command_slice_id: str
    start_ros_time_sec: float
    intended_duration_sec: float
    requested_v: float
    requested_w: float
    raw: List[Dict[str, Any]] = field(default_factory=list)
    final: List[Dict[str, Any]] = field(default_factory=list)
    odom: List[Dict[str, Any]] = field(default_factory=list)
    end_ros_time_sec: Optional[float] = None


class Correlator:
    """Pure deterministic interval correlator used by the ROS sidecar and tests."""

    def __init__(self) -> None:
        self.slices: Dict[str, SliceRecord] = {}
        self.sequence = {name: 0 for name in ("cmd_raw", "cmd_final", "odom")}

    def start(self, event: Dict[str, Any]) -> None:
        self.slices[str(event["command_slice_id"])] = SliceRecord(
            str(event["command_slice_id"]), float(event["ros_time_sec"]),
            float(event["intended_duration_sec"]), float(event["requested_v"]), float(event["requested_w"]),
        )

    def end(self, event: Dict[str, Any]) -> None:
        record = self.slices.get(str(event["command_slice_id"]))
        if record is not None:
            record.end_ros_time_sec = float(event["end_ros_time_sec"])

    def observe(self, stream: str, ros_time_sec: float, payload: Dict[str, Any]) -> None:
        self.sequence[stream] += 1
        row = {"sequence": self.sequence[stream], "ros_time_sec": float(ros_time_sec), **payload}
        for record in self.slices.values():
            end = record.end_ros_time_sec
            if end is None:
                end = record.start_ros_time_sec + record.intended_duration_sec
            if record.start_ros_time_sec <= float(ros_time_sec) <= end:
                getattr(record, "raw" if stream == "cmd_raw" else "final" if stream == "cmd_final" else "odom").append(row)

    def result(self, command_slice_id: str) -> Dict[str, Any]:
        record = self.slices[str(command_slice_id)]
        missing = [name for name, values in (("cmd_raw", record.raw), ("cmd_final", record.final), ("odom", record.odom)) if not values]
        return {
            "command_slice_id": record.command_slice_id,
            "requested_v": record.requested_v, "requested_w": record.requested_w,
            "raw": record.raw, "final": record.final, "odom": record.odom,
            "attribution_status": "COMPLETE" if not missing else "MISSING_CRITICAL_STREAM",
            "missing_critical_streams": missing,
        }


def intent_identity(intent: Any) -> Optional[str]:
    """Identity is evidence only: target plus the intent's recorded route anchor."""
    if not isinstance(intent, dict):
        return None
    keys = {key: intent.get(key) for key in ("target_key", "anchor_odom_xy", "anchor_tangent_odom_rad")}
    return json.dumps(keys, sort_keys=True, separators=(",", ":"))


class ValidationContractMonitor:
    """Pure observer contract; it never selects, changes, or publishes motion."""

    def __init__(self, run_id: str) -> None:
        self.run_id = str(run_id)
        self.correlator = Correlator()
        self.events: List[Dict[str, Any]] = []
        self.runner_invocation_id: Optional[str] = None
        self.control_failure_reason: Optional[str] = None
        self.evidence_status = "ONLINE_EVIDENCE_COMPLETE"
        self.evidence_status_updates: List[Dict[str, Any]] = []
        self.audit_discrepancies: List[Dict[str, Any]] = []
        self.last_decision: Optional[Dict[str, Any]] = None
        self.no_productive_without_fresh = False
        self.rotate_requires_fresh: Optional[Dict[str, int]] = None
        self.closed_slices: set = set()
        self.consumed_intents: set = set()

    def record_control_failure(self, reason: str) -> bool:
        """Record runner-native control failure without granting observer abort authority."""
        if self.control_failure_reason is not None:
            return False
        self.control_failure_reason = str(reason)
        return True

    def mark_evidence_insufficient(
        self,
        reason: str,
        *,
        event: Optional[Dict[str, Any]] = None,
        command_slice_id: Optional[str] = None,
        missing_streams: Optional[List[str]] = None,
        attribution_status: str = "UNRESOLVED",
        scope: str = "RUN_INVOCATION",
    ) -> bool:
        """Publishable evidence fact only; it cannot stop the current command slice."""
        if self.evidence_status == "ONLINE_EVIDENCE_INSUFFICIENT":
            return False
        row = event if isinstance(event, dict) else {}
        self.evidence_status = "ONLINE_EVIDENCE_INSUFFICIENT"
        self.evidence_status_updates.append({
            "event_type": "ONLINE_EVIDENCE_STATUS",
            "run_id": self.run_id,
            "runner_invocation_id": row.get("runner_invocation_id", self.runner_invocation_id),
            "command_slice_id": command_slice_id,
            "timestamp": row.get("ros_time_sec", row.get("end_ros_time_sec")),
            "evidence_status": self.evidence_status,
            "reason": str(reason),
            "missing_streams": list(missing_streams or []),
            "attribution_status": str(attribution_status),
            "first_detected_event": row.get("event_type"),
            "scope": str(scope),
        })
        return True

    def record_audit_discrepancy(self, reason: str, event: Dict[str, Any]) -> None:
        self.audit_discrepancies.append({
            "event_type": "AUDIT_DISCREPANCY",
            "run_id": self.run_id,
            "runner_invocation_id": event.get("runner_invocation_id"),
            "reason": str(reason),
            "source_event_type": event.get("event_type"),
        })

    def _fresh_decision(self, event: Dict[str, Any]) -> bool:
        baseline = self.rotate_requires_fresh or {}
        for key in ("input_acquisition_sequence", "grid_status_acquisition_sequence", "astar_evaluation_sequence", "path_anchor_sequence"):
            if not isinstance(event.get(key), int) or int(event[key]) <= int(baseline.get(key, -1)):
                return False
        return True

    def _finalize_closed_slices(self) -> None:
        for slice_id, record in self.correlator.slices.items():
            if record.end_ros_time_sec is None or slice_id in self.closed_slices:
                continue
            self.closed_slices.add(slice_id)
            result = self.correlator.result(slice_id)
            if result["attribution_status"] != "COMPLETE":
                self.mark_evidence_insufficient(
                    "CRITICAL_TELEMETRY_LOSS:" + ",".join(result["missing_critical_streams"]),
                    event={"event_type": "COMMAND_SLICE_END", "end_ros_time_sec": record.end_ros_time_sec},
                    command_slice_id=slice_id,
                    missing_streams=result["missing_critical_streams"],
                    attribution_status=result["attribution_status"],
                    scope="COMMAND_SLICE",
                )

    def _observe_intent(self, event: Dict[str, Any]) -> None:
        after = event.get("orientation_intent_after")
        identity = intent_identity(after)
        if identity is None:
            return
        if identity in self.consumed_intents and not bool(after.get("consumed")):
            self.record_audit_discrepancy("ORIENTATION_INTENT_REARM_WITHOUT_MATERIAL_ROUTE_CHANGE", event)
        if bool(after.get("consumed")):
            self.consumed_intents.add(identity)

    def on_event(self, event: Dict[str, Any]) -> bool:
        if event.get("run_id") not in (None, self.run_id):
            return False
        self.events.append(dict(event))
        if event.get("runner_invocation_id") is not None:
            self.runner_invocation_id = str(event.get("runner_invocation_id"))
        event_type = event.get("event_type")
        if event_type == "VALIDATION_CONTRACT_VIOLATION":
            self.record_control_failure(str(event.get("reason") or "ROOM_LOCAL_ONLINE_VALIDATION_ABORT"))
        elif event_type == "ROOM_LOCAL_DECISION":
            if self.rotate_requires_fresh is not None:
                if not self._fresh_decision(event):
                    self.record_audit_discrepancy("PHASE3_FRESH_REPLAN_CONTRACT_VIOLATION", event)
                else:
                    self.rotate_requires_fresh = None
            self.last_decision = dict(event)
            self.no_productive_without_fresh = event.get("phase2_productivity_status") == "NO_PRODUCTIVE_TRANSLATION"
        elif event_type == "PHASE3_RECOVERYSET_RESULT":
            self._observe_intent(event)
        elif event_type == "COMMAND_SLICE_START":
            self._finalize_closed_slices()
            v, w = float(event.get("requested_v", 0.0)), float(event.get("requested_w", 0.0))
            if abs(v) > EPS and self.last_decision is None:
                self.record_audit_discrepancy("MISSING_CRITICAL_DECISION", event)
            if self.rotate_requires_fresh is not None and (abs(v) > EPS or abs(w) > EPS):
                self.record_audit_discrepancy("PHASE3_ONE_SLICE_CONTRACT_VIOLATION", event)
            if self.no_productive_without_fresh and v > EPS:
                self.record_audit_discrepancy("NONPRODUCTIVE_TRANSLATION_FALLBACK_VIOLATION", event)
            self.correlator.start(event)
            if abs(v) <= EPS and abs(w) > EPS:
                decision = self.last_decision or {}
                self.rotate_requires_fresh = {
                    key: int(decision.get(key, -1))
                    for key in ("input_acquisition_sequence", "grid_status_acquisition_sequence", "astar_evaluation_sequence", "path_anchor_sequence")
                }
        elif event_type == "COMMAND_SLICE_END":
            self.correlator.end(event)
        elif event_type == "RUNNER_INVOCATION_END":
            self.finalize()
        return self.control_failure_reason is not None

    def observe(self, stream: str, ros_time_sec: float, payload: Dict[str, Any]) -> None:
        self.correlator.observe(stream, ros_time_sec, payload)

    def finalize(self) -> None:
        self._finalize_closed_slices()
        if self.rotate_requires_fresh is not None:
            self.record_audit_discrepancy("PHASE3_FRESH_REPLAN_MISSING_AT_RUN_END", {
                "event_type": "RUNNER_INVOCATION_END",
            })

    def result(self) -> Dict[str, Any]:
        self.finalize()
        return {
            "run_id": self.run_id,
            "events": self.events,
            "slices": [self.correlator.result(key) for key in self.correlator.slices],
            "control_contract_status": "CONTROL_CONTRACT_FAILURE" if self.control_failure_reason else "CONTROL_CONTRACT_PASS",
            "control_failure_reason": self.control_failure_reason,
            "online_evidence_status": self.evidence_status,
            "evidence_status_updates": self.evidence_status_updates,
            "audit_discrepancies": self.audit_discrepancies,
            "readiness": "PASS" if self.control_failure_reason is None and self.evidence_status == "ONLINE_EVIDENCE_COMPLETE" else "FAIL",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--event-topic", default="/audit/room_local_validation/event")
    parser.add_argument("--evidence-status-topic", default=EVIDENCE_STATUS_TOPIC)
    parser.add_argument("--ready-file", default="")
    args = parser.parse_args()
    import rospy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    monitor = ValidationContractMonitor(args.run_id)
    rospy.init_node("room_local_online_validation_observer", anonymous=True)
    evidence_pub = rospy.Publisher(args.evidence_status_topic, String, queue_size=10)
    published_evidence_count = 0

    def publish_new_evidence_status() -> None:
        nonlocal published_evidence_count
        while published_evidence_count < len(monitor.evidence_status_updates):
            payload = dict(monitor.evidence_status_updates[published_evidence_count])
            payload["ros_time_sec"] = rospy.Time.now().to_sec()
            evidence_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
            published_evidence_count += 1

    def flush() -> None:
        monitor.finalize()
        publish_new_evidence_status()
        out.write_text(json.dumps(monitor.result(), indent=2, sort_keys=True) + "\n")

    def event_cb(msg: Any) -> None:
        try:
            event = json.loads(msg.data)
            if event.get("run_id") != args.run_id:
                return
            monitor.on_event(event)
            publish_new_evidence_status()
        except Exception as exc:
            monitor.events.append({"event_type": "OBSERVER_EVENT_PARSE_ERROR", "error": type(exc).__name__})
            monitor.mark_evidence_insufficient(
                "OBSERVER_EVENT_PARSE_ERROR:%s" % type(exc).__name__,
                event={"event_type": "OBSERVER_EVENT_PARSE_ERROR"},
                scope="RUN_INVOCATION",
            )
            publish_new_evidence_status()

    def twist_cb(stream: str):
        def callback(msg: Any) -> None:
            monitor.observe(stream, rospy.Time.now().to_sec(), {"v": msg.linear.x, "w": msg.angular.z})
        return callback

    def odom_cb(msg: Any) -> None:
        pose = msg.pose.pose.position
        monitor.observe("odom", rospy.Time.now().to_sec(), {"x": pose.x, "y": pose.y})

    rospy.Subscriber(args.event_topic, String, event_cb, queue_size=100)
    rospy.Subscriber("/cmd_vel_raw", Twist, twist_cb("cmd_raw"), queue_size=100)
    rospy.Subscriber("/cmd_vel", Twist, twist_cb("cmd_final"), queue_size=100)
    rospy.Subscriber("/team/livox/icp_odom_gated", Odometry, odom_cb, queue_size=100)
    if args.ready_file:
        Path(args.ready_file).write_text(json.dumps({"run_id": args.run_id, "status": "READY"}) + "\n")
    rospy.on_shutdown(flush)
    rospy.spin()
    flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
