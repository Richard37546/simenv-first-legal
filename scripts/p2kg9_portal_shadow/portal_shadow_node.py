#!/usr/bin/env python3
"""Default-off P2K-G9 passive Portal Shadow; audit topics only."""
from __future__ import annotations

import json
import os
import queue
import resource
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# The frozen Ray Evidence Core is an audit asset under scripts/, not under
# the repository root. Keep this import independent from production launch.
sys.path.insert(0, str(ROOT / "scripts" / "l3v_ray_evidence_shadow"))
sys.path.insert(0, str(HERE))
from ray_evidence_core import RayEvidenceCore  # noqa: E402
from p2kg9_portal_core import FrozenPortalMethod  # noqa: E402
from portal_audit_writer import PortalAuditWriter  # noqa: E402
from event_journal import GlobalEventJournal  # noqa: E402
from p2kg11_portal_frame_contract import (  # noqa: E402
    PORTAL_FRAME_CONTRACT,
    portal_frame_payload_v1,
)


TOPICS = {
    "candidate": "/audit/p2kg9/portal_candidate",
    "status": "/audit/p2kg9/portal_status",
    "journal": "/audit/p2kg9/event_journal",
    "frame": "/audit/p2kg11/portal_frame",
}
COMPUTE_QUEUE_MAX = 128
WRITER_QUEUE_MAX = 64
PORTAL_FRAME_CONTRACT_VERSION = PORTAL_FRAME_CONTRACT
# The frame topic has a bounded asynchronous publisher queue. This value does
# not constitute a delivery guarantee; legacy candidate-topic behavior stays
# unchanged.
PORTAL_FRAME_PUBLISHER_QUEUE_SIZE = 2


def portal_frame_payload(run_id: str, frame_sequence: int, source_stamp: float, frame_id: str,
                         outputs: list[dict]) -> str:
    """Serialize both side results from one Portal source frame as one contract."""
    return portal_frame_payload_v1(run_id, frame_sequence, source_stamp, frame_id, outputs)


class PortalFramePublisher:
    """Audit-only atomic Portal frame publisher with attempt diagnostics."""

    def __init__(self, publisher, run_id: str, counts: Counter, lock: threading.Lock, emit) -> None:
        self.publisher = publisher
        self.run_id = run_id
        self.counts = counts
        self.lock = lock
        self.emit = emit

    def publish(self, frame_sequence: int, source_stamp: float, frame_id: str,
                outputs: list[dict]) -> str | None:
        try:
            message = portal_frame_payload(
                self.run_id, frame_sequence, source_stamp, frame_id, outputs
            )
        except (TypeError, ValueError) as exc:
            with self.lock:
                self.counts["serialization_errors"] += 1
            self.emit(
                "portal_frame_serialization_error",
                frame_sequence=frame_sequence,
                source_stamp=source_stamp,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            return None
        with self.lock:
            self.counts["frame_messages_serialized"] += 1
            self.counts["frame_publish_attempted"] += 1
            self.counts.setdefault("first_published_sequence", frame_sequence)
            self.counts["last_published_sequence"] = frame_sequence
        try:
            self.publisher.publish(String(data=message))
        except Exception as exc:
            with self.lock:
                self.counts["publish_exceptions"] += 1
            self.emit(
                "portal_frame_publish_exception",
                frame_sequence=frame_sequence,
                source_stamp=source_stamp,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            return None
        return message


class PortalShadow:
    def __init__(self, out: Path, run_id: str) -> None:
        self.out = out
        self.run_id = run_id
        self.core = RayEvidenceCore()
        self.portal = FrozenPortalMethod()
        self.tf_listener = tf.TransformListener()
        self.events: queue.Queue = queue.Queue(maxsize=COMPUTE_QUEUE_MAX)
        self.lock = threading.Lock()
        self.stop_requested = False
        self.callback_event_id = 0
        self.frame_sequence = 0
        self.counts: Counter = Counter({
            "source_frames_received": 0,
            "frame_messages_serialized": 0,
            "frame_publish_attempted": 0,
            "serialization_errors": 0,
            "publish_exceptions": 0,
        })
        self.last_header_seq = None
        self.header_seq_gaps: list[dict] = []
        self.publishers = {
            key: rospy.Publisher(topic, String,
                                 queue_size=PORTAL_FRAME_PUBLISHER_QUEUE_SIZE if key == "frame" else 1,
                                 latch=False)
            for key, topic in TOPICS.items()
        }
        self.journal = GlobalEventJournal(out / "event_journal.jsonl", lambda data: self.publishers["journal"].publish(String(data=data)))
        self.frame_publisher = PortalFramePublisher(
            self.publishers["frame"], self.run_id, self.counts, self.lock, self._emit
        )
        self.writer = PortalAuditWriter(out, WRITER_QUEUE_MAX, self._emit)
        rospy.Subscriber("/team/livox/scan_cloud_filtered", PointCloud2, self.cloud_callback, queue_size=COMPUTE_QUEUE_MAX)
        self.worker = threading.Thread(target=self._loop, name="p2kg9-portal-compute", daemon=True)
        self.worker.start()
        rospy.Timer(rospy.Duration(1.0), self.heartbeat)

    def _emit(self, event_type: str, **fields) -> int:
        return self.journal.emit(event_type, **fields)

    def cloud_callback(self, msg: PointCloud2) -> None:
        receipt = time.monotonic()
        stamp = msg.header.stamp.to_sec()
        self._emit("cloud_callback_enter", source_stamp=stamp, receipt_monotonic_time=receipt, portal_track_version=self.portal.track_version)
        with self.lock:
            self.callback_event_id += 1
            callback_id = self.callback_event_id
            self.counts["cloud_received"] += 1
            if isinstance(msg.header.seq, int) and self.last_header_seq is not None and msg.header.seq > self.last_header_seq + 1:
                self.header_seq_gaps.append({"after": self.last_header_seq, "received": msg.header.seq, "missing": msg.header.seq - self.last_header_seq - 1})
            self.last_header_seq = msg.header.seq
        self._emit("cloud_received", source_stamp=stamp, receipt_monotonic_time=receipt, callback_event_id=callback_id, header_seq=msg.header.seq, input_state_version=self.frame_sequence, portal_track_version=self.portal.track_version, queue_size=self.events.qsize())
        try:
            self.events.put_nowait((callback_id, msg, receipt, stamp))
        except queue.Full:
            with self.lock:
                self.counts["queue_overflow"] += 1
            self._emit("compute_queue_overflow", source_stamp=stamp, receipt_monotonic_time=receipt, callback_event_id=callback_id, input_state_version=self.frame_sequence, portal_track_version=self.portal.track_version, queue_size=self.events.qsize(), failure_reason="COMPUTE_QUEUE_FULL")

    def _loop(self) -> None:
        while not self.stop_requested or not self.events.empty():
            try:
                callback_id, msg, receipt, stamp = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            started = time.monotonic()
            with self.lock:
                input_version = self.frame_sequence
            row = {"callback_event_id": callback_id, "source_stamp": stamp, "receipt_monotonic_time": receipt, "processing_start": started, "input_state_version": input_version, "portal_track_version": self.portal.track_version}
            try:
                self._process_cloud(msg, row)
            except Exception as exc:
                with self.lock:
                    self.counts["processing_failure"] += 1
                row["failure_reason"] = repr(exc)
                self._emit("processing_failure", **row)
            finally:
                row["processing_end"] = time.monotonic()
                self._emit("cloud_processing_complete", **row)
                self.events.task_done()

    def _process_cloud(self, msg: PointCloud2, event: dict) -> None:
        source_frame = msg.header.frame_id.lstrip("/")
        try:
            transform_time = self.tf_listener.getLatestCommonTime("base", source_frame)
            translation, quaternion = self.tf_listener.lookupTransform("base", source_frame, transform_time)
        except Exception as exc:
            with self.lock:
                self.counts["tf_failure"] += 1
            event.update({"event_type": "TF_FAILURE", "failure_reason": repr(exc)})
            self._emit("tf_failure", **event)
            return
        self._emit("tf_success", source_stamp=msg.header.stamp.to_sec(), receipt_monotonic_time=event["receipt_monotonic_time"], processing_start=event["processing_start"], input_state_version=event["input_state_version"], portal_track_version=self.portal.track_version)
        raw = np.asarray(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float64)
        if raw.size == 0:
            raw = raw.reshape((0, 3))
        matrix = tf.transformations.quaternion_matrix(quaternion)
        points_base = raw @ matrix[:3, :3].T + np.asarray(translation, dtype=np.float64)
        rays, metadata = self.core.observe_sparse(points_base, msg.header.stamp.to_sec(), transform_time.to_sec(), source_frame)
        self._emit("core_processed", source_stamp=msg.header.stamp.to_sec(), input_state_version=event["input_state_version"], portal_track_version=self.portal.track_version, valid_ray_count=int(len(rays)))
        outputs = self.portal.process(rays, msg.header.stamp.to_sec(), metadata["source_provenance"])
        self._emit("portal_processed", source_stamp=msg.header.stamp.to_sec(), input_state_version=event["input_state_version"], portal_track_version=self.portal.track_version, candidate_count=len(outputs))
        ended = time.monotonic()
        with self.lock:
            self.frame_sequence += 1
            sequence = self.frame_sequence
            self.counts["tf_success"] += 1
            self.counts["core_processed"] += 1
            self.counts["portal_processed"] += 1
            self.counts["source_frames_received"] += 1
        payload = {"frame_sequence": sequence, "source_stamp": msg.header.stamp.to_sec(), "transform_stamp": transform_time.to_sec(), "source_frame": source_frame, "target_frame": "base", "ray_origin_base": [0.0, 0.0, 0.0], "portal_candidates": outputs, "input_quality_flags": [], "core_valid_ray_count": int(len(rays)), "processing_ms": (ended - event["processing_start"]) * 1000.0}
        enqueued = self.writer.enqueue(sequence, payload)
        with self.lock:
            self.counts["audit_enqueued"] += int(enqueued)
            self.counts["queue_overflow"] += int(not enqueued)
            self.counts["candidate_published"] += len(outputs)
        self.frame_publisher.publish(sequence, msg.header.stamp.to_sec(), "base", outputs)
        for output in outputs:
            self.publishers["candidate"].publish(String(data=json.dumps(output, sort_keys=True)))
        self._emit("candidate_published", source_stamp=msg.header.stamp.to_sec(), input_state_version=event["input_state_version"], portal_track_version=self.portal.track_version, candidate_count=len(outputs))
        if enqueued:
            self._emit("audit_enqueued", source_stamp=msg.header.stamp.to_sec(), input_state_version=event["input_state_version"], portal_track_version=self.portal.track_version, queue_size=self.writer.q.qsize())
        event.update({"tf_stamp": transform_time.to_sec(), "valid_ray_count": int(len(rays)), "candidate_count": len(outputs), "audit_enqueued": enqueued, "portal_track_version": self.portal.track_version, "processing_ms": payload["processing_ms"]})

    def heartbeat(self, _timer) -> None:
        vmrss = 0
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                vmrss = int(line.split()[1])
                break
        payload = {"audit_only": True, "portal_or_control_output": False, "cloud_received": self.counts["cloud_received"], "compute_queue": self.events.qsize(), "writer_queue": self.writer.q.qsize(), "vmrss_kb": vmrss, "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        self.publishers["status"].publish(String(data=json.dumps(payload, sort_keys=True)))

    def close(self) -> None:
        self._emit("shutdown_requested", source_stamp=None, input_state_version=self.frame_sequence, portal_track_version=self.portal.track_version, queue_size=self.events.qsize())
        self.stop_requested = True
        self.worker.join(timeout=30.0)
        self._emit("compute_worker_stopped", source_stamp=None, input_state_version=self.frame_sequence, portal_track_version=self.portal.track_version, queue_size=self.events.qsize(), worker_alive=self.worker.is_alive())
        writer = self.writer.close()
        accounting = dict(self.counts)
        accounting.update({"cloud_header_seq_gaps": self.header_seq_gaps, "audit_persisted": writer["persisted_frames"], "writer_queue_overflow": writer["queue_overflow"], "writer_errors": writer["writer_errors"], "compute_queue_remaining": self.events.qsize(), "compute_worker_alive": self.worker.is_alive(), "publisher_emitted_count": "PUBLISHER_EMITTED_COUNT_UNKNOWN"})
        (self.out / "frame_accounting.json").write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (self.out / "queue_and_shutdown.json").write_text(json.dumps({**writer, **accounting, "status": "STOPPED"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._emit("shutdown_complete", source_stamp=None, input_state_version=self.frame_sequence, portal_track_version=self.portal.track_version, queue_size=self.events.qsize())
        self.journal.close()


def main() -> None:
    rospy.init_node("p2kg9_passive_portal_shadow", anonymous=True)
    run_id = rospy.get_param("~run_id", os.environ.get("P2KG9_RUN_ID", time.strftime("online_%Y%m%d_%H%M%S")))
    default_root = ROOT / "debug/odom_accuracy_audit_v1/p2kg9_current_door_portal_shadow_048/online_run"
    out_root = Path(os.environ.get("P2KG9_OUTPUT_ROOT", str(default_root))).resolve()
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_manifest.json").write_text(json.dumps({"run_id": run_id, "audit_only": True, "default_started": False, "truth_or_world_input": False, "production_code_modified": False}, indent=2) + "\n", encoding="utf-8")
    (out / "shadow_topic_contract.json").write_text(json.dumps({"subscribes": ["/team/livox/scan_cloud_filtered", "/tf", "/tf_static"], "publishes": list(TOPICS.values()), "portal_frame_contract": PORTAL_FRAME_CONTRACT_VERSION, "legacy_candidate_topic": TOPICS["candidate"], "legacy_candidate_topic_downstream_forbidden": True, "forbidden_outputs": ["/cmd_vel", "/cmd_vel_raw", "entry pose", "target", "state event", "production grid"], "truth_inputs": False, "production_parameter_writes": False}, indent=2) + "\n", encoding="utf-8")
    node = PortalShadow(out, run_id)
    rospy.on_shutdown(node.close)
    rospy.spin()


if __name__ == "__main__":
    main()
