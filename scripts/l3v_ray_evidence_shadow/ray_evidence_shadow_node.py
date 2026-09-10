#!/usr/bin/env python3
"""P2K-G7R audit-only passive ray-evidence shadow.

This file is deliberately isolated from the production L3V node.  It only
subscribes, emits audit status, and writes evidence to its own debug tree.
"""
from __future__ import annotations

import json
import queue
import resource
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from ray_evidence_core import NavGridCompat, RayEvidenceCore
from sparse_persistence import SparseFrameWriter


ROOT = Path(__file__).resolve().parents[2]
AUDIT_TOPIC = "/audit/p2kg7r/ray_evidence_status"
COMPUTE_QUEUE_MAX = 256
WRITER_QUEUE_MAX = 64


class Shadow:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.core = RayEvidenceCore()
        self.nav = NavGridCompat()
        self.tf_listener = tf.TransformListener()
        self.events: queue.Queue = queue.Queue(maxsize=COMPUTE_QUEUE_MAX)
        self.writer = SparseFrameWriter(out, WRITER_QUEUE_MAX)
        self.stop_requested = False
        self.counter_lock = threading.Lock()
        self.callback_event_id = 0
        self.event_id = 0
        self.state_version = 0
        self.cloud_sequence = 0
        self.decay_publish_count = 0
        self.latest_cloud_stamp = None
        self.latest_odom_stamp = None
        self.counts: Counter = Counter()
        self.seq_gaps: list[dict] = []
        self.last_cloud_header_seq = None
        self.logs = {
            name: (out / name).open("a", encoding="utf-8")
            for name in (
                "event_journal.jsonl", "frame_summary.jsonl", "side_evidence_summary.jsonl",
                "nav_grid_parity.jsonl", "performance.jsonl", "process_resource.jsonl",
            )
        }
        self.publisher = rospy.Publisher(AUDIT_TOPIC, String, queue_size=1, latch=False)
        rospy.Subscriber("/team/livox/scan_cloud_filtered", PointCloud2, self.cloud_callback, queue_size=256)
        rospy.Subscriber("/team/livox/icp_odom_gated", Odometry, self.odom_callback, queue_size=256)
        rospy.Subscriber("/team/local_traversability_grid", OccupancyGrid, self.grid_callback, queue_size=256)
        self.worker = threading.Thread(target=self._compute_loop, name="p2kg7r-compute", daemon=True)
        self.worker.start()
        rospy.Timer(rospy.Duration(1.0), self.heartbeat)

    def _log(self, name: str, row: dict) -> None:
        self.logs[name].write(json.dumps(row, sort_keys=True) + "\n")
        self.logs[name].flush()

    @staticmethod
    def _stamp(msg) -> float | None:
        header = getattr(msg, "header", None)
        return header.stamp.to_sec() if header is not None and header.stamp else None

    def _enqueue(self, event_type: str, msg) -> None:
        receipt = time.monotonic()
        stamp = self._stamp(msg)
        header_seq = getattr(getattr(msg, "header", None), "seq", None)
        with self.counter_lock:
            self.callback_event_id += 1
            callback_event_id = self.callback_event_id
            self.counts[f"{event_type}_callback_received"] += 1
            if event_type == "cloud" and isinstance(header_seq, int):
                if self.last_cloud_header_seq is not None and header_seq > self.last_cloud_header_seq + 1:
                    self.seq_gaps.append({"after": self.last_cloud_header_seq, "received": header_seq, "missing": header_seq - self.last_cloud_header_seq - 1})
                self.last_cloud_header_seq = header_seq
        item = (callback_event_id, event_type, msg, receipt, stamp, header_seq)
        try:
            self.events.put_nowait(item)
        except queue.Full:
            with self.counter_lock:
                self.counts[f"{event_type}_compute_queue_overflow"] += 1
            self._log("event_journal.jsonl", {
                "callback_event_id": callback_event_id, "event_type": "QUEUE_OVERFLOW",
                "dropped_event_type": event_type, "ros_stamp": stamp, "header_seq": header_seq,
                "receipt_monotonic_time": receipt,
            })

    def cloud_callback(self, msg: PointCloud2) -> None:
        self._enqueue("cloud", msg)

    def odom_callback(self, msg: Odometry) -> None:
        self._enqueue("odom", msg)

    def grid_callback(self, msg: OccupancyGrid) -> None:
        self._enqueue("grid", msg)

    def _compute_loop(self) -> None:
        while not self.stop_requested or not self.events.empty():
            try:
                callback_event_id, event_type, msg, receipt, stamp, header_seq = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            self.event_id += 1
            started = time.monotonic()
            row = {
                "event_id": self.event_id,
                "callback_event_id": callback_event_id,
                "event_type": event_type,
                "ros_stamp": stamp,
                "header_seq": header_seq,
                "receipt_monotonic_time": receipt,
                "processing_start_monotonic_time": started,
                "state_version_before": self.state_version,
            }
            try:
                if event_type == "cloud":
                    self._apply_cloud(msg, row)
                elif event_type == "odom":
                    self._apply_odom(msg, row)
                else:
                    self._compare_grid(msg, row)
            except Exception as exc:  # Audit failures are visible, never treated as a valid frame.
                with self.counter_lock:
                    self.counts[f"{event_type}_processing_failure"] += 1
                row.update({"classification": "PROCESSING_FAILURE", "error": repr(exc)})
            finally:
                row["processing_end_monotonic_time"] = time.monotonic()
                row["state_version_after"] = self.state_version
                self._log("event_journal.jsonl", row)
                self.events.task_done()

    def _apply_odom(self, msg: Odometry, row: dict) -> None:
        pose = msg.pose.pose
        q = pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.nav.add_odom(float(pose.position.x), float(pose.position.y), float(yaw))
        self.latest_odom_stamp = self._stamp(msg)
        self.state_version += 1
        with self.counter_lock:
            self.counts["odom_applied"] += 1
        row.update({"classification": "ODOM_APPLIED", "odom_event_id": self.event_id})

    def _apply_cloud(self, msg: PointCloud2, row: dict) -> None:
        source_frame = msg.header.frame_id.lstrip("/")
        try:
            transform_time = self.tf_listener.getLatestCommonTime("base", source_frame)
            translation, quaternion = self.tf_listener.lookupTransform("base", source_frame, transform_time)
        except Exception as exc:
            with self.counter_lock:
                self.counts["cloud_tf_failure"] += 1
            row.update({"classification": "TF_FAILURE", "error": repr(exc)})
            return

        raw = np.asarray(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float64)
        if raw.size == 0:
            raw = raw.reshape((0, 3))
        matrix = tf.transformations.quaternion_matrix(quaternion)
        points_base = raw @ matrix[:3, :3].T + np.asarray(translation, dtype=np.float64)
        rays, metadata = self.core.observe_sparse(
            points_base, msg.header.stamp.to_sec(), transform_time.to_sec(), source_frame
        )
        self.cloud_sequence += 1
        self.nav.add_sparse_rays(rays)
        self.state_version += 1
        ended = time.monotonic()
        metadata.update({
            "frame_sequence": self.cloud_sequence,
            "source_header_seq": msg.header.seq,
            "callback_receive_monotonic_time": row["receipt_monotonic_time"],
            "processing_start_monotonic_time": row["processing_start_monotonic_time"],
            "processing_end_monotonic_time": ended,
            "persisted_ray_count": int(len(rays)),
            "left_count": int(np.sum(rays["side"] == 0)),
            "right_count": int(np.sum(rays["side"] == 1)),
            "clipped_count": int(np.sum(rays["clipped"])),
            "occupancy_eligible_count": int(np.sum(rays["occupancy_eligible"])),
            "quality_flags": [],
        })
        persisted_enqueued = self.writer.enqueue(self.cloud_sequence, rays, metadata, ended)
        self.latest_cloud_stamp = msg.header.stamp.to_sec()
        with self.counter_lock:
            self.counts["cloud_tf_success"] += 1
            self.counts["cloud_core_processed"] += 1
            self.counts["evidence_enqueued"] += int(persisted_enqueued)
            self.counts["evidence_enqueue_failed"] += int(not persisted_enqueued)
        summary = {**metadata, "persisted_enqueued": persisted_enqueued}
        self._log("frame_summary.jsonl", summary)
        self._log("side_evidence_summary.jsonl", {
            "frame_sequence": self.cloud_sequence, "source_stamp": metadata["source_stamp"],
            "left_count": metadata["left_count"], "right_count": metadata["right_count"],
            "clipped_count": metadata["clipped_count"], "ray_count": int(len(rays)),
            "sparse_geometry_file": f"side_evidence_frames/frame_{self.cloud_sequence:06d}.npz",
            "portal_or_control_fields_present": False,
        })
        self._log("performance.jsonl", {
            "event": "cloud", "source_stamp": self.latest_cloud_stamp,
            "core_and_nav_ms": (ended - row["processing_start_monotonic_time"]) * 1000.0,
            "compute_queue": self.events.qsize(), "writer_queue": self.writer.q.qsize(),
        })
        row.update({
            "classification": "CLOUD_APPLIED", "cloud_event_id": self.event_id,
            "valid_ray_count": int(len(rays)), "persisted_enqueued": persisted_enqueued,
            "transform_stamp": transform_time.to_sec(),
        })

    def _compare_grid(self, msg: OccupancyGrid, row: dict) -> None:
        if msg.info.width != 60 or msg.info.height != 60 or len(msg.data) != 3600:
            row.update({"classification": "GRID_METADATA_MISMATCH"})
            return
        expected = self.nav.publish_and_decay()
        actual = np.asarray(msg.data, dtype=np.int8).reshape((60, 60))
        different = int(np.sum(expected != actual))
        self.state_version += 1
        self.decay_publish_count += 1
        with self.counter_lock:
            self.counts["nav_grid_compared"] += 1
        row.update({
            "classification": "GRID_COMPARISON", "grid_event_id": self.event_id,
            "grid_stamp": self._stamp(msg), "different_cells": different,
            "cell_agreement": float(np.mean(expected == actual)),
            "latest_cloud_stamp": self.latest_cloud_stamp,
            "latest_odom_stamp": self.latest_odom_stamp,
            "decay_publish_count": self.decay_publish_count,
        })
        self._log("nav_grid_parity.jsonl", row)

    def heartbeat(self, _event) -> None:
        vmrss = 0
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                vmrss = int(line.split()[1])
                break
        self._log("process_resource.jsonl", {
            "wall_time": time.time(), "vmrss_kb": vmrss,
            "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "compute_queue": self.events.qsize(), "writer_queue": self.writer.q.qsize(),
        })
        self.publisher.publish(String(data=json.dumps({
            "audit_only": True, "portal_or_control_output": False,
            "compute_queue": self.events.qsize(), "writer_queue": self.writer.q.qsize(),
        })))

    def close(self) -> None:
        self.stop_requested = True
        self.worker.join(timeout=30.0)
        writer = self.writer.close()
        with self.counter_lock:
            accounting = dict(self.counts)
            accounting["cloud_header_seq_gaps"] = self.seq_gaps
        accounting.update({
            "evidence_persisted": writer["persisted_frames"],
            "persistence_queue_overflow": writer["queue_overflow"],
            "persistence_errors": writer["writer_errors"],
            "compute_queue_remaining": self.events.qsize(),
            "compute_worker_alive": self.worker.is_alive(),
            "publisher_emitted_frame_count": "UNKNOWN",
        })
        (self.out / "input_frame_accounting.json").write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
        stop = {**writer, "status": "STOPPED", "portal_or_control_output": False, **accounting}
        (self.out / "stop_summary.json").write_text(json.dumps(stop, indent=2, sort_keys=True) + "\n")
        for file in self.logs.values():
            file.close()


def main() -> None:
    rospy.init_node("p2kg7r_ray_evidence_shadow", anonymous=True)
    run_id = rospy.get_param("~run_id", time.strftime("online_%Y%m%d_%H%M%S"))
    out = ROOT / "debug/odom_accuracy_audit_v1/p2kg7r_ray_evidence_shadow_closure_045" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id, "audit_only": True, "production_code_modified": False,
        "default_started": False, "truth_or_world_input": False,
    }, indent=2) + "\n")
    (out / "topic_contract.json").write_text(json.dumps({
        "subscribes": ["/team/livox/scan_cloud_filtered", "/team/livox/icp_odom_gated", "/team/local_traversability_grid", "/tf"],
        "publishes": [AUDIT_TOPIC],
        "forbidden_outputs": ["/cmd_vel", "/cmd_vel_raw", "portal", "target", "state event", "production grid"],
        "services": [], "production_parameter_writes": False,
    }, indent=2) + "\n")
    node = Shadow(out)
    rospy.on_shutdown(node.close)
    rospy.spin()


if __name__ == "__main__":
    main()
