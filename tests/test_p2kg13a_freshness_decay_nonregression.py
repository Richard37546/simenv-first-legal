#!/usr/bin/env python3
"""Offline P2KG13A freshness and publish-timed decay regression tests."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "local_subgoal_runner_mvp"))


def load_module():
    path = ROOT / "scripts" / "l3v_local_traversability_diagnostic_node" / "l3v_local_traversability_node.py"
    spec = importlib.util.spec_from_file_location("p2kg13a_l3v_freshness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FakeStamp:
    def __init__(self, seconds):
        self.seconds = float(seconds)

    def to_sec(self):
        return self.seconds

    def __sub__(self, other):
        return FakeStamp(self.seconds - other.seconds)


class FakeRosTime:
    now_seconds = 0.0

    @classmethod
    def now(cls):
        return FakeStamp(cls.now_seconds)


class CapturePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FreshnessDecayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.l3v = load_module()
        baseline_path = ROOT / "audit_snapshots" / "p2kg13a_prechange_064" / "l3v_local_traversability_node.py"
        spec = importlib.util.spec_from_file_location("p2kg13a_l3v_prechange_064", baseline_path)
        cls.prechange = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(cls.prechange)

    def make_status(self, generation=1):
        return {
            "contract_version": "local_grid_navigation_contract_v1",
            "schema_version": "local_grid_status_v1",
            "producer_instance_id": "offline-fixture",
            "content_generation_id": generation,
            "grid_content_stamp": float(generation),
            "grid_content_hash": f"hash-{generation}",
            "frame_id": "base",
            "tf_valid": True,
            "input_time_monotonic": True,
            "all_required_inputs_fresh": True,
            "stale_reasons": [],
            "input_freshness": {},
            "diagnostic_only": True,
            "safe_for_navigation": False,
            "upstream_navigation_allowed": False,
            "rejection_reasons": ["diagnostic_only", "footprint_qualification_unqualified"],
            "local_traversability_status": "FREE_SUPPORTED",
        }

    def make_node(self, decay_per_publish=0.5):
        node = object.__new__(self.l3v.L3VLocalTraversabilityNode)
        node.rows = node.cols = 2
        node.resolution = 1.0
        node.y_min, node.y_max, node.x_max = -1.0, 1.0, 2.0
        node.frame_id = "base"
        node.self_footprint_x_m = 0.0
        node.self_footprint_abs_y_m = 0.0
        node.decay_per_publish = decay_per_publish
        node.lock = threading.Lock()
        node.reset_evidence()
        node.latest_odom = {"stamp": 1.0}
        node.last_odom_wall_time = 100.0
        node.last_lidar_wall_time = 100.0
        node.last_depth_points_wall_time = None
        node.last_depth_image_wall_time = None
        node.input_stale_timeout_sec = 2.0
        # Keep this lightweight __new__ fixture aligned with the current L3V
        # node defaults; these flags remain disabled in this freshness-only
        # contract and are not part of the behaviour under test.
        node.use_rgbd_obstacle_evidence = False
        node.doorway_provenance_enabled = False
        node.heartbeat_tick_count = 0
        node.heartbeat_rate_hz = 2.0
        node.last_publish_wall_time_sec = None
        node.last_publish_ros_time_sec = None
        node.content_generation_id = 0
        node.content_dirty = True
        node.cached_grid = None
        node.cached_status = None
        node.grid_pub = CapturePublisher()
        node.evidence_pub = CapturePublisher()
        node.front_pub = CapturePublisher()
        node.status_pub = CapturePublisher()
        node.labels_seen = []

        def build_content(_now_wall):
            node.labels_seen.append(node.labels_locked().copy())
            node.content_generation_id += 1
            node.cached_grid = SimpleNamespace(name=f"grid-{node.content_generation_id}")
            node.cached_status = self.make_status(node.content_generation_id)
            node.content_dirty = False
            node.freshness_revoked_content_generation = None

        node.build_content_locked = build_content
        node.sector_summary_locked = lambda _labels: {}
        return node

    def published_status(self, node):
        return json.loads(node.status_pub.messages[-1].data)

    def publish_at(self, node, wall_seconds, ros_seconds=None):
        FakeRosTime.now_seconds = wall_seconds if ros_seconds is None else ros_seconds
        with patch.object(self.l3v.time, "monotonic", return_value=wall_seconds), patch.object(self.l3v.rospy, "Time", FakeRosTime):
            node.publish(None)

    def add_cloud_fields(self, node):
        node.last_cloud_time = FakeStamp(0.0)
        node.last_depth_time = FakeStamp(0.0)
        node.last_source_stamp = {"lidar": None, "rgbd": None, "odom": None}
        node.latest_lidar_source_stamp = None
        node.pending_time_regression_reasons = []
        node.pending_tf_failure_reasons = []
        node.lidar_cloud_count = 0
        node.depth_points_count = 0
        node.depth_stride = 1
        node.max_points_per_cloud = 100
        node.use_rgbd_obstacle_evidence = False

    def call_empty_cloud(self, node, source, now_seconds, stamp_seconds):
        message = SimpleNamespace(header=SimpleNamespace(stamp=FakeStamp(stamp_seconds), frame_id="base"))
        FakeRosTime.now_seconds = now_seconds
        with patch.object(self.l3v.rospy, "Time", FakeRosTime), patch.object(self.l3v.pc2, "read_points", return_value=[]):
            node.process_cloud(message, source)

    def test_publish_calls_decay_once_without_callbacks(self):
        node = self.make_node()
        node.decay = Mock(return_value=False)
        for now in (100.0, 101.0, 102.0):
            self.publish_at(node, now)
        self.assertEqual(node.decay.call_count, 3)
        statuses = [json.loads(message.data) for message in node.status_pub.messages]
        self.assertEqual([status["content_generation_id"] for status in statuses], [1, 1, 1])
        self.assertEqual([status["publication_sequence"] for status in statuses], [1, 2, 3])

    def test_cloud_callbacks_never_decay_and_one_publish_decays_once(self):
        node = self.make_node()
        self.add_cloud_fields(node)
        node.decay = Mock(return_value=False)
        for index in range(5):
            self.call_empty_cloud(node, "lidar", 10.0 + index, 1.0 + index)
        self.assertEqual(node.decay.call_count, 0)
        self.publish_at(node, 100.0)
        self.assertEqual(node.decay.call_count, 1)

    def test_lidar_and_rgbd_callbacks_still_decay_only_at_publish(self):
        node = self.make_node()
        self.add_cloud_fields(node)
        node.decay = Mock(return_value=False)
        self.call_empty_cloud(node, "lidar", 10.0, 1.0)
        self.call_empty_cloud(node, "rgbd", 20.0, 2.0)
        self.assertEqual(node.decay.call_count, 0)
        self.publish_at(node, 100.0)
        self.assertEqual(node.decay.call_count, 1)

    def test_current_output_precedes_decay_and_next_generation_sees_decay(self):
        node = self.make_node(decay_per_publish=0.5)
        node.lidar_free[1, 1] = 1.0
        self.publish_at(node, 100.0)
        self.assertEqual(node.labels_seen[0][1, 1], "free")
        self.assertEqual(node.lidar_free[1, 1], 0.5)
        self.assertTrue(node.content_dirty)
        self.publish_at(node, 101.0)
        self.assertEqual(node.labels_seen[1][1, 1], "unknown")
        self.assertEqual(node.lidar_free[1, 1], 0.25)

    def test_decay_matches_pre_g13a_publish_evolution(self):
        node = self.make_node(decay_per_publish=0.8)
        baseline = object.__new__(self.prechange.L3VLocalTraversabilityNode)
        baseline.rows = baseline.cols = 2
        baseline.resolution = 1.0
        baseline.y_min, baseline.y_max, baseline.x_max = -1.0, 1.0, 2.0
        baseline.self_footprint_x_m = 0.0
        baseline.self_footprint_abs_y_m = 0.0
        baseline.decay_per_publish = 0.8
        baseline.reset_evidence()
        node.lidar_free[:, :] = [[0.5, 1.0], [2.0, 0.0]]
        node.lidar_occ[:, :] = [[0.0, 1.0], [0.0, 3.0]]
        node.rgbd_free[:, :] = [[1.0, 0.0], [0.0, 0.0]]
        node.rgbd_occ[:, :] = [[0.0, 0.0], [4.0, 0.0]]
        node.traversed[:, :] = [[0.0, 2.0], [0.0, 1.0]]
        for current, old in zip(
            (node.lidar_free, node.lidar_occ, node.rgbd_free, node.rgbd_occ, node.traversed),
            (baseline.lidar_free, baseline.lidar_occ, baseline.rgbd_free, baseline.rgbd_occ, baseline.traversed),
        ):
            old[:, :] = current
        for _ in range(3):
            labels_before = node.labels_locked().copy()
            np.testing.assert_array_equal(labels_before, baseline.labels_locked())
            self.assertTrue(node.decay())
            baseline.decay()
            for actual, old in zip(
                (node.lidar_free, node.lidar_occ, node.rgbd_free, node.rgbd_occ, node.traversed),
                (baseline.lidar_free, baseline.lidar_occ, baseline.rgbd_free, baseline.rgbd_occ, baseline.traversed),
            ):
                np.testing.assert_allclose(actual, old)
            self.assertEqual(labels_before.shape, node.labels_locked().shape)
            self.assertIn(labels_before[0, 1], {"free", "occupied", "conflict", "unknown"})

    def test_heartbeat_revokes_stale_cached_freshness_without_new_generation(self):
        node = self.make_node()
        node.decay = Mock(return_value=False)
        self.publish_at(node, 101.0)
        fresh = self.published_status(node)
        self.publish_at(node, 103.0)
        stale = self.published_status(node)
        self.assertTrue(fresh["all_required_inputs_fresh"])
        self.assertFalse(stale["all_required_inputs_fresh"])
        self.assertIn("odom_stale_or_missing", stale["stale_reasons"])
        # This fixture explicitly disables RGB-D evidence, so the current
        # status contract reports the precise LiDAR-only stale reason.
        self.assertIn("lidar_stale_or_missing", stale["stale_reasons"])
        self.assertFalse(stale["safe_for_navigation"])
        self.assertEqual(stale["content_generation_id"], fresh["content_generation_id"])
        self.assertEqual(stale["grid_content_stamp"], fresh["grid_content_stamp"])
        self.assertEqual(stale["grid_content_hash"], fresh["grid_content_hash"])

    def test_stale_heartbeat_cannot_restore_and_new_content_can_restore(self):
        node = self.make_node()
        node.decay = Mock(return_value=False)
        self.publish_at(node, 103.0)
        stale = self.published_status(node)
        self.publish_at(node, 104.0)
        self.assertFalse(self.published_status(node)["all_required_inputs_fresh"])
        self.publish_at(node, 101.0)
        self.assertFalse(self.published_status(node)["all_required_inputs_fresh"])
        node.last_odom_wall_time = 105.0
        node.last_lidar_wall_time = 105.0
        node.content_dirty = True
        self.publish_at(node, 105.0)
        recovered = self.published_status(node)
        self.assertFalse(stale["all_required_inputs_fresh"])
        self.assertTrue(recovered["all_required_inputs_fresh"])
        self.assertFalse(recovered["safe_for_navigation"])
        self.assertGreater(recovered["content_generation_id"], stale["content_generation_id"])

    def test_pure_heartbeat_preserves_content_identity(self):
        node = self.make_node()
        self.publish_at(node, 100.0, 10.0)
        first = self.published_status(node)
        self.publish_at(node, 101.0, 11.0)
        second = self.published_status(node)
        for field in ("producer_instance_id", "content_generation_id", "grid_content_stamp", "grid_content_hash"):
            self.assertEqual(first[field], second[field])
        self.assertLess(first["publication_sequence"], second["publication_sequence"])


if __name__ == "__main__":
    unittest.main()
