#!/usr/bin/env python3
import hashlib
import json
import math
import tempfile
import time
import unittest
import ast
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ray_evidence_core import (
    AXIS_X, AXIS_Y, HEIGHT_NEAR_GROUND, NavGridCompat, RayEvidenceCore,
    SideEvidence, local_to_cell,
)
from sparse_persistence import SparseFrameWriter


class RayEvidenceCoreTests(unittest.TestCase):
    def ray(self, point):
        return RayEvidenceCore().observe([point], 12.5, 12.6, "laser_livox")[0]

    def sparse_one(self, point):
        rays, _ = RayEvidenceCore().observe_sparse(np.asarray([point]), 12.5, 12.6, "laser_livox")
        return rays[0]

    def test_in_grid_endpoint(self):
        self.assertFalse(self.ray((2, 0, .3)).clipped)

    def test_x_clipped_endpoint_is_measured_and_not_boundary_occupied(self):
        ray = self.ray((3.1, 0, .3)); sparse = self.sparse_one((3.1, 0, .3))
        self.assertEqual(ray.clipped_axes, ("x",))
        np.testing.assert_allclose(sparse["endpoint_xyz"], (3.1, 0.0, 0.3), rtol=0, atol=1e-6)
        self.assertTrue(sparse["intersection_valid"])
        self.assertLess(float(sparse["grid_intersection_xy"][0]), 3.0)
        nav = NavGridCompat(); nav.add_sparse_rays(np.asarray([sparse]))
        self.assertEqual(float(nav.occ[-1, 30]), 0.0)

    def test_left_y_clipped(self):
        ray = self.ray((1, 1.7, .3)); self.assertEqual(ray.clipped_axes, ("y",))
        self.assertEqual(int(self.sparse_one((1, 1.7, .3))["clipped_axes"]), AXIS_Y)

    def test_right_y_clipped(self):
        ray = self.ray((1, -1.7, .3)); self.assertEqual(ray.clipped_axes, ("y",))
        self.assertEqual(int(self.sparse_one((1, -1.7, .3))["clipped_axes"]), AXIS_Y)

    def test_x_and_y_corner_clipped(self):
        ray = self.ray((4.0, 2.0, .3)); sparse = self.sparse_one((4.0, 2.0, .3))
        self.assertEqual(ray.clipped_axes, ("x", "y"))
        self.assertEqual(int(sparse["clipped_axes"]), AXIS_X | AXIS_Y)

    def test_exact_upper_boundary(self):
        self.assertEqual(self.ray((2, 1.5, .3)).clipped_axes, ("y",))

    def test_exact_lower_boundary_is_inside(self):
        self.assertFalse(self.ray((2, -1.5, .3)).clipped)

    def test_nonfinite_and_range_dropped(self):
        core = RayEvidenceCore()
        self.assertEqual(core.observe([(math.nan, 0, 0)], 1, 1, "x"), [])
        self.assertEqual(core.observe([(0.01, 0, 0)], 1, 1, "x"), [])
        self.assertEqual(core.observe([(9, 0, 0)], 1, 1, "x"), [])

    def test_height_not_occupied(self):
        self.assertFalse(self.ray((2, 0, 2)).endpoint_occupancy_eligible)
        self.assertEqual(int(self.sparse_one((2, 0, 0))["height_class"]), HEIGHT_NEAR_GROUND)

    def test_repeated_ray_has_ids_and_point_indexes(self):
        rays = RayEvidenceCore().observe([(2, 0, .3)] * 2, 1, 1, "x")
        self.assertEqual([x.ray_id for x in rays], [0, 1])
        self.assertEqual([x.point_index for x in rays], [0, 1])

    def test_stamp_frame_and_side_sign(self):
        ray = self.ray((2, 0, .3)); self.assertEqual((ray.source_stamp, ray.transform_stamp, ray.target_frame), (12.5, 12.6, "base"))
        rays = RayEvidenceCore().observe([(2, 1, .3), (2, -1, .3)], 1, 1, "x")
        self.assertEqual(SideEvidence().summarize(rays, 1)["sides"]["left"]["source_count"], 1)

    def test_bounds_are_half_open(self):
        self.assertIsNone(local_to_cell(3, 0))
        self.assertIsNone(local_to_cell(1, 1.5))
        self.assertIsNotNone(local_to_cell(1, -1.5))

    def test_legacy_and_sparse_geometry_equivalent(self):
        points = np.asarray([
            (2, 0, .3), (3.1, 0, .3), (1, 1.7, .3), (1, -1.7, .3),
            (4, 2, .3), (2, 1.5, .3), (2, -1.5, .3), (-1, 0, .3),
            (0, 4, .3), (0, -4, .3), (1, 0, 2), (math.nan, 0, 0),
        ])
        legacy = RayEvidenceCore().observe(points, 1, 2, "src")
        sparse, _ = RayEvidenceCore().observe_sparse(points, 1, 2, "src")
        self.assertEqual(len(legacy), len(sparse))
        for old, new in zip(legacy, sparse):
            self.assertEqual(old.point_index, int(new["point_index"]))
            np.testing.assert_allclose(old.ray_endpoint_base, new["endpoint_xyz"], rtol=0, atol=1e-6)
            self.assertAlmostEqual(old.measured_range, float(new["measured_range"]), places=5)
            self.assertEqual(old.clipped, bool(new["clipped"]))
            old_axes = sum({"x": AXIS_X, "y": AXIS_Y}[axis] for axis in old.clipped_axes)
            self.assertEqual(old_axes, int(new["clipped_axes"]))
            self.assertEqual(old.current_grid_intersection is not None, bool(new["intersection_valid"]))
            if old.current_grid_intersection:
                np.testing.assert_allclose(old.current_grid_intersection, new["grid_intersection_xy"], rtol=0, atol=1e-6)

    def test_sparse_nav_consumer_matches_legacy_ray_updates(self):
        rng = np.random.default_rng(7)
        points = rng.uniform([-.5, -2.0, -.6], [4.0, 2.0, 1.8], size=(2000, 3))
        points[:8] = [(2, 0, .3), (2, 0, 0), (3.2, 0, .3), (1, 1.8, .3), (1, -1.8, .3), (1, 0, 2), (0.05, 0, .3), (2.9, 1.4, .3)]
        legacy_core = RayEvidenceCore(); sparse_core = RayEvidenceCore()
        legacy = legacy_core.observe(points, 1, 1, "src")
        sparse, _ = sparse_core.observe_sparse(points, 1, 1, "src")
        expected, actual = NavGridCompat(), NavGridCompat()
        self.assertEqual(expected.add_rays(legacy), actual.add_sparse_rays(sparse))
        np.testing.assert_array_equal(expected.free, actual.free)
        np.testing.assert_array_equal(expected.occ, actual.occ)
        np.testing.assert_array_equal(expected.labels(), actual.labels())

    def test_sparse_persistence_roundtrip_and_sha_index(self):
        rays, meta = RayEvidenceCore().observe_sparse(np.asarray([(3.1, 0, .3), (1, -1.7, 0)]), 7.5, 7.6, "src")
        meta.update({"frame_sequence": 1, "source_header_seq": 4, "processing_end_monotonic_time": 3.0})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); writer = SparseFrameWriter(root, max_queue=1)
            self.assertTrue(writer.enqueue(1, rays, meta, time.monotonic()))
            result = writer.close(); self.assertEqual(result["persisted_frames"], 1); self.assertEqual(result["queue_remaining"], 0)
            row = json.loads((root / "side_evidence_index.jsonl").read_text().strip())
            target = root / row["file"]
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), row["sha256"])
            with np.load(target, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["rays"], rays)
                loaded_meta = json.loads(str(loaded["metadata"].item()))
            self.assertEqual(loaded_meta["source_stamp"], 7.5)

    def test_writer_error_has_no_index_item_or_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); writer = SparseFrameWriter(root, max_queue=1)
            writer.frames = root / "not_a_directory"; writer.frames.write_text("block writes")
            rays, meta = RayEvidenceCore().observe_sparse(np.asarray([(2, 0, .3)]), 1, 1, "src")
            self.assertTrue(writer.enqueue(7, rays, meta, time.monotonic()))
            result = writer.close()
            self.assertEqual(result["persisted_frames"], 0)
            self.assertTrue(result["writer_errors"])
            self.assertEqual((root / "side_evidence_index.jsonl").read_text(), "")

    def test_no_portal_target_state_or_control_fields(self):
        summary = SideEvidence().summarize([], 1)
        self.assertFalse(summary["portal_fields_present"])
        self.assertFalse(summary["control_fields_present"])
        forbidden = {"portal", "target", "state", "cmd_vel", "control"}
        self.assertFalse(forbidden & set(summary))

    def test_shadow_node_static_topic_and_output_isolation(self):
        node_path = Path(__file__).resolve().parents[1] / "ray_evidence_shadow_node.py"
        source = node_path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('AUDIT_TOPIC = "/audit/p2kg7r/ray_evidence_status"', source)
        self.assertNotIn('Publisher("/cmd_vel"', source)
        self.assertNotIn('Publisher("/cmd_vel_raw"', source)
        self.assertNotIn('Publisher("/team/local_traversability_grid"', source)
        self.assertNotIn("doorway_candidate", source)
        self.assertNotIn("entry_pose", source)


if __name__ == "__main__":
    unittest.main()
