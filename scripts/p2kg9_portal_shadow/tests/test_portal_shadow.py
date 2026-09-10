#!/usr/bin/env python3
import ast
import json
import tempfile
import time
import unittest
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "l3v_ray_evidence_shadow"))
from p2kg9_portal_core import FrozenPortalMethod  # noqa: E402
from portal_audit_writer import PortalAuditWriter  # noqa: E402
from ray_evidence_core import RayEvidenceCore  # noqa: E402


def doorway_rays():
    wall = [(x, 1.0, 0.4) for x in np.r_[np.arange(.2, 1.5, .05), np.arange(2.7, 4.0, .05)]]
    behind = [(2.0 * x, 2.0, .4) for x in np.arange(1.55, 2.65, .05)]
    rays, _ = RayEvidenceCore().observe_sparse(np.asarray(wall + behind), 1.0, 0.0, "laser_livox")
    return rays


class PortalShadowTests(unittest.TestCase):
    def test_direct_frozen_candidate_equivalence(self):
        method = FrozenPortalMethod()
        rays = doorway_rays()
        frame = {"source_stamp": 1.0, "frame_id": "base", "rays": rays, "source": "unit"}
        expected = method.g8.candidate(frame, "left", 3)
        actual = method.process(rays, 1.0, "unit")[0]
        for field in ("portal_center_base", "portal_width", "portal_normal_base", "left_boundary", "right_boundary", "wall_gap_support", "free_space_support", "traversability_support"):
            self.assertEqual(json.dumps(actual[field], sort_keys=True), json.dumps(expected[field], sort_keys=True))

    def test_confirmed_only_after_frozen_temporal_duration(self):
        method = FrozenPortalMethod()
        frames = method.g8.dev_frames()[:5]
        first = method.process(frames[0]["rays"], frames[0]["source_stamp"], frames[0]["source"])[0]
        self.assertEqual(first["observation_state"], "partial")
        final = None
        for frame in frames[1:]:
            final = method.process(frame["rays"], frame["source_stamp"], frame["source"])[0]
        self.assertIsNotNone(final)
        self.assertTrue(final["candidate_available"])
        self.assertEqual(final["observation_state"], "confirmed")
        self.assertTrue(final["temporal_support"])

    def test_unknown_cannot_be_candidate(self):
        rays, _ = RayEvidenceCore().observe_sparse(np.empty((0, 3)), 1.0, 0.0, "laser_livox")
        row = FrozenPortalMethod().process(rays, 1.0)[0]
        self.assertFalse(row["candidate_available"])
        self.assertIn(row["observation_state"], ("unknown", "rejected", "partial"))

    def test_out_of_order_stamp_does_not_promote_track(self):
        method, rays = FrozenPortalMethod(), doorway_rays()
        method.process(rays, 2.0)
        row = method.process(rays, 1.9)[0]
        self.assertEqual(row["observation_state"], "unknown")
        self.assertFalse(row["candidate_available"])
        self.assertEqual(row["rejection_reason"], "OUT_OF_ORDER_SOURCE_STAMP")

    def test_stale_gap_starts_a_new_partial_track(self):
        method, rays = FrozenPortalMethod(), doorway_rays()
        method.process(rays, 1.0)
        method.process(rays, 1.3)
        row = method.process(rays, 1.6)[0]
        self.assertEqual(row["observation_state"], "partial")
        self.assertEqual(row["stable_duration"], 0.0)

    def test_writer_is_bounded_atomic_and_drains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = PortalAuditWriter(root, max_queue=2)
            self.assertTrue(writer.enqueue(1, {"source_stamp": 1.0, "portal_candidates": []}))
            result = writer.close()
            self.assertEqual(result["persisted_frames"], 1)
            self.assertEqual(result["queue_remaining"], 0)
            index = json.loads((root / "portal_outputs.jsonl").read_text())
            self.assertTrue((root / index["file"]).exists())
            self.assertFalse(list((root / "portal_frames").glob("*.tmp")))

    def test_no_truth_target_or_control_node_input(self):
        source = (HERE / "portal_shadow_node.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertNotIn("/gazebo/model_states", source)
        self.assertNotIn('Publisher("/cmd_vel"', source)
        self.assertNotIn('Publisher("/cmd_vel_raw"', source)
        self.assertNotIn("doorway_candidate", source)
        self.assertNotIn("navigation_state_machine", source)
        self.assertIn('/audit/p2kg9/portal_candidate', source)

    def test_no_portal_output_can_be_marked_available_without_confirmed(self):
        method = FrozenPortalMethod()
        for row in method.process(doorway_rays(), 1.0):
            self.assertFalse(row["candidate_available"] and row["observation_state"] != "confirmed")


if __name__ == "__main__":
    unittest.main()
