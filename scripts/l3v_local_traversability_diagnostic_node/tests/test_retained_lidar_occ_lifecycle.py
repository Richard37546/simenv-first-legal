#!/usr/bin/env python3
"""Generic current-LiDAR support arbitration tests for retained lidar_occ."""

import ast
import sys
import unittest
from pathlib import Path

import numpy as np


NODE = Path(__file__).resolve().parents[1] / "l3v_local_traversability_node.py"
sys.path.insert(0, str(NODE.parent))
from motion_consistent_evidence import warp_array  # noqa: E402
SOURCE = NODE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def load_method(name):
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == "L3VLocalTraversabilityNode":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    module = ast.Module(body=[item], type_ignores=[])
                    namespace = {"np": np, "warp_array": warp_array}
                    exec(compile(ast.fix_missing_locations(module), str(NODE), "exec"), namespace)
                    return namespace[name]
    raise AssertionError(f"method not found: {name}")


RECONCILE = load_method("reconcile_current_lidar_support_locked")
REPROJECT_SUPPORT = load_method("reproject_current_lidar_support_locked")
BEGIN_SCAN = load_method("begin_current_lidar_support_locked")


class Fixture:
    def __init__(self):
        self.lidar_occ = np.zeros((20, 40), dtype=np.float32)
        self.current_lidar_free_support = np.zeros((20, 40), dtype=bool)
        self.current_lidar_occ_support = np.zeros((20, 40), dtype=bool)
        self.resolution = 1.0
        self.x_min = 0.0
        self.y_min = 0.0


class RetainedLidarOccLifecycleTests(unittest.TestCase):
    def reconcile(self, fixture):
        return RECONCILE(fixture)

    def reproject(self, fixture, old_pose=(0.0, 0.0, 0.0), new_pose=(1.0, 0.0, 0.0)):
        fixture.lidar_occ = warp_array(
            fixture.lidar_occ, old_pose, new_pose,
            fixture.resolution, fixture.x_min, fixture.y_min,
        )
        return REPROJECT_SUPPORT(fixture, old_pose, new_pose)

    def test_p1_stale_obstacle_is_invalidated_by_current_direct_free(self):
        f = Fixture()
        f.lidar_occ[1, 1] = 0.8
        f.current_lidar_free_support[1, 1] = True
        self.assertTrue(self.reconcile(f)[1, 1])
        self.assertEqual(float(f.lidar_occ[1, 1]), 0.0)

    def test_p2_unobserved_stale_obstacle_is_retained(self):
        f = Fixture()
        f.lidar_occ[1, 1] = 0.8
        self.reconcile(f)
        self.assertAlmostEqual(float(f.lidar_occ[1, 1]), 0.8)

    def test_p3_current_obstacle_wins_over_current_free(self):
        f = Fixture()
        f.lidar_occ[1, 1] = 1.8
        f.current_lidar_free_support[1, 1] = True
        f.current_lidar_occ_support[1, 1] = True
        self.assertFalse(self.reconcile(f)[1, 1])
        self.assertAlmostEqual(float(f.lidar_occ[1, 1]), 1.8)

    def test_p4_current_new_obstacle_is_preserved(self):
        f = Fixture()
        f.lidar_occ[2, 2] = 1.0
        f.current_lidar_occ_support[2, 2] = True
        self.reconcile(f)
        self.assertAlmostEqual(float(f.lidar_occ[2, 2]), 1.0)

    def test_p5_current_free_without_history_stays_non_obstacle(self):
        f = Fixture()
        f.current_lidar_free_support[1, 2] = True
        self.reconcile(f)
        self.assertEqual(float(f.lidar_occ[1, 2]), 0.0)

    def test_p6_reconciliation_does_not_change_max_reprojection_module(self):
        motion_source = (Path(__file__).resolve().parents[1] / "motion_consistent_evidence.py").read_text(encoding="utf-8")
        self.assertIn("max(", motion_source)
        self.assertIn("output[new_row, new_column]", motion_source)

    def test_p7_true_wall_unobserved_is_retained(self):
        f = Fixture()
        f.lidar_occ[3, 3] = 2.0
        self.reconcile(f)
        self.assertAlmostEqual(float(f.lidar_occ[3, 3]), 2.0)

    def test_p8_true_wall_currently_reobserved_is_preserved(self):
        f = Fixture()
        f.lidar_occ[3, 3] = 2.0
        f.current_lidar_occ_support[3, 3] = True
        self.reconcile(f)
        self.assertAlmostEqual(float(f.lidar_occ[3, 3]), 2.0)

    def test_p9_unknown_semantics_are_not_changed(self):
        self.assertNotIn("unknown", ast.get_source_segment(SOURCE, next(
            item for node in TREE.body if isinstance(node, ast.ClassDef)
            for item in node.body if isinstance(item, ast.FunctionDef)
            and item.name == "reconcile_current_lidar_support_locked"
        )).lower())

    def test_o1_cloud_free_then_odom_warp_cannot_reintroduce_history(self):
        f = Fixture()
        f.lidar_occ[13, 20] = 0.96
        f.current_lidar_free_support[13, 20] = True
        self.reproject(f)
        self.assertTrue(f.current_lidar_free_support[12, 20])
        self.reconcile(f)
        self.assertEqual(float(f.lidar_occ[12, 20]), 0.0)

    def test_o2_stale_neighbor_reentry_is_reconciled_after_warp(self):
        f = Fixture()
        f.lidar_occ[13, 34] = 0.96
        f.current_lidar_free_support[13, 34] = True
        self.reproject(f)
        self.assertTrue(f.current_lidar_free_support[12, 34])
        self.reconcile(f)
        self.assertEqual(float(f.lidar_occ[12, 34]), 0.0)

    def test_o3_current_obstacle_support_moves_with_odom(self):
        f = Fixture()
        f.lidar_occ[13, 20] = 1.0
        f.current_lidar_occ_support[13, 20] = True
        self.reproject(f)
        self.assertTrue(f.current_lidar_occ_support[12, 20])
        self.reconcile(f)
        self.assertAlmostEqual(float(f.lidar_occ[12, 20]), 1.0)

    def test_o4_obstacle_wins_when_warped_supports_overlap(self):
        f = Fixture()
        f.lidar_occ[13, 20] = 1.0
        f.current_lidar_free_support[13, 20] = True
        f.current_lidar_occ_support[13, 20] = True
        self.reproject(f)
        self.reconcile(f)
        self.assertAlmostEqual(float(f.lidar_occ[12, 20]), 1.0)

    def test_o5_unobserved_history_survives_odom_warp(self):
        f = Fixture()
        f.lidar_occ[13, 20] = 0.96
        self.reproject(f)
        self.reconcile(f)
        self.assertAlmostEqual(float(f.lidar_occ[12, 20]), 0.96)

    def test_o6_multiple_odom_updates_keep_support_aligned(self):
        f = Fixture()
        f.lidar_occ[13, 20] = 0.96
        f.current_lidar_free_support[13, 20] = True
        self.reproject(f, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.reproject(f, (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        self.assertTrue(f.current_lidar_free_support[11, 20])
        self.reconcile(f)
        self.assertEqual(float(f.lidar_occ[11, 20]), 0.0)

    def test_o7_new_scan_resets_old_support(self):
        f = Fixture()
        f.current_lidar_free_support[4, 4] = True
        f.current_lidar_occ_support[5, 5] = True
        BEGIN_SCAN(f)
        self.assertFalse(bool(f.current_lidar_free_support.any()))
        self.assertFalse(bool(f.current_lidar_occ_support.any()))

    def test_o8_new_scan_obstacle_overrides_prior_free(self):
        f = Fixture()
        f.current_lidar_free_support[6, 6] = True
        BEGIN_SCAN(f)
        f.lidar_occ[6, 6] = 1.0
        f.current_lidar_occ_support[6, 6] = True
        self.reconcile(f)
        self.assertAlmostEqual(float(f.lidar_occ[6, 6]), 1.0)

    def test_o9_max_evidence_merge_is_unchanged(self):
        merged = warp_array(
            np.array([[0.4, 0.9]], dtype=np.float32),
            (0.0, 0.0, 0.0), (0.49, 0.0, 0.0), 1.0, 0.0, 0.0,
        )
        self.assertAlmostEqual(float(merged.max()), 0.9)

    def test_o10_support_logic_does_not_create_unknown_values(self):
        source = ast.get_source_segment(SOURCE, next(
            item for node in TREE.body if isinstance(node, ast.ClassDef)
            for item in node.body if isinstance(item, ast.FunctionDef)
            and item.name in {"reproject_current_lidar_support_locked", "reconcile_current_lidar_support_locked"}
        )).lower()
        self.assertNotIn("unknown", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
