#!/usr/bin/env python3
"""Focused formal-freshness contract tests; no ROS node is started."""

import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
from l3v_local_traversability_node import L3VLocalTraversabilityNode  # noqa: E402


class Fixture:
    input_stale_timeout_sec = 1.0
    latest_odom = {"x": 0.0}
    last_odom_wall_time = 10.0
    last_lidar_wall_time = None
    last_depth_points_wall_time = 10.0
    last_depth_image_wall_time = 10.0


class FreshnessContractTests(unittest.TestCase):
    def test_t32_depth_cannot_mask_stale_lidar_when_rgbd_is_disabled(self):
        fixture = Fixture()
        fixture.use_rgbd_obstacle_evidence = False
        result = L3VLocalTraversabilityNode.input_freshness_locked(fixture, 10.1)
        self.assertFalse(result["all_required_inputs_fresh"])
        self.assertIn("lidar_stale_or_missing", result["stale_reasons"])

    def test_t33_fresh_odom_and_lidar_remain_eligible(self):
        fixture = Fixture()
        fixture.use_rgbd_obstacle_evidence = False
        fixture.last_lidar_wall_time = 10.0
        result = L3VLocalTraversabilityNode.input_freshness_locked(fixture, 10.1)
        self.assertTrue(result["all_required_inputs_fresh"])

    def test_t34_rgbd_obstacle_authority_is_not_enabled_by_the_repair(self):
        source = (MODULE_DIR / "l3v_local_traversability_node.py").read_text(encoding="utf-8")
        self.assertIn('rospy.get_param("~use_rgbd_obstacle_evidence", False)', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
