#!/usr/bin/env python3
"""R43 wall-evidence semantics tests; no ROS nodes, publishers, or motion."""

import sys
import unittest
from pathlib import Path

import numpy as np
from nav_msgs.msg import OccupancyGrid


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

from block_astar_dwa_mature_runner import BlockAStarDwaRunner, build_arg_parser
from local_grid_contract import flatten_index


class CorridorWallEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.runner = BlockAStarDwaRunner.__new__(BlockAStarDwaRunner)
        self.runner.args = build_arg_parser().parse_args([])

    def grid(self):
        msg = OccupancyGrid()
        msg.header.frame_id = "base"
        msg.info.resolution = 0.05
        msg.info.width = 66
        msg.info.height = 60
        msg.info.origin.position.x = -0.3
        msg.info.origin.position.y = -1.5
        msg.data = [0] * (msg.info.width * msg.info.height)
        return msg

    def set_cell(self, msg, x_m, y_m, value):
        cell = self.runner.local_xy_to_cell(x_m, y_m, msg)
        self.assertIsNotNone(cell)
        x_index, y_index = cell
        msg.data[flatten_index(x_index, y_index, msg.info.width, msg.info.height)] = value

    def test_unknown_remains_collision_blocked_but_is_not_wall_evidence(self):
        msg = self.grid()
        for x_m in np.arange(0.8, 2.41, 0.05):
            for y_m in (0.5, 0.55, 0.6, -0.5, -0.55, -0.6):
                self.set_cell(msg, x_m, y_m, -1)

        raw = self.runner.grid_array(msg)
        collision_blocked = self.runner.inflate_obstacles(raw, msg.info.resolution)
        observed_wall_evidence = self.runner.occupied_inflated_mask(raw, msg.info.resolution)
        report = self.runner.estimate_corridor_center(msg, observed_wall_evidence)

        unknown_cell = self.runner.local_xy_to_cell(1.0, 0.5, msg)
        self.assertIsNotNone(unknown_cell)
        unknown_x, unknown_y = unknown_cell
        self.assertTrue(collision_blocked[unknown_y, unknown_x])
        self.assertFalse(observed_wall_evidence.any())
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "insufficient_bilateral_wall_support")
        self.assertEqual(report["wall_evidence_semantics"], "occupied_euclidean_inflated_only_unknown_excluded")

    def test_observed_occupied_walls_still_support_a_center(self):
        msg = self.grid()
        for x_m in np.arange(0.9, 1.21, 0.05):
            self.set_cell(msg, x_m, 0.8, 100)
            self.set_cell(msg, x_m, -0.8, 100)

        raw = self.runner.grid_array(msg)
        observed_wall_evidence = self.runner.occupied_inflated_mask(raw, msg.info.resolution)
        report = self.runner.estimate_corridor_center(msg, observed_wall_evidence)

        self.assertTrue(report["applied"])
        self.assertEqual(report["reason"], "bilateral_wall_center_estimated")
        self.assertLessEqual(abs(report["estimated_center_y_m"]), msg.info.resolution)

    def test_additional_clearance_margin_only_expands_occupied_collision_mask(self):
        raw = np.zeros((21, 21), dtype=np.int16)
        raw[10, 16] = 100  # 0.30 m away at 0.05 m resolution.
        baseline = self.runner.occupied_inflated_mask(raw, 0.05)
        conservative = BlockAStarDwaRunner.__new__(BlockAStarDwaRunner)
        conservative.args = build_arg_parser().parse_args(["--additional-clearance-margin-m", "0.05"])
        margin_mask = conservative.occupied_inflated_mask(raw, 0.05)
        self.assertFalse(baseline[10, 10])
        self.assertTrue(margin_mask[10, 10])

    def test_large_opening_asymmetry_cannot_replace_centerline_target(self):
        report = {
            "applied": True,
            "estimated_center_y_m": -0.35,
        }

        adjusted_y, report = self.runner.apply_corridor_center_target(0.0, report)

        self.assertEqual(adjusted_y, 0.0)
        self.assertFalse(report["target_adjustment_applied"])
        self.assertEqual(report["target_adjustment_reason"], "target_correction_exceeds_limit")

    def test_small_bilateral_correction_remains_enabled(self):
        report = {
            "applied": True,
            "estimated_center_y_m": -0.10,
        }

        adjusted_y, report = self.runner.apply_corridor_center_target(0.0, report)

        self.assertAlmostEqual(adjusted_y, -0.075)
        self.assertTrue(report["target_adjustment_applied"])
        self.assertEqual(report["target_adjustment_reason"], "bounded_bilateral_wall_center")

    def test_correction_is_explicitly_limited_to_pre_room_zone_targets(self):
        pre_room_target = {
            "source": "state_machine_corridor_centerline_door_search",
            "subgoal_source": "state_machine_centerline",
            "corridor_center_target_scope": "pre_room_zone_only",
        }
        room_zone_target = dict(pre_room_target, corridor_center_target_scope="disabled_in_room_zone")

        self.assertTrue(self.runner.pre_room_zone_centerline_correction_authorized(pre_room_target))
        self.assertFalse(self.runner.pre_room_zone_centerline_correction_authorized(room_zone_target))

    def test_pre_room_scope_reuses_bounded_centerline_feedback_only(self):
        original_w = 0.0
        corrected_w, report = self.runner.apply_scoped_centerline_tracking_correction(
            0.60,
            original_w,
            {
                "source": "state_machine_corridor_centerline_door_search",
                "subgoal_source": "state_machine_centerline",
                "corridor_center_target_scope": "pre_room_zone_only",
                "entry_anchor_line": {"x": 0.0, "y": 0.0, "heading_rad": 0.0},
            },
            (0.0, 0.20, 0.0),
            True,
            {},
            pre_room_zone_centerline_correction=True,
        )

        self.assertLess(corrected_w, original_w)
        self.assertTrue(report["enabled"])
        self.assertTrue(report["active"])
        self.assertEqual(report["reason"], "centerline_tracking_applied")
        self.assertEqual(report["scope"], "pre_room_zone_only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
