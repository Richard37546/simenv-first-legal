#!/usr/bin/env python3
"""Deterministic R33 helper tests; no ROS publisher or robot motion."""

import copy
import json
import sys
import unittest
from pathlib import Path

import rosbag


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

import navigation_state_machine as nav
from local_grid_contract import grid_content_hash


BAG = ROOT / "debug/odom_accuracy_audit_v1/p2kg9_current_door_portal_shadow_048/online_run/p2kg15_093r22_online_20260809_184320/bag/p2kg15_093r22_online_20260809_184320.bag"
SUMMARY = ROOT / "debug/state_machine_navigation/run_archives/run_0072_20260809_184340_361324836_pid2600123/state_machine_navigation_summary.json"
SELECTED = ROOT / "audit_reports/p2kg15_swept_geometry_r33_offline_search.json"


def frozen_target(value):
    if isinstance(value, dict):
        if value.get("P_through_odom") and value.get("frozen_geometry", {}).get("geometry_valid"):
            return value
        for child in value.values():
            found = frozen_target(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = frozen_target(child)
            if found:
                return found
    return None


class StairMovingTurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected_pose = tuple(json.loads(SELECTED.read_text())["selected_p_pre"]["pose_odom"])
        source = frozen_target(json.loads(SUMMARY.read_text()))
        cls.target = {
            "P_pre_odom": list(cls.selected_pose),
            "P_through_odom": source["P_through_odom"],
            "portal_width_m": source["portal_width_m"],
            "frozen_geometry": source["frozen_geometry"],
        }
        grids, statuses = [], []
        with rosbag.Bag(str(BAG)) as bag:
            for topic, message, stamp in bag.read_messages(topics=["/team/local_traversability_grid", "/team/traversability_status"]):
                if topic == "/team/local_traversability_grid" and 40.30 < stamp.to_sec() < 40.36:
                    grids.append(message)
                elif topic == "/team/traversability_status" and 40.30 < stamp.to_sec() < 40.36:
                    statuses.append(json.loads(message.data))
        cls.grid = grids[-1]
        cls.status = min(statuses, key=lambda item: abs(item["grid_content_stamp"] - cls.grid.header.stamp.to_sec()))

    def formal_grid(self, occupied=False):
        grid = copy.deepcopy(self.grid)
        grid.data = [100 if occupied else 0] * len(grid.data)
        status = dict(self.status)
        status["grid_content_hash"] = grid_content_hash(
            grid,
            status["producer_instance_id"],
            status["content_generation_id"],
            status["grid_content_stamp"],
        )
        return grid, status

    def test_portal_relative_staging_is_not_old_normal_stop(self):
        normal = (0.0, 1.0)
        tangent = (-1.0, 0.0)
        pre = (
            -nav.STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M * normal[0] + nav.STAIR_MOVING_TURN_P_PRE_TANGENT_M * tangent[0],
            -nav.STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M * normal[1] + nav.STAIR_MOVING_TURN_P_PRE_TANGENT_M * tangent[1],
        )
        self.assertNotAlmostEqual(pre[0], 0.0)
        self.assertLess(pre[1], -nav.DISCRETE_INFLATION_REACH_M)

    def test_p_pre_crossing_stays_inside_discrete_portal_clearance(self):
        """A fixed historical runway must not put the quarter-arc on a jamb."""
        candidate = {
            "portal_width": 1.15,
            "portal_run_id": "test",
            "portal_frame_sequence": 1,
            "portal_source_stamp": 1.0,
            "portal_track_id": "left-test",
            "side": "left",
            "portal_center_base": [0.0, 0.0],
            "portal_normal_base": [0.0, 1.0],
            "left_boundary": [-0.575, 0.0],
            "right_boundary": [0.575, 0.0],
        }
        binding = {
            "binding_valid": True,
            "source_pose_x_y_yaw": [0.0, 0.0, 0.0],
        }
        result = nav.build_g14_shadow_target(
            candidate,
            binding,
            corridor_approach_direction=[1.0, 0.0],
        )
        relative = result["p_pre_portal_relative"]
        expected_limit = (
            0.5 * 1.15
            - nav.DISCRETE_INFLATION_REACH_M
            - 0.5 * nav.GRID_RESOLUTION_M
            - nav.PORTAL_JAMB_RASTER_GUARD_M
        )
        self.assertTrue(relative["width_aware_tangent_clamp_applied"])
        self.assertAlmostEqual(relative["crossing_tangent_limit_m"], expected_limit)
        self.assertEqual(relative["portal_jamb_raster_guard_m"], nav.GRID_RESOLUTION_M)
        self.assertAlmostEqual(relative["projected_crossing_tangent_offset_m"], expected_limit)
        self.assertLessEqual(
            relative["projected_crossing_tangent_offset_m"],
            relative["crossing_tangent_limit_m"] + 1e-12,
        )

    def test_forward_path_is_bounded_and_swept_safe(self):
        grid, status = self.formal_grid()
        result = nav.build_stair_moving_turn_entry_path(
            self.target, self.selected_pose, grid, status, nav.ROBOT_STATIC_RADIUS_M,
        )
        self.assertTrue(result["feasible"])
        self.assertGreater(abs(result["kappa_m_inv"]), 0.0)
        self.assertLessEqual(abs(result["kappa_m_inv"]), nav.STAIR_MOVING_TURN_KAPPA_MAX_M_INV)
        self.assertTrue(result["selected"]["swept_grid"]["safe"])

    def test_occupied_grid_fails_closed(self):
        grid, status = self.formal_grid(occupied=True)
        result = nav.build_stair_moving_turn_entry_path(
            self.target, self.selected_pose, grid, status, nav.ROBOT_STATIC_RADIUS_M,
        )
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "NO_FULL_SWEPT_FORWARD_STAIR_PATH")


if __name__ == "__main__":
    unittest.main(verbosity=2)
