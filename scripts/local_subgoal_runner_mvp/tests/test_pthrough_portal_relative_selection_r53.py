#!/usr/bin/env python3
"""R53 Portal-dominance selection tests; pure DWA scoring, no ROS runtime."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

import block_astar_dwa_mature_runner as runner_module


class PThroughPortalRelativeSelectionTests(unittest.TestCase):
    def make_runner(self):
        args = runner_module.build_arg_parser().parse_args(
            [
                "--max-linear-x", "0.60",
                "--max-angular-z", "0.35",
                "--min-linear-x", "0.30",
                "--enforce-min-forward-speed",
                "--dwa-predict-time", "1.0",
                "--max-target-heading-correction-rad", "0.85",
                "--target-heading-blend-weight", "0.90",
                "--target-lateral-correction-angular-z", "0.24",
                "--target-lateral-correction-weight", "0.32",
            ]
        )
        runner = object.__new__(runner_module.BlockAStarDwaRunner)
        runner.args = args
        runner.prev_cmd = (0.30, 0.245)
        runner.dynamic_window = lambda: (np.array([0.30, 0.60]), np.array([0.245, 0.35]))
        runner.collision_free_arc = lambda _grid, _blocked, _v, _w: (True, 0.50)
        return runner

    def choose(self, portal_enabled=True, pose_odom=(0.0, 0.0, 0.0), p_pre_odom=None):
        return self.make_runner().choose_dwa(
            None,
            np.zeros((1, 1), dtype=bool),
            (0.825, 0.025),
            (1.804, 0.835),
            1.988,
            p_through_safe_moving_eligibility=True,
            p_through_target_relative_angular_scoring=True,
            portal_relative_selection=portal_enabled,
            portal_center_odom=(0.0, 3.0),
            portal_normal_odom=(0.0, 1.0),
            portal_p_pre_odom=p_pre_odom,
            pose_odom=pose_odom,
            target_in_front=True,
            astar_path_exists=True,
        )

    def test_late_pre_crossing_prefers_tangent_reduction_within_near_equal_inward_heading_candidates(self):
        _v, _w, detail = self.choose(True)
        self.assertTrue(detail["portal_relative_selection_active"])
        self.assertTrue(detail["portal_relative_override_applied"])
        self.assertTrue(detail["portal_pre_crossing_early_turn_selection_applied"])
        self.assertEqual(
            detail["portal_selection_reason"],
            "pre_crossing_soft_tangent_reduction_within_yaw_slack",
        )
        selected = detail["portal_selected_winner"]
        self.assertGreaterEqual(selected["portal_normal_progress_m"], 0.0)
        self.assertGreater(detail["portal_heading_safe_candidate_count"], 0)
        self.assertGreater(detail["portal_tangent_preference_candidate_count"], 0)

    def test_distant_pre_crossing_keeps_existing_inward_heading_selection_without_p_pre_geometry(self):
        _v, _w, detail = self.choose(True, pose_odom=(1.0, 0.0, 0.0))
        self.assertTrue(detail["portal_pre_crossing_early_turn_selection_applied"])
        self.assertEqual(detail["portal_selection_reason"], "pre_crossing_min_portal_inward_yaw_error")
        self.assertEqual(detail["portal_tangent_preference_candidate_count"], 0)

    def test_p_pre_geometry_defines_continuous_arc_reference_without_distance_threshold(self):
        portal_center = (0.0, 0.0)
        portal_normal = (0.0, 1.0)
        p_pre = (1.0, -1.0)
        start = runner_module.p_through_arc_reference(
            (1.0, -1.0, 0.0), portal_center, portal_normal, p_pre,
        )
        midpoint = runner_module.p_through_arc_reference(
            (1.0 - 2.0 ** -0.5, -1.0 + (1.0 - 2.0 ** -0.5), 0.0),
            portal_center, portal_normal, p_pre,
        )
        end = runner_module.p_through_arc_reference(
            (0.0, 0.0, 0.0), portal_center, portal_normal, p_pre,
        )
        self.assertAlmostEqual(start["radius_m"], 1.0)
        self.assertAlmostEqual(start["theta_rad"], 0.0)
        self.assertAlmostEqual(abs(start["heading_rad"]), 3.141592653589793)
        self.assertAlmostEqual(midpoint["theta_rad"], 0.25 * 3.141592653589793, places=5)
        self.assertAlmostEqual(midpoint["heading_rad"], 0.75 * 3.141592653589793, places=5)
        self.assertAlmostEqual(end["theta_rad"], 0.5 * 3.141592653589793, places=5)
        self.assertAlmostEqual(end["heading_rad"], 0.5 * 3.141592653589793, places=5)

    def test_p_pre_arc_ranks_safe_candidates_by_predicted_endpoint_error(self):
        _v, _w, detail = self.choose(
            True,
            pose_odom=(-1.0, 2.0, 0.0),
            p_pre_odom=(-1.0, 2.0),
        )
        self.assertTrue(detail["portal_arc_reference_selection_applied"])
        self.assertGreater(detail["portal_arc_tracking_candidate_count"], 0)
        self.assertEqual(
            detail["portal_selection_reason"],
            "pre_crossing_p_pre_portal_arc_endpoint_tracking",
        )
        selected = detail["portal_selected_winner"]
        self.assertIn("portal_arc_endpoint_error_m", selected)
        self.assertIn("portal_arc_target_odom_xy", selected)

    def test_post_crossing_preserves_existing_portal_dominance(self):
        _v, _w, detail = self.choose(True, pose_odom=(0.0, 4.0, 0.0))
        self.assertFalse(detail["portal_pre_crossing_early_turn_selection_applied"])
        self.assertEqual(detail["portal_selection_reason"], "post_crossing_existing_portal_dominance")
        self.assertGreater(detail["portal_dominating_candidate_count"], 0)

    def test_non_pthrough_portal_logic_is_inactive(self):
        runner = self.make_runner()
        _v, _w, detail = runner.choose_dwa(
            None, np.zeros((1, 1), dtype=bool), (0.825, 0.025), (1.804, 0.835), 1.988,
            portal_relative_selection=False, target_in_front=True, astar_path_exists=True,
        )
        self.assertFalse(detail["portal_relative_selection_active"])
        self.assertFalse(detail["portal_relative_override_applied"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
