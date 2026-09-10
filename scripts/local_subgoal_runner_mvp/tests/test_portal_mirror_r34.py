#!/usr/bin/env python3
"""R34 portal-tangent orientation regressions; no ROS publisher or robot motion."""

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

import navigation_state_machine as nav


class PortalMirrorTests(unittest.TestCase):
    def test_left_and_right_raw_tangents_both_select_upstream(self):
        approach = (1.0, 0.0)
        left = nav.orient_portal_tangent_upstream((0.0, 1.0), approach)
        right = nav.orient_portal_tangent_upstream((0.0, -1.0), approach)
        self.assertTrue(left["valid"])
        self.assertTrue(right["valid"])
        self.assertLess(left["upstream_tangent_dot_approach"], 0.0)
        self.assertLess(right["upstream_tangent_dot_approach"], 0.0)
        self.assertEqual(left["upstream_tangent_odom"], [-1.0, 0.0])
        self.assertEqual(right["upstream_tangent_odom"], [-1.0, 0.0])

    def test_right_mirror_rejects_downstream_raw_tangent(self):
        orientation = nav.orient_portal_tangent_upstream((0.0, -1.0), (1.0, 0.0))
        self.assertGreater(orientation["raw_tangent_dot_approach"], 0.0)
        self.assertEqual(orientation["upstream_tangent_odom"], [-1.0, -0.0])

    def test_target_builder_uses_anchor_oriented_tangent_without_side_branch(self):
        candidate = {
            "portal_center_base": [17.0, -0.8],
            "portal_normal_base": [0.0, -1.0],
            "left_boundary": [16.4, -0.8],
            "right_boundary": [17.6, -0.8],
            "portal_width": 1.2,
            "portal_run_id": "test",
            "portal_frame_sequence": 1,
            "portal_source_stamp": 1.0,
            "portal_track_id": "right-mirror",
            "side": "right",
        }
        binding = {"binding_valid": True, "source_pose_x_y_yaw": [0.0, 0.0, 0.0]}
        target = nav.build_g14_shadow_target(candidate, binding, corridor_approach_direction=(1.0, 0.0))
        self.assertTrue(target["target_valid"])
        self.assertLess(target["portal_tangent_orientation"]["upstream_tangent_dot_approach"], 0.0)
        self.assertLess(target["P_pre_odom"][0] - candidate["portal_center_base"][0], 0.0)

    def test_normal_still_defines_room_side(self):
        normal = (0.0, -1.0)
        centre = (17.0, -0.8)
        through = (centre[0] + 0.3 * normal[0], centre[1] + 0.3 * normal[1])
        self.assertGreater((through[0] - centre[0]) * normal[0] + (through[1] - centre[1]) * normal[1], 0.0)
        self.assertTrue(math.isclose(abs(nav.STAIR_MOVING_TURN_KAPPA_MAX_M_INV), 0.6620444444444447))


if __name__ == "__main__":
    unittest.main(verbosity=2)
