#!/usr/bin/env python3
"""R41 profile-binding tests; no ROS nodes, publishers, or robot motion."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

import navigation_state_machine as nav


def option_value(command, option):
    return command[command.index(option) + 1]


class PortalPThroughProfileTests(unittest.TestCase):
    def setUp(self):
        self.args = nav.build_arg_parser().parse_args([])

    def test_portal_pthrough_uses_only_low_level_room_entry_profile(self):
        command = nav.runner_cmd(self.args, state="PORTAL_P_THROUGH", runtime_sec=35.0, max_steps=10)
        self.assertTrue(nav.uses_room_entry_local_nav_profile("PORTAL_P_THROUGH"))
        self.assertIn("--disable-pointcloud-wall-heading", command)
        self.assertEqual(option_value(command, "--min-linear-x"), "0.3")
        self.assertEqual(option_value(command, "--max-angular-z"), str(self.args.p_through_max_angular_z))
        self.assertEqual(option_value(command, "--max-angular-accel"), str(self.args.enter_room_max_angular_accel))
        self.assertEqual(option_value(command, "--target-heading-blend-weight"), str(self.args.enter_room_target_heading_blend_weight))
        self.assertEqual(option_value(command, "--max-target-heading-correction-rad"), str(self.args.enter_room_max_target_heading_correction_rad))

    def test_follow_corridor_does_not_receive_room_entry_profile(self):
        command = nav.runner_cmd(self.args, state="FOLLOW_CORRIDOR", runtime_sec=35.0, max_steps=10)
        self.assertFalse(nav.uses_room_entry_local_nav_profile("FOLLOW_CORRIDOR"))
        self.assertNotIn("--target-heading-blend-weight", command)
        self.assertNotIn("--max-target-heading-correction-rad", command)
        self.assertNotIn("--target-lateral-correction-angular-z", command)

    def test_portal_phase_budgets_are_independent_from_general_runner_budget(self):
        self.assertEqual(nav.portal_g14_p_pre_max_steps(self.args), 20)
        self.assertEqual(nav.portal_g14_p_through_max_steps(self.args), 16)
        self.assertEqual(self.args.enter_room_max_steps, 10)
        p_pre_command = nav.runner_cmd(
            self.args,
            state="FOLLOW_CORRIDOR",
            runtime_sec=self.args.runner_runtime_sec,
            max_steps=nav.portal_g14_p_pre_max_steps(self.args),
        )
        p_through_command = nav.runner_cmd(
            self.args,
            state="PORTAL_P_THROUGH",
            runtime_sec=self.args.runner_runtime_sec,
            max_steps=nav.portal_g14_p_through_max_steps(self.args),
        )
        self.assertEqual(option_value(p_pre_command, "--max-steps"), "20")
        self.assertEqual(option_value(p_through_command, "--max-steps"), "16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
