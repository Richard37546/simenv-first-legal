#!/usr/bin/env python3
"""R48 candidate-eligibility tests; pure DWA scoring only, no ROS runtime."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

import block_astar_dwa_mature_runner as runner_module


class PThroughSafeMovingEligibilityTests(unittest.TestCase):
    def make_runner(self, forward_is_safe=True):
        args = runner_module.build_arg_parser().parse_args(
            [
                "--max-linear-x", "0.60",
                "--max-angular-z", "0.35",
                "--min-linear-x", "0.30",
                "--enforce-min-forward-speed",
            ]
        )
        runner = object.__new__(runner_module.BlockAStarDwaRunner)
        runner.args = args
        runner.prev_cmd = (0.35, -0.242858705)
        runner.dynamic_window = lambda: (
            np.array([0.0, 0.30]),
            np.array([-0.242858705]),
        )

        def collision_free_arc(_grid, _blocked, v, _w):
            if v == 0.0:
                return True, 0.40
            return (forward_is_safe, 0.10 if forward_is_safe else 0.0)

        runner.collision_free_arc = collision_free_arc
        return runner

    def select(self, runner, **eligibility):
        return runner.choose_dwa(
            None,
            np.zeros((1, 1), dtype=bool),
            (0.425, -0.575),
            (0.486, -0.523),
            0.714,
            None,
            **eligibility,
        )

    def test_pthrough_excludes_zero_only_when_existing_safe_moving_candidate_exists(self):
        v, w, detail = self.select(
            self.make_runner(forward_is_safe=True),
            p_through_safe_moving_eligibility=True,
            target_in_front=True,
            astar_path_exists=True,
        )
        self.assertEqual((v, w), (0.30, -0.242858705))
        self.assertTrue(detail["p_through_safe_moving_available"])
        self.assertEqual(detail["safe_moving_candidate_count"], 1)
        self.assertTrue(detail["zero_speed_excluded_by_p_through_eligibility"])
        self.assertTrue(detail["native_best_was_zero_speed"])

    def test_pthrough_retains_zero_speed_when_no_safe_moving_candidate_exists(self):
        v, w, detail = self.select(
            self.make_runner(forward_is_safe=False),
            p_through_safe_moving_eligibility=True,
            target_in_front=True,
            astar_path_exists=True,
        )
        self.assertEqual((v, w), (0.0, -0.242858705))
        self.assertFalse(detail["p_through_safe_moving_available"])
        self.assertEqual(detail["safe_moving_candidate_count"], 0)
        self.assertFalse(detail["zero_speed_excluded_by_p_through_eligibility"])

    def test_non_pthrough_keeps_native_zero_speed_winner(self):
        v, w, detail = self.select(
            self.make_runner(forward_is_safe=True),
            p_through_safe_moving_eligibility=False,
            target_in_front=True,
            astar_path_exists=True,
        )
        self.assertEqual((v, w), (0.0, -0.242858705))
        self.assertFalse(detail["zero_speed_excluded_by_p_through_eligibility"])

    def test_room_search_reuses_only_existing_safe_moving_candidate(self):
        v, w, detail = self.select(
            self.make_runner(forward_is_safe=True),
            room_search_safe_moving_eligibility=True,
            target_in_front=True,
            astar_path_exists=True,
        )
        self.assertEqual((v, w), (0.30, -0.242858705))
        self.assertTrue(detail["zero_speed_excluded_by_room_search_eligibility"])

    def test_room_search_keeps_zero_when_no_safe_moving_candidate(self):
        v, w, detail = self.select(
            self.make_runner(forward_is_safe=False),
            room_search_safe_moving_eligibility=True,
            target_in_front=True,
            astar_path_exists=True,
        )
        self.assertEqual((v, w), (0.0, -0.242858705))
        self.assertFalse(detail["zero_speed_excluded_by_room_search_eligibility"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
