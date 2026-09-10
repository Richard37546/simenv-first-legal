#!/usr/bin/env python3
"""R51 pure scoring tests; no ROS runtime or production mutation."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

import block_astar_dwa_mature_runner as runner_module


class PThroughTargetRelativeAngularScoringTests(unittest.TestCase):
    def make_runner(self):
        args = runner_module.build_arg_parser().parse_args(
            [
                "--max-linear-x", "0.60",
                "--max-angular-z", "0.35",
                "--max-angular-accel", "0.50",
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
        runner.prev_cmd = (0.35, -0.24286235)
        runner.dynamic_window = lambda: (
            np.array([0.30]),
            np.array([-0.35, -0.2787, -0.24305]),
        )
        runner.collision_free_arc = lambda _grid, _blocked, _v, _w: (True, 0.50)
        return runner

    def choose(self, enabled):
        return self.make_runner().choose_dwa(
            None,
            np.zeros((1, 1), dtype=bool),
            (0.825, -0.375),
            (1.389, -1.062),
            1.75,
            p_through_safe_moving_eligibility=enabled,
            p_through_target_relative_angular_scoring=enabled,
            target_in_front=True,
            astar_path_exists=True,
        )

    def test_pthrough_uses_existing_target_relative_reference(self):
        _v, w, detail = self.choose(True)
        expected_time = max(1.0, 0.85 / 0.35)
        self.assertAlmostEqual(detail["angular_reference_time_sec"], expected_time)
        self.assertAlmostEqual(
            detail["w_reference"],
            max(-0.35, min(0.35, detail["target_heading_used_rad"] / expected_time)),
        )
        self.assertEqual(detail["angular_penalty_mode"], "p_through_target_relative")
        self.assertAlmostEqual(w, -0.2787)
        self.assertTrue(detail["p_through_safe_moving_available"])

    def test_non_pthrough_retains_absolute_magnitude_penalty(self):
        _v, w, detail = self.choose(False)
        self.assertEqual(detail["angular_penalty_mode"], "absolute_w_magnitude")
        self.assertIsNone(detail["angular_reference_time_sec"])
        self.assertIsNone(detail["w_reference"])
        self.assertAlmostEqual(w, -0.24305)

    def test_reference_penalty_still_penalizes_overshoot(self):
        _v, _w, detail = self.choose(True)
        reference = detail["w_reference"]
        self.assertLess(reference, 0.0)
        self.assertLess(abs(-0.2787 - reference), abs(-0.35 - reference))


if __name__ == "__main__":
    unittest.main(verbosity=2)
