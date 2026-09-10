#!/usr/bin/env python3
"""R55 lookahead handoff tests; no ROS runtime or motion."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

import block_astar_dwa_mature_runner as runner_module


class PThroughExecutableLookaheadTests(unittest.TestCase):
    def make_runner(self, collision_safe=True):
        args = runner_module.build_arg_parser().parse_args(
            [
                "--max-linear-x", "0.60",
                "--min-linear-x", "0.30",
                "--dwa-predict-time", "1.0",
                "--dwa-dt", "0.1",
            ]
        )
        runner = object.__new__(runner_module.BlockAStarDwaRunner)
        runner.args = args
        runner.dynamic_window = lambda: (np.array([0.30]), np.array([0.0]))
        runner.collision_free_arc = lambda _grid, _blocked, _v, _w: (collision_safe, 0.50)
        runner.cell_to_local_xy = lambda cell, _grid: tuple(cell)
        return runner

    def test_native_lookahead_is_preserved_when_it_is_forward_executable(self):
        runner = self.make_runner(True)
        path = [(0.0, 0.0), (0.25, -0.10), (0.80, -0.20)]
        result = runner.select_p_through_executable_lookahead(
            path, None, np.zeros((1, 1), dtype=bool), (0.80, -0.20, 2), 1.0
        )
        self.assertTrue(result["native_lookahead_forward_executable"])
        self.assertEqual(result["selected_lookahead_path_index"], 2)
        self.assertFalse(result["lookahead_changed"])
        self.assertEqual(result["lookahead_selection_reason"], "native_lookahead_forward_executable")

    def test_same_path_forward_alternative_is_used_only_when_native_is_not_executable(self):
        runner = self.make_runner(True)
        path = [(0.0, 0.0), (0.30, 0.0), (-0.20, 0.0)]
        result = runner.select_p_through_executable_lookahead(
            path, None, np.zeros((1, 1), dtype=bool), (-0.20, 0.0, 2), 1.0
        )
        self.assertFalse(result["native_lookahead_forward_executable"])
        self.assertEqual(result["selected_lookahead_path_index"], 1)
        self.assertTrue(result["lookahead_changed"])
        self.assertGreater(result["safe_forward_count_for_selected_lookahead"], 0)
        self.assertEqual(result["lookahead_selection_reason"], "same_astar_path_forward_executable_alternative")

    def test_fail_closed_when_existing_collision_checker_accepts_no_forward_arc(self):
        runner = self.make_runner(False)
        path = [(0.0, 0.0), (0.30, 0.0), (0.80, -0.20)]
        result = runner.select_p_through_executable_lookahead(
            path, None, np.zeros((1, 1), dtype=bool), (0.80, -0.20, 2), 1.0
        )
        self.assertIsNone(result["selected_lookahead_xy"])
        self.assertEqual(result["safe_forward_count_for_selected_lookahead"], 0)
        self.assertEqual(result["lookahead_selection_reason"], "P_THROUGH_ASTAR_PATH_NOT_FORWARD_EXECUTABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
