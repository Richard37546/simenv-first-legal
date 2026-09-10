#!/usr/bin/env python3
"""Minimal regressions for the approved speed and false-NO_PATH repairs."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

import block_astar_dwa_mature_runner as runner_module


class SpeedBoundarySamplingTests(unittest.TestCase):
    def assert_contains_once(self, values, expected):
        hits = [value for value in values if abs(float(value) - expected) <= runner_module.DWA_NUMERIC_EPS]
        self.assertEqual(len(hits), 1, values)

    def test_a1_collapsed_legal_interval_includes_single_minimum_boundary(self):
        values = runner_module.v_samples_with_legal_boundaries(
            np.array([0.10, 0.225, 0.35, 0.475, 0.60]), 0.30, 0.30
        )
        self.assert_contains_once(values, 0.30)

    def test_a2_includes_both_nonempty_legal_interval_boundaries(self):
        values = runner_module.v_samples_with_legal_boundaries(np.array([0.10, 0.225, 0.35, 0.60]), 0.30, 0.48)
        self.assert_contains_once(values, 0.30)
        self.assert_contains_once(values, 0.48)

    def test_a3_preserves_valid_samples_above_the_minimum(self):
        values = runner_module.v_samples_with_legal_boundaries(np.array([0.35, 0.45, 0.60]), 0.30, 0.60)
        for expected in (0.35, 0.45, 0.60):
            self.assert_contains_once(values, expected)

    def test_a4_state_minimums_below_contract_remain_point_three(self):
        self.assertEqual([runner_module.effective_minimum_forward(value) for value in (0.08, 0.12, 0.18)], [0.30, 0.30, 0.30])

    def test_a5_empty_window_does_not_create_an_illegal_boundary(self):
        original = np.array([0.10, 0.20])
        values = runner_module.v_samples_with_legal_boundaries(original, 0.30, 0.60)
        self.assertTrue(np.array_equal(values, original))

    def test_a6_run0097_saved_speed_state_recovers_point_three(self):
        minimum = runner_module.effective_minimum_forward(0.18)
        speed_limit = runner_module.speed_limit_for_distance(0.60, minimum, 0.2503, 1.0)
        values = runner_module.v_samples_with_legal_boundaries(
            np.array([0.10, 0.225, 0.35, 0.475, 0.60]), minimum, speed_limit
        )
        self.assertEqual(speed_limit, 0.30)
        self.assert_contains_once(values, 0.30)


class FineFallbackSafetyTests(unittest.TestCase):
    def make_runner(self):
        args = runner_module.build_arg_parser().parse_args(["--block-size-cells", "4"])
        runner = object.__new__(runner_module.BlockAStarDwaRunner)
        runner.args = args
        return runner

    def test_b1_fine_connected_coarse_disconnected_returns_fine_path(self):
        runner = self.make_runner()
        blocked = np.zeros((12, 12), dtype=bool)
        blocked[:4, :] = True
        blocked[8:, :] = True
        blocked[4, 4] = True
        blocked[4, 8] = True
        start, goal = (1, 5), (10, 5)
        self.assertLess(len(runner.block_astar(blocked, start, goal)), 2)
        path = runner.fine_grid_astar_fallback(blocked, start, goal)
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertTrue(all(not blocked[y_index, x_index] for x_index, y_index in path))
        self.assertTrue(all(abs(b[0] - a[0]) + abs(b[1] - a[1]) == 1 for a, b in zip(path, path[1:])))

    def test_coarse_path_remains_available_without_invoking_the_fallback(self):
        runner = self.make_runner()
        blocked = np.zeros((12, 12), dtype=bool)
        self.assertGreaterEqual(len(runner.block_astar(blocked, (1, 5), (10, 5))), 2)

    def test_b2_true_blocked_wall_remains_no_path(self):
        blocked = np.zeros((12, 12), dtype=bool)
        blocked[:, 6] = True
        self.assertEqual(self.make_runner().fine_grid_astar_fallback(blocked, (1, 5), (10, 5)), [])

    def test_b3_unknown_wall_remains_no_path(self):
        runner = self.make_runner()
        raw = np.zeros((12, 12), dtype=np.int16)
        raw[:, 6] = -1
        self.assertEqual(runner.fine_grid_astar_fallback(runner.inflate_obstacles(raw, 0.05), (1, 5), (10, 5)), [])

    def test_b4_inflated_wall_remains_no_path(self):
        runner = self.make_runner()
        raw = np.zeros((12, 12), dtype=np.int16)
        raw[:, 6] = 100
        self.assertEqual(runner.fine_grid_astar_fallback(runner.inflate_obstacles(raw, 0.05), (1, 5), (10, 5)), [])

    def test_b5_blocked_goal_fails_closed(self):
        blocked = np.zeros((8, 8), dtype=bool)
        blocked[5, 6] = True
        self.assertEqual(self.make_runner().fine_grid_astar_fallback(blocked, (1, 5), (6, 5)), [])

    def test_b6_blocked_start_fails_closed(self):
        blocked = np.zeros((8, 8), dtype=bool)
        blocked[5, 1] = True
        self.assertEqual(self.make_runner().fine_grid_astar_fallback(blocked, (1, 5), (6, 5)), [])

    def test_b7_no_diagonal_corner_cut_is_possible(self):
        blocked = np.zeros((2, 2), dtype=bool)
        blocked[0, 1] = True
        blocked[1, 0] = True
        self.assertEqual(self.make_runner().fine_grid_astar_fallback(blocked, (0, 0), (1, 1)), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
