#!/usr/bin/env python3
"""Offline deterministic tests for base-local SE(2) evidence reprojection."""

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "motion_consistent_evidence.py"
SPEC = importlib.util.spec_from_file_location("motion_consistent_evidence", MODULE)
MOTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOTION)

RESOLUTION = 0.05
X_MIN = -0.30
Y_MIN = -1.50
SHAPE = (66, 60)


def cell_for(x, y):
    return (
        int(math.floor((x - X_MIN) / RESOLUTION)),
        int(math.floor((y - Y_MIN) / RESOLUTION)),
    )


def cell_center(row, column):
    return (X_MIN + (row + 0.5) * RESOLUTION, Y_MIN + (column + 0.5) * RESOLUTION)


def transformed_local(point, old_pose, new_pose):
    ox, oy, oyaw = old_pose
    nx, ny, nyaw = new_pose
    world_x = ox + math.cos(oyaw) * point[0] - math.sin(oyaw) * point[1]
    world_y = oy + math.sin(oyaw) * point[0] + math.cos(oyaw) * point[1]
    dx, dy = world_x - nx, world_y - ny
    return (
        math.cos(nyaw) * dx + math.sin(nyaw) * dy,
        -math.sin(nyaw) * dx + math.cos(nyaw) * dy,
    )


class MotionConsistentEvidenceTests(unittest.TestCase):
    def single(self, x=0.525, y=0.025, value=1.0):
        array = np.zeros(SHAPE, dtype=np.float32)
        row, column = cell_for(x, y)
        array[row, column] = value
        return array, (x, y)

    def warp(self, array, old_pose, new_pose):
        return MOTION.warp_array(array, old_pose, new_pose, RESOLUTION, X_MIN, Y_MIN)

    def assert_world_fixed_cell(self, old_pose, new_pose):
        array, point = self.single()
        warped = self.warp(array, old_pose, new_pose)
        row, column = cell_for(*transformed_local(point, old_pose, new_pose))
        self.assertAlmostEqual(float(warped[row, column]), 1.0)
        self.assertAlmostEqual(float(warped.sum()), 1.0)

    def test_stationary_is_bitwise_identical(self):
        array, _point = self.single(value=2.0)
        warped = self.warp(array, (1.0, 2.0, 0.3), (1.0, 2.0, 0.3))
        np.testing.assert_array_equal(warped, array)

    def test_translation_keeps_obstacle_world_fixed(self):
        self.assert_world_fixed_cell((0.0, 0.0, 0.0), (0.20, -0.10, 0.0))

    def test_rotation_keeps_obstacle_world_fixed(self):
        self.assert_world_fixed_cell((0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2.0))

    def test_translation_and_rotation_keep_obstacle_world_fixed(self):
        self.assert_world_fixed_cell((1.0, -2.0, -0.4), (1.2, -1.7, 0.8))

    def test_historical_obstacle_leaving_window_is_dropped(self):
        array, _point = self.single()
        warped = self.warp(array, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        self.assertEqual(float(warped.sum()), 0.0)

    def test_free_evidence_uses_the_same_motion_contract(self):
        free, point = self.single(x=0.925, y=-0.475, value=3.0)
        old_pose, new_pose = (0.0, 0.0, 0.0), (0.15, 0.20, -math.pi / 2.0)
        warped = self.warp(free, old_pose, new_pose)
        row, column = cell_for(*transformed_local(point, old_pose, new_pose))
        self.assertAlmostEqual(float(warped[row, column]), 3.0)
        self.assertAlmostEqual(float(warped.sum()), 3.0)

    def test_current_obstacle_is_not_suppressed(self):
        current, _point = self.single(value=1.0)
        self.assertEqual(int(np.count_nonzero(current)), 1)
        self.assertEqual(float(current.max()), 1.0)

    def test_quantized_reprojection_does_not_add_historical_support(self):
        """Two source cells may land in one cell, but are not two new scans."""
        array = np.zeros(SHAPE, dtype=np.float32)
        first = cell_for(0.525, 0.025)
        second = cell_for(0.575, 0.025)
        array[first] = 1.0
        array[second] = 1.0
        # A 20-degree heading change quantizes both source cell centres to one
        # target cell at this map resolution.
        warped = self.warp(array, (0.0, 0.0, 0.0), (0.0, 0.0, math.radians(20.0)))
        self.assertEqual(float(warped.max()), 1.0)
        self.assertEqual(float(warped.sum()), 1.0)
        self.assertEqual(int(np.count_nonzero(warped)), 1)

    def test_invalid_odom_is_fail_closed(self):
        self.assertEqual(
            MOTION.motion_is_valid((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), 1.0),
            (False, "odom_nonfinite"),
        )
        self.assertEqual(
            MOTION.motion_is_valid((0.0, 0.0, 0.0), (1.01, 0.0, 0.0), 1.0),
            (False, "odom_motion_jump_exceeds_evidence_window"),
        )
        self.assertEqual(
            MOTION.motion_is_valid((0.0, 0.0, 0.0), (0.0, 0.0, math.pi), 1.0),
            (False, "odom_yaw_jump_exceeds_evidence_window"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
