#!/usr/bin/env python3
"""Focused pure fixtures for the L3V current-cloud low-obstacle classifier."""

import ast
import math
import unittest
from pathlib import Path


NODE = Path(__file__).resolve().parents[1] / "l3v_local_traversability_node.py"
TREE = ast.parse(NODE.read_text(encoding="utf-8"))
NAMES = {
    "OBSTACLE_MIN_BASE_Z_M",
    "OBSTACLE_MAX_BASE_Z_M",
    "NEAR_GROUND_MIN_BASE_Z_M",
    "NEAR_GROUND_MAX_BASE_Z_M",
    "height_classes",
    "low_obstacle_cells_from_current_cloud",
}
NODES = [
    node for node in TREE.body
    if isinstance(node, (ast.Assign, ast.FunctionDef))
    and (
        isinstance(node, ast.FunctionDef) and node.name in NAMES
        or isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in NAMES for target in node.targets)
    )
]
NAMESPACE = {"math": math}
exec(compile(ast.fix_missing_locations(ast.Module(body=NODES, type_ignores=[])), str(NODE), "exec"), NAMESPACE)
HEIGHT_CLASSES = NAMESPACE["height_classes"]
LOW_CELLS = NAMESPACE["low_obstacle_cells_from_current_cloud"]


class LowObstacleHeightReliefTests(unittest.TestCase):
    def test_flat_floor_remains_free(self):
        floor = {(x, y): [-0.31, -0.30, -0.29] for x in range(5) for y in range(5)}
        self.assertEqual(LOW_CELLS(floor, 0.05), set())

    def test_low_cube_above_local_floor_is_supported(self):
        points = {(x, y): [-0.32] for x in range(9) for y in range(9)}
        points[(4, 4)].extend([-0.05, -0.03])
        self.assertIn((4, 4), LOW_CELLS(points, 0.05))

    def test_existing_tall_obstacle_class_is_preserved(self):
        obstacle, near_ground = HEIGHT_CLASSES(0.40)
        self.assertTrue(obstacle)
        self.assertFalse(near_ground)

    def test_noisy_near_ground_does_not_make_sheet(self):
        noise = {(x, y): [-0.34 + 0.01 * ((x + y) % 3)] for x in range(9) for y in range(9)}
        self.assertEqual(LOW_CELLS(noise, 0.05), set())

    def test_low_solid_relief_is_shape_agnostic(self):
        points = {(x, y): [-0.30] for x in range(9) for y in range(9)}
        points[(4, 4)].append(-0.08)
        points[(4, 5)].append(-0.10)
        supported = LOW_CELLS(points, 0.05)
        self.assertTrue({(4, 4), (4, 5)} <= supported)


if __name__ == "__main__":
    unittest.main(verbosity=2)
