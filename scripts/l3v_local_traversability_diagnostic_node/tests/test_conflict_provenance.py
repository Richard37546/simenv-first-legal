#!/usr/bin/env python3
"""Synthetic audit-only checks for bounded L3V conflict provenance."""

import ast
import sys
import unittest
from pathlib import Path

import numpy as np


NODE = Path(__file__).resolve().parents[1] / "l3v_local_traversability_node.py"
SOURCE = NODE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def load_method(name):
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == "L3VLocalTraversabilityNode":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    module = ast.Module(body=[item], type_ignores=[])
                    namespace = {
                        "np": np,
                        "ratio": lambda numerator, denominator: float(numerator / denominator) if denominator else 0.0,
                        "CONFLICT_RATIO_THRESHOLD": 0.05,
                        "CONFLICT_PROVENANCE_SCHEMA_VERSION": "l3v_conflict_provenance_v1",
                    }
                    exec(compile(ast.fix_missing_locations(module), str(NODE), "exec"), namespace)
                    return namespace[name]
    raise AssertionError(f"method missing: {name}")


LABELS = load_method("labels_locked")
PROVENANCE = load_method("conflict_provenance_locked")


class Fixture:
    def __init__(self, shape=(10, 20)):
        self.rows, self.cols = shape
        self.lidar_free = np.zeros(shape, dtype=np.float32)
        self.rgbd_free = np.zeros(shape, dtype=np.float32)
        self.traversed = np.zeros(shape, dtype=np.float32)
        self.lidar_occ = np.zeros(shape, dtype=np.float32)
        self.rgbd_occ = np.zeros(shape, dtype=np.float32)
        self.current_lidar_free_support = np.zeros(shape, dtype=bool)
        self.current_lidar_occ_support = np.zeros(shape, dtype=bool)
        self.latest_lidar_source_stamp = 10.0
        self.latest_odom_source_stamp = 9.0
        self.last_source_stamp = {"rgbd": 8.0}
        self.conflict_provenance_enabled = True
        self.conflict_provenance_max_cells = 3
        self.conflict_provenance_run_id = "synthetic-run"
        self.conflict_cell_lifecycle = {}
        self.x_min = -0.3
        self.y_min = -1.5
        self.resolution = 0.05


class ConflictProvenanceTests(unittest.TestCase):
    def test_free_only_and_occupied_only_remain_nonconflict(self):
        fixture = Fixture()
        fixture.lidar_free[1, 1] = 1.0
        fixture.lidar_occ[2, 2] = 1.0
        labels = LABELS(fixture)
        self.assertEqual(labels[1, 1], "free")
        self.assertEqual(labels[2, 2], "lidar_supported_obstacle")

    def test_conflict_record_preserves_source_support_and_time_limits(self):
        fixture = Fixture()
        fixture.traversed[2, 3] = 1.0
        fixture.lidar_occ[2, 3] = 2.0
        fixture.current_lidar_occ_support[2, 3] = True
        labels = LABELS(fixture)
        self.assertEqual(labels[2, 3], "conflict")
        record = PROVENANCE(fixture, labels, content_stamp_sec=11.0, status_transition=True)
        self.assertEqual(record["conflict_cell_count"], 1)
        self.assertAlmostEqual(record["conflict_ratio"], 1.0)
        self.assertEqual(record["conflict_ratio_threshold"], 0.05)
        cell = record["representative_conflict_cells"][0]
        self.assertEqual(cell["cell_index_xy"], [2, 3])
        self.assertEqual(cell["occupied_evidence"][0]["source"], "LIDAR_OCC_ENDPOINT_ACCUMULATED")
        self.assertIn("PER_CELL_EVIDENCE_TIMESTAMPS_UNAVAILABLE", cell["cell_time_limit"])

    def test_conflict_ratio_below_and_above_existing_threshold_are_distinguishable(self):
        below = Fixture()
        below.traversed[0, :] = 1.0
        below.traversed[1, :] = 1.0
        below.lidar_occ[0, 0] = 1.0
        self.assertLess(PROVENANCE(below, LABELS(below), content_stamp_sec=1.0, status_transition=True)["conflict_ratio"], 0.05)
        above = Fixture()
        above.traversed[0, :] = 1.0
        above.lidar_occ[0, 0] = 1.0
        above.lidar_occ[0, 1] = 1.0
        self.assertGreater(PROVENANCE(above, LABELS(above), content_stamp_sec=1.0, status_transition=True)["conflict_ratio"], 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
