#!/usr/bin/env python3
"""Focused tests for task-aware representative selection inside one sector."""

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

from room_search_v1 import (  # noqa: E402
    LOCAL_MAX_RADIUS_M,
    PREFERRED_MAX_RADIUS_M,
    PREFERRED_MIN_RADIUS_M,
    RAW_SECTOR_REPRESENTATIVE_CAP,
    PortalAnchor,
    RoomSearchV2,
)


def make_search():
    target = {
        "portal_width_m": 1.2,
        "frozen_geometry": {"portal_center_odom": [0.0, 0.0], "portal_normal_odom": [1.0, 0.0]},
    }
    return RoomSearchV2(PortalAnchor.from_frozen_target(target, (0.0, 0.0, 0.0), 0.0))


def enrich(occlusion, coverage):
    return lambda row: {
        **row,
        "occlusion_reveal_room_points": [(10.0 + index, 0.0) for index in range(occlusion)],
        "visible_room_points": [(20.0 + index, 0.0) for index in range(coverage)],
    }


class TaskAwareRepresentativeTests(unittest.TestCase):
    def test_t1_occlusion_value_beats_farther_same_sector_point(self):
        search = make_search()
        values = {
            -0.225: enrich(0, 2),
            -0.025: enrich(1, 6),
        }
        candidates = search.candidates_from_planning_free_base(
            (0.0, 0.0, 0.0),
            [(1.175, -0.225), (1.175, -0.025)],
            lambda row: values[round(float(row["base_xy"][1]), 3)](row),
        )
        self.assertEqual(candidates[0]["base_xy"], [1.175, -0.025])
        self.assertEqual(candidates[0]["occlusion_reveal_cells"], 1)
        self.assertEqual(candidates[0]["new_observable_cells"], 6)

    def test_t2_larger_normal_coverage_beats_occlusion(self):
        search = make_search()
        candidates = search.candidates_from_planning_free_base(
            (0.0, 0.0, 0.0),
            [(1.0, -0.20), (1.0, -0.05)],
            lambda row: enrich(1, 3)(row) if float(row["base_xy"][1]) < -0.1 else enrich(0, 20)(row),
        )
        self.assertEqual(candidates[0]["base_xy"], [1.0, -0.05])

    def test_t3_coverage_breaks_equal_occlusion_tie(self):
        search = make_search()
        candidates = search.candidates_from_planning_free_base(
            (0.0, 0.0, 0.0),
            [(1.0, -0.20), (1.0, -0.05)],
            lambda row: enrich(1, 3)(row) if float(row["base_xy"][1]) < -0.1 else enrich(1, 4)(row),
        )
        self.assertEqual(candidates[0]["base_xy"], [1.0, -0.05])

    def test_t4_equal_task_values_keep_existing_farthest_radius_tie_break(self):
        search = make_search()
        candidates = search.candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(1.0, -0.10), (1.1, -0.11)], enrich(0, 0)
        )
        self.assertEqual(candidates[0]["base_xy"], [1.1, -0.11])

    def test_t5_different_sectors_keep_independent_representatives(self):
        candidates = make_search().candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(1.0, -0.1), (1.0, 0.1)], enrich(1, 2)
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len({candidate["sector"] for candidate in candidates}), 2)

    def test_t6_representative_cap_stays_twelve(self):
        points = [(math.cos(-math.pi + (index + 0.5) * math.pi / 6.0), math.sin(-math.pi + (index + 0.5) * math.pi / 6.0)) for index in range(12)]
        candidates = make_search().candidates_from_planning_free_base((0.0, 0.0, 0.0), points, enrich(0, 0))
        self.assertEqual(len(candidates), RAW_SECTOR_REPRESENTATIVE_CAP)

    def test_t7_horizon_and_preferred_band_are_unchanged(self):
        candidates = make_search().candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(0.24, 0.0), (0.25, 0.0), (LOCAL_MAX_RADIUS_M + 0.01, 0.0)], enrich(0, 0)
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual((PREFERRED_MIN_RADIUS_M, PREFERRED_MAX_RADIUS_M, LOCAL_MAX_RADIUS_M), (0.8, 1.2, 1.5))

    def test_t8_sector_boundaries_are_unchanged(self):
        candidates = make_search().candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(1.0, -0.01), (1.0, 0.01)], enrich(0, 0)
        )
        self.assertEqual({candidate["sector"] for candidate in candidates}, {5, 6})

    def test_t9_raw_selection_has_no_formal_preflight_hook(self):
        calls = []
        make_search().candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(1.0, 0.0), (1.0, 0.1)],
            lambda row: calls.append(tuple(row["base_xy"])) or enrich(0, 1)(row),
        )
        self.assertEqual(len(calls), 2)

    def test_t10_existing_occlusion_semantics_are_reused(self):
        candidate = make_search().candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(1.0, 0.0)], enrich(2, 4)
        )[0]
        self.assertEqual((candidate["occlusion_reveal_cells"], candidate["new_observable_cells"]), (2, 4))

    def test_t11_existing_generic_coverage_semantics_are_reused(self):
        candidate = make_search().candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(1.0, 0.0)], enrich(0, 4)
        )[0]
        self.assertEqual((candidate["target_priority_class"], candidate["new_observable_cells"]), ("GENERIC_COVERAGE", 4))

    def test_t12_no_occlusion_still_retains_generic_candidate(self):
        candidate = make_search().candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(1.0, 0.0)], enrich(0, 0)
        )[0]
        self.assertEqual(candidate["target_priority_class"], "GENERIC_COVERAGE")

    def test_t13_empty_sector_has_no_representative(self):
        self.assertEqual(make_search().candidates_from_planning_free_base((0.0, 0.0, 0.0), [], enrich(0, 0)), [])

    def test_t14_repeated_input_is_deterministic(self):
        points = [(1.175, -0.225), (1.175, -0.025), (1.0, 0.2)]
        values = lambda row: enrich(1, 6)(row) if round(float(row["base_xy"][1]), 3) == -0.025 else enrich(0, 2)(row)
        first = make_search().candidates_from_planning_free_base((0.0, 0.0, 0.0), points, values)
        second = make_search().candidates_from_planning_free_base((0.0, 0.0, 0.0), points, values)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
