#!/usr/bin/env python3
"""Focused Stage-1 tests: normal ROOM_SEARCH is coverage-first only."""

import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

from room_search_v1 import PortalAnchor, RoomSearchV2  # noqa: E402


def make_search():
    target = {
        "portal_width_m": 1.2,
        "frozen_geometry": {"portal_center_odom": [0.0, 0.0], "portal_normal_odom": [1.0, 0.0]},
    }
    return RoomSearchV2(PortalAnchor.from_frozen_target(target, (0.0, 0.0, 0.0), 0.0))


def candidate(identifier, target_xy, *, new, occlusion, danger=False, sector=0, heading=0.0):
    return {
        "candidate_id": identifier,
        "target_xy_team_livox_odom": list(target_xy),
        "room_target_xy": list(target_xy),
        "base_xy": list(target_xy),
        "candidate_radius_m": max(0.25, (target_xy[0] ** 2 + target_xy[1] ** 2) ** 0.5),
        "heading_change_rad": heading,
        "sector": sector,
        "visible_room_points": [(100.0 + index, 0.0) for index in range(new)],
        "occlusion_reveal_room_points": [(200.0 + index, 0.0) for index in range(occlusion)],
        "danger_reobserve_supported": danger,
        "danger_reobserve_abs_bearing_rad": 0.1 if danger else None,
        "audit_token": identifier,
    }


class RoomSearchStage1ReductionTests(unittest.TestCase):
    def test_a_danger_remains_ahead_of_higher_normal_coverage(self):
        search = make_search()
        danger = candidate("danger", (1.0, 0.1), new=1, occlusion=0, danger=True, sector=1)
        normal = candidate("normal", (1.0, -0.1), new=20, occlusion=0, sector=2)
        self.assertEqual(search.cheap_rank_candidates([normal, danger])[0]["candidate_id"], "danger")

    def test_b_normal_primary_coverage_beats_occlusion(self):
        search = make_search()
        coverage = candidate("coverage", (1.0, 0.1), new=6, occlusion=0, sector=1)
        occlusion = candidate("occlusion", (1.0, -0.1), new=3, occlusion=2, sector=2)
        ranked = search.cheap_rank_candidates([occlusion, coverage])
        self.assertEqual([row["candidate_id"] for row in ranked], ["coverage", "occlusion"])

    def test_c_within_sector_uses_same_primary_coverage(self):
        search = make_search()

        def enrich(row):
            if row["base_xy"][1] < -0.1:
                return {**row, "visible_room_points": [(index, 0.0) for index in range(3)],
                        "occlusion_reveal_room_points": [(10.0 + index, 0.0) for index in range(2)]}
            return {**row, "visible_room_points": [(20.0 + index, 0.0) for index in range(6)],
                    "occlusion_reveal_room_points": []}

        representatives = search.candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(1.0, -0.20), (1.0, -0.05)], enrich,
        )
        self.assertEqual(representatives[0]["base_xy"], [1.0, -0.05])
        self.assertEqual(representatives[0]["new_observable_cells"], 6)

    def test_d_equal_coverage_remains_deterministic(self):
        search = make_search()
        low_occ = candidate("low_occ", (1.0, 0.1), new=4, occlusion=0, sector=1)
        high_occ = candidate("high_occ", (1.0, -0.1), new=4, occlusion=1, sector=2)
        first = search.cheap_rank_candidates([low_occ, high_occ])
        second = search.cheap_rank_candidates([low_occ, high_occ])
        self.assertEqual(first[0]["candidate_id"], "high_occ")
        self.assertEqual(first, second)

    def test_e_candidate_schema_is_preserved_through_preflight(self):
        search = make_search()
        original = candidate("schema", (1.0, 0.1), new=6, occlusion=0, sector=1, heading=0.2)
        selected, attempts = search.admit_ranked_candidates(
            search.cheap_rank_candidates([original]), lambda row: {"legal": True, "path_length_m": 0.5},
        )
        self.assertEqual(attempts, 1)
        self.assertEqual(selected["candidate_id"], "schema")
        self.assertEqual(selected["target_xy_team_livox_odom"], [1.0, 0.1])
        self.assertEqual(selected["heading_change_rad"], 0.2)
        self.assertEqual(selected["audit_token"], "schema")

    def test_f_preflight_rejection_still_advances_to_lower_legal_candidate(self):
        search = make_search()
        high = candidate("high", (1.0, 0.1), new=6, occlusion=0, sector=1)
        lower = candidate("lower", (1.0, -0.1), new=3, occlusion=2, sector=2)
        selected, attempts = search.admit_ranked_candidates(
            search.cheap_rank_candidates([lower, high]),
            lambda row: {"legal": row["candidate_id"] == "lower"},
        )
        self.assertEqual(attempts, 2)
        self.assertEqual(selected["candidate_id"], "lower")

    def test_g_completion_semantics_are_not_changed(self):
        self.assertEqual(make_search().evaluate_marginal_value({"nbv_value": 0.0, "path_length_m": 1.0}, 1)["completion_reason"], "NON_POSITIVE_GAIN")


if __name__ == "__main__":
    unittest.main()
