#!/usr/bin/env python3
"""Focused P2 tests for soft room-local trajectory revisit preference."""

import sys
import unittest
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


def candidate(target_xy, visible, *, reveal=(), danger=False, radius=1.0, sector=0):
    return {
        "target_xy_team_livox_odom": list(target_xy),
        "room_target_xy": [2.0, 0.0],
        "visible_room_points": list(visible),
        "occlusion_reveal_room_points": list(reveal),
        "danger_reobserve_supported": danger,
        "candidate_radius_m": radius,
        "heading_change_rad": 0.0,
        "sector": sector,
    }


class RoomSearchRevisitEfficiencyTests(unittest.TestCase):
    def test_t1_equal_value_generic_away_from_recent_route_is_preferred(self):
        search = make_search()
        search.record_actual_breadcrumb((1.0, 1.0, 0.0), 0.0)
        recent = candidate((1.25, 1.25), [(10.0, 0.0), (10.6, 0.0)], radius=0.5, sector=0)
        away = candidate((3.0, 1.0), [(20.0, 0.0), (20.6, 0.0)], radius=1.4, sector=1)
        ranked = search.cheap_rank_candidates([recent, away])
        self.assertTrue(ranked[1]["generic_trajectory_revisit_preference_applied"])
        self.assertFalse(ranked[0]["generic_trajectory_revisit_preference_applied"])
        self.assertEqual(ranked[0]["target_xy_team_livox_odom"], [3.0, 1.0])

    def test_t2_all_generic_revisits_remain_selectable_and_stronger_value_wins(self):
        search = make_search()
        search.record_actual_breadcrumb((1.0, 1.0, 0.0), 0.0)
        strong = candidate((1.25, 1.25), [(10.0 + index, 0.0) for index in range(5)], sector=0)
        weak = candidate((1.75, 1.25), [(20.0, 0.0)], sector=1)
        ranked = search.cheap_rank_candidates([weak, strong])
        selected, attempts = search.admit_ranked_candidates(ranked, lambda row: {"legal": True})
        self.assertEqual(len(ranked), 2)
        self.assertEqual(selected["target_xy_team_livox_odom"], [1.25, 1.25])
        self.assertEqual(attempts, 1)

    def test_t3_danger_priority_is_unchanged_near_old_route(self):
        search = make_search()
        search.record_actual_breadcrumb((1.0, 1.0, 0.0), 0.0)
        danger = candidate((1.25, 1.25), [(10.0, 0.0)], danger=True, sector=0)
        generic = candidate((3.0, 1.0), [(20.0 + index, 0.0) for index in range(10)], sector=1)
        ranked = search.cheap_rank_candidates([generic, danger])
        self.assertTrue(ranked[0]["trajectory_revisit_coarse_location"])
        self.assertFalse(ranked[0]["generic_trajectory_revisit_preference_applied"])
        self.assertEqual(ranked[0]["target_priority_class"], "DANGER_REOBSERVE")

    def test_t4_normal_coverage_beats_occlusion_near_old_route(self):
        search = make_search()
        search.record_actual_breadcrumb((1.0, 1.0, 0.0), 0.0)
        occlusion = candidate((1.25, 1.25), [(10.0, 0.0)], reveal=[(30.0, 0.0)], sector=0)
        generic = candidate((3.0, 1.0), [(20.0 + index, 0.0) for index in range(10)], sector=1)
        ranked = search.cheap_rank_candidates([generic, occlusion])
        self.assertEqual(ranked[0]["target_priority_class"], "GENERIC_COVERAGE")
        self.assertEqual(ranked[0]["target_xy_team_livox_odom"], [3.0, 1.0])

    def test_t5_new_room_search_has_no_previous_room_trajectory_penalty(self):
        old_room = make_search()
        old_room.record_actual_breadcrumb((1.0, 1.0, 0.0), 0.0)
        self.assertTrue(old_room.cheap_rank_candidates([candidate((1.25, 1.25), [(10.0, 0.0)])])[0]["trajectory_revisit_coarse_location"])
        new_room = make_search()
        ranked = new_room.cheap_rank_candidates([candidate((1.25, 1.25), [(10.0, 0.0)])])
        self.assertFalse(ranked[0]["trajectory_revisit_coarse_location"])
        self.assertFalse(ranked[0]["generic_trajectory_revisit_preference_applied"])

    def test_t6_revisit_history_cannot_create_no_safe_candidate_or_completion(self):
        search = make_search()
        search.record_actual_breadcrumb((1.0, 1.0, 0.0), 0.0)
        ranked = search.cheap_rank_candidates([candidate((1.25, 1.25), [(10.0, 0.0)])])
        selected, attempts = search.admit_ranked_candidates(ranked, lambda row: {"legal": True})
        self.assertIsNotNone(selected)
        self.assertEqual(attempts, 1)
        self.assertIsNone(search.evaluate_marginal_value({"nbv_value": 1.0}, 1)["completion_reason"])

    def test_t7_run0126_d10_to_d12_are_coarse_route_revisits_but_d13_is_not(self):
        search = make_search()
        terminals = [
            (11.891404151916504, -3.2197861671447754, 0.0),
            (11.61826229095459, -4.382874488830566, 0.0),
            (11.553107261657715, -5.474282264709473, 0.0),
            (11.530644416809082, -6.566725254058838, 0.0),
            (11.308499336242676, -7.2811598777771, 0.0),
            (11.529056549072266, -8.242772102355957, 0.0),
        ]
        for pose in terminals:
            search.record_actual_breadcrumb(pose, 0.0)
        for target_xy in ((11.261383826925682, -4.758203227247507), (11.102487492490019, -4.53475439069199), (11.016674378716806, -3.6115878980826905)):
            self.assertTrue(search.cheap_rank_candidates([candidate(target_xy, [(10.0, 0.0)])])[0]["trajectory_revisit_coarse_location"])
        self.assertFalse(search.cheap_rank_candidates([candidate((16.005457827292684, -3.8177912045268183), [(10.0, 0.0)])])[0]["trajectory_revisit_coarse_location"])


if __name__ == "__main__":
    unittest.main()
