#!/usr/bin/env python3
"""Focused offline contract tests for mission-local visited Portal memory V1."""

import unittest
from pathlib import Path

from test_odom_cache import FakeRos, load_module


class VisitedPortalMemoryV1Tests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())
        self.anchor = {"x": 0.0, "y": 0.0, "heading_rad": 0.0}
        self.tolerance = self.module.VISITED_PORTAL_IDENTITY_TOLERANCE_M
        self.centre = [17.077644660332535, -1.454312995325242]

    def record(self, centre=None):
        value = list(centre if centre is not None else self.centre)
        return {
            "corridor_progress_m": self.module.portal_corridor_progress_m(value, self.anchor),
            "portal_center_odom": value,
            "entered": True,
            "completed": False,
        }

    def test_a_identical_portal_matches(self):
        self.assertTrue(self.module.same_physical_portal(self.centre, self.record(), self.anchor))

    def test_b_variation_inside_tolerance_matches(self):
        fresh = [self.centre[0] + self.tolerance - 1.0e-6, self.centre[1]]
        self.assertTrue(self.module.same_physical_portal(fresh, self.record(), self.anchor))

    def test_c_progress_difference_above_tolerance_does_not_match(self):
        fresh = [self.centre[0] + self.tolerance + 1.0e-6, self.centre[1]]
        self.assertFalse(self.module.same_physical_portal(fresh, self.record(), self.anchor))

    def test_d_centre_distance_above_tolerance_does_not_match(self):
        fresh = [self.centre[0], self.centre[1] + self.tolerance + 1.0e-6]
        self.assertFalse(self.module.same_physical_portal(fresh, self.record(), self.anchor))

    def test_e_exactly_at_tolerance_matches(self):
        fresh = [self.centre[0] + self.tolerance, self.centre[1]]
        self.assertTrue(self.module.same_physical_portal(fresh, self.record(), self.anchor))

    def test_f_same_station_opposite_portal_does_not_match(self):
        fresh = [self.centre[0], self.centre[1] + 2.20]
        self.assertFalse(self.module.same_physical_portal(fresh, self.record(), self.anchor))

    def test_g_different_station_does_not_match(self):
        fresh = [self.centre[0] + 14.03, self.centre[1]]
        self.assertFalse(self.module.same_physical_portal(fresh, self.record(), self.anchor))

    def test_h_side_flip_is_not_an_identity_input(self):
        stored = self.record()
        stored["side"] = "right"
        self.assertTrue(self.module.same_physical_portal(self.centre, stored, self.anchor))

    def test_i_track_id_change_is_not_an_identity_input(self):
        stored = self.record()
        stored["track_id"] = 99
        self.assertTrue(self.module.same_physical_portal(self.centre, stored, self.anchor))

    def test_run0132_reobservation_bound_matches_but_opposite_opening_does_not(self):
        reobserved = [self.centre[0] + 0.15989977061929084, self.centre[1]]
        self.assertTrue(self.module.same_physical_portal(reobserved, self.record(), self.anchor))
        opposite_opening = [self.centre[0], self.centre[1] + 2.20]
        self.assertFalse(self.module.same_physical_portal(opposite_opening, self.record(), self.anchor))

    def test_lifecycle_is_passive_and_completion_requires_door_return_success(self):
        visited = []
        # Detection alone has no lifecycle callback and cannot create a record.
        self.assertEqual(visited, [])
        entered = self.module.record_entered_visited_portal(visited, self.centre, self.anchor)
        self.assertEqual(entered["state"], "VISITED_PORTAL_CREATED")
        self.assertTrue(entered["record"]["entered"])
        self.assertFalse(entered["record"]["completed"])
        reobserved = [self.centre[0] + 0.10, self.centre[1]]
        matched = self.module.record_entered_visited_portal(visited, reobserved, self.anchor)
        self.assertEqual(matched["state"], "VISITED_PORTAL_MATCHED_EXISTING")
        self.assertEqual(len(visited), 1)
        different = [self.centre[0] + 14.03, self.centre[1]]
        separate = self.module.record_entered_visited_portal(visited, different, self.anchor)
        self.assertEqual(separate["state"], "VISITED_PORTAL_CREATED")
        self.assertEqual(len(visited), 2)
        self.module.mark_visited_portal_completed(entered["record"])
        self.assertTrue(entered["record"]["completed"])

    def test_parent_lifecycle_wires_completion_only_to_existing_return_success(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        reached_hook = source.index('if p_through_outcome.get("reached"):')
        return_gate = source.index('room_search_v2_result.get("final_decision") == "ROOM_SEARCH_V2_RETURNED_TO_DOOR_ANCHOR"')
        completion_update = source.index('mark_visited_portal_completed(entered_portal_memory["record"])')
        self.assertLess(reached_hook, return_gate)
        self.assertLess(return_gate, completion_update)

    def test_ambiguous_match_is_not_silently_selected(self):
        visited = [self.record(), self.record()]
        result = self.module.record_entered_visited_portal(visited, self.centre, self.anchor)
        self.assertEqual(result["state"], "VISITED_PORTAL_IDENTITY_AMBIGUOUS_MATCH")
        self.assertIsNone(result["record"])


if __name__ == "__main__":
    unittest.main()
