#!/usr/bin/env python3
"""Focused offline contracts for post-room-return next-Portal dispatch V1."""

from pathlib import Path
import unittest

from test_odom_cache import FakeRos, load_module


class NextPortalDispatchV1Tests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())
        self.anchor = {"x": 0.0, "y": 0.0, "heading_rad": 0.0}
        self.a_centre = [17.077644660332535, -1.454312995325242]
        self.b_centre = [17.077644660332535, 0.745687004674758]
        self.a_record = {
            "corridor_progress_m": self.module.portal_corridor_progress_m(self.a_centre, self.anchor),
            "portal_center_odom": list(self.a_centre),
            "entered": True,
            "completed": True,
        }

    @staticmethod
    def prepared(centre, side):
        return {
            "candidate": {"side": side, "portal_track_id": "fresh-observation-only"},
            "portal_center_odom": list(centre),
            "target": {"frozen_geometry": {"portal_center_odom": list(centre)}},
        }

    def decide(self, candidates, visited=None):
        return self.module.next_portal_dispatch_decision(
            candidates,
            [self.a_record] if visited is None else visited,
            self.a_record,
            self.anchor,
        )

    def decide_with_accounting(self, candidates, visited=None):
        accounting = {}
        decision = self.module.next_portal_dispatch_decision(
            candidates,
            [self.a_record] if visited is None else visited,
            self.a_record,
            self.anchor,
            accounting,
        )
        return decision, accounting

    def test_a_completed_portal_reentry_is_blocked_when_fresh_side_flips(self):
        decision = self.decide([self.prepared(self.a_centre, "right")])
        self.assertEqual(decision["state"], "FOLLOW_CORRIDOR_DEEPER")
        self.assertIsNone(decision["candidate"])

    def test_b_unique_same_station_portal_is_selected_when_side_equals_a_history(self):
        decision = self.decide([self.prepared(self.b_centre, "left")])
        self.assertEqual(decision["state"], "NEXT_PORTAL_DISPATCH_COMMIT")
        self.assertEqual(decision["candidate"]["portal_center_odom"], self.b_centre)
        self.assertTrue(self.a_record["completed"])

    def test_changing_candidate_side_alone_does_not_change_dispatch(self):
        left = self.decide([self.prepared(self.b_centre, "left")])
        right = self.decide([self.prepared(self.b_centre, "right")])
        self.assertEqual(left["state"], "NEXT_PORTAL_DISPATCH_COMMIT")
        self.assertEqual(right["state"], "NEXT_PORTAL_DISPATCH_COMMIT")
        self.assertEqual(left["candidate"]["portal_center_odom"], right["candidate"]["portal_center_odom"])

    def test_b_return_marks_new_portal_complete_then_resumes_deeper(self):
        visited = [dict(self.a_record)]
        entered = self.module.record_entered_visited_portal(visited, self.b_centre, self.anchor)
        self.assertTrue(entered["record"]["entered"])
        self.assertFalse(entered["record"]["completed"])
        self.module.mark_visited_portal_completed(entered["record"])
        decision = self.decide(
            [self.prepared(self.a_centre, "left"), self.prepared(self.b_centre, "right")],
            visited=visited,
        )
        self.assertEqual(decision["state"], "FOLLOW_CORRIDOR_DEEPER")

    def test_no_opposite_portal_resumes_deeper_without_error(self):
        decision = self.decide([])
        self.assertEqual(decision["state"], "FOLLOW_CORRIDOR_DEEPER")

    def test_different_station_candidate_is_not_selected(self):
        different_station = [self.b_centre[0] + self.module.VISITED_PORTAL_IDENTITY_TOLERANCE_M + 0.01, self.b_centre[1]]
        decision = self.decide([self.prepared(different_station, "left")])
        self.assertEqual(decision["state"], "FOLLOW_CORRIDOR_DEEPER")

    def test_multiple_fresh_unvisited_opposites_fail_closed(self):
        second_b = [self.b_centre[0], self.b_centre[1] + 0.10]
        decision = self.decide([
            self.prepared(self.b_centre, "left"),
            self.prepared(second_b, "right"),
        ])
        self.assertEqual(decision["state"], "NEXT_PORTAL_DISPATCH_AMBIGUOUS_CURRENT_STATION")
        self.assertIsNone(decision["candidate"])

    def test_reset_contains_only_portal_entry_attempt_state(self):
        reset = self.module.reset_portal_entry_attempt_state()
        self.assertEqual(set(reset), {
            "portal_bound_candidate", "portal_g14_shadow_target",
            "portal_g14_p_pre_admissibility", "portal_g14_p_pre_switch",
            "portal_g14_p_pre_runner", "room_search_v2_result",
        })
        self.assertTrue(all(value is None for value in reset.values()))

    def test_deeper_corridor_target_uses_current_pose_not_room_zone_origin(self):
        original_read_odom = self.module.read_odom
        original_write_absolute_target = self.module.write_absolute_target
        try:
            self.module.read_odom = lambda: {"pose_x_y_yaw": [9.0, 0.0, 0.0]}
            self.module.write_absolute_target = lambda xy, source, target_type, extra: {
                "target_xy_team_livox_odom": xy, "source": source, "extra": extra,
            }
            target = self.module.write_anchor_target(self.anchor, 0.60, "state_machine_corridor_centerline_door_search")
        finally:
            self.module.read_odom = original_read_odom
            self.module.write_absolute_target = original_write_absolute_target
        self.assertEqual(target["target_xy_team_livox_odom"], [9.6, 0.0])

    def test_fresh_preparation_uses_current_legal_effect_and_existing_target_builder(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        start = source.index("def prepare_fresh_post_return_portal_candidates")
        end = source.index("def target_distance_from_pose", start)
        helper = source[start:end]
        self.assertIn("sequence > minimum_frame_sequence", helper)
        self.assertIn("PortalBoundDoorCandidateAuthority._candidate_from_effect", helper)
        self.assertIn("build_g14_shadow_target", helper)
        self.assertIn('target.get("target_valid") is True', helper)

    def test_dispatch_selection_helper_has_no_side_authority(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        start = source.index("def next_portal_dispatch_decision")
        end = source.index("def reset_portal_entry_attempt_state", start)
        helper = source[start:end]
        self.assertNotIn("returned_observation_side", helper)
        self.assertNotIn('candidate.get("side") ==', helper)
        self.assertNotIn('candidate.get("side") !=', helper)

    def test_accounting_unique_b_records_visited_exclusion_and_selected_b(self):
        decision, accounting = self.decide_with_accounting([
            self.prepared(self.a_centre, "right"),
            self.prepared(self.b_centre, "left"),
        ])
        self.assertEqual(decision["state"], "NEXT_PORTAL_DISPATCH_COMMIT")
        self.assertEqual(accounting["final_branch"], "NEXT_PORTAL_DISPATCH_COMMIT")
        self.assertEqual(accounting["selected_index"], 1)
        self.assertEqual(accounting["eligible_current_station_candidates"], 1)
        a_entry, b_entry = accounting["prepared_candidates"]
        self.assertTrue(a_entry["visited_match"])
        self.assertEqual(a_entry["matched_record_index"], 0)
        self.assertTrue(a_entry["matched_record_entered"])
        self.assertTrue(a_entry["matched_record_completed"])
        self.assertEqual(a_entry["disposition"], "EXCLUDED_VISITED")
        self.assertFalse(b_entry["visited_match"])
        self.assertTrue(b_entry["current_station_passed"])
        self.assertEqual(b_entry["disposition"], "SELECTED")

    def test_accounting_zero_after_visited_filter_records_deeper_branch(self):
        decision, accounting = self.decide_with_accounting([self.prepared(self.a_centre, "left")])
        self.assertEqual(decision["state"], "FOLLOW_CORRIDOR_DEEPER")
        self.assertEqual(accounting["final_branch"], "FOLLOW_CORRIDOR_DEEPER")
        self.assertIsNone(accounting["selected_index"])
        self.assertEqual(accounting["eligible_current_station_candidates"], 0)
        entry = accounting["prepared_candidates"][0]
        self.assertTrue(entry["visited_match"])
        self.assertTrue(entry["matched_record_entered"])
        self.assertTrue(entry["matched_record_completed"])
        self.assertEqual(entry["disposition"], "EXCLUDED_VISITED")

    def test_accounting_not_current_station_records_existing_progress_delta(self):
        different_station = [
            self.b_centre[0] + self.module.VISITED_PORTAL_IDENTITY_TOLERANCE_M + 0.01,
            self.b_centre[1],
        ]
        decision, accounting = self.decide_with_accounting([self.prepared(different_station, "left")])
        self.assertEqual(decision["state"], "FOLLOW_CORRIDOR_DEEPER")
        entry = accounting["prepared_candidates"][0]
        self.assertFalse(entry["visited_match"])
        self.assertAlmostEqual(
            entry["progress_delta_to_returned_portal_m"],
            self.module.VISITED_PORTAL_IDENTITY_TOLERANCE_M + 0.01,
        )
        self.assertFalse(entry["current_station_passed"])
        self.assertEqual(entry["disposition"], "EXCLUDED_NOT_CURRENT_STATION")

    def test_accounting_ambiguity_records_both_eligible_without_selection(self):
        second_b = [self.b_centre[0], self.b_centre[1] + 0.10]
        decision, accounting = self.decide_with_accounting([
            self.prepared(self.b_centre, "left"),
            self.prepared(second_b, "right"),
        ])
        self.assertEqual(decision["state"], "NEXT_PORTAL_DISPATCH_AMBIGUOUS_CURRENT_STATION")
        self.assertEqual(accounting["final_branch"], "NEXT_PORTAL_DISPATCH_AMBIGUOUS_CURRENT_STATION")
        self.assertIsNone(accounting["selected_index"])
        self.assertEqual(accounting["eligible_current_station_candidates"], 2)
        self.assertTrue(all(entry["current_station_passed"] for entry in accounting["prepared_candidates"]))
        self.assertEqual(
            [entry["disposition"] for entry in accounting["prepared_candidates"]],
            ["ELIGIBLE_CURRENT_STATION", "ELIGIBLE_CURRENT_STATION"],
        )

    def test_accounting_side_is_diagnostic_only(self):
        left_decision, left_accounting = self.decide_with_accounting([self.prepared(self.b_centre, "left")])
        right_decision, right_accounting = self.decide_with_accounting([self.prepared(self.b_centre, "right")])
        self.assertEqual(left_decision["state"], right_decision["state"])
        self.assertEqual(left_decision["candidate"]["portal_center_odom"], right_decision["candidate"]["portal_center_odom"])
        for key in (
            "visited_match", "corridor_progress_m", "progress_delta_to_returned_portal_m",
            "current_station_passed", "disposition",
        ):
            self.assertEqual(
                left_accounting["prepared_candidates"][0][key],
                right_accounting["prepared_candidates"][0][key],
            )
        self.assertNotEqual(
            left_accounting["prepared_candidates"][0]["side"],
            right_accounting["prepared_candidates"][0]["side"],
        )

    def test_target_preparation_rejection_retains_existing_reason(self):
        original_candidate_from_effect = self.module.PortalBoundDoorCandidateAuthority._candidate_from_effect
        original_target_builder = self.module.build_g14_shadow_target
        try:
            self.module.PortalBoundDoorCandidateAuthority._candidate_from_effect = staticmethod(
                lambda _payload, side, _receive_time: {
                    "side": side,
                    "portal_track_id": "candidate-rejected-by-existing-target-gate",
                    "portal_source_stamp": 10.0,
                }
            )
            self.module.build_g14_shadow_target = lambda *_args: {
                "target_valid": False,
                "rejection_reason": "PORTAL_WIDTH_DISCRETE_ENVELOPE_INSUFFICIENT",
                "frozen_geometry": {"portal_center_odom": [3.0, 4.0]},
            }
            snapshot = {
                "latest_effect_state": {
                    "contract_version": self.module.PORTAL_EFFECT_CONTRACT_VERSION,
                    "input_valid": True,
                    "room_zone_active": True,
                    "left": {"effect_eligible": True, "portal": {"track_id": "candidate-rejected-by-existing-target-gate"}},
                    "right": {"effect_eligible": False},
                },
                "last_portal_frame_sequence": 7,
                "last_effect_receive_time": 11.0,
            }
            accounting = {}
            odom_cache = type(
                "PreparationRejectionOdomCache", (),
                {"pose_at_source_stamp": lambda _self, _stamp: {"binding_valid": True}},
            )()
            prepared = self.module.prepare_fresh_post_return_portal_candidates(
                snapshot, 6, odom_cache, (0.0, 0.0, 0.0), self.anchor, 0.2, accounting,
            )
        finally:
            self.module.PortalBoundDoorCandidateAuthority._candidate_from_effect = staticmethod(original_candidate_from_effect)
            self.module.build_g14_shadow_target = original_target_builder
        self.assertEqual(prepared, [])
        self.assertEqual(accounting["total_effect_candidates_seen"], 1)
        self.assertEqual(accounting["total_prepared_candidates"], 0)
        self.assertEqual(accounting["preparation_rejections"], [{
            "side": "left",
            "track_id": "candidate-rejected-by-existing-target-gate",
            "portal_center_odom": [3.0, 4.0],
            "disposition": "OBSERVED_BUT_REJECTED_BY_EXISTING_GATE",
            "existing_rejection_reason": "PORTAL_WIDTH_DISCRETE_ENVELOPE_INSUFFICIENT",
            "reason_availability": "EXISTING_TARGET_REJECTION_REASON",
        }])

    def test_main_reuses_existing_follow_corridor_entry_chain(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        dispatch = source[source.index('elif state == "NEXT_PORTAL_DISPATCH":'):source.index('elif state == "STUCK_RECOVERY":')]
        self.assertIn('next_state, reason = "FOLLOW_CORRIDOR", "next_portal_dispatch_unique_fresh_unvisited_opposite_portal"', dispatch)
        self.assertIn("portal_candidate_authority.commit_post_room_return_candidate", dispatch)
        self.assertNotIn("run_runner(", dispatch)


if __name__ == "__main__":
    unittest.main()
