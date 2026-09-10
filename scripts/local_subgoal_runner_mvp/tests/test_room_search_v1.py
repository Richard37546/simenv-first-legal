#!/usr/bin/env python3
"""Focused ROS-free ROOM_SEARCH V2-A decision-core tests."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

from room_search_v1 import (
    ANTI_INFINITE_DECISION_GUARD,
    DOOR_NEAR_UTILITY_FACTOR,
    LOW_GAIN_PATIENCE,
    LOW_GAIN_RATIO,
    PortalAnchor,
    RoomSearchV2,
)


def make_search(entry=(10.3, 20.0, 0.2)):
    target = {
        "portal_width_m": 1.2,
        "frozen_geometry": {"portal_center_odom": [10.0, 20.0], "portal_normal_odom": [1.0, 0.0]},
    }
    return RoomSearchV2(PortalAnchor.from_frozen_target(target, entry, 12.0))


class RoomSearchV2Tests(unittest.TestCase):
    def test_t1_actual_odom_is_door_return_anchor(self):
        search = make_search((10.31, 20.02, 0.7))
        self.assertEqual(search.anchor.door_return_anchor_xy_yaw, (10.31, 20.02, 0.7))
        self.assertEqual(search.anchor.door_return_anchor_stamp_sec, 12.0)

    def test_t2_room_local_transform_remains_correct(self):
        search = make_search()
        self.assertEqual(search.anchor.local_xy([12.0, 23.0]), (2.0, 3.0))
        self.assertEqual(search.anchor.odom_xy([2.0, -1.0]), (12.0, 19.0))

    def test_t3_candidates_come_from_current_free_space_not_left_right_route(self):
        search = make_search()
        candidates = search.candidates_from_planning_free_base((10.3, 20.0, 0.0), [(1.0, 0.1), (0.9, -0.7), (1.1, 0.75)])
        self.assertTrue(candidates)
        self.assertNotIn("left", str(candidates).lower())
        self.assertTrue(all("sector" in item and "base_xy" in item for item in candidates))

    def test_t4_candidates_are_local_and_only_use_supplied_free_points(self):
        search = make_search()
        candidates = search.candidates_from_planning_free_base((10.3, 20.0, 0.0), [(1.0, 0.0), (2.0, 0.0), (0.1, 0.0)])
        self.assertEqual(len(candidates), 1)
        self.assertLessEqual(candidates[0]["candidate_radius_m"], 1.5)

    def test_t5_higher_gain_per_path_cost_wins(self):
        search = make_search()
        best = search.select_best([
            {"room_target_xy": [2.0, 0.0], "visible_room_points": [(0, 0)], "path_length_m": 0.5, "heading_change_rad": 0.5},
            {"room_target_xy": [2.0, 1.0], "visible_room_points": [(1, 0), (1, 1), (1, 2)], "path_length_m": 1.0, "heading_change_rad": 0.1},
        ])
        self.assertEqual(best["room_target_xy"], [2.0, 1.0])

    def test_t6_door_keepout_is_soft_not_forbidden(self):
        search = make_search()
        best = search.select_best([
            {"room_target_xy": [0.5, 0.0], "visible_room_points": [(x, 0) for x in range(10)], "path_length_m": 1.0, "heading_change_rad": 0.0},
            {"room_target_xy": [2.0, 0.0], "visible_room_points": [(1, 0)], "path_length_m": 1.0, "heading_change_rad": 0.0},
        ])
        self.assertEqual(best["room_target_xy"], [0.5, 0.0])
        self.assertEqual(best["door_keepout_soft_factor"], DOOR_NEAR_UTILITY_FACTOR)

    def test_t7_observed_area_reduces_later_gain(self):
        search = make_search()
        search.update_actual_view((10.3, 20, 0), [(1, 1), (1.1, 1.1)])
        scored = search.score_candidates([{"room_target_xy": [2, 0], "visible_room_points": [(1, 1)], "path_length_m": 1.0}])
        self.assertEqual(scored[0]["new_observable_cells"], 0)

    def test_t8_search_ends_with_no_positive_gain_not_perfect_coverage(self):
        search = make_search()
        search.update_actual_view((0, 0, 0), [(1, 1)])
        best = search.select_best([{"room_target_xy": [2, 0], "visible_room_points": [(1, 1)], "path_length_m": 1.0}])
        self.assertEqual(search.evaluate_marginal_value(best, 1)["completion_reason"], "NON_POSITIVE_GAIN")

    def test_t9_guard_is_unchanged_24_decisions(self):
        self.assertEqual(ANTI_INFINITE_DECISION_GUARD, 24)

    def test_t10_failures_only_reject_current_context(self):
        search = make_search()
        search.reject_context_target((11.0, 20.0))
        candidates = search.candidates_from_planning_free_base((10.3, 20, 0), [(0.7, 0), (1.0, 0.7)])
        self.assertEqual(len(candidates), 1)

    def test_t11_failed_attempt_motion_can_be_actual_breadcrumb(self):
        search = make_search()
        self.assertTrue(search.record_actual_breadcrumb((11.0, 20.0, 0.0), 0.3))

    def test_t12_actual_breadcrumb_is_not_commanded_target(self):
        search = make_search()
        search.record_actual_breadcrumb((10.8, 20.1, 0.0), 0.3)
        self.assertEqual(search.breadcrumbs[0], (10.8, 20.1, 0.0))

    def test_t13_breadcrumbs_are_available_as_actual_safe_anchors(self):
        search = make_search()
        search.record_actual_breadcrumb((11.0, 20.0, 0.0), 0.3)
        search.record_actual_breadcrumb((12.0, 21.0, 0.0), 0.3)
        self.assertEqual(search.breadcrumbs[0], (11.0, 20.0, 0.0))
        self.assertEqual(search.breadcrumbs[1], (12.0, 21.0, 0.0))

    def test_t14_door_return_target_keeps_navigation_coordinates(self):
        self.assertIn("target_xy_team_livox_odom", make_search().door_return_target())

    def test_t15_final_return_target_is_actual_door_anchor(self):
        target = make_search((10.31, 20.02, 0.7)).door_return_target()
        self.assertEqual(target["kind"], "DOOR_RETURN_ANCHOR")
        self.assertEqual(target["target_xy_team_livox_odom"], [10.31, 20.02])

    def test_t16_no_corridor_exit_target_is_generated(self):
        self.assertNotIn("PORTAL_CORRIDOR_EXIT", str(make_search().door_return_target()))

    def test_t17_anchor_distance_is_the_final_return_predicate_input(self):
        anchor = make_search((10.31, 20.02, 0.7)).anchor
        self.assertLess(((10.4-anchor.door_return_anchor_xy_yaw[0])**2 + (20.02-anchor.door_return_anchor_xy_yaw[1])**2)**0.5, 0.3)

    def test_t18_high_positive_value_does_not_terminate_search(self):
        search = make_search()
        result = search.evaluate_marginal_value({"nbv_value": 10.0, "path_length_m": 0.2}, 1)
        self.assertIsNone(result["completion_reason"])
        self.assertEqual(result["low_gain_streak"], 0)

    def test_t19_one_low_gain_decision_continues(self):
        search = make_search()
        search.evaluate_marginal_value({"nbv_value": 10.0, "path_length_m": 0.2}, 1)
        result = search.evaluate_marginal_value({"nbv_value": 2.0, "path_length_m": 0.2}, 2)
        self.assertEqual(result["low_gain_streak"], 1)
        self.assertIsNone(result["completion_reason"])

    def test_t20_two_low_gain_decisions_return(self):
        search = make_search()
        search.evaluate_marginal_value({"nbv_value": 10.0, "path_length_m": 0.2}, 1)
        search.evaluate_marginal_value({"nbv_value": 2.0, "path_length_m": 0.2}, 2)
        result = search.evaluate_marginal_value({"nbv_value": 1.0, "path_length_m": 0.2}, 3)
        self.assertEqual(result["completion_reason"], "DIMINISHING_RETURN")

    def test_t21_meaningful_gain_resets_low_gain_streak(self):
        search = make_search()
        search.evaluate_marginal_value({"nbv_value": 10.0, "path_length_m": 0.2}, 1)
        search.evaluate_marginal_value({"nbv_value": 2.0, "path_length_m": 0.2}, 2)
        result = search.evaluate_marginal_value({"nbv_value": 3.0, "path_length_m": 0.2}, 3)
        self.assertEqual(result["low_gain_streak"], 0)
        self.assertGreater(result["relative_gain"], LOW_GAIN_RATIO)

    def test_t22_non_positive_gain_returns_immediately(self):
        result = make_search().evaluate_marginal_value({"nbv_value": 0.0, "path_length_m": 0.2}, 1)
        self.assertEqual(result["completion_reason"], "NON_POSITIVE_GAIN")

    def test_t23_guard_remains_24_decisions(self):
        self.assertEqual(ANTI_INFINITE_DECISION_GUARD, 24)

    def test_t24_remaining_unseen_cells_do_not_override_diminishing_return(self):
        search = make_search()
        search.evaluate_marginal_value({"nbv_value": 10.0, "path_length_m": 0.2}, 1)
        search.evaluate_marginal_value({"nbv_value": 2.0, "path_length_m": 0.2}, 2)
        result = search.evaluate_marginal_value({"nbv_value": 2.0, "path_length_m": 0.2}, 3)
        self.assertEqual(result["completion_reason"], "DIMINISHING_RETURN")
        self.assertFalse(search.observation.seen)

    def test_t25_target_failure_does_not_increment_low_gain_streak(self):
        search = make_search()
        search.evaluate_marginal_value({"nbv_value": 10.0, "path_length_m": 0.2}, 1)
        before = search.low_gain_streak
        search.reject_context_target((11.0, 20.0))  # NO_PATH/MAX_STEPS refresh path only.
        self.assertEqual(search.low_gain_streak, before)

    def test_t26_new_session_resets_peak_and_streak(self):
        search = make_search()
        search.evaluate_marginal_value({"nbv_value": 10.0, "path_length_m": 0.2}, 1)
        search.evaluate_marginal_value({"nbv_value": 2.0, "path_length_m": 0.2}, 2)
        new_session = make_search()
        self.assertIsNone(new_session.peak_best_nbv_value)
        self.assertEqual(new_session.low_gain_streak, 0)

    def test_t26a_zero_path_keeps_nbv_telemetry_but_cannot_create_baseline(self):
        search = make_search()
        result = search.evaluate_marginal_value({"nbv_value": 4000000.0, "path_length_m": 0.0}, 1)
        self.assertFalse(result["termination_baseline_eligible"])
        self.assertEqual(result["termination_baseline_ineligibility_reason"], "ZERO_PATH")
        self.assertEqual(result["best_candidate_nbv_value"], 4000000.0)
        self.assertIsNone(search.peak_best_nbv_value)
        self.assertEqual(search.low_gain_streak, 0)

    def test_t26b_missing_or_nonfinite_path_cannot_create_baseline(self):
        for path in (None, float("nan"), float("inf")):
            search = make_search()
            result = search.evaluate_marginal_value({"nbv_value": 4000000.0, "path_length_m": path}, 1)
            self.assertFalse(result["termination_baseline_eligible"])
            self.assertIsNone(search.peak_best_nbv_value)
            self.assertEqual(search.low_gain_streak, 0)

    def test_t26c_actual_progress_resets_one_marginal_epoch_once(self):
        search = make_search()
        search.evaluate_marginal_value({"nbv_value": 10.0, "path_length_m": 0.2}, 1)
        search.evaluate_marginal_value({"nbv_value": 2.0, "path_length_m": 0.2}, 2)
        reset = search.reset_marginal_completion_epoch(
            decision_index=2, run_id="run-test", actual_new_observation_cells=3,
        )
        self.assertTrue(reset["reset_applied"])
        self.assertEqual(reset["reason"], "ACTUAL_OBSERVATION_PROGRESS")
        self.assertIsNone(search.peak_best_nbv_value)
        self.assertEqual(search.low_gain_streak, 0)
        duplicate = search.reset_marginal_completion_epoch(
            decision_index=2, run_id="run-test", actual_new_observation_cells=3,
        )
        self.assertFalse(duplicate["reset_applied"])
        self.assertEqual(duplicate["reason"], "DUPLICATE_EXECUTION")

    def test_t26d_zero_actual_progress_does_not_reset_marginal_epoch(self):
        search = make_search()
        search.evaluate_marginal_value({"nbv_value": 10.0, "path_length_m": 0.2}, 1)
        reset = search.reset_marginal_completion_epoch(
            decision_index=1, run_id="run-test", actual_new_observation_cells=0,
        )
        self.assertFalse(reset["reset_applied"])
        self.assertEqual(reset["reason"], "NO_SUBSTANTIVE_PROGRESS")
        self.assertEqual(search.peak_best_nbv_value, 10.0)

    def test_t27_nbv_ranking_is_unchanged(self):
        search = make_search()
        best = search.select_best([
            {"room_target_xy": [2.0, 0.0], "visible_room_points": [(0, 0)], "path_length_m": 0.5, "heading_change_rad": 0.5},
            {"room_target_xy": [2.0, 1.0], "visible_room_points": [(1, 0), (1, 1), (1, 2)], "path_length_m": 1.0, "heading_change_rad": 0.1},
        ])
        self.assertEqual(best["room_target_xy"], [2.0, 1.0])
        self.assertEqual(LOW_GAIN_PATIENCE, 2)

    def test_t28_at_door_anchor_succeeds_without_a_navigation_target(self):
        search = make_search((10.31, 20.02, 0.7))
        self.assertTrue(search.at_door_return_anchor((10.40, 20.02, 0.0), 0.1))

    def test_t29_direct_door_wins_even_when_breadcrumbs_exist(self):
        search = make_search()
        selected = search.choose_return_target(
            {"legal": True, "kind": "DOOR_RETURN_ANCHOR", "target_xy_team_livox_odom": [10.3, 20.0]},
            [{"legal": True, "kind": "BREADCRUMB", "breadcrumb_index": 1, "target_distance_to_door": 0.2, "path_length_m": 0.2}],
        )
        self.assertEqual(selected["selected_target_type"], "DOOR_DIRECT")

    def test_t30_strict_reverse_breadcrumb_replay_is_not_required(self):
        search = make_search()
        selected = search.choose_return_target(
            {"legal": False},
            [{"legal": True, "kind": "BREADCRUMB", "breadcrumb_index": 0, "target_distance_to_door": 0.4, "path_length_m": 1.0},
             {"legal": True, "kind": "BREADCRUMB", "breadcrumb_index": 2, "target_distance_to_door": 1.0, "path_length_m": 0.2}],
        )
        self.assertEqual(selected["breadcrumb_index"], 0)

    def test_t31_executable_breadcrumb_closest_to_door_is_selected(self):
        search = make_search()
        selected = search.choose_return_target(
            {"legal": False},
            [{"legal": True, "kind": "BREADCRUMB", "breadcrumb_index": 0, "target_distance_to_door": 2.0, "path_length_m": 0.2},
             {"legal": True, "kind": "BREADCRUMB", "breadcrumb_index": 1, "target_distance_to_door": 0.5, "path_length_m": 1.0}],
        )
        self.assertEqual(selected["breadcrumb_index"], 1)

    def test_t32_intermediate_breadcrumbs_can_be_skipped(self):
        search = make_search()
        selected = search.choose_return_target(
            {"legal": False},
            [{"legal": False, "kind": "BREADCRUMB", "breadcrumb_index": 3, "target_distance_to_door": 1.2},
             {"legal": True, "kind": "BREADCRUMB", "breadcrumb_index": 1, "target_distance_to_door": 0.4, "path_length_m": 1.0}],
        )
        self.assertEqual(selected["breadcrumb_index"], 1)

    def test_t33_after_fallback_door_is_retried_before_another_breadcrumb(self):
        search = make_search()
        selected = search.choose_return_target(
            {"legal": True, "kind": "DOOR_RETURN_ANCHOR", "target_xy_team_livox_odom": [10.3, 20.0]},
            [{"legal": True, "kind": "BREADCRUMB", "breadcrumb_index": 0, "target_distance_to_door": 0.4, "path_length_m": 0.2}],
        )
        self.assertEqual(selected["selected_target_type"], "DOOR_DIRECT")

    def test_t34_close_but_not_executable_door_is_not_selected_as_direct(self):
        selected = make_search().choose_return_target({"legal": False, "target_distance_to_door": 0.01}, [])
        self.assertIsNone(selected)

    def test_t35_no_safe_door_or_breadcrumb_requests_transition_or_fail_closed(self):
        self.assertIsNone(make_search().choose_return_target({"legal": False}, [{"legal": False, "kind": "BREADCRUMB"}]))

    def test_t36_behind_or_out_of_support_uses_existing_short_safe_transition(self):
        transition = RoomSearchV2.choose_safe_return_transition(
            (2.0, 0.0), (0.0, 0.0),
            [{"legal": True, "target_xy_team_livox_odom": [1.0, 0.0]}, {"legal": True, "target_xy_team_livox_odom": [1.8, 0.0]}],
        )
        self.assertEqual(transition["target_xy_team_livox_odom"], [1.0, 0.0])

    def test_t37_no_safe_transition_fails_closed(self):
        self.assertIsNone(RoomSearchV2.choose_safe_return_transition(
            (2.0, 0.0), (0.0, 0.0), [{"legal": False, "target_xy_team_livox_odom": [1.0, 0.0]}]
        ))

    def test_t38_final_success_remains_actual_odom_within_goal_tolerance(self):
        search = make_search((10.31, 20.02, 0.7))
        self.assertTrue(search.at_door_return_anchor((10.40, 20.02, 0.0), 0.1))
        self.assertFalse(search.at_door_return_anchor((10.42, 20.02, 0.0), 0.1))

    def test_t39_low_gain_settings_and_completion_are_unchanged(self):
        search = make_search()
        self.assertEqual((LOW_GAIN_RATIO, LOW_GAIN_PATIENCE), (0.20, 2))
        self.assertEqual(
            search.evaluate_marginal_value({"nbv_value": 0.0, "path_length_m": 0.2}, 1)["completion_reason"],
            "NON_POSITIVE_GAIN",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
