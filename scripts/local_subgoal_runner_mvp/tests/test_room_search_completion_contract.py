#!/usr/bin/env python3
"""Focused offline regressions for ROOM_SEARCH mission-completion semantics."""

from pathlib import Path
import inspect
import unittest

from test_odom_cache import FakeRos, load_module


class RoomSearchCompletionContractTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())

    def test_a_no_candidate_without_independent_evidence_is_incomplete(self):
        state = self.module.room_search_v2_completion_contract("NO_SAFE_USEFUL_CANDIDATE", None)
        self.assertFalse(state["mission_complete"])
        self.assertIsNone(state["mission_completion_reason"])
        self.assertEqual(state["next_control_flow"], "ROOM_SEARCH_INCOMPLETE")

    def test_b_existing_finite_low_gain_completion_still_allows_room_return(self):
        state = self.module.room_search_v2_completion_contract("ADMITTED_CANDIDATE", "DIMINISHING_RETURN")
        self.assertTrue(state["mission_complete"])
        self.assertEqual(state["mission_completion_reason"], "DIMINISHING_RETURN")
        self.assertEqual(state["next_control_flow"], "ROOM_RETURN")

    def test_c_candidate_available_without_completion_evidence_keeps_ordinary_search_active(self):
        state = self.module.room_search_v2_completion_contract("ADMITTED_CANDIDATE", None)
        self.assertFalse(state["mission_complete"])
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        self.assertIn('target_xy = candidate["target_xy_team_livox_odom"]', source)
        self.assertIn('attempt = execute_target(target_xy, "ROOM_SEARCH_TARGET"', source)

    def test_d_repair1_constrained_arrival_remains_noncompletion(self):
        treatment = self.module.room_search_v2_postarrival_control("CONSTRAINED_ARRIVAL")
        self.assertFalse(treatment["normal_continuation"])
        self.assertFalse(treatment["search_completed"])
        self.assertFalse(treatment["enter_room_return"])

    def test_e_run0125_decision2_counterfactual_is_local_unavailability_not_completion(self):
        # Frozen Decision-2 facts: candidate=None, low-gain streak=0, and
        # decision budget was not exhausted, hence no accepted evidence reason.
        state = self.module.room_search_v2_completion_contract("NO_SAFE_USEFUL_CANDIDATE", None)
        self.assertFalse(state["mission_complete"])
        self.assertEqual(state["next_control_flow"], "ROOM_SEARCH_INCOMPLETE")

    def test_finite_nonproductive_guard_is_incomplete_not_task_completion(self):
        state = self.module.room_search_v2_completion_contract("DECISION_BUDGET_EXHAUSTED", "ANTI_INFINITE_GUARD")
        self.assertFalse(state["mission_complete"])
        self.assertTrue(state["search_aborted_incomplete"])
        self.assertEqual(state["next_control_flow"], "ROOM_SEARCH_INCOMPLETE")

    def test_run0131_decision24_actual_new_two_resets_nonproductive_budget(self):
        # Frozen RUN0131 decision-24 evidence: actual_new_observation_cells=2.
        before = self.module.room_search_v2_progress_guard_update(23, 0)
        productive = self.module.room_search_v2_progress_guard_update(before["nonproductive_progress_cycles"], 2)
        self.assertTrue(productive["substantive_progress"])
        self.assertEqual(productive["nonproductive_progress_cycles"], 0)
        self.assertFalse(productive["finite_abort"])

    def test_substantive_progress_seam_preserves_existing_guard_before_epoch_reset(self):
        source = inspect.getsource(self.module.execute_room_search_v2)
        guard = source.index('guard_fired = record_nonproductive_cycle("EXECUTED_CANDIDATE", actual_new_observation_cells)')
        reset = source.index("marginal_epoch_reset = search.reset_marginal_completion_epoch", guard)
        self.assertLess(guard, reset)
        self.assertIn('if bool(progress_guard.get("substantive_progress")):', source[guard:reset + 300])

    def test_twenty_four_productive_iterations_do_not_consume_the_guard(self):
        state = {"nonproductive_progress_cycles": 0}
        for _ in range(24):
            state = self.module.room_search_v2_progress_guard_update(state["nonproductive_progress_cycles"], 1)
        self.assertEqual(state["nonproductive_progress_cycles"], 0)
        self.assertFalse(state["finite_abort"])

    def test_repeated_nonproductive_cycles_remain_bounded(self):
        state = {"nonproductive_progress_cycles": 0}
        for _ in range(self.module.ROOM_SEARCH_NONPRODUCTIVE_PROGRESS_GUARD_LIMIT):
            state = self.module.room_search_v2_progress_guard_update(state["nonproductive_progress_cycles"], 0)
        self.assertTrue(state["finite_abort"])
        self.assertEqual(state["nonproductive_progress_cycles"], 24)

    def test_normal_completion_reasons_are_unchanged(self):
        for reason in ("NON_POSITIVE_GAIN", "DIMINISHING_RETURN"):
            state = self.module.room_search_v2_completion_contract("ADMITTED_CANDIDATE", reason)
            self.assertTrue(state["mission_complete"])
            self.assertFalse(state["search_aborted_incomplete"])

    def test_total_decision_loop_bound_is_removed_from_room_search(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("while decisions < 24 and not rospy.is_shutdown():", source)
        self.assertIn("while not rospy.is_shutdown():", source)

    def test_no_safe_candidate_branch_preserves_incomplete_result_then_requests_return(self):
        source = inspect.getsource(self.module.execute_room_search_v2)
        branch_start = source.index('if candidate is None:')
        branch_end = source.index('decisions += 1', branch_start)
        branch = source[branch_start:branch_end]
        self.assertIn('finish_room_search_incomplete', branch)
        self.assertIn('return request_room_return_exit(completion_contract)', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
