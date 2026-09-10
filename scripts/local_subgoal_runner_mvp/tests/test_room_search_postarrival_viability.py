#!/usr/bin/env python3
"""Focused offline contract checks for ROOM_SEARCH post-arrival viability."""

import types
import unittest
from pathlib import Path

from test_odom_cache import FakeGrid, FakeRos, load_module


class _RunnerProbe:
    def __init__(self, safe_count):
        self.safe_count = safe_count
        self.calls = []

    def choose_dwa(self, grid_msg, blocked, waypoint_xy, target_base_xy, distance_to_goal, **kwargs):
        self.calls.append((grid_msg, blocked, waypoint_xy, target_base_xy, distance_to_goal, kwargs))
        return 0.0, 0.0, {"safe_moving_candidate_count": self.safe_count}


class RoomSearchPostArrivalViabilityTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())
        self.original_runner_cmd = self.module.runner_cmd
        # The probe owns no runner configuration.  The production helper still
        # parses the existing runner parser from this deliberately empty profile.
        self.module.runner_cmd = lambda *_args, **_kwargs: ["python3", "runner.py"]
        self.args = types.SimpleNamespace(runner_runtime_sec=35.0)
        self.terminal_odom = {"stamp_sec": 47.82, "pose_x_y_yaw": [17.169, 3.294, 0.998]}

    def tearDown(self):
        self.module.runner_cmd = self.original_runner_cmd

    @staticmethod
    def context(runner, *, qualified=True, matched=True):
        return {
            "matched_pair_found": matched,
            "qualified": qualified,
            "failure_reason": "ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE" if not matched else "ROOM_SEARCH_GRID_STATUS_CONTEXT_UNQUALIFIED",
            "grid_header_stamp_sec": 47.80,
            "grid_content_stamp": 47.80,
            "content_generation_id": 866,
            "grid_content_hash": "exact-pair-hash",
            "grid_msg": FakeGrid(),
            "blocked": object(),
            "runner": runner,
        }

    @staticmethod
    def runner_summary():
        return {"last_dwa": {"selected_linear_x": 0.30, "selected_angular_z": -0.12}}

    def test_a_qualified_positive_safe_count_is_viable_and_keeps_normal_continuation(self):
        probe = _RunnerProbe(3)
        context = self.context(probe)
        result = self.module.room_search_v2_postarrival_viability(
            self.args, context, self.terminal_odom, self.runner_summary(),
        )
        treatment = self.module.room_search_v2_postarrival_control(result["viability_result"])
        self.assertEqual(result["viability_result"], "POST_ARRIVAL_VIABLE")
        self.assertEqual(result["safe_moving_candidate_count"], 3)
        self.assertTrue(treatment["normal_continuation"])
        self.assertTrue(treatment["safe_history_accept_terminal_pose"])
        self.assertEqual(len(probe.calls), 1)
        self.assertIs(probe.calls[0][0], context["grid_msg"])

    def test_b_zero_safe_count_is_constrained_without_completion_return_or_reposition(self):
        result = self.module.room_search_v2_postarrival_viability(
            self.args, self.context(_RunnerProbe(0)), self.terminal_odom, self.runner_summary(),
        )
        treatment = self.module.room_search_v2_postarrival_control(result["viability_result"])
        self.assertEqual(result["viability_result"], "CONSTRAINED_ARRIVAL")
        self.assertFalse(treatment["normal_continuation"])
        self.assertFalse(treatment["search_completed"])
        self.assertFalse(treatment["enter_room_return"])
        self.assertFalse(treatment["call_reposition"])
        self.assertTrue(treatment["safe_stop_required"])

    def test_c_missing_or_unqualified_pair_fails_closed(self):
        for context in (
            self.context(_RunnerProbe(9), matched=False, qualified=False),
            self.context(_RunnerProbe(9), matched=True, qualified=False),
        ):
            result = self.module.room_search_v2_postarrival_viability(
                self.args, context, self.terminal_odom, self.runner_summary(),
            )
            self.assertEqual(result["viability_result"], "POST_ARRIVAL_VIABILITY_INPUT_UNAVAILABLE")
            self.assertIsNone(result["safe_moving_candidate_count"])
            self.assertFalse(self.module.room_search_v2_postarrival_control(result["viability_result"])["normal_continuation"])

    def test_d_observation_retention_is_independent_from_constrained_continuation(self):
        treatment = self.module.room_search_v2_postarrival_control("CONSTRAINED_ARRIVAL")
        self.assertTrue(treatment["preserve_observation"])
        self.assertFalse(treatment["normal_continuation"])

    def test_e_constrained_arrival_is_not_a_safe_recovery_anchor(self):
        treatment = self.module.room_search_v2_postarrival_control("CONSTRAINED_ARRIVAL")
        self.assertFalse(treatment["safe_history_accept_terminal_pose"])
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        constrained_branch = source[source.index('if not treatment["normal_continuation"]:'):source.index('record["accepted"] = True')]
        self.assertNotIn("record_safe_actual_pose(terminal_pose", constrained_branch)

    def test_run0125_counterfactual_zero_count_is_not_false_completion(self):
        # Frozen audit input: RUN0125 actual terminal safe-moving count was 0.
        result = self.module.room_search_v2_postarrival_viability(
            self.args, self.context(_RunnerProbe(0)), self.terminal_odom, self.runner_summary(),
        )
        treatment = self.module.room_search_v2_postarrival_control(result["viability_result"])
        self.assertEqual(result["safe_moving_candidate_count"], 0)
        self.assertEqual(result["viability_result"], "CONSTRAINED_ARRIVAL")
        self.assertFalse(treatment["search_completed"])
        self.assertFalse(treatment["enter_room_return"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
