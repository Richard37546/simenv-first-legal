#!/usr/bin/env python3
"""Focused offline contract tests for the P1 ROOM_RETURN continuation branch."""

import json
import math
import unittest
from pathlib import Path

from test_odom_cache import FakeRos, load_module


ROOT = Path(__file__).resolve().parents[3]
RUN0126_SUMMARY = ROOT / "debug" / "state_machine_navigation" / "run_archives" / "run_0126_20260815_234044_767004534_pid1214239" / "state_machine_navigation_summary.json"


class RoomReturnContinuationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())

    @staticmethod
    def qualified_context(*, matched=True, qualified=True):
        return {"matched_pair_found": matched, "qualified": qualified}

    @staticmethod
    def preflight(*, legal=True, safe_count=22):
        return {"legal": legal, "runner": {"last_dwa": {"safe_moving_candidate_count": safe_count}}}

    def decide(self, decision="BLOCK_ASTAR_DWA_MAX_STEPS", *, start=1.185, terminal=0.660,
               context=None, preflight=None):
        return self.module.room_return_max_steps_continuation_contract(
            decision, start, terminal, 0.30,
            self.qualified_context() if context is None else context,
            self.preflight() if preflight is None else preflight,
        )

    def test_a_run0126_productive_max_steps_reenters_from_actual_pose(self):
        archive = json.loads(RUN0126_SUMMARY.read_text(encoding="utf-8"))
        room = archive["room_search_v2"]
        attempt = room["return_attempts"][0]
        runner = attempt["runner"]
        start = attempt["current_actual_pose"]
        target = attempt["selected_target"]
        terminal = runner["last_pose_x_y_yaw"]
        start_error = math.hypot(target[0] - start[0], target[1] - start[1])
        terminal_error = math.hypot(target[0] - terminal[0], target[1] - terminal[1])
        self.assertEqual(room["final_decision"], "ROOM_RETURN_FAILED")
        self.assertEqual(runner["runner_final_decision"], "BLOCK_ASTAR_DWA_MAX_STEPS")
        self.assertEqual(runner["step_count"], 10)
        self.assertGreater(runner["observed_straight_line_displacement_m"], 0.0)
        self.assertGreater(start_error, terminal_error)
        self.assertEqual(runner["last_dwa"]["safe_moving_candidate_count"], 22)
        result = self.decide(start=start_error, terminal=terminal_error)
        self.assertTrue(result["continuable"])
        self.assertEqual(result["classification"], "CONTINUE_ROOM_RETURN_FROM_ACTUAL_POSE")

    def test_b_no_safe_next_return_motion_fails_closed(self):
        result = self.decide(preflight=self.preflight(legal=True, safe_count=0))
        self.assertFalse(result["continuable"])
        self.assertEqual(result["reason"], "NO_SAFE_MOVING_NEXT_RETURN_ACTION")

    def test_c_stale_or_unqualified_navigation_input_fails_closed(self):
        for context in (self.qualified_context(matched=False, qualified=False), self.qualified_context(matched=True, qualified=False)):
            result = self.decide(context=context)
            self.assertFalse(result["continuable"])
            self.assertEqual(result["reason"], "FRESH_QUALIFIED_NAVIGATION_INPUT_UNAVAILABLE")

    def test_d_no_path_hard_failure_never_becomes_continuation(self):
        result = self.decide(decision="BLOCK_ASTAR_DWA_BLOCKED_NO_PATH")
        self.assertFalse(result["continuable"])
        self.assertEqual(result["reason"], "RUNNER_RESULT_NOT_MAX_STEPS")

    def test_e_no_command_hard_failure_never_becomes_continuation(self):
        result = self.decide(decision="BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD")
        self.assertFalse(result["continuable"])
        self.assertEqual(result["reason"], "RUNNER_RESULT_NOT_MAX_STEPS")

    def test_f_reached_goal_stays_outside_continuation_branch(self):
        result = self.decide(decision="BLOCK_ASTAR_DWA_REACHED_GOAL")
        self.assertFalse(result["continuable"])
        self.assertEqual(result["reason"], "RUNNER_RESULT_NOT_MAX_STEPS")
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        continuation_index = source.index('if runner_decision == "BLOCK_ASTAR_DWA_MAX_STEPS":')
        reached_goal_index = source.index('if runner_decision != "BLOCK_ASTAR_DWA_REACHED_GOAL":', continuation_index)
        self.assertGreater(reached_goal_index, continuation_index)

    def test_g_no_progress_fails_closed(self):
        result = self.decide(start=0.660, terminal=0.660)
        self.assertFalse(result["continuable"])
        self.assertEqual(result["reason"], "NO_PRODUCTIVE_RETURN_PROGRESS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
