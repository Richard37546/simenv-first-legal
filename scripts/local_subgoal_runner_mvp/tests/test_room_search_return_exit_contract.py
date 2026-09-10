#!/usr/bin/env python3
"""Focused offline checks for terminal ROOM_SEARCH exit-action routing."""

from pathlib import Path
import unittest

from test_odom_cache import FakeRos, load_module


class RoomSearchReturnExitContractTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())

    def contract(self, availability, reason):
        return self.module.room_search_v2_completion_contract(availability, reason)

    def exit_action(self, contract, *, alive=True, anchor=True):
        return self.module.room_search_v2_exit_action_contract(
            contract, control_alive=alive, portal_anchor_available=anchor,
        )

    def test_a_normal_completion_requests_existing_room_return(self):
        action = self.exit_action(self.contract("ADMITTED_CANDIDATE", "DIMINISHING_RETURN"))
        self.assertTrue(action["search_complete"])
        self.assertTrue(action["mission_complete"])
        self.assertTrue(action["room_return_requested"])
        self.assertEqual(action["next_control_flow"], "ROOM_RETURN")

    def test_b_finite_abort_is_incomplete_and_requests_return(self):
        action = self.exit_action(self.contract("NONPRODUCTIVE_PROGRESS_GUARD_EXHAUSTED", "ANTI_INFINITE_GUARD"))
        self.assertFalse(action["search_complete"])
        self.assertFalse(action["mission_complete"])
        self.assertTrue(action["search_aborted_incomplete"])
        self.assertTrue(action["room_return_requested"])

    def test_c_return_request_cannot_change_incomplete_search_result(self):
        contract = self.contract("NONPRODUCTIVE_PROGRESS_GUARD_EXHAUSTED", "ANTI_INFINITE_GUARD")
        self.exit_action(contract)
        self.assertFalse(contract["mission_complete"])
        self.assertTrue(contract["search_aborted_incomplete"])

    def test_d_terminal_incomplete_paths_request_return_while_control_is_alive(self):
        for availability in ("NO_SAFE_USEFUL_CANDIDATE", "ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE", "ROOM_SEARCH_GRID_STATUS_CONTEXT_UNQUALIFIED", "CONSTRAINED_ARRIVAL"):
            action = self.exit_action(self.contract(availability, None))
            self.assertEqual(action["search_outcome"], "SEARCH_INCOMPLETE_LOCAL_FAILURE")
            self.assertFalse(action["mission_complete"])
            self.assertTrue(action["room_return_requested"])

    def test_e_lifecycle_shutdown_is_exempt(self):
        action = self.exit_action(self.contract("ROS_SHUTDOWN", None), alive=False)
        self.assertFalse(action["room_return_requested"])
        self.assertEqual(action["next_control_flow"], "TERMINATE_WITHOUT_RETURN")

    def test_f_missing_anchor_does_not_invent_return_target(self):
        action = self.exit_action(self.contract("NO_SAFE_USEFUL_CANDIDATE", None), anchor=False)
        self.assertFalse(action["room_return_requested"])
        self.assertEqual(action["next_control_flow"], "TERMINATE_WITHOUT_RETURN")

    def test_g_existing_return_algorithm_is_reused_by_one_exit_helper(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        start = source.index("def execute_room_return_exit()")
        end = source.index("def request_room_return_exit", start)
        return_body = source[start:end]
        self.assertIn("search.door_return_target()", return_body)
        self.assertIn("search.choose_return_target", return_body)
        self.assertIn("search.choose_safe_return_transition", return_body)
        self.assertIn("room_return_max_steps_continuation_contract", return_body)
        self.assertIn("BLOCK_ASTAR_DWA_BLOCKED_NO_PATH", Path(__file__).with_name("test_room_return_continuation.py").read_text(encoding="utf-8"))

    def test_h_terminal_incomplete_and_normal_completion_share_exit_request(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        self.assertIn("return request_room_return_exit(completion_contract)", source)
        self.assertIn("return request_room_return_exit(mission_completion)", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
