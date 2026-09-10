#!/usr/bin/env python3
"""Focused regressions for decision-global L3V preflight invalidation."""

import unittest
from pathlib import Path

from test_odom_cache import FakeRos, load_module


L3V = "BLOCK_ASTAR_DWA_BLOCKED_BY_L3V_STATUS"


def candidate(identifier):
    return {"_room_search_audit_candidate_id": identifier, "target_xy_team_livox_odom": [1.0, 0.0]}


def result(terminal, legal=False):
    return {
        "legal": legal,
        "runner": {
            "runner_final_decision": terminal,
            "status_payload": {
                "content_generation_id": 42,
                "grid_content_stamp": 12.3,
                "grid_content_hash": "exact-hash",
                "local_traversability_status": "CONFLICT_NEEDS_CAUTION",
            },
        },
    }


def frozen_global_l3v_result():
    return {
        "legal": False,
        "formal_status": "EVALUATION_NOT_AVAILABLE",
        "runner_final_decision": "DECISION_GLOBAL_L3V_STATUS",
    }


class RoomSearchL3vDecisionConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())

    def test_a_l3v_transient_invalidates_before_lower_ranks(self):
        seen = []

        def preflight(row):
            seen.append(row["_room_search_audit_candidate_id"])
            return result(L3V)

        outcome = self.module.room_search_v2_admit_with_l3v_consistency(
            [candidate("rank1"), candidate("rank2"), candidate("rank3")], preflight, 5,
        )
        self.assertEqual(seen, ["rank1"])
        self.assertIsNone(outcome["candidate"])
        self.assertEqual(outcome["preflight_attempt_count"], 1)
        self.assertEqual(outcome["decision_global_l3v_invalidation"]["global_rank"], 1)
        self.assertEqual(outcome["decision_global_l3v_invalidation"]["remaining_lower_ranked_candidates_not_preflighted"], 2)

    def test_b_next_free_decision_evaluates_high_rank_normally_without_blacklist(self):
        seen = []

        def preflight(row):
            seen.append(row["_room_search_audit_candidate_id"])
            return result("BLOCK_ASTAR_DWA_MAX_STEPS", legal=True)

        outcome = self.module.room_search_v2_admit_with_l3v_consistency(
            [candidate("rank1"), candidate("rank2")], preflight, 5,
        )
        self.assertEqual(seen, ["rank1"])
        self.assertEqual(outcome["candidate"]["_room_search_audit_candidate_id"], "rank1")
        self.assertIsNone(outcome["decision_global_l3v_invalidation"])

    def test_c_target_specific_no_path_still_scans_next_candidate(self):
        seen = []

        def preflight(row):
            seen.append(row["_room_search_audit_candidate_id"])
            return result("BLOCK_ASTAR_DWA_BLOCKED_NO_PATH", legal=row["_room_search_audit_candidate_id"] == "rank2")

        outcome = self.module.room_search_v2_admit_with_l3v_consistency(
            [candidate("rank1"), candidate("rank2")], preflight, 5,
        )
        self.assertEqual(seen, ["rank1", "rank2"])
        self.assertEqual(outcome["candidate"]["_room_search_audit_candidate_id"], "rank2")

    def test_d_dwa_no_cmd_still_scans_next_candidate(self):
        seen = []

        def preflight(row):
            seen.append(row["_room_search_audit_candidate_id"])
            terminal = "BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD" if row["_room_search_audit_candidate_id"] == "rank1" else "BLOCK_ASTAR_DWA_MAX_STEPS"
            return result(terminal, legal=row["_room_search_audit_candidate_id"] == "rank2")

        outcome = self.module.room_search_v2_admit_with_l3v_consistency(
            [candidate("rank1"), candidate("rank2")], preflight, 5,
        )
        self.assertEqual(seen, ["rank1", "rank2"])
        self.assertEqual(outcome["candidate"]["_room_search_audit_candidate_id"], "rank2")

    def test_e_l3v_never_admits_a_candidate(self):
        outcome = self.module.room_search_v2_admit_with_l3v_consistency(
            [candidate("rank1")], lambda _row: result(L3V), 5,
        )
        self.assertIsNone(outcome["candidate"])
        self.assertEqual(outcome["decision_global_l3v_invalidation"]["runner_final_decision"], L3V)

    def test_f_transient_l3v_cycle_consumes_only_nonproductive_progress_guard(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        start = source.index("if decision_global_l3v_invalidation is not None:")
        branch = source[start:source.index("if candidate is None:", start)]
        self.assertIn("decisions += 1", branch)
        self.assertIn("record_nonproductive_cycle", branch)
        self.assertIn("continue", branch)
        self.assertIn("ROOM_SEARCH_NONPRODUCTIVE_PROGRESS_GUARD_LIMIT = 24", source)
        self.assertNotIn("while decisions < 24 and not rospy.is_shutdown():", source)

    def test_g_l3v_transient_is_not_a_completion_reason(self):
        state = self.module.room_search_v2_completion_contract(
            self.module.ROOM_SEARCH_DECISION_TRANSIENT_L3V_UNAVAILABLE, None,
        )
        self.assertFalse(state["mission_complete"])
        self.assertEqual(state["next_control_flow"], "ROOM_SEARCH_INCOMPLETE")

    def test_h_frozen_normalized_global_l3v_is_decision_global(self):
        self.assertTrue(self.module.room_search_v2_is_decision_global_l3v_transient(frozen_global_l3v_result()))

    def test_i_candidate_specific_terminals_are_not_global_l3v(self):
        for preflight in (
            {"legal": False, "formal_status": "FORMALLY_ILLEGAL", "runner_final_decision": "BLOCK_ASTAR_DWA_BLOCKED_NO_PATH"},
            {"legal": False, "formal_status": "EVALUATION_NOT_AVAILABLE", "runner_final_decision": "INPUT_UNAVAILABLE"},
            {"legal": False, "formal_status": "FORMALLY_ILLEGAL", "runner_final_decision": "BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD"},
        ):
            self.assertFalse(self.module.room_search_v2_is_decision_global_l3v_transient(preflight), preflight)


if __name__ == "__main__":
    unittest.main(verbosity=2)
