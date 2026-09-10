#!/usr/bin/env python3
"""Offline C0 tests.  They do not require ROS, Gazebo, a runner, or a planner."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


THIS = Path(__file__).resolve()
MODULE_ROOT = THIS.parents[1]
REPO_ROOT = THIS.parents[3]
sys.path.insert(0, str(MODULE_ROOT))

import replay_room_search_observation_arrival_c0 as replay  # noqa: E402
from room_search_observation_arrival_contract import (  # noqa: E402
    NAVIGATION_NOT_REACHED,
    OBSERVATION_VALID_REACHED_CANDIDATE,
    OBSERVATION_VALIDITY_UNKNOWN,
    OBSERVATION_VALUE_COLLAPSED,
    OBSERVATION_VALUE_PARTIAL,
    evaluate_observation_arrival_shadow,
)


def evaluate(predicted, actual, reached=True, complete=True, candidate_id="candidate"):
    return evaluate_observation_arrival_shadow(
        navigation_reached=reached,
        candidate_id=candidate_id,
        predicted_new_cell_ids=predicted,
        predicted_new_count=len(predicted),
        actual_new_cell_ids=actual,
        actual_new_count=len(actual),
        terminal_evidence_complete=complete,
        terminal_evidence_qualification="TEST",
        post_arrival_viability="POST_ARRIVAL_VIABLE",
    )


class ObservationArrivalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage_root = REPO_ROOT / "debug/room_search/stage_b2_online/stage_b2_whole_room_20260824_163358"
        cls.raw_replay = replay.replay_run0145(REPO_ROOT, cls.stage_root)
        cls.by_decision = {row["decision"]: row for row in cls.raw_replay["decisions"]}

    def test_t1_pure_helper_truth_table(self):
        self.assertEqual(evaluate([(1, 1)], [], reached=False).observation_state, NAVIGATION_NOT_REACHED)
        self.assertEqual(evaluate([(1, 1)], [], complete=False).observation_state, OBSERVATION_VALIDITY_UNKNOWN)
        self.assertEqual(evaluate([(1, 1)], []).observation_state, OBSERVATION_VALUE_COLLAPSED)
        self.assertEqual(evaluate([(1, 1)], [(2, 2)]).observation_state, OBSERVATION_VALID_REACHED_CANDIDATE)
        self.assertEqual(evaluate([(1, 1), (2, 2)], [(1, 1)]).observation_state, OBSERVATION_VALUE_PARTIAL)

    def test_t2_d9_collapsed(self):
        row = self.by_decision["D9"]
        self.assertTrue(row["navigation_reached"])
        self.assertEqual(row["observation_state"], OBSERVATION_VALUE_COLLAPSED)
        self.assertEqual((row["predicted_new_count"], row["actual_new_count"]), (2, 0))

    def test_t3_d10_collapsed(self):
        row = self.by_decision["D10"]
        self.assertTrue(row["navigation_reached"])
        self.assertEqual(row["observation_state"], OBSERVATION_VALUE_COLLAPSED)
        self.assertEqual((row["predicted_new_count"], row["actual_new_count"]), (2, 0))

    def test_t4_d8_different_set_but_valid(self):
        row = self.by_decision["D8"]
        self.assertNotEqual(row["predicted_new_cell_ids"], row["actual_new_cell_ids"])
        self.assertEqual((row["predicted_new_count"], row["actual_new_count"]), (4, 4))
        self.assertEqual(row["observation_state"], OBSERVATION_VALID_REACHED_CANDIDATE)
        self.assertTrue(row["unexpected_useful_actual_cell_ids"])

    def test_t5_d4_partial(self):
        row = self.by_decision["D4"]
        self.assertEqual((row["predicted_new_count"], row["actual_new_count"]), (6, 2))
        self.assertEqual(row["observation_state"], OBSERVATION_VALUE_PARTIAL)

    def test_t6_d6_partial(self):
        row = self.by_decision["D6"]
        self.assertEqual((row["predicted_new_count"], row["actual_new_count"]), (6, 4))
        self.assertEqual(row["observation_state"], OBSERVATION_VALUE_PARTIAL)

    def test_t7_unknown_has_no_credit_or_authority(self):
        result = evaluate([(1, 1)], [], complete=False)
        self.assertEqual(result.observation_state, OBSERVATION_VALIDITY_UNKNOWN)
        self.assertFalse(result.authority_enabled)

    def test_t8_seen_no_double_update_or_input_mutation(self):
        predicted, actual = [(1, 1), (2, 2)], [(2, 2)]
        predicted_before, actual_before = list(predicted), list(actual)
        result = evaluate(predicted, actual)
        self.assertEqual(predicted, predicted_before)
        self.assertEqual(actual, actual_before)
        self.assertEqual(result.actual_new_cell_ids, ((2, 2),))
        self.assertEqual(result.actual_new_count, 1)

    def test_t9_navigation_truth_is_preserved(self):
        for name in ("D9", "D10"):
            row = self.by_decision[name]
            self.assertTrue(row["navigation_reached"])
            self.assertEqual(row["observation_state"], OBSERVATION_VALUE_COLLAPSED)
            self.assertFalse(row["authority_enabled"])

    def test_t10_behavior_neutral_no_production_import_or_call_site(self):
        helper_source = (MODULE_ROOT / "room_search_observation_arrival_contract.py").read_text(encoding="utf-8")
        replay_source = (MODULE_ROOT / "replay_room_search_observation_arrival_c0.py").read_text(encoding="utf-8")
        self.assertNotIn("rospy", helper_source)
        self.assertNotIn("import navigation_state_machine", helper_source)
        self.assertNotIn("from navigation_state_machine", helper_source)
        self.assertNotIn("import navigation_state_machine", replay_source)
        self.assertNotIn("from navigation_state_machine", replay_source)
        self.assertNotIn("run_runner", replay_source)
        self.assertNotIn("ObservationMemory", helper_source)

    def test_t11_historical_hash_is_preserved_as_provenance_not_current_behavior_gate(self):
        self.assertEqual(replay.PROTECTED_HASHES, {
            "scripts/local_subgoal_runner_mvp/navigation_state_machine.py": "cf2a01426560694672754787ffd448eff8f7ae6ead87cc74675cdc1bf7e11f72",
            "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py": "468e735acc8bb0bdf03453906fccd70b41394b751bc3638e0f62f40b48a01a5b",
            "scripts/local_subgoal_runner_mvp/room_search_v1.py": "485140cbe3ce9cd985a3d27ea9a50cb83782e73293fd80003799df899d0d01a9",
        })
        provenance = replay.historical_provenance(REPO_ROOT)
        self.assertFalse(provenance["current_source_match_required"])
        self.assertEqual(provenance["historical_expected_hashes"], replay.PROTECTED_HASHES)
        self.assertEqual(set(provenance["current_source_hashes"]), set(replay.PROTECTED_HASHES))
        self.assertTrue(all(
            isinstance(item["actual"], str) and len(item["actual"]) == 64
            for item in provenance["hashes"].values()
        ))

    def test_t12_no_completion_or_recovery_authority(self):
        result = evaluate([(1, 1)], [])
        payload = result.to_dict()
        self.assertFalse(payload["authority_enabled"])
        self.assertEqual(payload["post_arrival_viability"], "POST_ARRIVAL_VIABLE")
        self.assertEqual(payload["observation_state"], OBSERVATION_VALUE_COLLAPSED)

    def test_t13_d1_d10_raw_replay_classification(self):
        expected = {
            "D1": OBSERVATION_VALUE_PARTIAL, "D2": OBSERVATION_VALUE_PARTIAL,
            "D3": OBSERVATION_VALUE_PARTIAL, "D4": OBSERVATION_VALUE_PARTIAL,
            "D5": OBSERVATION_VALUE_PARTIAL, "D6": OBSERVATION_VALUE_PARTIAL,
            "D7": OBSERVATION_VALUE_PARTIAL, "D8": OBSERVATION_VALID_REACHED_CANDIDATE,
            "D9": OBSERVATION_VALUE_COLLAPSED, "D10": OBSERVATION_VALUE_COLLAPSED,
        }
        self.assertTrue(self.raw_replay["raw_evidence_delta_count_closure"])
        self.assertEqual({row["decision"]: row["observation_state"] for row in self.raw_replay["decisions"]}, expected)
        self.assertTrue(all(row["navigation_reached"] and row["raw_seen_delta_count_closed"] for row in self.raw_replay["decisions"]))

    def test_t14_low_opportunity_remains_uncalibrated(self):
        result = evaluate([(1, 1)], [(1, 1)])
        self.assertIsNone(result.low_opportunity_candidate)
        self.assertEqual(result.low_opportunity_reason, "BOUNDARY_NOT_CALIBRATED")
        self.assertEqual(result.observation_state, OBSERVATION_VALID_REACHED_CANDIDATE)


if __name__ == "__main__":
    unittest.main()
