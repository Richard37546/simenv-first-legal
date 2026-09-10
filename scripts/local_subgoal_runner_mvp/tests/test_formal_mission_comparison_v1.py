#!/usr/bin/env python3
"""Focused authority and ordering tests for the pure Formal Comparison V1."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_mission_comparison_v1 import (  # noqa: E402
    CONTINUATION_NON_VIABLE,
    CONTINUATION_UNKNOWN,
    CONTINUATION_VIABLE,
    MISSION_INTENT_DANGER_REOBSERVE,
    evaluate_formal_mission_comparison_v1,
    replay_stage_b_shadow_result,
)


def candidate(candidate_id, **overrides):
    row = {
        "candidate_id": candidate_id,
        "mission_intent": "EXPLORATION",
        "formal_preflight": {"legal": True, "path_cost_m": 2.0},
        "exploration_action": {"intent_valid": True, "intended_observation_cell_ids": [[0, 0]]},
        "continuation": {"status": CONTINUATION_UNKNOWN, "complete": False},
        "geometry": {"geometric_distance_m": 1.0, "heading_change_rad": 0.4, "door_factor": 0.7},
        "cheap_rank": 1,
        "nbv_value": 999.0,
        "predicted_new_count": 99,
        "predicted_occlusion_reveal_count": 99,
    }
    row.update(overrides)
    return row


class FormalMissionComparisonV1Tests(unittest.TestCase):
    def evaluate(self, rows, **kwargs):
        return evaluate_formal_mission_comparison_v1("epoch-test", rows, cohort_complete=True, **kwargs)

    def test_01_complete_cohort_checks_every_candidate_not_first_legal(self):
        first = candidate("first", continuation={"status": CONTINUATION_NON_VIABLE, "complete": True})
        alternative = candidate("alternative", cheap_rank=99, continuation={"status": CONTINUATION_VIABLE, "complete": True})
        result = self.evaluate([first, alternative])
        comparison = result["comparison"]
        self.assertEqual(2, len(comparison["all_candidates"]))
        self.assertEqual(["alternative"], comparison["mission_admissible_candidate_ids"])
        self.assertEqual("alternative", result["shadow_selection"]["winner_candidate_id"])
        self.assertTrue(result["production_first_legal_unchanged"])

    def test_02_danger_priority_is_formal_and_uses_h1_identity(self):
        exploration = candidate("explore", continuation={"status": CONTINUATION_VIABLE, "complete": True})
        danger = candidate(
            "h1", mission_intent=MISSION_INTENT_DANGER_REOBSERVE,
            exploration_action={}, danger_reobserve={"available": True, "hypothesis_id": "H1", "repeat_guard_closed": False},
            continuation={"status": CONTINUATION_UNKNOWN, "complete": False},
        )
        result = self.evaluate([exploration, danger])
        self.assertEqual("h1", result["shadow_selection"]["winner_candidate_id"])
        self.assertEqual("DANGER_PRIORITY", result["shadow_selection"]["win_reason"])

    def test_03_closed_danger_episode_is_not_admitted(self):
        danger = candidate(
            "h1", mission_intent=MISSION_INTENT_DANGER_REOBSERVE, exploration_action={},
            danger_reobserve={"available": True, "hypothesis_id": "H1", "repeat_guard_closed": True},
        )
        normal = candidate("normal")
        result = self.evaluate([danger, normal])
        record = result["comparison"]["all_candidates"][0]
        self.assertIn("DANGER_EPISODE_ALREADY_CLOSED", record["hard_admission_reasons"])
        self.assertEqual("normal", result["shadow_selection"]["winner_candidate_id"])

    def test_04_viable_beats_unknown_within_same_intent(self):
        unknown = candidate("unknown", continuation={"status": CONTINUATION_UNKNOWN, "complete": False})
        viable = candidate("viable", continuation={"status": CONTINUATION_VIABLE, "complete": True}, cheap_rank=50)
        result = self.evaluate([unknown, viable])
        self.assertEqual("viable", result["shadow_selection"]["winner_candidate_id"])
        self.assertEqual("CONTINUATION_VIABLE", result["shadow_selection"]["win_reason"])

    def test_05_unknown_is_not_failure(self):
        unknown = candidate("unknown", continuation={"status": CONTINUATION_UNKNOWN, "complete": False})
        result = self.evaluate([unknown])
        self.assertEqual(["unknown"], result["comparison"]["mission_admissible_candidate_ids"])
        self.assertEqual("unknown", result["shadow_selection"]["winner_candidate_id"])

    def test_06_strict_superset_beats_lower_cost_subset(self):
        subset = candidate("subset", exploration_action={"intent_valid": True, "intended_observation_cell_ids": [[0, 0]]}, formal_preflight={"legal": True, "path_cost_m": 0.1})
        superset = candidate("superset", exploration_action={"intent_valid": True, "intended_observation_cell_ids": [[0, 0], [1, 0]]}, formal_preflight={"legal": True, "path_cost_m": 9.0})
        result = self.evaluate([subset, superset])
        self.assertEqual("superset", result["shadow_selection"]["winner_candidate_id"])
        self.assertEqual("STRICT_CELL_SUPERSET", result["shadow_selection"]["win_reason"])

    def test_07_non_dominated_cells_fall_to_formal_path_cost(self):
        left = candidate("left", exploration_action={"intent_valid": True, "intended_observation_cell_ids": [[0, 0]]}, formal_preflight={"legal": True, "path_cost_m": 4.0})
        right = candidate("right", exploration_action={"intent_valid": True, "intended_observation_cell_ids": [[1, 0]]}, formal_preflight={"legal": True, "path_cost_m": 2.0})
        result = self.evaluate([left, right])
        self.assertEqual("right", result["shadow_selection"]["winner_candidate_id"])
        self.assertEqual("LOWER_FORMAL_PATH_COST", result["shadow_selection"]["win_reason"])

    def test_08_door_factor_keeps_existing_higher_is_better_direction(self):
        low = candidate("low", formal_preflight={"legal": True, "path_cost_m": 2.0}, geometry={"geometric_distance_m": 1.0, "heading_change_rad": 0.4, "door_factor": 0.7})
        high = candidate("high", formal_preflight={"legal": True, "path_cost_m": 2.0}, geometry={"geometric_distance_m": 1.0, "heading_change_rad": 0.4, "door_factor": 1.0})
        result = self.evaluate([low, high])
        self.assertEqual("high", result["shadow_selection"]["winner_candidate_id"])
        self.assertEqual("HIGHER_DOOR_FACTOR", result["shadow_selection"]["win_reason"])

    def test_09_telemetry_cannot_change_winner(self):
        left = candidate("left", cheap_rank=99, nbv_value=9999.0, predicted_new_count=999)
        right = candidate("right", cheap_rank=1, nbv_value=0.0, predicted_new_count=0)
        result = self.evaluate([left, right])
        self.assertEqual("left", result["shadow_selection"]["winner_candidate_id"])
        self.assertEqual("STABLE_CANDIDATE_ID", result["shadow_selection"]["win_reason"])

    def test_10_incomplete_cohort_never_selects(self):
        result = evaluate_formal_mission_comparison_v1("epoch-test", [candidate("a")], cohort_complete=False)
        self.assertEqual("NO_SHADOW_WINNER", result["shadow_selection"]["status"])
        self.assertEqual("FORMAL_COMPARISON_SET_INCOMPLETE", result["shadow_selection"]["reason"])

    def test_11_stage_b_mismatch_cannot_be_reinterpreted_as_a_complete_cohort(self):
        stage_b = {
            "evaluation": {
                "status": "STAGE_B_SHADOW_COMPLETE",
                "room_search_decision_epoch": {"epoch_id": "e1"},
                "legal_comparison_set": {
                    "comparison_completeness": "COMPLETE",
                    "all_preflight_results": [{"candidate_id": "chosen", "legality": "ILLEGAL_NOW"}],
                },
            },
            "production_events": [{"event": "PRODUCTION_ADMISSION", "selected_candidate_id": "chosen"}],
        }
        frozen = [{"candidate_id": "chosen", "candidate_type": "GENERIC_COVERAGE", "opportunity": {"predicted_new_cell_ids": [[1, 1]]}}]
        result = replay_stage_b_shadow_result(stage_b, frozen)
        self.assertTrue(result["replay_context_mismatch"])
        self.assertEqual("NO_SHADOW_WINNER", result["shadow_selection"]["status"])

    def test_12_matching_stage_b_cohort_is_materialized_without_granting_commands(self):
        stage_b = {
            "evaluation": {
                "status": "STAGE_B_SHADOW_COMPLETE",
                "room_search_decision_epoch": {"epoch_id": "e2"},
                "legal_comparison_set": {
                    "comparison_completeness": "COMPLETE",
                    "all_preflight_results": [{"candidate_id": "chosen", "legality": "LEGAL_NOW", "path_length_m": 1.0}],
                },
            },
            "production_events": [{"event": "PRODUCTION_ADMISSION", "selected_candidate_id": "chosen"}],
        }
        frozen = [{"candidate_id": "chosen", "candidate_type": "GENERIC_COVERAGE", "distance_m": 1.0, "heading_rad": 0.1, "door_keepout_soft_factor": 0.5, "opportunity": {"predicted_new_cell_ids": [[1, 1]]}}]
        result = replay_stage_b_shadow_result(stage_b, frozen)
        self.assertEqual("chosen", result["shadow_selection"]["winner_candidate_id"])
        self.assertTrue(result["same_as_production"])
        self.assertFalse(result["command_authority"])


if __name__ == "__main__":
    unittest.main()
