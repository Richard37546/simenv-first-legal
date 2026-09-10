#!/usr/bin/env python3
"""Focused ROS-free OA-1 MissionAction semantic tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

from room_search_v1 import (  # noqa: E402
    MISSION_ACTION_INTENT_GENERIC,
    MISSION_ACTION_INTENT_OCCLUSION,
    MISSION_ACTION_TRANSLATE_AND_OBSERVE,
    ObservationMemory,
    canonical_observation_cell_ids,
    evaluate_mission_action_observation,
    freeze_mission_action_spec,
)


def generic_candidate() -> dict:
    return {
        "_room_search_audit_candidate_id": "generic-1",
        "target_priority_class": "GENERIC_COVERAGE",
        "target_xy_team_livox_odom": [23.169545, -6.884],
        "visible_room_points": [(5.25, 5.75), (5.75, 5.25), (5.75, 5.75)],
        "occlusion_reveal_room_points": [],
        "heading_change_rad": 1.304544,
        "new_observable_cells": 3,
        "nbv_value": 2.75,
        "cheap_rank": 1,
        "sector": 2,
    }


def d16_candidate(candidate_id: str) -> dict:
    if candidate_id == "raw-0057":
        return {
            **generic_candidate(),
            "_room_search_audit_candidate_id": candidate_id,
            "heading_change_rad": 1.304544283145938,
            "cheap_rank": 3,
            "sector": 8,
        }
    if candidate_id == "raw-0054":
        return {
            **generic_candidate(),
            "_room_search_audit_candidate_id": candidate_id,
            "visible_room_points": [(5.75, 5.25), (5.75, 5.75)],
            "heading_change_rad": 1.8370483401830382,
            "new_observable_cells": 2,
            "cheap_rank": 4,
            "sector": 9,
        }
    raise ValueError(candidate_id)


class MissionActionSemanticsTests(unittest.TestCase):
    def test_t1_canonical_ids_use_only_room_local_coarse_cells(self):
        self.assertEqual(
            canonical_observation_cell_ids([(0.01, 0.49), (0.49, 0.01), (0.50, 0.01), (-0.01, 0.01)]),
            ((-1, 0), (0, 0), (1, 0)),
        )

    def test_t2_generic_intent_is_predicted_cells_minus_pre_freeze_seen(self):
        spec = freeze_mission_action_spec(
            generic_candidate(), (20.0, -10.0, 0.0), "decision-1", ((10, 11),),
        )
        self.assertEqual(spec.action_type, MISSION_ACTION_TRANSLATE_AND_OBSERVE)
        self.assertEqual(spec.observation_intent_type, MISSION_ACTION_INTENT_GENERIC)
        self.assertEqual(spec.intended_observation_cell_ids, ((11, 10), (11, 11)))
        self.assertTrue(spec.intent_valid)

    def test_t3_occlusion_intent_uses_only_existing_reveal_evidence(self):
        candidate = generic_candidate()
        candidate.update({
            "target_priority_class": "OCCLUSION",
            "occlusion_reveal_room_points": [(2.01, 1.99), (2.25, 1.75), (2.51, 2.01)],
        })
        spec = freeze_mission_action_spec(candidate, (0.0, 0.0, 0.0), "decision-2", ((4, 3),))
        self.assertEqual(spec.observation_intent_type, MISSION_ACTION_INTENT_OCCLUSION)
        self.assertEqual(spec.intended_observation_cell_ids, ((4, 3), (5, 4)))

    def test_t4_freeze_does_not_follow_candidate_or_seen_mutation(self):
        candidate = generic_candidate()
        seen = [(10, 11)]
        spec = freeze_mission_action_spec(candidate, (0.0, 0.0, 0.0), "decision-3", seen)
        candidate["visible_room_points"].append((9.25, 9.25))
        seen.append((11, 10))
        self.assertEqual(spec.intended_observation_cell_ids, ((11, 10), (11, 11)))

    def test_t5_pure_predicate_truth_table(self):
        spec = freeze_mission_action_spec(generic_candidate(), (0.0, 0.0, 0.0), "decision-4", ())
        unknown = evaluate_mission_action_observation(
            spec, terminal_evidence_complete=False, actual_visible_cell_ids=((11, 10),), actual_new_observation_cells=1,
        )
        unsatisfied = evaluate_mission_action_observation(
            spec, terminal_evidence_complete=True, actual_visible_cell_ids=((7, 7),), actual_new_observation_cells=1,
        )
        new_information = evaluate_mission_action_observation(
            spec, terminal_evidence_complete=True, actual_visible_cell_ids=((11, 10),), actual_new_observation_cells=1,
        )
        no_information = evaluate_mission_action_observation(
            spec, terminal_evidence_complete=True, actual_visible_cell_ids=((11, 11),), actual_new_observation_cells=0,
        )
        self.assertEqual(unknown.mission_action_observation_outcome, "OBSERVATION_INTENT_UNKNOWN")
        self.assertEqual(unsatisfied.mission_action_observation_outcome, "OBSERVATION_INTENT_UNSATISFIED")
        self.assertEqual(new_information.mission_action_observation_outcome, "OBSERVATION_INTENT_SATISFIED_WITH_NEW_INFORMATION")
        self.assertEqual(no_information.mission_action_observation_outcome, "ACTION_COMPLETE_NO_NEW_INFORMATION")

    def test_t6_wrong_region_with_new_cells_is_not_credited(self):
        spec = freeze_mission_action_spec(generic_candidate(), (0.0, 0.0, 0.0), "decision-5", ())
        result = evaluate_mission_action_observation(
            spec, terminal_evidence_complete=True, actual_visible_cell_ids=((2, 2),), actual_new_observation_cells=6,
        )
        self.assertEqual(result.observation_intent_status, "OBSERVATION_INTENT_UNSATISFIED")
        self.assertEqual(result.intended_visible_intersection, ())

    def test_t7_heading_and_telemetry_do_not_change_completion(self):
        spec = freeze_mission_action_spec(generic_candidate(), (0.0, 0.0, 0.0), "decision-6", ())
        altered = replace(spec, predicted_view_heading_base_rad=-2.0, aim_yaw_odom_rad=2.0, nbv_value=99.0, cheap_rank=99)
        actual = ((11, 10),)
        baseline = evaluate_mission_action_observation(
            spec, terminal_evidence_complete=True, actual_visible_cell_ids=actual, actual_new_observation_cells=1,
        )
        result = evaluate_mission_action_observation(
            altered, terminal_evidence_complete=True, actual_visible_cell_ids=actual, actual_new_observation_cells=1,
        )
        self.assertEqual(result, baseline)

    def test_t8_predicate_does_not_mutate_seen_memory(self):
        memory = ObservationMemory(seen={(10, 11)})
        spec = freeze_mission_action_spec(generic_candidate(), (0.0, 0.0, 0.0), "decision-7", memory.seen)
        before = set(memory.seen)
        evaluate_mission_action_observation(
            spec, terminal_evidence_complete=True, actual_visible_cell_ids=((11, 10),), actual_new_observation_cells=1,
        )
        self.assertEqual(memory.seen, before)

    def test_t9_run0151_d16_raw_0057_preserves_observation_pending_after_position_satisfied(self):
        spec = freeze_mission_action_spec(d16_candidate("raw-0057"), (0.0, 0.0, 0.0), "run0151-d16-0057", ())
        result = evaluate_mission_action_observation(
            spec, terminal_evidence_complete=True, actual_visible_cell_ids=(), actual_new_observation_cells=0,
        )
        position_satisfied = 0.285043883 <= 0.30  # Historical D16 distance and unchanged goal tolerance.
        self.assertTrue(position_satisfied)
        self.assertEqual(spec.intended_observation_cell_ids, ((10, 11), (11, 10), (11, 11)))
        self.assertEqual(result.observation_intent_status, "OBSERVATION_INTENT_UNSATISFIED")

    def test_t10_run0151_d16_raw_0054_preserves_observation_pending_after_position_satisfied(self):
        spec = freeze_mission_action_spec(d16_candidate("raw-0054"), (0.0, 0.0, 0.0), "run0151-d16-0054", ())
        result = evaluate_mission_action_observation(
            spec, terminal_evidence_complete=True, actual_visible_cell_ids=(), actual_new_observation_cells=0,
        )
        position_satisfied = 0.285043883 <= 0.30  # Historical D16 distance and unchanged goal tolerance.
        self.assertTrue(position_satisfied)
        self.assertEqual(spec.intended_observation_cell_ids, ((11, 10), (11, 11)))
        self.assertEqual(result.observation_intent_status, "OBSERVATION_INTENT_UNSATISFIED")

    def test_t11_source_level_integration_is_after_selection_and_has_no_runner_or_command_authority(self):
        module_root = ROOT / "scripts/local_subgoal_runner_mvp"
        navigation = (module_root / "navigation_state_machine.py").read_text(encoding="utf-8")
        runner = (module_root / "block_astar_dwa_mature_runner.py").read_text(encoding="utf-8")
        selected_at = navigation.index("candidate = search.score_candidates([candidate])[0]")
        freeze_at = navigation.index("mission_action_spec: MissionActionSpec = freeze_mission_action_spec")
        terminal_at = navigation.index("mission_observation = evaluate_mission_action_observation")
        end_at = navigation.index("shadow_event: Optional[Dict[str, Any]]", terminal_at)
        semantic_block = navigation[freeze_at:end_at]
        self.assertLess(selected_at, freeze_at)
        self.assertIn("result[\"mission_actions\"].append(mission_action_record)", semantic_block)
        self.assertNotIn("publish_zero_to_topic", semantic_block)
        self.assertNotIn("runner_cmd(", semantic_block)
        self.assertNotIn("MissionActionSpec", runner)
        self.assertNotIn("evaluate_mission_action_observation", runner)


if __name__ == "__main__":
    unittest.main()
