#!/usr/bin/env python3
"""Focused regression for mission-local zero-observation opportunity feedback."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

from room_search_v1 import (  # noqa: E402
    ANTI_INFINITE_DECISION_GUARD,
    PortalAnchor,
    RoomSearchV2,
    evaluate_mission_action_observation,
    freeze_mission_action_spec,
)


def make_search():
    target = {
        "portal_width_m": 1.2,
        "frozen_geometry": {"portal_center_odom": [10.0, 20.0], "portal_normal_odom": [1.0, 0.0]},
    }
    return RoomSearchV2(PortalAnchor.from_frozen_target(target, (10.3, 20.0, 0.2), 12.0))


def candidate(target=(11.0, 20.0), visible=((1.1, 1.1),)):
    return {
        "_room_search_audit_candidate_id": "raw-test",
        "target_xy_team_livox_odom": list(target),
        "target_priority_class": "GENERIC_COVERAGE",
        "visible_room_points": list(visible),
        "new_observable_cells": len(visible),
        "candidate_radius_m": 0.7,
        "heading_change_rad": 0.0,
    }


def zero_observation(search, action):
    spec = freeze_mission_action_spec(action, (10.3, 20.0, 0.2), "action-test", search.observation.seen)
    result = evaluate_mission_action_observation(
        spec,
        terminal_evidence_complete=True,
        actual_visible_cell_ids=((9, 9),),
        actual_new_observation_cells=0,
    )
    return spec, result


class ZeroObservationFeedbackTests(unittest.TestCase):
    def test_positive_actual_gain_does_not_create_suppression(self):
        search = make_search()
        spec = freeze_mission_action_spec(candidate(), (10.3, 20.0, 0.2), "positive", search.observation.seen)
        result = evaluate_mission_action_observation(
            spec, terminal_evidence_complete=True,
            actual_visible_cell_ids=spec.intended_observation_cell_ids,
            actual_new_observation_cells=1,
        )
        record = search.record_failed_observation_opportunity(spec, result, decision_id=1)
        self.assertFalse(record["recorded"])
        self.assertEqual(record["reason"], "ACTUAL_OBSERVATION_PROGRESS")
        self.assertFalse(search.failed_observation_opportunities)

    def test_zero_unsatisfied_action_records_only_local_failure(self):
        search = make_search()
        spec, result = zero_observation(search, candidate())
        record = search.record_failed_observation_opportunity(spec, result, decision_id=33)
        self.assertTrue(record["recorded"])
        self.assertEqual(record["reason"], "OBSERVATION_OPPORTUNITY_ZERO_ACTUAL_GAIN")
        self.assertEqual(record["decision_id"], 33)
        self.assertEqual(record["candidate_id"], "raw-test")
        self.assertEqual(record["actual_new_observation_cells"], 0)

    def test_same_cells_and_same_local_viewpoint_are_not_eligible_next_decision(self):
        search = make_search()
        spec, result = zero_observation(search, candidate())
        search.record_failed_observation_opportunity(spec, result, decision_id=33)
        next_spec = freeze_mission_action_spec(candidate((11.1, 20.0)), (10.3, 20.0, 0.2), "next", search.observation.seen)
        self.assertIsNotNone(search.failed_observation_opportunity_suppression_for_spec(next_spec))

    def test_different_viewpoint_or_opportunity_is_not_killed(self):
        search = make_search()
        spec, result = zero_observation(search, candidate())
        search.record_failed_observation_opportunity(spec, result, decision_id=33)
        distant = freeze_mission_action_spec(candidate((12.2, 20.0)), (10.3, 20.0, 0.2), "distant", search.observation.seen)
        different_cells = freeze_mission_action_spec(candidate((11.0, 20.0), ((2.1, 2.1),)), (10.3, 20.0, 0.2), "different", search.observation.seen)
        self.assertIsNone(search.failed_observation_opportunity_suppression_for_spec(distant))
        self.assertIsNone(search.failed_observation_opportunity_suppression_for_spec(different_cells))

    def test_failed_opportunity_does_not_mutate_seen_or_completion_lifecycle(self):
        search = make_search()
        spec, result = zero_observation(search, candidate())
        before_seen = set(search.observation.seen)
        search.record_failed_observation_opportunity(spec, result, decision_id=33)
        self.assertEqual(search.observation.seen, before_seen)
        marginal = search.evaluate_marginal_value({"nbv_value": 10.0, "path_length_m": 0.5}, 1)
        self.assertIsNone(marginal["completion_reason"])
        self.assertEqual(ANTI_INFINITE_DECISION_GUARD, 24)

    def test_candidate_generation_filters_the_same_failed_claim(self):
        search = make_search()
        first = candidate()
        spec, result = zero_observation(search, first)
        search.record_failed_observation_opportunity(spec, result, decision_id=33)

        def enrich(row):
            row.update({"target_priority_class": "GENERIC_COVERAGE", "visible_room_points": [(1.1, 1.1)]})
            return row

        generated = search.candidates_from_planning_free_base(
            (10.3, 20.0, 0.0), [(0.7, 0.0)], task_enricher=enrich,
        )
        self.assertEqual(generated, [])

    def test_suppressed_claim_cannot_hide_a_different_same_sector_claim(self):
        search = make_search()
        spec, result = zero_observation(search, candidate((11.0, 20.0), ((1.1, 1.1),)))
        search.record_failed_observation_opportunity(spec, result, decision_id=33)

        def enrich(row):
            if row["base_xy"][0] < 0.8:
                row.update({"target_priority_class": "GENERIC_COVERAGE", "visible_room_points": [(1.1, 1.1)]})
            else:
                row.update({"target_priority_class": "GENERIC_COVERAGE", "visible_room_points": [(2.1, 2.1)]})
            return row

        generated = search.candidates_from_planning_free_base(
            (10.3, 20.0, 0.0), [(0.7, 0.0), (0.9, 0.0)], task_enricher=enrich,
        )
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0]["new_observable_cells"], 1)


if __name__ == "__main__":
    unittest.main()
