#!/usr/bin/env python3
"""Historical eligibility replay only; it never invents an alternate execution."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

from room_search_v1 import (  # noqa: E402
    MissionActionSpec,
    PortalAnchor,
    RoomSearchV2,
    MissionActionObservationResult,
)


ARCHIVE = ROOT / "debug/state_machine_navigation/run_archives/controlled_yaw_shadow_20260903_091716/room_search_candidate_audit.jsonl"


def make_search():
    target = {"portal_width_m": 1.2, "frozen_geometry": {"portal_center_odom": [1.0, 17.0], "portal_normal_odom": [1.0, 0.0]}}
    return RoomSearchV2(PortalAnchor.from_frozen_target(target, (1.3, 16.9, 0.0), 1.0))


def rows_by_decision():
    rows = {}
    for line in ARCHIVE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("decision_index") in {33, 34, 43, 44}:
            rows[int(row["decision_index"])] = row
    return rows


def spec_from_row(row):
    data = row["mission_action"]["spec"]
    return MissionActionSpec(
        action_id=data["action_id"], candidate_id=data["candidate_id"], action_type=data["action_type"],
        target_odom_xy=tuple(data["target_odom_xy"]), observation_intent_type=data["observation_intent_type"],
        intended_observation_cell_ids=tuple(tuple(cell) for cell in data["intended_observation_cell_ids"]),
        cell_frame=data["cell_frame"], cell_resolution_m=data["cell_resolution_m"], cell_indexing=data["cell_indexing"],
        predicted_view_heading_base_rad=data["predicted_view_heading_base_rad"], aim_yaw_odom_rad=data["aim_yaw_odom_rad"],
        predicted_new_count=data["predicted_new_count"], predicted_visible_count=data["predicted_visible_count"],
        predicted_occlusion_reveal_count=data["predicted_occlusion_reveal_count"], nbv_value=data["nbv_value"],
        cheap_rank=data["cheap_rank"], sector=data["sector"],
    )


class Run091716ReplayTests(unittest.TestCase):
    def test_d33_blocks_original_d34_same_failed_claim(self):
        rows = rows_by_decision()
        d33, d34 = rows[33], rows[34]
        self.assertEqual(d33["post_execution"]["actual_new_observation_cells"], 0)
        search = make_search()
        spec = spec_from_row(d33)
        terminal = d33["mission_action"]["terminal_observation"]
        observation = MissionActionObservationResult(
            terminal["observation_intent_status"], terminal["mission_action_observation_outcome"],
            tuple(tuple(cell) for cell in terminal["actual_visible_cell_ids"]),
            tuple(tuple(cell) for cell in terminal["intended_visible_intersection"]), terminal["actual_new_observation_cells"],
        )
        search.record_failed_observation_opportunity(spec, observation, decision_id=33)
        self.assertIsNotNone(search.failed_observation_opportunity_suppression_for_spec(spec_from_row(d34)))

    def test_d43_blocks_original_d44_second_failed_claim(self):
        rows = rows_by_decision()
        d43, d44 = rows[43], rows[44]
        self.assertEqual(d43["post_execution"]["actual_new_observation_cells"], 0)
        search = make_search()
        spec = spec_from_row(d43)
        terminal = d43["mission_action"]["terminal_observation"]
        observation = MissionActionObservationResult(
            terminal["observation_intent_status"], terminal["mission_action_observation_outcome"],
            tuple(tuple(cell) for cell in terminal["actual_visible_cell_ids"]),
            tuple(tuple(cell) for cell in terminal["intended_visible_intersection"]), terminal["actual_new_observation_cells"],
        )
        search.record_failed_observation_opportunity(spec, observation, decision_id=43)
        self.assertIsNotNone(search.failed_observation_opportunity_suppression_for_spec(spec_from_row(d44)))


if __name__ == "__main__":
    unittest.main()
