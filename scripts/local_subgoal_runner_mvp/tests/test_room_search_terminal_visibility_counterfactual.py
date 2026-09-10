#!/usr/bin/env python3
"""Focused offline checks for terminal visibility counterfactual instrumentation."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_search_v1 import PortalAnchor, RoomSearchV2
from test_odom_cache import FakeRos, load_module


def make_search():
    target = {
        "portal_width_m": 1.2,
        "frozen_geometry": {"portal_center_odom": [0.0, 0.0], "portal_normal_odom": [1.0, 0.0]},
    }
    return RoomSearchV2(PortalAnchor.from_frozen_target(target, (0.0, 0.0, 0.0), 0.0))


def context(*, qualified=True, matched=True):
    return {
        "matched_pair_found": matched,
        "qualified": qualified,
        "failure_reason": "ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE",
        "grid_header_stamp_sec": 47.80,
        "grid_content_stamp": 47.80,
        "content_generation_id": 866,
        "grid_content_hash": "exact-pair-hash",
        "local_traversability_status": "QUALIFIED_FOR_NAVIGATION",
        "qualification_errors": [],
    }


class RoomSearchTerminalVisibilityCounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())
        self.search = make_search()
        self.anchor = self.search.anchor
        self.candidate = {"new_observable_cells": 7, "occlusion_reveal_cells": 2}
        self.terminal_pose = (3.0, 4.0, 0.75)
        self.terminal_odom = {"stamp_sec": 47.82, "pose_x_y_yaw": list(self.terminal_pose)}
        self.original_visible = self.module.room_search_v2_visible_room_points

    def tearDown(self):
        self.module.room_search_v2_visible_room_points = self.original_visible

    def test_a_uses_actual_terminal_pose_and_body_heading_before_seen_mutation(self):
        self.search.observation.seen.add((0, 0))
        calls = []

        def visible(provided_context, anchor, pose, viewpoint, heading, hfov):
            calls.append((provided_context, anchor, pose, viewpoint, heading, hfov))
            return [(0.1, 0.1), (1.1, 0.1)]

        self.module.room_search_v2_visible_room_points = visible
        before_hash = self.module.room_search_v2_seen_state_hash(self.search)
        audit, actual_visible = self.module.room_search_v2_terminal_visibility_counterfactual(
            self.search, context(), self.anchor, self.terminal_pose, self.terminal_odom, 1.2, self.candidate,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], self.terminal_pose)
        self.assertEqual(calls[0][3], (0.0, 0.0))
        self.assertEqual(calls[0][4], 0.0)
        self.assertEqual(audit["terminal_body_heading_odom_rad"], 0.75)
        self.assertEqual(audit["seen_hash_before_terminal_update"], before_hash)
        self.assertEqual(audit["seen_count_before_terminal_update"], 1)
        self.assertEqual(audit["terminal_counterfactual_new_observable_cells"], 1)
        self.assertEqual(self.module.room_search_v2_seen_state_hash(self.search), before_hash)
        actual_new = self.search.update_actual_view(self.terminal_pose, actual_visible)
        self.module.room_search_v2_finalize_terminal_visibility_counterfactual(audit, actual_new)
        self.assertEqual(actual_new, 1)
        self.assertEqual(audit["actual_new_observation_cells"], 1)
        self.assertEqual(audit["viewpoint_explained_component"], 6)
        self.assertNotEqual(self.module.room_search_v2_seen_state_hash(self.search), before_hash)

    def test_b_unavailable_exact_pair_fails_closed_diagnostically_without_seen_mutation(self):
        self.search.observation.seen.add((0, 0))
        before_hash = self.module.room_search_v2_seen_state_hash(self.search)
        audit, actual_visible = self.module.room_search_v2_terminal_visibility_counterfactual(
            self.search, context(qualified=False), self.anchor, self.terminal_pose, self.terminal_odom, 1.2, self.candidate,
        )
        self.assertEqual(audit["status"], "TERMINAL_COUNTERFACTUAL_UNAVAILABLE")
        self.assertEqual(audit["grid_identity"]["navigation_qualification"], "UNAVAILABLE_OR_UNQUALIFIED")
        self.assertEqual(actual_visible, [])
        self.assertEqual(self.module.room_search_v2_seen_state_hash(self.search), before_hash)

    def test_c_observer_has_no_candidate_ranking_or_selection_authority(self):
        candidates = [
            {"_room_search_audit_candidate_id": "a", "new_observable_cells": 2, "occlusion_reveal_cells": 0,
             "candidate_radius_m": 1.0, "heading_change_rad": 0.2, "sector": 1, "room_target_xy": [1.0, 0.0]},
            {"_room_search_audit_candidate_id": "b", "new_observable_cells": 6, "occlusion_reveal_cells": 1,
             "candidate_radius_m": 1.0, "heading_change_rad": 0.1, "sector": 2, "room_target_xy": [1.0, 0.0]},
        ]
        baseline = make_search().cheap_rank_candidates(candidates)
        observed = make_search()
        audit, _ = self.module.room_search_v2_terminal_visibility_counterfactual(
            observed, context(qualified=False), observed.anchor, self.terminal_pose, self.terminal_odom, 1.2, self.candidate,
        )
        self.assertEqual(audit["status"], "TERMINAL_COUNTERFACTUAL_UNAVAILABLE")
        after = observed.cheap_rank_candidates(candidates)
        self.assertEqual([row["_room_search_audit_candidate_id"] for row in after], [row["_room_search_audit_candidate_id"] for row in baseline])


if __name__ == "__main__":
    unittest.main(verbosity=2)
