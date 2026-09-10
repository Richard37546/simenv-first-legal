#!/usr/bin/env python3
"""Focused regressions for candidate-evidence observation only."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from room_search_v1 import PortalAnchor, RoomSearchV2  # noqa: E402
from test_odom_cache import FakeRos, load_module  # noqa: E402


def make_search():
    target = {
        "portal_width_m": 1.2,
        "frozen_geometry": {"portal_center_odom": [0.0, 0.0], "portal_normal_odom": [1.0, 0.0]},
    }
    return RoomSearchV2(PortalAnchor.from_frozen_target(target, (0.0, 0.0, 0.0), 0.0))


def enrich(occlusion, coverage):
    return lambda row: {
        **row,
        "occlusion_reveal_room_points": [(10.0 + index, 0.0) for index in range(occlusion)],
        "visible_room_points": [(20.0 + index, 0.0) for index in range(coverage)],
    }


class RoomSearchCandidateEvidenceInstrumentationTests(unittest.TestCase):
    def test_a_observing_raw_candidates_does_not_change_selected_representative(self):
        points = [(1.0, -0.20), (1.0, -0.05), (1.0, 0.20)]
        chooser = lambda row: enrich(1, 6)(row) if float(row["base_xy"][1]) > -0.1 else enrich(0, 2)(row)
        baseline = make_search().candidates_from_planning_free_base((0.0, 0.0, 0.0), points, chooser)
        audit = {}
        observed = make_search().candidates_from_planning_free_base((0.0, 0.0, 0.0), points, chooser, audit=audit)
        self.assertEqual(
            [{key: value for key, value in row.items() if key != "_room_search_audit_candidate_id"} for row in observed],
            baseline,
        )
        self.assertEqual(audit["raw_candidate_count_before_sector_compression"], 3)
        self.assertEqual(len(audit["raw_candidates"]), 3)

    def test_b_sector_loser_keeps_existing_reason_and_winner(self):
        audit = {}
        candidates = make_search().candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(1.0, -0.20), (1.0, -0.05)],
            lambda row: enrich(0, 2)(row) if float(row["base_xy"][1]) < -0.1 else enrich(1, 6)(row), audit=audit,
        )
        self.assertEqual(candidates[0]["base_xy"], [1.0, -0.05])
        discarded = next(row for row in audit["raw_candidates"] if not row["survived_as_sector_representative"])
        self.assertEqual(discarded["representative_comparison_reason"], "LOWER_OCCLUSION")
        self.assertEqual(discarded["representative_winner_candidate_id"], "raw-0001")

    def test_c_rejected_preflight_evidence_is_observed_without_changing_admission(self):
        ranked = [{"_room_search_audit_candidate_id": "a"}, {"_room_search_audit_candidate_id": "b"}]
        preflight = lambda row: {"legal": row["_room_search_audit_candidate_id"] == "b", "runner": {"runner_final_decision": "OK"}}
        baseline, baseline_attempts = RoomSearchV2.admit_ranked_candidates(ranked, preflight)
        observed = []
        selected, attempts = RoomSearchV2.admit_ranked_candidates(
            ranked, preflight, audit_observer=lambda rank, row, result: observed.append((rank, row["_room_search_audit_candidate_id"], result["legal"])),
        )
        self.assertEqual((selected, attempts), (baseline, baseline_attempts))
        self.assertEqual(observed, [(1, "a", False), (2, "b", True)])

    def test_d_terminal_expansion_observes_every_existing_rejected_candidate(self):
        ranked = [{"_room_search_audit_candidate_id": f"raw-{index:04d}"} for index in range(11)]
        observed = []
        selected, attempts, expansion = RoomSearchV2.admit_with_terminal_expansion(
            ranked, lambda _row: {"legal": False}, normal_cap=5,
            audit_observer=lambda rank, row, result: observed.append((rank, row["_room_search_audit_candidate_id"], result["legal"])),
        )
        self.assertIsNone(selected)
        self.assertEqual((attempts, expansion), (11, 6))
        self.assertEqual(observed, [(index + 1, f"raw-{index:04d}", False) for index in range(11)])

    def test_e_p2_revisit_rank_remains_the_existing_soft_tiebreak(self):
        search = make_search()
        search.record_actual_breadcrumb((1.0, 1.0, 0.0), 0.0)
        recent = {"target_xy_team_livox_odom": [1.25, 1.25], "room_target_xy": [2.0, 0.0], "visible_room_points": [(1.0, 0.0)], "occlusion_reveal_room_points": [], "candidate_radius_m": 1.0, "heading_change_rad": 0.0, "sector": 0}
        away = {"target_xy_team_livox_odom": [3.0, 1.0], "room_target_xy": [2.0, 0.0], "visible_room_points": [(2.0, 0.0)], "occlusion_reveal_room_points": [], "candidate_radius_m": 1.0, "heading_change_rad": 0.0, "sector": 1}
        ranked = search.cheap_rank_candidates([recent, away])
        self.assertEqual(ranked[0]["target_xy_team_livox_odom"], [3.0, 1.0])
        self.assertTrue(ranked[1]["generic_trajectory_revisit_preference_applied"])

    def test_f_completion_contract_is_unchanged_and_unavailable_fields_are_explicit(self):
        module = load_module(FakeRos())
        state = module.room_search_v2_completion_contract("NO_SAFE_USEFUL_CANDIDATE", None)
        self.assertFalse(state["mission_complete"])
        record = module.room_search_v2_preflight_audit_record(1, {}, {"legal": False, "runner": {}})
        self.assertEqual(record["target_grid_value"], "UNAVAILABLE_EXISTING_INTERFACE")
        self.assertEqual(record["first_rejection_reason"], "UNAVAILABLE_EXISTING_INTERFACE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
