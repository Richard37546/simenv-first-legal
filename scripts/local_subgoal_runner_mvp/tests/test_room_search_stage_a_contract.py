#!/usr/bin/env python3
"""Offline/static acceptance tests for the ROOM_SEARCH Stage-A contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "scripts/local_subgoal_runner_mvp"
sys.path.insert(0, str(MODULE_DIR))

import frozen_decision_audit
import room_search_stage_a_contract as stage_a
from room_search_v1 import PortalAnchor, RoomSearchV2


BUNDLE = ROOT / (
    "debug/state_machine_navigation/run_archives/"
    "run_0139_20260822_224215_396672930_pid413263/frozen_decisions/"
    "run_0139_20260822_224215_396672930_pid413263/frozen_decision_0001"
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_historical_replay(bundle):
    # Runtime bytecode is not part of the immutable bundle manifest. Clean
    # only that derived cache before running the archived source itself.
    shutil.rmtree(bundle / "sources/scripts/local_subgoal_runner_mvp/__pycache__", ignore_errors=True)
    script = bundle / "sources/scripts/local_subgoal_runner_mvp/frozen_decision_audit.py"
    completed = subprocess.run(
        [sys.executable, str(script), "replay", str(bundle)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode != 0:
        raise AssertionError(f"historical_replay_failed:{completed.stdout}:{completed.stderr}")
    return json.loads(completed.stdout)


def claim(candidate_id, cells, grid_hash="grid-a"):
    return stage_a.CandidateOpportunityClaim(
        candidate_id=candidate_id,
        rank=1,
        visible_cell_ids=tuple(cells),
        new_cell_ids=tuple(cells),
        occlusion_reveal_cell_ids=(),
        danger_task_identities=(),
        status=stage_a.OPPORTUNITY_NONEMPTY if cells else stage_a.OPPORTUNITY_EMPTY,
        mission_value_semantics=(
            "DIRECT_OBSERVATION_OR_TASK_VALUE_EVIDENCED" if cells else "REPOSITION_OR_UNKNOWN_MISSION_VALUE"
        ),
        grid_hash=grid_hash,
        seen_hash="seen",
        provenance="test",
    )


class RoomSearchStageAContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BUNDLE.is_dir():
            raise unittest.SkipTest(f"RUN0139 fixture missing: {BUNDLE}")
        cls.epoch = stage_a.RoomSearchDecisionEpoch.from_bundle(BUNDLE)
        cls.candidates = read_json(BUNDLE / "candidates.json")["ranked_candidates"]
        cls.historical_replay = exact_historical_replay(BUNDLE)
        cls.run_result = stage_a.run_frozen_stage_a(
            BUNDLE,
            historical_provenance_replay=cls.historical_replay,
        )

    def test_a_collector_does_not_stop_at_first_legal_or_candidate_illegal(self):
        calls = []

        def preflight(candidate):
            calls.append(candidate["candidate_id"])
            legality = "ILLEGAL" if candidate["rank"] == 2 else "LEGAL_NOW"
            return {
                "rank": candidate["rank"],
                "candidate_id": candidate["candidate_id"],
                "legality": legality,
                "astar_result": "PATH" if legality == "LEGAL_NOW" else "NO_PATH",
                "candidate_specific_rejection": None if legality == "LEGAL_NOW" else "NO_PATH",
                "commands_published": False,
            }

        results = stage_a.collect_same_epoch_preflights(self.epoch, self.candidates[:3], preflight)
        self.assertEqual(calls, [candidate["candidate_id"] for candidate in self.candidates[:3]])
        self.assertEqual([result.legality for result in results], ["LEGAL_NOW", "ILLEGAL", "LEGAL_NOW"])
        self.assertEqual({result.epoch_id for result in results}, {self.epoch.epoch_id})
        self.assertEqual({result.grid_hash for result in results}, {self.epoch.grid_hash})
        self.assertFalse(any(result.commands_published for result in results))

    def test_b_global_invalid_stops_epoch(self):
        calls = []

        def preflight(candidate):
            calls.append(candidate["candidate_id"])
            return {
                "rank": candidate["rank"],
                "candidate_id": candidate["candidate_id"],
                "legality": "INVALID",
                "failure_reason": "DECISION_GLOBAL_L3V_STATUS",
                "commands_published": False,
            }

        results = stage_a.collect_same_epoch_preflights(self.epoch, self.candidates, preflight)
        comparison = stage_a.build_legal_comparison_set(self.epoch, results)
        self.assertEqual(len(calls), 1)
        self.assertEqual(comparison.comparison_completeness, "INCOMPLETE")
        self.assertEqual(comparison.decision_global_invalidation, "DECISION_GLOBAL_L3V_STATUS")

    def test_c_dry_preflight_command_is_rejected(self):
        candidate = self.candidates[0]
        with self.assertRaisesRegex(ValueError, "published_command"):
            stage_a.collect_same_epoch_preflights(
                self.epoch,
                [candidate],
                lambda row: {
                    "rank": row["rank"], "candidate_id": row["candidate_id"],
                    "legality": "LEGAL_NOW", "commands_published": True,
                },
            )

    def test_d_exact_opportunity_relations(self):
        by_id = {item["candidate_id"]: item for item in self.run_result["room_search_decision_epoch"]["opportunity_claims"]}
        relations = {
            (item["a_candidate_id"], item["b_candidate_id"]): item["relation"]
            for item in self.run_result["opportunity_exact_set_relations"]
        }
        self.assertEqual(relations[("raw-0827", "raw-0896")], "PARTIAL_OVERLAP_WITH_UNIQUE_CELLS")
        self.assertEqual(relations[("raw-0827", "raw-0721")], "STRICT_SUPERSET")
        self.assertEqual(relations[("raw-0827", "raw-0548")], "DISJOINT")
        self.assertEqual(by_id["raw-0676"]["status"], stage_a.OPPORTUNITY_EMPTY)
        self.assertEqual(by_id["raw-0676"]["mission_value_semantics"], "REPOSITION_OR_UNKNOWN_MISSION_VALUE")

    def test_e_commit_context_matrix_never_silently_demotes(self):
        old = claim("selected", ((1, 1), (2, 2)))
        identical = claim("selected", ((1, 1), (2, 2)), grid_hash="grid-b")
        changed = claim("selected", ((1, 1), (3, 3)), grid_hash="grid-b")
        same_new_but_changed_occlusion = stage_a.CandidateOpportunityClaim(
            **{
                **stage_a.asdict(identical),
                "occlusion_reveal_cell_ids": ((9, 9),),
            }
        )
        epoch_identity = {"generation": 996, "hash": "grid-a"}
        same_identity = dict(epoch_identity)
        changed_identity = {"generation": 999, "hash": "grid-b"}
        cases = [
            stage_a.validate_commit_context(epoch_identity, same_identity, old, old, "LEGAL_NOW"),
            stage_a.validate_commit_context(epoch_identity, changed_identity, old, identical, "ILLEGAL"),
            stage_a.validate_commit_context(epoch_identity, changed_identity, old, changed, "LEGAL_NOW"),
            stage_a.validate_commit_context(epoch_identity, changed_identity, old, identical, "LEGAL_NOW"),
            stage_a.validate_commit_context(
                epoch_identity, changed_identity, old, identical, "LEGAL_NOW", commit_global_status="TRANSIENT",
            ),
            stage_a.validate_commit_context(
                epoch_identity, changed_identity, old, same_new_but_changed_occlusion, "LEGAL_NOW",
            ),
        ]
        self.assertEqual(
            [case["result"] for case in cases],
            [
                "CONTEXT_CONSISTENT",
                "DISCARD_EPOCH_AND_REGENERATE",
                "DISCARD_EPOCH_AND_REGENERATE",
                "COMMIT_CONTEXT_VALID",
                "DISCARD_EPOCH_AND_REGENERATE",
                "DISCARD_EPOCH_AND_REGENERATE",
            ],
        )
        self.assertTrue(all(case["silent_demotion_performed"] is False for case in cases))
        self.assertTrue(all(case["fallback_candidate_id"] is None for case in cases))

    def test_f_run0139_exact_legal_set_and_rank1_runner_evidence(self):
        self.assertEqual(self.run_result["status"], "STAGE_A_OFFLINE_COMPARISON_COMPLETE")
        comparison = self.run_result["legal_comparison_set"]
        by_rank = {result["rank"]: result for result in comparison["all_preflight_results"]}
        legal_ranks = {result["rank"] for result in comparison["all_preflight_results"] if result["legality"] == "LEGAL_NOW"}
        illegal_ranks = {result["rank"] for result in comparison["all_preflight_results"] if result["legality"] == "ILLEGAL"}
        self.assertEqual(legal_ranks, {1, 2, 3, 5, 7, 8, 9, 10, 11})
        self.assertEqual(illegal_ranks, {4, 6})
        self.assertEqual(by_rank[1]["astar_result"], "PATH")
        self.assertEqual(by_rank[1]["path_cell_count"], 5)
        self.assertAlmostEqual(by_rank[1]["path_length_m"], 0.25, places=6)
        self.assertEqual((by_rank[1]["dwa_safe_moving"], by_rank[1]["dwa_admitted_count"]), (22, 33))
        self.assertFalse(any(result["commands_published"] for result in comparison["all_preflight_results"]))

    def test_g_empty_opportunity_legal_candidates_are_retained(self):
        comparison = self.run_result["legal_comparison_set"]
        self.assertEqual(
            comparison["empty_opportunity_legal_candidates"],
            ["raw-0676", "raw-0806", "raw-0918", "raw-0479", "raw-1164"],
        )
        self.assertTrue(set(comparison["empty_opportunity_legal_candidates"]).issubset(comparison["LEGAL_EXECUTABLE_SET"]))

    def test_h_current_nbv_scores_multiple_candidates_without_selection_or_mutation(self):
        nbv = self.run_result["nbv"]
        self.assertEqual(nbv["authority"], "ADVISORY_MULTI_CANDIDATE_EVIDENCE")
        self.assertEqual([row["cheap_rank"] for row in nbv["candidate_evidence"]], [1, 2, 3, 5])
        self.assertEqual([row["new_observable_cells"] for row in nbv["candidate_evidence"]], [5, 4, 4, 3])
        self.assertFalse(nbv["selection_performed"])
        self.assertFalse(nbv["select_best_called"])
        self.assertFalse(nbv["state_mutated"])

    def test_i_supplied_snapshot_matches_direct_existing_replay(self):
        replay = self.historical_replay
        self.assertEqual(replay["status"], "DRY_PREFLIGHT_COMPLETE")
        stage_results = self.run_result["legal_comparison_set"]["all_preflight_results"]
        for direct, staged in zip(replay["results"], stage_results):
            for key in (
                "rank", "candidate_id", "legality", "astar_result", "path_cell_count",
                "path_length_m", "dwa_safe_moving", "dwa_admitted_count", "failure_reason",
                "commands_published",
            ):
                self.assertEqual(direct.get(key), staged.get(key), (key, direct, staged))

    def test_j_production_first_legal_and_planner_sources_are_unchanged(self):
        source_manifest = read_json(BUNDLE / "source_manifest.json")
        frozen_hashes = {row["repo_relative_path"]: row["sha256"] for row in source_manifest["source_files"]}
        production_paths = [
            "scripts/local_subgoal_runner_mvp/room_search_v1.py",
            "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py",
        ]
        for relative in production_paths:
            frozen_copy = BUNDLE / "sources" / relative
            self.assertEqual(sha256(frozen_copy), frozen_hashes[relative])
            self.assertNotIn("room_search_stage_a_contract", (ROOT / relative).read_text(encoding="utf-8"))

        navigation = (ROOT / "scripts/local_subgoal_runner_mvp/navigation_state_machine.py").read_text(encoding="utf-8")
        self.assertIn("RoomSearchStageBShadowCapture", navigation)
        self.assertIn("room_search_v2_admit_with_l3v_consistency", navigation)
        self.assertNotIn("room_search_stage_a_contract", navigation)

        anchor = PortalAnchor(
            center_xy=(0.0, 0.0), inward_normal=(1.0, 0.0), tangent=(0.0, 1.0), width_m=1.0,
            door_return_anchor_xy_yaw=(0.0, 0.0, 0.0), door_return_anchor_stamp_sec=0.0,
        )
        search = RoomSearchV2(anchor)
        calls = []
        candidates = [{"candidate_id": "rank1"}, {"candidate_id": "rank2"}]
        selected, attempts = search.admit_ranked_candidates(
            candidates,
            lambda candidate: calls.append(candidate["candidate_id"]) or {"legal": True},
        )
        self.assertEqual((selected["candidate_id"], attempts, calls), ("rank1", 1, ["rank1"]))
        self.assertEqual(self.run_result["first_legal_baseline"]["candidate_id"], "raw-0827")
        self.assertFalse(self.run_result["first_legal_baseline"]["production_selected_target_changed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
