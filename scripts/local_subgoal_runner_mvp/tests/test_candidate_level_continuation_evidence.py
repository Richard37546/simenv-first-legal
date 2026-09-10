#!/usr/bin/env python3
"""Focused tests for the Stage-B candidate-level, shadow-only C0 adapter."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "scripts/local_subgoal_runner_mvp"
sys.path.insert(0, str(MODULE_DIR))

import room_search_stage_a_contract as stage_a  # noqa: E402


class FakeCandidate:
    @classmethod
    def from_mapping(cls, row):
        target = row.get("target_odom_xy")
        if not isinstance(target, list):
            raise ValueError("TARGET_MISSING")
        return types.SimpleNamespace(candidate_id=row["candidate_id"], target=tuple(target), candidate_type=row.get("candidate_type"))


class FakeEpoch:
    @classmethod
    def from_live_inputs(cls, **kwargs):
        return types.SimpleNamespace(epoch_id=kwargs["epoch_id"], args=kwargs["args"])


class FakeRunner:
    FrozenRoomLocalEpoch = FakeEpoch
    FrozenRoomLocalCandidate = FakeCandidate
    ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_OFFLINE_FROZEN = "offline_frozen"
    CONTINUATION_LOCAL_SELECTION_AUTHORITY_DISABLED = "disabled"

    @staticmethod
    def evaluate_frozen_room_local_candidate(epoch, candidate):
        assert epoch.args.execute is False
        assert epoch.args.room_local_phase2_productive_admission == "offline_frozen"
        return {"motion_candidates": [{"candidate": candidate.candidate_id, "score_eligible": True}]}

    @staticmethod
    def evaluate_frozen_room_local_continuation_cohort(epoch, candidate, motions):
        status = {
            "viable": "CONTINUATION_VIABLE",
            "nonviable": "CONTINUATION_NON_VIABLE",
            "unknown": "CONTINUATION_UNKNOWN",
        }[candidate.candidate_id]
        reason = {
            "viable": "AT_LEAST_ONE_CHECKED_MOTION_VIABLE",
            "nonviable": "ALL_PRODUCTIVE_MOTIONS_CHECKED_NON_VIABLE",
            "unknown": "CURRENT_PRODUCTIVE_SET_EMPTY",
        }[candidate.candidate_id]
        return {"continuation_status": status, "reason": reason, "records": list(motions)}


class CandidateLevelContinuationEvidenceTests(unittest.TestCase):
    def evaluate(self, candidates, legal, *, epoch="epoch-a", status="FREE_SUPPORTED"):
        args = types.SimpleNamespace(execute=True, room_local_phase2_productive_admission="disabled")
        return stage_a.candidate_level_continuation_evidence(
            runner_module=FakeRunner, runner_args=args, grid_msg=object(),
            status={"local_traversability_status": status}, pose_odom_xy_yaw=(0.0, 0.0, 0.0),
            epoch_id=epoch, candidates=candidates, formal_legal_candidate_ids=legal,
        )

    def test_arbitrary_legal_candidates_receive_independent_v_n_u_evidence(self):
        rows = [
            {"candidate_id": "viable", "target_odom_xy": [1.0, 0.0]},
            {"candidate_id": "nonviable", "target_odom_xy": [2.0, 0.0]},
            {"candidate_id": "unknown", "target_odom_xy": [3.0, 0.0], "candidate_type": "DANGER_REOBSERVE"},
            {"candidate_id": "illegal", "target_odom_xy": [4.0, 0.0]},
        ]
        evidence = self.evaluate(rows, ["viable", "nonviable", "unknown"])
        self.assertEqual(evidence["viable"]["status"], "CONTINUATION_VIABLE")
        self.assertTrue(evidence["viable"]["complete"])
        self.assertEqual(evidence["nonviable"]["status"], "CONTINUATION_NON_VIABLE")
        self.assertTrue(evidence["nonviable"]["complete"])
        self.assertEqual(evidence["unknown"]["status"], "CONTINUATION_UNKNOWN")
        self.assertEqual(evidence["unknown"]["reason"], "INSUFFICIENT_LOCAL_MOTION_EVIDENCE")
        self.assertNotIn("illegal", evidence)
        self.assertFalse(any(row["commands_published"] or row["selection_authority"] for row in evidence.values()))

    def test_epoch_identity_and_candidate_identity_cannot_be_mixed(self):
        rows = [{"candidate_id": "viable", "target_odom_xy": [1.0, 0.0]}]
        first = self.evaluate(rows, ["viable"], epoch="epoch-one")["viable"]
        second = self.evaluate(rows, ["viable"], epoch="epoch-two")["viable"]
        self.assertEqual(first["candidate_id"], second["candidate_id"])
        self.assertNotEqual(first["epoch_id"], second["epoch_id"])

    def test_grid_unqualified_is_explicit_unknown_not_a_false_nonviable(self):
        evidence = self.evaluate(
            [{"candidate_id": "viable", "target_odom_xy": [1.0, 0.0]}], ["viable"],
            status="CONFLICT_NEEDS_CAUTION",
        )
        self.assertEqual(evidence["viable"]["status"], "CONTINUATION_UNKNOWN")
        self.assertEqual(evidence["viable"]["reason"], "GRID_STATUS_UNQUALIFIED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
