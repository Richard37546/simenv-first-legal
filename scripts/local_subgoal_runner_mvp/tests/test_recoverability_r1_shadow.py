#!/usr/bin/env python3
"""Offline contracts for Recoverability R1 shadow evidence only."""

import copy
import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

from room_search_recoverability import (  # noqa: E402
    MAX_RECOVERABILITY_PREDECESSORS,
    NON_RECOVERABLE,
    RECOVERABLE,
    TERMINAL_STATE_UNKNOWN,
    UNKNOWN,
    PredictedTerminalState,
    RecoverabilityCertificate,
    evaluate_predecessor_set,
    evaluate_retreat,
    make_door_anchor_root_certificate,
    prepare_candidate_terminal_state,
    terminal_accuracy_telemetry,
)


EPOCH = {
    "grid_header_stamp_sec": 10.0,
    "grid_content_stamp": 10.0,
    "content_generation_id": "g10",
    "grid_content_hash": "h10",
}


class Anchor:
    door_return_anchor_xy_yaw = (0.0, 0.0, 0.0)
    door_return_anchor_stamp_sec = 1.0


def terminal(candidate_id="candidate_a"):
    return PredictedTerminalState(
        candidate_id=candidate_id, decision_id=7, pose_xy_yaw=(1.0, 0.0, 0.0),
        completion_semantics="FULL_CANDIDATE_ACTION_COMPLETION", prediction_horizon_sec=2.0,
        source="TEST_FULL_ACTION", source_evidence={"frozen": True},
        status="TERMINAL_STATE_AVAILABLE", reason="TEST", epoch_identity=dict(EPOCH),
    )


def predecessor(state_id="R0", recency_index=0, *, fresh=True, epoch=EPOCH):
    return RecoverabilityCertificate(
        state_id=state_id, pose_xy_yaw=(0.0, 0.0, 0.0), status=RECOVERABLE,
        predecessor_state_id=None, epoch_identity=dict(epoch), retreat_transition_type="ROOT",
        fresh=fresh, reason="TEST_ROOT", evidence={"recency_index": recency_index},
    )


def complete(executable, reason="TEST"):
    return {"complete": True, "executable": executable, "reason": reason}


class RecoverabilityR1ShadowTests(unittest.TestCase):
    def test_door_anchor_is_root_certificate_not_breadcrumb_alias(self):
        root = make_door_anchor_root_certificate(Anchor(), EPOCH)
        self.assertEqual(root.status, RECOVERABLE)
        self.assertEqual(root.state_id, "R0_DOOR_ANCHOR")
        self.assertEqual(root.pose_xy_yaw, Anchor.door_return_anchor_xy_yaw)
        self.assertEqual(root.evidence["recency_index"], 0)

    def test_current_preflight_refuses_to_relabel_one_slice_as_full_terminal(self):
        prediction = prepare_candidate_terminal_state({
            "_room_search_audit_candidate_id": "A",
            "runner": {"runner_final_decision": "BLOCK_ASTAR_DWA_REACHED_GOAL", "last_dwa": {"slice_endpoint_base_xy": [0.1, 0.0]}},
        }, 3, EPOCH)
        self.assertEqual(prediction.status, TERMINAL_STATE_UNKNOWN)
        self.assertIsNone(prediction.pose_xy_yaw)
        self.assertTrue(prediction.source_evidence["last_dwa_not_used_as_terminal"])

    def test_explicit_full_action_terminal_is_the_only_current_accepted_source(self):
        prediction = prepare_candidate_terminal_state({
            "_room_search_audit_candidate_id": "A",
            "runner": {"recoverability_candidate_terminal_prediction": {
                "candidate_id": "A", "pose_xy_yaw": [1.0, 2.0, 0.3],
                "completion_semantics": "FULL_CANDIDATE_ACTION_COMPLETION",
                "prediction_horizon_sec": 4.0, "source": "FROZEN_RUNNER_ACTION_TERMINAL",
            }},
        }, 4, EPOCH)
        self.assertEqual(prediction.status, "TERMINAL_STATE_AVAILABLE")
        self.assertEqual(prediction.pose_xy_yaw, (1.0, 2.0, 0.3))

    def test_direct_retreat_success(self):
        certificate = evaluate_retreat(terminal(), predecessor(), {
            "direct_translation": complete(True, "DIRECT_OK"),
            "orientation_then_translation": complete(False),
        })
        self.assertEqual(certificate.status, RECOVERABLE)
        self.assertEqual(certificate.retreat_transition_type, "DIRECT_TRANSLATION")

    def test_behind_predecessor_can_use_one_orientation_then_translation(self):
        certificate = evaluate_retreat(terminal(), predecessor(), {
            "direct_translation": complete(False, "PREDECESSOR_NOT_FORWARD_SUPPORTED"),
            "orientation_then_translation": complete(True, "ONE_ORIENTATION_AND_TRANSLATION_OK"),
        })
        self.assertEqual(certificate.status, RECOVERABLE)
        self.assertEqual(certificate.retreat_transition_type, "ORIENTATION_THEN_TRANSLATION")

    def test_collision_blocked_complete_retreat_is_non_recoverable(self):
        certificate = evaluate_retreat(terminal(), predecessor(), {
            "direct_translation": complete(False, "DIRECT_COLLISION_BLOCKED"),
            "orientation_then_translation": complete(False, "ORIENTATION_TRANSLATION_COLLISION_BLOCKED"),
        })
        self.assertEqual(certificate.status, NON_RECOVERABLE)

    def test_stale_predecessor_is_unknown(self):
        certificate = evaluate_retreat(terminal(), predecessor(fresh=False), {
            "direct_translation": complete(True), "orientation_then_translation": complete(False),
        })
        self.assertEqual(certificate.status, UNKNOWN)
        self.assertEqual(certificate.reason, "PREDECESSOR_CERTIFICATE_STALE_OR_UNCERTIFIED")

    def test_terminal_missing_is_unknown_not_non_recoverable(self):
        missing = prepare_candidate_terminal_state({"candidate_id": "A", "runner": {}}, 1, EPOCH)
        certificate = evaluate_predecessor_set(missing, [predecessor()], {})
        self.assertEqual(certificate.status, UNKNOWN)
        self.assertEqual(certificate.reason, "TERMINAL_STATE_UNKNOWN_NO_COMPLETE_ACTION_PREDICTION")

    def test_missing_transition_input_is_unknown(self):
        certificate = evaluate_retreat(terminal(), predecessor(), {})
        self.assertEqual(certificate.status, UNKNOWN)
        self.assertNotEqual(certificate.status, NON_RECOVERABLE)

    def test_first_predecessor_failure_does_not_hide_second_success(self):
        newer = predecessor("R2", recency_index=2)
        older = predecessor("R1", recency_index=1)
        certificate = evaluate_predecessor_set(terminal(), [older, newer], {
            "R2": {"direct_translation": complete(False), "orientation_then_translation": complete(False)},
            "R1": {"direct_translation": complete(True), "orientation_then_translation": complete(False)},
        }, max_predecessors=2)
        self.assertEqual(certificate.status, RECOVERABLE)
        self.assertEqual(certificate.predecessor_state_id, "R1")

    def test_cap_truncation_is_unknown(self):
        predecessors = [predecessor("R%d" % index, recency_index=index) for index in range(4)]
        evidence = {
            item.state_id: {"direct_translation": complete(False), "orientation_then_translation": complete(False)}
            for item in predecessors
        }
        certificate = evaluate_predecessor_set(terminal(), predecessors, evidence, max_predecessors=3)
        self.assertEqual(certificate.status, UNKNOWN)
        self.assertEqual(certificate.reason, "PREDECESSOR_CAP_TRUNCATED")
        self.assertEqual(MAX_RECOVERABILITY_PREDECESSORS, 3)

    def test_complete_negative_predecessor_set_is_non_recoverable(self):
        certificate = evaluate_predecessor_set(terminal(), [predecessor()], {
            "R0": {"direct_translation": complete(False), "orientation_then_translation": complete(False)},
        })
        self.assertEqual(certificate.status, NON_RECOVERABLE)

    def test_selected_vs_alternative_fixture_remains_evidence_only(self):
        candidate_a = evaluate_retreat(terminal("A"), predecessor(), {
            "direct_translation": complete(False), "orientation_then_translation": complete(False),
        })
        candidate_b = evaluate_retreat(terminal("B"), predecessor(), {
            "direct_translation": complete(True), "orientation_then_translation": complete(False),
        })
        candidate_unknown = evaluate_retreat(terminal("U"), predecessor(), {})
        self.assertEqual((candidate_a.status, candidate_b.status), (NON_RECOVERABLE, RECOVERABLE))
        self.assertEqual((candidate_unknown.status, candidate_b.status), (UNKNOWN, RECOVERABLE))
        # R1 records this contrast only.  The first-legal consumer remains
        # unchanged until a later Formal Mission Comparison/admission review.
        source = (ROOT / "scripts/local_subgoal_runner_mvp/navigation_state_machine.py").read_text(encoding="utf-8")
        start = source.index("def room_search_v2_admit_with_l3v_consistency(")
        end = source.index("\n\nROOM_SEARCH_TASK_COMPLETION_REASONS", start)
        self.assertNotIn("recoverability", source[start:end].lower())

    def test_evaluator_is_side_effect_free(self):
        item = terminal()
        pred = predecessor()
        evidence = {"direct_translation": complete(False), "orientation_then_translation": complete(True)}
        before = copy.deepcopy((item, pred, evidence))
        evaluate_retreat(item, pred, evidence)
        self.assertEqual((item, pred, evidence), before)

    def test_accuracy_telemetry_keeps_prediction_actual_and_errors(self):
        telemetry = terminal_accuracy_telemetry(terminal(), (1.3, 0.4, 0.2), "BLOCK_ASTAR_DWA_REACHED_GOAL")
        self.assertEqual(telemetry["candidate_id"], "candidate_a")
        self.assertAlmostEqual(telemetry["position_error_m"], 0.5)
        self.assertFalse(telemetry["physical_accuracy_validated"])

    def test_missing_prediction_telemetry_is_explicit_not_claimed_accurate(self):
        missing = prepare_candidate_terminal_state({"candidate_id": "A", "runner": {}}, 1, EPOCH)
        telemetry = terminal_accuracy_telemetry(missing, (1.0, 0.0, 0.0), "MAX_STEPS")
        self.assertIsNone(telemetry["position_error_m"])
        self.assertFalse(telemetry["physical_accuracy_validated"])

    def test_phase3_and_oa_state_machines_are_not_called_by_pure_evaluator(self):
        source = inspect.getsource(sys.modules["room_search_recoverability"])
        self.assertNotIn("run_runner(", source)
        self.assertNotIn("room_search_observation_arrival_contract", source)
        self.assertNotIn("BlockAStarDwaRunner", source)

    def test_selection_and_room_return_authority_seams_remain_unchanged(self):
        source = (ROOT / "scripts/local_subgoal_runner_mvp/navigation_state_machine.py").read_text(encoding="utf-8")
        start = source.index("def room_search_v2_admit_with_l3v_consistency(")
        end = source.index("\n\nROOM_SEARCH_TASK_COMPLETION_REASONS", start)
        admission = source[start:end]
        self.assertNotIn("recoverability", admission.lower())
        return_start = source.index("def execute_room_return_exit()")
        return_end = source.index("def request_room_return_exit", return_start)
        room_return = source[return_start:return_end]
        self.assertIn("search.choose_return_target", room_return)
        self.assertIn("search.choose_safe_return_transition", room_return)
        self.assertIn('"ROOM_SEARCH_RECOVERABILITY_R1_SHADOW"', source)
        self.assertIn('"authority_enabled": False', source)
        self.assertIn("terminal_accuracy_telemetry(", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
