#!/usr/bin/env python3
"""Focused production-admission tests for Continuation-aware local selection."""

import ast
import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_odom_cache import FakeRos, install_message_stubs


class ContinuationLocalSelectionAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.ros = FakeRos()
        install_message_stubs(self.ros)
        sys.modules.pop("block_astar_dwa_mature_runner", None)
        self.module = importlib.import_module("block_astar_dwa_mature_runner")

    @staticmethod
    def motion(v, w, score):
        return {"v": v, "w": w, "score_eligible": True, "final_dwa_score": score}

    def evidence(self, motions, statuses, *, unscheduled=0):
        return {
            "continuation_status": self.module.CONTINUATION_UNKNOWN,
            "reason": "TEST_EVIDENCE",
            "unscheduled_motion_count": unscheduled,
            "records": [
                {
                    "motion": dict(motion),
                    "continuation_status": status,
                    "reason": "TEST_%s" % status,
                }
                for motion, status in zip(motions, statuses)
            ],
        }

    def decide(self, motions, statuses, *, native_index=0, phase3=False, unscheduled=0):
        native = motions[native_index]
        return self.module.continuation_aware_admissible_selection(
            motions,
            self.evidence(motions, statuses, unscheduled=unscheduled),
            native_v=native["v"], native_w=native["w"],
            phase3_recovery_context=phase3,
        )

    def test_nonviable_native_yields_to_viable_alternative(self):
        motions = [self.motion(0.30, 0.0, 9.0), self.motion(0.30, 0.2, 4.0)]
        result = self.decide(motions, [self.module.CONTINUATION_NON_VIABLE, self.module.CONTINUATION_VIABLE])
        self.assertEqual(result["final_winner"]["candidate_index"], 1)
        self.assertEqual(result["final_winner"]["continuation_status"], self.module.CONTINUATION_VIABLE)
        self.assertEqual(result["alternative_counts"][self.module.CONTINUATION_VIABLE], 1)

    def test_nonviable_native_yields_to_unknown_alternative(self):
        motions = [self.motion(0.30, 0.0, 9.0), self.motion(0.30, 0.2, 4.0)]
        result = self.decide(motions, [self.module.CONTINUATION_NON_VIABLE, self.module.CONTINUATION_UNKNOWN])
        self.assertEqual(result["final_winner"]["candidate_index"], 1)
        self.assertEqual(result["outcome"], "UNVERIFIED_BUT_TRANSLATION_ALLOWED")

    def test_ordinary_room_local_retains_unknown_against_viable(self):
        motions = [self.motion(0.30, 0.0, 9.0), self.motion(0.30, 0.2, 4.0)]
        result = self.decide(motions, [self.module.CONTINUATION_UNKNOWN, self.module.CONTINUATION_VIABLE])
        self.assertEqual(result["admission_policy"], "ORDINARY_ROOM_LOCAL_UNKNOWN_RETAINED")
        self.assertEqual(result["final_winner"]["candidate_index"], 0)
        self.assertEqual(result["outcome"], "UNVERIFIED_BUT_TRANSLATION_ALLOWED")

    def test_phase3_prefers_viable_over_higher_scoring_unknown(self):
        motions = [self.motion(0.30, 0.0, 9.0), self.motion(0.30, 0.2, 4.0)]
        result = self.decide(
            motions, [self.module.CONTINUATION_UNKNOWN, self.module.CONTINUATION_VIABLE], phase3=True,
        )
        self.assertEqual(result["admission_policy"], "PHASE3_VERIFIED_ONLY_WHEN_AVAILABLE")
        self.assertEqual(result["final_winner"]["candidate_index"], 1)
        self.assertEqual(result["outcome"], "VERIFIED_RECOVERY_SUCCESS")

    def test_viable_native_remains_when_alternative_is_nonviable(self):
        motions = [self.motion(0.30, 0.0, 9.0), self.motion(0.30, 0.2, 4.0)]
        result = self.decide(motions, [self.module.CONTINUATION_VIABLE, self.module.CONTINUATION_NON_VIABLE])
        self.assertEqual(result["final_winner"]["candidate_index"], 0)
        self.assertEqual(result["outcome"], "VERIFIED_RECOVERY_SUCCESS")

    def test_complete_all_nonviable_has_no_fallback(self):
        motions = [self.motion(0.30, 0.0, 9.0), self.motion(0.30, 0.2, 4.0)]
        result = self.decide(motions, [self.module.CONTINUATION_NON_VIABLE, self.module.CONTINUATION_NON_VIABLE], phase3=True)
        self.assertFalse(result["translation_allowed"])
        self.assertIsNone(result["final_winner"])
        self.assertEqual(result["terminal_reason"], "ROOM_LOCAL_NO_ADMISSIBLE_CONTINUATION")

    def test_unscheduled_motion_is_unknown_not_false_all_nonviable(self):
        motions = [self.motion(0.30, 0.0, 9.0), self.motion(0.30, 0.2, 4.0)]
        evidence = self.evidence(motions[:1], [self.module.CONTINUATION_NON_VIABLE], unscheduled=1)
        result = self.module.continuation_aware_admissible_selection(
            motions, evidence, native_v=motions[0]["v"], native_w=motions[0]["w"], phase3_recovery_context=True,
        )
        self.assertTrue(result["translation_allowed"])
        self.assertEqual(result["final_winner"]["candidate_index"], 1)
        self.assertEqual(result["final_winner"]["continuation_status"], self.module.CONTINUATION_UNKNOWN)

    def test_equal_score_preserves_native_first_encountered_tie_behavior(self):
        motions = [self.motion(0.30, -0.2, 5.0), self.motion(0.30, 0.2, 5.0)]
        result = self.decide(motions, [self.module.CONTINUATION_VIABLE, self.module.CONTINUATION_VIABLE], native_index=0)
        self.assertEqual(result["final_winner"]["candidate_index"], 0)
        self.assertEqual(result["final_winner_source"], "EXISTING_DWA_SCORE_AND_NATIVE_ORDER")

    def test_final_winner_is_provisional_until_final_command_authority(self):
        motions = [self.motion(0.30, 0.0, 9.0), self.motion(0.30, 0.2, 4.0)]
        runner = object.__new__(self.module.BlockAStarDwaRunner)
        runner.prev_cmd = (motions[0]["v"], motions[0]["w"])
        dwa = {"room_local_productivity_candidates": motions}
        v, w, decision = runner.apply_continuation_aware_local_selection(
            dwa, motions[0]["v"], motions[0]["w"],
            self.evidence(motions, [self.module.CONTINUATION_NON_VIABLE, self.module.CONTINUATION_VIABLE]),
            phase3_recovery_context=True,
        )
        self.assertEqual((v, w), (motions[1]["v"], motions[1]["w"]))
        self.assertEqual(runner.prev_cmd, (motions[0]["v"], motions[0]["w"]))
        self.assertEqual((dwa["selected_linear_x"], dwa["selected_angular_z"]), (v, w))
        self.assertEqual(decision["native_provisional_winner"], {"v": 0.3, "w": 0.0})
        self.assertEqual(dwa["continuation_final_winner"]["authority_scope"], "PROVISIONAL_LOCAL_SELECTION")
        self.assertEqual(
            dwa["continuation_final_winner"]["execution_binding"],
            "PENDING_FINAL_COMMAND_AUTHORITY",
        )

    def test_final_command_authority_binds_unoverridden_continuation_winner(self):
        runner = object.__new__(self.module.BlockAStarDwaRunner)
        runner.prev_cmd = (0.30, 0.0)
        dwa = {
            "continuation_final_winner": {
                "v": 0.30, "w": 0.20,
                "continuation_status": self.module.CONTINUATION_VIABLE,
            },
        }
        action = runner.finalize_command_authority(
            dwa,
            provisional_v=0.30, provisional_w=0.20,
            final_v=0.30, final_w=0.20,
            final_command_reason="CONTINUATION_LOCAL_SELECTION",
            mutation_chain=[],
        )
        self.assertEqual((runner.prev_cmd, dwa["selected_linear_x"], dwa["selected_angular_z"]), ((0.30, 0.20), 0.30, 0.20))
        self.assertTrue(action["provisional_local_motion_executed"])
        self.assertEqual(action["final_command_reason"], "CONTINUATION_LOCAL_SELECTION")
        self.assertEqual(dwa["continuation_final_winner"]["execution_binding"], "EXECUTED_AS_FINAL_COMMAND")
        runner.record_final_command_publish_result(action, 3, execute=True)
        self.assertEqual((action["publish_result"], action["published_count"]), ("PUBLISHED", 3))

    def test_no_progress_override_has_final_command_authority_over_viable_provisional(self):
        runner = object.__new__(self.module.BlockAStarDwaRunner)
        runner.prev_cmd = (0.30, 0.0)
        dwa = {
            "continuation_final_winner": {
                "v": 0.30, "w": 0.20,
                "continuation_status": self.module.CONTINUATION_VIABLE,
            },
        }
        action = runner.finalize_command_authority(
            dwa,
            provisional_v=0.30, provisional_w=0.20,
            final_v=0.0, final_w=-0.60,
            final_command_reason="NO_PROGRESS_RECOVERY",
            mutation_chain=[{
                "reason": "NO_PROGRESS_RECOVERY",
                "before": {"v": 0.30, "w": 0.20},
                "after": {"v": 0.0, "w": -0.60},
            }],
        )
        self.assertEqual((runner.prev_cmd, dwa["selected_linear_x"], dwa["selected_angular_z"]), ((0.0, -0.60), 0.0, -0.60))
        self.assertEqual(action["final_command_reason"], "NO_PROGRESS_RECOVERY")
        self.assertFalse(action["provisional_local_motion_executed"])
        self.assertEqual(action["provisional_local_motion"], {"v": 0.30, "w": 0.20})
        self.assertEqual(dwa["continuation_final_winner"]["continuation_status"], self.module.CONTINUATION_VIABLE)
        self.assertEqual(
            dwa["continuation_final_winner"]["execution_binding"],
            "NOT_EXECUTED_HIGHER_PRIORITY_OVERRIDE",
        )

    def test_phase3_consumed_intent_stays_cleared_when_no_progress_overrides_translation(self):
        runner = object.__new__(self.module.BlockAStarDwaRunner)
        intent = self.module.OrientationIntent(
            ("ROOM_SEARCH", 1, 0), (0.0, 0.0), 0.0, 0.0, (), 0, 0.3, 0.3,
        )
        runner.orientation_intent = intent
        runner.close_phase3_orientation_intent_for_continuation({
            "outcome": "VERIFIED_RECOVERY_SUCCESS", "translation_allowed": True,
        })
        action = runner.finalize_command_authority(
            {},
            provisional_v=0.30, provisional_w=0.20,
            final_v=0.0, final_w=0.60,
            final_command_reason="NO_PROGRESS_RECOVERY",
            mutation_chain=[{"reason": "NO_PROGRESS_RECOVERY"}],
        )
        self.assertTrue(intent.consumed)
        self.assertIsNone(runner.orientation_intent)
        self.assertEqual((action["v"], action["w"]), (0.0, 0.60))

    def test_authority_off_uses_same_final_command_bookkeeping_without_continuation(self):
        runner = object.__new__(self.module.BlockAStarDwaRunner)
        runner.prev_cmd = (0.30, 0.0)
        dwa = {}
        action = runner.finalize_command_authority(
            dwa,
            provisional_v=0.30, provisional_w=0.0,
            final_v=0.0, final_w=0.60,
            final_command_reason="NO_PROGRESS_RECOVERY",
            mutation_chain=[{"reason": "NO_PROGRESS_RECOVERY"}],
        )
        self.assertEqual((runner.prev_cmd, action["v"], action["w"]), ((0.0, 0.60), 0.0, 0.60))
        self.assertNotIn("continuation_final_winner", dwa)
        self.assertFalse(action["provisional_local_motion_executed"])

    def test_next_dynamic_window_uses_final_command_not_provisional_local_motion(self):
        runner = object.__new__(self.module.BlockAStarDwaRunner)
        runner.args = type("Args", (), {
            "command_slice_sec": 0.5,
            "max_linear_accel": 1.0,
            "max_angular_accel": 1.0,
            "max_linear_x": 1.0,
            "max_angular_z": 1.0,
            "min_linear_x": 0.30,
            "linear_samples": 3,
            "angular_samples": 3,
        })()
        runner.prev_cmd = (0.0, -0.60)
        _v_window, w_window = runner.dynamic_window()
        self.assertAlmostEqual(float(w_window[1]), -0.55)

    def test_phase3_intent_is_consumed_and_cleared_for_translation_or_terminal_outcome(self):
        runner = object.__new__(self.module.BlockAStarDwaRunner)
        intent = self.module.OrientationIntent(
            ("ROOM_RETURN", 1, 0), (0.0, 0.0), 0.0, 0.0, (), 0, 0.3, 0.3,
        )
        runner.orientation_intent = intent
        result = runner.close_phase3_orientation_intent_for_continuation({
            "outcome": "UNVERIFIED_BUT_TRANSLATION_ALLOWED", "translation_allowed": True,
        })
        self.assertTrue(result["translation_allowed"])
        self.assertTrue(result["orientation_intent_before"]["consumed"] is False)
        self.assertTrue(intent.consumed)
        self.assertIsNone(runner.orientation_intent)
        terminal = runner.close_phase3_orientation_intent_for_continuation({
            "outcome": "NO_ADMISSIBLE_CONTINUATION", "translation_allowed": False,
        })
        self.assertFalse(terminal["translation_allowed"])
        self.assertIsNone(terminal["orientation_intent_before"])

    def test_authority_guard_is_default_off_and_requires_existing_execute_guard(self):
        disabled = self.module.build_arg_parser().parse_args([])
        self.assertFalse(self.module.continuation_local_selection_authority_is_active(disabled))
        guarded = self.module.build_arg_parser().parse_args([
            "--execute", "--local-control-mode", "ROOM_LOCAL",
            "--room-local-phase2-productive-admission", "phase3_guarded_execute",
            "--room-local-phase3-orientation-recovery", "phase3_guarded_execute",
            "--continuation-local-selection-authority", "guarded_execute",
        ])
        self.module.validate_phase2_productive_admission_activation(guarded)
        self.assertTrue(self.module.continuation_local_selection_authority_is_active(guarded))
        incomplete = self.module.build_arg_parser().parse_args([
            "--local-control-mode", "ROOM_LOCAL",
            "--room-local-phase2-productive-admission", "phase3_guarded_execute",
            "--room-local-phase3-orientation-recovery", "phase3_guarded_execute",
            "--continuation-local-selection-authority", "guarded_execute",
        ])
        with self.assertRaisesRegex(ValueError, "continuation_local_selection_authority_requires_execute"):
            self.module.validate_phase2_productive_admission_activation(incomplete)

    def test_run_freezes_final_command_after_all_overrides_and_before_translation_publish(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        runner_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BlockAStarDwaRunner")
        run_method = next(node for node in runner_class.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        selection_calls = [
            node.lineno for node in ast.walk(run_method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply_continuation_aware_local_selection"
        ]
        translation_publishes = [
            node.lineno for node in ast.walk(run_method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "publish_twist" and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name) and node.args[0].id == "v"
            and isinstance(node.args[1], ast.Name) and node.args[1].id == "w"
        ]
        self.assertEqual(len(selection_calls), 1)
        self.assertEqual(len(translation_publishes), 1)
        finalization_calls = [
            node.lineno for node in ast.walk(run_method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "finalize_command_authority"
        ]
        post_selection_mutations = {
            name: [
                node.lineno for node in ast.walk(run_method)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == name
            ]
            for name in (
                "apply_p_pre_goal_region_motion_guard",
                "apply_scoped_centerline_tracking_correction",
                "apply_imu_heading_hold",
            )
        }
        self.assertEqual(len(finalization_calls), 1)
        self.assertLess(selection_calls[0], finalization_calls[0])
        for calls in post_selection_mutations.values():
            self.assertEqual(len(calls), 1)
            self.assertLess(calls[0], finalization_calls[0])
        self.assertLess(finalization_calls[0], translation_publishes[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
