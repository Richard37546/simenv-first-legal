#!/usr/bin/env python3
"""Focused OA-2/OA-3 tests; no ROS node, simulator, or command is started."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = ROOT / "scripts/local_subgoal_runner_mvp"
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_astar_dwa_mature_runner as runner_module  # noqa: E402
from room_search_v1 import (  # noqa: E402
    evaluate_mission_action_observation,
    freeze_mission_action_spec,
)
from test_odom_cache import FakeRos, load_module  # noqa: E402


def candidate(candidate_id="raw-0057"):
    return {
        "_room_search_audit_candidate_id": candidate_id,
        "target_priority_class": "GENERIC_COVERAGE",
        "target_xy_team_livox_odom": [23.1695456365, -6.8844465755],
        "visible_room_points": [(5.25, 5.75), (5.75, 5.25), (5.75, 5.75)],
        "occlusion_reveal_room_points": [],
        "heading_change_rad": 1.304544283145938,
        "new_observable_cells": 3,
        "cheap_rank": 3,
        "sector": 8,
    }


class OA23PrimitiveTests(unittest.TestCase):
    def test_s1_safe_lattice_slice_moves_toward_frozen_aim(self):
        selection = runner_module.select_observation_orientation_slice(
            (-0.30, 0.0, 0.30), (-0.30, 0.30),
            current_yaw_odom_rad=0.0, aim_yaw_odom_rad=0.50, command_slice_sec=0.50,
        )
        self.assertIsNotNone(selection)
        self.assertEqual(selection["w_radps"], 0.30)
        self.assertGreater(selection["aim_error_improvement_rad"], 0.0)

    def test_s2_unsafe_or_nonimproving_samples_cannot_create_slice(self):
        self.assertIsNone(runner_module.select_observation_orientation_slice(
            (-0.30, 0.0, 0.30), (), current_yaw_odom_rad=0.0, aim_yaw_odom_rad=0.50, command_slice_sec=0.50,
        ))
        self.assertIsNone(runner_module.select_observation_orientation_slice(
            (-0.30, 0.0, 0.30), (-0.30, 0.30), current_yaw_odom_rad=0.0, aim_yaw_odom_rad=0.0, command_slice_sec=0.50,
        ))

    def test_s3_runner_admission_uses_existing_dynamic_and_collision_seam_without_phase3_state(self):
        runner = object.__new__(runner_module.BlockAStarDwaRunner)
        runner.args = runner_module.build_arg_parser().parse_args([
            "--observation-orientation-aim-yaw-odom-rad", "0.5", "--max-angular-z", "0.35",
        ])
        runner.prev_cmd = (0.0, 0.0)
        runner.collision_free_arc = lambda _grid, _blocked, v, w: (v == 0.0 and abs(w) > 0.0, 0.42)
        result = runner.observation_orientation_slice_action(
            pose=(0.0, 0.0, 0.0), grid_msg=object(), blocked=object(),
        )
        self.assertEqual(result["action"], "OBSERVATION_ORIENTATION_ONE_SLICE")
        self.assertFalse(result["phase3_state_touched"])
        self.assertNotIn("orientation_intent", runner.__dict__)

    def test_s4_aim_is_not_completion_authority(self):
        spec = freeze_mission_action_spec(candidate(), (0.0, 0.0, 0.0), "oa23-aim", ())
        result = evaluate_mission_action_observation(
            spec, terminal_evidence_complete=True, actual_visible_cell_ids=(), actual_new_observation_cells=0,
        )
        self.assertEqual(result.observation_intent_status, "OBSERVATION_INTENT_UNSATISFIED")


class OA23ActionAwareAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())
        self.args = self.module.build_arg_parser().parse_args([])

    def test_s5_near_position_valid_intent_uses_only_action_aware_admission(self):
        current = (23.1695456365 - 0.075, -6.8844465755 - 0.275, 0.0)
        result = self.module.room_search_v2_action_aware_preflight(self.args, candidate(), current, ())
        self.assertTrue(result["legal"])
        self.assertEqual(result["action_aware_admission"], "POSITION_SATISFIED_OBSERVATION_ACTION")
        self.assertLessEqual(result["position_distance_m"], self.args.goal_tolerance_m)
        self.assertEqual(result["runner"]["runner_final_decision"], "POSITION_SATISFIED_NO_TRANSLATION_PREFLIGHT")

    def test_s6_near_position_without_intent_does_not_use_action_aware_branch(self):
        bad = candidate()
        bad["visible_room_points"] = []
        current = (23.1695456365 - 0.075, -6.8844465755 - 0.275, 0.0)
        original = self.module.room_search_v2_preflight
        self.module.room_search_v2_preflight = lambda *_args: {"legal": False, "runner": {}}
        try:
            result = self.module.room_search_v2_action_aware_preflight(self.args, bad, current, ())
        finally:
            self.module.room_search_v2_preflight = original
        self.assertFalse(result["legal"])
        self.assertNotIn("action_aware_admission", result)

    def test_s7_far_position_preserves_translation_preflight(self):
        original = self.module.room_search_v2_preflight
        sentinel = {"legal": True, "runner": {"runner_final_decision": "TRANSLATION_PREFLIGHT"}}
        self.module.room_search_v2_preflight = lambda *_args: sentinel
        try:
            result = self.module.room_search_v2_action_aware_preflight(self.args, candidate(), (0.0, 0.0, 0.0), ())
        finally:
            self.module.room_search_v2_preflight = original
        self.assertIs(result, sentinel)

    def test_s8_d16_raw_0057_and_raw_0054_remove_only_moving_proof_mismatch(self):
        current = (23.1695456365 - 0.075, -6.8844465755 - 0.275, 0.0)
        raw_0057 = self.module.room_search_v2_action_aware_preflight(self.args, candidate("raw-0057"), current, ())
        raw_0054 = candidate("raw-0054")
        raw_0054["visible_room_points"] = [(5.75, 5.25), (5.75, 5.75)]
        raw_0054["heading_change_rad"] = 1.8370483401830382
        raw_0054["new_observable_cells"] = 2
        raw_0054_result = self.module.room_search_v2_action_aware_preflight(self.args, raw_0054, current, ())
        self.assertTrue(raw_0057["position_satisfied"])
        self.assertTrue(raw_0054_result["position_satisfied"])
        self.assertEqual(raw_0057["action_aware_admission"], "POSITION_SATISFIED_OBSERVATION_ACTION")
        self.assertEqual(raw_0054_result["action_aware_admission"], "POSITION_SATISFIED_OBSERVATION_ACTION")


class OA23SourceIsolationTests(unittest.TestCase):
    def test_s9_one_slice_runner_mode_has_one_nonzero_publish_site_and_post_inputs(self):
        source = (MODULE_ROOT / "block_astar_dwa_mature_runner.py").read_text(encoding="utf-8")
        start = source.index("def run_observation_orientation(")
        end = source.index("    def run(self)", start)
        body = source[start:end]
        self.assertEqual(body.count("self.publish_twist(0.0, w"), 1)
        self.assertIn("post_odom = rospy.wait_for_message", body)
        self.assertIn("post_grid = rospy.wait_for_message", body)
        self.assertIn("post_status_msg = rospy.wait_for_message", body)
        self.assertNotIn("self.orientation_intent", body)
        self.assertNotIn("phase3_recovery_action(", body)

    def test_s10_state_machine_calls_orientation_only_after_unsatisfied_position_handoff(self):
        source = (MODULE_ROOT / "navigation_state_machine.py").read_text(encoding="utf-8")
        trigger = source.index('position_satisfied_for_action\n            and terminal_evidence["mission_observation"].observation_intent_status == "OBSERVATION_INTENT_UNSATISFIED"')
        call = source.index("orientation_attempt = execute_observation_orientation_slice", trigger)
        self.assertLess(trigger, call)
        self.assertIn("if position_satisfied_for_action:", source)
        self.assertNotIn("observation_orientation_aim_yaw_odom_rad", source[source.index("def execute_room_return_exit"):])

    def test_s11_transit_command_has_no_observation_orientation_flag(self):
        module = load_module(FakeRos())
        args = module.build_arg_parser().parse_args([])
        command = module.runner_cmd(args, state="FOLLOW_CORRIDOR", runtime_sec=1.0, max_steps=1)
        self.assertNotIn("--observation-orientation-aim-yaw-odom-rad", command)

    def test_s12_completion_is_evaluated_before_the_single_seen_update(self):
        source = (MODULE_ROOT / "navigation_state_machine.py").read_text(encoding="utf-8")
        evidence = source.index("terminal_evidence = mission_action_terminal_evidence(mission_action_spec, candidate)")
        update = source.index("actual_new_observation_cells = search.update_actual_view(terminal_pose, actual_visible)", evidence)
        self.assertLess(evidence, update)
        helper_start = source.index("def mission_action_terminal_evidence(")
        helper_end = source.index("    def finish(", helper_start)
        helper = source[helper_start:helper_end]
        self.assertIn("actual_new = search.observation.new_count(actual_visible)", helper)
        self.assertIn("mission_observation = evaluate_mission_action_observation(", helper)
        self.assertNotIn("update_actual_view", helper)

    def test_s13_orientation_mode_never_selects_forward_or_phase3_recovery(self):
        source = (MODULE_ROOT / "block_astar_dwa_mature_runner.py").read_text(encoding="utf-8")
        start = source.index("def run_observation_orientation(")
        end = source.index("    def run(self)", start)
        body = source[start:end]
        self.assertIn("self.publish_twist(0.0, w", body)
        self.assertNotIn("self.publish_twist(v,", body)
        self.assertNotIn("phase3_recovery_action(", body)
        self.assertNotIn("build_recovery_set(", body)

    def test_s14_post_slice_visibility_uses_a_new_grid_status_acquisition(self):
        source = (MODULE_ROOT / "navigation_state_machine.py").read_text(encoding="utf-8")
        helper_start = source.index("def mission_action_terminal_evidence(")
        helper_end = source.index("    def finish(", helper_start)
        helper = source[helper_start:helper_end]
        self.assertIn("room_search_v2_fresh_planning_context(args, grid_status_pairs)", helper)
        self.assertIn("require_fresh_grid_status=True", source)
        fresh_start = source.index("def room_search_v2_fresh_planning_context(")
        fresh_end = source.index("def room_search_v2_context_record(", fresh_start)
        fresh = source[fresh_start:fresh_end]
        self.assertIn("grid_msg = read_grid(args.input_timeout_sec)", fresh)
        self.assertIn("status = read_traversability_status(args.input_timeout_sec)", fresh)


if __name__ == "__main__":
    unittest.main(verbosity=2)
