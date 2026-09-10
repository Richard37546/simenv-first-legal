#!/usr/bin/env python3
"""Phase-1 ROOM_LOCAL foundations and TRANSIT isolation tests."""

import math
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

import block_astar_dwa_mature_runner as runner_module
from test_odom_cache import FakeRos, load_module


class RoomLocalPhase1Tests(unittest.TestCase):
    def capability(self):
        return runner_module.room_local_capability_by_id("forward_turn_current_room_profile")

    def make_transit_runner(self):
        args = runner_module.build_arg_parser().parse_args(
            [
                "--max-linear-x", "0.60", "--max-angular-z", "0.35",
                "--max-angular-accel", "0.50", "--min-linear-x", "0.30",
                "--enforce-min-forward-speed", "--dwa-predict-time", "1.0",
                "--max-target-heading-correction-rad", "0.85",
                "--target-heading-blend-weight", "0.90",
                "--target-lateral-correction-angular-z", "0.24",
                "--target-lateral-correction-weight", "0.32",
            ]
        )
        runner = object.__new__(runner_module.BlockAStarDwaRunner)
        runner.args = args
        runner.prev_cmd = (0.30, 0.245)
        runner.dynamic_window = lambda: (
            np.array([0.30, 0.60]), np.array([0.245, 0.35]),
        )
        runner.collision_free_arc = lambda *_args: (True, 0.50)
        return runner

    @staticmethod
    def anchor(path, *, robot=(0.0, 0.0), anchor_segment=0, local_window=8.0, rollout_step=0.03):
        result = runner_module.build_path_progress_anchor(
            path,
            path_identity="phase1-test",
            robot_xy=robot,
            anchor_segment_index=anchor_segment,
            local_window_arc_m=local_window,
            grid_resolution_m=0.05,
            rollout_step_m=rollout_step,
        )
        assert result is not None
        return result

    def test_a_default_mode_is_transit(self):
        args = runner_module.build_arg_parser().parse_args([])
        self.assertEqual(args.local_control_mode, runner_module.LOCAL_CONTROL_MODE_TRANSIT)

    def test_b_explicit_room_local_mode_resolves(self):
        args = runner_module.build_arg_parser().parse_args(["--local-control-mode", "ROOM_LOCAL"])
        self.assertEqual(args.local_control_mode, runner_module.LOCAL_CONTROL_MODE_ROOM_LOCAL)

    def test_b2_malformed_mode_fails_fast(self):
        with self.assertRaises(SystemExit):
            runner_module.build_arg_parser().parse_args(["--local-control-mode", "IMPLICIT"])

    def test_c_supported_curvature_maps_to_dynamic_primitive(self):
        primitive, reason = runner_module.room_local_curvature_primitive(
            0.30, 0.5, self.capability(), (0.30, 0.60), (-0.35, 0.35),
        )
        self.assertEqual(reason, "QUALIFIED")
        self.assertIsNotNone(primitive)
        self.assertAlmostEqual(primitive.w_radps, 0.15)

    def test_d_unsupported_capability_is_rejected(self):
        primitive, reason = runner_module.room_local_curvature_primitive(
            0.20, 0.5, self.capability(), (0.0, 0.60), (-0.35, 0.35),
        )
        self.assertIsNone(primitive)
        self.assertEqual(reason, "CAPABILITY_UNQUALIFIED")

    def test_e_dynamic_w_window_rejects_mapped_command(self):
        primitive, reason = runner_module.room_local_curvature_primitive(
            0.30, 1.0, self.capability(), (0.30, 0.60), (-0.20, 0.20),
        )
        self.assertIsNone(primitive)
        self.assertEqual(reason, "DYNAMIC_W_WINDOW_REJECTED")

    def test_f_capability_w_bound_rejects_mapped_command(self):
        primitive, reason = runner_module.room_local_curvature_primitive(
            0.30, 2.0, self.capability(), (0.30, 0.60), (-0.60, 0.60),
        )
        self.assertIsNone(primitive)
        self.assertEqual(reason, "CAPABILITY_UNQUALIFIED")

    def test_g_straight_path_station_and_progress_are_positive(self):
        path = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
        anchor = self.anchor(path)
        projection = runner_module.project_endpoint_to_path_station(path, anchor, (0.30, 0.0), rollout_arc_length_m=0.30)
        evidence = runner_module.evaluate_candidate_progress(
            anchor, projection, current_target_xy=(1.0, 0.0), current_local_xy=(0.8, 0.0), endpoint_xy=(0.30, 0.0),
        )
        self.assertTrue(projection.valid)
        self.assertAlmostEqual(evidence.p_path_m, 0.30)
        self.assertGreater(evidence.p_target_m, 0.0)
        self.assertGreater(evidence.p_local_m, 0.0)

    def test_h_backward_endpoint_has_negative_path_progress(self):
        path = ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))
        anchor = self.anchor(path, robot=(0.0, 0.0), anchor_segment=1, rollout_step=0.04)
        projection = runner_module.project_endpoint_to_path_station(path, anchor, (-0.04, 0.0), rollout_arc_length_m=0.04)
        evidence = runner_module.evaluate_candidate_progress(
            anchor, projection, current_target_xy=(0.8, 0.0), current_local_xy=(0.5, 0.0), endpoint_xy=(-0.04, 0.0),
        )
        self.assertTrue(projection.valid)
        self.assertLess(evidence.p_path_m, 0.0)

    def test_i_corner_can_project_to_next_ordered_segment(self):
        path = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
        anchor = self.anchor(path)
        projection = runner_module.project_endpoint_to_path_station(path, anchor, (1.0, 0.25), rollout_arc_length_m=1.25)
        self.assertTrue(projection.valid)
        self.assertAlmostEqual(projection.station_m, 1.25)

    def test_j_self_near_future_segment_cannot_claim_jump(self):
        path = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.02), (1.0, 0.02))
        anchor = self.anchor(path)
        projection = runner_module.project_endpoint_to_path_station(path, anchor, (0.20, 0.02), rollout_arc_length_m=0.20)
        self.assertTrue(projection.valid)
        self.assertLessEqual(projection.station_m, anchor.s_current_m + 0.20 + anchor.projection_tolerance_m)

    def test_k_parallel_future_segment_cannot_steal_projection(self):
        path = ((0.0, 0.0), (1.0, 0.0), (1.0, 0.50), (0.0, 0.50), (0.0, 0.05), (1.0, 0.05))
        anchor = self.anchor(path)
        projection = runner_module.project_endpoint_to_path_station(path, anchor, (0.60, 0.05), rollout_arc_length_m=0.60)
        self.assertTrue(projection.valid)
        self.assertLess(projection.station_m, 1.0)

    def test_l_rollout_arc_upper_bound_rejects_unreachable_station(self):
        path = ((0.0, 0.0), (2.0, 0.0))
        anchor = self.anchor(path)
        projection = runner_module.project_endpoint_to_path_station(path, anchor, (1.50, 0.0), rollout_arc_length_m=0.20)
        self.assertFalse(projection.valid)
        self.assertEqual(projection.reason, "NO_ORDERED_ARC_REACHABLE_SEGMENT")

    def test_m_target_and_local_progress_values_are_exact(self):
        path = ((0.0, 0.0), (1.0, 0.0))
        anchor = self.anchor(path)
        projection = runner_module.project_endpoint_to_path_station(path, anchor, (0.20, 0.0), rollout_arc_length_m=0.20)
        evidence = runner_module.evaluate_candidate_progress(
            anchor, projection, current_target_xy=(1.0, 0.0), current_local_xy=(0.5, 0.0), endpoint_xy=(0.20, 0.0),
        )
        self.assertAlmostEqual(evidence.p_target_m, 0.20)
        self.assertAlmostEqual(evidence.p_local_m, 0.20)

    def test_n_run0150_step_0_progress_helper_ranks_tighter_endpoint_higher(self):
        path = ((0.0, 0.0), (0.225, 0.625), (0.45, 1.25))
        anchor = self.anchor(path, local_window=2.0)
        target = (0.229823, 0.609249)
        tighter = runner_module.evaluate_candidate_progress(
            anchor,
            runner_module.project_endpoint_to_path_station(path, anchor, (0.12, 0.30), rollout_arc_length_m=0.33),
            current_target_xy=target, current_local_xy=(0.225, 0.625), endpoint_xy=(0.12, 0.30),
        )
        wider = runner_module.evaluate_candidate_progress(
            anchor,
            runner_module.project_endpoint_to_path_station(path, anchor, (0.08, 0.20), rollout_arc_length_m=0.24),
            current_target_xy=target, current_local_xy=(0.225, 0.625), endpoint_xy=(0.08, 0.20),
        )
        self.assertGreater(tighter.p_target_m, wider.p_target_m)
        self.assertGreater(tighter.p_local_m, wider.p_local_m)

    def test_o_run0150_step_2_helper_identifies_regressive_forward_endpoint(self):
        path = ((0.0, 0.0), (0.04, 0.56), (0.08, 1.12))
        anchor = self.anchor(path, local_window=1.5)
        target = (0.041662, 0.556276)
        endpoint = (0.30, 0.0)
        evidence = runner_module.evaluate_candidate_progress(
            anchor,
            runner_module.project_endpoint_to_path_station(path, anchor, endpoint, rollout_arc_length_m=0.30),
            current_target_xy=target, current_local_xy=target, endpoint_xy=endpoint,
        )
        self.assertLess(evidence.p_target_m, 0.0)
        self.assertLess(evidence.p_local_m, 0.0)

    def test_p_transit_winner_and_failure_fixture_remain_legacy(self):
        runner = self.make_transit_runner()
        with mock.patch.object(runner_module, "classify_room_local_productivity", side_effect=AssertionError("ROOM_LOCAL gate called")):
            v, w, detail = runner.choose_dwa(
                None, np.zeros((1, 1), dtype=bool), (0.225, 0.625), (0.229823, 0.609249), 0.651,
                None, target_in_front=True, astar_path_exists=True,
            )
        self.assertEqual(runner.args.local_control_mode, "TRANSIT")
        self.assertAlmostEqual(v, 0.30)
        self.assertAlmostEqual(w, 0.245)
        self.assertFalse(detail["blocked"])
        runner.collision_free_arc = lambda *_args: (False, 0.0)
        blocked_v, blocked_w, blocked = runner.choose_dwa(
            None, np.zeros((1, 1), dtype=bool), (0.225, 0.625), (0.229823, 0.609249), 0.651,
            None, target_in_front=True, astar_path_exists=True,
        )
        self.assertEqual((blocked_v, blocked_w), (0.0, 0.0))
        self.assertTrue(blocked["blocked"])

    def test_q_state_machine_mode_wiring_is_explicit_and_transit_default_is_unchanged(self):
        module = load_module(FakeRos())
        args = module.build_arg_parser().parse_args([])
        transit = module.runner_cmd(args, state="FOLLOW_CORRIDOR", runtime_sec=1.0, max_steps=1)
        room_local = module.runner_cmd(
            args, state="ROOM_SEARCH", runtime_sec=1.0, max_steps=1, local_control_mode="ROOM_LOCAL",
        )
        self.assertNotIn("--local-control-mode", transit)
        mode_index = room_local.index("--local-control-mode")
        self.assertEqual(room_local[mode_index + 1], "ROOM_LOCAL")
        with self.assertRaises(ValueError):
            module.runner_cmd(args, state="ROOM_SEARCH", runtime_sec=1.0, max_steps=1, local_control_mode="IMPLICIT")


class RoomLocalPhase2Tests(unittest.TestCase):
    @staticmethod
    def evidence(p_path, p_target, p_local, reason="ORDERED_ARC_REACHABLE"):
        return runner_module.ProgressEvidence(
            p_path_m=p_path,
            p_target_m=p_target,
            p_local_m=p_local,
            projected_station_m=1.0 if p_path is not None else None,
            endpoint_target_distance_m=0.0,
            endpoint_local_distance_m=0.0,
            cross_track_m=0.0 if p_path is not None else None,
            projection_reason=reason,
        )

    @staticmethod
    def active_runner(endpoint=(0.30, 0.0), velocities=(0.30, 0.60), angular=(0.0, 0.35)):
        args = runner_module.build_arg_parser().parse_args([
            "--local-control-mode", "ROOM_LOCAL",
            "--room-local-phase2-productive-admission", "offline_frozen",
            "--max-linear-x", "0.60", "--max-angular-z", "0.35",
            "--min-linear-x", "0.30", "--enforce-min-forward-speed",
            "--dwa-predict-time", "1.0", "--dwa-dt", "0.10",
        ])
        runner = object.__new__(runner_module.BlockAStarDwaRunner)
        runner.args = args
        runner.prev_cmd = (0.0, 0.0)
        runner.phase2_productive_admission_active = True
        runner.dynamic_window = lambda: (np.array(velocities), np.array(angular))

        def collision(_grid, _blocked, _v, _w, trace=None):
            if trace is not None:
                trace["endpoint_base_xy"] = [float(endpoint[0]), float(endpoint[1])]
                trace["endpoint_base_yaw_rad"] = 0.0
            return True, 0.50

        runner.collision_free_arc = collision
        return runner

    def classify(self, evidence, *, direct=False, margin=0.025):
        return runner_module.classify_room_local_productivity(
            evidence, direct_or_short_context=direct, meaningful_progress_margin_m=margin,
        )

    def test_01_margin_is_derived_from_grid_and_minimum_rollout_step(self):
        self.assertAlmostEqual(runner_module.derive_meaningful_progress_margin(0.05, 0.30, 0.10), 0.025)
        self.assertAlmostEqual(runner_module.derive_meaningful_progress_margin(0.02, 0.60, 0.10), 0.030)
        self.assertIsNone(runner_module.derive_meaningful_progress_margin(0.0, 0.30, 0.10))

    def test_02_valid_path_progress_is_admitted(self):
        result = self.classify(self.evidence(0.10, -0.01, -0.01))
        self.assertEqual(result.classification, "PATH_PRODUCTIVE")
        self.assertTrue(result.score_eligible)

    def test_03_all_negative_safe_translation_is_rejected(self):
        result = self.classify(self.evidence(-0.10, -0.03, -0.04, "CROSS_TRACK_INCONSISTENT"))
        self.assertEqual(result.classification, "SAFE_BUT_NONPRODUCTIVE_TRANSLATION")
        self.assertFalse(result.score_eligible)

    def test_04_path_detour_survives_slightly_negative_target_progress(self):
        result = self.classify(self.evidence(0.08, -0.01, -0.02))
        self.assertTrue(result.score_eligible)

    def test_05_short_direct_target_progress_is_admitted(self):
        result = self.classify(self.evidence(None, 0.04, 0.03, "PATH_CONTEXT_UNAVAILABLE"), direct=True)
        self.assertEqual(result.classification, "DIRECT_OR_SHORT_PRODUCTIVE")
        self.assertTrue(result.score_eligible)

    def test_06_invalid_projection_cannot_create_path_productivity(self):
        result = self.classify(self.evidence(0.20, -0.03, -0.03, "CROSS_TRACK_INCONSISTENT"))
        self.assertFalse(result.score_eligible)
        self.assertNotEqual(result.classification, "PATH_PRODUCTIVE")

    def test_07_speed_score_cannot_revive_rejected_candidate(self):
        runner = self.active_runner(endpoint=(-0.30, 0.0), velocities=(0.30, 0.60), angular=(0.0,))
        v, w, detail = runner.choose_dwa(
            None, np.zeros((1, 1), dtype=bool), (0.80, 0.0), (1.00, 0.0), 1.0,
            room_local_path_xy=((0.0, 0.0), (1.0, 0.0)),
        )
        self.assertEqual((v, w), (0.0, 0.0))
        self.assertTrue(detail["blocked"])
        self.assertEqual(detail["room_local_productivity_set_status"], "NO_PRODUCTIVE_TRANSLATION")
        self.assertEqual(detail["sample_count"], 0)

    def test_08_angular_preference_cannot_revive_rejected_candidate(self):
        runner = self.active_runner(endpoint=(-0.30, 0.0), angular=(0.0, 0.35))
        _, _, detail = runner.choose_dwa(
            None, np.zeros((1, 1), dtype=bool), (0.80, 0.0), (1.00, 0.0), 1.0,
            room_local_path_xy=((0.0, 0.0), (1.0, 0.0)),
        )
        self.assertEqual(detail["room_local_productive_translational_candidate_count"], 0)
        self.assertTrue(all(not record["score_eligible"] for record in detail["room_local_productivity_candidates"]))

    def test_09_productive_set_empty_has_no_silent_legacy_fallback(self):
        runner = self.active_runner(endpoint=(-0.30, 0.0))
        _, _, detail = runner.choose_dwa(
            None, np.zeros((1, 1), dtype=bool), (0.80, 0.0), (1.00, 0.0), 1.0,
            room_local_path_xy=((0.0, 0.0), (1.0, 0.0)),
        )
        self.assertEqual(detail["room_local_safe_translational_candidate_count"], 4)
        self.assertEqual(detail["room_local_productivity_set_status"], "NO_PRODUCTIVE_TRANSLATION")
        self.assertEqual(detail["selected_linear_x"], 0.0)

    def test_10_corner_path_progress_is_productive(self):
        result = self.classify(self.evidence(0.25, -0.02, -0.01))
        self.assertTrue(result.score_eligible)

    def test_11_self_near_projection_jump_remains_nonproductive(self):
        result = self.classify(self.evidence(None, -0.02, -0.02, "NO_ORDERED_ARC_REACHABLE_SEGMENT"))
        self.assertFalse(result.score_eligible)

    def test_12_run0150_step_0_eligibility(self):
        self.assertTrue(self.classify(self.evidence(None, 0.062616, 0.059255, "PATH_CONTEXT_UNAVAILABLE"), direct=True).score_eligible)

    def test_13_run0150_step_1_high_curvature_candidate_remains_eligible(self):
        self.assertTrue(self.classify(self.evidence(None, 0.046568, 0.081887, "PATH_CONTEXT_UNAVAILABLE"), direct=True).score_eligible)

    def test_14_run0150_step_2_forward_candidates_are_rejected(self):
        result = self.classify(self.evidence(None, -0.011044, -0.012530, "PATH_CONTEXT_UNAVAILABLE"), direct=True)
        self.assertEqual(result.classification, "SAFE_BUT_NONPRODUCTIVE_TRANSLATION")

    def test_15_run0150_step_3_forward_candidates_are_rejected(self):
        result = self.classify(self.evidence(None, -0.052909, -0.012530, "PATH_CONTEXT_UNAVAILABLE"), direct=True)
        self.assertEqual(result.classification, "SAFE_BUT_NONPRODUCTIVE_TRANSLATION")

    def test_16_rule_is_source_independent_for_room_return_style_target(self):
        runner = self.active_runner(endpoint=(0.30, 0.0), velocities=(0.30,), angular=(0.0,))
        v, _, detail = runner.choose_dwa(
            None, np.zeros((1, 1), dtype=bool), (0.70, 0.0), (0.90, 0.0), 0.9,
            room_local_path_xy=((0.0, 0.0), (1.0, 0.0)),
        )
        self.assertEqual(v, 0.30)
        self.assertEqual(detail["room_local_productivity_set_status"], "PRODUCTIVE_TRANSLATIONAL_SET_NONEMPTY")

    def test_17_offline_activation_disallows_execute(self):
        args = runner_module.build_arg_parser().parse_args([
            "--execute", "--local-control-mode", "ROOM_LOCAL",
            "--room-local-phase2-productive-admission", "offline_frozen",
        ])
        with self.assertRaisesRegex(ValueError, "offline_frozen_disallows_execute"):
            runner_module.validate_phase2_productive_admission_activation(args)

    def test_18_transit_never_activates_the_phase2_gate(self):
        args = runner_module.build_arg_parser().parse_args([
            "--local-control-mode", "TRANSIT",
            "--room-local-phase2-productive-admission", "offline_frozen",
        ])
        self.assertFalse(runner_module.phase2_productive_admission_is_active(args))


class RoomLocalPhase3Tests(unittest.TestCase):
    def args(self, *extra):
        return runner_module.build_arg_parser().parse_args([
            "--local-control-mode", "ROOM_LOCAL",
            "--room-local-phase2-productive-admission", "offline_frozen",
            "--room-local-phase3-orientation-recovery", "offline_frozen",
            "--grid-resolution-m", "0.05", "--command-slice-sec", "0.10",
            "--dwa-predict-time", "1.0", "--max-angular-z", "0.40", *extra,
        ])

    def recovery_runner(self, recovery_set):
        runner = object.__new__(runner_module.BlockAStarDwaRunner)
        runner.args = self.args()
        runner.orientation_intent = None
        runner.dynamic_window = lambda: (np.array([0.0, 0.30]), np.array([0.0, 0.40]))
        runner.collision_free_arc = lambda *_args: (True, 0.50)
        runner.build_recovery_set = lambda *_args: recovery_set
        runner.orientation_route_context = lambda _target, _pose, _path: (("ROOM_RETURN", 20, 0), 0.0)
        return runner

    @staticmethod
    def recovery_call(runner, pose=(0.0, 0.0, 0.0), target=None):
        return runner.phase3_recovery_action(
            target=target or {"source": "ROOM_RETURN", "target_xy_team_livox_odom": [1.0, 0.0]},
            pose=pose, grid_msg=None, blocked=np.zeros((1, 1), dtype=bool),
            waypoint_xy=(0.8, 0.0), target_base_xy=(1.0, 0.0), distance_to_goal=1.0,
            room_local_path_xy=((0.0, 0.0), (1.0, 0.0)),
        )

    def test_01_phase3_is_transit_isolated(self):
        args = runner_module.build_arg_parser().parse_args([
            "--local-control-mode", "TRANSIT",
            "--room-local-phase2-productive-admission", "offline_frozen",
            "--room-local-phase3-orientation-recovery", "offline_frozen",
        ])
        self.assertFalse(runner_module.phase3_orientation_recovery_is_active(args))

    def test_02_angular_margin_comes_from_current_lattice_and_slice(self):
        self.assertAlmostEqual(runner_module.derive_angular_recovery_margin((-0.4, 0.0, 0.4), 0.1), 0.02)
        self.assertIsNone(runner_module.derive_angular_recovery_margin((0.0,), 0.1))

    def test_03_successful_adjacent_hypotheses_merge_to_one_interval(self):
        intervals = runner_module.recovery_intervals_from_hypotheses(
            ((-0.4, False), (0.0, True), (0.4, True), (0.8, False)), 0.02,
        )
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0].center_rad, 0.2)
        self.assertAlmostEqual(intervals[0].half_width_rad, 0.4)

    def test_04_recovery_distance_handles_pi_wrap(self):
        interval = runner_module.CircularYawInterval(-math.pi + 0.03, 0.05)
        self.assertAlmostEqual(runner_module.recovery_distance(math.pi - 0.02, (interval,)), 0.0)

    def test_05_recovery_distance_uses_union_of_multiple_intervals(self):
        intervals = (
            runner_module.CircularYawInterval(-0.5, 0.05),
            runner_module.CircularYawInterval(0.4, 0.05),
        )
        self.assertAlmostEqual(runner_module.recovery_distance(0.30, intervals), 0.05)

    def test_06_selection_moves_toward_recovery_set(self):
        result = runner_module.select_orientation_slice(
            (0.0, 0.4), (0.4,), (runner_module.CircularYawInterval(0.4, 0.01),),
            current_relative_yaw_rad=0.0, command_slice_sec=0.1, angular_margin_rad=0.02,
        )
        self.assertEqual(result["w_radps"], 0.4)
        self.assertGreater(result["improvement_rad"], 0.02)

    def test_07_selection_rejects_away_from_recovery_set(self):
        result = runner_module.select_orientation_slice(
            (-0.4, 0.0), (-0.4,), (runner_module.CircularYawInterval(0.4, 0.01),),
            current_relative_yaw_rad=0.0, command_slice_sec=0.1, angular_margin_rad=0.02,
        )
        self.assertIsNone(result)

    def test_08_selection_requires_collision_safe_w(self):
        result = runner_module.select_orientation_slice(
            (0.0, 0.4), (), (runner_module.CircularYawInterval(0.4, 0.01),),
            current_relative_yaw_rad=0.0, command_slice_sec=0.1, angular_margin_rad=0.02,
        )
        self.assertIsNone(result)

    def test_09_phase3_offline_frozen_disallows_execute(self):
        args = runner_module.build_arg_parser().parse_args([
            "--execute", "--local-control-mode", "ROOM_LOCAL",
            "--room-local-phase2-productive-admission", "phase3_guarded_execute",
            "--room-local-phase3-orientation-recovery", "offline_frozen",
        ])
        with self.assertRaisesRegex(ValueError, "phase3_orientation_recovery_offline_frozen_disallows_execute"):
            runner_module.validate_phase2_productive_admission_activation(args)

    def test_10_guarded_execute_requires_both_explicit_flags(self):
        args = runner_module.build_arg_parser().parse_args([
            "--execute", "--local-control-mode", "ROOM_LOCAL",
            "--room-local-phase2-productive-admission", "phase3_guarded_execute",
            "--room-local-phase3-orientation-recovery", "phase3_guarded_execute",
        ])
        runner_module.validate_phase2_productive_admission_activation(args)
        self.assertTrue(runner_module.phase3_orientation_recovery_is_active(args))

    def test_11_phase3_guarded_execute_cannot_be_enabled_alone(self):
        args = runner_module.build_arg_parser().parse_args([
            "--local-control-mode", "ROOM_LOCAL",
            "--room-local-phase3-orientation-recovery", "phase3_guarded_execute",
        ])
        with self.assertRaisesRegex(ValueError, "requires_phase2_productive_admission"):
            runner_module.validate_phase2_productive_admission_activation(args)

    def test_12_orientation_intent_ignores_epoch_only_replan(self):
        runner = self.recovery_runner(runner_module.RecoverySet((), None, (), True))
        intent = runner_module.OrientationIntent(("ROOM_RETURN", 20, 0), (0.0, 0.0), 0.0, 0.0, (), 0, 0.3, 0.3)
        self.assertFalse(runner.orientation_intent_materially_changed(intent, intent.target_key, (0.0, 0.0, 0.0), 0.0, 0.02))

    def test_13_orientation_intent_rearms_on_target_anchor_or_tangent_change(self):
        runner = self.recovery_runner(runner_module.RecoverySet((), None, (), True))
        intent = runner_module.OrientationIntent(("ROOM_RETURN", 20, 0), (0.0, 0.0), 0.0, 0.0, (), 0, 0.3, 0.3)
        self.assertTrue(runner.orientation_intent_materially_changed(intent, ("ROOM_RETURN", 21, 0), (0.0, 0.0, 0.0), 0.0, 0.02))
        self.assertTrue(runner.orientation_intent_materially_changed(intent, intent.target_key, (0.08, 0.0, 0.0), 0.0, 0.02))
        self.assertTrue(runner.orientation_intent_materially_changed(intent, intent.target_key, (0.0, 0.0, 0.0), 0.04, 0.02))

    def test_14_recoveryset_empty_is_truthful_failure(self):
        runner = self.recovery_runner(runner_module.RecoverySet((), 0.02, (), True))
        result = self.recovery_call(runner)
        self.assertEqual(result["action"], "FAIL")
        self.assertEqual(result["reason"], "RECOVERYSET_EMPTY_OR_UNRESOLVED")

    def test_15_one_safe_slice_then_fresh_replan_is_required(self):
        recovery = runner_module.RecoverySet((runner_module.CircularYawInterval(0.4, 0.01),), 0.02, (), True)
        runner = self.recovery_runner(recovery)
        result = self.recovery_call(runner)
        self.assertEqual(result["action"], "ORIENTATION_SLICE")
        self.assertEqual(result["selection"]["w_radps"], 0.4)
        self.assertTrue(runner.orientation_intent.awaiting_fresh_replan)

    def test_16_reached_recoveryset_without_new_translation_fails(self):
        recovery = runner_module.RecoverySet((runner_module.CircularYawInterval(0.4, 0.01),), 0.02, (), True)
        runner = self.recovery_runner(recovery)
        self.assertEqual(self.recovery_call(runner)["action"], "ORIENTATION_SLICE")
        result = self.recovery_call(runner, pose=(0.0, 0.0, 0.4))
        self.assertEqual(result["action"], "FAIL")
        self.assertEqual(result["reason"], "RECOVERYSET_REACHED_TRANSLATION_STILL_EMPTY")

    def test_17_consumed_intent_cannot_silently_rearm(self):
        recovery = runner_module.RecoverySet((runner_module.CircularYawInterval(0.4, 0.01),), 0.02, (), True)
        runner = self.recovery_runner(recovery)
        intent = runner_module.OrientationIntent(("ROOM_RETURN", 20, 0), (0.0, 0.0), 0.0, 0.0, recovery.intervals, 0, 0.3, 0.3, consumed=True)
        runner.orientation_intent = intent
        result = self.recovery_call(runner)
        self.assertEqual(result["reason"], "ORIENTATION_INTENT_ALREADY_CONSUMED")

    def test_18_recoveryset_builder_marks_counterfactual_approximate(self):
        runner = self.recovery_runner(runner_module.RecoverySet((), None, (), True))
        del runner.build_recovery_set
        runner.args.dwa_predict_time = 1.0
        runner.dynamic_window = lambda: (np.array([0.0, 0.30]), np.array([-0.4, 0.0, 0.4]))
        runner.room_local_productive_count_at_yaw = lambda *_args: (1 if _args[-1] > 0.0 else 0, [])
        result = runner.build_recovery_set(None, np.zeros((1, 1), dtype=bool), (0.8, 0.0), (1.0, 0.0), 1.0, ((0.0, 0.0), (1.0, 0.0)))
        self.assertTrue(result.approximate)
        self.assertEqual(len(result.hypothesis_records), 3)
        self.assertTrue(result.intervals)

    def test_19_unsafe_rotate_hypothesis_is_excluded_before_productivity_replay(self):
        runner = self.recovery_runner(runner_module.RecoverySet((), None, (), True))
        del runner.build_recovery_set
        runner.dynamic_window = lambda: (np.array([0.0, 0.30]), np.array([0.4]))
        runner.collision_free_arc = lambda *_args: (False, 0.0)
        with mock.patch.object(runner, "room_local_productive_count_at_yaw", wraps=runner.room_local_productive_count_at_yaw) as replay:
            result = runner.build_recovery_set(None, np.zeros((1, 1), dtype=bool), (0.8, 0.0), (1.0, 0.0), 1.0, ((0.0, 0.0), (1.0, 0.0)))
        self.assertFalse(result.intervals)
        self.assertFalse(result.hypothesis_records[0]["rotate_only_collision_safe"])
        replay.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
