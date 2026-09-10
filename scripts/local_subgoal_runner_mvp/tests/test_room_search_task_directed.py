#!/usr/bin/env python3
"""Focused offline tests for the task-directed ROOM_SEARCH repair."""

import ast
import inspect
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

import navigation_state_machine as navigation
from room_search_v1 import (
    MAX_CANDIDATES,
    RAW_SECTOR_REPRESENTATIVE_CAP,
    ObservationMemory,
    PortalAnchor,
    RoomSearchV2,
    occlusion_reveal_room_points,
    strategic_reposition_needed,
)


def make_search(entry=(10.3, 20.0, 0.2)):
    target = {
        "portal_width_m": 1.2,
        "frozen_geometry": {"portal_center_odom": [10.0, 20.0], "portal_normal_odom": [1.0, 0.0]},
    }
    return RoomSearchV2(PortalAnchor.from_frozen_target(target, entry, 12.0))


class TaskDirectedRoomSearchTests(unittest.TestCase):
    def test_preonline_t1_camera_fallback_is_finite_sixty_degrees(self):
        cache = object.__new__(navigation.RoomSearchCameraInfoCache)
        import threading
        cache._lock, cache._hfov_rad = threading.Lock(), None
        hfov, source = cache.effective_hfov()
        self.assertAlmostEqual(hfov, math.radians(60.0))
        self.assertEqual(source, "FALLBACK_60_DEG")

    def test_preonline_t2_fallback_excludes_outside_thirty_degrees(self):
        self.assertGreater(abs(math.radians(31.0)), navigation.ROOM_SEARCH_CAMERA_HFOV_FALLBACK_RAD / 2.0)

    def test_preonline_t3_camera_info_updates_without_wait(self):
        cache = object.__new__(navigation.RoomSearchCameraInfoCache)
        import threading
        cache._lock, cache._hfov_rad = threading.Lock(), None
        info = type("Info", (), {"width": 640, "K": [554.0]})()
        cache._callback(info)
        self.assertEqual(cache.effective_hfov()[1], "CAMERA_INFO")

    def test_preonline_t6_raw_sector_representatives_are_not_pretruncated(self):
        search = make_search()
        points = [(math.cos(-math.pi + (index + 0.5) * math.pi / 6.0), math.sin(-math.pi + (index + 0.5) * math.pi / 6.0)) for index in range(12)]
        self.assertEqual(len(search.candidates_from_planning_free_base((10.3, 20.0, 0.0), points)), RAW_SECTOR_REPRESENTATIVE_CAP)

    def test_preonline_t8_formal_preflight_cap_remains_five(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn("cheap_ranked = cheap_ranked_all[:MAX_CANDIDATES]", source)

    def test_mission_r1_tentative_hypothesis_parser_is_fail_soft(self):
        self.assertEqual(navigation.parse_tentative_danger_hypotheses("not-json"), [])
        self.assertEqual(navigation.parse_tentative_danger_hypotheses({"schema_version": 1, "frame_id": "wrong", "hypotheses": []}), [])

    def test_mission_r2_danger_reobserve_outranks_occlusion(self):
        ranked = make_search().cheap_rank_candidates([
            {"sector": 1, "base_xy": [1.0, 0.0], "room_target_xy": [2.0, 0.0], "occlusion_reveal_room_points": [(2.0, 0.0)]},
            {"sector": 2, "base_xy": [1.0, 0.1], "room_target_xy": [2.0, 0.1], "danger_reobserve_supported": True, "danger_reobserve_abs_bearing_rad": 0.1},
        ])
        self.assertEqual(ranked[0]["target_priority_class"], "DANGER_REOBSERVE")

    def test_mission_r11_t14_terminal_expansion_is_bounded_and_checks_sixth(self):
        calls = []
        selected, attempts, expansion = RoomSearchV2.admit_with_terminal_expansion(
            [{"id": index} for index in range(1, 13)],
            lambda row: calls.append(row["id"]) or {"legal": row["id"] == 6},
        )
        self.assertEqual(selected["id"], 6)
        self.assertEqual((attempts, expansion, calls), (6, 1, [1, 2, 3, 4, 5, 6]))

    def test_mission_r15_all_candidates_fail_only_after_full_bounded_exhaustion(self):
        selected, attempts, expansion = RoomSearchV2.admit_with_terminal_expansion(
            [{"id": index} for index in range(12)], lambda _row: {"legal": False},
        )
        self.assertIsNone(selected)
        self.assertEqual((attempts, expansion), (12, 7))

    def test_mission_r3_r4_reobserve_stays_behind_existing_preflight_and_falls_through(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn("room_search_v2_preflight", source)
        self.assertIn("danger_reobserve_supported", source)
        rank_source = inspect.getsource(RoomSearchV2.cheap_rank_candidates)
        self.assertIn("new_observable_cells", rank_source)
        self.assertIn("occlusion_reveal_cells", rank_source)

    def test_mission_r6_r8_reobserve_hold_is_single_and_bounded(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn("held_hypothesis_ids", source)
        self.assertIn("ROOM_SEARCH_REOBSERVE_HOLD_SIM_SEC", source)
        self.assertAlmostEqual(navigation.ROOM_SEARCH_REOBSERVE_HOLD_SIM_SEC, 0.6)

    def test_mission_r9_confirmed_track_is_not_a_reobserve_hypothesis(self):
        payload = {"schema_version": 1, "frame_id": "team_livox_odom", "hypotheses": [
            {"hypothesis_id": "danger-1", "state": "CONFIRMED", "position_xyz_m": [1.0, 0.0, 0.0], "confidence": 0.9, "support_count": 2, "last_observed_stamp_sec": 1.0},
        ]}
        self.assertEqual(navigation.parse_tentative_danger_hypotheses(payload), [])

    def test_mission_r16_r17_execution_failure_repositions_before_next_completion(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn("attempt_strategic_reposition(candidate, runner_decision)", source)
        self.assertIn("REGENERATE_CANDIDATES_THEN_TERMINAL_EXPAND", source)

    def test_mission_r19_r20_tentative_blocks_low_gain_stop_but_keeps_progress_guard(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn('marginal["completion_reason"] is not None and not active_hypotheses', source)
        self.assertIn("record_nonproductive_cycle", source)
        self.assertIn("while not rospy.is_shutdown():", source)
    def test_t1_occluded_unseen_cell_is_revealed(self):
        revealed = occlusion_reveal_room_points(ObservationMemory(), [(1.0, 1.0)], lambda _point: False)
        self.assertEqual(revealed, [(1.0, 1.0)])

    def test_t2_seen_cell_has_no_occlusion_credit(self):
        memory = ObservationMemory()
        memory.update([(1.0, 1.0)])
        self.assertEqual(occlusion_reveal_room_points(memory, [(1.0, 1.0)], lambda _point: False), [])

    def test_t3_candidate_without_clear_view_gets_no_credit(self):
        # A caller supplies only candidate-visible points; an empty set means
        # it cannot expose the hidden cell.
        self.assertEqual(occlusion_reveal_room_points(ObservationMemory(), [], lambda _point: False), [])

    def test_t4_normal_coverage_beats_occlusion_provenance(self):
        ranked = make_search().cheap_rank_candidates([
            {"sector": 1, "room_target_xy": [2.0, 0.0], "base_xy": [1.0, 0.0],
             "visible_room_points": [(x, 0.0) for x in range(5)], "occlusion_reveal_room_points": []},
            {"sector": 2, "room_target_xy": [2.0, 1.0], "base_xy": [1.1, 0.1],
             "visible_room_points": [(9.0, 0.0)], "occlusion_reveal_room_points": [(9.0, 0.0)]},
        ])
        self.assertEqual(ranked[0]["target_priority_class"], "GENERIC_COVERAGE")
        self.assertEqual(ranked[0]["sector"], 1)

    def test_t5_generic_coverage_fallback_preserves_more_area_first(self):
        ranked = make_search().cheap_rank_candidates([
            {"sector": 1, "room_target_xy": [2.0, 0.0], "base_xy": [1.0, 0.0],
             "visible_room_points": [(1.0, 0.0)], "occlusion_reveal_room_points": []},
            {"sector": 2, "room_target_xy": [2.0, 1.0], "base_xy": [1.1, 0.1],
             "visible_room_points": [(2.0, 0.0), (2.0, 1.0)], "occlusion_reveal_room_points": []},
        ])
        self.assertEqual(ranked[0]["target_priority_class"], "GENERIC_COVERAGE")
        self.assertEqual(ranked[0]["sector"], 2)

    def test_t6_ranking_has_no_motion_authority(self):
        candidate, attempts = RoomSearchV2.admit_ranked_candidates(
            make_search().cheap_rank_candidates([{"sector": 1, "base_xy": [1.0, 0.0], "room_target_xy": [2.0, 0.0]}]),
            lambda _candidate: {"legal": False},
        )
        self.assertIsNone(candidate)
        self.assertEqual(attempts, 1)

    def test_t7_safe_first_candidate_stops_after_one_preflight(self):
        calls = []
        candidate, attempts = RoomSearchV2.admit_ranked_candidates([{"id": 1}, {"id": 2}], lambda row: calls.append(row["id"]) or {"legal": True})
        self.assertEqual((candidate["id"], attempts, calls), (1, 1, [1]))

    def test_t8_second_candidate_needs_exactly_two_preflights(self):
        calls = []
        candidate, attempts = RoomSearchV2.admit_ranked_candidates(
            [{"id": 1}, {"id": 2}, {"id": 3}], lambda row: calls.append(row["id"]) or {"legal": row["id"] == 2},
        )
        self.assertEqual((candidate["id"], attempts, calls), (2, 2, [1, 2]))

    def test_t9_all_unsafe_candidates_fail_closed_and_stay_bounded(self):
        candidates = [{"id": number} for number in range(MAX_CANDIDATES)]
        selected, attempts = RoomSearchV2.admit_ranked_candidates(candidates, lambda _row: {"legal": False})
        self.assertIsNone(selected)
        self.assertEqual(attempts, MAX_CANDIDATES)

    def test_t10_candidate_population_contract_is_five(self):
        self.assertEqual(MAX_CANDIDATES, 5)

    def test_t11_selected_preflight_path_cost_reaches_exact_nbv_score(self):
        scored = make_search().score_candidates([{
            "room_target_xy": [2.0, 0.0], "visible_room_points": [(1.0, 1.0), (1.0, 1.5)], "path_length_m": 2.0,
        }])[0]
        self.assertEqual(scored["nbv_value"], 1.0)

    def test_preonline_t12_occlusion_does_not_trigger_generic_low_gain(self):
        search = make_search()
        first = search.evaluate_marginal_value({"target_priority_class": "OCCLUSION", "nbv_value": 0.01}, 1)
        second = search.evaluate_marginal_value({"target_priority_class": "OCCLUSION", "nbv_value": 0.01}, 2)
        self.assertIsNone(first["completion_reason"])
        self.assertIsNone(second["completion_reason"])
        self.assertEqual(search.low_gain_streak, 0)

    def test_preonline_t15_dry_preflight_is_one_step_and_execution_is_not(self):
        def calls_to_run_runner(function):
            tree = ast.parse(inspect.getsource(function))
            return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "run_runner"]

        def is_args_attribute(node, attribute):
            return (isinstance(node, ast.Attribute) and node.attr == attribute
                    and isinstance(node.value, ast.Name) and node.value.id == "args")

        preflight_source = ast.parse(inspect.getsource(navigation.room_search_v2_preflight))
        preflight_calls = calls_to_run_runner(navigation.room_search_v2_preflight)
        self.assertEqual(len(preflight_calls), 1)
        preflight_call = preflight_calls[0]
        self.assertEqual(len(preflight_call.args), 4)
        self.assertIsInstance(preflight_call.args[0], ast.Name)
        self.assertEqual(preflight_call.args[0].id, "dry_args")
        self.assertIsInstance(preflight_call.args[1], ast.Constant)
        self.assertEqual(preflight_call.args[1].value, "ROOM_SEARCH")
        self.assertTrue(is_args_attribute(preflight_call.args[2], "runner_runtime_sec"))
        self.assertIsInstance(preflight_call.args[3], ast.Constant)
        self.assertEqual(preflight_call.args[3].value, 1)
        self.assertEqual(
            [(item.arg, item.value.value) for item in preflight_call.keywords
             if item.arg == "local_control_mode" and isinstance(item.value, ast.Constant)],
            [("local_control_mode", "ROOM_LOCAL")],
        )
        self.assertTrue(any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "dry_args" for target in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "copy" and node.value.func.attr == "copy"
            and len(node.value.args) == 1 and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == "args"
            for node in ast.walk(preflight_source)
        ))
        self.assertTrue(any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == "execute"
                    and isinstance(target.value, ast.Name) and target.value.id == "dry_args"
                    for target in node.targets)
            and isinstance(node.value, ast.Constant) and node.value.value is False
            for node in ast.walk(preflight_source)
        ))

        execution_tree = ast.parse(inspect.getsource(navigation.execute_room_search_v2))
        execute_target = next(
            node for node in ast.walk(execution_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "execute_target"
        )
        self.assertEqual(execute_target.args.args[3].arg, "state")
        self.assertIsInstance(execute_target.args.defaults[0], ast.Constant)
        self.assertEqual(execute_target.args.defaults[0].value, "ROOM_SEARCH")
        self.assertEqual(execute_target.args.kwonlyargs[0].arg, "local_control_mode")
        self.assertIsInstance(execute_target.args.kw_defaults[0], ast.Constant)
        self.assertEqual(execute_target.args.kw_defaults[0].value, "ROOM_LOCAL")
        target_calls = [node for node in ast.walk(execute_target) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == "run_runner"]
        self.assertEqual(len(target_calls), 1)
        target_call = target_calls[0]
        self.assertEqual([node.id if isinstance(node, ast.Name) else None for node in target_call.args[:2]], ["args", "state"])
        self.assertTrue(is_args_attribute(target_call.args[2], "runner_runtime_sec"))
        self.assertTrue(is_args_attribute(target_call.args[3], "runner_max_steps"))
        self.assertEqual(
            [(item.arg, item.value.id) for item in target_call.keywords
             if item.arg == "local_control_mode" and isinstance(item.value, ast.Name)],
            [("local_control_mode", "local_control_mode")],
        )
        room_search_target_calls = [
            node for node in ast.walk(execution_tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "execute_target"
            and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "ROOM_SEARCH_TARGET"
        ]
        self.assertEqual(len(room_search_target_calls), 1)

    def test_t13_sensor_smoke_is_not_called_by_normal_execution(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertNotIn("smoke.capture", source)
        self.assertIn("FormalGridStatusPairSubscriber", source)

    def test_t14_exact_grid_status_pair_is_still_required(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn("room_search_v2_planning_context(args, grid_status_pairs)", source)
        self.assertIn("ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE", source)

    def test_t15_explicit_runner_failure_triggers_reposition_only(self):
        self.assertTrue(strategic_reposition_needed("BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD"))
        self.assertFalse(strategic_reposition_needed("BLOCK_ASTAR_DWA_REACHED_GOAL"))

    def test_t16_reposition_uses_actual_pose_anchors_reverse_chronologically(self):
        search = make_search()
        search.record_safe_actual_pose((10.3, 20.0, 0.0), 0.0)
        search.record_safe_actual_pose((10.8, 20.0, 0.0), 0.0)
        search.record_safe_actual_pose((11.2, 20.0, 0.0), 0.0)
        anchors = search.reposition_anchor_candidates((12.0, 20.0, 0.0), 0.1)
        self.assertEqual([row["actual_pose_xy_yaw"] for row in anchors], [[11.2, 20.0, 0.0], [10.8, 20.0, 0.0]])

    def test_t17_t21_reposition_preflight_and_cap_are_bounded(self):
        search = make_search()
        for offset in (0.0, 0.4, 0.8, 1.2):
            search.record_safe_actual_pose((10.3 + offset, 20.0, 0.0), 0.0)
        self.assertLessEqual(len(search.reposition_anchor_candidates((12.0, 20.0, 0.0), 0.1)), 2)
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn("room_search_v2_preflight(\n                args, target_xy", source)
        self.assertIn("maximum_attempts=2", source)

    def test_t18_success_requires_actual_displacement(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn("displacement >= float(args.stuck_min_displacement_m)", source)

    def test_t19_success_restarts_normal_candidate_selection(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn('if record.get("strategic_reposition", {}).get("successful"):\n            continue', source)

    def test_t20_sector_cooldown_expires_after_one_generation(self):
        search = make_search()
        search.arm_one_decision_sector_cooldown(6)
        free = [(1.0, 0.0), (-1.0, 0.0)]
        first = search.candidates_from_planning_free_base((10.3, 20.0, 0.0), free)
        second = search.candidates_from_planning_free_base((10.3, 20.0, 0.0), free)
        self.assertEqual(len(second), len(first) + 1)

    def test_t23_final_return_remains_a_separate_helper(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn('write_stage("room_search", "navigation_state_machine.py", {"mode": "ROOM_RETURN"})', source)
        self.assertIn("search.door_return_target()", source)

    def test_preonline_t26_behind_target_can_use_heading_improving_transition(self):
        transition = RoomSearchV2.choose_safe_return_transition(
            (0.0, 0.0, 0.0), (-1.0, 0.0),
            [{"legal": True, "target_xy_team_livox_odom": [0.0, 1.0]}],
        )
        self.assertIsNotNone(transition)
        self.assertGreater(transition["return_bearing_gain_rad"], 0.0)

    def test_preonline_t35_t39_startup_is_rgbd_only_and_nav_command_is_preserved(self):
        source = (ROOT / "scripts/local_subgoal_runner_mvp/run_state_machine_navigation.sh").read_text(encoding="utf-8")
        self.assertIn("run_rgbd_danger_perception.sh", source)
        self.assertNotIn("vision_scene_semantics_node.py", source)
        self.assertNotIn("doorway_candidate_detector.py", source)
        self.assertNotIn("room_frontier_viewpoint_selector.py", source)
        self.assertIn('python3 scripts/local_subgoal_runner_mvp/navigation_state_machine.py "$@"', source)

    def test_t22_unavailable_reposition_has_a_bounded_terminal_state(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn('"state": "STRATEGIC_REPOSITION_UNAVAILABLE"', source)

    def test_t24_no_blind_negative_reverse_is_introduced(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertNotIn("publish_zero_and_backoff", source)
        self.assertNotIn("linear_x=-", source)

    def test_t25_shared_dwa_parameters_are_not_tuned_here(self):
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertNotIn("max_angular_z", source)
        self.assertNotIn("dwa_linear", source)

    def test_t26_t27_absent_or_malformed_danger_snapshot_is_empty(self):
        self.assertEqual(navigation.parse_confirmed_danger_tracks("not-json"), [])
        self.assertEqual(navigation.parse_confirmed_danger_tracks({"schema_version": 2, "frame_id": "team_livox_odom", "tracks": []}), [])
        self.assertEqual(navigation.parse_confirmed_danger_tracks({
            "schema_version": 1, "frame_id": "team_livox_odom", "tracks": [
                {"track_id": "bad", "state": "CONFIRMED", "position_xyz_m": [float("nan"), 0.0, 0.0]},
            ],
        }), [])

    def test_t28_confirmed_danger_is_recorded_but_has_no_motion_field(self):
        tracks = navigation.parse_confirmed_danger_tracks({
            "schema_version": 1, "frame_id": "team_livox_odom", "tracks": [
                {"track_id": "danger-1", "state": "CONFIRMED", "position_xyz_m": [1.0, 2.0, 0.5], "last_observed_stamp_sec": 4.0},
            ],
        })
        self.assertEqual(tracks[0]["track_id"], "danger-1")
        self.assertNotIn("target", tracks[0])

    def test_t29_incremental_snapshot_writer_is_independently_available(self):
        self.assertTrue(callable(navigation.write_atomic_json))
        source = inspect.getsource(navigation.execute_room_search_v2)
        self.assertIn("persist_diagnostic", source)
        self.assertIn("ROOM_SEARCH_DIAGNOSTIC_PATH", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
