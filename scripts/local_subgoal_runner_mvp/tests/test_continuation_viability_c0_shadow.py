#!/usr/bin/env python3
"""Offline C0 continuation-shadow tests: evidence only, never authority."""

import copy
import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_grid_contract import (
    GRID_CONTRACT_VERSION, GRID_STATUS_SCHEMA_VERSION, POINT_PLANNING_WITH_OBSTACLE_INFLATION,
    STATIC_PLANNING_FOOTPRINT_RADIUS_M, grid_content_hash,
)
from test_odom_cache import FakeGrid, FakeRos, FakeStamp, install_message_stubs


def status_for(grid):
    status = {
        "contract_version": GRID_CONTRACT_VERSION, "schema_version": GRID_STATUS_SCHEMA_VERSION,
        "producer_instance_id": "continuation-c0-fixture", "content_generation_id": 23,
        "grid_content_stamp": 1.0, "tf_valid": True, "all_required_inputs_fresh": True,
        "frame_id": "base", "width": grid.info.width, "height": grid.info.height,
        "resolution": grid.info.resolution,
        "origin": {"x": grid.info.origin.position.x, "y": grid.info.origin.position.y},
        "input_time_monotonic": True, "diagnostic_only": False, "safe_for_navigation": True,
        "upstream_navigation_allowed": True, "rejection_reasons": [],
        "static_footprint_radius_m": STATIC_PLANNING_FOOTPRINT_RADIUS_M,
        "planning_collision_model": POINT_PLANNING_WITH_OBSTACLE_INFLATION,
        "local_traversability_status": "FREE_SUPPORTED", "input_freshness_window_sec": 2.0,
    }
    status["grid_content_hash"] = grid_content_hash(grid, status["producer_instance_id"], 23, 1.0)
    return status


class ContinuationViabilityC0ShadowTests(unittest.TestCase):
    def setUp(self):
        self.ros = FakeRos()
        install_message_stubs(self.ros)
        sys.modules.pop("block_astar_dwa_mature_runner", None)
        self.module = importlib.import_module("block_astar_dwa_mature_runner")
        self.grid = FakeGrid(width=100, height=100, origin_x=-2.5, origin_y=-2.5)
        self.grid.header.stamp = FakeStamp(1.0)
        self.status = status_for(self.grid)
        self.args = self.module.build_arg_parser().parse_args([
            "--local-control-mode", "ROOM_LOCAL",
            "--room-local-phase2-productive-admission", "offline_frozen",
            "--disable-pointcloud-wall-heading", "--disable-imu-heading-hold",
            "--min-linear-x", "0.30", "--enforce-min-forward-speed",
            "--max-linear-x", "0.60", "--max-angular-z", "0.35",
        ])
        self.epoch = self.module.FrozenRoomLocalEpoch.from_live_inputs(
            epoch_id="c0-fixture", pose_odom_xy_yaw=(0.0, 0.0, 0.0), grid_msg=self.grid,
            status_payload=self.status, args=self.args, previous_cmd=(0.0, 0.0), wall_heading_prior=None,
        )
        self.candidate = self.module.FrozenRoomLocalCandidate.from_mapping({
            "candidate_id": "c0-target", "target_odom_xy": [1.4, 0.0],
        })

    def productive_motion(self):
        current = self.module.evaluate_frozen_room_local_candidate(self.epoch, self.candidate)
        motions = current["motion_candidates"]
        productive = [row for row in motions if row.get("score_eligible")]
        self.assertTrue(productive, "fixture must expose an existing Phase-2 productive motion")
        return copy.deepcopy(productive[0])

    def shared_context(self, epoch=None):
        return self.module.build_frozen_continuation_grid_context(epoch or self.epoch, self.candidate)

    def assert_legacy_and_shared_mask_equal(self, context, start_xy):
        legacy_runner = self.module._pure_room_local_runner(self.epoch)
        shared_runner = self.module._pure_room_local_runner(self.epoch)
        legacy_mask, legacy_report = legacy_runner.apply_start_footprint_clearance(
            context.grid_msg, context.raw_grid, context.base_blocked, context.occupied_inflated,
            qualification_passed=True, start_base_xy=start_xy,
        )
        shared_mask, shared_report = shared_runner.apply_start_footprint_clearance_from_shared_context(
            context.grid_msg, context.raw_grid, context.base_blocked, context.occupied_inflated,
            qualification_passed=True, cell_center_x_m=context.cell_center_x_m,
            cell_center_y_m=context.cell_center_y_m, raw_grid_sha256=context.raw_grid_sha256,
            start_base_xy=start_xy,
        )
        self.assertTrue(np.array_equal(legacy_mask, shared_mask))
        self.assertEqual(legacy_report, shared_report)

    def test_a_actual_frozen_s1_replay_is_pure_and_reports_a_tristate(self):
        motion = self.productive_motion()
        before = (self.epoch, tuple(self.grid.data), copy.deepcopy(self.status))
        record = self.module.evaluate_frozen_room_local_continuation(self.epoch, self.candidate, motion)
        self.assertEqual(record["continuation_status"], self.module.CONTINUATION_VIABLE)
        self.assertGreater(record["next_safe_count"], 0)
        self.assertGreater(record["next_productive_count"], 0)
        self.assertIn("slice_state_s1", record)
        self.assertFalse(record["commands_published"])
        self.assertFalse(record["persistent_state_mutated"])
        self.assertEqual(before[0], self.epoch)
        self.assertEqual(before[1], tuple(self.grid.data))
        self.assertEqual(before[2], self.status)

    def test_b_missing_slice_endpoint_is_unknown_not_a_failure(self):
        motion = self.productive_motion()
        motion.pop("slice_endpoint_base_xy", None)
        result = self.module.evaluate_frozen_room_local_continuation(self.epoch, self.candidate, motion)
        self.assertEqual(result["continuation_status"], self.module.CONTINUATION_UNKNOWN)
        self.assertEqual(result["reason"], "SLICE_ENDPOINT_MISSING_OR_INVALID")

    def test_b2_real_s1_no_path_is_nonviable_when_the_frozen_grid_is_complete(self):
        blocked_grid = FakeGrid(width=100, height=100, origin_x=-2.5, origin_y=-2.5, value=100)
        blocked_grid.header.stamp = FakeStamp(1.0)
        blocked_epoch = self.module.FrozenRoomLocalEpoch.from_live_inputs(
            epoch_id="c0-no-path", pose_odom_xy_yaw=(0.0, 0.0, 0.0), grid_msg=blocked_grid,
            status_payload=status_for(blocked_grid), args=self.args,
            previous_cmd=(0.0, 0.0), wall_heading_prior=None,
        )
        result = self.module.evaluate_frozen_room_local_continuation(
            blocked_epoch, self.candidate, self.productive_motion(),
        )
        self.assertEqual(result["continuation_status"], self.module.CONTINUATION_NON_VIABLE)
        self.assertEqual(result["reason"], "S1_ASTAR_NO_PATH")

    def test_c_s1_is_real_slice_endpoint_not_one_second_terminal(self):
        slice_state = self.module.dwa_rollout_state_at_duration(0.4, 0.3, 0.5, 0.1)
        horizon_state = self.module.dwa_rollout_state_at_duration(0.4, 0.3, 1.0, 0.1)
        short_slice_state = self.module.dwa_rollout_state_at_duration(0.4, 0.3, 0.2, 0.1)
        self.assertEqual(slice_state["time_sec"], 0.5)
        self.assertEqual(horizon_state["time_sec"], 1.0)
        self.assertEqual(short_slice_state["time_sec"], 0.2)
        self.assertNotEqual(slice_state["endpoint_base_xy"], horizon_state["endpoint_base_xy"])
        self.assertNotEqual(slice_state["endpoint_base_xy"], short_slice_state["endpoint_base_xy"])
        self.assertTrue(slice_state["partial_terminal_step"] is False)

    def test_d_cohort_checks_every_scheduled_motion_and_does_not_stop_at_first(self):
        motions = [
            {"v": 0.3, "w": -0.2, "score_eligible": True},
            {"v": 0.3, "w": 0.0, "score_eligible": True},
        ]
        responses = [
            {"continuation_status": self.module.CONTINUATION_NON_VIABLE},
            {"continuation_status": self.module.CONTINUATION_VIABLE},
        ]
        with mock.patch.object(self.module, "evaluate_frozen_room_local_continuation", side_effect=responses) as evaluate:
            result = self.module.evaluate_frozen_room_local_continuation_cohort(
                self.epoch, self.candidate, motions,
            )
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(result["continuation_status"], self.module.CONTINUATION_VIABLE)
        self.assertEqual(result["counts"][self.module.CONTINUATION_NON_VIABLE], 1)

    def test_e_incomplete_or_capped_evidence_is_unknown(self):
        motions = [
            {"v": 0.3, "w": value, "score_eligible": True}
            for value in (-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5)
        ]
        response = {"continuation_status": self.module.CONTINUATION_NON_VIABLE}
        with mock.patch.object(self.module, "evaluate_frozen_room_local_continuation", return_value=response):
            result = self.module.evaluate_frozen_room_local_continuation_cohort(
                self.epoch, self.candidate, motions, cap=2,
            )
        self.assertEqual(result["continuation_status"], self.module.CONTINUATION_UNKNOWN)
        self.assertGreater(result["unscheduled_motion_count"], 0)

    def test_f_transit_is_isolated_and_cannot_become_a_room_local_result(self):
        transit_args = self.module.build_arg_parser().parse_args(["--local-control-mode", "TRANSIT"])
        transit_epoch = self.module.FrozenRoomLocalEpoch.from_live_inputs(
            epoch_id="transit", pose_odom_xy_yaw=(0.0, 0.0, 0.0), grid_msg=self.grid,
            status_payload=self.status, args=transit_args, previous_cmd=(0.0, 0.0), wall_heading_prior=None,
        )
        result = self.module.evaluate_frozen_room_local_continuation(
            transit_epoch, self.candidate, self.productive_motion(),
        )
        self.assertEqual(result["continuation_status"], self.module.CONTINUATION_UNKNOWN)
        self.assertEqual(result["reason"], "ROOM_LOCAL_PHASE2_AUTHORITY_UNAVAILABLE")

    def test_g_evaluator_performs_no_ros_access_or_publish(self):
        forbidden = RuntimeError("ROS access is forbidden in C0")
        with mock.patch.object(self.module.rospy, "Publisher", side_effect=forbidden, create=True), \
             mock.patch.object(self.module.rospy, "Subscriber", side_effect=forbidden, create=True), \
             mock.patch.object(self.module.rospy, "get_param", side_effect=forbidden, create=True):
            result = self.module.evaluate_frozen_room_local_continuation(
                self.epoch, self.candidate, self.productive_motion(),
            )
        self.assertFalse(result["commands_published"])
        self.assertFalse(result["ros_read"])
        self.assertFalse(result["ros_write"])

    def test_h_shadow_is_default_off_and_remains_separate_from_local_selection_authority(self):
        args = self.module.build_arg_parser().parse_args([])
        self.assertFalse(args.continuation_viability_shadow)
        self.assertFalse(self.module.continuation_local_selection_authority_is_active(args))
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        self.assertIn('"continuation_viability_shadow"', source)
        self.assertIn('"continuation_viability_authority_evidence"', source)
        self.assertIn('continuation_local_selection_authority_active', source)

    def test_i_shared_footprint_mask_matches_legacy_for_normal_and_cell_boundary_s1(self):
        context = self.shared_context()
        motion = self.productive_motion()
        normal = tuple(motion["slice_endpoint_base_xy"])
        cell_boundary = (0.500000000001, -0.250000000001)
        self.assert_legacy_and_shared_mask_equal(context, normal)
        self.assert_legacy_and_shared_mask_equal(context, cell_boundary)

    def test_j_shared_footprint_mask_matches_legacy_near_grid_edge_and_occupied_cells(self):
        grid = FakeGrid(width=100, height=100, origin_x=-2.5, origin_y=-2.5)
        grid.header.stamp = FakeStamp(1.0)
        # An occupied seed inside the S1 footprint exercises both occupied and
        # inflated preservation; the edge point exercises clipping semantics.
        grid.data[50 * grid.info.width + 2] = 100
        epoch = self.module.FrozenRoomLocalEpoch.from_live_inputs(
            epoch_id="c0-edge", pose_odom_xy_yaw=(0.0, 0.0, 0.0), grid_msg=grid,
            status_payload=status_for(grid), args=self.args, previous_cmd=(0.0, 0.0), wall_heading_prior=None,
        )
        context = self.shared_context(epoch)
        self.assert_legacy_and_shared_mask_equal(context, (-2.499, 0.025))
        self.assertGreater(int(context.occupied_inflated.sum()), 0)

    def test_k_shared_context_is_read_only_and_reused_by_multiple_s1_evaluations(self):
        context = self.shared_context()
        before = {
            "raw": context.raw_grid.copy(), "planning": context.planning_grid.copy(),
            "inflated": context.occupied_inflated.copy(), "blocked": context.base_blocked.copy(),
            "x": context.cell_center_x_m.copy(), "y": context.cell_center_y_m.copy(),
            "target": context.target_s0_xy, "grid_data": tuple(context.grid_msg.data),
        }
        motions = [row for row in self.module.evaluate_frozen_room_local_candidate(
            self.epoch, self.candidate,
        )["motion_candidates"] if row.get("score_eligible")][:3]
        self.assertEqual(len(motions), 3)
        for motion in motions:
            result = self.module.evaluate_frozen_room_local_continuation(
                self.epoch, self.candidate, motion, shared_grid_context=context,
            )
            self.assertIn(result["continuation_status"], {
                self.module.CONTINUATION_VIABLE, self.module.CONTINUATION_NON_VIABLE,
                self.module.CONTINUATION_UNKNOWN,
            })
        self.assertFalse(context.raw_grid.flags.writeable)
        self.assertFalse(context.base_blocked.flags.writeable)
        self.assertTrue(np.array_equal(before["raw"], context.raw_grid))
        self.assertTrue(np.array_equal(before["planning"], context.planning_grid))
        self.assertTrue(np.array_equal(before["inflated"], context.occupied_inflated))
        self.assertTrue(np.array_equal(before["blocked"], context.base_blocked))
        self.assertTrue(np.array_equal(before["x"], context.cell_center_x_m))
        self.assertTrue(np.array_equal(before["y"], context.cell_center_y_m))
        self.assertEqual(before["target"], context.target_s0_xy)
        self.assertEqual(before["grid_data"], tuple(context.grid_msg.data))

    def test_l_new_shared_path_matches_legacy_mask_for_full_continuation_result(self):
        def legacy_adapter(
            runner, grid_msg, raw_grid, base_blocked, occupied_inflated, qualification_passed,
            cell_center_x_m, cell_center_y_m, raw_grid_sha256, start_base_xy,
        ):
            return runner.apply_start_footprint_clearance(
                grid_msg, raw_grid, base_blocked, occupied_inflated,
                qualification_passed, start_base_xy,
            )

        motions = [row for row in self.module.evaluate_frozen_room_local_candidate(
            self.epoch, self.candidate,
        )["motion_candidates"] if row.get("score_eligible")][:3]
        for motion in motions:
            shared = self.module.evaluate_frozen_room_local_continuation(self.epoch, self.candidate, motion)
            with mock.patch.object(
                self.module.BlockAStarDwaRunner,
                "apply_start_footprint_clearance_from_shared_context",
                new=legacy_adapter,
            ):
                legacy = self.module.evaluate_frozen_room_local_continuation(self.epoch, self.candidate, motion)
            for key in (
                "continuation_status", "reason", "next_safe_count", "next_productive_count",
                "path_cell_count", "lookahead_path_index", "start_footprint_clearance", "target_shape", "dwa",
            ):
                self.assertEqual(shared[key], legacy[key], key)

    def test_m_cohort_builds_one_context_and_preserves_tristate_result(self):
        motions = [row for row in self.module.evaluate_frozen_room_local_candidate(
            self.epoch, self.candidate,
        )["motion_candidates"] if row.get("score_eligible")][:3]
        cohort = self.module.evaluate_frozen_room_local_continuation_cohort(
            self.epoch, self.candidate, motions,
        )
        self.assertTrue(cohort["shared_context_built"])
        self.assertGreater(cohort["shared_preprocessing_ns"], 0)
        self.assertEqual(len(cohort["records"]), 3)
        self.assertEqual(cohort["continuation_status"], self.module.CONTINUATION_VIABLE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
