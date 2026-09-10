#!/usr/bin/env python3
"""Offline purity, determinism and cohort tests for the ROOM_LOCAL evaluator."""

import copy
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

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
        "producer_instance_id": "pure-evaluator-fixture", "content_generation_id": 11,
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
    status["grid_content_hash"] = grid_content_hash(grid, status["producer_instance_id"], 11, 1.0)
    return status


class PureSameEpochRoomLocalEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.ros = FakeRos()
        install_message_stubs(self.ros)
        sys.modules.pop("block_astar_dwa_mature_runner", None)
        self.module = importlib.import_module("block_astar_dwa_mature_runner")
        self.grid = FakeGrid(width=80, height=80, origin_x=-2.0, origin_y=-2.0)
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
            epoch_id="fixture-epoch", pose_odom_xy_yaw=(0.0, 0.0, 0.0), grid_msg=self.grid,
            status_payload=self.status, args=self.args, previous_cmd=(0.0, 0.0), wall_heading_prior=None,
        )
        self.candidates = [
            self.module.FrozenRoomLocalCandidate.from_mapping({
                "candidate_id": "c1", "rank": 1, "sector": 0, "target_odom_xy": [1.0, 0.0],
                "heading_rad": 0.0, "candidate_type": "GENERIC_COVERAGE", "cheap_rank_components": {"sector": 0},
            }),
            self.module.FrozenRoomLocalCandidate.from_mapping({
                "candidate_id": "c2", "rank": 2, "sector": 1, "target_odom_xy": [1.0, 0.4],
                "heading_rad": 0.38, "candidate_type": "GENERIC_COVERAGE", "cheap_rank_components": {"sector": 1},
            }),
        ]

    def test_01_same_input_is_deterministic_and_nonmutating(self):
        before = (tuple(self.grid.data), copy.deepcopy(self.status), self.epoch)
        first = self.module.evaluate_frozen_room_local_candidate(self.epoch, self.candidates[0])
        second = self.module.evaluate_frozen_room_local_candidate(self.epoch, self.candidates[0])
        self.assertEqual(first, second)
        self.assertEqual(before[0], tuple(self.grid.data))
        self.assertEqual(before[1], self.status)
        self.assertEqual(before[2], self.epoch)
        self.assertFalse(first["commands_published"])
        self.assertFalse(first["persistent_state_mutated"])

    def test_02_evaluator_performs_no_ros_read_or_write_or_publish(self):
        forbidden = RuntimeError("ROS access is forbidden in pure evaluator")
        with mock.patch.object(self.module.rospy, "Publisher", side_effect=forbidden, create=True), \
             mock.patch.object(self.module.rospy, "Subscriber", side_effect=forbidden, create=True), \
             mock.patch.object(self.module.rospy, "get_param", side_effect=forbidden, create=True):
            record = self.module.evaluate_frozen_room_local_candidate(self.epoch, self.candidates[0])
        self.assertFalse(record["ros_read"])
        self.assertFalse(record["ros_write"])
        self.assertFalse(record["commands_published"])

    def test_03_cohort_has_one_identity_and_every_candidate_record(self):
        cohort = self.module.evaluate_frozen_room_local_cohort(self.epoch, self.candidates)
        self.assertTrue(cohort["complete"])
        self.assertTrue(cohort["same_epoch_identity"])
        self.assertEqual([row["candidate_id"] for row in cohort["records"]], ["c1", "c2"])
        self.assertTrue(all("motion_candidates" in row for row in cohort["records"]))

    def test_04_candidate_order_does_not_change_individual_result(self):
        forward = self.module.evaluate_frozen_room_local_cohort(self.epoch, self.candidates)
        reverse = self.module.evaluate_frozen_room_local_cohort(self.epoch, list(reversed(self.candidates)))
        by_id_a = {row["candidate_id"]: row for row in forward["records"]}
        by_id_b = {row["candidate_id"]: row for row in reverse["records"]}
        self.assertEqual(by_id_a, by_id_b)

    def test_05_transit_is_explicitly_isolated(self):
        transit_args = self.module.build_arg_parser().parse_args(["--local-control-mode", "TRANSIT"])
        transit_epoch = self.module.FrozenRoomLocalEpoch.from_live_inputs(
            epoch_id="transit", pose_odom_xy_yaw=(0.0, 0.0, 0.0), grid_msg=self.grid,
            status_payload=self.status, args=transit_args, previous_cmd=(0.0, 0.0), wall_heading_prior=None,
        )
        result = self.module.evaluate_frozen_room_local_candidate(transit_epoch, self.candidates[0])
        self.assertEqual(result["terminal_reason"], "TRANSIT_ROOM_LOCAL_EVALUATOR_NOT_APPLICABLE")
        self.assertEqual(result["formal_status"], "EVALUATION_NOT_AVAILABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
