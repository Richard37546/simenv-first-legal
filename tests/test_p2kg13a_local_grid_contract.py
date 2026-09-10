#!/usr/bin/env python3
"""Offline P2KG13A fixtures; no ROS master, publishers, or control output."""
from __future__ import annotations

import importlib.util
import hashlib
import math
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = ROOT / "scripts" / "local_subgoal_runner_mvp"
sys.path.insert(0, str(RUNNER_DIR))
from local_grid_contract import (  # noqa: E402
    GRID_CONTRACT_VERSION,
    GRID_STATUS_SCHEMA_VERSION,
    cell_to_metric,
    flatten_index,
    grid_content_hash,
    grid_metadata,
    metric_to_cell,
    POINT_PLANNING_WITH_OBSTACLE_INFLATION,
    qualified_for_navigation,
    STATIC_PLANNING_FOOTPRINT_RADIUS_M,
    static_footprint_fits_map,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Python 3.8 dataclass annotation resolution requires dynamically loaded
    # modules to be registered while their class bodies execute.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Stamp:
    def __init__(self, value: float) -> None:
        self.value = value

    def to_sec(self) -> float:
        return self.value


def make_grid(width: int = 60, height: int = 60, resolution: float = 0.05, generation: int = 1, stamp: float = 10.0, origin_x: float = 0.0):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="base", seq=generation, stamp=Stamp(stamp)),
        info=SimpleNamespace(
            resolution=resolution,
            width=width,
            height=height,
            origin=SimpleNamespace(position=SimpleNamespace(x=origin_x, y=-1.5)),
        ),
        data=[0] * (width * height),
    )


def set_cell(grid, x_index: int, y_index: int, value: int) -> None:
    index = flatten_index(x_index, y_index, grid.info.width, grid.info.height)
    assert index is not None
    grid.data[index] = value


def qualified_status(grid, producer: str = "fixture-producer", generation: int = 7):
    metadata = grid_metadata(grid)
    status = {
        "contract_version": GRID_CONTRACT_VERSION,
        "schema_version": GRID_STATUS_SCHEMA_VERSION,
        "producer_instance_id": producer,
        "content_generation_id": generation,
        "grid_content_stamp": metadata["content_stamp"],
        "frame_id": metadata["frame_id"],
        "origin": {"x": metadata["origin_x"], "y": metadata["origin_y"]},
        "resolution": metadata["resolution"],
        "width": metadata["width"],
        "height": metadata["height"],
        "tf_valid": True,
        "input_time_monotonic": True,
        "all_required_inputs_fresh": True,
        "diagnostic_only": False,
        "safe_for_navigation": True,
        "upstream_navigation_allowed": True,
        "rejection_reasons": [],
        "static_footprint_radius_m": STATIC_PLANNING_FOOTPRINT_RADIUS_M,
        "planning_collision_model": POINT_PLANNING_WITH_OBSTACLE_INFLATION,
    }
    status["grid_content_hash"] = grid_content_hash(
        grid, producer, status["content_generation_id"], status["grid_content_stamp"]
    )
    return status


class LocalGridContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.l3v = load_module(
            "p2kg13a_l3v",
            ROOT / "scripts/l3v_local_traversability_diagnostic_node/l3v_local_traversability_node.py",
        )
        cls.runner_module = load_module(
            "p2kg13a_runner", RUNNER_DIR / "block_astar_dwa_mature_runner.py"
        )
        cls.state = load_module(
            "p2kg13a_state", RUNNER_DIR / "navigation_state_machine.py"
        )

    def runner(self):
        runner = object.__new__(self.runner_module.BlockAStarDwaRunner)
        runner.args = SimpleNamespace(
            robot_radius_m=STATIC_PLANNING_FOOTPRINT_RADIUS_M,
            grid_resolution_m=0.05,
            block_size_cells=1,
            astar_lateral_bias_weight=0.0,
            dwa_predict_time=1.0,
            dwa_dt=0.05,
            max_clearance_score_m=3.0,
        )
        return runner

    def clearance_fixture(self):
        runner = self.runner()
        grid = make_grid(width=66, height=60, origin_x=-0.30)
        for y_index in range(grid.info.height):
            for x_index in range(grid.info.width):
                x, y = runner.cell_to_local_xy((x_index, y_index), grid)
                if x * x + y * y <= STATIC_PLANNING_FOOTPRINT_RADIUS_M ** 2:
                    set_cell(grid, x_index, y_index, -1)
        raw = runner.grid_array(grid)
        occupied_inflated = runner.occupied_inflated_mask(raw)
        blocked = runner.inflate_obstacles(raw)
        return runner, grid, raw, blocked, occupied_inflated

    def apply_clearance(self, runner, grid, raw, blocked, occupied_inflated):
        return runner.apply_start_footprint_clearance(
            grid, raw, blocked, occupied_inflated, qualification_passed=True
        )

    def test_spatial_direction_fixtures_are_consistent_across_all_readers(self):
        fixtures = {
            "front": (1.00, 0.00), "left": (1.00, 1.00), "right": (1.00, -1.00),
            "left_front": (1.30, 0.60), "right_front": (1.30, -0.60),
            "left_doorway": (1.80, 1.10), "right_doorway": (1.80, -1.10),
            "l_corner": (2.20, 0.70), "left_wall": (2.60, 1.30),
            "right_wall": (2.60, -1.30), "asymmetric_multi": (0.70, -0.35),
        }
        node = object.__new__(self.l3v.L3VLocalTraversabilityNode)
        node.rows, node.cols, node.resolution, node.x_min, node.y_min, node.frame_id = 60, 60, 0.05, 0.0, -1.5, "base"
        labels = np.full((node.rows, node.cols), "unknown", dtype=object)
        metadata = {
            "origin_x": 0.0, "origin_y": -1.5, "resolution": 0.05, "width": 60, "height": 60,
        }
        for x, y in fixtures.values():
            cell = metric_to_cell(x, y, metadata)
            self.assertIsNotNone(cell)
            labels[cell[0], cell[1]] = "occupied"
        grid = node.occupancy_grid_locked(labels)
        grid.header.seq = 1
        grid.header.stamp = self.l3v.rospy.Time.from_sec(10.0)
        metadata = grid_metadata(grid)
        runner = self.runner()
        runner_grid = runner.grid_array(grid)
        state_grid = self.state.grid_array(grid)
        blocked = runner.inflate_obstacles(runner_grid)
        for name, (x, y) in fixtures.items():
            with self.subTest(name=name):
                cell = metric_to_cell(x, y, metadata)
                self.assertEqual(cell_to_metric(cell[0], cell[1], metadata) and metric_to_cell(*cell_to_metric(cell[0], cell[1], metadata), metadata), cell)
                index = flatten_index(cell[0], cell[1], grid.info.width, grid.info.height)
                self.assertEqual(grid.data[index], 100)
                self.assertEqual(runner_grid[cell[1], cell[0]], 100)
                self.assertEqual(state_grid[cell[1], cell[0]], 100)
                self.assertEqual(runner.local_xy_to_cell(x, y, grid), cell)
                self.assertTrue(blocked[cell[1], cell[0]])
                self.assertFalse(runner.block_window_stats(blocked, cell)["block_free"])
        self.assertFalse(runner.collision_free_arc(grid, blocked, 0.35, 0.0)[0])

    def test_l3v_public_serialization_is_standard_y_by_x(self):
        node = object.__new__(self.l3v.L3VLocalTraversabilityNode)
        node.rows, node.cols, node.resolution, node.x_min, node.y_min, node.frame_id = 60, 60, 0.05, 0.0, -1.5, "base"
        labels = np.full((node.rows, node.cols), "unknown", dtype=object)
        x_index, y_index = 26, 42
        labels[x_index, y_index] = "occupied"
        grid = node.occupancy_grid_locked(labels)
        self.assertEqual((grid.info.width, grid.info.height), (60, 60))
        self.assertEqual(grid.data[flatten_index(x_index, y_index, 60, 60)], 100)
        self.assertEqual(np.array(grid.data).reshape((60, 60))[y_index, x_index], 100)

    def test_rear_map_geometry_and_static_footprint_coverage(self):
        node = object.__new__(self.l3v.L3VLocalTraversabilityNode)
        node.rows, node.cols, node.resolution = 66, 60, 0.05
        node.x_min, node.x_max, node.y_min, node.y_max, node.frame_id = -0.30, 3.0, -1.5, 1.5, "base"
        self.assertEqual(node.local_to_cell(-0.125, 0.0), (3, 30))
        self.assertEqual(node.local_to_cell(-0.05, 0.0), (5, 30))
        self.assertIsNone(node.local_to_cell(-0.3000001, 0.0))
        self.assertIsNone(node.local_to_cell(3.0, 0.0))
        labels = np.full((66, 60), "unknown", dtype=object)
        grid = node.occupancy_grid_locked(labels)
        self.assertEqual((grid.info.width, grid.info.height), (66, 60))
        self.assertEqual((grid.info.origin.position.x, grid.info.origin.position.y), (-0.30, -1.5))
        self.assertTrue(static_footprint_fits_map(grid_metadata(grid)))
        old_grid = make_grid()
        self.assertFalse(static_footprint_fits_map(grid_metadata(old_grid)))

    def test_pairing_rejects_adjacent_generation_and_selects_exact_identity(self):
        runner = self.runner()
        runner.grid_cache = deque()
        runner.status_cache = deque()
        matching_grid = make_grid(width=66, height=60, origin_x=-0.30, generation=100, stamp=20.0)
        adjacent_grid = make_grid(width=66, height=60, origin_x=-0.30, generation=101, stamp=20.5)
        matching_status = qualified_status(matching_grid, generation=100)
        matching_status["input_freshness_window_sec"] = 5.0
        adjacent_status = qualified_status(adjacent_grid, generation=101)
        adjacent_status["input_freshness_window_sec"] = 5.0
        now = self.runner_module.time.monotonic()
        runner.grid_cache.extend(((now, matching_grid),))
        runner.status_cache.extend(((now, adjacent_status), (now, matching_status)))
        grid, status = runner.matching_grid_status_pair()
        self.assertIs(grid, matching_grid)
        self.assertEqual(status["content_generation_id"], 100)
        runner.status_cache.clear()
        runner.status_cache.append((now, adjacent_status))
        self.assertIsNone(runner.matching_grid_status_pair())

    def test_ros_float32_resolution_policy_and_static_qualified_status(self):
        grid = make_grid(width=66, height=60, resolution=0.05000000074505806, origin_x=-0.30)
        status = qualified_status(grid)
        status["resolution"] = 0.05
        self.assertEqual(qualified_for_navigation(grid, status), (True, []))
        status["resolution"] = 0.10
        self.assertIn("status_grid_resolution_mismatch", qualified_for_navigation(grid, status)[1])

    def test_l3v_derives_qualified_static_status_from_real_conditions(self):
        node = object.__new__(self.l3v.L3VLocalTraversabilityNode)
        node.rows, node.cols, node.resolution = 66, 60, 0.05
        node.x_min, node.x_max, node.y_min, node.y_max, node.frame_id = -0.30, 3.0, -1.5, 1.5, "base"
        node.input_stale_timeout_sec = 2.0
        node.latest_odom = {"stamp": 1.0}
        node.last_odom_wall_time = 100.0
        node.last_lidar_wall_time = 100.0
        node.last_depth_points_wall_time = None
        node.last_depth_image_wall_time = None
        node.use_rgbd_obstacle_evidence = False
        node.doorway_provenance_enabled = False
        node.motion_evidence_invalidated = False
        node.motion_evidence_reason = None
        node.motion_evidence_max_translation_m = 1.0
        node.motion_evidence_max_yaw_delta_rad = math.pi / 2.0
        node.doorway_lidar_occ_meta = {}
        node.doorway_current_direct_lidar = {}
        node.latest_depth_valid_ratio = 0.0
        node.latest_lidar_source_stamp = 1.0
        node.latest_odom_source_stamp = 1.0
        node.pending_time_regression_reasons = []
        node.pending_tf_failure_reasons = []
        node.content_generation_id = 0
        node.producer_instance_id = "fixture-producer"
        node.reset_evidence()
        with patch.object(self.l3v.rospy.Time, "now", return_value=self.l3v.rospy.Time.from_sec(10.0)):
            node.build_content_locked(100.0)
        self.assertEqual((node.cached_grid.info.width, node.cached_grid.info.height), (66, 60))
        self.assertTrue(node.cached_status["static_footprint_fits_map"])
        self.assertFalse(node.cached_status["diagnostic_only"])
        self.assertEqual(node.cached_status["rejection_reasons"], [])
        self.assertEqual(qualified_for_navigation(node.cached_grid, node.cached_status), (True, []))
        self.assertEqual(self.runner().grid_qualification_errors(node.cached_grid, node.cached_status), [])

    def test_unknown_invalid_and_bounds_are_motion_blocked(self):
        grid = make_grid()
        runner = self.runner()
        unknown = metric_to_cell(0.20, 0.0, grid_metadata(grid))
        set_cell(grid, unknown[0], unknown[1], -1)
        blocked = runner.inflate_obstacles(runner.grid_array(grid))
        self.assertTrue(blocked[unknown[1], unknown[0]])
        self.assertFalse(runner.collision_free_arc(grid, blocked, 0.3, 0.0)[0])
        self.assertIsNone(metric_to_cell(3.0, 0.0, grid_metadata(grid)))
        self.assertFalse(self.state.grid_line_pass(runner.grid_array(grid), blocked, grid, (3.0, 0.0), 0.05)["pass"])
        invalid = make_grid()
        invalid.data[0] = 42
        self.assertRaises(ValueError, runner.grid_array, invalid)
        malformed = make_grid()
        malformed.data.pop()
        self.assertRaises(ValueError, self.state.grid_array, malformed)

    def test_tf_failure_has_no_coordinate_fallback(self):
        node = object.__new__(self.l3v.L3VLocalTraversabilityNode)
        node.frame_id = "base"
        node.lookup_to_base = lambda _frame: (None, None, False)
        point, reason = node.transform_to_base((1.0, 2.0, 3.0), "camera_optical_frame")
        self.assertIsNone(point)
        self.assertEqual(reason, "tf_unavailable:camera_optical_frame")

    def test_status_pair_generation_heartbeat_and_fail_closed_cases(self):
        grid = make_grid(generation=7, stamp=25.0)
        status = qualified_status(grid)
        self.assertEqual(qualified_for_navigation(grid, status), (True, []))
        heartbeat = dict(status)
        heartbeat["status_heartbeat_stamp"] = 26.0
        heartbeat["publication_sequence"] = 99
        self.assertEqual(qualified_for_navigation(grid, heartbeat), (True, []))
        cases = {
            "generation_mismatch": {"content_generation_id": 8},
            "tf_invalid": {"tf_valid": False},
            "diagnostic": {"diagnostic_only": True},
            "unsafe": {"safe_for_navigation": False},
            "time_rollback": {"input_time_monotonic": False},
            "old_status_new_grid": {"grid_content_stamp": 24.0},
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                bad = dict(status)
                bad.update(change)
                self.assertFalse(qualified_for_navigation(grid, bad)[0])
        next_grid = make_grid(generation=8, stamp=26.0)
        self.assertFalse(qualified_for_navigation(next_grid, status)[0])

    def test_centerline_regression_uses_standard_free_cells(self):
        grid = make_grid()
        runner = self.runner()
        status = qualified_status(grid)
        self.assertEqual(runner.grid_qualification_errors(grid, status), [])
        array = runner.grid_array(grid)
        blocked = runner.inflate_obstacles(array)
        start = runner.local_xy_to_cell(0.0, 0.0, grid)
        goal = runner.local_xy_to_cell(1.0, 0.0, grid)
        path = runner.block_astar(blocked, start, goal, grid)
        self.assertGreaterEqual(len(path), 2)
        self.assertTrue(all(array[y_index, x_index] == 0 for x_index, y_index in path))

    def test_start_clearance_t1_clears_only_circle_unknown_in_planning_copy(self):
        runner, grid, raw, blocked, occupied_inflated = self.clearance_fixture()
        planning, report = self.apply_clearance(runner, grid, raw, blocked, occupied_inflated)
        self.assertTrue(report["applied"])
        self.assertEqual(report["reason"], "APPLIED")
        self.assertEqual(report["raw_unknown_inside_count"], report["eligible_unknown_cleared_count"])
        self.assertEqual(report["retained_unknown_inside_count"], 0)
        self.assertTrue(all(
            not planning[y_index, x_index]
            for y_index in range(raw.shape[0]) for x_index in range(raw.shape[1])
            if raw[y_index, x_index] == -1
            and sum(value * value for value in runner.cell_to_local_xy((x_index, y_index), grid))
            <= STATIC_PLANNING_FOOTPRINT_RADIUS_M ** 2
        ))

    def test_start_clearance_t2_keeps_unknown_outside_circle_blocked(self):
        runner, grid, raw, blocked, occupied_inflated = self.clearance_fixture()
        outside = runner.local_xy_to_cell(-0.275, 0.0, grid)
        self.assertIsNotNone(outside)
        set_cell(grid, outside[0], outside[1], -1)
        raw = runner.grid_array(grid)
        planning, _report = self.apply_clearance(
            runner, grid, raw, runner.inflate_obstacles(raw), runner.occupied_inflated_mask(raw)
        )
        self.assertTrue(planning[outside[1], outside[0]])

    def test_start_clearance_t3_keeps_raw_occupied_inside_circle_blocked(self):
        runner, grid, raw, _blocked, _occupied_inflated = self.clearance_fixture()
        inside = runner.local_xy_to_cell(0.0, 0.0, grid)
        self.assertIsNotNone(inside)
        set_cell(grid, inside[0], inside[1], 100)
        raw = runner.grid_array(grid)
        planning, report = self.apply_clearance(
            runner, grid, raw, runner.inflate_obstacles(raw), runner.occupied_inflated_mask(raw)
        )
        self.assertTrue(planning[inside[1], inside[0]])
        self.assertGreaterEqual(report["raw_occupied_inside_count"], 1)

    def test_start_clearance_t4_keeps_occupied_inflation_inside_circle_blocked(self):
        runner, grid, raw, _blocked, _occupied_inflated = self.clearance_fixture()
        obstacle = runner.local_xy_to_cell(0.275, 0.0, grid)
        protected_unknown = runner.local_xy_to_cell(0.025, 0.025, grid)
        self.assertIsNotNone(obstacle)
        self.assertIsNotNone(protected_unknown)
        set_cell(grid, obstacle[0], obstacle[1], 100)
        raw = runner.grid_array(grid)
        occupied_inflated = runner.occupied_inflated_mask(raw)
        planning, report = self.apply_clearance(runner, grid, raw, runner.inflate_obstacles(raw), occupied_inflated)
        self.assertTrue(occupied_inflated[protected_unknown[1], protected_unknown[0]])
        self.assertTrue(planning[protected_unknown[1], protected_unknown[0]])
        self.assertGreater(report["occupied_inflated_inside_count"], 0)

    def test_start_clearance_t5_preserves_raw_grid_shape_and_content_hash(self):
        runner, grid, raw, blocked, occupied_inflated = self.clearance_fixture()
        raw_shape = raw.shape
        raw_hash = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
        _planning, report = self.apply_clearance(runner, grid, raw, blocked, occupied_inflated)
        self.assertEqual(raw.shape, raw_shape)
        self.assertEqual(hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(), raw_hash)
        self.assertTrue(report["raw_grid_unchanged"])

    def test_start_clearance_t6_084_start_block_becomes_free_and_block_astar_runs(self):
        runner, grid, raw, blocked, occupied_inflated = self.clearance_fixture()
        runner.args.block_size_cells = 4
        planning, report = self.apply_clearance(runner, grid, raw, blocked, occupied_inflated)
        start = runner.local_xy_to_cell(0.0, 0.0, grid)
        goal = runner.local_xy_to_cell(1.0, 0.0, grid)
        self.assertFalse(report["start_block_before"]["block_free"])
        self.assertTrue(report["start_block_after"]["block_free"])
        self.assertGreaterEqual(len(runner.block_astar(planning, start, goal, grid)), 2)

    def test_start_clearance_t7_free_path_has_dwa_candidate(self):
        runner, grid, raw, blocked, occupied_inflated = self.clearance_fixture()
        runner.args = self.runner_module.build_arg_parser().parse_args([])
        runner.prev_cmd = (0.0, 0.0)
        planning, _report = self.apply_clearance(runner, grid, raw, blocked, occupied_inflated)
        _v, _w, dwa = runner.choose_dwa(grid, planning, (1.0, 0.0), (1.0, 0.0), 1.0)
        self.assertFalse(dwa["blocked"])
        self.assertGreater(dwa["sample_count"], 0)

    def test_start_clearance_t8_is_nonpublishing_planner_only_operation(self):
        runner, grid, raw, blocked, occupied_inflated = self.clearance_fixture()
        self.assertFalse(hasattr(runner, "pub"))
        _planning, report = self.apply_clearance(runner, grid, raw, blocked, occupied_inflated)
        self.assertFalse(hasattr(runner, "pub"))
        self.assertTrue(report["raw_grid_unchanged"])

    def test_start_clearance_fails_closed_without_occupied_inflated_mask(self):
        runner, grid, raw, blocked, _occupied_inflated = self.clearance_fixture()
        planning, report = runner.apply_start_footprint_clearance(
            grid, raw, blocked, None, qualification_passed=True
        )
        self.assertEqual(report["reason"], "GRID_OR_MASK_INVALID")
        self.assertTrue(np.array_equal(planning, blocked))


if __name__ == "__main__":
    unittest.main()
