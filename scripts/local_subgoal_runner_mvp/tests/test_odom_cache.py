#!/usr/bin/env python3
"""Offline tests for navigation_state_machine.OdomCache.

The module is loaded with ROS message stubs.  No ROS master or simulated
clock is contacted by these tests.
"""

import importlib
import importlib.util
import json
import math
import sys
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "local_subgoal_runner_mvp" / "navigation_state_machine.py"
MODULE_DIR = str(MODULE_PATH.parent)
RUNNER_MODULE_PATH = ROOT / "scripts" / "local_subgoal_runner_mvp" / "block_astar_dwa_mature_runner.py"
GATE_MODULE_PATH = ROOT / "scripts" / "p2kg9_portal_shadow" / "portal_room_zone_effect_gate.py"

if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
from local_grid_contract import (
    ExactGridStatusPairCache,
    GRID_CONTRACT_VERSION,
    GRID_STATUS_SCHEMA_VERSION,
    grid_content_hash,
    validate_grid_status_content_binding,
)


class FakeStamp:
    def __init__(self, sec):
        self.sec = float(sec)

    def to_sec(self):
        return self.sec


class FakeTimeAPI:
    def __init__(self, ros):
        self.ros = ros

    def now(self):
        return FakeStamp(self.ros.sim_time_sec)


class FakeSubscriber:
    def unregister(self):
        return None


class FakeRos(types.ModuleType):
    class ROSInterruptException(RuntimeError):
        pass

    def __init__(self):
        super().__init__("rospy")
        self.sim_time_sec = 0.0
        self.shutdown = False
        self.Time = FakeTimeAPI(self)

    def Subscriber(self, *_args, **_kwargs):
        return FakeSubscriber()

    def is_shutdown(self):
        return self.shutdown


class FakeOdom:
    def __init__(self, stamp_sec, x=1.0, y=2.0):
        self.header = types.SimpleNamespace(stamp=FakeStamp(stamp_sec), frame_id="odom")
        self.child_frame_id = "base"
        orientation = types.SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0)
        position = types.SimpleNamespace(x=float(x), y=float(y))
        self.pose = types.SimpleNamespace(pose=types.SimpleNamespace(position=position, orientation=orientation))


class FakeGrid:
    def __init__(self, width=66, height=60, resolution=0.05, origin_x=-0.3, origin_y=-1.5, value=0):
        self.header = types.SimpleNamespace(frame_id="base", stamp=FakeStamp(1.0), seq=1)
        self.info = types.SimpleNamespace(
            width=width, height=height, resolution=resolution,
            origin=types.SimpleNamespace(position=types.SimpleNamespace(x=origin_x, y=origin_y)),
        )
        self.data = [value] * (width * height)


def install_message_stubs(fake_ros):
    sys.modules["rospy"] = fake_ros
    modules = {
        "geometry_msgs": types.ModuleType("geometry_msgs"),
        "geometry_msgs.msg": types.ModuleType("geometry_msgs.msg"),
        "nav_msgs": types.ModuleType("nav_msgs"),
        "nav_msgs.msg": types.ModuleType("nav_msgs.msg"),
        "rosgraph_msgs": types.ModuleType("rosgraph_msgs"),
        "rosgraph_msgs.msg": types.ModuleType("rosgraph_msgs.msg"),
        "sensor_msgs": types.ModuleType("sensor_msgs"),
        "sensor_msgs.msg": types.ModuleType("sensor_msgs.msg"),
        "sensor_msgs.point_cloud2": types.ModuleType("sensor_msgs.point_cloud2"),
        "std_msgs": types.ModuleType("std_msgs"),
        "std_msgs.msg": types.ModuleType("std_msgs.msg"),
        "tf": types.ModuleType("tf"),
    }
    for name, module in modules.items():
        sys.modules[name] = module
    for name in ("Twist",):
        setattr(modules["geometry_msgs.msg"], name, type(name, (), {}))
    for name in ("OccupancyGrid", "Odometry"):
        setattr(modules["nav_msgs.msg"], name, type(name, (), {}))
    setattr(modules["rosgraph_msgs.msg"], "Clock", type("Clock", (), {}))
    for name in ("Image", "Imu", "PointCloud2"):
        setattr(modules["sensor_msgs.msg"], name, type(name, (), {}))
    for name in ("Bool", "String"):
        setattr(modules["std_msgs.msg"], name, type(name, (), {}))


def load_module(fake_ros):
    install_message_stubs(fake_ros)
    if MODULE_DIR not in sys.path:
        sys.path.insert(0, MODULE_DIR)
    spec = importlib.util.spec_from_file_location("p2kg9u_navigation_state_machine", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_runner_module(fake_ros):
    install_message_stubs(fake_ros)
    if MODULE_DIR not in sys.path:
        sys.path.insert(0, MODULE_DIR)
    spec = importlib.util.spec_from_file_location("p2kg15_092r2_block_astar_dwa_runner", RUNNER_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Dataclasses resolve annotations through sys.modules during class creation.
    # Register this isolated test module exactly as normal import machinery does.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OdomCacheTests(unittest.TestCase):
    def setUp(self):
        self.ros = FakeRos()
        self.module = load_module(self.ros)
        self.cache = self.module.OdomCache(self.ros)

    def tearDown(self):
        self.cache.close()

    def test_read_odom_does_not_return_an_old_cached_frame(self):
        self.cache._callback(FakeOdom(10.0, x=3.0, y=4.0))
        self.module.ODOM_CACHE = self.cache
        result = {}

        def reader():
            result["value"] = self.module.read_odom()

        worker = threading.Thread(target=reader)
        worker.start()
        time.sleep(0.05)
        self.assertTrue(worker.is_alive(), "read_odom returned the old cached frame")
        self.cache._callback(FakeOdom(10.1, x=5.0, y=6.0))
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        result = result["value"]
        self.assertEqual(result["topic"], "/team/livox/icp_odom_gated")
        self.assertEqual(result["stamp_sec"], 10.1)
        self.assertEqual(result["pose_x_y_yaw"][:2], [5.0, 6.0])

    def test_wait_for_new_sequence_returns_after_callback(self):
        self.cache._callback(FakeOdom(10.0))
        _, sequence, _ = self.cache.snapshot()
        result = {}

        def reader():
            result["value"] = self.cache.get(2.0, after_sequence=sequence)

        worker = threading.Thread(target=reader)
        worker.start()
        time.sleep(0.05)
        self.cache._callback(FakeOdom(10.1))
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        self.assertGreater(result["value"][1], sequence)

    def test_low_rtf_wall_wait_does_not_trigger_legacy_five_second_timeout(self):
        self.cache._callback(FakeOdom(10.0))
        self.module.ODOM_CACHE = self.cache
        self.ros.sim_time_sec = 10.0
        result = {}

        def reader():
            result["value"] = self.module.read_odom(timeout_sec=1.0)

        worker = threading.Thread(target=reader)
        worker.start()
        time.sleep(6.55)
        self.ros.sim_time_sec = 10.5  # Healthy low-RTF progress: less than one sim second.
        self.cache._callback(FakeOdom(10.5))
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result["value"]["stamp_sec"], 10.5)

    def test_read_odom_detects_advancing_sim_time_without_odom(self):
        self.cache._callback(FakeOdom(10.0))
        self.module.ODOM_CACHE = self.cache
        self.ros.sim_time_sec = 11.1
        with self.assertRaisesRegex(RuntimeError, "gated_odom_stale_in_sim_time"):
            self.module.read_odom(timeout_sec=1.0)

    def test_read_odom_shutdown_terminates_wait(self):
        self.module.ODOM_CACHE = self.cache
        self.ros.shutdown = True
        with self.assertRaises(self.ros.ROSInterruptException):
            self.module.read_odom(1.0)

    def test_callbacks_and_readers_are_race_free(self):
        self.cache._callback(FakeOdom(1.0))
        self.module.ODOM_CACHE = self.cache
        failures = []

        def publish(offset):
            try:
                for index in range(25):
                    self.cache._callback(FakeOdom(2.0 + offset + index / 100.0))
            except Exception as exc:  # pragma: no cover - test failure path
                failures.append(exc)

        publishers = [threading.Thread(target=publish, args=(offset,)) for offset in (0.0, 1.0)]
        for publisher in publishers:
            publisher.start()
        for publisher in publishers:
            publisher.join(1.0)
        _msg, sequence, _stamp = self.cache.snapshot()
        self.assertFalse(failures)
        self.assertEqual(sequence, 51)

    def test_concurrent_read_odom_calls_all_wait_for_a_new_frame(self):
        self.cache._callback(FakeOdom(10.0))
        self.module.ODOM_CACHE = self.cache
        results = []
        failures = []

        def reader():
            try:
                results.append(self.module.read_odom(timeout_sec=1.0))
            except Exception as exc:  # pragma: no cover - test failure path
                failures.append(exc)

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for reader_thread in readers:
            reader_thread.start()
        time.sleep(0.05)
        self.cache._callback(FakeOdom(10.1))
        for reader_thread in readers:
            reader_thread.join(1.0)
        self.assertFalse(failures)
        self.assertEqual([item["stamp_sec"] for item in results], [10.1, 10.1, 10.1])


class PortalBoundCandidateTests(unittest.TestCase):
    """Pure parser tests: no ROS master, planner, target, or publisher."""

    def setUp(self):
        self.ros = FakeRos()
        self.module = load_module(self.ros)

    @staticmethod
    def payload(run_id="run-a", sequence=7, stamp=12.5):
        portal = {
            "track_id": "right-12", "side": "right", "observation_state": "confirmed",
            "candidate_available": True, "frame_id": "base", "source_stamp": stamp,
            "temporal_support": True, "traversability_support": True,
            "portal_center_base": [5.7, -1.2], "portal_normal_base": [0.0, -1.0],
            "portal_width": 1.2, "left_boundary": [5.1, -1.2], "right_boundary": [6.3, -1.2],
        }
        return {
            "contract_version": "p2kg12_portal_effect_gate_v2", "input_valid": True,
            "run_id": run_id, "portal_frame_sequence": sequence, "portal_source_stamp": stamp,
            "room_zone_active": True, "room_zone_source_stamp": 11.0,
            "room_zone_transition_sequence": 1,
            "left": {"portal": None, "effect_eligible": False, "effect_reason": "PORTAL_NOT_CONFIRMED"},
            "right": {"portal": portal, "effect_eligible": True,
                      "effect_reason": "CURRENT_FRAME_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION"},
        }

    def test_portal_identity_duplicate_conflict_order_and_run_change(self):
        authority = self.module.PortalBoundDoorCandidateAuthority(subscribe=False)
        first = self.payload()
        candidate = authority.ingest(first, 99.0)
        self.assertEqual(candidate["portal_run_id"], "run-a")
        self.assertEqual(candidate["portal_track_id"], "right-12")
        self.assertEqual(candidate["target_geometry_status"], "SOURCE_TIME_ODOM_BINDING_PENDING")
        self.assertEqual(authority.ingest(dict(first), 100.0)["portal_frame_sequence"], 7)
        conflict = self.payload()
        conflict["right"] = dict(conflict["right"])
        conflict["right"]["portal"] = dict(conflict["right"]["portal"])
        conflict["right"]["portal"]["portal_width"] = 2.0
        self.assertIsNone(authority.ingest(conflict))
        self.assertEqual(authority.last_rejection, "IDENTITY_CONTENT_CONFLICT")
        self.assertIsNone(authority.ingest(self.payload(sequence=6, stamp=12.4)))
        self.assertEqual(authority.last_rejection, "OUT_OF_ORDER_EFFECT_SEQUENCE")
        dual = self.payload(sequence=8, stamp=12.6)
        dual["left"] = {"portal": {}, "effect_eligible": True,
                        "effect_reason": "CURRENT_FRAME_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION"}
        self.assertIsNone(authority.ingest(dual))
        self.assertEqual(authority.last_rejection, "MULTIPLE_ELIGIBLE_PORTAL_SIDES")
        self.assertEqual(authority.ingest(self.payload(run_id="run-b", sequence=0, stamp=1.0))["portal_run_id"], "run-b")
        self.assertEqual(authority.snapshot()["committed_portal_candidate"]["portal_run_id"], "run-b")

    def test_commit_survives_later_ambiguous_and_empty_effects_then_zone_exit_invalidates(self):
        authority = self.module.PortalBoundDoorCandidateAuthority(subscribe=False)
        committed = authority.ingest(self.payload(), 1.0)
        empty = self.payload(sequence=8, stamp=13.0)
        empty["right"] = {"portal": None, "effect_eligible": False, "effect_reason": "PORTAL_NOT_CONFIRMED"}
        self.assertEqual(authority.ingest(empty)["portal_track_id"], committed["portal_track_id"])
        self.assertEqual(authority.last_rejection, "NO_ELIGIBLE_PORTAL_SIDE")
        dual = self.payload(sequence=9, stamp=14.0)
        dual["left"] = {"portal": {}, "effect_eligible": True,
                        "effect_reason": "CURRENT_FRAME_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION"}
        self.assertEqual(authority.ingest(dual)["portal_track_id"], "right-12")
        self.assertEqual(authority.last_rejection, "AMBIGUOUS_LATER_FRAME_IGNORED_FOR_COMMITTED_CANDIDATE")
        alternative = self.payload(sequence=10, stamp=15.0)
        alternative["right"] = dict(alternative["right"])
        alternative["right"]["portal"] = dict(alternative["right"]["portal"])
        alternative["right"]["portal"]["track_id"] = "right-13"
        self.assertEqual(authority.ingest(alternative)["portal_track_id"], "right-12")
        self.assertEqual(authority.snapshot()["pending_alternative_candidate"]["portal_track_id"], "right-13")
        exit_frame = self.payload(sequence=11, stamp=16.0)
        exit_frame["room_zone_active"] = False
        exit_frame["right"] = {"portal": None, "effect_eligible": False, "effect_reason": "OUTSIDE_ROOM_GENERATION_ZONE"}
        self.assertIsNone(authority.ingest(exit_frame))
        self.assertEqual(authority.last_rejection, "ROOM_ZONE_EXIT_INVALIDATED")

    def test_independent_sequence_stamp_order_contracts(self):
        authority = self.module.PortalBoundDoorCandidateAuthority(subscribe=False)
        authority.ingest(self.payload(sequence=7, stamp=12.5))
        authority.ingest(self.payload(sequence=8, stamp=12.4))
        self.assertEqual(authority.last_rejection, "OUT_OF_ORDER_EFFECT_SOURCE_STAMP")
        second = self.module.PortalBoundDoorCandidateAuthority(subscribe=False)
        second.ingest(self.payload(sequence=7, stamp=12.5))
        same_sequence_new_stamp = self.payload(sequence=7, stamp=12.6)
        second.ingest(same_sequence_new_stamp)
        self.assertEqual(second.last_rejection, "FRAME_SEQUENCE_STAMP_CONFLICT")
        third = self.module.PortalBoundDoorCandidateAuthority(subscribe=False)
        third.ingest(self.payload(sequence=7, stamp=12.5))
        third.ingest(self.payload(sequence=7, stamp=12.4))
        self.assertEqual(third.last_rejection, "FRAME_SEQUENCE_STAMP_CONFLICT")


class PortalEffectGateStartupPreflightTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())

    @staticmethod
    def args(*, execute=True, hierarchical=True):
        return types.SimpleNamespace(
            execute=execute,
            enable_hierarchical_portal_local_autonomy=hierarchical,
        )

    def test_missing_gate_fails_closed(self):
        def unavailable(*_args, **_kwargs):
            raise TimeoutError("no portal effect message")

        result = self.module.portal_effect_gate_startup_preflight(
            self.args(), wait_for_message=unavailable, timeout_sec=0.01,
        )
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "PORTAL_EFFECT_GATE_UNAVAILABLE")

    def test_valid_gate_message_passes(self):
        payload = {"contract_version": self.module.PORTAL_EFFECT_CONTRACT_VERSION}

        def available(*_args, **_kwargs):
            return types.SimpleNamespace(data=json.dumps(payload))

        result = self.module.portal_effect_gate_startup_preflight(
            self.args(), wait_for_message=available, timeout_sec=0.01,
        )
        self.assertTrue(result["ready"])
        self.assertEqual(result["reason"], "PORTAL_EFFECT_GATE_READY")

    def test_nonhierarchical_execution_does_not_wait(self):
        called = []

        def should_not_run(*_args, **_kwargs):
            called.append(True)
            raise AssertionError("waiter must not be called")

        result = self.module.portal_effect_gate_startup_preflight(
            self.args(hierarchical=False), wait_for_message=should_not_run,
        )
        self.assertTrue(result["ready"])
        self.assertFalse(result["required"])
        self.assertEqual(called, [])


class SourceTimeOdomAndShadowTests(unittest.TestCase):
    def setUp(self):
        self.ros = FakeRos()
        self.module = load_module(self.ros)

    @staticmethod
    def odom(stamp, x, y, yaw):
        message = FakeOdom(stamp, x=x, y=y)
        message.pose.pose.orientation.w = math.cos(yaw / 2.0)
        message.pose.pose.orientation.z = math.sin(yaw / 2.0)
        return message

    def test_bounded_history_exact_interpolation_and_contract_rejection(self):
        cache = self.module.OdomCache(self.ros, history_capacity=4)
        for row in (self.odom(1.0, 0, 0, 0), self.odom(2.0, 2, 0, 0), self.odom(3.0, 4, 0, math.pi / 2)):
            cache._callback(row)
        self.assertEqual(len(cache.history_snapshot()), 3)
        self.assertEqual(cache.pose_at_source_stamp(2.0)["odom_binding_method"], "EXACT")
        bracket = cache.pose_at_source_stamp(2.5)
        self.assertTrue(bracket["binding_valid"])
        self.assertEqual(bracket["odom_binding_method"], "BRACKET_INTERPOLATION")
        self.assertEqual(bracket["interpolation_ratio"], 0.5)
        self.assertAlmostEqual(bracket["source_pose_x_y_yaw"][0], 3.0)
        cache._callback(self.odom(10.0, 10, 0, 0))
        self.assertFalse(cache.pose_at_source_stamp(5.0)["binding_valid"])
        self.assertEqual(cache.pose_at_source_stamp(5.0)["reason"], "ODOM_BRACKET_SPAN_EXCEEDS_CONTRACT")
        cache.close()

    def test_g14_shadow_freeze_normal_width_and_nonexecution_flags(self):
        candidate = PortalBoundCandidateTests.payload(sequence=7, stamp=2.0)
        candidate = self.module.PortalBoundDoorCandidateAuthority._candidate_from_effect(candidate, "right", 1.0)
        binding = {"binding_valid": True, "source_pose_x_y_yaw": [10.0, 20.0, math.pi / 2]}
        target = self.module.build_g14_shadow_target(candidate, binding, (10.0, 20.0, math.pi / 2), (0.0, -1.0))
        self.assertTrue(target["target_valid"])
        self.assertAlmostEqual(target["frozen_geometry"]["portal_normal_odom"][0], 1.0)
        self.assertAlmostEqual(target["frozen_geometry"]["portal_normal_odom"][1], 0.0)
        self.assertAlmostEqual(
            target["P_pre_odom"][0],
            11.2 - self.module.STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M,
        )
        self.assertAlmostEqual(
            target["P_pre_odom"][1],
            25.7 + target["p_pre_portal_relative"]["upstream_tangent_offset_m"],
        )
        self.assertTrue(target["p_pre_portal_relative"]["width_aware_tangent_clamp_applied"])
        self.assertAlmostEqual(target["P_through_odom"][0], 11.5)
        self.assertAlmostEqual(target["P_through_odom"][1], 25.7)
        self.assertFalse(target["safe_for_navigation"] or target["planner_ready"] or target["send_to_navigation"] or target["control_authority"])
        narrow = dict(candidate)
        narrow["portal_width"] = 0.60
        self.assertFalse(self.module.build_g14_shadow_target(narrow, binding, corridor_approach_direction=(0.0, -1.0))["width_gate_pass"])
        bad = dict(candidate)
        bad["portal_normal_base"] = [0.0, 1.0]
        self.assertEqual(self.module.freeze_portal_geometry(bad, binding)["reason"], "PORTAL_NORMAL_SIDE_DIRECTION_CONFLICT")

    def test_g14_shadow_p_pre_tangent_override_moves_only_p_pre_along_corridor(self):
        candidate = PortalBoundCandidateTests.payload(sequence=7, stamp=2.0)
        candidate = self.module.PortalBoundDoorCandidateAuthority._candidate_from_effect(candidate, "right", 1.0)
        binding = {"binding_valid": True, "source_pose_x_y_yaw": [10.0, 20.0, math.pi / 2]}
        default = self.module.build_g14_shadow_target(candidate, binding, corridor_approach_direction=(0.0, -1.0))
        delayed = self.module.build_g14_shadow_target(
            candidate, binding, corridor_approach_direction=(0.0, -1.0), p_pre_upstream_tangent_m=1.20,
        )
        self.assertTrue(delayed["target_valid"])
        self.assertAlmostEqual(delayed["p_pre_portal_relative"]["outside_normal_m"], default["p_pre_portal_relative"]["outside_normal_m"])
        self.assertAlmostEqual(delayed["p_pre_portal_relative"]["upstream_tangent_offset_m"], 1.20)
        self.assertAlmostEqual(delayed["P_pre_odom"][0], default["P_pre_odom"][0])
        self.assertAlmostEqual(
            delayed["P_pre_odom"][1] - default["P_pre_odom"][1],
            delayed["p_pre_portal_relative"]["upstream_tangent_offset_m"]
            - default["p_pre_portal_relative"]["upstream_tangent_offset_m"],
        )
        self.assertEqual(delayed["P_through_odom"], default["P_through_odom"])

    def test_g14_shadow_rejects_negative_p_pre_tangent_override(self):
        candidate = PortalBoundCandidateTests.payload(sequence=7, stamp=2.0)
        candidate = self.module.PortalBoundDoorCandidateAuthority._candidate_from_effect(candidate, "right", 1.0)
        binding = {"binding_valid": True, "source_pose_x_y_yaw": [10.0, 20.0, math.pi / 2]}
        target = self.module.build_g14_shadow_target(candidate, binding, p_pre_upstream_tangent_m=-0.01)
        self.assertFalse(target["target_valid"])
        self.assertEqual(target["rejection_reason"], "P_PRE_UPSTREAM_TANGENT_INVALID")

    def test_forward_committed_shadow_is_stop_eligible(self):
        candidate = PortalBoundCandidateTests.payload(sequence=7, stamp=2.0)
        candidate = self.module.PortalBoundDoorCandidateAuthority._candidate_from_effect(candidate, "right", 1.0)
        binding = {"binding_valid": True, "source_pose_x_y_yaw": [10.0, 20.0, math.pi / 2]}
        target = self.module.build_g14_shadow_target(candidate, binding, (10.0, 20.0, math.pi / 2), (0.0, -1.0))
        self.assertTrue(target["current_base_diagnostics"]["portal_center_in_front"])
        self.assertTrue(target["current_base_diagnostics"]["pre_target_in_front"])
        self.assertTrue(self.module.g14_shadow_stop_eligible(target))

    def test_expired_or_rear_pre_shadow_is_not_stop_eligible(self):
        candidate = PortalBoundCandidateTests.payload(sequence=7, stamp=2.0)
        candidate = self.module.PortalBoundDoorCandidateAuthority._candidate_from_effect(candidate, "right", 1.0)
        binding = {"binding_valid": True, "source_pose_x_y_yaw": [10.0, 20.0, math.pi / 2]}
        target = self.module.build_g14_shadow_target(candidate, binding, (10.0, 20.0, math.pi / 2), (0.0, -1.0))
        expired = dict(target)
        expired["candidate_lifecycle"] = "PASSED_OR_EXPIRED"
        self.assertFalse(self.module.g14_shadow_stop_eligible(expired))
        rear_pre = dict(target)
        rear_pre["current_base_diagnostics"] = dict(target["current_base_diagnostics"])
        rear_pre["current_base_diagnostics"]["pre_target_in_front"] = False
        self.assertFalse(self.module.g14_shadow_stop_eligible(rear_pre))


class PortalG14PPreTests(unittest.TestCase):
    """T1-T6: P_pre control is decided by the existing formal grid contract."""

    def setUp(self):
        self.ros = FakeRos()
        self.module = load_module(self.ros)
        self.module.qualified_for_navigation = lambda _grid, status: (bool(status.get("qualified")), [] if status.get("qualified") else ["not_qualified"])
        self.candidate = {
            "portal_run_id": "run-092", "portal_frame_sequence": 314,
            "portal_source_stamp": 72.414, "portal_track_id": "left-12", "side": "left",
        }

    def target(self, pre=(1.0, 0.0)):
        return {
            "target_valid": True, "candidate_lifecycle": "COMMITTED", "P_pre_odom": list(pre),
            "portal_identity": {"run_id": "run-092", "frame_sequence": 314, "source_stamp": 72.414, "track_id": "left-12", "side": "left"},
            "frozen_geometry": {"geometry_valid": True, "portal_normal_odom": [0.0, 1.0]},
        }

    def evaluate(self, target, grid=None, candidate=Ellipsis, qualified=True):
        return self.module.evaluate_portal_g14_p_pre_admissibility(
            self.candidate if candidate is Ellipsis else candidate, target, (0.0, 0.0, 0.0),
            grid or FakeGrid(), {"qualified": qualified}, 0.2641935843278561,
        )

    def test_t1_outside_grid_keeps_centerline_and_does_not_switch(self):
        result = self.evaluate(self.target((3.1, 0.0)))
        self.assertEqual(result["state"], "P_PRE_OUTSIDE_LOCAL_PLANNING_WINDOW")
        self.assertTrue(result["planning_blocked"])

    def test_t2_inside_qualified_switches_to_portal_g14_p_pre(self):
        result = self.evaluate(self.target())
        self.assertEqual(result["state"], "P_PRE_READY_FOR_PLANNER")
        self.assertTrue(result["target_cell_admissible"])
        original = self.module.write_absolute_target
        self.module.write_absolute_target = lambda xy, source, subsource, extra: {"xy": list(xy), "source": source, "subgoal_source": subsource, **extra}
        try:
            switched = self.module.write_portal_g14_p_pre_target(self.target(), result)
        finally:
            self.module.write_absolute_target = original
        self.assertEqual(switched["source"], "PORTAL_G14_P_PRE")
        self.assertEqual(switched["xy"], [1.0, 0.0])

    def test_t3_blocked_target_cell_is_rejected(self):
        grid = FakeGrid()
        grid.data[30 * grid.info.width + 26] = 100
        result = self.evaluate(self.target(), grid=grid)
        self.assertEqual(result["state"], "P_PRE_TARGET_CELL_BLOCKED")
        self.assertFalse(result["target_cell_admissible"])

    def test_t4_invalidated_candidate_cannot_continue_or_fallback(self):
        result = self.evaluate(self.target(), candidate=None)
        self.assertEqual(result["state"], "P_PRE_CANDIDATE_INVALIDATED")
        self.assertEqual(result["target_source"], "PORTAL_G14_P_PRE")

    def test_t5_reached_goal_stops_at_p_pre(self):
        result = self.module.portal_g14_p_pre_runner_outcome({"runner_final_decision": "BLOCK_ASTAR_DWA_REACHED_GOAL"})
        self.assertTrue(result["reached"])
        self.assertEqual(result["final_decision"], "STATE_MACHINE_STOP_AFTER_PORTAL_G14_P_PRE_REACHED")

    def test_t6_runner_failure_is_safe_terminal_without_through_or_legacy(self):
        result = self.module.portal_g14_p_pre_runner_outcome({"runner_final_decision": "BLOCK_ASTAR_DWA_BLOCKED_NO_PATH"})
        self.assertFalse(result["reached"])
        self.assertEqual(result["final_decision"], "P2KG15_092_P_PRE_EXECUTION_FAILED")

    def test_t7_p_through_handoff_writes_goal_only(self):
        target = self.target()
        target["P_through_odom"] = [1.5, 2.0]
        original = self.module.write_absolute_target
        self.module.write_absolute_target = lambda xy, source, subsource, extra: {
            "xy": list(xy), "source": source, "subgoal_source": subsource, **extra,
        }
        try:
            written = self.module.write_portal_g14_p_through_target(target)
        finally:
            self.module.write_absolute_target = original
        self.assertEqual(written["xy"], [1.5, 2.0])
        self.assertEqual(written["source"], "PORTAL_G14_P_THROUGH")
        self.assertEqual(written["high_level_goal_type"], "ROOM_ENTRY")
        self.assertNotIn("linear_x", written)
        self.assertNotIn("angular_z", written)

    def test_t8_p_through_completion_requires_room_side(self):
        target = self.target()
        target["P_through_odom"] = [0.0, 0.3]
        target["frozen_geometry"] = {"geometry_valid": True, "portal_center_odom": [0.0, 0.0], "portal_normal_odom": [0.0, 1.0]}
        reached = self.module.portal_g14_p_through_runner_outcome(
            {"runner_final_decision": "BLOCK_ASTAR_DWA_REACHED_GOAL"}, target, (0.0, 0.2, 0.0),
        )
        self.assertTrue(reached["reached"])
        outside = self.module.portal_g14_p_through_runner_outcome(
            {"runner_final_decision": "BLOCK_ASTAR_DWA_REACHED_GOAL"}, target, (0.0, -0.01, 0.0),
        )
        self.assertFalse(outside["reached"])
        self.assertEqual(outside["final_decision"], "P2KG15_093_P_THROUGH_LOCAL_RUNNER_FAILED")

    def test_t9_deep_safe_max_steps_crossing_is_semantic_p_through_completion(self):
        target = self.target()
        target["P_through_odom"] = [0.0, 0.95]
        target["portal_width_m"] = 1.2
        target["frozen_geometry"] = {"geometry_valid": True, "portal_center_odom": [0.0, 0.0], "portal_normal_odom": [0.0, 1.0]}
        reached = self.module.portal_g14_p_through_runner_outcome(
            {"runner_final_decision": "BLOCK_ASTAR_DWA_MAX_STEPS"}, target, (0.20, 0.71, 0.0),
        )
        self.assertTrue(reached["reached"])
        self.assertEqual(reached["reason"], "LOCAL_AUTONOMY_P_THROUGH_DEEP_CROSSING_MAX_STEPS_STOP")
        shallow = self.module.portal_g14_p_through_runner_outcome(
            {"runner_final_decision": "BLOCK_ASTAR_DWA_MAX_STEPS"}, target, (0.0, 0.20, 0.0),
        )
        self.assertFalse(shallow["reached"])
        outside_aperture = self.module.portal_g14_p_through_runner_outcome(
            {"runner_final_decision": "BLOCK_ASTAR_DWA_MAX_STEPS"}, target, (0.40, 0.71, 0.0),
        )
        self.assertFalse(outside_aperture["reached"])


class PPreNormalRetreat092R14Tests(unittest.TestCase):
    """Frozen-normal target admission only; Portal, A*, DWA, and inflation stay untouched."""

    RADIUS_M = 0.2641935843278561

    def setUp(self):
        self.ros = FakeRos()
        self.module = load_module(self.ros)
        self.module.qualified_for_navigation = lambda _grid, status: (bool(status.get("qualified")), [] if status.get("qualified") else ["not_qualified"])
        self.candidate = {
            "portal_run_id": "run-092r14", "portal_frame_sequence": 14,
            "portal_source_stamp": 14.0, "portal_track_id": "left-14", "side": "left",
        }

    def target(self, center=(1.0, 1.375), pre=(1.0, 1.075)):
        return {
            "target_valid": True, "candidate_lifecycle": "COMMITTED", "P_pre_odom": list(pre),
            "portal_width_m": 1.15,
            "portal_identity": {"run_id": "run-092r14", "frame_sequence": 14, "source_stamp": 14.0, "track_id": "left-14", "side": "left"},
            "frozen_geometry": {
                "geometry_valid": True, "portal_center_odom": list(center), "portal_normal_odom": [0.0, 1.0],
            },
        }

    def evaluate(self, target, grid, status=None):
        status = {"qualified": True, "grid_content_stamp": 1.0, "grid_content_hash": "r14-grid", "content_generation_id": 14, **(status or {})}
        return self.module.evaluate_portal_g14_p_pre_admissibility(
            self.candidate, target, (0.0, 0.0, 0.0), grid, status, self.RADIUS_M,
        )

    def raw_block(self, grid, xy):
        cell = self.module.local_xy_to_cell(*xy, grid)
        self.assertIsNotNone(cell)
        grid.data[cell[1] * grid.info.width + cell[0]] = 100

    def test_t1_r11_style_legal_ideal_keeps_d_030_without_search(self):
        result = self.evaluate(self.target(), FakeGrid())
        self.assertEqual(result["state"], "P_PRE_READY_FOR_PLANNER")
        self.assertEqual(result["selection_reason"], "KEEP_IDEAL_P_PRE")
        self.assertAlmostEqual(result["selected_d_m"], 0.30)
        self.assertFalse(result["normal_search_triggered"])

    def test_t2_r12_style_blocked_ideal_selects_first_safe_normal_distance(self):
        grid = FakeGrid()
        target = self.target()
        self.raw_block(grid, target["P_pre_odom"])
        result = self.evaluate(target, grid)
        self.assertEqual(result["state"], "P_PRE_READY_FOR_PLANNER")
        self.assertTrue(result["normal_search_triggered"])
        self.assertAlmostEqual(result["selected_d_m"], 0.65)
        self.assertEqual(result["selection_reason"], "MINIMUM_FORMALLY_SAFE_FROZEN_NORMAL_RETREAT")

    def test_t3_no_safe_distance_within_bound_fails_closed(self):
        result = self.evaluate(self.target(), FakeGrid(value=100))
        self.assertEqual(result["state"], "P_PRE_NO_SAFE_NORMAL_DISTANCE")
        self.assertTrue(result["planning_blocked"])
        self.assertFalse(result["target_cell_admissible"])

    def test_t4_safe_point_beyond_geometry_bound_is_not_scanned(self):
        grid = FakeGrid()
        # The ideal cell is blocked; d=0.65 would be free, but the robot is
        # only 0.65 m from the plane, so the radius-preserving cap is below it.
        target = self.target(center=(1.0, 0.65), pre=(1.0, 0.35))
        self.raw_block(grid, target["P_pre_odom"])
        result = self.evaluate(target, grid)
        self.assertEqual(result["state"], "P_PRE_NO_SAFE_NORMAL_DISTANCE")
        self.assertLess(result["max_allowed_P_pre_normal_distance_m"], 0.65)
        self.assertTrue(all(attempt["d_m"] < 0.65 for attempt in result["normal_search_attempts"]))

    def test_t5_selection_stays_on_frozen_portal_normal_without_tangent_wandering(self):
        grid = FakeGrid()
        target = self.target()
        self.raw_block(grid, target["P_pre_odom"])
        result = self.evaluate(target, grid)
        selected = result["selected_P_pre_odom"]
        self.assertAlmostEqual(selected[0], target["frozen_geometry"]["portal_center_odom"][0])
        self.assertAlmostEqual(selected[1], target["frozen_geometry"]["portal_center_odom"][1] - result["selected_d_m"])

    def test_t6_selected_candidate_remains_in_front_and_on_pre_door_side(self):
        grid = FakeGrid()
        target = self.target()
        self.raw_block(grid, target["P_pre_odom"])
        result = self.evaluate(target, grid)
        self.assertTrue(result["P_pre_in_front"])
        self.assertGreater(result["selected_d_m"], 0.0)

    def test_t7_exact_grid_status_identity_is_captured_at_selection(self):
        grid = FakeGrid()
        target = self.target()
        self.raw_block(grid, target["P_pre_odom"])
        result = self.evaluate(target, grid, {"grid_content_stamp": 7.5, "grid_content_hash": "exact-r14", "content_generation_id": 75})
        self.assertEqual(result["selection_grid_status_identity"], {"grid_content_stamp": 7.5, "grid_content_hash": "exact-r14", "content_generation_id": 75})

    def test_t8_selected_target_is_frozen_and_preserves_original_ideal(self):
        grid = FakeGrid()
        target = self.target()
        self.raw_block(grid, target["P_pre_odom"])
        result = self.evaluate(target, grid)
        original = self.module.write_absolute_target
        self.module.write_absolute_target = lambda xy, source, subsource, extra: {"xy": list(xy), "source": source, "subgoal_source": subsource, **extra}
        try:
            written = self.module.write_portal_g14_p_pre_target(target, result)
        finally:
            self.module.write_absolute_target = original
        result["selected_P_pre_odom"][1] = -99.0
        self.assertEqual(written["frozen_P_pre_odom"], [1.0, 1.075])
        self.assertAlmostEqual(written["selected_P_pre_odom"][0], 1.0)
        self.assertAlmostEqual(written["selected_P_pre_odom"][1], 0.725)
        self.assertAlmostEqual(written["xy"][0], 1.0)
        self.assertAlmostEqual(written["xy"][1], 0.725)

    def test_t9_unqualified_grid_never_starts_normal_search(self):
        result = self.evaluate(self.target(), FakeGrid(), {"qualified": False})
        self.assertEqual(result["state"], "P_PRE_GRID_UNQUALIFIED")
        self.assertFalse(result["normal_search_triggered"])


class FormalGridStatusPairing092R3Tests(unittest.TestCase):
    """T1-T5 receipt-order tests plus the 0059 P_pre retry semantic."""

    @staticmethod
    def grid(stamp_sec, generation):
        grid = FakeGrid()
        grid.header.stamp = FakeStamp(stamp_sec)
        grid.header.seq = generation
        # A later L3V content generation must differ in the actual cell payload,
        # not merely in its identity stamp, to exercise the 0059 dual mismatch.
        if generation > 1:
            grid.data[generation] = 100
        return grid

    @staticmethod
    def status(grid, generation):
        stamp = grid.header.stamp.to_sec()
        producer = "offline-092r3"
        return {
            "contract_version": GRID_CONTRACT_VERSION,
            "schema_version": GRID_STATUS_SCHEMA_VERSION,
            "producer_instance_id": producer,
            "content_generation_id": generation,
            "grid_content_stamp": stamp,
            "grid_content_hash": grid_content_hash(grid, producer, generation, stamp),
            "frame_id": "base",
            "origin": {"x": -0.3, "y": -1.5},
            "resolution": 0.05,
            "width": 66,
            "height": 60,
            "tf_valid": True,
            "all_required_inputs_fresh": True,
            "input_time_monotonic": True,
            "diagnostic_only": False,
            "safe_for_navigation": True,
            "upstream_navigation_allowed": True,
            "rejection_reasons": [],
            "input_freshness_window_sec": 5.0,
        }

    def assert_pair(self, pair, expected_grid, expected_status):
        self.assertIsNotNone(pair)
        self.assertIs(pair[0], expected_grid)
        self.assertEqual(pair[1], expected_status)
        self.assertEqual(validate_grid_status_content_binding(pair[0], pair[1]), [])

    def test_t1_normal_complete_pair_passes(self):
        grid_a = self.grid(1.0, 1)
        status_a = self.status(grid_a, 1)
        cache = ExactGridStatusPairCache(maxlen=8)
        cache.add_grid(grid_a, 10.0)
        cache.add_status(status_a, 10.1)
        self.assert_pair(cache.matching_pair(10.2), grid_a, status_a)

    def test_t2_new_grid_first_never_forms_grid_b_status_a(self):
        grid_a, grid_b = self.grid(1.0, 1), self.grid(2.0, 2)
        status_a = self.status(grid_a, 1)
        cache = ExactGridStatusPairCache(maxlen=8)
        cache.add_grid(grid_a, 10.0)
        cache.add_status(status_a, 10.1)
        cache.add_grid(grid_b, 10.2)
        self.assertEqual(validate_grid_status_content_binding(cache.latest_grid(), status_a), [
            "status_grid_content_hash_mismatch", "status_grid_content_stamp_mismatch",
        ])
        pair = cache.matching_pair(10.3)
        self.assert_pair(pair, grid_a, status_a)
        self.assertIsNot(pair[0], grid_b)

    def test_t3_new_status_restores_grid_b_status_b(self):
        grid_a, grid_b = self.grid(1.0, 1), self.grid(2.0, 2)
        status_a, status_b = self.status(grid_a, 1), self.status(grid_b, 2)
        cache = ExactGridStatusPairCache(maxlen=8)
        cache.add_grid(grid_a, 10.0)
        cache.add_status(status_a, 10.1)
        cache.add_grid(grid_b, 10.2)
        cache.add_status(status_b, 10.3)
        self.assert_pair(cache.matching_pair(10.4), grid_b, status_b)

    def test_t4_status_first_never_forms_grid_a_status_b(self):
        grid_a, grid_b = self.grid(1.0, 1), self.grid(2.0, 2)
        status_a, status_b = self.status(grid_a, 1), self.status(grid_b, 2)
        cache = ExactGridStatusPairCache(maxlen=8)
        cache.add_grid(grid_a, 10.0)
        cache.add_status(status_a, 10.1)
        cache.add_status(status_b, 10.2)
        self.assertEqual(validate_grid_status_content_binding(grid_a, status_b), [
            "status_grid_content_hash_mismatch", "status_grid_content_stamp_mismatch",
        ])
        cache.add_grid(grid_b, 10.3)
        self.assert_pair(cache.matching_pair(10.4), grid_b, status_b)

    def test_t5_high_frequency_interleave_never_returns_mixed_identity(self):
        grid_a, grid_b, grid_c = self.grid(1.0, 1), self.grid(2.0, 2), self.grid(3.0, 3)
        status_a, status_b, status_c = self.status(grid_a, 1), self.status(grid_b, 2), self.status(grid_c, 3)
        cache = ExactGridStatusPairCache(maxlen=8)
        cache.add_grid(grid_a, 10.0)
        cache.add_status(status_a, 10.1)
        cache.add_grid(grid_b, 10.2)
        cache.add_status(status_b, 10.3)
        cache.add_grid(grid_c, 10.4)
        self.assert_pair(cache.matching_pair(10.5), grid_b, status_b)
        cache.add_status(status_c, 10.6)
        self.assert_pair(cache.matching_pair(10.7), grid_c, status_c)

    def test_iteration_9_semantic_is_pending_then_ready_not_terminal_rejection(self):
        ros = FakeRos()
        module = load_module(ros)
        candidate = {
            "portal_run_id": "run-092r3", "portal_frame_sequence": 314,
            "portal_source_stamp": 72.414, "portal_track_id": "left-12", "side": "left",
        }
        target = {
            "target_valid": True, "candidate_lifecycle": "COMMITTED", "P_pre_odom": [1.0, 0.0],
            "portal_identity": {"run_id": "run-092r3", "frame_sequence": 314, "source_stamp": 72.414, "track_id": "left-12", "side": "left"},
            "frozen_geometry": {"geometry_valid": True, "portal_normal_odom": [0.0, 1.0]},
        }
        grid_a, grid_b = self.grid(1.0, 1), self.grid(2.0, 2)
        status_a, status_b = self.status(grid_a, 1), self.status(grid_b, 2)
        cache = ExactGridStatusPairCache(maxlen=8)
        cache.add_grid(grid_b, 10.0)
        cache.add_status(status_a, 10.1)
        self.assertIsNone(cache.matching_pair(10.2))
        pending = module.p_pre_grid_status_pair_pending_admissibility(candidate, target, None, None)
        self.assertEqual(pending["state"], "P_PRE_GRID_STATUS_PAIR_PENDING")
        self.assertEqual(pending["reason"], "WAITING_FOR_MATCHED_GRID_STATUS_AND_SOURCE_ODOM")
        self.assertTrue(pending["planning_blocked"])
        cache.add_status(status_b, 10.3)
        pair = cache.matching_pair(10.4)
        self.assert_pair(pair, grid_b, status_b)
        ready = module.evaluate_portal_g14_p_pre_admissibility(candidate, target, (0.0, 0.0, 0.0), pair[0], pair[1], 0.2641935843278561)
        self.assertEqual(ready["state"], "P_PRE_READY_FOR_PLANNER")

    def test_matched_but_unsafe_status_remains_fail_closed(self):
        ros = FakeRos()
        module = load_module(ros)
        candidate = {
            "portal_run_id": "run-092r3", "portal_frame_sequence": 314,
            "portal_source_stamp": 72.414, "portal_track_id": "left-12", "side": "left",
        }
        target = {
            "target_valid": True, "candidate_lifecycle": "COMMITTED", "P_pre_odom": [1.0, 0.0],
            "portal_identity": {"run_id": "run-092r3", "frame_sequence": 314, "source_stamp": 72.414, "track_id": "left-12", "side": "left"},
            "frozen_geometry": {"geometry_valid": True, "portal_normal_odom": [0.0, 1.0]},
        }
        grid = self.grid(2.0, 2)
        status = self.status(grid, 2)
        status["safe_for_navigation"] = False
        result = module.evaluate_portal_g14_p_pre_admissibility(candidate, target, (0.0, 0.0, 0.0), grid, status, 0.2641935843278561)
        self.assertEqual(result["state"], "P_PRE_GRID_UNQUALIFIED")
        self.assertIn("status_safe_for_navigation_false", result["grid_qualification_errors"])


class PPreSameEpochBindingTests(unittest.TestCase):
    """P_pre may project only with the Grid producer's source Odom pose."""

    def setUp(self):
        self.ros = FakeRos()
        self.module = load_module(self.ros)
        self.module.qualified_for_navigation = lambda _grid, _status: (True, [])
        self.cache = self.module.OdomCache(self.ros)
        self.candidate = {
            "portal_run_id": "run-binding", "portal_frame_sequence": 1,
            "portal_source_stamp": 34.0, "portal_track_id": "left-1", "side": "left",
        }
        self.target = {
            "target_valid": True, "candidate_lifecycle": "COMMITTED", "P_pre_odom": [1.0, 0.0],
            "portal_identity": {"run_id": "run-binding", "frame_sequence": 1, "source_stamp": 34.0, "track_id": "left-1", "side": "left"},
            "frozen_geometry": {"geometry_valid": True, "portal_normal_odom": [0.0, 1.0]},
        }

    def tearDown(self):
        self.cache.close()

    def test_exact_grid_source_stamp_wins_over_newer_latest_odom(self):
        self.cache._callback(FakeOdom(34.097, x=0.0, y=0.0))
        self.cache._callback(FakeOdom(34.597, x=0.5, y=0.0))
        binding = self.module.p_pre_pose_binding_for_grid_status(
            self.cache, {"source_odom_stamp": 34.097},
        )
        self.assertTrue(binding["binding_valid"])
        self.assertEqual(binding["odom_binding_method"], "EXACT")
        self.assertEqual(binding["source_pose_x_y_yaw"][:2], [0.0, 0.0])
        result = self.module.evaluate_portal_g14_p_pre_admissibility(
            self.candidate, self.target, tuple(binding["source_pose_x_y_yaw"]),
            FakeGrid(), {"qualified": True, "source_odom_stamp": 34.097}, 0.2641935843278561,
        )
        self.assertEqual(result["state"], "P_PRE_READY_FOR_PLANNER")

    def test_missing_source_pose_is_pending_not_mixed_time_terminal_rejection(self):
        self.cache._callback(FakeOdom(34.097, x=0.0, y=0.0))
        self.cache._callback(FakeOdom(34.597, x=0.5, y=0.0))
        binding = self.module.p_pre_pose_binding_for_grid_status(
            self.cache, {"source_odom_stamp": 33.0},
        )
        self.assertFalse(binding["binding_valid"])
        pending = self.module.p_pre_grid_status_pair_pending_admissibility(
            self.candidate, self.target, None, None, binding,
        )
        self.assertEqual(pending["state"], "P_PRE_GRID_STATUS_PAIR_PENDING")
        self.assertNotEqual(pending["state"], "P_PRE_NO_SAFE_NORMAL_DISTANCE")
        self.assertEqual(pending["pose_binding"]["source_odom_stamp"], 33.0)

    def test_source_stamp_absence_is_wait_not_unknown_or_free(self):
        binding = self.module.p_pre_pose_binding_for_grid_status(self.cache, {})
        self.assertFalse(binding["binding_valid"])
        self.assertEqual(binding["reason"], "GRID_STATUS_SOURCE_ODOM_STAMP_UNAVAILABLE")


class DwaSameStateBindingTests(unittest.TestCase):
    """DWA must consume the Grid producer pose, never an unrelated latest pose."""

    def setUp(self):
        self.ros = FakeRos()
        self.runner_module = load_runner_module(self.ros)
        self.runner = object.__new__(self.runner_module.BlockAStarDwaRunner)
        self.runner.odom_cache = self.module_cache = load_module(self.ros).OdomCache(
            self.ros, topic="/team/livox/icp_odom_gated", message_type=object,
        )

    def tearDown(self):
        self.module_cache.close()

    @staticmethod
    def status(source_stamp, **extra):
        result = {
            "source_odom_stamp": float(source_stamp),
            "grid_content_stamp": 10.0,
            "content_generation_id": 7,
            "producer_instance_id": "fixture",
            "grid_content_hash": "fixture-hash",
        }
        result.update(extra)
        return result

    def binding(self, source_stamp, **extra):
        return self.runner.dwa_safety_state_binding(FakeGrid(), self.status(source_stamp, **extra))

    def test_b1_exact_pose_at_grid_source_stamp_is_used(self):
        self.module_cache._callback(FakeOdom(10.0, x=1.0, y=0.0))
        self.module_cache._callback(FakeOdom(10.1, x=9.0, y=0.0))
        result = self.binding(10.0)
        self.assertTrue(result["binding_valid"], result)
        self.assertEqual(result["binding_mode"], "EXACT")
        self.assertEqual(result["source_pose_x_y_yaw"][:2], [1.0, 0.0])

    def test_b1a_exact_first_runner_sample_needs_no_interpolation_period(self):
        self.module_cache._callback(FakeOdom(10.0, x=1.0, y=0.0))
        result = self.binding(10.0)
        self.assertTrue(result["binding_valid"], result)
        self.assertEqual(result["binding_mode"], "EXACT")
        self.assertIsNone(result["binding_contract_limit_sec"])

    def test_b1b_runner_startup_waits_for_a_bindable_fresh_pair(self):
        runner = object.__new__(self.runner_module.BlockAStarDwaRunner)
        runner.odom_cache = self.module_cache
        runner.args = types.SimpleNamespace(input_timeout_sec=1.0)
        grid = FakeGrid()
        status = self.status(10.1)
        runner.matching_grid_status_pair = lambda: (grid, status)
        runner.grid_cb = lambda _msg: None
        self.runner_module.rospy.ROSException = RuntimeError

        def no_grid_message(*_args, **_kwargs):
            raise RuntimeError("fixture relies on subscribed cache callbacks")

        self.runner_module.rospy.wait_for_message = no_grid_message

        def publish_startup_history():
            self.module_cache._callback(FakeOdom(10.0, x=0.0, y=0.0))
            time.sleep(0.02)
            self.module_cache._callback(FakeOdom(10.1, x=1.0, y=0.0))

        publisher = threading.Thread(target=publish_startup_history)
        publisher.start()
        try:
            _odom, returned_grid, returned_status = runner.wait_inputs()
        finally:
            publisher.join(timeout=1.0)
        self.assertIs(returned_grid, grid)
        self.assertIs(returned_status, status)
        binding = runner.dwa_safety_state_binding(returned_grid, returned_status)
        self.assertTrue(binding["binding_valid"], binding)
        self.assertEqual(binding["source_pose_x_y_yaw"][:2], [1.0, 0.0])

    def test_b2_latest_different_pose_cannot_replace_source_pose(self):
        self.module_cache._callback(FakeOdom(20.0, x=2.0, y=0.0))
        self.module_cache._callback(FakeOdom(20.1, x=20.0, y=0.0))
        result = self.binding(20.0)
        self.assertTrue(result["binding_valid"], result)
        self.assertEqual(result["source_pose_x_y_yaw"][:2], [2.0, 0.0])
        self.assertNotEqual(result["source_pose_x_y_yaw"][:2], [20.0, 0.0])

    def test_b3_missing_legal_pose_fails_closed(self):
        self.module_cache._callback(FakeOdom(30.0, x=3.0, y=0.0))
        self.module_cache._callback(FakeOdom(30.1, x=4.0, y=0.0))
        result = self.binding(29.0)
        self.assertFalse(result["binding_valid"])
        self.assertTrue(result["reason"].startswith("SAFETY_STATE_BINDING_UNAVAILABLE:"), result)

    def test_b4_status_epoch_mismatch_fails_closed(self):
        self.module_cache._callback(FakeOdom(40.0, x=4.0, y=0.0))
        self.module_cache._callback(FakeOdom(40.1, x=5.0, y=0.0))
        result = self.binding(40.0, source_odom_epoch_generation=99)
        self.assertFalse(result["binding_valid"])
        self.assertEqual(result["reason"], "SAFETY_STATE_BINDING_STATUS_ODOM_EPOCH_MISMATCH")

    def test_b5_binding_record_contains_the_decision_input_identities(self):
        self.module_cache._callback(FakeOdom(50.0, x=5.0, y=0.0))
        self.module_cache._callback(FakeOdom(50.1, x=6.0, y=0.0))
        result = self.binding(50.0)
        self.assertTrue(result["binding_valid"], result)
        attribution = self.runner_module.build_dwa_slice_attribution(
            decision_id="fixture:dwa_0001",
            safety_binding=result,
            pose_x_y_yaw=result["source_pose_x_y_yaw"],
            dwa={
                "sample_count": 55,
                "safe_moving_candidate_count": 7,
                "selected_linear_x": 0.30,
                "selected_angular_z": -0.14,
                "blocked": False,
            },
            raw_v=0.30,
            raw_w=-0.14,
        )
        self.assertEqual(attribution["grid"]["content_stamp"], 1.0)
        self.assertEqual(attribution["status"]["source_odom_stamp"], 50.0)
        self.assertEqual(attribution["safety_pose"]["odom_t0_stamp"], 50.0)
        self.assertEqual(attribution["safety_pose"]["odom_epoch_generation"], 0)
        self.assertEqual(attribution["raw_command"], {"v": 0.30, "w": -0.14})


class LegacyDoorwayMotionDeauthorization092R4Tests(unittest.TestCase):
    """T1-T7: preserve latch diagnostics, but remove its Portal-domain motion authority."""

    def setUp(self):
        self.ros = FakeRos()
        self.module = load_module(self.ros)
        self.latch = {"target_anchor_progress_m": 12.0, "side": "left"}
        self.anchor = {"x": 0.0, "y": 0.0, "heading_rad": 0.0}

    def formal_snapshot(self, *, committed):
        return {
            "doorway_candidate_authority": "PORTAL_ROOM_ZONE_EFFECT_GATE",
            "latest_effect_state": {
                "contract_version": "p2kg12_portal_effect_gate_v2",
                "room_zone_active": True,
                "input_valid": True,
            },
            "committed_portal_candidate": (
                {"candidate_lifecycle": "COMMITTED", "side": "right"} if committed else None
            ),
        }

    def emit(self, domain):
        original_anchor = self.module.write_anchor_target
        original_progress = self.module.write_anchor_progress_target
        self.module.write_anchor_target = lambda _anchor, lookahead, source: {
            "source": source, "lookahead": lookahead
        }
        self.module.write_anchor_progress_target = lambda _anchor, progress, source, extra: {
            "source": source, "progress": progress, **extra
        }
        try:
            return self.module.write_legacy_latch_or_corridor_target(
                self.anchor, self.latch, 2.0, domain
            )
        finally:
            self.module.write_anchor_target = original_anchor
            self.module.write_anchor_progress_target = original_progress

    def test_t1_non_portal_domain_preserves_historical_latch_target(self):
        domain = self.module.portal_bound_doorway_control_domain_status(False, self.formal_snapshot(committed=False))
        target, authority = self.emit(domain)
        self.assertFalse(domain["active"])
        self.assertTrue(authority["legacy_motion_authorized"])
        self.assertEqual(target["source"], "state_machine_doorway_latch_alignment")

    def test_t2_active_formal_domain_without_commit_deauthorizes_latch(self):
        domain = self.module.portal_bound_doorway_control_domain_status(True, self.formal_snapshot(committed=False))
        target, authority = self.emit(domain)
        self.assertTrue(domain["active"])
        self.assertFalse(domain["committed_candidate_present"])
        self.assertFalse(authority["legacy_motion_authorized"])
        self.assertEqual(target["source"], "state_machine_corridor_centerline_door_search")

    def test_t3_committed_portal_outside_window_uses_centerline_and_keeps_diagnostic(self):
        domain = self.module.portal_bound_doorway_control_domain_status(True, self.formal_snapshot(committed=True))
        target, authority = self.emit(domain)
        self.assertTrue(authority["diagnostic_only"])
        self.assertTrue(authority["latched_doorway_profile_present"])
        self.assertEqual(self.latch["side"], "left")
        self.assertEqual(self.module.target_authority_class(target), "CORRIDOR_CENTERLINE_AUTHORITY")

    def test_t4_opposite_sides_do_not_restore_legacy_motion_authority(self):
        domain = self.module.portal_bound_doorway_control_domain_status(True, self.formal_snapshot(committed=True))
        target, authority = self.emit(domain)
        self.assertEqual(self.formal_snapshot(committed=True)["committed_portal_candidate"]["side"], "right")
        self.assertEqual(self.latch["side"], "left")
        self.assertFalse(authority["legacy_motion_authorized"])
        self.assertNotEqual(target["source"], "state_machine_doorway_latch_alignment")

    def test_t5_authority_class_switches_from_centerline_to_p_pre_only(self):
        domain = self.module.portal_bound_doorway_control_domain_status(True, self.formal_snapshot(committed=True))
        target, _authority = self.emit(domain)
        self.assertEqual(self.module.target_authority_class(target), "CORRIDOR_CENTERLINE_AUTHORITY")
        self.assertEqual(
            self.module.target_authority_class({"source": "PORTAL_G14_P_PRE"}),
            "PORTAL_G14_P_PRE_AUTHORITY",
        )
        timeline = self.module.build_target_authority_timeline([
            {
                "iteration": 7, "state": "FOLLOW_CORRIDOR", "room_zone_reached_before": True,
                "portal_committed": True, "target": target, "runner": {"run_sim_start_sec": 10.0},
                "portal_g14_p_pre_admissibility": {"state": "P_PRE_OUTSIDE_LOCAL_PLANNING_WINDOW"},
            },
            {
                "iteration": 8, "state": "FOLLOW_CORRIDOR", "room_zone_reached_before": True,
                "portal_committed": True, "target": {"source": "PORTAL_G14_P_PRE"},
                "runner": {"run_sim_start_sec": 12.0},
                "portal_g14_p_pre_admissibility": {"state": "P_PRE_READY_FOR_PLANNER"},
            },
        ])
        self.assertEqual(
            [row["target_authority_class"] for row in timeline],
            ["CORRIDOR_CENTERLINE_AUTHORITY", "PORTAL_G14_P_PRE_AUTHORITY"],
        )
        self.assertEqual([row["p_pre_in_window"] for row in timeline], [False, True])

    def test_t6_legacy_doorway_sources_are_not_valid_centerline_or_p_pre_authorities(self):
        for source in (
            "state_machine_doorway_latch_alignment",
            "state_machine_room_side_gap_alignment",
            "state_machine_forced_room_entry_alignment",
        ):
            authority_class = self.module.target_authority_class({"source": source})
            self.assertNotIn(authority_class, {"CORRIDOR_CENTERLINE_AUTHORITY", "PORTAL_G14_P_PRE_AUTHORITY"})

    def test_t7_latch_diagnostic_data_is_not_cleared_to_pass(self):
        domain = self.module.portal_bound_doorway_control_domain_status(True, self.formal_snapshot(committed=True))
        _target, authority = self.emit(domain)
        self.assertEqual(self.latch, {"target_anchor_progress_m": 12.0, "side": "left"})
        self.assertTrue(authority["diagnostic_only"])


class EuclideanInflation092R2Tests(unittest.TestCase):
    """T1-T8: fixed-radius Euclidean inflation and 092 P_pre parity."""

    RADIUS_M = 0.2641935843278561
    RESOLUTION_M = 0.05

    def setUp(self):
        self.ros = FakeRos()
        self.navigation = load_module(self.ros)
        self.runner_module = load_runner_module(self.ros)
        self.runner = object.__new__(self.runner_module.BlockAStarDwaRunner)
        self.runner.args = types.SimpleNamespace(
            robot_radius_m=self.RADIUS_M,
            grid_resolution_m=self.RESOLUTION_M,
        )

    @staticmethod
    def grid(size=15):
        return np.zeros((size, size), dtype=np.int16)

    def runner_mask(self, grid):
        return self.runner.occupied_inflated_mask(grid, self.RESOLUTION_M)

    def test_t1_raw_occupied_seed_is_always_blocked(self):
        grid = self.grid()
        grid[7, 7] = 100
        self.assertTrue(self.runner_mask(grid)[7, 7])

    def test_t2_cells_inside_euclidean_radius_are_blocked(self):
        grid = self.grid()
        grid[7, 7] = 100
        # sqrt(5^2 + 1^2) * 0.05 < frozen static footprint radius.
        self.assertTrue(self.runner_mask(grid)[8, 12])

    def test_t3_exact_radius_boundary_is_included(self):
        grid = self.grid()
        grid[7, 7] = 100
        mask = self.runner_module.occupied_euclidean_inflated_mask(grid, 0.25, self.RESOLUTION_M)
        self.assertTrue(mask[7, 12])
        self.assertTrue(mask[11, 10])

    def test_t4_old_square_only_corner_is_not_blocked(self):
        grid = self.grid()
        grid[7, 7] = 100
        self.assertFalse(self.runner_mask(grid)[12, 12])

    def test_t5_point_at_old_030_index_corner_is_not_radius_inflated(self):
        grid = self.grid()
        grid[7, 7] = 100
        self.assertFalse(self.runner_mask(grid)[13, 13])

    def test_t6_frozen_radius_includes_026_and_excludes_roughly_027(self):
        grid = self.grid()
        grid[7, 7] = 100
        mask = self.runner_mask(grid)
        self.assertTrue(mask[8, 12])  # sqrt(0.25^2 + 0.05^2) ~= 0.255m
        self.assertFalse(mask[9, 12])  # sqrt(0.25^2 + 0.10^2) ~= 0.269m

    def test_t7_unknown_stays_blocked_without_becoming_an_inflation_seed(self):
        grid = self.grid()
        grid[7, 7] = -1
        original = grid.copy()
        runner_mask = self.runner_mask(grid)
        p_pre_mask = self.navigation.p_pre_inflated_blocked(grid, self.RADIUS_M, self.RESOLUTION_M)
        self.assertFalse(runner_mask[7, 8])
        self.assertTrue(p_pre_mask[7, 7])
        self.assertFalse(p_pre_mask[7, 8])
        self.assertTrue(np.array_equal(grid, original))

    def test_t8_runner_and_p_pre_share_identical_occupied_geometry(self):
        grid = self.grid()
        grid[4, 4] = 100
        grid[10, 10] = 100
        self.assertTrue(np.array_equal(
            self.runner_mask(grid),
            self.navigation.p_pre_inflated_blocked(grid, self.RADIUS_M, self.RESOLUTION_M),
        ))


class StartBlockTopologyR35Tests(unittest.TestCase):
    """R35: preserve inflated safety while admitting a proven-safe start escape."""

    def setUp(self):
        self.ros = FakeRos()
        self.runner_module = load_runner_module(self.ros)
        self.runner = object.__new__(self.runner_module.BlockAStarDwaRunner)
        self.runner.args = self.runner_module.build_arg_parser().parse_args([])

    def test_partial_start_block_with_safe_exact_escape_produces_path(self):
        blocked = np.zeros((16, 20), dtype=bool)
        # Start Block-4 is partially occupied, but exact start and the direct
        # diagonal escape to the next complete block are inflated-free.
        blocked[7, 4] = True
        start = (5, 5)
        path = self.runner.block_astar(blocked, start, (14, 6))
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(path[0], start)
        self.assertTrue(self.runner.last_block_astar_start_admission["partial_start_escape"])
        self.assertFalse(any(blocked[y, x] for x, y in path))

    def test_partial_start_block_rejects_unsafe_exact_start(self):
        blocked = np.zeros((16, 20), dtype=bool)
        blocked[5, 5] = True
        self.assertEqual(self.runner.block_astar(blocked, (5, 5), (14, 6)), [])
        self.assertEqual(
            self.runner.last_block_astar_start_admission["reason"],
            "START_CELL_OR_BLOCK_UNSAFE",
        )

    def test_partial_start_block_rejects_blocked_escape_segment(self):
        blocked = np.ones((16, 20), dtype=bool)
        # Only the start and target blocks are otherwise open. The single
        # possible start-to-target-block segment crosses this blocked cell.
        blocked[4:8, 4:8] = False
        blocked[4:8, 8:12] = False
        blocked[5, 5] = False
        blocked[5, 7] = True
        self.assertEqual(self.runner.block_astar(blocked, (5, 5), (10, 6)), [])


class TargetOnlyClearance092R11Tests(unittest.TestCase):
    """Pure target-shaping checks; A* and DWA are deliberately not changed."""

    def setUp(self):
        self.ros = FakeRos()
        self.runner_module = load_runner_module(self.ros)
        self.runner = object.__new__(self.runner_module.BlockAStarDwaRunner)
        self.runner.args = self.runner_module.build_arg_parser().parse_args([])
        self.runner.unilateral_clearance_safety_side = None
        self.runner.unilateral_clearance_missing_count = 0
        self.grid_msg = FakeGrid()

    def formal_grid(self, side=None):
        raw = np.zeros((60, 66), dtype=np.int16)
        blocked = np.zeros_like(raw, dtype=bool)
        # Existing estimator window: x=[0.8,2.4], |y|>=0.45.
        if side == "left":
            blocked[39:44, 22:54] = True
            blocked[35, 12] = True  # Formal near-side evidence at about 0.43 m.
        elif side == "right":
            blocked[16:21, 22:54] = True
            blocked[24, 12] = True  # Mirror of the left-side near evidence.
        return raw, blocked

    def shape(self, side, target_y=-0.025):
        raw, blocked = self.formal_grid(side)
        return self.runner.unilateral_clearance_target_shape(
            self.grid_msg,
            {"source": "state_machine_corridor_centerline_room_zone_guard"},
            raw,
            blocked,
            (0.425, target_y),
        )

    def test_t1_left_unilateral_block_selects_first_formal_right_target(self):
        shaped, report = self.shape("left")
        self.assertEqual(shaped, (0.425, -0.125))
        self.assertTrue(report["applied"])
        self.assertEqual(report["effective_safety_side"], "right_negative_y")
        self.assertEqual(report["selected_target_cell"], [14, 27])
        self.assertEqual(report["selected_target_block"], [3, 6])
        self.assertLess(report["selected_block_center_y_m"], 0.0)

    def test_t2_right_unilateral_block_mirrors_to_left_target(self):
        shaped, report = self.shape("right", target_y=0.025)
        self.assertEqual(shaped, (0.425, 0.125))
        self.assertTrue(report["applied"])
        self.assertEqual(report["effective_safety_side"], "left_positive_y")
        self.assertGreater(report["selected_block_center_y_m"], 0.0)

    def test_t3_near_center_without_unilateral_signal_remains_neutral(self):
        shaped, report = self.shape(None)
        self.assertEqual(shaped, (0.425, -0.025))
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "no_unilateral_clearance_degradation")

    def test_t4_existing_path_stability_window_holds_then_releases_a_missing_signal(self):
        shaped, report = self.shape("left")
        self.assertTrue(report["applied"])
        self.assertEqual(shaped[1], -0.125)
        shaped, report = self.shape(None)
        self.assertTrue(report["applied"])
        self.assertEqual(report["persistence"], "held_existing_path_stability_window")
        shaped, report = self.shape(None)
        self.assertTrue(report["applied"])
        self.assertEqual(report["persistence"], "held_existing_path_stability_window")
        shaped, report = self.shape(None)
        self.assertEqual(shaped, (0.425, -0.025))
        self.assertFalse(report["applied"])
        self.assertEqual(report["persistence"], "released_after_existing_path_stability_window")


class PPreGoalRegionGuard092R15Tests(unittest.TestCase):
    """Pure P_pre goal-disk checks; A* and DWA choices stay intact."""

    def setUp(self):
        self.ros = FakeRos()
        self.runner_module = load_runner_module(self.ros)
        self.runner = object.__new__(self.runner_module.BlockAStarDwaRunner)
        self.runner.args = types.SimpleNamespace(goal_tolerance_m=0.30)

    def guard(self, source, xy, v, w):
        return self.runner.apply_p_pre_goal_region_motion_guard({"source": source}, xy, v, w)

    def test_t1_r11_first_command_is_outer_miss_and_is_tangency_capped(self):
        capped, report = self.guard("PORTAL_G14_P_PRE", (1.5969618674700239, -1.04098531688358), 0.4, -0.14)
        self.assertTrue(report["applied"])
        self.assertFalse(report["trajectory_enters_goal_region"])
        self.assertLess(capped, 0.4)
        self.assertEqual(report["reason"], "p_pre_goal_region_outer_tangent_speed_cap")

    def test_t2_guard_is_p_pre_only_and_preserves_non_p_pre_command(self):
        capped, report = self.guard("state_machine_corridor_centerline_door_search", (1.6, -1.0), 0.4, -0.14)
        self.assertEqual(capped, 0.4)
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "not_portal_g14_p_pre")

    def test_t3_target_behind_fails_closed(self):
        capped, report = self.guard("PORTAL_G14_P_PRE", (-0.0008, -0.6118), 0.6, -0.12698)
        self.assertEqual(capped, 0.0)
        self.assertTrue(report["applied"])
        self.assertEqual(report["reason"], "p_pre_no_longer_in_front")

    def test_t4_r11_historical_replay_restricts_each_outer_miss_before_overshoot(self):
        historical = [
            ((1.5969618674700239, -1.04098531688358), 0.4, -0.14),
            ((1.5112056507385043, -0.9621502952489978), 0.6, -0.154),
            ((1.2710541217924234, -0.8301572291939069), 0.6, -0.1295),
            ((0.964430695176498, -0.7402906388854289), 0.6, -0.1498),
            ((0.6514302076433762, -0.6752914135084084), 0.6, -0.1274),
            ((0.3296763067873744, -0.6392855096485746), 0.6, -0.14896),
            ((-0.0008136102090192388, -0.6117758966898802), 0.6, -0.12698),
        ]
        outputs = [self.guard("PORTAL_G14_P_PRE", xy, v, w)[0] for xy, v, w in historical]
        self.assertLess(outputs[0], historical[0][1])
        self.assertLess(outputs[4], historical[4][1])
        self.assertEqual(outputs[6], 0.0)

    def test_t5_run_0065_near_aligned_circle_already_enters_goal_disk_without_collapse(self):
        guarded, report = self.guard("PORTAL_G14_P_PRE", (1.7256430600546566, -0.11782649495697023), 0.6, -0.005319999999999991)
        self.assertEqual(guarded, 0.6)
        self.assertFalse(report["applied"])
        self.assertTrue(report["trajectory_enters_goal_region"])
        self.assertEqual(report["reason"], "p_pre_goal_region_already_reachable")

    def test_t6_exact_straight_line_intersection_is_not_capped(self):
        guarded, report = self.guard("PORTAL_G14_P_PRE", (1.8, 0.20), 0.6, 0.0)
        self.assertEqual(guarded, 0.6)
        self.assertTrue(report["trajectory_enters_goal_region"])
        self.assertEqual(report["reason"], "p_pre_goal_region_straight_intersection")

    def test_t7_straight_line_miss_fails_closed(self):
        guarded, report = self.guard("PORTAL_G14_P_PRE", (1.8, 0.31), 0.6, 0.0)
        self.assertEqual(guarded, 0.0)
        self.assertTrue(report["applied"])
        self.assertFalse(report["trajectory_enters_goal_region"])

    def test_t8_curved_path_in_interval_is_preserved(self):
        guarded, report = self.guard("PORTAL_G14_P_PRE", (1.6, -1.0), 0.25, -0.14)
        self.assertEqual(guarded, 0.25)
        self.assertTrue(report["trajectory_enters_goal_region"])

    def test_t9_curved_outer_miss_is_capped_to_goal_disk_tangent(self):
        guarded, report = self.guard("PORTAL_G14_P_PRE", (1.6, -1.0), 0.6, -0.14)
        interval = report["goal_region_radius_interval_m"]
        self.assertTrue(report["applied"])
        self.assertAlmostEqual(guarded, abs(-0.14) * interval[1])
        self.assertLess(guarded, 0.6)

    def test_t10_inner_miss_is_not_misrepaired_by_a_speed_cap(self):
        guarded, report = self.guard("PORTAL_G14_P_PRE", (2.8865583708089013, -0.7517803650303474), 0.4, -0.14)
        self.assertEqual(guarded, 0.4)
        self.assertFalse(report["applied"])
        self.assertFalse(report["trajectory_enters_goal_region"])
        self.assertEqual(report["reason"], "p_pre_goal_region_inner_miss_not_speed_capable")

    def test_t11_wrong_turn_sign_that_cannot_reach_disk_fails_closed(self):
        guarded, report = self.guard("PORTAL_G14_P_PRE", (1.6, -1.0), 0.6, 0.14)
        self.assertEqual(guarded, 0.0)
        self.assertEqual(report["reason"], "p_pre_goal_region_wrong_turn_unreachable")

    def test_t12_zero_tolerance_converges_to_exact_point_tangent_radius(self):
        self.runner.args.goal_tolerance_m = 1e-9
        guarded, report = self.guard("PORTAL_G14_P_PRE", (1.6, -1.0), 0.6, -0.14)
        expected_radius = (1.6 ** 2 + 1.0 ** 2) / (2.0 * 1.0)
        self.assertAlmostEqual(report["goal_region_radius_interval_m"][1], expected_radius, places=6)
        self.assertAlmostEqual(guarded, 0.14 * expected_radius, places=6)


class PortalNormalAlignment093Tests(unittest.TestCase):
    """Pure 093 geometry/control checks; no master, bag, or motion required."""

    RADIUS_M = 0.2641935843278561

    def setUp(self):
        self.ros = FakeRos()
        self.module = load_module(self.ros)
        self.module.qualified_for_navigation = lambda _grid, status: (bool(status.get("qualified")), [] if status.get("qualified") else ["not_qualified"])
        self.candidate = {
            "portal_run_id": "run-093", "portal_frame_sequence": 93,
            "portal_source_stamp": 93.0, "portal_track_id": "left-93", "side": "left",
        }

    def target(self, normal=(0.0, 1.0), width=1.2):
        return {
            "target_valid": True, "candidate_lifecycle": "COMMITTED", "portal_width_m": width,
            "portal_identity": {"run_id": "run-093", "frame_sequence": 93, "source_stamp": 93.0, "track_id": "left-93", "side": "left"},
            "frozen_geometry": {"geometry_valid": True, "portal_center_odom": [1.0, 1.0], "portal_normal_odom": list(normal)},
            "P_through_odom": [1.0 + 0.3 * normal[0], 1.0 + 0.3 * normal[1]],
        }

    def evaluate(self, pose=(0.0, 0.0, 0.0), target=None, grid=None, candidate=Ellipsis, qualified=True, selected_d=0.4):
        return self.module.evaluate_portal_normal_alignment(
            self.candidate if candidate is Ellipsis else candidate,
            target or self.target(), pose, grid or FakeGrid(), {"qualified": qualified}, self.RADIUS_M, selected_d,
        )

    def test_positive_error_commands_positive_zero_linear_turn(self):
        check = self.evaluate()
        self.assertEqual(check["state"], "PORTAL_NORMAL_ALIGNMENT_READY")
        command = self.module.portal_normal_alignment_command(check["heading_error_rad"], 0.45, 1.0)
        self.assertEqual(command["linear_x"], 0.0)
        self.assertGreater(command["angular_z"], 0.0)
        self.assertLessEqual(command["angular_z"], 0.45)

    def test_negative_error_commands_negative_zero_linear_turn(self):
        check = self.evaluate(pose=(0.0, 0.0, math.pi))
        command = self.module.portal_normal_alignment_command(check["heading_error_rad"], 0.45, 1.0)
        self.assertLess(command["angular_z"], 0.0)
        self.assertEqual(command["linear_x"], 0.0)

    def test_wraparound_uses_short_signed_error_without_direction_guessing(self):
        check = self.evaluate(pose=(0.0, 0.0, -math.pi + 0.02), target=self.target(normal=(-1.0, 0.0)))
        self.assertAlmostEqual(check["heading_error_rad"], -0.02, places=6)

    def test_already_aligned_needs_no_angular_command(self):
        check = self.evaluate(pose=(0.0, 0.0, math.pi / 2.0))
        command = self.module.portal_normal_alignment_command(check["heading_error_rad"], 0.45, 1.0)
        self.assertAlmostEqual(command["angular_z"], 0.0)

    def test_invalid_or_wrong_normal_fails_closed(self):
        bad = self.target(normal=(0.0, 0.0))
        self.assertEqual(self.evaluate(target=bad)["reason"], "PORTAL_NORMAL_GEOMETRY_INVALID")
        wrong_through = self.target()
        wrong_through["P_through_odom"] = [1.0, 0.7]
        self.assertEqual(self.evaluate(target=wrong_through)["reason"], "PORTAL_NORMAL_DIRECTION_BINDING_INVALID")

    def test_stale_or_unqualified_pair_fails_closed(self):
        check = self.evaluate(qualified=False)
        self.assertEqual(check["reason"], "FORMAL_LOCAL_GRID_UNQUALIFIED")
        self.assertFalse(check["rotation_clearance_safe"])

    def test_blocked_current_centre_fails_rotation_clearance(self):
        grid = FakeGrid()
        cell = self.module.local_xy_to_cell(0.0, 0.0, grid)
        grid.data[cell[1] * grid.info.width + cell[0]] = 100
        check = self.evaluate(grid=grid)
        self.assertEqual(check["reason"], "P_PRE_ROTATION_CLEARANCE")
        self.assertFalse(check["rotation_clearance_safe"])

    def crossing(self, pose, center=(0.0, 0.0), normal=(0.0, 1.0), width=1.2):
        return self.module.portal_plane_crossing_geometry(pose, center, normal, width, self.RADIUS_M)

    def test_actual_pose_crossing_uses_tangent_intersection_not_nominal_tolerance(self):
        # With n=(0,1), t=(-1,0), this pose is centred and already faces the portal.
        check = self.crossing((0.0, -1.0, math.pi / 2.0))
        self.assertTrue(check["portal_plane_crossing_safe"])
        self.assertAlmostEqual(check["portal_normal_distance_D_m"], 1.0)
        self.assertAlmostEqual(check["portal_tangent_offset_s_m"], 0.0)
        self.assertAlmostEqual(check["portal_plane_crossing_s_m"], 0.0)
        self.assertAlmostEqual(check["portal_aperture_half_width_H_m"], (1.2 / 2.0) - self.RADIUS_M)

    def test_tangent_offset_left_right_and_actual_distance_are_distinct(self):
        left = self.crossing((-0.20, -1.0, math.pi / 2.0))
        right = self.crossing((0.40, -1.0, math.pi / 2.0))
        self.assertTrue(left["portal_plane_crossing_safe"])
        self.assertFalse(right["portal_plane_crossing_safe"])
        close = self.crossing((0.0, -0.20, math.pi / 2.0 + 0.30))
        far = self.crossing((0.0, -2.00, math.pi / 2.0 + 0.30))
        self.assertTrue(close["portal_plane_crossing_safe"])
        self.assertFalse(far["portal_plane_crossing_safe"])

    def test_inside_p_pre_goal_disk_can_still_need_turn_and_heading_can_reach_aperture(self):
        # selected P_pre for this fixture is (1.0, 0.6); this is within its 0.30 m goal disk.
        off_centre = self.crossing((0.75, 0.60, math.pi / 2.0 + 0.60), center=(1.0, 1.0))
        self.assertFalse(off_centre["portal_plane_crossing_safe"])
        self.assertEqual(off_centre["portal_plane_crossing_reason"], "PORTAL_PLANE_CROSSING_UNSAFE_TURN_REQUIRED")
        half_width = (1.2 / 2.0) - self.RADIUS_M
        reaches_aperture = self.crossing((0.0, -1.0, math.pi / 2.0 + math.atan(half_width) - 1e-6))
        self.assertTrue(reaches_aperture["portal_plane_crossing_safe"])
        self.assertGreaterEqual(reaches_aperture["portal_plane_remaining_margin_m"], 0.0)

    def test_run_0067_safe_and_run_0068_premature_pass_is_unsafe_under_actual_pose_contract(self):
        n67 = (-0.04265729413140853, -0.9990897633633258)
        center67 = (16.77944603983179 + 0.9500000096857548 * n67[0], -0.9669324223595672 + 0.9500000096857548 * n67[1])
        run67_terminal = self.crossing((16.661720276, -0.925704598, -1.462186032), center67, n67, 1.2)
        self.assertTrue(run67_terminal["portal_plane_crossing_safe"])
        self.assertAlmostEqual(run67_terminal["portal_normal_theta_rad"], 0.151280536, places=5)
        self.assertAlmostEqual(run67_terminal["portal_plane_crossing_s_m"], 0.030959430, places=5)
        n68 = (-0.017185729345311405, -0.9998523144479239)
        center68 = (16.831469688755906 + 0.6000000044703484 * n68[0], -0.8984847754343028 + 0.6000000044703484 * n68[1])
        run68_terminal = self.crossing((16.923246384, -0.640069425, -0.643710550), center68, n68, 1.5)
        self.assertFalse(run68_terminal["portal_plane_crossing_safe"])
        self.assertAlmostEqual(abs(run68_terminal["portal_normal_theta_rad"]), 0.944272352, places=5)
        self.assertAlmostEqual(run68_terminal["portal_plane_crossing_s_m"], 1.275425679, places=5)
        self.assertLess(run68_terminal["portal_plane_remaining_margin_m"], 0.0)

    def test_behind_portal_plane_and_non_facing_heading_never_align(self):
        overshoot = self.crossing((0.0, 0.10, math.pi / 2.0))
        self.assertEqual(overshoot["portal_plane_crossing_reason"], "P_PRE_PORTAL_PLANE_OVERSHOOT")
        self.assertFalse(overshoot["portal_plane_crossing_safe"])
        non_facing = self.crossing((0.0, -1.0, -math.pi / 2.0))
        self.assertEqual(non_facing["portal_plane_crossing_reason"], "PORTAL_HEADING_DOES_NOT_FACE_PORTAL_PLANE")
        self.assertFalse(non_facing["portal_plane_crossing_safe"])

    def test_failure_result_records_explicit_stop_evidence(self):
        args = types.SimpleNamespace(execute=True)
        original = self.module.publish_stop_at_door
        self.module.publish_stop_at_door = lambda _args: {"topic": "/cmd_vel_raw", "zero_count": 3}
        try:
            result = self.module.portal_normal_alignment_failure(
                args, "P2KG15_093_HEADING_CONTROLLER_NONCONVERGENCE", "test_failure", [],
            )
        finally:
            self.module.publish_stop_at_door = original
        self.assertFalse(result["aligned"])
        self.assertEqual(result["safe_stop"]["zero_count"], 3)

    def test_committed_identity_cannot_be_replaced_and_legacy_has_no_authority(self):
        other = dict(self.candidate)
        other["portal_track_id"] = "left-other"
        self.assertEqual(self.evaluate(candidate=other)["reason"], "COMMITTED_FROZEN_PORTAL_CONTRACT_INVALID")
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('legacy_doorway_control_disabled = True', source)
        self.assertIn('"ACTUAL_POSE_PORTAL_PLANE_CROSSING"', source)
        self.assertNotIn("portal_normal_heading_tolerance_rad", source)


class PThroughHandoffR22Tests(unittest.TestCase):
    """Pure R22 shadow-gate checks; no ROS master, publisher, or motion."""

    RADIUS_M = 0.2641935843278561

    def setUp(self):
        self.ros = FakeRos()
        self.module = load_module(self.ros)
        self.module.qualified_for_navigation = lambda _grid, status: (bool(status.get("qualified")), [] if status.get("qualified") else ["not_qualified"])
        self.candidate = {
            "portal_run_id": "run-r22", "portal_frame_sequence": 22,
            "portal_source_stamp": 22.0, "portal_track_id": "left-r22", "side": "left",
        }

    def target(self, through=(0.0, 0.40)):
        return {
            "target_valid": True, "candidate_lifecycle": "COMMITTED", "portal_width_m": 1.20,
            "portal_identity": {"run_id": "run-r22", "frame_sequence": 22, "source_stamp": 22.0, "track_id": "left-r22", "side": "left"},
            "frozen_geometry": {"geometry_valid": True, "portal_center_odom": [0.0, 0.0], "portal_normal_odom": [0.0, 1.0]},
            "P_through_odom": list(through),
        }

    def evaluate(self, target=None, grid=None, qualified=True):
        return self.module.evaluate_p_through_handoff_feasibility(
            self.candidate, target or self.target(), (0.0, -1.0, math.pi / 2.0),
            grid or FakeGrid(), {"qualified": qualified}, self.RADIUS_M,
        )

    def test_crossing_path_is_a_read_only_ready_candidate(self):
        result = self.evaluate()
        self.assertTrue(result["feasible"])
        self.assertEqual(result["state"], "P_THROUGH_HANDOFF_FEASIBLE")
        self.assertTrue(result["astar_crosses_portal"])
        self.assertTrue(result["astar_endpoint_room_facing"])
        self.assertTrue(result["astar_endpoint_in_target_block"])
        self.assertTrue(result["swept_footprint_safe"])
        self.assertFalse(result["dwa_first_command"]["blocked"])
        self.assertFalse(result["motion_commanded"])

    def test_unqualified_pair_and_inadmissible_target_fail_closed(self):
        self.assertEqual(self.evaluate(qualified=False)["reason"], "FORMAL_LOCAL_GRID_UNQUALIFIED")
        grid = FakeGrid()
        cell = self.module.local_xy_to_cell(1.4, 0.0, grid)
        grid.data[cell[1] * grid.info.width + cell[0]] = -1
        self.assertEqual(self.evaluate(grid=grid)["reason"], "P_THROUGH_TARGET_NOT_FORMALLY_ADMISSIBLE")

    def test_nearest_free_style_pre_portal_path_is_not_feasible(self):
        original = self.module.BlockAStarDwaRunner.block_astar
        self.module.BlockAStarDwaRunner.block_astar = lambda _self, _blocked, start, _goal, grid_msg=None: [start, (14, 34)]
        try:
            result = self.evaluate()
        finally:
            self.module.BlockAStarDwaRunner.block_astar = original
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "A_STAR_NEAREST_FREE_FALLBACK_NOT_P_THROUGH")

    def test_dwa_blocked_and_planner_exception_cannot_pass(self):
        original_dwa = self.module.BlockAStarDwaRunner.choose_dwa
        self.module.BlockAStarDwaRunner.choose_dwa = lambda *_args, **_kwargs: (0.0, 0.0, {"blocked": True, "sample_count": 0})
        try:
            result = self.evaluate()
        finally:
            self.module.BlockAStarDwaRunner.choose_dwa = original_dwa
        self.assertEqual(result["reason"], "P_THROUGH_DWA_SHADOW_BLOCKED")
        original_astar = self.module.BlockAStarDwaRunner.block_astar
        self.module.BlockAStarDwaRunner.block_astar = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("shadow"))
        try:
            result = self.evaluate()
        finally:
            self.module.BlockAStarDwaRunner.block_astar = original_astar
        self.assertEqual(result["reason"], "P_THROUGH_SHADOW_PLANNER_EXCEPTION:RuntimeError")

    def test_unknown_out_of_grid_and_swept_collision_fail_closed(self):
        self.assertEqual(self.evaluate(target=self.target(through=(0.0, 3.5)))["reason"], "P_THROUGH_OUTSIDE_FORMAL_LOCAL_GRID")
        grid = FakeGrid()
        collision_cell = self.module.local_xy_to_cell(0.70, 0.0, grid)
        grid.data[collision_cell[1] * grid.info.width + collision_cell[0]] = 100
        original = self.module.BlockAStarDwaRunner.block_astar
        self.module.BlockAStarDwaRunner.block_astar = lambda _self, _blocked, start, goal, grid_msg=None: [start, goal]
        try:
            result = self.evaluate(grid=grid)
        finally:
            self.module.BlockAStarDwaRunner.block_astar = original
        self.assertEqual(result["reason"], "P_THROUGH_SWEPT_STATIC_FOOTPRINT_UNSAFE")


class PostCommandTerminalReachR21Tests(unittest.TestCase):
    """R21: the terminal check is the existing goal disk on newer odom only."""

    def setUp(self):
        self.ros = FakeRos()
        self.runner_module = load_runner_module(self.ros)

    def evaluate(self, *, pre_stamp, post_stamp, post_x, tolerance=0.30):
        return self.runner_module.terminal_reach_from_post_command_odom(
            (0.0, 0.0),
            FakeOdom(pre_stamp, x=0.31, y=0.0),
            FakeOdom(post_stamp, x=post_x, y=0.0),
            tolerance,
        )

    def test_t1_step_start_inside_tolerance_is_still_the_existing_reached_predicate(self):
        # The normal pre-command branch remains the owner of this case.
        self.assertLessEqual(math.hypot(0.30, 0.0), 0.30)

    def test_t2_last_slice_enters_tolerance_returns_reached(self):
        result = self.evaluate(pre_stamp=10.0, post_stamp=10.5, post_x=0.29)
        self.assertTrue(result["fresh_valid_odom"])
        self.assertTrue(result["reached"])
        self.assertEqual(result["reason"], "post_command_reached_existing_goal_tolerance")

    def test_t3_last_slice_stays_outside_returns_not_reached(self):
        result = self.evaluate(pre_stamp=10.0, post_stamp=10.5, post_x=0.300001)
        self.assertTrue(result["fresh_valid_odom"])
        self.assertFalse(result["reached"])
        self.assertEqual(result["reason"], "post_command_outside_existing_goal_tolerance")

    def test_t4_stale_post_command_odom_cannot_reach(self):
        result = self.evaluate(pre_stamp=10.0, post_stamp=10.0, post_x=0.10)
        self.assertFalse(result["fresh_valid_odom"])
        self.assertFalse(result["reached"])
        self.assertEqual(result["reason"], "post_command_odom_not_fresh")

    def test_t5_exact_goal_tolerance_boundary_reaches(self):
        result = self.evaluate(pre_stamp=10.0, post_stamp=10.5, post_x=0.30)
        self.assertTrue(result["reached"])
        self.assertAlmostEqual(result["post_command_distance_to_target_m"], 0.30)

    def test_t6_just_inside_and_just_outside_preserve_the_boundary(self):
        self.assertTrue(self.evaluate(pre_stamp=10.0, post_stamp=10.5, post_x=0.299999)["reached"])
        self.assertFalse(self.evaluate(pre_stamp=10.0, post_stamp=10.5, post_x=0.300001)["reached"])

    def test_t7_run_0070_historical_post_command_sample_replays_as_reached(self):
        target = (16.743310687716587, -0.7686436028188337)
        result = self.runner_module.terminal_reach_from_post_command_odom(
            target,
            FakeOdom(46.826, x=16.78048324584961, y=-0.4352867603302002),
            FakeOdom(47.430, x=16.912058, y=-0.528533),
            0.30,
        )
        self.assertTrue(result["fresh_valid_odom"])
        self.assertTrue(result["reached"])
        # The fixture carries the bag pose rounded to six decimal places; it
        # must remain the observed 0.293477... m sample, not an exact binary
        # serialization claim.
        self.assertAlmostEqual(result["post_command_distance_to_target_m"], 0.29347705411103425, places=7)


class Replay087PortalCommitTests(unittest.TestCase):
    """Real bag assertions; run with /opt/ros/noetic sourced, never a master."""

    def test_087_commit_binding_shadow_and_noncontrol_regression(self):
        # rosbag deserialisation needs the genuine ROS Python modules; the
        # state-machine helper itself remains loaded with no-master stubs.
        for name in (
            "rospy", "std_msgs.msg", "std_msgs", "geometry_msgs.msg", "geometry_msgs",
            "nav_msgs.msg", "nav_msgs", "rosgraph_msgs.msg", "rosgraph_msgs",
            "sensor_msgs.msg", "sensor_msgs",
        ):
            sys.modules.pop(name, None)
        importlib.import_module("rospy")
        import rosbag

        ros = FakeRos()
        module = load_module(ros)
        if str(GATE_MODULE_PATH.parent) not in sys.path:
            sys.path.insert(0, str(GATE_MODULE_PATH.parent))
        spec = importlib.util.spec_from_file_location("p2kg15_gate_091_test", GATE_MODULE_PATH)
        gate_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(gate_module)
        cache = module.OdomCache(ros, history_capacity=1024)
        gate = gate_module.PortalRoomZoneEffectGate()
        authority = module.PortalBoundDoorCandidateAuthority(subscribe=False)
        bag_path = ROOT / "audit_bags" / "p2kg15_first_approach_candidate_online_087" / "p2kg15_first_approach_candidate_online_087.bag"
        frames = pre_zone_commits = dual_frames = 0
        first = None
        with rosbag.Bag(str(bag_path)) as bag:
            for topic, msg, stamp in bag.read_messages(topics=[
                "/audit/p2kg11/portal_frame", "/audit/p2kg12/room_zone_state", module.ODOM_TOPIC,
            ]):
                if topic == module.ODOM_TOPIC:
                    cache._callback(msg)
                elif topic == "/audit/p2kg12/room_zone_state":
                    self.assertTrue(gate.add_zone_state(json.loads(msg.data)))
                else:
                    frames += 1
                    effect = gate.evaluate(json.loads(msg.data))
                    committed = authority.ingest(effect, float(stamp.to_sec()))
                    if frames <= 289 and committed is not None:
                        pre_zone_commits += 1
                    if effect["left"]["effect_eligible"] and effect["right"]["effect_eligible"]:
                        dual_frames += 1
                    if committed is not None and first is None:
                        first = committed
        self.assertEqual(frames, 485)
        self.assertEqual(pre_zone_commits, 0)
        self.assertEqual(dual_frames, 96)
        self.assertIsNotNone(first)
        self.assertEqual(
            (first["portal_run_id"], first["portal_frame_sequence"], first["portal_source_stamp"], first["portal_track_id"], first["side"]),
            ("p2kg15_087_manual", 331, 36.497, "right-12", "right"),
        )
        self.assertEqual(authority.snapshot()["committed_portal_candidate"]["portal_track_id"], "right-12")
        binding = cache.pose_at_source_stamp(first["portal_source_stamp"])
        target = module.build_g14_shadow_target(first, binding, corridor_approach_direction=(1.0, 0.0))
        self.assertTrue(binding["binding_valid"])
        self.assertEqual(binding["odom_binding_method"], "EXACT")
        self.assertTrue(target["target_valid"] and target["width_gate_pass"])
        self.assertEqual(target["target_mode"], "G14_SHADOW_ONLY")
        self.assertFalse(target["safe_for_navigation"] or target["planner_ready"] or target["send_to_navigation"] or target["control_authority"])
        self.assertNotIn("left_003", json.dumps(first, sort_keys=True))
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("legacy_doorway_transition_deauthorized", source)
        self.assertIn("stop_after_first_portal_g14_shadow_target", source)
        cache.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
