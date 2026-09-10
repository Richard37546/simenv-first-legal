#!/usr/bin/env python3
"""Offline contracts for default-off ROOM_LOCAL online validation plumbing."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_odom_cache import FakeRos, load_module
import room_local_online_validation_observer as observer


class OnlineEnablementTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())

    def args(self, *extra):
        return self.module.build_arg_parser().parse_args(list(extra))

    def test_01_default_off_room_search_has_no_guarded_flags(self):
        args = self.args()
        cmd = self.module.runner_cmd(args, state="ROOM_SEARCH", runtime_sec=1.0, max_steps=1, local_control_mode="ROOM_LOCAL")
        self.assertNotIn("--room-local-phase2-productive-admission", cmd)
        self.assertNotIn("--room-local-phase3-orientation-recovery", cmd)

    def test_02_validation_room_search_forwards_paired_flags(self):
        args = self.args("--enable-room-local-guarded-online-validation")
        with mock.patch.dict(os.environ, {"STATE_MACHINE_RUN_ID": "run_enablement"}):
            cmd = self.module.runner_cmd(args, state="ROOM_SEARCH", runtime_sec=1.0, max_steps=1, local_control_mode="ROOM_LOCAL", validation_invocation_id="room_local_0001")
        p2 = cmd.index("--room-local-phase2-productive-admission")
        p3 = cmd.index("--room-local-phase3-orientation-recovery")
        self.assertEqual(cmd[p2 + 1], "phase3_guarded_execute")
        self.assertEqual(cmd[p3 + 1], "phase3_guarded_execute")
        self.assertIn("room_local_0001", cmd)

    def test_03_validation_room_return_profile_forwards_paired_flags(self):
        args = self.args("--enable-room-local-guarded-online-validation")
        cmd = self.module.runner_cmd(args, state="ROOM_SEARCH", runtime_sec=1.0, max_steps=1, local_control_mode="ROOM_LOCAL", validation_invocation_id="room_local_0002")
        self.assertEqual(cmd.count("phase3_guarded_execute"), 2)

    def test_04_transit_command_is_byte_equivalent_with_validation_on(self):
        before = self.module.runner_cmd(self.args(), state="FOLLOW_CORRIDOR", runtime_sec=1.0, max_steps=1)
        after = self.module.runner_cmd(self.args("--enable-room-local-guarded-online-validation"), state="FOLLOW_CORRIDOR", runtime_sec=1.0, max_steps=1)
        self.assertEqual(before, after)

    def test_05_preflight_remains_nonexecuting_with_validation_on(self):
        args = self.args("--enable-room-local-guarded-online-validation")
        cmd = self.module.runner_cmd(args, state="ROOM_SEARCH", runtime_sec=1.0, max_steps=1, local_control_mode="ROOM_LOCAL", validation_invocation_id="room_local_0003")
        self.assertNotIn("--execute", cmd)
        self.assertIn("phase3_guarded_execute", cmd)

    def test_06_invocation_ids_are_unique_and_room_local_only(self):
        args = self.args("--enable-room-local-guarded-online-validation")
        with mock.patch.dict(os.environ, {"STATE_MACHINE_RUN_ID": "run_enablement"}), \
             mock.patch.object(self.module, "run_command", return_value={}), \
             mock.patch.object(self.module, "summarize_runner", return_value={}):
            first = self.module.run_runner(args, "ROOM_SEARCH", 1.0, 1, local_control_mode="ROOM_LOCAL")
            second = self.module.run_runner(args, "ROOM_SEARCH", 1.0, 1, local_control_mode="ROOM_LOCAL")
            transit = self.module.run_runner(args, "FOLLOW_CORRIDOR", 1.0, 1)
        self.assertNotEqual(first["validation_runner_invocation_id"], second["validation_runner_invocation_id"])
        self.assertIsNone(transit["validation_runner_invocation_id"])

    def test_07_correlator_binds_only_messages_inside_slice_window(self):
        c = observer.Correlator()
        c.start({"command_slice_id": "s1", "ros_time_sec": 10.0, "intended_duration_sec": 0.5, "requested_v": 0.3, "requested_w": 0.0})
        c.end({"command_slice_id": "s1", "end_ros_time_sec": 10.5})
        c.observe("cmd_raw", 10.1, {"v": 0.3})
        c.observe("cmd_final", 10.2, {"v": 0.3})
        c.observe("odom", 10.4, {"x": 0.1})
        c.observe("odom", 10.6, {"x": 0.2})
        result = c.result("s1")
        self.assertEqual(result["attribution_status"], "COMPLETE")
        self.assertEqual(len(result["odom"]), 1)

    def test_08_correlator_reports_missing_critical_stream(self):
        c = observer.Correlator()
        c.start({"command_slice_id": "s2", "ros_time_sec": 2.0, "intended_duration_sec": 0.5, "requested_v": 0.0, "requested_w": 0.3})
        c.observe("cmd_raw", 2.1, {})
        self.assertEqual(c.result("s2")["missing_critical_streams"], ["cmd_final", "odom"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
