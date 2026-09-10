#!/usr/bin/env python3
"""Offline C1-C8 closure contracts; no ROS master, Gazebo, or mission is started."""

import json
import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_odom_cache import FakeRos, RUNNER_MODULE_PATH, install_message_stubs, load_module
import room_local_online_validation_observer as observer


RUN_ID = "closure_run"


def event(event_type, **fields):
    return {"run_id": RUN_ID, "runner_invocation_id": "room_local_0001", "event_type": event_type, **fields}


def decision(sequence=1, **extra):
    data = {
        "decision_id": "d%s" % sequence,
        "input_acquisition_sequence": sequence,
        "grid_status_acquisition_sequence": sequence,
        "astar_evaluation_sequence": sequence,
        "path_anchor_sequence": sequence,
        "phase2_productivity_status": "NO_PRODUCTIVE_TRANSLATION",
    }
    data.update(extra)
    return event("ROOM_LOCAL_DECISION", **data)


def slice_start(name, v=0.0, w=0.3, t=10.0):
    return event("COMMAND_SLICE_START", command_slice_id=name, ros_time_sec=t, intended_duration_sec=0.5, requested_v=v, requested_w=w)


def slice_end(name, t=10.5):
    return event("COMMAND_SLICE_END", command_slice_id=name, end_ros_time_sec=t)


def complete_slice(monitor, name, t=10.0):
    monitor.observe("cmd_raw", t + 0.1, {"v": 0.0})
    monitor.observe("cmd_final", t + 0.2, {"v": 0.0})
    monitor.observe("odom", t + 0.3, {"x": 0.0})
    monitor.on_event(slice_end(name, t + 0.5))


class ClosureMonitorTests(unittest.TestCase):
    def monitor(self):
        return observer.ValidationContractMonitor(RUN_ID)

    def assert_evidence_insufficient(self, monitor, reason):
        result = monitor.result()
        self.assertEqual(result["control_contract_status"], "CONTROL_CONTRACT_PASS")
        self.assertEqual(result["online_evidence_status"], "ONLINE_EVIDENCE_INSUFFICIENT")
        self.assertEqual(result["evidence_status_updates"][0]["reason"], reason)
        self.assertEqual(result["evidence_status_updates"][0]["run_id"], RUN_ID)
        self.assertEqual(result["evidence_status_updates"][0]["runner_invocation_id"], "room_local_0001")

    def test_01_runner_native_control_violation_remains_control_failure(self):
        monitor = self.monitor()
        monitor.on_event(event("VALIDATION_CONTRACT_VIOLATION", reason="UNSUPPORTED_COMMAND_ONLINE_VIOLATION:REVERSE"))
        result = monitor.result()
        self.assertEqual(result["control_contract_status"], "CONTROL_CONTRACT_FAILURE")
        self.assertEqual(result["control_failure_reason"], "UNSUPPORTED_COMMAND_ONLINE_VIOLATION:REVERSE")
        self.assertEqual(result["online_evidence_status"], "ONLINE_EVIDENCE_COMPLETE")

    def test_02_duplicate_runner_control_violation_is_idempotent(self):
        monitor = self.monitor()
        monitor.on_event(event("VALIDATION_CONTRACT_VIOLATION", reason="FIRST"))
        monitor.on_event(event("VALIDATION_CONTRACT_VIOLATION", reason="SECOND"))
        self.assertEqual(monitor.result()["control_failure_reason"], "FIRST")

    def test_03_missing_cmd_raw_is_evidence_insufficient_not_control_abort(self):
        monitor = self.monitor()
        monitor.on_event(decision(phase2_productivity_status="PRODUCTIVE_TRANSLATIONAL_SET_NONEMPTY"))
        monitor.on_event(slice_start("s", v=0.3, w=0.0))
        monitor.observe("cmd_final", 10.2, {})
        monitor.observe("odom", 10.3, {})
        monitor.on_event(slice_end("s"))
        self.assert_evidence_insufficient(monitor, "CRITICAL_TELEMETRY_LOSS:cmd_raw")

    def test_04_missing_cmd_final_is_evidence_insufficient_not_control_abort(self):
        monitor = self.monitor()
        monitor.on_event(decision(phase2_productivity_status="PRODUCTIVE_TRANSLATIONAL_SET_NONEMPTY"))
        monitor.on_event(slice_start("s", v=0.3, w=0.0))
        monitor.observe("cmd_raw", 10.1, {})
        monitor.observe("odom", 10.3, {})
        monitor.on_event(slice_end("s"))
        self.assert_evidence_insufficient(monitor, "CRITICAL_TELEMETRY_LOSS:cmd_final")

    def test_05_missing_odom_is_evidence_insufficient_not_control_abort(self):
        monitor = self.monitor()
        monitor.on_event(decision(phase2_productivity_status="PRODUCTIVE_TRANSLATIONAL_SET_NONEMPTY"))
        monitor.on_event(slice_start("s", v=0.3, w=0.0))
        monitor.observe("cmd_raw", 10.1, {})
        monitor.observe("cmd_final", 10.2, {})
        monitor.on_event(slice_end("s"))
        self.assert_evidence_insufficient(monitor, "CRITICAL_TELEMETRY_LOSS:odom")

    def test_06_unresolved_attribution_is_evidence_insufficient(self):
        monitor = self.monitor()
        monitor.on_event(decision(phase2_productivity_status="PRODUCTIVE_TRANSLATIONAL_SET_NONEMPTY"))
        monitor.on_event(slice_start("s", v=0.3, w=0.0))
        monitor.on_event(slice_end("s"))
        self.assert_evidence_insufficient(monitor, "CRITICAL_TELEMETRY_LOSS:cmd_raw,cmd_final,odom")

    def test_07_all_evidence_complete_remains_complete(self):
        monitor = self.monitor()
        monitor.on_event(decision(phase2_productivity_status="PRODUCTIVE_TRANSLATIONAL_SET_NONEMPTY"))
        monitor.on_event(slice_start("s", v=0.3, w=0.0))
        complete_slice(monitor, "s")
        result = monitor.result()
        self.assertEqual(result["control_contract_status"], "CONTROL_CONTRACT_PASS")
        self.assertEqual(result["online_evidence_status"], "ONLINE_EVIDENCE_COMPLETE")

    def test_08_redundant_one_slice_detection_is_audit_only(self):
        monitor = self.monitor()
        monitor.on_event(decision(1))
        monitor.on_event(slice_start("rotate1"))
        complete_slice(monitor, "rotate1")
        monitor.on_event(slice_start("rotate2", t=11.0))
        result = monitor.result()
        self.assertEqual(result["control_contract_status"], "CONTROL_CONTRACT_PASS")
        self.assertTrue(any(row["reason"] == "PHASE3_ONE_SLICE_CONTRACT_VIOLATION" for row in result["audit_discrepancies"]))

    def test_09_rearm_detection_is_audit_only(self):
        monitor = self.monitor()
        intent = {"target_key": ["ROOM", 1, 2], "anchor_odom_xy": [1.0, 2.0], "anchor_tangent_odom_rad": 0.0, "consumed": True}
        monitor.on_event(event("PHASE3_RECOVERYSET_RESULT", orientation_intent_after=intent))
        monitor.on_event(event("PHASE3_RECOVERYSET_RESULT", orientation_intent_after={**intent, "consumed": False}))
        result = monitor.result()
        self.assertEqual(result["control_contract_status"], "CONTROL_CONTRACT_PASS")
        self.assertTrue(any(row["reason"] == "ORIENTATION_INTENT_REARM_WITHOUT_MATERIAL_ROUTE_CHANGE" for row in result["audit_discrepancies"]))


class ClosureIntegrationTests(unittest.TestCase):
    def test_16_state_machine_translates_only_validation_abort(self):
        module = load_module(FakeRos())
        self.assertEqual(module.room_local_validation_abort_reason({"runner": {"runner_final_decision": "ROOM_LOCAL_ONLINE_VALIDATION_ABORT", "validation_abort_reason": "C1"}}), "C1")
        self.assertIsNone(module.room_local_validation_abort_reason({"runner": {"runner_final_decision": "BLOCK_ASTAR_DWA_MAX_STEPS"}}))

    def test_17_runner_stop_issues_and_retains_five_zero_commands(self):
        fake_ros = FakeRos()
        install_message_stubs(fake_ros)
        spec = importlib.util.spec_from_file_location("closure_runner_module", RUNNER_MODULE_PATH)
        runner_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner_module
        assert spec.loader is not None
        spec.loader.exec_module(runner_module)
        class ZeroTwist:
            def __init__(self):
                self.linear = types.SimpleNamespace(x=0.0)
                self.angular = types.SimpleNamespace(z=0.0)
        runner_module.Twist = ZeroTwist
        published = []
        runner = object.__new__(runner_module.BlockAStarDwaRunner)
        runner.args = types.SimpleNamespace(execute=True)
        runner.pub = types.SimpleNamespace(publish=lambda value: published.append(value))
        with mock.patch.object(runner_module.time, "sleep"):
            runner.stop()
        self.assertEqual(len(published), 5)
        self.assertTrue(all(command.linear.x == 0.0 and command.angular.z == 0.0 for command in published))

    def test_18_transit_validation_on_is_unchanged(self):
        module = load_module(FakeRos())
        before = module.runner_cmd(module.build_arg_parser().parse_args([]), state="FOLLOW_CORRIDOR", runtime_sec=1.0, max_steps=1)
        after = module.runner_cmd(module.build_arg_parser().parse_args(["--enable-room-local-guarded-online-validation"]), state="FOLLOW_CORRIDOR", runtime_sec=1.0, max_steps=1)
        self.assertEqual(before, after)

    def test_19_transit_validation_off_is_unchanged(self):
        module = load_module(FakeRos())
        command = module.runner_cmd(module.build_arg_parser().parse_args([]), state="FOLLOW_CORRIDOR", runtime_sec=1.0, max_steps=1)
        self.assertNotIn("--validation-run-id", command)

    def test_20_current_run_complete_allows_next_guarded_room_local_invocation(self):
        module = load_module(FakeRos())
        args = module.build_arg_parser().parse_args(["--execute", "--enable-room-local-guarded-online-validation"])
        cache = module.RoomLocalValidationEvidenceStatusCache(RUN_ID)
        self.addCleanup(cache.close)
        self.assertIsNone(module.guarded_room_local_validation_evidence_gate_reason(args, "ROOM_LOCAL", cache))

    def test_21_current_run_insufficient_blocks_next_guarded_room_local_invocation(self):
        module = load_module(FakeRos())
        args = module.build_arg_parser().parse_args(["--execute", "--enable-room-local-guarded-online-validation"])
        cache = module.RoomLocalValidationEvidenceStatusCache(RUN_ID)
        self.addCleanup(cache.close)
        cache._callback(types.SimpleNamespace(data=json.dumps({"run_id": RUN_ID, "evidence_status": "ONLINE_EVIDENCE_INSUFFICIENT", "reason": "cmd_final"})))
        self.assertEqual(module.guarded_room_local_validation_evidence_gate_reason(args, "ROOM_LOCAL", cache), "cmd_final")

    def test_22_validation_off_and_transit_ignore_known_evidence_insufficient(self):
        module = load_module(FakeRos())
        cache = module.RoomLocalValidationEvidenceStatusCache(RUN_ID)
        self.addCleanup(cache.close)
        cache._callback(types.SimpleNamespace(data=json.dumps({"run_id": RUN_ID, "evidence_status": "ONLINE_EVIDENCE_INSUFFICIENT", "reason": "odom"})))
        validation_off = module.build_arg_parser().parse_args(["--execute"])
        validation_on = module.build_arg_parser().parse_args(["--execute", "--enable-room-local-guarded-online-validation"])
        self.assertIsNone(module.guarded_room_local_validation_evidence_gate_reason(validation_off, "ROOM_LOCAL", cache))
        self.assertIsNone(module.guarded_room_local_validation_evidence_gate_reason(validation_on, "TRANSIT", cache))

    def test_23_dry_preflight_ignores_known_evidence_insufficient(self):
        module = load_module(FakeRos())
        args = module.build_arg_parser().parse_args(["--enable-room-local-guarded-online-validation"])
        cache = module.RoomLocalValidationEvidenceStatusCache(RUN_ID)
        self.addCleanup(cache.close)
        cache._callback(types.SimpleNamespace(data=json.dumps({"run_id": RUN_ID, "evidence_status": "ONLINE_EVIDENCE_INSUFFICIENT", "reason": "raw"})))
        self.assertIsNone(module.guarded_room_local_validation_evidence_gate_reason(args, "ROOM_LOCAL", cache))

    def test_24_stale_previous_run_status_does_not_block_current_run(self):
        module = load_module(FakeRos())
        args = module.build_arg_parser().parse_args(["--execute", "--enable-room-local-guarded-online-validation"])
        cache = module.RoomLocalValidationEvidenceStatusCache(RUN_ID)
        self.addCleanup(cache.close)
        cache._callback(types.SimpleNamespace(data=json.dumps({"run_id": "previous_run", "evidence_status": "ONLINE_EVIDENCE_INSUFFICIENT", "reason": "stale"})))
        self.assertIsNone(module.guarded_room_local_validation_evidence_gate_reason(args, "ROOM_LOCAL", cache))


class ArchiveFinalizerTests(unittest.TestCase):
    def finalize(self, result="ROOM_SEARCH_COMPLETED", control_failure_reason=None, evidence_status="ONLINE_EVIDENCE_COMPLETE", updates=None, missing=()):
        temp = tempfile.TemporaryDirectory()
        archive = Path(temp.name) / "archive"
        telemetry = archive / "telemetry"
        telemetry.mkdir(parents=True)
        files = {
            "state_machine_navigation_summary.json": {"final_decision": result},
            "block_astar_dwa_mature_summary.json": {"final_decision": result, "validation_abort_reason": control_failure_reason},
            "telemetry/command_odom_correlation.json": {
                "control_contract_status": "CONTROL_CONTRACT_FAILURE" if control_failure_reason else "CONTROL_CONTRACT_PASS",
                "control_failure_reason": control_failure_reason,
                "online_evidence_status": evidence_status,
                "evidence_status_updates": list(updates or []),
            },
        }
        for relative, payload in files.items():
            if relative not in missing:
                path = archive / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload))
        (archive / "resolved_parameters.json").write_text(json.dumps({"local_control_mode": "ROOM_LOCAL"}))
        (archive / "terminal_output.log").write_text("synthetic\n")
        command = [sys.executable, str(ROOT / "scripts/local_subgoal_runner_mvp/finalize_room_local_online_validation_archive.py"), "--repo-root", str(ROOT), "--archive-dir", str(archive), "--run-id", RUN_ID, "--exit-status", "0", "--launch-command", "synthetic", "--validation-enabled", "true"]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return temp, json.loads((archive / "room_local_online_validation_manifest.json").read_text())

    def test_20_normal_archive_finalization(self):
        temp, manifest = self.finalize()
        self.addCleanup(temp.cleanup)
        self.assertTrue(manifest["finalized"])
        self.assertEqual(manifest["result"], "ROOM_SEARCH_COMPLETED")
        self.assertEqual(manifest["control_contract_status"], "CONTROL_CONTRACT_PASS")
        self.assertEqual(manifest["online_evidence_status"], "ONLINE_EVIDENCE_COMPLETE")
        self.assertEqual(manifest["violation_status"], "NONE")

    def test_21_runner_control_failure_remains_control_failure(self):
        temp, manifest = self.finalize(result="ROOM_LOCAL_ONLINE_VALIDATION_ABORT", control_failure_reason="UNSUPPORTED_COMMAND_ONLINE_VIOLATION:REVERSE")
        self.addCleanup(temp.cleanup)
        self.assertEqual(manifest["violation_status"], "VIOLATION")
        self.assertEqual(manifest["control_contract_status"], "CONTROL_CONTRACT_FAILURE")
        self.assertEqual(manifest["abort_reason"], "UNSUPPORTED_COMMAND_ONLINE_VIOLATION:REVERSE")

    def test_22_late_evidence_insufficient_is_not_retroactive_control_failure(self):
        temp, manifest = self.finalize(
            evidence_status="ONLINE_EVIDENCE_INSUFFICIENT",
            updates=[{"reason": "CRITICAL_TELEMETRY_LOSS:odom"}],
        )
        self.addCleanup(temp.cleanup)
        self.assertEqual(manifest["control_contract_status"], "CONTROL_CONTRACT_PASS")
        self.assertEqual(manifest["online_evidence_status"], "ONLINE_EVIDENCE_INSUFFICIENT")
        self.assertEqual(manifest["violation_status"], "NONE")
        self.assertEqual(manifest["evidence_insufficient_reason"], "CRITICAL_TELEMETRY_LOSS:odom")
        self.assertEqual(manifest["readiness"], "FAIL")

    def test_23_partial_archive_is_truthful_and_fails_readiness(self):
        temp, manifest = self.finalize(missing=("telemetry/command_odom_correlation.json",))
        self.addCleanup(temp.cleanup)
        self.assertIn("observer_telemetry", manifest["critical_evidence_missing"])
        self.assertEqual(manifest["readiness"], "FAIL")

    def test_24_manifest_records_dirty_tree_source_identity(self):
        temp, manifest = self.finalize()
        self.addCleanup(temp.cleanup)
        identity = manifest["source_identity"]
        self.assertIn("git_status_short", identity)
        self.assertIn("scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py", identity["files"])
        self.assertEqual(identity["relevant_config"]["status"], "PRESENT")
        self.assertTrue(identity["relevant_config"]["sha256"])
        self.assertTrue(identity["head"])

    def test_25_runner_abort_terminal_without_reason_is_still_control_failure(self):
        temp, manifest = self.finalize(result="ROOM_LOCAL_ONLINE_VALIDATION_ABORT")
        self.addCleanup(temp.cleanup)
        self.assertEqual(manifest["control_contract_status"], "CONTROL_CONTRACT_FAILURE")
        self.assertEqual(manifest["control_failure_reason"], "ROOM_LOCAL_ONLINE_VALIDATION_ABORT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
