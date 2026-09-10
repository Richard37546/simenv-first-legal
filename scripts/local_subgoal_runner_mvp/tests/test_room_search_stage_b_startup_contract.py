#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Dict, Optional
import unittest


ROOT = Path(__file__).resolve().parents[3]
CHECKER = ROOT / "scripts/room_search_stage_b_shadow/verify_room_search_stage_b_navigation_contract.sh"
WRAPPER = ROOT / "scripts/p2kg9_portal_shadow/start_p2kg9u_navigation_runner.sh"


class StageBStartupContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.watch = self.root / "snapshots"
        self.results = self.root / "results"
        self.watch.mkdir()
        self.results.mkdir()
        self.pid_file = self.root / "sidecar.pid"
        self.ready_file = self.root / "sidecar.ready.json"
        self.process = subprocess.Popen([
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            str(ROOT / "scripts/local_subgoal_runner_mvp/room_search_stage_b_shadow_sidecar.py"),
            "--watch-dir", str(self.watch),
            "--result-dir", str(self.results),
            "--ros-grid-observer",
            "--ready-file", str(self.ready_file),
            "--one-shot",
        ])
        self.pid_file.write_text(f"{self.process.pid}\n", encoding="utf-8")
        self.ready = {
            "schema_version": "room_search_stage_b_shadow_result_v1",
            "status": "SIDECAR_READY",
            "pid": self.process.pid,
            "one_shot": True,
            "read_only_grid_status_observer": True,
            "watch_dir": str(self.watch),
            "result_dir": str(self.results),
            "production_authority": False,
            "selection_authority": False,
            "command_authority": False,
            "completion_authority": False,
            "recoverability_authority": False,
            "fallback_authority": False,
        }
        self.write_ready()

    def tearDown(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self.temp.cleanup()

    def write_ready(self) -> None:
        self.ready_file.write_text(json.dumps(self.ready), encoding="utf-8")

    def environment(self) -> Dict[str, str]:
        env = dict(os.environ)
        env.update({
            "ROOM_SEARCH_STAGE_B_SHADOW": "true",
            "ROOM_SEARCH_FROZEN_DECISION_CAPTURE": "false",
            "ROOM_SEARCH_STAGE_B_SHADOW_DIR": str(self.watch),
            "ROOM_SEARCH_STAGE_B_SHADOW_RESULT_DIR": str(self.results),
            "ROOM_SEARCH_STAGE_B_SHADOW_PID_FILE": str(self.pid_file),
            "ROOM_SEARCH_STAGE_B_SHADOW_READY_FILE": str(self.ready_file),
        })
        return env

    def run_checker(self, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(CHECKER)],
            cwd=ROOT,
            env=env or self.environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_matching_live_sidecar_contract_passes(self) -> None:
        completed = self.run_checker()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("navigation contract READY", completed.stdout)
        self.assertIn("PATHS_AND_AUTHORITY_MATCH", completed.stdout)

    def test_matching_multi_decision_live_sidecar_contract_passes(self) -> None:
        process = subprocess.Popen([
            sys.executable, "-c", "import time; time.sleep(60)",
            str(ROOT / "scripts/local_subgoal_runner_mvp/room_search_stage_b_shadow_sidecar.py"),
            "--watch-dir", str(self.watch), "--result-dir", str(self.results),
            "--ros-grid-observer", "--ready-file", str(self.ready_file), "--mode", "MULTI_DECISION",
        ])
        try:
            self.pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
            self.ready.update({"pid": process.pid, "mode": "MULTI_DECISION", "one_shot": False})
            self.write_ready()
            completed = self.run_checker()
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("PATHS_AND_AUTHORITY_MATCH", completed.stdout)
        finally:
            process.terminate()
            process.wait(timeout=2)

    def test_stale_capture_directory_fails_closed(self) -> None:
        stale = self.root / "old-run" / "snapshots"
        stale.mkdir(parents=True)
        env = self.environment()
        env["ROOM_SEARCH_STAGE_B_SHADOW_DIR"] = str(stale)
        completed = self.run_checker(env)
        self.assertEqual(completed.returncode, 78)
        self.assertIn("capture dir", completed.stderr)
        self.assertIn("sidecar watch_dir", completed.stderr)

    def test_ready_pid_mismatch_fails_closed(self) -> None:
        self.ready["pid"] = self.process.pid + 1
        self.write_ready()
        completed = self.run_checker()
        self.assertEqual(completed.returncode, 78)
        self.assertIn("READY pid", completed.stderr)

    def test_unrelated_live_pid_fails_closed(self) -> None:
        unrelated = subprocess.Popen(["sleep", "60"])
        try:
            self.pid_file.write_text(f"{unrelated.pid}\n", encoding="utf-8")
            self.ready["pid"] = unrelated.pid
            self.write_ready()
            completed = self.run_checker()
            self.assertEqual(completed.returncode, 78)
            self.assertIn("not room_search_stage_b_shadow_sidecar.py", completed.stderr)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=2)

    def test_authority_violation_fails_closed(self) -> None:
        self.ready["selection_authority"] = True
        self.write_ready()
        completed = self.run_checker()
        self.assertEqual(completed.returncode, 78)
        self.assertIn("authority isolation violated", completed.stderr)

    def test_conflicting_frozen_capture_fails_closed(self) -> None:
        env = self.environment()
        env["ROOM_SEARCH_FROZEN_DECISION_CAPTURE"] = "true"
        completed = self.run_checker(env)
        self.assertEqual(completed.returncode, 78)
        self.assertIn("conflicting", completed.stderr)

    def test_disabled_shadow_preserves_existing_navigation_path(self) -> None:
        env = self.environment()
        env["ROOM_SEARCH_STAGE_B_SHADOW"] = "false"
        completed = self.run_checker(env)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("SKIPPED", completed.stdout)

    def test_navigation_wrapper_checks_contract_before_exec(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        check_position = source.index("verify_room_search_stage_b_navigation_contract.sh")
        odom_preflight_position = source.index("wait_for_navigation_odom_ready.py")
        exec_position = source.index("exec bash")
        self.assertLess(odom_preflight_position, check_position)
        self.assertLess(check_position, exec_position)
        self.assertIn("P2KG9U_NAVIGATION_ODOM_WAIT_SEC", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
