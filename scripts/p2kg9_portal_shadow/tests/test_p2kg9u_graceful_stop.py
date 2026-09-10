#!/usr/bin/env python3
"""Offline tests for the P2K-G9U graceful audit-bundle stop wrapper."""

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STOP = ROOT / "scripts" / "p2kg9_portal_shadow" / "stop_p2kg9u_audit_bundle.sh"


class GracefulStopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="p2kg9u-stop-")
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "run"
        self.bag_dir = self.run_dir / "bag"
        self.yaw_dir = self.run_dir / "continuous_yaw_shadow"
        self.bag_dir.mkdir(parents=True)
        self.yaw_dir.mkdir()
        (self.bag_dir / "test_run.bag").write_bytes(b"fixture")
        (self.yaw_dir / "ready.json").write_text("{}\n", encoding="utf-8")
        (self.yaw_dir / "shadow_events.jsonl").write_text("{}\n", encoding="utf-8")
        (self.yaw_dir / "shadow_result.json").write_text(
            json.dumps(
                {
                    "status": "CLOSED",
                    "audit_authority": True,
                    "command_authority": False,
                    "odom_authority": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.write_shutdown(0)
        self.panes = self.root / "panes.txt"
        self.panes.write_text(
            "%shadow 101 p2kg9u-shadow\n"
            "%bag 102 p2kg9u-rosbag\n"
            "%yaw 103 continuous-odom-imu-yaw-shadow\n",
            encoding="utf-8",
        )
        self.env_file = self.root / "bundle.env"
        self.env_file.write_text(
            "\n".join(
                [
                    "export REPO_ROOT=" + str(ROOT),
                    "export SHADOW_RUN_ID=test_run",
                    "export SHADOW_OUTPUT_ROOT=" + str(self.root),
                    "export SHADOW_RUN_DIR=" + str(self.run_dir),
                    "export BAG_DIR=" + str(self.bag_dir),
                    "export CONTINUOUS_YAW_SHADOW_DIR=" + str(self.yaw_dir),
                    "export P2KG9U_TMUX_SESSION=test-session",
                    "export P2KG9U_PANE_FILE=" + str(self.panes),
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def write_shutdown(self, remaining):
        (self.run_dir / "frame_accounting.json").write_text("{}\n", encoding="utf-8")
        (self.run_dir / "queue_and_shutdown.json").write_text(
            json.dumps(
                {
                    "compute_queue_remaining": remaining,
                    "queue_remaining": remaining,
                    "compute_worker_alive": False,
                    "worker_alive": False,
                    "writer_errors": [],
                }
            ),
            encoding="utf-8",
        )

    def run_stop(self, *, dry_run=False, fake_tmux=False):
        env = os.environ.copy()
        env["P2KG9U_STOP_WAIT_SEC"] = "1"
        if fake_tmux:
            fake_bin = self.root / "bin"
            fake_bin.mkdir()
            log = self.root / "tmux.log"
            state = self.root / "panes"
            state.mkdir()
            for pane in ("%shadow", "%bag", "%yaw"):
                (state / pane).touch()
            tmux = fake_bin / "tmux"
            tmux.write_text(
                "#!/usr/bin/env bash\n"
                "echo \"$*\" >> \"$TMUX_LOG\"\n"
                "case \"$1\" in\n"
                "  has-session) exit 0 ;;\n"
                "  send-keys) rm -f \"$TMUX_STATE/$3\"; exit 0 ;;\n"
                "  list-panes) for p in \"$TMUX_STATE\"/*; do [[ -e \"$p\" ]] && basename \"$p\"; done; exit 0 ;;\n"
                "  kill-session) exit 0 ;;\n"
                "  *) exit 90 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            tmux.chmod(tmux.stat().st_mode | stat.S_IXUSR)
            env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
            env["TMUX_LOG"] = str(log)
            env["TMUX_STATE"] = str(state)
        command = ["bash", str(STOP), "--env-file", str(self.env_file)]
        if dry_run:
            command.append("--dry-run")
        return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

    def test_dry_run_prints_rosbag_before_shadow_without_invoking_tmux(self):
        result = self.run_stop(dry_run=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertLess(result.stdout.index("rosbag pane %bag"), result.stdout.index("Shadow pane %shadow"))
        self.assertIn("queue_and_shutdown", result.stdout)

    def test_actual_stop_sends_ctrl_c_in_order_then_cleans_session(self):
        result = self.run_stop(fake_tmux=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        log = (self.root / "tmux.log").read_text(encoding="utf-8")
        self.assertLess(log.index("send-keys -t %bag C-c"), log.index("send-keys -t %shadow C-c"))
        self.assertLess(log.index("send-keys -t %shadow C-c"), log.index("send-keys -t %yaw C-c"))
        self.assertLess(log.index("send-keys -t %yaw C-c"), log.index("kill-session -t test-session"))
        self.assertIn("Shadow shutdown verified", result.stdout)
        self.assertIn("Continuous yaw Shadow shutdown verified", result.stdout)

    def test_failed_shadow_shutdown_preserves_session(self):
        self.write_shutdown(1)
        result = self.run_stop(fake_tmux=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Shadow shutdown verification failed", result.stdout)
        log = (self.root / "tmux.log").read_text(encoding="utf-8")
        self.assertNotIn("kill-session", log)

    def test_source_has_no_broad_or_production_stop(self):
        source = STOP.read_text(encoding="utf-8")
        self.assertNotIn("pkill", source)
        self.assertNotIn("start_runtime_stack_tmux", source)
        self.assertNotIn("run_state_machine_navigation.sh", source)
        self.assertNotIn("rostopic pub", source)
        self.assertNotIn("xdotool", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
