#!/usr/bin/env python3
"""Offline fake-launcher tests for the P2K-G9U-R2 audit bundle."""
import os
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREPARE = ROOT / "scripts/p2kg9_portal_shadow/prepare_p2kg9u_audit_bundle.sh"
STATUS = ROOT / "scripts/p2kg9_portal_shadow/status_p2kg9u_audit_bundle.sh"
STOP = ROOT / "scripts/p2kg9_portal_shadow/stop_p2kg9u_audit_bundle.sh"
SHADOW_LAUNCHER = ROOT / "scripts/p2kg9_portal_shadow/start_passive_portal_shadow_audit.sh"
CREATE_ENV = ROOT / "scripts/p2kg9_portal_shadow/create_p2kg9_online_env.sh"


class StartupWrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="p2kg9u-r2-")
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.envfile = self.root / "run.env"
        self.envfile.write_text("\n".join([
            f"export REPO_ROOT={ROOT}", "export SHADOW_RUN_ID=fake", f"export SHADOW_OUTPUT_ROOT={self.root}",
            f"export SHADOW_RUN_DIR={self.run}", f"export BAG_DIR={self.run / 'bag'}", "export P2KG9U_TMUX_SESSION=fake", f"export P2KG9U_PANE_FILE={self.root / 'panes'}", "",
        ]), encoding="utf-8")
        (self.root / "panes").write_text(
            "%1 1 p2kg9u-shadow 0 0\n%2 2 p2kg9u-rosbag 0 0\n%3 3 continuous-odom-imu-yaw-shadow 0 0\n",
            encoding="utf-8",
        )

    def tearDown(self): self.tmp.cleanup()

    def launcher(self, name, body):
        path = self.root / name
        path.write_text("#!/usr/bin/env bash\nset -e\n" + body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_fake_shadow_and_fake_rosbag_panes_persist_logs_and_bag(self):
        shadow = self.launcher("shadow", 'mkdir -p "$P2KG9U_FAKE_RUN"\necho shadow\n')
        bag = self.launcher("rosbag", 'while [[ "$#" -gt 0 ]]; do\n  if [[ "$1" == "-O" ]]; then touch "$2"; exit 0; fi\n  shift\ndone\nexit 2\n')
        env = os.environ.copy(); env.update({"P2KG9U_SHADOW_LAUNCHER": str(shadow), "P2KG9U_ROSBAG_LAUNCHER": str(bag), "P2KG9U_FAKE_RUN": str(self.run), "P2KG9U_FAKE_BAG": str(self.run / "bag"), "P2KG9U_START_WAIT_TENTHS": "3"})
        first = subprocess.run(["bash", str(PREPARE), "--shadow-pane", str(self.envfile)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        second = subprocess.run(["bash", str(PREPARE), "--rosbag-pane", str(self.envfile)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        self.assertEqual(first.returncode, 0, first.stdout)
        self.assertEqual(second.returncode, 0, second.stdout)
        self.assertTrue((self.run / "logs/portal_shadow.log").is_file())
        self.assertTrue((self.run / "logs/rosbag.log").is_file())
        self.assertTrue((self.run / "bag/fake.bag").is_file())

    def test_stop_wrapper_closes_supervised_bag_after_session_disappears(self):
        bag = self.launcher(
            "rosbag_supervised",
            'out="$1"\n'
            'touch "$out.active"\n'
            'trap \'mv "$out.active" "$out"; exit 0\' INT\n'
            'while true; do sleep 0.05; done\n',
        )
        completed_bag = self.run / "bag/fake.bag"
        completed_bag.parent.mkdir(parents=True)
        proc = subprocess.Popen([str(bag), str(completed_bag)])
        try:
            deadline = time.monotonic() + 3
            while not completed_bag.with_suffix(".bag.active").is_file() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(completed_bag.with_suffix(".bag.active").is_file())
            (self.run / "rosbag.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
            stop_env = os.environ.copy()
            stop_env["P2KG9U_STOP_WAIT_SEC"] = "3"
            stopped = subprocess.run(
                ["bash", str(STOP), "--env-file", str(self.envfile)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=stop_env,
                timeout=5,
            )
            self.assertEqual(stopped.returncode, 0, stopped.stdout)
            self.assertTrue((self.run / "bag/fake.bag").is_file())
            self.assertFalse((self.run / "bag/fake.bag.active").exists())
            proc.wait(timeout=3)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=3)

    def test_outer_gate_has_bounded_topic_retry_and_rosbag_health_gate(self):
        source = PREPARE.read_text(encoding="utf-8")
        self.assertIn("for attempt in $(seq 1 \"$START_WAIT_TENTHS\")", source)
        self.assertIn("WAITING_FOR_SHADOW_TOPICS", source)
        self.assertIn("ROSBAG_FAILED", source)
        self.assertLess(source.index("STARTING_ROSBAG"), source.index("P2KG9U audit bundle READY"))
        self.assertIn("bag_dead", source)

    def test_angular_telemetry_is_recorded_but_not_stair_readiness(self):
        prepare = PREPARE.read_text(encoding="utf-8")
        status = STATUS.read_text(encoding="utf-8")
        self.assertIn("/audit/p2kg15/angular_actuation_state", prepare)
        self.assertNotIn(
            "/audit/p2kg15/selected_rl_policy /audit/p2kg15/angular_actuation_state /unitree/rl_mode_ready; do",
            prepare,
        )
        self.assertNotIn("/audit/p2kg15/angular_actuation_state /unitree/rl_mode_ready; do", status)

    def test_default_future_bag_root_is_d_drive_and_explicit_output_roots_remain_local(self):
        prepare = PREPARE.read_text(encoding="utf-8")
        self.assertIn('DEFAULT_BAG_OUTPUT_ROOT="/mnt/d/data/simenv_audit_bags/', prepare)
        self.assertIn('BAG_DIR="$BAG_OUTPUT_ROOT/$RUN_ID/bag"', prepare)
        self.assertIn("--bag-output-root", prepare)

        envfile = self.root / "created.env"
        created = subprocess.run(
            ["bash", str(CREATE_ENV), "--run-id", "isolated-test", "--output-root", str(self.root / "output"), "--env-file", str(envfile)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        self.assertEqual(created.returncode, 0, created.stderr)
        self.assertIn(f"export BAG_DIR={self.root / 'output' / 'isolated-test' / 'bag'}", envfile.read_text(encoding="utf-8"))

    def test_icp_odom_info_and_continuity_telemetry_are_audit_only_bag_topics(self):
        prepare = PREPARE.read_text(encoding="utf-8")
        self.assertIn("/team/livox/icp_odom_info", prepare)
        self.assertIn("/team/livox/icp_odom_gate_status", prepare)
        self.assertIn("/audit/startup_anchor/odom_epoch_event", prepare)
        policy_gate = prepare[prepare.index("for topic in /audit/p2kg15/selected_rl_policy"):]
        self.assertNotIn("/team/livox/icp_odom_info", policy_gate)

    def test_status_and_stop_cover_failure_and_partial_start_without_control(self):
        status = STATUS.read_text(encoding="utf-8"); stop = STOP.read_text(encoding="utf-8")
        for token in ("NOT_PREPARED", "STARTING_SHADOW", "PARTIAL_START", "SHADOW_FAILED", "ROSBAG_FAILED", "P2KG9U READY"):
            self.assertIn(token, status)
        self.assertLess(stop.index('"$ROS_BAG_PANE" C-c'), stop.index('"$SHADOW_PANE" C-c'))
        for source in (status, stop, PREPARE.read_text(encoding="utf-8")):
            self.assertNotIn("pkill", source); self.assertNotIn("rostopic pub", source); self.assertNotIn("xdotool", source)

    def test_all_strict_mode_ros_panes_source_setup_without_nounset(self):
        """A clean pane must not fail on ROS_DISTRO being initially unset."""
        prepare = PREPARE.read_text(encoding="utf-8")
        launcher = SHADOW_LAUNCHER.read_text(encoding="utf-8")
        for source in (prepare, launcher):
            self.assertIn("source_ros_workspace()", source)
            function_start = source.index("source_ros_workspace()")
            function_end = source.index("\n}", function_start)
            function = source[function_start:function_end]
            self.assertIn("set +u", function)
            self.assertIn("source /opt/ros/noetic/setup.bash", function)
            self.assertIn("set -u", function)
            self.assertLess(function.index("set +u"), function.index("source /opt/ros/noetic/setup.bash"))
            self.assertLess(function.index("source /opt/ros/noetic/setup.bash"), function.index("set -u"))
        for mode in ("--gate-pane", "--geometry-pane", "--continuous-yaw-shadow-pane", "--rosbag-pane"):
            block = prepare[prepare.index(f'if [[ "$MODE" == "{mode}" ]]'):]
            block = block[:block.index("fi")]
            self.assertIn("source_ros_workspace", block)


if __name__ == "__main__": unittest.main(verbosity=2)
