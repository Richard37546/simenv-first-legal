#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import tempfile
import unittest
import copy


ROOT = pathlib.Path(__file__).resolve().parents[3]
RECORDER_PATH = ROOT / "scripts/startup_anchor_evidence/startup_anchor_audit_recorder.py"
ANALYZER_PATH = ROOT / "scripts/startup_anchor_evidence/analyze_spawn_anchor_capture.py"
GATE_PATH = ROOT / "scripts/l2_livox_icp_rtabmap_readiness/l2_livox_odom_gate.py"
AUTO_PATH = ROOT / "auto.sh"
RUNTIME_PATH = ROOT / "scripts/local_subgoal_runner_mvp/start_runtime_stack_tmux.sh"
P2KG_PATH = ROOT / "scripts/p2kg9_portal_shadow/prepare_p2kg9u_audit_bundle.sh"
PREPARE_PATH = ROOT / "scripts/controlled_online_yaw_shadow/prepare_manual_run.sh"
SLAM_PATH = ROOT / "scripts/slam_bev_runtime/run_slam_bev_runtime.sh"

SPEC = importlib.util.spec_from_file_location("startup_anchor_recorder", str(RECORDER_PATH))
RECORDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECORDER)

ANALYZER_SPEC = importlib.util.spec_from_file_location("spawn_anchor_analyzer", str(ANALYZER_PATH))
ANALYZER = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(ANALYZER)

GATE_SPEC = importlib.util.spec_from_file_location("l2_livox_odom_gate", str(GATE_PATH))
GATE = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(GATE)


class StartupAnchorEvidenceTests(unittest.TestCase):
    def test_archive_is_run_scoped_and_oracle_is_separate(self):
        with tempfile.TemporaryDirectory(prefix="startup-anchor-") as tmp:
            root = pathlib.Path(tmp)
            run1 = RECORDER.ArchiveWriter("RUN1", root / "run1")
            tracker = RECORDER.SimTimeTracker("RUN1")
            run1.mark_ready(root / "run1" / "external.ready.json", **tracker.callback_fields())
            tracker.observe_clock(0.0)
            run1.production_event("FIRST_ACCEPT", **tracker.callback_fields())
            run1.oracle_event("GAZEBO_MODEL_STATE", model_name="a1_gazebo", **tracker.callback_fields())
            run1.launcher_marker("SPAWN_AND_PHYSICS_LAUNCH_REQUESTED", paused="false")
            run1.finalize("TEST_STOP")
            run2 = RECORDER.ArchiveWriter("RUN2", root / "run2")
            run2.production_event("FIRST_ACCEPT")
            rows1 = [json.loads(line) for line in (root / "run1" / "production_evidence.jsonl").read_text().splitlines()]
            oracle = [json.loads(line) for line in (root / "run1" / "test_only_gazebo_oracle.jsonl").read_text().splitlines()]
            rows2 = [json.loads(line) for line in (root / "run2" / "production_evidence.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["run_id"] == "RUN1" for row in rows1 + oracle))
            self.assertTrue(all(row["provenance"] == "TEST_ONLY_ORACLE" for row in oracle))
            self.assertEqual(rows1[0]["sim_time_status"], "UNAVAILABLE_NO_CLOCK")
            self.assertEqual(oracle[0]["sim_time_at_callback_sec"], 0.0)
            self.assertIn("wall_time_unix_sec", oracle[0])
            self.assertEqual([row["run_id"] for row in rows2], ["RUN2"])
            self.assertTrue((root / "run1" / "recorder.final.json").is_file())
            marker = json.loads((root / "run1" / "launcher_events.jsonl").read_text().strip())
            self.assertEqual(marker["event_type"], "SPAWN_AND_PHYSICS_LAUNCH_REQUESTED")
            self.assertEqual(marker["provenance"], "PRODUCTION_LAUNCH_METADATA")

    def test_archive_rejects_cross_run_reuse(self):
        with tempfile.TemporaryDirectory(prefix="startup-anchor-") as tmp:
            archive = pathlib.Path(tmp) / "one"
            RECORDER.ArchiveWriter("RUN1", archive)
            with self.assertRaisesRegex(RuntimeError, "archive_run_id_mismatch"):
                RECORDER.ArchiveWriter("RUN2", archive)

    def test_effective_spawn_provenance_requires_explicit_complete_input(self):
        complete = RECORDER.effective_spawn_provenance({
            "STARTUP_ANCHOR_SPAWN_SOURCE": "EXPLICIT_AUDIT_WRAPPER_ENV",
            "ROBOT_X": "0.0", "ROBOT_Y": "-2.2", "ROBOT_Z": "0.6", "ROBOT_YAW": "1.5708",
        })
        self.assertEqual(complete["status"], "RECORDED_EXPLICIT_EFFECTIVE_INPUT")
        self.assertEqual(complete["values"]["ROBOT_Y"], "-2.2")
        missing = RECORDER.effective_spawn_provenance({"ROBOT_X": "0.0"})
        self.assertEqual(missing["status"], "UNAVAILABLE")

    def test_spawn_candidate_uses_only_spawn_and_odom(self):
        anchor = {"pose": {"position_xyz": [1.0, 2.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}}
        candidate = ANALYZER.candidate_transform([10.0, -3.0, 0.5, 0.0], anchor)
        estimated = ANALYZER.estimate_world_pose(candidate, anchor)
        self.assertEqual([round(value, 8) for value in estimated["position_xyz"]], [10.0, -3.0, 0.5])
        source = ANALYZER_PATH.read_text(encoding="utf-8")
        estimate_section = source[source.index("def candidate_transform"):source.index("def _rows")]
        self.assertNotIn("oracle_rows", estimate_section)
        self.assertNotIn("_nearest_oracle", estimate_section)

    def test_offline_analysis_scores_oracle_only_after_candidate_construction(self):
        with tempfile.TemporaryDirectory(prefix="spawn-anchor-analysis-") as tmp:
            archive = pathlib.Path(tmp)
            epoch = "RUN1:sim-0001"
            provenance = {
                "run_id": "RUN1",
                "effective_spawn_provenance": {
                    "status": "RECORDED_EXPLICIT_EFFECTIVE_INPUT",
                    "source": "EXPLICIT_AUDIT_WRAPPER_ENV",
                    "values": {"ROBOT_X": "1.0", "ROBOT_Y": "2.0", "ROBOT_Z": "0.5", "ROBOT_YAW": "0.0"},
                },
            }
            gated = {
                "event_type": "GATED_ODOM", "message_stamp_sec": 1.0,
                "header_frame_id": "team_livox_odom", "child_frame_id": "base", "sim_time_epoch_id": epoch,
                "pose": {"position_xyz": [0.0, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
            }
            accepted = {"event_type": "EXTERNAL_AUDIT_EVENT", "source": "ODOM_EPOCH_AUDIT", "payload": json.dumps({"event_type": "FIRST_ACCEPT", "continuity_id": "RUN1:continuity-0001", "continuity_state": "CONTINUOUS"})}
            oracle = {"provenance": "TEST_ONLY_ORACLE", "event_type": "GAZEBO_MODEL_STATE", "model_identity": "a1_gazebo", "sim_time_epoch_id": epoch, "sim_time_at_callback_sec": 1.0, "position_xyz": [1.0, 2.0, 0.5]}
            (archive / "provenance.json").write_text(json.dumps(provenance))
            (archive / "production_evidence.jsonl").write_text(json.dumps(accepted) + "\n" + json.dumps(gated) + "\n")
            (archive / "test_only_gazebo_oracle.jsonl").write_text(json.dumps(oracle) + "\n")
            result = ANALYZER.analyze(archive)
            self.assertEqual(result["statistics"]["sample_count"], 1)
            self.assertEqual(result["statistics"]["anchor_error_3d_m"], 0.0)
            self.assertEqual(result["oracle_role"], "TEST_ONLY_ORACLE_AFTER_ESTIMATE_CONSTRUCTION")

    def test_sim_time_distinguishes_unavailable_zero_pause_unpause_and_reset(self):
        tracker = RECORDER.SimTimeTracker("RUN1")
        self.assertEqual(tracker.callback_fields()["sim_time_status"], "UNAVAILABLE_NO_CLOCK")
        zero = tracker.observe_clock(0.0)
        self.assertTrue(zero["accepted"])
        self.assertEqual(zero["sim_time_at_callback_sec"], 0.0)
        self.assertEqual(zero["sim_time_epoch_id"], "RUN1:sim-0001")
        advancing = tracker.observe_clock(4.0)
        paused = tracker.observe_clock(4.0)
        self.assertFalse(paused["clock_regressed"])
        self.assertEqual(paused["sim_time_at_callback_sec"], advancing["sim_time_at_callback_sec"])
        unpaused = tracker.observe_clock(4.25)
        self.assertEqual(unpaused["sim_time_at_callback_sec"], 4.25)
        reset = tracker.observe_clock(0.25)
        self.assertTrue(reset["clock_regressed"])
        self.assertEqual(reset["sim_time_epoch_id"], "RUN1:sim-0002")

    def test_oracle_samples_bracket_odom_only_within_the_same_sim_epoch(self):
        rows = [
            {"provenance": "TEST_ONLY_ORACLE", "sim_time_epoch_id": "RUN1:sim-0001", "sim_time_at_callback_sec": 1.0, "model_identity": "a1_gazebo"},
            {"provenance": "TEST_ONLY_ORACLE", "sim_time_epoch_id": "RUN1:sim-0001", "sim_time_at_callback_sec": 1.2, "model_identity": "a1_gazebo"},
            {"provenance": "TEST_ONLY_ORACLE", "sim_time_epoch_id": "RUN1:sim-0002", "sim_time_at_callback_sec": 0.1, "model_identity": "a1_gazebo"},
        ]
        bracket = RECORDER.nearest_oracle_samples(rows, 1.1, "RUN1:sim-0001")
        self.assertEqual(bracket["earlier_or_equal"]["sim_time_at_callback_sec"], 1.0)
        self.assertEqual(bracket["later_or_equal"]["sim_time_at_callback_sec"], 1.2)

    def test_gated_odom_keeps_message_stamp_and_callback_sim_time_separate(self):
        tracker = RECORDER.SimTimeTracker("RUN1")
        tracker.observe_clock(3.0)
        msg = GATE.Odometry()
        msg.header.stamp = GATE.rospy.Time.from_sec(2.875)
        msg.header.frame_id = "team_livox_odom"; msg.child_frame_id = "base"
        msg.pose.pose.position.x = 1.0
        row = RECORDER.odom_evidence_row(msg, tracker)
        self.assertEqual(row["message_stamp_sec"], 2.875)
        self.assertEqual(row["sim_time_at_callback_sec"], 3.0)
        self.assertEqual(row["sim_time_epoch_id"], "RUN1:sim-0001")

    def test_gate_continuity_identity_is_explicit_and_invalidates_without_rebase(self):
        identity = GATE.ContinuityAuthority("RUN1")
        first = identity.event("FIRST_ACCEPT")
        continuous = identity.event("GATED_ACCEPT")
        identity.invalidate("delta_translation_exceeded")
        discontinuity = identity.event("CONTINUITY_INVALID")
        next_run = GATE.ContinuityAuthority("RUN2").event("FIRST_ACCEPT")
        self.assertEqual(first["continuity_id"], "RUN1:continuity-0001")
        self.assertEqual(continuous["continuity_id"], first["continuity_id"])
        self.assertEqual(discontinuity["continuity_state"], "REBASE_REQUIRED")
        self.assertEqual(discontinuity["localization_authority"], "UNAVAILABLE")
        self.assertEqual(next_run["run_epoch_id"], "RUN2")
        self.assertEqual(next_run["event_sequence"], 1)

    def test_launcher_orders_recorder_ready_before_roslaunch_and_never_waits_for_gated_odom(self):
        auto = AUTO_PATH.read_text(encoding="utf-8")
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertLess(auto.index("start_startup_anchor_audit"), auto.rindex("roslaunch unitree_guide multi_floor_gazeboSim.launch"))
        sidecar_start = auto.index("\n# The audit wrapper has already made ROS available here")
        recorder_block = auto[auto.index("start_startup_anchor_audit() {"):sidecar_start]
        self.assertIn("STARTUP_ANCHOR_AUDIT_FAILED: recorder did not become ready before spawn", recorder_block)
        self.assertNotIn("icp_odom_gated", recorder_block)
        self.assertIn("SPAWN_AND_PHYSICS_LAUNCH_REQUESTED", auto)
        self.assertLess(sidecar_start, auto.rindex("roslaunch unitree_guide multi_floor_gazeboSim.launch"))
        self.assertIn("run_world_result_coordinate.sh", auto[sidecar_start:])
        self.assertIn("STARTUP_ANCHOR_AUDIT_ENABLED='$STARTUP_ANCHOR_AUDIT_ENABLED'", runtime)
        self.assertGreater(P2KG_PATH.read_text(encoding="utf-8").index("/team/livox/icp_odom_gated"), 0)

    def test_controlled_run_identity_reaches_recorder_and_gate(self):
        prepare = PREPARE_PATH.read_text(encoding="utf-8")
        slam = SLAM_PATH.read_text(encoding="utf-8")
        for token in ("STARTUP_ANCHOR_AUDIT_ENABLED=1", "STARTUP_ANCHOR_RUN_ID", "STARTUP_ANCHOR_ARCHIVE_DIR", "STARTUP_ANCHOR_READY_FILE"):
            self.assertIn(token, prepare)
        self.assertIn('_audit_run_epoch_id:="${STARTUP_ANCHOR_RUN_ID:-UNSCOPED_RUN}"', slam)
        self.assertIn("_audit_event_topic:=/audit/startup_anchor/odom_epoch_event", slam)
        self.assertIn("stop_prior_odom_gates", slam)
        self.assertIn("/slam_bev_odom_gate", slam)
        self.assertIn("/l2_livox_odom_gate", slam)
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("export STARTUP_ANCHOR_RUN_ID='$STARTUP_ANCHOR_RUN_ID'", runtime)

    def test_controlled_wrapper_requires_explicit_spawn_input(self):
        prepare = PREPARE_PATH.read_text(encoding="utf-8")
        for token in ("SPAWN_ROBOT_X", "SPAWN_ROBOT_Y", "SPAWN_ROBOT_Z", "SPAWN_ROBOT_YAW"):
            self.assertIn(token, prepare)
        self.assertIn("STARTUP_ANCHOR_SPAWN_SOURCE=EXPLICIT_AUDIT_WRAPPER_ENV", prepare)

    def test_recorder_has_no_production_authority_publishers_and_gate_output_precedes_audit(self):
        recorder = RECORDER_PATH.read_text(encoding="utf-8")
        gate = GATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("rospy.Publisher", recorder)
        for forbidden in ("tf.TransformBroadcaster", "detected_danger.json"):
            self.assertNotIn(forbidden, recorder)
        accept = gate[gate.index("    def accept("):gate.index("    def callback(")]
        self.assertLess(accept.index("self.pub.publish(out)"), accept.index("self.audit_event("))
        self.assertIn("out = copy.deepcopy(msg)", accept)

    def test_audit_event_does_not_change_the_accepted_gated_odom_payload(self):
        class Publisher:
            def __init__(self): self.rows = []
            def publish(self, row): self.rows.append(row)
        with tempfile.TemporaryDirectory(prefix="startup-anchor-") as tmp:
            gate = GATE.LivoxOdomGate.__new__(GATE.LivoxOdomGate)
            gate.valid_count = 0
            gate.reject_count = 0
            gate.output_frame_id = "team_livox_odom"
            gate.output_child_frame_id = "base"
            gate.pub = Publisher(); gate.status_pub = Publisher(); gate.audit_event_pub = Publisher()
            gate.continuity = GATE.ContinuityAuthority("RUN1")
            gate.events_path = str(pathlib.Path(tmp) / "gate.jsonl")
            msg = GATE.Odometry()
            msg.header.frame_id = "raw_frame"; msg.child_frame_id = "raw_child"
            msg.header.stamp = GATE.rospy.Time.from_sec(12.5)
            msg.pose.pose.position.x = 1.25; msg.pose.pose.position.y = -0.5; msg.pose.pose.position.z = 0.75
            msg.pose.pose.orientation.z = 0.2; msg.pose.pose.orientation.w = 0.98
            before = copy.deepcopy(msg)
            original_now = GATE.rospy.Time.now
            GATE.rospy.Time.now = lambda: GATE.rospy.Time.from_sec(12.5)
            try:
                gate.accept(msg, delta_translation=0.0, delta_yaw_deg=0.0)
            finally:
                GATE.rospy.Time.now = original_now
            out = gate.pub.rows[0]
            self.assertEqual(msg.header.frame_id, before.header.frame_id)
            self.assertEqual(msg.child_frame_id, before.child_frame_id)
            self.assertEqual(out.header.frame_id, "team_livox_odom")
            self.assertEqual(out.child_frame_id, "base")
            self.assertEqual(out.pose.pose.position.x, before.pose.pose.position.x)
            self.assertEqual(out.pose.pose.position.y, before.pose.pose.position.y)
            self.assertEqual(out.pose.pose.orientation.z, before.pose.pose.orientation.z)
            event = json.loads(gate.audit_event_pub.rows[0].data)
            self.assertEqual(event["event_type"], "FIRST_ACCEPT")
            self.assertEqual(event["run_epoch_id"], "RUN1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
