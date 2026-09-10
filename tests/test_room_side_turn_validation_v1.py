#!/usr/bin/env python3
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "local_subgoal_runner_mvp"))

from room_side_turn_validation_v1 import (  # noqa: E402
    FeedbackYawTurnCore,
    classify_stream_age,
    post_turn_transition,
    select_follow_corridor_opening_action,
)


class OpeningPriorityTests(unittest.TestCase):
    def decide(self, forced, gap, full=False):
        return select_follow_corridor_opening_action(
            doorway_control_enabled=True,
            fully_bounded_doorway_ready=full,
            room_side_gap_enabled=True,
            room_side_gap_trigger_ready=gap,
            forced_room_entry_enabled=forced,
        )

    def test_forced_true_gap_ready(self):
        value = self.decide(True, True)
        self.assertEqual(value["action"], "ROOM_SIDE_TURN")
        self.assertFalse(value["forced_mode_suppresses_gap"])

    def test_forced_true_gap_unavailable(self):
        self.assertEqual(self.decide(True, False)["action"], "FOLLOW_CORRIDOR")

    def test_forced_false_gap_ready(self):
        self.assertEqual(self.decide(False, True)["action"], "ROOM_SIDE_TURN")

    def test_full_doorway_wins_over_gap(self):
        value = self.decide(True, True, full=True)
        self.assertEqual(value["action"], "DOORWAY_VERIFY")
        self.assertEqual(value["reason"], "fully_bounded_doorway_ready_has_priority")


class FeedbackYawTests(unittest.TestCase):
    def core(self, side="left", target=1.0, minimum=0.7, watchdog=90.0):
        return FeedbackYawTurnCore(
            side=side,
            target_yaw_rad=target,
            min_yaw_rad=minimum,
            angular_z=0.3,
            watchdog_sec=watchdog,
            no_response_sample_limit=4,
            yaw_response_epsilon_rad=0.005,
        )

    def test_yaw_unwrap_across_positive_pi(self):
        core = self.core(target=0.25, minimum=0.2)
        core.update(odom_state="FRESH", yaw_rad=3.05, wall_elapsed_sec=0.0)
        core.update(odom_state="FRESH", yaw_rad=-3.08, wall_elapsed_sec=1.0)
        result = core.update(odom_state="FRESH", yaw_rad=-2.94, wall_elapsed_sec=2.0)
        self.assertGreater(result["directed_yaw_rad"], 0.25)
        self.assertEqual(result["stop_reason"], "target_yaw_reached")

    def test_right_turn_sign(self):
        core = self.core(side="right", target=0.3, minimum=0.2)
        first = core.update(odom_state="FRESH", yaw_rad=0.1, wall_elapsed_sec=0.0)
        self.assertLess(first["angular_z"], 0.0)
        core.update(odom_state="FRESH", yaw_rad=-0.1, wall_elapsed_sec=1.0)
        result = core.update(odom_state="FRESH", yaw_rad=-0.25, wall_elapsed_sec=2.0)
        self.assertTrue(result["target_reached"])

    def test_low_rtf_wall_progress_does_not_define_angle_completion(self):
        core = self.core(target=1.0, minimum=0.8, watchdog=90.0)
        result = None
        for index, yaw in enumerate([0.0, 0.22, 0.45, 0.7, 1.02]):
            # 0.1 seconds simulated per 10 seconds wall => RTF 0.01.
            result = core.update(odom_state="FRESH", yaw_rad=yaw, wall_elapsed_sec=index * 10.0)
        self.assertTrue(result["target_reached"])
        self.assertEqual(result["stop_reason"], "target_yaw_reached")

    def test_stale_means_zero_and_wait(self):
        core = self.core()
        core.update(odom_state="FRESH", yaw_rad=0.0, wall_elapsed_sec=0.0)
        result = core.update(odom_state="STALE", yaw_rad=None, wall_elapsed_sec=2.0)
        self.assertEqual(result["angular_z"], 0.0)
        self.assertFalse(result["stopped"])
        self.assertEqual(result["waiting_reason"], "odom_stale_wait_zero")

    def test_lost_is_failed_hold(self):
        core = self.core()
        result = core.update(odom_state="LOST", yaw_rad=None, wall_elapsed_sec=9.0)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["stop_reason"], "odom_lost")

    def test_fresh_samples_without_yaw_response_stop(self):
        core = self.core()
        for index in range(5):
            result = core.update(odom_state="FRESH", yaw_rad=0.0, wall_elapsed_sec=float(index))
        self.assertEqual(result["stop_reason"], "yaw_no_response")

    def test_wall_age_classification(self):
        self.assertEqual(classify_stream_age(0.5, 2.5, 8.0), "FRESH")
        self.assertEqual(classify_stream_age(3.0, 2.5, 8.0), "STALE")
        self.assertEqual(classify_stream_age(8.0, 2.5, 8.0), "LOST")


class PostTurnTransitionTests(unittest.TestCase):
    def test_partial_yaw_never_enters_room(self):
        result = post_turn_transition(
            turn_stop_reason="wall_watchdog",
            minimum_yaw_sufficient=False,
            target_reached=False,
            fresh_grid_count=4,
            required_fresh_grids=2,
            grid_safe_for_navigation=True,
            fresh_side_observation=True,
        )
        self.assertEqual(result["next_state"], "FOLLOW_CORRIDOR")
        self.assertNotEqual(result["next_state"], "ENTER_ROOM")
        self.assertFalse(result["entry_target_allowed"])

    def test_full_yaw_can_only_reach_doorway_verify(self):
        result = post_turn_transition(
            turn_stop_reason="target_yaw_reached",
            minimum_yaw_sufficient=True,
            target_reached=True,
            fresh_grid_count=2,
            required_fresh_grids=2,
            grid_safe_for_navigation=True,
            fresh_side_observation=True,
        )
        self.assertEqual(result["next_state"], "DOORWAY_VERIFY")
        self.assertFalse(result["entry_target_allowed"])

    def test_unsafe_grid_blocks_verify(self):
        result = post_turn_transition(
            turn_stop_reason="target_yaw_reached",
            minimum_yaw_sufficient=True,
            target_reached=True,
            fresh_grid_count=2,
            required_fresh_grids=2,
            grid_safe_for_navigation=False,
            fresh_side_observation=True,
        )
        self.assertEqual(result["next_state"], "FOLLOW_CORRIDOR")


class OnlineAdapterStaticGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "scripts" / "local_subgoal_runner_mvp" / "navigation_state_machine.py").read_text(encoding="utf-8")

    def test_gap_generation_not_guarded_by_forced_mode(self):
        self.assertNotIn(
            "args.enable_room_side_gap_trigger\n                    and not args.enable_forced_room_entry_mvp",
            self.source,
        )

    def test_room_side_turn_block_has_no_entry_target_or_enter_room(self):
        start = self.source.index('elif state == "ROOM_SIDE_TURN":')
        end = self.source.index('elif state == "ENTER_ROOM":', start)
        block = self.source[start:end]
        self.assertNotIn("write_base_target", block)
        self.assertNotIn('next_state, reason = "ENTER_ROOM"', block)
        self.assertIn('item["entry_target_generated"] = False', block)

    def test_global_enter_room_guard_exists(self):
        self.assertIn('args.room_side_turn_validation_v1 and next_state == "ENTER_ROOM"', self.source)


class ArchiveIsolationStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            ROOT / "scripts" / "local_subgoal_runner_mvp" / "run_state_machine_navigation.sh"
        ).read_text(encoding="utf-8")

    def test_archive_name_has_timestamp_and_process_entropy(self):
        self.assertIn('RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S_%N)"', self.source)
        self.assertIn('RUN_NAME="run_${RUN_ID}_${RUN_TIMESTAMP}_pid$$"', self.source)

    def test_shared_latest_copy_requires_matching_run_id(self):
        self.assertIn('data.get("run_id") == sys.argv[2]', self.source)
        self.assertIn('run_id_mismatch', self.source)

    def test_archive_copy_never_clobbers_existing_file(self):
        self.assertIn('cp --no-clobber "$src" "$RUN_ARCHIVE_DIR/$dst_name"', self.source)
        self.assertIn('shared_output_baseline.sha256', self.source)


if __name__ == "__main__":
    unittest.main()
