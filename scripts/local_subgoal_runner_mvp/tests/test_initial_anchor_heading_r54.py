#!/usr/bin/env python3
"""R54 initial corridor-axis regression tests; no ROS runtime or motion."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

import navigation_state_machine as nav


class _MatureEvidence:
    def __init__(self, result):
        self.result = result
        self.reset_count = 0

    def mature_bound_axis(self, *_args):
        return dict(self.result)

    def reset_maturity(self):
        self.reset_count += 1


class _EpochCache:
    def __init__(self, epoch):
        self.epoch = epoch

    def current_epoch_generation(self):
        return self.epoch


class InitialAnchorHeadingTests(unittest.TestCase):
    def setUp(self):
        self.args = nav.build_arg_parser().parse_args([])

    def test_historical_small_heading_keeps_odom_x_stabilization(self):
        heading, applied, reason = nav.select_initial_anchor_heading(0.07648868174923161, self.args)
        self.assertEqual(heading, 0.0)
        self.assertTrue(applied)
        self.assertEqual(reason, "default_odom_x_corridor_axis")

    def test_threshold_boundaries_and_angle_wrap(self):
        for raw_heading in (0.0, 0.05, -0.05, 0.119, -0.119, 2.0 * nav.math.pi - 0.05, -2.0 * nav.math.pi + 0.05):
            heading, applied, _reason = nav.select_initial_anchor_heading(raw_heading, self.args)
            self.assertEqual(heading, 0.0)
            self.assertTrue(applied)
        for raw_heading in (0.120001, -0.120001, nav.math.pi - 0.05, -nav.math.pi + 0.05):
            heading, applied, reason = nav.select_initial_anchor_heading(raw_heading, self.args)
            self.assertEqual(heading, raw_heading)
            self.assertFalse(applied)
            self.assertEqual(reason, "raw_heading_outside_odom_x_snap_threshold")

    def test_current_93_degree_heading_is_retained(self):
        raw_heading = 1.6250094716206724
        heading, applied, reason = nav.select_initial_anchor_heading(raw_heading, self.args)
        self.assertEqual(heading, raw_heading)
        self.assertFalse(applied)
        self.assertEqual(reason, "raw_heading_outside_odom_x_snap_threshold")

    def test_explicit_launch_heading_prevents_n5_bearing_from_redefining_axis(self):
        args = nav.build_arg_parser().parse_args([
            "--entry-anchor-configured-odom-heading-rad", str(nav.math.pi / 2.0),
        ])
        heading, applied, reason = nav.select_initial_anchor_heading(1.6438317633827753, args)
        self.assertAlmostEqual(heading, nav.math.pi / 2.0)
        self.assertFalse(applied)
        self.assertEqual(reason, "configured_initial_odom_heading")

    def test_configured_launch_heading_is_normalized(self):
        args = nav.build_arg_parser().parse_args([
            "--entry-anchor-configured-odom-heading-rad", str(2.0 * nav.math.pi + 0.2),
        ])
        heading, _applied, reason = nav.select_initial_anchor_heading(0.0, args)
        self.assertAlmostEqual(heading, 0.2)
        self.assertEqual(reason, "configured_initial_odom_heading")

    def test_explicit_opt_out_preserves_raw_heading_experiment(self):
        args = nav.build_arg_parser().parse_args(["--no-entry-anchor-snap-heading-to-odom-x"])
        heading, applied, reason = nav.select_initial_anchor_heading(0.1215778657529317, args)
        self.assertEqual(heading, 0.1215778657529317)
        self.assertFalse(applied)
        self.assertEqual(reason, "raw_heading_explicitly_requested")

    def test_room_zone_preserves_snapped_axis_until_portal_handoff(self):
        raw_heading = 0.07648868174923161
        anchor = {
            "heading_rad": 0.0,
            "raw_heading_rad": raw_heading,
            "heading_snap_applied": True,
            "room_zone_heading_restored": False,
        }
        report = nav.restore_raw_anchor_heading_at_room_zone(anchor, self.args)
        self.assertFalse(report["restored"])
        self.assertEqual(
            report["reason"],
            "room_zone_preserves_snapped_odom_x_corridor_axis_until_portal_handoff",
        )
        self.assertEqual(anchor["heading_rad"], 0.0)
        self.assertFalse(anchor["room_zone_heading_restored"])
        repeated = nav.restore_raw_anchor_heading_at_room_zone(anchor, self.args)
        self.assertFalse(repeated["restored"])
        self.assertEqual(
            repeated["reason"],
            "room_zone_preserves_snapped_odom_x_corridor_axis_until_portal_handoff",
        )

    def test_room_zone_does_not_change_explicit_raw_heading_mode(self):
        args = nav.build_arg_parser().parse_args(["--no-entry-anchor-snap-heading-to-odom-x"])
        anchor = {"heading_rad": 0.121, "raw_heading_rad": 0.121, "room_zone_heading_restored": False}
        report = nav.restore_raw_anchor_heading_at_room_zone(anchor, args)
        self.assertFalse(report["restored"])
        self.assertEqual(report["reason"], "raw_heading_already_explicitly_requested")
        self.assertEqual(anchor["heading_rad"], 0.121)

    def test_room_zone_does_not_mislabel_unsnapped_93_degree_anchor(self):
        raw_heading = 1.6250094716206724
        anchor = {
            "heading_rad": raw_heading,
            "raw_heading_rad": raw_heading,
            "heading_snap_applied": False,
            "room_zone_heading_restored": False,
        }
        report = nav.restore_raw_anchor_heading_at_room_zone(anchor, self.args)
        self.assertFalse(report["restored"])
        self.assertEqual(report["reason"], "raw_heading_retained_no_odom_x_snap")
        self.assertEqual(anchor["heading_rad"], raw_heading)

    def test_rotated_odom_frame_keeps_anchor_target_forward_in_robot_base(self):
        for yaw in (0.0, nav.math.pi / 2.0):
            heading, applied, _reason = nav.select_initial_anchor_heading(yaw, self.args)
            self.assertEqual(applied, yaw == 0.0)
            target = nav.transform_base_xy((2.0, 0.0), (0.0, 0.0, heading))
            base_target = nav.target_base_xy(target, (0.0, 0.0, yaw))
            self.assertAlmostEqual(base_target[0], 2.0, places=9)
            self.assertAlmostEqual(base_target[1], 0.0, places=9)

    def test_anchor_target_consumes_final_selected_heading(self):
        raw_heading = 1.6250094716206724
        selected_heading, applied, _reason = nav.select_initial_anchor_heading(raw_heading, self.args)
        self.assertFalse(applied)
        anchor = {"x": 0.05888795852661133, "y": 0.049908969551324844, "heading_rad": selected_heading}
        pose = (0.05335087329149246, 0.05005223676562309, 1.622213918918339)
        with patch.object(nav, "read_odom", return_value={"pose_x_y_yaw": list(pose)}), patch.object(
            nav, "anchor_metrics", return_value={"anchor_progress_m": 0.0}
        ), patch.object(
            nav,
            "write_absolute_target",
            side_effect=lambda target_xy, source, subgoal_source, extra: {"target_xy": target_xy, "source": source, "subgoal_source": subgoal_source, "extra": extra},
        ):
            result = nav.write_anchor_target(anchor, 2.374675005674362, "test", room_zone_active=False)
        base_target = nav.target_base_xy(result["target_xy"], pose)
        self.assertAlmostEqual(base_target[0], 2.3742380704697146, places=6)
        self.assertAlmostEqual(base_target[1], 0.0011161162312167544, places=6)

    def test_mature_geometry_handoffs_once_and_n5_can_no_longer_override_axis(self):
        legacy = {
            "x": 0.0, "y": 0.0, "yaw_rad": 1.63757, "heading_rad": 1.63757,
            "raw_heading_rad": 1.63757, "legacy_bootstrap_heading_rad": 1.63757,
            "corridor_axis_lifecycle": "LEGACY_BOOTSTRAP", "corridor_axis_odom_epoch_generation": 0,
        }
        evidence = {
            "valid": True, "heading_odom_rad": 1.5605,
            "odom_binding": {"source_pose_x_y_yaw": [0.1, 3.2, 1.66], "odom_epoch_generation": 0},
        }
        handoff = nav.maybe_handoff_corridor_axis(legacy, _MatureEvidence(evidence), _EpochCache(0))
        self.assertEqual(handoff["action"], "HANDOFF_TO_CORRIDOR_AXIS")
        self.assertEqual(
            handoff["lifecycle_transition"],
            ["LEGACY_BOOTSTRAP", "CORRIDOR_AXIS_MATURE", "CORRIDOR_BOUND"],
        )
        bound = handoff["anchor"]
        self.assertEqual(bound["corridor_axis_lifecycle"], "CORRIDOR_BOUND")
        self.assertAlmostEqual(bound["heading_rad"], 1.5605)
        bound["legacy_bootstrap_heading_rad"] = 1.2  # Later N5 output has no handoff path.
        kept = nav.maybe_handoff_corridor_axis(bound, _MatureEvidence(evidence), _EpochCache(0))
        self.assertEqual(kept["action"], "KEEP_BOUND")
        self.assertAlmostEqual(bound["heading_rad"], 1.5605)

    def test_handoff_preserves_room_zone_progress_without_changing_control_geometry(self):
        legacy = {
            "x": 0.0, "y": 0.0, "yaw_rad": 0.0, "heading_rad": 0.0,
            "raw_heading_rad": 0.0, "legacy_bootstrap_heading_rad": 0.0,
            "room_zone_progress_offset_m": 0.0,
            "corridor_axis_lifecycle": "LEGACY_BOOTSTRAP", "corridor_axis_odom_epoch_generation": 0,
        }
        evidence = {
            "valid": True, "heading_odom_rad": 0.0,
            "odom_binding": {"source_pose_x_y_yaw": [7.705, 0.0, 0.0], "odom_epoch_generation": 0},
        }
        handoff = nav.maybe_handoff_corridor_axis(legacy, _MatureEvidence(evidence), _EpochCache(0))
        bound = handoff["anchor"]
        self.assertAlmostEqual(bound["room_zone_progress_offset_m"], 7.705, places=9)
        self.assertAlmostEqual(nav.anchor_metrics(bound, (7.721, 0.0, 0.0))["anchor_progress_m"], 0.016, places=9)
        self.assertAlmostEqual(
            nav.effective_room_zone_progress_m(bound, 0.016),
            7.721,
            places=9,
        )
        progress_event = handoff["room_zone_progress_handoff"]
        self.assertAlmostEqual(progress_event["pre_handoff_current_anchor_progress_m"], 7.705, places=9)
        self.assertAlmostEqual(progress_event["new_anchor_progress_m"], 0.0, places=9)

    def test_frozen_20260831_handoff_room_zone_replay_crosses_existing_threshold(self):
        # Frozen RUN 20260831_103206 values: 7.705 m just before handoff and
        # 0.016 m in the newly-bound control frame.  This verifies only the
        # repaired timing semantic, not the full online doorway trajectory.
        current_run_pre_handoff_m = 7.705
        current_run_post_handoff_current_m = 0.016
        bound_anchor = {"room_zone_progress_offset_m": current_run_pre_handoff_m}
        self.assertAlmostEqual(
            nav.effective_room_zone_progress_m(bound_anchor, current_run_post_handoff_current_m),
            7.721,
            places=9,
        )
        self.assertLess(7.85 - 7.721, 0.15)
        self.assertGreaterEqual(
            nav.effective_room_zone_progress_m(bound_anchor, 0.145),
            7.85,
        )

    def test_bootstrap_without_handoff_retains_historical_anchor_progress_semantics(self):
        bootstrap = {"room_zone_progress_offset_m": 0.0}
        self.assertAlmostEqual(nav.effective_room_zone_progress_m(bootstrap, 7.981), 7.981)

    def test_bound_axis_requires_rebootstrap_after_odom_epoch_change(self):
        anchor = {
            "heading_rad": 1.56, "legacy_bootstrap_heading_rad": 1.63,
            "corridor_axis_lifecycle": "CORRIDOR_BOUND", "corridor_axis_odom_epoch_generation": 2,
        }
        evidence = _MatureEvidence({"valid": False})
        result = nav.maybe_handoff_corridor_axis(anchor, evidence, _EpochCache(3))
        self.assertEqual(result["action"], "REBOOTSTRAP_REQUIRED")
        self.assertEqual(result["reason"], "CORRIDOR_AXIS_ODOM_EPOCH_CHANGED")
        self.assertEqual(evidence.reset_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
