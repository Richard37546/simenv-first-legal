#!/usr/bin/env python3
"""Pure regression coverage for the pre-anchor CorridorAxisEvidence contract."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

from block_astar_dwa_mature_runner import build_arg_parser
from corridor_axis_evidence import (
    bind_axis_to_odom,
    estimate_bilateral_corridor_axis,
    make_mature_certificate,
    mature_bound_axis,
    mature_certificate_fresh,
    normalize_angle,
)


def wall_points(heading_base, length=3.5, half_width=1.0):
    axis = np.array([math.cos(heading_base), math.sin(heading_base)])
    normal = np.array([-axis[1], axis[0]])
    rows = []
    for along in np.arange(0.45, length, 0.04):
        for side in (-1.0, 1.0):
            point = axis * along + normal * (side * half_width)
            rows.append([point[0], point[1], 0.55])
    return np.asarray(rows, dtype=float)


def binding(pose, stamp=10.0):
    return {
        "binding_valid": True, "frame_id": "team_livox_odom",
        "odom_callback_sequence_t1": 17, "source_pose_x_y_yaw": list(pose),
        "odom_binding_method": "EXACT", "odom_t0_stamp": stamp, "odom_t1_stamp": stamp,
    }


class CorridorAxisEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.args = build_arg_parser().parse_args([])

    def bind(self, base_axis, pose, intent):
        axis = estimate_bilateral_corridor_axis(wall_points(base_axis), self.args)
        self.assertTrue(axis["valid"], axis)
        result = bind_axis_to_odom(axis, 10.0, binding(pose), intent)
        self.assertTrue(result["valid"], result)
        return result

    def test_current_193445_left_drift_uses_geometry_not_n5_bearing(self):
        # Archived run: N5 froze 1.6375707802 rad while the physical axis was pi/2.
        pose_yaw = 1.6375707802
        true_axis_odom = math.pi / 2.0
        result = self.bind(true_axis_odom - pose_yaw, (0.0, 0.0, pose_yaw), 1.6375707802)
        old_projected_drift = 13.0 * abs(math.sin(1.6375707802 - true_axis_odom))
        new_projected_drift = 13.0 * abs(math.sin(result["heading_odom_rad"] - true_axis_odom))
        self.assertGreater(old_projected_drift, 0.85)
        self.assertLess(new_projected_drift, 0.01)
        self.assertAlmostEqual(result["heading_odom_rad"], true_axis_odom, places=3)

    def test_93_degree_odom_yaw_is_not_snapped_to_zero(self):
        pose_yaw = math.radians(93.0)
        physical_axis_odom = math.pi / 2.0
        result = self.bind(physical_axis_odom - pose_yaw, (0.0, 0.0, pose_yaw), physical_axis_odom)
        self.assertAlmostEqual(result["heading_odom_rad"], physical_axis_odom, places=3)
        self.assertNotAlmostEqual(result["heading_odom_rad"], 0.0, places=3)

    def test_frame_rotation_equivalence(self):
        base_axis = -0.061
        physical_axis = 1.571
        for rotation in (0.0, math.pi / 2.0, -math.pi / 4.0):
            pose_yaw = physical_axis + rotation - base_axis
            result = self.bind(base_axis, (0.0, 0.0, pose_yaw), physical_axis + rotation)
            self.assertAlmostEqual(normalize_angle(result["heading_odom_rad"] - pose_yaw), base_axis, places=3)
            self.assertAlmostEqual(result["heading_odom_rad"], normalize_angle(physical_axis + rotation), places=3)

    def test_n5_offset_selects_sign_only_not_axis(self):
        pose_yaw = 1.4
        base_axis = -0.08
        expected = normalize_angle(pose_yaw + base_axis)
        left = self.bind(base_axis, (0.0, 0.0, pose_yaw), expected + 0.12)
        right = self.bind(base_axis, (0.0, 0.0, pose_yaw), expected - 0.12)
        self.assertAlmostEqual(left["heading_odom_rad"], expected, places=3)
        self.assertAlmostEqual(right["heading_odom_rad"], expected, places=3)
        self.assertEqual(left["forward_sign"], right["forward_sign"])

    def test_unilateral_or_unbound_evidence_fails_closed(self):
        points = wall_points(0.0)
        unilateral = points[points[:, 1] > 0.0]
        axis = estimate_bilateral_corridor_axis(unilateral, self.args)
        self.assertFalse(axis["valid"])
        self.assertEqual(axis["reason"], "BILATERAL_WALL_SUPPORT_UNAVAILABLE")
        self.assertFalse(bind_axis_to_odom(estimate_bilateral_corridor_axis(wall_points(0.0), self.args), 10.0, {"binding_valid": False, "reason": "ODOM_SOURCE_TIME_NOT_BRACKETED"}, 0.0)["valid"])

    def test_first_valid_sequence_does_not_mature(self):
        sample = self.bind(0.0, (0.0, 0.0, 0.0), 0.0)
        result = mature_bound_axis([sample], minimum_samples=8, max_heading_deviation_rad=0.10)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "CORRIDOR_AXIS_MATURITY_INSUFFICIENT_CONSECUTIVE_SAMPLES")

    def test_stable_consecutive_sequence_matures(self):
        samples = []
        for index in range(8):
            sample = self.bind(0.03 + (index - 3.5) * 0.01, (0.0, 0.0, 1.2), 1.25)
            sample["source_stamp"] = 10.0 + index * 0.1
            samples.append(sample)
        result = mature_bound_axis(samples, minimum_samples=8, max_heading_deviation_rad=0.10)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["maturity_consecutive_sample_count"], 8)
        self.assertLess(result["maturity_max_heading_deviation_rad"], 0.10)
        self.assertAlmostEqual(result["heading_odom_rad"], result["maturity_reference_heading_odom_rad"])
        self.assertAlmostEqual(result["maturity_frozen_heading_odom_rad"], result["maturity_reference_heading_odom_rad"])
        self.assertNotAlmostEqual(result["heading_odom_rad"], result["maturity_last_sample_heading_odom_rad"])

    def test_edge_last_sample_is_diagnostic_not_frozen_axis_authority(self):
        # The final observation remains inside the 0.10-rad maturity gate,
        # but the frozen long axis must stay the eight-sample circular mean.
        samples = [
            {"valid": True, "heading_odom_rad": 0.0, "source_stamp": 10.0 + index * 0.1}
            for index in range(7)
        ] + [{"valid": True, "heading_odom_rad": 0.09, "source_stamp": 10.7}]
        result = mature_bound_axis(samples, minimum_samples=8, max_heading_deviation_rad=0.10)
        self.assertTrue(result["valid"], result)
        self.assertAlmostEqual(result["heading_odom_rad"], result["maturity_reference_heading_odom_rad"])
        self.assertAlmostEqual(result["maturity_last_sample_heading_odom_rad"], 0.09)
        self.assertNotAlmostEqual(result["heading_odom_rad"], 0.09)

    def test_mature_certificate_survives_later_raw_queue_reset_until_next_poll(self):
        samples = [
            {
                "valid": True,
                "heading_odom_rad": 1.57 + (index - 3.5) * 0.01,
                "source_stamp": 18.397 - (7 - index) * 0.1,
                "odom_binding": {"odom_epoch_generation": 4},
            }
            for index in range(8)
        ]
        mature = mature_bound_axis(samples, minimum_samples=8, max_heading_deviation_rad=0.10)
        certificate = make_mature_certificate(mature, source_time_valid_for_sec=4.0)
        self.assertTrue(certificate["valid"], certificate)
        # The raw queue may now be empty after an invalid cloud, but the
        # certificate is a separate, one-consume lifecycle record.
        poll = mature_certificate_fresh(
            certificate,
            sim_now_sec=21.038,
            current_odom_epoch_generation=4,
        )
        self.assertTrue(poll["valid"], poll)
        self.assertAlmostEqual(poll["certificate_age_sec"], 21.038 - 18.397, places=6)
        self.assertAlmostEqual(
            poll["certificate_heading_odom_rad"],
            mature["maturity_reference_heading_odom_rad"],
        )

    def test_certificate_is_single_epoch_and_source_time_bounded(self):
        mature = {
            "valid": True,
            "source_stamp": 24.397,
            "maturity_frozen_heading_odom_rad": 1.5680,
            "odom_binding": {"odom_epoch_generation": 2},
        }
        certificate = make_mature_certificate(mature, source_time_valid_for_sec=4.0)
        self.assertEqual(
            mature_certificate_fresh(certificate, sim_now_sec=24.8, current_odom_epoch_generation=3)["reason"],
            "CORRIDOR_AXIS_CERTIFICATE_ODOM_EPOCH_CHANGED",
        )
        self.assertEqual(
            mature_certificate_fresh(certificate, sim_now_sec=28.5, current_odom_epoch_generation=2)["reason"],
            "CORRIDOR_AXIS_CERTIFICATE_STALE",
        )

    def test_eight_samples_with_a_pi_wrap_do_not_mature(self):
        # The baseline bag's early eight-sample run is geometrically valid but
        # flips direction.  Count alone must not promote it to a long axis.
        samples = []
        for index in range(8):
            heading = 0.04 if index < 4 else math.pi - 0.04
            samples.append({"valid": True, "heading_odom_rad": heading, "source_stamp": 10.0 + index * 0.1})
        result = mature_bound_axis(samples, minimum_samples=8, max_heading_deviation_rad=0.10)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "CORRIDOR_AXIS_MATURITY_HEADING_INCONSISTENT")

    def test_eight_samples_just_over_observed_baseline_spread_do_not_mature(self):
        # 0.108 rad is the archived baseline early-run spread.  The chosen
        # 0.10 rad bound must preserve its later, more stable handoff.
        samples = [
            {"valid": True, "heading_odom_rad": (-0.108 if index == 0 else 0.108 if index == 1 else 0.0)}
            for index in range(8)
        ]
        result = mature_bound_axis(samples, minimum_samples=8, max_heading_deviation_rad=0.10)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "CORRIDOR_AXIS_MATURITY_HEADING_INCONSISTENT")

    def test_production_seam_is_anchor_only_and_has_no_helper_ros_authority(self):
        helper_source = (ROOT / "scripts/local_subgoal_runner_mvp/corridor_axis_evidence.py").read_text(encoding="utf-8")
        self.assertNotIn("rospy", helper_source)
        self.assertNotIn("Publisher", helper_source)
        self.assertNotIn("cmd_vel", helper_source)
        nav_source = (ROOT / "scripts/local_subgoal_runner_mvp/navigation_state_machine.py").read_text(encoding="utf-8")
        start = nav_source.index("def build_initial_anchor(")
        end = nav_source.index("\ndef write_anchor_target(", start)
        seam = nav_source[start:end]
        self.assertIn("select_initial_anchor_heading(", seam)
        self.assertIn('"corridor_axis_lifecycle": "LEGACY_BOOTSTRAP"', seam)
        self.assertNotIn("wait_for_bound_axis", seam)
        handoff_start = nav_source.index("def maybe_handoff_corridor_axis(")
        handoff_end = nav_source.index("\ndef write_anchor_target(", handoff_start)
        handoff = nav_source[handoff_start:handoff_end]
        self.assertIn('"CORRIDOR_AXIS_MATURE"', handoff)
        self.assertNotIn("publish", handoff)
        self.assertNotIn("run_runner", handoff)


if __name__ == "__main__":
    unittest.main(verbosity=2)
