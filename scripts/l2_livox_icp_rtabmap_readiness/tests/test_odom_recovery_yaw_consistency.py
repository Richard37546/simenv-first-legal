#!/usr/bin/env python3
import importlib.util
import math
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "l2_livox_odom_gate.py"
SPEC = importlib.util.spec_from_file_location("l2_livox_odom_gate", str(MODULE_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeDuration:
    def __init__(self, sec): self.sec = float(sec)
    def to_sec(self): return self.sec


class FakeStamp:
    def __init__(self, sec): self.sec = float(sec)
    def __sub__(self, other): return FakeDuration(self.sec - other.sec)


def fake_odom(stamp_sec, x, yaw_deg=0.0):
    position = type("Position", (), {"x": float(x), "y": 0.0, "z": 0.0})()
    half_yaw = math.radians(float(yaw_deg)) / 2.0
    orientation = type("Orientation", (), {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(half_yaw),
        "w": math.cos(half_yaw),
    })()
    pose = type("Pose", (), {"position": position, "orientation": orientation})()
    return type("Odom", (), {
        "header": type("Header", (), {"stamp": FakeStamp(stamp_sec)})(),
        "pose": type("PoseWithCovariance", (), {"pose": pose})(),
    })()


def bare_gate(required_samples=3):
    gate = MODULE.LivoxOdomGate.__new__(MODULE.LivoxOdomGate)
    gate.max_delta_translation = 0.5
    gate.max_delta_yaw = math.radians(30.0)
    gate.max_input_gap_sec = 0.25
    gate.startup_stable_samples = 1
    gate.startup_stable_count = 0
    gate.startup_bootstrap_active = False
    gate.recovery_consistent_samples = required_samples
    gate.recovery = MODULE.RecoveryYawConsistency(required_samples, math.radians(2.5))
    gate.continuity = MODULE.ContinuityAuthority("test-run")
    gate.last_msg = None
    gate.audit_event = lambda *_args, **_kwargs: None
    gate.fresh_imu_yaw = lambda _stamp: (0.0, 0.0)
    accepted, rejected = [], []
    gate.accept = lambda msg, **extra: accepted.append((msg, extra))
    gate.reject = lambda reason, **extra: rejected.append((reason, extra))
    return gate, accepted, rejected


class RecoveryYawConsistencyTest(unittest.TestCase):
    def test_releases_after_ten_consistent_samples(self):
        guard = MODULE.RecoveryYawConsistency(10, math.radians(2.5))
        guard.arm()
        result = None
        for index in range(10):
            yaw = math.radians(index * 0.4)
            result = guard.observe(yaw, yaw + math.radians(12.0))
        self.assertTrue(result["release"])
        self.assertFalse(guard.active)

    def test_run0140_like_divergence_stays_quarantined(self):
        guard = MODULE.RecoveryYawConsistency(10, math.radians(2.5))
        guard.arm()
        result = None
        for index in range(12):
            odom_yaw = math.radians(-0.45 * index)
            imu_yaw = math.radians(0.05 * index)
            result = guard.observe(odom_yaw, imu_yaw)
        self.assertFalse(result["release"])
        self.assertTrue(guard.active)
        self.assertGreater(result["yaw_error_rad"], math.radians(2.5))

    def test_mismatch_rebases_a_new_consecutive_window(self):
        guard = MODULE.RecoveryYawConsistency(10, math.radians(2.5))
        guard.arm()
        guard.observe(math.radians(0.0), math.radians(0.0))

        mismatch = guard.observe(math.radians(4.0), math.radians(0.0))
        self.assertFalse(mismatch["release"])
        self.assertEqual(mismatch["consistent_samples"], 1)

        result = mismatch
        for index in range(1, 10):
            result = guard.observe(
                math.radians(4.0 + index * 0.2),
                math.radians(index * 0.2),
            )
        self.assertTrue(result["release"])
        self.assertFalse(guard.active)

    def test_new_discontinuity_rearms_an_active_recovery(self):
        guard = MODULE.RecoveryYawConsistency(3, math.radians(2.5))
        guard.arm()
        guard.observe(math.radians(0.0), math.radians(0.0))
        guard.observe(math.radians(0.2), math.radians(0.2))
        self.assertEqual(guard.consistent_samples, 2)

        gate = MODULE.LivoxOdomGate.__new__(MODULE.LivoxOdomGate)
        gate.recovery = guard
        gate.continuity = MODULE.ContinuityAuthority("test-run")
        gate.arm_recovery()

        self.assertTrue(guard.active)
        self.assertIsNone(guard.start_odom_yaw)
        self.assertIsNone(guard.start_imu_yaw)
        self.assertEqual(guard.consistent_samples, 0)

    def test_angle_wrap_does_not_create_false_failure(self):
        guard = MODULE.RecoveryYawConsistency(3, math.radians(2.5))
        guard.arm()
        samples = [
            (math.radians(179.0), math.radians(169.0)),
            (math.radians(-179.0), math.radians(171.0)),
            (math.radians(-177.0), math.radians(173.0)),
        ]
        result = None
        for odom_yaw, imu_yaw in samples:
            result = guard.observe(odom_yaw, imu_yaw)
        self.assertTrue(result["release"])


class ContinuityAuthorityTests(unittest.TestCase):
    def test_startup_reinitialization_is_quarantined_before_first_publish(self):
        gate, accepted, rejected = bare_gate(required_samples=3)
        gate.startup_stable_samples = 4
        gate.startup_bootstrap_active = True

        # Three provisional poses are followed by the RTAB-Map IMU-prior
        # startup yaw reset observed in the controlled run.  No provisional
        # pose may be published or turn into a false old-epoch discontinuity.
        gate.callback(fake_odom(1.0, 0.0, 0.0))
        gate.callback(fake_odom(1.1, 0.01, 0.0))
        gate.callback(fake_odom(1.2, 0.02, 0.0))
        gate.callback(fake_odom(1.3, 0.03, 90.0))
        self.assertEqual(accepted, [])
        self.assertTrue(gate.continuity.continuous)
        self.assertEqual(rejected[-1][0], "startup_stabilization_reset")

        for stamp, x in ((1.4, 0.04), (1.5, 0.05), (1.6, 0.06)):
            gate.callback(fake_odom(stamp, x, 90.0))
        self.assertEqual(len(accepted), 1)
        self.assertTrue(accepted[0][1]["startup_stabilized"])
        self.assertTrue(gate.continuity.continuous)

    def test_normal_continuous_authority_is_unchanged(self):
        authority = MODULE.ContinuityAuthority("run-a")
        self.assertTrue(authority.continuous)
        self.assertEqual(authority.metadata()["localization_authority"], "CONTINUOUS")

    def test_translation_discontinuity_invalidates_old_authority(self):
        authority = MODULE.ContinuityAuthority("run-a")
        authority.invalidate("delta_translation_exceeded")
        metadata = authority.metadata()
        self.assertFalse(authority.continuous)
        self.assertEqual(metadata["continuity_state"], "REBASE_REQUIRED")
        self.assertEqual(metadata["continuity_invalid_reason"], "delta_translation_exceeded")

    def test_yaw_release_cannot_restore_old_authority(self):
        authority = MODULE.ContinuityAuthority("run-a")
        authority.invalidate("delta_translation_exceeded")
        guard = MODULE.RecoveryYawConsistency(3, math.radians(2.5))
        guard.arm()
        for degrees in (0.0, 0.2, 0.4):
            result = guard.observe(math.radians(degrees), math.radians(degrees))
        self.assertTrue(result["release"])
        self.assertFalse(authority.continuous)
        self.assertEqual(authority.metadata()["localization_authority"], "UNAVAILABLE")

    def test_new_run_does_not_inherit_invalid_authority(self):
        previous = MODULE.ContinuityAuthority("run-a")
        previous.invalidate("delta_translation_exceeded")
        fresh = MODULE.ContinuityAuthority("run-b")
        self.assertFalse(previous.continuous)
        self.assertTrue(fresh.continuous)
        self.assertEqual(fresh.event("GATE_READY")["event_sequence"], 1)

    def test_bad_run_yaw_recovery_never_republishes_old_epoch(self):
        gate, accepted, rejected = bare_gate(required_samples=3)
        # RUN155322 reconstruction: normal raw -> 10.297 zero pose ->
        # 10.597 input gap -> yaw-consistent raw measurements.
        gate.callback(fake_odom(10.197, 1.925))
        gate.callback(fake_odom(10.297, 0.0))  # delta_translation_exceeded
        gate.callback(fake_odom(10.597, 1.925))  # input-gap quarantine
        for stamp, x in ((10.697, 1.968), (10.797, 1.985), (10.897, 2.000)):
            gate.callback(fake_odom(stamp, x))
        self.assertEqual(len(accepted), 1, "post-discontinuity raw Odom was republished")
        self.assertFalse(gate.continuity.continuous)
        self.assertEqual(rejected[0][0], "delta_translation_exceeded")
        self.assertEqual(rejected[-1][0], "continuity_invalid_rebase_required")

    def test_normal_raw_sequence_remains_publishable(self):
        gate, accepted, rejected = bare_gate(required_samples=3)
        gate.callback(fake_odom(10.0, 0.0))
        gate.callback(fake_odom(10.1, 0.1))
        self.assertEqual(len(accepted), 2)
        self.assertEqual(rejected, [])
        self.assertTrue(gate.continuity.continuous)


if __name__ == "__main__":
    unittest.main()
