#!/usr/bin/env python3
"""Continuous relative-yaw agreement monitor for ICP odometry and IMU.

This is deliberately transport-agnostic: it neither imports ROS nor publishes
or rejects messages.  The production gate can consume it later only after its
offline and read-only-shadow contracts have been accepted.
"""

import math


def angle_diff(a, b):
    """Return the signed wrapped difference a - b in [-pi, pi]."""
    delta = float(a) - float(b)
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


class ContinuousYawConsistency:
    """Detect a persistent change in ICP-vs-IMU relative yaw.

    A fixed mounting-frame yaw offset is accepted at the first paired sample.
    Every later sample is compared by change-from-baseline, so a constant
    offset cannot produce an alarm.  Unlike recovery validation, this monitor
    does not rebase after a mismatch: a slowly accumulated drift must remain
    visible until an explicit reset starts a new odometry epoch.
    """

    def __init__(self, max_yaw_error_rad, required_consecutive_samples):
        self.max_yaw_error_rad = float(max_yaw_error_rad)
        self.required_consecutive_samples = max(1, int(required_consecutive_samples))
        self.reset()

    def reset(self):
        self.baseline_odom_yaw = None
        self.baseline_imu_yaw = None
        self.consecutive_mismatch_samples = 0
        self.max_yaw_error_rad_seen = 0.0
        self.triggered = False

    def observe(self, odom_yaw, imu_yaw):
        odom_yaw = float(odom_yaw)
        imu_yaw = float(imu_yaw)
        if self.baseline_odom_yaw is None:
            self.baseline_odom_yaw = odom_yaw
            self.baseline_imu_yaw = imu_yaw
            return {
                "baseline_initialized": True,
                "yaw_error_rad": 0.0,
                "consecutive_mismatch_samples": 0,
                "triggered": False,
            }

        odom_delta = angle_diff(odom_yaw, self.baseline_odom_yaw)
        imu_delta = angle_diff(imu_yaw, self.baseline_imu_yaw)
        yaw_error_rad = abs(angle_diff(odom_delta, imu_delta))
        self.max_yaw_error_rad_seen = max(self.max_yaw_error_rad_seen, yaw_error_rad)
        if yaw_error_rad > self.max_yaw_error_rad:
            self.consecutive_mismatch_samples += 1
        else:
            self.consecutive_mismatch_samples = 0
        if self.consecutive_mismatch_samples >= self.required_consecutive_samples:
            self.triggered = True
        return {
            "baseline_initialized": False,
            "yaw_error_rad": yaw_error_rad,
            "consecutive_mismatch_samples": self.consecutive_mismatch_samples,
            "triggered": self.triggered,
        }
