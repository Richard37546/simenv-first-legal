#!/usr/bin/env python3
import json
import math
import os
import copy
import sys
import time

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def finite_odom(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    vals = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    return all(math.isfinite(float(v)) for v in vals)


def finite_imu_orientation(msg):
    q = msg.orientation
    vals = [q.x, q.y, q.z, q.w]
    return all(math.isfinite(float(v)) for v in vals) and sum(float(v) ** 2 for v in vals) > 1e-6


class RecoveryYawConsistency:
    """Require post-reset odom yaw motion to agree with the production IMU."""

    def __init__(self, required_samples, max_yaw_error_rad):
        self.required_samples = max(1, int(required_samples))
        self.max_yaw_error_rad = float(max_yaw_error_rad)
        self.active = False
        self.start_odom_yaw = None
        self.start_imu_yaw = None
        self.consistent_samples = 0

    def arm(self):
        self.active = True
        self.start_odom_yaw = None
        self.start_imu_yaw = None
        self.consistent_samples = 0

    def observe(self, odom_yaw, imu_yaw):
        if not self.active:
            return {
                "release": True,
                "consistent_samples": self.required_samples,
                "yaw_error_rad": 0.0,
            }
        if self.start_odom_yaw is None:
            self.start_odom_yaw = float(odom_yaw)
            self.start_imu_yaw = float(imu_yaw)
            self.consistent_samples = 1
            yaw_error = 0.0
        else:
            odom_delta = angle_diff(float(odom_yaw), self.start_odom_yaw)
            imu_delta = angle_diff(float(imu_yaw), self.start_imu_yaw)
            yaw_error = abs(angle_diff(odom_delta, imu_delta))
            if yaw_error <= self.max_yaw_error_rad:
                self.consistent_samples += 1
            else:
                # A mismatch breaks the consecutive-sample window.  Rebase the
                # next window at this valid pair instead of comparing forever
                # against a stale pre-reset epoch.
                self.start_odom_yaw = float(odom_yaw)
                self.start_imu_yaw = float(imu_yaw)
                self.consistent_samples = 1
        release = self.consistent_samples >= self.required_samples
        if release:
            self.active = False
        return {
            "release": release,
            "consistent_samples": self.consistent_samples,
            "yaw_error_rad": yaw_error,
        }


def append_jsonl(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class ContinuityAuthority:
    """The only authority that may label gated Odom as continuous.

    A raw discontinuity proves that the old coordinate continuity is no longer
    established.  This gate has no producer-supplied rebase transform, so a
    later yaw-consistent measurement is useful audit evidence but cannot
    restore the old localization authority.
    """

    def __init__(self, run_epoch_id):
        self.run_epoch_id = str(run_epoch_id or "UNSCOPED_RUN")
        self.continuity_id = self.run_epoch_id + ":continuity-0001"
        self.event_sequence = 0
        self.state = "CONTINUOUS"
        self.invalid_reason = None

    @property
    def continuous(self):
        return self.state == "CONTINUOUS"

    def invalidate(self, reason):
        if self.continuous:
            self.state = "REBASE_REQUIRED"
            self.invalid_reason = str(reason)

    def metadata(self):
        return {
            "continuity_id": self.continuity_id,
            "continuity_state": self.state,
            "localization_authority": "CONTINUOUS" if self.continuous else "UNAVAILABLE",
            "continuity_invalid_reason": self.invalid_reason,
        }

    def event(self, event_type, **extra):
        self.event_sequence += 1
        return {
            "run_epoch_id": self.run_epoch_id,
            "event_sequence": self.event_sequence,
            "event_type": str(event_type),
            **self.metadata(),
            **extra,
        }


class LivoxOdomGate:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/team/livox/icp_odom_raw")
        self.output_topic = rospy.get_param("~output_topic", "/team/livox/icp_odom_gated")
        self.status_topic = rospy.get_param("~status_topic", "/team/livox/icp_odom_gate_status")
        self.output_frame_id = rospy.get_param("~output_frame_id", "team_livox_odom")
        self.output_child_frame_id = rospy.get_param("~output_child_frame_id", "base")
        self.max_delta_translation = float(rospy.get_param("~max_delta_translation", 0.5))
        self.max_delta_yaw = math.radians(float(rospy.get_param("~max_delta_yaw_deg", 30.0)))
        # RTAB-Map can publish a few provisional startup poses before its IMU
        # prior and scan registration settle.  They must never become a
        # consumer-visible odom epoch: establish the first epoch only after a
        # consecutive stable raw window.  Once this window has been published,
        # the normal fail-closed discontinuity handling below remains strict.
        self.startup_stable_samples = max(1, int(rospy.get_param("~startup_stable_samples", 10)))
        self.startup_stable_count = 0
        self.startup_bootstrap_active = True
        self.imu_topic = rospy.get_param("~imu_topic", "/trunk_imu")
        self.max_input_gap_sec = float(rospy.get_param("~max_input_gap_sec", 0.25))
        self.recovery_imu_max_age_sec = float(rospy.get_param("~recovery_imu_max_age_sec", 0.05))
        self.recovery_consistent_samples = int(rospy.get_param("~recovery_consistent_samples", 10))
        self.recovery_max_yaw_error = math.radians(
            float(rospy.get_param("~recovery_max_yaw_error_deg", 2.5))
        )
        self.events_path = rospy.get_param(
            "~events_path",
            "/home/richard/simenv_official_clean/debug/l2_livox_icp_rtabmap_readiness/readiness_events.jsonl",
        )
        self.audit_event_topic = rospy.get_param("~audit_event_topic", "/audit/startup_anchor/odom_epoch_event")
        self.continuity = ContinuityAuthority(rospy.get_param("~audit_run_epoch_id", "UNSCOPED_RUN"))
        self.last_msg = None
        self.latest_imu_yaw = None
        self.latest_imu_stamp = None
        self.recovery = RecoveryYawConsistency(
            self.recovery_consistent_samples,
            self.recovery_max_yaw_error,
        )
        self.valid_count = 0
        self.reject_count = 0
        self.reject_reasons = {}
        self.pub = rospy.Publisher(self.output_topic, Odometry, queue_size=10)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=20)
        self.audit_event_pub = rospy.Publisher(self.audit_event_topic, String, queue_size=20)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=100)
        rospy.Subscriber(self.input_topic, Odometry, self.callback, queue_size=30)
        self.audit_event("GATE_READY")

    def audit_event(self, event_type, **extra):
        """Best-effort audit output: no return value is used by gate logic."""
        try:
            payload = self.continuity.event(event_type, stamp=rospy.Time.now().to_sec(), **extra)
            self.audit_event_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        except Exception:
            pass

    def imu_callback(self, msg):
        if not finite_imu_orientation(msg):
            return
        self.latest_imu_yaw = yaw_from_quat(msg.orientation)
        self.latest_imu_stamp = msg.header.stamp

    def fresh_imu_yaw(self, odom_stamp):
        if self.latest_imu_yaw is None or self.latest_imu_stamp is None:
            return None, None
        age_sec = abs((odom_stamp - self.latest_imu_stamp).to_sec())
        if age_sec > self.recovery_imu_max_age_sec:
            return None, age_sec
        return self.latest_imu_yaw, age_sec

    def arm_recovery(self, reason="discontinuity"):
        # Every discontinuity starts a distinct recovery epoch, including one
        # observed while an earlier recovery is still quarantined.
        self.recovery.arm()
        # No current producer supplies T_oldO_newO or any other legal rebase
        # identity.  Do not fabricate one, and do not let yaw agreement turn a
        # post-discontinuity measurement back into old-frame localization.
        self.continuity.invalidate(reason)
        self.audit_event("RECOVERY_DISCONTINUITY", reason=str(reason))
        self.audit_event("CONTINUITY_INVALID", reason=str(reason))

    def status_payload(self, action, reason, **extra):
        payload = {
            "stamp": rospy.Time.now().to_sec(),
            "action": str(action),
            "reason": str(reason),
            "valid_count": self.valid_count,
            "reject_count": self.reject_count,
            **self.continuity.metadata(),
        }
        payload.update(extra)
        return payload

    def reject(self, reason, msg=None, **extra):
        self.reject_count += 1
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1
        payload = self.status_payload("reject", reason, **extra)
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        append_jsonl(self.events_path, payload)

    def accept(self, msg, **extra):
        self.valid_count += 1
        out = copy.deepcopy(msg)
        out.header.frame_id = self.output_frame_id
        out.child_frame_id = self.output_child_frame_id
        self.pub.publish(out)
        self.audit_event(
            "FIRST_ACCEPT" if self.valid_count == 1 else "GATED_ACCEPT",
            header_stamp_sec=msg.header.stamp.to_sec(),
            header_frame_id=out.header.frame_id,
            child_frame_id=out.child_frame_id,
            recovery_released=bool(extra.get("recovery_released", False)),
        )
        payload = self.status_payload("publish", "ok", **extra)
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        append_jsonl(self.events_path, payload)

    def callback(self, msg):
        if not finite_odom(msg):
            self.reject("nan_or_inf")
            return
        p = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        if self.last_msg is None:
            self.last_msg = msg
            if self.startup_bootstrap_active:
                self.startup_stable_count = 1
                if self.startup_stable_count < self.startup_stable_samples:
                    self.reject(
                        "startup_stabilizing",
                        startup_stable_samples=self.startup_stable_count,
                        startup_required_samples=self.startup_stable_samples,
                    )
                    return
                self.startup_bootstrap_active = False
                self.accept(
                    msg,
                    delta_translation=0.0,
                    delta_yaw_deg=0.0,
                    startup_stabilized=True,
                    startup_stable_samples=self.startup_stable_count,
                )
                return
            self.accept(msg, delta_translation=0.0, delta_yaw_deg=0.0)
            return
        last_p = self.last_msg.pose.pose.position
        last_yaw = yaw_from_quat(self.last_msg.pose.pose.orientation)
        dt = (msg.header.stamp - self.last_msg.header.stamp).to_sec()
        if dt < 0.0:
            self.reject("non_monotonic_stamp", dt=dt)
            return
        delta_translation = math.sqrt(
            (p.x - last_p.x) ** 2 + (p.y - last_p.y) ** 2 + (p.z - last_p.z) ** 2
        )
        delta_yaw = abs(angle_diff(yaw, last_yaw))
        if self.startup_bootstrap_active:
            self.last_msg = msg
            stable = (
                0.0 <= dt <= self.max_input_gap_sec
                and delta_translation <= self.max_delta_translation
                and delta_yaw <= self.max_delta_yaw
            )
            if not stable:
                self.startup_stable_count = 1
                self.reject(
                    "startup_stabilization_reset",
                    reset_reason=(
                        "non_monotonic_stamp" if dt < 0.0
                        else "input_gap" if dt > self.max_input_gap_sec
                        else "delta_translation_exceeded" if delta_translation > self.max_delta_translation
                        else "delta_yaw_exceeded"
                    ),
                    dt=dt,
                    delta_translation=delta_translation,
                    delta_yaw_deg=math.degrees(delta_yaw),
                    startup_stable_samples=self.startup_stable_count,
                    startup_required_samples=self.startup_stable_samples,
                )
                return
            self.startup_stable_count += 1
            if self.startup_stable_count < self.startup_stable_samples:
                self.reject(
                    "startup_stabilizing",
                    dt=dt,
                    delta_translation=delta_translation,
                    delta_yaw_deg=math.degrees(delta_yaw),
                    startup_stable_samples=self.startup_stable_count,
                    startup_required_samples=self.startup_stable_samples,
                )
                return
            self.startup_bootstrap_active = False
            self.accept(
                msg,
                delta_translation=delta_translation,
                delta_yaw_deg=math.degrees(delta_yaw),
                startup_stabilized=True,
                startup_stable_samples=self.startup_stable_count,
            )
            return
        if dt > self.max_input_gap_sec:
            self.arm_recovery("input_gap_recovery_quarantine")
            self.last_msg = msg
            self.reject("input_gap_recovery_quarantine", dt=dt)
            return
        if delta_translation > self.max_delta_translation:
            self.arm_recovery("delta_translation_exceeded")
            self.reject("delta_translation_exceeded", delta_translation=delta_translation)
            self.last_msg = msg
            return
        if delta_yaw > self.max_delta_yaw:
            self.arm_recovery("delta_yaw_exceeded")
            self.reject("delta_yaw_exceeded", delta_yaw_deg=math.degrees(delta_yaw))
            self.last_msg = msg
            return
        self.last_msg = msg

        if self.recovery.active:
            imu_yaw, imu_age_sec = self.fresh_imu_yaw(msg.header.stamp)
            if imu_yaw is None:
                self.reject("recovery_imu_unavailable", imu_age_sec=imu_age_sec)
                return
            recovery = self.recovery.observe(yaw, imu_yaw)
            recovery_extra = {
                "recovery_consistent_samples": recovery["consistent_samples"],
                "recovery_required_samples": self.recovery_consistent_samples,
                "recovery_yaw_error_deg": math.degrees(recovery["yaw_error_rad"]),
                "imu_age_sec": imu_age_sec,
            }
            if not recovery["release"]:
                self.reject("recovery_consistency_quarantine", **recovery_extra)
                return
            # Yaw agreement is a measurement check only.  It is not evidence
            # of continuous coordinates or of T_oldO_newO.
            self.audit_event("RECOVERY_YAW_EVIDENCE_COMPLETE", **recovery_extra)
            self.reject(
                "continuity_invalid_rebase_required",
                delta_translation=delta_translation,
                delta_yaw_deg=math.degrees(delta_yaw),
                recovery_released=True,
                legal_rebase_evidence=False,
                **recovery_extra,
            )
            return
        if not self.continuity.continuous:
            # Continue to observe raw measurements for audit, but never publish
            # a pose that falsely claims old-epoch continuity.
            self.reject(
                "continuity_invalid_rebase_required",
                delta_translation=delta_translation,
                delta_yaw_deg=math.degrees(delta_yaw),
                legal_rebase_evidence=False,
            )
            return
        self.accept(msg, delta_translation=delta_translation, delta_yaw_deg=math.degrees(delta_yaw))


def main():
    rospy.init_node("l2_livox_odom_gate")
    LivoxOdomGate()
    rospy.spin()


if __name__ == "__main__":
    main()
