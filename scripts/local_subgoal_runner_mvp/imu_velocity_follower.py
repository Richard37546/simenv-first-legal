#!/usr/bin/env python3
"""ROS1 IMU-based velocity follower.

This node is intentionally a thin middle layer:

    planner / runner  ->  /cmd_vel_raw  ->  imu_velocity_follower  ->  /cmd_vel
                                    ^             ^
                                    |             |
                                desired       /trunk_imu
                                velocity

It does not know the target point. The planner owns target selection and path
planning. This node only tries to execute the requested velocity more stably by
using the body IMU yaw, yaw rate, roll, and pitch.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Optional

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import String


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quat_to_rpy(q) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


@dataclass
class ImuState:
    stamp: rospy.Time
    roll: float
    pitch: float
    yaw: float
    yaw_rate_z: float


class ImuVelocityFollower:
    def __init__(self) -> None:
        self.raw_cmd_topic = rospy.get_param("~raw_cmd_topic", "/cmd_vel_raw")
        self.imu_topic = rospy.get_param("~imu_topic", "/trunk_imu")
        self.output_cmd_topic = rospy.get_param("~output_cmd_topic", "/cmd_vel")
        self.status_topic = rospy.get_param("~status_topic", "/imu_velocity_follower/status")

        self.input_mode = rospy.get_param("~input_mode", "body_twist")
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.cmd_timeout_sec = float(rospy.get_param("~cmd_timeout_sec", 0.5))
        self.imu_timeout_sec = float(rospy.get_param("~imu_timeout_sec", 0.5))

        self.max_linear_x = float(rospy.get_param("~max_linear_x", 0.35))
        self.max_angular_z = float(rospy.get_param("~max_angular_z", 0.45))
        self.min_speed_for_lock = float(rospy.get_param("~min_speed_for_lock", 0.05))
        self.straight_angular_deadband = float(rospy.get_param("~straight_angular_deadband", 0.08))

        self.kp_yaw = float(rospy.get_param("~kp_yaw", 1.15))
        self.kd_yaw_rate = float(rospy.get_param("~kd_yaw_rate", 0.18))
        self.ki_yaw = float(rospy.get_param("~ki_yaw", 0.0))
        self.max_integral_correction = float(rospy.get_param("~max_integral_correction", 0.08))
        self.feedforward_angular_scale = float(rospy.get_param("~feedforward_angular_scale", 1.0))
        self.heading_bias_rad = float(rospy.get_param("~heading_bias_rad", 0.0))

        self.large_error_slowdown_rad = float(rospy.get_param("~large_error_slowdown_rad", 0.35))
        self.max_roll_deg = float(rospy.get_param("~max_roll_deg", 18.0))
        self.max_pitch_deg = float(rospy.get_param("~max_pitch_deg", 18.0))
        self.publish_when_stopped = bool(rospy.get_param("~publish_when_stopped", True))

        self.latest_cmd: Optional[Twist] = None
        self.latest_cmd_stamp: Optional[rospy.Time] = None
        self.imu: Optional[ImuState] = None
        self.yaw_reference: Optional[float] = None
        self.yaw_error_integral = 0.0
        self.last_control_wall_time: Optional[float] = None

        self.cmd_pub = rospy.Publisher(self.output_cmd_topic, Twist, queue_size=10)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10)
        rospy.Subscriber(self.raw_cmd_topic, Twist, self.on_cmd, queue_size=10)
        rospy.Subscriber(self.imu_topic, Imu, self.on_imu, queue_size=100)

        rospy.loginfo(
            "imu_velocity_follower ready: %s + %s -> %s, mode=%s",
            self.raw_cmd_topic,
            self.imu_topic,
            self.output_cmd_topic,
            self.input_mode,
        )

    def on_cmd(self, msg: Twist) -> None:
        self.latest_cmd = msg
        self.latest_cmd_stamp = rospy.Time.now()

    def on_imu(self, msg: Imu) -> None:
        roll, pitch, yaw = quat_to_rpy(msg.orientation)
        self.imu = ImuState(
            stamp=rospy.Time.now(),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            yaw_rate_z=float(msg.angular_velocity.z),
        )

    def raw_command_fresh(self, now: rospy.Time) -> bool:
        if self.latest_cmd is None or self.latest_cmd_stamp is None:
            return False
        return (now - self.latest_cmd_stamp).to_sec() <= self.cmd_timeout_sec

    def imu_fresh(self, now: rospy.Time) -> bool:
        if self.imu is None:
            return False
        return (now - self.imu.stamp).to_sec() <= self.imu_timeout_sec

    def zero(self, reason: str) -> tuple[Twist, dict]:
        self.yaw_reference = None
        self.reset_integral()
        return Twist(), {"mode": "STOP", "reason": reason, "stop": True}

    def reset_integral(self) -> None:
        self.yaw_error_integral = 0.0
        self.last_control_wall_time = None

    def integral_correction(self, yaw_error: float) -> float:
        if self.ki_yaw <= 0.0:
            self.reset_integral()
            return 0.0

        now = time.monotonic()
        if self.last_control_wall_time is None:
            dt = 0.0
        else:
            dt = clamp(now - self.last_control_wall_time, 0.0, 0.2)
        self.last_control_wall_time = now

        self.yaw_error_integral += yaw_error * dt
        max_integral = self.max_integral_correction / max(self.ki_yaw, 1e-6)
        self.yaw_error_integral = clamp(self.yaw_error_integral, -max_integral, max_integral)
        return self.ki_yaw * self.yaw_error_integral

    def compute_body_twist(self, raw: Twist, imu: ImuState) -> tuple[Twist, dict]:
        desired_vx = clamp(float(raw.linear.x), -self.max_linear_x, self.max_linear_x)
        desired_wz = clamp(float(raw.angular.z), -self.max_angular_z, self.max_angular_z)

        moving_forward = desired_vx > self.min_speed_for_lock
        requesting_straight = abs(desired_wz) <= self.straight_angular_deadband

        if moving_forward and requesting_straight:
            if self.yaw_reference is None:
                self.yaw_reference = imu.yaw
            yaw_error = normalize_angle(self.yaw_reference - imu.yaw)
            integral = self.integral_correction(yaw_error)
            correction = self.kp_yaw * yaw_error + integral - self.kd_yaw_rate * imu.yaw_rate_z
            mode = "IMU_STRAIGHT_LOCK"
        else:
            # During intentional turns, follow the requested angular command and
            # refresh the straight-line reference after the turn ends.
            self.yaw_reference = None
            self.reset_integral()
            yaw_error = 0.0
            integral = 0.0
            correction = 0.0
            mode = "FOLLOW_RAW_TURN" if abs(desired_wz) > self.straight_angular_deadband else "IDLE_OR_REVERSE"

        cmd = Twist()
        cmd.linear.x = desired_vx
        cmd.angular.z = clamp(
            self.feedforward_angular_scale * desired_wz + correction,
            -self.max_angular_z,
            self.max_angular_z,
        )

        if abs(yaw_error) > self.large_error_slowdown_rad:
            cmd.linear.x *= 0.35
            mode = "SLOW_REALIGN"

        return cmd, {
            "mode": mode,
            "desired_vx": desired_vx,
            "desired_wz": desired_wz,
            "yaw_reference": self.yaw_reference,
            "yaw_error_rad": yaw_error,
            "imu_yaw_rad": imu.yaw,
            "imu_yaw_rate_z": imu.yaw_rate_z,
            "integral_wz": integral,
            "correction_wz": correction,
            "stop": False,
        }

    def compute_world_vector(self, raw: Twist, imu: ImuState) -> tuple[Twist, dict]:
        vx = float(raw.linear.x)
        vy = float(raw.linear.y)
        desired_speed = clamp(math.hypot(vx, vy), 0.0, self.max_linear_x)
        if desired_speed < self.min_speed_for_lock:
            self.yaw_reference = None
            return Twist(), {
                "mode": "STOP",
                "reason": "world_vector_speed_too_small",
                "stop": True,
                "desired_speed": desired_speed,
            }

        desired_heading = normalize_angle(math.atan2(vy, vx) + self.heading_bias_rad)
        yaw_error = normalize_angle(desired_heading - imu.yaw)
        integral = self.integral_correction(yaw_error)
        correction = self.kp_yaw * yaw_error + integral - self.kd_yaw_rate * imu.yaw_rate_z

        cmd = Twist()
        cmd.linear.x = desired_speed * max(0.0, math.cos(yaw_error))
        cmd.angular.z = clamp(correction, -self.max_angular_z, self.max_angular_z)
        if abs(yaw_error) > self.large_error_slowdown_rad:
            cmd.linear.x *= 0.35

        return cmd, {
            "mode": "WORLD_VECTOR_TRACK",
            "desired_speed": desired_speed,
            "desired_heading_rad": desired_heading,
            "yaw_error_rad": yaw_error,
            "imu_yaw_rad": imu.yaw,
            "imu_yaw_rate_z": imu.yaw_rate_z,
            "heading_bias_rad": self.heading_bias_rad,
            "integral_wz": integral,
            "correction_wz": correction,
            "stop": False,
        }

    def safety_ok(self, imu: ImuState) -> tuple[bool, str]:
        roll_deg = abs(math.degrees(imu.roll))
        pitch_deg = abs(math.degrees(imu.pitch))
        if roll_deg > self.max_roll_deg:
            return False, f"roll_too_large:{roll_deg:.2f}deg"
        if pitch_deg > self.max_pitch_deg:
            return False, f"pitch_too_large:{pitch_deg:.2f}deg"
        return True, "ok"

    def spin(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            cmd = Twist()
            status: dict

            if not self.raw_command_fresh(now):
                cmd, status = self.zero("raw_cmd_stale_or_missing")
            elif not self.imu_fresh(now):
                cmd, status = self.zero("imu_stale_or_missing")
            else:
                assert self.latest_cmd is not None
                assert self.imu is not None
                safe, reason = self.safety_ok(self.imu)
                if not safe:
                    cmd, status = self.zero(reason)
                elif self.input_mode == "world_vector":
                    cmd, status = self.compute_world_vector(self.latest_cmd, self.imu)
                else:
                    cmd, status = self.compute_body_twist(self.latest_cmd, self.imu)

                status.update(
                    {
                        "roll_deg": math.degrees(self.imu.roll),
                        "pitch_deg": math.degrees(self.imu.pitch),
                    }
                )

            if self.publish_when_stopped or abs(cmd.linear.x) > 1e-6 or abs(cmd.angular.z) > 1e-6:
                self.cmd_pub.publish(cmd)

            status.update(
                {
                    "linear_x": cmd.linear.x,
                    "angular_z": cmd.angular.z,
                    "input_mode": self.input_mode,
                    "raw_cmd_topic": self.raw_cmd_topic,
                    "imu_topic": self.imu_topic,
                    "output_cmd_topic": self.output_cmd_topic,
                }
            )
            msg = String()
            msg.data = json.dumps(status, sort_keys=True)
            self.status_pub.publish(msg)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break


def main() -> None:
    rospy.init_node("imu_velocity_follower")
    ImuVelocityFollower().spin()


if __name__ == "__main__":
    main()
