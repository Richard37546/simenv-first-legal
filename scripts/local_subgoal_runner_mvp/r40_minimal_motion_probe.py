#!/usr/bin/env python3
"""One open-area R40 qualification probe for the minimal Stair motion set."""

import argparse
import time

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def twist(linear_x: float, angular_z: float) -> Twist:
    msg = Twist()
    msg.linear.x = linear_x
    msg.angular.z = angular_z
    return msg


def publish_segment(pub, phase_pub, phase: str, msg: Twist, duration: float, rate_hz: float, deadline: float) -> None:
    phase_pub.publish(String(data=phase))
    started = rospy.Time.now().to_sec()
    while not rospy.is_shutdown() and rospy.Time.now().to_sec() - started < duration:
        if time.monotonic() >= deadline:
            raise RuntimeError("wall_watchdog_expired")
        pub.publish(msg)
        time.sleep(1.0 / rate_hz)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cmd_vel_raw")
    parser.add_argument("--phase-topic", default="/p2kg15/r40_probe_phase")
    parser.add_argument("--motion-sim-sec", type=float, default=2.0)
    parser.add_argument("--zero-sim-sec", type=float, default=4.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--wall-watchdog-sec", type=float, default=900.0)
    args = parser.parse_args()

    rospy.init_node("p2kg15_r40_minimal_motion_probe", anonymous=True)
    pub = rospy.Publisher(args.topic, Twist, queue_size=2)
    phase_pub = rospy.Publisher(args.phase_topic, String, queue_size=10, latch=True)
    deadline = time.monotonic() + args.wall_watchdog_sec
    zero = twist(0.0, 0.0)
    primitives = (
        ("FORWARD_1", 0.30, 0.0),
        ("FORWARD_2", 0.30, 0.0),
        ("LEFT_2", 0.30, 0.14),
        ("RIGHT_1", 0.30, -0.14),
        ("RIGHT_2", 0.30, -0.14),
    )
    try:
        while rospy.Time.now().to_sec() <= 0.0:
            if time.monotonic() >= deadline:
                raise RuntimeError("sim_clock_unavailable")
            pub.publish(zero)
            time.sleep(0.05)
        publish_segment(pub, phase_pub, "PRE_ZERO", zero, args.zero_sim_sec, args.rate_hz, deadline)
        for name, linear_x, angular_z in primitives:
            rospy.loginfo("R40 minimal-motion probe: %s", name)
            publish_segment(pub, phase_pub, name, twist(linear_x, angular_z), args.motion_sim_sec, args.rate_hz, deadline)
            publish_segment(pub, phase_pub, name + "_ZERO", zero, args.zero_sim_sec, args.rate_hz, deadline)
        phase_pub.publish(String(data="COMPLETE"))
        return 0
    finally:
        phase_pub.publish(String(data="ABORT_OR_COMPLETE_ZERO"))
        for _ in range(10):
            pub.publish(zero)
            time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
