#!/usr/bin/env python3
"""One open-area R39 probe for the Stair command-to-motion contract."""

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
    start = rospy.Time.now().to_sec()
    while not rospy.is_shutdown() and rospy.Time.now().to_sec() - start < duration:
        if time.monotonic() >= deadline:
            raise RuntimeError("wall_watchdog_expired")
        pub.publish(msg)
        time.sleep(1.0 / rate_hz)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cmd_vel_raw")
    parser.add_argument("--phase-topic", default="/p2kg15/r39_probe_phase")
    parser.add_argument("--motion-sim-sec", type=float, default=4.0)
    parser.add_argument("--zero-sim-sec", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--wall-watchdog-sec", type=float, default=600.0)
    args = parser.parse_args()

    rospy.init_node("p2kg15_r39_execution_contract_probe", anonymous=True)
    pub = rospy.Publisher(args.topic, Twist, queue_size=2)
    phase_pub = rospy.Publisher(args.phase_topic, String, queue_size=10, latch=True)
    deadline = time.monotonic() + args.wall_watchdog_sec
    zero = twist(0.0, 0.0)
    phases = (("A_0175_014", 0.175, 0.14), ("B_0225_014", 0.225, 0.14), ("C_0300_014", 0.30, 0.14))

    try:
        while rospy.Time.now().to_sec() <= 0.0:
            if time.monotonic() >= deadline:
                raise RuntimeError("sim_clock_unavailable")
            pub.publish(zero)
            time.sleep(0.05)
        publish_segment(pub, phase_pub, "PRE_ZERO", zero, args.zero_sim_sec, args.rate_hz, deadline)
        for name, linear_x, angular_z in phases:
            rospy.loginfo("R39 execution probe: %s", name)
            publish_segment(pub, phase_pub, name, twist(linear_x, angular_z), args.motion_sim_sec, args.rate_hz, deadline)
            rospy.loginfo("R39 execution probe: %s_ZERO", name)
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
