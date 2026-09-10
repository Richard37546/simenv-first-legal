#!/usr/bin/env python3
"""One controlled R38 stopping-envelope probe; it never starts navigation."""

import argparse
import time

import rospy
from geometry_msgs.msg import Twist


def command(linear_x: float, angular_z: float) -> Twist:
    msg = Twist()
    msg.linear.x = linear_x
    msg.angular.z = angular_z
    return msg


def publish_for_sim_time(pub, msg: Twist, duration_sec: float, rate_hz: float, wall_deadline: float) -> None:
    started = rospy.Time.now().to_sec()
    while not rospy.is_shutdown() and rospy.Time.now().to_sec() - started < duration_sec:
        if time.monotonic() >= wall_deadline:
            raise RuntimeError("wall_watchdog_expired")
        pub.publish(msg)
        time.sleep(1.0 / rate_hz)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cmd_vel_raw")
    parser.add_argument("--linear-x", type=float, default=0.175)
    parser.add_argument("--angular-z", type=float, default=0.14)
    parser.add_argument("--motion-sim-sec", type=float, default=2.0)
    parser.add_argument("--zero-sim-sec", type=float, default=4.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--wall-watchdog-sec", type=float, default=180.0)
    args = parser.parse_args()

    rospy.init_node("p2kg15_r38_stopping_probe", anonymous=True)
    pub = rospy.Publisher(args.topic, Twist, queue_size=2)
    deadline = time.monotonic() + args.wall_watchdog_sec
    zero = command(0.0, 0.0)
    try:
        while rospy.Time.now().to_sec() <= 0.0:
            if time.monotonic() >= deadline:
                raise RuntimeError("sim_clock_unavailable")
            pub.publish(zero)
            time.sleep(0.05)
        rospy.loginfo("R38 stopping probe: moving")
        publish_for_sim_time(pub, command(args.linear_x, args.angular_z), args.motion_sim_sec, args.rate_hz, deadline)
        rospy.loginfo("R38 stopping probe: zero hold")
        publish_for_sim_time(pub, zero, args.zero_sim_sec, args.rate_hz, deadline)
        return 0
    finally:
        for _ in range(10):
            pub.publish(zero)
            time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
