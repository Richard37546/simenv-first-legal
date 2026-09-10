#!/usr/bin/env python3
"""Independent, offline-only Gazebo truth collector for V1.1 acceptance.

It has no publisher and no connection to supervisor decisions.  The output is
read only after the run has stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import rospy
from gazebo_msgs.msg import ModelStates
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu


def rpy(q):
    return (
        math.atan2(2 * (q.w*q.x + q.y*q.z), 1 - 2 * (q.x*q.x + q.y*q.y)),
        math.asin(max(-1.0, min(1.0, 2 * (q.w*q.y - q.z*q.x)))),
        math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z)),
    )


def upright_score(q):
    return 1.0 - 2.0 * (q.x*q.x + q.y*q.y)


class Collector:
    def __init__(self):
        self.sim_time = 0.0
        self.truth, self.imu = [], []
        self.subscribers = [
            rospy.Subscriber('/clock', Clock, self.clock_cb, queue_size=1000),
            rospy.Subscriber('/gazebo/model_states', ModelStates, self.truth_cb, queue_size=1000),
            rospy.Subscriber('/trunk_imu', Imu, self.imu_cb, queue_size=1000),
        ]

    def clock_cb(self, msg):
        self.sim_time = msg.clock.to_sec()

    def truth_cb(self, msg):
        if 'a1_gazebo' not in msg.name:
            return
        pose = msg.pose[msg.name.index('a1_gazebo')]
        roll, pitch, yaw = rpy(pose.orientation)
        self.truth.append({'sim_time': self.sim_time, 'position': [pose.position.x, pose.position.y, pose.position.z],
                           'quaternion_xyzw': [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                           'rpy': [roll, pitch, yaw], 'upright_score': upright_score(pose.orientation)})

    def imu_cb(self, msg):
        roll, pitch, yaw = rpy(msg.orientation)
        norm = math.sqrt(sum(value * value for value in (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)))
        self.imu.append({'sim_time': self.sim_time, 'frame_id': msg.header.frame_id,
                         'quaternion_xyzw': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                         'norm': norm, 'rpy': [roll, pitch, yaw], 'upright_score': upright_score(msg.orientation)})

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rospy.init_node('a1_safe_startup_offline_truth_collector', anonymous=True)
    collector = Collector()
    try:
        while not rospy.is_shutdown():
            time.sleep(.05)
    finally:
        output.write_text(json.dumps({'truth_used_for_control': False, 'truth': collector.truth, 'imu': collector.imu}, indent=2) + '\n', encoding='utf-8')
        collector.close()


if __name__ == '__main__':
    main()
