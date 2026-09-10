#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import rospy
import yaml
from nav_msgs.msg import OccupancyGrid
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


FREE = 0
UNKNOWN = 1
OCCUPIED = 2


class BevNode:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_occupancy_topic", "/map")
        self.output_topic = rospy.get_param("~output_occupancy_topic", "/bev/occupancy_grid")
        self.cloud_topic = rospy.get_param("~output_obstacle_cloud_topic", "/bev/obstacle_cloud")
        self.save_dir = Path(rospy.get_param("~save_dir", "/home/richard/.ros/results/bev_maps"))
        self.frame_id = rospy.get_param("~frame_id", "base")
        self.semantic_version = rospy.get_param("~semantic_version", "v3_fixed")
        self.grid_semantics_version = rospy.get_param("~grid_semantics_version", "v3_fixed")
        self.occupied_threshold = int(rospy.get_param("~occupied_threshold", 50))
        self.save_every_n = max(1, int(rospy.get_param("~save_every_n", 1)))
        self.frame_count = 0

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.grid_pub = rospy.Publisher(self.output_topic, OccupancyGrid, queue_size=1, latch=True)
        self.cloud_pub = rospy.Publisher(self.cloud_topic, PointCloud2, queue_size=1, latch=True)
        rospy.Subscriber(self.input_topic, OccupancyGrid, self.callback, queue_size=1)
        rospy.loginfo("bev_perception listening on %s, writing %s", self.input_topic, self.save_dir)

    def accgrid_from_occupancy(self, msg):
        h = int(msg.info.height)
        w = int(msg.info.width)
        raw = np.array(msg.data, dtype=np.int16).reshape((h, w))
        acc = np.full((h, w), UNKNOWN, dtype=np.uint8)
        acc[raw == 0] = FREE
        acc[raw >= self.occupied_threshold] = OCCUPIED
        return acc

    def publish_grid(self, msg, acc):
        out = OccupancyGrid()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.info = msg.info
        encoded = np.full(acc.shape, -1, dtype=np.int8)
        encoded[acc == FREE] = 0
        encoded[acc == OCCUPIED] = 100
        out.data = encoded.reshape(-1).astype(int).tolist()
        self.grid_pub.publish(out)

    def publish_cloud(self, msg, acc):
        resolution = float(msg.info.resolution)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)
        points = []
        rows, cols = np.where(acc == OCCUPIED)
        for row, col in zip(rows.tolist(), cols.tolist()):
            x = ox + (col + 0.5) * resolution
            y = oy + (row + 0.5) * resolution
            points.append((x, y, 0.0))
        header = msg.header
        header.frame_id = self.frame_id
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, points))

    def save_outputs(self, msg, acc):
        stamp = msg.header.stamp.to_sec()
        prefix = "frame%06d_%013d" % (self.frame_count, int(stamp * 1000.0))
        resolution = float(msg.info.resolution)
        origin_xy = [
            float(msg.info.origin.position.x),
            float(msg.info.origin.position.y),
        ]
        np.save(str(self.save_dir / ("%s_accgrid.npy" % prefix)), acc)
        with (self.save_dir / ("%s_accmap.yaml" % prefix)).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "stamp": stamp,
                    "frame_id": self.frame_id,
                    "grid_shape": [int(acc.shape[0]), int(acc.shape[1])],
                    "resolution": resolution,
                    "grid_origin_xy": origin_xy,
                    "accgrid": "%s_accgrid.npy" % prefix,
                    "grid_semantics": {"0": "free", "1": "unknown", "2": "occupied"},
                    "grid_semantics_version": self.grid_semantics_version,
                    "semantic_version": self.semantic_version,
                },
                handle,
                sort_keys=True,
            )
        with (self.save_dir / ("%s_meta.json" % prefix)).open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "stamp": stamp,
                    "frame_id": self.frame_id,
                    "grid_shape": [int(acc.shape[0]), int(acc.shape[1])],
                    "resolution": resolution,
                    "grid_origin_xy": origin_xy,
                    "grid_semantics": {"0": "free", "1": "unknown", "2": "occupied"},
                    "grid_semantics_version": self.grid_semantics_version,
                    "semantic_version": self.semantic_version,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    def callback(self, msg):
        self.frame_count += 1
        acc = self.accgrid_from_occupancy(msg)
        self.publish_grid(msg, acc)
        self.publish_cloud(msg, acc)
        if self.frame_count % self.save_every_n == 0:
            self.save_outputs(msg, acc)


def main():
    rospy.init_node("bev_perception")
    BevNode()
    rospy.spin()


if __name__ == "__main__":
    main()
