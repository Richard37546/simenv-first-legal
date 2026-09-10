#!/usr/bin/env python3
import json
import math
import os
import statistics
import time
from collections import defaultdict

import rospy
from sensor_msgs.msg import PointCloud, PointCloud2
from sensor_msgs import point_cloud2


COMPLIANCE_FLAGS = {
    "used_odom_gazebo": False,
    "used_ground_truth": False,
    "used_gazebo_model_states": False,
    "used_gazebo_link_states": False,
    "used_generated_metadata": False,
    "used_livox_pointcloud2_as_input": False,
    "used_livox_lidar2_as_input": False,
    "published_cmd_vel": False,
    "published_odom_frame_cloud": False,
    "published_map_frame_cloud": False,
    "published_world_frame_cloud": False,
}


def _param_bool(name, default):
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _ratio(count, total):
    return float(count) / float(total) if total else 0.0


class L1SScanCleanCloudNode:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/scan")
        self.raw_output_topic = rospy.get_param(
            "~raw_output_topic", "/team/livox/scan_cloud_raw_pcl2"
        )
        self.filtered_output_topic = rospy.get_param(
            "~filtered_output_topic", "/team/livox/scan_cloud_filtered"
        )
        self.min_range = float(rospy.get_param("~min_range", 0.2))
        self.max_range = float(rospy.get_param("~max_range", 30.0))
        self.enable_voxel_filter = _param_bool("~enable_voxel_filter", True)
        self.voxel_leaf_size = float(rospy.get_param("~voxel_leaf_size", 0.08))
        self.enable_radius_outlier_filter = _param_bool(
            "~enable_radius_outlier_filter", False
        )
        self.radius_search = float(rospy.get_param("~radius_search", 0.4))
        self.min_neighbors = int(rospy.get_param("~min_neighbors", 2))
        self.stats_window_sec = float(rospy.get_param("~stats_window_sec", 5.0))
        self.save_root = rospy.get_param(
            "~save_root", "debug/l1s_livox_scan_clean_cloud"
        )
        self.require_frame_id = rospy.get_param("~require_frame_id", "laser_livox")
        self.publish_rate_limit = float(rospy.get_param("~publish_rate_limit", 0.0))

        self.raw_pub = rospy.Publisher(
            self.raw_output_topic, PointCloud2, queue_size=1
        )
        self.filtered_pub = rospy.Publisher(
            self.filtered_output_topic, PointCloud2, queue_size=1
        )
        self.sub = rospy.Subscriber(self.input_topic, PointCloud, self._callback, queue_size=1)

        self.last_publish_time = None
        self.window_started = time.time()
        self.publish_raw_pcl2_count = 0
        self.publish_filtered_count = 0
        self.latest_stats = self._empty_stats("waiting_for_scan")

        os.makedirs(self.save_root, exist_ok=True)
        self._write_json("summary.json", self.latest_stats)
        rospy.loginfo("L1S scan clean cloud node listening on %s", self.input_topic)

    def _empty_stats(self, failure_reason=None):
        stats = {
            "input_topic": self.input_topic,
            "input_type": "sensor_msgs/PointCloud",
            "input_frame_id": None,
            "raw_point_count": 0,
            "finite_point_count": 0,
            "filtered_point_count": 0,
            "finite_ratio": 0.0,
            "removed_nan_inf_count": 0,
            "removed_near_count": 0,
            "removed_far_count": 0,
            "removed_outlier_count": 0,
            "range_min": None,
            "range_mean": None,
            "range_median": None,
            "range_p95": None,
            "range_p99": None,
            "range_max": None,
            "range_gt_10m_ratio": 0.0,
            "range_gt_20m_ratio": 0.0,
            "range_gt_30m_ratio": 0.0,
            "range_gt_50m_ratio": 0.0,
            "range_gt_100m_ratio": 0.0,
            "near_lt_min_range_ratio": 0.0,
            "publish_raw_pcl2_count": self.publish_raw_pcl2_count,
            "publish_filtered_count": self.publish_filtered_count,
            "failure_reason": failure_reason,
            "warnings": [],
            "parameters": self._parameter_snapshot(),
        }
        stats.update(COMPLIANCE_FLAGS)
        return stats

    def _parameter_snapshot(self):
        return {
            "input_topic": self.input_topic,
            "raw_output_topic": self.raw_output_topic,
            "filtered_output_topic": self.filtered_output_topic,
            "min_range": self.min_range,
            "max_range": self.max_range,
            "enable_voxel_filter": self.enable_voxel_filter,
            "voxel_leaf_size": self.voxel_leaf_size,
            "enable_radius_outlier_filter": self.enable_radius_outlier_filter,
            "radius_search": self.radius_search,
            "min_neighbors": self.min_neighbors,
            "stats_window_sec": self.stats_window_sec,
            "save_root": self.save_root,
            "require_frame_id": self.require_frame_id,
            "publish_rate_limit": self.publish_rate_limit,
        }

    def _callback(self, msg):
        now = time.time()
        if self.publish_rate_limit > 0.0 and self.last_publish_time is not None:
            if now - self.last_publish_time < 1.0 / self.publish_rate_limit:
                return
        self.last_publish_time = now

        raw_points = [(p.x, p.y, p.z) for p in msg.points]
        raw_pcl2 = point_cloud2.create_cloud_xyz32(msg.header, raw_points)
        self.raw_pub.publish(raw_pcl2)
        self.publish_raw_pcl2_count += 1

        filtered_points, stats = self._filter_points(raw_points)
        filtered_pcl2 = point_cloud2.create_cloud_xyz32(msg.header, filtered_points)
        self.filtered_pub.publish(filtered_pcl2)
        self.publish_filtered_count += 1

        stats["input_frame_id"] = msg.header.frame_id
        stats["publish_raw_pcl2_count"] = self.publish_raw_pcl2_count
        stats["publish_filtered_count"] = self.publish_filtered_count
        stats["failure_reason"] = None
        stats["warnings"] = []
        if msg.header.frame_id != self.require_frame_id:
            stats["warnings"].append(
                "input frame_id is %s, expected %s"
                % (msg.header.frame_id, self.require_frame_id)
            )
        stats["parameters"] = self._parameter_snapshot()
        stats.update(COMPLIANCE_FLAGS)
        self.latest_stats = stats

        self._write_json("latest_stats.json", stats)
        if now - self.window_started >= self.stats_window_sec:
            self.window_started = now
            self._write_json("summary.json", stats)

    def _filter_points(self, points):
        raw_count = len(points)
        finite_points = []
        finite_ranges = []
        removed_nan_inf = 0
        removed_near = 0
        removed_far = 0

        for x, y, z in points:
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                removed_nan_inf += 1
                continue
            distance = math.sqrt(x * x + y * y + z * z)
            finite_points.append((x, y, z, distance))
            finite_ranges.append(distance)
            if distance < self.min_range:
                removed_near += 1
            elif distance > self.max_range:
                removed_far += 1

        range_filtered = [
            (x, y, z) for x, y, z, distance in finite_points
            if self.min_range <= distance <= self.max_range
        ]

        voxel_filtered = self._voxel_downsample(range_filtered)
        final_points, removed_outlier = self._radius_outlier_filter(voxel_filtered)

        stats = self._empty_stats()
        stats.update({
            "raw_point_count": raw_count,
            "finite_point_count": len(finite_points),
            "filtered_point_count": len(final_points),
            "finite_ratio": _ratio(len(finite_points), raw_count),
            "removed_nan_inf_count": removed_nan_inf,
            "removed_near_count": removed_near,
            "removed_far_count": removed_far,
            "removed_outlier_count": removed_outlier,
            "range_min": min(finite_ranges) if finite_ranges else None,
            "range_mean": statistics.mean(finite_ranges) if finite_ranges else None,
            "range_median": statistics.median(finite_ranges) if finite_ranges else None,
            "range_p95": _percentile(finite_ranges, 95),
            "range_p99": _percentile(finite_ranges, 99),
            "range_max": max(finite_ranges) if finite_ranges else None,
            "range_gt_10m_ratio": _ratio(sum(1 for r in finite_ranges if r > 10.0), len(finite_ranges)),
            "range_gt_20m_ratio": _ratio(sum(1 for r in finite_ranges if r > 20.0), len(finite_ranges)),
            "range_gt_30m_ratio": _ratio(sum(1 for r in finite_ranges if r > 30.0), len(finite_ranges)),
            "range_gt_50m_ratio": _ratio(sum(1 for r in finite_ranges if r > 50.0), len(finite_ranges)),
            "range_gt_100m_ratio": _ratio(sum(1 for r in finite_ranges if r > 100.0), len(finite_ranges)),
            "near_lt_min_range_ratio": _ratio(removed_near, len(finite_ranges)),
        })
        return final_points, stats

    def _voxel_downsample(self, points):
        if not self.enable_voxel_filter or self.voxel_leaf_size <= 0.0:
            return points
        buckets = {}
        leaf = self.voxel_leaf_size
        for x, y, z in points:
            key = (
                int(math.floor(x / leaf)),
                int(math.floor(y / leaf)),
                int(math.floor(z / leaf)),
            )
            if key not in buckets:
                buckets[key] = [0.0, 0.0, 0.0, 0]
            bucket = buckets[key]
            bucket[0] += x
            bucket[1] += y
            bucket[2] += z
            bucket[3] += 1
        return [
            (sx / count, sy / count, sz / count)
            for sx, sy, sz, count in buckets.values()
        ]

    def _radius_outlier_filter(self, points):
        if not self.enable_radius_outlier_filter or len(points) < 2:
            return points, 0

        cell_size = max(self.radius_search, 1e-6)
        radius_sq = self.radius_search * self.radius_search
        grid = defaultdict(list)
        for idx, (x, y, z) in enumerate(points):
            key = (
                int(math.floor(x / cell_size)),
                int(math.floor(y / cell_size)),
                int(math.floor(z / cell_size)),
            )
            grid[key].append(idx)

        kept = []
        removed = 0
        for idx, (x, y, z) in enumerate(points):
            cx = int(math.floor(x / cell_size))
            cy = int(math.floor(y / cell_size))
            cz = int(math.floor(z / cell_size))
            neighbors = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for other_idx in grid.get((cx + dx, cy + dy, cz + dz), []):
                            if other_idx == idx:
                                continue
                            ox, oy, oz = points[other_idx]
                            if (x - ox) ** 2 + (y - oy) ** 2 + (z - oz) ** 2 <= radius_sq:
                                neighbors += 1
                                if neighbors >= self.min_neighbors:
                                    break
                        if neighbors >= self.min_neighbors:
                            break
                    if neighbors >= self.min_neighbors:
                        break
                if neighbors >= self.min_neighbors:
                    break
            if neighbors >= self.min_neighbors:
                kept.append((x, y, z))
            else:
                removed += 1
        return kept, removed

    def _write_json(self, filename, payload):
        path = os.path.join(self.save_root, filename)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)


def main():
    rospy.init_node("l1s_scan_clean_cloud_node")
    L1SScanCleanCloudNode()
    rospy.spin()


if __name__ == "__main__":
    main()
