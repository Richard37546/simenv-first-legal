#!/usr/bin/env python3
"""Isolated rospy smoke test for strict content binding after Header.seq rewrite."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "local_subgoal_runner_mvp"))

from local_grid_contract import (  # noqa: E402
    GRID_CONTRACT_VERSION,
    GRID_STATUS_SCHEMA_VERSION,
    grid_content_hash,
    validate_grid_status_content_binding,
)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_grid(OccupancyGrid, rospy, stamp, sequence, first_cell=0):
    message = OccupancyGrid()
    message.header.seq = sequence
    message.header.stamp = stamp
    message.header.frame_id = "base"
    message.info.resolution = 0.05
    message.info.width = 2
    message.info.height = 2
    message.info.origin.orientation.w = 1.0
    message.data = [first_cell, 0, 0, 0]
    return message


def make_status(grid, generation):
    stamp = grid.header.stamp.to_sec()
    status = {
        "contract_version": GRID_CONTRACT_VERSION,
        "schema_version": GRID_STATUS_SCHEMA_VERSION,
        "producer_instance_id": "p2kg15-smoke",
        "content_generation_id": generation,
        "grid_content_stamp": stamp,
        "tf_valid": True,
        "all_required_inputs_fresh": True,
    }
    status["grid_content_hash"] = grid_content_hash(grid, status["producer_instance_id"], generation, stamp)
    return status


def wait_for_master(master_uri):
    import rosgraph

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            rosgraph.Master("p2kg15_grid_status_smoke").getPid()
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("isolated_ros_master_not_ready:" + master_uri)


def main():
    port = free_port()
    master_uri = "http://127.0.0.1:%d" % port
    os.environ["ROS_MASTER_URI"] = master_uri
    roscore = subprocess.Popen(["roscore", "-p", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_master(master_uri)
        import rospy
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import String

        rospy.init_node("p2kg15_grid_status_smoke", anonymous=True, disable_signals=True)
        received_grids, received_statuses = [], []
        grid_topic = "/p2kg15_grid_status_smoke/grid"
        status_topic = "/p2kg15_grid_status_smoke/status"
        rospy.Subscriber(grid_topic, OccupancyGrid, received_grids.append, queue_size=10)
        rospy.Subscriber(status_topic, String, received_statuses.append, queue_size=10)
        grid_publisher = rospy.Publisher(grid_topic, OccupancyGrid, queue_size=10)
        status_publisher = rospy.Publisher(status_topic, String, queue_size=10)
        deadline = time.monotonic() + 5.0
        while (grid_publisher.get_num_connections() < 1 or status_publisher.get_num_connections() < 1) and time.monotonic() < deadline:
            time.sleep(0.02)
        if grid_publisher.get_num_connections() < 1 or status_publisher.get_num_connections() < 1:
            raise RuntimeError("isolated_subscribers_not_connected")

        stamp = rospy.Time.now()
        source = make_grid(OccupancyGrid, rospy, stamp, sequence=123456)
        good_status = make_status(source, generation=41)
        status_publisher.publish(String(data=json.dumps(good_status)))
        grid_publisher.publish(source)

        modified = make_grid(OccupancyGrid, rospy, stamp, sequence=123457, first_cell=100)
        status_publisher.publish(String(data=json.dumps(good_status)))
        grid_publisher.publish(modified)

        changed_stamp = rospy.Time.from_sec(stamp.to_sec() + 1.0)
        stamped = make_grid(OccupancyGrid, rospy, changed_stamp, sequence=123458)
        status_publisher.publish(String(data=json.dumps(good_status)))
        grid_publisher.publish(stamped)

        deadline = time.monotonic() + 5.0
        while (len(received_grids) < 3 or len(received_statuses) < 3) and time.monotonic() < deadline:
            time.sleep(0.02)
        if len(received_grids) != 3 or len(received_statuses) != 3:
            raise RuntimeError("isolated_messages_missing")
        decoded = [json.loads(message.data) for message in received_statuses]
        good_errors = validate_grid_status_content_binding(received_grids[0], decoded[0])
        content_errors = validate_grid_status_content_binding(received_grids[1], decoded[1])
        stamp_errors = validate_grid_status_content_binding(received_grids[2], decoded[2])
        if good_errors:
            raise RuntimeError("good_binding_rejected:" + ",".join(good_errors))
        if "status_grid_content_hash_mismatch" not in content_errors:
            raise RuntimeError("content_change_not_rejected:" + ",".join(content_errors))
        if "status_grid_content_stamp_mismatch" not in stamp_errors:
            raise RuntimeError("stamp_change_not_rejected:" + ",".join(stamp_errors))
        if received_grids[0].header.seq == 41:
            raise RuntimeError("transport_sequence_unexpectedly_matches_application_generation")
        print("P2KG15_GRID_STATUS_ROS_SMOKE_PASS")
        return 0
    finally:
        try:
            import rospy
            rospy.signal_shutdown("p2kg15_grid_status_smoke_complete")
        except Exception:
            pass
        roscore.terminate()
        try:
            roscore.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            roscore.kill()
            roscore.wait(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
