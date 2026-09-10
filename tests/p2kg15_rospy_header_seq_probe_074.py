#!/usr/bin/env python3
"""Isolated rospy transport probe for OccupancyGrid.header.seq."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audit_reports" / "p2kg15_rospy_header_seq_probe_074.json"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    port = free_port()
    master_uri = "http://127.0.0.1:%d" % port
    env = dict(os.environ, ROS_MASTER_URI=master_uri)
    roscore = subprocess.Popen(["roscore", "-p", str(port)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = {"master_uri": master_uri, "sent": [], "received": [], "subscriber_received_seq_rewritten": None, "error": None}
    try:
        deadline = time.monotonic() + 10.0
        import rosgraph
        while time.monotonic() < deadline:
            try:
                rosgraph.Master("p2kg15_header_seq_probe").getPid()
                break
            except Exception:
                time.sleep(0.05)
        else:
            raise RuntimeError("isolated_ros_master_not_ready")

        import rospy
        from nav_msgs.msg import OccupancyGrid

        rospy.init_node("p2kg15_header_seq_probe", anonymous=True, disable_signals=True)

        def received(message):
            result["received"].append({"header_seq": int(message.header.seq), "header_stamp": message.header.stamp.to_sec()})

        publisher = rospy.Publisher("/p2kg15_header_seq_probe/grid", OccupancyGrid, queue_size=10)
        rospy.Subscriber("/p2kg15_header_seq_probe/grid", OccupancyGrid, received, queue_size=10)
        deadline = time.monotonic() + 5.0
        while publisher.get_num_connections() < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        if publisher.get_num_connections() < 1:
            raise RuntimeError("isolated_subscriber_not_connected")

        for index in range(3):
            message = OccupancyGrid()
            message.header.seq = 123456 + index
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = "base"
            message.info.resolution = 1.0
            message.info.width = 1
            message.info.height = 1
            message.info.origin.orientation.w = 1.0
            message.data = [0]
            result["sent"].append({"header_seq": int(message.header.seq), "header_stamp": message.header.stamp.to_sec()})
            publisher.publish(message)
            time.sleep(0.15)

        deadline = time.monotonic() + 5.0
        while len(result["received"]) < len(result["sent"]) and time.monotonic() < deadline:
            time.sleep(0.02)
        if len(result["received"]) != len(result["sent"]):
            raise RuntimeError("isolated_messages_missing")
        result["subscriber_received_seq_rewritten"] = any(
            sent["header_seq"] != received["header_seq"]
            for sent, received in zip(result["sent"], result["received"])
        )
    except Exception as exc:
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        try:
            import rospy
            rospy.signal_shutdown("p2kg15_header_seq_probe_complete")
        except Exception:
            pass
        roscore.terminate()
        try:
            roscore.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            roscore.kill()
            roscore.wait(timeout=5.0)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["error"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
