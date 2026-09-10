#!/usr/bin/env python3
import argparse
import json
import os
import threading
import time

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String


GRID_TOPIC = "/team/local_traversability_grid"
EVIDENCE_TOPIC = "/team/local_traversability_evidence"
FRONT_TOPIC = "/team/front_auxiliary_evidence"
STATUS_TOPIC = "/team/traversability_status"
GRID_VALUES = {-1, 0, 100}


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_pgm(path, grid_msg):
    data = np.asarray(grid_msg.data, dtype=np.int16).reshape((grid_msg.info.height, grid_msg.info.width))
    image = np.full(data.shape, 205, dtype=np.uint8)
    image[data == 0] = 254
    image[data == 100] = 0
    image[data < 0] = 205
    with open(path, "wb") as handle:
        handle.write(("P5\n%d %d\n255\n" % (image.shape[1], image.shape[0])).encode("ascii"))
        handle.write(np.flipud(image).tobytes())


def parse_json_string(msg):
    try:
        return json.loads(msg.data)
    except Exception as exc:
        return {"parse_error": str(exc), "raw": msg.data}


def wait(topic, msg_type, timeout):
    return rospy.wait_for_message(topic, msg_type, timeout=timeout)


def collect_grid_samples(timeout, min_samples=3):
    samples = []
    lock = threading.Lock()

    def cb(msg):
        with lock:
            samples.append(msg)

    sub = rospy.Subscriber(GRID_TOPIC, OccupancyGrid, cb, queue_size=10)
    deadline = time.monotonic() + float(timeout)
    try:
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with lock:
                if len(samples) >= min_samples:
                    break
            time.sleep(0.05)
        with lock:
            return list(samples)
    finally:
        sub.unregister()


def monotonic_non_decreasing(values):
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def strictly_changes(values):
    return len(set(values)) > 1


def audit_freshness(samples):
    union_values = sorted({int(v) for sample in samples for v in sample.data})
    stamp_values = [float(sample.header.stamp.to_sec()) for sample in samples]
    seq_values = [int(sample.header.seq) for sample in samples]
    frame_ids = [sample.header.frame_id for sample in samples]
    data_lengths = [len(sample.data) for sample in samples]
    dimensions_valid = [
        int(sample.info.width) > 0
        and int(sample.info.height) > 0
        and float(sample.info.resolution) > 0.0
        and len(sample.data) == int(sample.info.width) * int(sample.info.height)
        for sample in samples
    ]
    stamp_monotonic = monotonic_non_decreasing(stamp_values) if len(stamp_values) >= 2 else False
    stamp_changed = strictly_changes(stamp_values)
    seq_monotonic = monotonic_non_decreasing(seq_values) if len(seq_values) >= 2 else False
    seq_changed = strictly_changes(seq_values)
    schema_pass = bool(samples) and set(union_values).issubset(GRID_VALUES) and all(dimensions_valid) and all(data_lengths)
    frame_pass = bool(samples) and all(frame == "base" for frame in frame_ids)
    now_ros = float(rospy.Time.now().to_sec())
    latest_grid_stamp_age_sec = max(0.0, now_ros - stamp_values[-1]) if stamp_values else None
    elapsed = stamp_values[-1] - stamp_values[0] if len(stamp_values) >= 2 else 0.0
    heartbeat_rate_observed_hz = (len(stamp_values) - 1) / elapsed if elapsed > 0.0 else None
    freshness_pass = bool(
        len(samples) >= 3
        and stamp_monotonic
        and stamp_changed
        and seq_monotonic
        and seq_changed
        and latest_grid_stamp_age_sec is not None
        and latest_grid_stamp_age_sec <= 2.0
    )
    return {
        "topic": GRID_TOPIC,
        "sampler_window_sec": None,
        "message_count_total": len(samples),
        "sample_count": len(samples),
        "valid_sample_count": len(samples) if schema_pass and frame_pass else 0,
        "required_sample_count": 3,
        "distinct_stamp_count": len(set(stamp_values)),
        "latest_grid_stamp_age_sec": latest_grid_stamp_age_sec,
        "heartbeat_rate_observed_hz": heartbeat_rate_observed_hz,
        "heartbeat_pass": bool(freshness_pass and heartbeat_rate_observed_hz is not None and heartbeat_rate_observed_hz >= 1.0),
        "frame_ids": frame_ids,
        "frame_id_pass": frame_pass,
        "seq_values": seq_values,
        "seq_monotonic": seq_monotonic,
        "seq_changed": seq_changed,
        "stamp_values_sec": stamp_values,
        "stamp_monotonic": stamp_monotonic,
        "stamp_changed": stamp_changed,
        "data_lengths": data_lengths,
        "dimensions_valid": dimensions_valid,
        "unique_values": union_values,
        "allowed_values": sorted(GRID_VALUES),
        "schema_pass": schema_pass and frame_pass,
        "grid_stamp_freshness_pass": freshness_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="debug/l3v_local_traversability_diagnostic_node")
    parser.add_argument("--report", default="audit_reports/l3v_local_traversability_diagnostic_node_report.md")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    rospy.init_node("l3v_diagnostic_readiness_check", anonymous=True)

    topics = {
        GRID_TOPIC: "nav_msgs/OccupancyGrid",
        EVIDENCE_TOPIC: "std_msgs/String",
        FRONT_TOPIC: "std_msgs/String",
        STATUS_TOPIC: "std_msgs/String",
    }
    result = {
        "topics_expected": topics,
        "topics_received": {},
        "stable_publish_check_pass": False,
        "local_traversability_grid_published": False,
        "front_auxiliary_evidence_published": False,
        "rgbd_front_auxiliary_effective": False,
        "l3u_main_conclusion_preserved": False,
        "connected_navigation_or_frontier": False,
        "published_cmd_vel": False,
        "safe_for_navigation": False,
        "safe_for_frontier": False,
        "autonomous_l4_allowed": False,
        "final_decision": "L3V_DIAGNOSTIC_NODE_BLOCKED_BY_TOPIC",
    }

    try:
        samples = collect_grid_samples(8.0, min_samples=3)
        if not samples:
            raise RuntimeError("no local traversability grid samples received")
        grid = samples[-1]
        evidence_msg = wait(EVIDENCE_TOPIC, String, min(args.timeout, 5.0))
        front_msg = wait(FRONT_TOPIC, String, min(args.timeout, 5.0))
        status_msg = wait(STATUS_TOPIC, String, min(args.timeout, 5.0))
    except Exception as exc:
        result["failure_reason"] = "missing_output_topic: %s" % exc
        write_json(os.path.join(args.output_dir, "l3v_replay_smoke_topics.json"), result)
        write_json(os.path.join(args.output_dir, "l3v_freshness_audit.json"), {"sampler_window_sec": 8.0, "sample_count": 0, "error": str(exc)})
        write_report(args.report, result, None, None, None, None)
        print(result["final_decision"])
        return

    freshness = audit_freshness(samples)
    freshness["sampler_window_sec"] = 8.0
    evidence = parse_json_string(evidence_msg)
    front = parse_json_string(front_msg)
    status = parse_json_string(status_msg)
    write_pgm(os.path.join(args.output_dir, "l3v_local_traversability_grid_snapshot.pgm"), grid)
    write_json(os.path.join(args.output_dir, "l3v_local_traversability_evidence_snapshot.json"), evidence)
    write_json(os.path.join(args.output_dir, "l3v_front_auxiliary_evidence_snapshot.json"), front)
    write_json(os.path.join(args.output_dir, "l3v_traversability_status_snapshot.json"), status)

    data = np.asarray(grid.data, dtype=np.int16)
    grid_nonempty = bool(np.any(data == 0) or np.any(data == 100))
    rgbd_useful = bool(front.get("rgbd_front_auxiliary_useful") or status.get("rgbd_front_auxiliary_useful"))
    l3u_preserved = bool(
        status.get("lidar_front_weak")
        and status.get("rgbd_front_auxiliary_useful")
        and status.get("odom_traversed_free_support_ratio", 0.0) > 0.0
    )
    safe_flags_clean = bool(
        status.get("safe_for_navigation") is False
        and status.get("safe_for_frontier") is False
        and status.get("autonomous_l4_allowed") is False
        and status.get("published_cmd_vel") is False
        and status.get("connected_navigation_or_frontier") is False
    )
    heartbeat_payload_ok = isinstance(status.get("heartbeat"), dict)
    input_freshness = status.get("input_freshness") if isinstance(status.get("input_freshness"), dict) else {}
    input_freshness_payload_ok = bool(input_freshness)
    input_freshness_pass = input_freshness.get("all_required_inputs_fresh") is True
    free_status_has_fresh_inputs = not (
        status.get("local_traversability_status") == "FREE_SUPPORTED" and not input_freshness_pass
    )
    freshness["input_freshness_pass"] = bool(input_freshness_pass)
    result.update({
        "topics_received": {
            GRID_TOPIC: True,
            EVIDENCE_TOPIC: True,
            FRONT_TOPIC: True,
            STATUS_TOPIC: True,
        },
        "grid_frame_id": grid.header.frame_id,
        "grid_width": int(grid.info.width),
        "grid_height": int(grid.info.height),
        "grid_resolution": float(grid.info.resolution),
        "grid_nonempty": grid_nonempty,
        "grid_stamp_freshness_pass": freshness["grid_stamp_freshness_pass"],
        "grid_stamp_monotonic": freshness["stamp_monotonic"],
        "grid_seq_monotonic": freshness["seq_monotonic"],
        "grid_unique_values": freshness["unique_values"],
        "local_traversability_grid_published": True,
        "front_auxiliary_evidence_published": True,
        "rgbd_front_auxiliary_effective": rgbd_useful,
        "l3u_main_conclusion_preserved": l3u_preserved,
        "safe_flags_clean": safe_flags_clean,
        "status_snapshot": status,
        "front_snapshot": front,
        "freshness_audit": freshness,
        "heartbeat_payload_ok": heartbeat_payload_ok,
        "input_freshness_payload_ok": input_freshness_payload_ok,
        "input_freshness_pass": input_freshness_pass,
        "free_status_has_fresh_inputs": free_status_has_fresh_inputs,
    })

    if not freshness["schema_pass"]:
        result["final_decision"] = "L3V_DIAGNOSTIC_NODE_BLOCKED_BY_SCHEMA"
    elif not freshness["grid_stamp_freshness_pass"]:
        result["final_decision"] = "L3V_DIAGNOSTIC_NODE_BLOCKED_BY_STALE_GRID_STAMP"
    elif not freshness.get("heartbeat_pass"):
        result["final_decision"] = "L3V_DIAGNOSTIC_NODE_BLOCKED_BY_HEARTBEAT"
        result["failure_reason"] = "heartbeat_observed_rate_below_1hz_or_freshness_failed"
    elif not heartbeat_payload_ok or not input_freshness_payload_ok:
        result["final_decision"] = "L3V_DIAGNOSTIC_NODE_BLOCKED_BY_SCHEMA"
        result["failure_reason"] = "status_missing_heartbeat_or_input_freshness"
    elif not input_freshness_pass or not free_status_has_fresh_inputs:
        result["final_decision"] = "L3V_DIAGNOSTIC_NODE_BLOCKED_BY_STALE_INPUT"
        result["failure_reason"] = "input_freshness_not_pass_or_free_when_stale"
    elif not grid_nonempty:
        result["final_decision"] = "L3V_DIAGNOSTIC_NODE_BLOCKED_BY_SCHEMA"
        result["failure_reason"] = "grid_has_no_free_or_occupied_cells"
    elif not rgbd_useful or not l3u_preserved:
        result["final_decision"] = "L3V_DIAGNOSTIC_NODE_BLOCKED_BY_SCHEMA"
        result["failure_reason"] = "diagnostic_payload_missing_required_l3u_evidence"
    elif safe_flags_clean:
        result["stable_publish_check_pass"] = True
        result["final_decision"] = "L3V_DIAGNOSTIC_NODE_READY"
    else:
        result["final_decision"] = "L3V_DIAGNOSTIC_NODE_BLOCKED_BY_SCHEMA"
        result["failure_reason"] = "safety_boundary_flags_not_clean"

    write_json(os.path.join(args.output_dir, "l3v_replay_smoke_topics.json"), result)
    write_json(os.path.join(args.output_dir, "l3v_freshness_audit.json"), freshness)
    write_report(args.report, result, front, evidence, status, freshness)
    print(result["final_decision"])


def write_report(path, result, front, evidence, status, freshness):
    decision = result.get("final_decision")
    freshness = freshness or {}
    lines = [
        "# L3V Local Traversability Diagnostic Node Report",
        "",
        "## Conclusion",
        "",
        f"- final_decision: `{decision}`",
        f"- online diagnostic node stable: `{result.get('stable_publish_check_pass', False)}`",
        f"- local_traversability_grid published: `{result.get('local_traversability_grid_published', False)}`",
        f"- grid_stamp_freshness_pass: `{result.get('grid_stamp_freshness_pass', False)}`",
        f"- grid_stamp_monotonic: `{result.get('grid_stamp_monotonic', False)}`",
        f"- grid_unique_values: `{result.get('grid_unique_values')}`",
        f"- front_auxiliary_evidence published: `{result.get('front_auxiliary_evidence_published', False)}`",
        f"- RGB-D front auxiliary effective in replay: `{result.get('rgbd_front_auxiliary_effective', False)}`",
        f"- L3U main conclusion preserved: `{result.get('l3u_main_conclusion_preserved', False)}`",
        "- connected navigation/frontier/L4: `False / False / False`",
        "",
        "## Freshness Audit",
        "",
        f"- sample_count: `{freshness.get('sample_count')}`",
        f"- sampler_window_sec: `{freshness.get('sampler_window_sec')}`",
        f"- message_count_total: `{freshness.get('message_count_total')}`",
        f"- valid_sample_count: `{freshness.get('valid_sample_count')}`",
        f"- distinct_stamp_count: `{freshness.get('distinct_stamp_count')}`",
        f"- latest_grid_stamp_age_sec: `{freshness.get('latest_grid_stamp_age_sec')}`",
        f"- heartbeat_rate_observed_hz: `{freshness.get('heartbeat_rate_observed_hz')}`",
        f"- heartbeat_pass: `{freshness.get('heartbeat_pass')}`",
        f"- input_freshness_pass: `{freshness.get('input_freshness_pass')}`",
        f"- frame_ids: `{freshness.get('frame_ids')}`",
        f"- seq_values: `{freshness.get('seq_values')}`",
        f"- stamp_values_sec: `{freshness.get('stamp_values_sec')}`",
        f"- stamp_changed: `{freshness.get('stamp_changed')}`",
        f"- schema_pass: `{freshness.get('schema_pass')}`",
        "",
        "## Required Answers",
        "",
        f"A. online diagnostic node can run stably: `{result.get('stable_publish_check_pass', False)}`",
        "- B. grid publish mode: `heartbeat`",
        f"- C. heartbeat payload present: `{result.get('heartbeat_payload_ok')}`",
        f"- D. 8s grid samples: `{freshness.get('message_count_total')}`",
        f"- E. stamp changed: `{freshness.get('stamp_changed')}`",
        f"- F. latest_grid_stamp_age_sec: `{freshness.get('latest_grid_stamp_age_sec')}`",
        f"- G. input freshness pass: `{result.get('input_freshness_pass')}`",
        f"- H. stale input prevents FREE_SUPPORTED: `{result.get('free_status_has_fresh_inputs')}`",
        f"- I. local_traversability_grid published: `{result.get('local_traversability_grid_published', False)}`",
        f"- J. front_auxiliary_evidence published: `{result.get('front_auxiliary_evidence_published', False)}`",
        f"- K. RGB-D front auxiliary still effective: `{result.get('rgbd_front_auxiliary_effective', False)}`",
        f"- L. local traversability layer preserves L3U conclusion: `{result.get('l3u_main_conclusion_preserved', False)}`",
        "- M. navigation/frontier/L4 connected: `False`",
        f"- N. can enter manual low-speed turn smoke pre-audit: `{decision == 'L3V_DIAGNOSTIC_NODE_READY'}`",
        "",
        "## Interface Snapshot",
        "",
        f"- grid_frame_id: `{result.get('grid_frame_id')}`",
        f"- grid_width: `{result.get('grid_width')}`",
        f"- grid_height: `{result.get('grid_height')}`",
        f"- grid_resolution: `{result.get('grid_resolution')}`",
    ]
    if front:
        lines.extend([
            "",
            "## Front Auxiliary Evidence",
            "",
            f"- front_lidar_point_count: `{front.get('front_lidar_point_count')}`",
            f"- front_rgbd_depth_valid_count: `{front.get('front_rgbd_depth_valid_count')}`",
            f"- front_to_side_density_ratio: `{front.get('front_to_side_density_ratio')}`",
            f"- lidar_front_weak: `{front.get('lidar_front_weak')}`",
            f"- depth_valid_ratio: `{front.get('depth_valid_ratio')}`",
            f"- rgbd_front_auxiliary_useful: `{front.get('rgbd_front_auxiliary_useful')}`",
        ])
    if status:
        lines.extend([
            "",
            "## Traversability Status",
            "",
            f"- sensor_inputs: `{status.get('sensor_inputs')}`",
            f"- grid_header_stamp_sec: `{status.get('grid_header_stamp_sec')}`",
            f"- grid_header_seq: `{status.get('grid_header_seq')}`",
            f"- publish_ros_time_sec: `{status.get('publish_ros_time_sec')}`",
            f"- odom_traversed_cell_count: `{status.get('odom_traversed_cell_count')}`",
            f"- odom_traversed_free_support_ratio: `{status.get('odom_traversed_free_support_ratio')}`",
            f"- odom_traversed_conflict_ratio: `{status.get('odom_traversed_conflict_ratio')}`",
            f"- local_traversability_status: `{status.get('local_traversability_status')}`",
            f"- heartbeat: `{status.get('heartbeat')}`",
            f"- input_freshness: `{status.get('input_freshness')}`",
            f"- grid_publish: `{status.get('grid_publish')}`",
        ])
    lines.extend([
        "",
        "## Compliance",
        "",
        "- forbidden topics used: `[]`",
        "- used Gazebo truth: `False`",
        "- published /cmd_vel: `False`",
        "- sent navigation goal: `False`",
        "- connected move_base/frontier/L4: `False`",
        "- planner-ready map claimed: `False`",
        "",
        "## Outputs",
        "",
        "- `debug/l3v_local_traversability_diagnostic_node/l3v_replay_smoke_topics.json`",
        "- `debug/l3v_local_traversability_diagnostic_node/l3v_freshness_audit.json`",
        "- `debug/l3v_local_traversability_diagnostic_node/l3v_local_traversability_grid_snapshot.pgm`",
        "- `debug/l3v_local_traversability_diagnostic_node/l3v_local_traversability_evidence_snapshot.json`",
        "- `debug/l3v_local_traversability_diagnostic_node/l3v_front_auxiliary_evidence_snapshot.json`",
        "- `debug/l3v_local_traversability_diagnostic_node/l3v_traversability_status_snapshot.json`",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


if __name__ == "__main__":
    main()
