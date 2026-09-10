#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


FORBIDDEN_SOURCES = [
    "/Odometry_gazebo",
    "/ground_truth/*",
    "/gazebo/model_states",
    "/gazebo/link_states",
    "danger_truth",
    "generated_building_runtime_metadata",
    "gazebo_truth_pose",
]

LOCAL_XY_AXIS_CONVENTION = "bev_grid_xy_lateral_forward"
LOCAL_XY_TO_ROS_BASE_FORMULA = "x_ros_base = local_y_forward; y_ros_base = local_x_lateral"
TRANSFORM_FORMULA = (
    "target_x = robot_x + cos(yaw) * x_ros_base - sin(yaw) * y_ros_base; "
    "target_y = robot_y + sin(yaw) * x_ros_base + cos(yaw) * y_ros_base"
)

META_POSE_KEYS = [
    "robot_pose",
    "robot_pose_target",
    "base_pose",
    "base_pose_xy_yaw",
    "odom_pose",
    "odom_pose_xy_yaw",
    "map_pose",
    "map_pose_xy_yaw",
    "transform",
    "transforms",
    "tf",
    "frame_tree",
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def discover_meta_files(input_dir, prefixes):
    base = Path(input_dir)
    meta_files = []
    for prefix in prefixes:
        path = base / ("%s_meta.json" % prefix)
        if path.exists():
            meta_files.append(path)
    return meta_files


def load_l3zf(l3zf_dir):
    base = Path(l3zf_dir)
    top1_path = base / "l3zf_top1_sequence.json"
    top3_path = base / "l3zf_top3_sequence.json"
    frames_path = base / "l3zf_frame_summaries.json"
    metrics_path = base / "l3zf_stability_metrics.json"
    missing = [str(path) for path in [top1_path, top3_path, frames_path, metrics_path] if not path.exists()]
    if missing:
        return None, missing
    return {
        "top1": read_json(top1_path),
        "top3": read_json(top3_path),
        "frames": read_json(frames_path),
        "metrics": read_json(metrics_path),
    }, []


def meta_pose_audit(meta_files):
    records = []
    usable_sources = []
    for path in meta_files:
        data = read_json(path)
        present_pose_keys = [key for key in META_POSE_KEYS if key in data]
        record = {
            "meta_file": str(path),
            "frame": data.get("frame"),
            "stamp": data.get("stamp"),
            "frame_id": data.get("frame_id") or data.get("frame_id_local") or data.get("frame_id_world"),
            "present_pose_keys": present_pose_keys,
            "timestamp_available": data.get("stamp") is not None,
            "diagnostic_only": data.get("diagnostic_only"),
        }
        records.append(record)
        # A compliant shadow transform source must explicitly provide target-frame pose.
        for key in present_pose_keys:
            target = None
            if "odom" in key:
                target = "odom"
            elif "map" in key:
                target = "map"
            elif key in {"robot_pose_target", "transform", "transforms", "tf", "frame_tree"}:
                target = "unknown"
            if target in {"odom", "map"} and isinstance(data.get(key), list) and len(data.get(key)) >= 3:
                usable_sources.append({
                    "source_name": "meta:%s:%s" % (path.name, key),
                    "source_type": "meta_pose",
                    "source_frame": "base",
                    "target_frame": target,
                    "available": True,
                    "compliant": True,
                    "reason": "meta contains explicit %s pose with x/y/yaw" % key,
                    "timestamp_available": data.get("stamp") is not None,
                    "usable_for_shadow_transform": True,
                    "transform_timestamp": data.get("stamp"),
                    "pose": data.get(key),
                })
    return records, usable_sources


def build_source_audit(meta_records, meta_usable_sources):
    sources = [
        {
            "source_name": "/tf base->odom",
            "source_type": "tf",
            "source_frame": "base",
            "target_frame": "odom",
            "available": False,
            "compliant": False,
            "reason": "offline audit only; no captured /tf transform source provided in L3Zf/L3Ze artifacts",
            "timestamp_available": False,
            "usable_for_shadow_transform": False,
        },
        {
            "source_name": "/tf base->map",
            "source_type": "tf",
            "source_frame": "base",
            "target_frame": "map",
            "available": False,
            "compliant": False,
            "reason": "offline audit only; no captured /tf transform source provided in L3Zf/L3Ze artifacts",
            "timestamp_available": False,
            "usable_for_shadow_transform": False,
        },
        {
            "source_name": "/tf odom->base",
            "source_type": "tf",
            "source_frame": "odom",
            "target_frame": "base",
            "available": False,
            "compliant": False,
            "reason": "offline audit only; no captured inverse /tf transform source provided",
            "timestamp_available": False,
            "usable_for_shadow_transform": False,
        },
        {
            "source_name": "/tf map->base",
            "source_type": "tf",
            "source_frame": "map",
            "target_frame": "base",
            "available": False,
            "compliant": False,
            "reason": "offline audit only; no captured inverse /tf transform source provided",
            "timestamp_available": False,
            "usable_for_shadow_transform": False,
        },
        {
            "source_name": "SLAM pose topic",
            "source_type": "slam_pose",
            "source_frame": "base",
            "target_frame": "odom or map",
            "available": False,
            "compliant": False,
            "reason": "no explicit compliant SLAM pose topic snapshot is present in current L3Zf inputs",
            "timestamp_available": False,
            "usable_for_shadow_transform": False,
        },
        {
            "source_name": "BEV meta pose/transform",
            "source_type": "meta_pose",
            "source_frame": "base",
            "target_frame": "odom or map",
            "available": bool(any(record["present_pose_keys"] for record in meta_records)),
            "compliant": bool(meta_usable_sources),
            "reason": "new BEV meta files contain no explicit odom/map robot pose or base->odom/map transform" if not meta_usable_sources else "usable meta pose found",
            "timestamp_available": any(record["timestamp_available"] for record in meta_records),
            "usable_for_shadow_transform": bool(meta_usable_sources),
            "meta_records": meta_records,
        },
    ]
    sources.extend(meta_usable_sources)
    forbidden = [
        {
            "source_name": name,
            "source_type": "forbidden",
            "source_frame": None,
            "target_frame": None,
            "available": False,
            "compliant": False,
            "reason": "forbidden by task boundary; not read and not used",
            "timestamp_available": False,
            "usable_for_shadow_transform": False,
        }
        for name in FORBIDDEN_SOURCES
    ]
    return sources, forbidden


def normalize_local_xy_to_ros_base(local_xy):
    lateral = float(local_xy[0])
    forward = float(local_xy[1])
    return [forward, lateral]


def transform_xy(local_xy, pose):
    x_base, y_base = normalize_local_xy_to_ros_base(local_xy)
    x_robot, y_robot, yaw_robot = float(pose[0]), float(pose[1]), float(pose[2])
    return [
        x_robot + math.cos(yaw_robot) * x_base - math.sin(yaw_robot) * y_base,
        y_robot + math.sin(yaw_robot) * x_base + math.cos(yaw_robot) * y_base,
    ]


def skipped_candidate(candidate, source_frame="base", target_frame="odom"):
    local_xy = candidate.get("local_xy_base") or candidate.get("local_xy")
    return {
        "candidate_id": candidate.get("candidate_id"),
        "source_frame": source_frame,
        "target_frame": target_frame,
        "local_xy_base": local_xy,
        "target_xy_odom": [None, None] if target_frame == "odom" else None,
        "target_xy_map": [None, None] if target_frame == "map" else None,
        "transform_ready": False,
        "transform_source": None,
        "transform_timestamp": None,
        "transform_compliant": False,
        "failure_reason": "no compliant base->odom/map transform source",
        "send_to_navigation": False,
        "safe_for_navigation": False,
        "planner_ready": False,
        "diagnostic_only": True,
    }


def transformed_candidate(candidate, source):
    target_frame = source["target_frame"]
    local_xy = candidate.get("local_xy_base") or candidate.get("local_xy")
    ros_base_xy = normalize_local_xy_to_ros_base(local_xy)
    target_xy = transform_xy(local_xy, source["pose"])
    payload = {
        "candidate_id": candidate.get("candidate_id"),
        "source_frame": "base",
        "target_frame": target_frame,
        "numeric_source_frame": "local_xy_base_bev_grid",
        "axis_convention": LOCAL_XY_AXIS_CONVENTION,
        "local_xy_to_ros_base_formula": LOCAL_XY_TO_ROS_BASE_FORMULA,
        "local_xy_base": local_xy,
        "ros_base_xy": ros_base_xy,
        "transform_formula": TRANSFORM_FORMULA,
        "transform_ready": True,
        "transform_source": source["source_name"],
        "transform_timestamp": source.get("transform_timestamp"),
        "transform_compliant": True,
        "matched_pose_yaw": source["pose"][2] if source.get("pose") and len(source.get("pose")) >= 3 else None,
        "startup_anchor_used": False,
        "frame_contract_validated": True,
        "send_to_navigation": False,
        "safe_for_navigation": False,
        "planner_ready": False,
        "diagnostic_only": True,
    }
    if target_frame == "odom":
        payload["target_xy_odom"] = target_xy
    else:
        payload["target_xy_map"] = target_xy
    return payload


def apply_shadow_transform(l3zf, selected_source):
    if not selected_source:
        top1 = []
        for item in l3zf["top1"]:
            candidate = item.get("top1_candidate")
            top1.append({
                "prefix": item.get("prefix"),
                "frame": item.get("frame"),
                "transformed_candidate": skipped_candidate(candidate) if candidate else None,
                "navigation_frame_ready": False,
                "send_to_navigation": False,
                "safe_for_navigation": False,
                "planner_ready": False,
                "diagnostic_only": True,
            })
        top3 = []
        for item in l3zf["top3"]:
            transformed = [skipped_candidate(candidate) for candidate in item.get("top3_candidates", [])]
            top3.append({
                "prefix": item.get("prefix"),
                "frame": item.get("frame"),
                "transformed_candidates": transformed,
                "navigation_frame_ready": False,
                "send_to_navigation": False,
                "safe_for_navigation": False,
                "planner_ready": False,
                "diagnostic_only": True,
            })
        return top1, top3

    top1 = []
    for item in l3zf["top1"]:
        candidate = item.get("top1_candidate")
        top1.append({
            "prefix": item.get("prefix"),
            "frame": item.get("frame"),
            "transformed_candidate": transformed_candidate(candidate, selected_source) if candidate else None,
            "navigation_frame_ready": True,
            "send_to_navigation": False,
            "safe_for_navigation": False,
            "planner_ready": False,
            "diagnostic_only": True,
        })
    top3 = []
    for item in l3zf["top3"]:
        transformed = [transformed_candidate(candidate, selected_source) for candidate in item.get("top3_candidates", [])]
        top3.append({
            "prefix": item.get("prefix"),
            "frame": item.get("frame"),
            "transformed_candidates": transformed,
            "navigation_frame_ready": True,
            "send_to_navigation": False,
            "safe_for_navigation": False,
            "planner_ready": False,
            "diagnostic_only": True,
        })
    return top1, top3


def collect_failures(top1, top3, selected_source):
    failures = []
    if not selected_source:
        failures.append({
            "type": "transform_source_not_ready",
            "reason": "no compliant base->odom/map transform source found",
        })
    for item in top1:
        candidate = item.get("transformed_candidate")
        if candidate and not candidate.get("transform_ready"):
            failures.append({
                "type": "top1_transform_skipped",
                "prefix": item.get("prefix"),
                "frame": item.get("frame"),
                "candidate_id": candidate.get("candidate_id"),
                "reason": candidate.get("failure_reason"),
            })
    for item in top3:
        for candidate in item.get("transformed_candidates", []):
            if not candidate.get("transform_ready"):
                failures.append({
                    "type": "top3_transform_skipped",
                    "prefix": item.get("prefix"),
                    "frame": item.get("frame"),
                    "candidate_id": candidate.get("candidate_id"),
                    "reason": candidate.get("failure_reason"),
                })
    return failures


def choose_conclusion(l3zf, compliant_sources, forbidden_sources, top1, top3):
    if not l3zf or not l3zf.get("top1") or not l3zf.get("top3"):
        return "L3ZG_INPUT_CANDIDATES_MISSING"
    if not compliant_sources and any(src["available"] for src in forbidden_sources):
        return "L3ZG_FORBIDDEN_SOURCE_ONLY"
    if not compliant_sources:
        return "L3ZG_TRANSFORM_SOURCE_NOT_READY"
    transformed_top1 = sum(1 for item in top1 if item.get("transformed_candidate", {}).get("transform_ready"))
    transformed_top3 = sum(
        1
        for item in top3
        for candidate in item.get("transformed_candidates", [])
        if candidate.get("transform_ready")
    )
    total_top3 = sum(len(item.get("transformed_candidates", [])) for item in top3)
    if transformed_top1 == len(top1) and transformed_top3 == total_top3:
        return "L3ZG_SHADOW_TRANSFORM_READY"
    return "L3ZG_TRANSFORM_PARTIAL_FAILURE"


def write_report(path, audit, metrics, conclusion):
    lines = [
        "# L3Zg Transform Audit + Shadow Adapter Report",
        "",
        "## Final Conclusion",
        "",
        "`%s`" % conclusion,
        "",
        "## Scope",
        "",
        "- Stage 1: transform source audit.",
        "- Stage 2: shadow transform adapter only if a compliant source exists.",
        "- No `/cmd_vel`, no `move_base`, no navigation goal, no L4.",
        "- Local `base` candidates are not treated as global goals.",
        "",
        "## Transform Source Audit",
        "",
        "- transform_source_count: `%s`" % metrics["transform_source_count"],
        "- compliant_transform_source_count: `%s`" % metrics["compliant_transform_source_count"],
        "- forbidden_source_detected_count: `%s`" % metrics["forbidden_source_detected_count"],
        "- selected_transform_source: `%s`" % metrics["selected_transform_source"],
        "- target_frame: `%s`" % metrics["target_frame"],
        "",
        "## Shadow Transform Metrics",
        "",
        "- transformed_top1_count: `%s`" % metrics["transformed_top1_count"],
        "- transformed_top3_count: `%s`" % metrics["transformed_top3_count"],
        "- transform_success_ratio: `%s`" % metrics["transform_success_ratio"],
        "- navigation_frame_ready: `%s`" % metrics["navigation_frame_ready"],
        "- planner_ready: `%s`" % metrics["planner_ready"],
        "- send_to_navigation: `%s`" % metrics["send_to_navigation"],
        "",
        "## Source Details",
        "",
    ]
    for source in audit["sources"]:
        lines.append("- `%s`: available=`%s`, compliant=`%s`, usable=`%s`, reason=%s" % (
            source.get("source_name"),
            source.get("available"),
            source.get("compliant"),
            source.get("usable_for_shadow_transform"),
            source.get("reason"),
        ))
    lines += [
        "",
        "## Forbidden Sources",
        "",
    ]
    for source in audit["forbidden_sources"]:
        lines.append("- `%s`: available=`%s`, compliant=`%s`, reason=%s" % (
            source.get("source_name"),
            source.get("available"),
            source.get("compliant"),
            source.get("reason"),
        ))
    lines += [
        "",
        "## Compliance",
        "",
        "- send_to_navigation: `false`",
        "- safe_for_navigation: `false`",
        "- planner_ready: `false`",
        "- diagnostic_only: `true`",
        "- forbidden topics / Gazebo truth: not read and not used.",
    ]
    if conclusion == "L3ZG_TRANSFORM_SOURCE_NOT_READY":
        lines += [
            "",
            "## Blocking Reason",
            "",
            "No compliant base->odom or base->map transform source is present in the offline L3Zf/L3Ze artifacts or BEV meta files. Candidate shadow transform was therefore skipped.",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l3zf-dir", default="debug/l3zf_multiframe_replay")
    parser.add_argument("--input-dir", default="/home/richard/.ros/results/bev_maps")
    parser.add_argument("--output-dir", default="debug/l3zg_transform_adapter")
    parser.add_argument("--report", default="audit_reports/l3zg_transform_adapter_report.md")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    l3zf, missing = load_l3zf(args.l3zf_dir)
    if l3zf:
        prefixes = [frame["prefix"] for frame in l3zf["frames"]]
    else:
        prefixes = []
    meta_files = discover_meta_files(args.input_dir, prefixes)
    meta_records, meta_sources = meta_pose_audit(meta_files)
    sources, forbidden_sources = build_source_audit(meta_records, meta_sources)
    compliant_sources = [source for source in sources if source.get("usable_for_shadow_transform") and source.get("compliant")]
    selected_source = compliant_sources[0] if compliant_sources else None
    top1, top3 = apply_shadow_transform(l3zf, selected_source) if l3zf else ([], [])
    failures = collect_failures(top1, top3, selected_source)
    transformed_top1_count = sum(1 for item in top1 if item.get("transformed_candidate", {}).get("transform_ready"))
    transformed_top3_count = sum(
        1
        for item in top3
        for candidate in item.get("transformed_candidates", [])
        if candidate.get("transform_ready")
    )
    total_transform_attempts = len(top1) + sum(len(item.get("transformed_candidates", [])) for item in top3)
    transform_success_ratio = (transformed_top1_count + transformed_top3_count) / float(max(total_transform_attempts, 1))
    conclusion = choose_conclusion(l3zf, compliant_sources, forbidden_sources, top1, top3)
    metrics = {
        "final_conclusion": conclusion,
        "input_candidates_missing": bool(missing),
        "missing_l3zf_files": missing,
        "transform_source_count": len(sources),
        "compliant_transform_source_count": len(compliant_sources),
        "forbidden_source_detected_count": sum(1 for source in forbidden_sources if source.get("available")),
        "selected_transform_source": selected_source.get("source_name") if selected_source else None,
        "target_frame": selected_source.get("target_frame") if selected_source else None,
        "transformed_top1_count": transformed_top1_count,
        "transformed_top3_count": transformed_top3_count,
        "transform_success_ratio": transform_success_ratio,
        "transform_failure_reasons": sorted(set(failure["reason"] for failure in failures if failure.get("reason"))),
        "navigation_frame_ready": bool(selected_source and transform_success_ratio > 0.0),
        "planner_ready": False,
        "send_to_navigation": False,
        "safe_for_navigation": False,
        "diagnostic_only": True,
    }
    audit = {
        "final_conclusion": conclusion,
        "input_dir": args.input_dir,
        "l3zf_dir": args.l3zf_dir,
        "sources": sources,
        "forbidden_sources": forbidden_sources,
        "meta_files_checked": [str(path) for path in meta_files],
        "meta_pose_audit": meta_records,
        "selected_transform_source": selected_source,
        "send_to_navigation": False,
        "safe_for_navigation": False,
        "planner_ready": False,
        "diagnostic_only": True,
    }

    write_json(output_dir / "l3zg_transform_source_audit.json", audit)
    write_json(output_dir / "l3zg_transformed_top1_sequence.json", top1)
    write_json(output_dir / "l3zg_transformed_top3_sequence.json", top3)
    write_json(output_dir / "l3zg_transform_metrics.json", metrics)
    write_json(output_dir / "l3zg_transform_failures.json", failures)
    write_report(Path(args.report), audit, metrics, conclusion)
    print(conclusion)


if __name__ == "__main__":
    main()
