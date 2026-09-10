#!/usr/bin/env python3
"""Read-only point-cloud observability diagnostic for corridor following.

This complements centerline_input_quality_diagnostic.py. The grid may not show
bilateral wall support, so this script checks whether raw compliant point-cloud
inputs expose wall geometry that can support corridor-center or wall-following
control. It does not publish /cmd_vel and does not read Gazebo truth.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
from sensor_msgs.msg import PointCloud2


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "debug" / "centerline_input_quality"
REPORT_PATH = ROOT / "audit_reports" / "pointcloud_centerline_observability_report.md"

TOPIC_LIDAR = "/team/livox/scan_cloud_filtered"
TOPIC_DEPTH_POINTS = "/real_sense/depth/points"


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def fit_line(points_xy: np.ndarray, side: str, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if points_xy.shape[0] < 2:
        return None
    bin_size = max(0.05, float(args.x_bin_size_m))
    x_min = float(np.min(points_xy[:, 0]))
    x_max = float(np.max(points_xy[:, 0]))
    percentile = args.left_inner_percentile if side == "left" else args.right_inner_percentile
    boundary: List[Tuple[float, float]] = []
    cur = math.floor(x_min / bin_size) * bin_size
    while cur <= x_max:
        nxt = cur + bin_size
        in_bin = points_xy[(points_xy[:, 0] >= cur) & (points_xy[:, 0] < nxt)]
        if in_bin.shape[0] >= args.min_points_per_bin:
            boundary.append((float(np.median(in_bin[:, 0])), float(np.percentile(in_bin[:, 1], percentile))))
        cur = nxt
    if len(boundary) >= args.min_boundary_bins:
        boundary_xy = np.array(boundary, dtype=float)
        if float(np.max(boundary_xy[:, 0]) - np.min(boundary_xy[:, 0])) >= args.min_x_span_m:
            fit_xy = boundary_xy
        else:
            fit_xy = points_xy
    else:
        fit_xy = points_xy
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    try:
        slope, intercept = np.polyfit(fit_xy[:, 0], fit_xy[:, 1], 1)
    except Exception:
        return None
    pred = slope * fit_xy[:, 0] + intercept
    residual = fit_xy[:, 1] - pred
    return {
        "side": side,
        "slope_dy_dx": float(slope),
        "intercept_y_m": float(intercept),
        "heading_parallel_rad": float(math.atan2(slope, 1.0)),
        "heading_parallel_deg": float(math.degrees(math.atan2(slope, 1.0))),
        "residual_rmse_m": float(math.sqrt(float(np.mean(residual * residual)))),
        "point_count": int(points_xy.shape[0]),
        "fit_point_count": int(fit_xy.shape[0]),
        "boundary_bin_count": int(len(boundary)),
        "used_boundary_fit": bool(fit_xy is not points_xy),
    }


class PointCloudSampler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.listener = tf.TransformListener()
        self.samples: List[Dict[str, Any]] = []
        self.raw_message_count = 0
        self.transform_failure_count = 0

    def lookup_to_base(self, source_frame: str) -> Tuple[Any, Any, bool]:
        source = source_frame.lstrip("/")
        if source == self.args.target_frame:
            return None, None, True
        try:
            trans, rot = self.listener.lookupTransform(self.args.target_frame, source, rospy.Time(0))
            return trans, rot, True
        except Exception:
            return None, None, False

    def transform_to_base(self, point: Sequence[float], source_frame: str) -> Optional[Tuple[float, float, float]]:
        trans, rot, ok = self.lookup_to_base(source_frame)
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        if ok and trans is not None:
            m = tf.transformations.quaternion_matrix(rot)
            v = np.dot(m, np.array([x, y, z, 1.0]))
            return float(v[0] + trans[0]), float(v[1] + trans[1]), float(v[2] + trans[2])
        if ok:
            return x, y, z
        source = source_frame.lower()
        if "optical" in source or "camera" in source or "realsense" in source:
            return float(z), float(-x), float(-y)
        self.transform_failure_count += 1
        return None

    def cloud_cb(self, msg: PointCloud2, source: str) -> None:
        self.raw_message_count += 1
        accepted: List[Tuple[float, float, float]] = []
        seen = 0
        for raw in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            seen += 1
            if seen % max(1, self.args.point_stride) != 0:
                continue
            p = self.transform_to_base(raw, msg.header.frame_id)
            if p is None:
                continue
            x, y, z = p
            if not (self.args.x_min_m <= x <= self.args.x_max_m):
                continue
            if not (-self.args.y_abs_max_m <= y <= self.args.y_abs_max_m):
                continue
            if not (self.args.z_min_m <= z <= self.args.z_max_m):
                continue
            accepted.append((x, y, z))
            if len(accepted) >= self.args.max_points_total:
                break
        self.samples.append(
            {
                "source": source,
                "frame_id": msg.header.frame_id,
                "stamp_sec": float(msg.header.stamp.to_sec()),
                "seen_points": int(seen),
                "accepted_points": accepted,
            }
        )

    def lidar_cb(self, msg: PointCloud2) -> None:
        self.cloud_cb(msg, "lidar")

    def depth_cb(self, msg: PointCloud2) -> None:
        if self.args.include_depth_points:
            self.cloud_cb(msg, "depth_points")


def analyze_points(points: np.ndarray, args: argparse.Namespace) -> Dict[str, Any]:
    if points.size == 0:
        return {
            "total_point_count": 0,
            "left_point_count": 0,
            "right_point_count": 0,
            "bilateral_wall_observable": False,
            "single_side_wall_observable": False,
            "reason": "no_points_in_window",
        }

    left = points[points[:, 1] >= args.wall_min_abs_y_m]
    right = points[points[:, 1] <= -args.wall_min_abs_y_m]
    center = points[np.abs(points[:, 1]) < args.wall_min_abs_y_m]
    left_fit = fit_line(left[:, :2], "left", args) if left.shape[0] >= args.min_side_points else None
    right_fit = fit_line(right[:, :2], "right", args) if right.shape[0] >= args.min_side_points else None

    result: Dict[str, Any] = {
        "x_range_m": [args.x_min_m, args.x_max_m],
        "y_abs_max_m": args.y_abs_max_m,
        "z_range_m": [args.z_min_m, args.z_max_m],
        "wall_min_abs_y_m": args.wall_min_abs_y_m,
        "total_point_count": int(points.shape[0]),
        "left_point_count": int(left.shape[0]),
        "right_point_count": int(right.shape[0]),
        "center_band_point_count": int(center.shape[0]),
        "left_y_percentiles_m": np.percentile(left[:, 1], [5, 50, 95]).astype(float).tolist() if left.size else None,
        "right_y_percentiles_m": np.percentile(right[:, 1], [5, 50, 95]).astype(float).tolist() if right.size else None,
        "left_wall_line": left_fit,
        "right_wall_line": right_fit,
        "bilateral_wall_observable": left_fit is not None and right_fit is not None,
        "single_side_wall_observable": left_fit is not None or right_fit is not None,
        "estimated_center_y_m": None,
        "estimated_width_m": None,
        "mean_wall_heading_rad": None,
        "mean_wall_heading_deg": None,
        "reason": None,
    }

    if left_fit is not None and right_fit is not None:
        # Estimate inner walls at the middle of the forward observation band.
        x_mid = 0.5 * (args.x_min_m + args.x_max_m)
        left_y = left_fit["slope_dy_dx"] * x_mid + left_fit["intercept_y_m"]
        right_y = right_fit["slope_dy_dx"] * x_mid + right_fit["intercept_y_m"]
        width = left_y - right_y
        center_y = 0.5 * (left_y + right_y)
        mean_heading = 0.5 * (left_fit["heading_parallel_rad"] + right_fit["heading_parallel_rad"])
        result.update(
            {
                "estimated_center_y_m": float(center_y),
                "estimated_width_m": float(width),
                "mean_wall_heading_rad": float(mean_heading),
                "mean_wall_heading_deg": float(math.degrees(mean_heading)),
            }
        )
        if width < args.min_corridor_width_m or width > args.max_corridor_width_m:
            result["bilateral_wall_observable"] = False
            result["reason"] = "bilateral_lines_width_out_of_range"
        else:
            result["reason"] = "bilateral_pointcloud_wall_lines_observable"
    elif left_fit is not None:
        result["mean_wall_heading_rad"] = left_fit["heading_parallel_rad"]
        result["mean_wall_heading_deg"] = left_fit["heading_parallel_deg"]
        result["reason"] = "left_wall_only_observable"
    elif right_fit is not None:
        result["mean_wall_heading_rad"] = right_fit["heading_parallel_rad"]
        result["mean_wall_heading_deg"] = right_fit["heading_parallel_deg"]
        result["reason"] = "right_wall_only_observable"
    else:
        result["reason"] = "insufficient_side_wall_points"

    return result


def summarize(args: argparse.Namespace, sampler: PointCloudSampler) -> Dict[str, Any]:
    all_points: List[Tuple[float, float, float]] = []
    by_source: Dict[str, int] = {}
    frames: Dict[str, int] = {}
    for sample in sampler.samples:
        pts = sample["accepted_points"]
        all_points.extend(pts)
        by_source[sample["source"]] = by_source.get(sample["source"], 0) + len(pts)
        frames[sample["frame_id"]] = frames.get(sample["frame_id"], 0) + 1
    if len(all_points) > args.max_points_total:
        all_points = all_points[: args.max_points_total]
    points = np.array(all_points, dtype=float) if all_points else np.empty((0, 3), dtype=float)
    analysis = analyze_points(points, args)

    if analysis.get("bilateral_wall_observable"):
        classification = "pointcloud_centerline_observable"
    elif analysis.get("single_side_wall_observable"):
        classification = "pointcloud_wall_direction_observable_centerline_not_direct"
    else:
        classification = "pointcloud_centerline_unobservable"

    return {
        "final_decision": "POINTCLOUD_CENTERLINE_OBSERVABILITY_DIAGNOSTIC_COMPLETE",
        "diagnostic_only": True,
        "published_cmd_vel": False,
        "forbidden_sources_used": [],
        "called_move_base": False,
        "sent_navigation_goal": False,
        "topics_read": [TOPIC_LIDAR] + ([TOPIC_DEPTH_POINTS] if args.include_depth_points else []),
        "sample_duration_sec": args.sample_sec,
        "raw_message_count": sampler.raw_message_count,
        "sample_batch_count": len(sampler.samples),
        "transform_failure_count": sampler.transform_failure_count,
        "accepted_points_by_source": by_source,
        "source_frame_message_counts": frames,
        "pointcloud_wall_observability": analysis,
        "input_quality_classification": classification,
        "recommendation": recommendation(classification),
    }


def recommendation(classification: str) -> str:
    if classification == "pointcloud_centerline_observable":
        return "Point cloud can provide bilateral wall centerline evidence; next step is to feed this lateral error into continuous centerline control."
    if classification == "pointcloud_wall_direction_observable_centerline_not_direct":
        return "Point cloud can provide wall direction but not full centerline; use it first for heading stabilization, and only use one-sided lateral control with conservative confidence."
    return "Point cloud does not currently expose reliable wall geometry in this window; inspect sensor pose, crop ranges, and L3V projection before controller changes."


def write_report(summary: Dict[str, Any]) -> None:
    obs = summary["pointcloud_wall_observability"]
    lines = [
        "# Pointcloud Centerline Observability Diagnostic",
        "",
        f"- final_decision: `{summary.get('final_decision')}`",
        f"- input_quality_classification: `{summary.get('input_quality_classification')}`",
        f"- recommendation: {summary.get('recommendation')}",
        f"- raw_message_count: `{summary.get('raw_message_count')}`",
        f"- accepted_points_by_source: `{summary.get('accepted_points_by_source')}`",
        f"- source_frame_message_counts: `{summary.get('source_frame_message_counts')}`",
        f"- transform_failure_count: `{summary.get('transform_failure_count')}`",
        f"- total_point_count: `{obs.get('total_point_count')}`",
        f"- left_point_count: `{obs.get('left_point_count')}`",
        f"- right_point_count: `{obs.get('right_point_count')}`",
        f"- bilateral_wall_observable: `{obs.get('bilateral_wall_observable')}`",
        f"- single_side_wall_observable: `{obs.get('single_side_wall_observable')}`",
        f"- estimated_center_y_m: `{obs.get('estimated_center_y_m')}`",
        f"- estimated_width_m: `{obs.get('estimated_width_m')}`",
        f"- mean_wall_heading_deg: `{obs.get('mean_wall_heading_deg')}`",
        f"- reason: `{obs.get('reason')}`",
        "",
        "## Boundary",
        "",
        "- published_cmd_vel: `false`",
        "- called_move_base: `false`",
        "- sent_navigation_goal: `false`",
        "- forbidden_sources_used: `[]`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sec", type=float, default=3.0)
    parser.add_argument("--target-frame", default="base")
    parser.add_argument("--include-depth-points", action="store_true")
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--max-points-total", type=int, default=12000)
    parser.add_argument("--x-min-m", type=float, default=0.40)
    parser.add_argument("--x-max-m", type=float, default=4.00)
    parser.add_argument("--y-abs-max-m", type=float, default=2.20)
    parser.add_argument("--z-min-m", type=float, default=0.15)
    parser.add_argument("--z-max-m", type=float, default=1.60)
    parser.add_argument("--wall-min-abs-y-m", type=float, default=0.45)
    parser.add_argument("--min-side-points", type=int, default=35)
    parser.add_argument("--x-bin-size-m", type=float, default=0.25)
    parser.add_argument("--min-points-per-bin", type=int, default=4)
    parser.add_argument("--min-boundary-bins", type=int, default=4)
    parser.add_argument("--min-x-span-m", type=float, default=0.90)
    parser.add_argument("--left-inner-percentile", type=float, default=10.0)
    parser.add_argument("--right-inner-percentile", type=float, default=90.0)
    parser.add_argument("--min-corridor-width-m", type=float, default=0.90)
    parser.add_argument("--max-corridor-width-m", type=float, default=3.20)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rospy.init_node("pointcloud_centerline_observability_diagnostic", anonymous=True, disable_signals=True)
    sampler = PointCloudSampler(args)
    rospy.Subscriber(TOPIC_LIDAR, PointCloud2, sampler.lidar_cb, queue_size=3)
    if args.include_depth_points:
        rospy.Subscriber(TOPIC_DEPTH_POINTS, PointCloud2, sampler.depth_cb, queue_size=3)
    rospy.sleep(0.5)

    deadline = time.monotonic() + max(0.1, args.sample_sec)
    rate = rospy.Rate(20.0)
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        rate.sleep()

    summary = summarize(args, sampler)
    out_path = OUT_DIR / "pointcloud_centerline_observability_summary.json"
    write_json(out_path, summary)
    write_report(summary)
    print(json.dumps({"final_decision": summary["final_decision"], "summary": str(out_path), "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
