#!/usr/bin/env python3
"""Pure, production-side bilateral wall geometry for an initial corridor axis.

This module deliberately has no ROS subscriptions, TF, planner, or command
authority.  Callers provide points already expressed in ``base`` and bind the
returned *unsigned* wall axis to their own same-epoch odometry sample.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def blend_angles(a: float, b: float, weight_b: float) -> float:
    weight_b = max(0.0, min(1.0, float(weight_b)))
    return math.atan2(
        (1.0 - weight_b) * math.sin(float(a)) + weight_b * math.sin(float(b)),
        (1.0 - weight_b) * math.cos(float(a)) + weight_b * math.cos(float(b)),
    )


def fit_wall_line(points_xy: np.ndarray, side: str, args: Any) -> Optional[Dict[str, Any]]:
    """Fit the existing per-side wall boundary rule without choosing an axis sign."""
    if points_xy.shape[0] < args.pointcloud_wall_min_side_points:
        return None
    bin_size = max(0.05, float(args.pointcloud_wall_x_bin_size_m))
    x_min = float(np.min(points_xy[:, 0]))
    x_max = float(np.max(points_xy[:, 0]))
    boundary: List[tuple] = []
    percentile = args.pointcloud_wall_left_inner_percentile if side == "left" else args.pointcloud_wall_right_inner_percentile
    current = math.floor(x_min / bin_size) * bin_size
    while current <= x_max:
        in_bin = points_xy[(points_xy[:, 0] >= current) & (points_xy[:, 0] < current + bin_size)]
        if in_bin.shape[0] >= args.pointcloud_wall_min_points_per_bin:
            boundary.append((float(np.median(in_bin[:, 0])), float(np.percentile(in_bin[:, 1], percentile))))
        current += bin_size
    if len(boundary) < args.pointcloud_wall_min_boundary_bins:
        return None
    boundary_xy = np.array(boundary, dtype=float)
    span = float(np.max(boundary_xy[:, 0]) - np.min(boundary_xy[:, 0]))
    if span < args.pointcloud_wall_min_x_span_m:
        return None
    try:
        slope, intercept = np.polyfit(boundary_xy[:, 0], boundary_xy[:, 1], 1)
    except Exception:
        return None
    residual = boundary_xy[:, 1] - (slope * boundary_xy[:, 0] + intercept)
    rmse = float(math.sqrt(float(np.mean(residual * residual))))
    if rmse > args.pointcloud_wall_max_line_rmse_m:
        return None
    point_residual = points_xy[:, 1] - (slope * points_xy[:, 0] + intercept)
    heading = float(math.atan2(slope, 1.0))
    return {
        "side": side, "slope_dy_dx": float(slope), "intercept_y_m": float(intercept),
        "heading_parallel_rad": heading, "heading_parallel_deg": float(math.degrees(heading)),
        "residual_rmse_m": rmse,
        "raw_point_residual_rmse_m": float(math.sqrt(float(np.mean(point_residual * point_residual)))),
        "point_count": int(points_xy.shape[0]), "boundary_bin_count": int(len(boundary)),
        "boundary_x_span_m": span,
        "boundary_points_xy": [[float(x), float(y)] for x, y in boundary[:12]],
    }


def estimate_bilateral_corridor_axis(points_base_xyz: np.ndarray, args: Any) -> Dict[str, Any]:
    """Return only a bilateral, unsigned physical corridor axis in ``base``."""
    arr = np.asarray(points_base_xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        arr = np.empty((0, 3), dtype=float)
    if arr.size:
        finite = np.isfinite(arr).all(axis=1)
        arr = arr[finite]
        arr = arr[
            (arr[:, 0] >= args.pointcloud_wall_x_min_m)
            & (arr[:, 0] <= args.pointcloud_wall_x_max_m)
            & (np.abs(arr[:, 1]) <= args.pointcloud_wall_y_abs_max_m)
            & (arr[:, 2] >= args.pointcloud_wall_z_min_m)
            & (arr[:, 2] <= args.pointcloud_wall_z_max_m)
        ]
    left = arr[arr[:, 1] >= args.pointcloud_wall_min_abs_y_m] if arr.size else np.empty((0, 3), dtype=float)
    right = arr[arr[:, 1] <= -args.pointcloud_wall_min_abs_y_m] if arr.size else np.empty((0, 3), dtype=float)
    left_fit = fit_wall_line(left[:, :2], "left", args) if left.shape[0] else None
    right_fit = fit_wall_line(right[:, :2], "right", args) if right.shape[0] else None
    if not left_fit or not right_fit:
        return {
            "valid": False, "reason": "BILATERAL_WALL_SUPPORT_UNAVAILABLE", "heading_base_rad": None,
            "confidence": 0.0, "left_support": int(left.shape[0]), "right_support": int(right.shape[0]),
            "left_wall_line": left_fit, "right_wall_line": right_fit,
        }
    heading = blend_angles(left_fit["heading_parallel_rad"], right_fit["heading_parallel_rad"], 0.5)
    disagreement = abs(normalize_angle(left_fit["heading_parallel_rad"] - right_fit["heading_parallel_rad"]))
    valid = abs(heading) <= float(args.pointcloud_wall_max_abs_heading_rad)
    support = min(float(left_fit["point_count"]), float(right_fit["point_count"])) / float(args.pointcloud_wall_min_side_points)
    quality = 1.0 - max(float(left_fit["residual_rmse_m"]), float(right_fit["residual_rmse_m"])) / float(args.pointcloud_wall_max_line_rmse_m)
    return {
        "valid": bool(valid), "reason": "BILATERAL_WALL_AXIS_OBSERVED" if valid else "WALL_AXIS_OUTSIDE_ALLOWED_BASE_RANGE",
        "heading_base_rad": float(heading) if valid else None, "confidence": max(0.0, min(1.0, support * quality)),
        "left_support": int(left_fit["point_count"]), "right_support": int(right_fit["point_count"]),
        "left_wall_line": left_fit, "right_wall_line": right_fit, "side_heading_disagreement_rad": float(disagreement),
    }


def bind_axis_to_odom(
    axis: Dict[str, Any],
    source_stamp: float,
    odom_binding: Dict[str, Any],
    forward_intent_heading_odom_rad: float,
) -> Dict[str, Any]:
    """Freeze a geometric axis in odom; the short target supplies sign only."""
    if not axis.get("valid"):
        return {"valid": False, "reason": str(axis.get("reason") or "CORRIDOR_AXIS_INVALID")}
    if not odom_binding.get("binding_valid"):
        return {"valid": False, "reason": str(odom_binding.get("reason") or "ODOM_BINDING_INVALID")}
    pose = odom_binding.get("source_pose_x_y_yaw")
    heading_base = axis.get("heading_base_rad")
    if not (isinstance(pose, list) and len(pose) == 3 and heading_base is not None):
        return {"valid": False, "reason": "CORRIDOR_AXIS_BINDING_INPUT_INVALID"}
    unsigned_heading_odom = normalize_angle(float(pose[2]) + float(heading_base))
    sign = 1 if math.cos(float(forward_intent_heading_odom_rad) - unsigned_heading_odom) >= 0.0 else -1
    heading_odom = normalize_angle(unsigned_heading_odom if sign > 0 else unsigned_heading_odom + math.pi)
    return {
        "valid": True, "reason": "BILATERAL_WALL_AXIS_BOUND_TO_GATED_ODOM",
        "heading_base_rad": float(heading_base), "heading_odom_rad": heading_odom,
        "source_stamp": float(source_stamp),
        "odom_epoch": "%s:seq-%s" % (odom_binding.get("frame_id"), odom_binding.get("odom_callback_sequence_t1")),
        "confidence": float(axis.get("confidence") or 0.0),
        "left_support": int(axis.get("left_support") or 0), "right_support": int(axis.get("right_support") or 0),
        "forward_sign": sign, "unsigned_heading_odom_rad": unsigned_heading_odom,
        "odom_binding": dict(odom_binding), "left_wall_line": axis.get("left_wall_line"),
        "right_wall_line": axis.get("right_wall_line"),
    }


def mature_bound_axis(
    samples: Sequence[Dict[str, Any]],
    *,
    minimum_samples: int,
    max_heading_deviation_rad: float,
) -> Dict[str, Any]:
    """Certify a short consecutive sequence before handing off long-axis authority.

    ``samples`` are already source-time-bound corridor axes.  This intentionally
    contains no ROS or motion authority: it only says whether the caller may
    freeze its most recent geometrically observed axis.
    """
    required = max(1, int(minimum_samples))
    recent = [dict(sample) for sample in samples[-required:]]
    if len(recent) < required:
        return {
            "valid": False,
            "reason": "CORRIDOR_AXIS_MATURITY_INSUFFICIENT_CONSECUTIVE_SAMPLES",
            "observed_consecutive_sample_count": len(recent),
            "required_consecutive_sample_count": required,
        }
    if any(not sample.get("valid") for sample in recent):
        return {
            "valid": False,
            "reason": "CORRIDOR_AXIS_MATURITY_SAMPLE_INVALID",
            "observed_consecutive_sample_count": len(recent),
            "required_consecutive_sample_count": required,
        }
    headings = [sample.get("heading_odom_rad") for sample in recent]
    if not all(heading is not None and math.isfinite(float(heading)) for heading in headings):
        return {
            "valid": False,
            "reason": "CORRIDOR_AXIS_MATURITY_HEADING_UNAVAILABLE",
            "observed_consecutive_sample_count": len(recent),
            "required_consecutive_sample_count": required,
        }
    reference = math.atan2(
        sum(math.sin(float(heading)) for heading in headings),
        sum(math.cos(float(heading)) for heading in headings),
    )
    max_deviation = max(abs(normalize_angle(float(heading) - reference)) for heading in headings)
    if max_deviation > float(max_heading_deviation_rad):
        return {
            "valid": False,
            "reason": "CORRIDOR_AXIS_MATURITY_HEADING_INCONSISTENT",
            "observed_consecutive_sample_count": len(recent),
            "required_consecutive_sample_count": required,
            "reference_heading_odom_rad": float(reference),
            "max_heading_deviation_rad": float(max_deviation),
            "allowed_heading_deviation_rad": float(max_heading_deviation_rad),
        }
    # The window's circular mean is both the consistency reference and the
    # only long-horizon authority.  Keep the newest observation's provenance
    # below solely as diagnostic context; it must not decide the frozen axis.
    result = dict(recent[-1])
    last_sample_heading = float(result["heading_odom_rad"])
    result.update({
        "valid": True,
        "reason": "CORRIDOR_AXIS_MATURE_CONSECUTIVE_BOUND_SAMPLES",
        "heading_odom_rad": float(reference),
        "maturity_frozen_heading_odom_rad": float(reference),
        "maturity_reference_heading_odom_rad": float(reference),
        "maturity_last_sample_heading_odom_rad": last_sample_heading,
        "maturity_max_heading_deviation_rad": float(max_deviation),
        "maturity_allowed_heading_deviation_rad": float(max_heading_deviation_rad),
        "maturity_consecutive_sample_count": len(recent),
        "maturity_required_consecutive_sample_count": required,
    })
    return result


def make_mature_certificate(
    mature_evidence: Dict[str, Any],
    *,
    source_time_valid_for_sec: float,
) -> Dict[str, Any]:
    """Freeze one already-mature axis for a delayed, single handoff consumer.

    This is deliberately a lifecycle record, not a second maturity rule.  The
    caller must first obtain ``mature_bound_axis()`` using the existing
    consecutive-sample and circular-mean contract.  An invalid later raw cloud
    may clear the *next* candidate window, but cannot retroactively invalidate
    this source-time-bounded certificate.
    """
    if mature_evidence.get("valid") is not True:
        return {"valid": False, "reason": "CORRIDOR_AXIS_CERTIFICATE_MATURITY_INVALID"}
    source_stamp = mature_evidence.get("source_stamp")
    heading = mature_evidence.get("maturity_frozen_heading_odom_rad")
    binding = mature_evidence.get("odom_binding")
    epoch = binding.get("odom_epoch_generation") if isinstance(binding, dict) else None
    if not (
        isinstance(source_stamp, (int, float)) and math.isfinite(float(source_stamp))
        and isinstance(heading, (int, float)) and math.isfinite(float(heading))
        and isinstance(epoch, int) and not isinstance(epoch, bool)
        and isinstance(source_time_valid_for_sec, (int, float))
        and math.isfinite(float(source_time_valid_for_sec)) and float(source_time_valid_for_sec) > 0.0
    ):
        return {"valid": False, "reason": "CORRIDOR_AXIS_CERTIFICATE_INPUT_INVALID"}
    result = dict(mature_evidence)
    result.update({
        "valid": True,
        "reason": "CORRIDOR_AXIS_MATURE_CERTIFICATE_CREATED",
        "certificate_id": "corridor-axis:e%s:t%.9f" % (int(epoch), float(source_stamp)),
        "certificate_source_stamp": float(source_stamp),
        "certificate_odom_epoch_generation": int(epoch),
        "certificate_heading_odom_rad": float(heading),
        "certificate_source_time_valid_for_sec": float(source_time_valid_for_sec),
        "certificate_consumed": False,
    })
    return result


def mature_certificate_fresh(
    certificate: Dict[str, Any],
    *,
    sim_now_sec: float,
    current_odom_epoch_generation: int,
) -> Dict[str, Any]:
    """Validate source-time age and continuity epoch before a handoff.

    The maximum age is supplied by the caller's existing source-time contract;
    this helper does not introduce a navigation- or scene-specific threshold.
    """
    if certificate.get("valid") is not True:
        return {"valid": False, "reason": "CORRIDOR_AXIS_CERTIFICATE_INVALID"}
    if certificate.get("certificate_consumed") is True:
        return {"valid": False, "reason": "CORRIDOR_AXIS_CERTIFICATE_ALREADY_CONSUMED"}
    source_stamp = certificate.get("certificate_source_stamp")
    max_age = certificate.get("certificate_source_time_valid_for_sec")
    certificate_epoch = certificate.get("certificate_odom_epoch_generation")
    if not (
        isinstance(source_stamp, (int, float)) and math.isfinite(float(source_stamp))
        and isinstance(max_age, (int, float)) and math.isfinite(float(max_age)) and float(max_age) > 0.0
        and isinstance(sim_now_sec, (int, float)) and math.isfinite(float(sim_now_sec))
        and isinstance(certificate_epoch, int) and not isinstance(certificate_epoch, bool)
        and isinstance(current_odom_epoch_generation, int) and not isinstance(current_odom_epoch_generation, bool)
    ):
        return {"valid": False, "reason": "CORRIDOR_AXIS_CERTIFICATE_FRESHNESS_INPUT_INVALID"}
    if int(certificate_epoch) != int(current_odom_epoch_generation):
        return {
            "valid": False,
            "reason": "CORRIDOR_AXIS_CERTIFICATE_ODOM_EPOCH_CHANGED",
            "certificate_odom_epoch_generation": int(certificate_epoch),
            "current_odom_epoch_generation": int(current_odom_epoch_generation),
        }
    age = float(sim_now_sec) - float(source_stamp)
    if age < 0.0 or age > float(max_age):
        return {
            "valid": False,
            "reason": "CORRIDOR_AXIS_CERTIFICATE_STALE",
            "certificate_age_sec": float(age),
            "certificate_source_time_valid_for_sec": float(max_age),
        }
    result = dict(certificate)
    result.update({
        "valid": True,
        "reason": "CORRIDOR_AXIS_MATURE_CERTIFICATE_FRESH",
        "certificate_age_sec": float(age),
    })
    return result
