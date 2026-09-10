"""Pure SE(2) reprojection for base-local traversability evidence."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


MOTION_CONSISTENCY_CONTRACT = "BASE_LOCAL_EVIDENCE_SE2_REPROJECT_V1"


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def pose_delta(old_pose: Tuple[float, float, float], new_pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Return translation and yaw change in the old odom/world frame."""
    return (
        float(new_pose[0]) - float(old_pose[0]),
        float(new_pose[1]) - float(old_pose[1]),
        normalize_angle(float(new_pose[2]) - float(old_pose[2])),
    )


def motion_is_valid(
    old_pose: Tuple[float, float, float],
    new_pose: Tuple[float, float, float],
    max_translation_m: float,
    max_yaw_delta_rad: float = math.pi / 2.0,
) -> Tuple[bool, str]:
    values = tuple(float(value) for value in (*old_pose, *new_pose))
    if not all(math.isfinite(value) for value in values):
        return False, "odom_nonfinite"
    dx, dy, dyaw = pose_delta(old_pose, new_pose)
    if math.hypot(dx, dy) > float(max_translation_m):
        return False, "odom_motion_jump_exceeds_evidence_window"
    if abs(dyaw) > float(max_yaw_delta_rad):
        return False, "odom_yaw_jump_exceeds_evidence_window"
    return True, "ok"


def warp_array(
    array: np.ndarray,
    old_pose: Tuple[float, float, float],
    new_pose: Tuple[float, float, float],
    resolution_m: float,
    x_min_m: float,
    y_min_m: float,
) -> np.ndarray:
    """Move cell-centre evidence from old base coordinates to new base coordinates.

    Evidence is conservatively nearest-cell splatted.  A reprojection must not
    manufacture observation strength: when several old cells quantize to one
    target cell, retain their strongest support rather than summing unrelated
    historical cells.  A direct obstacle observation of strength >= 1 remains
    occupied under the producer's existing > 0.5 threshold.
    """
    if array.ndim != 2:
        raise ValueError("evidence_array_must_be_2d")
    resolution = float(resolution_m)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution_m_must_be_positive")
    output = np.zeros_like(array)
    old_yaw = float(old_pose[2])
    new_yaw = float(new_pose[2])
    old_c, old_s = math.cos(old_yaw), math.sin(old_yaw)
    new_c, new_s = math.cos(new_yaw), math.sin(new_yaw)
    old_x0, old_y0 = float(old_pose[0]), float(old_pose[1])
    new_x0, new_y0 = float(new_pose[0]), float(new_pose[1])
    rows, cols = array.shape
    for row, column in zip(*np.nonzero(array)):
        old_x = float(x_min_m) + (int(row) + 0.5) * resolution
        old_y = float(y_min_m) + (int(column) + 0.5) * resolution
        world_x = old_x0 + old_c * old_x - old_s * old_y
        world_y = old_y0 + old_s * old_x + old_c * old_y
        dx, dy = world_x - new_x0, world_y - new_y0
        new_x = new_c * dx + new_s * dy
        new_y = -new_s * dx + new_c * dy
        new_row = int(math.floor((new_x - float(x_min_m)) / resolution))
        new_column = int(math.floor((new_y - float(y_min_m)) / resolution))
        if 0 <= new_row < rows and 0 <= new_column < cols:
            output[new_row, new_column] = max(
                output[new_row, new_column], array[row, column]
            )
    return output


def warp_sensor_evidence(
    arrays: Dict[str, np.ndarray],
    old_pose: Tuple[float, float, float],
    new_pose: Tuple[float, float, float],
    resolution_m: float,
    x_min_m: float,
    y_min_m: float,
) -> Dict[str, np.ndarray]:
    return {
        name: warp_array(array, old_pose, new_pose, resolution_m, x_min_m, y_min_m)
        for name, array in arrays.items()
    }
