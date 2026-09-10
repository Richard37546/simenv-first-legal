#!/usr/bin/env python3
"""Current-chain, local RGB-D danger perception sidecar.

This module deliberately has no dependency on legacy semantic observers, room
search, navigation, Gazebo truth, or world coordinates.  It consumes only RGB,
RGB CameraInfo, depth PointCloud2, and gated odometry to publish confirmed
danger tracks in ``team_livox_odom``.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


RGB_TOPIC = "/real_sense/rgb/image_raw"
RGB_INFO_TOPIC = "/real_sense/rgb/camera_info"
POINTS_TOPIC = "/real_sense/depth/points"
ODOM_TOPIC = "/team/livox/icp_odom_gated"
TRACK_TOPIC = "/team/danger_tracks"
HYPOTHESIS_TOPIC = "/team/danger_hypotheses"
TARGET_FRAME = "team_livox_odom"
CAMERA_FRAME = "real_sense"
SNAPSHOT_PATH = Path("debug/rgbd_danger_perception/latest_danger_tracks.json")
HYPOTHESIS_SNAPSHOT_PATH = Path("debug/rgbd_danger_perception/latest_danger_hypotheses.json")
SHADOW_VALIDATION_PATH = Path("debug/rgbd_danger_perception/shadow_validation.json")


@dataclass
class DetectorConfig:
    """All non-contract tuning values are explicit engineering constraints."""

    hue_low_max: float = 12.0
    hue_high_min: float = 168.0
    min_saturation: float = 100.0
    min_value: float = 70.0
    min_component_area_px: int = 40
    cleanup_neighbor_min: int = 2
    min_range_m: float = 0.40
    max_range_m: float = 8.0
    min_associated_points: int = 24
    expected_sphere_radius_m: float = 0.15
    sphere_radius_tolerance_m: float = 0.075
    max_sphere_residual_m: float = 0.025
    min_plane_residual_m: float = 0.004
    # P0: a real sphere is not explained by two substantial orthogonal planes.
    # Values are derived from the frozen red-cube replay and existing sphere fixture.
    multi_plane_inlier_distance_m: float = 0.010
    multi_plane_first_min_support_fraction: float = 0.44
    multi_plane_second_min_support_fraction: float = 0.30
    multi_plane_max_abs_normal_dot: float = 0.50
    multi_plane_ransac_trials: int = 128
    # Pixel-grid exposed-edge perimeter underestimates digital-circle
    # circularity; 0.55 is only an image pre-gate and is paired with 3-D checks.
    min_circularity: float = 0.55
    max_aspect_ratio: float = 1.45
    association_distance_m: float = 0.30
    confirmation_observations: int = 2
    min_observation_separation_sec: float = 0.04
    tentative_hypothesis_ttl_sec: float = 2.0
    explicit_sync_tolerance_sec: float = 0.0
    max_buffer_size: int = 40


@dataclass
class CameraModel:
    frame_id: str
    width: int
    height: int
    K: Sequence[float]
    distortion_model: str = ""
    D: Sequence[float] = field(default_factory=list)

    @property
    def fx(self) -> float:
        return float(self.K[0])

    @property
    def fy(self) -> float:
        return float(self.K[4])

    @property
    def cx(self) -> float:
        return float(self.K[2])

    @property
    def cy(self) -> float:
        return float(self.K[5])

    def valid(self) -> bool:
        return (
            self.frame_id == CAMERA_FRAME
            and self.width > 0
            and self.height > 0
            and len(self.K) == 9
            and all(math.isfinite(float(v)) for v in self.K)
            and self.fx > 0.0
            and self.fy > 0.0
            and self.distortion_model in ("", "plumb_bob")
            and all(math.isfinite(float(v)) for v in self.D)
        )


@dataclass
class RedCandidate:
    bbox_xywh: Tuple[int, int, int, int]
    centroid_uv: Tuple[float, float]
    area_px: int
    circularity: float
    aspect_ratio: float
    roi_mask: np.ndarray

    def contains_projected_pixel(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        x, y, width, height = self.bbox_xywh
        u_i = np.rint(u).astype(np.int64)
        v_i = np.rint(v).astype(np.int64)
        inside = (u_i >= x) & (u_i < x + width) & (v_i >= y) & (v_i < y + height)
        result = np.zeros(inside.shape, dtype=bool)
        if np.any(inside):
            result[inside] = self.roi_mask[v_i[inside] - y, u_i[inside] - x]
        return result


@dataclass
class GeometryEvidence:
    sphere_like: bool
    confidence: float
    center_camera_xyz_m: Optional[Tuple[float, float, float]]
    fitted_radius_m: Optional[float]
    sphere_residual_m: Optional[float]
    plane_residual_m: Optional[float]
    point_count: int
    rejection_reason: Optional[str]


@dataclass(frozen=True)
class MultiPlaneEvidence:
    """Deterministic two-plane evidence over the already-associated ROI points."""

    first_support_fraction: float
    second_support_fraction: float
    normal_abs_dot: Optional[float]
    multi_face_planar: bool


@dataclass
class OdomSample:
    stamp_sec: float
    position_xyz: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    frame_id: str = TARGET_FRAME
    child_frame_id: str = "base"

    def valid(self) -> bool:
        return (
            self.frame_id == TARGET_FRAME
            and self.child_frame_id == "base"
            and all(math.isfinite(float(v)) for v in self.position_xyz + self.quaternion_xyzw)
            and np.linalg.norm(np.asarray(self.quaternion_xyzw, dtype=float)) > 1e-9
        )


@dataclass
class DangerObservation:
    stamp_sec: float
    position_odom_xyz_m: Tuple[float, float, float]
    confidence: float
    evidence: GeometryEvidence


@dataclass
class DangerTrack:
    track_id: str
    position_xyz_m: np.ndarray
    confidence: float
    confirmation_count: int
    last_observed_stamp_sec: float
    state: str = "TENTATIVE"

    def public_row(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "position_xyz_m": [float(v) for v in self.position_xyz_m],
            "state": self.state,
            "confidence": float(self.confidence),
            "confirmation_count": int(self.confirmation_count),
            "last_observed_stamp_sec": float(self.last_observed_stamp_sec),
        }


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, str(path))
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def rgb_to_hsv_opencv_scale(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert uint8 RGB to OpenCV-compatible H[0,180), S/V[0,255]."""
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    maximum = np.maximum(np.maximum(r, g), b)
    minimum = np.minimum(np.minimum(r, g), b)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-8
    red = nonzero & (maximum == r)
    green = nonzero & (maximum == g)
    blue = nonzero & (maximum == b)
    hue[red] = np.mod((g[red] - b[red]) / delta[red], 6.0)
    hue[green] = (b[green] - r[green]) / delta[green] + 2.0
    hue[blue] = (r[blue] - g[blue]) / delta[blue] + 4.0
    hue = np.mod(hue * 30.0, 180.0)
    saturation = np.zeros_like(maximum)
    np.divide(delta, maximum, out=saturation, where=maximum > 1e-8)
    saturation *= 255.0
    return hue, saturation, maximum * 255.0


def cleanup_red_mask(mask: np.ndarray, neighbor_min: int) -> np.ndarray:
    """Remove isolated pixels without relying on OpenCV or SciPy."""
    source = np.asarray(mask, dtype=bool)
    padded = np.pad(source.astype(np.uint8), 1, mode="constant")
    neighborhood = np.zeros_like(source, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            neighborhood += padded[dy : dy + source.shape[0], dx : dx + source.shape[1]]
    return source & (neighborhood >= max(1, int(neighbor_min)))


def connected_components(mask: np.ndarray, min_area: int) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    source = np.asarray(mask, dtype=bool)
    visited = np.zeros(source.shape, dtype=bool)
    components: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    height, width = source.shape
    for y0, x0 in np.argwhere(source):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        pixels: List[Tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width and source[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        if len(pixels) < int(min_area):
            continue
        array = np.asarray(pixels, dtype=np.int64)
        y_min, x_min = np.min(array, axis=0)
        y_max, x_max = np.max(array, axis=0)
        roi = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
        roi[array[:, 0] - y_min, array[:, 1] - x_min] = True
        components.append((roi, (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))))
    return components


def extract_red_candidates(rgb: np.ndarray, config: DetectorConfig) -> List[RedCandidate]:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] < 3:
        return []
    hue, saturation, value = rgb_to_hsv_opencv_scale(rgb[..., :3])
    red = ((hue <= config.hue_low_max) | (hue >= config.hue_high_min)) & (saturation >= config.min_saturation) & (value >= config.min_value)
    result: List[RedCandidate] = []
    for roi, bbox in connected_components(cleanup_red_mask(red, config.cleanup_neighbor_min), config.min_component_area_px):
        area = int(np.count_nonzero(roi))
        ys, xs = np.nonzero(roi)
        x, y, width, height = bbox
        perimeter = int(np.count_nonzero(roi & ~np.pad(roi, ((1, 0), (0, 0)), mode="constant")[:-1, :]))
        perimeter += int(np.count_nonzero(roi & ~np.pad(roi, ((0, 1), (0, 0)), mode="constant")[1:, :]))
        perimeter += int(np.count_nonzero(roi & ~np.pad(roi, ((0, 0), (1, 0)), mode="constant")[:, :-1]))
        perimeter += int(np.count_nonzero(roi & ~np.pad(roi, ((0, 0), (0, 1)), mode="constant")[:, 1:]))
        circularity = float(4.0 * math.pi * area / max(1, perimeter * perimeter))
        aspect = max(float(width) / max(1, height), float(height) / max(1, width))
        result.append(RedCandidate(
            bbox_xywh=bbox,
            centroid_uv=(float(x + np.mean(xs)), float(y + np.mean(ys))),
            area_px=area,
            circularity=circularity,
            aspect_ratio=aspect,
            roi_mask=roi,
        ))
    return result


def project_points(points_xyz: np.ndarray, camera: CameraModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points, accounting for supported plumb-bob D."""
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not camera.valid():
        return np.empty(0), np.empty(0), np.empty(0, dtype=bool)
    finite = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0.0)
    u = np.full(len(points), np.nan, dtype=float)
    v = np.full(len(points), np.nan, dtype=float)
    if not np.any(finite):
        return u, v, finite
    x = points[finite, 0] / points[finite, 2]
    y = points[finite, 1] / points[finite, 2]
    if camera.distortion_model == "plumb_bob" and any(abs(float(value)) > 1e-12 for value in camera.D):
        d = list(camera.D) + [0.0] * max(0, 5 - len(camera.D))
        k1, k2, p1, p2, k3 = (float(value) for value in d[:5])
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xy = x * y
        x, y = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x), y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy
    u[finite] = camera.fx * x + camera.cx
    v[finite] = camera.fy * y + camera.cy
    return u, v, finite


def rounded_image_indices(u: np.ndarray, v: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest-pixel indices and a mask safe for NumPy image indexing."""
    col = np.rint(np.asarray(u, dtype=float)).astype(np.int64)
    row = np.rint(np.asarray(v, dtype=float)).astype(np.int64)
    valid = np.isfinite(u) & np.isfinite(v) & (col >= 0) & (col < int(width)) & (row >= 0) & (row < int(height))
    return row, col, valid


def associated_candidate_points(
    candidate: RedCandidate, points_xyz: np.ndarray, camera: CameraModel, config: DetectorConfig
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    finite_xyz = np.all(np.isfinite(points), axis=1)
    distance = np.zeros(len(points), dtype=float)
    distance[finite_xyz] = np.linalg.norm(points[finite_xyz], axis=1)
    valid_range = finite_xyz & (distance >= config.min_range_m) & (distance <= config.max_range_m)
    u, v, forward = project_points(points, camera)
    projected_in_image = np.zeros(len(points), dtype=bool)
    valid_projection = np.isfinite(u) & np.isfinite(v)
    projected_in_image[valid_projection] = (
        (u[valid_projection] >= 0.0) & (u[valid_projection] < camera.width)
        & (v[valid_projection] >= 0.0) & (v[valid_projection] < camera.height)
    )
    visible = forward & valid_range & projected_in_image
    if not np.any(visible):
        return np.empty((0, 3), dtype=float)
    selected = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(visible)
    selected[indices] = candidate.contains_projected_pixel(u[indices], v[indices])
    return points[selected]


def fit_sphere(points_xyz: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    """Algebraic sphere fit; returns centre, radius, radial RMS residual."""
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        return None, None, None
    matrix = np.column_stack((2.0 * points, np.ones(len(points))))
    target = np.sum(points * points, axis=1)
    try:
        coeff, _residuals, rank, _singular = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None
    if rank < 4:
        return None, None, None
    centre = coeff[:3]
    radius_sq = float(coeff[3] + np.dot(centre, centre))
    if not math.isfinite(radius_sq) or radius_sq <= 0.0:
        return None, None, None
    radius = math.sqrt(radius_sq)
    radial = np.linalg.norm(points - centre, axis=1)
    residual = float(np.sqrt(np.mean((radial - radius) ** 2)))
    return centre, radius, residual


def plane_residual(points_xyz: np.ndarray) -> Optional[float]:
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        return None
    centred = points - np.mean(points, axis=0)
    try:
        singular = np.linalg.svd(centred, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    return float(singular[-1] / math.sqrt(len(points))) if len(singular) else None


def _fit_plane(points_xyz: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        return None
    centre = np.mean(points, axis=0)
    try:
        _u, _singular, vectors = np.linalg.svd(points - centre, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vectors[-1]
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= 1e-12:
        return None
    return centre, normal / normal_norm


def _strongest_plane_inliers(points_xyz: np.ndarray, inlier_distance_m: float, trials: int, seed: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return a refitted strongest-plane mask and normal using deterministic RANSAC."""
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        return np.zeros(len(points), dtype=bool), None
    generator = np.random.default_rng(int(seed))
    best_mask = np.zeros(len(points), dtype=bool)
    best_count = 0
    best_mean = math.inf
    for _ in range(max(1, int(trials))):
        triple = points[generator.choice(len(points), size=3, replace=False)]
        normal = np.cross(triple[1] - triple[0], triple[2] - triple[0])
        normal_norm = float(np.linalg.norm(normal))
        if not math.isfinite(normal_norm) or normal_norm <= 1e-12:
            continue
        normal /= normal_norm
        distances = np.abs((points - triple[0]).dot(normal))
        mask = distances <= float(inlier_distance_m)
        count = int(np.count_nonzero(mask))
        mean = float(np.mean(distances[mask])) if count else math.inf
        if count > best_count or (count == best_count and mean < best_mean):
            best_mask, best_count, best_mean = mask, count, mean
    if best_count < 3:
        return best_mask, None
    plane = _fit_plane(points[best_mask])
    if plane is None:
        return best_mask, None
    centre, normal = plane
    for _ in range(2):
        refined_mask = np.abs((points - centre).dot(normal)) <= float(inlier_distance_m)
        if int(np.count_nonzero(refined_mask)) < 3:
            break
        refined = _fit_plane(points[refined_mask])
        if refined is None:
            break
        centre, normal = refined
        best_mask = refined_mask
    return best_mask, normal


def multi_face_plane_evidence(points_xyz: np.ndarray, config: DetectorConfig) -> MultiPlaneEvidence:
    """Detect two substantial near-orthogonal planar supports without a cuboid fitter."""
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[0] < 6 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        return MultiPlaneEvidence(0.0, 0.0, None, False)
    first_mask, first_normal = _strongest_plane_inliers(
        points, config.multi_plane_inlier_distance_m, config.multi_plane_ransac_trials, seed=0
    )
    first_fraction = float(np.count_nonzero(first_mask) / len(points))
    remaining = points[~first_mask]
    second_mask, second_normal = _strongest_plane_inliers(
        remaining, config.multi_plane_inlier_distance_m, config.multi_plane_ransac_trials, seed=1
    )
    second_fraction = float(np.count_nonzero(second_mask) / len(points))
    if first_normal is None or second_normal is None:
        return MultiPlaneEvidence(first_fraction, second_fraction, None, False)
    normal_abs_dot = float(abs(np.dot(first_normal, second_normal)))
    multi_face = (
        first_fraction >= config.multi_plane_first_min_support_fraction
        and second_fraction >= config.multi_plane_second_min_support_fraction
        and normal_abs_dot <= config.multi_plane_max_abs_normal_dot
    )
    return MultiPlaneEvidence(first_fraction, second_fraction, normal_abs_dot, bool(multi_face))


def classify_sphere_geometry(candidate: RedCandidate, points_xyz: np.ndarray, config: DetectorConfig) -> GeometryEvidence:
    points = np.asarray(points_xyz, dtype=float)
    count = int(len(points)) if points.ndim == 2 else 0
    if count < config.min_associated_points:
        return GeometryEvidence(False, 0.0, None, None, None, None, count, "insufficient_associated_points")
    centre, radius, sphere_rms = fit_sphere(points)
    planar_rms = plane_residual(points)
    if centre is None or radius is None or sphere_rms is None or planar_rms is None:
        return GeometryEvidence(False, 0.0, None, radius, sphere_rms, planar_rms, count, "sphere_fit_degenerate")
    image_score = min(1.0, candidate.circularity / max(1e-6, config.min_circularity))
    image_score *= min(1.0, config.max_aspect_ratio / max(1e-6, candidate.aspect_ratio))
    radius_error = abs(radius - config.expected_sphere_radius_m)
    radius_score = max(0.0, 1.0 - radius_error / config.sphere_radius_tolerance_m)
    residual_score = max(0.0, 1.0 - sphere_rms / config.max_sphere_residual_m)
    curvature_score = min(1.0, planar_rms / max(1e-6, config.min_plane_residual_m))
    confidence = float(max(0.0, min(1.0, 0.30 * image_score + 0.30 * radius_score + 0.25 * residual_score + 0.15 * curvature_score)))
    if candidate.circularity < config.min_circularity or candidate.aspect_ratio > config.max_aspect_ratio:
        reason = "image_shape_not_sphere_like"
    elif radius_error > config.sphere_radius_tolerance_m:
        reason = "fitted_radius_outside_constraint"
    elif sphere_rms > config.max_sphere_residual_m:
        reason = "sphere_residual_too_large"
    elif planar_rms < config.min_plane_residual_m:
        reason = "planar_box_like_geometry"
    else:
        multi_plane = multi_face_plane_evidence(points, config)
        reason = "multi_face_box_like_geometry" if multi_plane.multi_face_planar else None
    return GeometryEvidence(reason is None, confidence, tuple(float(v) for v in centre) if reason is None else None,
                            float(radius), float(sphere_rms), float(planar_rms), count, reason)


def quaternion_rotation_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12 or not math.isfinite(norm):
        raise ValueError("invalid_odom_quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def camera_point_to_odom(point_camera_xyz: Sequence[float], odom: OdomSample) -> Tuple[float, float, float]:
    if not odom.valid():
        raise ValueError("invalid_gated_odom")
    # PointCloud XYZ is in the RealSense optical convention: right, down,
    # forward.  Convert it before applying the physical base mount offset.
    optical = np.asarray(point_camera_xyz, dtype=float)
    if optical.shape != (3,) or not np.all(np.isfinite(optical)):
        raise ValueError("invalid_optical_point")
    point_base = np.asarray([optical[2], -optical[0], -optical[1]], dtype=float)
    camera_to_base = np.asarray([0.28, 0.0, 0.043], dtype=float)
    point_base = point_base + camera_to_base
    point_odom = quaternion_rotation_matrix(odom.quaternion_xyzw).dot(point_base) + np.asarray(odom.position_xyz, dtype=float)
    if not np.all(np.isfinite(point_odom)):
        raise ValueError("nonfinite_odom_point")
    return tuple(float(value) for value in point_odom)


def adaptive_sync_tolerance(stamps_a: Sequence[float], stamps_b: Sequence[float], explicit_sec: float) -> Optional[float]:
    if explicit_sec > 0.0:
        return float(explicit_sec)
    def period(stamps: Sequence[float]) -> Optional[float]:
        ordered = sorted(float(value) for value in stamps if math.isfinite(float(value)))
        deltas = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
        return float(np.median(deltas)) if len(deltas) >= 2 else None
    period_a, period_b = period(stamps_a), period(stamps_b)
    if period_a is None or period_b is None:
        return None
    return 0.45 * max(period_a, period_b)


def sample_stamp_sec(sample: Any) -> float:
    if hasattr(sample, "stamp_sec"):
        return float(sample.stamp_sec)
    if isinstance(sample, tuple) and sample:
        return float(sample[0])
    return float("nan")


def nearest_sample(samples: Sequence[Any], stamp_sec: float, tolerance_sec: Optional[float]) -> Optional[Any]:
    if tolerance_sec is None or tolerance_sec <= 0.0:
        return None
    eligible = [sample for sample in samples if math.isfinite(sample_stamp_sec(sample))]
    if not eligible:
        return None
    selected = min(eligible, key=lambda sample: abs(sample_stamp_sec(sample) - stamp_sec))
    sample_stamp = sample_stamp_sec(selected)
    return selected if abs(sample_stamp - stamp_sec) <= tolerance_sec else None


class DangerTrackManager:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        # This is a safety/semantic invariant, not merely a CLI default.
        self.config.confirmation_observations = max(2, int(self.config.confirmation_observations))
        self.config.tentative_hypothesis_ttl_sec = max(0.1, float(self.config.tentative_hypothesis_ttl_sec))
        self.tracks: List[DangerTrack] = []
        self._next_id = 1

    def expire_tentative(self, stamp_sec: float) -> bool:
        """Remove only stale unconfirmed evidence; confirmed output is retained."""
        before = len(self.tracks)
        self.tracks = [
            track for track in self.tracks
            if track.state == "CONFIRMED"
            or float(stamp_sec) - track.last_observed_stamp_sec <= self.config.tentative_hypothesis_ttl_sec
        ]
        return len(self.tracks) != before

    def add_observation(self, observation: DangerObservation) -> DangerTrack:
        self.expire_tentative(observation.stamp_sec)
        position = np.asarray(observation.position_odom_xyz_m, dtype=float)
        eligible = [track for track in self.tracks if np.linalg.norm(track.position_xyz_m - position) <= self.config.association_distance_m]
        if eligible:
            track = min(eligible, key=lambda row: float(np.linalg.norm(row.position_xyz_m - position)))
            time_is_new = observation.stamp_sec - track.last_observed_stamp_sec >= self.config.min_observation_separation_sec
            if time_is_new:
                total = track.confirmation_count + 1
                track.position_xyz_m = (track.position_xyz_m * track.confirmation_count + position) / total
                track.confidence = (track.confidence * track.confirmation_count + observation.confidence) / total
                track.confirmation_count = total
                track.last_observed_stamp_sec = observation.stamp_sec
        else:
            track = DangerTrack(
                track_id="danger-{0:04d}".format(self._next_id),
                position_xyz_m=position.copy(),
                confidence=float(observation.confidence),
                confirmation_count=1,
                last_observed_stamp_sec=float(observation.stamp_sec),
            )
            self._next_id += 1
            self.tracks.append(track)
        if track.confirmation_count >= self.config.confirmation_observations:
            track.state = "CONFIRMED"
        return track

    def snapshot(self, stamp_sec: float) -> Dict[str, Any]:
        rows = [track.public_row() for track in self.tracks if track.state == "CONFIRMED" and np.all(np.isfinite(track.position_xyz_m))]
        return {
            "schema_version": 1,
            "stamp_sec": float(stamp_sec),
            "frame_id": TARGET_FRAME,
            "tracks": rows,
        }

    def hypotheses_snapshot(self, stamp_sec: float) -> Dict[str, Any]:
        self.expire_tentative(stamp_sec)
        rows = [
            {
                "hypothesis_id": track.track_id,
                "state": "TENTATIVE",
                "position_xyz_m": [float(value) for value in track.position_xyz_m],
                "confidence": float(track.confidence),
                "support_count": int(track.confirmation_count),
                "last_observed_stamp_sec": float(track.last_observed_stamp_sec),
            }
            for track in self.tracks
            if track.state == "TENTATIVE" and np.all(np.isfinite(track.position_xyz_m))
        ]
        return {
            "schema_version": 1,
            "stamp_sec": float(stamp_sec),
            "frame_id": TARGET_FRAME,
            "hypotheses": rows,
        }


def summary_ms(values: Sequence[float]) -> Dict[str, Optional[float]]:
    valid = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if valid.size == 0:
        return {"count": 0, "median_ms": None, "p90_ms": None, "max_ms": None}
    return {
        "count": int(valid.size),
        "median_ms": float(np.median(valid)),
        "p90_ms": float(np.percentile(valid, 90)),
        "max_ms": float(np.max(valid)),
    }


class ShadowValidationRecorder:
    """Bounded sidecar observability; not a navigation or audit framework."""

    def __init__(self, path: Path, projection_dir: Path, max_projection_samples: int = 5,
                 red_event_path: Optional[Path] = None) -> None:
        self.path = path
        self.projection_dir = projection_dir
        self.red_event_path = red_event_path or path.with_name("red_candidate_events.jsonl")
        self.max_projection_samples = max(0, int(max_projection_samples))
        self.started_wall_sec = time.time()
        self.last_persist_monotonic = 0.0
        self.metadata: Dict[str, Any] = {}
        self.counts: Dict[str, int] = {"rgb_callbacks": 0, "cloud_callbacks": 0, "odom_callbacks": 0, "processed_frames": 0}
        self.skipped: Dict[str, int] = {}
        self.stage_ms: Dict[str, Deque[float]] = {name: deque(maxlen=256) for name in (
            "red_candidate_extraction", "pointcloud_decode", "projection_association", "sphere_plane_geometry",
            "odom_transform", "tracking_output", "full_processed_frame",
        )}
        self.cloud_point_counts: Deque[float] = deque(maxlen=256)
        self.geometry_point_counts: Deque[float] = deque(maxlen=256)
        self.processed_stamps: Deque[float] = deque(maxlen=256)
        self.events: Deque[Dict[str, Any]] = deque(maxlen=64)
        self.projection_samples: List[Dict[str, Any]] = []

    def set_metadata(self, key: str, value: Dict[str, Any]) -> None:
        self.metadata[key] = value

    def increment(self, key: str) -> None:
        self.counts[key] = int(self.counts.get(key, 0)) + 1

    def skip(self, reason: str) -> None:
        self.skipped[reason] = int(self.skipped.get(reason, 0)) + 1
        self.persist_if_due()

    def event(self, payload: Dict[str, Any]) -> None:
        # Event-triggered packet only: no RGB-D frame, cloud, crop, or mask is
        # retained here.  It is correlation evidence, never a control input.
        packet = dict(payload)
        bbox = packet.get("bbox_xywh") or [None, None, None, None]
        packet.update({
            "event_type": "RED_CANDIDATE",
            "frame_id": CAMERA_FRAME,
            "roi_center_xy": (
                [float(bbox[0]) + float(bbox[2]) / 2.0, float(bbox[1]) + float(bbox[3]) / 2.0]
                if len(bbox) == 4 and all(value is not None for value in bbox) else None
            ),
            "accepted": bool(packet.get("sphere_like") and packet.get("track_id")),
            "tentative_or_confirmed_track_id": packet.get("track_id"),
            "track_state": packet.get("track_state"),
            "danger_reobserve_affected": "NOT_AVAILABLE_IN_PERCEPTION_SIDECAR",
            "nearest_simulator_object_association": None,
        })
        self.events.append(packet)
        self.red_event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.red_event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")

    def add_stage_ms(self, name: str, elapsed_sec: float) -> None:
        if name in self.stage_ms and math.isfinite(elapsed_sec):
            self.stage_ms[name].append(float(elapsed_sec) * 1000.0)

    def add_projection_sample(self, payload: Dict[str, Any]) -> None:
        if len(self.projection_samples) < self.max_projection_samples:
            self.projection_samples.append(payload)

    def payload(self) -> Dict[str, Any]:
        stamps = list(self.processed_stamps)
        periods = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        return {
            "schema_version": 1,
            "started_wall_sec": self.started_wall_sec,
            "metadata": self.metadata,
            "counts": self.counts,
            "skipped": self.skipped,
            "timing_ms": {name: summary_ms(values) for name, values in self.stage_ms.items()},
            "processed_frame_rate_hz": (1.0 / float(np.median(periods))) if periods else None,
            "average_valid_cloud_points": float(np.mean(self.cloud_point_counts)) if self.cloud_point_counts else None,
            "average_geometry_points": float(np.mean(self.geometry_point_counts)) if self.geometry_point_counts else None,
            "projection_samples": self.projection_samples,
            "recent_events": list(self.events),
        }

    def persist_if_due(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self.last_persist_monotonic >= 0.5:
            atomic_write_json(self.path, self.payload())
            self.last_persist_monotonic = now


def ros_image_to_rgb(msg: Any) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    if encoding not in ("rgb8", "bgr8"):
        raise ValueError("unsupported_rgb_encoding:{0}".format(msg.encoding))
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    expected = int(msg.height) * int(msg.step)
    if len(data) < expected or int(msg.step) < int(msg.width) * 3:
        raise ValueError("invalid_rgb_layout")
    image = np.empty((int(msg.height), int(msg.width), 3), dtype=np.uint8)
    for row in range(int(msg.height)):
        start = row * int(msg.step)
        image[row] = data[start : start + int(msg.width) * 3].reshape((int(msg.width), 3))
    return image[..., ::-1] if encoding == "bgr8" else image


def camera_from_ros(msg: Any) -> CameraModel:
    return CameraModel(
        frame_id=str(msg.header.frame_id),
        width=int(msg.width),
        height=int(msg.height),
        K=[float(value) for value in msg.K],
        distortion_model=str(getattr(msg, "distortion_model", "") or ""),
        D=[float(value) for value in getattr(msg, "D", [])],
    )


def odom_from_ros(msg: Any) -> OdomSample:
    pose = msg.pose.pose
    return OdomSample(
        stamp_sec=float(msg.header.stamp.to_sec()),
        position_xyz=(float(pose.position.x), float(pose.position.y), float(pose.position.z)),
        quaternion_xyzw=(float(pose.orientation.x), float(pose.orientation.y), float(pose.orientation.z), float(pose.orientation.w)),
        frame_id=str(msg.header.frame_id),
        child_frame_id=str(msg.child_frame_id),
    )


class RosDangerPerceptionNode:
    """ROS wrapper. It has no publisher capable of commanding the robot."""

    def __init__(self, rospy: Any, config: DetectorConfig, snapshot_path: Path, validation_path: Path,
                 projection_dir: Path, red_event_path: Path, max_projection_samples: int) -> None:
        from nav_msgs.msg import Odometry  # type: ignore
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2  # type: ignore
        from std_msgs.msg import String  # type: ignore

        self.rospy = rospy
        self.config = config
        self.snapshot_path = snapshot_path
        self.validation = ShadowValidationRecorder(validation_path, projection_dir, max_projection_samples, red_event_path)
        self.camera: Optional[CameraModel] = None
        self.clouds: Deque[Tuple[float, Any]] = deque(maxlen=config.max_buffer_size)
        self.odometry: Deque[OdomSample] = deque(maxlen=config.max_buffer_size)
        self.rgb_stamps: Deque[float] = deque(maxlen=config.max_buffer_size)
        self.cloud_stamps: Deque[float] = deque(maxlen=config.max_buffer_size)
        self.odom_stamps: Deque[float] = deque(maxlen=config.max_buffer_size)
        self.tracks = DangerTrackManager(config)
        self.publisher = rospy.Publisher(TRACK_TOPIC, String, queue_size=4, latch=True)
        self.hypothesis_publisher = rospy.Publisher(HYPOTHESIS_TOPIC, String, queue_size=4, latch=True)
        self._string_type = String
        self._last_public_signature = ""
        self._last_hypothesis_signature = ""
        self.subscribers = [
            rospy.Subscriber(RGB_INFO_TOPIC, CameraInfo, self.camera_info_callback, queue_size=3),
            rospy.Subscriber(POINTS_TOPIC, PointCloud2, self.cloud_callback, queue_size=4),
            rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_callback, queue_size=20),
            rospy.Subscriber(RGB_TOPIC, Image, self.rgb_callback, queue_size=2),
        ]
        self.publish_snapshot(float(rospy.Time.now().to_sec()), force_persist=True)
        self.validation.persist_if_due(force=True)

    def log(self, level: str, reason: str) -> None:
        getattr(self.rospy, "log" + level)("[rgbd_danger_perception] " + reason)

    def camera_info_callback(self, msg: Any) -> None:
        candidate = camera_from_ros(msg)
        if candidate.valid():
            self.camera = candidate
            self.validation.set_metadata("rgb_camera_info", {
                "frame_id": candidate.frame_id, "width": candidate.width, "height": candidate.height,
                "K": [float(value) for value in candidate.K], "distortion_model": candidate.distortion_model,
                "D": [float(value) for value in candidate.D],
            })
        else:
            self.log("warn", "camera_info_rejected")

    def cloud_callback(self, msg: Any) -> None:
        self.validation.increment("cloud_callbacks")
        if str(msg.header.frame_id) != CAMERA_FRAME:
            self.log("warn", "pointcloud_frame_rejected:" + str(msg.header.frame_id))
            return
        stamp = float(msg.header.stamp.to_sec())
        if not math.isfinite(stamp):
            return
        self.clouds.append((stamp, msg))
        self.cloud_stamps.append(stamp)
        self.validation.set_metadata("pointcloud", {
            "frame_id": str(msg.header.frame_id), "width": int(msg.width), "height": int(msg.height),
            "point_step": int(msg.point_step), "row_step": int(msg.row_step), "is_dense": bool(msg.is_dense),
            "organized": bool(int(msg.height) > 1),
            "fields": [{"name": str(field.name), "offset": int(field.offset), "datatype": int(field.datatype), "count": int(field.count)} for field in msg.fields],
        })

    def odom_callback(self, msg: Any) -> None:
        self.validation.increment("odom_callbacks")
        sample = odom_from_ros(msg)
        if not sample.valid():
            self.log("warn", "gated_odom_rejected")
            return
        self.odometry.append(sample)
        self.odom_stamps.append(sample.stamp_sec)
        self.validation.set_metadata("gated_odom", {"frame_id": sample.frame_id, "child_frame_id": sample.child_frame_id})

    def _cloud_tolerance(self) -> Optional[float]:
        return adaptive_sync_tolerance(self.rgb_stamps, self.cloud_stamps, self.config.explicit_sync_tolerance_sec)

    def _odom_tolerance(self) -> Optional[float]:
        return adaptive_sync_tolerance(self.cloud_stamps, self.odom_stamps, self.config.explicit_sync_tolerance_sec)

    @staticmethod
    def cloud_points_xyz(msg: Any) -> np.ndarray:
        from sensor_msgs import point_cloud2  # type: ignore
        rows = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        return np.asarray(rows, dtype=float).reshape((-1, 3)) if rows else np.empty((0, 3), dtype=float)

    def publish_snapshot(self, stamp_sec: float, force_persist: bool = False) -> None:
        payload = self.tracks.snapshot(stamp_sec)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.publisher.publish(self._string_type(data=encoded))
        hypothesis_payload = self.tracks.hypotheses_snapshot(stamp_sec)
        hypothesis_encoded = json.dumps(hypothesis_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.hypothesis_publisher.publish(self._string_type(data=hypothesis_encoded))
        if force_persist or encoded != self._last_public_signature:
            atomic_write_json(self.snapshot_path, payload)
            self._last_public_signature = encoded
        if force_persist or hypothesis_encoded != self._last_hypothesis_signature:
            atomic_write_json(HYPOTHESIS_SNAPSHOT_PATH, hypothesis_payload)
            self._last_hypothesis_signature = hypothesis_encoded

    @staticmethod
    def diagnostic_counter_for_reason(reason: Optional[str]) -> str:
        text = str(reason or "")
        if text == "image_shape_not_sphere_like":
            return "image_shape_reject"
        if text == "insufficient_3d_points":
            return "insufficient_3d_points"
        if "radius" in text:
            return "radius_reject"
        if "sphere_residual" in text:
            return "sphere_residual_reject"
        if "plane" in text or "box" in text:
            return "plane_or_box_reject"
        return "association_reject"

    def save_projection_overlay(self, rgb: np.ndarray, points: np.ndarray, stamp_sec: float) -> None:
        if self.camera is None or len(self.validation.projection_samples) >= self.validation.max_projection_samples:
            return
        u, v, forward = project_points(points, self.camera)
        image_height, image_width = np.asarray(rgb).shape[:2]
        rows, cols, index_safe = rounded_image_indices(u, v, image_width, image_height)
        valid = forward & index_safe
        in_frame = int(np.count_nonzero(valid))
        sampled = np.flatnonzero(valid)
        if len(sampled) > 2500:
            sampled = sampled[::max(1, len(sampled) // 2500)]
        overlay = np.asarray(rgb, dtype=np.uint8).copy()
        if len(sampled):
            overlay[rows[sampled], cols[sampled]] = [0, 255, 0]
        try:
            from PIL import Image as PILImage
            self.validation.projection_dir.mkdir(parents=True, exist_ok=True)
            path = self.validation.projection_dir / "projection_{0:02d}_{1:.6f}.png".format(len(self.validation.projection_samples) + 1, stamp_sec)
            PILImage.fromarray(overlay, "RGB").save(str(path))
            image_path: Optional[str] = str(path)
        except Exception as exc:
            image_path = None
            self.log("warn", "projection_overlay_save_failed:" + str(exc))
        self.validation.add_projection_sample({
            "stamp_sec": float(stamp_sec), "valid_cloud_points": int(len(points)), "in_image_projected_points": in_frame,
            "in_image_projected_fraction": float(in_frame / len(points)) if len(points) else 0.0,
            "overlay_path": image_path,
        })

    def rgb_callback(self, msg: Any) -> None:
        self.validation.increment("rgb_callbacks")
        stamp = float(msg.header.stamp.to_sec())
        self.rgb_stamps.append(stamp)
        if self.camera is None or not self.camera.valid():
            self.log("debug", "rgb_skipped_camera_info_missing")
            self.validation.skip("camera_info_missing")
            return
        cloud_item = nearest_sample(self.clouds, stamp, self._cloud_tolerance())
        if cloud_item is None:
            self.log("debug", "rgb_skipped_pointcloud_sync_unavailable")
            self.validation.skip("pointcloud_sync_unavailable")
            return
        cloud_stamp, cloud_msg = cloud_item
        odom = nearest_sample(self.odometry, cloud_stamp, self._odom_tolerance())
        if odom is None:
            self.log("debug", "rgb_skipped_odom_sync_unavailable")
            self.validation.skip("odom_sync_unavailable")
            return
        try:
            full_start = time.perf_counter()
            rgb = ros_image_to_rgb(msg)
            red_start = time.perf_counter()
            candidates = extract_red_candidates(rgb, self.config)
            self.validation.add_stage_ms("red_candidate_extraction", time.perf_counter() - red_start)
            decode_start = time.perf_counter()
            points = self.cloud_points_xyz(cloud_msg)
            self.validation.add_stage_ms("pointcloud_decode", time.perf_counter() - decode_start)
        except Exception as exc:
            self.log("warn", "rgb_or_cloud_decode_failed:" + str(exc))
            self.validation.skip("rgb_or_cloud_decode_failed")
            return
        self.save_projection_overlay(rgb, points, cloud_stamp)
        self.validation.cloud_point_counts.append(float(len(points)))
        changed = self.tracks.expire_tentative(cloud_stamp)
        for candidate in candidates:
            self.validation.increment("red_candidate")
            association_start = time.perf_counter()
            associated = associated_candidate_points(candidate, points, self.camera, self.config)
            self.validation.add_stage_ms("projection_association", time.perf_counter() - association_start)
            self.validation.geometry_point_counts.append(float(len(associated)))
            geometry_start = time.perf_counter()
            evidence = classify_sphere_geometry(candidate, associated, self.config)
            self.validation.add_stage_ms("sphere_plane_geometry", time.perf_counter() - geometry_start)
            event: Dict[str, Any] = {
                "stamp_sec": float(cloud_stamp), "bbox_xywh": list(candidate.bbox_xywh), "area_px": int(candidate.area_px),
                "circularity": float(candidate.circularity), "aspect_ratio": float(candidate.aspect_ratio),
                "associated_point_count": int(len(associated)), "sphere_like": bool(evidence.sphere_like),
                "fitted_center_real_sense": None if evidence.center_camera_xyz_m is None else list(evidence.center_camera_xyz_m),
                "fitted_radius_m": evidence.fitted_radius_m, "sphere_residual_m": evidence.sphere_residual_m,
                "plane_residual_m": evidence.plane_residual_m, "confidence": evidence.confidence,
                "classification_reason": evidence.rejection_reason,
            }
            if not evidence.sphere_like or evidence.center_camera_xyz_m is None:
                self.validation.increment(self.diagnostic_counter_for_reason(evidence.rejection_reason))
                self.validation.event(event)
                continue
            try:
                transform_start = time.perf_counter()
                position = camera_point_to_odom(evidence.center_camera_xyz_m, odom)
                self.validation.add_stage_ms("odom_transform", time.perf_counter() - transform_start)
            except ValueError as exc:
                self.log("warn", "odom_composition_rejected:" + str(exc))
                event["classification_reason"] = "odom_composition_rejected:" + str(exc)
                self.validation.event(event)
                self.validation.increment("association_reject")
                continue
            before = self.tracks.snapshot(cloud_stamp)
            tracking_start = time.perf_counter()
            track = self.tracks.add_observation(DangerObservation(cloud_stamp, position, evidence.confidence, evidence))
            self.validation.add_stage_ms("tracking_output", time.perf_counter() - tracking_start)
            self.validation.increment("strong_sphere_support")
            event.update({"position_team_livox_odom": list(position), "track_id": track.track_id, "track_state": track.state,
                          "track_confirmation_count": track.confirmation_count})
            self.validation.event(event)
            changed = changed or before != self.tracks.snapshot(cloud_stamp)
        if changed:
            self.publish_snapshot(cloud_stamp)
        self.validation.increment("processed_frames")
        self.validation.processed_stamps.append(float(cloud_stamp))
        self.validation.add_stage_ms("full_processed_frame", time.perf_counter() - full_start)
        self.validation.persist_if_due(force=changed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone local RGB-D danger perception sidecar")
    parser.add_argument("--snapshot-path", default=str(SNAPSHOT_PATH))
    parser.add_argument("--shadow-validation-path", default=str(SHADOW_VALIDATION_PATH))
    parser.add_argument("--red-event-path", default="debug/rgbd_danger_perception/red_candidate_events.jsonl",
                        help="event-triggered compact red-candidate packets; no RGB-D payload")
    parser.add_argument("--projection-samples-dir", default="debug/rgbd_danger_perception/projection_samples")
    parser.add_argument("--max-projection-samples", type=int, default=5)
    parser.add_argument("--sync-tolerance-sec", type=float, default=0.0,
                        help="0 derives a conservative tolerance from observed topic cadence")
    parser.add_argument("--confirmation-observations", type=int, default=2)
    parser.add_argument("--tentative-hypothesis-ttl-sec", type=float, default=2.0)
    parser.add_argument("--association-distance-m", type=float, default=0.30)
    parser.add_argument("--min-component-area-px", type=int, default=40)
    parser.add_argument("--hue-low-max", type=float, default=12.0)
    parser.add_argument("--hue-high-min", type=float, default=168.0)
    parser.add_argument("--min-saturation", type=float, default=100.0)
    parser.add_argument("--min-value", type=float, default=70.0)
    parser.add_argument("--min-associated-points", type=int, default=24)
    parser.add_argument("--min-circularity", type=float, default=0.55)
    parser.add_argument("--max-aspect-ratio", type=float, default=1.45)
    parser.add_argument("--expected-sphere-radius-m", type=float, default=0.15)
    parser.add_argument("--sphere-radius-tolerance-m", type=float, default=0.075)
    parser.add_argument("--max-sphere-residual-m", type=float, default=0.025)
    parser.add_argument("--min-plane-residual-m", type=float, default=0.004)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        import rospy  # type: ignore
    except ImportError as exc:
        raise SystemExit("ROS Noetic environment required: {0}".format(exc))
    config = DetectorConfig(
        explicit_sync_tolerance_sec=max(0.0, float(args.sync_tolerance_sec)),
        confirmation_observations=max(2, int(args.confirmation_observations)),
        tentative_hypothesis_ttl_sec=max(0.1, float(args.tentative_hypothesis_ttl_sec)),
        association_distance_m=max(0.01, float(args.association_distance_m)),
        min_component_area_px=max(1, int(args.min_component_area_px)),
        hue_low_max=float(args.hue_low_max),
        hue_high_min=float(args.hue_high_min),
        min_saturation=max(0.0, float(args.min_saturation)),
        min_value=max(0.0, float(args.min_value)),
        min_associated_points=max(4, int(args.min_associated_points)),
        min_circularity=max(0.0, min(1.0, float(args.min_circularity))),
        max_aspect_ratio=max(1.0, float(args.max_aspect_ratio)),
        expected_sphere_radius_m=max(0.01, float(args.expected_sphere_radius_m)),
        sphere_radius_tolerance_m=max(0.001, float(args.sphere_radius_tolerance_m)),
        max_sphere_residual_m=max(0.0001, float(args.max_sphere_residual_m)),
        min_plane_residual_m=max(0.0001, float(args.min_plane_residual_m)),
    )
    rospy.init_node("rgbd_danger_perception", anonymous=False)
    RosDangerPerceptionNode(
        rospy, config, Path(args.snapshot_path), Path(args.shadow_validation_path),
        Path(args.projection_samples_dir), Path(args.red_event_path), max(0, int(args.max_projection_samples)),
    )
    rospy.loginfo("[rgbd_danger_perception] started; passive sidecar only")
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
