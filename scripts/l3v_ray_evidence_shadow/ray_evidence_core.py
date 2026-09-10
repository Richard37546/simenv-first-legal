#!/usr/bin/env python3
"""Pure P2K-G7 ray evidence and A0-compatible grid reference.

No ROS imports, publishers, portal semantics, targets, or control outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


RESOLUTION, X_MAX, Y_MIN, Y_MAX = 0.05, 3.0, -1.5, 1.5
ROWS, COLS, POINT_CAP, DECAY = 60, 60, 4000, 0.96
SPARSE_RAY_DTYPE = np.dtype([
    ("ray_id", "<i8"), ("point_index", "<i4"), ("side", "u1"),
    # Float64 prevents a valid endpoint from crossing a 5 cm cell boundary
    # merely because it was serialized into the sparse audit representation.
    ("endpoint_xyz", "<f8", (3,)), ("measured_range", "<f8"),
    ("height_class", "u1"), ("occupancy_eligible", "u1"),
    # Float64 keeps a clipped half-open boundary below X_MAX/Y_MAX on disk.
    ("grid_intersection_xy", "<f8", (2,)), ("intersection_valid", "u1"),
    ("clipped", "u1"), ("clipped_axes", "u1"),
])
HEIGHT_OTHER, HEIGHT_NEAR_GROUND, HEIGHT_OCCUPIED = 0, 1, 2
AXIS_X, AXIS_Y = 1, 2


def local_to_cell(x: float, y: float) -> Optional[Tuple[int, int]]:
    if not (math.isfinite(x) and math.isfinite(y)) or x < 0.0 or x >= X_MAX or y < Y_MIN or y >= Y_MAX:
        return None
    row, col = int(math.floor(x / RESOLUTION)), int(math.floor((y - Y_MIN) / RESOLUTION))
    return (row, col) if 0 <= row < ROWS and 0 <= col < COLS else None


def bresenham(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    r0, c0 = a; r1, c1 = b; dr, dc = abs(r1-r0), abs(c1-c0)
    sr, sc, err, out = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1), dc-dr, []
    while True:
        out.append((r0, c0))
        if (r0, c0) == (r1, c1): return out
        twice = 2 * err
        if twice > -dr: err -= dr; c0 += sc
        if twice < dc: err += dc; r0 += sr


def grid_intersection(endpoint: Tuple[float, float, float]) -> Tuple[Optional[Tuple[float, float]], List[str]]:
    """Return the measured segment's last point inside the half-open grid.

    The returned point is geometry only.  It is never an occupied endpoint.
    A ray pointing rearward from the base-frame origin has no positive-length
    intersection with the forward-only grid and is therefore recorded as
    unprojectable for that consumer.
    """
    x, y, _ = endpoint
    if local_to_cell(x, y) is not None:
        return (x, y), []
    if x < 0.0:
        return None, ["x"]

    hi_x = float(np.nextafter(X_MAX, -math.inf))
    hi_y = float(np.nextafter(Y_MAX, -math.inf))
    candidates: List[Tuple[float, str]] = []
    if x > 0.0:
        candidates.append((hi_x / x, "x"))
    if y > hi_y:
        candidates.append((hi_y / y, "y"))
    elif y < Y_MIN:
        candidates.append((Y_MIN / y, "y"))
    usable = [(t, axis) for t, axis in candidates if 0.0 <= t < 1.0]
    if not usable:
        return None, []
    t = min(item[0] for item in usable)
    axes = sorted(axis for candidate, axis in usable if math.isclose(candidate, t, rel_tol=0.0, abs_tol=1e-12))
    return (x * t, y * t), axes


@dataclass(frozen=True)
class RayEvidence:
    source_stamp: float
    transform_stamp: float
    source_frame: str
    target_frame: str
    ray_id: int
    point_index: int
    ray_origin_base: Tuple[float, float, float]
    ray_endpoint_base: Tuple[float, float, float]
    measured_range: float
    endpoint_valid: bool
    endpoint_height_class: str
    endpoint_occupancy_eligible: bool
    ray_observed: bool
    current_grid_intersection: Optional[Tuple[float, float]]
    clipped: bool
    clipped_axes: Tuple[str, ...]
    source_provenance: str
    input_quality_flags: Tuple[str, ...]

    def compact(self) -> Dict[str, Any]:
        return asdict(self)


class RayEvidenceCore:
    """Forms only finite, measured base-frame rays; it makes no decisions."""
    def __init__(self) -> None:
        self._next_id = 0

    def observe(self, points_base: Iterable[Tuple[float, float, float]], source_stamp: float, transform_stamp: float,
                source_frame: str, provenance: str = "l1s_filtered_cloud") -> List[RayEvidence]:
        result: List[RayEvidence] = []
        for index, raw in enumerate(points_base):
            x, y, z = map(float, raw)
            if not all(math.isfinite(v) for v in (x, y, z)):
                continue
            distance = math.sqrt(x*x+y*y+z*z)
            if distance < .10 or distance > 8.0:
                continue
            height = "occupied_eligible" if .15 < z <= 1.5 else ("near_ground" if -.45 <= z <= .20 else "other")
            hit, axes = grid_intersection((x, y, z))
            item = RayEvidence(source_stamp, transform_stamp, source_frame.lstrip("/"), "base", self._next_id, index,
                               (0.0, 0.0, 0.0), (x, y, z), distance, True, height, height == "occupied_eligible", True,
                               hit, bool(axes), tuple(axes), provenance, tuple())
            self._next_id += 1; result.append(item)
        return result

    def observe_sparse(self, points_base: np.ndarray, source_stamp: float, transform_stamp: float,
                       source_frame: str, provenance: str = "l1s_filtered_cloud") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Vectorized equivalent of observe(); stores only measured-ray geometry."""
        pts = np.asarray(points_base, dtype=np.float64).reshape((-1, 3))
        finite = np.isfinite(pts).all(axis=1); dist = np.linalg.norm(pts, axis=1)
        with np.errstate(invalid="ignore"):
            valid = finite & (dist >= .10) & (dist <= 8.0)
        indices = np.flatnonzero(valid)
        out = np.empty(len(indices), dtype=SPARSE_RAY_DTYPE)
        if not len(indices):
            return out, {"source_stamp": source_stamp, "transform_stamp": transform_stamp, "source_frame": source_frame.lstrip("/"), "target_frame": "base", "ray_origin_base": [0.0,0.0,0.0], "source_provenance": provenance, "valid_ray_count": 0}
        p, d = pts[indices], dist[indices]; x, y, z = p[:,0], p[:,1], p[:,2]
        inside = (x >= 0.0) & (x < X_MAX) & (y >= Y_MIN) & (y < Y_MAX)
        hi_x, hi_y = np.nextafter(X_MAX, -np.inf), np.nextafter(Y_MAX, -np.inf)
        tx = np.full(len(p), np.inf); ty = np.full(len(p), np.inf)
        np.divide(hi_x, x, out=tx, where=x > 0.0)
        np.divide(hi_y, y, out=ty, where=y > hi_y)
        np.divide(Y_MIN, y, out=ty, where=y < Y_MIN)
        t = np.minimum(tx, ty)
        intersection_valid = inside | ((x >= 0.0) & (t >= 0.0) & (t < 1.0))
        t = np.where(inside, 1.0, np.where(intersection_valid, t, 0.0))
        ix, iy = x * t, y * t
        clipped = ~inside
        axes = np.zeros(len(p), dtype=np.uint8)
        axes[clipped & intersection_valid & np.isclose(tx, t, rtol=0.0, atol=1e-12)] |= AXIS_X
        axes[clipped & intersection_valid & np.isclose(ty, t, rtol=0.0, atol=1e-12)] |= AXIS_Y
        axes[clipped & ~intersection_valid & (x < 0.0)] |= AXIS_X
        height = np.full(len(p), HEIGHT_OTHER, dtype=np.uint8); height[(-.45 <= z) & (z <= .20)] = HEIGHT_NEAR_GROUND; height[(.15 < z) & (z <= 1.5)] = HEIGHT_OCCUPIED
        out["ray_id"] = np.arange(self._next_id, self._next_id + len(p)); self._next_id += len(p)
        out["point_index"] = indices; out["side"] = (y < 0).astype(np.uint8); out["endpoint_xyz"] = p; out["measured_range"] = d
        out["height_class"] = height; out["occupancy_eligible"] = (height == HEIGHT_OCCUPIED); out["grid_intersection_xy"] = np.column_stack((ix,iy))
        out["intersection_valid"] = intersection_valid.astype(np.uint8); out["clipped"] = clipped; out["clipped_axes"] = axes
        return out, {"source_stamp": source_stamp, "transform_stamp": transform_stamp, "source_frame": source_frame.lstrip("/"), "target_frame": "base", "ray_origin_base": [0.0,0.0,0.0], "source_provenance": provenance, "valid_ray_count": int(len(out))}


class NavGridCompat:
    """A0-compatible in-memory consumer; whole-ray rejection is deliberate."""
    def __init__(self) -> None:
        self.free = np.zeros((ROWS, COLS), dtype=np.float32)
        self.occ = np.zeros((ROWS, COLS), dtype=np.float32)
        self.traversed = np.zeros((ROWS, COLS), dtype=np.float32)
        self.history: deque[Tuple[float, float, float]] = deque(maxlen=300)
        source = local_to_cell(0.0, 0.0)
        # The finite 60x60 navigation grid lets us pay this bounded allocation
        # before callbacks begin, rather than growing a cache during a run.
        self._ray_path_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for row in range(ROWS):
            for col in range(COLS):
                cells = bresenham(source, (row, col))[:-1]
                self._ray_path_cache[(row, col)] = (
                    np.asarray([item[0] for item in cells], dtype=np.intp),
                    np.asarray([item[1] for item in cells], dtype=np.intp),
                )

    def add_odom(self, x: float, y: float, yaw: float) -> None:
        self.history.append((x, y, yaw)); c, s, prior = math.cos(-yaw), math.sin(-yaw), None
        for px, py, _ in self.history:
            item = local_to_cell(c*(px-x)-s*(py-y), s*(px-x)+c*(py-y))
            if item is None: continue
            if prior is None: self.traversed[item] += 1.0
            else:
                for p in bresenham(prior, item): self.traversed[p] += 1.0
            prior = item

    def add_rays(self, rays: Iterable[RayEvidence]) -> Dict[str, int]:
        accepted = rejected = 0
        for ray in rays:
            target = local_to_cell(ray.ray_endpoint_base[0], ray.ray_endpoint_base[1])
            if target is None:
                rejected += 1; continue
            for p in bresenham(local_to_cell(0.0, 0.0), target)[:-1]: self.free[p] += 1.0
            if ray.endpoint_occupancy_eligible: self.occ[target] += 1.0
            if ray.endpoint_height_class == "near_ground": self.free[target] += 1.0
            accepted += 1
            if accepted >= POINT_CAP: break
        return {"accepted_grid_endpoints": accepted, "rejected_outside_grid_before_cap": rejected}

    def add_sparse_rays(self, rays: np.ndarray) -> Dict[str, int]:
        """Exact A0 update with cached paths, preserving input/cap ordering."""
        accepted_indices: List[int] = []
        rejected = 0
        for index, ray in enumerate(rays):
            x, y, _ = map(float, ray["endpoint_xyz"])
            if local_to_cell(x, y) is None:
                rejected += 1
                continue
            accepted_indices.append(index)
            if len(accepted_indices) >= POINT_CAP:
                break
        accepted = len(accepted_indices)
        if not accepted:
            return {"accepted_grid_endpoints": accepted, "rejected_outside_grid_before_cap": rejected}
        selected = rays[np.asarray(accepted_indices, dtype=np.intp)]
        endpoints = selected["endpoint_xyz"]
        rows = np.floor(endpoints[:, 0] / RESOLUTION).astype(np.intp)
        cols = np.floor((endpoints[:, 1] - Y_MIN) / RESOLUTION).astype(np.intp)
        flat = rows * COLS + cols
        total = np.bincount(flat, minlength=ROWS * COLS)
        occupied = np.bincount(flat, weights=selected["occupancy_eligible"], minlength=ROWS * COLS)
        near_ground = np.bincount(flat, weights=(selected["height_class"] == HEIGHT_NEAR_GROUND), minlength=ROWS * COLS)
        for flat_index in np.flatnonzero(total):
            target = (int(flat_index // COLS), int(flat_index % COLS))
            path = self._ray_path_cache[target]
            if len(path[0]):
                self.free[path] += total[flat_index]
            self.occ[target] += occupied[flat_index]
            self.free[target] += near_ground[flat_index]
        return {"accepted_grid_endpoints": accepted, "rejected_outside_grid_before_cap": rejected}

    def labels(self) -> np.ndarray:
        has_free, has_occ = (self.free+self.traversed) > .5, self.occ > .5
        data = np.full((ROWS, COLS), -1, dtype=np.int8); data[has_free] = 0; data[has_occ] = 100
        for r in range(ROWS):
            for c in range(COLS):
                if (r+.5)*RESOLUTION <= .30 and abs(Y_MIN+(c+.5)*RESOLUTION) <= .35: data[r, c] = 0
        return data

    def publish_and_decay(self) -> np.ndarray:
        data = self.labels(); self.free *= DECAY; self.occ *= DECAY; self.traversed *= DECAY
        return data


class SideEvidence:
    """Bounded aggregate, deliberately without candidate/portal fields."""
    def summarize(self, rays: Iterable[RayEvidence], now_stamp: float) -> Dict[str, Any]:
        sides = {"left": {"source_count": 0, "outside_grid": 0, "occupancy_eligible": 0, "clipped_axes": set()},
                 "right": {"source_count": 0, "outside_grid": 0, "occupancy_eligible": 0, "clipped_axes": set()}}
        for ray in rays:
            side = "left" if ray.ray_endpoint_base[1] >= 0.0 else "right"
            row = sides[side]; row["source_count"] += 1; row["outside_grid"] += int(ray.clipped)
            row["occupancy_eligible"] += int(ray.endpoint_occupancy_eligible); row["clipped_axes"].update(ray.clipped_axes)
        return {"source_stamp": now_stamp, "target_frame": "base", "sides": {key: {**value, "clipped_axes": sorted(value["clipped_axes"]), "evidence_age_sec": 0.0} for key, value in sides.items()}, "portal_fields_present": False, "control_fields_present": False}
