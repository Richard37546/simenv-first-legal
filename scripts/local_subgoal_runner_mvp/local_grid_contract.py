"""Shared P2KG13A contract for the formal local OccupancyGrid interface.

The public representation is always standard ROS OccupancyGrid: matrix[y][x]
and data[x + width * y].  This module deliberately has no rospy dependency so
production code and offline fixtures execute the exact same checks.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple


GRID_CONTRACT_VERSION = "local_grid_navigation_contract_v1"
GRID_STATUS_SCHEMA_VERSION = "local_grid_status_v1"
FORMAL_GRID_FRAME = "base"
FREE = "FREE"
OCCUPIED = "OCCUPIED"
UNKNOWN = "UNKNOWN"
OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
INVALID = "INVALID"
LEGAL_VALUES = {-1, 0, 100}
STATIC_PLANNING_FOOTPRINT_RADIUS_M = 0.2641935843278561
POINT_PLANNING_WITH_OBSTACLE_INFLATION = "POINT_PLANNING_WITH_OBSTACLE_INFLATION"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _stamp_seconds(value: Any) -> Optional[float]:
    if _finite(value):
        return float(value)
    if hasattr(value, "to_sec"):
        try:
            stamp = float(value.to_sec())
            return stamp if math.isfinite(stamp) else None
        except Exception:
            return None
    return None


def _ros_float32_wire_value(value: Any) -> float:
    """Return the value as transmitted by ROS's float32 OccupancyGrid field."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def same_ros_float32(left: Any, right: Any) -> bool:
    """Compare a ROS float32 field at its wire representation, fail-closed."""
    if not (_finite(left) and _finite(right)):
        return False
    try:
        return _ros_float32_wire_value(left) == _ros_float32_wire_value(right)
    except (OverflowError, struct.error, ValueError):
        return False


def static_footprint_fits_map(metadata: Dict[str, Any], radius_m: float = STATIC_PLANNING_FOOTPRINT_RADIUS_M) -> bool:
    """Whether a base-origin static circular envelope is wholly inside the map."""
    resolution = metadata.get("resolution")
    origin_x = metadata.get("origin_x")
    origin_y = metadata.get("origin_y")
    width = metadata.get("width")
    height = metadata.get("height")
    if not (_finite(radius_m) and float(radius_m) > 0.0 and _finite(resolution) and float(resolution) > 0.0
            and _finite(origin_x) and _finite(origin_y) and isinstance(width, int) and isinstance(height, int)
            and not isinstance(width, bool) and not isinstance(height, bool) and width > 0 and height > 0):
        return False
    x_max = float(origin_x) + int(width) * float(resolution)
    y_max = float(origin_y) + int(height) * float(resolution)
    radius = float(radius_m)
    return float(origin_x) <= -radius and x_max >= radius and float(origin_y) <= -radius and y_max >= radius


def grid_metadata(grid: Any) -> Dict[str, Any]:
    info = getattr(grid, "info", None)
    origin = getattr(info, "origin", None)
    position = getattr(origin, "position", None)
    header = getattr(grid, "header", None)
    return {
        "frame_id": getattr(header, "frame_id", None),
        "content_stamp": _stamp_seconds(getattr(header, "stamp", None)),
        # rospy assigns Header.seq on publication.  It is transport diagnostic
        # data, not the L3V application-level content generation identity.
        "ros_transport_sequence_diagnostic": getattr(header, "seq", None),
        "origin_x": getattr(position, "x", None),
        "origin_y": getattr(position, "y", None),
        "resolution": getattr(info, "resolution", None),
        "width": getattr(info, "width", None),
        "height": getattr(info, "height", None),
    }


def in_bounds(x_index: Any, y_index: Any, width: Any, height: Any) -> bool:
    return (
        isinstance(x_index, int) and not isinstance(x_index, bool)
        and isinstance(y_index, int) and not isinstance(y_index, bool)
        and isinstance(width, int) and not isinstance(width, bool)
        and isinstance(height, int) and not isinstance(height, bool)
        and 0 <= x_index < width and 0 <= y_index < height
    )


def metric_to_cell(x: float, y: float, metadata: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Return `(x_index, y_index)`; lower bounds inclusive, upper exclusive."""
    resolution = metadata.get("resolution")
    origin_x = metadata.get("origin_x")
    origin_y = metadata.get("origin_y")
    width = metadata.get("width")
    height = metadata.get("height")
    if not (_finite(x) and _finite(y) and _finite(resolution) and float(resolution) > 0.0 and _finite(origin_x) and _finite(origin_y)):
        return None
    x_index = int(math.floor((float(x) - float(origin_x)) / float(resolution)))
    y_index = int(math.floor((float(y) - float(origin_y)) / float(resolution)))
    return (x_index, y_index) if in_bounds(x_index, y_index, width, height) else None


def cell_to_metric(x_index: int, y_index: int, metadata: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if not in_bounds(x_index, y_index, metadata.get("width"), metadata.get("height")):
        return None
    resolution = metadata.get("resolution")
    origin_x = metadata.get("origin_x")
    origin_y = metadata.get("origin_y")
    if not (_finite(resolution) and float(resolution) > 0.0 and _finite(origin_x) and _finite(origin_y)):
        return None
    return (
        float(origin_x) + (x_index + 0.5) * float(resolution),
        float(origin_y) + (y_index + 0.5) * float(resolution),
    )


def flatten_index(x_index: int, y_index: int, width: int, height: int) -> Optional[int]:
    return x_index + width * y_index if in_bounds(x_index, y_index, width, height) else None


def classify_cell(value: Any) -> str:
    if value == 0:
        return FREE
    if value == 100:
        return OCCUPIED
    if value == -1:
        return UNKNOWN
    return INVALID


def is_cell_traversable(value: Any) -> bool:
    return classify_cell(value) == FREE


def validate_grid_metadata(grid: Any, expected_frame: str = FORMAL_GRID_FRAME) -> List[str]:
    metadata = grid_metadata(grid)
    errors: List[str] = []
    if metadata["frame_id"] != expected_frame:
        errors.append("grid_frame_invalid")
    if not (_finite(metadata["resolution"]) and float(metadata["resolution"]) > 0.0):
        errors.append("grid_resolution_invalid")
    for key in ("width", "height"):
        value = metadata[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"grid_{key}_invalid")
    if not (_finite(metadata["origin_x"]) and _finite(metadata["origin_y"])):
        errors.append("grid_origin_invalid")
    if not _finite(metadata["content_stamp"]):
        errors.append("grid_content_stamp_invalid")
    data = getattr(grid, "data", None)
    if not isinstance(data, (list, tuple)) or not isinstance(metadata["width"], int) or not isinstance(metadata["height"], int):
        errors.append("grid_data_invalid")
    elif len(data) != metadata["width"] * metadata["height"]:
        errors.append("grid_data_length_invalid")
    elif any(classify_cell(value) == INVALID for value in data):
        errors.append("grid_cell_value_invalid")
    return errors


def grid_content_hash(grid: Any, producer_instance_id: str, content_generation_id: int, content_stamp: float) -> str:
    metadata = grid_metadata(grid)
    payload = {
        "producer_instance_id": producer_instance_id,
        "content_generation_id": content_generation_id,
        "grid_content_stamp": content_stamp,
        "frame_id": metadata["frame_id"],
        "origin": [metadata["origin_x"], metadata["origin_y"]],
        # MapMetaData.resolution is a ROS float32. Canonicalize before hashing
        # so the producer's Python value and received wire value bind exactly.
        "resolution": _ros_float32_wire_value(metadata["resolution"]),
        "width": metadata["width"],
        "height": metadata["height"],
        "data": list(getattr(grid, "data", [])),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_grid_status_content_binding(
    grid: Any, status: Any, expected_frame: str = FORMAL_GRID_FRAME
) -> List[str]:
    """Validate the exact grid/status content identity without navigation policy.

    Header.seq is deliberately excluded: rospy replaces it during publication.
    The status application generation remains part of the content hash, while
    the received grid binds to that status through its exact content stamp and
    a recomputed hash of the actual received cell payload.
    """
    errors = validate_grid_metadata(grid, expected_frame)
    if not isinstance(status, dict):
        return errors + ["status_invalid"]
    if status.get("contract_version") != GRID_CONTRACT_VERSION:
        errors.append("status_contract_version_invalid")
    if status.get("schema_version") != GRID_STATUS_SCHEMA_VERSION:
        errors.append("status_schema_version_invalid")
    producer = status.get("producer_instance_id")
    if not isinstance(producer, str) or not producer:
        errors.append("status_producer_instance_invalid")
    generation = status.get("content_generation_id")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        errors.append("status_content_generation_invalid")
    metadata = grid_metadata(grid)
    stamp = status.get("grid_content_stamp")
    if not _finite(stamp) or float(stamp) != float(metadata["content_stamp"]):
        errors.append("status_grid_content_stamp_mismatch")
    if not isinstance(producer, str) or not producer or not isinstance(generation, int) or isinstance(generation, bool) or generation < 1 or not _finite(stamp):
        errors.append("status_content_identity_invalid")
    else:
        try:
            expected_hash = grid_content_hash(grid, producer, generation, float(stamp))
            if status.get("grid_content_hash") != expected_hash:
                errors.append("status_grid_content_hash_mismatch")
        except Exception:
            errors.append("status_grid_content_hash_invalid")
    if status.get("tf_valid") is not True:
        errors.append("status_tf_invalid")
    if status.get("all_required_inputs_fresh") is not True:
        errors.append("status_inputs_not_fresh")
    return sorted(set(errors))


class ExactGridStatusPairCache:
    """Bounded receipt cache that returns only a formally identical Grid/Status.

    ROS delivers the two topics independently.  Receipt order is therefore not
    an identity contract: callers must select a pair by the formal content
    stamp and hash, never by callback timing.  The cache is intentionally
    small and has no rospy dependency so the production consumer and offline
    callback-order tests execute identical matching logic.
    """

    def __init__(self, maxlen: int = 8) -> None:
        if not isinstance(maxlen, int) or isinstance(maxlen, bool) or maxlen < 1:
            raise ValueError("maxlen must be a positive integer")
        self._lock = threading.Lock()
        self._grids = deque(maxlen=maxlen)
        self._statuses = deque(maxlen=maxlen)

    def add_grid(self, grid: Any, received_wall_sec: Optional[float] = None) -> None:
        with self._lock:
            self._grids.append((time.monotonic() if received_wall_sec is None else float(received_wall_sec), grid))

    def add_status(self, status: Dict[str, Any], received_wall_sec: Optional[float] = None) -> None:
        with self._lock:
            self._statuses.append((time.monotonic() if received_wall_sec is None else float(received_wall_sec), status))

    def latest_grid(self) -> Optional[Any]:
        with self._lock:
            return self._grids[-1][1] if self._grids else None

    def receipt_watermark(self) -> float:
        """Return a receipt boundary for an action's terminal-after evidence."""
        return time.monotonic()

    def matching_pair_record(
        self,
        now_wall_sec: Optional[float] = None,
        *,
        not_before_wall_sec: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return one recent exact pair plus receipt provenance.

        ``not_before_wall_sec`` is deliberately a receipt boundary, rather
        than a ROS header-stamp comparison: the terminal consumer needs both
        independently delivered topic messages to have arrived after its
        terminal event, while retaining the existing content-identity test.
        """
        now = time.monotonic() if now_wall_sec is None else float(now_wall_sec)
        boundary = None if not_before_wall_sec is None else float(not_before_wall_sec)
        with self._lock:
            grids = tuple(self._grids)
            statuses = tuple(self._statuses)
        for grid_received, grid in reversed(grids):
            for status_received, status in reversed(statuses):
                if boundary is not None and (
                    float(grid_received) < boundary or float(status_received) < boundary
                ):
                    continue
                window = status.get("input_freshness_window_sec") if isinstance(status, dict) else None
                if not _finite(window) or float(window) <= 0.0:
                    continue
                if now - float(grid_received) > float(window) or now - float(status_received) > float(window):
                    continue
                if not validate_grid_status_content_binding(grid, status):
                    return {
                        "grid": grid,
                        "status": status,
                        "grid_received_wall_sec": float(grid_received),
                        "status_received_wall_sec": float(status_received),
                    }
        return None

    def matching_pair(
        self,
        now_wall_sec: Optional[float] = None,
        *,
        not_before_wall_sec: Optional[float] = None,
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """Return one recent exact pair, or ``None`` while an identity is pending."""
        record = self.matching_pair_record(
            now_wall_sec,
            not_before_wall_sec=not_before_wall_sec,
        )
        if record is None:
            return None
        return record["grid"], record["status"]


def validate_grid_status_pair(grid: Any, status: Any, expected_frame: str = FORMAL_GRID_FRAME) -> List[str]:
    """Return all fail-closed reasons for a formal motion consumer."""
    errors = validate_grid_status_content_binding(grid, status, expected_frame)
    if not isinstance(status, dict):
        return errors
    metadata = grid_metadata(grid)
    for key in ("frame_id", "width", "height"):
        if status.get(key) != metadata[key]:
            errors.append(f"status_grid_{key}_mismatch")
    if not same_ros_float32(status.get("resolution"), metadata["resolution"]):
        errors.append("status_grid_resolution_mismatch")
    origin = status.get("origin")
    if not isinstance(origin, dict) or origin.get("x") != metadata["origin_x"] or origin.get("y") != metadata["origin_y"]:
        errors.append("status_grid_origin_mismatch")
    if status.get("input_time_monotonic") is not True:
        errors.append("status_input_time_invalid")
    if status.get("diagnostic_only") is not False:
        errors.append("status_diagnostic_only")
    if status.get("safe_for_navigation") is not True:
        errors.append("status_safe_for_navigation_false")
    if status.get("upstream_navigation_allowed") is not True:
        errors.append("status_upstream_navigation_not_allowed")
    rejection_reasons = status.get("rejection_reasons")
    if not isinstance(rejection_reasons, list):
        errors.append("status_rejection_reasons_invalid")
    elif rejection_reasons:
        errors.append("status_rejection_reasons_present")
    return sorted(set(errors))


def qualified_for_navigation(grid: Any, status: Any, expected_frame: str = FORMAL_GRID_FRAME) -> Tuple[bool, List[str]]:
    reasons = validate_grid_status_pair(grid, status, expected_frame)
    return not reasons, reasons
