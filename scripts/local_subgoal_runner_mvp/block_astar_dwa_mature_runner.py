#!/usr/bin/env python3
"""Matured Block A* + DWA local runner for more_road.

This script is a replacement experiment for local_subgoal_runner.py. It keeps
the same adapted inputs:
- /team/livox/icp_odom_gated
- /team/local_traversability_grid
- /team/traversability_status
- debug/short_horizon_target_selection/short_horizon_target_override.json

It publishes the configured cmd topic only when --execute is supplied.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Imu, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String
import tf

from inflation_geometry import occupied_euclidean_inflated_mask
from corridor_axis_evidence import fit_wall_line as fit_corridor_wall_line
from local_grid_contract import (
    cell_to_metric,
    flatten_index,
    grid_metadata,
    is_cell_traversable,
    metric_to_cell,
    POINT_PLANNING_WITH_OBSTACLE_INFLATION,
    ExactGridStatusPairCache,
    qualified_for_navigation,
    STATIC_PLANNING_FOOTPRINT_RADIUS_M,
    validate_grid_status_content_binding,
    validate_grid_metadata,
)
from source_time_odom_cache import OdomCache


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "debug" / "block_astar_dwa_mature"
TARGET_PATH = ROOT / "debug" / "short_horizon_target_selection" / "short_horizon_target_override.json"

TOPIC_ODOM = "/team/livox/icp_odom_gated"
TOPIC_GRID = "/team/local_traversability_grid"
TOPIC_STATUS = "/team/traversability_status"
TOPIC_CMD = "/cmd_vel"
TOPIC_LIDAR = "/team/livox/scan_cloud_filtered"
TOPIC_TRUNK_IMU = "/trunk_imu"
TOPIC_ROOM_LOCAL_VALIDATION_EVENT = "/audit/room_local_validation/event"
TOPIC_ROOM_LOCAL_VALIDATION_ABORT = "/audit/room_local_validation/abort"
ROOM_ENTRY_MIN_TARGET_X_BASE_M = 0.15
MIN_EXECUTABLE_FORWARD_MPS = 0.30
DWA_NUMERIC_EPS = 1e-9
LOCAL_CONTROL_MODE_TRANSIT = "TRANSIT"
LOCAL_CONTROL_MODE_ROOM_LOCAL = "ROOM_LOCAL"
LOCAL_CONTROL_MODE_CHOICES = (
    LOCAL_CONTROL_MODE_TRANSIT,
    LOCAL_CONTROL_MODE_ROOM_LOCAL,
)
ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_DISABLED = "disabled"
ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_OFFLINE_FROZEN = "offline_frozen"
ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_PHASE3_GUARDED_EXECUTE = "phase3_guarded_execute"
ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_CHOICES = (
    ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_DISABLED,
    ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_OFFLINE_FROZEN,
    ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_PHASE3_GUARDED_EXECUTE,
)
ROOM_LOCAL_PHASE3_RECOVERY_DISABLED = "disabled"
ROOM_LOCAL_PHASE3_RECOVERY_OFFLINE_FROZEN = "offline_frozen"
ROOM_LOCAL_PHASE3_RECOVERY_GUARDED_EXECUTE = "phase3_guarded_execute"
ROOM_LOCAL_PHASE3_RECOVERY_CHOICES = (
    ROOM_LOCAL_PHASE3_RECOVERY_DISABLED,
    ROOM_LOCAL_PHASE3_RECOVERY_OFFLINE_FROZEN,
    ROOM_LOCAL_PHASE3_RECOVERY_GUARDED_EXECUTE,
)
CONTINUATION_LOCAL_SELECTION_AUTHORITY_DISABLED = "disabled"
CONTINUATION_LOCAL_SELECTION_AUTHORITY_GUARDED_EXECUTE = "guarded_execute"
CONTINUATION_LOCAL_SELECTION_AUTHORITY_CHOICES = (
    CONTINUATION_LOCAL_SELECTION_AUTHORITY_DISABLED,
    CONTINUATION_LOCAL_SELECTION_AUTHORITY_GUARDED_EXECUTE,
)
ROOM_LOCAL_CAPABILITY_VERSION = "room_local_phase1_v1"
# P_through remains an obstacle-safe local planner.  In the late approach to
# a portal it may, however, choose between several similarly inward-facing
# safe arcs.  These constants make a small portal-tangent reduction a
# tie-break only; they never reject an otherwise safe arc or require exact
# portal centring.
P_THROUGH_LATE_APPROACH_TANGENT_TRIGGER_M = 0.65
# The run_0110 final safe alternatives differed from the heading-only winner
# by 0.20--0.25 rad while reducing the predicted jamb-side offset by 6--9 cm.
# Keep that bounded trade-off available only in the late approach.
P_THROUGH_TANGENT_YAW_SLACK_RAD = 0.30
P_THROUGH_TANGENT_MIN_IMPROVEMENT_M = 0.01
ROOM_ENTRY_SUBGOAL_SOURCES = {
    "room_side_gap_entry",
    "room_side_gap_partial_turn_entry",
}
ROOM_ENTRY_TARGET_SOURCES = {
    "state_machine_room_side_gap_entry",
    "state_machine_room_side_gap_partial_turn_direct_entry",
}


@dataclass(frozen=True)
class RoomLocalCapability:
    """Frozen Phase-1 authority for a future ROOM_LOCAL primitive family.

    This table is intentionally not consulted by the live DWA winner path in
    Phase 1.  Its bounds only restate the currently frozen ROOM_LOCAL profile;
    it neither creates low-speed, reverse, nor higher-angular authority.
    """

    capability_id: str
    motion_class: str
    v_min_mps: float
    v_max_mps: float
    w_min_radps: float
    w_max_radps: float
    qualification: str
    evidence_source: str
    version: str = ROOM_LOCAL_CAPABILITY_VERSION

    def contains(self, v_mps: float, w_radps: float) -> bool:
        return (
            finite_number(v_mps)
            and finite_number(w_radps)
            and self.qualification == "EVIDENCE_QUALIFIED"
            and float(self.v_min_mps) - DWA_NUMERIC_EPS <= float(v_mps) <= float(self.v_max_mps) + DWA_NUMERIC_EPS
            and float(self.w_min_radps) - DWA_NUMERIC_EPS <= float(w_radps) <= float(self.w_max_radps) + DWA_NUMERIC_EPS
        )


@dataclass(frozen=True)
class RoomLocalPrimitive:
    """Pure Phase-1 representation; it has no winner authority yet."""

    v_mps: float
    w_radps: float
    kappa_m_inv: float
    capability_id: str


@dataclass(frozen=True)
class PathProgressAnchor:
    """One decision-slice anchor for an ordered robot-centred local path."""

    path_identity: str
    anchor_segment_index: int
    local_end_segment_index: int
    stations_m: Tuple[float, ...]
    s_current_m: float
    current_cross_track_m: float
    projection_tolerance_m: float


@dataclass(frozen=True)
class PathStationProjection:
    valid: bool
    station_m: Optional[float]
    cross_track_m: Optional[float]
    reason: str


@dataclass(frozen=True)
class ProgressEvidence:
    """Candidate-local evidence only; Phase 1 does not apply it to scoring."""

    p_path_m: Optional[float]
    p_target_m: float
    p_local_m: float
    projected_station_m: Optional[float]
    endpoint_target_distance_m: float
    endpoint_local_distance_m: float
    cross_track_m: Optional[float]
    projection_reason: str


@dataclass(frozen=True)
class RoomLocalProductivityAdmission:
    """Pure Phase-2 eligibility result for one collision-safe translation."""

    classification: str
    reason: str
    score_eligible: bool
    meaningful_progress_margin_m: float


@dataclass(frozen=True)
class FrozenRoomLocalEpoch:
    """Immutable, ROS-free input contract for one ROOM_LOCAL evaluation epoch.

    The live runner owns acquisition.  This value owns only JSON-safe snapshots
    of the acquired Grid/status, pose, runner arguments, previous command and
    optional wall-heading prior.  A pure evaluator rehydrates private plain
    objects from it and therefore cannot read ROS or mutate runner state.
    """

    epoch_id: str
    pose_odom_xy_yaw: Tuple[float, float, float]
    grid_header_stamp_sec: float
    grid_frame_id: str
    grid_resolution_m: float
    grid_width: int
    grid_height: int
    grid_origin_xy: Tuple[float, float]
    grid_data: Tuple[int, ...]
    status_json: str
    args_json: str
    previous_cmd: Tuple[float, float]
    wall_heading_prior_json: str

    @classmethod
    def from_live_inputs(
        cls,
        *,
        epoch_id: str,
        pose_odom_xy_yaw: Sequence[float],
        grid_msg: OccupancyGrid,
        status_payload: Dict[str, Any],
        args: argparse.Namespace,
        previous_cmd: Sequence[float],
        wall_heading_prior: Optional[Dict[str, Any]] = None,
    ) -> "FrozenRoomLocalEpoch":
        if len(pose_odom_xy_yaw) != 3 or len(previous_cmd) != 2:
            raise ValueError("frozen_room_local_epoch_shape_invalid")
        info = grid_msg.info
        if len(grid_msg.data) != int(info.width) * int(info.height):
            raise ValueError("frozen_room_local_epoch_grid_shape_invalid")
        stamp = getattr(getattr(grid_msg, "header", None), "stamp", None)
        stamp_sec = float(stamp.to_sec()) if hasattr(stamp, "to_sec") else float(stamp or 0.0)
        args_payload = vars(copy.deepcopy(args))
        return cls(
            epoch_id=str(epoch_id),
            pose_odom_xy_yaw=tuple(float(value) for value in pose_odom_xy_yaw),
            grid_header_stamp_sec=stamp_sec,
            grid_frame_id=str(getattr(getattr(grid_msg, "header", None), "frame_id", "")),
            grid_resolution_m=float(info.resolution),
            grid_width=int(info.width), grid_height=int(info.height),
            grid_origin_xy=(float(info.origin.position.x), float(info.origin.position.y)),
            grid_data=tuple(int(value) for value in grid_msg.data),
            status_json=json.dumps(copy.deepcopy(status_payload), sort_keys=True, allow_nan=False),
            args_json=json.dumps(args_payload, sort_keys=True, allow_nan=False),
            previous_cmd=(float(previous_cmd[0]), float(previous_cmd[1])),
            wall_heading_prior_json=json.dumps(copy.deepcopy(wall_heading_prior), sort_keys=True, allow_nan=False),
        )

    def grid_message(self) -> Any:
        """Return a private plain message-shaped value; it is not a ROS read."""
        stamp_sec = float(self.grid_header_stamp_sec)
        return SimpleNamespace(
            header=SimpleNamespace(frame_id=self.grid_frame_id, stamp=SimpleNamespace(to_sec=lambda: stamp_sec)),
            info=SimpleNamespace(
                resolution=float(self.grid_resolution_m), width=int(self.grid_width), height=int(self.grid_height),
                origin=SimpleNamespace(
                    position=SimpleNamespace(x=float(self.grid_origin_xy[0]), y=float(self.grid_origin_xy[1]), z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ),
            data=list(self.grid_data),
        )

    def status_payload(self) -> Dict[str, Any]:
        return json.loads(self.status_json)

    def runner_args(self) -> argparse.Namespace:
        args = argparse.Namespace(**json.loads(self.args_json))
        args.execute = False
        # A frozen evaluator is evidence-only even when the live caller has
        # explicitly enabled production local-selection authority.
        args.continuation_local_selection_authority = CONTINUATION_LOCAL_SELECTION_AUTHORITY_DISABLED
        return args

    def wall_heading_prior(self) -> Optional[Dict[str, Any]]:
        value = json.loads(self.wall_heading_prior_json)
        return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class FrozenRoomLocalCandidate:
    """Candidate-local input; rank metadata is evidence, never physics input."""

    candidate_id: str
    rank: int
    sector_id: Optional[int]
    target_odom_xy: Tuple[float, float]
    heading_rad: Optional[float]
    candidate_type: Optional[str]
    cheap_rank_metadata_json: str

    @classmethod
    def from_mapping(cls, candidate: Dict[str, Any]) -> "FrozenRoomLocalCandidate":
        target = candidate.get("target_odom_xy") or candidate.get("target_xy_team_livox_odom")
        if not isinstance(target, (list, tuple)) or len(target) != 2 or not all(finite_number(v) for v in target):
            raise ValueError("frozen_room_local_candidate_target_invalid")
        return cls(
            candidate_id=str(candidate.get("candidate_id") or candidate.get("_room_search_audit_candidate_id") or ""),
            rank=int(candidate.get("rank") or candidate.get("cheap_rank") or 0),
            sector_id=(None if candidate.get("sector") is None else int(candidate.get("sector"))),
            target_odom_xy=(float(target[0]), float(target[1])),
            heading_rad=(None if candidate.get("heading_rad") is None and candidate.get("heading_change_rad") is None
                         else float(candidate.get("heading_rad", candidate.get("heading_change_rad")))),
            candidate_type=(None if candidate.get("candidate_type") is None and candidate.get("target_priority_class") is None
                            else str(candidate.get("candidate_type", candidate.get("target_priority_class")))),
            cheap_rank_metadata_json=json.dumps(copy.deepcopy(candidate.get("cheap_rank_components") or {}), sort_keys=True, allow_nan=False),
        )


@dataclass(frozen=True)
class CircularYawInterval:
    center_rad: float
    half_width_rad: float


@dataclass(frozen=True)
class RecoverySet:
    intervals: Tuple[CircularYawInterval, ...]
    angular_margin_rad: Optional[float]
    hypothesis_records: Tuple[Dict[str, Any], ...]
    approximate: bool = True


@dataclass
class OrientationIntent:
    target_key: Tuple[Any, ...]
    anchor_odom_xy: Tuple[float, float]
    anchor_tangent_odom_rad: Optional[float]
    entry_yaw_rad: float
    recovery_intervals: Tuple[CircularYawInterval, ...]
    chosen_direction: int
    recovery_distance_before_rad: float
    last_recovery_distance_rad: float
    consumed: bool = False
    awaiting_fresh_replan: bool = False


# Frozen from the current ROOM_SEARCH/ROOM_RETURN runner profile.  This is an
# allow-list foundation for later phases, not a new command envelope.
ROOM_LOCAL_CAPABILITIES = (
    RoomLocalCapability(
        capability_id="forward_turn_current_room_profile",
        motion_class="FORWARD_TURN",
        v_min_mps=MIN_EXECUTABLE_FORWARD_MPS,
        v_max_mps=0.60,
        w_min_radps=-0.35,
        w_max_radps=0.35,
        qualification="EVIDENCE_QUALIFIED",
        evidence_source="RUN0150_CURRENT_ROOM_PROFILE_FROZEN",
    ),
)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def below_minimum_forward(v: float, minimum_forward: float) -> bool:
    """Treat only numerical round-off around the documented forward boundary as equal."""
    return float(v) > DWA_NUMERIC_EPS and float(v) < float(minimum_forward) - DWA_NUMERIC_EPS


def exceeds_speed_limit(v: float, speed_limit: float) -> bool:
    """Keep the existing limit while avoiding a round-off-only rejection at its boundary."""
    return float(v) > float(speed_limit) + DWA_NUMERIC_EPS


def meets_minimum_forward(v: float, minimum_forward: float) -> bool:
    return float(v) >= float(minimum_forward) - DWA_NUMERIC_EPS


def effective_minimum_forward(min_linear_x: float) -> float:
    """Return the frozen minimum executable forward-speed contract."""
    return max(MIN_EXECUTABLE_FORWARD_MPS, float(min_linear_x))


def speed_limit_for_distance(max_linear_x: float, minimum_forward: float, distance_to_goal: float, distance_speed_gain: float) -> float:
    """Apply the existing distance-based speed cap without changing its policy."""
    return min(
        float(max_linear_x),
        max(float(minimum_forward), float(distance_to_goal) * float(distance_speed_gain)),
    )


def v_samples_with_legal_boundaries(
    v_samples: Sequence[float],
    minimum_forward: float,
    speed_limit: float,
) -> np.ndarray:
    """Preserve uniform DWA samples and add only executable interval boundaries.

    The dynamic-window samples remain authoritative for the window itself.  This
    helper only adds an endpoint when the intersection of that existing window
    with the already-existing minimum/speed gates is non-empty.
    """
    raw = sorted(float(v) for v in v_samples if finite_number(v))
    if not raw:
        return np.array([], dtype=float)
    dynamic_v_min, dynamic_v_max = raw[0], raw[-1]
    legal_v_low = max(dynamic_v_min, float(minimum_forward))
    legal_v_high = min(dynamic_v_max, float(speed_limit))
    merged = list(raw)
    if legal_v_low <= legal_v_high + DWA_NUMERIC_EPS:
        # In the epsilon-sized numerical-boundary case, the lower value is
        # within the existing gates and is the sole deterministic representative.
        boundaries = [legal_v_low] if legal_v_low > legal_v_high else [legal_v_low, legal_v_high]
        merged.extend(boundaries)
    deduplicated: List[float] = []
    for value in sorted(merged):
        if not deduplicated or abs(value - deduplicated[-1]) > DWA_NUMERIC_EPS:
            deduplicated.append(value)
    return np.array(deduplicated, dtype=float)


def room_local_capability_by_id(capability_id: str) -> Optional[RoomLocalCapability]:
    """Return a Phase-1 frozen capability entry without changing live DWA."""
    return next((entry for entry in ROOM_LOCAL_CAPABILITIES if entry.capability_id == str(capability_id)), None)


def room_local_curvature_primitive(
    v_mps: float,
    kappa_m_inv: float,
    capability: RoomLocalCapability,
    dynamic_v_bounds: Tuple[float, float],
    dynamic_w_bounds: Tuple[float, float],
) -> Tuple[Optional[RoomLocalPrimitive], str]:
    """Map a ROOM_LOCAL ``(v, kappa)`` request into a legal future primitive.

    The helper is deliberately pure and inactive in Phase 1: it verifies an
    already-authorized capability and the existing dynamic window, but it does
    not generate, filter, score, or publish any live DWA candidate.
    """
    if not (finite_number(v_mps) and finite_number(kappa_m_inv)):
        return None, "NONFINITE_CURVATURE_REQUEST"
    if float(v_mps) <= DWA_NUMERIC_EPS:
        return None, "NONPOSITIVE_TRANSLATION"
    if not all(finite_number(value) for value in (*dynamic_v_bounds, *dynamic_w_bounds)):
        return None, "INVALID_DYNAMIC_WINDOW"
    v_low, v_high = sorted((float(dynamic_v_bounds[0]), float(dynamic_v_bounds[1])))
    w_low, w_high = sorted((float(dynamic_w_bounds[0]), float(dynamic_w_bounds[1])))
    w_radps = float(v_mps) * float(kappa_m_inv)
    if not capability.contains(float(v_mps), w_radps):
        return None, "CAPABILITY_UNQUALIFIED"
    if not (v_low - DWA_NUMERIC_EPS <= float(v_mps) <= v_high + DWA_NUMERIC_EPS):
        return None, "DYNAMIC_V_WINDOW_REJECTED"
    if not (w_low - DWA_NUMERIC_EPS <= w_radps <= w_high + DWA_NUMERIC_EPS):
        return None, "DYNAMIC_W_WINDOW_REJECTED"
    return RoomLocalPrimitive(
        v_mps=float(v_mps),
        w_radps=w_radps,
        kappa_m_inv=float(kappa_m_inv),
        capability_id=capability.capability_id,
    ), "QUALIFIED"


def derive_progress_projection_tolerance(
    grid_resolution_m: float,
    path_sampling_resolution_m: float,
    rollout_step_m: float,
) -> Optional[float]:
    """Derive a physical projection allowance without weakening arc reachability.

    Path sampling is validated here because it determines ordered-segment
    structure, but a coarse path segment is not endpoint uncertainty and must
    not enlarge the candidate's physically reachable station bound.
    """
    if not all(finite_number(value) and float(value) >= 0.0 for value in (
        grid_resolution_m, path_sampling_resolution_m, rollout_step_m,
    )):
        return None
    grid_cell_diagonal_m = math.sqrt(2.0) * float(grid_resolution_m)
    return max(grid_cell_diagonal_m, float(rollout_step_m), DWA_NUMERIC_EPS)


def derive_meaningful_progress_margin(
    grid_resolution_m: float,
    minimum_forward_mps: float,
    dwa_dt_sec: float,
) -> Optional[float]:
    """Derive Phase-2 progress significance from existing spatial quantities."""
    if not all(finite_number(value) and float(value) > 0.0 for value in (
        grid_resolution_m, minimum_forward_mps, dwa_dt_sec,
    )):
        return None
    # This is half of the coarser current grid sample or minimum-speed rollout
    # integration step, not a hand-picked numerical epsilon.
    return 0.5 * max(float(grid_resolution_m), float(minimum_forward_mps) * float(dwa_dt_sec))


def _path_stations(path_xy: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, ...]]:
    if len(path_xy) < 2:
        return None
    stations = [0.0]
    for first, second in zip(path_xy, path_xy[1:]):
        if not all(finite_number(value) for value in (*first, *second)):
            return None
        segment_length = math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))
        if segment_length <= DWA_NUMERIC_EPS:
            return None
        stations.append(stations[-1] + segment_length)
    return tuple(stations)


def _project_point_to_segment(
    point_xy: Tuple[float, float],
    first_xy: Tuple[float, float],
    second_xy: Tuple[float, float],
) -> Tuple[float, float]:
    """Return clamped segment fraction and cross-track distance."""
    dx = float(second_xy[0]) - float(first_xy[0])
    dy = float(second_xy[1]) - float(first_xy[1])
    length_sq = dx * dx + dy * dy
    if length_sq <= DWA_NUMERIC_EPS:
        return 0.0, float("inf")
    fraction = ((float(point_xy[0]) - float(first_xy[0])) * dx + (float(point_xy[1]) - float(first_xy[1])) * dy) / length_sq
    fraction = max(0.0, min(1.0, fraction))
    projected_x = float(first_xy[0]) + fraction * dx
    projected_y = float(first_xy[1]) + fraction * dy
    return fraction, math.hypot(float(point_xy[0]) - projected_x, float(point_xy[1]) - projected_y)


def build_path_progress_anchor(
    path_xy: Sequence[Tuple[float, float]],
    *,
    path_identity: str,
    robot_xy: Tuple[float, float] = (0.0, 0.0),
    anchor_segment_index: int = 0,
    local_window_arc_m: Optional[float] = None,
    grid_resolution_m: float,
    rollout_step_m: float,
) -> Optional[PathProgressAnchor]:
    """Anchor the current robot to an ordered, contiguous local path window.

    The caller supplies the current local path after ``smooth_path()``.  The
    function deliberately never performs a global nearest-segment search.
    """
    stations = _path_stations(path_xy)
    if stations is None or not all(finite_number(value) for value in robot_xy):
        return None
    segment_count = len(path_xy) - 1
    if anchor_segment_index < 0 or anchor_segment_index >= segment_count:
        return None
    sampling_resolution_m = max(
        stations[index + 1] - stations[index] for index in range(segment_count)
    )
    tolerance_m = derive_progress_projection_tolerance(
        grid_resolution_m, sampling_resolution_m, rollout_step_m,
    )
    if tolerance_m is None:
        return None
    fraction, cross_track_m = _project_point_to_segment(
        robot_xy,
        path_xy[anchor_segment_index],
        path_xy[anchor_segment_index + 1],
    )
    s_current_m = stations[anchor_segment_index] + fraction * (
        stations[anchor_segment_index + 1] - stations[anchor_segment_index]
    )
    if local_window_arc_m is None:
        local_end_segment_index = segment_count - 1
    elif not finite_number(local_window_arc_m) or float(local_window_arc_m) < 0.0:
        return None
    else:
        station_limit = s_current_m + float(local_window_arc_m) + tolerance_m
        local_end_segment_index = anchor_segment_index
        while (
            local_end_segment_index + 1 < segment_count
            and stations[local_end_segment_index + 1] < station_limit
        ):
            local_end_segment_index += 1
    return PathProgressAnchor(
        path_identity=str(path_identity),
        anchor_segment_index=int(anchor_segment_index),
        local_end_segment_index=int(local_end_segment_index),
        stations_m=stations,
        s_current_m=float(s_current_m),
        current_cross_track_m=float(cross_track_m),
        projection_tolerance_m=float(tolerance_m),
    )


def project_endpoint_to_path_station(
    path_xy: Sequence[Tuple[float, float]],
    anchor: PathProgressAnchor,
    endpoint_xy: Tuple[float, float],
    *,
    rollout_arc_length_m: float,
) -> PathStationProjection:
    """Project only onto the anchor's ordered, arc-reachable path window."""
    if not all(finite_number(value) for value in (*endpoint_xy, rollout_arc_length_m)):
        return PathStationProjection(False, None, None, "INVALID_PROJECTION_INPUT")
    if float(rollout_arc_length_m) < 0.0 or len(path_xy) != len(anchor.stations_m):
        return PathStationProjection(False, None, None, "INVALID_PATH_OR_ARC_LENGTH")
    lower_station = anchor.s_current_m - anchor.projection_tolerance_m
    upper_station = min(
        anchor.stations_m[-1],
        anchor.s_current_m + float(rollout_arc_length_m) + anchor.projection_tolerance_m,
    )
    candidates: List[Tuple[float, float]] = []
    first_segment = max(0, anchor.anchor_segment_index - 1)
    for index in range(first_segment, anchor.local_end_segment_index + 1):
        segment_start = anchor.stations_m[index]
        segment_end = anchor.stations_m[index + 1]
        if segment_end < lower_station - DWA_NUMERIC_EPS or segment_start > upper_station + DWA_NUMERIC_EPS:
            continue
        fraction, cross_track_m = _project_point_to_segment(endpoint_xy, path_xy[index], path_xy[index + 1])
        station_m = segment_start + fraction * (segment_end - segment_start)
        if lower_station - DWA_NUMERIC_EPS <= station_m <= upper_station + DWA_NUMERIC_EPS:
            candidates.append((cross_track_m, station_m))
    if not candidates:
        return PathStationProjection(False, None, None, "NO_ORDERED_ARC_REACHABLE_SEGMENT")
    cross_track_m, station_m = min(candidates, key=lambda item: (item[0], item[1]))
    if cross_track_m > anchor.current_cross_track_m + anchor.projection_tolerance_m + DWA_NUMERIC_EPS:
        return PathStationProjection(False, None, cross_track_m, "CROSS_TRACK_INCONSISTENT")
    return PathStationProjection(True, station_m, cross_track_m, "ORDERED_ARC_REACHABLE")


def evaluate_candidate_progress(
    anchor: PathProgressAnchor,
    projection: PathStationProjection,
    *,
    current_target_xy: Tuple[float, float],
    current_local_xy: Tuple[float, float],
    endpoint_xy: Tuple[float, float],
) -> ProgressEvidence:
    """Compute Phase-1 candidate-local progress without winner authority."""
    current_target_distance_m = math.hypot(float(current_target_xy[0]), float(current_target_xy[1]))
    endpoint_target_distance_m = math.hypot(
        float(current_target_xy[0]) - float(endpoint_xy[0]),
        float(current_target_xy[1]) - float(endpoint_xy[1]),
    )
    current_local_distance_m = math.hypot(float(current_local_xy[0]), float(current_local_xy[1]))
    endpoint_local_distance_m = math.hypot(
        float(current_local_xy[0]) - float(endpoint_xy[0]),
        float(current_local_xy[1]) - float(endpoint_xy[1]),
    )
    return ProgressEvidence(
        p_path_m=(float(projection.station_m) - anchor.s_current_m) if projection.valid and projection.station_m is not None else None,
        p_target_m=current_target_distance_m - endpoint_target_distance_m,
        p_local_m=current_local_distance_m - endpoint_local_distance_m,
        projected_station_m=projection.station_m if projection.valid else None,
        endpoint_target_distance_m=endpoint_target_distance_m,
        endpoint_local_distance_m=endpoint_local_distance_m,
        cross_track_m=projection.cross_track_m,
        projection_reason=projection.reason,
    )


def evaluate_candidate_direct_progress(
    *,
    current_target_xy: Tuple[float, float],
    current_local_xy: Tuple[float, float],
    endpoint_xy: Tuple[float, float],
    projection_reason: str = "PATH_CONTEXT_UNAVAILABLE",
) -> ProgressEvidence:
    """Retain direct target/local evidence when no ordered path is available."""
    current_target_distance_m = math.hypot(float(current_target_xy[0]), float(current_target_xy[1]))
    endpoint_target_distance_m = math.hypot(
        float(current_target_xy[0]) - float(endpoint_xy[0]),
        float(current_target_xy[1]) - float(endpoint_xy[1]),
    )
    current_local_distance_m = math.hypot(float(current_local_xy[0]), float(current_local_xy[1]))
    endpoint_local_distance_m = math.hypot(
        float(current_local_xy[0]) - float(endpoint_xy[0]),
        float(current_local_xy[1]) - float(endpoint_xy[1]),
    )
    return ProgressEvidence(
        p_path_m=None,
        p_target_m=current_target_distance_m - endpoint_target_distance_m,
        p_local_m=current_local_distance_m - endpoint_local_distance_m,
        projected_station_m=None,
        endpoint_target_distance_m=endpoint_target_distance_m,
        endpoint_local_distance_m=endpoint_local_distance_m,
        cross_track_m=None,
        projection_reason=str(projection_reason),
    )


def classify_room_local_productivity(
    evidence: ProgressEvidence,
    *,
    direct_or_short_context: bool,
    meaningful_progress_margin_m: float,
) -> RoomLocalProductivityAdmission:
    """Classify candidate productivity before the unchanged preference score."""
    if not finite_number(meaningful_progress_margin_m) or float(meaningful_progress_margin_m) <= 0.0:
        return RoomLocalProductivityAdmission(
            "INSUFFICIENT_PROGRESS_EVIDENCE", "INVALID_PROGRESS_MARGIN", False, 0.0,
        )
    margin = float(meaningful_progress_margin_m)
    if (
        evidence.projection_reason == "ORDERED_ARC_REACHABLE"
        and evidence.p_path_m is not None
        and float(evidence.p_path_m) > margin
    ):
        return RoomLocalProductivityAdmission(
            "PATH_PRODUCTIVE", "VALID_ORDERED_PATH_PROGRESS", True, margin,
        )
    if bool(direct_or_short_context) and (
        (float(evidence.p_target_m) > margin and float(evidence.p_local_m) >= -margin)
        or (float(evidence.p_local_m) > margin and float(evidence.p_target_m) >= -margin)
    ):
        return RoomLocalProductivityAdmission(
            "DIRECT_OR_SHORT_PRODUCTIVE", "DIRECT_TARGET_OR_LOCAL_PROGRESS", True, margin,
        )
    if (
        (evidence.p_path_m is None or float(evidence.p_path_m) <= margin)
        and float(evidence.p_target_m) <= margin
        and float(evidence.p_local_m) <= margin
    ):
        return RoomLocalProductivityAdmission(
            "SAFE_BUT_NONPRODUCTIVE_TRANSLATION", "NO_MEANINGFUL_PATH_TARGET_OR_LOCAL_PROGRESS", False, margin,
        )
    return RoomLocalProductivityAdmission(
        "INSUFFICIENT_PROGRESS_EVIDENCE", "AMBIGUOUS_OR_NONAUTHORITATIVE_PROGRESS", False, margin,
    )


def phase2_productive_admission_is_active(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "local_control_mode", LOCAL_CONTROL_MODE_TRANSIT)) == LOCAL_CONTROL_MODE_ROOM_LOCAL
        and str(getattr(args, "room_local_phase2_productive_admission", ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_DISABLED))
        in {
            ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_OFFLINE_FROZEN,
            ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_PHASE3_GUARDED_EXECUTE,
        }
    )


def phase3_orientation_recovery_is_active(args: argparse.Namespace) -> bool:
    return (
        phase2_productive_admission_is_active(args)
        and str(getattr(args, "room_local_phase3_orientation_recovery", ROOM_LOCAL_PHASE3_RECOVERY_DISABLED))
        in {
            ROOM_LOCAL_PHASE3_RECOVERY_OFFLINE_FROZEN,
            ROOM_LOCAL_PHASE3_RECOVERY_GUARDED_EXECUTE,
        }
    )


def continuation_local_selection_authority_is_active(args: argparse.Namespace) -> bool:
    """Return the explicit guarded production authority, never the C0 shadow."""
    return (
        bool(getattr(args, "execute", False))
        and str(getattr(args, "local_control_mode", LOCAL_CONTROL_MODE_TRANSIT)) == LOCAL_CONTROL_MODE_ROOM_LOCAL
        and str(getattr(args, "room_local_phase2_productive_admission", ""))
        == ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_PHASE3_GUARDED_EXECUTE
        and str(getattr(args, "room_local_phase3_orientation_recovery", ""))
        == ROOM_LOCAL_PHASE3_RECOVERY_GUARDED_EXECUTE
        and str(getattr(args, "continuation_local_selection_authority", CONTINUATION_LOCAL_SELECTION_AUTHORITY_DISABLED))
        == CONTINUATION_LOCAL_SELECTION_AUTHORITY_GUARDED_EXECUTE
    )


def dwa_rollout_states(
    v: float,
    w: float,
    horizon_sec: float,
    dt_sec: float,
    *,
    include_partial_terminal: bool = False,
) -> List[Tuple[float, float, float, float]]:
    """Return the existing DWA Euler rollout states without safety authority.

    The normal collision scorer keeps its historic full-step behavior by using
    ``include_partial_terminal=False``.  C0 continuation uses the same
    recurrence at the real command-slice horizon; a final partial step is
    explicit rather than a separately invented interpolation model.
    """
    if not finite_number(horizon_sec) or not finite_number(dt_sec) or float(horizon_sec) <= 0.0 or float(dt_sec) <= 0.0:
        return []
    full_steps = (
        int(float(horizon_sec) / float(dt_sec))
        if include_partial_terminal else max(1, int(float(horizon_sec) / float(dt_sec)))
    )
    x = y = yaw = 0.0
    states: List[Tuple[float, float, float, float]] = []
    for index in range(full_steps):
        elapsed = min(float(horizon_sec), float(index + 1) * float(dt_sec))
        step_dt = float(dt_sec)
        x += float(v) * math.cos(yaw) * step_dt
        y += float(v) * math.sin(yaw) * step_dt
        yaw += float(w) * step_dt
        states.append((elapsed, x, y, yaw))
    remainder = float(horizon_sec) - float(full_steps) * float(dt_sec)
    if include_partial_terminal and remainder > DWA_NUMERIC_EPS:
        x += float(v) * math.cos(yaw) * remainder
        y += float(v) * math.sin(yaw) * remainder
        yaw += float(w) * remainder
        states.append((float(horizon_sec), x, y, yaw))
    return states


def dwa_rollout_state_at_duration(v: float, w: float, duration_sec: float, dt_sec: float) -> Optional[Dict[str, Any]]:
    """Expose one command-slice state from the same DWA rollout recurrence."""
    states = dwa_rollout_states(v, w, duration_sec, dt_sec, include_partial_terminal=True)
    if not states:
        return None
    elapsed, x, y, yaw = states[-1]
    return {
        "time_sec": float(elapsed),
        "endpoint_base_xy": [float(x), float(y)],
        "endpoint_base_yaw_rad": float(yaw),
        "partial_terminal_step": bool(abs(float(elapsed) / float(dt_sec) - round(float(elapsed) / float(dt_sec))) > DWA_NUMERIC_EPS),
    }


def validate_phase2_productive_admission_activation(args: argparse.Namespace) -> None:
    """Reject a half-complete Phase-2 controller before ROS/publisher startup."""
    phase2_mode = str(getattr(args, "room_local_phase2_productive_admission", ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_DISABLED))
    phase3_mode = str(getattr(args, "room_local_phase3_orientation_recovery", ROOM_LOCAL_PHASE3_RECOVERY_DISABLED))
    continuation_authority_mode = str(getattr(
        args, "continuation_local_selection_authority", CONTINUATION_LOCAL_SELECTION_AUTHORITY_DISABLED,
    ))
    if phase2_mode == ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_OFFLINE_FROZEN and bool(getattr(args, "execute", False)):
        raise ValueError("room_local_phase2_productive_admission_offline_frozen_disallows_execute")
    if phase3_mode == ROOM_LOCAL_PHASE3_RECOVERY_OFFLINE_FROZEN and bool(getattr(args, "execute", False)):
        raise ValueError("room_local_phase3_orientation_recovery_offline_frozen_disallows_execute")
    if phase3_mode == ROOM_LOCAL_PHASE3_RECOVERY_OFFLINE_FROZEN and phase2_mode != ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_OFFLINE_FROZEN:
        raise ValueError("phase3_offline_frozen_requires_phase2_offline_frozen")
    if phase2_mode == ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_PHASE3_GUARDED_EXECUTE:
        if phase3_mode != ROOM_LOCAL_PHASE3_RECOVERY_GUARDED_EXECUTE:
            raise ValueError("phase3_guarded_execute_requires_phase3_orientation_recovery")
    elif phase3_mode == ROOM_LOCAL_PHASE3_RECOVERY_GUARDED_EXECUTE:
        raise ValueError("phase3_guarded_execute_requires_phase2_productive_admission")
    if continuation_authority_mode == CONTINUATION_LOCAL_SELECTION_AUTHORITY_GUARDED_EXECUTE:
        if not bool(getattr(args, "execute", False)):
            raise ValueError("continuation_local_selection_authority_requires_execute")
        if str(getattr(args, "local_control_mode", LOCAL_CONTROL_MODE_TRANSIT)) != LOCAL_CONTROL_MODE_ROOM_LOCAL:
            raise ValueError("continuation_local_selection_authority_requires_room_local")
        if (
            phase2_mode != ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_PHASE3_GUARDED_EXECUTE
            or phase3_mode != ROOM_LOCAL_PHASE3_RECOVERY_GUARDED_EXECUTE
        ):
            raise ValueError("continuation_local_selection_authority_requires_phase3_guarded_execute")


def derive_angular_recovery_margin(w_samples: Sequence[float], command_slice_sec: float) -> Optional[float]:
    """Use the existing angular lattice, not a hand-picked degree threshold."""
    if not finite_number(command_slice_sec) or float(command_slice_sec) <= 0.0:
        return None
    ordered = sorted(set(float(value) for value in w_samples if finite_number(value)))
    spacings = [second - first for first, second in zip(ordered, ordered[1:]) if second - first > DWA_NUMERIC_EPS]
    if not spacings:
        return None
    return 0.5 * min(spacings) * float(command_slice_sec)


def rotate_xy(point_xy: Tuple[float, float], yaw_rad: float) -> Tuple[float, float]:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return (
        c * float(point_xy[0]) - s * float(point_xy[1]),
        s * float(point_xy[0]) + c * float(point_xy[1]),
    )


def yaw_distance_to_interval(yaw_rad: float, interval: CircularYawInterval) -> float:
    return max(0.0, abs(normalize_angle(float(yaw_rad) - float(interval.center_rad))) - float(interval.half_width_rad))


def recovery_distance(yaw_rad: float, intervals: Sequence[CircularYawInterval]) -> Optional[float]:
    if not intervals:
        return None
    return min(yaw_distance_to_interval(yaw_rad, interval) for interval in intervals)


def recovery_intervals_from_hypotheses(
    hypotheses: Sequence[Tuple[float, bool]],
    angular_margin_rad: Optional[float],
) -> Tuple[CircularYawInterval, ...]:
    """Bound successful current-DWA yaw hypotheses by their native spacing."""
    ordered = sorted((float(yaw), bool(success)) for yaw, success in hypotheses if finite_number(yaw))
    if not ordered or angular_margin_rad is None or float(angular_margin_rad) <= 0.0:
        return ()
    raw_intervals: List[Tuple[float, float]] = []
    for index, (yaw, success) in enumerate(ordered):
        if not success:
            continue
        left = (yaw - ordered[index - 1][0]) * 0.5 if index else float(angular_margin_rad)
        right = (ordered[index + 1][0] - yaw) * 0.5 if index + 1 < len(ordered) else float(angular_margin_rad)
        half_width = max(DWA_NUMERIC_EPS, min(left, right))
        raw_intervals.append((yaw - half_width, yaw + half_width))
    if not raw_intervals:
        return ()
    # The hypothesis sweep is linear around the current yaw, so merge adjacent
    # successful samples before turning the bounded intervals into circular data.
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(raw_intervals):
        if merged and start <= merged[-1][1] + DWA_NUMERIC_EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(
        CircularYawInterval(normalize_angle((start + end) * 0.5), max(DWA_NUMERIC_EPS, (end - start) * 0.5))
        for start, end in merged
    )


def select_orientation_slice(
    w_samples: Sequence[float],
    safe_w_samples: Sequence[float],
    intervals: Sequence[CircularYawInterval],
    *,
    current_relative_yaw_rad: float,
    command_slice_sec: float,
    angular_margin_rad: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Select one current-lattice rotation only when recoverability improves."""
    before = recovery_distance(current_relative_yaw_rad, intervals)
    if before is None or angular_margin_rad is None:
        return None
    options: List[Dict[str, Any]] = []
    safe = {float(value) for value in safe_w_samples if finite_number(value)}
    for w in sorted(float(value) for value in w_samples if finite_number(value)):
        if abs(w) <= DWA_NUMERIC_EPS or w not in safe:
            continue
        after_yaw = normalize_angle(float(current_relative_yaw_rad) + w * float(command_slice_sec))
        after = recovery_distance(after_yaw, intervals)
        if after is None:
            continue
        improvement = float(before) - float(after)
        if improvement + DWA_NUMERIC_EPS < float(angular_margin_rad):
            continue
        options.append({
            "w_radps": w,
            "before_rad": float(before),
            "after_rad": float(after),
            "improvement_rad": improvement,
            "direction": 1 if w > 0.0 else -1,
        })
    if not options:
        return None
    return min(options, key=lambda item: (-float(item["improvement_rad"]), abs(float(item["w_radps"])), float(item["w_radps"])))


def select_observation_orientation_slice(
    w_samples: Sequence[float],
    safe_w_samples: Sequence[float],
    *,
    current_yaw_odom_rad: float,
    aim_yaw_odom_rad: float,
    command_slice_sec: float,
) -> Optional[Dict[str, Any]]:
    """Choose one existing safe angular-lattice sample toward a frozen aim.

    This is deliberately a low-level primitive, not a Phase-3 RecoverySet
    decision.  The aim selects a safe direction only; it has no observation
    completion authority and it creates no persistent recovery state.
    """
    if not (
        finite_number(current_yaw_odom_rad)
        and finite_number(aim_yaw_odom_rad)
        and finite_number(command_slice_sec)
        and float(command_slice_sec) > 0.0
    ):
        return None
    before = abs(normalize_angle(float(aim_yaw_odom_rad) - float(current_yaw_odom_rad)))
    safe = {float(value) for value in safe_w_samples if finite_number(value)}
    options: List[Dict[str, Any]] = []
    for w in sorted(float(value) for value in w_samples if finite_number(value)):
        if abs(w) <= DWA_NUMERIC_EPS or w not in safe:
            continue
        after_yaw = normalize_angle(float(current_yaw_odom_rad) + w * float(command_slice_sec))
        after = abs(normalize_angle(float(aim_yaw_odom_rad) - after_yaw))
        improvement = before - after
        if improvement <= DWA_NUMERIC_EPS:
            continue
        options.append({
            "w_radps": float(w),
            "aim_error_before_rad": float(before),
            "aim_error_after_rad": float(after),
            "aim_error_improvement_rad": float(improvement),
            "direction": 1 if w > 0.0 else -1,
        })
    if not options:
        return None
    return min(options, key=lambda item: (-float(item["aim_error_improvement_rad"]), abs(float(item["w_radps"])), float(item["w_radps"])))


def recovery_set_record(recovery_set: RecoverySet) -> Dict[str, Any]:
    return {
        "counterfactual_status": "RECOVERYSET_COUNTERFACTUAL_APPROXIMATE" if recovery_set.approximate else "EXACT",
        "angular_margin_rad": recovery_set.angular_margin_rad,
        "intervals": [
            {"center_rad": interval.center_rad, "half_width_rad": interval.half_width_rad}
            for interval in recovery_set.intervals
        ],
        "hypotheses": list(recovery_set.hypothesis_records),
    }


def orientation_intent_record(intent: Optional[OrientationIntent]) -> Optional[Dict[str, Any]]:
    if intent is None:
        return None
    return {
        "target_key": list(intent.target_key), "anchor_odom_xy": list(intent.anchor_odom_xy),
        "anchor_tangent_odom_rad": intent.anchor_tangent_odom_rad,
        "entry_yaw_rad": intent.entry_yaw_rad, "chosen_direction": intent.chosen_direction,
        "recovery_distance_before_rad": intent.recovery_distance_before_rad,
        "last_recovery_distance_rad": intent.last_recovery_distance_rad,
        "consumed": intent.consumed, "awaiting_fresh_replan": intent.awaiting_fresh_replan,
    }


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def blend_angles(a: float, b: float, weight_b: float) -> float:
    weight_b = max(0.0, min(1.0, float(weight_b)))
    weight_a = 1.0 - weight_b
    return math.atan2(
        weight_a * math.sin(a) + weight_b * math.sin(b),
        weight_a * math.cos(a) + weight_b * math.cos(b),
    )


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def euler_from_quat(q: Any) -> Tuple[float, float, float]:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"_read_error": str(exc)}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

def pose_tuple(msg: Odometry) -> Tuple[float, float, float]:
    pose = msg.pose.pose
    return float(pose.position.x), float(pose.position.y), yaw_from_quat(pose.orientation)


def target_to_base(target_xy: Sequence[float], pose: Tuple[float, float, float]) -> Tuple[float, float]:
    dx = float(target_xy[0]) - pose[0]
    dy = float(target_xy[1]) - pose[1]
    c = math.cos(pose[2])
    s = math.sin(pose[2])
    return c * dx + s * dy, -s * dx + c * dy


def base_to_odom(point_base_xy: Sequence[float], pose: Tuple[float, float, float]) -> Tuple[float, float]:
    """Transform a predicted base-frame endpoint into the frozen odom frame."""
    c = math.cos(pose[2])
    s = math.sin(pose[2])
    return (
        pose[0] + c * float(point_base_xy[0]) - s * float(point_base_xy[1]),
        pose[1] + s * float(point_base_xy[0]) + c * float(point_base_xy[1]),
    )


def dwa_predicted_endpoint_odom(
    pose: Tuple[float, float, float],
    v: float,
    w: float,
    predict_time_sec: float,
    dt_sec: float,
) -> Tuple[float, float, float]:
    """Integrate one DWA sample with the same discrete horizon as its scoring."""
    x_base = 0.0
    y_base = 0.0
    yaw_delta = 0.0
    for _ in range(max(1, int(float(predict_time_sec) / max(float(dt_sec), 1e-6)))):
        x_base += float(v) * math.cos(yaw_delta) * float(dt_sec)
        y_base += float(v) * math.sin(yaw_delta) * float(dt_sec)
        yaw_delta += float(w) * float(dt_sec)
    endpoint_x, endpoint_y = base_to_odom((x_base, y_base), pose)
    return endpoint_x, endpoint_y, normalize_angle(float(pose[2]) + yaw_delta)


def portal_relative_geometry(
    pose: Tuple[float, float, float],
    v: float,
    w: float,
    predict_time_sec: float,
    dt_sec: float,
    portal_center_odom: Sequence[float],
    portal_normal_odom: Sequence[float],
) -> Optional[Dict[str, Any]]:
    """Return parameter-free Portal-frame endpoint geometry for one DWA arc."""
    if not (
        len(portal_center_odom) == 2
        and len(portal_normal_odom) == 2
        and all(finite_number(value) for value in portal_center_odom)
        and all(finite_number(value) for value in portal_normal_odom)
    ):
        return None
    normal_norm = math.hypot(float(portal_normal_odom[0]), float(portal_normal_odom[1]))
    if normal_norm <= 1e-9:
        return None
    normal_x = float(portal_normal_odom[0]) / normal_norm
    normal_y = float(portal_normal_odom[1]) / normal_norm
    endpoint_x, endpoint_y, endpoint_yaw = dwa_predicted_endpoint_odom(
        pose, v, w, predict_time_sec, dt_sec,
    )
    start_x, start_y = pose[0], pose[1]
    tangent_x, tangent_y = -normal_y, normal_x
    portal_yaw = math.atan2(normal_y, normal_x)
    return {
        "portal_normal_progress_m": (endpoint_x - start_x) * normal_x + (endpoint_y - start_y) * normal_y,
        "portal_abs_tangent_offset_m": abs(
            (endpoint_x - float(portal_center_odom[0])) * tangent_x
            + (endpoint_y - float(portal_center_odom[1])) * tangent_y
        ),
        "portal_abs_yaw_error_rad": abs(normalize_angle(endpoint_yaw - portal_yaw)),
        "endpoint_odom_xy": [endpoint_x, endpoint_y],
        "endpoint_yaw_rad": endpoint_yaw,
    }


def p_through_arc_reference(
    pose_odom: Tuple[float, float, float],
    portal_center_odom: Sequence[float],
    portal_normal_odom: Sequence[float],
    p_pre_odom: Sequence[float],
) -> Optional[Dict[str, Any]]:
    """Return the physically feasible P_pre-to-Portal quarter-arc heading.

    P_pre is already defined Portal-relatively: it is upstream along the
    Portal tangent and outside along the Portal normal.  Its outside-normal
    distance is therefore the only non-arbitrary radius that brings a
    tangent-aligned robot to the Portal plane in a quarter turn.  This helper
    does not generate commands or relax collision checks; it supplies a
    heading reference for selection among DWA's existing safe candidates.
    """
    if not (
        len(portal_center_odom) == 2
        and len(portal_normal_odom) == 2
        and len(p_pre_odom) == 2
        and all(finite_number(value) for value in portal_center_odom)
        and all(finite_number(value) for value in portal_normal_odom)
        and all(finite_number(value) for value in p_pre_odom)
    ):
        return None
    normal_norm = math.hypot(float(portal_normal_odom[0]), float(portal_normal_odom[1]))
    if normal_norm <= 1e-9:
        return None
    normal_x = float(portal_normal_odom[0]) / normal_norm
    normal_y = float(portal_normal_odom[1]) / normal_norm
    tangent_x, tangent_y = -normal_y, normal_x

    def portal_frame(point_xy: Sequence[float]) -> Tuple[float, float]:
        dx = float(point_xy[0]) - float(portal_center_odom[0])
        dy = float(point_xy[1]) - float(portal_center_odom[1])
        return dx * tangent_x + dy * tangent_y, dx * normal_x + dy * normal_y

    pre_tangent_m, pre_normal_m = portal_frame(p_pre_odom)
    current_tangent_m, current_normal_m = portal_frame(pose_odom)
    # P_pre must be outside the Portal and have a definite upstream side.
    if pre_normal_m >= -1e-6 or abs(pre_tangent_m) <= 1e-6:
        return None
    radius_m = abs(pre_normal_m)
    # A quarter turn consumes one radius of tangent runway.  If P_pre was
    # configured nearer than that, no physical arc can reach the Portal plane.
    if abs(pre_tangent_m) + 1e-6 < radius_m:
        return None
    upstream_sign = 1.0 if pre_tangent_m > 0.0 else -1.0
    tangent_progress_m = upstream_sign * (pre_tangent_m - current_tangent_m)
    arc_center_normal_m = pre_normal_m + radius_m
    theta_rad = math.atan2(
        max(0.0, tangent_progress_m),
        max(0.0, arc_center_normal_m - current_normal_m),
    )
    theta_rad = max(0.0, min(0.5 * math.pi, theta_rad))
    desired_x = -upstream_sign * math.cos(theta_rad) * tangent_x + math.sin(theta_rad) * normal_x
    desired_y = -upstream_sign * math.cos(theta_rad) * tangent_y + math.sin(theta_rad) * normal_y
    return {
        "heading_rad": math.atan2(desired_y, desired_x),
        "radius_m": radius_m,
        "theta_rad": theta_rad,
        "current_tangent_m": current_tangent_m,
        "current_normal_m": current_normal_m,
        "p_pre_tangent_m": pre_tangent_m,
        "p_pre_normal_m": pre_normal_m,
        "upstream_sign": upstream_sign,
        "portal_tangent_xy": [tangent_x, tangent_y],
        "portal_normal_xy": [normal_x, normal_y],
    }


def p_through_arc_point_odom(
    reference: Dict[str, Any],
    portal_center_odom: Sequence[float],
    theta_rad: float,
) -> Tuple[float, float, float]:
    """Return a point and tangent heading on the proven P_pre-to-Portal arc."""
    theta = max(0.0, min(0.5 * math.pi, float(theta_rad)))
    radius = float(reference["radius_m"])
    upstream_sign = float(reference["upstream_sign"])
    tangent_x, tangent_y = (float(value) for value in reference["portal_tangent_xy"])
    normal_x, normal_y = (float(value) for value in reference["portal_normal_xy"])
    tangent_m = float(reference["p_pre_tangent_m"]) - upstream_sign * radius * math.sin(theta)
    normal_m = float(reference["p_pre_normal_m"]) + radius * (1.0 - math.cos(theta))
    point_x = float(portal_center_odom[0]) + tangent_m * tangent_x + normal_m * normal_x
    point_y = float(portal_center_odom[1]) + tangent_m * tangent_y + normal_m * normal_y
    heading_x = -upstream_sign * math.cos(theta) * tangent_x + math.sin(theta) * normal_x
    heading_y = -upstream_sign * math.cos(theta) * tangent_y + math.sin(theta) * normal_y
    return point_x, point_y, math.atan2(heading_y, heading_x)


def odom_stamp_sec(msg: Odometry) -> Optional[float]:
    try:
        stamp_sec = float(msg.header.stamp.to_sec())
    except Exception:
        return None
    return stamp_sec if finite_number(stamp_sec) else None


def terminal_reach_from_post_command_odom(
    target_xy: Sequence[float],
    pre_command_odom: Odometry,
    post_command_odom: Odometry,
    goal_tolerance_m: float,
) -> Dict[str, Any]:
    """Evaluate the existing goal disk from a demonstrably newer terminal odom.

    This is deliberately not a second success condition.  It only makes the
    same ``distance <= goal_tolerance_m`` predicate available after the final
    permitted command slice, before the loop reports MAX_STEPS.
    """
    pre_stamp_sec = odom_stamp_sec(pre_command_odom)
    post_stamp_sec = odom_stamp_sec(post_command_odom)
    report: Dict[str, Any] = {
        "attempted": True,
        "pre_command_odom_stamp_sec": pre_stamp_sec,
        "post_command_odom_stamp_sec": post_stamp_sec,
        "fresh_valid_odom": False,
        "target_xy_team_livox_odom": [float(target_xy[0]), float(target_xy[1])],
        "goal_tolerance_m": float(goal_tolerance_m),
        "post_command_distance_to_target_m": None,
        "reached": False,
        "reason": None,
    }
    if pre_stamp_sec is None or post_stamp_sec is None:
        report["reason"] = "post_command_odom_stamp_invalid"
        return report
    if post_stamp_sec <= pre_stamp_sec:
        report["reason"] = "post_command_odom_not_fresh"
        return report
    pose = pose_tuple(post_command_odom)
    tx_base, ty_base = target_to_base(target_xy, pose)
    distance = math.hypot(tx_base, ty_base)
    report.update(
        {
            "fresh_valid_odom": True,
            "post_command_pose_x_y_yaw": list(pose),
            "post_command_target_base_xy": [tx_base, ty_base],
            "post_command_distance_to_target_m": distance,
        }
    )
    if distance <= float(goal_tolerance_m):
        report.update({"reached": True, "reason": "post_command_reached_existing_goal_tolerance"})
    else:
        report["reason"] = "post_command_outside_existing_goal_tolerance"
    return report


def build_dwa_slice_attribution(
    *,
    decision_id: str,
    safety_binding: Mapping[str, Any],
    pose_x_y_yaw: Sequence[float],
    dwa: Mapping[str, Any],
    raw_v: float,
    raw_w: float,
) -> Dict[str, Any]:
    """Record the exact state identity behind one final DWA command slice."""
    return {
        "dwa_decision_id": str(decision_id),
        "grid": copy.deepcopy(dict(safety_binding["grid_identity"])),
        "status": copy.deepcopy(dict(safety_binding["status_identity"])),
        "safety_pose": {
            "source_odom_stamp": safety_binding.get("source_odom_stamp"),
            "odom_t0_stamp": safety_binding.get("odom_t0_stamp"),
            "odom_t1_stamp": safety_binding.get("odom_t1_stamp"),
            "odom_epoch_generation": safety_binding.get("odom_epoch_generation"),
            "binding_mode": safety_binding.get("binding_mode"),
            "pose_x_y_yaw": [float(value) for value in pose_x_y_yaw],
        },
        "dwa": {
            "total_candidate_count": int(dwa.get("sample_count") or 0),
            "safe_candidate_count": int(dwa.get("safe_moving_candidate_count") or 0),
            "selected_v": float(dwa.get("selected_linear_x") or 0.0),
            "selected_w": float(dwa.get("selected_angular_z") or 0.0),
            "selected_safety_outcome": "SAFE_SELECTED" if not bool(dwa.get("blocked")) else "NO_SAFE_TRAJECTORY",
        },
        "raw_command": {"v": float(raw_v), "w": float(raw_w)},
    }


class BlockAStarDwaRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.local_control_mode = str(getattr(args, "local_control_mode", LOCAL_CONTROL_MODE_TRANSIT))
        validate_phase2_productive_admission_activation(self.args)
        self.phase2_productive_admission_active = phase2_productive_admission_is_active(self.args)
        self.phase3_orientation_recovery_active = phase3_orientation_recovery_is_active(self.args)
        self.continuation_local_selection_authority_active = continuation_local_selection_authority_is_active(self.args)
        self.orientation_intent: Optional[OrientationIntent] = None
        self.validation_run_id = str(getattr(args, "validation_run_id", "")).strip()
        self.validation_runner_invocation_id = str(getattr(args, "validation_runner_invocation_id", "")).strip()
        self.validation_enabled = bool(self.validation_run_id and self.validation_runner_invocation_id)
        self.validation_event_sequence = 0
        self.validation_decision_sequence = 0
        self.validation_command_slice_sequence = 0
        self.safety_decision_sequence = 0
        # These counters describe observation/planning acquisition for the
        # validation sidecar only.  They do not feed DWA, A*, or recovery.
        self.validation_input_acquisition_sequence = 0
        self.validation_astar_evaluation_sequence = 0
        self.validation_path_anchor_sequence = 0
        self.validation_current_decision_id = ""
        self.validation_abort_reason: Optional[str] = None
        self.validation_events: List[Dict[str, Any]] = []
        self.validation_event_pub = (
            rospy.Publisher(str(args.validation_event_topic), String, queue_size=50)
            if self.validation_enabled else None
        )
        self.validation_abort_sub = (
            rospy.Subscriber(str(args.validation_abort_topic), String, self.validation_abort_cb, queue_size=10)
            if self.validation_enabled else None
        )
        self.pub = rospy.Publisher(self.args.cmd_topic, Twist, queue_size=10) if self.args.execute else None
        self.grid_status_pair_cache = ExactGridStatusPairCache(maxlen=8)
        # Same shared source-time cache used by the state machine.  It is the
        # sole resolver for collision-safety pose geometry in this process.
        self.odom_cache = OdomCache(rospy, topic=TOPIC_ODOM, message_type=Odometry)
        rospy.Subscriber(TOPIC_GRID, OccupancyGrid, self.grid_cb, queue_size=8)
        rospy.Subscriber(TOPIC_STATUS, String, self.status_cb, queue_size=8)
        self.prev_cmd = (0.0, 0.0)
        self.best_distance_seen = None
        self.no_progress_count = 0
        self.recovery_turn_sign = 1.0
        self.path_stability_hold_count = 0
        self.no_path_hold_count = 0
        self.unilateral_clearance_safety_side: Optional[int] = None
        self.unilateral_clearance_missing_count = 0
        self.run_wall_start_sec: Optional[float] = None
        self.run_sim_start_sec: Optional[float] = None
        self.run_wall_deadline_sec: Optional[float] = None
        self.use_sim_time = False
        self.sim_time_valid = False
        self.timeout_trigger: Optional[str] = None
        self.tf_listener = tf.TransformListener()
        self.latest_wall_heading_prior: Optional[Dict[str, Any]] = None
        self.wall_heading_point_buffer: List[Tuple[float, float, float, float]] = []
        self.last_wall_cloud_wall_time = 0.0
        self.latest_imu: Optional[Dict[str, Any]] = None
        self.imu_heading_anchor_yaw: Optional[float] = None
        self.imu_heading_anchor_wall_time: Optional[float] = None
        self.imu_heading_hold_active_last = False
        if not self.args.disable_pointcloud_wall_heading:
            rospy.Subscriber(TOPIC_LIDAR, PointCloud2, self.lidar_wall_heading_cb, queue_size=2)
        if not self.args.disable_imu_heading_hold:
            rospy.Subscriber(self.args.imu_topic, Imu, self.imu_cb, queue_size=200)

    def validation_abort_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if payload.get("run_id") == self.validation_run_id:
                self.validation_abort_reason = str(payload.get("reason") or "ROOM_LOCAL_VALIDATION_ABORT")
        except Exception:
            return

    def imu_cb(self, msg: Imu) -> None:
        _roll, _pitch, yaw = euler_from_quat(msg.orientation)
        self.latest_imu = {
            "stamp_sec": float(msg.header.stamp.to_sec()) if msg.header.stamp else float(rospy.Time.now().to_sec()),
            "wall_time_sec": time.monotonic(),
            "frame_id": msg.header.frame_id,
            "yaw_rad": float(yaw),
            "yaw_deg": float(math.degrees(yaw)),
            "angular_velocity_z": float(msg.angular_velocity.z),
        }

    def lookup_to_base(self, source_frame: str) -> Tuple[Any, Any, bool]:
        source = source_frame.lstrip("/")
        if source == "base":
            return None, None, True
        try:
            trans, rot = self.tf_listener.lookupTransform("base", source, rospy.Time(0))
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
        return None

    def fit_wall_line(self, points_xy: np.ndarray, side: str) -> Optional[Dict[str, Any]]:
        return fit_corridor_wall_line(points_xy, side, self.args)

    def lidar_wall_heading_cb(self, msg: PointCloud2) -> None:
        now = time.monotonic()
        if now - self.last_wall_cloud_wall_time < self.args.pointcloud_wall_min_interval_sec:
            return
        self.last_wall_cloud_wall_time = now
        points: List[Tuple[float, float, float]] = []
        seen = 0
        for raw in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            seen += 1
            if seen % max(1, self.args.pointcloud_wall_point_stride) != 0:
                continue
            p = self.transform_to_base(raw, msg.header.frame_id)
            if p is None:
                continue
            x, y, z = p
            if not (self.args.pointcloud_wall_x_min_m <= x <= self.args.pointcloud_wall_x_max_m):
                continue
            if abs(y) > self.args.pointcloud_wall_y_abs_max_m:
                continue
            if not (self.args.pointcloud_wall_z_min_m <= z <= self.args.pointcloud_wall_z_max_m):
                continue
            points.append((x, y, z))
            if len(points) >= self.args.pointcloud_wall_max_points:
                break

        self.wall_heading_point_buffer.extend((x, y, z, now) for x, y, z in points)
        min_time = now - self.args.pointcloud_wall_buffer_sec
        self.wall_heading_point_buffer = [
            item for item in self.wall_heading_point_buffer if item[3] >= min_time
        ][-self.args.pointcloud_wall_max_buffer_points :]
        buffer_points = [(x, y, z) for x, y, z, _t in self.wall_heading_point_buffer]
        arr = np.array(buffer_points, dtype=float) if buffer_points else np.empty((0, 3), dtype=float)
        left = arr[arr[:, 1] >= self.args.pointcloud_wall_min_abs_y_m] if arr.size else np.empty((0, 3), dtype=float)
        right = arr[arr[:, 1] <= -self.args.pointcloud_wall_min_abs_y_m] if arr.size else np.empty((0, 3), dtype=float)
        left_fit = self.fit_wall_line(left[:, :2], "left") if left.shape[0] else None
        right_fit = self.fit_wall_line(right[:, :2], "right") if right.shape[0] else None

        heading = None
        source = None
        if left_fit and right_fit:
            heading = blend_angles(left_fit["heading_parallel_rad"], right_fit["heading_parallel_rad"], 0.5)
            source = "bilateral_wall_lines"
        elif left_fit:
            heading = left_fit["heading_parallel_rad"]
            source = "left_wall_line"
        elif right_fit:
            heading = right_fit["heading_parallel_rad"]
            source = "right_wall_line"

        active = heading is not None and abs(heading) <= self.args.pointcloud_wall_max_abs_heading_rad
        self.latest_wall_heading_prior = {
            "enabled": not bool(self.args.disable_pointcloud_wall_heading),
            "active": bool(active),
            "source": source,
            "reason": "wall_heading_observed" if active else "insufficient_or_unstable_wall_line",
            "stamp_sec": float(msg.header.stamp.to_sec()),
            "wall_time_sec": now,
            "frame_id": msg.header.frame_id,
            "seen_points": int(seen),
            "current_cloud_accepted_point_count": int(len(points)),
            "accepted_point_count": int(arr.shape[0]),
            "left_point_count": int(left.shape[0]),
            "right_point_count": int(right.shape[0]),
            "left_wall_line": left_fit,
            "right_wall_line": right_fit,
            "heading_parallel_rad": float(heading) if heading is not None else None,
            "heading_parallel_deg": float(math.degrees(heading)) if heading is not None else None,
        }

    def current_wall_heading_prior(self) -> Dict[str, Any]:
        if self.args.disable_pointcloud_wall_heading:
            return {"enabled": False, "active": False, "reason": "disabled"}
        if self.latest_wall_heading_prior is None:
            return {"enabled": True, "active": False, "reason": "no_pointcloud_wall_heading_yet"}
        prior = dict(self.latest_wall_heading_prior)
        age = time.monotonic() - float(prior.get("wall_time_sec", 0.0))
        prior["age_sec"] = float(age)
        if age > self.args.pointcloud_wall_max_age_sec:
            prior["active"] = False
            prior["reason"] = "pointcloud_wall_heading_stale"
        return prior

    def apply_imu_heading_hold(self, v: float, w: float, dwa: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        report: Dict[str, Any] = {
            "enabled": not bool(self.args.disable_imu_heading_hold),
            "active": False,
            "reason": None,
            "base_cmd_angular_z": float(w),
            "corrected_cmd_angular_z": float(w),
            "correction_angular_z": 0.0,
            "anchor_yaw_rad": self.imu_heading_anchor_yaw,
            "imu_yaw_rad": None,
            "imu_yaw_error_rad": None,
            "imu_angular_velocity_z": None,
        }
        if self.args.disable_imu_heading_hold:
            report["reason"] = "disabled"
            return w, report
        if self.latest_imu is None:
            report["reason"] = "no_imu_sample"
            return w, report
        imu_age = time.monotonic() - float(self.latest_imu["wall_time_sec"])
        report["imu_age_sec"] = float(imu_age)
        report["imu_yaw_rad"] = float(self.latest_imu["yaw_rad"])
        report["imu_angular_velocity_z"] = float(self.latest_imu["angular_velocity_z"])
        if imu_age > self.args.imu_heading_hold_max_age_sec:
            report["reason"] = "imu_sample_stale"
            self.imu_heading_hold_active_last = False
            return w, report

        command_allows_hold = (
            v >= self.args.imu_heading_hold_min_linear_x
            and abs(w) <= self.args.imu_heading_hold_command_angular_deadband
            and not bool(dwa.get("recovery"))
        )
        report["command_allows_hold"] = bool(command_allows_hold)
        if not command_allows_hold:
            report["reason"] = "command_intent_turning_or_not_forward"
            self.imu_heading_anchor_yaw = None
            self.imu_heading_anchor_wall_time = None
            self.imu_heading_hold_active_last = False
            return w, report

        if self.imu_heading_anchor_yaw is None:
            self.imu_heading_anchor_yaw = float(self.latest_imu["yaw_rad"])
            self.imu_heading_anchor_wall_time = time.monotonic()
        yaw_error = normalize_angle(float(self.latest_imu["yaw_rad"]) - float(self.imu_heading_anchor_yaw))
        correction = (
            -self.args.imu_heading_hold_kp * yaw_error
            -self.args.imu_heading_hold_kd * float(self.latest_imu["angular_velocity_z"])
        )
        correction = max(
            -self.args.imu_heading_hold_max_correction_rad_s,
            min(self.args.imu_heading_hold_max_correction_rad_s, correction),
        )
        corrected = max(-self.args.max_angular_z, min(self.args.max_angular_z, w + correction))
        report.update(
            {
                "active": True,
                "reason": "heading_hold_applied",
                "anchor_yaw_rad": float(self.imu_heading_anchor_yaw),
                "anchor_age_sec": time.monotonic() - float(self.imu_heading_anchor_wall_time or time.monotonic()),
                "imu_yaw_error_rad": float(yaw_error),
                "imu_yaw_error_deg": float(math.degrees(yaw_error)),
                "correction_angular_z": float(correction),
                "corrected_cmd_angular_z": float(corrected),
            }
        )
        self.imu_heading_hold_active_last = True
        return corrected, report

    def grid_cb(self, msg: OccupancyGrid) -> None:
        self.grid_status_pair_cache.add_grid(msg)

    def status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {}
        self.grid_status_pair_cache.add_status(payload)

    def matching_grid_status_pair(self) -> Optional[Tuple[OccupancyGrid, Dict[str, Any]]]:
        """Return only an exact formal content-identity match inside L3V freshness."""
        cache = getattr(self, "grid_status_pair_cache", None)
        if cache is not None:
            return cache.matching_pair()
        # Offline contract fixtures intentionally construct this class with
        # object.__new__ and inject the historic receipt deques.  Preserve that
        # pure-test seam while routing the same records through the shared
        # matcher; normal production instances always take the branch above.
        cache = ExactGridStatusPairCache(maxlen=8)
        for received_wall_sec, grid in getattr(self, "grid_cache", []):
            cache.add_grid(grid, received_wall_sec)
        for received_wall_sec, status in getattr(self, "status_cache", []):
            cache.add_status(status, received_wall_sec)
        return cache.matching_pair()

    def dwa_safety_state_binding(
        self,
        grid_msg: OccupancyGrid,
        status_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve the only pose permitted for this formal Grid/status pair.

        Grid/status matching remains owned by ``ExactGridStatusPairCache``.
        This adds the missing third member of that physical-state identity:
        gated odom at ``status.source_odom_stamp``.  No latest-odom fallback
        exists; a missing or reset-crossing binding denies DWA publication.
        """
        metadata = grid_metadata(grid_msg)
        status = status_payload if isinstance(status_payload, dict) else {}
        grid_identity = {
            "transport_seq": getattr(getattr(grid_msg, "header", None), "seq", None),
            "content_stamp": metadata.get("content_stamp"),
            "content_generation_id": status.get("content_generation_id"),
            "content_hash": status.get("grid_content_hash"),
            "producer_instance_id": status.get("producer_instance_id"),
        }
        source_stamp = status.get("source_odom_stamp")
        status_identity = {
            "grid_content_stamp": status.get("grid_content_stamp"),
            "source_odom_stamp": source_stamp,
            "content_generation_id": status.get("content_generation_id"),
            "producer_instance_id": status.get("producer_instance_id"),
        }
        if not finite_number(source_stamp):
            return {
                "binding_valid": False,
                "reason": "SAFETY_STATE_BINDING_SOURCE_ODOM_STAMP_UNAVAILABLE",
                "grid_identity": grid_identity,
                "status_identity": status_identity,
            }
        binding = self.odom_cache.pose_at_source_stamp(float(source_stamp))
        binding = dict(binding) if isinstance(binding, dict) else {
            "binding_valid": False,
            "reason": "SAFETY_STATE_BINDING_ODOM_RESOLVER_INVALID_RETURN",
        }
        binding.update({
            "source_odom_stamp": float(source_stamp),
            "grid_identity": grid_identity,
            "status_identity": status_identity,
            "current_odom_epoch_generation": self.odom_cache.current_epoch_generation(),
        })
        if binding.get("binding_valid") is not True:
            binding["reason"] = "SAFETY_STATE_BINDING_UNAVAILABLE:" + str(
                binding.get("reason") or "ODOM_SOURCE_BINDING_UNAVAILABLE"
            )
            return binding
        source_epoch = binding.get("odom_epoch_generation")
        if source_epoch != self.odom_cache.current_epoch_generation():
            binding.update({
                "binding_valid": False,
                "reason": "SAFETY_STATE_BINDING_ODOM_EPOCH_MISMATCH",
            })
            return binding
        status_epoch = status.get("source_odom_epoch_generation")
        if status_epoch is not None and status_epoch != source_epoch:
            binding.update({
                "binding_valid": False,
                "reason": "SAFETY_STATE_BINDING_STATUS_ODOM_EPOCH_MISMATCH",
            })
            return binding
        pose = binding.get("source_pose_x_y_yaw")
        if not (isinstance(pose, list) and len(pose) == 3 and all(finite_number(value) for value in pose)):
            binding.update({
                "binding_valid": False,
                "reason": "SAFETY_STATE_BINDING_POSE_INVALID",
            })
            return binding
        binding["binding_mode"] = str(binding.get("odom_binding_method") or "UNKNOWN")
        return binding

    def wait_inputs(self) -> Tuple[Odometry, OccupancyGrid, Dict[str, Any]]:
        """Wait until one formal pair also has a lawful source-time pose.

        A short-lived runner subscribes after the runtime is already live.
        Its first formal Grid/status pair can therefore predate its local
        odom history.  Do not hand that pair to DWA as an immediate safety
        failure: wait for a newer exact pair and its lawful source-time pose.
        This is startup synchronization only; it never substitutes latest
        odom for the status source time and still times out fail-closed.
        """
        deadline = time.monotonic() + float(self.args.input_timeout_sec)
        last_odom_sequence = self.odom_cache.current_sequence()
        last_binding_reason = "SAFETY_STATE_BINDING_UNAVAILABLE"
        while time.monotonic() < deadline:
            pair = self.matching_grid_status_pair()
            if pair is not None:
                safety_binding = self.dwa_safety_state_binding(pair[0], pair[1])
                if safety_binding.get("binding_valid") is True:
                    odom, _sequence = self.odom_cache.get(
                        max(0.01, deadline - time.monotonic())
                    )
                    return odom, pair[0], pair[1]
                last_binding_reason = str(safety_binding.get("reason") or last_binding_reason)
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            # Receive another gated-odom sample first.  Grid/status callbacks
            # continue asynchronously, so the next iteration selects only the
            # newest exact pair that can be bound to this runner's own cache.
            try:
                _odom, last_odom_sequence = self.odom_cache.get(
                    min(0.10, remaining), after_sequence=last_odom_sequence
                )
            except (RuntimeError, rospy.ROSException):
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                self.grid_cb(rospy.wait_for_message(TOPIC_GRID, OccupancyGrid, timeout=min(0.10, remaining)))
            except rospy.ROSException:
                pass
        raise rospy.ROSException(
            "formal_grid_status_pose_binding_unavailable:" + last_binding_reason
        )

    def grid_array(self, msg: OccupancyGrid) -> np.ndarray:
        errors = validate_grid_metadata(msg)
        if errors:
            raise ValueError("local_grid_contract_invalid:" + ",".join(errors))
        width, height = int(msg.info.width), int(msg.info.height)
        arr = np.empty((height, width), dtype=np.int16)
        for y_index in range(height):
            for x_index in range(width):
                arr[y_index, x_index] = msg.data[flatten_index(x_index, y_index, width, height)]
        return arr

    def grid_qualification_errors(self, grid_msg: OccupancyGrid, status_payload: Dict[str, Any]) -> List[str]:
        _, errors = qualified_for_navigation(grid_msg, status_payload)
        if not math.isclose(float(self.args.robot_radius_m), STATIC_PLANNING_FOOTPRINT_RADIUS_M, rel_tol=0.0, abs_tol=1e-12):
            errors.append("runner_static_footprint_radius_mismatch")
        additional_margin = getattr(self.args, "additional_clearance_margin_m", 0.0)
        if not finite_number(additional_margin) or float(additional_margin) < 0.0:
            errors.append("additional_clearance_margin_invalid")
        if status_payload.get("static_footprint_radius_m") != STATIC_PLANNING_FOOTPRINT_RADIUS_M:
            errors.append("status_static_footprint_radius_mismatch")
        if status_payload.get("planning_collision_model") != POINT_PLANNING_WITH_OBSTACLE_INFLATION:
            errors.append("status_planning_collision_model_mismatch")
        return sorted(set(errors))

    def centerline_columns(self, grid_msg: OccupancyGrid, y_abs_m: float) -> List[int]:
        cols = []
        for y_index in range(int(grid_msg.info.height)):
            _, y = self.cell_to_local_xy((0, y_index), grid_msg)
            if abs(y) <= y_abs_m:
                cols.append(y_index)
        return cols

    def row_free_ratio(self, grid: np.ndarray, rows: Iterable[int], cols: Sequence[int]) -> Optional[float]:
        values: List[int] = []
        width = grid.shape[1]
        for x_index in rows:
            if 0 <= x_index < width:
                values.extend(int(grid[y_index, x_index]) for y_index in cols)
        if not values:
            return None
        return float(sum(1 for value in values if value == 0) / len(values))

    def apply_centerline_thin_barrier_clearing(
        self,
        grid_msg: OccupancyGrid,
        grid: np.ndarray,
        target_base_xy: Tuple[float, float],
        centerline_target: bool,
        status_payload: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        report: Dict[str, Any] = {
            "enabled": not bool(self.args.disable_centerline_thin_barrier_clearing),
            "applied": False,
            "reason": None,
            "candidate_groups": [],
            "cleared_cell_count": 0,
        }
        if self.args.disable_centerline_thin_barrier_clearing:
            report["reason"] = "disabled"
            return grid, report
        if not centerline_target:
            report["reason"] = "not_centerline_target"
            return grid, report
        if abs(target_base_xy[1]) > self.args.centerline_target_y_threshold_m:
            report["reason"] = "target_not_centered"
            return grid, report
        if status_payload.get("local_traversability_status") != "FREE_SUPPORTED":
            report["reason"] = "l3v_status_not_free_supported"
            return grid, report

        x_limit = min(float(target_base_xy[0]) + 0.10, self.args.thin_barrier_x_max_m)
        first = self.local_xy_to_cell(self.args.thin_barrier_x_min_m, 0.0, grid_msg)
        last = self.local_xy_to_cell(max(self.args.thin_barrier_x_min_m, x_limit), 0.0, grid_msg)
        if first is None or last is None:
            report["reason"] = "scan_window_out_of_grid"
            return grid, report
        row_min, _ = first
        row_max, _ = last
        row_min = max(0, row_min)
        row_max = min(grid.shape[1] - 1, row_max)
        cols = self.centerline_columns(grid_msg, self.args.centerline_clear_y_abs_m)
        if row_max < row_min or not cols:
            report["reason"] = "empty_scan_window"
            return grid, report

        candidate_rows: List[int] = []
        row_details: Dict[int, Dict[str, Any]] = {}
        for row in range(row_min, row_max + 1):
            values = [int(grid[y_index, row]) for y_index in cols]
            occupied = sum(1 for value in values if value == 100)
            unknown = sum(1 for value in values if value == -1)
            occ_ratio = occupied / max(len(values), 1)
            detail = {
                "row": int(row),
                "x_center_m": self.cell_to_local_xy((row, cols[0]), grid_msg)[0],
                "occupied_count": int(occupied),
                "unknown_count": int(unknown),
                "cell_count": len(values),
                "occupied_ratio": float(occ_ratio),
            }
            row_details[row] = detail
            if unknown == 0 and occ_ratio >= self.args.thin_barrier_row_occupied_ratio:
                candidate_rows.append(row)

        groups: List[List[int]] = []
        for row in candidate_rows:
            if not groups or row != groups[-1][-1] + 1:
                groups.append([row])
            else:
                groups[-1].append(row)

        out = grid.copy()
        max_cells = max(1, int(math.ceil(self.args.thin_barrier_max_thickness_m / float(grid_msg.info.resolution))))
        support_cells = max(1, int(math.ceil(self.args.thin_barrier_support_depth_m / float(grid_msg.info.resolution))))
        for group in groups:
            before_rows = range(group[0] - support_cells, group[0])
            after_rows = range(group[-1] + 1, group[-1] + support_cells + 1)
            before_free = self.row_free_ratio(grid, before_rows, cols)
            after_free = self.row_free_ratio(grid, after_rows, cols)
            accepted = (
                len(group) <= max_cells
                and before_free is not None
                and after_free is not None
                and before_free >= self.args.thin_barrier_free_support_ratio
                and after_free >= self.args.thin_barrier_free_support_ratio
            )
            group_report = {
                "rows": [int(group[0]), int(group[-1])],
                "x_range_m": [
                    self.cell_to_local_xy((group[0], cols[0]), grid_msg)[0],
                    self.cell_to_local_xy((group[-1], cols[0]), grid_msg)[0],
                ],
                "thickness_cells": len(group),
                "thickness_m": len(group) * float(grid_msg.info.resolution),
                "before_free_ratio": before_free,
                "after_free_ratio": after_free,
                "accepted": bool(accepted),
                "row_details": [row_details[row] for row in group],
            }
            if accepted:
                before_count = int((out[np.ix_(cols, group)] == 100).sum())
                out[np.ix_(cols, group)] = np.where(out[np.ix_(cols, group)] == 100, 0, out[np.ix_(cols, group)])
                cleared = before_count - int((out[np.ix_(cols, group)] == 100).sum())
                report["cleared_cell_count"] += int(cleared)
                report["applied"] = True
            report["candidate_groups"].append(group_report)

        if report["applied"]:
            report["reason"] = "thin_centerline_barrier_cleared"
        elif groups:
            report["reason"] = "candidate_groups_rejected"
        else:
            report["reason"] = "no_thin_barrier_candidate"
        return out, report

    def occupied_inflated_mask(self, grid: np.ndarray, resolution_m: Optional[float] = None) -> np.ndarray:
        resolution = float(self.args.grid_resolution_m) if resolution_m is None else float(resolution_m)
        additional_margin = float(getattr(self.args, "additional_clearance_margin_m", 0.0))
        return occupied_euclidean_inflated_mask(
            grid,
            float(self.args.robot_radius_m) + additional_margin,
            resolution,
        )

    def inflate_obstacles(self, grid: np.ndarray, resolution_m: Optional[float] = None) -> np.ndarray:
        inflated = self.occupied_inflated_mask(grid, resolution_m)
        unknown = grid == -1
        # Unknown is always blocking for formal motion; no opt-in escape hatch.
        inflated |= unknown
        return inflated

    def apply_start_footprint_clearance(
        self,
        grid_msg: OccupancyGrid,
        raw_grid: np.ndarray,
        existing_blocked: np.ndarray,
        occupied_inflated: np.ndarray,
        qualification_passed: bool,
        start_base_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Clear only protected-free unknown cells in the current static circle.

        This never mutates `raw_grid`, the OccupancyGrid message, or the formal
        grid/status qualification.  The returned mask is a planner-only copy.
        """
        raw_hash_before = (
            hashlib.sha256(np.ascontiguousarray(raw_grid).tobytes()).hexdigest()
            if isinstance(raw_grid, np.ndarray) else None
        )
        if isinstance(existing_blocked, np.ndarray):
            result = existing_blocked.copy()
        elif isinstance(raw_grid, np.ndarray):
            # A missing planning mask must never become an implicit free mask.
            result = np.ones(raw_grid.shape, dtype=bool)
        else:
            result = np.ones((0, 0), dtype=bool)
        report: Dict[str, Any] = {
            "enabled": True,
            "applied": False,
            "radius_m": float(self.args.robot_radius_m) if finite_number(self.args.robot_radius_m) else None,
            "footprint_cell_count": 0,
            "raw_unknown_inside_count": 0,
            "raw_occupied_inside_count": 0,
            "occupied_inflated_inside_count": 0,
            "eligible_unknown_cleared_count": 0,
            "retained_unknown_inside_count": 0,
            "retained_blocked_inside_count": 0,
            "start_block_before": None,
            "start_block_after": None,
            "raw_grid_unchanged": False,
            "reason": None,
        }
        if not qualification_passed:
            report["reason"] = "QUALIFICATION_NOT_PASSED"
        elif not finite_number(self.args.robot_radius_m) or float(self.args.robot_radius_m) <= 0.0:
            report["reason"] = "INVALID_RADIUS"
        elif validate_grid_metadata(grid_msg):
            report["reason"] = "GRID_OR_MASK_INVALID"
        elif (
            not isinstance(raw_grid, np.ndarray)
            or not isinstance(existing_blocked, np.ndarray)
            or not isinstance(occupied_inflated, np.ndarray)
            or raw_grid.shape != existing_blocked.shape
            or raw_grid.shape != occupied_inflated.shape
            or not np.isin(raw_grid, (-1, 0, 100)).all()
        ):
            report["reason"] = "GRID_OR_MASK_INVALID"
        else:
            start = self.local_xy_to_cell(float(start_base_xy[0]), float(start_base_xy[1]), grid_msg)
            if start is None:
                report["reason"] = "START_OUT_OF_GRID"
            else:
                start_block = (start[0] // max(1, int(self.args.block_size_cells)), start[1] // max(1, int(self.args.block_size_cells)))
                report["start_block_before"] = self.block_window_stats(existing_blocked, start_block)
                radius = float(self.args.robot_radius_m)
                for y_index in range(raw_grid.shape[0]):
                    for x_index in range(raw_grid.shape[1]):
                        x, y = self.cell_to_local_xy((x_index, y_index), grid_msg)
                        if (x - float(start_base_xy[0])) ** 2 + (y - float(start_base_xy[1])) ** 2 > radius * radius:
                            continue
                        report["footprint_cell_count"] += 1
                        raw_value = int(raw_grid[y_index, x_index])
                        if raw_value == -1:
                            report["raw_unknown_inside_count"] += 1
                        elif raw_value == 100:
                            report["raw_occupied_inside_count"] += 1
                        if bool(occupied_inflated[y_index, x_index]):
                            report["occupied_inflated_inside_count"] += 1
                        if raw_value == -1 and not bool(occupied_inflated[y_index, x_index]):
                            result[y_index, x_index] = False
                            report["eligible_unknown_cleared_count"] += 1
                report["applied"] = report["eligible_unknown_cleared_count"] > 0
                report["retained_unknown_inside_count"] = int(sum(
                    int(raw_grid[y_index, x_index]) == -1 and bool(result[y_index, x_index])
                    for y_index in range(raw_grid.shape[0]) for x_index in range(raw_grid.shape[1])
                    if ((self.cell_to_local_xy((x_index, y_index), grid_msg)[0] - float(start_base_xy[0])) ** 2
                        + (self.cell_to_local_xy((x_index, y_index), grid_msg)[1] - float(start_base_xy[1])) ** 2) <= radius * radius
                ))
                report["retained_blocked_inside_count"] = int(sum(
                    bool(result[y_index, x_index])
                    for y_index in range(raw_grid.shape[0]) for x_index in range(raw_grid.shape[1])
                    if ((self.cell_to_local_xy((x_index, y_index), grid_msg)[0] - float(start_base_xy[0])) ** 2
                        + (self.cell_to_local_xy((x_index, y_index), grid_msg)[1] - float(start_base_xy[1])) ** 2) <= radius * radius
                ))
                report["start_block_after"] = self.block_window_stats(result, start_block)
                if report["applied"]:
                    report["reason"] = "APPLIED"
                elif report["raw_occupied_inside_count"] or report["occupied_inflated_inside_count"]:
                    report["reason"] = "OCCUPIED_OR_INFLATED_CONFLICT"
                else:
                    report["reason"] = "NO_ELIGIBLE_UNKNOWN"
        raw_hash_after = (
            hashlib.sha256(np.ascontiguousarray(raw_grid).tobytes()).hexdigest()
            if isinstance(raw_grid, np.ndarray) else None
        )
        report["raw_grid_unchanged"] = raw_hash_before is not None and raw_hash_before == raw_hash_after
        return result, report

    def apply_start_footprint_clearance_from_shared_context(
        self,
        grid_msg: OccupancyGrid,
        raw_grid: np.ndarray,
        existing_blocked: np.ndarray,
        occupied_inflated: np.ndarray,
        qualification_passed: bool,
        cell_center_x_m: np.ndarray,
        cell_center_y_m: np.ndarray,
        raw_grid_sha256: Optional[str],
        start_base_xy: Tuple[float, float],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Equivalent S1 footprint clearance using immutable epoch geometry.

        This is deliberately C0-only.  It starts from a per-S1 copy of the
        existing planner mask, while the raw Grid, inflated mask and cell
        centres are read-only same-epoch inputs.  The legacy helper above is
        retained as the equivalence reference for every other call path.
        """
        if isinstance(existing_blocked, np.ndarray):
            result = existing_blocked.copy()
        elif isinstance(raw_grid, np.ndarray):
            result = np.ones(raw_grid.shape, dtype=bool)
        else:
            result = np.ones((0, 0), dtype=bool)
        report: Dict[str, Any] = {
            "enabled": True,
            "applied": False,
            "radius_m": float(self.args.robot_radius_m) if finite_number(self.args.robot_radius_m) else None,
            "footprint_cell_count": 0,
            "raw_unknown_inside_count": 0,
            "raw_occupied_inside_count": 0,
            "occupied_inflated_inside_count": 0,
            "eligible_unknown_cleared_count": 0,
            "retained_unknown_inside_count": 0,
            "retained_blocked_inside_count": 0,
            "start_block_before": None,
            "start_block_after": None,
            "raw_grid_unchanged": False,
            "reason": None,
        }
        axes_valid = (
            isinstance(cell_center_x_m, np.ndarray)
            and isinstance(cell_center_y_m, np.ndarray)
            and cell_center_x_m.ndim == 1
            and cell_center_y_m.ndim == 1
            and isinstance(raw_grid, np.ndarray)
            and len(cell_center_x_m) == raw_grid.shape[1]
            and len(cell_center_y_m) == raw_grid.shape[0]
        )
        if not qualification_passed:
            report["reason"] = "QUALIFICATION_NOT_PASSED"
        elif not finite_number(self.args.robot_radius_m) or float(self.args.robot_radius_m) <= 0.0:
            report["reason"] = "INVALID_RADIUS"
        elif validate_grid_metadata(grid_msg):
            report["reason"] = "GRID_OR_MASK_INVALID"
        elif (
            not isinstance(raw_grid, np.ndarray)
            or not isinstance(existing_blocked, np.ndarray)
            or not isinstance(occupied_inflated, np.ndarray)
            or raw_grid.shape != existing_blocked.shape
            or raw_grid.shape != occupied_inflated.shape
            or not np.isin(raw_grid, (-1, 0, 100)).all()
            or not axes_valid
        ):
            report["reason"] = "GRID_OR_MASK_INVALID"
        else:
            start = self.local_xy_to_cell(float(start_base_xy[0]), float(start_base_xy[1]), grid_msg)
            if start is None:
                report["reason"] = "START_OUT_OF_GRID"
            else:
                start_block = (
                    start[0] // max(1, int(self.args.block_size_cells)),
                    start[1] // max(1, int(self.args.block_size_cells)),
                )
                report["start_block_before"] = self.block_window_stats(existing_blocked, start_block)
                radius = float(self.args.robot_radius_m)
                inside = (
                    (cell_center_x_m[np.newaxis, :] - float(start_base_xy[0])) ** 2
                    + (cell_center_y_m[:, np.newaxis] - float(start_base_xy[1])) ** 2
                    <= radius * radius
                )
                raw_unknown_inside = inside & (raw_grid == -1)
                raw_occupied_inside = inside & (raw_grid == 100)
                inflated_inside = inside & occupied_inflated
                eligible_unknown = raw_unknown_inside & ~occupied_inflated
                report["footprint_cell_count"] = int(inside.sum())
                report["raw_unknown_inside_count"] = int(raw_unknown_inside.sum())
                report["raw_occupied_inside_count"] = int(raw_occupied_inside.sum())
                report["occupied_inflated_inside_count"] = int(inflated_inside.sum())
                result[eligible_unknown] = False
                report["eligible_unknown_cleared_count"] = int(eligible_unknown.sum())
                report["applied"] = report["eligible_unknown_cleared_count"] > 0
                report["retained_unknown_inside_count"] = int((raw_unknown_inside & result).sum())
                report["retained_blocked_inside_count"] = int((inside & result).sum())
                report["start_block_after"] = self.block_window_stats(result, start_block)
                if report["applied"]:
                    report["reason"] = "APPLIED"
                elif report["raw_occupied_inside_count"] or report["occupied_inflated_inside_count"]:
                    report["reason"] = "OCCUPIED_OR_INFLATED_CONFLICT"
                else:
                    report["reason"] = "NO_ELIGIBLE_UNKNOWN"
        # The shared raw array is read-only.  Retaining this check's legacy
        # output field avoids changing audit semantics without re-hashing the
        # same immutable bytes for every S1.
        report["raw_grid_unchanged"] = raw_grid_sha256 is not None
        return result, report

    def local_xy_to_cell(self, x: float, y: float, msg: OccupancyGrid) -> Optional[Tuple[int, int]]:
        return metric_to_cell(x, y, grid_metadata(msg))

    def cell_to_local_xy(self, cell: Tuple[int, int], msg: OccupancyGrid) -> Tuple[float, float]:
        point = cell_to_metric(cell[0], cell[1], grid_metadata(msg))
        if point is None:
            raise ValueError("cell_out_of_bounds")
        return point

    def grid_value_counts(self, grid: np.ndarray) -> Dict[str, int]:
        return {
            "free": int((grid == 0).sum()),
            "occupied": int((grid == 100).sum()),
            "unknown": int((grid == -1).sum()),
            "other": int(((grid != 0) & (grid != 100) & (grid != -1)).sum()),
        }

    def local_rect_counts(
        self,
        grid_msg: OccupancyGrid,
        grid: np.ndarray,
        blocked: np.ndarray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
    ) -> Dict[str, Any]:
        raw_values: List[int] = []
        blocked_count = 0
        height, width = grid.shape
        for y_index in range(height):
            for x_index in range(width):
                x, y = self.cell_to_local_xy((x_index, y_index), grid_msg)
                if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                    raw_values.append(int(grid[y_index, x_index]))
                    if bool(blocked[y_index, x_index]):
                        blocked_count += 1
        total = len(raw_values)
        return {
            "x_range_m": list(x_range),
            "y_range_m": list(y_range),
            "cell_count": total,
            "free_count": int(sum(1 for v in raw_values if v == 0)),
            "occupied_count": int(sum(1 for v in raw_values if v == 100)),
            "unknown_count": int(sum(1 for v in raw_values if v == -1)),
            "inflated_blocked_count": int(blocked_count),
            "inflated_blocked_ratio": float(blocked_count / total) if total else None,
        }

    def estimate_corridor_center(
        self,
        grid_msg: OccupancyGrid,
        observed_wall_evidence: np.ndarray,
    ) -> Dict[str, Any]:
        """Estimate a corridor center from observed obstacle geometry only.

        `UNKNOWN` remains collision-blocked in the separate planning mask, but
        lack of observation is not positive evidence of a physical wall.
        `observed_wall_evidence` is therefore the occupied-only inflated mask,
        rather than the broader collision-blocked mask.
        """
        report: Dict[str, Any] = {
            "enabled": not bool(self.args.disable_corridor_center_target),
            "wall_evidence_semantics": "occupied_euclidean_inflated_only_unknown_excluded",
            "applied": False,
            "reason": None,
            "x_range_m": [self.args.corridor_center_x_min_m, self.args.corridor_center_x_max_m],
            "wall_min_abs_y_m": self.args.corridor_center_wall_min_abs_y_m,
            "left_wall_cell_count": 0,
            "right_wall_cell_count": 0,
            "left_wall_inner_y_m": None,
            "right_wall_inner_y_m": None,
            "estimated_center_y_m": None,
            "estimated_width_m": None,
        }
        if self.args.disable_corridor_center_target:
            report["reason"] = "disabled"
            return report

        left_wall_y: List[float] = []
        right_wall_y: List[float] = []
        height, width = observed_wall_evidence.shape
        for y_index in range(height):
            for x_index in range(width):
                if not bool(observed_wall_evidence[y_index, x_index]):
                    continue
                x, y = self.cell_to_local_xy((x_index, y_index), grid_msg)
                if not (self.args.corridor_center_x_min_m <= x <= self.args.corridor_center_x_max_m):
                    continue
                if y >= self.args.corridor_center_wall_min_abs_y_m:
                    left_wall_y.append(float(y))
                elif y <= -self.args.corridor_center_wall_min_abs_y_m:
                    right_wall_y.append(float(y))

        report["left_wall_cell_count"] = len(left_wall_y)
        report["right_wall_cell_count"] = len(right_wall_y)
        if (
            len(left_wall_y) < self.args.corridor_center_min_wall_count
            or len(right_wall_y) < self.args.corridor_center_min_wall_count
        ):
            report["reason"] = "insufficient_bilateral_wall_support"
            return report

        left_inner = min(left_wall_y)
        right_inner = max(right_wall_y)
        width = left_inner - right_inner
        center_y = 0.5 * (left_inner + right_inner)
        report["left_wall_inner_y_m"] = float(left_inner)
        report["right_wall_inner_y_m"] = float(right_inner)
        report["estimated_center_y_m"] = float(center_y)
        report["estimated_width_m"] = float(width)

        if width < self.args.corridor_center_min_width_m or width > self.args.corridor_center_max_width_m:
            report["reason"] = "estimated_width_out_of_range"
            return report
        if abs(center_y) > self.args.corridor_center_max_abs_y_m:
            report["reason"] = "estimated_center_out_of_range"
            return report

        report["applied"] = True
        report["reason"] = "bilateral_wall_center_estimated"
        return report

    def apply_corridor_center_target(
        self,
        original_target_y_m: float,
        corridor_center_target: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Apply a bounded local-wall correction to a centreline target.

        The local observation is useful for small, symmetric centreline
        corrections.  Near an opening, however, the nearest evidence on one
        side can belong to the room rather than the corridor wall.  The
        resulting bilateral estimate is not permitted to replace the frozen
        centreline target by a large lateral displacement.
        """
        original_y = float(original_target_y_m)
        corridor_center_target["original_target_base_y_m"] = original_y
        corridor_center_target["blend_weight"] = float(self.args.corridor_center_blend_weight)
        corridor_center_target["target_correction_limit_m"] = float(
            self.args.corridor_center_target_max_correction_m
        )
        corridor_center_target["target_adjustment_applied"] = False
        corridor_center_target["target_adjustment_reason"] = "estimate_not_applied"
        corridor_center_target["adjusted_target_base_y_m"] = original_y

        if not corridor_center_target.get("applied"):
            return original_y, corridor_center_target

        estimated_center_y = float(corridor_center_target["estimated_center_y_m"])
        blended_y = (
            (1.0 - self.args.corridor_center_blend_weight) * original_y
            + self.args.corridor_center_blend_weight * estimated_center_y
        )
        bounded_y = max(
            -self.args.corridor_center_target_max_abs_y_m,
            min(self.args.corridor_center_target_max_abs_y_m, blended_y),
        )
        correction_m = bounded_y - original_y
        corridor_center_target["blended_target_base_y_m"] = float(blended_y)
        corridor_center_target["requested_target_correction_m"] = float(correction_m)
        if abs(correction_m) > self.args.corridor_center_target_max_correction_m:
            corridor_center_target["target_adjustment_reason"] = (
                "target_correction_exceeds_limit"
            )
            return original_y, corridor_center_target

        corridor_center_target["adjusted_target_base_y_m"] = float(bounded_y)
        corridor_center_target["target_adjustment_applied"] = True
        corridor_center_target["target_adjustment_reason"] = "bounded_bilateral_wall_center"
        return bounded_y, corridor_center_target

    def unilateral_clearance_target_shape(
        self,
        grid_msg: OccupancyGrid,
        target: Dict[str, Any],
        grid: np.ndarray,
        planning_blocked: np.ndarray,
        target_base_xy: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        """Shape only FOLLOW_CORRIDOR centreline targets away from a one-sided block.

        This deliberately does not estimate a physical centreline.  It consumes
        the already-formal inflated/unknown-blocked grid and selects the first
        grid-cell centre which the existing Block A* and DWA can represent:
        beyond DWA's existing lateral deadband and in a Block-4 centre with the
        same safety-side sign.  No map, inflation, A*, or DWA semantics change.
        """
        tx_base, ty_base = target_base_xy
        report: Dict[str, Any] = {
            "enabled": True,
            "applied": False,
            "reason": None,
            "target_source": target.get("source"),
            "original_target_base_xy": [float(tx_base), float(ty_base)],
            "shaped_target_base_xy": [float(tx_base), float(ty_base)],
            "left_blocked_count": 0,
            "right_blocked_count": 0,
            "left_nearest_blocked_m": None,
            "right_nearest_blocked_m": None,
            "direct_safety_side": None,
            "effective_safety_side": None,
            "selected_target_cell": None,
            "selected_target_block": None,
            "selected_block_center_y_m": None,
            "persistence": None,
        }
        source = str(target.get("source") or "")
        if not source.startswith("state_machine_corridor_centerline"):
            report["reason"] = "not_follow_corridor_centerline_target"
            return target_base_xy, report
        if not isinstance(grid, np.ndarray) or not isinstance(planning_blocked, np.ndarray) or grid.shape != planning_blocked.shape:
            report["reason"] = "grid_or_planning_mask_invalid"
            return target_base_xy, report

        left_count = 0
        right_count = 0
        left_nearest = float("inf")
        right_nearest = float("inf")
        for y_index in range(grid.shape[0]):
            for x_index in range(grid.shape[1]):
                if not bool(planning_blocked[y_index, x_index]):
                    continue
                x, y = self.cell_to_local_xy((x_index, y_index), grid_msg)
                if 0.0 <= x <= self.args.corridor_center_x_max_m:
                    distance = math.hypot(x, y)
                    if y >= 0.0:
                        left_nearest = min(left_nearest, distance)
                    else:
                        right_nearest = min(right_nearest, distance)
                if not (self.args.corridor_center_x_min_m <= x <= self.args.corridor_center_x_max_m):
                    continue
                if y >= self.args.corridor_center_wall_min_abs_y_m:
                    left_count += 1
                elif y <= -self.args.corridor_center_wall_min_abs_y_m:
                    right_count += 1
        report["left_blocked_count"] = int(left_count)
        report["right_blocked_count"] = int(right_count)
        report["left_nearest_blocked_m"] = float(left_nearest) if math.isfinite(left_nearest) else None
        report["right_nearest_blocked_m"] = float(right_nearest) if math.isfinite(right_nearest) else None

        minimum_support = int(self.args.corridor_center_min_wall_count)
        near_side_limit = float(self.args.corridor_center_wall_min_abs_y_m)
        direct_side: Optional[int] = None
        if (
            left_count >= minimum_support
            and right_count < minimum_support
            and left_nearest <= near_side_limit
            and right_nearest > left_nearest
        ):
            direct_side = -1  # local y<0 is the formal right/open side.
        elif (
            right_count >= minimum_support
            and left_count < minimum_support
            and right_nearest <= near_side_limit
            and left_nearest > right_nearest
        ):
            direct_side = 1
        report["direct_safety_side"] = "right_negative_y" if direct_side == -1 else (
            "left_positive_y" if direct_side == 1 else None
        )

        cached_side = getattr(self, "unilateral_clearance_safety_side", None)
        missing_count = int(getattr(self, "unilateral_clearance_missing_count", 0))
        effective_side = direct_side
        if direct_side is not None:
            self.unilateral_clearance_safety_side = direct_side
            self.unilateral_clearance_missing_count = 0
            report["persistence"] = "direct_observation"
        elif cached_side in (-1, 1):
            missing_count += 1
            if missing_count < int(self.args.path_stability_max_hold_steps):
                effective_side = int(cached_side)
                self.unilateral_clearance_missing_count = missing_count
                report["persistence"] = "held_existing_path_stability_window"
            else:
                self.unilateral_clearance_safety_side = None
                self.unilateral_clearance_missing_count = 0
                report["persistence"] = "released_after_existing_path_stability_window"
        else:
            report["persistence"] = "no_unilateral_signal"
        report["effective_safety_side"] = "right_negative_y" if effective_side == -1 else (
            "left_positive_y" if effective_side == 1 else None
        )
        if effective_side is None:
            report["reason"] = "no_unilateral_clearance_degradation"
            return target_base_xy, report

        x_cell = self.local_xy_to_cell(tx_base, 0.0, grid_msg)
        if x_cell is None:
            report["reason"] = "forward_target_out_of_grid"
            return target_base_xy, report
        x_index, _ = x_cell
        max_abs_y = min(
            float(self.args.corridor_center_target_max_abs_y_m),
            float(self.args.centerline_target_y_threshold_m),
        )
        block = max(1, int(self.args.block_size_cells))
        candidates: List[Tuple[float, int, int, float]] = []
        for y_index in range(grid.shape[0]):
            _x, y = self.cell_to_local_xy((x_index, y_index), grid_msg)
            if effective_side * y <= 0.0:
                continue
            if abs(y) + 1e-12 < float(self.args.target_lateral_correction_threshold_m):
                continue
            if abs(y) > max_abs_y + 1e-12:
                continue
            block_x, block_y = x_index // block, y_index // block
            x0, x1 = block_x * block, min(planning_blocked.shape[1], (block_x + 1) * block)
            y0, y1 = block_y * block, min(planning_blocked.shape[0], (block_y + 1) * block)
            center_x = min(planning_blocked.shape[1] - 1, block_x * block + block // 2)
            center_y = min(planning_blocked.shape[0] - 1, block_y * block + block // 2)
            _cx, block_center_y = self.cell_to_local_xy((center_x, center_y), grid_msg)
            if effective_side * block_center_y <= 0.0:
                continue
            if int(grid[y_index, x_index]) != 0 or bool(planning_blocked[y_index, x_index]):
                continue
            if bool(planning_blocked[y0:y1, x0:x1].any()):
                continue
            candidates.append((abs(y), x_index, y_index, block_center_y))
        if not candidates:
            report["reason"] = "no_formally_admissible_surviving_target_cell"
            return target_base_xy, report
        _abs_y, selected_x, selected_y, selected_block_center_y = min(candidates, key=lambda item: item[0])
        _selected_x_m, selected_y_m = self.cell_to_local_xy((selected_x, selected_y), grid_msg)
        report.update(
            {
                "applied": True,
                "reason": "unilateral_clearance_safety_side_target",
                "shaped_target_base_xy": [float(tx_base), float(selected_y_m)],
                "selected_target_cell": [int(selected_x), int(selected_y)],
                "selected_target_block": [int(selected_x // block), int(selected_y // block)],
                "selected_block_center_y_m": float(selected_block_center_y),
            }
        )
        return (float(tx_base), float(selected_y_m)), report

    def encoded_window(self, grid: np.ndarray, center: Tuple[int, int], radius: int = 4) -> Dict[str, Any]:
        height, width = grid.shape
        x_index, y_index = center
        y0, y1 = max(0, y_index - radius), min(height, y_index + radius + 1)
        x0, x1 = max(0, x_index - radius), min(width, x_index + radius + 1)
        chars = {-1: "?", 0: ".", 100: "#"}
        rows = []
        for yy in range(y0, y1):
            rows.append("".join(chars.get(int(grid[yy, xx]), "x") for xx in range(x0, x1)))
        return {
            "center_cell": [int(x_index), int(y_index)],
            "x_index_range": [int(x0), int(x1 - 1)],
            "y_index_range": [int(y0), int(y1 - 1)],
            "legend": {".": "free", "#": "occupied", "?": "unknown", "x": "other"},
            "rows": rows,
        }

    def block_window_stats(self, blocked: np.ndarray, block_cell: Tuple[int, int]) -> Dict[str, Any]:
        block = max(1, int(self.args.block_size_cells))
        height, width = blocked.shape
        block_x, block_y = block_cell
        x0 = block_x * block
        y0 = block_y * block
        x1 = min(width, x0 + block)
        y1 = min(height, y0 + block)
        if x0 < 0 or y0 < 0 or x0 >= width or y0 >= height:
            return {"block_cell": [int(block_x), int(block_y)], "in_grid": False}
        window = blocked[y0:y1, x0:x1]
        return {
            "block_cell": [int(block_x), int(block_y)],
            "in_grid": True,
            "x_index_range": [int(x0), int(x1 - 1)],
            "y_index_range": [int(y0), int(y1 - 1)],
            "blocked_count": int(window.sum()),
            "cell_count": int(window.size),
            "block_free": bool(not window.any()),
        }

    def path_local_xy(self, path: List[Tuple[int, int]], grid_msg: OccupancyGrid) -> List[List[float]]:
        return [[float(x), float(y)] for x, y in (self.cell_to_local_xy(cell, grid_msg) for cell in path)]

    def path_grid_diagnostic(
        self,
        grid_msg: OccupancyGrid,
        grid: np.ndarray,
        blocked: np.ndarray,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        raw_path: List[Tuple[int, int]],
        path: List[Tuple[int, int]],
        target_base_xy: Tuple[float, float],
        waypoint_xy: Optional[Tuple[float, float]],
        astar_lateral_bias_active: bool,
        corridor_center_target: Optional[Dict[str, Any]] = None,
        start_footprint_clearance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        block = max(1, int(self.args.block_size_cells))
        start_b = (start_cell[0] // block, start_cell[1] // block)
        goal_b = (goal_cell[0] // block, goal_cell[1] // block)
        waypoint_delta = None
        if waypoint_xy is not None:
            waypoint_delta = float(waypoint_xy[1] - target_base_xy[1])
        return {
            "grid_value_counts": self.grid_value_counts(grid),
            "start_footprint_clearance": start_footprint_clearance,
            "inflated_blocked_count": int(blocked.sum()),
            "inflated_blocked_ratio": float(blocked.sum() / blocked.size) if blocked.size else None,
            "start_cell": [int(start_cell[0]), int(start_cell[1])],
            "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
            "start_block": self.block_window_stats(blocked, start_b),
            "start_block_admission": getattr(self, "last_block_astar_start_admission", None),
            "goal_block": self.block_window_stats(blocked, goal_b),
            "raw_path_cell_count": len(raw_path),
            "smoothed_path_cell_count": len(path),
            "raw_path_local_xy": self.path_local_xy(raw_path[:12], grid_msg),
            "smoothed_path_local_xy": self.path_local_xy(path[:12], grid_msg),
            "target_base_xy": [float(target_base_xy[0]), float(target_base_xy[1])],
            "waypoint_base_xy": list(waypoint_xy) if waypoint_xy is not None else None,
            "waypoint_to_target_lateral_delta_m": waypoint_delta,
            "astar_lateral_bias_active": bool(astar_lateral_bias_active),
            "corridor_center_target": corridor_center_target,
            "center_corridor_counts": self.local_rect_counts(grid_msg, grid, blocked, (0.35, 1.35), (-0.30, 0.30)),
            "door_width_corridor_counts": self.local_rect_counts(grid_msg, grid, blocked, (0.60, 1.60), (-0.75, 0.75)),
            "left_side_doorway_probe_counts": self.local_rect_counts(
                grid_msg, grid, blocked, (0.60, 2.20), (0.95, 1.45)
            ),
            "right_side_doorway_probe_counts": self.local_rect_counts(
                grid_msg, grid, blocked, (0.60, 2.20), (-1.45, -0.95)
            ),
            "left_side_wall_context_counts": self.local_rect_counts(
                grid_msg, grid, blocked, (0.60, 2.60), (0.90, 1.50)
            ),
            "right_side_wall_context_counts": self.local_rect_counts(
                grid_msg, grid, blocked, (0.60, 2.60), (-1.50, -0.90)
            ),
            "start_window": self.encoded_window(grid, start_cell),
            "goal_window": self.encoded_window(grid, goal_cell),
        }

    def target_is_grid_centerline(self, target: Dict[str, Any]) -> bool:
        source = str(target.get("source") or "")
        return (
            target.get("subgoal_source") == "l3v_grid_centerline"
            or target.get("subgoal_source") == "state_machine_centerline"
            or target.get("source") == "post_frame_fix_regenerated_grid_centerline_subgoal"
            or (source.startswith("state_machine_") and "centerline" in source)
        )

    def pre_room_zone_centerline_correction_authorized(self, target: Dict[str, Any]) -> bool:
        """Permit the local-wall correction only for explicitly pre-room targets."""
        return bool(
            self.target_is_grid_centerline(target)
            and target.get("corridor_center_target_scope") == "pre_room_zone_only"
        )

    def anchor_line_metrics(self, target: Dict[str, Any], pose: Tuple[float, float, float]) -> Optional[Dict[str, Any]]:
        anchor = target.get("entry_anchor_line")
        if not isinstance(anchor, dict):
            return None
        if not all(finite_number(anchor.get(key)) for key in ("x", "y", "heading_rad")):
            return None
        heading = float(anchor["heading_rad"])
        dx = pose[0] - float(anchor["x"])
        dy = pose[1] - float(anchor["y"])
        along = dx * math.cos(heading) + dy * math.sin(heading)
        lateral = -dx * math.sin(heading) + dy * math.cos(heading)
        yaw_error = normalize_angle(pose[2] - heading)
        return {
            "anchor_progress_m": float(along),
            "anchor_lateral_error_m": float(lateral),
            "anchor_abs_lateral_error_m": float(abs(lateral)),
            "anchor_yaw_error_rad": float(yaw_error),
            "anchor_abs_yaw_error_rad": float(abs(yaw_error)),
        }

    def apply_centerline_tracking_correction(
        self,
        v: float,
        w: float,
        target: Dict[str, Any],
        pose: Tuple[float, float, float],
        centerline_target: bool,
        dwa: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        report: Dict[str, Any] = {
            "enabled": not bool(self.args.disable_centerline_tracking_correction),
            "active": False,
            "reason": None,
            "base_cmd_angular_z": float(w),
            "corrected_cmd_angular_z": float(w),
            "correction_angular_z": 0.0,
        }
        if self.args.disable_centerline_tracking_correction:
            report["reason"] = "disabled"
            return w, report
        if not centerline_target:
            report["reason"] = "not_centerline_target"
            return w, report
        if bool(dwa.get("recovery")):
            report["reason"] = "recovery_command"
            return w, report
        if v < self.args.centerline_tracking_min_linear_x:
            report["reason"] = "linear_speed_too_low"
            return w, report
        if abs(w) > self.args.centerline_tracking_command_angular_deadband:
            report["reason"] = "planner_intent_turning"
            return w, report
        metrics = self.anchor_line_metrics(target, pose)
        if metrics is None:
            report["reason"] = "anchor_line_unavailable"
            return w, report
        lateral = float(metrics["anchor_lateral_error_m"])
        yaw_error = float(metrics["anchor_yaw_error_rad"])
        correction = -self.args.centerline_tracking_lateral_kp * lateral - self.args.centerline_tracking_yaw_kp * yaw_error
        correction = max(
            -self.args.centerline_tracking_max_correction_rad_s,
            min(self.args.centerline_tracking_max_correction_rad_s, correction),
        )
        corrected = max(-self.args.max_angular_z, min(self.args.max_angular_z, w + correction))
        report.update(
            {
                "active": True,
                "reason": "centerline_tracking_applied",
                "anchor_line_metrics": metrics,
                "correction_angular_z": float(correction),
                "corrected_cmd_angular_z": float(corrected),
                "lateral_kp": float(self.args.centerline_tracking_lateral_kp),
                "yaw_kp": float(self.args.centerline_tracking_yaw_kp),
            }
        )
        return corrected, report

    def apply_scoped_centerline_tracking_correction(
        self,
        v: float,
        w: float,
        target: Dict[str, Any],
        pose: Tuple[float, float, float],
        centerline_target: bool,
        dwa: Dict[str, Any],
        *,
        pre_room_zone_centerline_correction: bool,
    ) -> Tuple[float, Dict[str, Any]]:
        """Apply the existing bounded centreline feedback only before the room.

        The caller authorizes this wrapper solely for targets whose scope is
        ``pre_room_zone_only``.  It deliberately reuses the normal feedback
        guards (recovery, low speed, and intentional planner turns), rather
        than creating a separate corridor controller.  Room-zone and
        room-interior targets never enter this branch.
        """
        if pre_room_zone_centerline_correction:
            corrected_w, report = self.apply_centerline_tracking_correction(
                v, w, target, pose, centerline_target, dwa
            )
            report["scope"] = "pre_room_zone_only"
            return corrected_w, report
        return self.apply_centerline_tracking_correction(
            v, w, target, pose, centerline_target, dwa
        )

    def block_astar(
        self,
        blocked: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid_msg: Optional[OccupancyGrid] = None,
        target_lateral_y: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        block = max(1, int(self.args.block_size_cells))
        height, width = blocked.shape

        def to_block(cell: Tuple[int, int]) -> Tuple[int, int]:
            return cell[0] // block, cell[1] // block

        def block_free(bcell: Tuple[int, int]) -> bool:
            x0 = bcell[0] * block
            y0 = bcell[1] * block
            x1 = min(width, x0 + block)
            y1 = min(height, y0 + block)
            return bool(not blocked[y0:y1, x0:x1].any())

        def block_center(bcell: Tuple[int, int]) -> Tuple[int, int]:
            return (
                min(width - 1, bcell[0] * block + block // 2),
                min(height - 1, bcell[1] * block + block // 2),
            )

        def raster_segment_free(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
            """Check every planning cell crossed by the one permitted start escape."""
            steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
            for index in range(steps + 1):
                fraction = float(index) / float(max(1, steps))
                x_index = int(round(a[0] + (b[0] - a[0]) * fraction))
                y_index = int(round(a[1] + (b[1] - a[1]) * fraction))
                if (
                    x_index < 0 or y_index < 0 or x_index >= width or y_index >= height
                    or bool(blocked[y_index, x_index])
                ):
                    return False
            return True

        def block_lateral_cost(bcell: Tuple[int, int]) -> float:
            if grid_msg is None or target_lateral_y is None or self.args.astar_lateral_bias_weight <= 0.0:
                return 0.0
            center_cell = (
                min(width - 1, bcell[0] * block + block // 2),
                min(height - 1, bcell[1] * block + block // 2),
            )
            _, local_y = self.cell_to_local_xy(center_cell, grid_msg)
            return self.args.astar_lateral_bias_weight * abs(float(local_y) - float(target_lateral_y))

        start_b = to_block(start)
        goal_b = to_block(goal)
        start_cell_free = (
            0 <= start[0] < width and 0 <= start[1] < height and not bool(blocked[start[1], start[0]])
        )
        start_block_free = block_free(start_b)
        # A local grid is robot-centred. Inflation can mark other hypothetical
        # centres in the start Block-4 unsafe while the actual current
        # footprint remains formally safe. Admit exactly one escape only when
        # the exact start cell and the discrete segment to a fully-free
        # neighbour are safe.
        partial_start_escape = bool(start_cell_free and not start_block_free)
        self.last_block_astar_start_admission = {
            "start_cell_free": bool(start_cell_free),
            "start_block_free": bool(start_block_free),
            "partial_start_escape": bool(partial_start_escape),
            "reason": "FULL_START_BLOCK_FREE" if start_block_free else (
                "PARTIAL_START_BLOCK_EXACT_CELL_FREE" if partial_start_escape else "START_CELL_OR_BLOCK_UNSAFE"
            ),
        }
        if not start_block_free and not partial_start_escape:
            return []
        if not block_free(goal_b):
            goal_b = self.nearest_free_block(blocked, goal_b, block)
            if goal_b is None:
                return []

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        open_heap: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start_b))
        came: Dict[Tuple[int, int], Tuple[int, int]] = {}
        cost = {start_b: 0.0}

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == goal_b:
                path_b = [cur]
                while cur in came:
                    cur = came[cur]
                    path_b.append(cur)
                path_b.reverse()
                return [start if partial_start_escape and index == 0 else block_center(b)
                        for index, b in enumerate(path_b)]
            for dr, dc in moves:
                nb = (cur[0] + dr, cur[1] + dc)
                if nb[0] < 0 or nb[1] < 0 or nb[0] * block >= width or nb[1] * block >= height:
                    continue
                if not block_free(nb):
                    continue
                if cur == start_b and partial_start_escape and not raster_segment_free(start, block_center(nb)):
                    continue
                step = math.sqrt(2.0) if dr and dc else 1.0
                new_cost = cost[cur] + step + block_lateral_cost(nb)
                if new_cost < cost.get(nb, float("inf")):
                    cost[nb] = new_cost
                    priority = new_cost + math.hypot(nb[0] - goal_b[0], nb[1] - goal_b[1])
                    came[nb] = cur
                    heapq.heappush(open_heap, (priority, nb))
        return []

    @staticmethod
    def fine_grid_astar_fallback(
        blocked: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Find a 4-neighbor route on the existing planning mask only.

        This is intentionally a representation fallback for a coarse NO_PATH,
        not a second mask, collision model, or goal-relocation policy.
        """
        if not isinstance(blocked, np.ndarray) or blocked.ndim != 2:
            return []
        height, width = blocked.shape

        def free(cell: Tuple[int, int]) -> bool:
            x_index, y_index = cell
            return (
                0 <= x_index < width
                and 0 <= y_index < height
                and not bool(blocked[y_index, x_index])
            )

        if not free(start) or not free(goal):
            return []
        moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, 0, start))
        came: Dict[Tuple[int, int], Tuple[int, int]] = {}
        cost: Dict[Tuple[int, int], int] = {start: 0}
        tie_break = 0
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came:
                    current = came[current]
                    path.append(current)
                return list(reversed(path))
            for dx, dy in moves:
                neighbor = (current[0] + dx, current[1] + dy)
                if not free(neighbor):
                    continue
                new_cost = cost[current] + 1
                if new_cost >= cost.get(neighbor, float("inf")):
                    continue
                cost[neighbor] = new_cost
                came[neighbor] = current
                tie_break += 1
                priority = new_cost + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                heapq.heappush(open_heap, (float(priority), tie_break, neighbor))
        return []

    def nearest_free_block(self, blocked: np.ndarray, goal_b: Tuple[int, int], block: int) -> Optional[Tuple[int, int]]:
        height, width = blocked.shape
        max_r = max(height // block, width // block)
        for rad in range(1, max_r + 1):
            for block_x in range(goal_b[0] - rad, goal_b[0] + rad + 1):
                for block_y in range(goal_b[1] - rad, goal_b[1] + rad + 1):
                    if block_x < 0 or block_y < 0 or block_x * block >= width or block_y * block >= height:
                        continue
                    x0, y0 = block_x * block, block_y * block
                    x1, y1 = min(width, x0 + block), min(height, y0 + block)
                    if not blocked[y0:y1, x0:x1].any():
                        return block_x, block_y
        return None

    def collision_free_arc(
        self,
        grid_msg: OccupancyGrid,
        blocked: np.ndarray,
        v: float,
        w: float,
        rejection_trace: Optional[Dict[str, Any]] = None,
        reference_pose_base: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[bool, float]:
        """Evaluate one production DWA arc, optionally retaining its first rejection.

        ``rejection_trace`` is an observability sink only.  The default path and
        every collision decision remain the original production implementation.
        """
        min_clearance = float("inf")
        states = dwa_rollout_states(v, w, self.args.dwa_predict_time, self.args.dwa_dt)
        obstacle_cells = np.argwhere(blocked)
        if rejection_trace is not None:
            rejection_trace.update({
                "sample_count": int(len(states)),
                "first_rejected_sample": None,
            })
        final_x = final_y = final_yaw = 0.0
        for sample_index, (_elapsed, x, y, yaw) in enumerate(states):
            final_x, final_y, final_yaw = float(x), float(y), float(yaw)
            lookup_x, lookup_y = float(x), float(y)
            if reference_pose_base is not None:
                ref_x, ref_y, ref_yaw = reference_pose_base
                lookup_x = float(ref_x) + math.cos(float(ref_yaw)) * float(x) - math.sin(float(ref_yaw)) * float(y)
                lookup_y = float(ref_y) + math.sin(float(ref_yaw)) * float(x) + math.cos(float(ref_yaw)) * float(y)
            cell = self.local_xy_to_cell(lookup_x, lookup_y, grid_msg)
            if cell is None:
                if rejection_trace is not None:
                    rejection_trace["first_rejected_sample"] = {
                        "sample_index": int(sample_index),
                        "base_xy": [float(x), float(y)],
                        "base_yaw_rad": float(yaw),
                        "cell": None,
                        "rejection_reason": "OOB",
                    }
                return False, 0.0
            x_index, y_index = cell
            if blocked[y_index, x_index]:
                if rejection_trace is not None:
                    rejection_trace["first_rejected_sample"] = {
                        "sample_index": int(sample_index),
                        "base_xy": [float(x), float(y)],
                        "base_yaw_rad": float(yaw),
                        "cell": [int(x_index), int(y_index)],
                        "rejection_reason": "BLOCKED_MASK",
                    }
                return False, 0.0
            if obstacle_cells.size:
                d = np.sqrt(((obstacle_cells - np.array([y_index, x_index])) ** 2).sum(axis=1)).min()
                min_clearance = min(min_clearance, float(d) * float(grid_msg.info.resolution))
        if min_clearance == float("inf"):
            min_clearance = self.args.max_clearance_score_m
        if rejection_trace is not None:
            rejection_trace["endpoint_base_xy"] = [final_x, final_y]
            rejection_trace["endpoint_base_yaw_rad"] = final_yaw
        return True, min_clearance

    def collision_free_arc_at_yaw_offset(
        self,
        grid_msg: OccupancyGrid,
        blocked: np.ndarray,
        v: float,
        w: float,
        yaw_offset_rad: float,
    ) -> Tuple[bool, float, Tuple[float, float]]:
        """Frozen base-grid counterfactual with normal collision semantics.

        The candidate is rolled in the hypothesized future body frame.  Each
        point is rigidly mapped back to the current base-frame grid for the
        same blocked-mask and out-of-bounds test used by ``collision_free_arc``.
        This helper never mutates the Grid and actual post-rotation input remains
        authoritative.
        """
        x = y = yaw = 0.0
        min_clearance = float("inf")
        steps = max(1, int(self.args.dwa_predict_time / self.args.dwa_dt))
        obstacle_cells = np.argwhere(blocked)
        for _ in range(steps):
            x += float(v) * math.cos(yaw) * self.args.dwa_dt
            y += float(v) * math.sin(yaw) * self.args.dwa_dt
            yaw += float(w) * self.args.dwa_dt
            lookup_x, lookup_y = rotate_xy((x, y), yaw_offset_rad)
            cell = self.local_xy_to_cell(lookup_x, lookup_y, grid_msg)
            if cell is None:
                return False, 0.0, (x, y)
            x_index, y_index = cell
            if blocked[y_index, x_index]:
                return False, 0.0, (x, y)
            if obstacle_cells.size:
                d = np.sqrt(((obstacle_cells - np.array([y_index, x_index])) ** 2).sum(axis=1)).min()
                min_clearance = min(min_clearance, float(d) * float(grid_msg.info.resolution))
        if min_clearance == float("inf"):
            min_clearance = self.args.max_clearance_score_m
        return True, float(min_clearance), (x, y)

    def room_local_productive_count_at_yaw(
        self,
        grid_msg: OccupancyGrid,
        blocked: np.ndarray,
        waypoint_xy: Tuple[float, float],
        target_base_xy: Tuple[float, float],
        distance_to_goal: float,
        room_local_path_xy: Sequence[Tuple[float, float]],
        yaw_offset_rad: float,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Run the Phase-2 authority at a frozen in-place yaw hypothesis."""
        raw_v_samples, w_samples = self.dynamic_window()
        minimum_forward = effective_minimum_forward(float(self.args.min_linear_x))
        speed_limit = speed_limit_for_distance(
            self.args.max_linear_x, minimum_forward, distance_to_goal, self.args.distance_speed_gain,
        )
        v_samples = v_samples_with_legal_boundaries(raw_v_samples, minimum_forward, speed_limit)
        margin = derive_meaningful_progress_margin(float(self.args.grid_resolution_m), minimum_forward, float(self.args.dwa_dt))
        capability = room_local_capability_by_id("forward_turn_current_room_profile")
        transformed_path = tuple(rotate_xy(point, -float(yaw_offset_rad)) for point in room_local_path_xy)
        transformed_target = rotate_xy(target_base_xy, -float(yaw_offset_rad))
        transformed_waypoint = rotate_xy(waypoint_xy, -float(yaw_offset_rad))
        anchor = None
        remaining = None
        if len(transformed_path) >= 2 and margin is not None:
            anchor = build_path_progress_anchor(
                transformed_path, path_identity="recoveryset_frozen_path", robot_xy=(0.0, 0.0),
                grid_resolution_m=float(self.args.grid_resolution_m),
                rollout_step_m=minimum_forward * float(self.args.dwa_dt),
            )
            if anchor is not None:
                remaining = max(0.0, float(anchor.stations_m[-1]) - float(anchor.s_current_m))
        direct_or_short = anchor is None or (remaining is not None and remaining <= float(self.args.lookahead_m) + float(margin or 0.0))
        records: List[Dict[str, Any]] = []
        productive = 0
        for v in v_samples:
            for w in w_samples:
                if float(v) <= DWA_NUMERIC_EPS or below_minimum_forward(float(v), minimum_forward):
                    continue
                if capability is None or not capability.contains(float(v), float(w)):
                    continue
                if exceeds_speed_limit(float(v), speed_limit):
                    continue
                ok, _clearance, endpoint = self.collision_free_arc_at_yaw_offset(
                    grid_msg, blocked, float(v), float(w), float(yaw_offset_rad),
                )
                if not ok:
                    continue
                if anchor is not None:
                    projection = project_endpoint_to_path_station(
                        transformed_path, anchor, endpoint,
                        rollout_arc_length_m=float(v) * float(self.args.dwa_predict_time),
                    )
                    evidence = evaluate_candidate_progress(
                        anchor, projection, current_target_xy=transformed_target,
                        current_local_xy=transformed_waypoint, endpoint_xy=endpoint,
                    )
                else:
                    evidence = evaluate_candidate_direct_progress(
                        current_target_xy=transformed_target,
                        current_local_xy=transformed_waypoint, endpoint_xy=endpoint,
                    )
                admission = classify_room_local_productivity(
                    evidence, direct_or_short_context=bool(direct_or_short),
                    meaningful_progress_margin_m=float(margin or 0.0),
                )
                records.append({
                    "v": float(v), "w": float(w), "p_path_m": evidence.p_path_m,
                    "p_target_m": evidence.p_target_m, "p_local_m": evidence.p_local_m,
                    "projection_reason": evidence.projection_reason,
                    "productivity_classification": admission.classification,
                    "score_eligible": admission.score_eligible,
                })
                productive += int(admission.score_eligible)
        return productive, records

    def build_recovery_set(
        self,
        grid_msg: OccupancyGrid,
        blocked: np.ndarray,
        waypoint_xy: Tuple[float, float],
        target_base_xy: Tuple[float, float],
        distance_to_goal: float,
        room_local_path_xy: Sequence[Tuple[float, float]],
    ) -> RecoverySet:
        """Bounded frozen counterfactual; fresh post-slice input remains truth."""
        _v_samples, w_samples = self.dynamic_window()
        margin = derive_angular_recovery_margin(w_samples, float(self.args.command_slice_sec))
        hypotheses: List[Tuple[float, bool]] = []
        records: List[Dict[str, Any]] = []
        for w in w_samples:
            yaw = float(w) * float(self.args.dwa_predict_time)
            rotation_safe = (
                abs(float(w)) <= float(self.args.max_angular_z) + DWA_NUMERIC_EPS
                and self.collision_free_arc(grid_msg, blocked, 0.0, float(w))[0]
            )
            if rotation_safe:
                productive, candidate_records = self.room_local_productive_count_at_yaw(
                    grid_msg, blocked, waypoint_xy, target_base_xy, distance_to_goal, room_local_path_xy, yaw,
                )
            else:
                productive, candidate_records = 0, []
            hypotheses.append((yaw, productive > 0))
            records.append({
                "yaw_hypothesis_rad": yaw, "angular_sample_radps": float(w),
                "rotate_only_collision_safe": bool(rotation_safe),
                "productive_translational_candidate_count": productive,
                "candidates": candidate_records,
            })
        return RecoverySet(
            intervals=recovery_intervals_from_hypotheses(hypotheses, margin),
            angular_margin_rad=margin,
            hypothesis_records=tuple(records),
            approximate=True,
        )

    def orientation_route_context(
        self,
        target: Dict[str, Any],
        pose: Tuple[float, float, float],
        room_local_path_xy: Sequence[Tuple[float, float]],
    ) -> Tuple[Tuple[Any, ...], Optional[float]]:
        resolution = max(float(self.args.grid_resolution_m), DWA_NUMERIC_EPS)
        target_xy = target.get("target_xy_team_livox_odom")
        if not (isinstance(target_xy, (list, tuple)) and len(target_xy) == 2 and all(finite_number(value) for value in target_xy)):
            target_xy = (0.0, 0.0)
        target_key = (
            str(target.get("source") or ""),
            int(round(float(target_xy[0]) / resolution)),
            int(round(float(target_xy[1]) / resolution)),
        )
        tangent = None
        for first, second in zip(room_local_path_xy, room_local_path_xy[1:]):
            dx, dy = float(second[0]) - float(first[0]), float(second[1]) - float(first[1])
            if math.hypot(dx, dy) > DWA_NUMERIC_EPS:
                tangent = normalize_angle(math.atan2(dy, dx) + float(pose[2]))
                break
        return target_key, tangent

    def orientation_intent_materially_changed(
        self,
        intent: OrientationIntent,
        target_key: Tuple[Any, ...],
        pose: Tuple[float, float, float],
        tangent_odom_rad: Optional[float],
        angular_margin_rad: Optional[float],
    ) -> bool:
        locality = math.sqrt(2.0) * float(self.args.grid_resolution_m)
        if target_key != intent.target_key:
            return True
        if math.hypot(float(pose[0]) - intent.anchor_odom_xy[0], float(pose[1]) - intent.anchor_odom_xy[1]) > locality + DWA_NUMERIC_EPS:
            return True
        if (intent.anchor_tangent_odom_rad is None) != (tangent_odom_rad is None):
            return True
        if intent.anchor_tangent_odom_rad is None:
            return False
        return abs(normalize_angle(float(tangent_odom_rad) - float(intent.anchor_tangent_odom_rad))) > float(angular_margin_rad or math.pi)

    def phase3_recovery_action(
        self,
        *,
        target: Dict[str, Any],
        pose: Tuple[float, float, float],
        grid_msg: OccupancyGrid,
        blocked: np.ndarray,
        waypoint_xy: Tuple[float, float],
        target_base_xy: Tuple[float, float],
        distance_to_goal: float,
        room_local_path_xy: Sequence[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """Return exactly one evidence-bounded rotate action or truthful failure."""
        recovery_set = self.build_recovery_set(
            grid_msg, blocked, waypoint_xy, target_base_xy, distance_to_goal, room_local_path_xy,
        )
        target_key, tangent = self.orientation_route_context(target, pose, room_local_path_xy)
        intent = getattr(self, "orientation_intent", None)
        if intent is not None and self.orientation_intent_materially_changed(
            intent, target_key, pose, tangent, recovery_set.angular_margin_rad,
        ):
            self.orientation_intent = None
            intent = None
        if intent is not None and intent.consumed:
            return {"action": "FAIL", "reason": "ORIENTATION_INTENT_ALREADY_CONSUMED", "recovery_set": recovery_set}
        if not recovery_set.intervals or recovery_set.angular_margin_rad is None:
            if intent is not None:
                intent.consumed = True
            return {"action": "FAIL", "reason": "RECOVERYSET_EMPTY_OR_UNRESOLVED", "recovery_set": recovery_set}
        if intent is None:
            before = recovery_distance(0.0, recovery_set.intervals)
            if before is None or before <= float(recovery_set.angular_margin_rad):
                return {"action": "FAIL", "reason": "RECOVERYSET_ALREADY_REACHED_WITHOUT_TRANSLATION", "recovery_set": recovery_set}
            intent = OrientationIntent(
                target_key=target_key,
                anchor_odom_xy=(float(pose[0]), float(pose[1])),
                anchor_tangent_odom_rad=tangent,
                entry_yaw_rad=float(pose[2]),
                recovery_intervals=recovery_set.intervals,
                chosen_direction=0,
                recovery_distance_before_rad=float(before),
                last_recovery_distance_rad=float(before),
            )
            self.orientation_intent = intent
        current_relative_yaw = normalize_angle(float(pose[2]) - float(intent.entry_yaw_rad))
        current_distance = recovery_distance(current_relative_yaw, intent.recovery_intervals)
        if current_distance is None:
            intent.consumed = True
            return {"action": "FAIL", "reason": "RECOVERYSET_LOST", "recovery_set": recovery_set, "intent": intent}
        if intent.awaiting_fresh_replan:
            if current_distance + DWA_NUMERIC_EPS >= intent.last_recovery_distance_rad - float(recovery_set.angular_margin_rad):
                intent.consumed = True
                return {"action": "FAIL", "reason": "RECOVERY_DISTANCE_NOT_MEANINGFULLY_REDUCED", "recovery_set": recovery_set, "intent": intent}
            intent.awaiting_fresh_replan = False
        if current_distance <= float(recovery_set.angular_margin_rad):
            intent.consumed = True
            return {"action": "FAIL", "reason": "RECOVERYSET_REACHED_TRANSLATION_STILL_EMPTY", "recovery_set": recovery_set, "intent": intent}
        _v_samples, w_samples = self.dynamic_window()
        safe_w_samples: List[float] = []
        for w in w_samples:
            if abs(float(w)) <= DWA_NUMERIC_EPS or abs(float(w)) > float(self.args.max_angular_z) + DWA_NUMERIC_EPS:
                continue
            safe, _clearance = self.collision_free_arc(grid_msg, blocked, 0.0, float(w))
            if safe:
                safe_w_samples.append(float(w))
        selection = select_orientation_slice(
            w_samples, safe_w_samples, intent.recovery_intervals,
            current_relative_yaw_rad=current_relative_yaw,
            command_slice_sec=float(self.args.command_slice_sec),
            angular_margin_rad=recovery_set.angular_margin_rad,
        )
        if selection is None:
            intent.consumed = True
            return {"action": "FAIL", "reason": "NO_SAFE_USEFUL_ORIENTATION_SLICE", "recovery_set": recovery_set, "intent": intent}
        intent.chosen_direction = int(selection["direction"])
        intent.last_recovery_distance_rad = float(selection["before_rad"])
        intent.awaiting_fresh_replan = True
        return {"action": "ORIENTATION_SLICE", "selection": selection, "recovery_set": recovery_set, "intent": intent}

    def apply_continuation_aware_local_selection(
        self,
        dwa: Dict[str, Any],
        native_v: float,
        native_w: float,
        evidence: Mapping[str, Any],
        *,
        phase3_recovery_context: bool,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Make the final local winner authoritative without changing DWA scoring."""
        decision = continuation_aware_admissible_selection(
            dwa.get("room_local_productivity_candidates") or [],
            evidence,
            native_v=float(native_v), native_w=float(native_w),
            phase3_recovery_context=bool(phase3_recovery_context),
        )
        dwa["continuation_native_provisional_winner"] = dict(decision["native_provisional_winner"])
        dwa["continuation_local_selection"] = decision
        winner = decision.get("final_winner")
        if not isinstance(winner, Mapping):
            return float(native_v), float(native_w), decision
        final_v, final_w = float(winner["v"]), float(winner["w"])
        # This is the final *local-selection* winner, not necessarily the
        # command that will be published.  Existing higher-priority command
        # guards run after this seam.  `finalize_command_authority()` writes
        # `prev_cmd` only after all of them have had their chance to act.
        dwa["selected_linear_x"] = final_v
        dwa["selected_angular_z"] = final_w
        dwa["continuation_final_winner"] = {
            "v": final_v,
            "w": final_w,
            "candidate_index": int(winner["candidate_index"]),
            "continuation_status": winner["continuation_status"],
            "final_dwa_score": float(winner["final_dwa_score"]),
            "authority_scope": "PROVISIONAL_LOCAL_SELECTION",
            "execution_binding": "PENDING_FINAL_COMMAND_AUTHORITY",
        }
        return final_v, final_w, decision

    def finalize_command_authority(
        self,
        dwa: Dict[str, Any],
        *,
        provisional_v: float,
        provisional_w: float,
        final_v: float,
        final_w: float,
        final_command_reason: str,
        mutation_chain: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Freeze the only command identity allowed to reach publication.

        DWA and Continuation may identify a useful local translation before
        existing command guards run.  That evidence remains valuable, but it
        must never be confused with the command actually requested from the
        publisher or used as the next dynamic-window history.
        """
        provisional = {"v": float(provisional_v), "w": float(provisional_w)}
        final = {"v": float(final_v), "w": float(final_w)}
        overridden = bool(
            abs(float(provisional_v) - float(final_v)) > DWA_NUMERIC_EPS
            or abs(float(provisional_w) - float(final_w)) > DWA_NUMERIC_EPS
        )
        local_winner = dwa.get("continuation_final_winner")
        if isinstance(local_winner, dict):
            local_winner["execution_binding"] = (
                "EXECUTED_AS_FINAL_COMMAND"
                if not overridden else "NOT_EXECUTED_HIGHER_PRIORITY_OVERRIDE"
            )
            local_winner["final_command_reason"] = str(final_command_reason)
        action = {
            "authority_scope": "FINAL_COMMAND_AUTHORITY",
            "v": float(final_v),
            "w": float(final_w),
            "final_command_reason": str(final_command_reason),
            "provisional_local_motion": provisional,
            "provisional_local_motion_executed": not overridden,
            "mutation_chain": [copy.deepcopy(dict(item)) for item in mutation_chain],
            "publish_result": "PENDING",
            "published_count": None,
        }
        dwa["provisional_local_motion"] = provisional
        dwa["final_command_action"] = action
        # `prev_cmd` determines the next epoch's dynamic window and reversal
        # penalty, so it must represent this frozen final command rather than
        # any provisional local winner.
        self.prev_cmd = (float(final_v), float(final_w))
        dwa["selected_linear_x"] = float(final_v)
        dwa["selected_angular_z"] = float(final_w)
        return action

    @staticmethod
    def record_final_command_publish_result(action: Dict[str, Any], published: int, *, execute: bool) -> None:
        """Attach factual publication outcome without changing command authority."""
        action["published_count"] = int(published)
        action["publish_result"] = (
            "PUBLISHED" if int(published) > 0 else
            "NOT_PUBLISHED_DRY_RUN" if not bool(execute) else
            "NOT_PUBLISHED"
        )

    def close_phase3_orientation_intent_for_continuation(
        self,
        selection: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Consume and detach exactly the current Phase3 recovery episode."""
        before = orientation_intent_record(getattr(self, "orientation_intent", None))
        if self.orientation_intent is not None:
            self.orientation_intent.consumed = True
        self.orientation_intent = None
        return {
            "outcome": selection.get("outcome"),
            "orientation_intent_before": before,
            "orientation_intent_after": None,
            "translation_allowed": bool(selection.get("translation_allowed")),
        }

    def observation_orientation_slice_action(
        self,
        *,
        pose: Tuple[float, float, float],
        grid_msg: OccupancyGrid,
        blocked: np.ndarray,
    ) -> Dict[str, Any]:
        """Admit one OA mission-orientation slice without Phase-3 state.

        The current runner owns the angular lattice, collision predicate, and
        command publisher.  This method deliberately does not inspect or
        mutate ``orientation_intent`` or any RecoverySet field.
        """
        aim = getattr(self.args, "observation_orientation_aim_yaw_odom_rad", None)
        if not finite_number(aim):
            return {"action": "ORIENTATION_UNAVAILABLE", "reason": "OBSERVATION_AIM_INVALID"}
        _v_samples, w_samples = self.dynamic_window()
        safe_w_samples: List[float] = []
        safety_records: List[Dict[str, Any]] = []
        for w in w_samples:
            if abs(float(w)) <= DWA_NUMERIC_EPS or abs(float(w)) > float(self.args.max_angular_z) + DWA_NUMERIC_EPS:
                continue
            safe, clearance = self.collision_free_arc(grid_msg, blocked, 0.0, float(w))
            safety_records.append({"w_radps": float(w), "collision_safe": bool(safe), "clearance_m": float(clearance)})
            if safe:
                safe_w_samples.append(float(w))
        selection = select_observation_orientation_slice(
            w_samples,
            safe_w_samples,
            current_yaw_odom_rad=float(pose[2]),
            aim_yaw_odom_rad=float(aim),
            command_slice_sec=float(self.args.command_slice_sec),
        )
        if selection is None:
            return {
                "action": "ORIENTATION_UNAVAILABLE",
                "reason": "NO_SAFE_AIM_IMPROVING_ORIENTATION_SLICE",
                "aim_yaw_odom_rad": float(aim),
                "safety_records": safety_records,
            }
        return {
            "action": "OBSERVATION_ORIENTATION_ONE_SLICE",
            "aim_yaw_odom_rad": float(aim),
            "selection": selection,
            "safety_records": safety_records,
            "phase3_state_touched": False,
        }

    @staticmethod
    def message_stamp_sec(msg: Any) -> Optional[float]:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return None
        try:
            return float(stamp.to_sec())
        except Exception:
            return None

    def build_dwa_no_cmd_failure_snapshot(
        self,
        *,
        step_index: int,
        odom: Odometry,
        grid_msg: OccupancyGrid,
        status_payload: Dict[str, Any],
        target: Dict[str, Any],
        pose: Tuple[float, float, float],
        target_base_xy: Tuple[float, float],
        waypoint_base_xy: Tuple[float, float],
        raw_path: Sequence[Tuple[int, int]],
        path: Sequence[Tuple[int, int]],
        raw_grid: np.ndarray,
        occupied_inflated: np.ndarray,
        planning_blocked: np.ndarray,
        distance_to_goal: float,
        dwa_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record a failed decision by rerunning the same collision checker only.

        This is called only after the production DWA has already returned
        ``blocked``.  It does not publish, score, select, or update ``prev_cmd``.
        """
        raw_v_samples, w_samples = self.dynamic_window()
        minimum_forward = effective_minimum_forward(float(self.args.min_linear_x))
        speed_limit = speed_limit_for_distance(
            self.args.max_linear_x,
            minimum_forward,
            distance_to_goal,
            self.args.distance_speed_gain,
        )
        v_samples = v_samples_with_legal_boundaries(raw_v_samples, minimum_forward, speed_limit)
        candidates: List[Dict[str, Any]] = []
        for v in v_samples:
            for w in w_samples:
                record: Dict[str, Any] = {"v": float(v), "w": float(w)}
                if below_minimum_forward(float(v), minimum_forward):
                    record.update({
                        "result": "SKIPPED_MIN_FORWARD",
                        "first_rejected_sample": None,
                    })
                    candidates.append(record)
                    continue
                trace: Dict[str, Any] = {}
                # Exact production checker and exact planning mask used above.
                collision_free, clearance_m = self.collision_free_arc(
                    grid_msg, planning_blocked, float(v), float(w), trace
                )
                if not collision_free:
                    sample = trace.get("first_rejected_sample")
                    if isinstance(sample, dict):
                        cell = sample.get("cell")
                        if isinstance(cell, list) and len(cell) == 2:
                            x_index, y_index = int(cell[0]), int(cell[1])
                            raw_value = int(raw_grid[y_index, x_index])
                            sample["cell_state"] = {
                                "raw_value": raw_value,
                                "raw_occupied": bool(raw_value > 0),
                                "unknown": bool(raw_value < 0),
                                "oob": False,
                                "occupied_inflated": bool(occupied_inflated[y_index, x_index]),
                                "planning_blocked": bool(planning_blocked[y_index, x_index]),
                            }
                        else:
                            sample["cell_state"] = {
                                "raw_value": None,
                                "raw_occupied": None,
                                "unknown": None,
                                "oob": True,
                                "occupied_inflated": None,
                                "planning_blocked": None,
                            }
                    record.update({
                        "result": "REJECTED_COLLISION",
                        "first_rejected_sample": sample,
                    })
                elif exceeds_speed_limit(float(v), speed_limit):
                    record.update({
                        "result": "REJECTED_SPEED_LIMIT",
                        "clearance_m": float(clearance_m),
                        "first_rejected_sample": None,
                    })
                else:
                    record.update({
                        "result": "ACCEPTED_BY_COLLISION_AND_SPEED_GATE",
                        "clearance_m": float(clearance_m),
                        "first_rejected_sample": None,
                    })
                candidates.append(record)
        return {
            "schema_version": 1,
            "trigger": "BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD",
            "step_index": int(step_index),
            "grid": {
                "header_stamp_sec": self.message_stamp_sec(grid_msg),
                "frame_id": str(getattr(getattr(grid_msg, "header", None), "frame_id", "")),
                "width": int(grid_msg.info.width),
                "height": int(grid_msg.info.height),
                "resolution_m": float(grid_msg.info.resolution),
            },
            "status_grid_identity": {
                "grid_content_stamp": status_payload.get("grid_content_stamp"),
                "content_generation_id": status_payload.get("content_generation_id"),
                "grid_content_hash": status_payload.get("grid_content_hash"),
            },
            "odom": {
                "header_stamp_sec": self.message_stamp_sec(odom),
                "pose_x_y_yaw": [float(pose[0]), float(pose[1]), float(pose[2])],
            },
            "target": {
                "source": target.get("source"),
                "subgoal_source": target.get("subgoal_source"),
                "target_odom_xy": target.get("target_xy_team_livox_odom"),
                "target_base_xy": [float(target_base_xy[0]), float(target_base_xy[1])],
                "waypoint_base_xy": [float(waypoint_base_xy[0]), float(waypoint_base_xy[1])],
                "distance_to_goal_m": float(distance_to_goal),
            },
            "astar": {
                "raw_path_cells": [[int(x), int(y)] for x, y in raw_path],
                "smoothed_path_cells": [[int(x), int(y)] for x, y in path],
            },
            "dynamic_window": {
                "previous_command": [float(self.prev_cmd[0]), float(self.prev_cmd[1])],
                "v_samples": [float(v) for v in v_samples],
                "w_samples": [float(w) for w in w_samples],
                "minimum_forward_mps": float(minimum_forward),
                "speed_limit_mps": float(speed_limit),
            },
            "dwa_result": dict(dwa_result),
            "candidates": candidates,
            "candidate_counts": {
                "total": len(candidates),
                "rejected_collision": sum(c["result"] == "REJECTED_COLLISION" for c in candidates),
                "rejected_speed_limit": sum(c["result"] == "REJECTED_SPEED_LIMIT" for c in candidates),
                "skipped_min_forward": sum(c["result"] == "SKIPPED_MIN_FORWARD" for c in candidates),
                "accepted_by_collision_and_speed_gate": sum(
                    c["result"] == "ACCEPTED_BY_COLLISION_AND_SPEED_GATE" for c in candidates
                ),
            },
            "observability_contract": {
                "grid_mask": "exact_planning_blocked_passed_to_choose_dwa",
                "collision_checker": "collision_free_arc",
                "navigation_output_modified": False,
            },
        }

    def write_dwa_no_cmd_failure_snapshot(self, snapshot: Dict[str, Any]) -> Path:
        archive_dir = os.environ.get("STATE_MACHINE_RUN_ARCHIVE_DIR")
        path = (Path(archive_dir) if archive_dir else OUT) / "dwa_no_cmd_failure_snapshot.json"
        write_json(path, snapshot)
        return path

    def execution_qualified_candidate_set(self) -> List[Tuple[float, float, str]]:
        """The R40 guarded mode exposes only the measured Stair primitives."""
        return [
            (0.0, 0.0, "STOP"),
            (0.30, 0.0, "FORWARD"),
            (0.30, 0.14, "LEFT"),
            (0.30, -0.14, "RIGHT"),
        ]

    def execution_qualified_primitive_name(self, v: float, w: float) -> Optional[str]:
        for candidate_v, candidate_w, name in self.execution_qualified_candidate_set():
            if abs(float(v) - candidate_v) <= 1e-6 and abs(float(w) - candidate_w) <= 1e-6:
                return name
        return None

    def execution_qualified_envelope_branches(self, primitive: str) -> List[List[Tuple[float, float]]]:
        """Conservative R40 command-slice plus zero-tail centre paths in base frame.

        The Grid is already inflated by the frozen robot radius.  These branches
        therefore represent centre paths, not a second footprint inflation.
        """
        if primitive == "STOP":
            return [[(0.0, 0.0)]]
        if primitive == "FORWARD":
            # Two .30/0 repetitions: 0.323 m/s peak slice progress, 0.123 m
            # observed zero-tail path, and small heading excursions on either side.
            return [
                [(0.0, 0.0), (0.162, 0.000), (0.285, 0.006)],
                [(0.0, 0.0), (0.162, 0.000), (0.285, -0.006)],
            ]
        # The R40 left command and its stop tail supply the positive branch.
        # RIGHT is its conservative geometric mirror in this user-authorized
        # experimental mode; it remains distinguishable in runner diagnostics.
        sign = 1.0 if primitive == "LEFT" else -1.0
        return [
            [(0.0, 0.0), (0.154, sign * 0.014), (0.247, sign * 0.042)],
            [(0.0, 0.0), (0.154, sign * 0.008), (0.247, sign * 0.036)],
        ]

    def collision_free_execution_qualified_envelope(
        self,
        grid_msg: OccupancyGrid,
        blocked: np.ndarray,
        v: float,
        w: float,
    ) -> Tuple[bool, float]:
        primitive = self.execution_qualified_primitive_name(v, w)
        if primitive is None:
            return False, 0.0
        if abs(float(self.args.command_slice_sec) - 0.50) > 1e-6:
            # R40 records a 0.50 s command slice followed by its measured tail.
            return False, 0.0
        obstacle_cells = np.argwhere(blocked)
        min_clearance = float("inf")
        for branch in self.execution_qualified_envelope_branches(primitive):
            for start, end in zip(branch, branch[1:]):
                segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
                steps = max(1, int(math.ceil(segment_length / max(float(grid_msg.info.resolution) * 0.5, 1e-6))))
                for index in range(steps + 1):
                    ratio = float(index) / float(steps)
                    x = start[0] + ratio * (end[0] - start[0])
                    y = start[1] + ratio * (end[1] - start[1])
                    cell = self.local_xy_to_cell(x, y, grid_msg)
                    if cell is None:
                        return False, 0.0
                    x_index, y_index = cell
                    if blocked[y_index, x_index]:
                        return False, 0.0
                    if obstacle_cells.size:
                        distance = np.sqrt(((obstacle_cells - np.array([y_index, x_index])) ** 2).sum(axis=1)).min()
                        min_clearance = min(min_clearance, float(distance) * float(grid_msg.info.resolution))
        if min_clearance == float("inf"):
            min_clearance = self.args.max_clearance_score_m
        return True, min_clearance

    def execution_qualified_predicted_heading(self, primitive: Optional[str]) -> Optional[float]:
        if primitive == "LEFT":
            return 0.164
        if primitive == "RIGHT":
            return -0.164
        if primitive in {"FORWARD", "STOP"}:
            return 0.0
        return None

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            prev_cell = smoothed[-1]
            cur = path[i]
            nxt = path[i + 1]
            prev_dir = (cur[0] - prev_cell[0], cur[1] - prev_cell[1])
            next_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            if prev_dir != next_dir:
                smoothed.append(cur)
        smoothed.append(path[-1])
        return smoothed

    def select_lookahead_waypoint(self, path: List[Tuple[int, int]], grid_msg: OccupancyGrid) -> Tuple[float, float, int]:
        chosen = path[-1]
        chosen_idx = len(path) - 1
        for i, cell in enumerate(path[1:], start=1):
            xy = self.cell_to_local_xy(cell, grid_msg)
            if math.hypot(xy[0], xy[1]) >= self.args.lookahead_m:
                chosen = cell
                chosen_idx = i
                break
        return (*self.cell_to_local_xy(chosen, grid_msg), chosen_idx)

    def predicted_arc_endpoint_base(self, v: float, w: float) -> Tuple[float, float]:
        """Integrate a candidate endpoint with the existing DWA horizon and step."""
        x = 0.0
        y = 0.0
        yaw = 0.0
        steps = max(1, int(self.args.dwa_predict_time / self.args.dwa_dt))
        for _ in range(steps):
            x += float(v) * math.cos(yaw) * self.args.dwa_dt
            y += float(v) * math.sin(yaw) * self.args.dwa_dt
            yaw += float(w) * self.args.dwa_dt
        return x, y

    def select_p_through_executable_lookahead(
        self,
        path: List[Tuple[int, int]],
        grid_msg: OccupancyGrid,
        blocked: np.ndarray,
        native_lookahead: Tuple[float, float, int],
        distance_to_goal: float,
    ) -> Dict[str, Any]:
        """Choose an existing A* waypoint that has a legal moving DWA direction.

        This is deliberately a feasibility gate, not a second DWA scorer.  It
        visits the native lookahead first and otherwise only points already on
        the same A* path.  Each gate candidate reuses the current dynamic
        window and the authoritative collision_free_arc() implementation.
        """
        native_x, native_y, native_idx = native_lookahead
        candidate_indices = [int(native_idx)]
        candidate_indices.extend(index for index in range(1, len(path)) if index != int(native_idx))
        minimum_forward = effective_minimum_forward(float(self.args.min_linear_x))
        raw_v_samples, w_samples = self.dynamic_window()
        speed_limit = speed_limit_for_distance(
            self.args.max_linear_x,
            minimum_forward,
            distance_to_goal,
            self.args.distance_speed_gain,
        )
        v_samples = v_samples_with_legal_boundaries(raw_v_samples, minimum_forward, speed_limit)

        def safe_forward_count(waypoint_xy: Tuple[float, float]) -> int:
            # G2 and G4: a waypoint must be in the usable forward half-plane,
            # and a predicted legal arc must make positive progress toward it.
            if float(waypoint_xy[0]) <= 0.0:
                return 0
            count = 0
            for v in v_samples:
                if below_minimum_forward(float(v), minimum_forward) or exceeds_speed_limit(float(v), speed_limit):
                    continue
                for w in w_samples:
                    ok, _clearance = self.collision_free_arc(grid_msg, blocked, float(v), float(w))
                    if not ok:
                        continue
                    end_x, end_y = self.predicted_arc_endpoint_base(float(v), float(w))
                    if end_x * float(waypoint_xy[0]) + end_y * float(waypoint_xy[1]) <= 0.0:
                        continue
                    count += 1
            return count

        native_count: Optional[int] = None
        for index in candidate_indices:
            waypoint_xy = self.cell_to_local_xy(path[index], grid_msg)
            count = safe_forward_count(waypoint_xy)
            if index == int(native_idx):
                native_count = count
            if count > 0:
                return {
                    "p_through_executable_lookahead_active": True,
                    "native_lookahead_xy": [float(native_x), float(native_y)],
                    "native_lookahead_forward_executable": bool((native_count or 0) > 0),
                    "selected_lookahead_xy": [float(waypoint_xy[0]), float(waypoint_xy[1])],
                    "selected_lookahead_path_index": int(index),
                    "lookahead_changed": bool(index != int(native_idx)),
                    "lookahead_candidate_count": len(candidate_indices),
                    "safe_forward_count_for_selected_lookahead": int(count),
                    "lookahead_selection_reason": (
                        "native_lookahead_forward_executable"
                        if index == int(native_idx)
                        else "same_astar_path_forward_executable_alternative"
                    ),
                }
        return {
            "p_through_executable_lookahead_active": True,
            "native_lookahead_xy": [float(native_x), float(native_y)],
            "native_lookahead_forward_executable": bool((native_count or 0) > 0),
            "selected_lookahead_xy": None,
            "selected_lookahead_path_index": None,
            "lookahead_changed": False,
            "lookahead_candidate_count": len(candidate_indices),
            "safe_forward_count_for_selected_lookahead": 0,
            "lookahead_selection_reason": "P_THROUGH_ASTAR_PATH_NOT_FORWARD_EXECUTABLE",
        }

    def dynamic_window(self) -> Tuple[np.ndarray, np.ndarray]:
        prev_v, prev_w = self.prev_cmd
        dt = max(self.args.command_slice_sec, 1e-3)
        v_min = max(0.0, prev_v - self.args.max_linear_accel * dt)
        v_max = min(self.args.max_linear_x, prev_v + self.args.max_linear_accel * dt)
        w_min = max(-self.args.max_angular_z, prev_w - self.args.max_angular_accel * dt)
        w_max = min(self.args.max_angular_z, prev_w + self.args.max_angular_accel * dt)
        minimum_forward = effective_minimum_forward(float(self.args.min_linear_x))
        if v_max < minimum_forward and float(self.args.max_linear_x) >= minimum_forward:
            v_max = minimum_forward
        return np.linspace(v_min, v_max, self.args.linear_samples), np.linspace(w_min, w_max, self.args.angular_samples)

    def choose_dwa(
        self,
        grid_msg: OccupancyGrid,
        blocked: np.ndarray,
        waypoint_xy: Tuple[float, float],
        target_base_xy: Tuple[float, float],
        distance_to_goal: float,
        wall_heading_prior: Optional[Dict[str, Any]] = None,
        *,
        p_through_safe_moving_eligibility: bool = False,
        room_search_safe_moving_eligibility: bool = False,
        p_through_target_relative_angular_scoring: bool = False,
        portal_relative_selection: bool = False,
        portal_center_odom: Optional[Sequence[float]] = None,
        portal_normal_odom: Optional[Sequence[float]] = None,
        portal_p_pre_odom: Optional[Sequence[float]] = None,
        pose_odom: Optional[Tuple[float, float, float]] = None,
        target_in_front: bool = False,
        astar_path_exists: bool = False,
        room_local_path_xy: Optional[Sequence[Tuple[float, float]]] = None,
        collision_reference_pose_base: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        best_score = -float("inf")
        best = (0.0, 0.0)
        samples: List[Dict[str, Any]] = []
        high_level_shadow_samples: List[Dict[str, Any]] = []
        portal_relative_selection_active = bool(
            portal_relative_selection
            and pose_odom is not None
            and portal_center_odom is not None
            and portal_normal_odom is not None
        )
        waypoint_heading = math.atan2(waypoint_xy[1], waypoint_xy[0])
        final_target_heading = math.atan2(target_base_xy[1], max(target_base_xy[0], 1e-6))
        target_lateral_abs = abs(target_base_xy[1])
        correction_active = target_lateral_abs >= self.args.target_lateral_correction_threshold_m
        correction_heading = max(
            -self.args.max_target_heading_correction_rad,
            min(self.args.max_target_heading_correction_rad, final_target_heading),
        )
        if correction_active:
            target_heading = normalize_angle(
                (1.0 - self.args.target_heading_blend_weight) * waypoint_heading
                + self.args.target_heading_blend_weight * correction_heading
            )
        else:
            target_heading = waypoint_heading
        wall_heading_active = bool(wall_heading_prior and wall_heading_prior.get("active"))
        if wall_heading_active:
            target_heading = blend_angles(
                target_heading,
                float(wall_heading_prior["heading_parallel_rad"]),
                self.args.pointcloud_wall_heading_blend_weight,
            )
        target_relative_angular_scoring_active = bool(p_through_target_relative_angular_scoring)
        angular_reference_time_sec: Optional[float] = None
        w_reference: Optional[float] = None
        angular_penalty_mode = "absolute_w_magnitude"
        if target_relative_angular_scoring_active:
            angular_reference_time_sec = max(
                float(self.args.dwa_predict_time),
                float(self.args.max_target_heading_correction_rad)
                / max(float(self.args.max_angular_z), 1e-6),
            )
            w_reference = max(
                -float(self.args.max_angular_z),
                min(
                    float(self.args.max_angular_z),
                    float(target_heading) / float(angular_reference_time_sec),
                ),
            )
            angular_penalty_mode = "p_through_target_relative"
        target_dist = math.hypot(waypoint_xy[0], waypoint_xy[1])
        raw_v_samples, w_samples = self.dynamic_window()
        minimum_forward = effective_minimum_forward(float(self.args.min_linear_x))
        speed_limit = speed_limit_for_distance(
            self.args.max_linear_x,
            minimum_forward,
            distance_to_goal,
            self.args.distance_speed_gain,
        )
        v_samples = v_samples_with_legal_boundaries(raw_v_samples, minimum_forward, speed_limit)
        candidate_samples = [(float(v), float(w), None) for v in v_samples for w in w_samples]
        accepted_moving_samples: List[Dict[str, Any]] = []
        phase2_productive_active = bool(getattr(self, "phase2_productive_admission_active", False))
        progress_margin_m = derive_meaningful_progress_margin(
            float(self.args.grid_resolution_m), minimum_forward, float(self.args.dwa_dt),
        ) if phase2_productive_active else None
        room_local_capability = room_local_capability_by_id("forward_turn_current_room_profile")
        path_xy = tuple(room_local_path_xy or ())
        path_anchor: Optional[PathProgressAnchor] = None
        path_remaining_m: Optional[float] = None
        if phase2_productive_active and len(path_xy) >= 2 and progress_margin_m is not None:
            path_anchor = build_path_progress_anchor(
                path_xy,
                path_identity="room_local_decision_path",
                robot_xy=(0.0, 0.0),
                grid_resolution_m=float(self.args.grid_resolution_m),
                rollout_step_m=minimum_forward * float(self.args.dwa_dt),
            )
            if path_anchor is not None:
                path_remaining_m = max(0.0, float(path_anchor.stations_m[-1]) - float(path_anchor.s_current_m))
        direct_or_short_context = bool(
            phase2_productive_active
            and (path_anchor is None or (path_remaining_m is not None and path_remaining_m <= float(self.args.lookahead_m) + float(progress_margin_m)))
        )
        room_local_candidate_records: List[Dict[str, Any]] = []
        safe_translational_candidate_count = 0
        productive_translational_candidate_count = 0
        # This opt-in sink observes the exact collision-accepted arcs below.
        # It never feeds score, eligibility, best selection, or publication.
        high_level_shadow_capture = bool(
            room_search_safe_moving_eligibility
            and os.environ.get("ROOM_SEARCH_HIGH_LEVEL_LOCOMOTION_COMPATIBILITY_SHADOW_V0", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        for v, w, primitive_name in candidate_samples:
                if below_minimum_forward(float(v), minimum_forward):
                    continue
                if phase2_productive_active and float(v) <= DWA_NUMERIC_EPS:
                    # Rotate-only is outside Phase-2 translational admission.
                    continue
                if phase2_productive_active and (
                    room_local_capability is None or not room_local_capability.contains(float(v), float(w))
                ):
                    room_local_candidate_records.append({
                        "v": float(v), "w": float(w), "motion_class": "FORWARD_TURN",
                        "productivity_classification": "CAPABILITY_UNQUALIFIED",
                        "productivity_reason": "CAPABILITY_UNQUALIFIED",
                        "score_eligible": False,
                    })
                    continue
                arc_trace: Optional[Dict[str, Any]] = {} if (high_level_shadow_capture or phase2_productive_active) else None
                # Keep the native call shape byte-for-byte equivalent when
                # C0's virtual S1 frame is not in use.  Besides avoiding
                # needless argument churn, existing test/dry seams may supply
                # the historic collision callable without the C0 extension.
                if arc_trace is None and collision_reference_pose_base is None:
                    ok, clearance = self.collision_free_arc(grid_msg, blocked, float(v), float(w))
                elif arc_trace is not None and collision_reference_pose_base is None:
                    ok, clearance = self.collision_free_arc(
                        grid_msg, blocked, float(v), float(w), arc_trace,
                    )
                elif arc_trace is None:
                    ok, clearance = self.collision_free_arc(
                        grid_msg, blocked, float(v), float(w),
                        reference_pose_base=collision_reference_pose_base,
                    )
                else:
                    ok, clearance = self.collision_free_arc(
                        grid_msg, blocked, float(v), float(w), arc_trace,
                        reference_pose_base=collision_reference_pose_base,
                    )
                if not ok:
                    if phase2_productive_active:
                        room_local_candidate_records.append({
                            "v": float(v), "w": float(w),
                            "curvature_m_inv": (float(w) / float(v)) if abs(float(v)) > DWA_NUMERIC_EPS else None,
                            "motion_class": "FORWARD_STRAIGHT" if abs(float(w)) <= DWA_NUMERIC_EPS else "FORWARD_TURN",
                            "capability_id": room_local_capability.capability_id if room_local_capability else None,
                            "collision_safe": False, "clearance_m": 0.0,
                            "endpoint_base_xy": arc_trace.get("endpoint_base_xy") if arc_trace is not None else None,
                            "endpoint_base_yaw_rad": arc_trace.get("endpoint_base_yaw_rad") if arc_trace is not None else None,
                            "collision_rejection": arc_trace.get("first_rejected_sample") if arc_trace is not None else None,
                            "productivity_classification": "COLLISION_REJECTED",
                            "productivity_reason": "COLLISION_OR_GRID_REJECTED",
                            "score_eligible": False,
                        })
                    continue
                predicted_heading = w * self.args.dwa_predict_time
                heading_error = abs(normalize_angle(target_heading - predicted_heading))
                if exceeds_speed_limit(float(v), speed_limit):
                    continue
                admission: Optional[RoomLocalProductivityAdmission] = None
                progress_evidence: Optional[ProgressEvidence] = None
                if phase2_productive_active:
                    safe_translational_candidate_count += 1
                    endpoint_xy = arc_trace.get("endpoint_base_xy") if arc_trace is not None else None
                    if isinstance(endpoint_xy, list) and len(endpoint_xy) == 2 and all(finite_number(value) for value in endpoint_xy):
                        endpoint = (float(endpoint_xy[0]), float(endpoint_xy[1]))
                        if path_anchor is not None:
                            projection = project_endpoint_to_path_station(
                                path_xy, path_anchor, endpoint,
                                rollout_arc_length_m=float(v) * float(self.args.dwa_predict_time),
                            )
                            progress_evidence = evaluate_candidate_progress(
                                path_anchor, projection,
                                current_target_xy=target_base_xy,
                                current_local_xy=waypoint_xy,
                                endpoint_xy=endpoint,
                            )
                        else:
                            progress_evidence = evaluate_candidate_direct_progress(
                                current_target_xy=target_base_xy,
                                current_local_xy=waypoint_xy,
                                endpoint_xy=endpoint,
                            )
                    else:
                        progress_evidence = evaluate_candidate_direct_progress(
                            current_target_xy=target_base_xy,
                            current_local_xy=waypoint_xy,
                            endpoint_xy=(0.0, 0.0),
                            projection_reason="MISSING_SAFE_ROLLOUT_ENDPOINT",
                        )
                    admission = classify_room_local_productivity(
                        progress_evidence,
                        direct_or_short_context=direct_or_short_context,
                        meaningful_progress_margin_m=float(progress_margin_m or 0.0),
                    )
                    room_local_candidate_records.append({
                        "v": float(v), "w": float(w),
                        "curvature_m_inv": float(w) / float(v),
                        "motion_class": "FORWARD_STRAIGHT" if abs(float(w)) <= DWA_NUMERIC_EPS else "FORWARD_TURN",
                        "capability_id": room_local_capability.capability_id if room_local_capability else None,
                        "collision_safe": True,
                        "clearance_m": float(clearance),
                        "endpoint_base_xy": list(endpoint_xy) if isinstance(endpoint_xy, list) else None,
                        "endpoint_base_yaw_rad": arc_trace.get("endpoint_base_yaw_rad") if arc_trace is not None else None,
                        "p_path_m": progress_evidence.p_path_m,
                        "p_target_m": progress_evidence.p_target_m,
                        "p_local_m": progress_evidence.p_local_m,
                        "projection_reason": progress_evidence.projection_reason,
                        "productivity_classification": admission.classification,
                        "productivity_reason": admission.reason,
                        "score_eligible": admission.score_eligible,
                    })
                    slice_state = dwa_rollout_state_at_duration(
                        float(v), float(w), float(self.args.command_slice_sec), float(self.args.dwa_dt),
                    )
                    if slice_state is not None:
                        room_local_candidate_records[-1].update({
                            "slice_endpoint_base_xy": slice_state["endpoint_base_xy"],
                            "slice_endpoint_base_yaw_rad": slice_state["endpoint_base_yaw_rad"],
                            "slice_endpoint_time_sec": slice_state["time_sec"],
                            "slice_endpoint_partial_terminal_step": slice_state["partial_terminal_step"],
                        })
                    if not admission.score_eligible:
                        continue
                    productive_translational_candidate_count += 1
                speed_score = float(v) / max(self.args.max_linear_x, 1e-6)
                clearance_score = min(clearance, self.args.max_clearance_score_m) / self.args.max_clearance_score_m
                heading_score = 1.0 - min(heading_error, math.pi) / math.pi
                distance_score = min(target_dist, self.args.lookahead_m) / self.args.lookahead_m
                if target_relative_angular_scoring_active:
                    angular_magnitude_penalty = abs(float(w) - float(w_reference)) / max(
                        self.args.max_angular_z, 1e-6
                    )
                else:
                    angular_magnitude_penalty = abs(float(w)) / max(self.args.max_angular_z, 1e-6)
                prev_w = float(self.prev_cmd[1])
                angular_reversal_penalty = (
                    1.0
                    if abs(prev_w) > self.args.angular_reversal_deadband
                    and abs(float(w)) > self.args.angular_reversal_deadband
                    and (prev_w * float(w)) < 0.0
                    else 0.0
                )
                target_lateral_correction_score = 0.0
                if correction_active:
                    desired_sign = 1.0 if target_base_xy[1] > 0.0 else -1.0
                    signed_w = desired_sign * float(w)
                    if signed_w > 0.0:
                        target_lateral_correction_score = min(
                            signed_w,
                            self.args.target_lateral_correction_angular_z,
                        ) / max(self.args.target_lateral_correction_angular_z, 1e-6)
                score = (
                    self.args.heading_weight * heading_score
                    + self.args.clearance_weight * clearance_score
                    + self.args.speed_weight * speed_score
                    + self.args.distance_weight * distance_score
                    + self.args.target_lateral_correction_weight * target_lateral_correction_score
                    - self.args.angular_magnitude_weight * angular_magnitude_penalty
                    - self.args.angular_reversal_weight * angular_reversal_penalty
                )
                sample = {
                    "v": float(v),
                    "w": float(w),
                    "execution_qualified_primitive": primitive_name,
                    "score": score,
                    "clearance_m": clearance,
                    "heading_error_rad": heading_error,
                    "angular_magnitude_penalty": angular_magnitude_penalty,
                    "angular_penalty_mode": angular_penalty_mode,
                    "angular_reference_time_sec": angular_reference_time_sec,
                    "w_reference": w_reference,
                    "angular_reversal_penalty": angular_reversal_penalty,
                    "target_lateral_correction_score": target_lateral_correction_score,
                }
                if phase2_productive_active and room_local_candidate_records:
                    room_local_candidate_records[-1].update({
                        "final_dwa_score": float(score),
                        "heading_error_rad": float(heading_error),
                        "angular_magnitude_penalty": float(angular_magnitude_penalty),
                        "angular_reversal_penalty": float(angular_reversal_penalty),
                        "target_lateral_correction_score": float(target_lateral_correction_score),
                    })
                if arc_trace is not None:
                    high_level_shadow_samples.append({
                        "v": float(v), "w": float(w),
                        "endpoint_base_xy": arc_trace.get("endpoint_base_xy"),
                        "endpoint_base_yaw_rad": arc_trace.get("endpoint_base_yaw_rad"),
                    })
                if portal_relative_selection_active:
                    geometry = portal_relative_geometry(
                        pose_odom,
                        float(v),
                        float(w),
                        self.args.dwa_predict_time,
                        self.args.dwa_dt,
                        portal_center_odom,
                        portal_normal_odom,
                    )
                    if geometry is None:
                        portal_relative_selection_active = False
                    else:
                        sample.update(geometry)
                samples.append(sample)
                if meets_minimum_forward(float(v), minimum_forward):
                    accepted_moving_samples.append(sample)
                if score > best_score:
                    best_score = score
                    best = (float(v), float(w))
        if best_score == -float("inf"):
            no_productive_translation = bool(
                phase2_productive_active and safe_translational_candidate_count > 0
            )
            return 0.0, 0.0, {
                "sample_count": len(samples),
                "blocked": True,
                "room_local_productivity_set_status": (
                    "NO_PRODUCTIVE_TRANSLATION" if no_productive_translation else None
                ),
                "room_local_phase2_productive_admission_active": phase2_productive_active,
                "room_local_meaningful_progress_margin_m": progress_margin_m,
                "room_local_safe_translational_candidate_count": safe_translational_candidate_count,
                "room_local_productive_translational_candidate_count": 0,
                "room_local_productivity_candidates": room_local_candidate_records,
                "p_through_safe_moving_available": False,
                "safe_moving_candidate_count": 0,
                "zero_speed_excluded_by_p_through_eligibility": False,
                "target_heading_used_rad": target_heading,
                "angular_reference_time_sec": angular_reference_time_sec,
                "w_reference": w_reference,
                "angular_penalty_mode": angular_penalty_mode,
                "selected_linear_x": 0.0,
                "selected_angular_z": 0.0,
            }
        eligibility_requested = bool(p_through_safe_moving_eligibility or room_search_safe_moving_eligibility)
        eligibility_active = bool(
            eligibility_requested
            and target_in_front
            and astar_path_exists
            and accepted_moving_samples
        )
        native_best = best
        native_best_sample = max(samples, key=lambda sample: float(sample["score"]))
        if eligibility_active:
            # R48 A2: this filters only candidates that have already passed the
            # original bounds, dynamic-window, speed-limit and collision checks.
            # It never makes a rejected arc legal.  The caller gates this reuse
            # to P_through or ROOM_SEARCH only.
            moving_best = max(accepted_moving_samples, key=lambda sample: float(sample["score"]))
            best = (float(moving_best["v"]), float(moving_best["w"]))
            best_score = float(moving_best["score"])
            native_best_sample = moving_best
        portal_relative_override_applied = False
        dominating_samples: List[Dict[str, Any]] = []
        selected_sample = native_best_sample
        portal_heading_safe_candidate_count = 0
        portal_pre_crossing_early_turn_selection_applied = False
        portal_arc_reference: Optional[Dict[str, Any]] = None
        portal_arc_reference_selection_applied = False
        portal_arc_tracking_candidate_count = 0
        current_portal_abs_tangent_offset_m: Optional[float] = None
        tangent_preference_candidates: List[Dict[str, Any]] = []
        portal_selection_reason = "native_dwa_selection"
        if portal_relative_selection_active and eligibility_active:
            required_geometry = (
                "portal_normal_progress_m",
                "portal_abs_tangent_offset_m",
                "portal_abs_yaw_error_rad",
            )
            normal_norm = math.hypot(
                float(portal_normal_odom[0]), float(portal_normal_odom[1])
            )
            signed_portal_normal_m = (
                (float(pose_odom[0]) - float(portal_center_odom[0]))
                * float(portal_normal_odom[0]) / normal_norm
                + (float(pose_odom[1]) - float(portal_center_odom[1]))
                * float(portal_normal_odom[1]) / normal_norm
            )
            signed_portal_tangent_m = (
                (float(pose_odom[0]) - float(portal_center_odom[0]))
                * (-float(portal_normal_odom[1]) / normal_norm)
                + (float(pose_odom[1]) - float(portal_center_odom[1]))
                * (float(portal_normal_odom[0]) / normal_norm)
            )
            current_portal_abs_tangent_offset_m = abs(signed_portal_tangent_m)
            pre_portal_crossing = signed_portal_normal_m <= DWA_NUMERIC_EPS
            if pre_portal_crossing:
                portal_arc_reference = (
                    p_through_arc_reference(
                        pose_odom,
                        portal_center_odom,
                        portal_normal_odom,
                        portal_p_pre_odom,
                    )
                    if portal_p_pre_odom is not None else None
                )
                if portal_arc_reference is not None:
                    prediction_steps = max(
                        1,
                        int(float(self.args.dwa_predict_time) / max(float(self.args.dwa_dt), 1e-6)),
                    )
                    prediction_duration_sec = prediction_steps * float(self.args.dwa_dt)
                    arc_tracking_samples: List[Dict[str, Any]] = []
                    for sample in accepted_moving_samples:
                        endpoint_xy = sample.get("endpoint_odom_xy")
                        endpoint_yaw = sample.get("endpoint_yaw_rad")
                        if (
                            not isinstance(endpoint_xy, list)
                            or len(endpoint_xy) != 2
                            or not finite_number(endpoint_xy[0])
                            or not finite_number(endpoint_xy[1])
                            or not finite_number(endpoint_yaw)
                        ):
                            continue
                        target_theta = min(
                            0.5 * math.pi,
                            float(portal_arc_reference["theta_rad"])
                            + float(sample["v"]) * prediction_duration_sec
                            / float(portal_arc_reference["radius_m"]),
                        )
                        target_x, target_y, target_heading = p_through_arc_point_odom(
                            portal_arc_reference, portal_center_odom, target_theta,
                        )
                        tracked = dict(sample)
                        tracked.update({
                            "portal_arc_target_theta_rad": target_theta,
                            "portal_arc_target_odom_xy": [target_x, target_y],
                            "portal_arc_endpoint_error_m": math.hypot(
                                float(endpoint_xy[0]) - target_x,
                                float(endpoint_xy[1]) - target_y,
                            ),
                            "portal_arc_endpoint_heading_error_rad": abs(
                                normalize_angle(float(endpoint_yaw) - target_heading)
                            ),
                        })
                        arc_tracking_samples.append(tracked)
                    portal_arc_tracking_candidate_count = len(arc_tracking_samples)
                    if arc_tracking_samples:
                        selected_sample = min(
                            arc_tracking_samples,
                            key=lambda sample: (
                                float(sample["portal_arc_endpoint_error_m"]),
                                float(sample["portal_arc_endpoint_heading_error_rad"]),
                                -float(sample["score"]),
                            ),
                        )
                        portal_arc_reference.update({
                            "selected_target_theta_rad": float(selected_sample["portal_arc_target_theta_rad"]),
                            "selected_target_odom_xy": list(selected_sample["portal_arc_target_odom_xy"]),
                            "selected_endpoint_error_m": float(selected_sample["portal_arc_endpoint_error_m"]),
                            "selected_endpoint_heading_error_rad": float(
                                selected_sample["portal_arc_endpoint_heading_error_rad"]
                            ),
                        })
                        best = (float(selected_sample["v"]), float(selected_sample["w"]))
                        best_score = float(selected_sample["score"])
                        portal_relative_override_applied = best != native_best
                        portal_arc_reference_selection_applied = True
                        portal_selection_reason = "pre_crossing_p_pre_portal_arc_endpoint_tracking"
                    else:
                        portal_arc_reference = None
                else:
                    # The existing safe-candidate set, speed gate, and collision
                    # semantics are unchanged.  Once close to the Portal along
                    # its tangent, establish the Portal-inward body heading.
                    heading_safe_samples = [
                        sample
                        for sample in accepted_moving_samples
                        if all(key in sample for key in required_geometry)
                        and float(sample["portal_normal_progress_m"]) >= -DWA_NUMERIC_EPS
                    ]
                    portal_heading_safe_candidate_count = len(heading_safe_samples)
                    if heading_safe_samples:
                        selected_sample = min(
                            heading_safe_samples,
                            key=lambda sample: (
                                float(sample["portal_abs_yaw_error_rad"]),
                                -float(sample["score"]),
                            ),
                        )
                        # Keep the established inward-heading winner unless a
                        # near-equally aligned safe arc makes a meaningful
                        # reduction in predicted portal-tangent offset.  This is
                        # deliberately a soft preference: no candidate is made
                        # illegal and no exact-centre condition is imposed.
                        tangent_preference_candidates: List[Dict[str, Any]] = []
                        tangent_preference_active = bool(
                            current_portal_abs_tangent_offset_m
                            <= P_THROUGH_LATE_APPROACH_TANGENT_TRIGGER_M
                        )
                        if tangent_preference_active:
                            best_heading_error = float(selected_sample["portal_abs_yaw_error_rad"])
                            baseline_tangent_offset = float(selected_sample["portal_abs_tangent_offset_m"])
                            tangent_preference_candidates = [
                                sample
                                for sample in heading_safe_samples
                                if float(sample["portal_abs_yaw_error_rad"])
                                <= best_heading_error + P_THROUGH_TANGENT_YAW_SLACK_RAD
                                and float(sample["portal_abs_tangent_offset_m"])
                                <= baseline_tangent_offset - P_THROUGH_TANGENT_MIN_IMPROVEMENT_M
                            ]
                            if tangent_preference_candidates:
                                selected_sample = min(
                                    tangent_preference_candidates,
                                    key=lambda sample: (
                                        float(sample["portal_abs_tangent_offset_m"]),
                                        float(sample["portal_abs_yaw_error_rad"]),
                                        -float(sample["score"]),
                                    ),
                                )
                        best = (float(selected_sample["v"]), float(selected_sample["w"]))
                        best_score = float(selected_sample["score"])
                        portal_relative_override_applied = best != native_best
                        portal_pre_crossing_early_turn_selection_applied = True
                        portal_selection_reason = (
                            "pre_crossing_soft_tangent_reduction_within_yaw_slack"
                            if tangent_preference_candidates
                            else "pre_crossing_min_portal_inward_yaw_error"
                        )
                    else:
                        portal_selection_reason = "pre_crossing_no_positive_progress_safe_candidate"
            elif all(key in native_best_sample for key in required_geometry):
                for sample in accepted_moving_samples:
                    if not all(key in sample for key in required_geometry):
                        continue
                    dominates = (
                        sample["portal_normal_progress_m"] >= native_best_sample["portal_normal_progress_m"]
                        and sample["portal_abs_tangent_offset_m"] <= native_best_sample["portal_abs_tangent_offset_m"]
                        and sample["portal_abs_yaw_error_rad"] <= native_best_sample["portal_abs_yaw_error_rad"]
                        and (
                            sample["portal_normal_progress_m"] > native_best_sample["portal_normal_progress_m"]
                            or sample["portal_abs_tangent_offset_m"] < native_best_sample["portal_abs_tangent_offset_m"]
                            or sample["portal_abs_yaw_error_rad"] < native_best_sample["portal_abs_yaw_error_rad"]
                        )
                    )
                    if dominates:
                        dominating_samples.append(sample)
                if dominating_samples:
                    selected_sample = max(dominating_samples, key=lambda sample: float(sample["score"]))
                    best = (float(selected_sample["v"]), float(selected_sample["w"]))
                    best_score = float(selected_sample["score"])
                    portal_relative_override_applied = best != native_best
                portal_selection_reason = "post_crossing_existing_portal_dominance"
        self.prev_cmd = (best[0], best[1])
        result = {
            "sample_count": len(samples),
            "best_score": best_score,
            "blocked": False,
            "execution_qualified_dwa_primitives": False,
            "selected_execution_qualified_primitive": None,
            "dynamic_window_v": [float(v_samples[0]), float(v_samples[-1])],
            "dynamic_window_w": [float(w_samples[0]), float(w_samples[-1])],
            "waypoint_heading_rad": waypoint_heading,
            "final_target_heading_rad": final_target_heading,
            "target_heading_used_rad": target_heading,
            "angular_reference_time_sec": angular_reference_time_sec,
            "w_reference": w_reference,
            "angular_penalty_mode": angular_penalty_mode,
            "selected_linear_x": best[0],
            "selected_angular_z": best[1],
            "target_lateral_error_m": target_base_xy[1],
            "target_lateral_correction_active": correction_active,
            "pointcloud_wall_heading_active": wall_heading_active,
            "pointcloud_wall_heading_used_rad": wall_heading_prior.get("heading_parallel_rad") if wall_heading_prior else None,
            "pointcloud_wall_heading_blend_weight": self.args.pointcloud_wall_heading_blend_weight,
            "p_through_safe_moving_eligibility_requested": bool(p_through_safe_moving_eligibility),
            "room_search_safe_moving_eligibility_requested": bool(room_search_safe_moving_eligibility),
            "p_through_target_relative_angular_scoring_requested": bool(
                p_through_target_relative_angular_scoring
            ),
            "p_through_target_in_front": bool(target_in_front),
            "p_through_astar_path_exists": bool(astar_path_exists),
            "p_through_safe_moving_available": bool(accepted_moving_samples),
            "safe_moving_candidate_count": len(accepted_moving_samples),
            "portal_relative_selection_active": bool(portal_relative_selection_active),
            "portal_relative_override_applied": bool(portal_relative_override_applied),
            "portal_safe_moving_candidate_count": len(accepted_moving_samples) if portal_relative_selection_active else 0,
            "portal_heading_safe_candidate_count": int(portal_heading_safe_candidate_count),
            "portal_pre_crossing_early_turn_selection_applied": bool(
                portal_pre_crossing_early_turn_selection_applied
            ),
            "portal_arc_reference": portal_arc_reference,
            "portal_arc_reference_selection_applied": bool(
                portal_arc_reference_selection_applied
            ),
            "portal_arc_tracking_candidate_count": int(portal_arc_tracking_candidate_count),
            "portal_current_abs_tangent_offset_m": (
                float(current_portal_abs_tangent_offset_m)
                if current_portal_abs_tangent_offset_m is not None else None
            ),
            "portal_late_approach_tangent_preference_trigger_m": (
                float(P_THROUGH_LATE_APPROACH_TANGENT_TRIGGER_M)
                if portal_relative_selection_active else None
            ),
            "portal_tangent_preference_candidate_count": len(tangent_preference_candidates),
            "portal_selection_reason": portal_selection_reason,
            "portal_dominating_candidate_count": len(dominating_samples),
            "portal_native_winner": dict(native_best_sample),
            "portal_selected_winner": dict(selected_sample),
            "zero_speed_excluded_by_p_through_eligibility": bool(
                p_through_safe_moving_eligibility and eligibility_active and abs(float(native_best[0])) <= 1e-12
            ),
            "zero_speed_excluded_by_room_search_eligibility": bool(
                room_search_safe_moving_eligibility and eligibility_active and abs(float(native_best[0])) <= 1e-12
            ),
            "native_best_was_zero_speed": bool(abs(float(native_best[0])) <= 1e-12),
            "room_local_phase2_productive_admission_active": phase2_productive_active,
            "room_local_productivity_set_status": (
                "PRODUCTIVE_TRANSLATIONAL_SET_NONEMPTY" if phase2_productive_active else None
            ),
            "room_local_meaningful_progress_margin_m": progress_margin_m,
            "room_local_safe_translational_candidate_count": safe_translational_candidate_count,
            "room_local_productive_translational_candidate_count": productive_translational_candidate_count,
            "room_local_productivity_candidates": room_local_candidate_records,
        }
        if high_level_shadow_capture:
            def _is_rotation(sample: Dict[str, Any]) -> bool:
                return abs(float(sample["v"])) <= 1e-12 and abs(float(sample["w"])) > 1e-12

            def _is_forward(sample: Dict[str, Any]) -> bool:
                return float(sample["v"]) > 1e-12

            forward = [sample for sample in high_level_shadow_samples if _is_forward(sample)]
            forward_turn = [sample for sample in forward if abs(float(sample["w"])) > 1e-12]
            rotations = [sample for sample in high_level_shadow_samples if _is_rotation(sample)]
            endpoint_progress = []
            target_norm = math.hypot(float(target_base_xy[0]), float(target_base_xy[1]))
            if target_norm > 0.0:
                for sample in forward:
                    endpoint = sample.get("endpoint_base_xy")
                    if isinstance(endpoint, list) and len(endpoint) == 2:
                        endpoint_progress.append((float(endpoint[0]) * float(target_base_xy[0]) + float(endpoint[1]) * float(target_base_xy[1])) / target_norm)
            selected_shadow_sample = next(
                (sample for sample in high_level_shadow_samples
                 if float(sample["v"]) == float(selected_sample["v"]) and float(sample["w"]) == float(selected_sample["w"])),
                None,
            )
            result["high_level_locomotion_shadow"] = {
                "evidence_source": "SAME_PRODUCTION_DWA_EVALUATION",
                "motion_class_counts": {"safe_forward_turn": len(forward_turn), "safe_forward": len(forward), "rotate_only": len(rotations)},
                "SAFE_FORWARD_TURN_PRESENT": bool(forward_turn),
                "SAFE_FORWARD_MOTION_PRESENT": bool(forward),
                "ROTATE_ONLY_PRESENT": bool(rotations),
                "ONLY_ROTATION_EVIDENCE_PRESENT": bool(rotations and not forward),
                "FORWARD_PROGRESS_EVIDENCE_PRESENT": bool(endpoint_progress),
                "forward_endpoint_progress_along_target_m": endpoint_progress,
                "selected_endpoint_base_xy": selected_shadow_sample.get("endpoint_base_xy") if selected_shadow_sample else None,
                "selected_endpoint_base_yaw_rad": selected_shadow_sample.get("endpoint_base_yaw_rad") if selected_shadow_sample else None,
            }
        return best[0], best[1], result

    def apply_p_pre_goal_region_motion_guard(
        self,
        target: Dict[str, Any],
        target_base_xy: Tuple[float, float],
        linear_x: float,
        angular_z: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """Preserve a P_pre command when its constant-curvature path reaches its goal disk.

        The P_pre success contract is Euclidean distance <= goal_tolerance_m,
        not passage through the target centre.  With fixed signed angular rate
        ``w``, positive speed chooses a circle radius ``r = v / |w|``.  For a
        correctly signed forward target, the radii whose forward circle meets
        the target disk form one interval.  A speed cap is needed only when
        the DWA circle is wider than that interval.  The zero-angular-rate
        limit is the corresponding straight-ray/disk intersection.

        DWA remains authoritative for path scoring, direction, and obstacle
        checks; this P_pre-only guard may only reduce forward speed, never
        alter angular velocity or create a replacement target.
        """
        x, y = float(target_base_xy[0]), float(target_base_xy[1])
        goal_tolerance = float(self.args.goal_tolerance_m)
        distance = math.hypot(x, y)
        report: Dict[str, Any] = {
            "applied": False,
            "target_source": target.get("source"),
            "original_linear_x": float(linear_x),
            "angular_z": float(angular_z),
            "target_base_xy": [x, y],
            "goal_tolerance_m": goal_tolerance,
            "distance_to_target_m": distance,
            "command_radius_m": None,
            "goal_region_radius_interval_m": None,
            "trajectory_enters_goal_region": None,
            "linear_cap_mps": None,
            "reason": None,
        }
        if target.get("source") != "PORTAL_G14_P_PRE":
            report["reason"] = "not_portal_g14_p_pre"
            return float(linear_x), report
        if linear_x <= 0.0:
            report["reason"] = "nonforward_command"
            return float(linear_x), report
        if x <= 0.0:
            report.update({"applied": True, "linear_cap_mps": 0.0, "reason": "p_pre_no_longer_in_front"})
            return 0.0, report
        if distance <= goal_tolerance:
            report.update({
                "applied": True,
                "linear_cap_mps": 0.0,
                "trajectory_enters_goal_region": True,
                "reason": "p_pre_already_within_goal_region",
            })
            return 0.0, report

        angular_epsilon = 1e-9
        if abs(angular_z) <= angular_epsilon:
            straight_enters = abs(y) <= goal_tolerance
            report.update({
                "trajectory_enters_goal_region": straight_enters,
                "reason": "p_pre_goal_region_straight_intersection" if straight_enters else "p_pre_goal_region_unreachable_straight",
            })
            if straight_enters:
                return float(linear_x), report
            report.update({"applied": True, "linear_cap_mps": 0.0})
            return 0.0, report

        turn_sign = 1.0 if angular_z > 0.0 else -1.0
        signed_lateral = turn_sign * y
        # If the disk lies fully on the opposite side of the selected turn,
        # no forward first-pass arc can reach it.  Reducing v would only make
        # that wrong turn tighter, so fail closed rather than release it.
        if signed_lateral <= -goal_tolerance:
            report.update({
                "trajectory_enters_goal_region": False,
                "applied": True,
                "linear_cap_mps": 0.0,
                "reason": "p_pre_goal_region_wrong_turn_unreachable",
            })
            return 0.0, report

        numerator = distance * distance - goal_tolerance * goal_tolerance
        minimum_radius = numerator / (2.0 * (signed_lateral + goal_tolerance))
        maximum_radius: Optional[float]
        if signed_lateral > goal_tolerance:
            maximum_radius = numerator / (2.0 * (signed_lateral - goal_tolerance))
        else:
            maximum_radius = None
        command_radius = float(linear_x) / abs(float(angular_z))
        report.update({
            "command_radius_m": command_radius,
            "goal_region_radius_interval_m": [minimum_radius, maximum_radius],
        })
        if command_radius < minimum_radius - 1e-9:
            report.update({
                "trajectory_enters_goal_region": False,
                # Reducing v would only tighten this already over-tight arc.
                # This guard has no authority to increase DWA's chosen speed,
                # so preserve the selected command and let the unchanged DWA
                # choose a different command on its next normal cycle.
                "reason": "p_pre_goal_region_inner_miss_not_speed_capable",
            })
            return float(linear_x), report
        if maximum_radius is None or command_radius <= maximum_radius + 1e-9:
            report.update({
                "trajectory_enters_goal_region": True,
                "reason": "p_pre_goal_region_already_reachable",
            })
            return float(linear_x), report

        linear_cap = abs(float(angular_z)) * maximum_radius
        report.update({
            "applied": True,
            "trajectory_enters_goal_region": False,
            "linear_cap_mps": float(linear_cap),
            "reason": "p_pre_goal_region_outer_tangent_speed_cap",
        })
        return min(float(linear_x), linear_cap), report

    def runtime_clock_status(self) -> Dict[str, Any]:
        now_wall = time.monotonic()
        now_sim = float(rospy.Time.now().to_sec())
        if not self.sim_time_valid and (not self.use_sim_time or now_sim > 0.0):
            self.sim_time_valid = True
            self.run_sim_start_sec = now_sim
        wall_elapsed = (
            now_wall - float(self.run_wall_start_sec)
            if self.run_wall_start_sec is not None
            else 0.0
        )
        sim_elapsed = (
            max(0.0, now_sim - float(self.run_sim_start_sec))
            if self.sim_time_valid and self.run_sim_start_sec is not None
            else None
        )
        rtf = (
            float(sim_elapsed) / wall_elapsed
            if sim_elapsed is not None and wall_elapsed > 1e-6
            else None
        )
        return {
            "run_wall_elapsed_sec": wall_elapsed,
            "run_sim_elapsed_sec": sim_elapsed,
            "real_time_factor_estimate": rtf,
            "now_wall_sec": now_wall,
            "now_sim_sec": now_sim,
        }

    def current_timeout_trigger(self, *, include_sim_timeout: bool = True) -> Optional[str]:
        status = self.runtime_clock_status()
        if status["run_wall_elapsed_sec"] >= float(self.args.wall_watchdog_sec):
            self.timeout_trigger = "wall_watchdog_timeout"
        elif (
            include_sim_timeout
            and self.sim_time_valid
            and finite_number(status.get("run_sim_elapsed_sec"))
            and float(status["run_sim_elapsed_sec"]) >= float(self.args.max_runtime_sec)
        ):
            self.timeout_trigger = "sim_timeout"
        return self.timeout_trigger

    def emit_validation_event(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        """Publish a small, authority-free validation event; never wait for it."""
        if not self.validation_enabled:
            return {}
        self.validation_event_sequence += 1
        event = {
            "event_type": str(event_type),
            "run_id": self.validation_run_id,
            "runner_invocation_id": self.validation_runner_invocation_id,
            "event_sequence": self.validation_event_sequence,
            "ros_time_sec": float(rospy.Time.now().to_sec()),
        }
        event.update(fields)
        self.validation_events.append(event)
        if self.validation_event_pub is not None:
            try:
                self.validation_event_pub.publish(String(data=json.dumps(event, sort_keys=True)))
            except Exception:
                # Observability cannot delay or replace the existing controller.
                pass
        return event

    def validation_command_violation(self, v: float, w: float) -> Optional[str]:
        """Validation-only producer-near guard; TRANSIT keeps legacy authority."""
        if not (self.validation_enabled and self.local_control_mode == LOCAL_CONTROL_MODE_ROOM_LOCAL):
            return None
        if float(v) < -DWA_NUMERIC_EPS:
            return "UNSUPPORTED_COMMAND_ONLINE_VIOLATION:REVERSE"
        if abs(float(w)) > float(self.args.max_angular_z) + DWA_NUMERIC_EPS:
            return "UNSUPPORTED_COMMAND_ONLINE_VIOLATION:ANGULAR_BOUND"
        if float(v) > DWA_NUMERIC_EPS:
            capability = room_local_capability_by_id("forward_turn_current_room_profile")
            if float(v) + DWA_NUMERIC_EPS < MIN_EXECUTABLE_FORWARD_MPS or capability is None or not capability.contains(float(v), float(w)):
                return "UNSUPPORTED_COMMAND_ONLINE_VIOLATION:FORWARD_CAPABILITY"
        return None

    def publish_twist(self, v: float, w: float, duration_sec: float) -> int:
        violation = self.validation_command_violation(v, w)
        if violation is not None:
            self.validation_abort_reason = violation
            self.emit_validation_event("VALIDATION_CONTRACT_VIOLATION", reason=violation, requested_v=float(v), requested_w=float(w))
            return 0
        if not self.args.execute:
            return 0
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        start_sim = float(rospy.Time.now().to_sec())
        command_slice_id = ""
        if self.validation_enabled:
            self.validation_command_slice_sequence += 1
            command_slice_id = "%s:cmd_%04d" % (self.validation_runner_invocation_id, self.validation_command_slice_sequence)
            self.emit_validation_event(
                "COMMAND_SLICE_START", command_slice_id=command_slice_id,
                requested_v=float(v), requested_w=float(w), intended_duration_sec=float(duration_sec),
                decision_id=self.validation_current_decision_id or None,
            )
        count = 0
        sim_period_sec = 1.0 / max(float(self.args.cmd_rate_hz), 1e-6)
        next_publish_sim = start_sim
        while not rospy.is_shutdown():
            if self.validation_abort_reason is not None:
                break
            if self.current_timeout_trigger(include_sim_timeout=False) == "wall_watchdog_timeout":
                break
            now_sim = float(rospy.Time.now().to_sec())
            if now_sim - start_sim >= duration_sec:
                break
            if now_sim + 1e-9 >= next_publish_sim:
                self.pub.publish(cmd)
                count += 1
                next_publish_sim += sim_period_sec
            time.sleep(min(0.01, sim_period_sec))
        if command_slice_id:
            self.emit_validation_event(
                "COMMAND_SLICE_END", command_slice_id=command_slice_id,
                requested_v=float(v), requested_w=float(w), published_count=count,
                start_ros_time_sec=start_sim, end_ros_time_sec=float(rospy.Time.now().to_sec()),
            )
        return count

    def post_command_terminal_reach_check(
        self,
        target_xy: Sequence[float],
        pre_command_odom: Odometry,
    ) -> Dict[str, Any]:
        """Fetch one new odom sample after the final command slice, fail closed."""
        try:
            post_command_odom = rospy.wait_for_message(
                TOPIC_ODOM,
                Odometry,
                timeout=self.args.input_timeout_sec,
            )
        except Exception as exc:
            return {
                "attempted": True,
                "fresh_valid_odom": False,
                "reached": False,
                "reason": f"post_command_odom_unavailable:{type(exc).__name__}",
            }
        return terminal_reach_from_post_command_odom(
            target_xy,
            pre_command_odom,
            post_command_odom,
            self.args.goal_tolerance_m,
        )

    def stop(self) -> None:
        if not self.args.execute:
            return
        zero = Twist()
        for _ in range(5):
            self.pub.publish(zero)
            time.sleep(0.03)

    @staticmethod
    def orientation_input_identity(
        odom: Odometry,
        grid_msg: OccupancyGrid,
        status_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        pose = pose_tuple(odom)
        return {
            "pose_x_y_yaw": [float(value) for value in pose],
            "odom_stamp_sec": BlockAStarDwaRunner.message_stamp_sec(odom),
            "grid_header_stamp_sec": BlockAStarDwaRunner.message_stamp_sec(grid_msg),
            "grid_content_stamp": status_payload.get("grid_content_stamp"),
            "content_generation_id": status_payload.get("content_generation_id"),
            "grid_content_hash": status_payload.get("grid_content_hash"),
        }

    def run_observation_orientation(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Execute at most one OA slice, then acquire fresh terminal inputs."""
        final_decision = "OBSERVATION_ORIENTATION_INPUT_UNAVAILABLE"
        status_payload: Dict[str, Any] = {}
        orientation: Dict[str, Any] = {
            "mode": "OBSERVATION_ORIENTATION_ONE_SLICE",
            "phase3_state_touched": False,
            "pre_slice": None,
            "post_slice": None,
        }
        try:
            odom, grid_msg, status_payload = self.wait_inputs()
            qualification_errors = self.grid_qualification_errors(grid_msg, status_payload)
            if qualification_errors or status_payload.get("local_traversability_status") != "FREE_SUPPORTED":
                orientation.update({
                    "action": "ORIENTATION_UNAVAILABLE",
                    "reason": "PRE_SLICE_GRID_STATUS_UNQUALIFIED",
                    "qualification_errors": qualification_errors,
                    "local_traversability_status": status_payload.get("local_traversability_status"),
                })
                final_decision = "OBSERVATION_ORIENTATION_UNAVAILABLE"
            else:
                pose = pose_tuple(odom)
                raw_grid = self.grid_array(grid_msg)
                blocked = self.inflate_obstacles(raw_grid, float(grid_msg.info.resolution))
                action = self.observation_orientation_slice_action(pose=pose, grid_msg=grid_msg, blocked=blocked)
                orientation.update(action)
                orientation["pre_slice"] = self.orientation_input_identity(odom, grid_msg, status_payload)
                if action.get("action") != "OBSERVATION_ORIENTATION_ONE_SLICE":
                    final_decision = "OBSERVATION_ORIENTATION_UNAVAILABLE"
                elif not bool(self.args.execute):
                    orientation["reason"] = "DRY_RUN_NO_COMMAND_PUBLISHED"
                    final_decision = "OBSERVATION_ORIENTATION_DRY_RUN"
                else:
                    selection = action["selection"]
                    w = float(selection["w_radps"])
                    self.prev_cmd = (0.0, w)
                    published = self.publish_twist(0.0, w, float(self.args.command_slice_sec))
                    summary["steps"].append({
                        "step": 0,
                        "pose_x_y_yaw": list(pose),
                        "cmd_linear_x": 0.0,
                        "cmd_angular_z": w,
                        "published_count": int(published),
                        "observation_orientation": copy.deepcopy(action),
                    })
                    # Direct post-command waits ensure each terminal source is
                    # acquired after the one command slice, not reused from the
                    # pre-slice cache.
                    post_odom = rospy.wait_for_message(TOPIC_ODOM, Odometry, timeout=self.args.input_timeout_sec)
                    post_grid = rospy.wait_for_message(TOPIC_GRID, OccupancyGrid, timeout=self.args.input_timeout_sec)
                    post_status_msg = rospy.wait_for_message(TOPIC_STATUS, String, timeout=self.args.input_timeout_sec)
                    try:
                        post_status = json.loads(post_status_msg.data)
                    except Exception:
                        post_status = {}
                    post_status = post_status if isinstance(post_status, dict) else {}
                    post_errors = self.grid_qualification_errors(post_grid, post_status)
                    orientation["post_slice"] = self.orientation_input_identity(post_odom, post_grid, post_status)
                    orientation["post_slice"].update({
                        "grid_status_qualification": "QUALIFIED_EXACT_PAIR" if not post_errors else "UNAVAILABLE_OR_UNQUALIFIED",
                        "qualification_errors": post_errors,
                    })
                    if post_errors or post_status.get("local_traversability_status") != "FREE_SUPPORTED":
                        final_decision = "OBSERVATION_ORIENTATION_POST_SLICE_EVIDENCE_UNAVAILABLE"
                    elif int(published) <= 0:
                        final_decision = "OBSERVATION_ORIENTATION_COMMAND_NOT_PUBLISHED"
                    else:
                        final_decision = "OBSERVATION_ORIENTATION_SLICE_EXECUTED"
        except Exception as exc:
            orientation.update({"action": "ORIENTATION_UNAVAILABLE", "reason": "%s:%s" % (type(exc).__name__, exc)})
            final_decision = "OBSERVATION_ORIENTATION_INPUT_UNAVAILABLE"
        finally:
            self.stop()

        summary["observation_orientation"] = orientation
        clock_status = self.runtime_clock_status()
        summary.update(clock_status)
        summary["run_wall_start_sec"] = self.run_wall_start_sec
        summary["run_sim_start_sec"] = self.run_sim_start_sec
        summary["use_sim_time"] = self.use_sim_time
        summary["sim_time_valid"] = self.sim_time_valid
        summary["timeout_trigger"] = self.timeout_trigger
        summary["final_decision"] = final_decision
        summary["legacy_timeout_decision"] = None
        summary["status_payload"] = status_payload
        summary["validation"]["event_count"] = len(self.validation_events)
        summary["validation"]["abort_reason"] = self.validation_abort_reason
        self.emit_validation_event("RUNNER_INVOCATION_END", final_decision=final_decision)
        write_json(OUT / "block_astar_dwa_mature_summary.json", summary)
        return summary

    def run(self) -> Dict[str, Any]:
        target = read_json(TARGET_PATH)
        target_xy = target.get("target_xy_team_livox_odom")
        if not (isinstance(target_xy, list) and len(target_xy) == 2 and all(finite_number(v) for v in target_xy)):
            return {"final_decision": "BLOCK_ASTAR_DWA_BLOCKED_BY_TARGET", "target": target}

        self.run_wall_start_sec = time.monotonic()
        self.run_wall_deadline_sec = self.run_wall_start_sec + float(self.args.wall_watchdog_sec)
        self.use_sim_time = bool(rospy.get_param("/use_sim_time", False))
        initial_sim_time = float(rospy.Time.now().to_sec())
        sim_wait_started_wall = time.monotonic()
        sim_wait_deadline_wall = min(
            self.run_wall_deadline_sec,
            sim_wait_started_wall + float(self.args.sim_time_start_wait_sec),
        )
        while (
            self.use_sim_time
            and initial_sim_time <= 0.0
            and not rospy.is_shutdown()
            and time.monotonic() < sim_wait_deadline_wall
        ):
            time.sleep(0.05)
            initial_sim_time = float(rospy.Time.now().to_sec())
        self.sim_time_valid = bool(not self.use_sim_time or initial_sim_time > 0.0)
        self.run_sim_start_sec = initial_sim_time if self.sim_time_valid else None
        sim_time_start_wait_wall_sec = time.monotonic() - sim_wait_started_wall

        status_payload = {}
        summary: Dict[str, Any] = {
            "execute": bool(self.args.execute),
            "local_control_mode": self.local_control_mode,
            "cmd_topic": self.args.cmd_topic,
            "target_xy_team_livox_odom": target_xy,
            "target_source": target.get("source"),
            "target_subgoal_source": target.get("subgoal_source"),
            "run_wall_start_sec": self.run_wall_start_sec,
            "run_sim_start_sec": self.run_sim_start_sec,
            "use_sim_time": self.use_sim_time,
            "sim_time_valid": self.sim_time_valid,
            "sim_time_start_wait_wall_sec": sim_time_start_wait_wall_sec,
            "timeout_clock": "sim_time_with_wall_watchdog",
            "max_runtime_sec_sim": float(self.args.max_runtime_sec),
            "wall_watchdog_sec": float(self.args.wall_watchdog_sec),
            "timeout_trigger": None,
            "validation": {
                "enabled": self.validation_enabled,
                "run_id": self.validation_run_id or None,
                "runner_invocation_id": self.validation_runner_invocation_id or None,
                "phase2_active": bool(self.phase2_productive_admission_active),
                "phase3_active": bool(self.phase3_orientation_recovery_active),
            },
                "path_stability_policy": {
                    "enabled": not bool(self.args.disable_path_stability),
                    "centerline_target_y_threshold_m": self.args.centerline_target_y_threshold_m,
                    "waypoint_lateral_anomaly_threshold_m": self.args.waypoint_lateral_anomaly_threshold_m,
                    "max_hold_steps": self.args.path_stability_max_hold_steps,
                    "astar_lateral_bias_weight": self.args.astar_lateral_bias_weight,
                    "centerline_thin_barrier_clearing_enabled": not bool(
                        self.args.disable_centerline_thin_barrier_clearing
                    ),
                    "corridor_center_target_enabled": not bool(self.args.disable_corridor_center_target),
                    "corridor_center_x_range_m": [
                        self.args.corridor_center_x_min_m,
                        self.args.corridor_center_x_max_m,
                    ],
                    "corridor_center_blend_weight": self.args.corridor_center_blend_weight,
                    "pointcloud_wall_heading_enabled": not bool(self.args.disable_pointcloud_wall_heading),
                    "pointcloud_wall_heading_blend_weight": self.args.pointcloud_wall_heading_blend_weight,
                    "imu_heading_hold_enabled": not bool(self.args.disable_imu_heading_hold),
                    "imu_topic": self.args.imu_topic,
                    "imu_heading_hold_kp": self.args.imu_heading_hold_kp,
                    "imu_heading_hold_kd": self.args.imu_heading_hold_kd,
                    "imu_heading_hold_max_correction_rad_s": self.args.imu_heading_hold_max_correction_rad_s,
                    "centerline_tracking_correction_enabled": not bool(
                        self.args.disable_centerline_tracking_correction
                    ),
                    "centerline_tracking_lateral_kp": self.args.centerline_tracking_lateral_kp,
                    "centerline_tracking_yaw_kp": self.args.centerline_tracking_yaw_kp,
                    "centerline_tracking_max_correction_rad_s": self.args.centerline_tracking_max_correction_rad_s,
                },
            "steps": [],
        }
        self.emit_validation_event(
            "RUNNER_INVOCATION_START", local_control_mode=self.local_control_mode,
            phase2_active=bool(self.phase2_productive_admission_active),
            phase3_active=bool(self.phase3_orientation_recovery_active), target=target_xy,
        )
        if getattr(self.args, "observation_orientation_aim_yaw_odom_rad", None) is not None:
            return self.run_observation_orientation(summary)

        final_decision = "BLOCK_ASTAR_DWA_SIM_TIMEOUT"
        try:
            for step_idx in range(self.args.max_steps):
                if self.validation_abort_reason is not None:
                    final_decision = "ROOM_LOCAL_ONLINE_VALIDATION_ABORT"
                    summary["validation_abort_reason"] = self.validation_abort_reason
                    break
                timeout_trigger = self.current_timeout_trigger()
                if timeout_trigger == "wall_watchdog_timeout":
                    final_decision = "BLOCK_ASTAR_DWA_WALL_WATCHDOG_TIMEOUT"
                    break
                if timeout_trigger == "sim_timeout":
                    final_decision = "BLOCK_ASTAR_DWA_SIM_TIMEOUT"
                    break
                odom, grid_msg, status_payload = self.wait_inputs()
                if self.validation_enabled:
                    self.validation_input_acquisition_sequence += 1
                    self.emit_validation_event(
                        "FRESH_INPUT_ACQUIRED",
                        input_acquisition_sequence=int(self.validation_input_acquisition_sequence),
                        grid_status_acquisition_sequence=int(self.validation_input_acquisition_sequence),
                    )
                qualification_errors = self.grid_qualification_errors(grid_msg, status_payload)
                if qualification_errors:
                    final_decision = "BLOCK_ASTAR_DWA_BLOCKED_BY_LOCAL_GRID_CONTRACT"
                    summary["local_grid_contract_rejection"] = qualification_errors
                    break
                if status_payload.get("local_traversability_status") != "FREE_SUPPORTED":
                    final_decision = "BLOCK_ASTAR_DWA_BLOCKED_BY_L3V_STATUS"
                    break
                self.safety_decision_sequence += 1
                safety_decision_id = "%s:dwa_%04d" % (
                    self.validation_runner_invocation_id or "runner",
                    self.safety_decision_sequence,
                )
                safety_binding = self.dwa_safety_state_binding(grid_msg, status_payload)
                if safety_binding.get("binding_valid") is not True:
                    summary["failure_diagnostic"] = {
                        "dwa_decision_id": safety_decision_id,
                        "same_state_safety_binding": safety_binding,
                        "command_authority": False,
                    }
                    summary["steps"].append({
                        "step": step_idx,
                        "dwa_decision_id": safety_decision_id,
                        "same_state_safety_binding": safety_binding,
                        "command_authority": False,
                    })
                    self.emit_validation_event(
                        "SAFETY_STATE_BINDING_UNAVAILABLE",
                        decision_id=safety_decision_id,
                        same_state_safety_binding=safety_binding,
                        command_authority=False,
                    )
                    final_decision = "BLOCK_ASTAR_DWA_SAFETY_STATE_BINDING_UNAVAILABLE"
                    break
                # All Grid-relative A*, target and collision geometry below is
                # expressed using the exact/bounded source-time pose that made
                # this formal Grid/status pair.  The newest receiver message
                # remains diagnostic only and can never substitute for it.
                pose = tuple(float(value) for value in safety_binding["source_pose_x_y_yaw"])
                tx_base, ty_base = target_to_base(target_xy, pose)
                room_entry_target = (
                    target.get("subgoal_source") in ROOM_ENTRY_SUBGOAL_SOURCES
                    or target.get("source") in ROOM_ENTRY_TARGET_SOURCES
                )
                if room_entry_target and tx_base <= ROOM_ENTRY_MIN_TARGET_X_BASE_M:
                    final_decision = "BLOCK_ASTAR_DWA_TARGET_NOT_IN_FRONT"
                    summary["target_not_in_front_guard"] = {
                        "target_base_xy": [tx_base, ty_base],
                        "min_target_x_base_m": ROOM_ENTRY_MIN_TARGET_X_BASE_M,
                        "target_source": target.get("source"),
                        "target_subgoal_source": target.get("subgoal_source"),
                        "pose_x_y_yaw": list(pose),
                    }
                    break
                distance = math.hypot(tx_base, ty_base)
                if distance <= self.args.goal_tolerance_m:
                    final_decision = "BLOCK_ASTAR_DWA_REACHED_GOAL"
                    break
                centerline_target = self.target_is_grid_centerline(target)
                pre_room_zone_centerline_correction = (
                    self.pre_room_zone_centerline_correction_authorized(target)
                )
                original_target_base_xy = (tx_base, ty_base)
                raw_grid = self.grid_array(grid_msg)
                grid, costmap_preprocess = self.apply_centerline_thin_barrier_clearing(
                    grid_msg,
                    raw_grid,
                    (tx_base, ty_base),
                    centerline_target,
                    status_payload,
                )
                occupied_inflated = self.occupied_inflated_mask(grid, float(grid_msg.info.resolution))
                blocked = self.inflate_obstacles(grid, float(grid_msg.info.resolution))
                # Preserve the existing target-adjustment order.  The planning
                # diagnostic below is recomputed from the clearance copy.
                corridor_center_target = self.estimate_corridor_center(grid_msg, occupied_inflated)
                if pre_room_zone_centerline_correction:
                    ty_base, corridor_center_target = self.apply_corridor_center_target(
                        ty_base,
                        corridor_center_target,
                    )
                else:
                    corridor_center_target["target_application"] = {
                        "target_adjustment_applied": False,
                        "target_adjustment_reason": "outside_pre_room_zone_scope",
                        "target_scope": target.get("corridor_center_target_scope"),
                    }
                wall_heading_prior = self.current_wall_heading_prior()
                start_cell = self.local_xy_to_cell(0.0, 0.0, grid_msg)
                if start_cell is None:
                    final_decision = "BLOCK_ASTAR_DWA_GRID_TARGET_OUT_OF_BOUNDS"
                    break
                planning_blocked, start_footprint_clearance = self.apply_start_footprint_clearance(
                    grid_msg,
                    raw_grid,
                    blocked,
                    occupied_inflated,
                    qualification_passed=True,
                )
                summary["start_footprint_clearance"] = start_footprint_clearance
                planning_corridor_center_target = self.estimate_corridor_center(
                    grid_msg,
                    occupied_inflated,
                )
                planning_corridor_center_target["target_application"] = corridor_center_target.get(
                    "target_application",
                    corridor_center_target,
                )
                corridor_center_target = planning_corridor_center_target
                (tx_base, ty_base), unilateral_clearance_target = self.unilateral_clearance_target_shape(
                    grid_msg,
                    target,
                    grid,
                    planning_blocked,
                    (tx_base, ty_base),
                )
                distance = math.hypot(tx_base, ty_base)
                goal_x = max(0.0, min(tx_base, self.args.max_goal_x_m))
                goal_y = max(-self.args.max_goal_abs_y_m, min(ty_base, self.args.max_goal_abs_y_m))
                goal_cell = self.local_xy_to_cell(goal_x, goal_y, grid_msg)
                if goal_cell is None:
                    final_decision = "BLOCK_ASTAR_DWA_GRID_TARGET_OUT_OF_BOUNDS"
                    break
                astar_lateral_bias_active = (
                    centerline_target and abs(ty_base) <= self.args.centerline_target_y_threshold_m
                )
                target_lateral_for_astar = ty_base if astar_lateral_bias_active else None
                raw_path = self.block_astar(planning_blocked, start_cell, goal_cell, grid_msg, target_lateral_for_astar)
                planner_path_source = "coarse_block_astar"
                fine_grid_fallback = {
                    "attempted": False,
                    "neighbor_rule": "4_neighbor",
                    "path_found": False,
                    "start_cell": [int(start_cell[0]), int(start_cell[1])],
                    "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
                }
                if len(raw_path) < 2:
                    # Reuse exactly the already-qualified planning mask.  Coarse
                    # success remains untouched; this runs only on coarse NO_PATH.
                    fine_grid_fallback["attempted"] = True
                    fine_path = self.fine_grid_astar_fallback(planning_blocked, start_cell, goal_cell)
                    fine_grid_fallback["path_found"] = bool(fine_path)
                    fine_grid_fallback["path_cell_count"] = int(len(fine_path))
                    if fine_path:
                        raw_path = fine_path
                        planner_path_source = "fine_grid_fallback"
                path = self.smooth_path(raw_path)
                if len(path) < 2:
                    diagnostic = self.path_grid_diagnostic(
                        grid_msg,
                        grid,
                        planning_blocked,
                        start_cell,
                        goal_cell,
                        raw_path,
                        path,
                        (tx_base, ty_base),
                        None,
                        astar_lateral_bias_active,
                        corridor_center_target,
                        start_footprint_clearance,
                    )
                    diagnostic["planner_path_source"] = planner_path_source
                    diagnostic["fine_grid_fallback"] = fine_grid_fallback
                    if (
                        centerline_target
                        and not self.args.disable_path_stability
                        and self.no_path_hold_count < self.args.path_stability_max_hold_steps
                    ):
                        self.no_path_hold_count += 1
                        self.prev_cmd = (0.0, 0.0)
                        published = self.publish_twist(0.0, 0.0, self.args.command_slice_sec)
                        summary["steps"].append(
                            {
                                "step": step_idx,
                                "pose_x_y_yaw": pose,
                                "target_base_xy": [tx_base, ty_base],
                                "original_target_base_xy": list(original_target_base_xy),
                                "unilateral_clearance_target": unilateral_clearance_target,
                                "corridor_center_target": corridor_center_target,
                                "pointcloud_wall_heading_prior": wall_heading_prior,
                                "distance_to_target_m": distance,
                                "path_cell_count": len(path),
                                "waypoint_base_xy": None,
                                "lookahead_path_index": None,
                                "cmd_linear_x": 0.0,
                                "cmd_angular_z": 0.0,
                                "published_count": published,
                                "dwa": {"blocked": False, "path_stability_hold": True, "hold_reason": "no_path_transient"},
                                "costmap_preprocess": costmap_preprocess,
                                "path_diagnostic": diagnostic,
                            }
                        )
                        continue
                    summary["failure_diagnostic"] = diagnostic
                    summary["failure_costmap_preprocess"] = costmap_preprocess
                    final_decision = "BLOCK_ASTAR_DWA_BLOCKED_NO_PATH"
                    break
                self.no_path_hold_count = 0
                native_lookahead = self.select_lookahead_waypoint(path, grid_msg)
                waypoint_x, waypoint_y, look_idx = native_lookahead
                waypoint_xy = (waypoint_x, waypoint_y)
                p_through_lookahead: Optional[Dict[str, Any]] = None
                if target.get("source") == "PORTAL_G14_P_THROUGH":
                    p_through_lookahead = self.select_p_through_executable_lookahead(
                        path,
                        grid_msg,
                        planning_blocked,
                        native_lookahead,
                        distance,
                    )
                    selected_xy = p_through_lookahead.get("selected_lookahead_xy")
                    if selected_xy is None:
                        summary["failure_diagnostic"] = p_through_lookahead
                        summary["failure_costmap_preprocess"] = costmap_preprocess
                        final_decision = "P_THROUGH_ASTAR_PATH_NOT_FORWARD_EXECUTABLE"
                        break
                    waypoint_xy = (float(selected_xy[0]), float(selected_xy[1]))
                    look_idx = int(p_through_lookahead["selected_lookahead_path_index"])
                path_diagnostic = self.path_grid_diagnostic(
                    grid_msg,
                    grid,
                    planning_blocked,
                    start_cell,
                    goal_cell,
                    raw_path,
                    path,
                    (tx_base, ty_base),
                    waypoint_xy,
                    astar_lateral_bias_active,
                    corridor_center_target,
                    start_footprint_clearance,
                )
                path_diagnostic["planner_path_source"] = planner_path_source
                path_diagnostic["fine_grid_fallback"] = fine_grid_fallback
                if p_through_lookahead is not None:
                    path_diagnostic["p_through_executable_lookahead"] = p_through_lookahead
                waypoint_lateral_delta = waypoint_y - ty_base
                waypoint_lateral_anomaly = (
                    centerline_target
                    and abs(ty_base) <= self.args.centerline_target_y_threshold_m
                    and abs(waypoint_lateral_delta) >= self.args.waypoint_lateral_anomaly_threshold_m
                )
                if waypoint_lateral_anomaly and not self.args.disable_path_stability:
                    if self.path_stability_hold_count < self.args.path_stability_max_hold_steps:
                        self.path_stability_hold_count += 1
                        self.prev_cmd = (0.0, 0.0)
                        published = self.publish_twist(0.0, 0.0, self.args.command_slice_sec)
                        summary["steps"].append(
                            {
                                "step": step_idx,
                                "pose_x_y_yaw": pose,
                                "target_base_xy": [tx_base, ty_base],
                                "original_target_base_xy": list(original_target_base_xy),
                                "unilateral_clearance_target": unilateral_clearance_target,
                                "corridor_center_target": corridor_center_target,
                                "pointcloud_wall_heading_prior": wall_heading_prior,
                                "distance_to_target_m": distance,
                                "path_cell_count": len(path),
                                "waypoint_base_xy": list(waypoint_xy),
                                "lookahead_path_index": look_idx,
                                "p_through_executable_lookahead": p_through_lookahead,
                                "cmd_linear_x": 0.0,
                                "cmd_angular_z": 0.0,
                                "published_count": published,
                                "dwa": {
                                    "blocked": False,
                                    "path_stability_hold": True,
                                    "hold_reason": "waypoint_lateral_anomaly",
                                    "waypoint_to_target_lateral_delta_m": waypoint_lateral_delta,
                                },
                                "costmap_preprocess": costmap_preprocess,
                                "path_diagnostic": path_diagnostic,
                            }
                        )
                        continue
                    summary["failure_diagnostic"] = path_diagnostic
                    summary["failure_costmap_preprocess"] = costmap_preprocess
                    final_decision = "BLOCK_ASTAR_DWA_BLOCKED_UNSTABLE_PATH"
                    break
                self.path_stability_hold_count = 0
                room_local_path_xy = (
                    tuple(self.cell_to_local_xy(cell, grid_msg) for cell in path)
                    if bool(getattr(self, "phase2_productive_admission_active", False))
                    else None
                )
                # choose_dwa updates prev_cmd as part of its existing native
                # dynamic-window bookkeeping.  C0 must freeze the command
                # that was actually authoritative at this decision epoch.
                previous_cmd_before_choose = tuple(self.prev_cmd)
                v, w, dwa = self.choose_dwa(
                    grid_msg,
                    planning_blocked,
                    waypoint_xy,
                    (tx_base, ty_base),
                    distance,
                    wall_heading_prior,
                    p_through_safe_moving_eligibility=(target.get("source") == "PORTAL_G14_P_THROUGH"),
                    room_search_safe_moving_eligibility=(target.get("source") == "ROOM_SEARCH_V2"),
                    p_through_target_relative_angular_scoring=(
                        target.get("source") == "PORTAL_G14_P_THROUGH"
                    ),
                    portal_relative_selection=(target.get("source") == "PORTAL_G14_P_THROUGH"),
                    portal_center_odom=(target.get("frozen_geometry") or {}).get("portal_center_odom"),
                    portal_normal_odom=(target.get("frozen_geometry") or {}).get("portal_normal_odom"),
                    portal_p_pre_odom=target.get("P_pre_odom"),
                    pose_odom=pose,
                    target_in_front=(tx_base > 0.0),
                    astar_path_exists=(len(path) >= 2),
                    room_local_path_xy=room_local_path_xy,
                )
                dwa["dwa_decision_id"] = safety_decision_id
                dwa["same_state_safety_binding"] = copy.deepcopy(safety_binding)
                continuation_evidence_requested = bool(
                    getattr(self.args, "continuation_viability_shadow", False)
                    or getattr(self, "continuation_local_selection_authority_active", False)
                )
                if (
                    continuation_evidence_requested
                    and self.local_control_mode == LOCAL_CONTROL_MODE_ROOM_LOCAL
                    and bool(getattr(self, "phase2_productive_admission_active", False))
                ):
                    # The frozen evaluator stays pure in both modes.  Shadow
                    # consumers receive observer-only evidence; the separate
                    # guarded authority consumes a second, explicitly named
                    # evidence field below.
                    try:
                        epoch = FrozenRoomLocalEpoch.from_live_inputs(
                            epoch_id="continuation_evidence_step_%04d" % int(step_idx),
                            pose_odom_xy_yaw=pose,
                            grid_msg=grid_msg,
                            status_payload=status_payload,
                            args=self.args,
                            previous_cmd=previous_cmd_before_choose,
                            wall_heading_prior=wall_heading_prior,
                        )
                        continuation_candidate = FrozenRoomLocalCandidate.from_mapping({
                            "candidate_id": "current_runner_target",
                            "target_xy_team_livox_odom": target_xy,
                        })
                        continuation_evidence = evaluate_frozen_room_local_continuation_cohort(
                            epoch,
                            continuation_candidate,
                            dwa.get("room_local_productivity_candidates") or [],
                        )
                    except Exception as exc:
                        # A missing or non-equivalent input is evidence
                        # incomplete, never a claim of non-viability.
                        continuation_evidence = {
                            "continuation_status": CONTINUATION_UNKNOWN,
                            "reason": "C0_EVALUATION_EXCEPTION",
                            "exception_type": type(exc).__name__,
                            "commands_published": False,
                            "selection_performed": False,
                            "phase3_state_mutated": False,
                        }
                    if bool(getattr(self.args, "continuation_viability_shadow", False)):
                        dwa["continuation_viability_shadow"] = copy.deepcopy(continuation_evidence)
                    if bool(getattr(self, "continuation_local_selection_authority_active", False)):
                        dwa["continuation_viability_authority_evidence"] = copy.deepcopy(continuation_evidence)
                continuation_selection_terminal: Optional[Dict[str, Any]] = None
                if (
                    bool(getattr(self, "continuation_local_selection_authority_active", False))
                    and dwa.get("room_local_productivity_set_status") == "PRODUCTIVE_TRANSLATIONAL_SET_NONEMPTY"
                ):
                    phase3_recovery_context = bool(
                        getattr(self, "phase3_orientation_recovery_active", False)
                        and self.orientation_intent is not None
                    )
                    v, w, continuation_selection = self.apply_continuation_aware_local_selection(
                        dwa,
                        v,
                        w,
                        dwa.get("continuation_viability_authority_evidence") or {
                            "continuation_status": CONTINUATION_UNKNOWN,
                            "reason": "CONTINUATION_AUTHORITY_EVIDENCE_UNAVAILABLE",
                            "records": [],
                        },
                        phase3_recovery_context=phase3_recovery_context,
                    )
                    if phase3_recovery_context:
                        # Each translation branch ends the old one-slice
                        # recovery episode before publication.  A terminal
                        # no-admissible result also consumes it, preventing an
                        # immediate stale-orientation loop.
                        dwa["phase3_continuation_authority"] = self.close_phase3_orientation_intent_for_continuation(
                            continuation_selection,
                        )
                    if not bool(continuation_selection["translation_allowed"]):
                        continuation_selection_terminal = continuation_selection
                decision_id = ""
                if self.validation_enabled:
                    self.validation_decision_sequence += 1
                    self.validation_astar_evaluation_sequence += 1
                    self.validation_path_anchor_sequence += 1
                    decision_id = "%s:decision_%04d" % (
                        self.validation_runner_invocation_id, self.validation_decision_sequence,
                    )
                    self.validation_current_decision_id = decision_id
                if continuation_selection_terminal is not None:
                    summary["failure_diagnostic"] = continuation_selection_terminal
                    summary["failure_costmap_preprocess"] = costmap_preprocess
                    final_decision = str(continuation_selection_terminal["terminal_reason"])
                    self.emit_validation_event(
                        final_decision,
                        decision_id=decision_id or None,
                        continuation_local_selection=continuation_selection_terminal,
                    )
                    break
                if isinstance(dwa.get("high_level_locomotion_shadow"), dict):
                    # Derived exclusively from the path just used for this DWA
                    # call.  No planner invocation or path mutation occurs here.
                    local_path = [self.cell_to_local_xy(cell, grid_msg) for cell in path[:5]]
                    deltas = []
                    for first, second in zip(local_path, local_path[1:]):
                        dx = float(second[0]) - float(first[0])
                        dy = float(second[1]) - float(first[1])
                        if dx != 0.0 or dy != 0.0:
                            deltas.append(math.atan2(dy, dx))
                    initial_heading = deltas[0] if deltas else None
                    target_bearing = math.atan2(float(ty_base), float(tx_base))
                    dwa["high_level_locomotion_shadow"].update({
                        "astar_approach_geometry_status": "SAME_PRODUCTION_ASTAR_PATH",
                        "PATH_APPROACH_EVIDENCE_PRESENT": bool(initial_heading is not None),
                        "raw_path_cell_count": int(len(raw_path)),
                        "smoothed_path_cell_count": int(len(path)),
                        "first_nontrivial_segment_heading_base_rad": initial_heading,
                        "robot_yaw_to_initial_path_delta_rad": initial_heading,
                        "candidate_bearing_base_rad": target_bearing,
                        "initial_path_to_candidate_bearing_delta_rad": (
                            normalize_angle(float(target_bearing) - float(initial_heading))
                            if initial_heading is not None else None
                        ),
                        "first_path_segment_headings_base_rad": deltas,
                    })
                if (
                    bool(getattr(self, "phase3_orientation_recovery_active", False))
                    and dwa.get("room_local_productivity_set_status") == "PRODUCTIVE_TRANSLATIONAL_SET_NONEMPTY"
                    and self.orientation_intent is not None
                    and not bool(getattr(self, "continuation_local_selection_authority_active", False))
                ):
                    # A fresh Phase-2 productive set is the only success
                    # authority; stop recovery before selecting translation.
                    self.orientation_intent = None
                    dwa["phase3_orientation_intent_cleared"] = "PRODUCTIVE_TRANSLATION_RESTORED"
                if (
                    bool(getattr(self, "phase3_orientation_recovery_active", False))
                    and dwa.get("room_local_productivity_set_status") == "NO_PRODUCTIVE_TRANSLATION"
                ):
                    validation_orientation_intent_before = orientation_intent_record(self.orientation_intent)
                    recovery = self.phase3_recovery_action(
                        target=target,
                        pose=pose,
                        grid_msg=grid_msg,
                        blocked=planning_blocked,
                        waypoint_xy=waypoint_xy,
                        target_base_xy=(tx_base, ty_base),
                        distance_to_goal=distance,
                        room_local_path_xy=room_local_path_xy or (),
                    )
                    dwa["phase3_recovery"] = {
                        "action": recovery["action"], "reason": recovery.get("reason"),
                        "selection": recovery.get("selection"),
                        "recovery_set": recovery_set_record(recovery["recovery_set"]),
                        "orientation_intent": orientation_intent_record(recovery.get("intent") or self.orientation_intent),
                    }
                    self.emit_validation_event(
                        "PHASE3_RECOVERYSET_RESULT", decision_id=decision_id or None,
                        recovery=dwa["phase3_recovery"], fresh_replan_required=True,
                        orientation_intent_before=validation_orientation_intent_before,
                        orientation_intent_after=orientation_intent_record(recovery.get("intent") or self.orientation_intent),
                    )
                    if recovery["action"] == "ORIENTATION_SLICE":
                        w = float(recovery["selection"]["w_radps"])
                        self.prev_cmd = (0.0, w)
                        published = self.publish_twist(0.0, w, self.args.command_slice_sec)
                        summary["steps"].append({
                            "step": step_idx, "pose_x_y_yaw": pose,
                            "target_base_xy": [tx_base, ty_base], "path_cell_count": len(path),
                            "waypoint_base_xy": list(waypoint_xy), "cmd_linear_x": 0.0,
                            "cmd_angular_z": w, "published_count": published, "dwa": dwa,
                            "costmap_preprocess": costmap_preprocess, "path_diagnostic": path_diagnostic,
                        })
                        # The next loop begins with fresh odom/grid/status and a
                        # fresh A* path before Phase-2 admission is run again.
                        continue
                    summary["failure_diagnostic"] = dwa["phase3_recovery"]
                    summary["failure_costmap_preprocess"] = costmap_preprocess
                    final_decision = "ROOM_LOCAL_NO_PRODUCTIVE_MOTION"
                    self.emit_validation_event(
                        "ROOM_LOCAL_NO_PRODUCTIVE_MOTION", decision_id=decision_id or None,
                        reason=recovery.get("reason"),
                    )
                    break
                if dwa.get("blocked"):
                    failure_snapshot = self.build_dwa_no_cmd_failure_snapshot(
                        step_index=step_idx,
                        odom=odom,
                        grid_msg=grid_msg,
                        status_payload=status_payload,
                        target=target,
                        pose=pose,
                        target_base_xy=(tx_base, ty_base),
                        waypoint_base_xy=waypoint_xy,
                        raw_path=raw_path,
                        path=path,
                        raw_grid=raw_grid,
                        occupied_inflated=occupied_inflated,
                        planning_blocked=planning_blocked,
                        distance_to_goal=distance,
                        dwa_result=dwa,
                    )
                    failure_snapshot_path = self.write_dwa_no_cmd_failure_snapshot(failure_snapshot)
                    summary["dwa_no_cmd_failure_snapshot"] = {
                        "path": str(failure_snapshot_path),
                        "step_index": int(step_idx),
                        "candidate_counts": failure_snapshot["candidate_counts"],
                    }
                    summary["failure_diagnostic"] = path_diagnostic
                    summary["failure_costmap_preprocess"] = costmap_preprocess
                    final_decision = "BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD"
                    break
                provisional_v, provisional_w = float(v), float(w)
                command_mutation_chain: List[Dict[str, Any]] = []
                final_command_reason = (
                    "CONTINUATION_LOCAL_SELECTION"
                    if isinstance(dwa.get("continuation_local_selection"), dict)
                    else "DWA_LOCAL_SELECTION"
                )
                if self.best_distance_seen is None or distance < self.best_distance_seen - self.args.min_progress_m:
                    self.best_distance_seen = distance
                    self.no_progress_count = 0
                else:
                    self.no_progress_count += 1
                if self.no_progress_count >= self.args.no_progress_steps:
                    before_override = {"v": float(v), "w": float(w)}
                    v = 0.0
                    w = self.recovery_turn_sign * self.args.recovery_angular_z
                    self.recovery_turn_sign *= -1.0
                    self.no_progress_count = 0
                    dwa["recovery"] = "in_place_turn"
                    command_mutation_chain.append({
                        "reason": "NO_PROGRESS_RECOVERY",
                        "before": before_override,
                        "after": {"v": float(v), "w": float(w)},
                    })
                    final_command_reason = "NO_PROGRESS_RECOVERY"
                before_p_pre_guard = {"v": float(v), "w": float(w)}
                v, p_pre_goal_region_motion_guard = self.apply_p_pre_goal_region_motion_guard(
                    target,
                    (tx_base, ty_base),
                    v,
                    w,
                )
                if abs(float(v) - before_p_pre_guard["v"]) > DWA_NUMERIC_EPS:
                    command_mutation_chain.append({
                        "reason": "P_PRE_GOAL_REGION_MOTION_GUARD",
                        "before": before_p_pre_guard,
                        "after": {"v": float(v), "w": float(w)},
                    })
                    if final_command_reason != "NO_PROGRESS_RECOVERY":
                        final_command_reason = "P_PRE_GOAL_REGION_MOTION_GUARD"
                before_centerline_correction = {"v": float(v), "w": float(w)}
                w, centerline_tracking_correction = self.apply_scoped_centerline_tracking_correction(
                    v,
                    w,
                    target,
                    pose,
                    centerline_target,
                    dwa,
                    pre_room_zone_centerline_correction=pre_room_zone_centerline_correction,
                )
                if abs(float(w) - before_centerline_correction["w"]) > DWA_NUMERIC_EPS:
                    command_mutation_chain.append({
                        "reason": "CENTERLINE_TRACKING_CORRECTION",
                        "before": before_centerline_correction,
                        "after": {"v": float(v), "w": float(w)},
                    })
                    if final_command_reason != "NO_PROGRESS_RECOVERY":
                        final_command_reason = "CENTERLINE_TRACKING_CORRECTION"
                before_imu_heading_hold = {"v": float(v), "w": float(w)}
                w, imu_heading_hold = self.apply_imu_heading_hold(v, w, dwa)
                if abs(float(w) - before_imu_heading_hold["w"]) > DWA_NUMERIC_EPS:
                    command_mutation_chain.append({
                        "reason": "IMU_HEADING_HOLD_CORRECTION",
                        "before": before_imu_heading_hold,
                        "after": {"v": float(v), "w": float(w)},
                    })
                    if final_command_reason != "NO_PROGRESS_RECOVERY":
                        final_command_reason = "IMU_HEADING_HOLD_CORRECTION"
                final_command_action = self.finalize_command_authority(
                    dwa,
                    provisional_v=provisional_v,
                    provisional_w=provisional_w,
                    final_v=float(v),
                    final_w=float(w),
                    final_command_reason=final_command_reason,
                    mutation_chain=command_mutation_chain,
                )
                final_command_action["same_state_safety_attribution"] = build_dwa_slice_attribution(
                    decision_id=safety_decision_id,
                    safety_binding=safety_binding,
                    pose_x_y_yaw=pose,
                    dwa=dwa,
                    raw_v=float(v),
                    raw_w=float(w),
                )
                if self.validation_enabled:
                    self.emit_validation_event(
                        "ROOM_LOCAL_DECISION", decision_id=decision_id, step_index=int(step_idx),
                        local_control_mode=self.local_control_mode, pose=list(pose), target_base_xy=[tx_base, ty_base],
                        input_acquisition_sequence=int(self.validation_input_acquisition_sequence),
                        grid_status_acquisition_sequence=int(self.validation_input_acquisition_sequence),
                        astar_evaluation_sequence=int(self.validation_astar_evaluation_sequence),
                        path_anchor_sequence=int(self.validation_path_anchor_sequence),
                        path_cell_count=int(len(path)), selected_v=float(v), selected_w=float(w),
                        phase2_productivity_status=dwa.get("room_local_productivity_set_status"),
                        phase2_safe_count=dwa.get("room_local_safe_translational_candidate_count"),
                        phase2_productive_count=dwa.get("room_local_productive_translational_candidate_count"),
                        phase2_margin_m=dwa.get("room_local_meaningful_progress_margin_m"),
                        phase3_recovery=dwa.get("phase3_recovery"),
                        continuation_local_selection=dwa.get("continuation_local_selection"),
                        phase3_continuation_authority=dwa.get("phase3_continuation_authority"),
                        provisional_local_motion=dwa.get("provisional_local_motion"),
                        final_command_action=final_command_action,
                    )
                published = self.publish_twist(v, w, self.args.command_slice_sec)
                self.record_final_command_publish_result(
                    final_command_action, published, execute=bool(self.args.execute),
                )
                summary["steps"].append(
                    {
                        "step": step_idx,
                        "pose_x_y_yaw": pose,
                        "target_base_xy": [tx_base, ty_base],
                        "original_target_base_xy": list(original_target_base_xy),
                        "unilateral_clearance_target": unilateral_clearance_target,
                        "corridor_center_target": corridor_center_target,
                        "pointcloud_wall_heading_prior": wall_heading_prior,
                        "distance_to_target_m": distance,
                        "path_cell_count": len(path),
                        "waypoint_base_xy": list(waypoint_xy),
                        "lookahead_path_index": look_idx,
                        "p_through_executable_lookahead": p_through_lookahead,
                        "cmd_linear_x": v,
                        "cmd_angular_z": w,
                        "published_count": published,
                        "dwa": dwa,
                        # Kept only as an audit-key compatibility alias.  The
                        # authoritative production decision is the goal-region
                        # guard immediately above.
                        "p_pre_arc_speed_cap": p_pre_goal_region_motion_guard,
                        "p_pre_goal_region_motion_guard": p_pre_goal_region_motion_guard,
                        "centerline_tracking_correction": centerline_tracking_correction,
                        "imu_heading_hold": imu_heading_hold,
                        "costmap_preprocess": costmap_preprocess,
                        "path_diagnostic": path_diagnostic,
                    }
                )
                timeout_trigger = self.current_timeout_trigger()
                if timeout_trigger == "wall_watchdog_timeout":
                    final_decision = "BLOCK_ASTAR_DWA_WALL_WATCHDOG_TIMEOUT"
                    break
                if timeout_trigger == "sim_timeout":
                    final_decision = "BLOCK_ASTAR_DWA_SIM_TIMEOUT"
                    break
                if step_idx == self.args.max_steps - 1:
                    terminal_reach = self.post_command_terminal_reach_check(target_xy, odom)
                    summary["post_command_terminal_reach_check"] = terminal_reach
                    if terminal_reach.get("reached"):
                        final_decision = "BLOCK_ASTAR_DWA_REACHED_GOAL"
                        break
            else:
                final_decision = "BLOCK_ASTAR_DWA_MAX_STEPS"
        finally:
            self.stop()
            self.orientation_intent = None

        clock_status = self.runtime_clock_status()
        summary.update(clock_status)
        summary["run_wall_start_sec"] = self.run_wall_start_sec
        summary["run_sim_start_sec"] = self.run_sim_start_sec
        summary["use_sim_time"] = self.use_sim_time
        summary["sim_time_valid"] = self.sim_time_valid
        summary["timeout_trigger"] = self.timeout_trigger
        summary["final_decision"] = final_decision
        summary["legacy_timeout_decision"] = (
            "BLOCK_ASTAR_DWA_TIMEOUT"
            if self.timeout_trigger in {"sim_timeout", "wall_watchdog_timeout"}
            else None
        )
        summary["status_payload"] = status_payload
        summary["validation"]["event_count"] = len(self.validation_events)
        summary["validation"]["abort_reason"] = self.validation_abort_reason
        self.emit_validation_event("RUNNER_INVOCATION_END", final_decision=final_decision)
        write_json(OUT / "block_astar_dwa_mature_summary.json", summary)
        return summary


def _pure_room_local_runner(epoch: FrozenRoomLocalEpoch) -> "BlockAStarDwaRunner":
    """Build a private, non-initialized runner shell for pure computation only."""
    runner = object.__new__(BlockAStarDwaRunner)
    runner.args = epoch.runner_args()
    runner.local_control_mode = str(getattr(runner.args, "local_control_mode", LOCAL_CONTROL_MODE_TRANSIT))
    validate_phase2_productive_admission_activation(runner.args)
    runner.phase2_productive_admission_active = phase2_productive_admission_is_active(runner.args)
    runner.phase3_orientation_recovery_active = False
    runner.orientation_intent = None
    runner.prev_cmd = tuple(epoch.previous_cmd)
    # These are the only mutable fields touched by the existing planning/DWA
    # helpers below.  They belong to this private shell, never a live runner.
    runner.path_stability_hold_count = 0
    runner.no_path_hold_count = 0
    runner.unilateral_clearance_safety_side = None
    runner.unilateral_clearance_missing_count = 0
    runner.latest_wall_heading_prior = epoch.wall_heading_prior()
    runner.wall_heading_point_buffer = []
    runner.latest_imu = None
    runner.imu_heading_anchor_yaw = None
    runner.imu_heading_hold_active_last = False
    runner.pub = None
    return runner


def evaluate_frozen_room_local_candidate(
    epoch: FrozenRoomLocalEpoch,
    candidate: FrozenRoomLocalCandidate,
) -> Dict[str, Any]:
    """Run existing ROOM_LOCAL formal + DWA + Phase-2 logic without ROS.

    It owns a new private shell and message-shaped copy for every call.  The
    computation invokes the live runner's A*, Grid semantics, collision arc,
    dynamic-window and `choose_dwa` methods directly; it has no publisher,
    subscriber, sleep, clock, state-machine, breadcrumb, or orientation path.
    """
    runner = _pure_room_local_runner(epoch)
    grid_msg = epoch.grid_message()
    status = epoch.status_payload()
    target_xy = tuple(candidate.target_odom_xy)
    result: Dict[str, Any] = {
        "epoch_id": epoch.epoch_id,
        "candidate_id": candidate.candidate_id,
        "rank": int(candidate.rank),
        "sector_id": candidate.sector_id,
        "candidate_type": candidate.candidate_type,
        "candidate_heading_rad": candidate.heading_rad,
        "formal_status": "EVALUATION_NOT_AVAILABLE",
        "terminal_reason": None,
        "exact_reject_stage": None,
        "path_exists": None,
        "path_cell_count": None,
        "path_length_m": None,
        "target_distance_m": None,
        "within_goal_tolerance": None,
        "p_through_astar_path_exists": None,
        "safe_moving_candidate_count": None,
        "productive_moving_candidate_count": None,
        "motion_candidates": [],
        "winner": None,
        "commands_published": False,
        "ros_read": False,
        "ros_write": False,
        "persistent_state_mutated": False,
    }
    if runner.local_control_mode != LOCAL_CONTROL_MODE_ROOM_LOCAL:
        result.update({"terminal_reason": "TRANSIT_ROOM_LOCAL_EVALUATOR_NOT_APPLICABLE",
                       "exact_reject_stage": "MODE_GUARD"})
        return result
    qualification_errors = runner.grid_qualification_errors(grid_msg, status)
    if qualification_errors:
        result.update({"formal_status": "FORMALLY_ILLEGAL", "terminal_reason": "GRID_STATUS_CONTRACT_INVALID",
                       "exact_reject_stage": "GRID_QUALIFICATION", "qualification_errors": qualification_errors})
        return result
    if status.get("local_traversability_status") != "FREE_SUPPORTED":
        result.update({"formal_status": "EVALUATION_NOT_AVAILABLE", "terminal_reason": "DECISION_GLOBAL_L3V_STATUS",
                       "exact_reject_stage": "GLOBAL_STATUS"})
        return result
    pose = tuple(epoch.pose_odom_xy_yaw)
    tx_base, ty_base = target_to_base(target_xy, pose)
    distance = math.hypot(tx_base, ty_base)
    result.update({"target_base_xy": [float(tx_base), float(ty_base)], "target_distance_m": float(distance),
                   "goal_tolerance_m": float(runner.args.goal_tolerance_m),
                   "within_goal_tolerance": bool(distance <= float(runner.args.goal_tolerance_m))})
    if distance <= float(runner.args.goal_tolerance_m):
        result.update({"formal_status": "FORMALLY_ILLEGAL", "terminal_reason": "BLOCK_ASTAR_DWA_REACHED_GOAL",
                       "exact_reject_stage": "GOAL_TOLERANCE", "path_exists": False,
                       "p_through_astar_path_exists": False})
        return result
    target = {"source": "ROOM_SEARCH_V2", "target_xy_team_livox_odom": list(target_xy)}
    raw_grid = runner.grid_array(grid_msg)
    grid, preprocess = runner.apply_centerline_thin_barrier_clearing(grid_msg, raw_grid, (tx_base, ty_base), False, status)
    occupied_inflated = runner.occupied_inflated_mask(grid, float(grid_msg.info.resolution))
    blocked = runner.inflate_obstacles(grid, float(grid_msg.info.resolution))
    start_cell = runner.local_xy_to_cell(0.0, 0.0, grid_msg)
    if start_cell is None:
        result.update({"formal_status": "EVALUATION_NOT_AVAILABLE", "terminal_reason": "START_OUT_OF_GRID",
                       "exact_reject_stage": "START_CELL"})
        return result
    planning_blocked, start_clearance = runner.apply_start_footprint_clearance(
        grid_msg, raw_grid, blocked, occupied_inflated, qualification_passed=True,
    )
    result["start_footprint_clearance"] = start_clearance
    (tx_base, ty_base), target_shape = runner.unilateral_clearance_target_shape(
        grid_msg, target, grid, planning_blocked, (tx_base, ty_base),
    )
    result["target_shape"] = target_shape
    distance = math.hypot(tx_base, ty_base)
    goal_x = max(0.0, min(tx_base, runner.args.max_goal_x_m))
    goal_y = max(-runner.args.max_goal_abs_y_m, min(ty_base, runner.args.max_goal_abs_y_m))
    goal_cell = runner.local_xy_to_cell(goal_x, goal_y, grid_msg)
    if goal_cell is None:
        result.update({"formal_status": "FORMALLY_ILLEGAL", "terminal_reason": "BLOCK_ASTAR_DWA_GRID_TARGET_OUT_OF_BOUNDS",
                       "exact_reject_stage": "GOAL_CELL", "path_exists": False, "p_through_astar_path_exists": False})
        return result
    raw_path = runner.block_astar(planning_blocked, start_cell, goal_cell, grid_msg, None)
    planner_path_source = "coarse_block_astar"
    if len(raw_path) < 2:
        fine_path = runner.fine_grid_astar_fallback(planning_blocked, start_cell, goal_cell)
        if fine_path:
            raw_path, planner_path_source = fine_path, "fine_grid_fallback"
    path = runner.smooth_path(raw_path)
    result.update({"path_exists": bool(len(path) >= 2), "path_cell_count": int(len(path)),
                   "path_length_m": max(0.05, len(path) * float(grid_msg.info.resolution)) if path else None,
                   "planner_path_source": planner_path_source, "costmap_preprocess": preprocess,
                   "p_through_astar_path_exists": bool(len(path) >= 2)})
    if len(path) < 2:
        result.update({"formal_status": "FORMALLY_ILLEGAL", "terminal_reason": "BLOCK_ASTAR_DWA_BLOCKED_NO_PATH",
                       "exact_reject_stage": "ASTAR"})
        return result
    waypoint_x, waypoint_y, look_index = runner.select_lookahead_waypoint(path, grid_msg)
    room_local_path_xy = tuple(runner.cell_to_local_xy(cell, grid_msg) for cell in path)
    v, w, dwa = runner.choose_dwa(
        grid_msg, planning_blocked, (waypoint_x, waypoint_y), (tx_base, ty_base), distance,
        epoch.wall_heading_prior(), room_search_safe_moving_eligibility=True,
        pose_odom=pose, target_in_front=(tx_base > 0.0), astar_path_exists=True,
        room_local_path_xy=room_local_path_xy,
    )
    result.update({
        "lookahead_path_index": int(look_index), "lookahead_xy": [float(waypoint_x), float(waypoint_y)],
        "safe_moving_candidate_count": int(dwa.get("safe_moving_candidate_count") or 0),
        "productive_moving_candidate_count": int(dwa.get("room_local_productive_translational_candidate_count") or 0),
        "motion_candidates": copy.deepcopy(dwa.get("room_local_productivity_candidates") or []),
        "dwa": copy.deepcopy(dwa),
        "winner": None if bool(dwa.get("blocked")) else {"v": float(v), "w": float(w), "reason": "PRODUCTION_CHOOSE_DWA"},
    })
    if bool(dwa.get("blocked")) or int(dwa.get("safe_moving_candidate_count") or 0) <= 0:
        result.update({"formal_status": "FORMALLY_ILLEGAL", "terminal_reason": "BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD",
                       "exact_reject_stage": "DWA"})
    else:
        result.update({"formal_status": "FORMALLY_EXECUTABLE", "terminal_reason": "FORMAL_DRY_PREFLIGHT_PASS",
                       "exact_reject_stage": None})
    return result


def evaluate_frozen_room_local_cohort(
    epoch: FrozenRoomLocalEpoch,
    candidates: Sequence[FrozenRoomLocalCandidate],
) -> Dict[str, Any]:
    """Evaluate every supplied bounded representative against one epoch only."""
    records: List[Dict[str, Any]] = []
    timings_ns: List[int] = []
    for candidate in candidates:
        started_ns = time.perf_counter_ns()
        record = evaluate_frozen_room_local_candidate(epoch, candidate)
        timings_ns.append(time.perf_counter_ns() - started_ns)
        if record.get("epoch_id") != epoch.epoch_id or record.get("candidate_id") != candidate.candidate_id:
            raise ValueError("frozen_room_local_cohort_identity_mismatch")
        records.append(record)
    return {
        "epoch_id": epoch.epoch_id,
        "candidate_count": len(records),
        "records": records,
        "candidate_elapsed_ns": timings_ns,
        "commands_published": False,
        "selection_performed": False,
        "same_epoch_identity": all(record.get("epoch_id") == epoch.epoch_id for record in records),
        "complete": len(records) == len(candidates),
    }


CONTINUATION_VIABLE = "CONTINUATION_VIABLE"
CONTINUATION_NON_VIABLE = "CONTINUATION_NON_VIABLE"
CONTINUATION_UNKNOWN = "CONTINUATION_UNKNOWN"
CONTINUATION_SHADOW_MAX_MOTIONS = 6


@dataclass(frozen=True)
class FrozenContinuationGridContext:
    """Read-only Grid facts shared only inside one frozen C0 cohort."""

    epoch_id: str
    candidate_id: str
    grid_msg: Any
    raw_grid: np.ndarray
    planning_grid: np.ndarray
    occupied_inflated: np.ndarray
    base_blocked: np.ndarray
    cell_center_x_m: np.ndarray
    cell_center_y_m: np.ndarray
    raw_grid_sha256: str
    target_s0_xy: Tuple[float, float]
    costmap_preprocess: Dict[str, Any]
    target_shape: Dict[str, Any]
    shared_preprocessing_ns: int


def _readonly_continuation_array(array: np.ndarray) -> np.ndarray:
    copied = np.ascontiguousarray(array).copy()
    copied.setflags(write=False)
    return copied


def build_frozen_continuation_grid_context(
    epoch: FrozenRoomLocalEpoch,
    candidate: FrozenRoomLocalCandidate,
) -> FrozenContinuationGridContext:
    """Build once-per-cohort immutable Grid facts for the exact C0 path."""
    started_ns = time.perf_counter_ns()
    runner = _pure_room_local_runner(epoch)
    if runner.local_control_mode != LOCAL_CONTROL_MODE_ROOM_LOCAL or not runner.phase2_productive_admission_active:
        raise ValueError("ROOM_LOCAL_PHASE2_AUTHORITY_UNAVAILABLE")
    grid_msg = epoch.grid_message()
    status = epoch.status_payload()
    qualification_errors = runner.grid_qualification_errors(grid_msg, status)
    if qualification_errors or status.get("local_traversability_status") != "FREE_SUPPORTED":
        raise ValueError("FROZEN_GRID_STATUS_UNQUALIFIED")
    current_pose = tuple(epoch.pose_odom_xy_yaw)
    target_s0_xy = target_to_base(tuple(candidate.target_odom_xy), current_pose)
    target = {"source": "ROOM_SEARCH_V2", "target_xy_team_livox_odom": list(candidate.target_odom_xy)}
    raw_grid = runner.grid_array(grid_msg)
    planning_grid, preprocess = runner.apply_centerline_thin_barrier_clearing(
        grid_msg, raw_grid, target_s0_xy, False, status,
    )
    occupied_inflated = runner.occupied_inflated_mask(planning_grid, float(grid_msg.info.resolution))
    base_blocked = runner.inflate_obstacles(planning_grid, float(grid_msg.info.resolution))
    # In the current C0 target contract unilateral shaping exits at the source
    # guard before consulting this mask; retaining the call preserves its
    # existing report and makes that condition testable.
    shaped_target_s0_xy, target_shape = runner.unilateral_clearance_target_shape(
        grid_msg, target, planning_grid, base_blocked, target_s0_xy,
    )
    width, height = raw_grid.shape[1], raw_grid.shape[0]
    cell_center_x_m = np.asarray(
        [runner.cell_to_local_xy((x_index, 0), grid_msg)[0] for x_index in range(width)], dtype=np.float64,
    )
    cell_center_y_m = np.asarray(
        [runner.cell_to_local_xy((0, y_index), grid_msg)[1] for y_index in range(height)], dtype=np.float64,
    )
    raw_readonly = _readonly_continuation_array(raw_grid)
    planning_readonly = _readonly_continuation_array(planning_grid)
    return FrozenContinuationGridContext(
        epoch_id=epoch.epoch_id,
        candidate_id=candidate.candidate_id,
        grid_msg=grid_msg,
        raw_grid=raw_readonly,
        planning_grid=planning_readonly,
        occupied_inflated=_readonly_continuation_array(occupied_inflated),
        base_blocked=_readonly_continuation_array(base_blocked),
        cell_center_x_m=_readonly_continuation_array(cell_center_x_m),
        cell_center_y_m=_readonly_continuation_array(cell_center_y_m),
        raw_grid_sha256=hashlib.sha256(np.ascontiguousarray(raw_readonly).tobytes()).hexdigest(),
        target_s0_xy=(float(shaped_target_s0_xy[0]), float(shaped_target_s0_xy[1])),
        costmap_preprocess=copy.deepcopy(preprocess),
        target_shape=copy.deepcopy(target_shape),
        shared_preprocessing_ns=int(time.perf_counter_ns() - started_ns),
    )


def _continuation_motion_key(record: Mapping[str, Any]) -> Tuple[float, float]:
    return (float(record.get("v") or 0.0), float(record.get("w") or 0.0))


def schedule_continuation_productive_motions(
    productive_motions: Sequence[Mapping[str, Any]],
    *,
    cap: int = CONTINUATION_SHADOW_MAX_MOTIONS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically schedule evidence without changing native selection."""
    unique: List[Dict[str, Any]] = []
    seen = set()
    for record in productive_motions:
        if not bool(record.get("score_eligible")):
            continue
        key = _continuation_motion_key(record)
        if key in seen:
            continue
        seen.add(key)
        unique.append(copy.deepcopy(dict(record)))
    if len(unique) <= int(cap):
        return unique, []
    chosen_indices: List[int] = []
    for predicate in (
        lambda item: float(item.get("w") or 0.0) < -DWA_NUMERIC_EPS,
        lambda item: abs(float(item.get("w") or 0.0)) <= DWA_NUMERIC_EPS,
        lambda item: float(item.get("w") or 0.0) > DWA_NUMERIC_EPS,
    ):
        index = next((i for i, item in enumerate(unique) if predicate(item)), None)
        if index is not None and index not in chosen_indices:
            chosen_indices.append(index)
    extrema = sorted(range(len(unique)), key=lambda i: (float(unique[i].get("w") or 0.0), i))
    for index in (extrema[0], extrema[-1]):
        if index not in chosen_indices:
            chosen_indices.append(index)
    for index in range(len(unique)):
        if len(chosen_indices) >= int(cap):
            break
        if index not in chosen_indices:
            chosen_indices.append(index)
    chosen_set = set(chosen_indices[:int(cap)])
    return [unique[index] for index in chosen_indices[:int(cap)]], [
        item for index, item in enumerate(unique) if index not in chosen_set
    ]


def evaluate_frozen_room_local_continuation(
    epoch: FrozenRoomLocalEpoch,
    candidate: FrozenRoomLocalCandidate,
    productive_motion: Mapping[str, Any],
    *,
    shared_grid_context: Optional[FrozenContinuationGridContext] = None,
) -> Dict[str, Any]:
    """Pure depth-1 S1 replay using current A*/DWA/Phase2 helpers only."""
    started_ns = time.perf_counter_ns()
    result: Dict[str, Any] = {
        "epoch_id": epoch.epoch_id,
        "candidate_id": candidate.candidate_id,
        "motion": copy.deepcopy(dict(productive_motion)),
        "continuation_status": CONTINUATION_UNKNOWN,
        "reason": None,
        "next_safe_count": None,
        "next_productive_count": None,
        "commands_published": False,
        "ros_read": False,
        "ros_write": False,
        "persistent_state_mutated": False,
    }
    if not bool(productive_motion.get("score_eligible")):
        result["reason"] = "CURRENT_MOTION_NOT_PHASE2_PRODUCTIVE"
        return result
    endpoint = productive_motion.get("slice_endpoint_base_xy")
    slice_yaw = productive_motion.get("slice_endpoint_base_yaw_rad")
    if not (isinstance(endpoint, (list, tuple)) and len(endpoint) == 2 and all(finite_number(value) for value in endpoint)
            and finite_number(slice_yaw)):
        result["reason"] = "SLICE_ENDPOINT_MISSING_OR_INVALID"
        return result
    shared_preprocessing_ns = 0
    try:
        context = shared_grid_context or build_frozen_continuation_grid_context(epoch, candidate)
        if context.epoch_id != epoch.epoch_id or context.candidate_id != candidate.candidate_id:
            raise ValueError("FROZEN_CONTINUATION_CONTEXT_IDENTITY_MISMATCH")
        if shared_grid_context is None:
            shared_preprocessing_ns = int(context.shared_preprocessing_ns)
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    runner = _pure_room_local_runner(epoch)
    grid_msg = context.grid_msg
    s1_x, s1_y, s1_yaw = float(endpoint[0]), float(endpoint[1]), float(slice_yaw)
    if not all(finite_number(value) for value in (s1_x, s1_y, s1_yaw)):
        result["reason"] = "S1_TRANSFORM_UNVERIFIABLE"
        return result
    current_pose = tuple(epoch.pose_odom_xy_yaw)
    s1_pose = (
        float(current_pose[0]) + math.cos(float(current_pose[2])) * s1_x - math.sin(float(current_pose[2])) * s1_y,
        float(current_pose[1]) + math.sin(float(current_pose[2])) * s1_x + math.cos(float(current_pose[2])) * s1_y,
        normalize_angle(float(current_pose[2]) + s1_yaw),
    )
    result["slice_state_s1"] = {"base_xy": [s1_x, s1_y], "base_yaw_rad": s1_yaw, "odom_xy_yaw": list(s1_pose)}
    next_v, next_w = float(productive_motion.get("v") or 0.0), float(productive_motion.get("w") or 0.0)
    if next_v <= DWA_NUMERIC_EPS:
        result["reason"] = "CURRENT_PRODUCTIVE_MOTION_NOT_TRANSLATIONAL"
        return result
    runner.prev_cmd = (next_v, next_w)
    target_s0 = tuple(context.target_s0_xy)
    start_cell = runner.local_xy_to_cell(s1_x, s1_y, grid_msg)
    if start_cell is None:
        result["reason"] = "S1_OUTSIDE_FROZEN_GRID_SUPPORT"
        return result
    footprint_started_ns = time.perf_counter_ns()
    planning_blocked, start_clearance = runner.apply_start_footprint_clearance_from_shared_context(
        grid_msg,
        context.raw_grid,
        context.base_blocked,
        context.occupied_inflated,
        qualification_passed=True,
        cell_center_x_m=context.cell_center_x_m,
        cell_center_y_m=context.cell_center_y_m,
        raw_grid_sha256=context.raw_grid_sha256,
        start_base_xy=(s1_x, s1_y),
    )
    footprint_done_ns = time.perf_counter_ns()
    target_shape = copy.deepcopy(context.target_shape)
    goal_cell = runner.local_xy_to_cell(
        max(0.0, min(float(target_s0[0]), runner.args.max_goal_x_m)),
        max(-runner.args.max_goal_abs_y_m, min(float(target_s0[1]), runner.args.max_goal_abs_y_m)), grid_msg,
    )
    if goal_cell is None:
        result["reason"] = "S1_GOAL_OUTSIDE_FROZEN_GRID_SUPPORT"
        return result
    raw_path = runner.block_astar(planning_blocked, start_cell, goal_cell, grid_msg, None)
    if len(raw_path) < 2:
        raw_path = runner.fine_grid_astar_fallback(planning_blocked, start_cell, goal_cell)
    path = runner.smooth_path(raw_path)
    if len(path) < 2:
        result.update({"continuation_status": CONTINUATION_NON_VIABLE, "reason": "S1_ASTAR_NO_PATH",
                       "next_safe_count": 0, "next_productive_count": 0})
        return result
    astar_done_ns = time.perf_counter_ns()

    def to_s1(point: Tuple[float, float]) -> Tuple[float, float]:
        dx, dy = float(point[0]) - s1_x, float(point[1]) - s1_y
        return (math.cos(s1_yaw) * dx + math.sin(s1_yaw) * dy,
                -math.sin(s1_yaw) * dx + math.cos(s1_yaw) * dy)

    path_s1 = tuple(to_s1(runner.cell_to_local_xy(cell, grid_msg)) for cell in path)
    look_index = next((i for i, point in enumerate(path_s1[1:], start=1)
                       if math.hypot(point[0], point[1]) >= runner.args.lookahead_m), len(path_s1) - 1)
    waypoint_s1 = path_s1[look_index]
    target_s1 = to_s1((float(target_s0[0]), float(target_s0[1])))
    wall_prior = epoch.wall_heading_prior()
    if wall_prior is not None:
        wall_prior = copy.deepcopy(wall_prior)
        if wall_prior.get("active"):
            heading = wall_prior.get("heading_parallel_rad")
            if not finite_number(heading):
                result["reason"] = "WALL_HEADING_PRIOR_UNTRANSFORMABLE"
                return result
            wall_prior["heading_parallel_rad"] = normalize_angle(float(heading) - s1_yaw)
    distance = math.hypot(target_s1[0], target_s1[1])
    _v, _w, dwa = runner.choose_dwa(
        grid_msg, planning_blocked, waypoint_s1, target_s1, distance, wall_prior,
        room_search_safe_moving_eligibility=True, pose_odom=s1_pose,
        target_in_front=(target_s1[0] > 0.0), astar_path_exists=True,
        room_local_path_xy=path_s1, collision_reference_pose_base=(s1_x, s1_y, s1_yaw),
    )
    dwa_phase2_done_ns = time.perf_counter_ns()
    productive_count = int(dwa.get("room_local_productive_translational_candidate_count") or 0)
    safe_count = int(dwa.get("room_local_safe_translational_candidate_count") or 0)
    result.update({
        "continuation_status": CONTINUATION_VIABLE if productive_count > 0 else CONTINUATION_NON_VIABLE,
        "reason": "S1_NEXT_PRODUCTIVE_PRESENT" if productive_count > 0 else "S1_NEXT_PRODUCTIVE_EMPTY",
        "next_safe_count": safe_count, "next_productive_count": productive_count,
        "path_cell_count": len(path), "lookahead_path_index": int(look_index),
        "costmap_preprocess": copy.deepcopy(context.costmap_preprocess), "start_footprint_clearance": start_clearance,
        "target_shape": target_shape, "dwa": copy.deepcopy(dwa),
        # Collision sampling and Phase-2 admission are intentionally inside
        # choose_dwa's existing candidate loop; this split preserves that
        # native coupling instead of inventing a second profiler path.
        "timing_ns": {
            "shared_preprocessing": int(shared_preprocessing_ns),
            "per_s1_footprint_preparation": int(footprint_done_ns - footprint_started_ns),
            "grid_preparation": int(shared_preprocessing_ns + footprint_done_ns - footprint_started_ns),
            "astar": int(astar_done_ns - footprint_done_ns),
            "dwa_rollout_collision_phase2": int(dwa_phase2_done_ns - astar_done_ns),
            "total": int(dwa_phase2_done_ns - started_ns),
        },
    })
    return result


def evaluate_frozen_room_local_continuation_cohort(
    epoch: FrozenRoomLocalEpoch,
    candidate: FrozenRoomLocalCandidate,
    productive_motions: Sequence[Mapping[str, Any]],
    *,
    cap: int = CONTINUATION_SHADOW_MAX_MOTIONS,
) -> Dict[str, Any]:
    """Bounded C0 evidence cohort; never provides control or selection authority."""
    scheduled, unscheduled = schedule_continuation_productive_motions(productive_motions, cap=cap)
    shared_context: Optional[FrozenContinuationGridContext] = None
    shared_context_error: Optional[str] = None
    shared_preprocessing_ns = 0
    if scheduled:
        try:
            shared_context = build_frozen_continuation_grid_context(epoch, candidate)
            shared_preprocessing_ns = int(shared_context.shared_preprocessing_ns)
        except ValueError as exc:
            shared_context_error = str(exc)
    records = [
        evaluate_frozen_room_local_continuation(
            epoch,
            candidate,
            motion,
            shared_grid_context=shared_context,
        ) if shared_context is not None else {
            "epoch_id": epoch.epoch_id,
            "candidate_id": candidate.candidate_id,
            "motion": copy.deepcopy(dict(motion)),
            "continuation_status": CONTINUATION_UNKNOWN,
            "reason": shared_context_error or "FROZEN_CONTINUATION_CONTEXT_UNAVAILABLE",
            "next_safe_count": None,
            "next_productive_count": None,
            "commands_published": False,
            "ros_read": False,
            "ros_write": False,
            "persistent_state_mutated": False,
        }
        for motion in scheduled
    ]
    counts = {
        CONTINUATION_VIABLE: sum(item.get("continuation_status") == CONTINUATION_VIABLE for item in records),
        CONTINUATION_NON_VIABLE: sum(item.get("continuation_status") == CONTINUATION_NON_VIABLE for item in records),
        CONTINUATION_UNKNOWN: sum(item.get("continuation_status") == CONTINUATION_UNKNOWN for item in records),
    }
    if not scheduled and not unscheduled:
        status, reason = CONTINUATION_UNKNOWN, "CURRENT_PRODUCTIVE_SET_EMPTY"
    elif counts[CONTINUATION_VIABLE]:
        status, reason = CONTINUATION_VIABLE, "AT_LEAST_ONE_CHECKED_MOTION_VIABLE"
    elif counts[CONTINUATION_UNKNOWN] or unscheduled:
        status, reason = CONTINUATION_UNKNOWN, "EVIDENCE_INCOMPLETE_OR_CAP_EXCEEDED"
    else:
        status, reason = CONTINUATION_NON_VIABLE, "ALL_PRODUCTIVE_MOTIONS_CHECKED_NON_VIABLE"
    return {
        "continuation_status": status, "reason": reason, "cap": int(cap),
        "scheduled_motion_count": len(scheduled), "unscheduled_motion_count": len(unscheduled),
        "records": records, "unscheduled_motion_identities": [_continuation_motion_key(item) for item in unscheduled],
        "counts": counts, "commands_published": False, "selection_performed": False,
        "phase3_state_mutated": False,
        "shared_preprocessing_ns": int(shared_preprocessing_ns),
        "shared_context_built": bool(shared_context is not None),
        "shared_context_error": shared_context_error,
    }


def _continuation_status_for_productive_motion(
    evidence: Mapping[str, Any],
    productive_motion: Mapping[str, Any],
) -> Tuple[str, str]:
    """Join C0 evidence to an existing Phase-2 record by its frozen `(v,w)` identity."""
    key = _continuation_motion_key(productive_motion)
    for record in evidence.get("records") or []:
        if not isinstance(record, Mapping) or _continuation_motion_key(record.get("motion") or {}) != key:
            continue
        status = record.get("continuation_status")
        if status in {CONTINUATION_VIABLE, CONTINUATION_NON_VIABLE, CONTINUATION_UNKNOWN}:
            return str(status), str(record.get("reason") or "CONTINUATION_RECORD_REASON_UNAVAILABLE")
        return CONTINUATION_UNKNOWN, "CONTINUATION_RECORD_STATUS_INVALID"
    if evidence.get("unscheduled_motion_count"):
        return CONTINUATION_UNKNOWN, "CONTINUATION_MOTION_UNSCHEDULED_OR_CAP_TRUNCATED"
    return CONTINUATION_UNKNOWN, "CONTINUATION_MOTION_EVIDENCE_MISSING"


def continuation_aware_admissible_selection(
    productive_motions: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    *,
    native_v: float,
    native_w: float,
    phase3_recovery_context: bool,
) -> Dict[str, Any]:
    """Filter only admission, then reuse native DWA order and strict-score comparison.

    `choose_dwa()` keeps the first encountered candidate for an equal score by
    updating `best` only when `score > best_score`.  This function uses the
    same candidate order and the same strict comparison; it never creates a
    continuation score or changes a candidate's score.
    """
    native_key = (float(native_v), float(native_w))
    classified: List[Dict[str, Any]] = []
    for candidate_index, motion in enumerate(productive_motions):
        if not bool(motion.get("score_eligible")):
            continue
        score = motion.get("final_dwa_score")
        if not finite_number(score):
            continue
        status, reason = _continuation_status_for_productive_motion(evidence, motion)
        classified.append({
            "candidate_index": int(candidate_index),
            "motion_key": list(_continuation_motion_key(motion)),
            "v": float(motion.get("v") or 0.0),
            "w": float(motion.get("w") or 0.0),
            "final_dwa_score": float(score),
            "continuation_status": status,
            "continuation_reason": reason,
            "is_native_provisional_winner": _continuation_motion_key(motion) == native_key,
        })
    counts = {
        status: sum(row["continuation_status"] == status for row in classified)
        for status in (CONTINUATION_VIABLE, CONTINUATION_UNKNOWN, CONTINUATION_NON_VIABLE)
    }
    if phase3_recovery_context and counts[CONTINUATION_VIABLE] > 0:
        admissible_statuses = {CONTINUATION_VIABLE}
        admission_policy = "PHASE3_VERIFIED_ONLY_WHEN_AVAILABLE"
    else:
        admissible_statuses = {CONTINUATION_VIABLE, CONTINUATION_UNKNOWN}
        admission_policy = (
            "PHASE3_UNKNOWN_ALLOWED_WITHOUT_VIABLE"
            if phase3_recovery_context else "ORDINARY_ROOM_LOCAL_UNKNOWN_RETAINED"
        )
    admissible = [row for row in classified if row["continuation_status"] in admissible_statuses]
    winner: Optional[Dict[str, Any]] = None
    for row in admissible:
        if winner is None or float(row["final_dwa_score"]) > float(winner["final_dwa_score"]):
            winner = row
    all_nonviable_complete = bool(classified) and counts[CONTINUATION_NON_VIABLE] == len(classified)
    outcome = (
        "NO_ADMISSIBLE_CONTINUATION"
        if winner is None and all_nonviable_complete else
        "VERIFIED_RECOVERY_SUCCESS"
        if winner is not None and winner["continuation_status"] == CONTINUATION_VIABLE else
        "UNVERIFIED_BUT_TRANSLATION_ALLOWED"
        if winner is not None else "CONTINUATION_EVIDENCE_UNUSABLE"
    )
    alternative_counts = {
        status: sum(
            row["continuation_status"] == status and not row["is_native_provisional_winner"]
            for row in classified
        )
        for status in (CONTINUATION_VIABLE, CONTINUATION_UNKNOWN, CONTINUATION_NON_VIABLE)
    }
    return {
        "authority_scope": "LOCAL_SELECTION_ADMISSION_ONLY",
        "phase3_recovery_context": bool(phase3_recovery_context),
        "admission_policy": admission_policy,
        "native_provisional_winner": {"v": float(native_v), "w": float(native_w)},
        "candidate_classifications": classified,
        "counts": counts,
        "alternative_counts": alternative_counts,
        "admissible_motion_count": len(admissible),
        "admissible_motion_keys": [row["motion_key"] for row in admissible],
        "final_winner": copy.deepcopy(winner),
        "final_winner_source": "EXISTING_DWA_SCORE_AND_NATIVE_ORDER" if winner is not None else None,
        "translation_allowed": bool(winner is not None),
        "outcome": outcome,
        "terminal_reason": "ROOM_LOCAL_NO_ADMISSIBLE_CONTINUATION" if winner is None else None,
        "cohort_status": evidence.get("continuation_status"),
        "cohort_reason": evidence.get("reason"),
        "cohort_unscheduled_motion_count": int(evidence.get("unscheduled_motion_count") or 0),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="publish cmd_vel commands")
    parser.add_argument(
        "--local-control-mode",
        choices=LOCAL_CONTROL_MODE_CHOICES,
        default=LOCAL_CONTROL_MODE_TRANSIT,
        help="explicit local-control authority; ROOM_LOCAL uses legacy selection unless Phase-2 offline admission is explicitly enabled",
    )
    parser.add_argument(
        "--room-local-phase2-productive-admission",
        choices=ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_CHOICES,
        default=ROOM_LOCAL_PHASE2_PRODUCTIVE_ADMISSION_DISABLED,
        help="ROOM_LOCAL productive admission; offline_frozen rejects execute and phase3_guarded_execute is explicit-only",
    )
    parser.add_argument(
        "--room-local-phase3-orientation-recovery",
        choices=ROOM_LOCAL_PHASE3_RECOVERY_CHOICES,
        default=ROOM_LOCAL_PHASE3_RECOVERY_DISABLED,
        help="RecoverySet/OrientationIntent activation; only phase3_guarded_execute may pair with execute",
    )
    parser.add_argument(
        "--continuation-viability-shadow",
        action="store_true",
        help="C0 only: bounded pure depth-1 continuation evidence; default-off and no control authority",
    )
    parser.add_argument(
        "--continuation-local-selection-authority",
        choices=CONTINUATION_LOCAL_SELECTION_AUTHORITY_CHOICES,
        default=CONTINUATION_LOCAL_SELECTION_AUTHORITY_DISABLED,
        help="explicit guarded production authority: filter only ROOM_LOCAL motion admission using C0 evidence",
    )
    parser.add_argument("--validation-run-id", default="", help="Validation-only mission identity supplied by the state machine.")
    parser.add_argument("--validation-runner-invocation-id", default="", help="Validation-only unique runner invocation identity.")
    parser.add_argument("--validation-event-topic", default=TOPIC_ROOM_LOCAL_VALIDATION_EVENT)
    parser.add_argument("--validation-abort-topic", default=TOPIC_ROOM_LOCAL_VALIDATION_ABORT)
    parser.add_argument("--cmd-topic", default=TOPIC_CMD)
    parser.add_argument(
        "--observation-orientation-aim-yaw-odom-rad",
        type=float,
        default=None,
        help="OA mission-only frozen yaw aim; executes at most one existing safe v=0 angular slice",
    )
    parser.add_argument("--input-timeout-sec", type=float, default=3.0)
    parser.add_argument("--max-runtime-sec", type=float, default=60.0, help="ROS/sim-time runtime limit in seconds")
    parser.add_argument("--wall-watchdog-sec", type=float, default=120.0, help="Wall-time safety watchdog in seconds")
    parser.add_argument(
        "--sim-time-start-wait-sec",
        type=float,
        default=3.0,
        help="Maximum wall-time wait for a valid ROS sim clock before continuing under the wall watchdog",
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--command-slice-sec", type=float, default=0.50)
    parser.add_argument("--cmd-rate-hz", type=float, default=20.0)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.30)
    parser.add_argument("--grid-resolution-m", type=float, default=0.05)
    parser.add_argument("--robot-radius-m", type=float, default=STATIC_PLANNING_FOOTPRINT_RADIUS_M)
    parser.add_argument(
        "--additional-clearance-margin-m",
        type=float,
        default=0.0,
        help="Additional occupied-obstacle inflation margin; physical robot radius contract remains unchanged.",
    )
    parser.add_argument("--block-size-cells", type=int, default=4)
    parser.add_argument("--lookahead-blocks", type=int, default=2)
    parser.add_argument("--lookahead-m", type=float, default=0.8)
    parser.add_argument("--max-goal-x-m", type=float, default=2.6)
    parser.add_argument("--max-goal-abs-y-m", type=float, default=1.35)
    parser.add_argument("--max-linear-x", type=float, default=0.35)
    parser.add_argument("--max-angular-z", type=float, default=0.28)
    parser.add_argument("--linear-samples", type=int, default=5)
    parser.add_argument("--angular-samples", type=int, default=11)
    parser.add_argument("--dwa-predict-time", type=float, default=1.0)
    parser.add_argument("--dwa-dt", type=float, default=0.1)
    parser.add_argument("--heading-weight", type=float, default=0.50)
    parser.add_argument("--clearance-weight", type=float, default=0.25)
    parser.add_argument("--speed-weight", type=float, default=0.10)
    parser.add_argument("--distance-weight", type=float, default=0.05)
    parser.add_argument("--max-clearance-score-m", type=float, default=1.0)
    parser.add_argument("--min-linear-x", type=float, default=0.04)
    parser.add_argument("--enforce-min-forward-speed", action="store_true")
    parser.add_argument(
        "--enable-execution-qualified-dwa-primitives",
        action="store_true",
        help="Deprecated compatibility flag; native continuous DWA remains authoritative.",
    )
    parser.add_argument("--max-linear-accel", type=float, default=0.35)
    parser.add_argument("--max-angular-accel", type=float, default=0.35)
    parser.add_argument("--distance-speed-gain", type=float, default=0.45)
    parser.add_argument("--min-progress-m", type=float, default=0.03)
    parser.add_argument("--no-progress-steps", type=int, default=8)
    parser.add_argument("--recovery-angular-z", type=float, default=0.18)
    parser.add_argument("--angular-magnitude-weight", type=float, default=0.18)
    parser.add_argument("--angular-reversal-weight", type=float, default=0.25)
    parser.add_argument("--angular-reversal-deadband", type=float, default=0.05)
    parser.add_argument("--target-lateral-correction-threshold-m", type=float, default=0.12)
    parser.add_argument("--target-heading-blend-weight", type=float, default=0.65)
    parser.add_argument("--max-target-heading-correction-rad", type=float, default=0.16)
    parser.add_argument("--target-lateral-correction-angular-z", type=float, default=0.14)
    parser.add_argument("--target-lateral-correction-weight", type=float, default=0.16)
    parser.add_argument("--centerline-target-y-threshold-m", type=float, default=0.20)
    parser.add_argument("--waypoint-lateral-anomaly-threshold-m", type=float, default=0.45)
    parser.add_argument("--path-stability-max-hold-steps", type=int, default=3)
    parser.add_argument("--astar-lateral-bias-weight", type=float, default=0.35)
    parser.add_argument("--disable-path-stability", action="store_true")
    parser.add_argument("--corridor-center-x-min-m", type=float, default=0.80)
    parser.add_argument("--corridor-center-x-max-m", type=float, default=2.40)
    parser.add_argument("--corridor-center-wall-min-abs-y-m", type=float, default=0.45)
    parser.add_argument("--corridor-center-min-wall-count", type=int, default=6)
    parser.add_argument("--corridor-center-min-width-m", type=float, default=0.90)
    parser.add_argument("--corridor-center-max-width-m", type=float, default=2.80)
    parser.add_argument("--corridor-center-max-abs-y-m", type=float, default=0.45)
    parser.add_argument("--corridor-center-target-max-abs-y-m", type=float, default=0.35)
    parser.add_argument(
        "--corridor-center-target-max-correction-m",
        type=float,
        default=0.15,
        help="maximum local-wall correction permitted relative to a centreline target",
    )
    parser.add_argument("--corridor-center-blend-weight", type=float, default=0.75)
    parser.add_argument("--disable-corridor-center-target", action="store_true")
    parser.add_argument("--pointcloud-wall-heading-blend-weight", type=float, default=0.35)
    parser.add_argument("--pointcloud-wall-min-interval-sec", type=float, default=0.20)
    parser.add_argument("--pointcloud-wall-max-age-sec", type=float, default=2.50)
    parser.add_argument("--pointcloud-wall-buffer-sec", type=float, default=3.00)
    parser.add_argument("--pointcloud-wall-max-buffer-points", type=int, default=12000)
    parser.add_argument("--pointcloud-wall-point-stride", type=int, default=4)
    parser.add_argument("--pointcloud-wall-max-points", type=int, default=2500)
    parser.add_argument("--pointcloud-wall-x-min-m", type=float, default=0.40)
    parser.add_argument("--pointcloud-wall-x-max-m", type=float, default=4.00)
    parser.add_argument("--pointcloud-wall-y-abs-max-m", type=float, default=2.20)
    parser.add_argument("--pointcloud-wall-z-min-m", type=float, default=0.15)
    parser.add_argument("--pointcloud-wall-z-max-m", type=float, default=1.60)
    parser.add_argument("--pointcloud-wall-min-abs-y-m", type=float, default=0.45)
    parser.add_argument("--pointcloud-wall-min-side-points", type=int, default=35)
    parser.add_argument("--pointcloud-wall-x-bin-size-m", type=float, default=0.25)
    parser.add_argument("--pointcloud-wall-min-points-per-bin", type=int, default=4)
    parser.add_argument("--pointcloud-wall-min-boundary-bins", type=int, default=4)
    parser.add_argument("--pointcloud-wall-min-x-span-m", type=float, default=0.90)
    parser.add_argument("--pointcloud-wall-left-inner-percentile", type=float, default=10.0)
    parser.add_argument("--pointcloud-wall-right-inner-percentile", type=float, default=90.0)
    parser.add_argument("--pointcloud-wall-max-line-rmse-m", type=float, default=0.50)
    parser.add_argument("--pointcloud-wall-max-abs-heading-rad", type=float, default=0.45)
    parser.add_argument("--disable-pointcloud-wall-heading", action="store_true")
    parser.add_argument("--imu-topic", default=TOPIC_TRUNK_IMU)
    parser.add_argument("--imu-heading-hold-kp", type=float, default=0.85)
    parser.add_argument("--imu-heading-hold-kd", type=float, default=0.08)
    parser.add_argument("--imu-heading-hold-max-correction-rad-s", type=float, default=0.08)
    parser.add_argument("--imu-heading-hold-min-linear-x", type=float, default=0.08)
    parser.add_argument("--imu-heading-hold-command-angular-deadband", type=float, default=0.08)
    parser.add_argument("--imu-heading-hold-max-age-sec", type=float, default=0.25)
    parser.add_argument("--disable-imu-heading-hold", action="store_true")
    parser.add_argument("--centerline-tracking-lateral-kp", type=float, default=0.45)
    parser.add_argument("--centerline-tracking-yaw-kp", type=float, default=0.90)
    parser.add_argument("--centerline-tracking-max-correction-rad-s", type=float, default=0.18)
    parser.add_argument("--centerline-tracking-min-linear-x", type=float, default=0.08)
    parser.add_argument("--centerline-tracking-command-angular-deadband", type=float, default=0.08)
    parser.add_argument("--disable-centerline-tracking-correction", action="store_true")
    parser.add_argument("--centerline-clear-y-abs-m", type=float, default=0.32)
    parser.add_argument("--thin-barrier-x-min-m", type=float, default=0.35)
    parser.add_argument("--thin-barrier-x-max-m", type=float, default=1.55)
    parser.add_argument("--thin-barrier-max-thickness-m", type=float, default=0.25)
    parser.add_argument("--thin-barrier-support-depth-m", type=float, default=0.15)
    parser.add_argument("--thin-barrier-row-occupied-ratio", type=float, default=0.55)
    parser.add_argument("--thin-barrier-free-support-ratio", type=float, default=0.60)
    parser.add_argument("--disable-centerline-thin-barrier-clearing", action="store_true")
    parser.add_argument("--treat-unknown-as-blocked", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        validate_phase2_productive_admission_activation(args)
    except ValueError as exc:
        raise SystemExit(str(exc))
    rospy.init_node("block_astar_dwa_runner", anonymous=True, disable_signals=True)
    runner = BlockAStarDwaRunner(args)
    summary = runner.run()
    print(json.dumps({"final_decision": summary.get("final_decision"), "summary": str(OUT / "block_astar_dwa_mature_summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
