#!/usr/bin/env python3
"""Lightweight local navigation state machine.

This is an MVP orchestration layer. It keeps the existing compliant runtime
pieces and makes phase transitions explicit:

ENTER_BUILDING -> FOLLOW_CORRIDOR, with optional room-door control states behind
an explicit flag.

It does not read Gazebo truth, call move_base, send navigation goals, or change
navigation safety flags.
"""

from __future__ import annotations

import argparse
import atexit
import copy
from collections import deque
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, Imu, PointCloud2
import sensor_msgs.point_cloud2 as pc2
try:
    from sensor_msgs.msg import CameraInfo
except ImportError:  # Fake ROS test stubs do not instantiate the sensor smoke collector.
    CameraInfo = Image
from std_msgs.msg import Bool, String
import tf

from inflation_geometry import occupied_euclidean_inflated_mask
from corridor_axis_evidence import (
    bind_axis_to_odom,
    estimate_bilateral_corridor_axis,
    make_mature_certificate,
    mature_bound_axis,
    mature_certificate_fresh,
)
from block_astar_dwa_mature_runner import (
    BlockAStarDwaRunner,
    build_arg_parser as build_block_astar_dwa_arg_parser,
)
from room_side_turn_validation_v1 import (
    FeedbackYawTurnCore,
    classify_stream_age,
    post_turn_transition,
    select_follow_corridor_opening_action,
    select_room_side_gap_candidate,
    update_room_side_gap_observation_from_progress,
)
from room_search_v1 import (
    COARSE_OBSERVATION_RESOLUTION_M,
    MAX_CANDIDATES,
    DangerReobserveEpisodeGuard,
    DangerReobserveMissionActionSpec,
    MissionActionSpec,
    PortalAnchor,
    RoomSearchV2,
    canonical_observation_cell_ids,
    candidate_observation_intent_cell_ids,
    evaluate_mission_action_observation,
    evaluate_danger_reobserve_terminal,
    freeze_danger_reobserve_mission_action_spec,
    freeze_mission_action_spec,
    occlusion_reveal_room_points,
    strategic_reposition_needed,
)
from frozen_decision_audit import FrozenDecisionCapture, _candidate_evidence
from room_search_observation_arrival_contract import evaluate_observation_arrival_shadow
from room_search_stage_b_shadow_capture import RoomSearchStageBShadowCapture
from high_level_locomotion_compatibility_shadow import HighLevelLocomotionCompatibilityShadow
from room_search_recoverability import (
    evaluate_predecessor_set,
    make_door_anchor_root_certificate,
    prepare_candidate_terminal_state,
    recoverability_epoch_identity,
    terminal_accuracy_telemetry,
)
from local_grid_contract import (
    ExactGridStatusPairCache,
    cell_to_metric,
    flatten_index,
    grid_metadata,
    metric_to_cell,
    qualified_for_navigation,
    STATIC_PLANNING_FOOTPRINT_RADIUS_M,
    validate_grid_metadata,
)
from source_time_odom_cache import OdomCache


ROOT = Path(__file__).resolve().parents[2]
DEBUG_DIR = ROOT / "debug" / "state_machine_navigation"
SUMMARY_PATH = DEBUG_DIR / "state_machine_navigation_summary.json"
CORRIDOR_MOTION_AUDIT_PATH = DEBUG_DIR / "corridor_motion_audit.json"
DOOR_LANDMARK_DEBUG_PATH = ROOT / "debug" / "door_landmark_tracker" / "door_landmarks_debug.json"
APPROACH_POINT_DEBUG_PATH = ROOT / "debug" / "door_landmark_tracker" / "approach_point_debug.json"
PORTAL_G14_SHADOW_TARGET_PATH = DEBUG_DIR / "portal_g14_shadow_target.json"
DOOR_OBSERVE_DEBUG_PATH = ROOT / "debug" / "door_landmark_tracker" / "door_observe_debug.json"
SIDE_GAP_NAV_DEBUG_PATH = ROOT / "debug" / "door_landmark_tracker" / "side_gap_nav_target_debug.json"
SIDE_GAP_SEGMENT_SWITCH_AUDIT_PATH = ROOT / "debug" / "door_landmark_tracker" / "side_gap_segment_switch_audit.json"
SIDE_GAP_VISUAL_AUDIT_DIR = ROOT / "debug" / "side_gap_visual_coordinate_audit"
SIDE_GAP_VISUAL_AUDIT_MANIFEST_PATH = SIDE_GAP_VISUAL_AUDIT_DIR / "manifest.json"
LOCAL_ENTRY_DEBUG_DIR = ROOT / "debug" / "local_free_space_entry_target"
LOCAL_ENTRY_DEBUG_PATH = LOCAL_ENTRY_DEBUG_DIR / "local_free_space_entry_target_debug.json"
FORCED_ENTRY_DEBUG_DIR = ROOT / "debug" / "forced_room_entry_mvp"
FORCED_ENTRY_SUMMARY_PATH = FORCED_ENTRY_DEBUG_DIR / "forced_room_entry_summary.json"
FORCED_ENTRY_TRIGGER_PATH = FORCED_ENTRY_DEBUG_DIR / "trigger_event.json"
REPORT_PATH = ROOT / "audit_reports" / "state_machine_navigation_report.md"
TARGET_PATH = ROOT / "debug" / "short_horizon_target_selection" / "short_horizon_target_override.json"
N5B_PATH = ROOT / "debug" / "short_horizon_target_selection" / "n5b_subgoal_audit_report.json"
N5_SUMMARY_PATH = ROOT / "debug" / "short_horizon_target_selection" / "n5_target_selection_subgoal_summary.json"
RUNNER_SUMMARY_PATH = ROOT / "debug" / "block_astar_dwa_mature" / "block_astar_dwa_mature_summary.json"
RUNNER_TIMEOUT_AUDIT_PATH = ROOT / "debug" / "block_astar_dwa_mature" / "runner_timeout_audit.json"
DOORWAY_PATH = ROOT / "debug" / "doorway_candidate_detector" / "latest_doorway_candidate.json"
ROOM_VIEWPOINT_PATH = ROOT / "debug" / "room_frontier_viewpoint_selector" / "latest_room_viewpoint.json"
VISION_PATH = ROOT / "debug" / "vision_scene_semantics" / "latest_project_scene_analysis.json"
STAGE_PATH = ROOT / "debug" / "navigation_stage" / "current_stage.json"
ROOM_SEARCH_DIAGNOSTIC_PATH = ROOT / "debug" / "room_search" / "latest_room_search_diagnostic.json"
ROOM_SEARCH_CANDIDATE_AUDIT_FALLBACK_PATH = ROOT / "debug" / "room_search" / "room_search_candidate_audit.jsonl"
ROOM_SEARCH_OBSERVATION_ARRIVAL_SHADOW_FLAG = "ROOM_SEARCH_OBSERVATION_ARRIVAL_SHADOW"
ROOM_SEARCH_C1_C2_RESULT_MD = "ROOM_SEARCH_C1_C2_OBSERVATION_ARRIVAL_ONLINE_RESULT.md"
ROOM_SEARCH_C1_C2_RESULT_JSON = "ROOM_SEARCH_C1_C2_OBSERVATION_ARRIVAL_ONLINE_RESULT.json"
ROOM_SEARCH_C1_C2_DECISIONS_JSON = "ROOM_SEARCH_C1_C2_OBSERVATION_ARRIVAL_DECISIONS.json"
DANGER_TRACKS_TOPIC = "/team/danger_tracks"
DANGER_HYPOTHESES_TOPIC = "/team/danger_hypotheses"
ROOM_SEARCH_TENTATIVE_HYPOTHESIS_TTL_SEC = 2.0
ROOM_SEARCH_REOBSERVE_HOLD_SIM_SEC = 0.6
ROOM_SEARCH_RGB_CAMERA_INFO_TOPIC = "/real_sense/rgb/camera_info"
ROOM_SEARCH_CAMERA_HFOV_FALLBACK_RAD = math.radians(60.0)
ROOM_ZONE_STATE_TOPIC = "/audit/p2kg12/room_zone_state"
ROOM_ZONE_STATE_CONTRACT_VERSION = "p2kg12_room_zone_state_v1"
PORTAL_EFFECT_TOPIC = "/audit/p2kg12/portal_effect_gate"
PORTAL_EFFECT_CONTRACT_VERSION = "p2kg12_portal_effect_gate_v2"
PORTAL_EFFECT_STARTUP_TIMEOUT_SEC = 5.0
DOORWAY_CANDIDATE_AUTHORITY = "PORTAL_ROOM_ZONE_EFFECT_GATE"
ROBOT_STATIC_RADIUS_M = 0.2641935843278561
GRID_RESOLUTION_M = 0.05
DISCRETE_INFLATION_CELLS = int(math.ceil(ROBOT_STATIC_RADIUS_M / GRID_RESOLUTION_M))
DISCRETE_INFLATION_REACH_M = DISCRETE_INFLATION_CELLS * GRID_RESOLUTION_M
# RUN0132 measured the same physical Portal's maximum retained reobservation
# deviation as 0.15989977061929084 m.  V1 uses twice that observed bound for
# deterministic mission-local physical-Portal identity matching.
VISITED_PORTAL_IDENTITY_TOLERANCE_M = 0.3197995412385817
# A portal width is continuous geometry, while a jamb is represented by full
# raw occupancy cells.  Keep one whole raster cell beyond the existing
# half-cell centre-of-strip allowance when converting a portal width into the
# P_pre quarter-arc crossing limit.
PORTAL_JAMB_RASTER_GUARD_M = GRID_RESOLUTION_M
# R33: a Portal-relative staging pose recovered from the historical Stair
# approach.  These are not world coordinates: each value is expressed in the
# frozen Portal normal/tangent frame and is revalidated against fresh odom and
# the formal local Grid before it can receive motion authority.
STAIR_MOVING_TURN_KAPPA_MAX_M_INV = 0.6620444444444447
STAIR_MOVING_TURN_MIN_LINEAR_X_MPS = 0.225
STAIR_MOVING_TURN_MAX_LINEAR_X_MPS = 0.500
STAIR_MOVING_TURN_MAX_ABS_ANGULAR_Z_RADPS = 0.149
STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M = 1.1648542852807815
STAIR_MOVING_TURN_P_PRE_TANGENT_M = 1.4515654787245793
STAIR_MOVING_TURN_PATH_STEP_M = 0.01
STAIR_MOVING_TURN_GOAL_NORMAL_M = 0.30
STAIR_MOVING_TURN_GOAL_TOLERANCE_M = 0.30
ODOM_TOPIC = "/team/livox/icp_odom_gated"
GRID_TOPIC = "/team/local_traversability_grid"
RAW_ODOM_TOPIC = "/team/livox/icp_odom_raw"
# The two archived runs show an early, non-durable valid run of at most three
# samples (CURRENT) / two samples (baseline).  Eight samples is the smallest
# observed sustained run in CURRENT.  Heading consistency below supplies the
# second guard: the baseline's earlier eight-sample run has 0.108 rad
# same-epoch spread and is rejected, while CURRENT's observed sustained run
# is 0.083 rad.  This is deliberately a sample-only rule, not a scene-time
# or distance trigger.
CORRIDOR_AXIS_MATURITY_SAMPLES = 8
CORRIDOR_AXIS_MATURITY_MAX_HEADING_DEVIATION_RAD = 0.10
FILTERED_CLOUD_TOPIC = "/team/livox/scan_cloud_filtered"
TRAVERSABILITY_STATUS_TOPIC = "/team/traversability_status"
RL_MODE_READY_TOPIC = "/unitree/rl_mode_ready"
ROOM_LOCAL_VALIDATION_EVIDENCE_STATUS_TOPIC = "/audit/room_local_validation/evidence_status"


class FastDebugDoorCueStop(RuntimeError):
    def __init__(self, observations: Sequence[Dict[str, Any]], stop_reason: str = "stop_after_first_door_cue") -> None:
        super().__init__(stop_reason)
        self.observations = list(observations)
        self.stop_reason = stop_reason


class RoomLocalValidationAbort(RuntimeError):
    """Validation-only terminal result; never a navigation recovery signal."""


class RoomLocalValidationEvidenceInsufficient(RuntimeError):
    """Stop further guarded validation at a high-level invocation boundary only."""


class RoomLocalValidationEvidenceStatusCache:
    """Read-only, run-scoped observer evidence cache; it never gates TRANSIT or a command slice."""

    def __init__(self, run_id: str, topic: str = ROOM_LOCAL_VALIDATION_EVIDENCE_STATUS_TOPIC) -> None:
        self.run_id = str(run_id)
        self.latest_insufficient: Optional[Dict[str, Any]] = None
        self.subscriber = rospy.Subscriber(topic, String, self._callback, queue_size=10)

    def _callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict) or str(payload.get("run_id") or "") != self.run_id:
            return
        if payload.get("evidence_status") == "ONLINE_EVIDENCE_INSUFFICIENT":
            self.latest_insufficient = dict(payload)

    def insufficient_reason(self) -> Optional[str]:
        if self.latest_insufficient is None:
            return None
        return str(self.latest_insufficient.get("reason") or "ONLINE_EVIDENCE_INSUFFICIENT")

    def close(self) -> None:
        self.subscriber.unregister()


class FastDebugSideGapNavStop(RuntimeError):
    def __init__(self, candidate: Dict[str, Any]) -> None:
        super().__init__("stop_after_first_side_gap_nav_debug")
        self.candidate = candidate


class FastDebugSideGapSegmentSwitchAuditStop(RuntimeError):
    def __init__(self, event: Dict[str, Any]) -> None:
        super().__init__("stop_after_first_side_gap_segment_switch_audit")
        self.event = event


class FastDebugSideGapVisualAuditStop(RuntimeError):
    def __init__(self, event: Dict[str, Any]) -> None:
        super().__init__("stop_after_first_side_gap_visual_audit")
        self.event = event


class FastDebugLocalEntryTargetStop(RuntimeError):
    def __init__(self, candidate: Dict[str, Any]) -> None:
        super().__init__("stop_after_first_local_entry_target")
        self.candidate = candidate


class FastDebugPortalBoundCandidateStop(RuntimeError):
    """Stop at identity freeze; never turns, targets, or enters a doorway."""

    def __init__(self, candidate: Dict[str, Any]) -> None:
        super().__init__("stop_after_first_portal_bound_candidate")
        self.candidate = candidate


class ForcedRoomEntryMVPFinished(RuntimeError):
    def __init__(self, result: Dict[str, Any], stop_after_done: bool) -> None:
        super().__init__("forced_room_entry_done" if result.get("forced_entry_done") else "forced_room_entry_failed")
        self.result = result
        self.stop_after_done = stop_after_done


class PortalG14PPreFinished(RuntimeError):
    """Terminate 092 at the P_pre boundary without entering doorway logic."""

    def __init__(self, final_decision: str, reason: str, details: Dict[str, Any]) -> None:
        super().__init__(reason)
        self.final_decision = final_decision
        self.reason = reason
        self.details = details


class RoomReturnNextPortalDispatch(RuntimeError):
    """Leave a completed room through the thin post-return dispatcher."""

    def __init__(self, details: Dict[str, Any]) -> None:
        super().__init__("door_return_anchor_reached_next_portal_dispatch")
        self.details = details


class FormalGridStatusPairSubscriber:
    """Persistent formal-pair subscriber for fail-closed navigation admission."""

    def __init__(self) -> None:
        self.cache = ExactGridStatusPairCache(maxlen=8)
        self.subscribers = [
            rospy.Subscriber(GRID_TOPIC, OccupancyGrid, self._grid_cb, queue_size=8),
            rospy.Subscriber(TRAVERSABILITY_STATUS_TOPIC, String, self._status_cb, queue_size=8),
        ]

    def close(self) -> None:
        for subscriber in self.subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass

    def _grid_cb(self, msg: OccupancyGrid) -> None:
        self.cache.add_grid(msg)

    def _status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {}
        self.cache.add_status(payload if isinstance(payload, dict) else {})

    def matching_pair(self) -> Optional[Tuple[OccupancyGrid, Dict[str, Any]]]:
        return self.cache.matching_pair()

    def receipt_watermark(self) -> float:
        """Mark the receipt boundary after which terminal evidence is valid."""
        return self.cache.receipt_watermark()

    def wait_for_matching_pair(
        self,
        timeout_sec: float,
        *,
        wall_watchdog_sec: float,
    ) -> Optional[Tuple[OccupancyGrid, Dict[str, Any]]]:
        """Wait only for the existing exact cache match; never mix latest inputs.

        The normal admission timeout is measured on the ROS simulation clock.
        The explicit wall watchdog only bounds a stalled runtime, so reduced
        real-time factor cannot turn a still-valid simulation wait into an
        early failure.
        """
        start_sim_sec = float(rospy.Time.now().to_sec())
        sim_timeout_sec = max(0.0, float(timeout_sec))
        wall_deadline = time.monotonic() + max(1.0, float(wall_watchdog_sec))
        while not rospy.is_shutdown():
            pair = self.matching_pair()
            if pair is not None:
                return pair
            sim_elapsed_sec = max(0.0, float(rospy.Time.now().to_sec()) - start_sim_sec)
            if sim_elapsed_sec >= sim_timeout_sec or time.monotonic() >= wall_deadline:
                return None
            time.sleep(0.01)
        return None

    def wait_for_matching_pair_after(
        self,
        receipt_watermark_wall_sec: float,
        timeout_sec: float,
        *,
        wall_watchdog_sec: float,
    ) -> Optional[Dict[str, Any]]:
        """Wait for an exact Grid/Status pair received after one terminal boundary."""
        start_sim_sec = float(rospy.Time.now().to_sec())
        sim_timeout_sec = max(0.0, float(timeout_sec))
        wall_deadline = time.monotonic() + max(1.0, float(wall_watchdog_sec))
        while not rospy.is_shutdown():
            record = self.cache.matching_pair_record(
                not_before_wall_sec=float(receipt_watermark_wall_sec),
            )
            if record is not None:
                return record
            sim_elapsed_sec = max(0.0, float(rospy.Time.now().to_sec()) - start_sim_sec)
            if sim_elapsed_sec >= sim_timeout_sec or time.monotonic() >= wall_deadline:
                return None
            time.sleep(0.01)
        return None

    def latest_grid(self) -> Optional[OccupancyGrid]:
        return self.cache.latest_grid()


class RoomSideTurnHealthCache:
    """Small persistent cache for the V1 side-turn safety gate.

    Wall receipt age classifies transport health. ROS stamps and /clock
    progression are retained separately so low RTF is visible instead of being
    confused with a frozen simulation clock.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.samples: Dict[str, Dict[str, Any]] = {}
        self.clock_progress_wall_sec: Optional[float] = None
        self.last_clock_ros_sec: Optional[float] = None
        self.subscribers = [
            rospy.Subscriber("/clock", Clock, self._clock_cb, queue_size=20),
            rospy.Subscriber(FILTERED_CLOUD_TOPIC, PointCloud2, lambda msg: self._header_cb("filtered_cloud", msg), queue_size=2),
            rospy.Subscriber(RAW_ODOM_TOPIC, Odometry, lambda msg: self._odom_cb("raw_odom", msg), queue_size=20),
            rospy.Subscriber(ODOM_TOPIC, Odometry, lambda msg: self._odom_cb("gated_odom", msg), queue_size=50),
            rospy.Subscriber(GRID_TOPIC, OccupancyGrid, self._grid_cb, queue_size=10),
            rospy.Subscriber(TRAVERSABILITY_STATUS_TOPIC, String, self._traversability_status_cb, queue_size=10),
            rospy.Subscriber(args.follower_status_topic, String, self._follower_status_cb, queue_size=20),
            rospy.Subscriber(args.follower_imu_topic, Imu, self._imu_cb, queue_size=50),
            rospy.Subscriber(args.follower_raw_cmd_topic, Twist, lambda msg: self._twist_cb("raw_cmd", msg), queue_size=20),
            rospy.Subscriber(args.cmd_topic, Twist, lambda msg: self._twist_cb("cmd_vel_output", msg), queue_size=50),
            rospy.Subscriber(RL_MODE_READY_TOPIC, Bool, self._rl_mode_cb, queue_size=10),
        ]

    def close(self) -> None:
        for subscriber in self.subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass

    def _record(self, name: str, ros_stamp_sec: Optional[float], payload: Optional[Dict[str, Any]] = None) -> None:
        now_wall = time.monotonic()
        with self.lock:
            previous = self.samples.get(name, {})
            self.samples[name] = {
                "count": int(previous.get("count") or 0) + 1,
                "last_wall_receipt_sec": now_wall,
                "last_ros_stamp_sec": ros_stamp_sec,
                "previous_wall_receipt_sec": previous.get("last_wall_receipt_sec"),
                "payload": payload or {},
            }

    def _clock_cb(self, msg: Clock) -> None:
        ros_sec = float(msg.clock.to_sec())
        now_wall = time.monotonic()
        with self.lock:
            if self.last_clock_ros_sec is not None and ros_sec > self.last_clock_ros_sec + 1e-9:
                self.clock_progress_wall_sec = now_wall
            self.last_clock_ros_sec = ros_sec
        self._record("clock", ros_sec, {"clock_ros_sec": ros_sec})

    def _header_cb(self, name: str, msg: Any) -> None:
        stamp = float(msg.header.stamp.to_sec()) if getattr(msg, "header", None) is not None else None
        self._record(name, stamp, {"frame_id": str(getattr(msg.header, "frame_id", ""))})

    def _odom_cb(self, name: str, msg: Odometry) -> None:
        pose = msg.pose.pose
        self._record(
            name,
            float(msg.header.stamp.to_sec()),
            {
                "frame_id": msg.header.frame_id,
                "child_frame_id": msg.child_frame_id,
                "yaw_rad": yaw_from_quat(pose.orientation),
                "pose_x": float(pose.position.x),
                "pose_y": float(pose.position.y),
            },
        )

    def _grid_cb(self, msg: OccupancyGrid) -> None:
        self._record(
            "local_grid",
            float(msg.header.stamp.to_sec()),
            {
                "frame_id": msg.header.frame_id,
                "seq": int(msg.header.seq),
                "width": int(msg.info.width),
                "height": int(msg.info.height),
                "resolution": float(msg.info.resolution),
            },
        )

    def _traversability_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"parse_error": True, "raw_prefix": msg.data[:200]}
        stamp = payload.get("grid_header_stamp_sec") if isinstance(payload, dict) else None
        self._record("grid_status", float(stamp) if finite_number(stamp) else float(rospy.Time.now().to_sec()), payload if isinstance(payload, dict) else {})

    def _follower_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"parse_error": True, "raw_prefix": msg.data[:200]}
        self._record("follower_status", float(rospy.Time.now().to_sec()), payload if isinstance(payload, dict) else {})

    def _imu_cb(self, msg: Imu) -> None:
        self._record(
            "imu",
            float(msg.header.stamp.to_sec()) if msg.header.stamp else float(rospy.Time.now().to_sec()),
            {"yaw_rad": yaw_from_quat(msg.orientation), "angular_velocity_z": float(msg.angular_velocity.z)},
        )

    def _twist_cb(self, name: str, msg: Twist) -> None:
        self._record(name, float(rospy.Time.now().to_sec()), {"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)})

    def _rl_mode_cb(self, msg: Bool) -> None:
        self._record("rl_mode", float(rospy.Time.now().to_sec()), {"ready": bool(msg.data)})

    def snapshot(self) -> Dict[str, Any]:
        now_wall = time.monotonic()
        fresh_sec = float(self.args.room_side_turn_health_fresh_wall_sec)
        lost_sec = float(self.args.room_side_turn_health_lost_wall_sec)
        with self.lock:
            samples = json.loads(json.dumps(self.samples))
            clock_progress_wall = self.clock_progress_wall_sec
        streams: Dict[str, Any] = {}
        for name, sample in samples.items():
            last_wall = sample.get("last_wall_receipt_sec")
            age = now_wall - float(last_wall) if finite_number(last_wall) else None
            streams[name] = {**sample, "age_wall_sec": age, "health_state": classify_stream_age(age, fresh_sec, lost_sec)}
        required = ["filtered_cloud", "raw_odom", "gated_odom", "local_grid", "grid_status", "follower_status", "cmd_vel_output", "rl_mode"]
        blocking: List[str] = []
        for name in required:
            state = (streams.get(name) or {}).get("health_state", "LOST")
            if state != "FRESH":
                blocking.append(f"{name.lower()}_{state.lower()}")
        clock_progress_age = now_wall - float(clock_progress_wall) if finite_number(clock_progress_wall) else None
        clock_state = classify_stream_age(clock_progress_age, fresh_sec, lost_sec)
        if clock_state != "FRESH":
            blocking.append(f"clock_not_advancing_{clock_state.lower()}")
        rl_ready = bool((((streams.get("rl_mode") or {}).get("payload") or {}).get("ready")))
        if not rl_ready:
            blocking.append("rl_control_mode_not_ready")
        grid_status = ((streams.get("grid_status") or {}).get("payload") or {})
        input_freshness = grid_status.get("input_freshness") if isinstance(grid_status.get("input_freshness"), dict) else {}
        if not bool(input_freshness.get("all_required_inputs_fresh")):
            blocking.append("grid_upstream_status_reports_stale_content")
        if not bool(grid_status.get("safe_for_navigation")):
            blocking.append("grid_safe_for_navigation_false")
        follower_payload = ((streams.get("follower_status") or {}).get("payload") or {})
        if follower_payload.get("parse_error"):
            blocking.append("imu_follower_status_unparseable")
        unique_blocking = list(dict.fromkeys(blocking))
        return {
            "ready": not unique_blocking,
            "blocking_reasons": unique_blocking,
            "primary_blocking_reason": unique_blocking[0] if unique_blocking else None,
            "fresh_wall_sec": fresh_sec,
            "lost_wall_sec": lost_sec,
            "clock_progress_age_wall_sec": clock_progress_age,
            "clock_health_state": clock_state,
            "streams": streams,
        }

    def stream(self, name: str) -> Dict[str, Any]:
        return (self.snapshot().get("streams") or {}).get(name, {})


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def portal_effect_gate_startup_preflight(
    args: argparse.Namespace,
    wait_for_message: Optional[Any] = None,
    timeout_sec: float = PORTAL_EFFECT_STARTUP_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Fail closed before motion when hierarchical Portal authority is absent."""
    required = bool(args.execute and args.enable_hierarchical_portal_local_autonomy)
    if not required:
        return {
            "required": False,
            "ready": True,
            "reason": "PORTAL_EFFECT_GATE_NOT_REQUIRED",
        }
    waiter = wait_for_message if wait_for_message is not None else rospy.wait_for_message
    try:
        message = waiter(PORTAL_EFFECT_TOPIC, String, timeout=max(0.0, float(timeout_sec)))
    except Exception as exc:
        return {
            "required": True,
            "ready": False,
            "reason": "PORTAL_EFFECT_GATE_UNAVAILABLE",
            "topic": PORTAL_EFFECT_TOPIC,
            "timeout_sec": float(timeout_sec),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    try:
        payload = json.loads(message.data)
    except Exception as exc:
        return {
            "required": True,
            "ready": False,
            "reason": "PORTAL_EFFECT_GATE_PAYLOAD_INVALID",
            "topic": PORTAL_EFFECT_TOPIC,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    if not isinstance(payload, dict) or payload.get("contract_version") != PORTAL_EFFECT_CONTRACT_VERSION:
        return {
            "required": True,
            "ready": False,
            "reason": "PORTAL_EFFECT_GATE_CONTRACT_INVALID",
            "topic": PORTAL_EFFECT_TOPIC,
            "expected_contract_version": PORTAL_EFFECT_CONTRACT_VERSION,
            "observed_contract_version": payload.get("contract_version") if isinstance(payload, dict) else None,
        }
    return {
        "required": True,
        "ready": True,
        "reason": "PORTAL_EFFECT_GATE_READY",
        "topic": PORTAL_EFFECT_TOPIC,
        "contract_version": PORTAL_EFFECT_CONTRACT_VERSION,
    }


def portal_bound_doorway_control_domain_status(
    room_zone_active: bool,
    portal_effect_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Describe the existing formal doorway-control domain without creating one.

    The decision is deliberately derived from the already-published effect-gate
    state plus the existing doorway authority label.  A committed candidate is
    reported for diagnostics, but is not required to deauthorize legacy motion:
    an active formal domain with no candidate must still not fall back to a
    legacy doorway target.
    """
    snapshot = portal_effect_snapshot if isinstance(portal_effect_snapshot, dict) else {}
    effect_state = snapshot.get("latest_effect_state")
    effect_state = effect_state if isinstance(effect_state, dict) else {}
    authority = snapshot.get("doorway_candidate_authority")
    effect_gate_valid = (
        effect_state.get("contract_version") == PORTAL_EFFECT_CONTRACT_VERSION
        and effect_state.get("room_zone_active") is True
        and effect_state.get("input_valid") is True
    )
    formal_authority_selected = authority == DOORWAY_CANDIDATE_AUTHORITY
    active = bool(room_zone_active and effect_gate_valid and formal_authority_selected)
    committed = snapshot.get("committed_portal_candidate")
    return {
        "active": active,
        "room_zone_active": bool(room_zone_active),
        "effect_gate_valid": effect_gate_valid,
        "formal_authority_selected": formal_authority_selected,
        "doorway_candidate_authority": authority,
        "committed_candidate_present": isinstance(committed, dict),
        "reason": (
            "PORTAL_ROOM_ZONE_EFFECT_GATE_ACTIVE"
            if active
            else "PORTAL_ROOM_ZONE_EFFECT_GATE_NOT_ACTIVE"
        ),
    }


def legacy_latch_motion_authority(
    latched_doorway_profile: Optional[Dict[str, Any]],
    portal_bound_domain: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep a legacy latch observable while removing only its motion authority."""
    latch_present = isinstance(latched_doorway_profile, dict)
    domain_active = bool((portal_bound_domain or {}).get("active"))
    return {
        "latched_doorway_profile_present": latch_present,
        "diagnostic_only": bool(latch_present and domain_active),
        "legacy_motion_authorized": bool(latch_present and not domain_active),
        "reason": (
            "PORTAL_BOUND_DOMAIN_DEAUTHORIZES_LEGACY_LATCH"
            if latch_present and domain_active
            else "PORTAL_BOUND_DOMAIN_NOT_ACTIVE"
        ),
    }


def write_legacy_latch_or_corridor_target(
    anchor: Dict[str, Any],
    latched_doorway_profile: Dict[str, Any],
    corridor_lookahead_m: float,
    portal_bound_domain: Dict[str, Any],
    room_zone_active: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Emit the historical latch target only while it still has motion authority."""
    latch_authority = legacy_latch_motion_authority(latched_doorway_profile, portal_bound_domain)
    if latch_authority["legacy_motion_authorized"]:
        return (
            write_anchor_progress_target(
                anchor,
                float(latched_doorway_profile["target_anchor_progress_m"]),
                "state_machine_doorway_latch_alignment",
                {"latched_doorway_profile": latched_doorway_profile},
            ),
            latch_authority,
        )
    corridor_target = write_anchor_target(
        anchor,
        float(corridor_lookahead_m),
        "state_machine_corridor_centerline_door_search",
    )
    if isinstance(corridor_target, dict):
        corridor_target.update({
            "room_zone_active": bool(room_zone_active),
            "corridor_center_target_scope": (
                "pre_room_zone_only" if not room_zone_active else "disabled_in_room_zone"
            ),
        })
        # write_anchor_target() has already persisted the target.  Persist the
        # explicit scope as well so the separately launched runner consumes it.
        if corridor_target.get("target_type") == "generated_subgoal":
            write_json(TARGET_PATH, corridor_target)
    return corridor_target, latch_authority


def target_authority_class(target: Any) -> str:
    """Classify an emitted target for the compact 092 authority timeline."""
    source = target.get("source") if isinstance(target, dict) else None
    if source == "PORTAL_G14_P_PRE":
        return "PORTAL_G14_P_PRE_AUTHORITY"
    if isinstance(source, str) and source.startswith("state_machine_corridor_centerline"):
        return "CORRIDOR_CENTERLINE_AUTHORITY"
    if source == "state_machine_doorway_latch_alignment":
        return "LEGACY_DOORWAY_LATCH_AUTHORITY"
    if isinstance(source, str) and (
        "room_side" in source or "forced_room_entry" in source or "doorway" in source
    ):
        return "LEGACY_DOORWAY_AUTHORITY"
    return "NON_DOORWAY_OR_UNCLASSIFIED"


def build_target_authority_timeline(trace: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize actual runner starts from room-zone entry through P_pre stop."""
    timeline: List[Dict[str, Any]] = []
    for item in trace:
        target = item.get("target") if isinstance(item, dict) else None
        runner = item.get("runner") if isinstance(item, dict) else None
        if not isinstance(target, dict) or not isinstance(runner, dict):
            continue
        room_zone_active = bool(
            item.get("room_zone_reached_before") or item.get("room_zone_reached_after")
        )
        admissibility = item.get("portal_g14_p_pre_admissibility")
        if isinstance(admissibility, dict):
            p_pre_in_window: Optional[bool] = (
                str(admissibility.get("state")) != "P_PRE_OUTSIDE_LOCAL_PLANNING_WINDOW"
            )
        else:
            p_pre_in_window = None
        if not room_zone_active and p_pre_in_window is None:
            continue
        timeline.append({
            "iteration": item.get("iteration"),
            "state": item.get("state"),
            "ros_stamp": runner.get("run_sim_start_sec"),
            "target_source": target.get("source"),
            "target_authority_class": target_authority_class(target),
            "portal_committed": bool(item.get("portal_committed")),
            "p_pre_in_window": p_pre_in_window,
            "runner_started": True,
        })
    return timeline


def _finite_shape(value: Any, length: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == length and all(finite_number(item) for item in value)


class PortalBoundDoorCandidateAuthority:
    """Receive-only Portal authority with a separately frozen first commit."""

    def __init__(self, subscribe: bool = True) -> None:
        self.lock = threading.Lock()
        self.latest_effect_state: Optional[Dict[str, Any]] = None
        self.committed_portal_candidate: Optional[Dict[str, Any]] = None
        self.pending_alternative_candidate: Optional[Dict[str, Any]] = None
        self.last_rejection: Optional[str] = None
        self.current_run_id: Optional[str] = None
        self.last_sequence: Optional[int] = None
        self.last_source_stamp: Optional[float] = None
        self.last_effect_receive_time: Optional[float] = None
        self._frame_content: Dict[Tuple[str, int, float], str] = {}
        self.subscriber = rospy.Subscriber(PORTAL_EFFECT_TOPIC, String, self._callback, queue_size=20) if subscribe else None

    def close(self) -> None:
        if self.subscriber is not None:
            try:
                self.subscriber.unregister()
            except Exception:
                pass

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "latest_effect_state": dict(self.latest_effect_state) if isinstance(self.latest_effect_state, dict) else None,
                "committed_portal_candidate": dict(self.committed_portal_candidate) if isinstance(self.committed_portal_candidate, dict) else None,
                "pending_alternative_candidate": dict(self.pending_alternative_candidate) if isinstance(self.pending_alternative_candidate, dict) else None,
                "last_rejection": self.last_rejection,
                "last_portal_frame_sequence": self.last_sequence,
                "last_portal_source_stamp": self.last_source_stamp,
                "last_effect_receive_time": self.last_effect_receive_time,
                "doorway_candidate_authority": DOORWAY_CANDIDATE_AUTHORITY,
            }

    def reset_post_room_return_candidate(self) -> None:
        """Clear only the previous entry attempt's one-shot candidate authority."""
        with self.lock:
            self.committed_portal_candidate = None
            self.pending_alternative_candidate = None
            self.last_rejection = None

    def commit_post_room_return_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Make one dispatcher-qualified fresh candidate available to existing entry code."""
        with self.lock:
            self.committed_portal_candidate = dict(candidate)
            self.pending_alternative_candidate = None
            self.last_rejection = None
            return dict(self.committed_portal_candidate)

    def _reject(self, reason: str, *, invalidate_commit: bool = False) -> Optional[Dict[str, Any]]:
        if invalidate_commit:
            self.committed_portal_candidate = None
            self.pending_alternative_candidate = None
        self.last_rejection = reason
        return dict(self.committed_portal_candidate) if isinstance(self.committed_portal_candidate, dict) else None

    def _clear_for_new_run(self, run_id: str) -> None:
        self.committed_portal_candidate = None
        self.pending_alternative_candidate = None
        self.last_sequence = None
        self.last_source_stamp = None
        self.last_effect_receive_time = None
        self._frame_content = {}
        self.current_run_id = run_id

    @staticmethod
    def _candidate_from_effect(payload: Dict[str, Any], side: str, receive_time: float) -> Optional[Dict[str, Any]]:
        row = payload[side]
        portal = row.get("portal")
        stamp = float(payload["portal_source_stamp"])
        required = (
            payload.get("room_zone_active") is True,
            row.get("effect_reason") == "CURRENT_FRAME_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION",
            isinstance(portal, dict),
            isinstance(portal.get("track_id") if isinstance(portal, dict) else None, str),
            portal.get("side") == side if isinstance(portal, dict) else False,
            portal.get("observation_state") == "confirmed" if isinstance(portal, dict) else False,
            portal.get("candidate_available") is True if isinstance(portal, dict) else False,
            portal.get("frame_id") == "base" if isinstance(portal, dict) else False,
            portal.get("source_stamp") == stamp if isinstance(portal, dict) else False,
            portal.get("temporal_support") is True if isinstance(portal, dict) else False,
            portal.get("traversability_support") is True if isinstance(portal, dict) else False,
            _finite_shape(portal.get("portal_center_base") if isinstance(portal, dict) else None, 2),
            _finite_shape(portal.get("portal_normal_base") if isinstance(portal, dict) else None, 2),
            finite_number(portal.get("portal_width") if isinstance(portal, dict) else None),
            _finite_shape(portal.get("left_boundary") if isinstance(portal, dict) else None, 2),
            _finite_shape(portal.get("right_boundary") if isinstance(portal, dict) else None, 2),
        )
        if not all(required):
            return None
        return {
            "contract_version": PORTAL_EFFECT_CONTRACT_VERSION,
            "portal_run_id": payload["run_id"],
            "portal_frame_sequence": payload["portal_frame_sequence"],
            "portal_source_stamp": stamp,
            "portal_track_id": portal["track_id"], "side": side,
            "observation_state": portal["observation_state"], "candidate_available": True,
            "effect_eligible": True, "effect_reason": row["effect_reason"],
            "room_zone_active": True, "room_zone_source_stamp": payload.get("room_zone_source_stamp"),
            "room_zone_transition_sequence": payload.get("room_zone_transition_sequence"),
            "portal_frame_id": portal["frame_id"],
            "portal_center_base": list(portal["portal_center_base"]),
            "portal_normal_base": list(portal["portal_normal_base"]),
            "portal_width": float(portal["portal_width"]),
            "left_boundary": list(portal["left_boundary"]), "right_boundary": list(portal["right_boundary"]),
            "temporal_support": True, "traversability_support": True,
            "candidate_receive_time": float(receive_time),
            "control_authority_source": DOORWAY_CANDIDATE_AUTHORITY,
            "commit_rule": "FIRST_UNIQUE_ELIGIBLE_CANDIDATE_IN_CURRENT_RUN_AND_ROOM_ZONE",
            "target_geometry_status": "SOURCE_TIME_ODOM_BINDING_PENDING",
            "candidate_lifecycle": "COMMITTED",
        }

    def _callback(self, message: String) -> None:
        try:
            self.ingest(json.loads(message.data), candidate_receive_time=time.time())
        except Exception:
            with self.lock:
                self._reject("EFFECT_PAYLOAD_PARSE_ERROR")

    def ingest(self, payload: Any, candidate_receive_time: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Update latest effect state and commit only the first unique side."""
        with self.lock:
            if not isinstance(payload, dict) or payload.get("contract_version") != PORTAL_EFFECT_CONTRACT_VERSION:
                return self._reject("EFFECT_CONTRACT_VERSION_INVALID")
            run_id, sequence, stamp = payload.get("run_id"), payload.get("portal_frame_sequence"), payload.get("portal_source_stamp")
            self.latest_effect_state = dict(payload)
            if payload.get("input_valid") is not True or not isinstance(run_id, str) or not run_id.strip():
                return self._reject("PORTAL_RUN_ID_MISSING_OR_INVALID")
            if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0 or not finite_number(stamp):
                return self._reject("EFFECT_FRAME_IDENTITY_INVALID")
            stamp = float(stamp)
            if self.current_run_id != run_id:
                self._clear_for_new_run(run_id)
            frame_identity = (run_id, sequence, stamp)
            serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            prior_content = self._frame_content.get(frame_identity)
            if prior_content is not None:
                if prior_content != serialized:
                    return self._reject("IDENTITY_CONTENT_CONFLICT", invalidate_commit=True)
                self.last_rejection = "IDEMPOTENT_DUPLICATE"
                return dict(self.committed_portal_candidate) if self.committed_portal_candidate else None
            if self.last_sequence is not None and sequence < self.last_sequence:
                return self._reject("OUT_OF_ORDER_EFFECT_SEQUENCE")
            if self.last_sequence is not None and sequence == self.last_sequence and stamp != self.last_source_stamp:
                return self._reject("FRAME_SEQUENCE_STAMP_CONFLICT")
            if self.last_source_stamp is not None and stamp < self.last_source_stamp:
                return self._reject("OUT_OF_ORDER_EFFECT_SOURCE_STAMP")
            self._frame_content[frame_identity] = serialized
            self.last_sequence, self.last_source_stamp = sequence, stamp
            self.last_effect_receive_time = float(candidate_receive_time if candidate_receive_time is not None else time.time())
            if payload.get("room_zone_active") is False and self.committed_portal_candidate is not None:
                return self._reject("ROOM_ZONE_EXIT_INVALIDATED", invalidate_commit=True)
            eligible_sides = [side for side in ("left", "right") if isinstance(payload.get(side), dict) and payload[side].get("effect_eligible") is True]
            if len(eligible_sides) > 1:
                if self.committed_portal_candidate is not None:
                    return self._reject("AMBIGUOUS_LATER_FRAME_IGNORED_FOR_COMMITTED_CANDIDATE")
                return self._reject("MULTIPLE_ELIGIBLE_PORTAL_SIDES")
            if not eligible_sides:
                return self._reject("NO_ELIGIBLE_PORTAL_SIDE")
            side = eligible_sides[0]
            candidate = self._candidate_from_effect(payload, side, candidate_receive_time if candidate_receive_time is not None else time.time())
            if candidate is None:
                return self._reject("PORTAL_BOUND_CANDIDATE_VALIDATION_FAILED")
            if self.committed_portal_candidate is None:
                self.committed_portal_candidate = candidate
                self.last_rejection = None
                return dict(candidate)
            if candidate["portal_track_id"] != self.committed_portal_candidate["portal_track_id"] or candidate["side"] != self.committed_portal_candidate["side"]:
                self.pending_alternative_candidate = candidate
                return self._reject("PENDING_ALTERNATIVE_CANDIDATE_IGNORED")
            self.last_rejection = None
            return dict(self.committed_portal_candidate)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    """Persist a compact diagnostic snapshot without making shutdown a dependency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def room_search_candidate_audit_path() -> Path:
    """Use the already-created run archive when the normal wrapper provides it."""
    archive_dir = os.environ.get("STATE_MACHINE_RUN_ARCHIVE_DIR", "").strip()
    if archive_dir:
        return Path(archive_dir) / "room_search_candidate_audit.jsonl"
    return ROOM_SEARCH_CANDIDATE_AUDIT_FALLBACK_PATH


def append_room_search_candidate_audit(payload: Dict[str, Any]) -> Path:
    """Append compact decision evidence; this observer never feeds navigation."""
    path = room_search_candidate_audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def room_search_observation_arrival_shadow_enabled() -> bool:
    """Read the default-OFF, audit-only C1/C2 switch."""
    return str(os.environ.get(ROOM_SEARCH_OBSERVATION_ARRIVAL_SHADOW_FLAG, "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def room_search_observation_arrival_source_hashes() -> Dict[str, str]:
    """Capture C1/C2 provenance without reading any planner or control result."""
    paths = (
        "scripts/local_subgoal_runner_mvp/navigation_state_machine.py",
        "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py",
        "scripts/local_subgoal_runner_mvp/room_search_v1.py",
        "scripts/local_subgoal_runner_mvp/room_search_observation_arrival_contract.py",
        "scripts/local_subgoal_runner_mvp/replay_room_search_observation_arrival_c0.py",
        "scripts/local_subgoal_runner_mvp/tests/test_room_search_observation_arrival_contract.py",
    )
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def write_room_search_c1_c2_observation_arrival_outputs(
    payload: Dict[str, Any], shadow_context: Dict[str, Any],
) -> None:
    """Persist observer evidence only; its return value is intentionally unused."""
    if not shadow_context.get("feature_flag_enabled"):
        return
    output_dir = Path(os.environ.get("STATE_MACHINE_RUN_ARCHIVE_DIR", "").strip() or DEBUG_DIR)
    events = list(shadow_context.get("events") or [])
    states = [row.get("observation_state") for row in events]
    errors = [row.get("shadow_evaluation_error") for row in events if row.get("shadow_evaluation_error")]
    reached = sum(1 for row in events if row.get("navigation_reached") is True)
    source_before = dict(shadow_context.get("source_hashes_before") or {})
    source_after = room_search_observation_arrival_source_hashes()
    exact_diff_path = os.environ.get("ROOM_SEARCH_C1_C2_EXACT_DIFF_FILE", "").strip()
    try:
        exact_diff = Path(exact_diff_path).read_text(encoding="utf-8") if exact_diff_path else None
    except OSError as exc:
        exact_diff = "UNAVAILABLE:%s:%s" % (type(exc).__name__, exc)
    protected_expected = dict(shadow_context.get("protected_hashes") or {})
    protected_matches = {
        path: source_after.get(path) == expected for path, expected in protected_expected.items()
    }
    if errors:
        verdict = "C1_C2_ONLINE_SHADOW_BLOCKED_BY_EVALUATOR_EXCEPTION"
    elif not all(protected_matches.values()):
        verdict = "C1_C2_ONLINE_SHADOW_BLOCKED_BY_PROTECTED_HASH_CHANGED"
    elif source_before != source_after:
        verdict = "C1_C2_ONLINE_SHADOW_BLOCKED_BY_SOURCE_HASH_CHANGED_DURING_RUN"
    elif reached < 2:
        verdict = "C1_C2_ONLINE_SHADOW_VALID_BUT_INSUFFICIENT_MULTI_DECISION_EVIDENCE"
    else:
        verdict = "C1_C2_ONLINE_SHADOW_PASS_READY_FOR_AUTHORITY_DESIGN"
    counts = {
        "room_search_terminal_decisions_total": len(events),
        "navigation_reached_count": reached,
        "shadow_evaluated_count": len(events) - len(errors),
        "valid_count": states.count("OBSERVATION_VALID_REACHED_CANDIDATE"),
        "partial_count": states.count("OBSERVATION_VALUE_PARTIAL"),
        "collapsed_count": states.count("OBSERVATION_VALUE_COLLAPSED"),
        "unknown_count": states.count("OBSERVATION_VALIDITY_UNKNOWN"),
        "missing_evidence_count": sum(1 for row in events if not row.get("terminal_evidence_complete")),
        "evaluation_failure_count": len(errors),
    }
    result = {
        "schema_version": "room_search_c1_c2_observation_arrival_online_result_v1",
        "final_verdict": verdict,
        "run_id": shadow_context.get("run_id"),
        "feature_flag": ROOM_SEARCH_OBSERVATION_ARRIVAL_SHADOW_FLAG,
        "feature_flag_enabled": True,
        "navigation_state_machine_before_hash": shadow_context.get("navigation_before_hash"),
        "navigation_state_machine_after_hash": source_after.get(
            "scripts/local_subgoal_runner_mvp/navigation_state_machine.py"
        ),
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "exact_c1_unified_diff": exact_diff,
        "exact_c1_unified_diff_sha256": (
            hashlib.sha256(exact_diff.encode("utf-8")).hexdigest()
            if isinstance(exact_diff, str) else None
        ),
        "protected_hashes": protected_expected,
        "protected_hash_matches_after": protected_matches,
        "exact_c1_diff_scope": {
            "file": "scripts/local_subgoal_runner_mvp/navigation_state_machine.py",
            "functions": [
                "room_search_observation_arrival_shadow_enabled",
                "room_search_observation_arrival_source_hashes",
                "write_room_search_c1_c2_observation_arrival_outputs",
                "execute_room_search_v2",
            ],
            "allowed_categories": [
                "feature_flag", "immutable_observation_arrival_input_capture",
                "c0_evaluator_call", "structured_shadow_telemetry", "evaluator_exception_isolation",
            ],
        },
        "shadow_seam": "after terminal visibility and exactly one search.update_actual_view; before existing progress/postarrival/continuation",
        "seen_audit": {
            "order": [
                "immutable_pre_update_seen", "terminal_visibility", "existing_update_actual_view_once",
                "immutable_post_update_seen", "post_minus_pre_delta", "c0_shadow_evaluator",
            ],
            "shadow_seen_mutation_authority": False,
        },
        "behavior_neutrality": {
            "authority_enabled": False,
            "selection": False, "target": False, "runner": False, "command": False,
            "seen": False, "progress": False, "completion": False,
            "recoverability": False, "room_return": False, "state_transition": False,
        },
        "counts": counts,
        "decision_ids": [row.get("decision_id") for row in events],
        "candidate_ids": [row.get("candidate_id") for row in events],
        "evaluation_latency_ms": [row.get("evaluation_wall_ms") for row in events],
        "errors": errors,
        "final_room_search_outcome": payload.get("final_decision"),
        "next_recommendation": (
            "Review C1/C2 evidence and only then authorize C3/C4 authority design."
            if verdict == "C1_C2_ONLINE_SHADOW_PASS_READY_FOR_AUTHORITY_DESIGN"
            else "Stop after this one run and review the recorded evidence."
        ),
    }
    write_json(output_dir / ROOM_SEARCH_C1_C2_DECISIONS_JSON, {
        "schema_version": "room_search_c1_c2_observation_arrival_decisions_v1",
        "run_id": shadow_context.get("run_id"),
        "decisions": events,
    })
    write_json(output_dir / ROOM_SEARCH_C1_C2_RESULT_JSON, result)
    lines = [
        "# ROOM_SEARCH C1/C2 Observation-Arrival Online Result", "",
        "- final_verdict: `%s`" % verdict,
        "- run_id: `%s`" % shadow_context.get("run_id"),
        "- authority: `NONE`", "- final_room_search_outcome: `%s`" % payload.get("final_decision"),
        "- terminal / reached / valid / partial / collapsed / unknown / errors: %s / %s / %s / %s / %s / %s / %s" % (
            counts["room_search_terminal_decisions_total"], counts["navigation_reached_count"],
            counts["valid_count"], counts["partial_count"], counts["collapsed_count"],
            counts["unknown_count"], counts["evaluation_failure_count"],
        ), "", "## Decisions", "",
    ]
    for row in events:
        lines.append("- D%s %s: reached=%s, %s->%s, `%s`, viability=`%s`, next=`%s`, error=%s." % (
            row.get("decision_id"), row.get("candidate_id"), row.get("navigation_reached"),
            row.get("predicted_new_count"), row.get("actual_new_count"), row.get("observation_state"),
            row.get("post_arrival_viability"), row.get("next_production_action"), row.get("shadow_evaluation_error"),
        ))
    (output_dir / ROOM_SEARCH_C1_C2_RESULT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


class CorridorAxisEvidenceCache:
    """Production-only LiDAR geometry cache for a delayed anchor handoff.

    It owns no target, command, TF, localization, or DWA authority.  A single
    fresh cloud is reduced to bilateral wall geometry.  The caller later
    certifies a short consecutive sequence against existing gated-odom history
    before it may replace the legacy bootstrap heading.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = build_block_astar_dwa_arg_parser().parse_args([])
        self._max_age_sec = float(args.corridor_axis_evidence_max_age_sec)
        # This existing source-time wait allowance was previously not consumed
        # by the evidence lifecycle.  It now bounds only a formed certificate:
        # it is not a new geometric threshold and does not extend raw-cloud
        # freshness.
        self._certificate_max_age_sec = float(args.corridor_axis_evidence_wait_sec)
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._consecutive_valid: deque = deque(maxlen=CORRIDOR_AXIS_MATURITY_SAMPLES)
        self._mature_certificate: Optional[Dict[str, Any]] = None
        self._forward_intent_heading_odom_rad: Optional[float] = None
        self._odom_cache: Optional[OdomCache] = None
        self._tf_listener = tf.TransformListener()
        self._subscriber = rospy.Subscriber(FILTERED_CLOUD_TOPIC, PointCloud2, self._callback, queue_size=2)

    def close(self) -> None:
        try:
            self._subscriber.unregister()
        except Exception:
            pass

    def configure_handoff_context(
        self,
        odom_cache: OdomCache,
        forward_intent_heading_odom_rad: float,
    ) -> None:
        """Give the callback the existing bootstrap sign and source-time cache."""
        with self._lock:
            self._odom_cache = odom_cache
            self._forward_intent_heading_odom_rad = float(forward_intent_heading_odom_rad)

    def _point_in_base(self, point: Sequence[float], frame_id: str, stamp: Any) -> Optional[Tuple[float, float, float]]:
        source = str(frame_id or "").lstrip("/")
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        if source == "base":
            return x, y, z
        try:
            trans, rotation = self._tf_listener.lookupTransform("base", source, stamp)
        except Exception:
            return None
        matrix = tf.transformations.quaternion_matrix(rotation)
        value = np.dot(matrix, np.array([x, y, z, 1.0]))
        return float(value[0] + trans[0]), float(value[1] + trans[1]), float(value[2] + trans[2])

    def _callback(self, msg: PointCloud2) -> None:
        stamp_sec = float(msg.header.stamp.to_sec())
        if not finite_number(stamp_sec) or stamp_sec <= 0.0:
            return
        points: List[Tuple[float, float, float]] = []
        seen = 0
        try:
            for raw in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                seen += 1
                if seen % max(1, int(self._args.pointcloud_wall_point_stride)) != 0:
                    continue
                point = self._point_in_base(raw, msg.header.frame_id, msg.header.stamp)
                if point is not None:
                    points.append(point)
                if len(points) >= int(self._args.pointcloud_wall_max_points):
                    break
        except Exception:
            return
        axis = estimate_bilateral_corridor_axis(np.array(points, dtype=float), self._args)
        axis.update({
            "source_stamp": stamp_sec, "source_frame": str(msg.header.frame_id),
            "accepted_point_count": int(len(points)), "seen_point_count": int(seen),
        })
        with self._lock:
            self._latest = axis
            if axis.get("valid"):
                self._consecutive_valid.append(dict(axis))
            else:
                self._consecutive_valid.clear()
            samples = [dict(sample) for sample in self._consecutive_valid]
            odom_cache = self._odom_cache
            forward_heading = self._forward_intent_heading_odom_rad
            certificate_present = self._mature_certificate is not None
        if (
            axis.get("valid")
            and not certificate_present
            and odom_cache is not None
            and finite_number(forward_heading)
        ):
            bound_samples: List[Dict[str, Any]] = []
            for sample in samples:
                source_stamp = float(sample.get("source_stamp") or 0.0)
                bound = bind_axis_to_odom(
                    sample,
                    source_stamp,
                    odom_cache.pose_at_source_stamp(source_stamp),
                    float(forward_heading),
                )
                if not bound.get("valid"):
                    return
                bound["source"] = "production_lidar_bilateral_wall_geometry"
                bound_samples.append(bound)
            mature = mature_bound_axis(
                bound_samples,
                minimum_samples=CORRIDOR_AXIS_MATURITY_SAMPLES,
                max_heading_deviation_rad=CORRIDOR_AXIS_MATURITY_MAX_HEADING_DEVIATION_RAD,
            )
            certificate = make_mature_certificate(
                mature,
                source_time_valid_for_sec=self._certificate_max_age_sec,
            )
            if certificate.get("valid"):
                with self._lock:
                    # The first valid window is the one whose source-time
                    # identity later reaches the single consumer.  Do not let
                    # later callbacks rewrite an unconsumed certificate.
                    if self._mature_certificate is None:
                        self._mature_certificate = certificate

    def reset_maturity(self) -> None:
        with self._lock:
            self._consecutive_valid.clear()
            self._mature_certificate = None

    def consume_mature_certificate(self, certificate_id: Any) -> bool:
        with self._lock:
            certificate = self._mature_certificate
            if not isinstance(certificate, dict) or certificate.get("certificate_id") != certificate_id:
                return False
            if certificate.get("certificate_consumed") is True:
                return False
            certificate["certificate_consumed"] = True
            return True

    def mature_bound_axis(
        self,
        odom_cache: OdomCache,
        forward_intent_heading_odom_rad: float,
    ) -> Dict[str, Any]:
        """Return a mature bound axis, never a first-valid axis."""
        with self._lock:
            samples = [dict(sample) for sample in self._consecutive_valid]
            latest = dict(self._latest) if self._latest is not None else None
            certificate = dict(self._mature_certificate) if self._mature_certificate is not None else None
        if certificate is not None:
            fresh = mature_certificate_fresh(
                certificate,
                sim_now_sec=float(rospy.Time.now().to_sec()),
                current_odom_epoch_generation=odom_cache.current_epoch_generation(),
            )
            if fresh.get("valid"):
                return fresh
            if fresh.get("reason") == "CORRIDOR_AXIS_CERTIFICATE_ODOM_EPOCH_CHANGED":
                self.reset_maturity()
            else:
                with self._lock:
                    if self._mature_certificate is not None and self._mature_certificate.get("certificate_id") == certificate.get("certificate_id"):
                        self._mature_certificate = None
            return fresh
        if latest is None:
            return {"valid": False, "reason": "CORRIDOR_AXIS_EVIDENCE_UNAVAILABLE:NO_EVIDENCE"}
        latest_stamp = float(latest.get("source_stamp") or 0.0)
        sim_now = float(rospy.Time.now().to_sec())
        if sim_now < latest_stamp or sim_now - latest_stamp > self._max_age_sec:
            return {"valid": False, "reason": "CORRIDOR_AXIS_EVIDENCE_STALE"}
        bound_samples: List[Dict[str, Any]] = []
        for sample in samples:
            stamp = float(sample.get("source_stamp") or 0.0)
            bound = bind_axis_to_odom(
                sample,
                stamp,
                odom_cache.pose_at_source_stamp(stamp),
                forward_intent_heading_odom_rad,
            )
            if not bound.get("valid"):
                self.reset_maturity()
                return {
                    "valid": False,
                    "reason": "CORRIDOR_AXIS_MATURITY_ODOM_BINDING_INVALID:"
                    + str(bound.get("reason") or "UNKNOWN"),
                }
            bound["source"] = "production_lidar_bilateral_wall_geometry"
            bound_samples.append(bound)
        return mature_bound_axis(
            bound_samples,
            minimum_samples=CORRIDOR_AXIS_MATURITY_SAMPLES,
            max_heading_deviation_rad=CORRIDOR_AXIS_MATURITY_MAX_HEADING_DEVIATION_RAD,
        )


ODOM_CACHE: Optional[OdomCache] = None


def initialize_odom_cache() -> OdomCache:
    global ODOM_CACHE
    if ODOM_CACHE is None:
        ODOM_CACHE = OdomCache(rospy, topic=ODOM_TOPIC, message_type=Odometry)
    return ODOM_CACHE


def read_odom(timeout_sec: float = 5.0) -> Dict[str, Any]:
    cache = initialize_odom_cache()
    # A pose sampled for a state transition must be newer than this call.  The
    # cache sequence is protected by the same Condition used by the callback.
    sequence_at_call_start = cache.current_sequence()
    msg, _sequence = cache.get(timeout_sec, after_sequence=sequence_at_call_start)
    pose = msg.pose.pose
    yaw = yaw_from_quat(pose.orientation)
    return {
        "topic": ODOM_TOPIC,
        "header_frame_id": msg.header.frame_id,
        "child_frame_id": msg.child_frame_id,
        "stamp_sec": float(msg.header.stamp.to_sec()),
        "pose_x_y_yaw": [float(pose.position.x), float(pose.position.y), yaw],
    }


def read_grid(timeout_sec: float = 5.0) -> OccupancyGrid:
    return rospy.wait_for_message(GRID_TOPIC, OccupancyGrid, timeout=timeout_sec)


def read_traversability_status(timeout_sec: float = 5.0) -> Dict[str, Any]:
    message = rospy.wait_for_message(TRAVERSABILITY_STATUS_TOPIC, String, timeout=timeout_sec)
    try:
        payload = json.loads(message.data)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def pose_tuple(odom: Dict[str, Any]) -> Tuple[float, float, float]:
    pose = odom.get("pose_x_y_yaw")
    if not (isinstance(pose, list) and len(pose) == 3 and all(finite_number(v) for v in pose)):
        raise RuntimeError("odom_pose_unavailable")
    return float(pose[0]), float(pose[1]), float(pose[2])


def transform_base_xy(base_xy: Sequence[float], pose: Tuple[float, float, float]) -> List[float]:
    x_base, y_base = float(base_xy[0]), float(base_xy[1])
    x, y, yaw = pose
    return [x + math.cos(yaw) * x_base - math.sin(yaw) * y_base, y + math.sin(yaw) * x_base + math.cos(yaw) * y_base]


def target_base_xy(target_xy: Sequence[float], pose: Tuple[float, float, float]) -> Tuple[float, float]:
    dx = float(target_xy[0]) - pose[0]
    dy = float(target_xy[1]) - pose[1]
    yaw = pose[2]
    return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def rotate_base_vector(base_vector: Sequence[float], yaw: float) -> List[float]:
    return [
        math.cos(yaw) * float(base_vector[0]) - math.sin(yaw) * float(base_vector[1]),
        math.sin(yaw) * float(base_vector[0]) + math.cos(yaw) * float(base_vector[1]),
    ]


def freeze_portal_geometry(candidate: Dict[str, Any], binding: Dict[str, Any]) -> Dict[str, Any]:
    if not binding.get("binding_valid"):
        return {"geometry_valid": False, "reason": str(binding.get("reason") or "ODOM_BINDING_INVALID")}
    normal = candidate.get("portal_normal_base")
    side = candidate.get("side")
    if not _finite_shape(normal, 2) or (side == "left" and float(normal[1]) <= 0) or (side == "right" and float(normal[1]) >= 0):
        return {"geometry_valid": False, "reason": "PORTAL_NORMAL_SIDE_DIRECTION_CONFLICT"}
    pose = binding.get("source_pose_x_y_yaw")
    if not _finite_shape(pose, 3):
        return {"geometry_valid": False, "reason": "SOURCE_POSE_INVALID"}
    normal_odom = rotate_base_vector(normal, float(pose[2]))
    norm = math.hypot(normal_odom[0], normal_odom[1])
    if not finite_number(norm) or norm <= 0:
        return {"geometry_valid": False, "reason": "PORTAL_NORMAL_NORM_INVALID"}
    normal_odom = [normal_odom[0] / norm, normal_odom[1] / norm]
    return {
        "geometry_valid": True,
        "portal_center_odom": transform_base_xy(candidate["portal_center_base"], tuple(pose)),
        "portal_normal_odom": normal_odom,
        "left_boundary_odom": transform_base_xy(candidate["left_boundary"], tuple(pose)),
        "right_boundary_odom": transform_base_xy(candidate["right_boundary"], tuple(pose)),
        "source_time_pose": list(pose),
    }


def orient_portal_tangent_upstream(
    portal_normal: Sequence[float],
    corridor_approach_direction: Sequence[float],
) -> Dict[str, Any]:
    """Choose the Portal tangent that points opposite formal corridor travel."""
    if not (_finite_shape(portal_normal, 2) and _finite_shape(corridor_approach_direction, 2)):
        return {"valid": False, "reason": "CORRIDOR_APPROACH_DIRECTION_UNAVAILABLE"}
    normal_norm = math.hypot(float(portal_normal[0]), float(portal_normal[1]))
    approach_norm = math.hypot(float(corridor_approach_direction[0]), float(corridor_approach_direction[1]))
    if normal_norm <= 0.0 or approach_norm <= 0.0:
        return {"valid": False, "reason": "CORRIDOR_APPROACH_DIRECTION_UNAVAILABLE"}
    normal = [float(portal_normal[0]) / normal_norm, float(portal_normal[1]) / normal_norm]
    approach = [float(corridor_approach_direction[0]) / approach_norm, float(corridor_approach_direction[1]) / approach_norm]
    raw = [-normal[1], normal[0]]
    raw_dot_approach = raw[0] * approach[0] + raw[1] * approach[1]
    upstream = raw if raw_dot_approach < 0.0 else [-raw[0], -raw[1]]
    return {
        "valid": True,
        "raw_tangent_odom": raw,
        "approach_direction_odom": approach,
        "raw_tangent_dot_approach": raw_dot_approach,
        "upstream_tangent_odom": upstream,
        "upstream_tangent_dot_approach": upstream[0] * approach[0] + upstream[1] * approach[1],
    }


def build_g14_shadow_target(
    candidate: Dict[str, Any],
    binding: Dict[str, Any],
    current_pose: Optional[Tuple[float, float, float]] = None,
    corridor_approach_direction: Optional[Sequence[float]] = None,
    p_pre_upstream_tangent_m: float = STAIR_MOVING_TURN_P_PRE_TANGENT_M,
) -> Dict[str, Any]:
    frozen = freeze_portal_geometry(candidate, binding)
    width = float(candidate.get("portal_width") or 0.0)
    required_width = 2.0 * DISCRETE_INFLATION_REACH_M
    target = {
        "target_mode": "G14_SHADOW_ONLY", "safe_for_navigation": False,
        "planner_ready": False, "send_to_navigation": False,
        "diagnostic_only": True, "control_authority": False,
        "portal_identity": {
            "run_id": candidate.get("portal_run_id"), "frame_sequence": candidate.get("portal_frame_sequence"),
            "source_stamp": candidate.get("portal_source_stamp"), "track_id": candidate.get("portal_track_id"), "side": candidate.get("side"),
        },
        "source_time_odom_binding": binding,
        "static_radius_m": ROBOT_STATIC_RADIUS_M, "static_diameter_m": 2.0 * ROBOT_STATIC_RADIUS_M,
        "grid_resolution_m": GRID_RESOLUTION_M, "discrete_inflation_cells": DISCRETE_INFLATION_CELLS,
        "discrete_inflation_reach_m": DISCRETE_INFLATION_REACH_M,
        "discrete_required_width_m": required_width, "portal_width_m": width,
        "residual_width_after_discrete_envelope_m": width - required_width,
        "width_gate_pass": width > required_width,
        "frozen_geometry": frozen,
    }
    if not frozen.get("geometry_valid"):
        target.update({"target_valid": False, "rejection_reason": frozen.get("reason")})
        return target
    if not target["width_gate_pass"]:
        target.update({"target_valid": False, "rejection_reason": "PORTAL_WIDTH_DISCRETE_ENVELOPE_INSUFFICIENT"})
        return target
    if not finite_number(p_pre_upstream_tangent_m) or float(p_pre_upstream_tangent_m) < 0.0:
        target.update({"target_valid": False, "rejection_reason": "P_PRE_UPSTREAM_TANGENT_INVALID"})
        return target
    # The quarter-arc used by P_through crosses the portal plane at
    # ``P_pre_tangent - P_pre_outside_normal``.  The historic staging value
    # did not account for the portal width, so it could place that crossing
    # at (or beyond) the discrete inflated clearance boundary.  Keep the
    # original request whenever it already fits; otherwise clamp only the
    # excess to the centre of the last formally usable fine-grid strip, plus
    # one occupied-cell raster guard.  The latter is required because the raw
    # jamb evidence used by collision can occupy the cell immediately inside
    # the nominal continuous portal boundary.
    requested_tangent_m = float(p_pre_upstream_tangent_m)
    requested_crossing_tangent_m = (
        requested_tangent_m - STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M
    )
    crossing_tangent_limit_m = max(
        0.0,
        0.5 * width
        - DISCRETE_INFLATION_REACH_M
        - 0.5 * GRID_RESOLUTION_M
        - PORTAL_JAMB_RASTER_GUARD_M,
    )
    selected_tangent_m = requested_tangent_m
    if requested_crossing_tangent_m > 0.0:
        selected_tangent_m = STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M + min(
            requested_crossing_tangent_m,
            crossing_tangent_limit_m,
        )
    centre, normal = frozen["portal_center_odom"], frozen["portal_normal_odom"]
    # R33 changes P_pre from a stop-only normal retreat to a Portal-relative
    # Stair moving-turn staging pose.  The historical values are a geometry
    # seed only; fresh-pose swept validation remains mandatory at handoff.
    orientation = orient_portal_tangent_upstream(normal, corridor_approach_direction or [])
    if not orientation.get("valid"):
        target.update({"target_valid": False, "rejection_reason": orientation.get("reason")})
        return target
    tangent = orientation["upstream_tangent_odom"]
    pre = [
        centre[0] - STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M * normal[0] + selected_tangent_m * tangent[0],
        centre[1] - STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M * normal[1] + selected_tangent_m * tangent[1],
    ]
    through = [centre[0] + DISCRETE_INFLATION_REACH_M * normal[0], centre[1] + DISCRETE_INFLATION_REACH_M * normal[1]]
    target.update({
        "target_valid": True,
        "candidate_lifecycle": "COMMITTED",
        "P_pre_odom": pre,
        "P_through_odom": through,
        "p_pre_contract": "STATIC_SAFE_REACHABLE_STAIR_TURNING_RUNWAY_FULL_SWEPT_PORTAL_SAFE",
        "p_pre_selection_mode": "PORTAL_RELATIVE_HISTORICAL_STAIR_STAGING",
        "p_pre_portal_relative": {
            "outside_normal_m": STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M,
            "requested_upstream_tangent_offset_m": requested_tangent_m,
            "upstream_tangent_offset_m": selected_tangent_m,
            "requested_crossing_tangent_offset_m": requested_crossing_tangent_m,
            "crossing_tangent_limit_m": crossing_tangent_limit_m,
            "portal_jamb_raster_guard_m": PORTAL_JAMB_RASTER_GUARD_M,
            "projected_crossing_tangent_offset_m": max(
                0.0,
                selected_tangent_m - STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M,
            ),
            "width_aware_tangent_clamp_applied": selected_tangent_m < requested_tangent_m,
        },
        "portal_tangent_orientation": orientation,
    })
    if current_pose is not None and _finite_shape(current_pose, 3):
        current_centre = target_base_xy(centre, current_pose)
        current_pre = target_base_xy(pre, current_pose)
        current_through = target_base_xy(through, current_pose)
        target["current_base_diagnostics"] = {
            "current_portal_center_base": list(current_centre), "current_pre_target_base": list(current_pre),
            "current_through_target_base": list(current_through), "portal_center_in_front": current_centre[0] > 0,
            "pre_target_in_front": current_pre[0] > 0, "through_target_in_front": current_through[0] > 0,
            "current_distance_to_portal_plane": current_centre[0],
            "current_heading_to_pre_target": math.atan2(current_pre[1], current_pre[0]),
        }
        if current_centre[0] <= 0:
            target["candidate_lifecycle"] = "PASSED_OR_EXPIRED"
    return target


def g14_shadow_stop_eligible(target: Any) -> bool:
    """The online stop is valid only for a still-forward committed shadow."""
    diagnostics = target.get("current_base_diagnostics") if isinstance(target, dict) else None
    return bool(
        isinstance(target, dict)
        and target.get("target_valid") is True
        and target.get("candidate_lifecycle") == "COMMITTED"
        and isinstance(diagnostics, dict)
        and diagnostics.get("portal_center_in_front") is True
        and diagnostics.get("pre_target_in_front") is True
    )


def evaluate_portal_g14_p_pre_admissibility(
    candidate: Any,
    target: Any,
    current_pose: Tuple[float, float, float],
    grid_msg: Any,
    status_payload: Any,
    robot_radius_m: float,
) -> Dict[str, Any]:
    """Admit frozen ideal P_pre, or one bounded normal retreat on its exact grid pair."""
    ideal_odom = None
    base = None
    if isinstance(target, dict) and _finite_shape(target.get("P_pre_odom"), 2):
        ideal_odom = [float(target["P_pre_odom"][0]), float(target["P_pre_odom"][1])]
        base = target_base_xy(ideal_odom, current_pose)
    result: Dict[str, Any] = {
        "target_source": "PORTAL_G14_P_PRE",
        "candidate_lifecycle": target.get("candidate_lifecycle") if isinstance(target, dict) else None,
        "P_pre_base_x": base[0] if base is not None else None,
        "P_pre_base_y": base[1] if base is not None else None,
        "P_pre_distance": math.hypot(*base) if base is not None else None,
        "P_pre_heading": math.atan2(base[1], base[0]) if base is not None else None,
        "P_pre_in_front": bool(base is not None and base[0] > 0.0),
        "planning_blocked": True,
        "target_cell_admissible": False,
        "ideal_d_m": DISCRETE_INFLATION_REACH_M,
        "P_ideal_odom": ideal_odom,
        "normal_search_triggered": False,
    }
    identity = target.get("portal_identity") if isinstance(target, dict) else None
    frozen = target.get("frozen_geometry") if isinstance(target, dict) else None
    identity_matches = bool(
        isinstance(candidate, dict) and isinstance(identity, dict)
        and candidate.get("portal_run_id") == identity.get("run_id")
        and candidate.get("portal_frame_sequence") == identity.get("frame_sequence")
        and candidate.get("portal_source_stamp") == identity.get("source_stamp")
        and candidate.get("portal_track_id") == identity.get("track_id")
        and candidate.get("side") == identity.get("side")
    )
    if not (
        isinstance(target, dict) and target.get("target_valid") is True
        and target.get("candidate_lifecycle") == "COMMITTED"
        and isinstance(frozen, dict) and frozen.get("geometry_valid") is True
        and identity_matches and base is not None and base[0] > 0.0
    ):
        result.update({"state": "P_PRE_CANDIDATE_INVALIDATED", "reason": "COMMITTED_FROZEN_P_PRE_CONTRACT_INVALID"})
        return result
    cell = local_xy_to_cell(base[0], base[1], grid_msg)
    result["target_cell"] = list(cell) if cell is not None else None
    if cell is None:
        result.update({"state": "P_PRE_OUTSIDE_LOCAL_PLANNING_WINDOW", "reason": "P_PRE_OUTSIDE_FORMAL_LOCAL_GRID"})
        return result
    grid_qualified, qualification_errors = qualified_for_navigation(grid_msg, status_payload)
    result["grid_qualified"] = bool(grid_qualified)
    result["grid_qualification_errors"] = list(qualification_errors)
    if not grid_qualified:
        result.update({"state": "P_PRE_GRID_UNQUALIFIED", "reason": "FORMAL_LOCAL_GRID_UNQUALIFIED"})
        return result
    try:
        grid = grid_array(grid_msg)
        inflated = p_pre_inflated_blocked(grid, robot_radius_m, float(grid_msg.info.resolution))
    except Exception as exc:
        result.update({"state": "P_PRE_GRID_UNQUALIFIED", "reason": f"FORMAL_GRID_READ_FAILED:{type(exc).__name__}"})
        return result
    def inspect(point_odom: Sequence[float], distance_m: float) -> Dict[str, Any]:
        point_base = target_base_xy(point_odom, current_pose)
        point_cell = local_xy_to_cell(point_base[0], point_base[1], grid_msg)
        entry: Dict[str, Any] = {
            "d_m": float(distance_m),
            "P_pre_odom": [float(point_odom[0]), float(point_odom[1])],
            "P_pre_base": [float(point_base[0]), float(point_base[1])],
            "in_front": bool(point_base[0] > 0.0),
            "target_cell": list(point_cell) if point_cell is not None else None,
            "raw_value": None,
            "inflated_blocked": True,
            "formally_safe": False,
        }
        if point_cell is None:
            return entry
        x_index, y_index = point_cell
        entry["raw_value"] = int(grid[y_index, x_index])
        entry["inflated_blocked"] = bool(inflated[y_index, x_index])
        entry["formally_safe"] = bool(entry["in_front"] and entry["raw_value"] == 0 and not entry["inflated_blocked"])
        return entry

    stair_turning_runway = isinstance(target, dict) and target.get("p_pre_selection_mode") == "PORTAL_RELATIVE_HISTORICAL_STAIR_STAGING"
    ideal_distance = STAIR_MOVING_TURN_P_PRE_OUTSIDE_NORMAL_M if stair_turning_runway else DISCRETE_INFLATION_REACH_M
    ideal = inspect(ideal_odom, ideal_distance)
    result.update({
        "target_cell": ideal["target_cell"],
        "target_cell_raw_value": ideal["raw_value"],
        "target_cell_inflated_blocked": ideal["inflated_blocked"],
        "ideal_admissibility": ideal,
    })
    selection_identity = {
        "grid_content_stamp": status_payload.get("grid_content_stamp") if isinstance(status_payload, dict) else None,
        "grid_content_hash": status_payload.get("grid_content_hash") if isinstance(status_payload, dict) else None,
        "content_generation_id": status_payload.get("content_generation_id") if isinstance(status_payload, dict) else None,
    }
    if ideal["formally_safe"]:
        result.update({
            "state": "P_PRE_READY_FOR_PLANNER",
            "reason": "FORMAL_LOCAL_GRID_AND_IDEAL_TARGET_CELL_ADMISSIBLE",
            "planning_blocked": False,
            "target_cell_admissible": True,
            "selected_d_m": ideal_distance,
            "selected_P_pre_odom": ideal["P_pre_odom"],
            "selection_reason": "KEEP_IDEAL_P_PRE",
            "selection_grid_status_identity": selection_identity,
        })
        return result

    # A normal-only retreat revives the pre-R33 stop-only P_pre contract and
    # cannot be substituted for a rejected turning-runway staging pose.
    if stair_turning_runway:
        result.update({
            "state": "P_PRE_NO_SAFE_NORMAL_DISTANCE",
            "reason": "P2KG15_P_PRE_TURNING_RUNWAY_UNAVAILABLE",
        })
        return result

    frozen_center = frozen.get("portal_center_odom") if isinstance(frozen, dict) else None
    frozen_normal = frozen.get("portal_normal_odom") if isinstance(frozen, dict) else None
    portal_width = target.get("portal_width_m") if isinstance(target, dict) else None
    if not (_finite_shape(frozen_center, 2) and _finite_shape(frozen_normal, 2) and finite_number(portal_width) and float(portal_width) > 0.0):
        result.update({"state": "P_PRE_TARGET_CELL_BLOCKED", "reason": "P_PRE_TARGET_CELL_NOT_FORMALLY_ADMISSIBLE"})
        return result
    normal_norm = math.hypot(float(frozen_normal[0]), float(frozen_normal[1]))
    if normal_norm <= 0.0:
        result.update({"state": "P_PRE_TARGET_CELL_BLOCKED", "reason": "P_PRE_TARGET_CELL_NOT_FORMALLY_ADMISSIBLE"})
        return result
    normal = [float(frozen_normal[0]) / normal_norm, float(frozen_normal[1]) / normal_norm]
    center_base = target_base_xy(frozen_center, current_pose)
    yaw = float(current_pose[2])
    normal_base = [
        math.cos(yaw) * normal[0] + math.sin(yaw) * normal[1],
        -math.sin(yaw) * normal[0] + math.cos(yaw) * normal[1],
    ]
    robot_distance_to_plane = center_base[0] * normal_base[0] + center_base[1] * normal_base[1]
    max_distance = min(float(portal_width), robot_distance_to_plane - float(robot_radius_m))
    resolution = float(grid_msg.info.resolution)
    result.update({
        "normal_search_triggered": True,
        "normal_search_policy": "FROZEN_PORTAL_NORMAL_OUTWARD_MINIMUM_D",
        "normal_interval_requirement": "TWO_CONSECUTIVE_RESOLUTION_STEPS_FORMALLY_SAFE",
        "max_allowed_P_pre_normal_distance_m": max_distance,
        "max_distance_components_m": {
            "portal_width_m": float(portal_width),
            "robot_to_portal_plane_minus_radius_m": robot_distance_to_plane - float(robot_radius_m),
        },
        "normal_search_attempts": [ideal],
    })
    if max_distance < DISCRETE_INFLATION_REACH_M + resolution - 1e-12:
        result.update({"state": "P_PRE_NO_SAFE_NORMAL_DISTANCE", "reason": "P_PRE_NORMAL_RETREAT_BOUND_INSUFFICIENT"})
        return result
    steps = int(math.floor((max_distance - DISCRETE_INFLATION_REACH_M + 1e-12) / resolution))
    previous_attempt_safe = False
    for step in range(1, steps + 1):
        distance_m = DISCRETE_INFLATION_REACH_M + step * resolution
        point_odom = [
            float(frozen_center[0]) - distance_m * normal[0],
            float(frozen_center[1]) - distance_m * normal[1],
        ]
        attempt = inspect(point_odom, distance_m)
        result["normal_search_attempts"].append(attempt)
        # A single free cell at the edge of an occupied band is not a stable
        # normal-distance interval.  Require this point and its immediately
        # portalward 0.05 m neighbour to be formally safe, then freeze the
        # first outer endpoint.  The ideal 0.30 m case returned above is
        # deliberately unaffected.
        if not (attempt["formally_safe"] and previous_attempt_safe):
            previous_attempt_safe = bool(attempt["formally_safe"])
            continue
        result.update({
            "P_pre_base_x": attempt["P_pre_base"][0],
            "P_pre_base_y": attempt["P_pre_base"][1],
            "P_pre_distance": math.hypot(*attempt["P_pre_base"]),
            "P_pre_heading": math.atan2(attempt["P_pre_base"][1], attempt["P_pre_base"][0]),
            "P_pre_in_front": True,
            "target_cell": attempt["target_cell"],
            "target_cell_raw_value": attempt["raw_value"],
            "target_cell_inflated_blocked": attempt["inflated_blocked"],
            "state": "P_PRE_READY_FOR_PLANNER",
            "reason": "FORMAL_LOCAL_GRID_AND_NORMAL_RETREAT_TARGET_ADMISSIBLE",
            "planning_blocked": False,
            "target_cell_admissible": True,
            "selected_d_m": distance_m,
            "selected_P_pre_odom": attempt["P_pre_odom"],
            "selection_reason": "MINIMUM_FORMALLY_SAFE_FROZEN_NORMAL_RETREAT",
            "selection_grid_status_identity": selection_identity,
        })
        return result
    result.update({"state": "P_PRE_NO_SAFE_NORMAL_DISTANCE", "reason": "NO_FORMALLY_SAFE_NORMAL_RETREAT_WITHIN_BOUND"})
    return result


def p_pre_pose_binding_for_grid_status(
    odom_cache: OdomCache,
    status_payload: Any,
) -> Dict[str, Any]:
    """Bind a formal base-frame Grid to its producer Odom time, never latest.

    L3V already records ``source_odom_stamp`` in the status paired with the
    Grid content.  The P_pre projection must use that pose (or the cache's
    existing bounded interpolation contract), otherwise a newer base pose is
    being queried against an older base-frame Grid.
    """
    status = status_payload if isinstance(status_payload, dict) else {}
    source_stamp = status.get("source_odom_stamp")
    if not finite_number(source_stamp):
        return {
            "binding_valid": False,
            "reason": "GRID_STATUS_SOURCE_ODOM_STAMP_UNAVAILABLE",
            "source_odom_stamp": None,
        }
    binding = odom_cache.pose_at_source_stamp(float(source_stamp))
    binding = dict(binding) if isinstance(binding, dict) else {
        "binding_valid": False, "reason": "ODOM_SOURCE_BINDING_INVALID_RETURN"
    }
    binding["source_odom_stamp"] = float(source_stamp)
    if binding.get("binding_valid") is not True:
        binding.setdefault("reason", "ODOM_SOURCE_BINDING_UNAVAILABLE")
        return binding
    pose = binding.get("source_pose_x_y_yaw")
    if not _finite_shape(pose, 3):
        return {
            "binding_valid": False,
            "reason": "ODOM_SOURCE_BINDING_POSE_INVALID",
            "source_odom_stamp": float(source_stamp),
        }
    return binding


def p_pre_grid_status_pair_pending_admissibility(
    candidate: Any,
    target: Any,
    current_pose: Optional[Tuple[float, float, float]],
    latest_grid: Any,
    pose_binding: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wait without projecting a P_pre onto unmatched Grid/Pose evidence."""
    base = None
    if (
        current_pose is not None
        and _finite_shape(current_pose, 3)
        and isinstance(target, dict)
        and _finite_shape(target.get("P_pre_odom"), 2)
    ):
        base = target_base_xy(target["P_pre_odom"], current_pose)
    result: Dict[str, Any] = {
        "target_source": "PORTAL_G14_P_PRE",
        "candidate_lifecycle": target.get("candidate_lifecycle") if isinstance(target, dict) else None,
        "P_pre_base_x": base[0] if base is not None else None,
        "P_pre_base_y": base[1] if base is not None else None,
        "P_pre_distance": math.hypot(*base) if base is not None else None,
        "P_pre_heading": math.atan2(base[1], base[0]) if base is not None else None,
        "P_pre_in_front": bool(base is not None and base[0] > 0.0),
        "planning_blocked": True,
        "target_cell_admissible": False,
        "grid_qualified": False,
        "pose_binding": dict(pose_binding) if isinstance(pose_binding, dict) else None,
    }
    identity = target.get("portal_identity") if isinstance(target, dict) else None
    frozen = target.get("frozen_geometry") if isinstance(target, dict) else None
    identity_matches = bool(
        isinstance(candidate, dict) and isinstance(identity, dict)
        and candidate.get("portal_run_id") == identity.get("run_id")
        and candidate.get("portal_frame_sequence") == identity.get("frame_sequence")
        and candidate.get("portal_source_stamp") == identity.get("source_stamp")
        and candidate.get("portal_track_id") == identity.get("track_id")
        and candidate.get("side") == identity.get("side")
    )
    if not (
        isinstance(target, dict) and target.get("target_valid") is True
        and target.get("candidate_lifecycle") == "COMMITTED"
        and isinstance(frozen, dict) and frozen.get("geometry_valid") is True
        and identity_matches
    ):
        result.update({"state": "P_PRE_CANDIDATE_INVALIDATED", "reason": "COMMITTED_FROZEN_P_PRE_CONTRACT_INVALID"})
        return result
    # Do not combine ``latest_grid`` with a latest/current pose here.  Pair
    # absence or missing source Odom is an evidence wait, not a grid verdict.
    result.update({
        "state": "P_PRE_GRID_STATUS_PAIR_PENDING",
        "reason": "WAITING_FOR_MATCHED_GRID_STATUS_AND_SOURCE_ODOM",
        "grid_qualification_errors": ["exact_grid_status_pair_or_source_odom_unavailable"],
    })
    return result


def write_portal_g14_p_pre_target(target: Dict[str, Any], admissibility: Dict[str, Any]) -> Dict[str, Any]:
    selected = admissibility.get("selected_P_pre_odom") if isinstance(admissibility, dict) else None
    selected_odom = selected if _finite_shape(selected, 2) else target["P_pre_odom"]
    return write_absolute_target(
        selected_odom,
        "PORTAL_G14_P_PRE",
        "portal_g14_p_pre",
        {
            "portal_identity": target.get("portal_identity"),
            "frozen_P_pre_odom": list(target["P_pre_odom"]),
            "selected_P_pre_odom": list(selected_odom),
            "selected_d_m": admissibility.get("selected_d_m"),
            "normal_search_triggered": admissibility.get("normal_search_triggered"),
            "selection_grid_status_identity": admissibility.get("selection_grid_status_identity"),
            "portal_g14_p_pre_admissibility": admissibility,
            "target_source": "PORTAL_G14_P_PRE",
        },
    )


def write_portal_g14_p_through_target(target: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the staging goal with the semantic room-side goal only.

    The local Block-A*/DWA runner owns all path, waypoint, and command
    decisions after this handoff.  Frozen portal geometry remains attached for
    high-level completion auditing, not for a second motion authority.
    """
    through = target.get("P_through_odom") if isinstance(target, dict) else None
    if not _finite_shape(through, 2):
        raise ValueError("P_through_odom_invalid")
    return write_absolute_target(
        through,
        "PORTAL_G14_P_THROUGH",
        "portal_g14_p_through",
        {
            "portal_identity": target.get("portal_identity"),
            "P_pre_odom": target.get("P_pre_odom"),
            "P_through_odom": list(through),
            "frozen_geometry": target.get("frozen_geometry"),
            "high_level_goal_type": "ROOM_ENTRY",
            "high_level_completion": "PORTAL_CROSSED_AND_P_THROUGH_REACHED",
        },
    )


def portal_g14_p_pre_runner_outcome(runner: Dict[str, Any]) -> Dict[str, Any]:
    decision = str(runner.get("runner_final_decision") or "")
    if decision == "BLOCK_ASTAR_DWA_REACHED_GOAL":
        return {"reached": True, "final_decision": "STATE_MACHINE_STOP_AFTER_PORTAL_G14_P_PRE_REACHED", "reason": "portal_g14_p_pre_reached"}
    return {"reached": False, "final_decision": "P2KG15_092_P_PRE_EXECUTION_FAILED", "reason": f"portal_g14_p_pre_runner_failed:{decision or 'UNKNOWN'}"}


def portal_g14_p_pre_max_steps(args: argparse.Namespace) -> int:
    """Return the P_pre-only runner budget without changing P_through."""
    return int(args.portal_p_pre_max_steps)


def portal_g14_p_through_max_steps(args: argparse.Namespace) -> int:
    """Return the P_through-only budget after the P_pre-to-Portal arc handoff."""
    return int(args.portal_p_through_max_steps)


def portal_g14_p_through_runner_outcome(
    runner: Dict[str, Any],
    target: Dict[str, Any],
    final_pose: Tuple[float, float, float],
    *,
    deep_crossing_min_progress_m: float = 0.50,
    deep_crossing_max_distance_m: float = 0.50,
    robot_radius_m: float = STATIC_PLANNING_FOOTPRINT_RADIUS_M,
) -> Dict[str, Any]:
    """Keep semantic completion above the generic local-motion authority."""
    decision = str(runner.get("runner_final_decision") or "")
    through = target.get("P_through_odom") if isinstance(target, dict) else None
    frozen = target.get("frozen_geometry") if isinstance(target, dict) else None
    centre = frozen.get("portal_center_odom") if isinstance(frozen, dict) else None
    normal = frozen.get("portal_normal_odom") if isinstance(frozen, dict) else None
    if not (_finite_shape(through, 2) and _finite_shape(centre, 2) and _finite_shape(normal, 2)):
        return {"reached": False, "final_decision": "P2KG15_093_P_THROUGH_CONTRACT_INVALID", "reason": "P_THROUGH_OR_FROZEN_PORTAL_INVALID"}
    normal_norm = math.hypot(float(normal[0]), float(normal[1]))
    if normal_norm <= 0.0:
        return {"reached": False, "final_decision": "P2KG15_093_P_THROUGH_CONTRACT_INVALID", "reason": "P_THROUGH_PORTAL_NORMAL_INVALID"}
    unit_normal = (float(normal[0]) / normal_norm, float(normal[1]) / normal_norm)
    tangent = (-unit_normal[1], unit_normal[0])
    normal_progress = (
        (float(final_pose[0]) - float(centre[0])) * unit_normal[0]
        + (float(final_pose[1]) - float(centre[1])) * unit_normal[1]
    )
    tangent_offset = (
        (float(final_pose[0]) - float(centre[0])) * tangent[0]
        + (float(final_pose[1]) - float(centre[1])) * tangent[1]
    )
    through_distance = math.hypot(float(final_pose[0]) - float(through[0]), float(final_pose[1]) - float(through[1]))
    if decision == "BLOCK_ASTAR_DWA_REACHED_GOAL" and normal_progress >= 0.0:
        return {
            "reached": True,
            "final_decision": "P2KG15_FULL_FOOTPRINT_SAFE_P_THROUGH_REACHED",
            "reason": "LOCAL_AUTONOMY_P_THROUGH_REACHED_ZERO_STOP",
            "portal_normal_progress_m": normal_progress,
            "distance_to_p_through_m": through_distance,
        }
    portal_width = target.get("portal_width_m") if isinstance(target, dict) else None
    aperture_half_width = (
        float(portal_width) / 2.0 - float(robot_radius_m)
        if finite_number(portal_width) and float(portal_width) > 0.0 else None
    )
    deep_crossing = bool(
        decision == "BLOCK_ASTAR_DWA_MAX_STEPS"
        and normal_progress >= float(deep_crossing_min_progress_m)
        and through_distance <= float(deep_crossing_max_distance_m)
        and aperture_half_width is not None
        and aperture_half_width > 0.0
        and abs(tangent_offset) <= aperture_half_width
    )
    if deep_crossing:
        return {
            "reached": True,
            "final_decision": "P2KG15_FULL_FOOTPRINT_SAFE_P_THROUGH_REACHED",
            "reason": "LOCAL_AUTONOMY_P_THROUGH_DEEP_CROSSING_MAX_STEPS_STOP",
            "runner_final_decision": decision,
            "portal_normal_progress_m": normal_progress,
            "portal_tangent_offset_m": tangent_offset,
            "portal_aperture_half_width_m": aperture_half_width,
            "distance_to_p_through_m": through_distance,
        }
    return {
        "reached": False,
        "final_decision": "P2KG15_093_P_THROUGH_LOCAL_RUNNER_FAILED",
        "reason": "runner_did_not_reach_room_side_p_through",
        "runner_final_decision": decision,
        "portal_normal_progress_m": normal_progress,
        "portal_tangent_offset_m": tangent_offset,
        "distance_to_p_through_m": through_distance,
    }


def portal_normal_alignment_command(heading_error_rad: float, max_angular_z: float, gain: float) -> Dict[str, float]:
    """One bounded feedback law for 093.  Alignment never commands translation."""
    bounded_w = max(-abs(float(max_angular_z)), min(abs(float(max_angular_z)), float(gain) * float(heading_error_rad)))
    return {"linear_x": 0.0, "angular_z": bounded_w}


def portal_plane_crossing_geometry(
    current_pose: Tuple[float, float, float],
    portal_center_odom: Sequence[float],
    unit_normal_odom: Sequence[float],
    portal_width_m: float,
    robot_radius_m: float,
) -> Dict[str, Any]:
    """Evaluate the current-pose straight-heading intersection with the frozen portal plane.

    This deliberately has no nominal P_pre distance or fixed heading tolerance:
    P_pre is a goal *region*, so 093 must use the pose that actually reached it.
    """
    tangent = [-float(unit_normal_odom[1]), float(unit_normal_odom[0])]
    delta_x = float(current_pose[0]) - float(portal_center_odom[0])
    delta_y = float(current_pose[1]) - float(portal_center_odom[1])
    normal_distance = -(delta_x * float(unit_normal_odom[0]) + delta_y * float(unit_normal_odom[1]))
    tangent_offset = delta_x * tangent[0] + delta_y * tangent[1]
    desired_yaw = math.atan2(float(unit_normal_odom[1]), float(unit_normal_odom[0]))
    theta = normalize_angle(float(current_pose[2]) - desired_yaw)
    aperture_half_width = float(portal_width_m) / 2.0 - float(robot_radius_m)
    result: Dict[str, Any] = {
        "portal_tangent_odom": tangent,
        "robot_yaw_rad": float(current_pose[2]),
        "desired_yaw_rad": desired_yaw,
        "heading_error_rad": normalize_angle(desired_yaw - float(current_pose[2])),
        "portal_normal_theta_rad": theta,
        "portal_normal_distance_D_m": normal_distance,
        "portal_tangent_offset_s_m": tangent_offset,
        "portal_aperture_half_width_H_m": aperture_half_width,
        "portal_plane_crossing_s_m": None,
        "portal_plane_remaining_margin_m": None,
        "portal_plane_crossing_safe": False,
        "portal_plane_crossing_reason": None,
    }
    if aperture_half_width <= 0.0:
        result["portal_plane_crossing_reason"] = "PORTAL_APERTURE_INSUFFICIENT_FOR_STATIC_FOOTPRINT"
        return result
    if normal_distance <= 0.0:
        result["portal_plane_crossing_reason"] = "P_PRE_PORTAL_PLANE_OVERSHOOT"
        return result
    if math.cos(theta) <= 0.0:
        result["portal_plane_crossing_reason"] = "PORTAL_HEADING_DOES_NOT_FACE_PORTAL_PLANE"
        return result
    crossing_s = tangent_offset + normal_distance * math.tan(theta)
    remaining_margin = aperture_half_width - abs(crossing_s)
    result.update({
        "portal_plane_crossing_s_m": crossing_s,
        "portal_plane_remaining_margin_m": remaining_margin,
        "portal_plane_crossing_safe": bool(abs(crossing_s) <= aperture_half_width),
        "portal_plane_crossing_reason": "PORTAL_PLANE_CROSSING_SAFE" if abs(crossing_s) <= aperture_half_width else "PORTAL_PLANE_CROSSING_UNSAFE_TURN_REQUIRED",
    })
    return result


def _stair_integrate_forward_segment(
    start: Tuple[float, float, float],
    kappa: float,
    length_m: float,
) -> List[Tuple[float, float, float]]:
    points = [tuple(float(v) for v in start)]
    x, y, yaw = points[0]
    travelled = 0.0
    while travelled < float(length_m) - 1e-9:
        ds = min(STAIR_MOVING_TURN_PATH_STEP_M, float(length_m) - travelled)
        next_yaw = yaw + float(kappa) * ds
        if abs(float(kappa)) < 1e-9:
            x += math.cos(yaw) * ds
            y += math.sin(yaw) * ds
        else:
            radius = 1.0 / float(kappa)
            x += radius * (math.sin(next_yaw) - math.sin(yaw))
            y += radius * (-math.cos(next_yaw) + math.cos(yaw))
        yaw = next_yaw
        travelled += ds
        points.append((x, y, yaw))
    return points


def _stair_portal_frame(point: Sequence[float], centre: Sequence[float], normal: Sequence[float]) -> Tuple[float, float]:
    tangent = (-float(normal[1]), float(normal[0]))
    dx = float(point[0]) - float(centre[0])
    dy = float(point[1]) - float(centre[1])
    return dx * float(normal[0]) + dy * float(normal[1]), dx * tangent[0] + dy * tangent[1]


def _stair_swept_grid_check(
    grid_msg: OccupancyGrid,
    grid: np.ndarray,
    current_pose: Tuple[float, float, float],
    path: Sequence[Tuple[float, float, float]],
    radius_m: float,
) -> Dict[str, Any]:
    """Conservative raw-cell circle sweep; unknown and outside are blocked."""
    resolution = float(grid_msg.info.resolution)
    radius_cells = int(math.ceil(float(radius_m) / max(resolution, 1e-6))) + 1
    for index, point in enumerate(path):
        point_base = target_base_xy(point, current_pose)
        cell = local_xy_to_cell(point_base[0], point_base[1], grid_msg)
        if cell is None:
            return {"safe": False, "state": "OUT_OF_GRID", "path_distance_m": index * STAIR_MOVING_TURN_PATH_STEP_M}
        covered: List[Tuple[int, int, int]] = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                x_index, y_index = cell[0] + dx, cell[1] + dy
                if not (0 <= x_index < grid.shape[1] and 0 <= y_index < grid.shape[0]):
                    continue
                local = cell_to_metric(x_index, y_index, grid_metadata(grid_msg))
                if local is None or math.hypot(local[0] - point_base[0], local[1] - point_base[1]) > float(radius_m):
                    continue
                covered.append((x_index, y_index, int(grid[y_index, x_index])))
        if not covered:
            return {"safe": False, "state": "OUT_OF_GRID", "path_distance_m": index * STAIR_MOVING_TURN_PATH_STEP_M}
        if any(value == 100 for _x, _y, value in covered):
            return {
                "safe": False, "state": "OCCUPIED", "path_distance_m": index * STAIR_MOVING_TURN_PATH_STEP_M,
                "triggering_cells": [[x, y] for x, y, value in covered if value == 100],
            }
        if any(value != 0 for _x, _y, value in covered):
            return {
                "safe": False, "state": "UNKNOWN", "path_distance_m": index * STAIR_MOVING_TURN_PATH_STEP_M,
                "triggering_cells": [[x, y] for x, y, value in covered if value != 0],
            }
    return {"safe": True, "state": "FREE", "path_distance_m": (len(path) - 1) * STAIR_MOVING_TURN_PATH_STEP_M}


def build_stair_moving_turn_entry_path(
    target: Dict[str, Any],
    current_pose: Tuple[float, float, float],
    grid_msg: OccupancyGrid,
    status_payload: Dict[str, Any],
    robot_radius_m: float,
    require_arrival_region: bool = True,
) -> Dict[str, Any]:
    """Build and validate the smallest bounded Stair-native forward entry arc."""
    result: Dict[str, Any] = {
        "state": "STAIR_MOVING_TURN_PATH_REJECTED", "feasible": False,
        "pure_yaw_required": False, "path": [], "candidates_tested": 0,
    }
    frozen = target.get("frozen_geometry") if isinstance(target, dict) else None
    centre = frozen.get("portal_center_odom") if isinstance(frozen, dict) else None
    normal = frozen.get("portal_normal_odom") if isinstance(frozen, dict) else None
    through = target.get("P_through_odom") if isinstance(target, dict) else None
    width = target.get("portal_width_m") if isinstance(target, dict) else None
    if not (isinstance(frozen, dict) and frozen.get("geometry_valid") is True and _finite_shape(centre, 2)
            and _finite_shape(normal, 2) and _finite_shape(through, 2) and finite_number(width)):
        result["reason"] = "FROZEN_PORTAL_GEOMETRY_INVALID"
        return result
    qualified, errors = qualified_for_navigation(grid_msg, status_payload)
    result["grid_qualified"] = bool(qualified)
    result["grid_qualification_errors"] = list(errors)
    if not qualified:
        result["reason"] = "FORMAL_GRID_STATUS_PAIR_UNQUALIFIED"
        return result
    norm = math.hypot(float(normal[0]), float(normal[1]))
    unit_normal = (float(normal[0]) / norm, float(normal[1]) / norm)
    half_aperture = float(width) / 2.0 - float(robot_radius_m)
    if half_aperture <= 0.0:
        result["reason"] = "PORTAL_APERTURE_INSUFFICIENT"
        return result
    grid = grid_array(grid_msg)
    current_to_pre = math.hypot(current_pose[0] - float(target["P_pre_odom"][0]), current_pose[1] - float(target["P_pre_odom"][1]))
    result["actual_arrival_distance_to_selected_p_pre_m"] = current_to_pre
    if require_arrival_region and current_to_pre > STAIR_MOVING_TURN_GOAL_TOLERANCE_M:
        result["reason"] = "ACTUAL_ARRIVAL_OUTSIDE_SELECTED_P_PRE_REGION"
        return result
    candidates: List[Dict[str, Any]] = []
    for step in range(-66, 67):
        kappa = float(step) * STAIR_MOVING_TURN_KAPPA_MAX_M_INV / 66.0
        path = _stair_integrate_forward_segment(current_pose, kappa, 4.0)
        goal_index = None
        for index, point in enumerate(path):
            normal_progress, _tangent = _stair_portal_frame(point, centre, unit_normal)
            if normal_progress >= STAIR_MOVING_TURN_GOAL_NORMAL_M:
                goal_index = index
                break
        if goal_index is None:
            continue
        path = path[:goal_index + 1]
        crossing = None
        for index in range(1, len(path)):
            n0, _ = _stair_portal_frame(path[index - 1], centre, unit_normal)
            n1, tangent = _stair_portal_frame(path[index], centre, unit_normal)
            if n0 < 0.0 <= n1:
                crossing = {"path_index": index, "tangent_m": tangent, "aperture_margin_m": half_aperture - abs(tangent)}
                break
        if crossing is None or crossing["aperture_margin_m"] < 0.0:
            continue
        final_n, final_t = _stair_portal_frame(path[-1], centre, unit_normal)
        if final_n < STAIR_MOVING_TURN_GOAL_NORMAL_M - GRID_RESOLUTION_M or abs(final_t) > STAIR_MOVING_TURN_GOAL_TOLERANCE_M - GRID_RESOLUTION_M:
            continue
        sweep = _stair_swept_grid_check(grid_msg, grid, current_pose, path, robot_radius_m)
        candidate = {
            "kappa_m_inv": kappa,
            "path_length_m": (len(path) - 1) * STAIR_MOVING_TURN_PATH_STEP_M,
            "crossing": crossing,
            "final_portal_normal_m": final_n,
            "final_portal_tangent_m": final_t,
            "p_through_goal_margin_m": STAIR_MOVING_TURN_GOAL_TOLERANCE_M - abs(final_t),
            "swept_grid": sweep,
            "path": [[float(x), float(y), float(yaw)] for x, y, yaw in path],
        }
        result["candidates_tested"] = int(result["candidates_tested"]) + 1
        if sweep.get("safe"):
            candidates.append(candidate)
    if not candidates:
        result["reason"] = "NO_FULL_SWEPT_FORWARD_STAIR_PATH"
        return result
    candidates.sort(key=lambda row: (abs(float(row["kappa_m_inv"])), float(row["path_length_m"])))
    selected = candidates[0]
    result.update({
        "state": "STAIR_MOVING_TURN_PATH_READY", "feasible": True,
        "reason": "FULL_SWEPT_FORWARD_STAIR_PATH_SAFE", "selected": selected,
        "path": selected["path"], "kappa_m_inv": selected["kappa_m_inv"],
    })
    return result


def execute_stair_moving_turn_portal_entry(
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    target: Dict[str, Any],
    formal_grid_status_pair: FormalGridStatusPairSubscriber,
) -> Dict[str, Any]:
    """R33's only doorway motion authority: bounded forward DWA path tracking."""
    topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
    pub = rospy.Publisher(topic, Twist, queue_size=2)
    dwa_args = build_block_astar_dwa_arg_parser().parse_args([])
    dwa_args.robot_radius_m = float(args.robot_radius_m)
    dwa_args.enforce_min_forward_speed = True
    dwa_args.min_linear_x = STAIR_MOVING_TURN_MIN_LINEAR_X_MPS
    dwa_args.max_linear_x = STAIR_MOVING_TURN_MAX_LINEAR_X_MPS
    dwa_args.max_angular_z = STAIR_MOVING_TURN_MAX_ABS_ANGULAR_Z_RADPS
    dwa_args.goal_tolerance_m = STAIR_MOVING_TURN_GOAL_TOLERANCE_M
    dwa_args.disable_pointcloud_wall_heading = True
    dwa_args.disable_imu_heading_hold = True
    dwa = object.__new__(BlockAStarDwaRunner)
    dwa.args = dwa_args
    dwa.prev_cmd = (0.0, 0.0)
    started_sim = float(rospy.Time.now().to_sec())
    rate = rospy.Rate(10.0)
    trace: List[Dict[str, Any]] = []
    crossing_stamp = None
    first_turn_yaw = None
    nonzero_turn_count = 0
    last_progress = None
    last_progress_sim = started_sim
    try:
        while not rospy.is_shutdown():
            now_sim = float(rospy.Time.now().to_sec())
            if now_sim - started_sim > float(args.runner_runtime_sec):
                return portal_normal_alignment_failure(args, "P2KG15_093_MOVING_TURN_TIMEOUT", "MOVING_TURN_TIMEOUT", trace, executed_safe_trace=trace)
            pair = formal_grid_status_pair.matching_pair()
            if pair is None:
                return portal_normal_alignment_failure(args, "P2KG15_093_MOVING_TURN_GRID_INVALID", "EXACT_GRID_STATUS_PAIR_UNAVAILABLE", trace, executed_safe_trace=trace)
            pose = pose_tuple(read_odom(timeout_sec=args.input_timeout_sec))
            path_check = build_stair_moving_turn_entry_path(
                target, pose, pair[0], pair[1], args.robot_radius_m, require_arrival_region=not trace,
            )
            if not path_check.get("feasible"):
                return portal_normal_alignment_failure(args, "P2KG15_093_MOVING_TURN_PATH_INVALID", str(path_check.get("reason")), trace, path_check=path_check, executed_safe_trace=trace)
            selected = path_check["selected"]
            path = [tuple(point) for point in selected["path"]]
            frozen = target["frozen_geometry"]
            normal = frozen["portal_normal_odom"]
            norm = math.hypot(float(normal[0]), float(normal[1]))
            unit_normal = (float(normal[0]) / norm, float(normal[1]) / norm)
            normal_progress, tangent_progress = _stair_portal_frame(pose, frozen["portal_center_odom"], unit_normal)
            through_base = target_base_xy(target["P_through_odom"], pose)
            if normal_progress >= 0.0 and crossing_stamp is None:
                crossing_stamp = now_sim
            if crossing_stamp is not None and math.hypot(*through_base) <= STAIR_MOVING_TURN_GOAL_TOLERANCE_M:
                stop = publish_stop_at_door(args)
                return {
                    "reached": True, "final_decision": "P2KG15_FULL_FOOTPRINT_SAFE_P_THROUGH_REACHED",
                    "reason": "P_THROUGH_GOAL_REGION_REACHED_ZERO_STOP", "portal_crossing_stamp": crossing_stamp,
                    "terminal_zero": stop, "executed_safe_trace": trace, "final_pose": list(pose),
                }
            if last_progress is None or normal_progress > last_progress + 0.02:
                last_progress, last_progress_sim = normal_progress, now_sim
            elif now_sim - last_progress_sim > 3.0:
                return portal_normal_alignment_failure(args, "P2KG15_093_MOVING_TURN_PHYSICAL_RESPONSE_FAILURE", "FORWARD_PROGRESS_STALLED", trace, executed_safe_trace=trace)
            waypoint = next((point for point in path[1:] if math.hypot(*target_base_xy(point, pose)) >= 0.30), path[-1])
            waypoint_base = target_base_xy(waypoint, pose)
            raw_grid = dwa.grid_array(pair[0])
            blocked = dwa.inflate_obstacles(raw_grid, float(pair[0].info.resolution))
            linear_x, angular_z, dwa_detail = dwa.choose_dwa(
                pair[0], blocked, waypoint_base, through_base, math.hypot(*through_base),
                wall_heading_prior={"active": False},
            )
            if linear_x <= 0.0 and abs(angular_z) > 1e-6:
                return portal_normal_alignment_failure(args, "P2KG15_093_PLANNER_REQUIRES_PURE_YAW", "DWA_PURE_YAW_REJECTED", trace, dwa=dwa_detail, executed_safe_trace=trace)
            if linear_x <= 0.0 or abs(angular_z) < 0.01 or abs(angular_z / linear_x) > STAIR_MOVING_TURN_KAPPA_MAX_M_INV + 1e-9:
                return portal_normal_alignment_failure(args, "P2KG15_093_MOVING_TURN_PATH_INVALID", "DWA_FORWARD_YAW_ENVELOPE_REJECTED", trace, dwa=dwa_detail, executed_safe_trace=trace)
            try:
                imu = rospy.wait_for_message(args.follower_imu_topic, Imu, timeout=0.05)
                roll, pitch, _yaw = R29_euler_from_quat(imu.orientation)
                if abs(roll) > math.radians(15.0) or abs(pitch) > math.radians(15.0):
                    return portal_normal_alignment_failure(args, "P2KG15_093_BODY_STABILITY_ABORT", "ROLL_OR_PITCH_LIMIT_EXCEEDED", trace, executed_safe_trace=trace)
            except rospy.ROSException:
                roll = pitch = None
            cmd = Twist()
            cmd.linear.x = float(linear_x)
            cmd.angular.z = float(angular_z)
            if args.execute:
                pub.publish(cmd)
            nonzero_turn_count += 1
            if first_turn_yaw is None:
                first_turn_yaw = pose[2]
            if nonzero_turn_count >= 8 and abs(normalize_angle(pose[2] - first_turn_yaw)) < 0.01:
                return portal_normal_alignment_failure(args, "P2KG15_093_MOVING_TURN_PHYSICAL_RESPONSE_FAILURE", "FORWARD_YAW_COMMAND_WITHOUT_YAW_RESPONSE", trace, executed_safe_trace=trace)
            trace.append({
                "sim_stamp": now_sim, "pose_odom": list(pose), "grid_content_stamp": pair[1].get("grid_content_stamp"),
                "content_generation_id": pair[1].get("content_generation_id"), "portal_normal_progress_m": normal_progress,
                "portal_tangent_m": tangent_progress, "local_clearance": selected["swept_grid"],
                "cmd_linear_x": float(linear_x), "cmd_angular_z": float(angular_z),
                "cmd_kappa_m_inv": float(angular_z / linear_x), "roll_rad": roll, "pitch_rad": pitch,
            })
            rate.sleep()
        return portal_normal_alignment_failure(args, "P2KG15_093_MOVING_TURN_PATH_INVALID", "ROS_SHUTDOWN", trace, executed_safe_trace=trace)
    except Exception as exc:
        return portal_normal_alignment_failure(args, "P2KG15_093_MOVING_TURN_PATH_INVALID", f"MOVING_TURN_EXCEPTION:{type(exc).__name__}", trace, executed_safe_trace=trace)


def R29_euler_from_quat(q: Any) -> Tuple[float, float, float]:
    sinr = 2.0 * (float(q.w) * float(q.x) + float(q.y) * float(q.z))
    cosr = 1.0 - 2.0 * (float(q.x) * float(q.x) + float(q.y) * float(q.y))
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (float(q.w) * float(q.y) - float(q.z) * float(q.x))
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)), 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z)))
    return roll, pitch, yaw


def portal_normal_alignment_failure(
    args: argparse.Namespace,
    final_decision: str,
    reason: str,
    samples: Sequence[Dict[str, Any]],
    **details: Any,
) -> Dict[str, Any]:
    """Fail closed and record the same explicit zero-stop evidence as success."""
    stop = publish_stop_at_door(args) if args.execute else {"execute": False, "reason": "dry_run_no_stop_command"}
    return {
        "aligned": False,
        "final_decision": final_decision,
        "reason": reason,
        "samples": list(samples),
        "safe_stop": stop,
        **details,
    }


def evaluate_portal_normal_alignment(
    candidate: Any,
    target: Any,
    current_pose: Tuple[float, float, float],
    grid_msg: Any,
    status_payload: Any,
    robot_radius_m: float,
    selected_d_m: Any,
) -> Dict[str, Any]:
    """Bind a single room-facing yaw and prove that rotating at the current centre is safe.

    The local grid is expressed in the current base frame, so the robot centre is
    always (0, 0).  Its cell must be explicit free after the same static-radius
    inflation already used for P_pre; unknown and out-of-grid are fail-closed.
    """
    result: Dict[str, Any] = {
        "state": "PORTAL_NORMAL_ALIGNMENT_REJECTED",
        "alignment_authority": "ACTUAL_POSE_PORTAL_PLANE_CROSSING",
        "candidate_lifecycle": target.get("candidate_lifecycle") if isinstance(target, dict) else None,
        "linear_x_required": 0.0,
        "rotation_clearance_safe": False,
        "heading_error_rad": None,
        "desired_yaw_rad": None,
        "portal_plane_crossing_safe": False,
    }
    identity = target.get("portal_identity") if isinstance(target, dict) else None
    frozen = target.get("frozen_geometry") if isinstance(target, dict) else None
    identity_matches = bool(
        isinstance(candidate, dict) and isinstance(identity, dict)
        and candidate.get("portal_run_id") == identity.get("run_id")
        and candidate.get("portal_frame_sequence") == identity.get("frame_sequence")
        and candidate.get("portal_source_stamp") == identity.get("source_stamp")
        and candidate.get("portal_track_id") == identity.get("track_id")
        and candidate.get("side") == identity.get("side")
    )
    if not (
        isinstance(target, dict) and target.get("target_valid") is True
        and target.get("candidate_lifecycle") == "COMMITTED"
        and isinstance(frozen, dict) and frozen.get("geometry_valid") is True
        and identity_matches
    ):
        result["reason"] = "COMMITTED_FROZEN_PORTAL_CONTRACT_INVALID"
        return result
    centre = frozen.get("portal_center_odom")
    normal = frozen.get("portal_normal_odom")
    through = target.get("P_through_odom")
    if not (_finite_shape(centre, 2) and _finite_shape(normal, 2) and _finite_shape(through, 2) and finite_number(selected_d_m)):
        result["reason"] = "PORTAL_NORMAL_GEOMETRY_UNAVAILABLE"
        return result
    normal_norm = math.hypot(float(normal[0]), float(normal[1]))
    if normal_norm <= 0.0 or float(selected_d_m) <= 0.0:
        result["reason"] = "PORTAL_NORMAL_GEOMETRY_INVALID"
        return result
    unit_normal = [float(normal[0]) / normal_norm, float(normal[1]) / normal_norm]
    selected_pre = [
        float(centre[0]) - float(selected_d_m) * unit_normal[0],
        float(centre[1]) - float(selected_d_m) * unit_normal[1],
    ]
    pre_to_centre = (float(centre[0]) - selected_pre[0]) * unit_normal[0] + (float(centre[1]) - selected_pre[1]) * unit_normal[1]
    centre_to_through = (float(through[0]) - float(centre[0])) * unit_normal[0] + (float(through[1]) - float(centre[1])) * unit_normal[1]
    if pre_to_centre <= 0.0 or centre_to_through <= 0.0:
        result["reason"] = "PORTAL_NORMAL_DIRECTION_BINDING_INVALID"
        return result
    qualified, qualification_errors = qualified_for_navigation(grid_msg, status_payload)
    result["grid_qualified"] = bool(qualified)
    result["grid_qualification_errors"] = list(qualification_errors)
    if not qualified:
        result["reason"] = "FORMAL_LOCAL_GRID_UNQUALIFIED"
        return result
    try:
        grid = grid_array(grid_msg)
        centre_cell = local_xy_to_cell(0.0, 0.0, grid_msg)
        if centre_cell is None:
            result["reason"] = "ROBOT_CENTRE_OUTSIDE_FORMAL_LOCAL_GRID"
            return result
        inflated = p_pre_inflated_blocked(grid, robot_radius_m, float(grid_msg.info.resolution))
        x_index, y_index = centre_cell
        raw_value = int(grid[y_index, x_index])
        inflated_blocked = bool(inflated[y_index, x_index])
    except Exception as exc:
        result["reason"] = f"FORMAL_GRID_READ_FAILED:{type(exc).__name__}"
        return result
    portal_width = target.get("portal_width_m")
    if not finite_number(portal_width):
        result["reason"] = "PORTAL_APERTURE_GEOMETRY_UNAVAILABLE"
        return result
    crossing = portal_plane_crossing_geometry(current_pose, centre, unit_normal, float(portal_width), robot_radius_m)
    result.update({
        "portal_identity": identity,
        "frozen_portal_center_odom": [float(centre[0]), float(centre[1])],
        "frozen_portal_normal_odom": unit_normal,
        "selected_P_pre_odom": selected_pre,
        "selected_d_m": float(selected_d_m),
        "normal_direction_check": {"P_pre_to_portal_dot": pre_to_centre, "portal_to_P_through_dot": centre_to_through},
        "rotation_centre_cell": list(centre_cell),
        "rotation_centre_raw_value": raw_value,
        "rotation_centre_inflated_blocked": inflated_blocked,
        "static_radius_m": float(robot_radius_m),
        "portal_width_m": float(portal_width),
        **crossing,
    })
    if raw_value != 0 or inflated_blocked:
        result["reason"] = "P_PRE_ROTATION_CLEARANCE"
        return result
    if crossing["portal_plane_crossing_reason"] == "PORTAL_APERTURE_INSUFFICIENT_FOR_STATIC_FOOTPRINT":
        result["reason"] = "PORTAL_APERTURE_INSUFFICIENT_FOR_STATIC_FOOTPRINT"
        return result
    if crossing["portal_plane_crossing_reason"] == "P_PRE_PORTAL_PLANE_OVERSHOOT":
        result["reason"] = "P_PRE_PORTAL_PLANE_OVERSHOOT"
        return result
    result.update({"state": "PORTAL_NORMAL_ALIGNMENT_READY", "reason": str(crossing["portal_plane_crossing_reason"]), "rotation_clearance_safe": True})
    return result


def evaluate_p_through_handoff_feasibility(
    candidate: Any,
    target: Any,
    current_pose: Tuple[float, float, float],
    grid_msg: Any,
    status_payload: Any,
    robot_radius_m: float,
) -> Dict[str, Any]:
    """Read-only P_through shadow gate using the unchanged Block-4/A*/DWA helpers.

    This function never creates a publisher or sends a command.  It rejects a
    nearest-free fallback unless the returned planner path actually enters the
    room-facing side of the frozen portal plane and remains footprint-safe.
    """
    result: Dict[str, Any] = {
        "state": "P_THROUGH_HANDOFF_NOT_FEASIBLE",
        "feasible": False,
        "reason": None,
        "shadow_only": True,
        "motion_commanded": False,
        "grid_qualified": False,
        "target_in_grid": False,
        "target_raw_free": False,
        "target_inflated_blocked": None,
        "astar_path": [],
        "astar_crosses_portal": False,
        "astar_endpoint_room_facing": False,
        "astar_endpoint_in_target_block": False,
        "dwa_first_command": None,
        "swept_footprint_safe": False,
    }
    identity = target.get("portal_identity") if isinstance(target, dict) else None
    frozen = target.get("frozen_geometry") if isinstance(target, dict) else None
    identity_matches = bool(
        isinstance(candidate, dict) and isinstance(identity, dict)
        and candidate.get("portal_run_id") == identity.get("run_id")
        and candidate.get("portal_frame_sequence") == identity.get("frame_sequence")
        and candidate.get("portal_source_stamp") == identity.get("source_stamp")
        and candidate.get("portal_track_id") == identity.get("track_id")
        and candidate.get("side") == identity.get("side")
    )
    through = target.get("P_through_odom") if isinstance(target, dict) else None
    centre = frozen.get("portal_center_odom") if isinstance(frozen, dict) else None
    normal = frozen.get("portal_normal_odom") if isinstance(frozen, dict) else None
    portal_width = target.get("portal_width_m") if isinstance(target, dict) else None
    if not (
        isinstance(target, dict) and target.get("target_valid") is True
        and target.get("candidate_lifecycle") == "COMMITTED"
        and isinstance(frozen, dict) and frozen.get("geometry_valid") is True
        and identity_matches
        and _finite_shape(through, 2) and _finite_shape(centre, 2) and _finite_shape(normal, 2)
        and finite_number(portal_width) and float(portal_width) > 0.0
    ):
        result["reason"] = "COMMITTED_FROZEN_P_THROUGH_CONTRACT_INVALID"
        return result
    if not finite_number(robot_radius_m) or not math.isclose(
        float(robot_radius_m), STATIC_PLANNING_FOOTPRINT_RADIUS_M, rel_tol=0.0, abs_tol=1e-12,
    ):
        result["reason"] = "STATIC_FOOTPRINT_RADIUS_MISMATCH"
        return result
    normal_norm = math.hypot(float(normal[0]), float(normal[1]))
    if normal_norm <= 0.0 or float(robot_radius_m) <= 0.0:
        result["reason"] = "P_THROUGH_GEOMETRY_OR_RADIUS_INVALID"
        return result
    unit_normal = (float(normal[0]) / normal_norm, float(normal[1]) / normal_norm)
    target_base = target_base_xy(through, current_pose)
    result.update({
        "portal_identity": identity,
        "frozen_portal_center_odom": [float(centre[0]), float(centre[1])],
        "frozen_portal_normal_odom": [unit_normal[0], unit_normal[1]],
        "P_through_odom": [float(through[0]), float(through[1])],
        "P_through_base": [float(target_base[0]), float(target_base[1])],
        "static_radius_m": float(robot_radius_m),
    })
    qualified, qualification_errors = qualified_for_navigation(grid_msg, status_payload)
    result["grid_qualified"] = bool(qualified)
    result["grid_qualification_errors"] = list(qualification_errors)
    if not qualified:
        result["reason"] = "FORMAL_LOCAL_GRID_UNQUALIFIED"
        return result
    if target_base[0] <= 0.0:
        result["reason"] = "P_THROUGH_NOT_IN_FRONT"
        return result
    try:
        # Deliberately avoid BlockAStarDwaRunner.__init__: it owns ROS
        # subscribers/publishers.  The following existing pure helpers are
        # used with their production defaults and no changed planner policy.
        runner = object.__new__(BlockAStarDwaRunner)
        runner.args = build_block_astar_dwa_arg_parser().parse_args([])
        runner.args.robot_radius_m = float(robot_radius_m)
        runner.prev_cmd = (0.0, 0.0)
        raw_grid = runner.grid_array(grid_msg)
        resolution_m = float(grid_msg.info.resolution)
        occupied_inflated = runner.occupied_inflated_mask(raw_grid, resolution_m)
        blocked = runner.inflate_obstacles(raw_grid, resolution_m)
        start_cell = runner.local_xy_to_cell(0.0, 0.0, grid_msg)
        target_cell = runner.local_xy_to_cell(float(target_base[0]), float(target_base[1]), grid_msg)
        result["start_cell"] = list(start_cell) if start_cell is not None else None
        result["target_cell"] = list(target_cell) if target_cell is not None else None
        if start_cell is None:
            result["reason"] = "ROBOT_CENTRE_OUTSIDE_FORMAL_LOCAL_GRID"
            return result
        if target_cell is None:
            result["reason"] = "P_THROUGH_OUTSIDE_FORMAL_LOCAL_GRID"
            return result
        tx, ty = target_cell
        result["target_in_grid"] = True
        result["target_raw_value"] = int(raw_grid[ty, tx])
        result["target_raw_free"] = bool(raw_grid[ty, tx] == 0)
        result["target_inflated_blocked"] = bool(blocked[ty, tx])
        if not result["target_raw_free"] or result["target_inflated_blocked"]:
            result["reason"] = "P_THROUGH_TARGET_NOT_FORMALLY_ADMISSIBLE"
            return result
        planning_blocked, start_clearance = runner.apply_start_footprint_clearance(
            grid_msg, raw_grid, blocked, occupied_inflated, qualification_passed=True,
        )
        result["start_footprint_clearance"] = start_clearance
        raw_path = runner.block_astar(planning_blocked, start_cell, target_cell, grid_msg=grid_msg)
        path = runner.smooth_path(raw_path)
        path_base = [(0.0, 0.0)] + [runner.cell_to_local_xy(cell, grid_msg) for cell in path]
        result["astar_path"] = [[float(x), float(y)] for x, y in path_base]
        block = max(1, int(runner.args.block_size_cells))
        target_block = [target_cell[0] // block, target_cell[1] // block]
        endpoint_block = [path[-1][0] // block, path[-1][1] // block] if path else None
        result["target_block"] = target_block
        result["astar_endpoint_block"] = endpoint_block
        result["astar_endpoint_in_target_block"] = bool(endpoint_block == target_block)
        if len(path_base) < 2:
            result["reason"] = "P_THROUGH_ASTAR_NO_PATH"
            return result

        def base_to_odom(point: Tuple[float, float]) -> Tuple[float, float]:
            yaw = float(current_pose[2])
            c, s = math.cos(yaw), math.sin(yaw)
            return (
                float(current_pose[0]) + c * point[0] - s * point[1],
                float(current_pose[1]) + s * point[0] + c * point[1],
            )

        tangent = (-unit_normal[1], unit_normal[0])
        half_aperture = float(portal_width) / 2.0 - float(robot_radius_m)
        result["portal_aperture_half_width_H_m"] = half_aperture
        if half_aperture <= 0.0:
            result["reason"] = "PORTAL_APERTURE_INSUFFICIENT_FOR_STATIC_FOOTPRINT"
            return result
        path_odom = [base_to_odom(point) for point in path_base]
        crossing: Optional[Dict[str, Any]] = None
        for index, (a, b) in enumerate(zip(path_odom, path_odom[1:])):
            side_a = (a[0] - float(centre[0])) * unit_normal[0] + (a[1] - float(centre[1])) * unit_normal[1]
            side_b = (b[0] - float(centre[0])) * unit_normal[0] + (b[1] - float(centre[1])) * unit_normal[1]
            if side_a < 0.0 and side_b >= 0.0 and side_b > side_a:
                ratio = -side_a / (side_b - side_a)
                point = (a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1]))
                tangent_offset = (point[0] - float(centre[0])) * tangent[0] + (point[1] - float(centre[1])) * tangent[1]
                crossing = {
                    "segment_index": index,
                    "outside_to_room": True,
                    "tangent_offset_m": tangent_offset,
                    "aperture_residual_m": half_aperture - abs(tangent_offset),
                }
                break
        endpoint = path_odom[-1]
        endpoint_side = (endpoint[0] - float(centre[0])) * unit_normal[0] + (endpoint[1] - float(centre[1])) * unit_normal[1]
        result["portal_plane_crossing"] = crossing
        result["astar_crosses_portal"] = crossing is not None
        result["astar_endpoint_room_facing"] = bool(endpoint_side > 0.0)
        if not result["astar_endpoint_in_target_block"]:
            result["reason"] = "A_STAR_NEAREST_FREE_FALLBACK_NOT_P_THROUGH"
            return result
        if crossing is None or crossing["aperture_residual_m"] < 0.0 or not result["astar_endpoint_room_facing"]:
            result["reason"] = "A_STAR_PATH_DOES_NOT_CROSS_PORTAL"
            return result

        swept_samples: List[Dict[str, Any]] = []
        sample_step = max(1e-6, min(resolution_m / 2.0, 0.025))
        swept_safe = True
        for a, b in zip(path_base, path_base[1:]):
            distance = math.hypot(b[0] - a[0], b[1] - a[1])
            for step in range(max(1, int(math.ceil(distance / sample_step))) + 1):
                ratio = float(step) / float(max(1, int(math.ceil(distance / sample_step))))
                point = (a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1]))
                cell = runner.local_xy_to_cell(point[0], point[1], grid_msg)
                entry: Dict[str, Any] = {"point_base": [point[0], point[1]], "cell": list(cell) if cell is not None else None, "safe": False}
                if cell is None:
                    swept_safe = False
                    entry["reason"] = "OUT_OF_GRID"
                else:
                    x_index, y_index = cell
                    if int(raw_grid[y_index, x_index]) != 0 or bool(planning_blocked[y_index, x_index]):
                        swept_safe = False
                        entry["reason"] = "INFLATED_OR_UNKNOWN_BLOCKED"
                    else:
                        odom_point = base_to_odom(point)
                        normal_side = (odom_point[0] - float(centre[0])) * unit_normal[0] + (odom_point[1] - float(centre[1])) * unit_normal[1]
                        tangent_offset = (odom_point[0] - float(centre[0])) * tangent[0] + (odom_point[1] - float(centre[1])) * tangent[1]
                        if normal_side >= 0.0 and abs(tangent_offset) > half_aperture:
                            swept_safe = False
                            entry["reason"] = "PORTAL_JAMB_CLEARANCE_VIOLATION"
                        else:
                            entry["safe"] = True
                swept_samples.append(entry)
                if not swept_safe:
                    break
            if not swept_safe:
                break
        result["swept_footprint_safe"] = swept_safe
        result["swept_sample_count"] = len(swept_samples)
        result["swept_first_failure"] = next((entry for entry in swept_samples if not entry["safe"]), None)
        if not swept_safe:
            result["reason"] = "P_THROUGH_SWEPT_STATIC_FOOTPRINT_UNSAFE"
            return result
        waypoint = runner.select_lookahead_waypoint(path, grid_msg)
        linear_x, angular_z, dwa = runner.choose_dwa(
            grid_msg, planning_blocked, waypoint[:2], target_base, math.hypot(*target_base),
        )
        result["dwa_first_command"] = {
            "linear_x": float(linear_x), "angular_z": float(angular_z),
            "blocked": bool(dwa.get("blocked")), "sample_count": int(dwa.get("sample_count", 0)),
        }
        if bool(dwa.get("blocked")) or int(dwa.get("sample_count", 0)) <= 0:
            result["reason"] = "P_THROUGH_DWA_SHADOW_BLOCKED"
            return result
    except Exception as exc:
        result["reason"] = f"P_THROUGH_SHADOW_PLANNER_EXCEPTION:{type(exc).__name__}"
        return result
    result.update({"state": "P_THROUGH_HANDOFF_FEASIBLE", "feasible": True, "reason": "P_THROUGH_HANDOFF_FEASIBLE"})
    return result


def execute_portal_normal_alignment(
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    target: Dict[str, Any],
    formal_grid_status_pair: FormalGridStatusPairSubscriber,
    selected_d_m: float,
) -> Dict[str, Any]:
    """The sole 093 motion authority: zero-linear closed-loop portal-normal alignment."""
    samples: List[Dict[str, Any]] = []
    pub = rospy.Publisher(args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic, Twist, queue_size=2)
    rospy.sleep(0.1)
    rate = rospy.Rate(float(args.portal_normal_alignment_rate_hz))
    start_sim = rospy.Time.now()
    consecutive_aligned = 0
    last_feasibility_check_sim_sec: Optional[float] = None
    last_feasibility: Optional[Dict[str, Any]] = None
    commanded_nonzero = 0
    first_command_yaw: Optional[float] = None
    try:
        while not rospy.is_shutdown():
            pair = formal_grid_status_pair.matching_pair()
            pose = pose_tuple(read_odom())
            if pair is None:
                return portal_normal_alignment_failure(args, "P2KG15_093_P_PRE_ROTATION_CLEARANCE", "EXACT_GRID_STATUS_PAIR_UNAVAILABLE", samples)
            check = evaluate_portal_normal_alignment(candidate, target, pose, pair[0], pair[1], args.robot_radius_m, selected_d_m)
            if check.get("state") != "PORTAL_NORMAL_ALIGNMENT_READY":
                decision = "P2KG15_093_P_PRE_PORTAL_PLANE_OVERSHOOT" if check.get("reason") == "P_PRE_PORTAL_PLANE_OVERSHOOT" else "P2KG15_093_P_PRE_ROTATION_CLEARANCE"
                return portal_normal_alignment_failure(args, decision, str(check.get("reason")), samples, rotation_clearance=check)
            error = float(check["heading_error_rad"])
            command = portal_normal_alignment_command(error, args.portal_normal_alignment_max_angular_z, args.portal_normal_alignment_gain)
            r17_safe = bool(check["portal_plane_crossing_safe"])
            handoff_ready = False
            feasibility_checked = False
            if r17_safe:
                now_sim_sec = float(rospy.Time.now().to_sec())
                if (
                    last_feasibility_check_sim_sec is None
                    or now_sim_sec - last_feasibility_check_sim_sec >= float(args.portal_normal_alignment_feasibility_min_interval_sec)
                ):
                    last_feasibility = evaluate_p_through_handoff_feasibility(
                        candidate, target, pose, pair[0], pair[1], args.robot_radius_m,
                    )
                    last_feasibility_check_sim_sec = now_sim_sec
                    feasibility_checked = True
                    if last_feasibility.get("feasible"):
                        consecutive_aligned += 1
                    else:
                        consecutive_aligned = 0
                if last_feasibility is not None and last_feasibility.get("feasible"):
                    handoff_ready = True
            else:
                consecutive_aligned = 0
                last_feasibility = None
                last_feasibility_check_sim_sec = None
            if handoff_ready:
                command = {"linear_x": 0.0, "angular_z": 0.0}
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = float(command["angular_z"])
            if args.execute:
                pub.publish(msg)
            if abs(msg.angular.z) > 0.0:
                commanded_nonzero += 1
                if first_command_yaw is None:
                    first_command_yaw = float(pose[2])
            sample = {
                "robot_yaw_rad": float(pose[2]), "desired_yaw_rad": check["desired_yaw_rad"],
                "heading_error_rad": error,
                "portal_normal_theta_rad": check["portal_normal_theta_rad"],
                "portal_normal_distance_D_m": check["portal_normal_distance_D_m"],
                "portal_tangent_offset_s_m": check["portal_tangent_offset_s_m"],
                "portal_plane_crossing_s_m": check["portal_plane_crossing_s_m"],
                "portal_aperture_half_width_H_m": check["portal_aperture_half_width_H_m"],
                "portal_plane_remaining_margin_m": check["portal_plane_remaining_margin_m"],
                "portal_plane_crossing_safe": bool(check["portal_plane_crossing_safe"]),
                "cmd_linear_x": 0.0, "cmd_angular_z": float(msg.angular.z),
                "rotation_clearance": check,
                "r17_safe": r17_safe,
                "p_through_feasibility_checked": feasibility_checked,
                "p_through_handoff": last_feasibility if r17_safe else None,
                "aligned_sample": handoff_ready,
            }
            samples.append(sample)
            if consecutive_aligned >= int(args.portal_normal_alignment_required_consecutive_samples):
                stop = publish_stop_at_door(args) if args.execute else {"execute": False, "reason": "dry_run_no_stop_command"}
                return {
                    "aligned": True,
                    "final_decision": "P2KG15_093_P_THROUGH_HANDOFF_READY",
                    "reason": "P_THROUGH_HANDOFF_FEASIBLE",
                    "samples": samples,
                    "safe_stop": stop,
                }
            if commanded_nonzero >= int(args.portal_normal_alignment_no_response_samples) and first_command_yaw is not None and abs(normalize_angle(float(pose[2]) - first_command_yaw)) < float(args.portal_normal_alignment_yaw_response_epsilon_rad):
                return portal_normal_alignment_failure(args, "P2KG15_093_ANGULAR_ACTUATION_INEFFECTIVE", "ANGULAR_COMMAND_WITHOUT_YAW_RESPONSE", samples)
            elapsed = max(0.0, float((rospy.Time.now() - start_sim).to_sec()))
            if elapsed >= float(args.portal_normal_alignment_max_sim_sec):
                return portal_normal_alignment_failure(args, "P2KG15_093_HEADING_CONTROLLER_NONCONVERGENCE", "HEADING_ERROR_DID_NOT_CONVERGE_WITHIN_BOUND", samples)
            rate.sleep()
        return portal_normal_alignment_failure(args, "P2KG15_093_HEADING_CONTROLLER_NONCONVERGENCE", "ROS_SHUTDOWN", samples)
    except Exception as exc:
        return portal_normal_alignment_failure(
            args,
            "P2KG15_093_ALIGNMENT_EXCEPTION",
            f"PORTAL_NORMAL_ALIGNMENT_EXCEPTION:{type(exc).__name__}",
            samples,
        )


def grid_array(msg: OccupancyGrid) -> np.ndarray:
    errors = validate_grid_metadata(msg)
    if errors:
        raise ValueError("local_grid_contract_invalid:" + ",".join(errors))
    width, height = int(msg.info.width), int(msg.info.height)
    result = np.empty((height, width), dtype=np.int16)
    for y_index in range(height):
        for x_index in range(width):
            result[y_index, x_index] = msg.data[flatten_index(x_index, y_index, width, height)]
    return result


def local_xy_to_cell(x: float, y: float, msg: OccupancyGrid) -> Optional[Tuple[int, int]]:
    return metric_to_cell(x, y, grid_metadata(msg))


def inflate_blocked(grid: np.ndarray, robot_radius_m: float, resolution_m: float, treat_unknown_as_blocked: bool = True) -> np.ndarray:
    # Formal motion accepts only explicit free cells, independent of a caller flag.
    blocked = grid != 0
    inflated = blocked.copy()
    radius = max(0, int(math.ceil(robot_radius_m / max(resolution_m, 1e-6))))
    y_indices, x_indices = np.where(blocked)
    for y_index, x_index in zip(y_indices.tolist(), x_indices.tolist()):
        y0 = max(0, y_index - radius)
        y1 = min(grid.shape[0], y_index + radius + 1)
        x0 = max(0, x_index - radius)
        x1 = min(grid.shape[1], x_index + radius + 1)
        inflated[y0:y1, x0:x1] = True
    return inflated


def p_pre_inflated_blocked(grid: np.ndarray, robot_radius_m: float, resolution_m: float) -> np.ndarray:
    """P_pre formal mask: shared occupied geometry plus unchanged unknown block."""
    inflated = occupied_euclidean_inflated_mask(grid, robot_radius_m, resolution_m)
    # Unknown remains blocked for P_pre, but is not an inflation seed.
    inflated |= grid == -1
    return inflated


def grid_window_metrics(grid: np.ndarray, cell: Optional[Tuple[int, int]], radius_cells: int) -> Dict[str, Any]:
    if cell is None:
        return {"in_grid": False, "cell": None}
    x_index, y_index = cell
    if x_index < 0 or y_index < 0 or x_index >= grid.shape[1] or y_index >= grid.shape[0]:
        return {"in_grid": False, "cell": [x_index, y_index]}
    y0 = max(0, y_index - radius_cells)
    y1 = min(grid.shape[0], y_index + radius_cells + 1)
    x0 = max(0, x_index - radius_cells)
    x1 = min(grid.shape[1], x_index + radius_cells + 1)
    window = grid[y0:y1, x0:x1]
    count = int(window.size)
    free = int((window == 0).sum())
    occupied = int((window == 100).sum())
    unknown = int((window == -1).sum())
    return {
        "in_grid": True,
        "cell": [x_index, y_index],
        "raw_cell_value": int(grid[y_index, x_index]),
        "cell_count": count,
        "free_count": free,
        "occupied_count": occupied,
        "unknown_count": unknown,
        "free_ratio": float(free / count) if count else None,
        "blocked_ratio": float((occupied + unknown) / count) if count else None,
    }


def grid_line_pass(
    grid: np.ndarray,
    inflated: np.ndarray,
    msg: OccupancyGrid,
    end_xy: Tuple[float, float],
    sample_step_m: float,
) -> Dict[str, Any]:
    dist = math.hypot(float(end_xy[0]), float(end_xy[1]))
    steps = max(1, int(math.ceil(dist / max(float(sample_step_m), 1e-3))))
    blocked_count = 0
    out_of_grid_count = 0
    first_blocked: Optional[Dict[str, Any]] = None
    for idx in range(steps + 1):
        t = idx / steps
        x = float(end_xy[0]) * t
        y = float(end_xy[1]) * t
        cell = local_xy_to_cell(x, y, msg)
        in_grid = cell is not None
        blocked = bool(in_grid and inflated[cell[1], cell[0]])
        if not in_grid:
            out_of_grid_count += 1
        if blocked or not in_grid:
            blocked_count += 1
            if first_blocked is None:
                first_blocked = {
                    "sample_index": idx,
                    "x_base": x,
                    "y_base": y,
                    "cell": list(cell) if cell is not None else None,
                    "in_grid": in_grid,
                    "value": int(grid[cell[1], cell[0]]) if in_grid else None,
                    "inflated_blocked": blocked,
                }
    return {
        "distance_m": float(dist),
        "sample_count": steps + 1,
        "blocked_or_out_of_grid_count": int(blocked_count),
        "out_of_grid_count": int(out_of_grid_count),
        "first_blocked": first_blocked,
        "pass": blocked_count == 0,
    }


def candidate_depths(max_depth_m: float, min_depth_m: float, step_m: float) -> List[float]:
    max_depth = max(0.0, float(max_depth_m))
    min_depth = max(0.0, min(float(min_depth_m), max_depth))
    step = max(float(step_m), 0.02)
    depths: List[float] = []
    depth = max_depth
    while depth >= min_depth - 1e-9:
        depths.append(round(depth, 4))
        depth -= step
    if not any(abs(d) <= 1e-9 for d in depths):
        depths.append(0.0)
    return depths


def select_grid_validated_doorway_target(
    entry: Dict[str, Any],
    doorway: Dict[str, Any],
    args: argparse.Namespace,
    *,
    commit_depth_m: float,
) -> Tuple[List[float], Dict[str, Any]]:
    if not finite_number(entry.get("yaw")):
        return [float(entry["x"]), float(entry["y"])], {"grid_validation_pass": False, "reason": "doorway_entry_yaw_unavailable"}

    pose = pose_tuple(read_odom(args.input_timeout_sec))
    grid_msg = read_grid(args.input_timeout_sec)
    status_payload = read_traversability_status(args.input_timeout_sec)
    grid_qualified, qualification_errors = qualified_for_navigation(grid_msg, status_payload)
    if not grid_qualified:
        return [float(entry["x"]), float(entry["y"])], {
            "grid_validation_pass": False,
            "reason": "local_grid_contract_not_qualified",
            "local_grid_contract_rejection": qualification_errors,
            "fallback": "no_commit_depth_selected",
        }
    grid = grid_array(grid_msg)
    inflated = inflate_blocked(grid, args.robot_radius_m, float(grid_msg.info.resolution), True)
    yaw = float(entry["yaw"])
    evaluations: List[Dict[str, Any]] = []
    best_rejected: Optional[Dict[str, Any]] = None
    for depth in candidate_depths(commit_depth_m, args.enter_room_commit_min_depth_m, args.enter_room_commit_depth_step_m):
        target_xy = [float(entry["x"]) + math.cos(yaw) * depth, float(entry["y"]) + math.sin(yaw) * depth]
        base_xy = target_base_xy(target_xy, pose)
        cell = local_xy_to_cell(base_xy[0], base_xy[1], grid_msg)
        metrics = grid_window_metrics(grid, cell, args.enter_room_commit_grid_window_radius_cells)
        line = grid_line_pass(grid, inflated, grid_msg, base_xy, args.enter_room_commit_line_sample_step_m)
        inflated_blocked = None
        if metrics.get("in_grid"):
            x_index, y_index = cell
            inflated_blocked = bool(inflated[y_index, x_index])
        blocked_ratio = metrics.get("blocked_ratio")
        pass_candidate = bool(
            base_xy[0] >= args.enter_room_commit_min_target_x_base_m
            and metrics.get("in_grid")
            and metrics.get("raw_cell_value") == 0
            and inflated_blocked is False
            and finite_number(blocked_ratio)
            and float(blocked_ratio) <= args.enter_room_commit_max_window_blocked_ratio
            and line.get("pass")
        )
        item = {
            "depth_m": float(depth),
            "target_xy_team_livox_odom": target_xy,
            "target_base_xy": [float(base_xy[0]), float(base_xy[1])],
            "target_cell_metrics": metrics,
            "target_inflated_blocked": inflated_blocked,
            "line_of_travel_check": line,
            "pass": pass_candidate,
        }
        evaluations.append(item)
        if pass_candidate:
            return target_xy, {
                "grid_validation_pass": True,
                "selected_depth_m": float(depth),
                "candidate_evaluations": evaluations,
                "grid_frame_id": grid_msg.header.frame_id,
                "grid_header_stamp_sec": float(grid_msg.header.stamp.to_sec()),
                "matched_pose_x_y_yaw": list(pose),
            }
        best_rejected = item

    return [float(entry["x"]), float(entry["y"])], {
        "grid_validation_pass": False,
        "reason": "no_grid_valid_doorway_commit_candidate",
        "fallback": "doorway_entry_pose_without_commit_depth",
        "candidate_evaluations": evaluations,
        "best_rejected_candidate": best_rejected,
        "grid_frame_id": grid_msg.header.frame_id,
        "grid_header_stamp_sec": float(grid_msg.header.stamp.to_sec()),
        "matched_pose_x_y_yaw": list(pose),
    }


def anchor_metrics(anchor: Dict[str, Any], pose: Tuple[float, float, float]) -> Dict[str, Any]:
    heading = float(anchor["heading_rad"])
    dx = pose[0] - float(anchor["x"])
    dy = pose[1] - float(anchor["y"])
    along = dx * math.cos(heading) + dy * math.sin(heading)
    lateral = -dx * math.sin(heading) + dy * math.cos(heading)
    yaw_error = math.atan2(math.sin(pose[2] - heading), math.cos(pose[2] - heading))
    return {
        "anchor_progress_m": along,
        "anchor_lateral_error_m": lateral,
        "anchor_abs_lateral_error_m": abs(lateral),
        "anchor_yaw_error_rad": yaw_error,
        "anchor_abs_yaw_error_rad": abs(yaw_error),
    }


def effective_room_zone_progress_m(
    anchor: Dict[str, Any],
    current_anchor_progress_m: Optional[float],
) -> Optional[float]:
    """Keep room-zone corridor projection continuous across control handoff.

    ``anchor_progress_m`` remains the current control-anchor geometry.  Only
    the room-zone authority adds the handoff-local offset retained on the new
    control anchor; bootstrap anchors intentionally default to zero.
    """
    if not finite_number(current_anchor_progress_m):
        return None
    offset_m = anchor.get("room_zone_progress_offset_m", 0.0)
    if not finite_number(offset_m):
        return None
    return float(offset_m) + float(current_anchor_progress_m)


def portal_corridor_progress_m(portal_center_odom: Sequence[float], corridor_anchor: Dict[str, Any]) -> Optional[float]:
    """Project a frozen Portal centre into the existing initial-corridor frame."""
    if not (_finite_shape(portal_center_odom, 2) and finite_number(corridor_anchor.get("x"))
            and finite_number(corridor_anchor.get("y")) and finite_number(corridor_anchor.get("heading_rad"))):
        return None
    metrics = anchor_metrics(
        corridor_anchor,
        (float(portal_center_odom[0]), float(portal_center_odom[1]), float(corridor_anchor["heading_rad"])),
    )
    return float(metrics["anchor_progress_m"])


def same_physical_portal(
    fresh_portal_center_odom: Sequence[float],
    stored_record: Dict[str, Any],
    corridor_anchor: Dict[str, Any],
) -> bool:
    """Deterministic V1 physical-Portal match using only progress and centre."""
    fresh_progress = portal_corridor_progress_m(fresh_portal_center_odom, corridor_anchor)
    stored_center = stored_record.get("portal_center_odom") if isinstance(stored_record, dict) else None
    stored_progress = stored_record.get("corridor_progress_m") if isinstance(stored_record, dict) else None
    if not (
        fresh_progress is not None
        and _finite_shape(stored_center, 2)
        and finite_number(stored_progress)
    ):
        return False
    progress_difference = abs(fresh_progress - float(stored_progress))
    centre_distance = math.hypot(
        float(fresh_portal_center_odom[0]) - float(stored_center[0]),
        float(fresh_portal_center_odom[1]) - float(stored_center[1]),
    )
    return progress_difference <= VISITED_PORTAL_IDENTITY_TOLERANCE_M and centre_distance <= VISITED_PORTAL_IDENTITY_TOLERANCE_M


def record_entered_visited_portal(
    visited_portals: List[Dict[str, Any]],
    fresh_portal_center_odom: Sequence[float],
    corridor_anchor: Dict[str, Any],
) -> Dict[str, Any]:
    """Passively mark the uniquely matched frozen Portal as entered at P_through."""
    progress = portal_corridor_progress_m(fresh_portal_center_odom, corridor_anchor)
    if progress is None or not _finite_shape(fresh_portal_center_odom, 2):
        return {"state": "VISITED_PORTAL_IDENTITY_INVALID_GEOMETRY", "record": None}
    matches = [
        record for record in visited_portals
        if same_physical_portal(fresh_portal_center_odom, record, corridor_anchor)
    ]
    if len(matches) > 1:
        return {"state": "VISITED_PORTAL_IDENTITY_AMBIGUOUS_MATCH", "record": None}
    if len(matches) == 1:
        record = matches[0]
        record["entered"] = True
        return {"state": "VISITED_PORTAL_MATCHED_EXISTING", "record": record}
    record = {
        "corridor_progress_m": float(progress),
        "portal_center_odom": [float(fresh_portal_center_odom[0]), float(fresh_portal_center_odom[1])],
        "entered": True,
        "completed": False,
    }
    visited_portals.append(record)
    return {"state": "VISITED_PORTAL_CREATED", "record": record}


def mark_visited_portal_completed(entered_portal_record: Dict[str, Any]) -> None:
    """Mark a record complete only after the existing door-return success result."""
    entered_portal_record["completed"] = True


def portal_matches_entered_or_completed_visit(
    fresh_portal_center_odom: Sequence[float],
    visited_portals: Sequence[Dict[str, Any]],
    corridor_anchor: Dict[str, Any],
    matched_record_details: Optional[Dict[str, Any]] = None,
) -> bool:
    """Exclude only durable records that were physically entered or completed."""
    for index, record in enumerate(visited_portals):
        if not isinstance(record, dict):
            continue
        if (
            bool(record.get("entered") or record.get("completed"))
            and same_physical_portal(fresh_portal_center_odom, record, corridor_anchor)
        ):
            if isinstance(matched_record_details, dict):
                matched_record_details.update({
                    "matched_record_index": int(index),
                    "matched_record_entered": bool(record.get("entered")),
                    "matched_record_completed": bool(record.get("completed")),
                })
            return True
    return False


def next_portal_dispatch_decision(
    fresh_prepared_candidates: Sequence[Dict[str, Any]],
    visited_portals: Sequence[Dict[str, Any]],
    returned_portal_record: Dict[str, Any],
    corridor_anchor: Dict[str, Any],
    candidate_accounting: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select one fresh, unvisited, same-station Portal after room return.

    Candidates are prepared from the current legal effect frame by the caller.
    This helper is pure: it neither commits an authority nor controls motion.
    """
    returned_progress = returned_portal_record.get("corridor_progress_m") if isinstance(returned_portal_record, dict) else None
    accounting = candidate_accounting if isinstance(candidate_accounting, dict) else None
    if accounting is not None:
        accounting.setdefault("prepared_candidates", [])
        accounting.setdefault("total_prepared_candidates", len(fresh_prepared_candidates))
        accounting.setdefault("eligible_current_station_candidates", 0)
        accounting.setdefault("selected_index", None)
    if not finite_number(returned_progress):
        if accounting is not None:
            accounting["final_branch"] = "FOLLOW_CORRIDOR_DEEPER"
        return {
            "state": "FOLLOW_CORRIDOR_DEEPER",
            "reason": "RETURNED_PORTAL_CONTEXT_UNAVAILABLE",
            "candidate": None,
            "candidate_accounting": accounting,
        }
    eligible: List[Tuple[Dict[str, Any], int, Dict[str, Any]]] = []
    for index, prepared in enumerate(fresh_prepared_candidates):
        candidate = prepared.get("candidate") if isinstance(prepared, dict) else None
        centre = prepared.get("portal_center_odom") if isinstance(prepared, dict) else None
        if not (isinstance(candidate, dict) and _finite_shape(centre, 2)):
            continue
        candidate_entry = {
            "index": int(index),
            "side": candidate.get("side"),
            "track_id": candidate.get("portal_track_id"),
            "portal_center_odom": [float(centre[0]), float(centre[1])],
            "corridor_progress_m": None,
            "visited_match": False,
            "matched_record_index": None,
            "matched_record_entered": None,
            "matched_record_completed": None,
            "progress_delta_to_returned_portal_m": None,
            "current_station_passed": None,
            "disposition": None,
        }
        matched_record_details: Dict[str, Any] = {}
        visited_match = portal_matches_entered_or_completed_visit(
            centre,
            visited_portals,
            corridor_anchor,
            matched_record_details,
        )
        candidate_entry["visited_match"] = bool(visited_match)
        candidate_entry.update(matched_record_details)
        if visited_match:
            candidate_entry["disposition"] = "EXCLUDED_VISITED"
            if accounting is not None:
                accounting["prepared_candidates"].append(candidate_entry)
            continue
        progress = portal_corridor_progress_m(centre, corridor_anchor)
        candidate_entry["corridor_progress_m"] = float(progress) if progress is not None else None
        progress_delta = (
            abs(float(progress) - float(returned_progress))
            if progress is not None else None
        )
        candidate_entry["progress_delta_to_returned_portal_m"] = progress_delta
        same_station = (
            progress is not None
            and progress_delta <= VISITED_PORTAL_IDENTITY_TOLERANCE_M
        )
        candidate_entry["current_station_passed"] = bool(same_station)
        if same_station:
            candidate_entry["disposition"] = "ELIGIBLE_CURRENT_STATION"
            eligible.append((prepared, int(index), candidate_entry))
        else:
            candidate_entry["disposition"] = "EXCLUDED_NOT_CURRENT_STATION"
        if accounting is not None:
            accounting["prepared_candidates"].append(candidate_entry)
    if accounting is not None:
        accounting["eligible_current_station_candidates"] = len(eligible)
    if len(eligible) > 1:
        if accounting is not None:
            accounting["final_branch"] = "NEXT_PORTAL_DISPATCH_AMBIGUOUS_CURRENT_STATION"
        return {
            "state": "NEXT_PORTAL_DISPATCH_AMBIGUOUS_CURRENT_STATION",
            "reason": "MULTIPLE_FRESH_UNVISITED_CURRENT_STATION_PORTALS",
            "candidate": None,
            "candidate_accounting": accounting,
        }
    if len(eligible) == 1:
        selected_prepared, selected_index, selected_entry = eligible[0]
        selected_entry["disposition"] = "SELECTED"
        if accounting is not None:
            accounting["selected_index"] = selected_index
            accounting["final_branch"] = "NEXT_PORTAL_DISPATCH_COMMIT"
        return {
            "state": "NEXT_PORTAL_DISPATCH_COMMIT",
            "reason": "UNIQUE_FRESH_UNVISITED_CURRENT_STATION_PORTAL",
            "candidate": selected_prepared,
            "candidate_accounting": accounting,
        }
    if accounting is not None:
        accounting["final_branch"] = "FOLLOW_CORRIDOR_DEEPER"
    return {
        "state": "FOLLOW_CORRIDOR_DEEPER",
        "reason": "NO_FRESH_UNVISITED_CURRENT_STATION_PORTAL",
        "candidate": None,
        "candidate_accounting": accounting,
    }


def reset_portal_entry_attempt_state() -> Dict[str, None]:
    """Return only entry-attempt-local state that must not leak to Portal B."""
    return {
        "portal_bound_candidate": None,
        "portal_g14_shadow_target": None,
        "portal_g14_p_pre_admissibility": None,
        "portal_g14_p_pre_switch": None,
        "portal_g14_p_pre_runner": None,
        "room_search_v2_result": None,
    }


def prepare_fresh_post_return_portal_candidates(
    portal_effect_snapshot: Dict[str, Any],
    minimum_frame_sequence: Optional[int],
    odom_cache: OdomCache,
    current_pose: Tuple[float, float, float],
    corridor_anchor: Dict[str, Any],
    p_pre_upstream_tangent_m: float,
    candidate_accounting: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Prepare only a newer legal effect frame for passive post-return dispatch."""
    snapshot = portal_effect_snapshot if isinstance(portal_effect_snapshot, dict) else {}
    payload = snapshot.get("latest_effect_state")
    sequence = snapshot.get("last_portal_frame_sequence")
    if not (
        isinstance(payload, dict)
        and payload.get("contract_version") == PORTAL_EFFECT_CONTRACT_VERSION
        and payload.get("input_valid") is True
        and payload.get("room_zone_active") is True
        and isinstance(sequence, int)
        and isinstance(minimum_frame_sequence, int)
        and sequence > minimum_frame_sequence
    ):
        return []
    approach_heading = float(corridor_anchor["heading_rad"])
    approach_direction = [math.cos(approach_heading), math.sin(approach_heading)]
    prepared: List[Dict[str, Any]] = []
    accounting = candidate_accounting if isinstance(candidate_accounting, dict) else None
    if accounting is not None:
        accounting.setdefault("total_effect_candidates_seen", 0)
        accounting.setdefault("preparation_rejections", [])
    receive_time = snapshot.get("last_effect_receive_time")
    for side in ("left", "right"):
        row = payload.get(side)
        if not isinstance(row, dict) or row.get("effect_eligible") is not True:
            continue
        if accounting is not None:
            accounting["total_effect_candidates_seen"] += 1
        candidate = PortalBoundDoorCandidateAuthority._candidate_from_effect(
            payload,
            side,
            float(receive_time) if finite_number(receive_time) else time.time(),
        )
        if candidate is None:
            if accounting is not None:
                portal = row.get("portal") if isinstance(row.get("portal"), dict) else {}
                accounting["preparation_rejections"].append({
                    "side": side,
                    "track_id": portal.get("track_id"),
                    "portal_center_odom": None,
                    "disposition": "OBSERVED_BUT_REJECTED_BY_EXISTING_GATE",
                    "existing_rejection_reason": None,
                    "reason_availability": "PREPARATION_REJECTION_DETAIL_NOT_AVAILABLE_WITHOUT_SCOPE_EXPANSION",
                })
            continue
        binding = odom_cache.pose_at_source_stamp(float(candidate["portal_source_stamp"]))
        target = build_g14_shadow_target(
            candidate,
            binding,
            current_pose,
            approach_direction,
            p_pre_upstream_tangent_m,
        )
        centre = (target.get("frozen_geometry") or {}).get("portal_center_odom")
        if target.get("target_valid") is True and _finite_shape(centre, 2):
            prepared.append({
                "candidate": candidate,
                "target": target,
                "portal_center_odom": [float(centre[0]), float(centre[1])],
            })
        elif accounting is not None:
            accounting["preparation_rejections"].append({
                "side": candidate.get("side"),
                "track_id": candidate.get("portal_track_id"),
                "portal_center_odom": [float(centre[0]), float(centre[1])] if _finite_shape(centre, 2) else None,
                "disposition": "OBSERVED_BUT_REJECTED_BY_EXISTING_GATE",
                "existing_rejection_reason": target.get("rejection_reason"),
                "reason_availability": "EXISTING_TARGET_REJECTION_REASON",
            })
    if accounting is not None:
        accounting["total_prepared_candidates"] = len(prepared)
    return prepared


def target_distance_from_pose(target: Dict[str, Any], pose: Tuple[float, float, float]) -> Optional[float]:
    target_xy = target.get("target_xy_team_livox_odom")
    if not (isinstance(target_xy, list) and len(target_xy) == 2 and all(finite_number(v) for v in target_xy)):
        return None
    return math.hypot(float(target_xy[0]) - pose[0], float(target_xy[1]) - pose[1])


def target_diagnostics(target: Dict[str, Any], pose: Tuple[float, float, float], anchor: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    target_xy = target.get("target_xy_team_livox_odom")
    diag: Dict[str, Any] = {"target_xy_valid": False}
    if not (isinstance(target_xy, list) and len(target_xy) == 2 and all(finite_number(v) for v in target_xy)):
        return diag
    base_xy = target_base_xy(target_xy, pose)
    diag.update(
        {
            "target_xy_valid": True,
            "target_xy_team_livox_odom": [float(target_xy[0]), float(target_xy[1])],
            "target_base_xy": [float(base_xy[0]), float(base_xy[1])],
            "target_distance_base_m": float(math.hypot(base_xy[0], base_xy[1])),
            "target_heading_base_rad": float(math.atan2(base_xy[1], base_xy[0])),
        }
    )
    if anchor is not None:
        metrics = anchor_metrics(anchor, pose)
        diag.update(
            {
                "anchor_heading_rad": float(anchor["heading_rad"]),
                "anchor_progress_m": metrics.get("anchor_progress_m"),
                "anchor_lateral_error_m": metrics.get("anchor_lateral_error_m"),
                "anchor_yaw_error_rad": metrics.get("anchor_yaw_error_rad"),
            }
        )
    return diag


def count_state(trace: Sequence[Dict[str, Any]], state_name: str) -> int:
    return sum(1 for item in trace if item.get("state") == state_name)


def room_scan_summary() -> Dict[str, Any]:
    room = read_json(ROOM_VIEWPOINT_PATH)
    vision = read_json(VISION_PATH)
    danger_visible = bool(
        room.get("danger_source_visible")
        or room.get("danger_visible")
        or vision.get("danger_source_visible")
        or vision.get("danger_visible")
    )
    return {
        "room_viewpoint_final_decision": room.get("final_decision"),
        "room_search_state": room.get("room_search_state"),
        "room_viewpoint": room,
        "vision_final_decision": vision.get("final_decision"),
        "vision_scene_label": vision.get("scene_label") or vision.get("primary_scene_label"),
        "vision_danger_source_visible": bool(vision.get("danger_source_visible") or vision.get("danger_visible")),
        "danger_source_visible": danger_visible,
    }


def run_command(cmd: Sequence[str], timeout_sec: float) -> Dict[str, Any]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "cmd": list(cmd),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1200:],
            "stderr_tail": proc.stderr[-1200:],
            "timed_out": False,
            "wall_duration_sec": time.monotonic() - start,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": list(cmd),
            "returncode": 124,
            "stdout_tail": (exc.stdout or "")[-1200:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-1200:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
            "wall_duration_sec": time.monotonic() - start,
        }


def write_stage(stage: str, source: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {"stage": stage, "source": source, "wall_time_sec": time.time()}
    if extra:
        payload.update(extra)
    write_json(STAGE_PATH, payload)


class RoomZoneStateAuditPublisher:
    """Expose the existing state-machine room-zone result without using it."""

    def __init__(self) -> None:
        self.publisher = rospy.Publisher(ROOM_ZONE_STATE_TOPIC, String, queue_size=10, latch=False)
        self.previous_active: Optional[bool] = None
        self.transition_sequence = 0

    def publish(
        self,
        room_zone_active: bool,
        state_machine_stage: str,
        reason: str,
        *,
        current_anchor_progress_m: Optional[float] = None,
        effective_room_zone_progress_m: Optional[float] = None,
        room_zone_progress_offset_m: Optional[float] = None,
    ) -> None:
        active = bool(room_zone_active)
        transition_type = "NONE"
        if self.previous_active is not None and active != self.previous_active:
            self.transition_sequence += 1
            transition_type = "ENTER" if active else "EXIT"
        self.previous_active = active
        self.publisher.publish(String(data=json.dumps({
            "contract_version": ROOM_ZONE_STATE_CONTRACT_VERSION,
            "source_stamp": rospy.Time.now().to_sec(),
            "room_zone_active": active,
            "authoritative_source": "navigation_state_machine.FOLLOW_CORRIDOR.anchor_progress_m",
            "state_machine_stage": state_machine_stage,
            "transition_sequence": self.transition_sequence,
            "transition_type": transition_type,
            "reason": reason,
            # Keep the existing authoritative-source label for the unchanged
            # Portal gate contract.  These fields make the handoff continuity
            # observable without changing the gate's producer identity.
            "current_anchor_progress_m": current_anchor_progress_m,
            "effective_room_zone_progress_m": effective_room_zone_progress_m,
            "room_zone_progress_offset_m": room_zone_progress_offset_m,
        }, sort_keys=True)))


def write_absolute_target(target_xy: Sequence[float], source: str, subgoal_source: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    target = {
        "target_type": "generated_subgoal",
        "candidate_id": f"state_machine_{subgoal_source}",
        "parent_candidate_id": subgoal_source,
        "source": source,
        "source_frame": "team_livox_odom",
        "target_frame": "team_livox_odom",
        "target_xy_team_livox_odom": [float(target_xy[0]), float(target_xy[1])],
        "subgoal_source": subgoal_source,
        "safe_for_navigation": False,
        "planner_ready": False,
        "send_to_navigation": False,
        "diagnostic_only": True,
    }
    if extra:
        target.update(extra)
    n5b = {
        "stage": "STATE_MACHINE_TARGET",
        "final_decision": "N5_TARGET_SELECTION_READY_WITH_SUBGOAL",
        "parent_candidate_id": subgoal_source,
        "subgoal_source": subgoal_source,
        "subgoal_safety_pass": True,
        "subgoal_safety_status": "PASS",
        "safe_for_navigation": False,
        "planner_ready": False,
        "send_to_navigation": False,
        "errors": [],
        "warnings": [],
    }
    write_json(TARGET_PATH, target)
    write_json(N5B_PATH, n5b)
    return target


def write_base_target(base_xy: Sequence[float], source: str, subgoal_source: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    pose = pose_tuple(read_odom())
    target_xy = transform_base_xy(base_xy, pose)
    payload_extra = dict(extra or {})
    payload_extra.update({"subgoal_base_xy": [float(base_xy[0]), float(base_xy[1])], "matched_pose_x_y_yaw": list(pose)})
    return write_absolute_target(target_xy, source, subgoal_source, payload_extra)


def select_initial_anchor_heading(raw_heading: float, args: argparse.Namespace) -> Tuple[float, bool, str]:
    """Stabilize only an initial heading already close to odom +X.

    The initial short-horizon target is expressed in ``team_livox_odom``.
    Consequently, an odom frame whose initialization yaw differs from a
    historical run must retain a non-small raw heading rather than silently
    treating global odom +X as the physical corridor direction.
    """
    wrapped_error = normalize_angle(float(raw_heading))
    if (
        bool(args.entry_anchor_snap_heading_to_odom_x)
        and abs(wrapped_error) <= float(args.entry_anchor_heading_snap_threshold_rad)
    ):
        return 0.0, True, "default_odom_x_corridor_axis"
    if bool(args.entry_anchor_snap_heading_to_odom_x):
        return float(raw_heading), False, "raw_heading_outside_odom_x_snap_threshold"
    return float(raw_heading), False, "raw_heading_explicitly_requested"


def restore_raw_anchor_heading_at_room_zone(anchor: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Preserve the actual per-anchor heading while no Portal target owns motion.

    Room-zone entry changes which local obstacle evidence DWA observes; it
    does not itself grant a new high-level direction.  Restoring a small raw
    N5 heading while the state machine still emits a corridor-search target
    creates an ever-more lateral target over distance.  Keep the odom-x
    reference until an actual Portal/P_pre/P_through target replaces it.

    The function name remains for archive/API compatibility and to make the
    explicit no-restore boundary observable in existing run summaries.
    """
    report = {
        "attempted": True,
        "restored": False,
        "heading_before_rad": anchor.get("heading_rad"),
        "raw_heading_rad": anchor.get("raw_heading_rad"),
        "reason": None,
    }
    if not bool(args.entry_anchor_snap_heading_to_odom_x):
        report["reason"] = "raw_heading_already_explicitly_requested"
        return report
    if anchor.get("heading_snap_applied") is not True:
        report["reason"] = "raw_heading_retained_no_odom_x_snap"
        return report
    report.update({
        "heading_after_rad": anchor.get("heading_rad"),
        "reason": "room_zone_preserves_snapped_odom_x_corridor_axis_until_portal_handoff",
    })
    return report


def build_initial_anchor(
    args: argparse.Namespace,
    commands: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Preserve the established legacy entry target as a temporary bootstrap."""
    n5_cmd = [sys.executable, "scripts/short_horizon_target_selection/n5_target_selection_subgoal_audit.py"]
    n5_result = run_command(n5_cmd, args.input_timeout_sec + args.command_timeout_margin_sec)
    commands.append(n5_result)
    if n5_result["returncode"] != 0:
        raise RuntimeError("n5_target_selection_failed")
    n5_summary = read_json(N5_SUMMARY_PATH)
    n5_decision = n5_summary.get("final_decision")
    if n5_decision not in {
        "N5_TARGET_SELECTION_READY_WITH_EXISTING_CANDIDATE",
        "N5_TARGET_SELECTION_READY_WITH_SUBGOAL",
    }:
        raise RuntimeError(f"n5_target_selection_not_ready:{n5_decision}")
    regen_cmd = [sys.executable, "scripts/local_subgoal_runner_mvp/regenerate_corrected_target_from_frame_contract.py"]
    regen_result = run_command(regen_cmd, args.input_timeout_sec + args.command_timeout_margin_sec)
    commands.append(regen_result)
    if regen_result["returncode"] != 0:
        raise RuntimeError("target_regeneration_failed")
    target = read_json(TARGET_PATH)
    target_xy = target.get("target_xy_team_livox_odom")
    if not (isinstance(target_xy, list) and len(target_xy) == 2 and all(finite_number(v) for v in target_xy)):
        raise RuntimeError("initial_target_xy_unavailable")
    odom = read_odom()
    pose = pose_tuple(odom)
    dx = float(target_xy[0]) - pose[0]
    dy = float(target_xy[1]) - pose[1]
    raw_heading = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else pose[2]
    heading, snap_applied, snap_reason = select_initial_anchor_heading(raw_heading, args)
    return {
        "x": pose[0],
        "y": pose[1],
        "yaw_rad": pose[2],
        "heading_rad": heading,
        "raw_heading_rad": raw_heading,
        "heading_snap_applied": snap_applied,
        "heading_snap_reason": snap_reason,
        "heading_snap_threshold_rad": float(args.entry_anchor_heading_snap_threshold_rad),
        "room_zone_heading_restored": False,
        "source": "initial_n5_centerline",
        "corridor_axis_lifecycle": "LEGACY_BOOTSTRAP",
        "legacy_bootstrap_heading_rad": heading,
        "room_zone_progress_offset_m": 0.0,
        "corridor_axis_odom_epoch_generation": initialize_odom_cache().current_epoch_generation(),
    }


def maybe_handoff_corridor_axis(
    anchor: Dict[str, Any],
    corridor_axis_evidence: CorridorAxisEvidenceCache,
    odom_cache: OdomCache,
) -> Dict[str, Any]:
    """Replace only a bootstrap long-axis reference after mature geometry evidence."""
    lifecycle = str(anchor.get("corridor_axis_lifecycle") or "LEGACY_BOOTSTRAP")
    current_epoch = odom_cache.current_epoch_generation()
    configure_handoff_context = getattr(corridor_axis_evidence, "configure_handoff_context", None)
    if callable(configure_handoff_context):
        configure_handoff_context(
            odom_cache,
            float(anchor["legacy_bootstrap_heading_rad"]),
        )
    anchor_epoch = anchor.get("corridor_axis_odom_epoch_generation")
    if lifecycle == "CORRIDOR_BOUND":
        if anchor_epoch != current_epoch:
            corridor_axis_evidence.reset_maturity()
            return {
                "action": "REBOOTSTRAP_REQUIRED",
                "reason": "CORRIDOR_AXIS_ODOM_EPOCH_CHANGED",
                "previous_epoch_generation": anchor_epoch,
                "current_epoch_generation": current_epoch,
            }
        return {"action": "KEEP_BOUND", "reason": "CORRIDOR_AXIS_FROZEN"}
    evidence = corridor_axis_evidence.mature_bound_axis(
        odom_cache,
        float(anchor["legacy_bootstrap_heading_rad"]),
    )
    if not evidence.get("valid"):
        return {
            "action": "KEEP_LEGACY_BOOTSTRAP",
            "reason": str(evidence.get("reason") or "CORRIDOR_AXIS_NOT_MATURE"),
        }
    source_pose = evidence["odom_binding"]["source_pose_x_y_yaw"]
    pre_handoff_metrics = anchor_metrics(anchor, tuple(float(value) for value in source_pose))
    pre_handoff_current_progress_m = pre_handoff_metrics.get("anchor_progress_m")
    pre_handoff_effective_progress_m = effective_room_zone_progress_m(
        anchor,
        pre_handoff_current_progress_m,
    )
    if pre_handoff_effective_progress_m is None:
        return {
            "action": "KEEP_LEGACY_BOOTSTRAP",
            "reason": "CORRIDOR_AXIS_HANDOFF_ROOM_ZONE_PROGRESS_UNAVAILABLE",
        }
    replacement = {
        "x": float(source_pose[0]),
        "y": float(source_pose[1]),
        "yaw_rad": float(source_pose[2]),
        "heading_rad": float(evidence["heading_odom_rad"]),
        "raw_heading_rad": anchor.get("raw_heading_rad"),
        "heading_snap_applied": False,
        "heading_snap_reason": "corridor_axis_mature_handoff",
        "heading_snap_threshold_rad": None,
        "room_zone_heading_restored": False,
        "source": "corridor_axis_evidence_handoff",
        "corridor_axis_lifecycle": "CORRIDOR_BOUND",
        "legacy_bootstrap_heading_rad": anchor.get("legacy_bootstrap_heading_rad"),
        "room_zone_progress_offset_m": pre_handoff_effective_progress_m,
        "corridor_axis_odom_epoch_generation": evidence["odom_binding"].get("odom_epoch_generation"),
        "corridor_axis_evidence": evidence,
    }
    certificate_id = evidence.get("certificate_id")
    consume_mature_certificate = getattr(corridor_axis_evidence, "consume_mature_certificate", None)
    certificate_consumed = (
        bool(consume_mature_certificate(certificate_id))
        if certificate_id is not None and callable(consume_mature_certificate) else False
    )
    if certificate_id is not None and not certificate_consumed:
        return {
            "action": "KEEP_LEGACY_BOOTSTRAP",
            "reason": "CORRIDOR_AXIS_CERTIFICATE_CONSUME_FAILED",
        }
    return {
        "action": "HANDOFF_TO_CORRIDOR_AXIS",
        "reason": "CORRIDOR_AXIS_MATURE",
        "lifecycle_transition": [
            "LEGACY_BOOTSTRAP",
            "CORRIDOR_AXIS_MATURE",
            "CORRIDOR_BOUND",
        ],
        "legacy_heading_rad": anchor.get("heading_rad"),
        "corridor_heading_rad": replacement["heading_rad"],
        "heading_delta_rad": normalize_angle(replacement["heading_rad"] - float(anchor["heading_rad"])),
        "room_zone_progress_handoff": {
            "pre_handoff_current_anchor_progress_m": pre_handoff_current_progress_m,
            "pre_handoff_effective_room_zone_progress_m": pre_handoff_effective_progress_m,
            "saved_room_zone_progress_offset_m": replacement["room_zone_progress_offset_m"],
            "new_anchor_progress_m": 0.0,
            "effective_room_zone_progress_m": pre_handoff_effective_progress_m,
        },
        "maturity": evidence,
        "certificate_consumed": bool(certificate_consumed),
        "anchor": replacement,
    }


def write_anchor_target(
    anchor: Dict[str, Any],
    lookahead_m: float,
    source: str,
    *,
    room_zone_active: bool = True,
) -> Dict[str, Any]:
    pose = pose_tuple(read_odom())
    metrics = anchor_metrics(anchor, pose)
    along = max(0.0, float(metrics["anchor_progress_m"]))
    heading = float(anchor["heading_rad"])
    target_along = along + float(lookahead_m)
    target_xy = [
        float(anchor["x"]) + math.cos(heading) * target_along,
        float(anchor["y"]) + math.sin(heading) * target_along,
    ]
    return write_absolute_target(
        target_xy,
        source,
        "state_machine_centerline",
        {
            "entry_anchor_line": anchor,
            "entry_anchor_metrics": metrics,
            "entry_anchor_lookahead_m": lookahead_m,
            "room_zone_active": bool(room_zone_active),
            "corridor_center_target_scope": (
                "pre_room_zone_only" if not room_zone_active else "disabled_in_room_zone"
            ),
        },
    )


def write_anchor_progress_target(
    anchor: Dict[str, Any],
    target_progress_m: float,
    source: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pose = pose_tuple(read_odom())
    metrics = anchor_metrics(anchor, pose)
    heading = float(anchor["heading_rad"])
    target_along = max(0.0, float(target_progress_m))
    target_xy = [
        float(anchor["x"]) + math.cos(heading) * target_along,
        float(anchor["y"]) + math.sin(heading) * target_along,
    ]
    payload = {
        "entry_anchor_line": anchor,
        "entry_anchor_metrics": metrics,
        "entry_anchor_target_progress_m": target_along,
    }
    payload.update(extra or {})
    return write_absolute_target(target_xy, source, "state_machine_centerline", payload)


def write_room_viewpoint_target() -> Dict[str, Any]:
    room = read_json(ROOM_VIEWPOINT_PATH)
    viewpoint = room.get("next_viewpoint_base") if isinstance(room.get("next_viewpoint_base"), dict) else None
    base_xy = viewpoint.get("base_xy") if isinstance(viewpoint, dict) else None
    if room.get("final_decision") != "ROOM_VIEWPOINT_READY":
        raise RuntimeError(f"room_viewpoint_not_ready:{room.get('final_decision')}")
    if not (isinstance(base_xy, list) and len(base_xy) == 2 and all(finite_number(v) for v in base_xy)):
        raise RuntimeError("room_viewpoint_base_xy_unavailable")
    return write_base_target(base_xy, "state_machine_room_frontier_viewpoint", "room_frontier_viewpoint", {"room_viewpoint": room})


def write_doorway_target(
    doorway: Optional[Dict[str, Any]] = None,
    *,
    locked: bool = False,
    commit_depth_m: float = 0.0,
    args: Optional[argparse.Namespace] = None,
) -> Dict[str, Any]:
    doorway = doorway or read_json(DOORWAY_PATH)
    entry = doorway.get("door_entry_pose_odom") if isinstance(doorway.get("door_entry_pose_odom"), dict) else None
    if doorway.get("final_decision") != "DOORWAY_CANDIDATE_READY" or not doorway.get("doorway_candidate"):
        raise RuntimeError(f"doorway_candidate_not_ready:{doorway.get('final_decision')}")
    if not entry or not (finite_number(entry.get("x")) and finite_number(entry.get("y"))):
        raise RuntimeError("doorway_entry_pose_unavailable")
    target_x = float(entry["x"])
    target_y = float(entry["y"])
    target_meta: Dict[str, Any] = {
        "door_side": doorway.get("door_side"),
        "doorway_candidate": doorway,
        "doorway_target_locked": bool(locked),
        "doorway_commit_depth_m": float(commit_depth_m),
        "doorway_entry_xy_odom": [target_x, target_y],
    }
    if args is not None and commit_depth_m > 0.0:
        target_xy, validation = select_grid_validated_doorway_target(entry, doorway, args, commit_depth_m=commit_depth_m)
        target_x = float(target_xy[0])
        target_y = float(target_xy[1])
        target_meta["doorway_commit_grid_validation"] = validation
        if finite_number(entry.get("yaw")):
            target_meta["doorway_commit_yaw_rad"] = float(entry["yaw"])
        if validation.get("grid_validation_pass"):
            target_meta["doorway_commit_depth_m"] = float(validation.get("selected_depth_m", commit_depth_m))
    elif commit_depth_m > 0.0 and finite_number(entry.get("yaw")):
        yaw = float(entry["yaw"])
        target_x += math.cos(yaw) * float(commit_depth_m)
        target_y += math.sin(yaw) * float(commit_depth_m)
        target_meta["doorway_commit_yaw_rad"] = yaw
    return write_absolute_target(
        [target_x, target_y],
        "state_machine_doorway_entry_pose",
        "doorway_entry_pose",
        target_meta,
    )


def doorway_target_grid_validated(target: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(target, dict):
        return False
    validation = target.get("doorway_commit_grid_validation")
    return bool(isinstance(validation, dict) and validation.get("grid_validation_pass"))


def doorway_approach_target_available(target: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(target, dict):
        return False
    target_xy = target.get("target_xy_team_livox_odom")
    if not (isinstance(target_xy, list) and len(target_xy) == 2 and all(finite_number(v) for v in target_xy)):
        return False
    commit_depth = target.get("doorway_commit_depth_m")
    if finite_number(commit_depth) and float(commit_depth) <= 0.0:
        return True
    return doorway_target_grid_validated(target)


def doorway_opening_status(doorway: Dict[str, Any]) -> Dict[str, Any]:
    side = doorway.get("door_side")
    profile_root = doorway.get("doorway_geometry_profile") if isinstance(doorway.get("doorway_geometry_profile"), dict) else {}
    side_profile = profile_root.get(side) if side in ("left", "right") else None
    if not isinstance(side_profile, dict):
        side_profile = profile_root.get("selected_by_candidate_side")
    selected = side_profile.get("selected_opening") if isinstance(side_profile, dict) else None
    entry = doorway.get("door_entry_pose_odom") if isinstance(doorway.get("door_entry_pose_odom"), dict) else {}
    center_estimated = bool(
        isinstance(selected, dict)
        and selected.get("opening_center_estimated")
        and entry.get("opening_center_estimated")
    )
    status = {
        "door_side": side,
        "opening_center_estimated": center_estimated,
        "entry_pose_source": entry.get("source") if isinstance(entry, dict) else None,
        "selected_opening": selected if isinstance(selected, dict) else None,
    }
    if not isinstance(side_profile, dict):
        status["reason"] = "doorway_profile_unavailable"
    elif not isinstance(selected, dict):
        status["reason"] = "selected_opening_unavailable"
    elif not selected.get("width_plausible"):
        status["reason"] = "opening_width_not_plausible"
    elif selected.get("partial_opening_observed"):
        status["reason"] = "partial_opening_observed"
    elif not selected.get("opening_center_estimated"):
        status["reason"] = "opening_center_not_estimated"
    elif not entry.get("opening_center_estimated"):
        status["reason"] = "entry_pose_not_profile_center"
    else:
        status["reason"] = "opening_center_estimated"
    return status


def doorway_opening_center_estimated(doorway: Dict[str, Any]) -> bool:
    return bool(doorway_opening_status(doorway).get("opening_center_estimated"))


def doorway_profile_opening_status(doorway: Dict[str, Any]) -> Dict[str, Any]:
    profile_root = doorway.get("doorway_geometry_profile") if isinstance(doorway.get("doorway_geometry_profile"), dict) else {}
    candidates: List[Dict[str, Any]] = []
    for side in ("left", "right"):
        side_profile = profile_root.get(side)
        if not isinstance(side_profile, dict):
            continue
        selected = side_profile.get("selected_opening")
        if not isinstance(selected, dict):
            continue
        center_x = selected.get("center_x_m")
        width = selected.get("width_m")
        if not (finite_number(center_x) and finite_number(width)):
            continue
        status = {
            "door_side": side,
            "selected_opening": selected,
            "center_x_base_m": float(center_x),
            "opening_width_m": float(width),
            "opening_width_plausible": bool(selected.get("width_plausible")),
            "opening_center_estimated": bool(selected.get("opening_center_estimated")),
            "partial_opening_observed": bool(selected.get("partial_opening_observed")),
            "before_wall_or_unknown": bool(selected.get("before_wall_or_unknown")),
            "after_wall_or_unknown": bool(selected.get("after_wall_or_unknown")),
            "doorway_final_decision": doorway.get("final_decision"),
        }
        if status["opening_width_plausible"]:
            candidates.append(status)
    if not candidates:
        return {"available": False, "reason": "no_plausible_profile_opening"}
    candidates.sort(
        key=lambda item: (
            1 if item.get("opening_center_estimated") else 0,
            1 if item.get("partial_opening_observed") else 0,
            -abs(float(item.get("opening_width_m") or 0.0) - 0.9),
            -float(item.get("center_x_base_m") or 0.0),
        ),
        reverse=True,
    )
    best = dict(candidates[0])
    best["available"] = True
    best["reason"] = (
        "opening_center_estimated"
        if best.get("opening_center_estimated")
        else ("partial_opening_observed" if best.get("partial_opening_observed") else "plausible_profile_opening")
    )
    return best


def _room_side_gap_candidate_legacy(doorway: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    profile_root = doorway.get("doorway_geometry_profile") if isinstance(doorway.get("doorway_geometry_profile"), dict) else {}
    candidates: List[Dict[str, Any]] = []
    for side in ("left", "right"):
        side_profile = profile_root.get(side)
        if not isinstance(side_profile, dict):
            continue
        side_candidate = doorway.get(f"{side}_candidate") if isinstance(doorway.get(f"{side}_candidate"), dict) else {}
        geometry_support = bool(side_candidate.get("geometry_pass") or side_candidate.get("soft_geometry_pass"))
        segments = side_profile.get("open_segments")
        if not isinstance(segments, list):
            selected = side_profile.get("selected_opening")
            segments = [selected] if isinstance(selected, dict) else []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            center_x = seg.get("center_x_m")
            width = seg.get("width_m")
            if not (finite_number(center_x) and finite_number(width)):
                continue
            center_x = float(center_x)
            width = float(width)
            if not (float(args.room_side_gap_min_center_x_m) <= center_x <= float(args.room_side_gap_max_center_x_m)):
                continue
            if not (float(args.room_side_gap_min_width_m) <= width <= float(args.room_side_gap_max_width_m)):
                continue
            if args.room_side_gap_require_profile_plausible and not bool(seg.get("width_plausible")):
                continue
            if args.room_side_gap_require_profile_door_signal and not (
                bool(seg.get("partial_opening_observed")) or bool(seg.get("opening_center_estimated"))
            ):
                continue
            wall_support_count = int(bool(seg.get("before_wall_or_unknown"))) + int(bool(seg.get("after_wall_or_unknown")))
            if wall_support_count < int(args.room_side_gap_min_wall_support_count) and not geometry_support:
                continue
            score = (
                3.0 * wall_support_count
                + (2.0 if geometry_support else 0.0)
                - abs(center_x - float(args.room_side_gap_preferred_center_x_m))
                - 0.25 * abs(width - float(args.room_side_gap_preferred_width_m))
            )
            candidates.append(
                {
                    "available": True,
                    "door_side": side,
                    "center_x_base_m": center_x,
                    "opening_width_m": width,
                    "start_x_m": float(seg["start_x_m"]) if finite_number(seg.get("start_x_m")) else None,
                    "end_x_m": float(seg["end_x_m"]) if finite_number(seg.get("end_x_m")) else None,
                    "width_plausible": bool(seg.get("width_plausible")),
                    "wall_support_count": wall_support_count,
                    "geometry_support": geometry_support,
                    "selected_opening": seg,
                    "score": score,
                    "source": "room_side_gap_profile",
                    "doorway_final_decision": doorway.get("final_decision"),
                    "doorway_decision_mode": doorway.get("decision_mode"),
                }
            )
    if not candidates:
        return {"available": False, "reason": "no_room_side_gap_candidate"}
    preferred_side = doorway.get("door_side")
    if (
        doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY"
        and preferred_side in {"left", "right"}
    ):
        preferred_candidates = [item for item in candidates if item.get("door_side") == preferred_side]
        if preferred_candidates:
            candidates = preferred_candidates
        else:
            return {
                "available": False,
                "reason": "room_side_gap_candidate_side_mismatch",
                "preferred_door_side": preferred_side,
                "candidate_sides": sorted({str(item.get("door_side")) for item in candidates}),
            }
    candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    best = dict(candidates[0])
    best["reason"] = "room_side_gap_candidate"
    return best


def room_side_gap_candidate(doorway: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """V1 pure adapter; forced-entry mode is intentionally not an input guard."""
    return select_room_side_gap_candidate(doorway, vars(args))


def _update_room_side_gap_observation_legacy(
    args: argparse.Namespace,
    current: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
    anchor: Optional[Dict[str, Any]],
    pose: Tuple[float, float, float],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    metrics = anchor_metrics(anchor, pose) if anchor is not None else {}
    if not candidate.get("available"):
        status = {"available": False, "reason": candidate.get("reason", "room_side_gap_unavailable")}
        if isinstance(current, dict) and finite_number(current.get("alignment_target_anchor_progress_m")):
            preferred_side = candidate.get("preferred_door_side")
            if (
                candidate.get("reason") == "room_side_gap_candidate_side_mismatch"
                and preferred_side in {"left", "right"}
                and current.get("door_side") != preferred_side
            ):
                status.update(
                    {
                        "door_side": current.get("door_side"),
                        "preferred_door_side": preferred_side,
                        "alignment_cleared": True,
                        "alignment_clear_reason": "room_side_gap_preferred_side_changed",
                    }
                )
                return None, status
            current_progress = metrics.get("anchor_progress_m")
            target_progress = float(current["alignment_target_anchor_progress_m"])
            longitudinal_error = target_progress - float(current_progress) if finite_number(current_progress) else None
            missed_count = int(current.get("missed_observation_count") or 0) + 1
            current["missed_observation_count"] = missed_count
            stable_count = int(current.get("stable_observation_count") or 0)
            status.update(
                {
                    "door_side": current.get("door_side"),
                    "center_x_base_m": current.get("center_x_base_m"),
                    "opening_width_m": current.get("opening_width_m"),
                    "selected_opening": current.get("selected_opening"),
                    "stable_observation_count": stable_count,
                    "alignment_locked": True,
                    "anchor_progress_m": current_progress,
                    "alignment_target_anchor_progress_m": target_progress,
                    "alignment_longitudinal_error_m": longitudinal_error,
                    "missed_observation_count": missed_count,
                    "trigger_ready": False,
                    "trigger_suppressed_reason": "room_side_gap_not_visible_current_frame",
                }
            )
            if missed_count > int(args.room_side_gap_max_missed_observations):
                status["alignment_cleared"] = True
                status["alignment_clear_reason"] = "room_side_gap_lost"
                return None, status
            return current, status
        return current, status
    previous_center = current.get("center_x_base_m") if isinstance(current, dict) else None
    previous_width = current.get("opening_width_m") if isinstance(current, dict) else None
    same_side = isinstance(current, dict) and current.get("door_side") == candidate.get("door_side")
    already_locked = bool(same_side and finite_number(current.get("alignment_target_anchor_progress_m")))
    center_delta = abs(float(candidate["center_x_base_m"]) - float(previous_center)) if finite_number(previous_center) else None
    width_delta = abs(float(candidate["opening_width_m"]) - float(previous_width)) if finite_number(previous_width) else None
    stable = bool(
        same_side
        and center_delta is not None
        and width_delta is not None
        and center_delta <= float(args.room_side_gap_center_stability_tolerance_m)
        and width_delta <= float(args.room_side_gap_width_stability_tolerance_m)
    )
    stable_count = int(current.get("stable_observation_count") or 0) + 1 if stable and isinstance(current, dict) else 1
    observation = dict(candidate)
    current_progress = metrics.get("anchor_progress_m")
    alignment_target = (
        float(current["alignment_target_anchor_progress_m"])
        if already_locked
        else (
            float(current_progress) + float(candidate["center_x_base_m"])
            if stable_count >= int(args.room_side_gap_required_stable_observations) and finite_number(current_progress)
            else None
        )
    )
    longitudinal_error = (
        float(alignment_target) - float(current_progress)
        if finite_number(alignment_target) and finite_number(current_progress)
        else None
    )
    observation.update(
        {
            "stable_observation_count": stable_count,
            "center_delta_m": center_delta,
            "width_delta_m": width_delta,
            "missed_observation_count": 0,
            "anchor_progress_m": current_progress,
            "alignment_locked": finite_number(alignment_target),
            "alignment_target_anchor_progress_m": alignment_target,
            "alignment_longitudinal_error_m": longitudinal_error,
            "trigger_ready": bool(
                finite_number(longitudinal_error)
                and abs(float(longitudinal_error)) <= float(args.room_side_gap_turn_alignment_tolerance_m)
            ),
        }
    )
    return observation, observation


def update_room_side_gap_observation(
    args: argparse.Namespace,
    current: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
    anchor: Optional[Dict[str, Any]],
    pose: Tuple[float, float, float],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    metrics = anchor_metrics(anchor, pose) if anchor is not None else {}
    progress = metrics.get("anchor_progress_m") if isinstance(metrics, dict) else None
    return update_room_side_gap_observation_from_progress(vars(args), current, candidate, progress)


def median_float(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) * 0.5)


def update_door_landmark_tracks(
    door_tracks: Dict[str, List[Dict[str, Any]]],
    next_track_ids: Dict[str, int],
    doorway: Dict[str, Any],
    gap_status: Dict[str, Any],
    pose: Tuple[float, float, float],
    anchor: Optional[Dict[str, Any]],
    iteration: int,
    observation_phase: str,
    corridor_half_width_m: float,
) -> List[Dict[str, Any]]:
    profile_root = doorway.get("doorway_geometry_profile") if isinstance(doorway.get("doorway_geometry_profile"), dict) else {}
    added: List[Dict[str, Any]] = []
    current_progress = anchor_metrics(anchor, pose).get("anchor_progress_m") if anchor is not None else None
    for side in ("left", "right"):
        side_profile = profile_root.get(side) if isinstance(profile_root.get(side), dict) else {}
        selected = side_profile.get("selected_opening") if isinstance(side_profile.get("selected_opening"), dict) else {}
        source = "doorway_geometry_profile.selected_opening"
        if not selected and gap_status.get("available") and gap_status.get("door_side") == side:
            selected = gap_status.get("selected_opening") if isinstance(gap_status.get("selected_opening"), dict) else {}
            source = "room_side_gap_status.selected_opening"

        start_x = selected.get("start_x_m", gap_status.get("start_x_m") if gap_status.get("door_side") == side else None)
        end_x = selected.get("end_x_m", gap_status.get("end_x_m") if gap_status.get("door_side") == side else None)
        center_x = selected.get(
            "center_x_m",
            gap_status.get("center_x_base_m") if gap_status.get("door_side") == side else None,
        )
        width_m = selected.get(
            "width_m",
            gap_status.get("opening_width_m") if gap_status.get("door_side") == side else None,
        )
        if not all(finite_number(value) for value in (start_x, end_x, center_x, width_m)):
            continue

        center_base = selected.get("opening_center_base_xy")
        approximation = not (
            isinstance(center_base, list)
            and len(center_base) == 2
            and all(finite_number(value) for value in center_base)
        )
        if approximation:
            lateral_m = float(corridor_half_width_m) if side == "left" else -float(corridor_half_width_m)
            center_base_xy = [float(center_x), lateral_m]
            center_base_source = "corridor_half_width_approximation"
        else:
            center_base_xy = [float(center_base[0]), float(center_base[1])]
            center_base_source = "opening_center_base_xy"
        center_odom_xy = transform_base_xy(center_base_xy, pose)

        side_candidate = doorway.get(f"{side}_candidate") if isinstance(doorway.get(f"{side}_candidate"), dict) else {}
        side_counts = side_candidate.get("side_probe_counts") if isinstance(side_candidate.get("side_probe_counts"), dict) else {}
        grid_stamp = doorway.get("grid_stamp_sec")
        observation_key = (
            f"grid:{float(grid_stamp):.9f}"
            if finite_number(grid_stamp)
            else f"iteration:{int(iteration)}"
        )
        side_tracks = door_tracks.setdefault(side, [])
        if any(
            item.get("observation_key") == observation_key
            for track in side_tracks
            for item in track.get("observations", [])
            if isinstance(item, dict)
        ):
            continue

        progress = float(current_progress) if finite_number(current_progress) else None
        center_progress = progress + float(center_x) if progress is not None else None
        observation_delta = center_progress - progress if center_progress is not None else None
        free_ratio = float(side_counts["free_ratio"]) if finite_number(side_counts.get("free_ratio")) else None
        geometry_pass = bool(side_candidate.get("geometry_pass"))
        soft_geometry_pass = bool(side_candidate.get("soft_geometry_pass"))
        door_cue_ahead = bool(
            finite_number(observation_delta)
            and 0.3 <= float(observation_delta) <= 2.5
            and (geometry_pass or soft_geometry_pass or (free_ratio is not None and free_ratio >= 0.65))
        )
        if not finite_number(observation_delta):
            door_cue_reason = "rejected_progress_unavailable"
        elif not 0.3 <= float(observation_delta) <= 2.5:
            door_cue_reason = "rejected_not_in_ahead_window"
        elif geometry_pass:
            door_cue_reason = "geometry_pass"
        elif soft_geometry_pass:
            door_cue_reason = "soft_geometry_pass"
        elif free_ratio is not None and free_ratio >= 0.65:
            door_cue_reason = "free_ratio_supported"
        else:
            door_cue_reason = "rejected_no_door_signal"
        observation = {
            "side": side,
            "center_base_xy": center_base_xy,
            "center_odom_xy": center_odom_xy,
            "coordinate_frame": "team_livox_odom",
            "start_x_m": float(start_x),
            "end_x_m": float(end_x),
            "center_x_m": float(center_x),
            "opening_width_m": float(width_m),
            "width_plausible": bool(selected.get("width_plausible")),
            "current_anchor_progress_m": progress,
            "observation_anchor_progress_m": progress,
            "start_progress_m": progress + float(start_x) if progress is not None else None,
            "end_progress_m": progress + float(end_x) if progress is not None else None,
            "center_progress_m": center_progress,
            "observation_center_progress_m": center_progress,
            "observation_door_progress_delta_m": observation_delta,
            "door_cue_ahead": door_cue_ahead,
            "raw_door_cue_ahead": door_cue_ahead,
            "door_cue_reason": door_cue_reason,
            "door_cue_class": "unclassified",
            "door_cue_stop_candidate": False,
            "actionable_door_cue": False,
            "actionable_reject_reason": "not_evaluated",
            "free_ratio": free_ratio,
            "geometry_pass": geometry_pass,
            "soft_geometry_pass": soft_geometry_pass,
            "geometry_confidence": (
                float(side_candidate["geometry_confidence"])
                if finite_number(side_candidate.get("geometry_confidence"))
                else None
            ),
            "stable_frame_count": int(side_candidate.get("stable_frame_count") or 0),
            "robot_pose": {
                "frame": "team_livox_odom",
                "x": float(pose[0]),
                "y": float(pose[1]),
                "yaw": float(pose[2]),
            },
            "source": source,
            "center_base_source": center_base_source,
            "approximation": approximation,
            "timestamp_sec": float(grid_stamp) if finite_number(grid_stamp) else time.time(),
            "timestamp_source": "doorway_grid_stamp" if finite_number(grid_stamp) else "wall_time_fallback",
            "observation_wall_time_sec": time.time(),
            "iteration": int(iteration),
            "observation_phase": observation_phase,
            "observation_key": observation_key,
        }
        track_distances = []
        for track in side_tracks:
            observations = track.get("observations") if isinstance(track.get("observations"), list) else []
            values = [
                float(item["center_progress_m"])
                for item in observations
                if isinstance(item, dict) and finite_number(item.get("center_progress_m"))
            ]
            median_progress = median_float(values)
            if median_progress is not None and progress is not None:
                track_distances.append((abs(progress + float(center_x) - median_progress), track))
        if track_distances:
            nearest_distance, selected_track = min(track_distances, key=lambda item: item[0])
        else:
            nearest_distance, selected_track = None, None
        if selected_track is None or nearest_distance is None or nearest_distance > 0.8:
            track_id = f"{side}_{int(next_track_ids.get(side, 0)):03d}"
            next_track_ids[side] = int(next_track_ids.get(side, 0)) + 1
            selected_track = {
                "track_id": track_id,
                "side": side,
                "observations": [],
                "created_iteration": int(iteration),
                "last_seen_iteration": int(iteration),
            }
            side_tracks.append(selected_track)
        selected_track["observations"].append(observation)
        selected_track["observations"] = selected_track["observations"][-5:]
        selected_track["last_seen_iteration"] = int(iteration)
        if len(side_tracks) > 3:
            side_tracks.sort(key=lambda track: int(track.get("last_seen_iteration", -1)), reverse=True)
            door_tracks[side] = side_tracks[:3]
        added.append(observation)
    return added


def annotate_door_cue_actionability(
    observations: Sequence[Dict[str, Any]],
    room_zone_reached: bool,
    room_zone_start_progress_m: float,
    current_anchor_progress_m: Optional[float],
) -> None:
    """Annotate existing cue observations only; this does not affect navigation decisions."""
    for observation in observations:
        raw_cue = bool(observation.get("raw_door_cue_ahead", observation.get("door_cue_ahead")))
        cue_class = str(observation.get("door_cue_class") or "unclassified")
        stop_candidate = bool(raw_cue and cue_class != "broad_bilateral_opening")
        if not raw_cue:
            reject_reason = "not_raw_door_cue"
        elif not room_zone_reached:
            reject_reason = "before_room_zone"
        elif cue_class == "broad_bilateral_opening":
            reject_reason = "broad_bilateral_opening"
        elif not stop_candidate:
            reject_reason = "not_stop_candidate"
        else:
            reject_reason = "pass"
        observation.update(
            {
                "raw_door_cue_ahead": raw_cue,
                "room_zone_reached": bool(room_zone_reached),
                "room_zone_start_progress_m": float(room_zone_start_progress_m),
                "current_anchor_progress_m": current_anchor_progress_m,
                "door_cue_class": cue_class,
                "door_cue_stop_candidate": stop_candidate,
                "actionable_door_cue": bool(raw_cue and room_zone_reached and stop_candidate),
                "actionable_reject_reason": reject_reason,
            }
        )


def build_side_gap_nav_debug_candidates(
    observations: Sequence[Dict[str, Any]],
    cache: Dict[str, List[Dict[str, Any]]],
    anchor: Optional[Dict[str, Any]],
    robot_pose: Tuple[float, float, float],
    args: argparse.Namespace,
    room_zone_reached: bool,
    current_anchor_progress_m: Optional[float],
) -> List[Dict[str, Any]]:
    """Build D_raw/D_nav diagnostics from side intervals without writing a target."""
    if not room_zone_reached or anchor is None:
        return []
    candidates: List[Dict[str, Any]] = []
    heading = float(anchor["heading_rad"])
    for observation in observations:
        side = observation.get("side")
        start_progress = observation.get("start_progress_m")
        center_progress = observation.get("center_progress_m")
        center_xy = observation.get("center_odom_xy")
        if not (
            side in {"left", "right"}
            and finite_number(start_progress)
            and finite_number(center_progress)
            and isinstance(center_xy, list)
            and len(center_xy) == 2
            and all(finite_number(value) for value in center_xy)
        ):
            continue
        side_cache = cache.setdefault(str(side), [])
        previous = side_cache[-1] if side_cache else None
        previous_center = previous.get("center_progress_m") if isinstance(previous, dict) else None
        center_jump = (
            abs(float(center_progress) - float(previous_center)) if finite_number(previous_center) else None
        )
        side_cache.append({"center_progress_m": float(center_progress), "iteration": observation.get("iteration")})
        del side_cache[:-2]
        seen_count = len(side_cache)

        d_nav_progress = float(start_progress) - float(args.door_nav_approach_offset_m)
        d_nav_xy = [
            float(anchor["x"]) + math.cos(heading) * d_nav_progress,
            float(anchor["y"]) + math.sin(heading) * d_nav_progress,
        ]
        base_xy = target_base_xy(d_nav_xy, robot_pose)
        nav_distance = math.hypot(base_xy[0], base_xy[1])
        nav_progress_delta = (
            d_nav_progress - float(current_anchor_progress_m)
            if finite_number(current_anchor_progress_m)
            else None
        )
        dx = d_nav_xy[0] - float(anchor["x"])
        dy = d_nav_xy[1] - float(anchor["y"])
        nav_lateral_error = -math.sin(heading) * dx + math.cos(heading) * dy
        nav_in_front = base_xy[0] > 0.3
        first_frame_mode = int(args.side_gap_confirm_frames) <= 1
        cache_confirmed = bool(
            seen_count >= 1
            if first_frame_mode
            else (
                seen_count >= int(args.side_gap_confirm_frames)
                and finite_number(center_jump)
                and float(center_jump) <= float(args.side_gap_confirm_progress_jump_m)
            )
        )
        lightweight_confirmed = bool(
            room_zone_reached
            and nav_in_front
            and finite_number(nav_progress_delta)
            and 0.3 <= float(nav_progress_delta) <= 2.5
            and cache_confirmed
        )
        candidates.append(
            {
                "side": side,
                "iteration": observation.get("iteration"),
                "observation_phase": observation.get("observation_phase"),
                "room_zone_reached": True,
                "room_zone_start_progress_m": float(args.room_zone_start_progress_m),
                "current_anchor_progress_m": current_anchor_progress_m,
                "robot_pose": {
                    "frame": observation.get("coordinate_frame", "team_livox_odom"),
                    "x": robot_pose[0],
                    "y": robot_pose[1],
                    "yaw": robot_pose[2],
                },
                "start_x_m": observation.get("start_x_m"),
                "end_x_m": observation.get("end_x_m"),
                "center_x_m": observation.get("center_x_m"),
                "start_progress_m": start_progress,
                "end_progress_m": observation.get("end_progress_m"),
                "center_progress_m": center_progress,
                "opening_width_m": observation.get("opening_width_m"),
                "center_base_xy": observation.get("center_base_xy"),
                "center_odom_xy": center_xy,
                "coordinate_frame": observation.get("coordinate_frame", "team_livox_odom"),
                "source": observation.get("source"),
                "timestamp_sec": observation.get("timestamp_sec"),
                "geometry_pass": observation.get("geometry_pass"),
                "soft_geometry_pass": observation.get("soft_geometry_pass"),
                "free_ratio": observation.get("free_ratio"),
                "width_plausible": observation.get("width_plausible"),
                "D_raw_xy": center_xy,
                "D_raw_progress_m": center_progress,
                "D_raw_side": side,
                "D_raw_frame": observation.get("coordinate_frame", "team_livox_odom"),
                "D_nav_xy": d_nav_xy,
                "D_nav_progress_m": d_nav_progress,
                "D_nav_frame": "team_livox_odom",
                "D_nav_target_base_xy": [base_xy[0], base_xy[1]],
                "D_nav_distance_m": nav_distance,
                "D_nav_heading_error_rad": normalize_angle(math.atan2(base_xy[1], base_xy[0])),
                "D_nav_in_front": nav_in_front,
                "D_nav_progress_delta_m": nav_progress_delta,
                "D_nav_lateral_error_to_corridor_center_m": nav_lateral_error,
                "D_nav_near_corridor_center": abs(nav_lateral_error) <= 0.35,
                "seen_count": seen_count,
                "previous_center_progress_m": previous_center,
                "center_progress_jump_m": center_jump,
                "lightweight_confirmed": lightweight_confirmed,
                "diagnostic_only": True,
            }
        )
    return candidates


def side_gap_segment_payload(
    segment: Dict[str, Any],
    side: str,
    segment_index: int,
    selected_center_x_m: Optional[float],
    side_candidate: Dict[str, Any],
    pose: Tuple[float, float, float],
    current_anchor_progress_m: Optional[float],
    corridor_half_width_m: float,
    is_selected: bool,
) -> Dict[str, Any]:
    start_x = segment.get("start_x_m")
    end_x = segment.get("end_x_m")
    center_x = segment.get("center_x_m")
    width = segment.get("width_m")
    opening_center = segment.get("opening_center_base_xy")
    if isinstance(opening_center, list) and len(opening_center) == 2 and all(finite_number(v) for v in opening_center):
        center_base_xy = [float(opening_center[0]), float(opening_center[1])]
    elif finite_number(center_x):
        center_base_xy = [float(center_x), float(corridor_half_width_m) if side == "left" else -float(corridor_half_width_m)]
    else:
        center_base_xy = None
    center_odom_xy = transform_base_xy(center_base_xy, pose) if center_base_xy is not None else None
    free_ratio = None
    counts = side_candidate.get("side_probe_counts") if isinstance(side_candidate.get("side_probe_counts"), dict) else {}
    if finite_number(counts.get("free_ratio")):
        free_ratio = float(counts["free_ratio"])
    return {
        "segment_index": segment_index,
        "side": side,
        "start_x_m": float(start_x) if finite_number(start_x) else None,
        "end_x_m": float(end_x) if finite_number(end_x) else None,
        "center_x_m": float(center_x) if finite_number(center_x) else None,
        "opening_width_m": float(width) if finite_number(width) else None,
        "start_progress_m": float(current_anchor_progress_m) + float(start_x)
        if finite_number(current_anchor_progress_m) and finite_number(start_x)
        else None,
        "end_progress_m": float(current_anchor_progress_m) + float(end_x)
        if finite_number(current_anchor_progress_m) and finite_number(end_x)
        else None,
        "center_progress_m": float(current_anchor_progress_m) + float(center_x)
        if finite_number(current_anchor_progress_m) and finite_number(center_x)
        else None,
        "center_base_xy": center_base_xy,
        "center_odom_xy": center_odom_xy,
        "geometry_pass": bool(side_candidate.get("geometry_pass")),
        "soft_geometry_pass": bool(side_candidate.get("soft_geometry_pass")),
        "free_ratio": free_ratio,
        "width_plausible": bool(segment.get("width_plausible")),
        "is_selected": is_selected,
        "selected_distance_to_this_segment_m": abs(float(center_x) - float(selected_center_x_m))
        if finite_number(center_x) and finite_number(selected_center_x_m)
        else None,
    }


def segment_center_x_present(segments: Sequence[Dict[str, Any]], center_x_m: Optional[float]) -> bool:
    return bool(
        finite_number(center_x_m)
        and any(
            finite_number(segment.get("center_x_m"))
            and abs(float(segment["center_x_m"]) - float(center_x_m)) <= 0.15
            for segment in segments
        )
    )


def build_side_gap_segment_switch_audit_events(
    doorway: Dict[str, Any],
    pose: Tuple[float, float, float],
    anchor: Optional[Dict[str, Any]],
    iteration: int,
    observation_phase: str,
    room_zone_reached: bool,
    args: argparse.Namespace,
    previous_selected_by_side: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not room_zone_reached or anchor is None:
        return []
    current_progress = anchor_metrics(anchor, pose).get("anchor_progress_m")
    profile_root = doorway.get("doorway_geometry_profile") if isinstance(doorway.get("doorway_geometry_profile"), dict) else {}
    events: List[Dict[str, Any]] = []
    threshold = float(args.side_gap_switch_threshold_m)
    for side in ("left", "right"):
        profile = profile_root.get(side) if isinstance(profile_root.get(side), dict) else {}
        all_segments_raw = profile.get("open_segments") if isinstance(profile.get("open_segments"), list) else None
        selected_raw = profile.get("selected_opening") if isinstance(profile.get("selected_opening"), dict) else None
        side_candidate = doorway.get(f"{side}_candidate") if isinstance(doorway.get(f"{side}_candidate"), dict) else {}
        selected_center_x = selected_raw.get("center_x_m") if isinstance(selected_raw, dict) else None
        all_segments_available = isinstance(all_segments_raw, list)
        all_segments = [
            side_gap_segment_payload(
                segment,
                side,
                index,
                selected_center_x,
                side_candidate,
                pose,
                current_progress,
                args.door_landmark_corridor_half_width_m,
                bool(
                    isinstance(selected_raw, dict)
                    and finite_number(segment.get("center_x_m"))
                    and finite_number(selected_center_x)
                    and abs(float(segment["center_x_m"]) - float(selected_center_x)) <= 1e-6
                ),
            )
            for index, segment in enumerate(all_segments_raw or [])
            if isinstance(segment, dict)
        ]
        selected_index = next((item["segment_index"] for item in all_segments if item.get("is_selected")), None)
        selected_segment = next((item for item in all_segments if item.get("is_selected")), None)
        if selected_segment is None and isinstance(selected_raw, dict):
            selected_segment = side_gap_segment_payload(
                selected_raw,
                side,
                selected_index if selected_index is not None else -1,
                selected_center_x,
                side_candidate,
                pose,
                current_progress,
                args.door_landmark_corridor_half_width_m,
                True,
            )
        if isinstance(selected_segment, dict):
            selected_segment["selected_index"] = selected_index
            selected_segment["selected_reason"] = profile.get("selected_reason")
            selected_segment["source"] = "doorway_geometry_profile.selected_opening"
        previous = previous_selected_by_side.get(side)
        previous = previous if isinstance(previous, dict) else {}
        previous_selected = previous.get("selected_segment") or {}
        previous_selected = previous_selected if isinstance(previous_selected, dict) else {}
        previous_all = previous.get("all_segments") or []
        previous_all = previous_all if isinstance(previous_all, list) else []
        previous_center_progress = previous_selected.get("center_progress_m")
        current_center_progress = selected_segment.get("center_progress_m") if isinstance(selected_segment, dict) else None
        jump = (
            abs(float(current_center_progress) - float(previous_center_progress))
            if finite_number(current_center_progress) and finite_number(previous_center_progress)
            else None
        )
        switch_detected = bool(finite_number(jump) and float(jump) > threshold)
        if not all_segments_available:
            possible_reason = "all_segments_unavailable"
        elif not switch_detected:
            possible_reason = "same_segment_continuation"
        elif segment_center_x_present(previous_all, selected_segment.get("center_x_m") if selected_segment else None):
            possible_reason = "far_segment_existed_previous_frame"
        elif not segment_center_x_present(all_segments, previous_selected.get("center_x_m") if isinstance(previous_selected, dict) else None):
            possible_reason = "previous_near_segment_missing"
        elif previous and previous.get("selected_segment_index") != selected_index:
            possible_reason = "selected_index_changed"
        else:
            possible_reason = "unknown"
        event = {
            "iteration": iteration,
            "observation_phase": observation_phase,
            "timestamp_sec": doorway.get("grid_stamp_sec") if finite_number(doorway.get("grid_stamp_sec")) else time.time(),
            "timestamp_source": "doorway_grid_stamp" if finite_number(doorway.get("grid_stamp_sec")) else "wall_time_fallback",
            "current_anchor_progress_m": current_progress,
            "room_zone_start_progress_m": float(args.room_zone_start_progress_m),
            "room_zone_reached": True,
            "robot_pose": {"x": pose[0], "y": pose[1], "yaw": pose[2]},
            "coordinate_frame": "team_livox_odom",
            "side": side,
            "all_segments_available": all_segments_available,
            "missing_reason": None if all_segments_available else "detector_does_not_expose_all_segments",
            "selected_segment": selected_segment,
            "all_segments": all_segments,
            "previous_selected_center_progress_m": previous_center_progress,
            "selected_center_progress_jump_m": jump,
            "selected_switch_detected": switch_detected,
            "selected_switch_threshold_m": threshold,
            "previous_selected_segment_index": previous.get("selected_segment_index") if previous else None,
            "current_selected_segment_index": selected_index,
            "possible_switch_reason": possible_reason,
        }
        previous_selected_by_side[side] = {
            "selected_segment": selected_segment,
            "selected_segment_index": selected_index,
            "all_segments": all_segments,
            "event": event,
        }
        events.append(event)
    return events


def side_gap_segment_switch_summary(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    switches = [event for event in events if event.get("selected_switch_detected")]
    left_switches = [event for event in switches if event.get("side") == "left"]
    right_switches = [event for event in switches if event.get("side") == "right"]

    def max_jump(items: Sequence[Dict[str, Any]]) -> Optional[float]:
        values = [float(item["selected_center_progress_jump_m"]) for item in items if finite_number(item.get("selected_center_progress_jump_m"))]
        return max(values) if values else None

    def first_large(items: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not items:
            return None
        current_index, current_event = next(
            (
                (index, event)
                for index, event in enumerate(events)
                if event.get("side") == items[0].get("side") and event.get("selected_switch_detected")
            ),
            (None, None),
        )
        if current_event is None or current_index is None:
            return None
        previous_event = next(
            (
                event
                for event in reversed(events[:current_index])
                if event.get("side") == current_event.get("side")
            ),
            None,
        )
        return {
            "previous_event": previous_event,
            "current_event": current_event,
            "previous_selected_segment": previous_event.get("selected_segment") if previous_event else None,
            "current_selected_segment": current_event.get("selected_segment"),
            "previous_all_segments": previous_event.get("all_segments") if previous_event else None,
            "current_all_segments": current_event.get("all_segments"),
            "inferred_cause": current_event.get("possible_switch_reason"),
        }

    return {
        "left_switch_count": len(left_switches),
        "right_switch_count": len(right_switches),
        "max_left_selected_center_progress_jump_m": max_jump(left_switches),
        "max_right_selected_center_progress_jump_m": max_jump(right_switches),
        "first_large_left_switch": first_large(left_switches),
        "first_large_right_switch": first_large(right_switches),
        "audit_event_count": len(events),
    }


def visual_candidate_from_segment_switch_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    segment = event.get("selected_segment") if isinstance(event.get("selected_segment"), dict) else None
    if segment is None:
        return None
    candidate = dict(segment)
    candidate.update(
        {
            "iteration": event.get("iteration"),
            "observation_phase": event.get("observation_phase"),
            "timestamp_sec": event.get("timestamp_sec"),
            "room_zone_reached": event.get("room_zone_reached"),
            "room_zone_start_progress_m": event.get("room_zone_start_progress_m"),
            "current_anchor_progress_m": event.get("current_anchor_progress_m"),
            "robot_pose": event.get("robot_pose"),
            "coordinate_frame": event.get("coordinate_frame"),
            "source": segment.get("source", "side_gap_segment_switch_audit"),
        }
    )
    return candidate


def image_stamp_sec(msg: Optional[Image]) -> Optional[float]:
    if msg is None:
        return None
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    try:
        return float(stamp.to_sec()) if stamp is not None else None
    except Exception:
        return None


def ros_rgb_to_pil(msg: Image) -> Any:
    from PIL import Image as PILImage

    encoding = (msg.encoding or "").lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding)
    if channels is None:
        raise RuntimeError(f"unsupported_rgb_encoding:{msg.encoding}")
    raw = bytes(msg.data)
    rows = []
    row_bytes = int(msg.width) * channels
    step = int(msg.step) or row_bytes
    for row in range(int(msg.height)):
        rows.append(raw[row * step : row * step + row_bytes])
    packed = b"".join(rows)
    mode = "L" if channels == 1 else ("RGBA" if channels == 4 else "RGB")
    image = PILImage.frombytes(mode, (int(msg.width), int(msg.height)), packed)
    if encoding == "bgr8":
        r, g, b = image.split()
        image = PILImage.merge("RGB", (b, g, r))
    elif encoding == "bgra8":
        r, g, b, a = image.split()
        image = PILImage.merge("RGBA", (b, g, r, a)).convert("RGB")
    return image.convert("RGB")


def ros_depth_to_colormap(msg: Image) -> Any:
    from PIL import Image as PILImage

    encoding = (msg.encoding or "").lower()
    dtype = np.uint16 if encoding in {"16uc1", "mono16"} else (np.float32 if encoding == "32fc1" else None)
    if dtype is None:
        raise RuntimeError(f"unsupported_depth_encoding:{msg.encoding}")
    row_values = int(msg.width)
    step = int(msg.step) or row_values * np.dtype(dtype).itemsize
    raw = bytes(msg.data)
    rows = [
        np.frombuffer(raw[row * step : row * step + row_values * np.dtype(dtype).itemsize], dtype=dtype)
        for row in range(int(msg.height))
    ]
    depth = np.vstack(rows).astype(np.float32)
    valid = depth[np.isfinite(depth)]
    valid = valid[valid > 0]
    if valid.size == 0:
        normalized = np.zeros(depth.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(valid, [2, 98])
        normalized = np.clip((depth - low) * 255.0 / max(high - low, 1e-6), 0, 255).astype(np.uint8)
    # Lightweight blue-to-red pseudo-color, avoiding an OpenCV dependency.
    color = np.stack((normalized, 255 - np.abs(normalized.astype(np.int16) - 128) * 2, 255 - normalized), axis=-1)
    return PILImage.fromarray(np.clip(color, 0, 255).astype(np.uint8), "RGB")


def ros_depth_to_meters(msg: Image) -> np.ndarray:
    encoding = (msg.encoding or "").lower()
    dtype = np.uint16 if encoding in {"16uc1", "mono16"} else (np.float32 if encoding == "32fc1" else None)
    if dtype is None:
        raise RuntimeError(f"unsupported_depth_encoding:{msg.encoding}")
    row_values = int(msg.width)
    step = int(msg.step) or row_values * np.dtype(dtype).itemsize
    raw = bytes(msg.data)
    rows = [
        np.frombuffer(raw[row * step : row * step + row_values * np.dtype(dtype).itemsize], dtype=dtype)
        for row in range(int(msg.height))
    ]
    depth = np.vstack(rows).astype(np.float32)
    if dtype == np.uint16:
        depth *= 0.001
    return depth


def forced_entry_front_clearance(
    visual_audit: Optional["SideGapVisualCoordinateAudit"],
    minimum_clearance_m: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "pass": False,
        "minimum_required_m": float(minimum_clearance_m),
    }
    if visual_audit is None or visual_audit.latest_depth is None:
        result["reject_reason"] = "front_depth_unavailable"
        return result
    try:
        depth = ros_depth_to_meters(visual_audit.latest_depth)
        height, width = depth.shape
        x0, x1 = int(width * 0.30), max(int(width * 0.70), int(width * 0.30) + 1)
        y0, y1 = int(height * 0.25), max(int(height * 0.70), int(height * 0.25) + 1)
        roi = depth[y0:y1, x0:x1]
        valid = roi[np.isfinite(roi)]
        valid = valid[(valid > 0.05) & (valid < 20.0)]
        result.update(
            {
                "available": bool(valid.size),
                "encoding": visual_audit.latest_depth.encoding,
                "image_size": [int(width), int(height)],
                "roi_xyxy": [x0, y0, x1, y1],
                "valid_pixel_count": int(valid.size),
            }
        )
        if valid.size == 0:
            result["reject_reason"] = "front_depth_has_no_valid_pixels"
            return result
        clearance = float(np.percentile(valid, 20))
        result["clearance_percentile"] = 20
        result["front_clearance_m"] = clearance
        result["median_depth_m"] = float(np.median(valid))
        result["pass"] = clearance >= float(minimum_clearance_m)
        result["reject_reason"] = None if result["pass"] else "front_clearance_below_threshold"
    except Exception as exc:
        result["reject_reason"] = "front_depth_parse_failed"
        result["error"] = repr(exc)
    return result


class SideGapVisualCoordinateAudit:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.latest_rgb: Optional[Image] = None
        self.latest_depth: Optional[Image] = None
        self.events: List[Dict[str, Any]] = []
        self.subscription_errors: List[str] = []
        try:
            rospy.Subscriber(args.side_gap_visual_audit_rgb_topic, Image, self._rgb_cb, queue_size=1)
        except Exception as exc:
            self.subscription_errors.append(f"rgb_subscribe_failed:{exc}")
        try:
            rospy.Subscriber(args.side_gap_visual_audit_depth_topic, Image, self._depth_cb, queue_size=1)
        except Exception as exc:
            self.subscription_errors.append(f"depth_subscribe_failed:{exc}")

    def _rgb_cb(self, msg: Image) -> None:
        self.latest_rgb = msg

    def _depth_cb(self, msg: Image) -> None:
        self.latest_depth = msg

    def capture(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if len(self.events) >= int(self.args.side_gap_visual_audit_max_events):
            return None
        broad = bool(
            finite_number(candidate.get("start_x_m"))
            and finite_number(candidate.get("end_x_m"))
            and finite_number(candidate.get("opening_width_m"))
            and float(candidate["start_x_m"]) <= 0.35
            and float(candidate["end_x_m"]) >= 2.8
            and float(candidate["opening_width_m"]) >= 2.4
        )
        if self.args.side_gap_visual_audit_broad_only and not broad:
            return None
        event_id = len(self.events) + 1
        event_dir = SIDE_GAP_VISUAL_AUDIT_DIR / f"event_{event_id:06d}"
        event_dir.mkdir(parents=True, exist_ok=True)
        raw_pose = candidate.get("robot_pose") if isinstance(candidate.get("robot_pose"), dict) else None
        pose_available = bool(
            isinstance(raw_pose, dict)
            and all(finite_number(raw_pose.get(key)) for key in ("x", "y", "yaw"))
        )
        pose = (
            {
                "frame": raw_pose.get("frame", candidate.get("coordinate_frame", "team_livox_odom")),
                "x": float(raw_pose["x"]),
                "y": float(raw_pose["y"]),
                "yaw": float(raw_pose["yaw"]),
            }
            if pose_available and isinstance(raw_pose, dict)
            else None
        )
        pose_tuple_value = (float(pose["x"]), float(pose["y"]), float(pose["yaw"])) if pose else None
        center_base = candidate.get("center_base_xy") if isinstance(candidate.get("center_base_xy"), list) else None
        lateral = float(center_base[1]) if center_base and len(center_base) == 2 and finite_number(center_base[1]) else 0.0
        start_base = [float(candidate["start_x_m"]), lateral] if finite_number(candidate.get("start_x_m")) else None
        end_base = [float(candidate["end_x_m"]), lateral] if finite_number(candidate.get("end_x_m")) else None
        event_timestamp = candidate.get("timestamp_sec") if finite_number(candidate.get("timestamp_sec")) else time.time()
        rgb_saved = False
        depth_saved = False
        errors: List[str] = list(self.subscription_errors)
        rgb_path = None
        depth_path = None
        try:
            if self.latest_rgb is None:
                raise RuntimeError("rgb_message_unavailable")
            rgb_path = event_dir / "rgb.png"
            ros_rgb_to_pil(self.latest_rgb).save(rgb_path)
            rgb_saved = True
        except Exception as exc:
            errors.append(f"rgb_capture_failed:{exc}")
        try:
            if self.latest_depth is None:
                raise RuntimeError("depth_message_unavailable")
            depth_path = event_dir / "depth_colormap.png"
            ros_depth_to_colormap(self.latest_depth).save(depth_path)
            depth_saved = True
        except Exception as exc:
            errors.append(f"depth_capture_failed:{exc}")
        image_time = image_stamp_sec(self.latest_rgb)
        event = {
            "event_id": event_id,
            "iteration": candidate.get("iteration"),
            "observation_phase": candidate.get("observation_phase"),
            "timestamp_sec": event_timestamp,
            "timestamp_source": "side_gap_candidate_timestamp",
            "image_timestamp_sec": image_time,
            "image_time_delta_sec": abs(float(image_time) - float(event_timestamp)) if image_time is not None else None,
            "image_sync_warning": bool(image_time is not None and abs(float(image_time) - float(event_timestamp)) > 1.0),
            "room_zone_reached": candidate.get("room_zone_reached"),
            "room_zone_start_progress_m": candidate.get("room_zone_start_progress_m"),
            "current_anchor_progress_m": candidate.get("current_anchor_progress_m"),
            "robot_pose": pose,
            "robot_pose_available": pose_available,
            "robot_pose_missing_reason": None if pose_available else "candidate_robot_pose_unavailable",
            "coordinate_frame": candidate.get("coordinate_frame", "team_livox_odom"),
            "side": candidate.get("side"),
            "source": candidate.get("source"),
            "start_x_m": candidate.get("start_x_m"),
            "center_x_m": candidate.get("center_x_m"),
            "end_x_m": candidate.get("end_x_m"),
            "opening_width_m": candidate.get("opening_width_m"),
            "start_progress_m": candidate.get("start_progress_m"),
            "center_progress_m": candidate.get("center_progress_m"),
            "end_progress_m": candidate.get("end_progress_m"),
            "start_odom_xy": transform_base_xy(start_base, pose_tuple_value) if start_base and pose_tuple_value else None,
            "center_odom_xy": candidate.get("center_odom_xy"),
            "end_odom_xy": transform_base_xy(end_base, pose_tuple_value) if end_base and pose_tuple_value else None,
            "center_base_xy": center_base,
            "geometry_pass": candidate.get("geometry_pass"),
            "soft_geometry_pass": candidate.get("soft_geometry_pass"),
            "width_plausible": candidate.get("width_plausible"),
            "free_ratio": candidate.get("free_ratio"),
            "window_limited_broad_opening": broad,
            "D_raw_xy": candidate.get("D_raw_xy"),
            "D_raw_progress_m": candidate.get("D_raw_progress_m"),
            "D_nav_xy": candidate.get("D_nav_xy"),
            "D_nav_progress_m": candidate.get("D_nav_progress_m"),
            "D_nav_in_front": candidate.get("D_nav_in_front"),
            "D_nav_distance_m": candidate.get("D_nav_distance_m"),
            "D_nav_progress_delta_m": candidate.get("D_nav_progress_delta_m"),
            "rgb_saved": rgb_saved,
            "rgb_path": str(rgb_path) if rgb_path else None,
            "depth_saved": depth_saved,
            "depth_path": str(depth_path) if depth_path else None,
            "visual_available": rgb_saved,
            "missing_reason": ";".join(errors) if errors else None,
            "diagnostic_only": True,
        }
        try:
            write_json(event_dir / "event.json", event)
        except Exception as exc:
            event["event_write_error"] = str(exc)
        self.events.append(event)
        return event

    def manifest(self) -> Dict[str, Any]:
        broad_events = [event for event in self.events if event.get("window_limited_broad_opening")]
        room_zone_events = [event for event in self.events if event.get("room_zone_reached")]
        visual_available = self.latest_rgb is not None
        return {
            "diagnostic_only": True,
            "controls_robot": False,
            "controls_next_state": False,
            "writes_real_target": False,
            "calls_runner": False,
            "visual_available": visual_available,
            "rgb_topic": self.args.side_gap_visual_audit_rgb_topic,
            "depth_topic": self.args.side_gap_visual_audit_depth_topic,
            "event_count": len(self.events),
            "broad_opening_event_count": len(broad_events),
            "room_zone_reached_event_count": len(room_zone_events),
            "first_broad_opening_event": broad_events[0] if broad_events else None,
            "events_summary": self.events,
            "subscription_errors": self.subscription_errors,
            "missing_reason": None
            if visual_available
            else (";".join(self.subscription_errors) if self.subscription_errors else "rgb_message_unavailable"),
        }


def build_local_free_space_entry_debug(
    args: argparse.Namespace,
    pose: Tuple[float, float, float],
    room_zone_reached: bool,
    progress: Optional[float],
) -> Dict[str, Any]:
    grid_wait_timeout_sec = float(getattr(args, "local_entry_grid_timeout_sec", 0.4))

    def empty_candidate(side: str, reject_reason: str) -> Dict[str, Any]:
        y_bounds = (
            [float(args.local_entry_lateral_min_m), float(args.local_entry_lateral_max_m)]
            if side == "left"
            else [-float(args.local_entry_lateral_max_m), -float(args.local_entry_lateral_min_m)]
        )
        return {
            "side": side,
            "sector_base_bounds_xy": {"x": [float(args.local_entry_forward_min_m), float(args.local_entry_forward_max_m)], "y": y_bounds},
            "sector_cell_bounds_raw": None,
            "sector_cell_bounds_clipped": None,
            "sector_in_grid": False,
            "sector_raw_bounds_intersects_grid": False,
            "sector_out_of_grid_reason": "grid_unavailable",
            "sector_total_cell_count": 0,
            "sector_free_cell_count": 0,
            "sector_occupied_cell_count": 0,
            "sector_unknown_cell_count": 0,
            "sector_other_cell_count": 0,
            "sector_value_histogram": {},
            "component_count": 0,
            "valid_component_count": 0,
            "selected_component": None,
            "E_entry_base_xy": None,
            "E_entry_odom_xy": None,
            "E_entry_is_free": False,
            "path_exists": False,
            "entry_target_valid": False,
            "reject_reason": reject_reason,
        }

    base = {
        "diagnostic_only": True,
        "controls_robot": False,
        "controls_next_state": False,
        "writes_real_target": False,
        "calls_runner": False,
        "room_zone_reached": room_zone_reached,
        "current_anchor_progress_m": progress,
        "timestamp_sec": time.time(),
        "robot_pose": {"frame": "team_livox_odom", "x": pose[0], "y": pose[1], "yaw": pose[2]},
        "grid_source": GRID_TOPIC,
        "grid_wait_timeout_sec": grid_wait_timeout_sec,
        "grid_frame": None,
        "grid_width": None,
        "grid_height": None,
        "grid_resolution_m": None,
        "grid_origin": None,
        "coordinate_frame": None,
        "robot_cell": None,
        "robot_cell_in_grid": False,
        "entry_target_valid": False,
    }
    if not room_zone_reached:
        return {
            **base,
            "grid_available": False,
            "left_entry_candidate": empty_candidate("left", "room_zone_not_reached"),
            "right_entry_candidate": empty_candidate("right", "room_zone_not_reached"),
            "selected_entry_candidate": None,
            "first_valid_entry_target": None,
            "reject_reasons": ["room_zone_not_reached"],
        }
    try:
        msg = read_grid(timeout_sec=grid_wait_timeout_sec)
        grid = grid_array(msg)
    except Exception as exc:
        return {
            **base,
            "grid_available": False,
            "missing_reason": f"local_traversability_grid_not_found:{exc}",
            "left_entry_candidate": empty_candidate("left", "local_traversability_grid_not_found"),
            "right_entry_candidate": empty_candidate("right", "local_traversability_grid_not_found"),
            "selected_entry_candidate": None,
            "first_valid_entry_target": None,
            "reject_reasons": ["local_traversability_grid_not_found"],
        }

    resolution = float(msg.info.resolution)
    origin_x = float(msg.info.origin.position.x)
    origin_y = float(msg.info.origin.position.y)
    coordinate_frame = msg.header.frame_id or "team_livox_odom"
    grid_frame_is_base = coordinate_frame in {"base", "base_link"}

    def grid_xy_to_cell(x: float, y: float) -> Optional[Tuple[int, int]]:
        return metric_to_cell(x, y, grid_metadata(msg))

    robot_cell = grid_xy_to_cell(0.0, 0.0) if grid_frame_is_base else grid_xy_to_cell(pose[0], pose[1])
    robot_cell_in_grid = robot_cell is not None

    def cell_xy(x_index: int, y_index: int) -> Tuple[float, float]:
        point = cell_to_metric(x_index, y_index, grid_metadata(msg))
        if point is None:
            raise ValueError("cell_out_of_bounds")
        return point

    def base_xy(x_index: int, y_index: int) -> Tuple[float, float]:
        return cell_xy(x_index, y_index) if grid_frame_is_base else target_base_xy(cell_xy(x_index, y_index), pose)

    def is_free(cell: Tuple[int, int]) -> bool:
        return cell is not None and 0 <= cell[0] < grid.shape[1] and 0 <= cell[1] < grid.shape[0] and grid[cell[1], cell[0]] == 0

    def bfs(start: Tuple[int, int], allowed: Optional[set] = None) -> Dict[Tuple[int, int], Optional[Tuple[int, int]]]:
        queue = [start]
        previous = {start: None}
        while queue:
            x_index, y_index = queue.pop(0)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_cell = (x_index + dx, y_index + dy)
                if is_free(next_cell) and next_cell not in previous and (allowed is None or next_cell in allowed):
                    previous[next_cell] = (x_index, y_index)
                    queue.append(next_cell)
        return previous

    reachable = bfs(robot_cell) if is_free(robot_cell) else {}

    def cell_clearance_m(cell: Tuple[int, int], max_radius_cells: int = 12) -> Optional[float]:
        for radius in range(1, max_radius_cells + 1):
            for dx in range(-radius, radius + 1):
                for dy in (-radius, radius):
                    x_index, y_index = cell[0] + dx, cell[1] + dy
                    if 0 <= x_index < grid.shape[1] and 0 <= y_index < grid.shape[0] and grid[y_index, x_index] != 0:
                        return radius * resolution
            for dy in range(-radius + 1, radius):
                for dx in (-radius, radius):
                    x_index, y_index = cell[0] + dx, cell[1] + dy
                    if 0 <= x_index < grid.shape[1] and 0 <= y_index < grid.shape[0] and grid[y_index, x_index] != 0:
                        return radius * resolution
        return None

    def side_candidate(side: str) -> Dict[str, Any]:
        y_bounds = (
            (float(args.local_entry_lateral_min_m), float(args.local_entry_lateral_max_m))
            if side == "left"
            else (-float(args.local_entry_lateral_max_m), -float(args.local_entry_lateral_min_m))
        )
        x_bounds = (float(args.local_entry_forward_min_m), float(args.local_entry_forward_max_m))
        raw_bounds = {
            "x_index": [int(math.floor((x_bounds[0] - origin_x) / resolution)), int(math.floor((x_bounds[1] - origin_x) / resolution))],
            "y_index": [int(math.floor((y_bounds[0] - origin_y) / resolution)), int(math.floor((y_bounds[1] - origin_y) / resolution))],
        }
        clipped_bounds = {
            "x_index": [max(0, raw_bounds["x_index"][0]), min(grid.shape[1] - 1, raw_bounds["x_index"][1])],
            "y_index": [max(0, raw_bounds["y_index"][0]), min(grid.shape[0] - 1, raw_bounds["y_index"][1])],
        }
        raw_intersects_grid = bool(
            raw_bounds["x_index"][1] >= 0
            and raw_bounds["x_index"][0] < grid.shape[1]
            and raw_bounds["y_index"][1] >= 0
            and raw_bounds["y_index"][0] < grid.shape[0]
        )
        x_out = raw_bounds["x_index"][1] < 0 or raw_bounds["x_index"][0] >= grid.shape[1]
        y_out = raw_bounds["y_index"][1] < 0 or raw_bounds["y_index"][0] >= grid.shape[0]
        out_of_grid_reason = (
            "x_and_y_out_of_grid" if x_out and y_out else "x_out_of_grid" if x_out else "y_out_of_grid" if y_out else None
        )
        sector_cells: List[Tuple[int, int]] = []
        sector_free_cells: set = set()
        values: List[int] = []
        for y_index in range(grid.shape[0]):
            for x_index in range(grid.shape[1]):
                base_x, base_y = base_xy(x_index, y_index)
                if x_bounds[0] <= base_x <= x_bounds[1] and y_bounds[0] <= base_y <= y_bounds[1]:
                    sector_cells.append((x_index, y_index))
                    value = int(grid[y_index, x_index])
                    values.append(value)
                    if value == 0:
                        sector_free_cells.add((x_index, y_index))
        histogram: Dict[str, int] = {}
        for value in values:
            histogram[str(value)] = histogram.get(str(value), 0) + 1
        sector_bounds = None
        if sector_cells:
            x_indices = [cell[0] for cell in sector_cells]
            y_indices = [cell[1] for cell in sector_cells]
            sector_bounds = {"x_index": [min(x_indices), max(x_indices)], "y_index": [min(y_indices), max(y_indices)]}
        sector_summary = {
            "sector_base_bounds_xy": {"x": list(x_bounds), "y": list(y_bounds)},
            "sector_cell_bounds_raw": raw_bounds,
            "sector_cell_bounds_clipped": clipped_bounds if raw_intersects_grid else None,
            "sector_cell_bounds": sector_bounds,
            "sector_in_grid": bool(sector_cells),
            "sector_raw_bounds_intersects_grid": raw_intersects_grid,
            "sector_out_of_grid_reason": out_of_grid_reason,
            "sector_total_cell_count": len(sector_cells),
            "sector_free_cell_count": len(sector_free_cells),
            "sector_occupied_cell_count": sum(1 for value in values if value > 0),
            "sector_unknown_cell_count": sum(1 for value in values if value == -1),
            "sector_other_cell_count": sum(1 for value in values if value != 0 and value != -1 and value <= 0),
            "sector_value_histogram": histogram,
            "free_value_definition": "value == 0",
            "occupied_value_definition": "value > 0",
            "unknown_value_definition": "value == -1",
        }
        components: List[Dict[str, Any]] = []
        remaining = set(sector_free_cells)
        while remaining:
            seed = next(iter(remaining))
            component = set(bfs(seed, remaining))
            remaining -= component
            points = [base_xy(row, col) for row, col in component]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            nearest_cell = min(component, key=lambda cell: math.hypot(*base_xy(*cell)))
            farthest_lateral_cell = max(component, key=lambda cell: abs(base_xy(*cell)[1]))
            is_valid = bool(
                len(component) * resolution * resolution >= float(args.local_entry_min_area_m2)
                and max(xs) - min(xs) >= float(args.local_entry_min_forward_extent_m)
                and max(ys) - min(ys) >= float(args.local_entry_min_lateral_extent_m)
            )
            components.append(
                {
                    "cells": component,
                    "component_cell_count": len(component),
                    "component_area_m2": len(component) * resolution * resolution,
                    "component_bounds_base_xy": [[min(xs), min(ys)], [max(xs), max(ys)]],
                    "forward_extent_m": max(xs) - min(xs),
                    "lateral_extent_m": max(ys) - min(ys),
                    "centroid_base_xy": [sum(xs) / len(xs), sum(ys) / len(ys)],
                    "nearest_point_base_xy": list(base_xy(*nearest_cell)),
                    "farthest_lateral_point_base_xy": list(base_xy(*farthest_lateral_cell)),
                    "valid": is_valid,
                }
            )
        valid_components = [component for component in components if component["valid"]]
        selected = max(valid_components, key=lambda component: component["component_area_m2"]) if valid_components else None
        if not raw_intersects_grid or not sector_cells:
            reject_reason = "sector_out_of_grid_or_empty"
        elif not sector_free_cells:
            reject_reason = "no_free_cells_in_sector"
        elif not components:
            reject_reason = "free_cells_not_connected_or_component_extraction_failed"
        elif not valid_components:
            reject_reason = "components_below_threshold"
        else:
            reject_reason = None
        output = {
            "side": side,
            "search_sector_base_xy": sector_summary["sector_base_bounds_xy"],
            **sector_summary,
            "component_count": len(components),
            "valid_component_count": len(valid_components),
            "components_summary": [{key: value for key, value in component.items() if key != "cells"} for component in components],
            "selected_component": None,
            "E_entry_base_xy": None,
            "E_entry_odom_xy": None,
            "E_entry_is_free": False,
            "path_exists": False,
            "entry_target_valid": False,
            "reject_reason": reject_reason,
        }
        if selected is None:
            return output
        ideal = (0.8, 1.0 if side == "left" else -1.0)
        entry_cell = min(
            selected["cells"],
            key=lambda cell: (base_xy(*cell)[0] - ideal[0]) ** 2 + (base_xy(*cell)[1] - ideal[1]) ** 2,
        )
        path: List[Tuple[int, int]] = []
        if entry_cell in reachable:
            current = entry_cell
            while current is not None:
                path.append(current)
                current = reachable[current]
            path.reverse()
        known_clearances = [value for value in (cell_clearance_m(cell) for cell in path) if value is not None]
        entry_base_xy = base_xy(*entry_cell)
        entry_odom_xy = transform_base_xy(entry_base_xy, pose) if grid_frame_is_base else list(cell_xy(*entry_cell))
        output.update(
            {
                "selected_component": {key: value for key, value in selected.items() if key != "cells"},
                "E_entry_base_xy": list(entry_base_xy),
                "E_entry_odom_xy": entry_odom_xy,
                "E_entry_frame": "team_livox_odom",
                "E_entry_distance_m": math.hypot(*entry_base_xy),
                "E_entry_side": side,
                "selected_component_index": components.index(selected),
                "selection_reason": "nearest_component_free_cell_to_side_entry_ideal",
                "E_entry_is_free": True,
                "path_exists": bool(path),
                "path_length_m": max(0, len(path) - 1) * resolution,
                "path_cell_count": len(path),
                "path_min_clearance_m": min(known_clearances) if known_clearances else None,
                "path_clearance_limited_to_m": 12 * resolution,
                "path_reject_reason": None if path else "robot_cell_not_connected",
                "entry_target_valid": bool(path),
                "reject_reason": None if path else "path_not_connected",
                "path_cells": path,
            }
        )
        return output

    left = side_candidate("left")
    right = side_candidate("right")
    valid_candidates = [candidate for candidate in (left, right) if candidate.get("entry_target_valid")]
    payload = {
        **base,
        "grid_available": True,
        "grid_frame": msg.header.frame_id,
        "coordinate_frame": coordinate_frame,
        "coordinate_frame_assumption": "base_grid_uses_origin_and_resolution_directly" if grid_frame_is_base else "local_grid_and_robot_odom_share_the_existing_target_transform_frame",
        "grid_width": int(msg.info.width),
        "grid_height": int(msg.info.height),
        "grid_resolution_m": resolution,
        "grid_origin": {"x": origin_x, "y": origin_y},
        "grid_bounds_base_xy": (
            [[origin_x, origin_y], [origin_x + grid.shape[1] * resolution, origin_y + grid.shape[0] * resolution]]
            if grid_frame_is_base
            else None
        ),
        "robot_cell": list(robot_cell) if robot_cell is not None else None,
        "robot_cell_in_grid": robot_cell_in_grid,
        "left_entry_candidate": left,
        "right_entry_candidate": right,
        "selected_entry_candidate": valid_candidates[0] if valid_candidates else None,
        "entry_target_valid_count": len(valid_candidates),
        "first_valid_entry_target": valid_candidates[0] if valid_candidates else None,
        "entry_target_valid": bool(valid_candidates),
        "reject_reasons": [] if valid_candidates else [left.get("reject_reason"), right.get("reject_reason")],
    }
    try:
        from PIL import Image as PILImage, ImageDraw

        image = np.full((grid.shape[0], grid.shape[1], 3), (90, 90, 90), dtype=np.uint8)
        image[grid == 0] = (245, 245, 245)
        image[grid > 0] = (45, 45, 45)
        image = np.flipud(image)
        image_object = PILImage.fromarray(image)
        draw = ImageDraw.Draw(image_object)

        def image_cell(cell: Tuple[int, int]) -> Tuple[int, int]:
            return cell[0], grid.shape[0] - 1 - cell[1]

        if robot_cell_in_grid:
            rx, ry = image_cell(robot_cell)
            draw.ellipse((rx - 3, ry - 3, rx + 3, ry + 3), fill=(0, 180, 255))
        colors = {"left": (255, 160, 0), "right": (160, 0, 255)}
        for candidate in (left, right):
            bounds = candidate.get("sector_cell_bounds") or {}
            x_indices, y_indices = bounds.get("x_index"), bounds.get("y_index")
            if isinstance(x_indices, list) and isinstance(y_indices, list):
                top_left = image_cell((x_indices[0], y_indices[1]))
                bottom_right = image_cell((x_indices[1], y_indices[0]))
                draw.rectangle((top_left, bottom_right), outline=colors[candidate["side"]], width=2)
                draw.text((top_left[0] + 2, top_left[1] + 2), f"{candidate['side']} free={candidate['sector_free_cell_count']}", fill=colors[candidate["side"]])
        selected = payload.get("selected_entry_candidate")
        if isinstance(selected, dict) and selected.get("E_entry_odom_xy"):
            target_cell = (
                grid_xy_to_cell(*selected["E_entry_base_xy"])
                if grid_frame_is_base
                else grid_xy_to_cell(*selected["E_entry_odom_xy"])
            )
            if is_free(target_cell):
                tx, ty = image_cell(target_cell)
                draw.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), fill=(0, 220, 80))
            for cell in selected.get("path_cells", []):
                if isinstance(cell, (list, tuple)) and len(cell) == 2:
                    draw.point(image_cell((int(cell[0]), int(cell[1]))), fill=(0, 220, 80))
        reasons = f"L:{left.get('reject_reason')} R:{right.get('reject_reason')}"
        draw.text((4, 4), f"frame={coordinate_frame} res={resolution:.3f} robot={list(robot_cell) if robot_cell else None}", fill=(255, 255, 0))
        draw.text((4, 18), reasons, fill=(255, 255, 0))
        LOCAL_ENTRY_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        image_object.save(LOCAL_ENTRY_DEBUG_DIR / "local_grid_debug.png")
        payload["local_grid_debug_path"] = str(LOCAL_ENTRY_DEBUG_DIR / "local_grid_debug.png")
    except Exception as exc:
        payload["local_grid_debug_error"] = repr(exc)
    return payload


def write_local_entry_debug_event(
    event_index: int,
    payload: Dict[str, Any],
    visual_audit: Optional["SideGapVisualCoordinateAudit"],
) -> Dict[str, Any]:
    """Persist one diagnostic-only grid event; optional camera captures never gate the state machine."""
    event_dir = LOCAL_ENTRY_DEBUG_DIR / f"event_{event_index:06d}"
    event_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "event_id": f"local_entry_{event_index:06d}",
        "event_index": event_index,
        **payload,
        "entry_target_valid": bool(payload.get("entry_target_valid")),
        "first_valid_entry_target": payload.get("first_valid_entry_target") if payload.get("entry_target_valid") else None,
        "rgb_path": None,
        "depth_path": None,
    }
    if visual_audit is None:
        event["image_missing_reason"] = "side_gap_visual_coordinate_audit_disabled"
    else:
        try:
            if visual_audit.latest_rgb is not None:
                rgb_path = event_dir / "rgb.png"
                ros_rgb_to_pil(visual_audit.latest_rgb).save(rgb_path)
                event["rgb_path"] = str(rgb_path)
            if visual_audit.latest_depth is not None:
                depth_path = event_dir / "depth.png"
                ros_depth_to_colormap(visual_audit.latest_depth).save(depth_path)
                event["depth_path"] = str(depth_path)
            if event["rgb_path"] is None and event["depth_path"] is None:
                event["image_missing_reason"] = "visual_messages_unavailable"
        except Exception as exc:
            event["image_capture_error"] = repr(exc)
    write_json(event_dir / "event.json", event)
    return event


def select_forced_room_entry_opening(
    doorway: Dict[str, Any],
    room_zone_reached: bool,
    current_progress: Optional[float],
    args: argparse.Namespace,
    pending_opening: Optional[Dict[str, Any]] = None,
    latched_doorway_profile: Optional[Dict[str, Any]] = None,
    opening_observation_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "forced_entry_enabled": bool(args.enable_forced_room_entry_mvp),
        "room_zone_reached": bool(room_zone_reached),
        "current_anchor_progress_m": current_progress,
        "trigger_ready": False,
        "selected_opening": None,
        "pending_forced_opening": pending_opening if isinstance(pending_opening, dict) else None,
    }
    if not args.enable_forced_room_entry_mvp:
        status["reject_reason"] = "forced_entry_disabled"
        return status
    if not room_zone_reached:
        status["reject_reason"] = "room_zone_not_reached"
        return status
    if (
        args.forced_entry_max_trigger_progress_m is not None
        and finite_number(current_progress)
        and float(current_progress) > float(args.forced_entry_max_trigger_progress_m)
    ):
        status["pending_forced_opening"] = None
        status["reject_reason"] = "max_trigger_progress_exceeded"
        return status
    if not finite_number(current_progress):
        status["reject_reason"] = "anchor_progress_unavailable"
        return status

    if isinstance(pending_opening, dict) and finite_number(pending_opening.get("target_progress_m")):
        target_progress = float(pending_opening["target_progress_m"])
        alignment_error = target_progress - float(current_progress)
        status["alignment_target_anchor_progress_m"] = target_progress
        status["alignment_longitudinal_error_m"] = alignment_error
        if float(current_progress) > target_progress + float(args.forced_entry_alignment_overshoot_m):
            status["pending_forced_opening"] = None
            status["pending_clear_reason"] = "forced_opening_alignment_overshot"
        elif abs(alignment_error) <= float(args.forced_entry_alignment_tolerance_m):
            pending_side = str(pending_opening.get("side"))
            side_payload = doorway.get(f"{pending_side}_candidate")
            side_payload = side_payload if isinstance(side_payload, dict) else {}
            side_probe = side_payload.get("side_probe_counts")
            side_probe = side_probe if isinstance(side_probe, dict) else {}
            current_side_free_ratio = side_probe.get("free_ratio")
            profile_root = doorway.get("doorway_geometry_profile")
            profile_root = profile_root if isinstance(profile_root, dict) else {}
            current_side_profile = profile_root.get(pending_side)
            current_side_profile = current_side_profile if isinstance(current_side_profile, dict) else {}
            current_opening = current_side_profile.get("selected_opening")
            current_opening = current_opening if isinstance(current_opening, dict) else {}
            current_overlap_ratio = 0.0
            if (
                finite_number(current_opening.get("start_x_m"))
                and finite_number(current_opening.get("end_x_m"))
                and finite_number(pending_opening.get("start_progress_m"))
                and finite_number(pending_opening.get("end_progress_m"))
            ):
                current_start = float(current_progress) + float(current_opening["start_x_m"])
                current_end = float(current_progress) + float(current_opening["end_x_m"])
                overlap = max(
                    0.0,
                    min(current_end, float(pending_opening["end_progress_m"]))
                    - max(current_start, float(pending_opening["start_progress_m"])),
                )
                current_overlap_ratio = overlap / max(
                    1e-6,
                    min(
                        current_end - current_start,
                        float(pending_opening["end_progress_m"])
                        - float(pending_opening["start_progress_m"]),
                    ),
                )
            current_side_support = bool(
                (
                    finite_number(current_side_free_ratio)
                    and float(current_side_free_ratio)
                    >= float(args.forced_entry_trigger_min_side_free_ratio)
                )
                or current_overlap_ratio
                >= float(args.forced_entry_trigger_min_current_overlap_ratio)
            )
            status["trigger_current_side_support"] = {
                "side": pending_side,
                "grid_stamp_sec": doorway.get("grid_stamp_sec"),
                "side_free_ratio": current_side_free_ratio,
                "current_opening_interval_overlap_ratio": current_overlap_ratio,
                "pass": current_side_support,
            }
            if not current_side_support:
                status["pending_forced_opening"] = None
                status["pending_clear_reason"] = "alignment_current_side_gap_not_supported"
                status["reject_reason"] = "alignment_current_side_gap_not_supported"
                status["trigger_source"] = "pending_forced_opening_alignment_rejected"
                return status
            status.update(
                {
                    "trigger_ready": True,
                    "reject_reason": None,
                    "entry_side": pending_opening.get("side"),
                    "selected_opening": pending_opening,
                    "selection_reason": pending_opening.get("selection_reason"),
                    "trigger_reason": "side_opening_alignment_reached",
                    "trigger_source": "pending_forced_opening_alignment",
                }
            )
            return status
        else:
            status["reject_reason"] = "forced_opening_alignment_pending"
            status["trigger_source"] = "pending_forced_opening_alignment"
            return status

    profile_root = doorway.get("doorway_geometry_profile")
    profile_root = profile_root if isinstance(profile_root, dict) else {}
    cache = opening_observation_cache if isinstance(opening_observation_cache, dict) else {}
    for side in ("left", "right"):
        if not isinstance(cache.get(side), list):
            cache[side] = []
    candidates: List[Dict[str, Any]] = []
    partial_candidates_awaiting_confirmation: List[Dict[str, Any]] = []
    below_threshold: List[Dict[str, Any]] = []
    quality_rejected: List[Dict[str, Any]] = []
    for side in ("left", "right"):
        side_profile = profile_root.get(side)
        side_profile = side_profile if isinstance(side_profile, dict) else {}
        selected = side_profile.get("selected_opening")
        if not isinstance(selected, dict):
            continue
        width = selected.get("width_m")
        center = selected.get("center_x_m")
        if not finite_number(width):
            continue
        candidate = {
            "side": side,
            "opening_width_m": float(width),
            "opening_start_x_m": float(selected["start_x_m"]) if finite_number(selected.get("start_x_m")) else None,
            "opening_center_x_m": float(center) if finite_number(center) else None,
            "opening_end_x_m": float(selected["end_x_m"]) if finite_number(selected.get("end_x_m")) else None,
            "window_limited_broad_opening": bool(
                selected.get("window_limited_broad_opening")
                or side_profile.get("window_limited_broad_opening")
                or (
                    finite_number(selected.get("start_x_m"))
                    and finite_number(selected.get("end_x_m"))
                    and float(selected["start_x_m"]) <= 0.35
                    and float(selected["end_x_m"]) >= 2.8
                    and float(width) >= 2.4
                )
            ),
            "selected_opening": selected,
            "source": "doorway_geometry_profile.selected_opening",
            "width_plausible": bool(selected.get("width_plausible")),
            "before_wall_or_unknown": bool(selected.get("before_wall_or_unknown")),
            "after_wall_or_unknown": bool(selected.get("after_wall_or_unknown")),
        }
        candidate["fully_bounded_opening"] = bool(
            candidate["before_wall_or_unknown"] and candidate["after_wall_or_unknown"]
        )
        if float(width) < float(args.forced_entry_min_opening_width_m):
            below_threshold.append(candidate)
            continue
        candidate_reject_reasons: List[str] = []
        if not finite_number(candidate.get("opening_start_x_m")) or not finite_number(
            candidate.get("opening_end_x_m")
        ):
            candidate_reject_reasons.append("opening_interval_unavailable")
        elif not finite_number(candidate.get("opening_center_x_m")):
            candidate_reject_reasons.append("opening_center_unavailable")
        elif float(candidate["opening_end_x_m"]) <= float(candidate["opening_start_x_m"]):
            candidate_reject_reasons.append("opening_interval_invalid")
        elif float(candidate["opening_end_x_m"]) < -float(args.forced_entry_alignment_tolerance_m):
            candidate_reject_reasons.append("opening_interval_behind_robot")
        if candidate_reject_reasons:
            candidate["candidate_reject_reasons"] = candidate_reject_reasons
            quality_rejected.append(candidate)
            continue
        if not candidate["before_wall_or_unknown"]:
            candidate["candidate_reject_reasons"] = ["opening_leading_boundary_not_supported"]
            quality_rejected.append(candidate)
            continue
        if candidate["window_limited_broad_opening"]:
            candidate["candidate_reject_reasons"] = ["window_limited_broad_opening_not_confirmable"]
            quality_rejected.append(candidate)
            continue
        observation = {
            **candidate,
            "grid_stamp_sec": doorway.get("grid_stamp_sec"),
            "observation_anchor_progress_m": float(current_progress),
            "start_progress_m": float(current_progress) + float(candidate["opening_start_x_m"]),
            "center_progress_m": float(current_progress) + float(candidate["opening_center_x_m"]),
            "end_progress_m": float(current_progress) + float(candidate["opening_end_x_m"]),
        }
        side_cache = cache[side]
        previous = side_cache[-1] if side_cache else None
        distinct_grid_stamp = bool(
            isinstance(previous, dict)
            and finite_number(observation.get("grid_stamp_sec"))
            and finite_number(previous.get("grid_stamp_sec"))
            and float(observation["grid_stamp_sec"]) != float(previous["grid_stamp_sec"])
        )
        center_jump = None
        start_progress_jump = None
        interval_overlap = 0.0
        interval_overlap_ratio = 0.0
        if isinstance(previous, dict):
            center_jump = abs(
                float(observation["center_progress_m"]) - float(previous["center_progress_m"])
            )
            start_progress_jump = abs(
                float(observation["start_progress_m"]) - float(previous["start_progress_m"])
            )
            interval_overlap = max(
                0.0,
                min(float(observation["end_progress_m"]), float(previous["end_progress_m"]))
                - max(float(observation["start_progress_m"]), float(previous["start_progress_m"])),
            )
            interval_overlap_ratio = interval_overlap / max(
                1e-6,
                min(
                    float(observation["end_progress_m"])
                    - float(observation["start_progress_m"]),
                    float(previous["end_progress_m"]) - float(previous["start_progress_m"]),
                ),
            )
        confirmed = bool(
            isinstance(previous, dict)
            and distinct_grid_stamp
            and bool(previous.get("before_wall_or_unknown"))
            and finite_number(start_progress_jump)
            and float(start_progress_jump) <= float(args.forced_entry_confirm_center_jump_m)
            and interval_overlap_ratio >= float(args.forced_entry_confirm_interval_overlap_ratio)
        )
        observation.update(
            {
                "previous_grid_stamp_sec": previous.get("grid_stamp_sec") if isinstance(previous, dict) else None,
                "distinct_grid_stamp": distinct_grid_stamp,
                "center_progress_jump_m": center_jump,
                "start_progress_jump_m": start_progress_jump,
                "interval_overlap_m": interval_overlap,
                "interval_overlap_ratio": interval_overlap_ratio,
                "two_frame_confirmed": confirmed,
            }
        )
        if (
            not side_cache
            or not finite_number(observation.get("grid_stamp_sec"))
            or not finite_number(side_cache[-1].get("grid_stamp_sec"))
            or float(observation["grid_stamp_sec"]) != float(side_cache[-1]["grid_stamp_sec"])
        ):
            side_cache.append(observation)
            del side_cache[:-2]
        if not confirmed:
            observation["candidate_reject_reasons"] = ["two_frame_absolute_position_not_confirmed"]
            partial_candidates_awaiting_confirmation.append(observation)
            continue
        overlap_start = max(
            float(observation["start_progress_m"]), float(previous["start_progress_m"])
        )
        overlap_end = min(
            float(observation["end_progress_m"]), float(previous["end_progress_m"])
        )
        fully_bounded_pair = bool(
            observation.get("fully_bounded_opening") and previous.get("fully_bounded_opening")
        )
        if fully_bounded_pair:
            fixed_target_progress = 0.5 * (overlap_start + overlap_end)
            target_progress_source = "bounded_interval_overlap_center"
        else:
            stable_start_progress = 0.5 * (
                float(observation["start_progress_m"]) + float(previous["start_progress_m"])
            )
            alignment_offset = min(
                0.5
                * min(
                    float(observation["opening_width_m"]),
                    float(previous["opening_width_m"]),
                ),
                float(args.forced_entry_alignment_max_inside_offset_m),
            )
            fixed_target_progress = stable_start_progress + alignment_offset
            target_progress_source = "stable_leading_boundary_plus_inside_offset"
        candidates.append(
            {
                **candidate,
                "source": "two_frame_absolute_leading_boundary",
                "fixed_target_progress_m": fixed_target_progress,
                "target_progress_source": target_progress_source,
                "confirmed_start_progress_m": overlap_start,
                "confirmed_end_progress_m": overlap_end,
                "two_frame_confirmed": True,
                "confirmation_observations": [previous, observation],
                "center_progress_jump_m": center_jump,
                "start_progress_jump_m": start_progress_jump,
                "interval_overlap_ratio": interval_overlap_ratio,
                "selection_reason": "two_frame_absolute_leading_boundary_confirmed",
            }
        )

    latch = latched_doorway_profile if isinstance(latched_doorway_profile, dict) else {}
    latch_selected = latch.get("selected_opening")
    latch_selected = latch_selected if isinstance(latch_selected, dict) else {}
    latch_side = latch.get("door_side")
    latch_width = latch_selected.get("width_m", latch.get("opening_width_m"))
    latch_target_progress = latch.get("target_anchor_progress_m")
    latch_stable_count = int(latch.get("stable_partial_observation_count") or 0)
    latch_broad = bool(
        finite_number(latch_selected.get("start_x_m"))
        and finite_number(latch_selected.get("end_x_m"))
        and finite_number(latch_width)
        and float(latch_selected["start_x_m"]) <= 0.35
        and float(latch_selected["end_x_m"]) >= 2.8
        and float(latch_width) >= 2.4
    )
    latch_candidate_ready = False
    if latch_candidate_ready:
        candidates.append(
            {
                "side": latch_side,
                "opening_width_m": float(latch_width),
                "opening_start_x_m": (
                    float(latch_selected["start_x_m"])
                    if finite_number(latch_selected.get("start_x_m"))
                    else None
                ),
                "opening_center_x_m": float(latch_target_progress) - float(current_progress),
                "opening_end_x_m": (
                    float(latch_selected["end_x_m"])
                    if finite_number(latch_selected.get("end_x_m"))
                    else None
                ),
                "window_limited_broad_opening": latch_broad,
                "selected_opening": latch_selected,
                "source": "stable_partial_latched_doorway_profile",
                "width_plausible": bool(latch_selected.get("width_plausible")),
                "before_wall_or_unknown": bool(latch_selected.get("before_wall_or_unknown")),
                "after_wall_or_unknown": bool(latch_selected.get("after_wall_or_unknown")),
                "fully_bounded_opening": bool(
                    latch_selected.get("before_wall_or_unknown")
                    and latch_selected.get("after_wall_or_unknown")
                ),
                "fixed_target_progress_m": float(latch_target_progress),
                "stable_partial_observation_count": latch_stable_count,
                "selection_reason": "stable_partial_latch_side_opening",
            }
        )
        status["stable_partial_latch_candidate_used"] = True
    status["candidate_count"] = len(candidates)
    status["opening_observation_cache"] = cache
    status["latched_doorway_profile_control_enabled"] = False
    status["partial_candidates_awaiting_confirmation"] = partial_candidates_awaiting_confirmation
    status["below_threshold_candidates"] = below_threshold
    status["quality_rejected_candidates"] = quality_rejected
    if not candidates:
        if partial_candidates_awaiting_confirmation:
            status["reject_reason"] = "two_frame_absolute_position_not_confirmed"
        elif quality_rejected:
            status["reject_reason"] = "no_valid_side_opening_interval"
        else:
            status["reject_reason"] = "opening_width_below_threshold" if below_threshold else "no_side_opening"
        return status
    candidates.sort(
        key=lambda candidate: (
            1 if candidate.get("window_limited_broad_opening") else 0,
            float(candidate["opening_start_x_m"])
            if finite_number(candidate.get("opening_start_x_m"))
            else float("inf"),
            -float(candidate["opening_width_m"]),
            0 if candidate["side"] == "left" else 1,
        )
    )
    selected = candidates[0]
    if finite_number(selected.get("fixed_target_progress_m")):
        target_progress = float(selected["fixed_target_progress_m"])
        alignment_offset_inside_opening = None
    else:
        alignment_offset_inside_opening = min(
            float(selected["opening_width_m"]) * 0.5,
            float(args.forced_entry_alignment_max_inside_offset_m),
        )
        target_progress = (
            float(current_progress)
            + float(selected["opening_start_x_m"])
            + alignment_offset_inside_opening
        )
    selected = {
        **selected,
        "target_progress_m": target_progress,
        "start_progress_m": (
            float(selected["confirmed_start_progress_m"])
            if finite_number(selected.get("confirmed_start_progress_m"))
            else float(current_progress) + float(selected["opening_start_x_m"])
            if finite_number(selected.get("opening_start_x_m"))
            else None
        ),
        "end_progress_m": (
            float(selected["confirmed_end_progress_m"])
            if finite_number(selected.get("confirmed_end_progress_m"))
            else float(current_progress) + float(selected["opening_end_x_m"])
            if finite_number(selected.get("opening_end_x_m"))
            else None
        ),
        "alignment_offset_inside_opening_m": alignment_offset_inside_opening,
        "created_anchor_progress_m": float(current_progress),
        "selection_reason": selected.get("selection_reason")
        or (
            "single_room_zone_side_opening"
            if len(candidates) == 1
            else "non_broad_then_nearest_start_then_width"
        ),
    }
    alignment_error = target_progress - float(current_progress)
    status.update(
        {
            "trigger_ready": abs(alignment_error) <= float(args.forced_entry_alignment_tolerance_m),
            "reject_reason": (
                None
                if abs(alignment_error) <= float(args.forced_entry_alignment_tolerance_m)
                else "forced_opening_alignment_pending"
            ),
            "entry_side": selected["side"] if abs(alignment_error) <= float(args.forced_entry_alignment_tolerance_m) else None,
            "selected_opening": selected,
            "pending_forced_opening": selected,
            "selection_reason": selected["selection_reason"],
            "alignment_target_anchor_progress_m": target_progress,
            "alignment_longitudinal_error_m": alignment_error,
            "trigger_reason": (
                "side_opening_alignment_reached"
                if abs(alignment_error) <= float(args.forced_entry_alignment_tolerance_m)
                else None
            ),
            "trigger_source": "new_room_zone_side_opening",
        }
    )
    return status


def select_simple_room_entry_opening(
    doorway: Dict[str, Any],
    room_zone_reached: bool,
    current_progress: Optional[float],
    args: argparse.Namespace,
    pending_opening: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """React to a room-zone side gap without committing to a distant estimated door center."""
    status: Dict[str, Any] = {
        "forced_entry_enabled": bool(args.enable_forced_room_entry_mvp),
        "simple_room_entry_enabled": bool(args.enable_simple_room_entry),
        "room_zone_reached": bool(room_zone_reached),
        "current_anchor_progress_m": current_progress,
        "trigger_ready": False,
        "stop_observe_required": False,
        "selected_opening": None,
        "pending_forced_opening": pending_opening if isinstance(pending_opening, dict) else None,
        "quality_fields_control_robot": False,
    }
    if not args.enable_forced_room_entry_mvp or not args.enable_simple_room_entry:
        status["reject_reason"] = "simple_room_entry_disabled"
        return status
    if not room_zone_reached:
        status["pending_forced_opening"] = None
        status["reject_reason"] = "room_zone_not_reached"
        return status
    if not finite_number(current_progress):
        status["reject_reason"] = "anchor_progress_unavailable"
        return status
    progress = float(current_progress)
    if (
        args.forced_entry_max_trigger_progress_m is not None
        and progress > float(args.forced_entry_max_trigger_progress_m)
    ):
        status["pending_forced_opening"] = None
        status["reject_reason"] = "max_trigger_progress_exceeded"
        return status

    profile_root = doorway.get("doorway_geometry_profile")
    profile_root = profile_root if isinstance(profile_root, dict) else {}

    def current_side_opening(side: str) -> Optional[Dict[str, Any]]:
        side_profile = profile_root.get(side)
        side_profile = side_profile if isinstance(side_profile, dict) else {}
        selected = side_profile.get("selected_opening")
        if not isinstance(selected, dict):
            return None
        start_x = selected.get("start_x_m")
        end_x = selected.get("end_x_m")
        width = selected.get("width_m")
        if not all(finite_number(value) for value in (start_x, end_x, width)):
            return None
        start_x = float(start_x)
        end_x = float(end_x)
        width = float(width)
        if end_x <= start_x or end_x <= 0.0 or width < float(args.forced_entry_min_opening_width_m):
            return None
        broad = bool(
            selected.get("window_limited_broad_opening")
            or side_profile.get("window_limited_broad_opening")
            or (start_x <= 0.35 and end_x >= 2.8 and width >= 2.4)
        )
        return {
            "side": side,
            "opening_width_m": width,
            "opening_start_x_m": start_x,
            "opening_center_x_m": (
                float(selected["center_x_m"])
                if finite_number(selected.get("center_x_m"))
                else 0.5 * (start_x + end_x)
            ),
            "opening_end_x_m": end_x,
            "start_progress_m": progress + start_x,
            "center_progress_m": progress
            + (
                float(selected["center_x_m"])
                if finite_number(selected.get("center_x_m"))
                else 0.5 * (start_x + end_x)
            ),
            "end_progress_m": progress + end_x,
            "window_limited_broad_opening": broad,
            "width_plausible": selected.get("width_plausible"),
            "before_wall_or_unknown": selected.get("before_wall_or_unknown"),
            "after_wall_or_unknown": selected.get("after_wall_or_unknown"),
            "fully_bounded_opening": bool(
                selected.get("before_wall_or_unknown")
                and selected.get("after_wall_or_unknown")
            ),
            "geometry_pass": side_profile.get("geometry_pass"),
            "soft_geometry_pass": side_profile.get("soft_geometry_pass"),
            "free_ratio": side_profile.get("free_ratio"),
            "source": "simple_room_entry_current_side_gap",
            "selection_reason": "room_zone_side_gap_interval",
            "selected_opening_debug": selected,
        }

    candidates = [
        candidate
        for side in ("left", "right")
        for candidate in [current_side_opening(side)]
        if isinstance(candidate, dict)
    ]
    status["current_side_gap_candidates"] = candidates
    nearfield_max_x = float(args.simple_room_entry_nearfield_max_x_m)
    max_opening_width = float(args.simple_room_entry_max_opening_width_m)
    visible_nearfield_candidates = [
        candidate
        for candidate in candidates
        if float(candidate["opening_start_x_m"]) <= nearfield_max_x
        and float(candidate["opening_width_m"]) <= max_opening_width
    ]
    status["simple_execution_gate"] = "room_zone_and_current_nearfield_side_gap"
    status["nearfield_max_x_m"] = nearfield_max_x
    status["opening_width_range_m"] = [
        float(args.forced_entry_min_opening_width_m),
        max_opening_width,
    ]
    status["visible_nearfield_candidates"] = visible_nearfield_candidates
    if not visible_nearfield_candidates:
        status["pending_forced_opening"] = None
        status["reject_reason"] = (
            "side_gap_outside_simple_nearfield_or_width_gate"
            if candidates
            else "no_room_zone_side_gap"
        )
        return status

    visible_nearfield_candidates.sort(
        key=lambda candidate: (
            max(0.0, float(candidate["opening_start_x_m"])),
            -float(candidate["opening_width_m"]),
            0 if candidate["side"] == "left" else 1,
        )
    )
    selected = dict(visible_nearfield_candidates[0])
    selected.update(
        {
            "source": "simple_room_entry_turn_on_sight",
            "simple_phase": "IMMEDIATE_TURN",
            "target_progress_m": progress,
            "cue_anchor_progress_m": progress,
            "last_update_anchor_progress_m": progress,
            "missed_frame_count": 0,
            "nearfield_max_x_m": nearfield_max_x,
        }
    )
    status.update(
        {
            "trigger_ready": True,
            "stop_observe_required": False,
            "entry_side": selected["side"],
            "selected_opening": selected,
            "pending_forced_opening": selected,
            "selection_reason": "current_room_zone_nearfield_side_gap",
            "alignment_target_anchor_progress_m": progress,
            "alignment_longitudinal_error_m": 0.0,
            "trigger_reason": "simple_room_entry_turn_on_sight",
            "trigger_source": "simple_room_entry_current_frame_nearfield_gap",
            "reject_reason": None,
        }
    )
    return status


def save_forced_entry_visual_snapshot(
    visual_audit: Optional["SideGapVisualCoordinateAudit"],
    prefix: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"rgb_path": None, "depth_path": None, "available": False}
    if visual_audit is None:
        result["missing_reason"] = "visual_subscriber_unavailable"
        return result
    try:
        if visual_audit.latest_rgb is not None:
            rgb_path = FORCED_ENTRY_DEBUG_DIR / f"{prefix}_rgb.png"
            ros_rgb_to_pil(visual_audit.latest_rgb).save(rgb_path)
            result["rgb_path"] = str(rgb_path)
        if visual_audit.latest_depth is not None:
            depth_path = FORCED_ENTRY_DEBUG_DIR / f"{prefix}_depth_colormap.png"
            ros_depth_to_colormap(visual_audit.latest_depth).save(depth_path)
            result["depth_path"] = str(depth_path)
        result["available"] = bool(result["rgb_path"] or result["depth_path"])
        if not result["available"]:
            result["missing_reason"] = "rgb_and_depth_messages_unavailable"
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def forced_entry_side_free_space_check(
    args: argparse.Namespace,
    pose: Tuple[float, float, float],
    current_progress: Optional[float],
    side: str,
    opening: Dict[str, Any],
) -> Dict[str, Any]:
    check_args = argparse.Namespace(**vars(args))
    remaining_opening_m = None
    if finite_number(current_progress) and finite_number(opening.get("end_progress_m")):
        remaining_opening_m = float(opening["end_progress_m"]) - float(current_progress)
    forward_max_m = 0.65
    if finite_number(remaining_opening_m):
        forward_max_m = min(forward_max_m, max(0.25, float(remaining_opening_m) - 0.05))
    check_args.local_entry_forward_min_m = 0.0
    check_args.local_entry_forward_max_m = forward_max_m
    check_args.local_entry_lateral_min_m = 0.30
    check_args.local_entry_lateral_max_m = 1.40
    check_args.local_entry_min_area_m2 = 0.08
    check_args.local_entry_min_forward_extent_m = 0.15
    check_args.local_entry_min_lateral_extent_m = 0.45
    check_args.local_entry_grid_timeout_sec = float(args.forced_entry_side_grid_timeout_sec)
    try:
        local_entry_check = build_local_free_space_entry_debug(
            check_args,
            pose,
            True,
            current_progress,
        )
    except Exception as exc:
        local_entry_check = {
            "grid_available": False,
            "grid_source": GRID_TOPIC,
            f"{side}_entry_candidate": {
                "entry_target_valid": False,
                "reject_reason": "side_free_space_check_exception",
                "exception": repr(exc),
            },
        }
    candidate = local_entry_check.get(f"{side}_entry_candidate") or {}
    selected_component = candidate.get("selected_component") or {}
    component_bounds = selected_component.get("component_bounds_base_xy")
    component_crosses_side_wall_line = False
    component_reaches_corridor_side = False
    component_reaches_room_side = False
    if (
        isinstance(component_bounds, list)
        and len(component_bounds) == 2
        and all(isinstance(point, list) and len(point) == 2 for point in component_bounds)
    ):
        min_y = float(component_bounds[0][1])
        max_y = float(component_bounds[1][1])
        if side == "left":
            component_reaches_corridor_side = min_y <= 0.45
            component_reaches_room_side = max_y >= 1.20
        else:
            component_reaches_corridor_side = max_y >= -0.45
            component_reaches_room_side = min_y <= -1.20
        component_crosses_side_wall_line = bool(
            component_reaches_corridor_side and component_reaches_room_side
        )
    generic_path_pass = bool(candidate.get("entry_target_valid"))
    side_throat_pass = bool(
        local_entry_check.get("grid_available")
        and candidate.get("valid_component_count", 0) >= 1
        and component_crosses_side_wall_line
    )
    side_free_space_pass = bool(generic_path_pass or side_throat_pass)
    return {
        "grid_available": local_entry_check.get("grid_available"),
        "grid_source": local_entry_check.get("grid_source"),
        "grid_frame": local_entry_check.get("grid_frame"),
        "grid_resolution_m": local_entry_check.get("grid_resolution_m"),
        "side": side,
        "opening_end_progress_m": opening.get("end_progress_m"),
        "current_anchor_progress_m": current_progress,
        "remaining_opening_m": remaining_opening_m,
        "forced_sector_base_bounds_xy": {
            "x": [0.0, forward_max_m],
            "y": [0.30, 1.40] if side == "left" else [-1.40, -0.30],
        },
        "generic_path_pass": generic_path_pass,
        "component_reaches_corridor_side": component_reaches_corridor_side,
        "component_reaches_room_side": component_reaches_room_side,
        "component_crosses_side_wall_line": component_crosses_side_wall_line,
        "pass_reason": (
            "generic_path_connected"
            if generic_path_pass
            else "continuous_side_throat_across_wall_line"
            if side_throat_pass
            else "no_continuous_side_throat"
        ),
        "candidate": candidate,
        "pass": side_free_space_pass,
    }


def execute_forced_room_entry_mvp(
    args: argparse.Namespace,
    trigger_status: Dict[str, Any],
    visual_audit: Optional["SideGapVisualCoordinateAudit"],
    linear_x_clamped_from: Optional[float],
    follower_process: Optional[subprocess.Popen],
    robot_pose_hint: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    FORCED_ENTRY_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    opening = trigger_status.get("selected_opening")
    opening = opening if isinstance(opening, dict) else {}
    side = str(opening.get("side"))
    sign = 1.0 if side == "left" else -1.0
    turn_angle_rad = math.radians(float(args.forced_entry_turn_angle_deg))
    turn_angular_z = sign * abs(float(args.forced_entry_angular_z))
    nominal_turn_duration = turn_angle_rad / max(abs(turn_angular_z), 1e-6)
    forward_linear_x = float(args.forced_entry_linear_x)
    turn_topic = args.cmd_topic
    forward_topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
    result: Dict[str, Any] = {
        "diagnostic_only": False,
        "controls_robot": True,
        "calls_runner": False,
        "writes_real_target": False,
        "forced_entry_enabled": True,
        "forced_entry_triggered": True,
        "forced_entry_done": False,
        "entry_side": side,
        "trigger_reason": trigger_status.get("trigger_reason"),
        "selection_reason": trigger_status.get("selection_reason"),
        "pending_forced_opening": trigger_status.get("pending_forced_opening"),
        "quality_rejected_candidates": trigger_status.get("quality_rejected_candidates", []),
        "room_zone_reached": trigger_status.get("room_zone_reached"),
        "current_anchor_progress_m": trigger_status.get("current_anchor_progress_m"),
        "opening_width_m": opening.get("opening_width_m"),
        "opening_start_x_m": opening.get("opening_start_x_m"),
        "opening_center_x_m": opening.get("opening_center_x_m"),
        "opening_end_x_m": opening.get("opening_end_x_m"),
        "window_limited_broad_opening": opening.get("window_limited_broad_opening"),
        "width_plausible": opening.get("width_plausible"),
        "before_wall_or_unknown": opening.get("before_wall_or_unknown"),
        "after_wall_or_unknown": opening.get("after_wall_or_unknown"),
        "alignment_target_anchor_progress_m": trigger_status.get("alignment_target_anchor_progress_m"),
        "alignment_longitudinal_error_m": trigger_status.get("alignment_longitudinal_error_m"),
        "turn_angle_deg": float(args.forced_entry_turn_angle_deg),
        "turn_angular_z": turn_angular_z,
        "turn_duration_sec": nominal_turn_duration,
        "turn_control_mode": None,
        "turn_started": False,
        "turn_completed": False,
        "forward_linear_x": forward_linear_x,
        "forward_duration_sec": float(args.forced_entry_forward_duration_sec),
        "expected_forward_distance_m": forward_linear_x * float(args.forced_entry_forward_duration_sec),
        "forward_started": False,
        "forward_completed": False,
        "front_clearance": None,
        "front_clearance_pass": False,
        "front_clearance_required_for_plan_m": max(
            float(args.forced_entry_front_clearance_min_m),
            forward_linear_x * float(args.forced_entry_forward_duration_sec) + 0.15,
        ),
        "side_free_space_check": None,
        "side_free_space_pass": False,
        "forward_stalled": False,
        "forward_actual_progress_m": None,
        "forward_progress_pass": False,
        "side_crossing_pass": False,
        "signed_entry_lateral_displacement_m": None,
        "forward_success_min_progress_m": float(args.forced_entry_success_min_forward_progress_m),
        "side_crossing_min_displacement_m": float(args.forced_entry_success_min_signed_lateral_m),
        "linear_x_clamped_from": linear_x_clamped_from,
        "linear_x_clamped_to": forward_linear_x if linear_x_clamped_from is not None else None,
        "command_publisher_available": False,
        "cmd_vel_topic": turn_topic,
        "forward_cmd_topic": forward_topic,
        "turn_command_chain": "direct_cmd_vel",
        "forward_command_chain": "cmd_vel_raw_to_imu_velocity_follower_to_cmd_vel" if args.use_imu_velocity_follower else "direct_cmd_vel",
        "imu_follower_paused_for_turn": False,
        "imu_follower_restarted_for_forward": False,
        "twist_publish_count": 0,
        "turn_publish_count": 0,
        "forward_publish_count": 0,
        "exception_type": None,
        "exception_message": None,
        "exception_traceback": None,
        "final_decision": "STATE_MACHINE_FORCED_ROOM_ENTRY_INCOMPLETE",
        "debug_stop_reason": None,
    }
    write_json(FORCED_ENTRY_TRIGGER_PATH, {**trigger_status, "timestamp_sec": time.time()})
    if not args.execute:
        result["final_decision"] = "STATE_MACHINE_FORCED_ROOM_ENTRY_DRY_RUN"
        result["debug_stop_reason"] = "execute_disabled"
        write_json(FORCED_ENTRY_SUMMARY_PATH, result)
        return result

    latest_pose: Dict[str, Optional[Tuple[float, float, float]]] = {"pose": robot_pose_hint}
    odom_callback_count = 0

    def odom_callback(msg: Odometry) -> None:
        nonlocal odom_callback_count
        ros_pose = msg.pose.pose
        latest_pose["pose"] = (
            float(ros_pose.position.x),
            float(ros_pose.position.y),
            yaw_from_quat(ros_pose.orientation),
        )
        odom_callback_count += 1

    subscriber = rospy.Subscriber(ODOM_TOPIC, Odometry, odom_callback, queue_size=20)
    try:
        if latest_pose["pose"] is None:
            try:
                latest_pose["pose"] = pose_tuple(read_odom(timeout_sec=1.0))
            except Exception:
                pass
        result["robot_pose_before"] = list(latest_pose["pose"]) if latest_pose["pose"] is not None else None
        result["robot_pose_before_available"] = latest_pose["pose"] is not None
        trigger_event = {**trigger_status, "timestamp_sec": time.time()}
        trigger_event["robot_pose_before"] = result["robot_pose_before"]
        trigger_event["robot_pose_before_available"] = result["robot_pose_before_available"]
        write_json(FORCED_ENTRY_TRIGGER_PATH, trigger_event)
        result["before_visual"] = save_forced_entry_visual_snapshot(visual_audit, "before_entry")
        if args.use_imu_velocity_follower and follower_process is not None:
            stop_follower(follower_process)
            result["imu_follower_paused_for_turn"] = True
        turn_pub = rospy.Publisher(turn_topic, Twist, queue_size=2)
        forward_pub = rospy.Publisher(forward_topic, Twist, queue_size=2)
        time.sleep(0.15)
        result["command_publisher_available"] = True
        zero = Twist()

        def publish_phase(
            publishers: Sequence[Any],
            command: Twist,
            duration_sim_sec: float,
            wall_watchdog_sec: float,
        ) -> Dict[str, Any]:
            sim_start = rospy.Time.now()
            wall_start = time.monotonic()
            count = 0
            sim_elapsed = 0.0
            watchdog = False
            while not rospy.is_shutdown():
                sim_elapsed = max(0.0, float((rospy.Time.now() - sim_start).to_sec()))
                if sim_elapsed >= duration_sim_sec:
                    break
                if time.monotonic() - wall_start >= wall_watchdog_sec:
                    watchdog = True
                    break
                for publisher in publishers:
                    publisher.publish(command)
                    count += 1
                time.sleep(0.05)
            command_publish_count = count
            for publisher in publishers:
                publisher.publish(zero)
                count += 1
            return {
                "publish_count": count,
                "command_publish_count": command_publish_count,
                "zero_publish_count": len(publishers),
                "sim_duration_sec": sim_elapsed,
                "wall_duration_sec": time.monotonic() - wall_start,
                "wall_watchdog_triggered": watchdog,
            }

        def publish_forward_phase(
            publisher: Any,
            command: Twist,
            duration_sim_sec: float,
            wall_watchdog_sec: float,
        ) -> Dict[str, Any]:
            sim_start = rospy.Time.now()
            wall_start = time.monotonic()
            start_pose = latest_pose["pose"]
            count = 0
            sim_elapsed = 0.0
            watchdog = False
            stalled = False
            stall_window_sec = float(args.forced_entry_stall_window_sec)
            stall_min_progress_m = float(args.forced_entry_stall_min_progress_m)
            pose_samples: List[Tuple[float, float, float]] = []
            last_sample_sim = -1.0
            forward_progress = None
            translation = None
            while not rospy.is_shutdown():
                sim_elapsed = max(0.0, float((rospy.Time.now() - sim_start).to_sec()))
                if sim_elapsed >= duration_sim_sec:
                    break
                if time.monotonic() - wall_start >= wall_watchdog_sec:
                    watchdog = True
                    break
                current_pose = latest_pose["pose"]
                if current_pose is not None:
                    if start_pose is None:
                        start_pose = current_pose
                    if sim_elapsed - last_sample_sim >= 0.05:
                        pose_samples.append((sim_elapsed, float(current_pose[0]), float(current_pose[1])))
                        last_sample_sim = sim_elapsed
                    if start_pose is not None:
                        dx = float(current_pose[0]) - float(start_pose[0])
                        dy = float(current_pose[1]) - float(start_pose[1])
                        forward_progress = math.cos(float(start_pose[2])) * dx + math.sin(float(start_pose[2])) * dy
                        translation = math.hypot(dx, dy)
                    if sim_elapsed >= max(1.0, stall_window_sec) and pose_samples:
                        cutoff = sim_elapsed - stall_window_sec
                        reference = None
                        for sample in pose_samples:
                            if sample[0] <= cutoff:
                                reference = sample
                            else:
                                break
                        if reference is not None:
                            window_displacement = math.hypot(
                                float(current_pose[0]) - reference[1],
                                float(current_pose[1]) - reference[2],
                            )
                            if window_displacement < stall_min_progress_m:
                                stalled = True
                                break
                publisher.publish(command)
                count += 1
                time.sleep(0.05)
            publisher.publish(zero)
            count += 1
            current_pose = latest_pose["pose"]
            if start_pose is not None and current_pose is not None:
                dx = float(current_pose[0]) - float(start_pose[0])
                dy = float(current_pose[1]) - float(start_pose[1])
                forward_progress = math.cos(float(start_pose[2])) * dx + math.sin(float(start_pose[2])) * dy
                translation = math.hypot(dx, dy)
            return {
                "publish_count": count,
                "command_publish_count": max(0, count - 1),
                "zero_publish_count": 1,
                "sim_duration_sec": sim_elapsed,
                "wall_duration_sec": time.monotonic() - wall_start,
                "wall_watchdog_triggered": watchdog,
                "duration_completed": sim_elapsed >= duration_sim_sec,
                "stalled": stalled,
                "stall_reason": "insufficient_odom_progress_during_forward_command" if stalled else None,
                "stall_window_sec": stall_window_sec,
                "stall_min_progress_m": stall_min_progress_m,
                "actual_forward_progress_m": forward_progress,
                "actual_translation_m": translation,
                "pose_sample_count": len(pose_samples),
            }

        stop_publishers = [turn_pub, forward_pub] if turn_topic != forward_topic else [turn_pub]
        result["pre_stop"] = publish_phase(
            stop_publishers,
            zero,
            float(args.forced_entry_pre_stop_sec),
            max(30.0, float(args.forced_entry_pre_stop_sec) * 30.0),
        )
        result["twist_publish_count"] += int(result["pre_stop"]["publish_count"])

        pose_for_side_check = latest_pose["pose"]
        if pose_for_side_check is not None:
            result["side_free_space_check"] = forced_entry_side_free_space_check(
                args,
                pose_for_side_check,
                trigger_status.get("current_anchor_progress_m"),
                side,
                opening,
            )
            result["side_free_space_pass"] = bool(result["side_free_space_check"].get("pass"))
        else:
            result["side_free_space_check"] = {
                "grid_available": False,
                "side": side,
                "reject_reason": "robot_pose_unavailable_for_side_free_space_check",
            }
        initial_pose = latest_pose["pose"]
        initial_yaw = initial_pose[2] if initial_pose is not None else None
        yaw_feedback = initial_yaw is not None
        result["turn_control_mode"] = "yaw_feedback" if yaw_feedback else "timed_fallback"
        turn_cmd = Twist()
        turn_cmd.angular.z = turn_angular_z
        turn_sim_start = rospy.Time.now()
        turn_wall_start = time.monotonic()
        turn_max_sim = max(nominal_turn_duration * 3.0, nominal_turn_duration + 2.0)
        turn_wall_watchdog = max(120.0, turn_max_sim * 30.0)
        result["turn_started"] = True
        actual_yaw_delta = None
        while not rospy.is_shutdown():
            sim_elapsed = max(0.0, float((rospy.Time.now() - turn_sim_start).to_sec()))
            if time.monotonic() - turn_wall_start >= turn_wall_watchdog:
                result["turn_wall_watchdog_triggered"] = True
                break
            if yaw_feedback and odom_callback_count == 0 and time.monotonic() - turn_wall_start >= 1.0:
                yaw_feedback = False
                result["turn_control_mode"] = "timed_fallback"
                result["yaw_feedback_fallback_reason"] = "odom_callback_not_updating"
            current_pose = latest_pose["pose"]
            if yaw_feedback and current_pose is not None and initial_yaw is not None:
                actual_yaw_delta = normalize_angle(current_pose[2] - initial_yaw)
                if sign * actual_yaw_delta >= turn_angle_rad:
                    result["turn_completed"] = True
                    break
                if sim_elapsed >= turn_max_sim:
                    result["turn_sim_timeout"] = True
                    break
            elif sim_elapsed >= nominal_turn_duration:
                result["turn_completed"] = True
                break
            turn_pub.publish(turn_cmd)
            result["turn_publish_count"] += 1
            result["twist_publish_count"] += 1
            time.sleep(0.05)
        turn_pub.publish(zero)
        result["twist_publish_count"] += 1
        result["turn_sim_duration_sec"] = max(0.0, float((rospy.Time.now() - turn_sim_start).to_sec()))
        result["turn_wall_duration_sec"] = time.monotonic() - turn_wall_start
        result["actual_yaw_delta_rad"] = actual_yaw_delta
        result["odom_callback_count"] = odom_callback_count
        result["robot_pose_after_turn"] = list(latest_pose["pose"]) if latest_pose["pose"] is not None else None
        result["turn_settle_stop"] = publish_phase([turn_pub], zero, 0.4, 30.0)
        result["twist_publish_count"] += int(result["turn_settle_stop"]["publish_count"])
        result["after_turn_visual"] = save_forced_entry_visual_snapshot(visual_audit, "after_turn")
        result["front_clearance"] = forced_entry_front_clearance(
            visual_audit,
            float(result["front_clearance_required_for_plan_m"]),
        )
        result["front_clearance"]["safety_scope"] = "center_depth_emergency_guard"
        result["front_clearance"]["verifies_doorway_crossing"] = False
        result["front_clearance_pass"] = bool(result["front_clearance"].get("pass"))

        if (
            result["turn_completed"]
            and result["side_free_space_pass"]
            and result["front_clearance_pass"]
        ):
            if args.use_imu_velocity_follower:
                restarted_follower = start_follower(args)
                atexit.register(stop_follower, restarted_follower)
                result["imu_follower_restarted_for_forward"] = restarted_follower is not None
            forward_cmd = Twist()
            forward_cmd.linear.x = forward_linear_x
            result["forward_started"] = True
            forward_result = publish_forward_phase(
                forward_pub,
                forward_cmd,
                float(args.forced_entry_forward_duration_sec),
                max(120.0, float(args.forced_entry_forward_duration_sec) * 30.0),
            )
            result["forward_publish_count"] = int(forward_result["command_publish_count"])
            result["twist_publish_count"] += int(forward_result["publish_count"])
            result["forward_control"] = forward_result
            result["forward_stalled"] = bool(forward_result["stalled"])
            result["forward_actual_progress_m"] = forward_result.get("actual_forward_progress_m")
            result["forward_completed"] = bool(
                forward_result["duration_completed"]
                and not forward_result["wall_watchdog_triggered"]
                and not forward_result["stalled"]
            )
        elif result["turn_completed"]:
            result["forward_block_reason"] = (
                (result["side_free_space_check"] or {}).get("reject_reason")
                if not result["side_free_space_pass"]
                else (result["front_clearance"] or {}).get("reject_reason")
            )
        result["post_stop"] = publish_phase(
            stop_publishers,
            zero,
            float(args.forced_entry_post_stop_sec),
            max(30.0, float(args.forced_entry_post_stop_sec) * 30.0),
        )
        result["twist_publish_count"] += int(result["post_stop"]["publish_count"])
        result["robot_pose_after_entry"] = list(latest_pose["pose"]) if latest_pose["pose"] is not None else None
        result["after_visual"] = save_forced_entry_visual_snapshot(visual_audit, "after_entry")
        before = result.get("robot_pose_before")
        after = result.get("robot_pose_after_entry")
        if isinstance(before, list) and isinstance(after, list) and len(before) >= 3 and len(after) >= 2:
            dx, dy = float(after[0]) - float(before[0]), float(after[1]) - float(before[1])
            result["estimated_displacement_m"] = math.hypot(dx, dy)
            result["actual_pose_delta_m"] = result["estimated_displacement_m"]
            result["estimated_lateral_displacement_m"] = -math.sin(float(before[2])) * dx + math.cos(float(before[2])) * dy
            result["signed_entry_lateral_displacement_m"] = (
                sign * float(result["estimated_lateral_displacement_m"])
            )
        else:
            result["estimated_displacement_m"] = None
            result["actual_pose_delta_m"] = None
            result["estimated_lateral_displacement_m"] = None
            result["signed_entry_lateral_displacement_m"] = None
        result["forward_progress_pass"] = bool(
            finite_number(result.get("forward_actual_progress_m"))
            and float(result["forward_actual_progress_m"]) >= float(args.forced_entry_success_min_forward_progress_m)
        )
        result["side_crossing_pass"] = bool(
            finite_number(result.get("signed_entry_lateral_displacement_m"))
            and float(result["signed_entry_lateral_displacement_m"])
            >= float(args.forced_entry_success_min_signed_lateral_m)
        )
        result["forced_entry_done"] = bool(
            result["turn_completed"]
            and result["front_clearance_pass"]
            and result["side_free_space_pass"]
            and result["forward_completed"]
            and result["forward_progress_pass"]
            and result["side_crossing_pass"]
        )
        if result["forced_entry_done"]:
            result["final_decision"] = "STATE_MACHINE_FORCED_ROOM_ENTRY_DONE"
            result["debug_stop_reason"] = "forced_room_entry_done"
        elif result["turn_completed"] and not result["side_free_space_pass"]:
            result["final_decision"] = "STATE_MACHINE_FORCED_ROOM_ENTRY_BLOCKED_SIDE_FREE_SPACE"
            result["debug_stop_reason"] = "forced_room_entry_side_free_space_blocked_after_turn"
        elif result["turn_completed"] and not result["front_clearance_pass"]:
            result["final_decision"] = "STATE_MACHINE_FORCED_ROOM_ENTRY_BLOCKED_FRONT_CLEARANCE"
            result["debug_stop_reason"] = "forced_room_entry_front_clearance_blocked"
        elif result["forward_stalled"]:
            result["final_decision"] = "STATE_MACHINE_FORCED_ROOM_ENTRY_BLOCKED_STALL"
            result["debug_stop_reason"] = "forced_room_entry_forward_stalled"
        elif result["forward_completed"] and not (
            result["forward_progress_pass"] and result["side_crossing_pass"]
        ):
            result["final_decision"] = "STATE_MACHINE_FORCED_ROOM_ENTRY_INSUFFICIENT_PROGRESS"
            result["debug_stop_reason"] = "forced_room_entry_insufficient_progress"
        else:
            result["final_decision"] = "STATE_MACHINE_FORCED_ROOM_ENTRY_FAILED"
            result["debug_stop_reason"] = "forced_room_entry_failed"
    except Exception as exc:
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["exception_traceback"] = traceback.format_exc()
        result["final_decision"] = "STATE_MACHINE_FORCED_ROOM_ENTRY_EXCEPTION"
        result["debug_stop_reason"] = "forced_room_entry_exception"
    finally:
        try:
            subscriber.unregister()
        except Exception:
            pass
        write_json(FORCED_ENTRY_SUMMARY_PATH, result)
    return result


def run_door_observe_debug(
    args: argparse.Namespace,
    trigger_cue: Dict[str, Any],
    door_tracks: Dict[str, List[Dict[str, Any]]],
    door_track_next_ids: Dict[str, int],
    anchor: Optional[Dict[str, Any]],
    iteration: int,
) -> Dict[str, Any]:
    """Stop and gather door observations only; no target or runner is invoked."""
    trigger_side = trigger_cue.get("side")
    observations: List[Dict[str, Any]] = [dict(trigger_cue)]
    topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
    pub = rospy.Publisher(topic, Twist, queue_size=2)
    zero = Twist()
    start_wall = time.monotonic()
    start_sim = rospy.Time.now()
    sim_elapsed = 0.0
    zero_count = 0
    loop_count = 0
    wall_watchdog_triggered = False
    doorway_final_decisions: List[Any] = []
    while not rospy.is_shutdown():
        now_wall = time.monotonic()
        sim_elapsed = max(0.0, float((rospy.Time.now() - start_sim).to_sec()))
        if sim_elapsed >= float(args.door_observe_sim_sec):
            break
        if now_wall - start_wall >= float(args.door_observe_wall_watchdog_sec):
            wall_watchdog_triggered = True
            break
        if args.execute:
            pub.publish(zero)
            zero_count += 1
        doorway = read_json(DOORWAY_PATH)
        doorway_final_decisions.append(doorway.get("final_decision"))
        try:
            pose = pose_tuple(read_odom(timeout_sec=0.2))
        except Exception:
            pose = None
        if pose is not None:
            current_progress = anchor_metrics(anchor, pose).get("anchor_progress_m") if anchor is not None else None
            effective_progress = effective_room_zone_progress_m(anchor, current_progress) if anchor is not None else None
            room_zone_reached = bool(
                effective_progress is not None
                and float(effective_progress) >= float(args.room_zone_start_progress_m)
            )
            new_observations = update_door_landmark_tracks(
                door_tracks,
                door_track_next_ids,
                doorway,
                {},
                pose,
                anchor,
                iteration,
                "door_observe",
                args.door_landmark_corridor_half_width_m,
            )
            annotate_door_cue_actionability(
                new_observations,
                room_zone_reached,
                args.room_zone_start_progress_m,
                float(current_progress) if finite_number(current_progress) else None,
            )
            observations.extend(new_observations)
        loop_count += 1
        time.sleep(0.1)
    if args.execute:
        pub.publish(zero)
        zero_count += 1

    same_side = [item for item in observations if item.get("side") == trigger_side]
    opposite_side = [item for item in observations if item.get("side") in {"left", "right"} and item.get("side") != trigger_side]
    centers = [float(item["center_progress_m"]) for item in same_side if finite_number(item.get("center_progress_m"))]
    widths = [float(item["opening_width_m"]) for item in same_side if finite_number(item.get("opening_width_m"))]
    center_range = max(centers) - min(centers) if centers else None
    width_range = max(widths) - min(widths) if widths else None
    geometry_pass_count = sum(1 for item in same_side if item.get("geometry_pass"))
    soft_geometry_pass_count = sum(1 for item in same_side if item.get("soft_geometry_pass"))
    free_ratio_support_count = sum(
        1 for item in same_side if finite_number(item.get("free_ratio")) and float(item["free_ratio"]) >= 0.65
    )
    confirmed = bool(
        len(same_side) >= 2
        and finite_number(center_range)
        and float(center_range) <= 0.8
        and (geometry_pass_count >= 1 or free_ratio_support_count >= 2 or soft_geometry_pass_count >= 2)
    )
    if confirmed:
        confirm_reason = "same_side_stable_with_geometry_free_ratio_or_soft_geometry_support"
        reject_reason = None
    elif wall_watchdog_triggered:
        confirm_reason = None
        reject_reason = "wall_watchdog_timeout"
    elif len(same_side) < 2:
        confirm_reason = None
        reject_reason = "insufficient_same_side_observations"
    elif not finite_number(center_range) or float(center_range) > 0.8:
        confirm_reason = None
        reject_reason = "center_progress_not_stable"
    else:
        confirm_reason = None
        reject_reason = "insufficient_geometry_free_ratio_or_soft_geometry_support"
    return {
        "diagnostic_only": True,
        "controls_robot": False,
        "controls_next_state": False,
        "calls_runner": False,
        "writes_navigation_target": False,
        "trigger_cue": trigger_cue,
        "observe_sim_sec": float(args.door_observe_sim_sec),
        "observe_sim_elapsed_sec": sim_elapsed,
        "observe_wall_elapsed_sec": time.monotonic() - start_wall,
        "wall_watchdog_sec": float(args.door_observe_wall_watchdog_sec),
        "wall_watchdog_triggered": wall_watchdog_triggered,
        "cmd_topic": topic,
        "zero_count": zero_count,
        "loop_count": loop_count,
        "doorway_read_count": len(doorway_final_decisions),
        "doorway_final_decisions": doorway_final_decisions,
        "observation_count": len(observations),
        "same_side_observation_count": len(same_side),
        "opposite_side_observation_count": len(opposite_side),
        "geometry_pass_count": geometry_pass_count,
        "soft_geometry_pass_count": soft_geometry_pass_count,
        "free_ratio_support_count": free_ratio_support_count,
        "median_center_progress": median_float(centers),
        "center_progress_values": centers,
        "center_progress_range_m": center_range,
        "median_width_m": median_float(widths),
        "width_values": widths,
        "width_range_m": width_range,
        "confirmed_door_after_observe": confirmed,
        "confirm_reason": confirm_reason,
        "reject_reason": reject_reason,
        "observations": observations,
    }


def door_track_summary(
    track: Dict[str, Any],
    anchor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    observations = track.get("observations") if isinstance(track.get("observations"), list) else []
    center_xy_values = [
        [float(item["center_odom_xy"][0]), float(item["center_odom_xy"][1])]
        for item in observations
        if isinstance(item.get("center_odom_xy"), list)
        and len(item["center_odom_xy"]) == 2
        and all(finite_number(value) for value in item["center_odom_xy"])
    ]
    centers = [float(item["center_progress_m"]) for item in observations if finite_number(item.get("center_progress_m"))]
    starts = [float(item["start_progress_m"]) for item in observations if finite_number(item.get("start_progress_m"))]
    ends = [float(item["end_progress_m"]) for item in observations if finite_number(item.get("end_progress_m"))]
    widths = [float(item["opening_width_m"]) for item in observations if finite_number(item.get("opening_width_m"))]
    width_plausible_values = [bool(item.get("width_plausible")) for item in observations]
    geometry_pass_values = [bool(item.get("geometry_pass")) for item in observations]
    free_ratio_values = [float(item["free_ratio"]) for item in observations if finite_number(item.get("free_ratio"))]
    center_range = max(centers) - min(centers) if centers else None
    width_range = max(widths) - min(widths) if widths else None
    median_center_xy = (
        [median_float([value[0] for value in center_xy_values]), median_float([value[1] for value in center_xy_values])]
        if center_xy_values
        else None
    )
    position_std_m = (
        math.sqrt(
            sum(
                (value[0] - float(median_center_xy[0])) ** 2
                + (value[1] - float(median_center_xy[1])) ** 2
                for value in center_xy_values
            )
            / len(center_xy_values)
        )
        if center_xy_values and median_center_xy is not None
        else None
    )
    median_width = median_float(widths)
    position_stable = bool(
        len(observations) >= 3
        and finite_number(position_std_m)
        and float(position_std_m) <= 0.45
        and finite_number(center_range)
        and float(center_range) <= 1.0
    )
    median_free_ratio = median_float(free_ratio_values)
    geometry_pass_count = sum(1 for value in geometry_pass_values if value)
    plausible_width_count = sum(1 for value in width_plausible_values if value)
    has_geometry_support = geometry_pass_count >= 1
    free_ratio_support_count = sum(1 for value in free_ratio_values if value >= 0.65)
    has_free_ratio_support = plausible_width_count >= 2 and free_ratio_support_count >= 2
    door_signal_pass = bool(
        has_geometry_support or has_free_ratio_support
    )
    if has_geometry_support:
        door_signal_reason = "geometry_pass_supported"
    elif has_free_ratio_support:
        door_signal_reason = "free_ratio_support_count_supported"
    elif plausible_width_count >= 2:
        door_signal_reason = "rejected_width_only"
    else:
        door_signal_reason = "rejected_no_geometry_or_free_support"
    summary: Dict[str, Any] = {
        "track_id": track.get("track_id"),
        "side": track.get("side"),
        "count": len(observations),
        "median_center_xy": median_center_xy,
        "median_center_xy_frame": "team_livox_odom",
        "position_std_m": position_std_m,
        "created_iteration": track.get("created_iteration"),
        "last_seen_iteration": track.get("last_seen_iteration"),
        "uses_approximation": any(bool(item.get("approximation")) for item in observations),
        "center_progress_values": centers,
        "center_range_m": center_range,
        "median_center_progress": median_float(centers),
        "median_start_progress": median_float(starts),
        "median_end_progress": median_float(ends),
        "median_width_m": median_width,
        "width_values": widths,
        "width_range_m": width_range,
        "geometry_pass_values": geometry_pass_values,
        "geometry_pass_count": geometry_pass_count,
        "plausible_width_count": plausible_width_count,
        "free_ratio_values": free_ratio_values,
        "median_free_ratio": median_free_ratio,
        "free_ratio_support_count": free_ratio_support_count,
        "position_stable": position_stable,
        "door_signal_pass": door_signal_pass,
        "door_signal_reason": door_signal_reason,
        "duplicate_suspect": False,
        "duplicate_suspect_with": None,
        "duplicate_reason": None,
        "duplicate_group_id": None,
        "duplicate_representative": True,
        "duplicate_representative_reason": "not_duplicate",
        "navigation_ready": False,
        "current_anchor_progress": None,
        "door_progress_delta_m": None,
        "passed_or_expired": False,
        "approach_candidate_valid": False,
        "stable": position_stable,
    }
    if not position_stable or median_center_xy is None or not observations:
        return summary

    side = track.get("side")
    if side not in {"left", "right"}:
        return summary
    d_xy = [float(median_center_xy[0]), float(median_center_xy[1])]
    uses_approximation = bool(summary["uses_approximation"])
    summary["D"] = {
        "xy": d_xy,
        "frame": "team_livox_odom",
        "source": "median_stable_door_track",
        "approximation": uses_approximation,
    }
    if anchor is not None:
        heading = float(anchor["heading_rad"])
        dx = d_xy[0] - float(anchor["x"])
        dy = d_xy[1] - float(anchor["y"])
        projected_progress = dx * math.cos(heading) + dy * math.sin(heading)
        approach_xy = [
            float(anchor["x"]) + math.cos(heading) * projected_progress,
            float(anchor["y"]) + math.sin(heading) * projected_progress,
        ]
        room_normal = (
            [-math.sin(heading), math.cos(heading)]
            if side == "left"
            else [math.sin(heading), -math.cos(heading)]
        )
        summary["A"] = {
            "xy": approach_xy,
            "yaw": heading,
            "frame": "team_livox_odom",
            "source": "door_center_projection_to_corridor_anchor",
            "approximation": uses_approximation,
        }
    else:
        last_observation = observations[-1] if isinstance(observations[-1], dict) else {}
        last_pose = last_observation.get("robot_pose") if isinstance(last_observation.get("robot_pose"), dict) else {}
        yaw = float(last_pose.get("yaw")) if finite_number(last_pose.get("yaw")) else 0.0
        center_x = float(last_observation.get("center_x_m") or 0.0)
        approach_xy = transform_base_xy([center_x, 0.0], (float(last_pose.get("x") or 0.0), float(last_pose.get("y") or 0.0), yaw))
        room_normal = [-math.sin(yaw), math.cos(yaw)] if side == "left" else [math.sin(yaw), -math.cos(yaw)]
        summary["A_estimated"] = {
            "xy": approach_xy,
            "yaw": yaw,
            "frame": "team_livox_odom",
            "source": "latest_robot_heading_centerline_estimate",
            "approximation": True,
        }
    summary["room_normal_odom_xy"] = room_normal
    summary["E_candidates"] = [
        {
            "distance_from_D_m": distance_m,
            "xy": [d_xy[0] + room_normal[0] * distance_m, d_xy[1] + room_normal[1] * distance_m],
            "frame": "team_livox_odom",
            "diagnostic_only": True,
            "approximation": uses_approximation,
        }
        for distance_m in (0.5, 0.7, 0.9, 1.1)
    ]
    return summary


def door_track_payload(track: Dict[str, Any], anchor: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = dict(track)
    payload["summary"] = door_track_summary(track, anchor)
    return payload


def apply_same_side_duplicate_audit(tracks: List[Dict[str, Any]]) -> None:
    track_by_id: Dict[str, Dict[str, Any]] = {}
    adjacency: Dict[str, set[str]] = {}
    for track in tracks:
        summary = track.get("summary") if isinstance(track.get("summary"), dict) else {}
        track_id = summary.get("track_id")
        if not isinstance(track_id, str):
            continue
        track_by_id[track_id] = track
        adjacency[track_id] = set()

    for index, track in enumerate(tracks):
        summary = track.get("summary") if isinstance(track.get("summary"), dict) else {}
        duplicate_pairs: List[Dict[str, Any]] = []
        for other in tracks[index + 1 :]:
            other_summary = other.get("summary") if isinstance(other.get("summary"), dict) else {}
            center_i = summary.get("median_center_progress")
            center_j = other_summary.get("median_center_progress")
            start_i = summary.get("median_start_progress")
            start_j = other_summary.get("median_start_progress")
            end_i = summary.get("median_end_progress")
            end_j = other_summary.get("median_end_progress")
            width_i = summary.get("median_width_m")
            width_j = other_summary.get("median_width_m")
            if not all(finite_number(value) for value in (center_i, center_j, start_i, start_j, end_i, end_j, width_i, width_j)):
                continue
            center_gap = abs(float(center_i) - float(center_j))
            overlap = max(0.0, min(float(end_i), float(end_j)) - max(float(start_i), float(start_j)))
            min_width = min(float(width_i), float(width_j))
            overlap_ratio = overlap / min_width if min_width > 1e-6 else 0.0
            reason = "center_gap_small" if center_gap < 1.0 else ("interval_overlap" if overlap_ratio > 0.4 else None)
            if reason is None:
                continue
            pair = {
                "other_track_id": other_summary.get("track_id"),
                "center_gap_m": center_gap,
                "interval_overlap_m": overlap,
                "interval_overlap_ratio": overlap_ratio,
                "duplicate_reason": reason,
            }
            duplicate_pairs.append(pair)
            other_pairs = other_summary.setdefault("duplicate_suspect_pairs", [])
            other_pairs.append(
                {
                    "other_track_id": summary.get("track_id"),
                    "center_gap_m": center_gap,
                    "interval_overlap_m": overlap,
                    "interval_overlap_ratio": overlap_ratio,
                    "duplicate_reason": reason,
                }
            )
            this_track_id = summary.get("track_id")
            other_track_id = other_summary.get("track_id")
            if isinstance(this_track_id, str) and isinstance(other_track_id, str):
                adjacency.setdefault(this_track_id, set()).add(other_track_id)
                adjacency.setdefault(other_track_id, set()).add(this_track_id)
        if duplicate_pairs:
            summary["duplicate_suspect_pairs"] = duplicate_pairs

    for track in tracks:
        summary = track.get("summary") if isinstance(track.get("summary"), dict) else {}
        pairs = summary.get("duplicate_suspect_pairs") if isinstance(summary.get("duplicate_suspect_pairs"), list) else []
        if pairs:
            primary = pairs[0]
            summary["duplicate_suspect"] = True
            summary["duplicate_suspect_with"] = primary.get("other_track_id")
            summary["duplicate_reason"] = primary.get("duplicate_reason")

    visited: set[str] = set()
    group_index = 0
    for track_id, neighbors in adjacency.items():
        if track_id in visited or not neighbors:
            continue
        component: List[str] = []
        pending = [track_id]
        visited.add(track_id)
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        group_tracks = [track_by_id[item] for item in component if item in track_by_id]
        if not group_tracks:
            continue
        side = str(group_tracks[0]["summary"].get("side") or "unknown")
        group_id = f"{side}_duplicate_group_{group_index:02d}"
        group_index += 1

        def representative_sort_key(candidate: Dict[str, Any]) -> Tuple[int, int, int, float, int, str]:
            candidate_summary = candidate["summary"]
            position_std = candidate_summary.get("position_std_m")
            return (
                -int(bool(candidate_summary.get("door_signal_pass"))),
                -int(candidate_summary.get("free_ratio_support_count") or 0),
                -int(candidate_summary.get("plausible_width_count") or 0),
                float(position_std) if finite_number(position_std) else float("inf"),
                -int(candidate_summary.get("count") or 0),
                str(candidate_summary.get("track_id") or ""),
            )

        representative = min(group_tracks, key=representative_sort_key)
        representative_id = representative["summary"].get("track_id")
        for candidate in group_tracks:
            candidate_summary = candidate["summary"]
            is_representative = candidate is representative
            candidate_summary["duplicate_group_id"] = group_id
            candidate_summary["duplicate_representative"] = is_representative
            candidate_summary["duplicate_representative_reason"] = (
                "selected_by_door_signal_free_ratio_support_count_plausible_width_position_std_count"
                if is_representative
                else f"lower_rank_than:{representative_id}"
            )

    for track in tracks:
        summary = track.get("summary") if isinstance(track.get("summary"), dict) else {}
        summary["navigation_ready"] = bool(
            summary.get("position_stable")
            and summary.get("door_signal_pass")
            and summary.get("duplicate_representative")
        )


def apply_door_approach_reachability(
    tracks: List[Dict[str, Any]],
    current_anchor_progress: Optional[float],
) -> None:
    for track in tracks:
        summary = track.get("summary") if isinstance(track.get("summary"), dict) else {}
        door_progress = summary.get("median_center_progress")
        if finite_number(current_anchor_progress) and finite_number(door_progress):
            delta = float(door_progress) - float(current_anchor_progress)
            passed_or_expired = delta < 0.3
            summary["current_anchor_progress"] = float(current_anchor_progress)
            summary["door_progress_delta_m"] = delta
            summary["passed_or_expired"] = passed_or_expired
            summary["approach_candidate_valid"] = bool(
                summary.get("navigation_ready") and not passed_or_expired
            )


def door_cue_ahead_observations(tracks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    for track in tracks:
        observations = track.get("observations") if isinstance(track.get("observations"), list) else []
        for observation in observations:
            if isinstance(observation, dict) and observation.get("raw_door_cue_ahead", observation.get("door_cue_ahead")):
                cues.append(observation)
    return cues


def door_landmark_debug_payload(
    door_tracks: Dict[str, List[Dict[str, Any]]],
    anchor: Optional[Dict[str, Any]],
    current_anchor_progress: Optional[float] = None,
) -> Dict[str, Any]:
    left_tracks = [door_track_payload(track, anchor) for track in door_tracks.get("left", [])]
    right_tracks = [door_track_payload(track, anchor) for track in door_tracks.get("right", [])]
    apply_same_side_duplicate_audit(left_tracks)
    apply_same_side_duplicate_audit(right_tracks)
    apply_door_approach_reachability(left_tracks + right_tracks, current_anchor_progress)
    position_stable_left_tracks = [track for track in left_tracks if track["summary"].get("position_stable")]
    position_stable_right_tracks = [track for track in right_tracks if track["summary"].get("position_stable")]
    navigation_ready_left_tracks = [track for track in left_tracks if track["summary"].get("navigation_ready")]
    navigation_ready_right_tracks = [track for track in right_tracks if track["summary"].get("navigation_ready")]
    navigation_ready_tracks = navigation_ready_left_tracks + navigation_ready_right_tracks
    approach_candidate_valid_tracks = [
        track for track in navigation_ready_tracks if track["summary"].get("approach_candidate_valid")
    ]
    passed_or_expired_tracks = [
        track for track in left_tracks + right_tracks if track["summary"].get("passed_or_expired")
    ]
    duplicate_suspect_tracks = [
        track for track in left_tracks + right_tracks if track["summary"].get("duplicate_suspect")
    ]
    door_cues = door_cue_ahead_observations(left_tracks + right_tracks)
    actionable_door_cues = [observation for observation in door_cues if observation.get("actionable_door_cue")]
    rejected_before_room_zone = [
        observation for observation in door_cues if observation.get("actionable_reject_reason") == "before_room_zone"
    ]
    rejected_broad_bilateral = [
        observation for observation in door_cues if observation.get("actionable_reject_reason") == "broad_bilateral_opening"
    ]
    all_stable = []
    for track in position_stable_left_tracks + position_stable_right_tracks:
        summary = track["summary"]
        all_stable.append(
            {
                "track_id": summary.get("track_id"),
                "side": summary.get("side"),
                "D": summary.get("D"),
                "A": summary.get("A") or summary.get("A_estimated"),
                "E_candidates": summary.get("E_candidates"),
            }
        )
    summary = {
        "coordinate_frame": "team_livox_odom",
        "odom_topic": ODOM_TOPIC,
        "diagnostic_only": True,
        "controls_robot": False,
        "controls_next_state": False,
        "controls_committed_room_side_gap": False,
        "left_track_count": len(left_tracks),
        "right_track_count": len(right_tracks),
        "position_stable_left_count": len(position_stable_left_tracks),
        "position_stable_right_count": len(position_stable_right_tracks),
        "navigation_ready_left_count": len(navigation_ready_left_tracks),
        "navigation_ready_right_count": len(navigation_ready_right_tracks),
        "navigation_ready_total": len(navigation_ready_tracks),
        "approach_candidate_valid_count": len(approach_candidate_valid_tracks),
        "door_cue_ahead_count": len(door_cues),
        "raw_door_cue_ahead_count": len(door_cues),
        "actionable_door_cue_count": len(actionable_door_cues),
        "rejected_before_room_zone_count": len(rejected_before_room_zone),
        "rejected_broad_bilateral_opening_count": len(rejected_broad_bilateral),
        "first_door_cue_ahead": door_cues[0] if door_cues else None,
        "first_actionable_door_cue": actionable_door_cues[0] if actionable_door_cues else None,
        "passed_or_expired_count": len(passed_or_expired_tracks),
        "current_anchor_progress": current_anchor_progress if finite_number(current_anchor_progress) else None,
        "duplicate_suspect_count": len(duplicate_suspect_tracks),
    }
    return {
        "summary": summary,
        "left_tracks": left_tracks,
        "right_tracks": right_tracks,
        "position_stable_left_tracks": position_stable_left_tracks,
        "position_stable_right_tracks": position_stable_right_tracks,
        "navigation_ready_left_tracks": navigation_ready_left_tracks,
        "navigation_ready_right_tracks": navigation_ready_right_tracks,
        "navigation_ready_tracks": navigation_ready_tracks,
        "approach_candidate_valid_tracks": approach_candidate_valid_tracks,
        "passed_or_expired_tracks": passed_or_expired_tracks,
        "duplicate_suspect_tracks": duplicate_suspect_tracks,
        "door_cue_ahead_observations": door_cues,
        "first_door_cue_ahead": door_cues[0] if door_cues else None,
        "actionable_door_cue_observations": actionable_door_cues,
        "first_actionable_door_cue": actionable_door_cues[0] if actionable_door_cues else None,
        "all_stable_door_landmarks": all_stable,
    }


def approach_point_debug_payload(
    approach_tracks: Sequence[Dict[str, Any]],
    anchor: Optional[Dict[str, Any]],
    robot_pose: Tuple[float, float, float],
    current_anchor_progress: Optional[float],
) -> Dict[str, Any]:
    """Build diagnostic-only A-point geometry; this never creates a navigation target."""
    candidates: List[Dict[str, Any]] = []
    for track in approach_tracks:
        summary = track.get("summary") if isinstance(track.get("summary"), dict) else {}
        side = summary.get("side")
        d = summary.get("D") if isinstance(summary.get("D"), dict) else {}
        a = summary.get("A") if isinstance(summary.get("A"), dict) else summary.get("A_estimated")
        if not isinstance(a, dict):
            a = {}
        d_xy = d.get("xy")
        a_xy = a.get("xy")
        frame = a.get("frame") if a.get("frame") == d.get("frame") else None
        entry: Dict[str, Any] = {
            "track_id": summary.get("track_id"),
            "side": side,
            "D_xy": d_xy,
            "A_xy": a_xy,
            "E_candidates": summary.get("E_candidates"),
            "frame": frame,
            "current_robot_pose": {"x": robot_pose[0], "y": robot_pose[1], "yaw": robot_pose[2]},
            "current_anchor_progress": current_anchor_progress if finite_number(current_anchor_progress) else None,
            "diagnostic_only": True,
        }
        if not (
            isinstance(d_xy, list)
            and isinstance(a_xy, list)
            and len(d_xy) == 2
            and len(a_xy) == 2
            and all(finite_number(value) for value in d_xy + a_xy)
        ):
            entry["approach_point_debug_pass"] = False
            entry["reason"] = "D_or_A_xy_unavailable"
            candidates.append(entry)
            continue

        a_base_x, a_base_y = target_base_xy(a_xy, robot_pose)
        a_distance = math.hypot(a_base_x, a_base_y)
        heading_error = normalize_angle(math.atan2(a_base_y, a_base_x))
        entry["A_target_base_xy"] = [a_base_x, a_base_y]
        entry["A_target_distance_m"] = a_distance
        entry["A_target_heading_error_rad"] = heading_error
        entry["A_target_in_front"] = a_base_x > 0.3
        entry["D_to_A_distance_m"] = math.hypot(float(d_xy[0]) - float(a_xy[0]), float(d_xy[1]) - float(a_xy[1]))
        entry["D_to_A_direction_expected"] = "projection_to_corridor_center"

        a_progress: Optional[float] = None
        a_lateral_error: Optional[float] = None
        if anchor is not None:
            heading = float(anchor["heading_rad"])
            dx = float(a_xy[0]) - float(anchor["x"])
            dy = float(a_xy[1]) - float(anchor["y"])
            a_progress = dx * math.cos(heading) + dy * math.sin(heading)
            a_lateral_error = -math.sin(heading) * dx + math.cos(heading) * dy
        entry["A_progress_m"] = a_progress
        entry["A_progress_delta_m"] = (
            a_progress - float(current_anchor_progress)
            if finite_number(a_progress) and finite_number(current_anchor_progress)
            else None
        )
        entry["A_lateral_error_to_corridor_center_m"] = a_lateral_error
        entry["A_near_corridor_center"] = bool(
            finite_number(a_lateral_error) and abs(float(a_lateral_error)) <= 0.35
        )

        expected_normal = summary.get("room_normal_odom_xy")
        e_candidates = summary.get("E_candidates") if isinstance(summary.get("E_candidates"), list) else []
        side_consistent = bool(
            side in {"left", "right"}
            and e_candidates
            and isinstance(expected_normal, list)
            and len(expected_normal) == 2
        )
        if side_consistent:
            for e_candidate in e_candidates:
                e_xy = e_candidate.get("xy") if isinstance(e_candidate, dict) else None
                if not (isinstance(e_xy, list) and len(e_xy) == 2 and all(finite_number(value) for value in e_xy)):
                    side_consistent = False
                    break
                lateral_projection = (float(e_xy[0]) - float(d_xy[0])) * float(expected_normal[0]) + (
                    float(e_xy[1]) - float(d_xy[1])
                ) * float(expected_normal[1])
                if lateral_projection <= 0.0:
                    side_consistent = False
                    break
        entry["side_consistency_check"] = side_consistent
        entry["approach_point_debug_pass"] = bool(
            entry["A_target_in_front"]
            and finite_number(entry["A_progress_delta_m"])
            and float(entry["A_progress_delta_m"]) > 0.3
            and entry["A_near_corridor_center"]
            and side_consistent
            and frame == "team_livox_odom"
        )
        candidates.append(entry)
    return {
        "coordinate_frame": "team_livox_odom",
        "odom_topic": ODOM_TOPIC,
        "diagnostic_only": True,
        "controls_robot": False,
        "controls_next_state": False,
        "calls_runner": False,
        "writes_navigation_target": False,
        "approach_point_debug_count": len(candidates),
        "approach_point_debug_pass_count": sum(1 for item in candidates if item.get("approach_point_debug_pass")),
        "first_approach_point_debug": candidates[0] if candidates else None,
        "candidates": candidates,
    }


def commit_room_side_gap_if_ready(args: argparse.Namespace, status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not status.get("available"):
        return None
    stable_count = int(status.get("stable_observation_count") or 0)
    if stable_count < int(args.room_side_gap_required_stable_observations):
        return None
    selected = status.get("selected_opening") if isinstance(status.get("selected_opening"), dict) else {}
    start_x = status.get("start_x_m", selected.get("start_x_m"))
    end_x = status.get("end_x_m", selected.get("end_x_m"))
    width_m = status.get("opening_width_m", selected.get("width_m"))
    target_progress = status.get("alignment_target_anchor_progress_m")
    last_seen_progress = status.get("anchor_progress_m")
    if not (
        finite_number(start_x)
        and finite_number(end_x)
        and finite_number(width_m)
        and finite_number(target_progress)
        and finite_number(last_seen_progress)
    ):
        return None
    if not bool(status.get("width_plausible", selected.get("width_plausible"))):
        return None
    if float(start_x) > 0.6 or float(end_x) < 1.0:
        return None
    if not (bool(status.get("geometry_support")) or status.get("doorway_final_decision") == "DOORWAY_CANDIDATE_READY"):
        return None
    side = status.get("door_side")
    if side not in {"left", "right"}:
        return None
    return {
        "source": "committed_room_side_gap",
        "side": side,
        "door_side": side,
        "target_progress_m": float(target_progress),
        "alignment_target_anchor_progress_m": float(target_progress),
        "last_seen_progress_m": float(last_seen_progress),
        "center_x_base_m": float(status["center_x_base_m"]) if finite_number(status.get("center_x_base_m")) else None,
        "start_x_m": float(start_x),
        "end_x_m": float(end_x),
        "width_m": float(width_m),
        "opening_width_m": float(width_m),
        "stable_count": stable_count,
        "created_wall_time_sec": time.time(),
        "selected_opening": selected,
    }


def update_committed_room_side_gap(
    args: argparse.Namespace,
    committed: Optional[Dict[str, Any]],
    status: Dict[str, Any],
    current_progress: Optional[float],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    result: Dict[str, Any] = {"committed_exists_before": isinstance(committed, dict)}
    if isinstance(committed, dict) and finite_number(current_progress) and finite_number(committed.get("target_progress_m")):
        overshoot = float(current_progress) - float(committed["target_progress_m"])
        result["overshoot_m"] = overshoot
        if overshoot > 0.45:
            result.update({"cleared": True, "clear_reason": "committed_room_side_gap_overshot"})
            committed = None
    if (
        isinstance(committed, dict)
        and status.get("available")
        and status.get("door_side") in {"left", "right"}
        and committed.get("side") in {"left", "right"}
        and status.get("door_side") != committed.get("side")
        and int(status.get("stable_observation_count") or 0) >= 2
    ):
        result.update({"cleared": True, "clear_reason": "opposite_side_candidate"})
        committed = None
    new_commit = commit_room_side_gap_if_ready(args, status)
    if new_commit is not None and not isinstance(committed, dict):
        committed = new_commit
        result.update({"committed": True, "commit_reason": "stable_room_side_gap_candidate"})
    elif new_commit is not None and isinstance(committed, dict) and committed.get("side") == new_commit.get("side"):
        committed["last_seen_progress_m"] = new_commit["last_seen_progress_m"]
        committed["stable_count"] = max(int(committed.get("stable_count") or 0), int(new_commit.get("stable_count") or 0))
        result.update({"kept": True, "keep_reason": "same_side_committed_target_locked"})
    elif isinstance(committed, dict) and status.get("available") and status.get("door_side") == committed.get("side"):
        if finite_number(status.get("anchor_progress_m")):
            committed["last_seen_progress_m"] = float(status["anchor_progress_m"])
        committed["stable_count"] = max(int(committed.get("stable_count") or 0), int(status.get("stable_observation_count") or 0))
        result.update({"kept": True, "keep_reason": "same_side_candidate_update"})
    elif isinstance(committed, dict):
        result.update({"kept": True, "keep_reason": "committed_room_side_gap_waiting_for_alignment"})
    result["committed_exists_after"] = isinstance(committed, dict)
    if isinstance(committed, dict):
        result["committed_room_side_gap"] = committed
    return committed, result


def committed_room_side_gap_trigger_status(
    args: argparse.Namespace,
    committed: Optional[Dict[str, Any]],
    current_progress: Optional[float],
) -> Dict[str, Any]:
    if not isinstance(committed, dict):
        return {"trigger_ready": False, "reason": "no_committed_room_side_gap"}
    if not (finite_number(current_progress) and finite_number(committed.get("target_progress_m"))):
        return {"trigger_ready": False, "reason": "committed_room_side_gap_progress_unavailable"}
    age = time.time() - float(committed.get("created_wall_time_sec") or time.time())
    longitudinal_error = float(committed["target_progress_m"]) - float(current_progress)
    overshoot_m = float(current_progress) - float(committed["target_progress_m"])
    progress_since_last_seen = (
        float(current_progress) - float(committed["last_seen_progress_m"])
        if finite_number(committed.get("last_seen_progress_m"))
        else None
    )
    within_alignment_window = abs(longitudinal_error) <= float(args.room_side_gap_turn_alignment_tolerance_m)
    stale_age_exceeded = age > float(args.room_side_gap_commit_max_age_sec)
    no_progress_since_seen = bool(
        progress_since_last_seen is not None
        and progress_since_last_seen <= float(args.room_side_gap_commit_stale_no_progress_epsilon_m)
    )
    status = dict(committed)
    status.update(
        {
            "source": "committed_room_side_gap",
            "trigger_ready": False,
            "trigger_source": "committed_room_side_gap",
            "anchor_progress_m": float(current_progress),
            "alignment_longitudinal_error_m": longitudinal_error,
            "overshoot_m": overshoot_m,
            "age_wall_time_sec": age,
            "stale_age_exceeded": stale_age_exceeded,
            "within_alignment_window": within_alignment_window,
            "progress_since_last_seen_m": progress_since_last_seen,
            "stale_no_progress_epsilon_m": float(args.room_side_gap_commit_stale_no_progress_epsilon_m),
        }
    )
    if overshoot_m > 0.45:
        status["reason"] = "committed_room_side_gap_overshot"
        return status
    if within_alignment_window:
        status["trigger_ready"] = True
        status["trigger_source"] = "committed_room_side_gap_alignment_reached"
        status["reason"] = "committed_room_side_gap_alignment_reached"
    elif stale_age_exceeded and no_progress_since_seen:
        status["reason"] = "committed_room_side_gap_stale_no_progress"
    else:
        status["reason"] = "committed_room_side_gap_alignment_pending"
    return status


def maybe_latch_doorway_profile(
    args: argparse.Namespace,
    anchor: Optional[Dict[str, Any]],
    pose: Tuple[float, float, float],
    doorway: Dict[str, Any],
    current_latch: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    status = doorway_profile_opening_status(doorway)
    if anchor is None or not status.get("available") or not status.get("partial_opening_observed"):
        return current_latch, status
    metrics = anchor_metrics(anchor, pose)
    current_progress = float(metrics["anchor_progress_m"])
    observed_center_x = float(status["center_x_base_m"])
    observed_forward_x = max(float(args.doorway_latch_min_forward_m), observed_center_x)
    if current_latch is not None:
        status["latched"] = True
        status["latch_reused"] = True
        if current_latch.get("door_side") != status.get("door_side"):
            status["latch_cleared"] = True
            status["latch_clear_reason"] = "door_side_changed"
            return None, status
        previous_observed = current_latch.get("last_observed_center_x_base_m")
        if not finite_number(previous_observed):
            previous_observed = current_latch.get("center_x_base_m_at_latch")
        center_delta = abs(observed_center_x - float(previous_observed)) if finite_number(previous_observed) else None
        stable_count = int(current_latch.get("stable_partial_observation_count") or 0)
        if center_delta is not None and center_delta <= float(args.doorway_partial_latch_center_stability_tolerance_m):
            stable_count += 1
        else:
            stable_count = 1
        observation_count = int(current_latch.get("partial_observation_count") or 1) + 1
        current_latch["partial_observation_count"] = observation_count
        current_latch["stable_partial_observation_count"] = stable_count
        current_latch["last_observed_center_x_base_m"] = observed_center_x
        current_latch["last_observed_opening_width_m"] = status.get("opening_width_m")
        current_latch["last_observed_anchor_progress_m"] = current_progress
        current_latch["last_observed_selected_opening"] = status.get("selected_opening")
        status["latch_center_delta_m"] = center_delta
        status["stable_partial_observation_count"] = stable_count
        status["partial_observation_count"] = observation_count
        if (
            not current_latch.get("partial_center_stable")
            and stable_count >= int(args.doorway_partial_latch_required_stable_observations)
            and observed_center_x <= float(args.doorway_partial_latch_max_stable_center_x_m)
        ):
            current_latch["partial_center_stable"] = True
            current_latch["partial_center_stable_wall_time_sec"] = time.time()
            current_latch["center_x_base_m_at_latch"] = observed_center_x
            current_latch["opening_width_m"] = status.get("opening_width_m")
            current_latch["selected_opening"] = status.get("selected_opening")
            current_latch["target_anchor_progress_m"] = current_progress + observed_forward_x
            current_latch["target_refined_from_stable_partial"] = True
            current_latch.pop("partial_latch_exhausted", None)
            status["latch_updated"] = True
            status["latch_update_reason"] = "stable_partial_center_refined_target"
        elif (
            not current_latch.get("partial_center_stable")
            and observed_center_x > float(args.doorway_partial_latch_max_stable_center_x_m)
        ):
            status["latch_update_suppressed_reason"] = "observed_center_too_far_for_stable_partial"
        else:
            status["latch_updated"] = False
        status["latched"] = True
        return current_latch, status
    latch = {
        "door_side": status.get("door_side"),
        "source": "partial_opening_profile",
        "latched_wall_time_sec": time.time(),
        "latched_anchor_progress_m": current_progress,
        "target_anchor_progress_m": current_progress + observed_forward_x,
        "center_x_base_m_at_latch": observed_center_x,
        "opening_width_m": status.get("opening_width_m"),
        "selected_opening": status.get("selected_opening"),
        "doorway_final_decision_at_latch": status.get("doorway_final_decision"),
        "opening_center_estimated_at_latch": bool(status.get("opening_center_estimated")),
        "partial_opening_observed_at_latch": bool(status.get("partial_opening_observed")),
        "partial_center_stable": False,
        "partial_observation_count": 1,
        "stable_partial_observation_count": 1,
        "last_observed_center_x_base_m": observed_center_x,
        "last_observed_anchor_progress_m": current_progress,
        "extension_count": 0,
    }
    status["latched"] = True
    status["latch_reused"] = False
    return latch, status


def doorway_latch_alignment(
    latch: Optional[Dict[str, Any]],
    anchor: Optional[Dict[str, Any]],
    pose: Tuple[float, float, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if latch is None or anchor is None:
        return {"available": False}
    metrics = anchor_metrics(anchor, pose)
    current_progress = float(metrics["anchor_progress_m"])
    target_progress = float(latch["target_anchor_progress_m"])
    longitudinal_error = target_progress - current_progress
    progress_target_reached = longitudinal_error <= float(args.doorway_latch_alignment_tolerance_m)
    partial_only = bool(latch.get("partial_opening_observed_at_latch")) and not bool(
        latch.get("opening_center_estimated_at_latch")
    )
    partial_center_stable = bool(latch.get("partial_center_stable"))
    alignment_ready = progress_target_reached and not partial_only
    return {
        "available": True,
        "door_side": latch.get("door_side"),
        "current_anchor_progress_m": current_progress,
        "target_anchor_progress_m": target_progress,
        "longitudinal_error_m": longitudinal_error,
        "progress_target_reached": progress_target_reached,
        "partial_only": partial_only,
        "partial_center_stable": partial_center_stable,
        "stable_partial_observation_count": latch.get("stable_partial_observation_count"),
        "needs_more_centerline_observation": progress_target_reached and partial_only,
        "alignment_ready": alignment_ready,
        "overshot": longitudinal_error < -float(args.doorway_latch_overshoot_tolerance_m),
        "alignment_tolerance_m": float(args.doorway_latch_alignment_tolerance_m),
    }


def extend_partial_latch_target_if_needed(
    latch: Optional[Dict[str, Any]],
    alignment: Dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if latch is None:
        return False
    if not (alignment.get("partial_only") and alignment.get("progress_target_reached")):
        return False
    extension_count = int(latch.get("extension_count") or 0)
    max_extensions = int(args.doorway_partial_latch_max_extensions)
    if extension_count >= max_extensions:
        alignment["partial_latch_extension_available"] = False
        latch["partial_latch_exhausted"] = True
        return False
    current_progress = float(alignment["current_anchor_progress_m"])
    latch["target_anchor_progress_m"] = current_progress + float(args.doorway_partial_latch_extra_observe_m)
    latch["extension_count"] = extension_count + 1
    latch["last_extension_wall_time_sec"] = time.time()
    alignment["partial_latch_extended"] = True
    alignment["extended_target_anchor_progress_m"] = latch["target_anchor_progress_m"]
    return True


def room_entry_target_danger(diagnostics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if not diagnostics.get("target_xy_valid"):
        return {"dangerous": False, "reason": "target_xy_unavailable"}
    base_xy = diagnostics.get("target_base_xy")
    if not (isinstance(base_xy, list) and len(base_xy) == 2 and all(finite_number(v) for v in base_xy)):
        return {"dangerous": False, "reason": "target_base_xy_unavailable"}
    x_base = float(base_xy[0])
    y_base = float(base_xy[1])
    heading = math.atan2(y_base, x_base)
    reasons: List[str] = []
    if x_base <= args.enter_room_danger_min_target_x_base_m:
        reasons.append("target_not_in_front")
    if abs(y_base) > args.enter_room_danger_max_abs_target_y_base_m:
        reasons.append("target_lateral_error_too_large")
    if abs(heading) > args.enter_room_danger_max_abs_heading_error_rad:
        reasons.append("target_heading_error_too_large")
    return {
        "dangerous": bool(reasons),
        "reason": ",".join(reasons) if reasons else "pass",
        "target_base_xy": [x_base, y_base],
        "target_heading_base_rad": heading,
    }


def uses_room_entry_local_nav_profile(state: str) -> bool:
    """Select only the low-level room-entry DWA profile for room-side targets.

    ``PORTAL_P_THROUGH`` remains an R37 high-level target semantic.  This
    helper deliberately does not restore the legacy ENTER_ROOM state-machine
    authority, a doorway target, or a fixed moving-turn trajectory.
    """
    return state in {"DOORWAY_VERIFY", "ENTER_ROOM", "PORTAL_P_THROUGH", "ROOM_SEARCH", "ROOM_EXIT"}


def room_local_guarded_online_validation_active(args: argparse.Namespace, local_control_mode: str) -> bool:
    """Keep Phase-2/3 online authority explicit and confined to ROOM_LOCAL."""
    return bool(getattr(args, "enable_room_local_guarded_online_validation", False)) and local_control_mode == "ROOM_LOCAL"


def guarded_room_local_validation_evidence_gate_reason(
    args: argparse.Namespace,
    local_control_mode: str,
    evidence_status: Optional[RoomLocalValidationEvidenceStatusCache],
) -> Optional[str]:
    """Return only a known current-run evidence stop at the next real ROOM_LOCAL boundary."""
    if not (
        bool(getattr(args, "execute", False))
        and room_local_guarded_online_validation_active(args, local_control_mode)
        and evidence_status is not None
    ):
        return None
    return evidence_status.insufficient_reason()


def room_local_validation_abort_reason(attempt: Dict[str, Any]) -> Optional[str]:
    """Translate only the validation terminal result; preserve all other runner semantics."""
    runner = attempt.get("runner") if isinstance(attempt, dict) else None
    if not isinstance(runner, dict) or str(runner.get("runner_final_decision") or "") != "ROOM_LOCAL_ONLINE_VALIDATION_ABORT":
        return None
    return str(runner.get("validation_abort_reason") or "ROOM_LOCAL_ONLINE_VALIDATION_ABORT")


def runner_cmd(
    args: argparse.Namespace,
    *,
    state: str,
    runtime_sec: float,
    max_steps: int,
    local_control_mode: str = "TRANSIT",
    validation_invocation_id: str = "",
    observation_orientation_aim_yaw_odom_rad: Optional[float] = None,
) -> List[str]:
    if local_control_mode not in {"TRANSIT", "ROOM_LOCAL"}:
        raise ValueError(f"unsupported_local_control_mode:{local_control_mode}")
    max_linear_x = float(args.max_linear_x)
    if state == "ENTER_BUILDING":
        max_linear_x = float(args.entry_max_linear_x)
    elif state == "FOLLOW_CORRIDOR":
        max_linear_x = float(args.corridor_max_linear_x)
    cmd = [
        sys.executable,
        "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py",
        "--input-timeout-sec",
        str(args.input_timeout_sec),
        "--max-runtime-sec",
        str(runtime_sec),
        "--wall-watchdog-sec",
        str(args.runner_wall_watchdog_sec),
        "--max-steps",
        str(max_steps),
        "--robot-radius-m",
        str(args.robot_radius_m),
        "--command-slice-sec",
        str(args.command_slice_sec),
        "--goal-tolerance-m",
        str(args.goal_tolerance_m),
        "--max-linear-x",
        str(max_linear_x),
    ]
    if args.use_imu_velocity_follower:
        cmd.extend(["--cmd-topic", args.follower_raw_cmd_topic, "--disable-imu-heading-hold"])
    if state == "ENTER_BUILDING" and args.disable_entry_pointcloud_wall_heading:
        cmd.append("--disable-pointcloud-wall-heading")
    if state == "FOLLOW_CORRIDOR" and args.disable_corridor_pointcloud_wall_heading:
        cmd.append("--disable-pointcloud-wall-heading")
    if uses_room_entry_local_nav_profile(state) and args.disable_enter_room_pointcloud_wall_heading:
        cmd.append("--disable-pointcloud-wall-heading")
    if state == "ENTER_BUILDING":
        cmd.extend(
            [
                "--enforce-min-forward-speed",
                "--min-linear-x",
                str(args.entry_min_linear_x),
                "--speed-weight",
                str(args.entry_speed_weight),
                "--clearance-weight",
                str(args.entry_clearance_weight),
                "--dwa-predict-time",
                str(args.entry_dwa_predict_time),
                "--max-linear-accel",
                str(args.entry_max_linear_accel),
                "--distance-speed-gain",
                str(args.entry_distance_speed_gain),
            ]
        )
    if state == "FOLLOW_CORRIDOR":
        cmd.extend(
            [
                "--enforce-min-forward-speed",
                "--min-linear-x",
                str(args.corridor_min_linear_x),
                "--speed-weight",
                str(args.corridor_speed_weight),
                "--clearance-weight",
                str(args.corridor_clearance_weight),
                "--dwa-predict-time",
                str(args.corridor_dwa_predict_time),
                "--max-linear-accel",
                str(args.corridor_max_linear_accel),
                "--distance-speed-gain",
                str(args.corridor_distance_speed_gain),
            ]
        )
    if uses_room_entry_local_nav_profile(state):
        min_linear_x = args.doorway_verify_min_linear_x if state == "DOORWAY_VERIFY" else args.enter_room_min_linear_x
        if state in {"PORTAL_P_THROUGH", "ROOM_SEARCH", "ROOM_EXIT"}:
            # R39/R40 freeze: any selected nonzero forward command is at least
            # the Stair-qualified 0.30 m/s; do not re-admit the historical
            # 0.12 m/s ENTER_ROOM command as an executable action.
            min_linear_x = max(float(min_linear_x), 0.30)
        max_angular_z = args.doorway_verify_max_angular_z if state == "DOORWAY_VERIFY" else args.enter_room_max_angular_z
        # P_through is the only frozen entry phase receiving this reduced cap.
        # P_pre is run as FOLLOW_CORRIDOR and all later ROOM_SEARCH targets keep
        # their existing room-local profile.
        if state == "PORTAL_P_THROUGH":
            max_angular_z = args.p_through_max_angular_z
        max_steps_runtime_gain = args.doorway_verify_distance_speed_gain if state == "DOORWAY_VERIFY" else args.enter_room_distance_speed_gain
        cmd.extend(
            [
                "--enforce-min-forward-speed",
                "--min-linear-x",
                str(min_linear_x),
                "--max-angular-z",
                str(max_angular_z),
                "--max-linear-accel",
                str(args.enter_room_max_linear_accel),
                "--max-angular-accel",
                str(args.enter_room_max_angular_accel),
                "--distance-speed-gain",
                str(max_steps_runtime_gain),
                "--target-heading-blend-weight",
                str(args.enter_room_target_heading_blend_weight),
                "--max-target-heading-correction-rad",
                str(args.enter_room_max_target_heading_correction_rad),
                "--target-lateral-correction-angular-z",
                str(args.enter_room_target_lateral_correction_angular_z),
                "--target-lateral-correction-weight",
                str(args.enter_room_target_lateral_correction_weight),
                "--dwa-predict-time",
                str(args.enter_room_dwa_predict_time),
            ]
        )
    if state == "ROOM_SEARCH":
        cmd.extend(["--additional-clearance-margin-m", str(args.room_search_extra_clearance_margin_m)])
    if local_control_mode == "ROOM_LOCAL":
        # Phase 1 wires an explicit contract only.  The runner keeps its legacy
        # DWA winner authority until productive admission is separately approved.
        cmd.extend(["--local-control-mode", "ROOM_LOCAL"])
        if room_local_guarded_online_validation_active(args, local_control_mode):
            # The runner rejects either half of this pair.  Keep activation at
            # the mission boundary rather than inferring it from target data.
            cmd.extend([
                "--room-local-phase2-productive-admission", "phase3_guarded_execute",
                "--room-local-phase3-orientation-recovery", "phase3_guarded_execute",
                "--validation-run-id", str(os.environ.get("STATE_MACHINE_RUN_ID", "")),
                "--validation-runner-invocation-id", str(validation_invocation_id),
            ])
    if observation_orientation_aim_yaw_odom_rad is not None:
        if not finite_number(observation_orientation_aim_yaw_odom_rad):
            raise ValueError("observation_orientation_aim_invalid")
        cmd.extend(["--observation-orientation-aim-yaw-odom-rad", str(float(observation_orientation_aim_yaw_odom_rad))])
    if args.execute:
        cmd.insert(2, "--execute")
    return cmd


def summarize_runner() -> Dict[str, Any]:
    data = read_json(RUNNER_SUMMARY_PATH)
    steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    first = steps[0] if steps else {}
    last = steps[-1] if steps else {}
    first_pose = first.get("pose_x_y_yaw") if isinstance(first, dict) else None
    last_pose = last.get("pose_x_y_yaw") if isinstance(last, dict) else None
    displacement = None
    if (
        isinstance(first_pose, list)
        and isinstance(last_pose, list)
        and len(first_pose) == 3
        and len(last_pose) == 3
        and all(finite_number(v) for v in first_pose + last_pose)
    ):
        displacement = math.hypot(float(last_pose[0]) - float(first_pose[0]), float(last_pose[1]) - float(first_pose[1]))
    published_total = sum(int(s.get("published_count") or 0) for s in steps if isinstance(s, dict))
    last_dwa = last.get("dwa") if isinstance(last.get("dwa"), dict) else {}
    last_centerline = last.get("corridor_center_target") if isinstance(last.get("corridor_center_target"), dict) else {}
    last_tracking = last.get("centerline_tracking_correction") if isinstance(last.get("centerline_tracking_correction"), dict) else {}
    last_imu_hold = last.get("imu_heading_hold") if isinstance(last.get("imu_heading_hold"), dict) else {}
    return {
        "runner_final_decision": data.get("final_decision"),
        "timeout_clock": data.get("timeout_clock"),
        "timeout_trigger": data.get("timeout_trigger"),
        "run_wall_start_sec": data.get("run_wall_start_sec"),
        "run_sim_start_sec": data.get("run_sim_start_sec"),
        "run_wall_elapsed_sec": data.get("run_wall_elapsed_sec"),
        "run_sim_elapsed_sec": data.get("run_sim_elapsed_sec"),
        "real_time_factor_estimate": data.get("real_time_factor_estimate"),
        "use_sim_time": data.get("use_sim_time"),
        "sim_time_valid": data.get("sim_time_valid"),
        "max_runtime_sec_sim": data.get("max_runtime_sec_sim"),
        "wall_watchdog_sec": data.get("wall_watchdog_sec"),
        "target_source": data.get("target_source"),
        "target_subgoal_source": data.get("target_subgoal_source"),
        "step_count": len(steps),
        "published_count_total": published_total,
        "observed_straight_line_displacement_m": displacement,
        "first_pose_x_y_yaw": first_pose,
        "last_pose_x_y_yaw": last_pose,
        "last_cmd_linear_x": last.get("cmd_linear_x") if isinstance(last, dict) else None,
        "last_cmd_angular_z": last.get("cmd_angular_z") if isinstance(last, dict) else None,
        "last_target_base_xy": last.get("target_base_xy") if isinstance(last, dict) else None,
        "last_waypoint_base_xy": last.get("waypoint_base_xy") if isinstance(last, dict) else None,
        "last_original_target_base_xy": last.get("original_target_base_xy") if isinstance(last, dict) else None,
        "last_path_cell_count": last.get("path_cell_count") if isinstance(last, dict) else None,
        "last_dwa": last_dwa,
        "last_corridor_center_target": last_centerline,
        "last_centerline_tracking_correction": last_tracking,
        "last_imu_heading_hold": last_imu_hold,
        "status_payload": data.get("status_payload") if isinstance(data.get("status_payload"), dict) else {},
        "observation_orientation": (
            data.get("observation_orientation") if isinstance(data.get("observation_orientation"), dict) else {}
        ),
    }


def trace_item_anchor_progress(item: Dict[str, Any]) -> Optional[float]:
    metric_sources = (
        item.get("corridor_anchor_metrics_after"),
        item.get("corridor_anchor_metrics_before"),
        item.get("entry_anchor_metrics_after"),
        item.get("target_diagnostics_after_runner"),
        item.get("target_diagnostics_before_runner"),
    )
    for metrics in metric_sources:
        if isinstance(metrics, dict) and finite_number(metrics.get("anchor_progress_m")):
            return float(metrics["anchor_progress_m"])
    runner = item.get("runner") if isinstance(item.get("runner"), dict) else {}
    tracking = runner.get("last_centerline_tracking_correction")
    tracking = tracking if isinstance(tracking, dict) else {}
    metrics = tracking.get("anchor_line_metrics") if isinstance(tracking.get("anchor_line_metrics"), dict) else {}
    if finite_number(metrics.get("anchor_progress_m")):
        return float(metrics["anchor_progress_m"])
    return None


def build_corridor_motion_audit(trace: Sequence[Dict[str, Any]], state_machine_final_decision: str) -> Dict[str, Any]:
    runner_data = read_json(RUNNER_SUMMARY_PATH)
    steps = runner_data.get("steps") if isinstance(runner_data.get("steps"), list) else []
    latest_trace_item = next(
        (item for item in reversed(trace) if isinstance(item.get("runner"), dict)),
        {},
    )
    diagnostics_before = latest_trace_item.get("target_diagnostics_before_runner")
    diagnostics_before = diagnostics_before if isinstance(diagnostics_before, dict) else {}
    diagnostics_after = latest_trace_item.get("target_diagnostics_after_runner")
    diagnostics_after = diagnostics_after if isinstance(diagnostics_after, dict) else {}
    lateral_samples: List[float] = []
    for diagnostics in (diagnostics_before, diagnostics_after):
        if finite_number(diagnostics.get("anchor_lateral_error_m")):
            lateral_samples.append(float(diagnostics["anchor_lateral_error_m"]))

    centerline_tracking_active_count = 0
    corridor_center_target_applied_count = 0
    center_target_failed_reasons: Dict[str, int] = {}
    unknown_grid_ratio_samples: List[Dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tracking = step.get("centerline_tracking_correction")
        tracking = tracking if isinstance(tracking, dict) else {}
        if tracking.get("active"):
            centerline_tracking_active_count += 1
        anchor_metrics = tracking.get("anchor_line_metrics") if isinstance(tracking.get("anchor_line_metrics"), dict) else {}
        if finite_number(anchor_metrics.get("anchor_lateral_error_m")):
            lateral_samples.append(float(anchor_metrics["anchor_lateral_error_m"]))

        center_target = step.get("corridor_center_target")
        center_target = center_target if isinstance(center_target, dict) else {}
        if center_target.get("applied"):
            corridor_center_target_applied_count += 1
        elif center_target.get("reason"):
            reason = str(center_target["reason"])
            center_target_failed_reasons[reason] = center_target_failed_reasons.get(reason, 0) + 1

        path_diagnostic = step.get("path_diagnostic") if isinstance(step.get("path_diagnostic"), dict) else {}
        counts = path_diagnostic.get("grid_value_counts") if isinstance(path_diagnostic.get("grid_value_counts"), dict) else {}
        total_cells = sum(int(counts.get(key) or 0) for key in ("free", "occupied", "unknown", "other"))
        if total_cells > 0:
            unknown_grid_ratio_samples.append(
                {
                    "step": step.get("step"),
                    "unknown_ratio": float(counts.get("unknown") or 0) / float(total_cells),
                    "unknown_count": int(counts.get("unknown") or 0),
                    "total_cell_count": total_cells,
                }
            )

    first_step = steps[0] if steps and isinstance(steps[0], dict) else {}
    last_step = steps[-1] if steps and isinstance(steps[-1], dict) else {}
    start_lateral = diagnostics_before.get("anchor_lateral_error_m")
    final_lateral = diagnostics_after.get("anchor_lateral_error_m")
    if not finite_number(start_lateral) and lateral_samples:
        start_lateral = lateral_samples[0]
    if not finite_number(final_lateral) and lateral_samples:
        final_lateral = lateral_samples[-1]
    runner_final_decision = runner_data.get("final_decision")
    return {
        "diagnostic_only": True,
        "source": "existing_runner_summary_and_state_trace",
        "target_source": runner_data.get("target_source"),
        "target_xy_team_livox_odom": runner_data.get("target_xy_team_livox_odom"),
        "final_decision": runner_final_decision,
        "state_machine_final_decision": state_machine_final_decision,
        "start_pose_x_y_yaw": first_step.get("pose_x_y_yaw"),
        "final_pose_x_y_yaw": last_step.get("pose_x_y_yaw"),
        "start_anchor_lateral_error_m": start_lateral if finite_number(start_lateral) else None,
        "final_anchor_lateral_error_m": final_lateral if finite_number(final_lateral) else None,
        "max_abs_anchor_lateral_error_m": max((abs(value) for value in lateral_samples), default=None),
        "centerline_tracking_active_count": centerline_tracking_active_count,
        "corridor_center_target_applied_count": corridor_center_target_applied_count,
        "corridor_center_target_failed_reasons": center_target_failed_reasons,
        "unknown_grid_ratio_samples": unknown_grid_ratio_samples,
        "runner_timeout": runner_final_decision
        in {
            "BLOCK_ASTAR_DWA_TIMEOUT",
            "BLOCK_ASTAR_DWA_SIM_TIMEOUT",
            "BLOCK_ASTAR_DWA_WALL_WATCHDOG_TIMEOUT",
        },
    }


def command_option(command: Any, option: str) -> Optional[str]:
    if not isinstance(command, list):
        return None
    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def optional_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def build_runner_timeout_audit(trace: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    runner_data = read_json(RUNNER_SUMMARY_PATH)
    steps = [step for step in runner_data.get("steps", []) if isinstance(step, dict)]
    latest_trace_item = next(
        (item for item in reversed(trace) if isinstance(item.get("runner"), dict)),
        {},
    )
    command_result = latest_trace_item.get("command") if isinstance(latest_trace_item.get("command"), dict) else {}
    command = command_result.get("cmd")
    target_xy = runner_data.get("target_xy_team_livox_odom")
    target_valid = isinstance(target_xy, list) and len(target_xy) == 2 and all(finite_number(value) for value in target_xy)
    first_step = steps[0] if steps else {}
    last_step = steps[-1] if steps else {}
    start_pose = first_step.get("pose_x_y_yaw") if isinstance(first_step.get("pose_x_y_yaw"), list) else None
    final_pose = last_step.get("pose_x_y_yaw") if isinstance(last_step.get("pose_x_y_yaw"), list) else None

    def target_distance(pose: Any) -> Optional[float]:
        if not (
            target_valid
            and isinstance(pose, list)
            and len(pose) >= 2
            and all(finite_number(value) for value in pose[:2])
        ):
            return None
        return math.hypot(float(target_xy[0]) - float(pose[0]), float(target_xy[1]) - float(pose[1]))

    target_distance_start = target_distance(start_pose)
    target_distance_final = target_distance(final_pose)
    distance_closed = (
        float(target_distance_start) - float(target_distance_final)
        if finite_number(target_distance_start) and finite_number(target_distance_final)
        else None
    )
    progress_ratio = (
        float(distance_closed) / float(target_distance_start)
        if finite_number(distance_closed) and finite_number(target_distance_start) and float(target_distance_start) > 1e-6
        else None
    )
    goal_tolerance = command_option(command, "--goal-tolerance-m")
    reach_tolerance = optional_float(goal_tolerance)
    max_runtime = command_option(command, "--max-runtime-sec")
    wall_watchdog = command_option(command, "--wall-watchdog-sec")
    max_steps = command_option(command, "--max-steps")
    command_slice = command_option(command, "--command-slice-sec")
    wall_duration = runner_data.get("run_wall_elapsed_sec")
    if not finite_number(wall_duration):
        wall_duration = command_result.get("wall_duration_sec")
    cmd_linear_values = [float(step["cmd_linear_x"]) for step in steps if finite_number(step.get("cmd_linear_x"))]
    avg_cmd_linear = sum(cmd_linear_values) / len(cmd_linear_values) if cmd_linear_values else None
    max_cmd_linear = max(cmd_linear_values) if cmd_linear_values else None
    avg_actual_speed = (
        float(distance_closed) / float(wall_duration)
        if finite_number(distance_closed) and finite_number(wall_duration) and float(wall_duration) > 1e-6
        else None
    )
    stuck_count = sum(
        1
        for step in steps
        if isinstance(step.get("dwa"), dict) and step["dwa"].get("recovery") == "in_place_turn"
    )
    min_obstacle_distance = None
    collision_count = None
    runner_final_decision = runner_data.get("final_decision")
    if (
        finite_number(distance_closed)
        and float(distance_closed) > 0.5
        and finite_number(target_distance_final)
        and finite_number(reach_tolerance)
        and float(target_distance_final) > float(reach_tolerance)
    ):
        timeout_reason_guess = "moving_toward_target_but_timeout_before_reach"
    elif finite_number(distance_closed) and float(distance_closed) < 0.2:
        timeout_reason_guess = "little_progress"
    elif finite_number(min_obstacle_distance) and float(min_obstacle_distance) < 0.3:
        timeout_reason_guess = "obstacle_or_wall_constraint"
    else:
        timeout_reason_guess = "unknown"
    return {
        "diagnostic_only": True,
        "source": "existing_runner_summary_and_state_trace",
        "target_source": runner_data.get("target_source"),
        "target_xy_team_livox_odom": target_xy,
        "start_pose_x_y_yaw": start_pose,
        "final_pose_x_y_yaw": final_pose,
        "target_distance_start_m": target_distance_start,
        "target_distance_final_m": target_distance_final,
        "distance_closed_m": distance_closed,
        "progress_ratio": progress_ratio,
        "runner_final_decision": runner_final_decision,
        "runner_stage": latest_trace_item.get("state"),
        "runner_steps": len(steps),
        "sim_duration_sec": runner_data.get("run_sim_elapsed_sec"),
        "sim_duration_available": finite_number(runner_data.get("run_sim_elapsed_sec")),
        "wall_duration_sec": wall_duration if finite_number(wall_duration) else None,
        "reach_tolerance_m": reach_tolerance,
        "timeout_limit": {
            "max_runtime_sec": optional_float(max_runtime),
            "wall_watchdog_sec": optional_float(wall_watchdog),
            "max_steps": int(max_steps) if max_steps is not None and max_steps.isdigit() else None,
            "command_slice_sec": optional_float(command_slice),
        },
        "avg_cmd_linear_x": avg_cmd_linear,
        "max_cmd_linear_x": max_cmd_linear,
        "avg_actual_speed_mps": avg_actual_speed,
        "avg_actual_speed_time_basis": "runner_command_wall_duration_sec" if avg_actual_speed is not None else None,
        "min_obstacle_distance_m": min_obstacle_distance,
        "min_obstacle_distance_available": False,
        "stuck_count": stuck_count,
        "collision_count": collision_count,
        "collision_count_available": False,
        "timeout_reason_guess": timeout_reason_guess,
    }


def read_follower_status(args: argparse.Namespace, timeout_sec: float = 0.15) -> Dict[str, Any]:
    if not args.use_imu_velocity_follower:
        return {"enabled": False}
    try:
        msg = rospy.wait_for_message(args.follower_status_topic, String, timeout=timeout_sec)
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"raw": msg.data}
        if isinstance(payload, dict):
            payload["enabled"] = True
            return payload
        return {"enabled": True, "payload": payload}
    except Exception as exc:
        return {"enabled": True, "available": False, "error": str(exc)}


def run_runner(
    args: argparse.Namespace,
    state: str,
    runtime_sec: float,
    max_steps: int,
    *,
    local_control_mode: str = "TRANSIT",
    observation_orientation_aim_yaw_odom_rad: Optional[float] = None,
) -> Dict[str, Any]:
    validation_active = room_local_guarded_online_validation_active(args, local_control_mode)
    invocation_id = ""
    if validation_active:
        next_id = int(getattr(args, "_room_local_validation_invocation_counter", 0)) + 1
        setattr(args, "_room_local_validation_invocation_counter", next_id)
        invocation_id = "room_local_%04d" % next_id
    cmd = runner_cmd(
        args,
        state=state,
        runtime_sec=runtime_sec,
        max_steps=max_steps,
        local_control_mode=local_control_mode,
        validation_invocation_id=invocation_id,
        observation_orientation_aim_yaw_odom_rad=observation_orientation_aim_yaw_odom_rad,
    )
    subprocess_timeout_sec = max(
        runtime_sec + args.command_timeout_margin_sec,
        args.runner_wall_watchdog_sec + 30.0,
    )
    result = run_command(cmd, subprocess_timeout_sec)
    runner = summarize_runner()
    return {
        "state": state, "local_control_mode": local_control_mode,
        "validation_run_id": os.environ.get("STATE_MACHINE_RUN_ID", "") if validation_active else None,
        "validation_runner_invocation_id": invocation_id or None,
        "command": result, "runner": runner,
    }


def runner_is_stuck(args: argparse.Namespace, runner: Dict[str, Any], *, state: str = "") -> bool:
    final_decision = str(runner.get("runner_final_decision") or "")
    displacement = runner.get("observed_straight_line_displacement_m")
    published = int(runner.get("published_count_total") or 0)
    if final_decision in {"BLOCK_ASTAR_DWA_BLOCKED_NO_PATH", "BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD", "BLOCK_ASTAR_DWA_BLOCKED_UNSTABLE_PATH"}:
        return True
    min_displacement = args.entry_stuck_min_displacement_m if state == "ENTER_BUILDING" else args.stuck_min_displacement_m
    if published > 0 and finite_number(displacement) and float(displacement) < min_displacement:
        return True
    return False


class RoomSearchSensorSmoke:
    """Minimal passive RGB-D contract capture; it never changes navigation."""

    def __init__(self) -> None:
        self.messages: Dict[str, List[Any]] = {name: [] for name in ("rgb", "depth", "rgb_info", "depth_info", "points")}
        self.subscribers = [
            rospy.Subscriber("/real_sense/rgb/image_raw", Image, lambda msg: self._append("rgb", msg), queue_size=8),
            rospy.Subscriber("/real_sense/depth/image_raw", Image, lambda msg: self._append("depth", msg), queue_size=8),
            rospy.Subscriber("/real_sense/rgb/camera_info", CameraInfo, lambda msg: self._append("rgb_info", msg), queue_size=8),
            rospy.Subscriber("/real_sense/depth/camera_info", CameraInfo, lambda msg: self._append("depth_info", msg), queue_size=8),
            rospy.Subscriber("/real_sense/depth/points", PointCloud2, lambda msg: self._append("points", msg), queue_size=8),
        ]

    def _append(self, key: str, msg: Any) -> None:
        samples = self.messages[key]
        samples.append(msg)
        if len(samples) > 6:
            del samples[0]

    def close(self) -> None:
        for subscriber in self.subscribers:
            subscriber.unregister()

    @staticmethod
    def _rate(samples: Sequence[Any]) -> Optional[float]:
        if len(samples) < 2:
            return None
        stamps = [float(item.header.stamp.to_sec()) for item in samples]
        elapsed = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / elapsed if elapsed > 0.0 else None

    @staticmethod
    def _image_payload(samples: Sequence[Image]) -> Dict[str, Any]:
        if not samples:
            return {"live": False, "encoding": "", "frame": "", "width": None, "height": None, "approx_rate_hz": None}
        msg = samples[-1]
        return {
            "live": True,
            "encoding": str(msg.encoding),
            "frame": str(msg.header.frame_id),
            "width": int(msg.width),
            "height": int(msg.height),
            "approx_rate_hz": RoomSearchSensorSmoke._rate(samples),
            "latest_stamp_sec": float(msg.header.stamp.to_sec()),
        }

    def capture(self, wait_sec: float) -> Dict[str, Any]:
        deadline = time.monotonic() + max(0.0, float(wait_sec))
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            rospy.sleep(0.1)
        rgb = self._image_payload(self.messages["rgb"])
        depth = self._image_payload(self.messages["depth"])
        rgb_info = self.messages["rgb_info"][-1] if self.messages["rgb_info"] else None
        depth_info = self.messages["depth_info"][-1] if self.messages["depth_info"] else None
        points = self.messages["points"][-1] if self.messages["points"] else None
        return {
            "rgb_topic_live": bool(rgb["live"]),
            "rgb_encoding": rgb["encoding"],
            "rgb_frame": rgb["frame"],
            "rgb_width": rgb["width"],
            "rgb_height": rgb["height"],
            "rgb_approx_rate_hz": rgb["approx_rate_hz"],
            "depth_topic_live": bool(depth["live"]),
            "depth_encoding": depth["encoding"],
            "depth_frame": depth["frame"],
            "depth_width": depth["width"],
            "depth_height": depth["height"],
            "depth_approx_rate_hz": depth["approx_rate_hz"],
            "depth_units_inferred": "millimeters_if_16UC1_or_meters_if_32FC1_else_UNRESOLVED",
            "depth_invalid_value_form": "UNRESOLVED: no detector may assume invalid-value semantics from metadata alone",
            "rgb_camera_info": None if rgb_info is None else {"frame": str(rgb_info.header.frame_id), "K": [float(value) for value in rgb_info.K]},
            "depth_camera_info": None if depth_info is None else {"frame": str(depth_info.header.frame_id), "K": [float(value) for value in depth_info.K]},
            "point_cloud_live": points is not None,
            "point_cloud_frame": "" if points is None else str(points.header.frame_id),
            "rgb_depth_pixel_aligned": "UNRESOLVED",
        }


def parse_confirmed_danger_tracks(payload: object) -> List[Dict[str, Any]]:
    """Validate the sidecar's public snapshot; malformed input is fail-soft."""
    try:
        document = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, ValueError):
        return []
    if not isinstance(document, dict):
        return []
    if document.get("schema_version") != 1 or document.get("frame_id") != "team_livox_odom":
        return []
    accepted: List[Dict[str, Any]] = []
    for item in document.get("tracks", []):
        if not isinstance(item, dict) or item.get("state") != "CONFIRMED":
            continue
        track_id, position = item.get("track_id"), item.get("position_xyz_m")
        if not isinstance(track_id, str) or not isinstance(position, (list, tuple)) or len(position) != 3:
            continue
        try:
            xyz = [float(value) for value in position]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in xyz):
            continue
        stamp_value = item.get("last_observed_stamp_sec")
        try:
            stamp_value = float(stamp_value) if stamp_value is not None else None
        except (TypeError, ValueError):
            stamp_value = None
        if stamp_value is not None and not math.isfinite(stamp_value):
            stamp_value = None
        accepted.append({
            "track_id": track_id,
            "position_xyz_m": xyz,
            "last_observed_stamp_sec": stamp_value,
        })
    return accepted


def parse_tentative_danger_hypotheses(payload: object, now_sec: Optional[float] = None) -> List[Dict[str, Any]]:
    """Validate optional pre-confirmation evidence without granting motion authority."""
    try:
        document = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, ValueError):
        return []
    if not isinstance(document, dict) or document.get("schema_version") != 1 or document.get("frame_id") != "team_livox_odom":
        return []
    accepted: List[Dict[str, Any]] = []
    for item in document.get("hypotheses", []):
        if not isinstance(item, dict) or item.get("state") != "TENTATIVE":
            continue
        identity, position = item.get("hypothesis_id"), item.get("position_xyz_m")
        try:
            xyz = [float(value) for value in position]
            support = int(item.get("support_count"))
            observed = float(item.get("last_observed_stamp_sec"))
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            continue
        if not isinstance(identity, str) or support < 1 or not all(math.isfinite(value) for value in xyz + [observed, confidence]):
            continue
        if now_sec is not None and math.isfinite(float(now_sec)) and float(now_sec) - observed > ROOM_SEARCH_TENTATIVE_HYPOTHESIS_TTL_SEC:
            continue
        accepted.append({
            "hypothesis_id": identity,
            "state": "TENTATIVE",
            "position_xyz_m": xyz,
            "confidence": confidence,
            "support_count": support,
            "last_observed_stamp_sec": observed,
        })
    return accepted


class DangerTrackSnapshotCache:
    """Optional, asynchronous awareness only; it has no motion authority."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tracks: List[Dict[str, Any]] = []
        self._subscriber = None
        # This is awareness only.  A missing topic, a late ROS teardown, or a
        # subscriber construction failure must be indistinguishable from an
        # empty snapshot to the navigation path.
        try:
            self._subscriber = rospy.Subscriber(DANGER_TRACKS_TOPIC, String, self._callback, queue_size=2)
        except Exception:
            self._subscriber = None

    def _callback(self, message: String) -> None:
        tracks = parse_confirmed_danger_tracks(getattr(message, "data", ""))
        with self._lock:
            self._tracks = tracks

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(track) for track in self._tracks]

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.unregister()


class DangerHypothesisSnapshotCache:
    """Fail-soft cache for active, unconfirmed RGB-D hypotheses only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hypotheses: List[Dict[str, Any]] = []
        self._subscriber = None
        try:
            self._subscriber = rospy.Subscriber(DANGER_HYPOTHESES_TOPIC, String, self._callback, queue_size=2)
        except Exception:
            self._subscriber = None

    def _callback(self, message: String) -> None:
        hypotheses = parse_tentative_danger_hypotheses(getattr(message, "data", ""))
        with self._lock:
            self._hypotheses = hypotheses

    def snapshot(self, now_sec: Optional[float] = None) -> List[Dict[str, Any]]:
        with self._lock:
            payload = {"schema_version": 1, "frame_id": "team_livox_odom", "hypotheses": list(self._hypotheses)}
        return parse_tentative_danger_hypotheses(payload, now_sec)

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.unregister()


class RoomSearchCameraInfoCache:
    """Read-only CameraInfo cache; absence deliberately has a finite FOV."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hfov_rad: Optional[float] = None
        self._subscriber = None
        try:
            self._subscriber = rospy.Subscriber(ROOM_SEARCH_RGB_CAMERA_INFO_TOPIC, CameraInfo, self._callback, queue_size=2)
        except Exception:
            self._subscriber = None

    def _callback(self, message: CameraInfo) -> None:
        try:
            width, fx = float(message.width), float(message.K[0])
            hfov = 2.0 * math.atan(width / (2.0 * fx))
        except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
            return
        if math.isfinite(hfov) and 0.0 < hfov < math.pi:
            with self._lock:
                self._hfov_rad = hfov

    def effective_hfov(self) -> Tuple[float, str]:
        with self._lock:
            hfov = self._hfov_rad
        return (hfov, "CAMERA_INFO") if hfov is not None else (ROOM_SEARCH_CAMERA_HFOV_FALLBACK_RAD, "FALLBACK_60_DEG")

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.unregister()


def room_search_v2_planning_context(
    args: argparse.Namespace,
    grid_status_pairs: FormalGridStatusPairSubscriber,
    *,
    pair_override: Optional[Tuple[OccupancyGrid, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Read the current production Grid and expose only its existing free mask.

    This is not a second map: the same ``inflate_obstacles`` implementation and
    ROOM_SEARCH safety margin used by the runner are reused before generating a
    small set of candidate points.  Final admission is still the unchanged
    runner preflight below.
    """
    pair = pair_override if pair_override is not None else grid_status_pairs.wait_for_matching_pair(
        args.input_timeout_sec,
        wall_watchdog_sec=args.runner_wall_watchdog_sec,
    )
    context: Dict[str, Any] = {
        "matched_pair_found": bool(pair is not None),
        "grid_content_stamp": None,
        "content_generation_id": None,
        "grid_content_hash": None,
        "qualification_errors": [],
        "candidate_generation_reached": False,
    }
    if pair is None:
        context.update({
            "qualified": False,
            "failure_reason": "ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE",
        })
        return context
    grid_msg, status = pair
    qualified, errors = qualified_for_navigation(grid_msg, status)
    context.update({
        "qualified": bool(qualified),
        "qualification_errors": list(errors),
        "local_traversability_status": status.get("local_traversability_status"),
        "grid_header_stamp_sec": float(grid_msg.header.stamp.to_sec()),
        "grid_content_stamp": status.get("grid_content_stamp"),
        "content_generation_id": status.get("content_generation_id"),
        "grid_content_hash": status.get("grid_content_hash"),
        "grid_msg": grid_msg,
        # Audit-only consumers may freeze the already-paired status verbatim.
        # Production planning continues to use the same scalar fields above.
        "status_payload": copy.deepcopy(status),
    })
    if not qualified:
        context["failure_reason"] = "ROOM_SEARCH_GRID_STATUS_CONTEXT_UNQUALIFIED"
        return context
    runner = object.__new__(BlockAStarDwaRunner)
    runner.args = build_block_astar_dwa_arg_parser().parse_args([])
    runner.args.robot_radius_m = float(args.robot_radius_m)
    runner.args.additional_clearance_margin_m = float(args.room_search_extra_clearance_margin_m)
    raw_grid = runner.grid_array(grid_msg)
    resolution = float(grid_msg.info.resolution)
    blocked = runner.inflate_obstacles(raw_grid, resolution)
    free_points: List[Tuple[float, float]] = []
    for y_index in range(blocked.shape[0]):
        for x_index in range(blocked.shape[1]):
            if not bool(blocked[y_index, x_index]):
                point = runner.cell_to_local_xy((x_index, y_index), grid_msg)
                if point is not None:
                    free_points.append((float(point[0]), float(point[1])))
    coarse_visibility: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for point in free_points:
        key = (
            int(math.floor(point[0] / COARSE_OBSERVATION_RESOLUTION_M)),
            int(math.floor(point[1] / COARSE_OBSERVATION_RESOLUTION_M)),
        )
        previous = coarse_visibility.get(key)
        if previous is None or (math.hypot(*point), point[0], point[1]) < (math.hypot(*previous), previous[0], previous[1]):
            coarse_visibility[key] = point
    context.update({
        "runner": runner,
        "blocked": blocked,
        "free_base_points": free_points,
        "coarse_visibility_base_points": [coarse_visibility[key] for key in sorted(coarse_visibility)],
        "fine_free_point_count": len(free_points),
        "coarse_visibility_point_count": len(coarse_visibility),
        "candidate_generation_reached": True,
    })
    return context


def room_search_v2_fresh_planning_context(
    args: argparse.Namespace,
    grid_status_pairs: FormalGridStatusPairSubscriber,
) -> Dict[str, Any]:
    """Acquire a terminal-after exact Grid/status pair, fail closed on timeout."""
    watermark = grid_status_pairs.receipt_watermark()
    pair_record = grid_status_pairs.wait_for_matching_pair_after(
        watermark,
        args.input_timeout_sec,
        wall_watchdog_sec=args.runner_wall_watchdog_sec,
    )
    acquisition = {
        "receipt_watermark_wall_sec": float(watermark),
        "pair_acquisition_result": "POST_TERMINAL_EXACT_PAIR_TIMEOUT",
        "matched_grid_received_wall_sec": None,
        "matched_status_received_wall_sec": None,
        "matched_grid_content_stamp": None,
        "matched_status_grid_content_stamp": None,
    }
    if pair_record is None:
        return {
            "matched_pair_found": False,
            "grid_content_stamp": None,
            "content_generation_id": None,
            "grid_content_hash": None,
            "qualification_errors": [],
            "candidate_generation_reached": False,
            "qualified": False,
            "failure_reason": "ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE",
            "post_arrival_pair_acquisition": acquisition,
        }
    grid_msg = pair_record["grid"]
    status = pair_record["status"]
    acquisition.update({
        "pair_acquisition_result": "POST_TERMINAL_EXACT_PAIR_ACQUIRED",
        "matched_grid_received_wall_sec": pair_record["grid_received_wall_sec"],
        "matched_status_received_wall_sec": pair_record["status_received_wall_sec"],
        "matched_grid_content_stamp": float(grid_msg.header.stamp.to_sec()),
        "matched_status_grid_content_stamp": status.get("grid_content_stamp"),
    })
    context = room_search_v2_planning_context(
        args,
        grid_status_pairs,
        pair_override=(grid_msg, status),
    )
    context["post_arrival_pair_acquisition"] = acquisition
    return context


def room_search_v2_context_record(context: Dict[str, Any], candidate_count: Optional[int] = None) -> Dict[str, Any]:
    """Keep the first ROOM_SEARCH context observable without serializing Grid data."""
    record = {
        key: context.get(key)
        for key in (
            "matched_pair_found", "grid_header_stamp_sec", "grid_content_stamp", "content_generation_id",
            "grid_content_hash", "local_traversability_status", "qualification_errors", "candidate_generation_reached",
            "failure_reason", "post_arrival_pair_acquisition",
        )
    }
    if candidate_count is not None:
        record["candidate_count"] = int(candidate_count)
    return record


def room_search_v2_rank_components(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Expose the existing cheap-rank inputs without recomputing a rank."""
    return {
        "action_class": candidate.get("target_priority_class"),
        "danger_reobserve_supported": candidate.get("danger_reobserve_supported"),
        "danger_reobserve_abs_bearing_rad": candidate.get("danger_reobserve_abs_bearing_rad"),
        "occlusion_reveal_cells": candidate.get("occlusion_reveal_cells"),
        "new_observable_cells": candidate.get("new_observable_cells"),
        "generic_trajectory_revisit_preference_applied": candidate.get("generic_trajectory_revisit_preference_applied"),
        "cheap_geometric_distance_m": candidate.get("cheap_geometric_distance_m"),
        "heading_change_rad": candidate.get("heading_change_rad"),
        "door_keepout_soft_factor": candidate.get("door_keepout_soft_factor"),
        "sector": candidate.get("sector"),
    }


def room_search_v2_preflight_audit_record(
    rank: int,
    candidate: Dict[str, Any],
    preflight: Dict[str, Any],
) -> Dict[str, Any]:
    """Serialize only fields already emitted by the existing dry runner.

    The runner does not expose per-target raw/inflated collision cells.  Those
    fields are deliberately marked unavailable rather than approximated here.
    """
    runner = preflight.get("runner") if isinstance(preflight.get("runner"), dict) else {}
    last_dwa = runner.get("last_dwa") if isinstance(runner.get("last_dwa"), dict) else {}
    status = runner.get("status_payload") if isinstance(runner.get("status_payload"), dict) else {}
    astar_path_exists = last_dwa.get("p_through_astar_path_exists")
    safe_count = last_dwa.get("safe_moving_candidate_count")
    sample_count = last_dwa.get("sample_count")
    return {
        "global_rank": int(rank),
        "candidate": RoomSearchV2._candidate_audit_summary(candidate),
        "preflight_result": "PASS" if bool(preflight.get("legal")) else "FAIL",
        "legal": bool(preflight.get("legal")),
        "action_aware_admission": preflight.get("action_aware_admission"),
        "position_satisfied": bool(preflight.get("position_satisfied", False)),
        "position_distance_m": preflight.get("position_distance_m"),
        "runner_final_decision": runner.get("runner_final_decision"),
        "local_traversability_status": status.get("local_traversability_status", "UNAVAILABLE_EXISTING_INTERFACE"),
        "astar_path_exists": bool(astar_path_exists) if isinstance(astar_path_exists, bool) else "UNAVAILABLE_EXISTING_INTERFACE",
        "astar_reason": "UNAVAILABLE_EXISTING_INTERFACE",
        "dwa_safe_moving_candidate_count": int(safe_count) if finite_number(safe_count) else "UNAVAILABLE_EXISTING_INTERFACE",
        "dwa_total_candidate_count": int(sample_count) if finite_number(sample_count) else "UNAVAILABLE_EXISTING_INTERFACE",
        "grid_identity": {
            "grid_header_stamp_sec": status.get("grid_header_stamp_sec", "UNAVAILABLE_EXISTING_INTERFACE"),
            "grid_content_stamp": status.get("grid_content_stamp", "UNAVAILABLE_EXISTING_INTERFACE"),
            "content_generation_id": status.get("content_generation_id", "UNAVAILABLE_EXISTING_INTERFACE"),
            "grid_content_hash": status.get("grid_content_hash", "UNAVAILABLE_EXISTING_INTERFACE"),
        },
        "target_grid_value": "UNAVAILABLE_EXISTING_INTERFACE",
        "target_grid_inflated": "UNAVAILABLE_EXISTING_INTERFACE",
        "target_grid_unknown": "UNAVAILABLE_EXISTING_INTERFACE",
        "target_grid_oob": "UNAVAILABLE_EXISTING_INTERFACE",
        "first_rejected_sample_or_cell": "UNAVAILABLE_EXISTING_INTERFACE",
        "first_rejection_reason": "UNAVAILABLE_EXISTING_INTERFACE",
    }


ROOM_SEARCH_DECISION_TRANSIENT_L3V_UNAVAILABLE = "DECISION_TRANSIENT_L3V_UNAVAILABLE"


def room_search_v2_is_decision_global_l3v_transient(preflight: Dict[str, Any]) -> bool:
    """Identify the only preflight terminal known to precede target evaluation.

    The runner emits this terminal after an exact, qualified Grid/Status pair
    is accepted but before target conversion, A*, or DWA.  It therefore cannot
    be attributed to the candidate currently being compared.
    """
    runner = preflight.get("runner") if isinstance(preflight.get("runner"), dict) else {}
    return runner.get("runner_final_decision") == "BLOCK_ASTAR_DWA_BLOCKED_BY_L3V_STATUS"


def room_search_v2_admit_with_l3v_consistency(
    candidates: Sequence[Dict[str, Any]],
    preflight: Any,
    normal_cap: int,
    audit_observer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the existing ranked admission order without cross-snapshot demotion.

    Candidate-specific failures retain the existing scan/terminal-expansion
    behavior. A global L3V transient ends the entire comparison before any
    lower-ranked candidate can be tested against a newer live generation.
    """
    attempts = 0
    expansion_attempts = 0
    for rank, original in enumerate(candidates, 1):
        candidate = dict(original)
        attempts += 1
        if rank > int(normal_cap):
            expansion_attempts += 1
        result = preflight(candidate)
        if audit_observer is not None:
            audit_observer(rank, candidate, result)
        if room_search_v2_is_decision_global_l3v_transient(result):
            return {
                "candidate": None,
                "preflight_attempt_count": attempts,
                "terminal_expansion_attempt_count": expansion_attempts,
                "decision_global_l3v_invalidation": {
                    "candidate": RoomSearchV2._candidate_audit_summary(candidate),
                    "global_rank": int(rank),
                    "runner_final_decision": "BLOCK_ASTAR_DWA_BLOCKED_BY_L3V_STATUS",
                    "status_payload": (
                        result.get("runner", {}).get("status_payload", {})
                        if isinstance(result.get("runner"), dict) else {}
                    ),
                    "remaining_lower_ranked_candidates_not_preflighted": max(0, len(candidates) - int(rank)),
                },
            }
        if not bool(result.get("legal")):
            continue
        selected = dict(candidate)
        selected.update(result)
        selected["cheap_rank"] = int(rank)
        return {
            "candidate": selected,
            "preflight_attempt_count": attempts,
            "terminal_expansion_attempt_count": expansion_attempts,
            "decision_global_l3v_invalidation": None,
        }
    return {
        "candidate": None,
        "preflight_attempt_count": attempts,
        "terminal_expansion_attempt_count": expansion_attempts,
        "decision_global_l3v_invalidation": None,
    }


ROOM_SEARCH_TASK_COMPLETION_REASONS = frozenset({
    "NON_POSITIVE_GAIN",
    "DIMINISHING_RETURN",
})
ROOM_SEARCH_NONPRODUCTIVE_PROGRESS_GUARD_LIMIT = 24


def room_search_v2_progress_guard_update(
    previous_nonproductive_cycles: int,
    actual_new_observation_cells: Any,
) -> Dict[str, Any]:
    """Update the bounded ROOM_SEARCH no-substantive-progress guard.

    The historical numerical bound remains 24, but it is no longer a total
    decision count and never proves task completion.  A terminal observation
    is the authoritative progress signal: any positive number of newly
    observed cells resets the consecutive nonproductive-cycle count.
    """
    actual_new = int(actual_new_observation_cells) if finite_number(actual_new_observation_cells) else 0
    actual_new = max(0, actual_new)
    substantive_progress = actual_new > 0
    nonproductive_cycles = 0 if substantive_progress else max(0, int(previous_nonproductive_cycles)) + 1
    return {
        "actual_new_observation_cells": actual_new,
        "substantive_progress": substantive_progress,
        "nonproductive_progress_cycles": nonproductive_cycles,
        "guard_limit": ROOM_SEARCH_NONPRODUCTIVE_PROGRESS_GUARD_LIMIT,
        "finite_abort": nonproductive_cycles >= ROOM_SEARCH_NONPRODUCTIVE_PROGRESS_GUARD_LIMIT,
    }


def room_search_v2_completion_contract(
    local_availability: str,
    completion_evidence_reason: Optional[str],
) -> Dict[str, Any]:
    """Keep local candidate availability distinct from mission completion.

    The accepted reasons are existing finite task-level conditions.  In
    particular, no local admission failure can imply that ROOM_SEARCH was
    completed or justify entering the unchanged ROOM_RETURN implementation.
    """
    reason = str(completion_evidence_reason) if completion_evidence_reason is not None else None
    mission_complete = reason in ROOM_SEARCH_TASK_COMPLETION_REASONS
    finite_abort = reason == "ANTI_INFINITE_GUARD"
    return {
        "local_availability": str(local_availability),
        "completion_predicates": {
            "existing_task_level_reason": reason,
            "is_existing_finite_completion_reason": mission_complete,
            "is_nonproductive_progress_guard_abort": finite_abort,
        },
        "mission_complete": mission_complete,
        "mission_completion_reason": reason if mission_complete else None,
        "search_aborted_incomplete": finite_abort,
        "finite_abort_reason": reason if finite_abort else None,
        "next_control_flow": "ROOM_RETURN" if mission_complete else "ROOM_SEARCH_INCOMPLETE",
    }


def room_search_v2_exit_action_contract(
    completion_contract: Dict[str, Any],
    *,
    control_alive: bool,
    portal_anchor_available: bool,
) -> Dict[str, Any]:
    """Separate a terminal ROOM_SEARCH result from its physical exit action.

    A valid, still-running entered-room episode always requests the existing
    ROOM_RETURN implementation.  This request is deliberately independent of
    whether the search completed successfully.
    """
    mission_complete = bool(completion_contract.get("mission_complete", False))
    search_aborted_incomplete = bool(completion_contract.get("search_aborted_incomplete", False))
    if mission_complete:
        search_outcome = "SEARCH_COMPLETE"
    elif search_aborted_incomplete:
        search_outcome = "SEARCH_ABORTED_INCOMPLETE"
    else:
        search_outcome = "SEARCH_INCOMPLETE_LOCAL_FAILURE"
    room_return_requested = bool(control_alive and portal_anchor_available)
    return {
        "search_outcome": search_outcome,
        "search_complete": mission_complete,
        "mission_complete": mission_complete,
        "search_aborted_incomplete": search_aborted_incomplete,
        "control_alive": bool(control_alive),
        "portal_anchor_available": bool(portal_anchor_available),
        "room_return_requested": room_return_requested,
        "next_control_flow": "ROOM_RETURN" if room_return_requested else "TERMINATE_WITHOUT_RETURN",
    }


def room_return_max_steps_continuation_contract(
    runner_decision: str,
    start_target_error_m: Any,
    terminal_target_error_m: Any,
    goal_tolerance_m: Any,
    terminal_context: Dict[str, Any],
    next_action_preflight: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify the one productive bounded-return case without adding a planner.

    ``BLOCK_ASTAR_DWA_MAX_STEPS`` is an invocation bound, not a proof that the
    return objective is infeasible.  Continuation is deliberately narrow: it
    requires strictly improved *actual* target error, an unreached target, a
    fresh qualified exact Grid/status pair, and the existing one-step preflight
    to find a legal safe next action.  The caller then restarts the ordinary
    ROOM_RETURN selection loop from the actual terminal odom pose.
    """
    result: Dict[str, Any] = {
        "classification": "TERMINAL_RETURN_FAILURE",
        "continuable": False,
        "runner_decision": str(runner_decision),
        "start_target_error_m": float(start_target_error_m) if finite_number(start_target_error_m) else None,
        "terminal_target_error_m": float(terminal_target_error_m) if finite_number(terminal_target_error_m) else None,
        "goal_tolerance_m": float(goal_tolerance_m) if finite_number(goal_tolerance_m) else None,
        "fresh_exact_grid_status_pair": bool(terminal_context.get("matched_pair_found") and terminal_context.get("qualified")),
        "next_action_preflight_legal": bool(next_action_preflight.get("legal")),
        "safe_moving_candidate_count": None,
    }
    preflight_runner = next_action_preflight.get("runner", {})
    preflight_runner = preflight_runner if isinstance(preflight_runner, dict) else {}
    last_dwa = preflight_runner.get("last_dwa", {})
    if isinstance(last_dwa, dict) and finite_number(last_dwa.get("safe_moving_candidate_count")):
        result["safe_moving_candidate_count"] = int(last_dwa["safe_moving_candidate_count"])
    if str(runner_decision) != "BLOCK_ASTAR_DWA_MAX_STEPS":
        result["reason"] = "RUNNER_RESULT_NOT_MAX_STEPS"
        return result
    if result["start_target_error_m"] is None or result["terminal_target_error_m"] is None or result["goal_tolerance_m"] is None:
        result["reason"] = "RETURN_PROGRESS_INPUT_INVALID"
        return result
    if result["terminal_target_error_m"] <= result["goal_tolerance_m"]:
        result["reason"] = "RETURN_TARGET_ALREADY_REACHED"
        return result
    # This is only a floating-point strict-improvement comparison, not a
    # navigation tuning threshold.
    if not result["terminal_target_error_m"] + 1e-9 < result["start_target_error_m"]:
        result["reason"] = "NO_PRODUCTIVE_RETURN_PROGRESS"
        return result
    if not result["fresh_exact_grid_status_pair"]:
        result["reason"] = "FRESH_QUALIFIED_NAVIGATION_INPUT_UNAVAILABLE"
        return result
    if not result["next_action_preflight_legal"]:
        result["reason"] = "NO_LEGAL_SAFE_NEXT_RETURN_ACTION"
        return result
    if result["safe_moving_candidate_count"] is None or result["safe_moving_candidate_count"] <= 0:
        result["reason"] = "NO_SAFE_MOVING_NEXT_RETURN_ACTION"
        return result
    result.update({
        "classification": "CONTINUE_ROOM_RETURN_FROM_ACTUAL_POSE",
        "continuable": True,
        "reason": "PRODUCTIVE_MAX_STEPS_WITH_FRESH_SAFE_NEXT_ACTION",
    })
    return result


def room_search_v2_postarrival_control(viability_result: str) -> Dict[str, Any]:
    """Return the bounded post-arrival control treatment without adding recovery.

    A reached target is usable only after the current local DWA contract has at
    least one safe moving sample.  The non-viable outcomes deliberately stop
    before ordinary completion, reposition, or ROOM_RETURN semantics.
    """
    if viability_result == "POST_ARRIVAL_VIABLE":
        return {
            "normal_continuation": True,
            "safe_history_accept_terminal_pose": True,
            "safe_stop_required": False,
            "search_completed": None,
            "enter_room_return": False,
            "call_reposition": False,
            "preserve_observation": True,
        }
    if viability_result in {"CONSTRAINED_ARRIVAL", "POST_ARRIVAL_VIABILITY_INPUT_UNAVAILABLE"}:
        return {
            "normal_continuation": False,
            "safe_history_accept_terminal_pose": False,
            "safe_stop_required": True,
            "search_completed": False,
            "enter_room_return": False,
            "call_reposition": False,
            "preserve_observation": True,
        }
    raise ValueError(f"room_search_unknown_postarrival_viability:{viability_result}")


def room_search_v2_constrained_arrival_recovery_handoff(
    viability_result: str,
    actual_pose_xy_yaw: Sequence[float],
    door_return_anchor_xy_yaw: Sequence[float],
    goal_tolerance_m: float,
) -> Dict[str, Any]:
    """Classify the one bounded recovery handoff before existing ROOM_RETURN.

    This is deliberately only a handoff decision.  It reuses the existing
    DoorAnchor XY completion tolerance and does not grant a historical pose
    motion authority: the caller must still invoke the existing strategic
    reposition preflight and then the existing fresh viability check.
    """
    result: Dict[str, Any] = {
        "viability_result": str(viability_result),
        "door_anchor_distance_m": None,
        "within_existing_door_anchor_tolerance": False,
        "action": "PRESERVE_EXISTING_ROOM_RETURN",
    }
    if str(viability_result) != "CONSTRAINED_ARRIVAL":
        result["reason"] = "NOT_A_CONSTRAINED_ARRIVAL"
        return result
    pose_values = list(actual_pose_xy_yaw) if isinstance(actual_pose_xy_yaw, (list, tuple)) else []
    anchor_values = list(door_return_anchor_xy_yaw) if isinstance(door_return_anchor_xy_yaw, (list, tuple)) else []
    if (
        len(pose_values) < 2 or len(anchor_values) < 2
        or not all(finite_number(value) for value in pose_values[:2] + anchor_values[:2])
        or not finite_number(goal_tolerance_m)
    ):
        result.update({
            "action": "FAIL_CLOSED_RECOVERY_INPUT_UNAVAILABLE",
            "reason": "DOOR_ANCHOR_PROXIMITY_INPUT_UNAVAILABLE",
        })
        return result
    distance_m = math.hypot(float(pose_values[0]) - float(anchor_values[0]), float(pose_values[1]) - float(anchor_values[1]))
    within_tolerance = distance_m <= float(goal_tolerance_m)
    result.update({
        "door_anchor_distance_m": distance_m,
        "within_existing_door_anchor_tolerance": within_tolerance,
    })
    if within_tolerance:
        result["reason"] = "WITHIN_EXISTING_DOOR_ANCHOR_GOAL_TOLERANCE"
        return result
    result.update({
        "action": "ATTEMPT_EXISTING_STRATEGIC_REPOSITION_ONCE",
        "reason": "FAR_CONSTRAINED_ARRIVAL",
    })
    return result


def room_search_v2_postarrival_viability(
    args: argparse.Namespace,
    context: Dict[str, Any],
    terminal_odom: Dict[str, Any],
    runner_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate local continuation with the current production DWA only.

    ``grid_msg`` is expressed in the robot's current base frame, so the Grid
    acquired after terminal odom already binds the collision evaluation to the
    actual terminal pose/yaw.  The neutral local vector below is *not* a
    committed target or an A* request: ``choose_dwa`` needs a score reference,
    while its existing ``safe_moving_candidate_count`` is determined before
    scoring from the unchanged dynamic-window, speed, and collision gates.
    """
    pose = pose_tuple(terminal_odom)
    result: Dict[str, Any] = {
        "actual_terminal_odom_stamp_sec": terminal_odom.get("stamp_sec"),
        "actual_terminal_pose_xy_yaw": list(pose),
        "grid_header_stamp_sec": context.get("grid_header_stamp_sec"),
        "grid_content_stamp": context.get("grid_content_stamp"),
        "content_generation_id": context.get("content_generation_id"),
        "grid_content_hash": context.get("grid_content_hash"),
        "grid_status_qualification_result": "QUALIFIED_EXACT_PAIR" if context.get("qualified") else "UNAVAILABLE_OR_UNQUALIFIED",
        "safe_moving_candidate_count": None,
        "previous_command_source": "RUNNER_LAST_DWA",
        "dwa_profile": "ROOM_SEARCH",
        "committed_target_created": False,
    }
    if not context.get("matched_pair_found") or not context.get("qualified"):
        result.update({
            "viability_result": "POST_ARRIVAL_VIABILITY_INPUT_UNAVAILABLE",
            "failure_reason": str(context.get("failure_reason") or "POST_ARRIVAL_VIABILITY_INPUT_UNAVAILABLE"),
        })
        return result

    try:
        runner = context["runner"]
        # Reconstruct exactly the existing ROOM_SEARCH runner profile.  This
        # does not execute it and does not alter the production runner binary.
        profile_cmd = runner_cmd(
            args, state="ROOM_SEARCH", runtime_sec=float(args.runner_runtime_sec), max_steps=1,
        )
        runner.args = build_block_astar_dwa_arg_parser().parse_args(profile_cmd[2:])
        last_dwa = runner_summary.get("last_dwa") if isinstance(runner_summary.get("last_dwa"), dict) else {}
        previous_v = last_dwa.get("selected_linear_x")
        previous_w = last_dwa.get("selected_angular_z")
        runner.prev_cmd = (
            float(previous_v) if finite_number(previous_v) else 0.0,
            float(previous_w) if finite_number(previous_w) else 0.0,
        )
        # ``distance_to_goal=inf`` retains the existing maximum legal local
        # speed cap; there is intentionally no new long-range target or A*.
        neutral_distance = max(float(runner.args.lookahead_m), float(runner.args.min_linear_x), 1e-6)
        _v, _w, dwa = runner.choose_dwa(
            context["grid_msg"], context["blocked"],
            (neutral_distance, 0.0), (neutral_distance, 0.0), float("inf"),
            room_search_safe_moving_eligibility=True,
            target_in_front=True,
            astar_path_exists=True,
        )
        safe_count = int(dwa.get("safe_moving_candidate_count") or 0)
        result.update({
            "safe_moving_candidate_count": safe_count,
            "dynamic_window": dwa.get("dynamic_window"),
            "viability_result": "POST_ARRIVAL_VIABLE" if safe_count > 0 else "CONSTRAINED_ARRIVAL",
        })
        return result
    except Exception as exc:
        result.update({
            "viability_result": "POST_ARRIVAL_VIABILITY_INPUT_UNAVAILABLE",
            "failure_reason": f"POST_ARRIVAL_VIABILITY_EVALUATION_UNAVAILABLE:{type(exc).__name__}",
        })
        return result


def room_search_v2_camera_hfov(sensor_smoke: Dict[str, Any]) -> Optional[float]:
    info = sensor_smoke.get("rgb_camera_info") if isinstance(sensor_smoke, dict) else None
    width = sensor_smoke.get("rgb_width") if isinstance(sensor_smoke, dict) else None
    matrix = info.get("K") if isinstance(info, dict) else None
    if not (finite_number(width) and isinstance(matrix, list) and len(matrix) >= 1 and finite_number(matrix[0]) and float(matrix[0]) > 0.0):
        return None
    return 2.0 * math.atan(float(width) / (2.0 * float(matrix[0])))


def room_search_v2_visible_room_points(
    context: Dict[str, Any], anchor: PortalAnchor, current_pose: Tuple[float, float, float],
    viewpoint_base_xy: Sequence[float], view_heading_base_rad: float, camera_hfov_rad: Optional[float],
) -> List[Tuple[float, float]]:
    """Coarse Grid line-of-sight proxy; blocked fine cells terminate rays."""
    return [
        anchor.local_xy(transform_base_xy(point, current_pose))
        for point in room_search_v2_visible_base_points(context, viewpoint_base_xy, view_heading_base_rad, camera_hfov_rad)
    ]


def room_search_v2_line_clear(
    context: Dict[str, Any], source_base_xy: Sequence[float], target_base_xy_value: Sequence[float],
) -> bool:
    """Read-only LOS query against the exact current formal blocked mask."""
    if not context.get("qualified"):
        return False
    runner, grid_msg, blocked = context["runner"], context["grid_msg"], context["blocked"]
    source = (float(source_base_xy[0]), float(source_base_xy[1]))
    dx, dy = float(target_base_xy_value[0]) - source[0], float(target_base_xy_value[1]) - source[1]
    distance = math.hypot(dx, dy)
    for fraction in np.linspace(0.0, 1.0, max(2, int(math.ceil(distance / 0.05)) + 1)):
        cell = runner.local_xy_to_cell(source[0] + fraction * dx, source[1] + fraction * dy, grid_msg)
        if cell is None or bool(blocked[cell[1], cell[0]]):
            return False
    return True


def room_search_v2_visible_base_points(
    context: Dict[str, Any], viewpoint_base_xy: Sequence[float], view_heading_base_rad: float, camera_hfov_rad: Optional[float],
) -> List[Tuple[float, float]]:
    """Existing formal-Grid visibility computation, expressed in current base."""
    if not context.get("qualified"):
        return []
    source = (float(viewpoint_base_xy[0]), float(viewpoint_base_xy[1]))
    visible: List[Tuple[float, float]] = []
    for point in context.get("coarse_visibility_base_points", []):
        dx, dy = float(point[0]) - source[0], float(point[1]) - source[1]
        distance = math.hypot(dx, dy)
        if not (0.25 <= distance <= 1.5):
            continue
        bearing = normalize_angle(math.atan2(dy, dx) - float(view_heading_base_rad))
        if camera_hfov_rad is not None and abs(bearing) > camera_hfov_rad / 2.0:
            continue
        if room_search_v2_line_clear(context, source, point):
            visible.append((float(point[0]), float(point[1])))
    return visible


def room_search_v2_seen_state_hash(search: RoomSearchV2) -> str:
    """Return a compact deterministic identity for diagnostic-only SEEN evidence."""
    cells = sorted((int(cell[0]), int(cell[1])) for cell in search.observation.seen)
    encoded = json.dumps(cells, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def room_search_v2_terminal_visibility_counterfactual(
    search: RoomSearchV2,
    context: Dict[str, Any],
    anchor: PortalAnchor,
    terminal_pose: Tuple[float, float, float],
    terminal_odom: Dict[str, Any],
    camera_hfov_rad: Optional[float],
    candidate: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Tuple[float, float]]]:
    """Observe the existing terminal visibility calculation before SEEN mutates.

    This helper deliberately calls the existing formal-Grid visibility function
    once and returns its result for the existing terminal update.  It has no
    motion, ranking, or ObservationMemory mutation authority.
    """
    matched_pair = bool(context.get("matched_pair_found"))
    qualified = bool(context.get("qualified"))
    exact_terminal_grid = bool(matched_pair and qualified)
    audit: Dict[str, Any] = {
        "status": "TERMINAL_COUNTERFACTUAL_READY" if exact_terminal_grid else "TERMINAL_COUNTERFACTUAL_UNAVAILABLE",
        "candidate_predicted_new": candidate.get("new_observable_cells"),
        "candidate_predicted_occlusion_reveal": candidate.get("occlusion_reveal_cells"),
        "actual_terminal_odom_stamp_sec": terminal_odom.get("stamp_sec"),
        "actual_terminal_pose_xy_yaw": list(terminal_pose),
        "terminal_body_heading_odom_rad": float(terminal_pose[2]),
        "terminal_view_heading_base_rad": 0.0,
        "seen_count_before_terminal_update": len(search.observation.seen),
        "seen_hash_before_terminal_update": room_search_v2_seen_state_hash(search),
        "grid_identity": {
            "grid_header_stamp_sec": context.get("grid_header_stamp_sec"),
            "grid_content_stamp": context.get("grid_content_stamp"),
            "content_generation_id": context.get("content_generation_id"),
            "grid_content_hash": context.get("grid_content_hash"),
            "local_traversability_status": context.get("local_traversability_status"),
            "navigation_qualification": "QUALIFIED_EXACT_PAIR" if exact_terminal_grid else "UNAVAILABLE_OR_UNQUALIFIED",
            "qualification_errors": list(context.get("qualification_errors") or []),
            "post_arrival_pair_acquisition": copy.deepcopy(context.get("post_arrival_pair_acquisition")),
        },
        "terminal_counterfactual_visible_cells": None,
        "terminal_counterfactual_new_observable_cells": None,
        "actual_terminal_visible_cells": None,
        "actual_new_observation_cells": None,
        "original_prediction_error": None,
        "terminal_proxy_error": None,
        "viewpoint_explained_component": None,
    }
    if not exact_terminal_grid:
        audit["unavailable_reason"] = str(
            context.get("failure_reason") or "TERMINAL_GRID_STATUS_PAIR_UNAVAILABLE_OR_UNQUALIFIED"
        )
        return audit, []

    terminal_visible = room_search_v2_visible_room_points(
        context, anchor, terminal_pose, (0.0, 0.0), 0.0, camera_hfov_rad,
    )
    terminal_new = search.observation.new_count(terminal_visible)
    audit.update({
        "terminal_counterfactual_visible_cells": len(terminal_visible),
        "terminal_counterfactual_new_observable_cells": int(terminal_new),
        "actual_terminal_visible_cells": len(terminal_visible),
    })
    return audit, terminal_visible


def room_search_v2_finalize_terminal_visibility_counterfactual(
    audit: Dict[str, Any], actual_new_observation_cells: int,
) -> Dict[str, Any]:
    """Attach derived diagnostic values after the unchanged SEEN update."""
    audit["actual_new_observation_cells"] = int(actual_new_observation_cells)
    candidate_prediction = audit.get("candidate_predicted_new")
    terminal_prediction = audit.get("terminal_counterfactual_new_observable_cells")
    actual = int(actual_new_observation_cells)
    if finite_number(candidate_prediction):
        audit["original_prediction_error"] = int(candidate_prediction) - actual
    if finite_number(terminal_prediction):
        audit["terminal_proxy_error"] = int(terminal_prediction) - actual
    if finite_number(candidate_prediction) and finite_number(terminal_prediction):
        audit["viewpoint_explained_component"] = int(candidate_prediction) - int(terminal_prediction)
    return audit


def room_search_v2_preflight(args: argparse.Namespace, target_xy: Sequence[float], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Use the existing runner without ``--execute`` for legal-path admission."""
    write_absolute_target(target_xy, "ROOM_SEARCH_V2", "nbv_preflight", extra)
    dry_args = copy.copy(args)
    dry_args.execute = False
    # A dry admission needs exactly one fresh planning/collision decision; it
    # cannot gain evidence by repeating while the robot remains stationary.
    attempt = run_runner(
        dry_args,
        "ROOM_SEARCH",
        args.runner_runtime_sec,
        1,
        local_control_mode="ROOM_LOCAL",
    )
    runner = attempt["runner"]
    terminal = str(runner.get("runner_final_decision") or "")
    last_dwa = runner.get("last_dwa") if isinstance(runner.get("last_dwa"), dict) else {}
    legal = terminal not in {
        "BLOCK_ASTAR_DWA_BLOCKED_NO_PATH", "BLOCK_ASTAR_DWA_BLOCKED_DWA_NO_CMD",
        "BLOCK_ASTAR_DWA_BLOCKED_UNSTABLE_PATH", "BLOCK_ASTAR_DWA_BLOCKED_BY_TARGET",
    } and bool(last_dwa.get("p_through_astar_path_exists")) and int(last_dwa.get("safe_moving_candidate_count") or 0) > 0
    path_cells = int(runner.get("last_path_cell_count") or 0)
    return {"legal": legal, "runner": runner, "path_length_m": max(0.05, path_cells * GRID_RESOLUTION_M)}


def room_search_v2_action_aware_preflight(
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    current_pose: Tuple[float, float, float],
    seen_cell_ids: Iterable[Sequence[int]],
) -> Dict[str, Any]:
    """Preserve translation admission except for a valid already-at-action pose."""
    target = candidate.get("target_xy_team_livox_odom")
    if isinstance(target, (list, tuple)) and len(target) == 2 and all(finite_number(value) for value in target):
        distance = math.hypot(float(target[0]) - float(current_pose[0]), float(target[1]) - float(current_pose[1]))
        intent_type, intent_cells = candidate_observation_intent_cell_ids(candidate, seen_cell_ids)
        if distance <= float(args.goal_tolerance_m) and bool(intent_cells):
            return {
                "legal": True,
                "runner": {
                    "runner_final_decision": "POSITION_SATISFIED_NO_TRANSLATION_PREFLIGHT",
                    "last_dwa": {},
                    "status_payload": {},
                },
                "path_length_m": 0.0,
                "action_aware_admission": "POSITION_SATISFIED_OBSERVATION_ACTION",
                "position_satisfied": True,
                "position_distance_m": float(distance),
                "observation_intent_type": intent_type,
                "intended_observation_cell_count": len(intent_cells),
            }
    return room_search_v2_preflight(
        args,
        candidate["target_xy_team_livox_odom"],
        {"room_search_candidate": candidate},
    )


def execute_room_search_v2(args: argparse.Namespace, portal_target: Dict[str, Any], _entry_pose: Tuple[float, float, float]) -> Dict[str, Any]:
    """NBV-lite ROOM_SEARCH followed by return to the saved actual entry pose."""
    entry_odom = read_odom()
    entry_pose = pose_tuple(entry_odom)
    anchor = PortalAnchor.from_frozen_target(portal_target, entry_pose, float(entry_odom["stamp_sec"]))
    search = RoomSearchV2(anchor)
    frozen_decision_capture = FrozenDecisionCapture.from_environment(ROOT)
    stage_b_shadow_capture = RoomSearchStageBShadowCapture.from_environment(ROOT)
    high_level_locomotion_shadow = HighLevelLocomotionCompatibilityShadow.from_environment(ROOT)
    observation_arrival_shadow_enabled = room_search_observation_arrival_shadow_enabled()
    observation_arrival_shadow_context: Dict[str, Any] = {
        "feature_flag_enabled": bool(observation_arrival_shadow_enabled),
        "run_id": os.environ.get("STATE_MACHINE_RUN_ID", ""),
        "navigation_before_hash": os.environ.get("ROOM_SEARCH_C1_C2_NAVIGATION_BEFORE_HASH", ""),
        "source_hashes_before": (
            room_search_observation_arrival_source_hashes() if observation_arrival_shadow_enabled else {}
        ),
        "protected_hashes": {
            "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py": "468e735acc8bb0bdf03453906fccd70b41394b751bc3638e0f62f40b48a01a5b",
            "scripts/local_subgoal_runner_mvp/room_search_v1.py": "485140cbe3ce9cd985a3d27ea9a50cb83782e73293fd80003799df899d0d01a9",
            "scripts/local_subgoal_runner_mvp/room_search_observation_arrival_contract.py": "450417b281fabb56700b481996440d1a8f4061fd0b7f73ef64cb3903bc9f8284",
            "scripts/local_subgoal_runner_mvp/replay_room_search_observation_arrival_c0.py": "058c4384d945281731126c7f31482fe524b38de8883438eabfafe88c9ef67e23",
            "scripts/local_subgoal_runner_mvp/tests/test_room_search_observation_arrival_contract.py": "5e5ff603e80b8282682b234334dd59ee5baad8a50c3867c178ab973dd66af27d",
        },
        "events": [],
    }
    grid_status_pairs = FormalGridStatusPairSubscriber()
    danger_tracks = DangerTrackSnapshotCache()
    danger_hypotheses = DangerHypothesisSnapshotCache()
    camera_info = RoomSearchCameraInfoCache()
    # The old sensor smoke collector remains available as a diagnostic helper,
    # but a passive camera check must not hold ROOM_SEARCH motion for four seconds.
    sensor_smoke = {"critical_path": "DISABLED_NONBLOCKING", "wait_sec": 0.0}
    camera_hfov, camera_hfov_source = camera_info.effective_hfov()
    entry_seen_initialized = False
    entry_seen_cell_count = 0
    latest_planning_context: Dict[str, Any] = {}
    held_hypothesis_ids = set()
    danger_reobserve_episode_guard = DangerReobserveEpisodeGuard(str(os.environ.get("STATE_MACHINE_RUN_ID", "")))
    search.record_safe_actual_pose(entry_pose, 0.0)
    online: Dict[str, Any] = {
        "started": True,
        "run_id": os.environ.get("STATE_MACHINE_RUN_ID", ""),
        "pthrough_success": True,
        "room_search_started": True,
        "waypoints_reached": 0,
        "search_completed": False,
        "search_aborted_incomplete": False,
        "room_return_started": False,
        "breadcrumbs_used": 0,
        "returned_to_door_return_anchor": False,
        "collision_safety_regression": False,
        "danger_reobserve_holds": 0,
    }
    validation_evidence_status = (
        RoomLocalValidationEvidenceStatusCache(str(os.environ.get("STATE_MACHINE_RUN_ID", "")))
        if bool(args.execute)
        and room_local_guarded_online_validation_active(args, "ROOM_LOCAL")
        and bool(os.environ.get("STATE_MACHINE_RUN_ID", ""))
        else None
    )
    result: Dict[str, Any] = {
        "portal_anchor": {
            "frame": "team_livox_odom",
            "center_xy": list(anchor.center_xy),
            "inward_normal": list(anchor.inward_normal),
            "tangent": list(anchor.tangent),
            "width_m": anchor.width_m,
            "door_return_anchor_xy_yaw": list(anchor.door_return_anchor_xy_yaw),
            "door_return_anchor_stamp_sec": anchor.door_return_anchor_stamp_sec,
        },
        "sensor_smoke": sensor_smoke,
        "search_decisions": [],
        "search_attempts": [],
        "mission_actions": [],
        "danger_reobserve_actions": [],
        "return_attempts": [],
        "recoverability_shadow_events": [],
        "danger_reobserve_holds": [],
        "online": online,
    }

    def execute_target(
        target_xy: Sequence[float],
        kind: str,
        extra: Dict[str, Any],
        state: str = "ROOM_SEARCH",
        *,
        local_control_mode: str = "ROOM_LOCAL",
    ) -> Dict[str, Any]:
        evidence_reason = guarded_room_local_validation_evidence_gate_reason(
            args, local_control_mode, validation_evidence_status,
        )
        if evidence_reason is not None:
            raise RoomLocalValidationEvidenceInsufficient(evidence_reason)
        write_absolute_target(target_xy, "ROOM_SEARCH_V2", kind.lower(), extra)
        attempt = run_runner(
            args,
            state,
            args.runner_runtime_sec,
            args.runner_max_steps,
            local_control_mode=local_control_mode,
        )
        validation_abort_reason = room_local_validation_abort_reason(attempt)
        if validation_abort_reason is not None:
            raise RoomLocalValidationAbort(validation_abort_reason)
        return attempt

    def execute_observation_orientation_slice(
        spec: MissionActionSpec,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call the runner's OA-only one-slice entry; never invoke Phase-3 state."""
        evidence_reason = guarded_room_local_validation_evidence_gate_reason(
            args, "ROOM_LOCAL", validation_evidence_status,
        )
        if evidence_reason is not None:
            raise RoomLocalValidationEvidenceInsufficient(evidence_reason)
        if not finite_number(spec.aim_yaw_odom_rad):
            return {
                "state": "ROOM_SEARCH", "local_control_mode": "ROOM_LOCAL",
                "command": {"cmd": [], "reason": "OBSERVATION_AIM_INVALID"},
                "runner": {"runner_final_decision": "OBSERVATION_ORIENTATION_UNAVAILABLE", "observation_orientation": {
                    "action": "ORIENTATION_UNAVAILABLE", "reason": "OBSERVATION_AIM_INVALID",
                }},
            }
        write_absolute_target(
            spec.target_odom_xy,
            "ROOM_SEARCH_V2",
            "observation_orientation",
            {"room_search_candidate": candidate, "mission_action": spec.to_dict()},
        )
        attempt = run_runner(
            args,
            "ROOM_SEARCH",
            args.runner_runtime_sec,
            1,
            local_control_mode="ROOM_LOCAL",
            observation_orientation_aim_yaw_odom_rad=float(spec.aim_yaw_odom_rad),
        )
        validation_abort_reason = room_local_validation_abort_reason(attempt)
        if validation_abort_reason is not None:
            raise RoomLocalValidationAbort(validation_abort_reason)
        return attempt

    def mission_action_terminal_evidence(
        spec: MissionActionSpec,
        candidate: Dict[str, Any],
        *,
        require_fresh_grid_status: bool = False,
    ) -> Dict[str, Any]:
        """Read terminal facts and classify them before factual SEEN mutation."""
        terminal_odom = read_odom()
        terminal_pose = pose_tuple(terminal_odom)
        terminal_context = (
            room_search_v2_fresh_planning_context(args, grid_status_pairs)
            if require_fresh_grid_status else room_search_v2_planning_context(args, grid_status_pairs)
        )
        terminal_visibility_audit, actual_visible = room_search_v2_terminal_visibility_counterfactual(
            search, terminal_context, anchor, terminal_pose, terminal_odom, camera_hfov, candidate,
        )
        terminal_grid_identity = terminal_visibility_audit.get("grid_identity", {})
        terminal_grid_identity = terminal_grid_identity if isinstance(terminal_grid_identity, dict) else {}
        terminal_evidence_complete = bool(
            terminal_visibility_audit.get("status") == "TERMINAL_COUNTERFACTUAL_READY"
            and terminal_grid_identity.get("navigation_qualification") == "QUALIFIED_EXACT_PAIR"
        )
        # ``new_count`` is a factual pre-update calculation.  The sole actual
        # update remains below, after the mission predicate has been recorded.
        actual_new = search.observation.new_count(actual_visible)
        mission_observation = evaluate_mission_action_observation(
            spec,
            terminal_evidence_complete=terminal_evidence_complete,
            actual_visible_cell_ids=canonical_observation_cell_ids(actual_visible),
            actual_new_observation_cells=actual_new,
        )
        return {
            "terminal_odom": terminal_odom,
            "terminal_pose": terminal_pose,
            "terminal_context": terminal_context,
            "terminal_visibility_audit": terminal_visibility_audit,
            "actual_visible": actual_visible,
            "actual_new_pre_update": actual_new,
            "mission_observation": mission_observation,
        }

    def finish(payload: Dict[str, Any]) -> Dict[str, Any]:
        if observation_arrival_shadow_enabled:
            try:
                write_room_search_c1_c2_observation_arrival_outputs(payload, observation_arrival_shadow_context)
            except Exception as exc:
                # An audit write fault must not change the existing mission result.
                payload["observation_arrival_shadow_output_error"] = "%s:%s" % (type(exc).__name__, exc)
        try:
            stage_b_shadow_capture.note_production_finish(
                decision_id=int(decisions),
                final_decision=payload.get("final_decision"),
                search_completed=online.get("search_completed"),
                room_return_started=online.get("room_return_started"),
                returned_to_door_return_anchor=online.get("returned_to_door_return_anchor"),
            )
        except Exception:
            pass
        grid_status_pairs.close()
        danger_tracks.close()
        danger_hypotheses.close()
        camera_info.close()
        if validation_evidence_status is not None:
            validation_evidence_status.close()
        return payload

    def execute_room_return_exit() -> Dict[str, Any]:
        """Run the existing ROOM_RETURN implementation without changing it."""
        write_stage("room_search", "navigation_state_machine.py", {"mode": "ROOM_RETURN"})
        online["room_return_started"] = True
        return_decisions = 0
        door_target = search.door_return_target()
        while not rospy.is_shutdown():
            if return_decisions >= 24:
                result.update({"final_decision": "ROOM_RETURN_FAILED", "first_failure_mechanism": "ANTI_INFINITE_SEARCH_GUARD"})
                return finish(result)
            pose = pose_tuple(read_odom())
            door_xy = door_target["target_xy_team_livox_odom"]
            current_distance_to_door = math.hypot(float(door_xy[0]) - pose[0], float(door_xy[1]) - pose[1])
            if search.at_door_return_anchor(pose, float(args.goal_tolerance_m)):
                result["return_attempts"].append({"return_decision_index": return_decisions + 1, "current_actual_pose": list(pose),
                                                  "door_direct_executable": True, "selected_target_type": "DOOR_DIRECT", "selected_target": door_xy,
                                                  "current_distance_to_door": current_distance_to_door,
                                                  "target_distance_to_door": 0.0, "planner_path_available": True,
                                                  "terminal_runner_result": "NOT_RUN_ALREADY_AT_DOOR_ANCHOR",
                                                  "final_return_completion_reason": "DOOR_RETURN_ANCHOR_REACHED"})
                online["returned_to_door_return_anchor"] = True
                result["final_decision"] = "ROOM_SEARCH_V2_RETURNED_TO_DOOR_ANCHOR"
                return finish(result)

            def executable_option(target: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
                target_xy = target["target_xy_team_livox_odom"]
                target_base = target_base_xy(target_xy, pose)
                local_support = target_base[0] >= 0.15 and math.hypot(*target_base) <= 1.5
                preflight = room_search_v2_preflight(args, target_xy, extra) if local_support else {"legal": False, "path_length_m": None}
                option = dict(target)
                option.update({"legal": bool(preflight.get("legal")), "local_planning_support": local_support,
                               "path_length_m": preflight.get("path_length_m"),
                               "target_distance_to_door": math.hypot(float(door_xy[0]) - float(target_xy[0]), float(door_xy[1]) - float(target_xy[1]))})
                return option

            door_option = executable_option(door_target, {"room_search_return_target": door_target})
            breadcrumb_options = []
            for index, point in enumerate(search.breadcrumbs):
                breadcrumb = {"kind": "BREADCRUMB", "breadcrumb_index": index, "target_xy_team_livox_odom": [point[0], point[1]]}
                breadcrumb_options.append(executable_option(breadcrumb, {"room_search_return_target": breadcrumb}))
            selected = search.choose_return_target(door_option, breadcrumb_options)
            transition = None
            if selected is None:
                context = room_search_v2_planning_context(args, grid_status_pairs)
                candidates = search.candidates_from_planning_free_base(pose, context.get("free_base_points", [])) if context.get("qualified") else []
                for candidate in candidates:
                    candidate["legal"] = bool(room_search_v2_preflight(
                        args, candidate["target_xy_team_livox_odom"], {"room_return_transition_for": door_target}
                    ).get("legal"))
                transition = search.choose_safe_return_transition(pose, door_xy, candidates)
                if transition is None:
                    result["return_attempts"].append({"return_decision_index": return_decisions + 1, "current_actual_pose": list(pose),
                                                      "door_direct_executable": False, "selected_target_type": "SAFE_TRANSITION",
                                                      "current_distance_to_door": current_distance_to_door,
                                                      "selected_target": None, "target_distance_to_door": None,
                                                      "planner_path_available": False,
                                                      "terminal_runner_result": "NOT_RUN_NO_SAFE_LOCAL_RETURN_TRANSITION",
                                                      "final_return_completion_reason": "NO_SAFE_LOCAL_RETURN_TRANSITION",
                                                      "failure": "NO_SAFE_LOCAL_RETURN_TRANSITION"})
                    result.update({"final_decision": "ROOM_RETURN_FAILED", "first_failure_mechanism": "NO_SAFE_LOCAL_RETURN_TRANSITION"})
                    return finish(result)
                transition_xy = transition["target_xy_team_livox_odom"]
                _gain = current_distance_to_door - math.hypot(float(door_xy[0]) - float(transition_xy[0]), float(door_xy[1]) - float(transition_xy[1]))
                selected = {"kind": "SAFE_TRANSITION", "selected_target_type": "SAFE_TRANSITION",
                            "target_xy_team_livox_odom": transition_xy,
                            "target_distance_to_door": current_distance_to_door - _gain}

            selected_target = selected["target_xy_team_livox_odom"]
            start_target_error_m = math.hypot(float(selected_target[0]) - pose[0], float(selected_target[1]) - pose[1])
            skipped_count = (len(search.breadcrumbs) - 1 - int(selected["breadcrumb_index"])) if selected["selected_target_type"] == "BREADCRUMB_FALLBACK" else 0
            attempt = execute_target(
                selected_target,
                "ROOM_RETURN_TRANSITION" if transition else str(selected["kind"]),
                {"room_search_return_target": selected, "room_return_transition": transition},
                "ROOM_SEARCH",
                local_control_mode="ROOM_LOCAL",
            )
            return_decisions += 1
            runner_decision = str(attempt["runner"].get("runner_final_decision") or "")
            record = {"return_decision_index": return_decisions, "current_actual_pose": list(pose),
                      "door_direct_executable": bool(door_option["legal"]), "selected_target_type": selected["selected_target_type"],
                      "selected_breadcrumb_index": selected.get("breadcrumb_index"), "breadcrumbs_skipped_count": skipped_count,
                      "selected_target": list(selected_target), "current_distance_to_door": current_distance_to_door,
                      "target_distance_to_door": selected.get("target_distance_to_door"),
                      "start_target_error_m": start_target_error_m,
                      "planner_path_available": bool(door_option["legal"] if selected["selected_target_type"] == "DOOR_DIRECT" else selected.get("legal", True)),
                      "terminal_runner_result": runner_decision, "transition": transition,
                      "runner": attempt["runner"], "runner_command": attempt["command"]}
            result["return_attempts"].append(record)
            if runner_decision == "BLOCK_ASTAR_DWA_MAX_STEPS":
                terminal_context: Dict[str, Any] = {}
                continuation_preflight: Dict[str, Any] = {"legal": False, "runner": {}}
                terminal_odom: Dict[str, Any] = {}
                terminal_pose: Optional[Tuple[float, float, float]] = None
                terminal_target_error_m: Optional[float] = None
                continuation_input_error: Optional[str] = None
                try:
                    terminal_odom = read_odom()
                    terminal_pose = pose_tuple(terminal_odom)
                    terminal_target_error_m = math.hypot(
                        float(selected_target[0]) - terminal_pose[0],
                        float(selected_target[1]) - terminal_pose[1],
                    )
                    terminal_context = room_search_v2_planning_context(args, grid_status_pairs)
                    if terminal_context.get("matched_pair_found") and terminal_context.get("qualified"):
                        continuation_preflight = room_search_v2_preflight(
                            args, selected_target,
                            {"room_return_continuation_from_actual_pose": list(terminal_pose)},
                        )
                except Exception as exc:
                    continuation_input_error = f"POST_MAX_STEPS_INPUT_UNAVAILABLE:{type(exc).__name__}"
                continuation = room_return_max_steps_continuation_contract(
                    runner_decision, start_target_error_m, terminal_target_error_m,
                    args.goal_tolerance_m, terminal_context, continuation_preflight,
                )
                record.update({
                    "actual_terminal_odom_stamp_sec": terminal_odom.get("stamp_sec"),
                    "actual_terminal_pose_xy_yaw": list(terminal_pose) if terminal_pose is not None else None,
                    "terminal_target_error_m": terminal_target_error_m,
                    "return_continuation_navigation_context": room_search_v2_context_record(terminal_context),
                    "return_continuation_preflight": continuation_preflight,
                    "return_continuation": continuation,
                    "return_continuation_input_error": continuation_input_error,
                })
                if continuation["continuable"]:
                    record["continuation_action"] = "REENTER_EXISTING_ROOM_RETURN_SELECTION_FROM_ACTUAL_POSE"
                    continue
            if runner_decision != "BLOCK_ASTAR_DWA_REACHED_GOAL":
                result["final_decision"] = "ROOM_RETURN_FAILED"
                result["first_failure_mechanism"] = runner_decision or "RUNNER_SUMMARY_MISSING"
                return finish(result)
            if selected["selected_target_type"] == "BREADCRUMB_FALLBACK":
                online["breadcrumbs_used"] = int(online["breadcrumbs_used"]) + 1
        result.update({"final_decision": "ROOM_RETURN_FAILED", "first_failure_mechanism": "ROS_SHUTDOWN"})
        return finish(result)

    def request_room_return_exit(completion_contract: Dict[str, Any]) -> Dict[str, Any]:
        anchor_xy_yaw = getattr(anchor, "door_return_anchor_xy_yaw", ())
        anchor_available = (
            isinstance(anchor_xy_yaw, (list, tuple))
            and len(anchor_xy_yaw) == 3
            and all(finite_number(value) for value in anchor_xy_yaw)
        )
        exit_action = room_search_v2_exit_action_contract(
            completion_contract,
            control_alive=not rospy.is_shutdown(),
            portal_anchor_available=anchor_available,
        )
        result["room_search_outcome"] = exit_action["search_outcome"]
        result["room_search_exit_action"] = exit_action
        if exit_action["room_return_requested"]:
            return execute_room_return_exit()
        return finish(result)

    def persist_diagnostic(
        decision_index: int, *, candidate: Optional[Dict[str, Any]] = None, preflight_attempt_count: int = 0,
        execution_result: Optional[str] = None, actual_displacement_m: Optional[float] = None,
        reposition_state: str = "NOT_REQUESTED",
        post_arrival_viability: Optional[Dict[str, Any]] = None,
        completion_contract: Optional[Dict[str, Any]] = None,
    ) -> None:
        candidate = candidate if isinstance(candidate, dict) else {}
        bearing = candidate.get("heading_change_rad")
        runner = candidate.get("runner") if isinstance(candidate.get("runner"), dict) else {}
        last_dwa = runner.get("last_dwa") if isinstance(runner.get("last_dwa"), dict) else {}
        linear_x, angular_z = last_dwa.get("selected_linear_x"), last_dwa.get("selected_angular_z")
        turn_radius = None
        if finite_number(linear_x) and finite_number(angular_z) and abs(float(angular_z)) > 1e-6:
            turn_radius = abs(float(linear_x) / float(angular_z))
        write_atomic_json(ROOM_SEARCH_DIAGNOSTIC_PATH, {
            "schema_version": 1,
            "search_decision_index": int(decision_index),
            "selected_target": candidate.get("target_xy_team_livox_odom"),
            "target_priority_class": candidate.get("target_priority_class"),
            "occlusion_reveal_cells": candidate.get("occlusion_reveal_cells"),
            "new_observable_cells": candidate.get("new_observable_cells"),
            "cheap_rank": candidate.get("cheap_rank"),
            "preflight_attempt_count": int(preflight_attempt_count),
            "selected_target_bearing_rad": bearing,
            "selected_linear_x": linear_x,
            "selected_angular_z": angular_z,
            "selected_nominal_turn_radius_m": turn_radius,
            "execution_result": execution_result,
            "actual_displacement_m": actual_displacement_m,
            "reposition_state": reposition_state,
            "confirmed_danger_track_count": len(danger_tracks.snapshot()),
            "active_tentative_hypothesis_count": len(danger_hypotheses.snapshot(float(rospy.Time.now().to_sec()))),
            "entry_seen_initialized": entry_seen_initialized,
            "entry_seen_cell_count": entry_seen_cell_count,
            "camera_hfov_rad_used": camera_hfov,
            "camera_hfov_source": camera_hfov_source,
            "fine_free_point_count": latest_planning_context.get("fine_free_point_count"),
            "coarse_visibility_point_count": latest_planning_context.get("coarse_visibility_point_count"),
            "post_arrival_viability": post_arrival_viability,
            "completion_contract": completion_contract,
        })

    def finish_room_search_incomplete(
        decision_index: int,
        completion_contract: Dict[str, Any],
        attempt_record: Dict[str, Any],
        *,
        candidate: Optional[Dict[str, Any]] = None,
        preflight_attempt_count: int = 0,
        request_room_return: bool = True,
        strategic_reposition_state: Optional[str] = None,
        post_arrival_viability: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Safely terminate a local-unavailability segment; recovery is deferred."""
        topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
        safe_stop = (
            publish_zero_to_topic(topic, 0.0, 10.0)
            if args.execute else {"topic": topic, "zero_count": 0, "reason": "dry_run_no_cmd_published"}
        )
        resolved_reposition_state = strategic_reposition_state or "NOT_CALLED_LOCAL_UNAVAILABILITY"
        attempt_record.update({
            "completion_contract": completion_contract,
            "safe_stop": safe_stop,
            "strategic_reposition_state": resolved_reposition_state,
        })
        result["search_attempts"].append(attempt_record)
        online["search_completed"] = False
        online["search_aborted_incomplete"] = bool(completion_contract.get("search_aborted_incomplete", False))
        online["room_return_started"] = False
        result["search_completion_reason"] = None
        result["local_search_availability"] = completion_contract["local_availability"]
        result["mission_completion"] = completion_contract
        result["search_aborted_incomplete"] = online["search_aborted_incomplete"]
        result["gain_history"] = search.gain_history
        result["final_decision"] = "ROOM_SEARCH_INCOMPLETE_LOCAL_UNAVAILABLE"
        diagnostic_kwargs: Dict[str, Any] = {
            "execution_result": str(completion_contract["local_availability"]),
            "completion_contract": completion_contract,
        }
        if strategic_reposition_state is not None:
            diagnostic_kwargs["reposition_state"] = str(strategic_reposition_state)
        if post_arrival_viability is not None:
            diagnostic_kwargs["post_arrival_viability"] = post_arrival_viability
        persist_diagnostic(
            decision_index, candidate=candidate, preflight_attempt_count=preflight_attempt_count,
            **diagnostic_kwargs,
        )
        write_stage("room_search", "navigation_state_machine.py", {
            "mode": "ROOM_SEARCH_INCOMPLETE", "local_availability": completion_contract["local_availability"],
        })
        if request_room_return:
            return request_room_return_exit(completion_contract)
        result["room_search_exit_action"] = {
            "search_outcome": "SEARCH_INCOMPLETE_LOCAL_FAILURE",
            "room_return_requested": False,
            "next_control_flow": "TERMINATE_WITHOUT_RETURN",
            "reason": "CONSTRAINED_ARRIVAL_RECOVERY_NOT_VIABLE",
        }
        return finish(result)

    def enrich_candidate(
        candidate: Dict[str, Any], context: Dict[str, Any], pose: Tuple[float, float, float], active_hypotheses: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        row = dict(candidate)
        base_xy = row["base_xy"]
        heading = math.atan2(float(base_xy[1]), float(base_xy[0]))
        visible_base_points = room_search_v2_visible_base_points(context, base_xy, heading, camera_hfov)
        visible_room_points = [anchor.local_xy(transform_base_xy(point, pose)) for point in visible_base_points]

        def current_los_clear(room_point: Sequence[float]) -> bool:
            return room_search_v2_line_clear(
                context, (0.0, 0.0), target_base_xy(anchor.odom_xy(room_point), pose)
            )

        danger_opportunities = []
        for hypothesis in active_hypotheses:
            position = hypothesis.get("position_xyz_m")
            if not isinstance(position, (list, tuple)) or len(position) < 2:
                continue
            hypothesis_base = target_base_xy(position, pose)
            dx, dy = float(hypothesis_base[0]) - float(base_xy[0]), float(hypothesis_base[1]) - float(base_xy[1])
            bearing = abs(normalize_angle(math.atan2(dy, dx) - heading))
            if room_search_v2_line_clear(context, base_xy, hypothesis_base) and bearing <= camera_hfov / 2.0:
                danger_opportunities.append({
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "position_xyz_m": list(position),
                    "confidence": hypothesis.get("confidence"),
                    "support_count": hypothesis.get("support_count"),
                    "last_observed_stamp_sec": hypothesis.get("last_observed_stamp_sec"),
                    "abs_bearing_rad": bearing,
                    "distance_m": math.hypot(dx, dy),
                })
        danger_opportunities.sort(key=lambda item: (float(item["abs_bearing_rad"]), float(item["distance_m"])))
        row.update({
            "heading_change_rad": heading,
            "visible_room_points": visible_room_points,
            "occlusion_reveal_room_points": occlusion_reveal_room_points(
                search.observation, visible_room_points, current_los_clear,
            ),
            "danger_reobserve_supported": bool(danger_opportunities),
            "danger_reobserve_opportunities": danger_opportunities,
            "danger_reobserve_abs_bearing_rad": None if not danger_opportunities else danger_opportunities[0]["abs_bearing_rad"],
        })
        return row

    def maybe_hold_for_danger_reobserve(
        context: Dict[str, Any], pose: Tuple[float, float, float], active_hypotheses: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Take one bounded, stationary observation chance when already centered on a hypothesis."""
        for hypothesis in active_hypotheses:
            hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
            position = hypothesis.get("position_xyz_m")
            if not hypothesis_id or hypothesis_id in held_hypothesis_ids or not isinstance(position, (list, tuple)) or len(position) < 2:
                continue
            base_xy = target_base_xy(position, pose)
            bearing = abs(math.atan2(float(base_xy[1]), float(base_xy[0])))
            if float(base_xy[0]) <= 0.0 or bearing > camera_hfov / 4.0 or not room_search_v2_line_clear(context, (0.0, 0.0), base_xy):
                continue
            held_hypothesis_ids.add(hypothesis_id)
            topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
            hold = publish_zero_to_topic(topic, ROOM_SEARCH_REOBSERVE_HOLD_SIM_SEC, 10.0)
            hold.update({"event": "DANGER_REOBSERVE_HOLD", "hypothesis_id": hypothesis_id, "bearing_rad": bearing})
            return hold
        return None

    def attempt_strategic_reposition(
        failed_candidate: Dict[str, Any], failure_decision: str,
    ) -> Dict[str, Any]:
        current_pose = pose_tuple(read_odom())
        anchors = search.reposition_anchor_candidates(
            current_pose, max(float(args.stuck_min_displacement_m), float(args.goal_tolerance_m)), maximum_attempts=2,
        )
        record: Dict[str, Any] = {
            "trigger": failure_decision,
            "failed_sector": failed_candidate.get("sector"),
            "anchors_considered": [],
            "preflight_attempt_count": 0,
            "state": "STRATEGIC_REPOSITION_UNAVAILABLE",
            "successful": False,
        }
        for anchor_option in anchors:
            if int(record["preflight_attempt_count"]) >= 2:
                return record
            target_xy = anchor_option["target_xy_team_livox_odom"]
            local_base = target_base_xy(target_xy, current_pose)
            local_support = local_base[0] >= 0.15 and math.hypot(*local_base) <= 1.5
            item = dict(anchor_option)
            item.update({"target_base_xy": list(local_base), "local_planning_support": local_support})
            if not local_support:
                # A real safe anchor can be behind the body.  Select at most
                # one formally admitted forward/local transition instead of
                # silently discarding the reposition episode.
                transition_context = room_search_v2_planning_context(args, grid_status_pairs)
                transition_candidates = search.candidates_from_planning_free_base(
                    current_pose, transition_context.get("free_base_points", []),
                ) if transition_context.get("qualified") else []
                target_bearing = math.atan2(local_base[1], local_base[0])
                transition_candidates = sorted(
                    transition_candidates,
                    key=lambda row: abs(normalize_angle(math.atan2(float(row["base_xy"][1]), float(row["base_xy"][0])) - target_bearing)),
                )
                item["preflight"] = "DIRECT_ANCHOR_UNSUPPORTED_TRY_LOCAL_TRANSITION"
                record["anchors_considered"].append(item)
                for transition_candidate in transition_candidates:
                    if int(record["preflight_attempt_count"]) >= 2:
                        return record
                    transition_xy = transition_candidate["target_xy_team_livox_odom"]
                    record["preflight_attempt_count"] += 1
                    transition_preflight = room_search_v2_preflight(
                        args, transition_xy, {"strategic_reposition_transition_for": item, "failed_sector": failed_candidate.get("sector")},
                    )
                    transition_candidate["legal"] = bool(transition_preflight.get("legal"))
                    transition = search.choose_safe_return_transition(current_pose, target_xy, [transition_candidate])
                    if transition is None:
                        continue
                    attempt = execute_target(
                        transition_xy, "ROOM_SEARCH_REPOSITION_TRANSITION",
                        {"strategic_reposition_transition_for": item, "failed_sector": failed_candidate.get("sector")},
                    )
                    terminal_odom = read_odom()
                    terminal_pose = pose_tuple(terminal_odom)
                    displacement = math.hypot(terminal_pose[0] - current_pose[0], terminal_pose[1] - current_pose[1])
                    runner_decision = str(attempt["runner"].get("runner_final_decision") or "")
                    if runner_decision == "BLOCK_ASTAR_DWA_REACHED_GOAL" and displacement >= float(args.stuck_min_displacement_m):
                        search.record_actual_breadcrumb(terminal_pose, float(args.goal_tolerance_m))
                        search.record_safe_actual_pose(terminal_pose, float(args.goal_tolerance_m))
                        search.arm_one_decision_sector_cooldown(failed_candidate.get("sector"))
                        record.update({
                            "state": "STRATEGIC_REPOSITION_TRANSITION_SUCCEEDED",
                            "successful": True,
                            "actual_displacement_m": displacement,
                            "actual_terminal_odom": terminal_odom,
                            "runner": attempt["runner"],
                        })
                        return record
                continue
            record["preflight_attempt_count"] += 1
            preflight = room_search_v2_preflight(
                args, target_xy, {"strategic_reposition_anchor": item, "failed_sector": failed_candidate.get("sector")},
            )
            item["preflight"] = {"legal": bool(preflight.get("legal")), "path_length_m": preflight.get("path_length_m")}
            record["anchors_considered"].append(item)
            if not preflight.get("legal"):
                continue
            attempt = execute_target(
                target_xy, "ROOM_SEARCH_REPOSITION", {"strategic_reposition_anchor": item, "failed_sector": failed_candidate.get("sector")},
            )
            terminal_odom = read_odom()
            terminal_pose = pose_tuple(terminal_odom)
            displacement = math.hypot(terminal_pose[0] - current_pose[0], terminal_pose[1] - current_pose[1])
            runner_decision = str(attempt["runner"].get("runner_final_decision") or "")
            item.update({
                "runner_final_decision": runner_decision,
                "actual_terminal_pose_xy_yaw": list(terminal_pose),
                "actual_displacement_m": displacement,
            })
            if runner_decision == "BLOCK_ASTAR_DWA_REACHED_GOAL" and displacement >= float(args.stuck_min_displacement_m):
                search.record_actual_breadcrumb(terminal_pose, float(args.goal_tolerance_m))
                search.record_safe_actual_pose(terminal_pose, float(args.goal_tolerance_m))
                search.arm_one_decision_sector_cooldown(failed_candidate.get("sector"))
                record.update({
                    "state": "STRATEGIC_REPOSITION_SUCCEEDED",
                    "successful": True,
                    "actual_displacement_m": displacement,
                    "actual_terminal_odom": terminal_odom,
                    "runner": attempt["runner"],
                })
                return record
        return record

    write_stage("room_search", "navigation_state_machine.py", {"mode": "ROOM_EXPLORE"})
    decisions = 0
    nonproductive_progress_cycles = 0
    progress_guard: Dict[str, Any] = {
        "actual_new_observation_cells": None,
        "substantive_progress": False,
        "nonproductive_progress_cycles": 0,
        "guard_limit": ROOM_SEARCH_NONPRODUCTIVE_PROGRESS_GUARD_LIMIT,
        "finite_abort": False,
    }
    completion_reason: Optional[str] = None
    mission_completion: Optional[Dict[str, Any]] = None

    def record_nonproductive_cycle(reason: str, actual_new_observation_cells: Any = 0) -> bool:
        """Account one no-progress cycle without changing navigation policy."""
        nonlocal nonproductive_progress_cycles, progress_guard
        progress_guard = room_search_v2_progress_guard_update(
            nonproductive_progress_cycles, actual_new_observation_cells,
        )
        nonproductive_progress_cycles = int(progress_guard["nonproductive_progress_cycles"])
        result["nonproductive_progress_guard"] = {
            **progress_guard,
            "last_cycle_reason": str(reason),
        }
        return bool(progress_guard["finite_abort"])

    while not rospy.is_shutdown():
        pose = pose_tuple(read_odom())
        camera_hfov, camera_hfov_source = camera_info.effective_hfov()
        context = room_search_v2_planning_context(args, grid_status_pairs)
        latest_planning_context.clear()
        latest_planning_context.update(context)
        if not context.get("matched_pair_found"):
            result["first_planning_context"] = room_search_v2_context_record(context)
            completion_contract = room_search_v2_completion_contract("ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE", None)
            return finish_room_search_incomplete(
                decisions, completion_contract, {"candidate": None, "reason": "ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE"},
            )
        if not context.get("qualified"):
            result["first_planning_context"] = room_search_v2_context_record(context)
            completion_contract = room_search_v2_completion_contract("ROOM_SEARCH_GRID_STATUS_CONTEXT_UNQUALIFIED", None)
            return finish_room_search_incomplete(
                decisions, completion_contract, {"candidate": None, "reason": "ROOM_SEARCH_GRID_STATUS_CONTEXT_UNQUALIFIED"},
            )
        active_hypotheses = danger_hypotheses.snapshot(float(rospy.Time.now().to_sec()))
        hold = maybe_hold_for_danger_reobserve(context, pose, active_hypotheses)
        if hold is not None:
            result["danger_reobserve_holds"].append(hold)
            online["danger_reobserve_holds"] = int(online["danger_reobserve_holds"]) + 1
            persist_diagnostic(decisions, execution_result="DANGER_REOBSERVE_HOLD")
            if record_nonproductive_cycle("DANGER_REOBSERVE_HOLD"):
                break
            continue
        if not entry_seen_initialized:
            entry_visible = room_search_v2_visible_room_points(
                context, anchor, pose, (0.0, 0.0), 0.0, camera_hfov,
            )
            entry_seen_cell_count = search.update_actual_view(pose, entry_visible)
            entry_seen_initialized = True
            persist_diagnostic(decisions, execution_result="ENTRY_SEEN_INITIALIZED")
        candidate_generation_audit: Dict[str, Any] = {}
        candidates = search.candidates_from_planning_free_base(
            pose,
            context.get("free_base_points", []),
            lambda raw: enrich_candidate(raw, context, pose, active_hypotheses),
            audit=candidate_generation_audit,
        ) if context.get("qualified") else []
        if "first_planning_context" not in result:
            result["first_planning_context"] = room_search_v2_context_record(context, len(candidates))
        cheap_ranked_all = search.cheap_rank_candidates(candidates)
        cheap_ranked = cheap_ranked_all[:MAX_CANDIDATES]

        ranked_representatives = []
        for rank, ranked_candidate in enumerate(cheap_ranked_all, 1):
            ranked_item = RoomSearchV2._candidate_audit_summary(ranked_candidate)
            ranked_item.update({
                "global_rank": int(rank),
                "rank_components": room_search_v2_rank_components(ranked_candidate),
            })
            ranked_representatives.append(ranked_item)
        preflight_evidence: List[Dict[str, Any]] = []

        # Stage-B is an authority-free, one-shot snapshot.  This hook is before
        # every production preflight and its return value is deliberately unused.
        try:
            dry_shadow_args = copy.copy(args)
            dry_shadow_args.execute = False
            stage_b_shadow_capture.try_capture_epoch(
                run_id=str(os.environ.get("STATE_MACHINE_RUN_ID", "")),
                decision_id=int(decisions + 1),
                ros_timestamp_sec=float(rospy.Time.now().to_sec()),
                pose_odom_xy_yaw=pose,
                pose_room_xy=search.room_xy(pose),
                portal_context=result["portal_anchor"],
                planning_context=context,
                seen_cells=search.observation.seen,
                ranked_candidates=cheap_ranked_all,
                parameters=vars(args),
                runner_command_template=runner_cmd(
                    dry_shadow_args, state="ROOM_SEARCH",
                    runtime_sec=args.runner_runtime_sec, max_steps=1,
                    local_control_mode="ROOM_LOCAL",
                ),
                breadcrumbs=search.breadcrumbs,
            )
        except Exception:
            pass

        def preflight_ranked(candidate: Dict[str, Any]) -> Dict[str, Any]:
            return room_search_v2_action_aware_preflight(
                args, candidate, pose, tuple(sorted(search.observation.seen)),
            )

        def observe_preflight(rank: int, observed_candidate: Dict[str, Any], preflight: Dict[str, Any]) -> None:
            preflight_evidence.append(room_search_v2_preflight_audit_record(rank, observed_candidate, preflight))
            try:
                high_level_locomotion_shadow.record_preflight(
                    decision_id=int(decisions + 1), rank=int(rank), candidate=observed_candidate,
                    preflight=preflight, pose_xy_yaw=pose, planning_context=context,
                )
            except Exception:
                # A telemetry failure is intentionally isolated from admission.
                pass

        stage_b_admission_started_ns = time.perf_counter_ns()
        admission = room_search_v2_admit_with_l3v_consistency(
            cheap_ranked_all, preflight_ranked, MAX_CANDIDATES, audit_observer=observe_preflight,
        )
        stage_b_admission_elapsed_ns = time.perf_counter_ns() - stage_b_admission_started_ns
        candidate = admission["candidate"]
        preflight_attempt_count = int(admission["preflight_attempt_count"])
        terminal_expansion_attempt_count = int(admission["terminal_expansion_attempt_count"])
        decision_global_l3v_invalidation = admission["decision_global_l3v_invalidation"]
        terminal_expansion_used = terminal_expansion_attempt_count > 0
        try:
            stage_b_shadow_capture.record_production_admission(
                decision_id=int(decisions + 1),
                admission_elapsed_ns=stage_b_admission_elapsed_ns,
                preflight_attempt_count=preflight_attempt_count,
                terminal_expansion_attempt_count=terminal_expansion_attempt_count,
                production_preflight_grid_identity={
                    "grid_header_stamp_sec": context.get("grid_header_stamp_sec"),
                    "grid_content_stamp": context.get("grid_content_stamp"),
                    "content_generation_id": context.get("content_generation_id"),
                    "grid_content_hash": context.get("grid_content_hash"),
                },
                preflight_sequence=[
                    {
                        "rank": row.get("global_rank"),
                        "candidate_id": (row.get("candidate") or {}).get("candidate_id"),
                        "legal": row.get("legal"),
                    }
                    for row in preflight_evidence
                ],
                selected_candidate_id=(
                    candidate.get("_room_search_audit_candidate_id") if candidate is not None else None
                ),
                selected_rank=(candidate.get("cheap_rank") if candidate is not None else None),
                decision_global_l3v_invalidation=decision_global_l3v_invalidation is not None,
            )
        except Exception:
            pass
        candidate_audit: Dict[str, Any] = {
            "schema_version": "room_search_candidate_evidence_v1",
            "run_id": os.environ.get("STATE_MACHINE_RUN_ID", ""),
            "decision_index": int(decisions + 1),
            "decision_timestamp_sec": float(rospy.Time.now().to_sec()),
            "actual_pose_odom_xy_yaw": list(pose),
            "actual_pose_room_xy": list(search.room_xy(pose)),
            "grid_identity": {
                "grid_header_stamp_sec": context.get("grid_header_stamp_sec"),
                "status_grid_content_stamp": context.get("grid_content_stamp"),
                "content_generation_id": context.get("content_generation_id"),
                "grid_content_hash": context.get("grid_content_hash"),
            },
            "candidate_generation_qualification": {
                "matched_pair_found": context.get("matched_pair_found"),
                "qualified": context.get("qualified"),
                "qualification_errors": list(context.get("qualification_errors") or []),
                "candidate_generation_reached": context.get("candidate_generation_reached"),
            },
            "counts": {
                "raw_candidates_before_sector_compression": candidate_generation_audit.get("raw_candidate_count_before_sector_compression", 0),
                "sector_representatives": candidate_generation_audit.get("sector_representative_count", 0),
                "ranked_representatives": len(cheap_ranked_all),
                "normal_preflight_candidates": len(cheap_ranked),
                "preflight_attempts": int(preflight_attempt_count),
                "terminal_expansion_attempts": int(terminal_expansion_attempt_count),
            },
            "raw_candidates": candidate_generation_audit.get("raw_candidates", []),
            "ranked_representatives": ranked_representatives,
            "preflight_attempts": preflight_evidence,
        }
        if decision_global_l3v_invalidation is not None:
            status = decision_global_l3v_invalidation.get("status_payload", {})
            status = status if isinstance(status, dict) else {}
            invalidation = {
                "decision_index": int(decisions + 1),
                "first_observed_candidate": decision_global_l3v_invalidation.get("candidate"),
                "first_observed_global_rank": decision_global_l3v_invalidation.get("global_rank"),
                "consumed_grid_status_generation": status.get("content_generation_id"),
                "consumed_grid_content_stamp": status.get("grid_content_stamp"),
                "consumed_grid_content_hash": status.get("grid_content_hash"),
                "local_traversability_status": status.get("local_traversability_status"),
                "remaining_lower_ranked_candidates_not_preflighted": decision_global_l3v_invalidation.get(
                    "remaining_lower_ranked_candidates_not_preflighted"
                ),
            }
            candidate_audit.update({
                "decision_outcome": ROOM_SEARCH_DECISION_TRANSIENT_L3V_UNAVAILABLE,
                "selected_candidate": None,
                "selection_reason": "DECISION_GLOBAL_L3V_STATUS_BEFORE_TARGET_ASTAR_DWA",
                "decision_global_l3v_invalidation": invalidation,
                "post_execution": "NOT_RUN_DECISION_INVALIDATED_BEFORE_TARGET_ASTAR_DWA",
            })
            append_room_search_candidate_audit(candidate_audit)
            decisions += 1
            topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
            safe_stop = (
                publish_zero_to_topic(topic, 0.0, 10.0)
                if args.execute else {"topic": topic, "zero_count": 0, "reason": "dry_run_no_cmd_published"}
            )
            result["search_attempts"].append({
                "decision_index": int(decisions),
                "reason": ROOM_SEARCH_DECISION_TRANSIENT_L3V_UNAVAILABLE,
                "invalidation": invalidation,
                "safe_stop": safe_stop,
            })
            persist_diagnostic(
                decisions, preflight_attempt_count=preflight_attempt_count,
                execution_result=ROOM_SEARCH_DECISION_TRANSIENT_L3V_UNAVAILABLE,
            )
            if record_nonproductive_cycle(ROOM_SEARCH_DECISION_TRANSIENT_L3V_UNAVAILABLE):
                break
            continue
        if candidate is None:
            candidate_audit.update({
                "decision_outcome": "NO_SAFE_USEFUL_CANDIDATE",
                "selected_candidate": None,
                "selection_reason": "ALL_EXISTING_PREFLIGHTS_REJECTED",
                "post_execution": "NOT_RUN_NO_ADMITTED_CANDIDATE",
            })
            append_room_search_candidate_audit(candidate_audit)
            completion_contract = room_search_v2_completion_contract("NO_SAFE_USEFUL_CANDIDATE", None)
            return finish_room_search_incomplete(decisions, completion_contract, {
                "candidate": None, "reason": "NO_SAFE_USEFUL_CANDIDATE", "raw_candidate_count": len(candidates),
                "cheap_ranked_count": len(cheap_ranked), "raw_sector_representative_count": len(candidates), "preflight_attempt_count": preflight_attempt_count,
                "terminal_expansion_used": terminal_expansion_used, "terminal_expansion_attempt_count": terminal_expansion_attempt_count,
            }, preflight_attempt_count=preflight_attempt_count)
        # The opt-in audit hook freezes only values already produced by this
        # decision.  It has no preflight, ranking, selection, or command
        # authority; all bundle I/O runs on its one-shot background writer.
        try:
            dry_identity_args = copy.copy(args)
            dry_identity_args.execute = False
            frozen_decision_capture.capture_once(
                run_id=str(os.environ.get("STATE_MACHINE_RUN_ID", "")),
                decision_id=int(decisions + 1),
                ros_timestamp_sec=float(candidate_audit["decision_timestamp_sec"]),
                pose_odom_xy_yaw=pose,
                pose_room_xy=search.room_xy(pose),
                portal_context=result["portal_anchor"],
                planning_context=context,
                seen_cells=search.observation.seen,
                ranked_candidates=cheap_ranked_all,
                selected_candidate=candidate,
                parameters=vars(args),
                runner_command_template=runner_cmd(
                    dry_identity_args,
                    state="ROOM_SEARCH",
                    runtime_sec=args.runner_runtime_sec,
                    max_steps=1,
                ),
                breadcrumbs=search.breadcrumbs,
                capture_trigger_reason="FIRST_QUALIFIED_NORMAL_ROOM_SEARCH_DECISION_WITH_MULTIPLE_RANKS",
            )
        except Exception:
            # Audit failure must never alter production decision authority.
            pass
        decisions += 1
        stage_b_nbv_started_ns = time.perf_counter_ns()
        candidate = search.score_candidates([candidate])[0]
        stage_b_nbv_elapsed_ns = time.perf_counter_ns() - stage_b_nbv_started_ns
        # R1 is evidence-only.  The current dry preflight does not expose a
        # full action-completion terminal state, so this records UNKNOWN rather
        # than reusing the one-slice DWA continuation endpoint.  Nothing below
        # reads this record for admission, selection, or command authority.
        recoverability_started_ns = time.perf_counter_ns()
        recoverability_epoch = recoverability_epoch_identity(context)
        recoverability_root = make_door_anchor_root_certificate(anchor, recoverability_epoch)
        recoverability_terminal = prepare_candidate_terminal_state(
            candidate, int(decisions), recoverability_epoch,
        )
        recoverability_certificate = evaluate_predecessor_set(
            recoverability_terminal, [recoverability_root], {},
        )
        recoverability_shadow: Dict[str, Any] = {
            "event": "ROOM_SEARCH_RECOVERABILITY_R1_SHADOW",
            "authority_enabled": False,
            "selection_consumed": False,
            "future_consumer": "FORMAL_MISSION_COMPARISON_OR_ADMISSION_REVIEW",
            "candidate_id": recoverability_terminal.candidate_id,
            "decision_id": int(decisions),
            "root_certificate": recoverability_root.to_dict(),
            "predicted_terminal_state": recoverability_terminal.to_dict(),
            "recoverability_certificate": recoverability_certificate.to_dict(),
            "evaluation_wall_ms": (time.perf_counter_ns() - recoverability_started_ns) / 1_000_000.0,
        }
        result["recoverability_shadow_events"].append(recoverability_shadow)
        candidate_audit["recoverability_shadow"] = recoverability_shadow
        mission_action_spec: MissionActionSpec = freeze_mission_action_spec(
            candidate,
            pose,
            "room_search_decision_%04d_%s" % (
                int(decisions), str(candidate.get("_room_search_audit_candidate_id") or "unknown"),
            ),
            tuple(sorted(search.observation.seen)),
        )
        mission_action_record: Dict[str, Any] = {"spec": mission_action_spec.to_dict()}
        result["mission_actions"].append(mission_action_record)
        candidate_audit["mission_action"] = {"spec": mission_action_spec.to_dict()}
        candidate_audit.update({
            "decision_index": int(decisions),
            "decision_outcome": "ADMITTED_CANDIDATE",
            "selected_candidate": RoomSearchV2._candidate_audit_summary(candidate),
            "selected_global_rank": candidate.get("cheap_rank"),
            "selection_reason": "FIRST_FORMALLY_ADMITTED_EXISTING_CHEAP_RANK",
        })
        stage_b_marginal_started_ns = time.perf_counter_ns()
        marginal = search.evaluate_marginal_value(candidate, decisions)
        stage_b_marginal_elapsed_ns = time.perf_counter_ns() - stage_b_marginal_started_ns
        try:
            stage_b_shadow_capture.record_production_nbv(
                decision_id=int(decisions + 1),
                selected_candidate_id=candidate.get("_room_search_audit_candidate_id"),
                selected_rank=candidate.get("cheap_rank"),
                singleton_nbv_elapsed_ns=stage_b_nbv_elapsed_ns,
                nbv_value=candidate.get("nbv_value"),
                marginal_elapsed_ns=stage_b_marginal_elapsed_ns,
                completion_reason=marginal.get("completion_reason"),
            )
        except Exception:
            pass
        result["search_decisions"].append(marginal)
        if marginal["completion_reason"] is not None and not active_hypotheses:
            completion_reason = str(marginal["completion_reason"])
            mission_completion = room_search_v2_completion_contract("ADMITTED_CANDIDATE", completion_reason)
            try:
                stage_b_shadow_capture.record_production_completion(
                    decision_id=int(decisions + 1), completion_reason=completion_reason,
                    completion_contract=mission_completion,
                )
                stage_b_shadow_capture.record_execution_handoff(
                    decision_id=int(decisions + 1),
                    selected_candidate_id=candidate.get("_room_search_audit_candidate_id"),
                    selected_rank=candidate.get("cheap_rank"),
                    handoff="HANDOFF_NOT_RUN_EXISTING_COMPLETION",
                    completion_reason=completion_reason,
                )
            except Exception:
                pass
            candidate_audit.update({
                "decision_outcome": "TASK_COMPLETION_BEFORE_EXECUTION",
                "selection_reason": "EXISTING_MARGINAL_COMPLETION_CONTRACT",
                "post_execution": "NOT_RUN_EXISTING_COMPLETION_CONTRACT",
                "completion_contract": mission_completion,
            })
            append_room_search_candidate_audit(candidate_audit)
            result["search_attempts"].append({
                "candidate": candidate, "reason": completion_reason, "marginal": marginal,
                "raw_candidate_count": len(candidates), "cheap_ranked_count": len(cheap_ranked), "raw_sector_representative_count": len(candidates),
                "preflight_attempt_count": preflight_attempt_count,
                "completion_contract": mission_completion,
            })
            persist_diagnostic(
                decisions, candidate=candidate, preflight_attempt_count=preflight_attempt_count,
                execution_result=completion_reason, completion_contract=mission_completion,
            )
            break
        target_xy = candidate["target_xy_team_livox_odom"]
        danger_action_spec: Optional[DangerReobserveMissionActionSpec] = freeze_danger_reobserve_mission_action_spec(
            candidate,
            action_id="danger_reobserve_decision_%04d_%s" % (
                int(decisions), str(candidate.get("_room_search_audit_candidate_id") or "unknown"),
            ),
            run_id=str(os.environ.get("STATE_MACHINE_RUN_ID", "")),
            dispatch_time_sec=float(rospy.Time.now().to_sec()),
        )
        danger_action_record: Optional[Dict[str, Any]] = None
        if danger_action_spec is not None:
            if danger_reobserve_episode_guard.admission_allowed(danger_action_spec):
                danger_action_record = {
                    "spec": danger_action_spec.to_dict(),
                    "formal_admission": "FORMAL_DANGER_REOBSERVE_EVIDENCE_AVAILABLE",
                    "authority": "FORMAL_MISSION_SEMANTICS_ONLY",
                }
                result["danger_reobserve_actions"].append(danger_action_record)
                mission_action_record["danger_reobserve"] = danger_action_record
                candidate_audit["mission_action"]["danger_reobserve"] = danger_action_record
            else:
                suppression = danger_reobserve_episode_guard.suppression_reason(danger_action_spec)
                mission_action_record["danger_reobserve"] = {
                    "spec": danger_action_spec.to_dict(),
                    "formal_admission": "SUPPRESSED",
                    "suppression_reason": suppression or "FORMAL_DANGER_ACTION_INVALID",
                    "authority": "FORMAL_MISSION_SEMANTICS_ONLY",
                }
                candidate_audit["mission_action"]["danger_reobserve"] = mission_action_record["danger_reobserve"]
        search.record_safe_actual_pose(pose, float(args.goal_tolerance_m))
        try:
            stage_b_shadow_capture.record_execution_handoff(
                decision_id=int(decisions + 1),
                selected_candidate_id=candidate.get("_room_search_audit_candidate_id"),
                selected_rank=candidate.get("cheap_rank"),
                target_xy_team_livox_odom=list(target_xy),
                handoff="EXISTING_EXECUTE_TARGET",
            )
        except Exception:
            pass
        position_satisfied_for_action = bool(candidate.get("position_satisfied", False))
        if position_satisfied_for_action:
            attempt = {
                "state": "ROOM_SEARCH", "local_control_mode": "ROOM_LOCAL",
                "command": {"cmd": [], "reason": "POSITION_SATISFIED_NO_TRANSLATION"},
                "runner": {
                    "runner_final_decision": "POSITION_SATISFIED_NO_TRANSLATION",
                    "last_dwa": {}, "observation_orientation": {},
                },
            }
        else:
            attempt = execute_target(target_xy, "ROOM_SEARCH_TARGET", {"room_search_candidate": candidate, "portal_anchor": result["portal_anchor"]})
        runner_decision = str(attempt["runner"].get("runner_final_decision") or "")
        record = {
            "candidate": candidate, "marginal": marginal, "runner": attempt["runner"], "runner_command": attempt["command"],
            "raw_candidate_count": len(candidates), "cheap_ranked_count": len(cheap_ranked), "raw_sector_representative_count": len(candidates),
            "preflight_attempt_count": preflight_attempt_count,
            "terminal_expansion_used": terminal_expansion_used,
            "terminal_expansion_attempt_count": terminal_expansion_attempt_count,
            "position_satisfied_for_action": position_satisfied_for_action,
        }
        terminal_evidence = mission_action_terminal_evidence(mission_action_spec, candidate)
        terminal_odom = terminal_evidence["terminal_odom"]
        terminal_pose = terminal_evidence["terminal_pose"]
        if position_satisfied_for_action:
            pre_orientation_status = str(
                terminal_evidence["mission_observation"].observation_intent_status
            )
            record["position_observation_handoff"] = {
                "lifecycle": (
                    "OBSERVATION_ORIENTATION_PENDING"
                    if pre_orientation_status == "OBSERVATION_INTENT_UNSATISFIED"
                    else "POSITION_SATISFIED_OBSERVATION_EVALUATED"
                ),
                "pre_orientation_observation_intent_status": pre_orientation_status,
                "pre_orientation_terminal_pose_xy_yaw": list(terminal_pose),
            }
        try:
            high_level_locomotion_shadow.record_outcome(
                decision_id=int(decisions), candidate=candidate, start_pose=pose,
                terminal_pose=terminal_pose, runner=attempt["runner"],
            )
        except Exception:
            # A telemetry failure is intentionally isolated from execution.
            pass
        breadcrumb_added = False
        terminal_context = terminal_evidence["terminal_context"]
        if (
            position_satisfied_for_action
            and terminal_evidence["mission_observation"].observation_intent_status == "OBSERVATION_INTENT_UNSATISFIED"
        ):
            orientation_attempt = execute_observation_orientation_slice(mission_action_spec, candidate)
            orientation_runner = orientation_attempt.get("runner") if isinstance(orientation_attempt.get("runner"), dict) else {}
            record["observation_orientation"] = {
                "trigger": "POSITION_SATISFIED_AND_OBSERVATION_INTENT_UNSATISFIED",
                "lifecycle": "OBSERVATION_ORIENTATION_PENDING",
                "runner": orientation_runner,
                "command": orientation_attempt.get("command"),
            }
            if str(orientation_runner.get("runner_final_decision") or "") == "OBSERVATION_ORIENTATION_SLICE_EXECUTED":
                attempt = orientation_attempt
                runner_decision = str(orientation_runner.get("runner_final_decision"))
                record["runner"] = orientation_runner
                record["runner_command"] = orientation_attempt.get("command")
                terminal_evidence = mission_action_terminal_evidence(
                    mission_action_spec, candidate, require_fresh_grid_status=True,
                )
                terminal_odom = terminal_evidence["terminal_odom"]
                terminal_pose = terminal_evidence["terminal_pose"]
                terminal_context = terminal_evidence["terminal_context"]
                record["observation_orientation"]["post_slice_terminal_evidence"] = {
                    "terminal_pose_xy_yaw": list(terminal_pose),
                    "terminal_grid_identity": terminal_evidence["terminal_visibility_audit"].get("grid_identity"),
                    "observation_intent_status": str(
                        terminal_evidence["mission_observation"].observation_intent_status
                    ),
                }
        breadcrumb_added = search.record_actual_breadcrumb(terminal_pose, float(args.goal_tolerance_m))
        if danger_action_spec is not None and danger_action_record is not None:
            terminal_time_sec = float(rospy.Time.now().to_sec())
            danger_terminal = evaluate_danger_reobserve_terminal(
                danger_action_spec,
                execution_viewpoint_obtained=bool(
                    position_satisfied_for_action or runner_decision == "BLOCK_ASTAR_DWA_REACHED_GOAL"
                ),
                terminal_time_sec=terminal_time_sec,
                confirmed_tracks=danger_tracks.snapshot(),
                tentative_hypotheses=danger_hypotheses.snapshot(terminal_time_sec),
            )
            danger_reobserve_episode_guard.close(danger_action_spec, danger_terminal)
            danger_action_record["terminal"] = danger_terminal
            danger_action_record["episode_closed"] = True
            danger_action_record["repeat_guard"] = "SAME_RUN_HYPOTHESIS_SUPPRESSED_AFTER_TERMINAL"
            record["danger_reobserve"] = danger_action_record
            candidate_audit["mission_action"]["danger_reobserve"] = danger_action_record
        # Freeze the existing SEEN identity before the one existing terminal
        # update.  This is audit data only and is never passed to control.
        shadow_pre_update_seen: Tuple[Tuple[int, int], ...] = ()
        shadow_prediction: Optional[Dict[str, Any]] = None
        shadow_preparation_error: Optional[str] = None
        if observation_arrival_shadow_enabled:
            shadow_pre_update_seen = tuple(sorted(search.observation.seen))
            try:
                evidence = _candidate_evidence(
                    dict(candidate), int(candidate.get("cheap_rank") or 0), shadow_pre_update_seen,
                )
                opportunity = evidence.get("opportunity")
                candidate_id = str(evidence.get("candidate_id") or "")
                predicted_cell_ids = list(opportunity.get("predicted_new_cell_ids") or []) if isinstance(opportunity, dict) else None
                predicted_count = opportunity.get("predicted_new_count") if isinstance(opportunity, dict) else None
                if not candidate_id or predicted_cell_ids is None:
                    raise ValueError("predicted_opportunity_identity_unavailable")
                if int(predicted_count) != len(predicted_cell_ids):
                    raise ValueError("predicted_new_cell_ids_count_mismatch")
                if not finite_number(candidate.get("new_observable_cells")):
                    raise ValueError("production_predicted_new_count_unavailable")
                if int(candidate["new_observable_cells"]) != int(predicted_count):
                    raise ValueError("production_predicted_count_identity_mismatch")
                shadow_prediction = {
                    "candidate_id": candidate_id,
                    "predicted_new_count": int(predicted_count),
                    "predicted_new_cell_ids": predicted_cell_ids,
                }
            except Exception as exc:
                shadow_preparation_error = "%s:%s" % (type(exc).__name__, exc)
        terminal_visibility_audit = terminal_evidence["terminal_visibility_audit"]
        actual_visible = terminal_evidence["actual_visible"]
        actual_new_observation_cells = search.update_actual_view(terminal_pose, actual_visible)
        if int(actual_new_observation_cells) != int(terminal_evidence["actual_new_pre_update"]):
            raise RuntimeError("mission_action_actual_new_pre_update_mismatch")
        mission_observation = terminal_evidence["mission_observation"]
        mission_action_record["terminal_observation"] = mission_observation.to_dict()
        candidate_audit["mission_action"]["terminal_observation"] = mission_observation.to_dict()
        record["mission_action_observation"] = mission_observation.to_dict()
        failed_observation_opportunity = search.record_failed_observation_opportunity(
            mission_action_spec,
            mission_observation,
            decision_id=int(decisions),
        )
        mission_action_record["failed_observation_opportunity"] = failed_observation_opportunity
        candidate_audit["mission_action"]["failed_observation_opportunity"] = failed_observation_opportunity
        record["failed_observation_opportunity"] = failed_observation_opportunity
        shadow_event: Optional[Dict[str, Any]] = None
        if observation_arrival_shadow_enabled:
            evaluation_started_ns = time.perf_counter_ns()
            grid_identity = terminal_visibility_audit.get("grid_identity", {})
            grid_identity = grid_identity if isinstance(grid_identity, dict) else {}
            qualification = grid_identity.get("navigation_qualification")
            evidence_complete = bool(
                terminal_visibility_audit.get("status") == "TERMINAL_COUNTERFACTUAL_READY"
                and qualification == "QUALIFIED_EXACT_PAIR"
            )
            try:
                if shadow_preparation_error is not None or shadow_prediction is None:
                    raise ValueError(shadow_preparation_error or "predicted_opportunity_identity_unavailable")
                shadow_post_update_seen = tuple(sorted(search.observation.seen))
                actual_cell_ids = [list(cell) for cell in sorted(
                    set(shadow_post_update_seen) - set(shadow_pre_update_seen)
                )]
                if int(actual_new_observation_cells) != len(actual_cell_ids):
                    raise ValueError("actual_new_cell_ids_count_mismatch")
                shadow_event = evaluate_observation_arrival_shadow(
                    navigation_reached=position_satisfied_for_action or runner_decision == "BLOCK_ASTAR_DWA_REACHED_GOAL",
                    candidate_id=str(shadow_prediction["candidate_id"]),
                    predicted_new_cell_ids=shadow_prediction["predicted_new_cell_ids"],
                    predicted_new_count=int(shadow_prediction["predicted_new_count"]),
                    actual_new_cell_ids=actual_cell_ids,
                    actual_new_count=int(actual_new_observation_cells),
                    terminal_evidence_complete=evidence_complete,
                    terminal_evidence_qualification=(str(qualification) if qualification is not None else None),
                    terminal_pose_xy_yaw=terminal_pose,
                ).to_dict()
                shadow_event["shadow_evaluation_error"] = None
            except Exception as exc:
                shadow_event = {
                    "candidate_id": shadow_prediction.get("candidate_id") if shadow_prediction else candidate.get("_room_search_audit_candidate_id"),
                    "navigation_reached": position_satisfied_for_action or runner_decision == "BLOCK_ASTAR_DWA_REACHED_GOAL",
                    "predicted_new_count": shadow_prediction.get("predicted_new_count") if shadow_prediction else None,
                    "predicted_new_cell_ids": shadow_prediction.get("predicted_new_cell_ids") if shadow_prediction else None,
                    "actual_new_count": int(actual_new_observation_cells),
                    "actual_new_cell_ids": None,
                    "retained_predicted_cell_ids": None,
                    "unexpected_useful_actual_cell_ids": None,
                    "observation_state": "OBSERVATION_VALIDITY_UNKNOWN",
                    "terminal_evidence_complete": evidence_complete,
                    "terminal_evidence_qualification": qualification,
                    "terminal_pose_xy_yaw": list(terminal_pose),
                    "post_arrival_viability": None,
                    "low_opportunity_candidate": None,
                    "low_opportunity_reason": "BOUNDARY_NOT_CALIBRATED",
                    "authority_enabled": False,
                    "shadow_evaluation_error": "%s:%s" % (type(exc).__name__, exc),
                }
            shadow_event.update({
                "event": "ROOM_SEARCH_OBSERVATION_ARRIVAL_SHADOW",
                "run_id": observation_arrival_shadow_context["run_id"],
                "decision_id": int(decisions),
                "grid_identity": dict(grid_identity),
                "feature_flag_enabled": True,
                "evaluation_wall_ms": (time.perf_counter_ns() - evaluation_started_ns) / 1_000_000.0,
                "next_production_action": "PENDING_EXISTING_PRODUCTION_FLOW",
            })
            observation_arrival_shadow_context["events"].append(shadow_event)
            candidate_audit["observation_arrival_shadow"] = shadow_event
            record["observation_arrival_shadow"] = shadow_event
        try:
            stage_b_shadow_capture.record_execution_outcome(
                decision_id=int(decisions + 1), selected_candidate_id=candidate.get("_room_search_audit_candidate_id"),
                selected_rank=candidate.get("cheap_rank"), runner_final_decision=runner_decision,
                actual_new_observation_cells=actual_new_observation_cells,
            )
        except Exception:
            pass
        guard_fired = record_nonproductive_cycle("EXECUTED_CANDIDATE", actual_new_observation_cells)
        marginal_epoch_reset = None
        if bool(progress_guard.get("substantive_progress")):
            marginal_epoch_reset = search.reset_marginal_completion_epoch(
                decision_index=int(decisions),
                run_id=str(os.environ.get("STATE_MACHINE_RUN_ID", "")),
                actual_new_observation_cells=actual_new_observation_cells,
            )
        record["marginal_epoch_reset"] = marginal_epoch_reset
        candidate_audit["marginal_epoch_reset"] = marginal_epoch_reset
        terminal_visibility_audit = room_search_v2_finalize_terminal_visibility_counterfactual(
            terminal_visibility_audit, actual_new_observation_cells,
        )
        record.update({"actual_terminal_pose_xy_yaw": list(terminal_pose), "actual_breadcrumb_added": breadcrumb_added,
                       "actual_new_observation_cells": actual_new_observation_cells,
                       "terminal_visibility_audit": terminal_visibility_audit,
                       "terminal_target_error_m": math.hypot(float(target_xy[0]) - terminal_pose[0], float(target_xy[1]) - terminal_pose[1])})
        # Audit-only terminal binding: this is appended with the candidate record
        # and is intentionally not read by selection, recovery, or control code.
        candidate_audit["terminal_record"] = {
            "terminal_event_ros_stamp_sec": float(rospy.Time.now().to_sec()),
            "terminal_reason": runner_decision,
            "gated_odom_stamp_sec": terminal_odom.get("stamp_sec"),
            "gated_odom_pose_xy_yaw": list(terminal_pose),
            "target_xy_team_livox_odom": [float(target_xy[0]), float(target_xy[1])],
            "target_error_m": float(record["terminal_target_error_m"]),
        }
        recoverability_shadow["terminal_accuracy"] = terminal_accuracy_telemetry(
            recoverability_terminal, terminal_pose, runner_decision,
        )
        record["recoverability_shadow"] = recoverability_shadow
        if position_satisfied_for_action or runner_decision == "BLOCK_ASTAR_DWA_REACHED_GOAL":
            viability = room_search_v2_postarrival_viability(args, terminal_context, terminal_odom, attempt["runner"])
            treatment = room_search_v2_postarrival_control(str(viability["viability_result"]))
            if shadow_event is not None:
                shadow_event["post_arrival_viability"] = viability.get("viability_result")
                shadow_event["next_production_action"] = (
                    "EXISTING_NORMAL_ROOM_SEARCH_CONTINUATION"
                    if treatment.get("normal_continuation")
                    else "EXISTING_CONSTRAINED_ARRIVAL_FLOW"
                )
            record.update({
                "post_arrival_viability": viability,
                "post_arrival_control": treatment,
                "observation_retained_despite_viability": bool(treatment["preserve_observation"]),
            })
            if not treatment["normal_continuation"]:
                record.update({"accepted": False, "failure_mechanism": str(viability["viability_result"])})
                topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
                safe_stop = (
                    publish_zero_to_topic(topic, 0.0, 10.0)
                    if args.execute else {"topic": topic, "zero_count": 0, "reason": "dry_run_no_cmd_published"}
                )
                record["post_arrival_safe_stop"] = safe_stop
                completion_contract = room_search_v2_completion_contract("CONSTRAINED_ARRIVAL", None)
                recovery_handoff = (
                    room_search_v2_constrained_arrival_recovery_handoff(
                        str(viability["viability_result"]), terminal_pose,
                        getattr(anchor, "door_return_anchor_xy_yaw", ()), float(args.goal_tolerance_m),
                    )
                    if str(viability["viability_result"]) == "CONSTRAINED_ARRIVAL"
                    else {"action": "PRESERVE_EXISTING_ROOM_RETURN", "reason": "NOT_A_CONSTRAINED_ARRIVAL"}
                )
                record["constrained_arrival_recovery_handoff"] = recovery_handoff
                if recovery_handoff["action"] == "FAIL_CLOSED_RECOVERY_INPUT_UNAVAILABLE":
                    record["constrained_arrival_recovery_state"] = "RECOVERY_INPUT_UNAVAILABLE"
                    candidate_audit.update({
                        "decision_outcome": "CONSTRAINED_ARRIVAL_RECOVERY_NOT_VIABLE",
                        "terminal_visibility_audit": terminal_visibility_audit,
                        "post_execution": {
                            "runner_final_decision": runner_decision,
                            "actual_terminal_pose_xy_yaw": list(terminal_pose),
                            "post_arrival_viability": viability.get("viability_result"),
                            "constrained_arrival_recovery": recovery_handoff,
                            "strategic_reposition_state": record["constrained_arrival_recovery_state"],
                        },
                    })
                    append_room_search_candidate_audit(candidate_audit)
                    return finish_room_search_incomplete(
                        decisions, completion_contract, record, candidate=candidate,
                        preflight_attempt_count=preflight_attempt_count,
                        request_room_return=False,
                        strategic_reposition_state=str(record["constrained_arrival_recovery_state"]),
                        post_arrival_viability=viability,
                    )
                if recovery_handoff["action"] == "ATTEMPT_EXISTING_STRATEGIC_REPOSITION_ONCE":
                    reposition = attempt_strategic_reposition(candidate, "CONSTRAINED_ARRIVAL")
                    record["strategic_reposition"] = reposition
                    if reposition.get("successful"):
                        recovered_odom = reposition.get("actual_terminal_odom")
                        recovered_runner = reposition.get("runner")
                        recovered_context = room_search_v2_planning_context(args, grid_status_pairs)
                        if isinstance(recovered_odom, dict) and isinstance(recovered_runner, dict):
                            recovered_viability = room_search_v2_postarrival_viability(
                                args, recovered_context, recovered_odom, recovered_runner,
                            )
                        else:
                            recovered_viability = {
                                "viability_result": "POST_ARRIVAL_VIABILITY_INPUT_UNAVAILABLE",
                                "failure_reason": "STRATEGIC_REPOSITION_TERMINAL_INPUT_UNAVAILABLE",
                            }
                        record["post_reposition_viability"] = recovered_viability
                        record["post_reposition_navigation_context"] = room_search_v2_context_record(recovered_context)
                        if recovered_viability.get("viability_result") == "POST_ARRIVAL_VIABLE":
                            record["constrained_arrival_recovery_state"] = "REPOSITION_SUCCEEDED_POST_ARRIVAL_VIABLE"
                            candidate_audit.update({
                                "decision_outcome": "CONSTRAINED_ARRIVAL_RECOVERED_FOR_ROOM_RETURN",
                                "terminal_visibility_audit": terminal_visibility_audit,
                                "post_execution": {
                                    "runner_final_decision": runner_decision,
                                    "actual_terminal_pose_xy_yaw": list(terminal_pose),
                                    "post_arrival_viability": viability.get("viability_result"),
                                    "constrained_arrival_recovery": recovery_handoff,
                                    "post_reposition_viability": recovered_viability.get("viability_result"),
                                },
                            })
                            append_room_search_candidate_audit(candidate_audit)
                            result["search_attempts"].append(record)
                            online["search_completed"] = False
                            online["room_return_started"] = False
                            result["search_completion_reason"] = str(viability["viability_result"])
                            result["mission_completion"] = completion_contract
                            result["search_aborted_incomplete"] = False
                            result["gain_history"] = search.gain_history
                            result["final_decision"] = str(viability["viability_result"])
                            persist_diagnostic(
                                decisions, candidate=candidate, preflight_attempt_count=preflight_attempt_count,
                                execution_result=runner_decision,
                                actual_displacement_m=math.hypot(terminal_pose[0] - pose[0], terminal_pose[1] - pose[1]),
                                reposition_state=str(reposition.get("state")),
                                post_arrival_viability=recovered_viability,
                                completion_contract=completion_contract,
                            )
                            write_stage("room_search", "navigation_state_machine.py", {
                                "mode": "CONSTRAINED_ARRIVAL_RECOVERED", "result": recovered_viability["viability_result"],
                            })
                            return request_room_return_exit(completion_contract)
                        record["constrained_arrival_recovery_state"] = "REPOSITION_SUCCEEDED_STILL_NOT_VIABLE"
                    else:
                        record["constrained_arrival_recovery_state"] = "REPOSITION_UNAVAILABLE"
                    candidate_audit.update({
                        "decision_outcome": "CONSTRAINED_ARRIVAL_RECOVERY_NOT_VIABLE",
                        "terminal_visibility_audit": terminal_visibility_audit,
                        "post_execution": {
                            "runner_final_decision": runner_decision,
                            "actual_terminal_pose_xy_yaw": list(terminal_pose),
                            "post_arrival_viability": viability.get("viability_result"),
                            "constrained_arrival_recovery": recovery_handoff,
                            "strategic_reposition_state": record["constrained_arrival_recovery_state"],
                            "post_reposition_viability": (record.get("post_reposition_viability") or {}).get("viability_result"),
                        },
                    })
                    append_room_search_candidate_audit(candidate_audit)
                    return finish_room_search_incomplete(
                        decisions, completion_contract, record, candidate=candidate,
                        preflight_attempt_count=preflight_attempt_count,
                        request_room_return=False,
                        strategic_reposition_state=str(record["constrained_arrival_recovery_state"]),
                        post_arrival_viability=record.get("post_reposition_viability") or viability,
                    )
                candidate_audit.update({
                    "decision_outcome": "CONSTRAINED_ARRIVAL",
                    "terminal_visibility_audit": terminal_visibility_audit,
                    "post_execution": {
                        "runner_final_decision": runner_decision,
                        "actual_terminal_pose_xy_yaw": list(terminal_pose),
                        "actual_new_observation_cells": record.get("actual_new_observation_cells"),
                        "post_arrival_viability": viability.get("viability_result"),
                    },
                })
                append_room_search_candidate_audit(candidate_audit)
                result["search_attempts"].append(record)
                online["search_completed"] = False
                online["room_return_started"] = False
                result["search_completion_reason"] = str(viability["viability_result"])
                result["mission_completion"] = completion_contract
                result["search_aborted_incomplete"] = False
                result["gain_history"] = search.gain_history
                result["final_decision"] = str(viability["viability_result"])
                persist_diagnostic(
                    decisions, candidate=candidate, preflight_attempt_count=preflight_attempt_count,
                    execution_result=runner_decision,
                    actual_displacement_m=math.hypot(terminal_pose[0] - pose[0], terminal_pose[1] - pose[1]),
                    post_arrival_viability=viability,
                )
                write_stage("room_search", "navigation_state_machine.py", {
                    "mode": "CONSTRAINED_ARRIVAL", "result": viability["viability_result"],
                })
                return request_room_return_exit(completion_contract)
            record["accepted"] = True
            online["waypoints_reached"] = int(online["waypoints_reached"]) + 1
            # ``safe_actual_poses`` are recovery anchors.  A historical
            # breadcrumb may record an arrival, but only a viable arrival may
            # become a future safe transition anchor.
            search.record_safe_actual_pose(terminal_pose, float(args.goal_tolerance_m))
            persist_diagnostic(decisions, candidate=candidate, preflight_attempt_count=preflight_attempt_count,
                               execution_result=runner_decision, actual_displacement_m=math.hypot(terminal_pose[0] - pose[0], terminal_pose[1] - pose[1]),
                               post_arrival_viability=viability)
        else:
            if shadow_event is not None:
                shadow_event["post_arrival_viability"] = "NOT_EVALUATED_RUNNER_NOT_REACHED_GOAL"
                shadow_event["next_production_action"] = "EXISTING_RUNNER_FAILURE_FLOW"
            search.reject_context_target(target_xy)
            record.update({"accepted": False, "failure_mechanism": runner_decision or "RUNNER_SUMMARY_MISSING"})
            reposition = (
                attempt_strategic_reposition(candidate, runner_decision)
                if strategic_reposition_needed(runner_decision) else {"state": "NOT_TRIGGERED", "successful": False, "preflight_attempt_count": 0}
            )
            record["strategic_reposition"] = reposition
            record["continuation_after_reposition_failure"] = "REGENERATE_CANDIDATES_THEN_TERMINAL_EXPAND" if not reposition.get("successful") else None
            persist_diagnostic(decisions, candidate=candidate, preflight_attempt_count=preflight_attempt_count,
                               execution_result=runner_decision, actual_displacement_m=math.hypot(terminal_pose[0] - pose[0], terminal_pose[1] - pose[1]),
                               reposition_state=str(reposition.get("state")))
        candidate_audit.update({
            "decision_outcome": "EXECUTED_CANDIDATE",
            "terminal_visibility_audit": terminal_visibility_audit,
            "post_execution": {
                "runner_final_decision": runner_decision,
                "actual_terminal_pose_xy_yaw": list(terminal_pose),
                "actual_new_observation_cells": record.get("actual_new_observation_cells"),
                "post_arrival_viability": (
                    (record.get("post_arrival_viability") or {}).get("viability_result")
                    if isinstance(record.get("post_arrival_viability"), dict) else "NOT_EVALUATED_RUNNER_NOT_REACHED_GOAL"
                ),
            },
        })
        append_room_search_candidate_audit(candidate_audit)
        result["search_attempts"].append(record)
        if guard_fired:
            break
        if record.get("strategic_reposition", {}).get("successful"):
            continue

    if rospy.is_shutdown():
        completion_contract = room_search_v2_completion_contract("ROS_SHUTDOWN", None)
        return finish_room_search_incomplete(
            decisions, completion_contract, {"candidate": None, "reason": "ROS_SHUTDOWN"},
        )
    if mission_completion is None and bool(progress_guard.get("finite_abort")):
        completion_reason = "ANTI_INFINITE_GUARD"
        completion_contract = room_search_v2_completion_contract("NONPRODUCTIVE_PROGRESS_GUARD_EXHAUSTED", completion_reason)
        return finish_room_search_incomplete(decisions, completion_contract, {
            "candidate": None,
            "reason": completion_reason,
            "nonproductive_progress_guard": result.get("nonproductive_progress_guard"),
        })
    if mission_completion is None or not mission_completion["mission_complete"]:
        completion_contract = room_search_v2_completion_contract("ROOM_SEARCH_LOOP_EXIT_UNCLASSIFIED", None)
        return finish_room_search_incomplete(
            decisions, completion_contract, {"candidate": None, "reason": "ROOM_SEARCH_LOOP_EXIT_UNCLASSIFIED"},
        )
    online["search_completed"] = True
    result["search_completion_reason"] = mission_completion["mission_completion_reason"]
    result["mission_completion"] = mission_completion
    result["gain_history"] = search.gain_history

    return request_room_return_exit(mission_completion)


def publish_zero_and_backoff(args: argparse.Namespace) -> Dict[str, Any]:
    topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
    pub = rospy.Publisher(topic, Twist, queue_size=2)
    rospy.sleep(0.2)
    rate = rospy.Rate(args.recovery_cmd_rate_hz)
    zero = Twist()
    zero_count = 0
    for _ in range(args.recovery_zero_count):
        pub.publish(zero)
        zero_count += 1
        rate.sleep()
    cmd = Twist()
    cmd.linear.x = -abs(float(args.recovery_backoff_linear_x))
    start = time.monotonic()
    backoff_count = 0
    while not rospy.is_shutdown() and time.monotonic() - start < args.recovery_backoff_sec:
        if args.execute:
            pub.publish(cmd)
            backoff_count += 1
        rate.sleep()
    for _ in range(args.recovery_zero_count):
        pub.publish(zero)
        zero_count += 1
        rate.sleep()
    time.sleep(args.recovery_resample_wait_sec)
    return {
        "topic": topic,
        "zero_count": zero_count,
        "backoff_count": backoff_count,
        "backoff_linear_x": -abs(float(args.recovery_backoff_linear_x)),
        "backoff_duration_sec": args.recovery_backoff_sec,
    }


def publish_stop_at_door(args: argparse.Namespace) -> Dict[str, Any]:
    topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
    pub = rospy.Publisher(topic, Twist, queue_size=2)
    rospy.sleep(0.1)
    rate_hz = max(1.0, float(args.room_side_gap_turn_cmd_rate_hz))
    rate = rospy.Rate(rate_hz)
    zero = Twist()
    start_wall = time.monotonic()
    start_sim = rospy.Time.now()
    count = 0
    sim_elapsed = 0.0
    wall_limit = max(2.0, float(args.room_side_gap_stop_before_turn_sec) * 20.0)
    while not rospy.is_shutdown():
        now_sim = rospy.Time.now()
        sim_elapsed = max(0.0, float((now_sim - start_sim).to_sec()))
        wall_elapsed = time.monotonic() - start_wall
        if sim_elapsed >= float(args.room_side_gap_stop_before_turn_sec) or wall_elapsed >= wall_limit:
            break
        pub.publish(zero)
        count += 1
        rate.sleep()
    pub.publish(zero)
    count += 1
    wall_duration = time.monotonic() - start_wall
    return {
        "topic": topic,
        "zero_count": count,
        "target_stop_sim_duration_sec": float(args.room_side_gap_stop_before_turn_sec),
        "sim_duration_sec": sim_elapsed,
        "wall_duration_sec": wall_duration,
        "average_publish_rate_wall_hz": float(count) / wall_duration if wall_duration > 1e-6 else None,
        "average_publish_rate_sim_hz": float(count) / sim_elapsed if sim_elapsed > 1e-6 else None,
    }


def rostopic_publisher_count(topic: str, timeout_sec: float = 1.0) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["rostopic", "info", topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception as exc:
        return {"topic": topic, "publisher_count": None, "reason": f"rostopic_info_failed:{exc}"}
    if result.returncode != 0:
        return {
            "topic": topic,
            "publisher_count": None,
            "reason": "rostopic_info_nonzero",
            "stderr_tail": result.stderr[-500:],
        }
    in_publishers = False
    count = 0
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Publishers:"):
            in_publishers = True
            continue
        if stripped.startswith("Subscribers:"):
            in_publishers = False
        if in_publishers and stripped.startswith("*"):
            count += 1
    return {"topic": topic, "publisher_count": count, "stdout_tail": result.stdout[-1000:]}


def publish_zero_to_topic(topic: str, duration_sec: float, rate_hz: float) -> Dict[str, Any]:
    pub = rospy.Publisher(topic, Twist, queue_size=2)
    rospy.sleep(0.1)
    rate = rospy.Rate(max(1.0, float(rate_hz)))
    zero = Twist()
    start_sim = rospy.Time.now()
    start_wall = time.monotonic()
    count = 0
    sim_elapsed = 0.0
    while not rospy.is_shutdown():
        sim_elapsed = max(0.0, float((rospy.Time.now() - start_sim).to_sec()))
        if sim_elapsed >= float(duration_sec):
            break
        pub.publish(zero)
        count += 1
        rate.sleep()
    pub.publish(zero)
    count += 1
    wall_duration = time.monotonic() - start_wall
    return {
        "topic": topic,
        "zero_count": count,
        "sim_duration_sec": sim_elapsed,
        "wall_duration_sec": wall_duration,
    }


def publish_fast_debug_stop(args: argparse.Namespace) -> Dict[str, Any]:
    """Publish a short zero command burst before a debug-only early exit."""
    topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
    result: Dict[str, Any] = {
        "topic": topic,
        "execute": bool(args.execute),
        "zero_count": 0,
        "clock": "wall_time_short_burst",
    }
    if not args.execute:
        result["reason"] = "dry_run_no_cmd_published"
        return result
    pub = rospy.Publisher(topic, Twist, queue_size=2)
    time.sleep(0.05)
    zero = Twist()
    for _ in range(3):
        pub.publish(zero)
        result["zero_count"] += 1
        time.sleep(0.03)
    return result


def cli_option_was_provided(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in argv)


def apply_fast_debug_profile(args: argparse.Namespace, argv: Sequence[str]) -> None:
    """Apply debug-only defaults without overriding explicit user CLI values."""
    if not args.fast_debug:
        return
    if not cli_option_was_provided(argv, "--runner-runtime-sec"):
        args.runner_runtime_sec = 8.0
    if not cli_option_was_provided(argv, "--runner-wall-watchdog-sec"):
        args.runner_wall_watchdog_sec = 60.0
    if not cli_option_was_provided(argv, "--command-timeout-margin-sec"):
        args.command_timeout_margin_sec = 15.0
    if args.debug_max_iterations is None and not cli_option_was_provided(argv, "--debug-max-iterations"):
        args.debug_max_iterations = 8


def _publish_room_side_turn_legacy(args: argparse.Namespace, side: str) -> Dict[str, Any]:
    direct_cmd_mode = bool(args.room_side_gap_turn_direct_cmd)
    topic = args.cmd_topic if direct_cmd_mode else (args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic)
    cmd_vel_publishers_before = rostopic_publisher_count(args.cmd_topic)
    raw_zero_before_turn = None
    if direct_cmd_mode and args.use_imu_velocity_follower:
        raw_zero_before_turn = publish_zero_to_topic(
            args.follower_raw_cmd_topic,
            args.room_side_gap_turn_raw_zero_before_direct_sec,
            args.room_side_gap_turn_cmd_rate_hz,
        )
    pub = rospy.Publisher(topic, Twist, queue_size=2)
    rospy.sleep(0.2)
    rate_hz = max(1.0, float(args.room_side_gap_turn_cmd_rate_hz))
    rate = rospy.Rate(rate_hz)
    latest_pose: Dict[str, Optional[Tuple[float, float, float]]] = {"pose": None}

    def on_odom(msg: Odometry) -> None:
        pose = msg.pose.pose
        latest_pose["pose"] = (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quat(pose.orientation),
        )

    sub = rospy.Subscriber(ODOM_TOPIC, Odometry, on_odom, queue_size=20)

    def cached_pose(timeout_sec: float = 0.5) -> Tuple[float, float, float]:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while latest_pose["pose"] is None and not rospy.is_shutdown() and time.monotonic() < deadline:
            time.sleep(0.01)
        if latest_pose["pose"] is not None:
            return latest_pose["pose"]  # type: ignore[return-value]
        return pose_tuple(read_odom(timeout_sec=timeout_sec))

    zero = Twist()
    zero_count = 0
    for _ in range(args.recovery_zero_count):
        pub.publish(zero)
        zero_count += 1
        rate.sleep()
    cmd = Twist()
    sign = 1.0 if side == "left" else -1.0
    angular_z = (
        float(args.room_side_gap_turn_direct_angular_z)
        if direct_cmd_mode
        else float(args.room_side_gap_turn_angular_z)
    )
    cmd.angular.z = sign * abs(angular_z)
    initial_pose = cached_pose()
    initial_yaw = float(initial_pose[2])
    start_wall = time.monotonic()
    start_sim = rospy.Time.now()
    turn_count = 0
    final_yaw = initial_yaw
    yaw_delta = 0.0
    sim_elapsed = 0.0
    try:
        while not rospy.is_shutdown():
            now_sim = rospy.Time.now()
            sim_elapsed = max(0.0, float((now_sim - start_sim).to_sec()))
            if sim_elapsed >= float(args.room_side_gap_turn_duration_sec):
                break
            if args.execute:
                pub.publish(cmd)
                turn_count += 1
            pose = latest_pose["pose"]
            if pose is not None:
                final_yaw = float(pose[2])
                yaw_delta = normalize_angle(final_yaw - initial_yaw)
                if abs(yaw_delta) >= float(args.room_side_gap_turn_target_yaw_rad):
                    break
            rate.sleep()
    finally:
        try:
            sub.unregister()
        except Exception:
            pass
    if latest_pose["pose"] is not None:
        final_yaw = float(latest_pose["pose"][2])
    yaw_delta = normalize_angle(final_yaw - initial_yaw)
    wall_duration = time.monotonic() - start_wall
    sim_duration = max(0.0, float((rospy.Time.now() - start_sim).to_sec()))
    for _ in range(args.recovery_zero_count):
        pub.publish(zero)
        zero_count += 1
        rate.sleep()
    cmd_vel_publishers_after = rostopic_publisher_count(args.cmd_topic)
    return {
        "topic": topic,
        "direct_cmd_mode": direct_cmd_mode,
        "cmd_vel_publishers_before": cmd_vel_publishers_before,
        "cmd_vel_publishers_after": cmd_vel_publishers_after,
        "raw_zero_before_turn": raw_zero_before_turn,
        "door_side": side,
        "turn_count": turn_count,
        "zero_count": zero_count,
        "angular_z": cmd.angular.z,
        "duration_sec": float(args.room_side_gap_turn_duration_sec),
        "wall_duration_sec": wall_duration,
        "sim_duration_sec": sim_duration,
        "average_publish_rate_wall_hz": float(turn_count) / wall_duration if wall_duration > 1e-6 else None,
        "average_publish_rate_sim_hz": float(turn_count) / sim_duration if sim_duration > 1e-6 else None,
        "initial_yaw_rad": initial_yaw,
        "final_yaw_rad": final_yaw,
        "actual_yaw_delta_rad": yaw_delta,
        "actual_abs_yaw_delta_rad": abs(yaw_delta),
        "target_yaw_delta_rad": float(args.room_side_gap_turn_target_yaw_rad),
        "min_required_yaw_delta_rad": float(args.room_side_gap_turn_min_yaw_rad),
        "turn_yaw_sufficient": abs(yaw_delta) >= float(args.room_side_gap_turn_min_yaw_rad),
    }


def publish_room_side_turn(
    args: argparse.Namespace,
    side: str,
    health_cache: RoomSideTurnHealthCache,
) -> Dict[str, Any]:
    """Closed-loop zero-linear turn using only fresh gated-odom yaw feedback."""
    topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
    result: Dict[str, Any] = {
        "validation_version": "room_side_turn_validation_v1",
        "execute": bool(args.execute),
        "topic": topic,
        "door_side": side,
        "linear_x_invariant": 0.0,
        "legacy_direct_cmd_requested": bool(args.room_side_gap_turn_direct_cmd),
        "legacy_direct_cmd_used": False,
        "feedback_source": ODOM_TOPIC,
        "imu_role": "cross_diagnostic_only",
        "timeline": [],
        "turn_count": 0,
        "zero_count": 0,
        "stop_reason": None,
    }
    pub = rospy.Publisher(topic, Twist, queue_size=5)
    time.sleep(0.1)
    zero = Twist()

    def publish_zero_once() -> None:
        if args.execute:
            pub.publish(zero)
            result["zero_count"] += 1

    initial_health = health_cache.snapshot()
    result["startup_health"] = initial_health
    if not initial_health.get("ready"):
        for _ in range(max(1, int(args.recovery_zero_count))):
            publish_zero_once()
            time.sleep(0.02)
        result.update(
            {
                "stop_reason": "startup_health_gate_blocked",
                "health_blocking_reason": initial_health.get("primary_blocking_reason"),
                "target_reached": False,
                "turn_yaw_sufficient": False,
                "actual_yaw_delta_rad": 0.0,
                "actual_abs_yaw_delta_rad": 0.0,
            }
        )
        return result

    angular = float(args.room_side_gap_turn_angular_z)
    core = FeedbackYawTurnCore(
        side=side,
        target_yaw_rad=float(args.room_side_gap_turn_target_yaw_rad),
        min_yaw_rad=float(args.room_side_gap_turn_min_yaw_rad),
        angular_z=abs(angular),
        watchdog_sec=float(args.room_side_turn_wall_watchdog_sec),
        no_response_sample_limit=max(1, int(args.room_side_turn_no_response_samples)),
        yaw_response_epsilon_rad=max(0.0, float(args.room_side_turn_yaw_response_epsilon_rad)),
    )
    start_wall = time.monotonic()
    start_sim = rospy.Time.now()
    previous_odom_count = -1
    last_control = core._result(0.0, 0.0, waiting_reason="waiting_for_first_fresh_odom")
    rate_sleep = 1.0 / max(1.0, float(args.room_side_gap_turn_cmd_rate_hz))
    try:
        while not rospy.is_shutdown():
            now_wall = time.monotonic()
            now_sim = rospy.Time.now()
            wall_elapsed = now_wall - start_wall
            sim_elapsed = max(0.0, float((now_sim - start_sim).to_sec()))
            health = health_cache.snapshot()
            odom = (health.get("streams") or {}).get("gated_odom", {})
            odom_count = int(odom.get("count") or 0)
            odom_state = str(odom.get("health_state") or "LOST")
            odom_payload = odom.get("payload") if isinstance(odom.get("payload"), dict) else {}
            yaw = odom_payload.get("yaw_rad")

            hard_block = next(
                (
                    value
                    for value in health.get("blocking_reasons", [])
                    if value.startswith("rl_")
                    or value.startswith("follower_")
                    or value.startswith("cmd_vel_output_")
                    or value.startswith("clock_not_advancing_lost")
                ),
                None,
            )
            if hard_block is not None:
                core.stopped = True
                core.stop_reason = "health_lost_during_turn:" + str(hard_block)
                last_control = core._result(0.0, 0.0)
            elif odom_count != previous_odom_count or odom_state != "FRESH":
                last_control = core.update(
                    odom_state=odom_state,
                    yaw_rad=float(yaw) if finite_number(yaw) else None,
                    wall_elapsed_sec=wall_elapsed,
                )
                previous_odom_count = odom_count
            elif wall_elapsed >= float(args.room_side_turn_wall_watchdog_sec):
                last_control = core.update(
                    odom_state=odom_state,
                    yaw_rad=None,
                    wall_elapsed_sec=wall_elapsed,
                )

            raw_cmd = Twist()
            raw_cmd.linear.x = 0.0
            raw_cmd.angular.z = float(last_control.get("angular_z") or 0.0)
            if odom_state != "FRESH" or core.stopped:
                raw_cmd.angular.z = 0.0
            if args.execute:
                pub.publish(raw_cmd)
                if abs(raw_cmd.angular.z) > 1e-9:
                    result["turn_count"] += 1
                else:
                    result["zero_count"] += 1

            output = ((health.get("streams") or {}).get("cmd_vel_output") or {}).get("payload") or {}
            imu = ((health.get("streams") or {}).get("imu") or {}).get("payload") or {}
            result["timeline"].append(
                {
                    "wall_elapsed_sec": wall_elapsed,
                    "sim_elapsed_sec": sim_elapsed,
                    "ros_time_sec": float(now_sim.to_sec()),
                    "rtf": sim_elapsed / wall_elapsed if wall_elapsed > 1e-6 else None,
                    "raw_command": {"linear_x": 0.0, "angular_z": float(raw_cmd.angular.z)},
                    "follower_output_command": {
                        "linear_x": output.get("linear_x"),
                        "angular_z": output.get("angular_z"),
                    },
                    "gated_odom_yaw_rad": yaw,
                    "imu_yaw_rad": imu.get("yaw_rad"),
                    "imu_angular_velocity_z": imu.get("angular_velocity_z"),
                    "odom_health_state": odom_state,
                    "odom_age_wall_sec": odom.get("age_wall_sec"),
                    "accumulated_yaw_rad": last_control.get("accumulated_yaw_rad"),
                    "directed_yaw_rad": last_control.get("directed_yaw_rad"),
                    "no_response_samples": last_control.get("no_response_samples"),
                    "stop_reason": last_control.get("stop_reason"),
                }
            )
            if core.stopped:
                break
            time.sleep(rate_sleep)
    finally:
        for _ in range(max(1, int(args.recovery_zero_count))):
            publish_zero_once()
            time.sleep(min(0.05, rate_sleep))

    wall_duration = time.monotonic() - start_wall
    sim_duration = max(0.0, float((rospy.Time.now() - start_sim).to_sec()))
    final = core._result(0.0, 0.0)
    result.update(
        {
            "stop_reason": core.stop_reason or "ros_shutdown",
            "wall_duration_sec": wall_duration,
            "sim_duration_sec": sim_duration,
            "rtf": sim_duration / wall_duration if wall_duration > 1e-6 else None,
            "actual_yaw_delta_rad": float(core.accumulated_yaw_rad),
            "actual_abs_yaw_delta_rad": abs(float(core.accumulated_yaw_rad)),
            "directed_yaw_delta_rad": final.get("directed_yaw_rad"),
            "target_yaw_delta_rad": float(args.room_side_gap_turn_target_yaw_rad),
            "min_required_yaw_delta_rad": float(args.room_side_gap_turn_min_yaw_rad),
            "target_reached": bool(final.get("target_reached")),
            "turn_yaw_sufficient": bool(final.get("minimum_yaw_sufficient")),
            "final_health": health_cache.snapshot(),
        }
    )
    return result


def wait_for_post_turn_observation(
    args: argparse.Namespace,
    health_cache: RoomSideTurnHealthCache,
    grid_count_before: int,
) -> Dict[str, Any]:
    """Hold zero until fresh grids and a same-run perception update are available."""
    topic = args.follower_raw_cmd_topic if args.use_imu_velocity_follower else args.cmd_topic
    pub = rospy.Publisher(topic, Twist, queue_size=5)
    zero = Twist()
    start_wall = time.monotonic()
    start_sim = rospy.Time.now()
    timeline: List[Dict[str, Any]] = []
    last_doorway: Dict[str, Any] = {}
    last_gap: Dict[str, Any] = {"available": False, "reason": "not_checked"}
    stop_reason = "ros_shutdown_during_post_turn_observation"
    required = max(1, int(args.room_side_turn_post_fresh_grid_count))
    while not rospy.is_shutdown():
        wall_elapsed = time.monotonic() - start_wall
        if wall_elapsed >= float(args.room_side_turn_post_grid_watchdog_sec):
            stop_reason = "post_turn_grid_watchdog"
            break
        if args.execute:
            pub.publish(zero)
        health = health_cache.snapshot()
        grid = (health.get("streams") or {}).get("local_grid", {})
        grid_count = int(grid.get("count") or 0)
        new_count = max(0, grid_count - int(grid_count_before))
        status = ((health.get("streams") or {}).get("grid_status") or {}).get("payload") or {}
        grid_safe = bool(status.get("safe_for_navigation"))
        upstream_fresh = bool((status.get("input_freshness") or {}).get("all_required_inputs_fresh"))
        last_doorway = read_json(DOORWAY_PATH)
        last_gap = room_side_gap_candidate(last_doorway, args)
        profile_status = doorway_profile_opening_status(last_doorway)
        doorway_grid_stamp = last_doorway.get("grid_stamp_sec")
        grid_ros_stamp = grid.get("last_ros_stamp_sec")
        perception_fresh = bool(
            finite_number(doorway_grid_stamp)
            and finite_number(grid_ros_stamp)
            and float(doorway_grid_stamp) >= float(grid_ros_stamp) - 1e-6
        )
        observation_available = bool(last_gap.get("available") or profile_status.get("available"))
        timeline.append(
            {
                "wall_elapsed_sec": wall_elapsed,
                "sim_elapsed_sec": max(0.0, float((rospy.Time.now() - start_sim).to_sec())),
                "grid_count_after_turn": new_count,
                "grid_health_state": grid.get("health_state"),
                "grid_safe_for_navigation": grid_safe,
                "grid_upstream_fresh": upstream_fresh,
                "doorway_grid_stamp_sec": doorway_grid_stamp,
                "latest_grid_stamp_sec": grid_ros_stamp,
                "perception_fresh": perception_fresh,
                "side_observation_available": observation_available,
            }
        )
        if new_count >= required and grid.get("health_state") == "FRESH" and grid_safe and upstream_fresh and perception_fresh and observation_available:
            stop_reason = "fresh_post_turn_observation_ready"
            break
        time.sleep(0.1)
    if args.execute:
        for _ in range(max(1, int(args.recovery_zero_count))):
            pub.publish(zero)
            time.sleep(0.02)
    final_health = health_cache.snapshot()
    grid = (final_health.get("streams") or {}).get("local_grid", {})
    status = ((final_health.get("streams") or {}).get("grid_status") or {}).get("payload") or {}
    fresh_count = max(0, int(grid.get("count") or 0) - int(grid_count_before))
    profile_status = doorway_profile_opening_status(last_doorway)
    observation_available = bool(last_gap.get("available") or profile_status.get("available"))
    return {
        "stop_reason": stop_reason,
        "required_fresh_grid_count": required,
        "fresh_grid_count": fresh_count,
        "grid_safe_for_navigation": bool(status.get("safe_for_navigation")),
        "grid_upstream_fresh": bool((status.get("input_freshness") or {}).get("all_required_inputs_fresh")),
        "fresh_side_observation": observation_available and stop_reason == "fresh_post_turn_observation_ready",
        "doorway": last_doorway,
        "room_side_gap_candidate": last_gap,
        "timeline": timeline,
        "final_health": final_health,
    }


def start_follower(args: argparse.Namespace) -> Optional[subprocess.Popen]:
    if not args.use_imu_velocity_follower:
        return None
    log_path = DEBUG_DIR / "imu_velocity_follower.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/local_subgoal_runner_mvp/imu_velocity_follower.py",
        f"_raw_cmd_topic:={args.follower_raw_cmd_topic}",
        f"_imu_topic:={args.follower_imu_topic}",
        f"_output_cmd_topic:={args.follower_output_cmd_topic}",
        f"_status_topic:={args.follower_status_topic}",
        f"_max_linear_x:={args.follower_max_linear_x}",
        f"_max_angular_z:={args.follower_max_angular_z}",
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
    setattr(proc, "_codex_log_handle", log)
    time.sleep(1.0)
    return proc


def stop_follower(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
    handle = getattr(proc, "_codex_log_handle", None)
    if handle is not None:
        handle.close()


def write_report(summary: Dict[str, Any]) -> None:
    lines = [
        "# State Machine Navigation Report",
        "",
        f"- final_decision: `{summary.get('final_decision')}`",
        f"- final_state: `{summary.get('final_state')}`",
        f"- execute: `{summary.get('execute')}`",
        f"- completed_iterations: `{summary.get('completed_iterations')}`",
        f"- forbidden_sources_used: `{summary.get('forbidden_sources_used')}`",
        f"- called_move_base: `{summary.get('called_move_base')}`",
        f"- sent_navigation_goal: `{summary.get('sent_navigation_goal')}`",
        "",
        "## State Trace",
    ]
    for item in summary.get("state_trace", []):
        runner = item.get("runner") if isinstance(item.get("runner"), dict) else {}
        target_diag = item.get("target_diagnostics_after_runner") or item.get("target_diagnostics_before_runner") or {}
        center = runner.get("last_corridor_center_target") if isinstance(runner.get("last_corridor_center_target"), dict) else {}
        dwa = runner.get("last_dwa") if isinstance(runner.get("last_dwa"), dict) else {}
        lines.append(
            "- "
            f"{item.get('iteration')}: `{item.get('state')}` -> `{item.get('next_state')}` "
            f"reason=`{item.get('transition_reason')}` "
            f"progress=`{target_diag.get('anchor_progress_m')}` "
            f"lat=`{target_diag.get('anchor_lateral_error_m')}` "
            f"yaw_err=`{target_diag.get('anchor_yaw_error_rad')}` "
            f"target_base=`{target_diag.get('target_base_xy')}` "
            f"disp=`{runner.get('observed_straight_line_displacement_m')}` "
            f"cmd=(`{runner.get('last_cmd_linear_x')}`, `{runner.get('last_cmd_angular_z')}`) "
            f"center_applied=`{center.get('applied')}` "
            f"center_reason=`{center.get('reason')}` "
            f"pc_wall=`{dwa.get('pointcloud_wall_heading_active')}`"
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--enable-room-local-guarded-online-validation",
        action="store_true",
        help="Default-off: forward paired Phase-2/3 guarded authority only to explicit ROOM_LOCAL runner calls.",
    )
    parser.add_argument("--max-iterations", type=int, default=24)
    parser.add_argument("--input-timeout-sec", type=float, default=20.0)
    parser.add_argument("--command-timeout-margin-sec", type=float, default=45.0)
    parser.add_argument("--robot-radius-m", type=float, default=STATIC_PLANNING_FOOTPRINT_RADIUS_M)
    parser.add_argument("--command-slice-sec", type=float, default=0.50)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.30)
    parser.add_argument("--max-linear-x", type=float, default=0.60)
    parser.add_argument("--entry-max-linear-x", type=float, default=0.60)
    parser.add_argument("--corridor-max-linear-x", type=float, default=0.60)
    parser.add_argument("--runner-runtime-sec", type=float, default=35.0)
    parser.add_argument("--runner-wall-watchdog-sec", type=float, default=120.0)
    parser.add_argument("--runner-max-steps", type=int, default=10)
    parser.add_argument(
        "--portal-p-pre-max-steps",
        type=int,
        default=20,
        help="P_pre-only local-runner step budget; P_through remains on its existing budget.",
    )
    parser.add_argument(
        "--portal-p-through-max-steps",
        type=int,
        default=16,
        help="P_through-only local-runner step budget; all other runner states keep --runner-max-steps.",
    )
    parser.add_argument(
        "--p-pre-upstream-tangent-m",
        type=float,
        default=STAIR_MOVING_TURN_P_PRE_TANGENT_M,
        help="Portal G14 P_pre upstream tangent offset in metres; changes only the P_pre target location.",
    )
    parser.add_argument("--fast-debug", action="store_true", help="Apply debug-only short runtime defaults.")
    parser.add_argument("--debug-max-iterations", type=int, default=None)
    parser.add_argument("--stop-after-first-runner", action="store_true")
    parser.add_argument("--stop-after-first-approach-candidate", action="store_true")
    parser.add_argument("--stop-after-first-portal-g14-shadow-target", action="store_true")
    parser.add_argument("--stop-after-portal-g14-p-pre-reached", action="store_true")
    parser.add_argument("--enable-stair-moving-turn-portal-entry", action="store_true")
    parser.add_argument("--enable-hierarchical-portal-local-autonomy", action="store_true")
    parser.add_argument(
        "--enable-room-search-v2",
        action="store_true",
        help="Run bounded ROOM_SEARCH V2-A only after frozen P_through success; disabled by default.",
    )
    parser.add_argument("--room-search-step-m", type=float, default=1.0)
    parser.add_argument("--room-search-sensor-smoke-sec", type=float, default=4.0)
    parser.add_argument(
        "--room-search-extra-clearance-margin-m",
        type=float,
        default=0.05,
        help="ROOM_SEARCH and breadcrumb-return-only occupied-obstacle safety margin in metres.",
    )
    parser.add_argument("--p-through-deep-crossing-min-progress-m", type=float, default=0.50)
    parser.add_argument("--p-through-deep-crossing-max-distance-m", type=float, default=0.50)
    parser.add_argument(
        "--enable-execution-qualified-dwa-primitives",
        action="store_true",
        help="Deprecated compatibility flag; native continuous DWA remains authoritative.",
    )
    parser.add_argument("--stop-after-portal-normal-aligned", action="store_true")
    parser.add_argument("--stop-after-p-through-handoff-ready", action="store_true")
    parser.add_argument("--portal-normal-alignment-max-angular-z", type=float, default=0.45)
    parser.add_argument("--portal-normal-alignment-gain", type=float, default=1.0)
    parser.add_argument("--portal-normal-alignment-rate-hz", type=float, default=10.0)
    parser.add_argument("--portal-normal-alignment-feasibility-min-interval-sec", type=float, default=0.50)
    parser.add_argument("--portal-normal-alignment-required-consecutive-samples", type=int, default=3)
    parser.add_argument("--portal-normal-alignment-no-response-samples", type=int, default=8)
    parser.add_argument("--portal-normal-alignment-yaw-response-epsilon-rad", type=float, default=0.01)
    parser.add_argument("--portal-normal-alignment-max-sim-sec", type=float, default=20.0)
    parser.add_argument("--stop-after-first-door-cue", action="store_true")
    parser.add_argument("--stop-after-first-actionable-door-cue", action="store_true")
    parser.add_argument("--observe-after-actionable-door-cue", action="store_true")
    parser.add_argument("--door-observe-sim-sec", type=float, default=2.0)
    parser.add_argument("--door-observe-wall-watchdog-sec", type=float, default=30.0)
    parser.add_argument("--stop-after-first-side-gap-nav-debug", action="store_true")
    parser.add_argument("--side-gap-confirm-frames", type=int, default=2)
    parser.add_argument("--side-gap-confirm-progress-jump-m", type=float, default=0.8)
    parser.add_argument("--door-nav-approach-offset-m", type=float, default=0.4)
    parser.add_argument("--stop-after-first-side-gap-segment-switch-audit", action="store_true")
    parser.add_argument("--side-gap-switch-threshold-m", type=float, default=None)
    parser.add_argument("--enable-side-gap-visual-coordinate-audit", action="store_true")
    parser.add_argument("--stop-after-first-side-gap-visual-audit", action="store_true")
    parser.add_argument("--side-gap-visual-audit-max-events", type=int, default=10)
    parser.add_argument("--side-gap-visual-audit-broad-only", action="store_true")
    parser.add_argument("--side-gap-visual-audit-rgb-topic", default="/real_sense/rgb/image_raw")
    parser.add_argument("--side-gap-visual-audit-depth-topic", default="/real_sense/depth/image_raw")
    parser.add_argument("--enable-local-free-space-entry-debug", action="store_true")
    parser.add_argument("--stop-after-first-local-entry-target-debug", action="store_true")
    parser.add_argument("--local-entry-max-events", type=int, default=5)
    parser.add_argument("--local-entry-forward-min-m", type=float, default=0.2)
    parser.add_argument("--local-entry-forward-max-m", type=float, default=1.8)
    parser.add_argument("--local-entry-lateral-min-m", type=float, default=0.4)
    parser.add_argument("--local-entry-lateral-max-m", type=float, default=1.8)
    parser.add_argument("--local-entry-min-area-m2", type=float, default=0.25)
    parser.add_argument("--local-entry-min-forward-extent-m", type=float, default=0.4)
    parser.add_argument("--local-entry-min-lateral-extent-m", type=float, default=0.4)
    parser.add_argument("--stop-after-door-landmark-debug", action="store_true")
    parser.add_argument("--debug-run-label", default=None)
    parser.add_argument("--entry-lookahead-m", type=float, default=2.0)
    parser.add_argument("--corridor-lookahead-m", type=float, default=2.0)
    parser.add_argument("--entry-progress-threshold-m", type=float, default=2.2)
    parser.add_argument("--room-zone-start-progress-m", type=float, default=7.85)
    parser.add_argument(
        "--room-entry-observer-log",
        default="",
        help=(
            "Optional JSONL audit sink. Appends completed state-machine iteration snapshots "
            "only; it does not publish, alter targets, or participate in state decisions."
        ),
    )
    parser.add_argument("--enable-forced-room-entry-mvp", action="store_true")
    parser.add_argument("--enable-simple-room-entry", action="store_true")
    parser.add_argument("--simple-room-entry-search-step-m", type=float, default=0.35)
    parser.add_argument("--simple-room-entry-creep-step-m", type=float, default=0.25)
    parser.add_argument("--simple-room-entry-nearfield-max-x-m", type=float, default=0.8)
    parser.add_argument("--simple-room-entry-max-opening-width-m", type=float, default=2.0)
    parser.add_argument("--simple-room-entry-min-nearfield-overlap-m", type=float, default=0.20)
    parser.add_argument("--simple-room-entry-max-missed-frames", type=int, default=2)
    parser.add_argument("--forced-entry-min-opening-width-m", type=float, default=0.8)
    parser.add_argument("--forced-entry-turn-angle-deg", type=float, default=85.0)
    parser.add_argument("--forced-entry-angular-z", type=float, default=0.60)
    parser.add_argument("--forced-entry-linear-x", type=float, default=0.32)
    parser.add_argument("--forced-entry-allow-low-speed", action="store_true")
    parser.add_argument("--forced-entry-forward-duration-sec", type=float, default=3.5)
    parser.add_argument("--forced-entry-stop-after-done", action="store_true", default=True)
    parser.add_argument(
        "--no-forced-entry-stop-after-done",
        dest="forced_entry_stop_after_done",
        action="store_false",
    )
    parser.add_argument("--forced-entry-max-trigger-progress-m", type=float, default=None)
    parser.add_argument("--forced-entry-pre-stop-sec", type=float, default=0.5)
    parser.add_argument("--forced-entry-post-stop-sec", type=float, default=0.5)
    parser.add_argument("--forced-entry-alignment-tolerance-m", type=float, default=0.30)
    parser.add_argument("--forced-entry-alignment-overshoot-m", type=float, default=0.45)
    parser.add_argument("--forced-entry-alignment-max-inside-offset-m", type=float, default=0.60)
    parser.add_argument("--forced-entry-confirm-center-jump-m", type=float, default=0.20)
    parser.add_argument("--forced-entry-confirm-interval-overlap-ratio", type=float, default=0.40)
    parser.add_argument("--forced-entry-trigger-min-side-free-ratio", type=float, default=0.25)
    parser.add_argument("--forced-entry-trigger-min-current-overlap-ratio", type=float, default=0.20)
    parser.add_argument("--forced-entry-side-grid-timeout-sec", type=float, default=20.0)
    parser.add_argument("--forced-entry-front-clearance-min-m", type=float, default=0.65)
    parser.add_argument("--forced-entry-stall-window-sec", type=float, default=0.8)
    parser.add_argument("--forced-entry-stall-min-progress-m", type=float, default=0.05)
    parser.add_argument("--forced-entry-success-min-forward-progress-m", type=float, default=0.65)
    parser.add_argument("--forced-entry-success-min-signed-lateral-m", type=float, default=0.80)
    parser.add_argument(
        "--entry-anchor-snap-heading-to-odom-x",
        dest="entry_anchor_snap_heading_to_odom_x",
        action="store_true",
        default=True,
        help="Snap small initial entry-anchor heading errors to odom x to avoid startup yaw drift accumulating in corridor.",
    )
    parser.add_argument(
        "--no-entry-anchor-snap-heading-to-odom-x",
        dest="entry_anchor_snap_heading_to_odom_x",
        action="store_false",
        help="Use the raw initial target heading for the entry/corridor anchor.",
    )
    parser.add_argument("--entry-anchor-heading-snap-threshold-rad", type=float, default=0.12)
    parser.add_argument("--corridor-axis-evidence-wait-sec", type=float, default=4.0)
    parser.add_argument("--corridor-axis-evidence-max-age-sec", type=float, default=1.0)
    parser.add_argument("--disable-entry-pointcloud-wall-heading", action="store_true", default=True)
    parser.add_argument("--enable-entry-pointcloud-wall-heading", dest="disable_entry_pointcloud_wall_heading", action="store_false")
    parser.add_argument("--disable-corridor-pointcloud-wall-heading", action="store_true", default=True)
    parser.add_argument(
        "--enable-corridor-pointcloud-wall-heading",
        dest="disable_corridor_pointcloud_wall_heading",
        action="store_false",
    )
    parser.add_argument("--disable-enter-room-pointcloud-wall-heading", action="store_true", default=True)
    parser.add_argument(
        "--enable-enter-room-pointcloud-wall-heading",
        dest="disable_enter_room_pointcloud_wall_heading",
        action="store_false",
    )
    parser.add_argument("--entry-min-linear-x", type=float, default=0.12)
    parser.add_argument("--entry-speed-weight", type=float, default=0.35)
    parser.add_argument("--entry-clearance-weight", type=float, default=0.10)
    parser.add_argument("--entry-dwa-predict-time", type=float, default=0.60)
    parser.add_argument("--entry-max-linear-accel", type=float, default=0.80)
    parser.add_argument("--entry-distance-speed-gain", type=float, default=1.00)
    parser.add_argument("--corridor-min-linear-x", type=float, default=0.18)
    parser.add_argument("--corridor-speed-weight", type=float, default=0.35)
    parser.add_argument("--corridor-clearance-weight", type=float, default=0.10)
    parser.add_argument("--corridor-dwa-predict-time", type=float, default=0.60)
    parser.add_argument("--corridor-max-linear-accel", type=float, default=0.80)
    parser.add_argument("--corridor-distance-speed-gain", type=float, default=1.00)
    parser.add_argument("--corridor-doorway-check-wait-sec", type=float, default=0.3)
    parser.add_argument("--doorway-latch-min-forward-m", type=float, default=0.35)
    parser.add_argument("--doorway-latch-alignment-tolerance-m", type=float, default=0.22)
    parser.add_argument("--doorway-latch-overshoot-tolerance-m", type=float, default=0.45)
    parser.add_argument("--doorway-partial-latch-extra-observe-m", type=float, default=0.60)
    parser.add_argument("--doorway-partial-latch-max-extensions", type=int, default=2)
    parser.add_argument("--doorway-partial-latch-required-stable-observations", type=int, default=2)
    parser.add_argument("--doorway-partial-latch-center-stability-tolerance-m", type=float, default=0.25)
    parser.add_argument("--doorway-partial-latch-max-stable-center-x-m", type=float, default=2.20)
    parser.add_argument("--enable-room-side-gap-trigger", action="store_true", default=True)
    parser.add_argument("--disable-room-side-gap-trigger", dest="enable_room_side_gap_trigger", action="store_false")
    parser.add_argument("--room-side-gap-required-stable-observations", type=int, default=2)
    parser.add_argument("--room-side-gap-center-stability-tolerance-m", type=float, default=0.65)
    parser.add_argument("--room-side-gap-width-stability-tolerance-m", type=float, default=1.20)
    parser.add_argument("--room-side-gap-min-center-x-m", type=float, default=0.35)
    parser.add_argument("--room-side-gap-max-center-x-m", type=float, default=1.35)
    parser.add_argument("--room-side-gap-preferred-center-x-m", type=float, default=0.85)
    parser.add_argument("--room-side-gap-min-width-m", type=float, default=0.75)
    parser.add_argument("--room-side-gap-max-width-m", type=float, default=2.00)
    parser.add_argument("--room-side-gap-preferred-width-m", type=float, default=1.00)
    parser.add_argument("--room-side-gap-min-wall-support-count", type=int, default=1)
    parser.add_argument("--room-side-gap-require-profile-plausible", action="store_true", default=True)
    parser.add_argument("--room-side-gap-allow-nonplausible-profile", dest="room_side_gap_require_profile_plausible", action="store_false")
    parser.add_argument("--room-side-gap-require-profile-door-signal", action="store_true", default=True)
    parser.add_argument(
        "--room-side-gap-allow-raw-profile-segment",
        dest="room_side_gap_require_profile_door_signal",
        action="store_false",
    )
    parser.add_argument("--room-side-gap-max-missed-observations", type=int, default=0)
    parser.add_argument("--room-side-gap-turn-alignment-tolerance-m", type=float, default=0.25)
    parser.add_argument(
        "--door-landmark-corridor-half-width-m",
        type=float,
        default=1.125,
        help="Debug-only fallback lateral distance when an opening has no explicit base-frame y coordinate.",
    )
    parser.add_argument("--room-side-gap-turn-angular-z", type=float, default=0.45)
    parser.add_argument("--room-side-gap-turn-direct-cmd", action="store_true", default=True)
    parser.add_argument("--no-room-side-gap-turn-direct-cmd", dest="room_side_gap_turn_direct_cmd", action="store_false")
    parser.add_argument("--room-side-gap-turn-direct-angular-z", type=float, default=0.30)
    parser.add_argument("--room-side-gap-turn-raw-zero-before-direct-sec", type=float, default=0.5)
    parser.add_argument("--room-side-gap-turn-duration-sec", type=float, default=45.0)
    parser.add_argument("--room-side-gap-turn-target-yaw-rad", type=float, default=1.35)
    parser.add_argument("--room-side-gap-turn-min-yaw-rad", type=float, default=1.00)
    parser.add_argument("--room-side-turn-validation-v1", action="store_true", default=True)
    parser.add_argument("--room-side-turn-health-fresh-wall-sec", type=float, default=2.5)
    parser.add_argument("--room-side-turn-health-lost-wall-sec", type=float, default=8.0)
    parser.add_argument("--room-side-turn-wall-watchdog-sec", type=float, default=90.0)
    parser.add_argument("--room-side-turn-no-response-samples", type=int, default=8)
    parser.add_argument("--room-side-turn-yaw-response-epsilon-rad", type=float, default=0.01)
    parser.add_argument("--room-side-turn-post-fresh-grid-count", type=int, default=2)
    parser.add_argument("--room-side-turn-post-grid-watchdog-sec", type=float, default=30.0)
    parser.add_argument("--room-side-gap-partial-turn-min-yaw-rad", type=float, default=0.25)
    parser.add_argument("--room-side-gap-direct-entry-forward-m", type=float, default=0.45)
    parser.add_argument("--room-side-gap-direct-entry-lateral-m", type=float, default=0.90)
    parser.add_argument("--room-side-gap-turn-cmd-rate-hz", type=float, default=10.0)
    parser.add_argument("--room-side-gap-stop-before-turn-sec", type=float, default=0.6)
    parser.add_argument("--room-side-gap-commit-max-age-sec", type=float, default=30.0)
    parser.add_argument("--room-side-gap-commit-stale-no-progress-epsilon-m", type=float, default=0.05)
    parser.add_argument("--room-side-gap-enter-forward-m", type=float, default=0.90)
    parser.add_argument("--room-side-gap-inside-min-lateral-offset-m", type=float, default=0.45)
    parser.add_argument(
        "--enable-doorway-control",
        dest="enable_doorway_control",
        action="store_true",
        default=True,
        help="Allow doorway detections to drive DOORWAY_VERIFY/ENTER_ROOM states. Enabled by default.",
    )
    parser.add_argument(
        "--disable-doorway-control",
        dest="enable_doorway_control",
        action="store_false",
        help="Keep doorway output diagnostic-only and do not use it for room entry control.",
    )
    parser.add_argument("--enter-room-runtime-sec", type=float, default=40.0)
    parser.add_argument("--enter-room-max-steps", type=int, default=10)
    parser.add_argument("--enter-room-progress-threshold-m", type=float, default=0.35)
    parser.add_argument("--doorway-verify-runtime-sec", type=float, default=16.0)
    parser.add_argument("--doorway-verify-max-steps", type=int, default=4)
    parser.add_argument("--doorway-verify-min-linear-x", type=float, default=0.08)
    parser.add_argument("--doorway-verify-max-angular-z", type=float, default=0.55)
    parser.add_argument("--doorway-verify-distance-speed-gain", type=float, default=0.65)
    parser.add_argument("--enter-room-commit-depth-m", type=float, default=0.75)
    parser.add_argument("--enter-room-commit-min-depth-m", type=float, default=0.15)
    parser.add_argument("--enter-room-commit-depth-step-m", type=float, default=0.10)
    parser.add_argument("--enter-room-commit-grid-window-radius-cells", type=int, default=2)
    parser.add_argument("--enter-room-commit-max-window-blocked-ratio", type=float, default=0.55)
    parser.add_argument("--enter-room-commit-line-sample-step-m", type=float, default=0.05)
    parser.add_argument("--enter-room-commit-min-target-x-base-m", type=float, default=0.15)
    parser.add_argument("--enter-room-min-linear-x", type=float, default=0.12)
    parser.add_argument("--enter-room-max-angular-z", type=float, default=0.35)
    parser.add_argument(
        "--p-through-max-angular-z",
        type=float,
        default=0.30,
        help="P_through-only angular-speed cap; does not alter P_pre or ROOM_SEARCH.",
    )
    parser.add_argument("--enter-room-max-linear-accel", type=float, default=0.80)
    parser.add_argument("--enter-room-max-angular-accel", type=float, default=0.50)
    parser.add_argument("--enter-room-distance-speed-gain", type=float, default=0.80)
    parser.add_argument("--enter-room-target-heading-blend-weight", type=float, default=0.90)
    parser.add_argument("--enter-room-max-target-heading-correction-rad", type=float, default=0.85)
    parser.add_argument("--enter-room-target-lateral-correction-angular-z", type=float, default=0.24)
    parser.add_argument("--enter-room-target-lateral-correction-weight", type=float, default=0.32)
    parser.add_argument("--enter-room-dwa-predict-time", type=float, default=1.00)
    parser.add_argument("--enter-room-danger-min-target-x-base-m", type=float, default=0.10)
    parser.add_argument("--enter-room-danger-max-abs-target-y-base-m", type=float, default=0.55)
    parser.add_argument("--enter-room-danger-max-abs-heading-error-rad", type=float, default=0.80)
    parser.add_argument("--room-entry-target-tolerance-m", type=float, default=0.45)
    parser.add_argument("--room-entry-max-attempts", type=int, default=3)
    parser.add_argument("--room-scan-wait-sec", type=float, default=1.0)
    parser.add_argument("--room-scan-max-cycles", type=int, default=3)
    parser.add_argument("--entry-stuck-min-displacement-m", type=float, default=0.02)
    parser.add_argument("--stuck-min-displacement-m", type=float, default=0.06)
    parser.add_argument("--max-recovery-count", type=int, default=5)
    parser.add_argument("--recovery-backoff-linear-x", type=float, default=0.12)
    parser.add_argument("--recovery-backoff-sec", type=float, default=0.8)
    parser.add_argument("--recovery-resample-wait-sec", type=float, default=1.2)
    parser.add_argument("--recovery-zero-count", type=int, default=5)
    parser.add_argument("--recovery-cmd-rate-hz", type=float, default=10.0)
    parser.add_argument("--cmd-topic", default="/cmd_vel")
    parser.set_defaults(use_imu_velocity_follower=True)
    parser.add_argument("--use-imu-velocity-follower", dest="use_imu_velocity_follower", action="store_true")
    parser.add_argument("--no-imu-velocity-follower", dest="use_imu_velocity_follower", action="store_false")
    parser.add_argument("--follower-raw-cmd-topic", default="/cmd_vel_raw")
    parser.add_argument("--follower-output-cmd-topic", default="/cmd_vel")
    parser.add_argument("--follower-status-topic", default="/imu_velocity_follower/status")
    parser.add_argument("--follower-imu-topic", default="/trunk_imu")
    parser.add_argument("--follower-max-linear-x", type=float, default=99.0)
    parser.add_argument("--follower-max-angular-z", type=float, default=0.45)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    apply_fast_debug_profile(args, sys.argv[1:])
    # 090: legacy doorway mechanisms remain diagnostic only.  They cannot
    # create a doorway target, change a room-door state, or command motion.
    legacy_doorway_control_disabled = True
    legacy_doorway_control_disable_reason = "LEGACY_DIAGNOSTIC_DISABLED_FOR_CONTROL"
    args.enable_doorway_control = False
    args.enable_room_side_gap_trigger = False
    args.enable_forced_room_entry_mvp = False
    args.enable_simple_room_entry = False
    forced_entry_linear_x_clamped_from: Optional[float] = None
    if not args.forced_entry_allow_low_speed and float(args.forced_entry_linear_x) < 0.30:
        forced_entry_linear_x_clamped_from = float(args.forced_entry_linear_x)
        args.forced_entry_linear_x = 0.30
    if args.side_gap_switch_threshold_m is None:
        args.side_gap_switch_threshold_m = float(args.side_gap_confirm_progress_jump_m)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    rospy.init_node("state_machine_navigation", anonymous=True, disable_signals=True)
    room_zone_state_audit = RoomZoneStateAuditPublisher()
    portal_candidate_authority = PortalBoundDoorCandidateAuthority()
    atexit.register(portal_candidate_authority.close)
    room_zone_state_audit.publish(
        False,
        "ENTER_BUILDING",
        "ANCHOR_UNAVAILABLE",
    )
    portal_effect_gate_preflight = portal_effect_gate_startup_preflight(args)
    if not portal_effect_gate_preflight["ready"]:
        print(json.dumps(portal_effect_gate_preflight, sort_keys=True), file=sys.stderr)
        return 79
    odom_cache = initialize_odom_cache()
    atexit.register(odom_cache.close)
    corridor_axis_evidence = CorridorAxisEvidenceCache(args)
    atexit.register(corridor_axis_evidence.close)
    follower = start_follower(args)
    atexit.register(stop_follower, follower)
    room_side_turn_health = RoomSideTurnHealthCache(args)
    atexit.register(room_side_turn_health.close)
    formal_grid_status_pair = FormalGridStatusPairSubscriber()
    atexit.register(formal_grid_status_pair.close)
    side_gap_visual_audit = (
        SideGapVisualCoordinateAudit(args)
        if args.enable_side_gap_visual_coordinate_audit or args.enable_forced_room_entry_mvp
        else None
    )
    commands: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    recovery_count = 0
    state = "ENTER_BUILDING"
    final_decision = "STATE_MACHINE_NAVIGATION_INCOMPLETE"
    anchor: Optional[Dict[str, Any]] = None
    locked_doorway_target: Optional[Dict[str, Any]] = None
    latched_doorway_profile: Optional[Dict[str, Any]] = None
    room_side_gap_observation: Optional[Dict[str, Any]] = None
    committed_room_side_gap: Optional[Dict[str, Any]] = None
    pending_room_side_gap: Optional[Dict[str, Any]] = None
    door_tracks: Dict[str, List[Dict[str, Any]]] = {"left": [], "right": []}
    door_track_next_ids: Dict[str, int] = {"left": 0, "right": 0}
    side_gap_nav_cache: Dict[str, List[Dict[str, Any]]] = {"left": [], "right": []}
    side_gap_nav_debug_records: List[Dict[str, Any]] = []
    side_gap_segment_previous_selected: Dict[str, Dict[str, Any]] = {}
    side_gap_segment_switch_audit_events: List[Dict[str, Any]] = []
    latest_door_anchor_progress: Optional[float] = None
    last_normal_state = "FOLLOW_CORRIDOR"
    debug_stop_reason: Optional[str] = None
    debug_stop_details: Dict[str, Any] = {}
    portal_bound_candidate: Optional[Dict[str, Any]] = None
    portal_g14_shadow_target: Optional[Dict[str, Any]] = None
    portal_g14_p_pre_admissibility: Optional[Dict[str, Any]] = None
    portal_g14_p_pre_switch: Optional[Dict[str, Any]] = None
    portal_g14_p_pre_runner: Optional[Dict[str, Any]] = None
    room_search_v2_result: Optional[Dict[str, Any]] = None
    # Mission-local passive memory.  It intentionally outlives a ROOM_SEARCH
    # episode, but is not persisted across process restart.
    visited_portals: List[Dict[str, Any]] = []
    post_room_return_context: Optional[Dict[str, Any]] = None
    post_room_return_resume_marker = False
    portal_bound_control_domain_entered = False
    approach_candidate_valid_count = 0
    approach_point_debug_count = 0
    approach_point_debug_pass_count = 0
    first_approach_point_debug: Optional[Dict[str, Any]] = None
    door_cue_ahead_count = 0
    raw_door_cue_ahead_count = 0
    first_door_cue_ahead: Optional[Dict[str, Any]] = None
    actionable_door_cue_count = 0
    first_actionable_door_cue: Optional[Dict[str, Any]] = None
    rejected_before_room_zone_count = 0
    rejected_broad_bilateral_opening_count = 0
    door_observe_debug_count = 0
    door_observe_confirmed: Optional[bool] = None
    first_door_observe_debug: Optional[Dict[str, Any]] = None
    side_gap_nav_debug_count = 0
    side_gap_nav_debug_confirmed_count = 0
    first_side_gap_nav_debug: Optional[Dict[str, Any]] = None
    side_gap_segment_switch_audit_count = 0
    left_selected_switch_count = 0
    right_selected_switch_count = 0
    max_left_selected_center_progress_jump_m: Optional[float] = None
    first_large_left_switch: Optional[Dict[str, Any]] = None
    first_side_gap_visual_audit_event: Optional[Dict[str, Any]] = None
    # Summary construction also runs when target preparation fails before the
    # first loop iteration.  Keep the diagnostic-only visual counters defined
    # for that zero-iteration path; later iterations refresh them from the
    # audit manifest.
    side_gap_visual_audit_event_count = 0
    side_gap_visual_audit_broad_event_count = 0
    local_entry_debug_events: List[Dict[str, Any]] = []
    latest_local_entry_debug: Optional[Dict[str, Any]] = None
    forced_room_entry_result: Optional[Dict[str, Any]] = None
    forced_room_entry_done = False
    pending_forced_room_entry_opening: Optional[Dict[str, Any]] = None
    forced_room_entry_opening_observation_cache: Dict[str, List[Dict[str, Any]]] = {
        "left": [],
        "right": [],
    }
    last_forced_room_entry_trigger_status: Optional[Dict[str, Any]] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None

    try:
        anchor = build_initial_anchor(args, commands)
    except Exception as exc:
        final_decision = f"STATE_MACHINE_TARGET_PREP_FAILED:{exc}"
        state = "FAILED"

    for iteration in range(args.max_iterations):
        if state in {"DONE", "FAILED"}:
            break
        if args.debug_max_iterations is not None and iteration >= int(args.debug_max_iterations):
            debug_stop_reason = "debug_max_iterations"
            debug_stop_details = {
                "debug_max_iterations": int(args.debug_max_iterations),
                "completed_iterations": len(trace),
                "safe_stop": publish_fast_debug_stop(args),
            }
            final_decision = "STATE_MACHINE_DEBUG_MAX_ITERATIONS_REACHED"
            state = "DONE"
            break
        item: Dict[str, Any] = {"iteration": iteration, "state": state, "started_wall_time_sec": time.time()}
        next_state = state
        reason = "continue"
        try:
            if anchor is not None:
                handoff = maybe_handoff_corridor_axis(anchor, corridor_axis_evidence, odom_cache)
                item["corridor_axis_handoff"] = handoff
                if handoff.get("action") == "HANDOFF_TO_CORRIDOR_AXIS":
                    anchor = handoff["anchor"]
                    item["corridor_axis_lifecycle"] = "CORRIDOR_BOUND"
                elif handoff.get("action") == "REBOOTSTRAP_REQUIRED":
                    anchor = build_initial_anchor(args, commands)
                    item["corridor_axis_rebootstrap"] = {
                        "reason": handoff.get("reason"),
                        "corridor_axis_lifecycle": "LEGACY_BOOTSTRAP",
                    }
            if state == "ENTER_BUILDING":
                write_stage("building_entry", "navigation_state_machine.py")
                if anchor is None:
                    raise RuntimeError("entry_anchor_unavailable")
                target = write_anchor_target(
                    anchor,
                    args.entry_lookahead_m,
                    "state_machine_entry_centerline",
                    room_zone_active=False,
                )
                item["target"] = target
                item["target_diagnostics_before_runner"] = target_diagnostics(target, pose_tuple(read_odom()), anchor)
                result = run_runner(args, state, args.runner_runtime_sec, args.runner_max_steps)
                item.update(result)
                runner = result["runner"]
                pose = pose_tuple(read_odom())
                metrics = anchor_metrics(anchor, pose)
                item["entry_anchor_metrics_after"] = metrics
                item["target_diagnostics_after_runner"] = target_diagnostics(target, pose, anchor)
                item["follower_status_after_runner"] = read_follower_status(args)
                if runner_is_stuck(args, runner, state=state):
                    next_state, reason = "STUCK_RECOVERY", "runner_stuck_during_enter_building"
                    last_normal_state = "ENTER_BUILDING"
                elif float(metrics["anchor_progress_m"]) >= args.entry_progress_threshold_m:
                    next_state, reason = "FOLLOW_CORRIDOR", "entry_anchor_progress_threshold_met"
                else:
                    next_state, reason = "ENTER_BUILDING", "entry_progress_not_yet_met"

            elif state == "FOLLOW_CORRIDOR":
                write_stage("inside_corridor", "navigation_state_machine.py")
                time.sleep(args.corridor_doorway_check_wait_sec)
                doorway = read_json(DOORWAY_PATH)
                item["doorway_before"] = doorway
                room_zone_reached = False
                room_zone_heading_restore = None
                room_zone_reason = "ANCHOR_UNAVAILABLE"
                if anchor is not None:
                    pose = pose_tuple(read_odom())
                    metrics = anchor_metrics(anchor, pose)
                    current_anchor_progress_m = metrics.get("anchor_progress_m")
                    effective_progress_m = effective_room_zone_progress_m(anchor, current_anchor_progress_m)
                    room_zone_reached = bool(
                        effective_progress_m is not None
                        and effective_progress_m >= args.room_zone_start_progress_m
                    )
                    room_zone_reason = (
                        "EFFECTIVE_ROOM_ZONE_PROGRESS_AT_OR_ABOVE_ROOM_ZONE_START"
                        if room_zone_reached else "EFFECTIVE_ROOM_ZONE_PROGRESS_BELOW_ROOM_ZONE_START"
                    )
                    if room_zone_reached:
                        room_zone_heading_restore = restore_raw_anchor_heading_at_room_zone(anchor, args)
                        item["room_zone_heading_restore"] = room_zone_heading_restore
                    item["corridor_anchor_metrics_before"] = metrics
                    item["room_zone_start_progress_m"] = args.room_zone_start_progress_m
                    item["current_anchor_progress_m_before"] = current_anchor_progress_m
                    item["effective_room_zone_progress_m_before"] = effective_progress_m
                    item["room_zone_progress_offset_m_before"] = anchor.get("room_zone_progress_offset_m", 0.0)
                    item["room_zone_reached_before"] = room_zone_reached
                    write_stage(
                        "inside_corridor",
                        "navigation_state_machine.py",
                        {
                            "room_zone_reached": room_zone_reached,
                            "anchor_progress_m": float(current_anchor_progress_m),
                            "current_anchor_progress_m": current_anchor_progress_m,
                            "effective_room_zone_progress_m": effective_progress_m,
                            "room_zone_progress_offset_m": anchor.get("room_zone_progress_offset_m", 0.0),
                            "room_zone_start_progress_m": args.room_zone_start_progress_m,
                        },
                    )
                room_zone_state_audit.publish(
                    room_zone_reached,
                    "FOLLOW_CORRIDOR",
                    room_zone_reason,
                    current_anchor_progress_m=(
                        metrics.get("anchor_progress_m") if anchor is not None else None
                    ),
                    effective_room_zone_progress_m=(
                        effective_room_zone_progress_m(anchor, metrics.get("anchor_progress_m"))
                        if anchor is not None else None
                    ),
                    room_zone_progress_offset_m=(
                        anchor.get("room_zone_progress_offset_m", 0.0) if anchor is not None else None
                    ),
                )
                portal_effect_snapshot = portal_candidate_authority.snapshot()
                item["doorway_candidate_authority"] = DOORWAY_CANDIDATE_AUTHORITY
                item["legacy_doorway_control_disabled"] = legacy_doorway_control_disabled
                item["legacy_doorway_control_disable_reason"] = legacy_doorway_control_disable_reason
                item["portal_effect_rejection"] = portal_effect_snapshot.get("last_rejection")
                portal_bound_candidate = portal_effect_snapshot.get("committed_portal_candidate")
                if post_room_return_resume_marker and isinstance(portal_bound_candidate, dict) and anchor is not None:
                    resumed_binding = odom_cache.pose_at_source_stamp(float(portal_bound_candidate["portal_source_stamp"]))
                    resumed_geometry = freeze_portal_geometry(portal_bound_candidate, resumed_binding)
                    resumed_centre = resumed_geometry.get("portal_center_odom") if isinstance(resumed_geometry, dict) else None
                    if (
                        _finite_shape(resumed_centre, 2)
                        and portal_matches_entered_or_completed_visit(resumed_centre, visited_portals, anchor)
                    ):
                        portal_candidate_authority.reset_post_room_return_candidate()
                        portal_bound_candidate = None
                        portal_g14_shadow_target = None
                        item["completed_portal_recommit_blocked"] = True
                portal_bound_control_domain = portal_bound_doorway_control_domain_status(
                    room_zone_reached,
                    portal_effect_snapshot,
                )
                portal_bound_control_domain_entered = (
                    portal_bound_control_domain_entered or bool(portal_bound_control_domain["active"])
                )
                item["portal_bound_doorway_control_domain"] = portal_bound_control_domain
                item["portal_committed"] = isinstance(portal_bound_candidate, dict)
                if portal_bound_candidate is not None and (
                    portal_g14_shadow_target is None or not portal_g14_shadow_target.get("target_valid")
                ):
                    latest_msg, _latest_sequence, _latest_stamp = odom_cache.snapshot()
                    current_pose = None
                    if latest_msg is not None:
                        latest_pose = latest_msg.pose.pose
                        current_pose = (float(latest_pose.position.x), float(latest_pose.position.y), yaw_from_quat(latest_pose.orientation))
                    binding = odom_cache.pose_at_source_stamp(float(portal_bound_candidate["portal_source_stamp"]))
                    approach_direction = None
                    if anchor is not None and finite_number(anchor.get("heading_rad")):
                        approach_heading = float(anchor["heading_rad"])
                        approach_direction = [math.cos(approach_heading), math.sin(approach_heading)]
                    portal_g14_shadow_target = build_g14_shadow_target(
                        portal_bound_candidate,
                        binding,
                        current_pose,
                        approach_direction,
                        args.p_pre_upstream_tangent_m,
                    )
                    item["portal_g14_shadow_target"] = portal_g14_shadow_target
                    write_json(PORTAL_G14_SHADOW_TARGET_PATH, portal_g14_shadow_target)
                    if g14_shadow_stop_eligible(portal_g14_shadow_target) and args.stop_after_first_portal_g14_shadow_target:
                        raise FastDebugPortalBoundCandidateStop(portal_bound_candidate)
                if (
                    args.enable_hierarchical_portal_local_autonomy
                    or args.enable_stair_moving_turn_portal_entry
                    or args.stop_after_portal_g14_p_pre_reached
                    or args.stop_after_portal_normal_aligned
                ) and portal_g14_shadow_target is not None:
                    if portal_bound_candidate is None:
                        portal_g14_p_pre_admissibility = {
                            "state": "P_PRE_CANDIDATE_INVALIDATED",
                            "reason": "COMMITTED_PORTAL_CANDIDATE_NO_LONGER_AVAILABLE",
                        }
                    else:
                        try:
                            pair = formal_grid_status_pair.matching_pair()
                            if pair is None:
                                portal_g14_p_pre_admissibility = p_pre_grid_status_pair_pending_admissibility(
                                    portal_bound_candidate,
                                    portal_g14_shadow_target,
                                    None,
                                    None,
                                )
                            else:
                                grid_msg, status_payload = pair
                                pose_binding = p_pre_pose_binding_for_grid_status(
                                    odom_cache,
                                    status_payload,
                                )
                                if pose_binding.get("binding_valid") is not True:
                                    portal_g14_p_pre_admissibility = p_pre_grid_status_pair_pending_admissibility(
                                        portal_bound_candidate,
                                        portal_g14_shadow_target,
                                        None,
                                        None,
                                        pose_binding,
                                    )
                                else:
                                    current_pose = tuple(pose_binding["source_pose_x_y_yaw"])
                                    portal_g14_p_pre_admissibility = evaluate_portal_g14_p_pre_admissibility(
                                        portal_bound_candidate,
                                        portal_g14_shadow_target,
                                        current_pose,
                                        grid_msg,
                                        status_payload,
                                        args.robot_radius_m,
                                    )
                                    portal_g14_p_pre_admissibility["pose_binding"] = pose_binding
                        except Exception as grid_exc:
                            portal_g14_p_pre_admissibility = {
                                "state": "P_PRE_GRID_UNQUALIFIED",
                                "reason": f"FORMAL_GRID_INPUT_UNAVAILABLE:{type(grid_exc).__name__}",
                            }
                    item["portal_g14_p_pre_admissibility"] = portal_g14_p_pre_admissibility
                    p_pre_state = str(portal_g14_p_pre_admissibility.get("state"))
                    if p_pre_state == "P_PRE_READY_FOR_PLANNER":
                        previous_target = read_json(TARGET_PATH)
                        target = write_portal_g14_p_pre_target(portal_g14_shadow_target, portal_g14_p_pre_admissibility)
                        portal_g14_p_pre_switch = {
                            "portal_identity": portal_g14_shadow_target.get("portal_identity"),
                            "switch_iteration": iteration,
                            "switch_wall_time_sec": time.time(),
                            "current_pose_x_y_yaw": list(current_pose),
                            "P_pre_current_base": [
                                portal_g14_p_pre_admissibility.get("P_pre_base_x"),
                                portal_g14_p_pre_admissibility.get("P_pre_base_y"),
                            ],
                            "previous_target_source": previous_target.get("source"),
                            "new_target_source": "PORTAL_G14_P_PRE",
                        }
                        item["portal_g14_p_pre_switch"] = portal_g14_p_pre_switch
                        item["target"] = target
                        item["target_diagnostics_before_runner"] = target_diagnostics(target, current_pose, anchor)
                        result = run_runner(
                            args,
                            state,
                            args.runner_runtime_sec,
                            portal_g14_p_pre_max_steps(args),
                        )
                        item.update(result)
                        runner = result["runner"]
                        portal_g14_p_pre_runner = runner
                        final_pose = pose_tuple(read_odom())
                        item["target_diagnostics_after_runner"] = target_diagnostics(target, final_pose, anchor)
                        item["follower_status_after_runner"] = read_follower_status(args)
                        if portal_candidate_authority.snapshot().get("committed_portal_candidate") is None:
                            outcome = {
                                "reached": False,
                                "final_decision": "P2KG15_092_CANDIDATE_INVALIDATED",
                                "reason": "committed_portal_candidate_invalidated_during_p_pre_runner",
                            }
                        else:
                            outcome = portal_g14_p_pre_runner_outcome(runner)
                        frozen = portal_g14_shadow_target.get("frozen_geometry") or {}
                        normal = frozen.get("portal_normal_odom") or [None, None]
                        normal_yaw = math.atan2(float(normal[1]), float(normal[0])) if _finite_shape(normal, 2) else None
                        terminal = {
                            "portal_g14_p_pre_admissibility": portal_g14_p_pre_admissibility,
                            "portal_g14_p_pre_switch": portal_g14_p_pre_switch,
                            "portal_g14_p_pre_runner": runner,
                            "robot_yaw_at_P_pre": final_pose[2] if outcome.get("reached") else None,
                            "actual_arrival_pose_odom": list(final_pose) if outcome.get("reached") else None,
                            "actual_arrival_pose_source": "pose_tuple(read_odom())" if outcome.get("reached") else None,
                            "anchor_progress_m": anchor_metrics(anchor, final_pose).get("anchor_progress_m") if outcome.get("reached") and anchor is not None else None,
                            "portal_room_normal_yaw": normal_yaw,
                            "heading_error_to_portal_normal": normalize_angle(final_pose[2] - normal_yaw) if outcome.get("reached") and normal_yaw is not None else None,
                        }
                        if outcome.get("reached") and args.enable_hierarchical_portal_local_autonomy:
                            pair = formal_grid_status_pair.matching_pair()
                            terminal["moving_turn_full_path_shadow"] = (
                                build_stair_moving_turn_entry_path(
                                    portal_g14_shadow_target,
                                    final_pose,
                                    pair[0],
                                    pair[1],
                                    args.robot_radius_m,
                                )
                                if pair is not None
                                else {"state": "STAIR_MOVING_TURN_SHADOW_UNAVAILABLE", "reason": "EXACT_GRID_STATUS_PAIR_UNAVAILABLE"}
                            )
                            p_through_target = write_portal_g14_p_through_target(portal_g14_shadow_target)
                            terminal["p_through_high_level_goal_replacement"] = {
                                "previous_goal_source": target.get("source"),
                                "new_goal_source": p_through_target.get("source"),
                                "goal_xy_team_livox_odom": p_through_target.get("target_xy_team_livox_odom"),
                                "motion_authority": "BLOCK_ASTAR_DWA",
                            }
                            item["p_through_high_level_goal_replacement"] = terminal["p_through_high_level_goal_replacement"]
                            p_through_result = run_runner(
                                args,
                                "PORTAL_P_THROUGH",
                                args.runner_runtime_sec,
                                portal_g14_p_through_max_steps(args),
                            )
                            terminal["p_through_local_runner"] = p_through_result["runner"]
                            p_through_final_pose = pose_tuple(read_odom())
                            terminal["actual_p_through_final_pose_odom"] = list(p_through_final_pose)
                            p_through_outcome = portal_g14_p_through_runner_outcome(
                                p_through_result["runner"],
                                portal_g14_shadow_target,
                                p_through_final_pose,
                                deep_crossing_min_progress_m=args.p_through_deep_crossing_min_progress_m,
                                deep_crossing_max_distance_m=args.p_through_deep_crossing_max_distance_m,
                                robot_radius_m=args.robot_radius_m,
                            )
                            terminal["p_through_outcome"] = p_through_outcome
                            entered_portal_memory: Optional[Dict[str, Any]] = None
                            if p_through_outcome.get("reached"):
                                frozen_geometry = portal_g14_shadow_target.get("frozen_geometry") or {}
                                entered_portal_memory = record_entered_visited_portal(
                                    visited_portals,
                                    frozen_geometry.get("portal_center_odom") or [],
                                    anchor or {},
                                )
                                terminal["visited_portal_memory"] = entered_portal_memory
                                if entered_portal_memory.get("state") == "VISITED_PORTAL_IDENTITY_AMBIGUOUS_MATCH":
                                    raise PortalG14PPreFinished(
                                        "VISITED_PORTAL_IDENTITY_AMBIGUOUS_MATCH",
                                        "visited_portal_identity_ambiguous_match",
                                        terminal,
                                    )
                            if p_through_outcome.get("reached") and args.enable_room_search_v2:
                                room_search_v2_result = execute_room_search_v2(
                                    args, portal_g14_shadow_target, p_through_final_pose,
                                )
                                terminal["room_search_v2"] = room_search_v2_result
                                if (
                                    entered_portal_memory is not None
                                    and entered_portal_memory.get("record") is not None
                                    and room_search_v2_result.get("final_decision") == "ROOM_SEARCH_V2_RETURNED_TO_DOOR_ANCHOR"
                                ):
                                    mark_visited_portal_completed(entered_portal_memory["record"])
                                    post_room_return_context = {
                                        "returned_portal_record": entered_portal_memory["record"],
                                        "minimum_fresh_portal_frame_sequence": portal_candidate_authority.snapshot().get("last_portal_frame_sequence"),
                                    }
                                    terminal["post_room_return_context"] = post_room_return_context
                                    raise RoomReturnNextPortalDispatch(terminal)
                                raise PortalG14PPreFinished(
                                    str(room_search_v2_result.get("final_decision")),
                                    "room_search_v2_finished",
                                    terminal,
                                )
                            raise PortalG14PPreFinished(
                                str(p_through_outcome["final_decision"]), str(p_through_outcome["reason"]), terminal,
                            )
                        if outcome.get("reached") and args.enable_stair_moving_turn_portal_entry:
                            terminal["stair_moving_turn_portal_entry"] = execute_stair_moving_turn_portal_entry(
                                args,
                                portal_bound_candidate,
                                portal_g14_shadow_target,
                                formal_grid_status_pair,
                            )
                            entry = terminal["stair_moving_turn_portal_entry"]
                            raise PortalG14PPreFinished(
                                str(entry.get("final_decision") or "P2KG15_093_MOVING_TURN_PATH_INVALID"),
                                str(entry.get("reason") or "stair_moving_turn_failed"),
                                terminal,
                            )
                        if outcome.get("reached") and (
                            args.stop_after_portal_normal_aligned or args.stop_after_p_through_handoff_ready
                        ):
                            selected_d_m = portal_g14_p_pre_admissibility.get("selected_d_m")
                            terminal["portal_normal_alignment"] = execute_portal_normal_alignment(
                                args,
                                portal_bound_candidate,
                                portal_g14_shadow_target,
                                formal_grid_status_pair,
                                float(selected_d_m),
                            )
                            alignment = terminal["portal_normal_alignment"]
                            if alignment.get("aligned"):
                                raise PortalG14PPreFinished(
                                    "P2KG15_093_P_THROUGH_HANDOFF_READY",
                                    "p_through_handoff_ready",
                                    terminal,
                                )
                            raise PortalG14PPreFinished(
                                str(alignment.get("final_decision") or "P2KG15_093_HEADING_CONTROLLER_NONCONVERGENCE"),
                                str(alignment.get("reason") or "portal_normal_alignment_failed"),
                                terminal,
                            )
                        terminal["safe_stop"] = publish_stop_at_door(args) if args.execute else {"execute": False, "reason": "dry_run_no_stop_command"}
                        raise PortalG14PPreFinished(str(outcome["final_decision"]), str(outcome["reason"]), terminal)
                    if p_pre_state == "P_PRE_GRID_STATUS_PAIR_PENDING":
                        item["portal_g14_p_pre_pair_pending_safe_stop"] = (
                            publish_stop_at_door(args)
                            if args.execute
                            else {"execute": False, "reason": "dry_run_no_stop_command"}
                        )
                        continue
                    if p_pre_state != "P_PRE_OUTSIDE_LOCAL_PLANNING_WINDOW":
                        rejection = {
                            "P_PRE_CANDIDATE_INVALIDATED": "P2KG15_092_CANDIDATE_INVALIDATED",
                            "P_PRE_GRID_UNQUALIFIED": "P2KG15_092_P_PRE_PLANNER_REJECTED",
                            "P_PRE_TARGET_CELL_BLOCKED": "P2KG15_092_P_PRE_PLANNER_REJECTED",
                            "P_PRE_NO_SAFE_NORMAL_DISTANCE": "P2KG15_092_P_PRE_PLANNER_REJECTED",
                        }.get(p_pre_state, "P2KG15_092_SCOPE_EXPANSION_REQUIRED")
                        terminal = {
                            "portal_g14_p_pre_admissibility": portal_g14_p_pre_admissibility,
                            "safe_stop": publish_stop_at_door(args) if args.execute else {"execute": False, "reason": "dry_run_no_stop_command"},
                        }
                        raise PortalG14PPreFinished(rejection, str(portal_g14_p_pre_admissibility.get("reason")), terminal)
                doorway_before_status = (
                    doorway_opening_status(doorway) if doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY" else {}
                )
                if doorway_before_status:
                    item["doorway_before_opening_status"] = doorway_before_status
                if args.enable_forced_room_entry_mvp:
                    if args.enable_simple_room_entry:
                        forced_trigger_before = select_simple_room_entry_opening(
                            doorway,
                            room_zone_reached,
                            metrics.get("anchor_progress_m") if isinstance(metrics, dict) else None,
                            args,
                            pending_forced_room_entry_opening,
                        )
                    else:
                        forced_trigger_before = select_forced_room_entry_opening(
                            doorway,
                            room_zone_reached,
                            metrics.get("anchor_progress_m") if isinstance(metrics, dict) else None,
                            args,
                            pending_forced_room_entry_opening,
                            latched_doorway_profile,
                            forced_room_entry_opening_observation_cache,
                        )
                    pending_forced_room_entry_opening = forced_trigger_before.get("pending_forced_opening")
                    if forced_trigger_before.get("stop_observe_required"):
                        stop_result = (
                            publish_stop_at_door(args)
                            if args.execute
                            else {"execute": False, "reason": "dry_run_no_stop_command"}
                        )
                        forced_trigger_before["stop_observe_result"] = stop_result
                        item["simple_room_entry_stop_observe_before"] = stop_result
                        if isinstance(pending_forced_room_entry_opening, dict):
                            pending_forced_room_entry_opening["stop_observe_completed"] = True
                    last_forced_room_entry_trigger_status = forced_trigger_before
                    item["forced_room_entry_trigger_before"] = forced_trigger_before
                    if (
                        forced_trigger_before.get("trigger_ready")
                        and not forced_room_entry_done
                        and not args.room_side_turn_validation_v1
                    ):
                        forced_room_entry_result = execute_forced_room_entry_mvp(
                            args,
                            forced_trigger_before,
                            side_gap_visual_audit,
                            forced_entry_linear_x_clamped_from,
                            follower,
                            pose,
                        )
                        forced_room_entry_done = bool(forced_room_entry_result.get("forced_entry_done"))
                        item["forced_room_entry_result"] = forced_room_entry_result
                        raise ForcedRoomEntryMVPFinished(
                            forced_room_entry_result,
                            bool(args.forced_entry_stop_after_done),
                        )
                    elif forced_trigger_before.get("trigger_ready") and args.room_side_turn_validation_v1:
                        item["forced_room_entry_execution_suppressed_reason"] = (
                            "room_side_turn_validation_v1_forbids_entry"
                        )
                if args.enable_local_free_space_entry_debug:
                    try:
                        latest_local_entry_debug = build_local_free_space_entry_debug(
                            args,
                            pose,
                            room_zone_reached,
                            metrics.get("anchor_progress_m") if isinstance(metrics, dict) else None,
                        )
                        item["local_free_space_entry_debug"] = latest_local_entry_debug
                        if (
                            room_zone_reached
                            and len(local_entry_debug_events) < int(args.local_entry_max_events)
                        ):
                            event = write_local_entry_debug_event(
                                len(local_entry_debug_events) + 1,
                                latest_local_entry_debug,
                                side_gap_visual_audit,
                            )
                            local_entry_debug_events.append(event)
                        first_valid_local_entry = latest_local_entry_debug.get("first_valid_entry_target")
                        if (
                            args.fast_debug
                            and args.stop_after_first_local_entry_target_debug
                            and isinstance(first_valid_local_entry, dict)
                        ):
                            raise FastDebugLocalEntryTargetStop(first_valid_local_entry)
                    except FastDebugLocalEntryTargetStop:
                        raise
                    except Exception as local_entry_debug_exc:
                        latest_local_entry_debug = {
                            "diagnostic_only": True,
                            "controls_robot": False,
                            "controls_next_state": False,
                            "writes_real_target": False,
                            "calls_runner": False,
                            "grid_available": False,
                            "missing_reason": "local_entry_debug_exception",
                            "error": repr(local_entry_debug_exc),
                        }
                        item["local_free_space_entry_debug_error"] = repr(local_entry_debug_exc)
                if anchor is not None and room_zone_reached:
                    side_gap_segment_audit_before = build_side_gap_segment_switch_audit_events(
                        doorway,
                        pose,
                        anchor,
                        iteration,
                        "before_corridor_motion",
                        room_zone_reached,
                        args,
                        side_gap_segment_previous_selected,
                    )
                    side_gap_segment_switch_audit_events.extend(side_gap_segment_audit_before)
                    item["side_gap_segment_switch_audit_before"] = side_gap_segment_audit_before
                    if side_gap_visual_audit is not None:
                        visual_switch_events = [
                            visual_event
                            for visual_event in (
                                side_gap_visual_audit.capture(candidate)
                                for candidate in (
                                    visual_candidate_from_segment_switch_event(event)
                                    for event in side_gap_segment_audit_before
                                    if event.get("selected_switch_detected")
                                )
                                if candidate is not None
                            )
                            if visual_event is not None
                        ]
                        if visual_switch_events:
                            item["side_gap_visual_audit_switch_events_before"] = visual_switch_events
                            first_side_gap_visual_audit_event = (
                                first_side_gap_visual_audit_event or visual_switch_events[0]
                            )
                            if args.fast_debug and args.stop_after_first_side_gap_visual_audit:
                                raise FastDebugSideGapVisualAuditStop(visual_switch_events[0])
                    first_detected_switch = next(
                        (event for event in side_gap_segment_audit_before if event.get("selected_switch_detected")),
                        None,
                    )
                    if (
                        args.fast_debug
                        and args.stop_after_first_side_gap_segment_switch_audit
                        and first_detected_switch
                    ):
                        raise FastDebugSideGapSegmentSwitchAuditStop(first_detected_switch)
                if anchor is not None:
                    door_landmark_observations_before = update_door_landmark_tracks(
                        door_tracks,
                        door_track_next_ids,
                        doorway,
                        {},
                        pose,
                        anchor,
                        iteration,
                        "before_corridor_motion",
                        args.door_landmark_corridor_half_width_m,
                    )
                    if door_landmark_observations_before:
                        annotate_door_cue_actionability(
                            door_landmark_observations_before,
                            room_zone_reached,
                            args.room_zone_start_progress_m,
                            metrics.get("anchor_progress_m") if isinstance(metrics, dict) else None,
                        )
                        side_gap_nav_before = build_side_gap_nav_debug_candidates(
                            door_landmark_observations_before,
                            side_gap_nav_cache,
                            anchor,
                            pose,
                            args,
                            room_zone_reached,
                            metrics.get("anchor_progress_m") if isinstance(metrics, dict) else None,
                        )
                        side_gap_nav_debug_records.extend(side_gap_nav_before)
                        if side_gap_nav_before:
                            item["side_gap_nav_debug_before"] = side_gap_nav_before
                            if side_gap_visual_audit is not None:
                                visual_events_before = [
                                    event
                                    for event in (side_gap_visual_audit.capture(candidate) for candidate in side_gap_nav_before)
                                    if event is not None
                                ]
                                if visual_events_before:
                                    item["side_gap_visual_audit_events_before"] = visual_events_before
                                    first_side_gap_visual_audit_event = (
                                        first_side_gap_visual_audit_event or visual_events_before[0]
                                    )
                                    if args.fast_debug and args.stop_after_first_side_gap_visual_audit:
                                        raise FastDebugSideGapVisualAuditStop(visual_events_before[0])
                        item["door_landmark_observations_before"] = door_landmark_observations_before
                        current_door_cues = [
                            observation
                            for observation in door_landmark_observations_before
                            if observation.get("raw_door_cue_ahead")
                        ]
                        current_actionable_door_cues = [
                            observation for observation in current_door_cues if observation.get("actionable_door_cue")
                        ]
                        if args.fast_debug and args.stop_after_first_door_cue and current_door_cues:
                            raise FastDebugDoorCueStop(current_door_cues)
                        if (
                            args.fast_debug
                            and args.stop_after_first_actionable_door_cue
                            and current_actionable_door_cues
                        ):
                            raise FastDebugDoorCueStop(
                                current_actionable_door_cues,
                                "stop_after_first_actionable_door_cue",
                            )
                        first_confirmed_side_gap = next(
                            (candidate for candidate in side_gap_nav_before if candidate.get("lightweight_confirmed")),
                            None,
                        )
                        if args.fast_debug and args.stop_after_first_side_gap_nav_debug and first_confirmed_side_gap:
                            raise FastDebugSideGapNavStop(first_confirmed_side_gap)
                doorway_latch_status_before: Dict[str, Any] = {}
                doorway_latch_alignment_before: Dict[str, Any] = {}
                if room_zone_reached and anchor is not None:
                    pose_for_latch = pose_tuple(read_odom())
                    latched_doorway_profile, doorway_latch_status_before = maybe_latch_doorway_profile(
                        args,
                        anchor,
                        pose_for_latch,
                        doorway,
                        latched_doorway_profile,
                    )
                    item["doorway_latch_status_before"] = doorway_latch_status_before
                    if latched_doorway_profile is not None:
                        item["latched_doorway_profile"] = latched_doorway_profile
                        doorway_latch_alignment_before = doorway_latch_alignment(
                            latched_doorway_profile,
                            anchor,
                            pose_for_latch,
                            args,
                        )
                        extend_partial_latch_target_if_needed(
                            latched_doorway_profile,
                            doorway_latch_alignment_before,
                            args,
                        )
                        item["doorway_latch_alignment_before"] = doorway_latch_alignment_before
                room_side_gap_transition = False
                if (
                    args.enable_room_side_gap_trigger
                    and room_zone_reached
                    and anchor is not None
                ):
                    pose_for_gap = pose_tuple(read_odom())
                    gap_candidate = room_side_gap_candidate(doorway, args)
                    room_side_gap_observation, gap_status = update_room_side_gap_observation(
                        args,
                        room_side_gap_observation,
                        gap_candidate,
                        anchor,
                        pose_for_gap,
                    )
                    item["room_side_gap_status_before"] = gap_status
                    current_progress = gap_status.get("anchor_progress_m")
                    if not finite_number(current_progress) and isinstance(metrics, dict):
                        current_progress = metrics.get("anchor_progress_m")
                    gap_landmark_observations = update_door_landmark_tracks(
                        door_tracks,
                        door_track_next_ids,
                        doorway,
                        gap_status,
                        pose_for_gap,
                        anchor,
                        iteration,
                        "room_side_gap_before",
                        args.door_landmark_corridor_half_width_m,
                    )
                    if gap_landmark_observations:
                        annotate_door_cue_actionability(
                            gap_landmark_observations,
                            room_zone_reached,
                            args.room_zone_start_progress_m,
                            float(current_progress) if finite_number(current_progress) else None,
                        )
                        side_gap_nav_from_gap = build_side_gap_nav_debug_candidates(
                            gap_landmark_observations,
                            side_gap_nav_cache,
                            anchor,
                            pose_for_gap,
                            args,
                            room_zone_reached,
                            float(current_progress) if finite_number(current_progress) else None,
                        )
                        side_gap_nav_debug_records.extend(side_gap_nav_from_gap)
                        if side_gap_nav_from_gap:
                            item.setdefault("side_gap_nav_debug_before", []).extend(side_gap_nav_from_gap)
                            if side_gap_visual_audit is not None:
                                visual_events_from_gap = [
                                    event
                                    for event in (side_gap_visual_audit.capture(candidate) for candidate in side_gap_nav_from_gap)
                                    if event is not None
                                ]
                                if visual_events_from_gap:
                                    item.setdefault("side_gap_visual_audit_events_before", []).extend(
                                        visual_events_from_gap
                                    )
                                    first_side_gap_visual_audit_event = (
                                        first_side_gap_visual_audit_event or visual_events_from_gap[0]
                                    )
                                    if args.fast_debug and args.stop_after_first_side_gap_visual_audit:
                                        raise FastDebugSideGapVisualAuditStop(visual_events_from_gap[0])
                        item.setdefault("door_landmark_observations_before", []).extend(gap_landmark_observations)
                        first_confirmed_side_gap = next(
                            (candidate for candidate in side_gap_nav_from_gap if candidate.get("lightweight_confirmed")),
                            None,
                        )
                        if args.fast_debug and args.stop_after_first_side_gap_nav_debug and first_confirmed_side_gap:
                            raise FastDebugSideGapNavStop(first_confirmed_side_gap)
                    committed_room_side_gap, commit_status = update_committed_room_side_gap(
                        args,
                        committed_room_side_gap,
                        gap_status,
                        float(current_progress) if finite_number(current_progress) else None,
                    )
                    item["committed_room_side_gap_status_before"] = commit_status
                    committed_trigger = committed_room_side_gap_trigger_status(
                        args,
                        committed_room_side_gap,
                        float(current_progress) if finite_number(current_progress) else None,
                    )
                    item["committed_room_side_gap_trigger_before"] = committed_trigger
                    trigger_gap = committed_trigger if committed_trigger.get("trigger_ready") else gap_status
                    opening_decision_before = select_follow_corridor_opening_action(
                        doorway_control_enabled=bool(args.enable_doorway_control),
                        fully_bounded_doorway_ready=bool(
                            doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY"
                            and doorway_before_status.get("opening_center_estimated")
                        ),
                        room_side_gap_enabled=bool(args.enable_room_side_gap_trigger),
                        room_side_gap_trigger_ready=bool(trigger_gap.get("trigger_ready")),
                        forced_room_entry_enabled=bool(args.enable_forced_room_entry_mvp),
                    )
                    item["follow_corridor_opening_decision_before"] = opening_decision_before
                    if opening_decision_before.get("action") == "ROOM_SIDE_TURN":
                        health_gate = room_side_turn_health.snapshot()
                        item["room_side_turn_health_gate_before"] = health_gate
                        item["stop_at_door_result"] = publish_stop_at_door(args)
                        room_side_gap_transition = True
                        if health_gate.get("ready"):
                            pending_room_side_gap = trigger_gap
                            latched_doorway_profile = None
                            locked_doorway_target = None
                            next_state, reason = "ROOM_SIDE_TURN", "room_side_gap_trigger_ready_before_corridor_motion"
                        else:
                            next_state, reason = "FOLLOW_CORRIDOR", (
                                "room_side_gap_health_gate_blocked_before:"
                                + str(health_gate.get("primary_blocking_reason"))
                            )
                if room_side_gap_transition:
                    pass
                elif (
                    args.enable_doorway_control
                    and doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY"
                    and room_zone_reached
                ):
                    if doorway_before_status.get("opening_center_estimated"):
                        locked_doorway_target = write_doorway_target(
                            doorway,
                            locked=True,
                            commit_depth_m=0.0,
                            args=args,
                        )
                        locked_doorway_target["doorway_verify_mode"] = "approach_threshold_before_commit"
                        item["locked_doorway_target"] = locked_doorway_target
                    else:
                        locked_doorway_target = None
                        item["doorway_before_suppressed_reason"] = (
                            "doorway_opening_center_not_estimated:"
                            + str(doorway_before_status.get("reason"))
                        )
                    if locked_doorway_target is not None and doorway_approach_target_available(locked_doorway_target):
                        next_state, reason = "DOORWAY_VERIFY", (
                            "doorway_ready_needs_verify_before_corridor_motion:"
                            + str(doorway_before_status.get("reason", "opening_center_estimated"))
                        )
                    elif (
                        doorway_latch_alignment_before.get("alignment_ready")
                        and not portal_bound_control_domain_entered
                    ):
                        next_state, reason = "DOORWAY_VERIFY", "doorway_partial_latch_alignment_ready_before_corridor_motion"
                    else:
                        if anchor is None:
                            raise RuntimeError("corridor_anchor_unavailable")
                        active_room_side_gap_alignment = (
                            committed_room_side_gap
                            if isinstance(committed_room_side_gap, dict)
                            else room_side_gap_observation
                        )
                        if (
                            args.enable_forced_room_entry_mvp
                            and isinstance(pending_forced_room_entry_opening, dict)
                            and finite_number(pending_forced_room_entry_opening.get("target_progress_m"))
                        ):
                            pending_source = str(pending_forced_room_entry_opening.get("source") or "")
                            pending_phase = str(
                                pending_forced_room_entry_opening.get("simple_phase") or "alignment"
                            ).lower()
                            target = write_anchor_progress_target(
                                anchor,
                                float(pending_forced_room_entry_opening["target_progress_m"]),
                                (
                                    f"state_machine_simple_room_entry_{pending_phase}"
                                    if pending_source == "simple_room_entry_side_gap"
                                    else "state_machine_forced_room_entry_alignment"
                                ),
                                {"pending_forced_room_entry_opening": pending_forced_room_entry_opening},
                            )
                            item["room_viewpoint_suppressed_reason"] = "forced_room_entry_alignment_pending"
                        elif (
                            isinstance(active_room_side_gap_alignment, dict)
                            and finite_number(
                                active_room_side_gap_alignment.get(
                                    "alignment_target_anchor_progress_m",
                                    active_room_side_gap_alignment.get("target_progress_m"),
                                )
                            )
                        ):
                            target = write_anchor_progress_target(
                                anchor,
                                float(
                                    active_room_side_gap_alignment.get(
                                        "alignment_target_anchor_progress_m",
                                        active_room_side_gap_alignment.get("target_progress_m"),
                                    )
                                ),
                                "state_machine_room_side_gap_alignment",
                                {"room_side_gap_observation": active_room_side_gap_alignment},
                            )
                            item["room_viewpoint_suppressed_reason"] = "room_side_gap_alignment_pending"
                        elif latched_doorway_profile is not None and not latched_doorway_profile.get("partial_latch_exhausted"):
                            target, latch_authority = write_legacy_latch_or_corridor_target(
                                anchor,
                                latched_doorway_profile,
                                args.corridor_lookahead_m,
                                portal_bound_control_domain,
                                room_zone_reached,
                            )
                            item["legacy_latch_motion_authority"] = latch_authority
                            if not latch_authority["legacy_motion_authorized"]:
                                item["room_viewpoint_suppressed_reason"] = "legacy_latch_diagnostic_only_portal_bound_domain"
                        else:
                            simple_search_active = bool(
                                args.enable_forced_room_entry_mvp
                                and args.enable_simple_room_entry
                                and room_zone_reached
                            )
                            search_lookahead_m = (
                                float(args.simple_room_entry_search_step_m)
                                if simple_search_active
                                else float(args.corridor_lookahead_m)
                            )
                            target = write_anchor_target(
                                anchor,
                                search_lookahead_m,
                                (
                                    "state_machine_simple_room_entry_gap_search"
                                    if simple_search_active
                                    else "state_machine_corridor_centerline_door_search"
                                ),
                                room_zone_active=room_zone_reached,
                            )
                            if simple_search_active:
                                item["simple_room_entry_search_step_m"] = search_lookahead_m
                        item["target"] = target
                        item["target_diagnostics_before_runner"] = target_diagnostics(target, pose_tuple(read_odom()), anchor)
                        result = run_runner(args, state, args.runner_runtime_sec, args.runner_max_steps)
                        item.update(result)
                        runner = result["runner"]
                        item["follower_status_after_runner"] = read_follower_status(args)
                        if runner_is_stuck(args, runner, state=state):
                            next_state, reason = "STUCK_RECOVERY", "runner_stuck_during_follow_corridor"
                            last_normal_state = "FOLLOW_CORRIDOR"
                        else:
                            next_state, reason = "FOLLOW_CORRIDOR", (
                                "continue_corridor_centerline_until_verified_doorway:"
                                + str(doorway_before_status.get("reason", "grid_not_valid"))
                            )
                else:
                    doorway_ready_diagnostic_only = (
                        doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY"
                        and room_zone_reached
                        and not args.enable_doorway_control
                    )
                    if doorway_ready_diagnostic_only:
                        item["doorway_before_diagnostic_only"] = True
                        item["doorway_before_control_suppressed_reason"] = "doorway_control_disabled"
                    if doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY" and not room_zone_reached:
                        item["doorway_before_suppressed_reason"] = "room_zone_not_reached"
                    elif doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY" and not doorway_ready_diagnostic_only:
                        item["doorway_before_suppressed_reason"] = (
                            "doorway_opening_center_not_estimated:"
                            + str(doorway_before_status.get("reason"))
                        )
                    if not room_zone_reached:
                        if anchor is None:
                            raise RuntimeError("corridor_anchor_unavailable")
                        current_progress_before_room_zone = (
                            metrics.get("anchor_progress_m") if isinstance(metrics, dict) else None
                        )
                        simple_room_zone_boundary_guard = bool(
                            args.enable_forced_room_entry_mvp
                            and args.enable_simple_room_entry
                            and finite_number(current_progress_before_room_zone)
                            and float(current_progress_before_room_zone)
                            + float(args.corridor_lookahead_m)
                            > float(args.room_zone_start_progress_m)
                            + float(args.goal_tolerance_m)
                            + 0.10
                        )
                        if simple_room_zone_boundary_guard:
                            boundary_target_progress = (
                                float(args.room_zone_start_progress_m)
                                + float(args.goal_tolerance_m)
                                + 0.10
                            )
                            target = write_anchor_progress_target(
                                anchor,
                                boundary_target_progress,
                                "state_machine_simple_room_entry_room_zone_boundary_guard",
                                {
                                    "room_zone_start_progress_m": float(args.room_zone_start_progress_m),
                                    "goal_tolerance_m": float(args.goal_tolerance_m),
                                    "boundary_margin_m": 0.10,
                                },
                            )
                            item["simple_room_entry_room_zone_boundary_target_progress_m"] = (
                                boundary_target_progress
                            )
                        else:
                            target = write_anchor_target(
                                anchor,
                                args.corridor_lookahead_m,
                                "state_machine_corridor_centerline_room_zone_guard",
                                room_zone_active=False,
                            )
                        item["room_viewpoint_suppressed_reason"] = "room_zone_not_reached"
                    else:
                        if anchor is None:
                            raise RuntimeError("corridor_anchor_unavailable")
                        active_room_side_gap_alignment = (
                            committed_room_side_gap
                            if isinstance(committed_room_side_gap, dict)
                            else room_side_gap_observation
                        )
                        if (
                            args.enable_forced_room_entry_mvp
                            and isinstance(pending_forced_room_entry_opening, dict)
                            and finite_number(pending_forced_room_entry_opening.get("target_progress_m"))
                        ):
                            pending_source = str(pending_forced_room_entry_opening.get("source") or "")
                            pending_phase = str(
                                pending_forced_room_entry_opening.get("simple_phase") or "alignment"
                            ).lower()
                            target = write_anchor_progress_target(
                                anchor,
                                float(pending_forced_room_entry_opening["target_progress_m"]),
                                (
                                    f"state_machine_simple_room_entry_{pending_phase}"
                                    if pending_source == "simple_room_entry_side_gap"
                                    else "state_machine_forced_room_entry_alignment"
                                ),
                                {"pending_forced_room_entry_opening": pending_forced_room_entry_opening},
                            )
                            item["room_viewpoint_suppressed_reason"] = "forced_room_entry_alignment_pending"
                        elif (
                            isinstance(active_room_side_gap_alignment, dict)
                            and finite_number(
                                active_room_side_gap_alignment.get(
                                    "alignment_target_anchor_progress_m",
                                    active_room_side_gap_alignment.get("target_progress_m"),
                                )
                            )
                        ):
                            target = write_anchor_progress_target(
                                anchor,
                                float(
                                    active_room_side_gap_alignment.get(
                                        "alignment_target_anchor_progress_m",
                                        active_room_side_gap_alignment.get("target_progress_m"),
                                    )
                                ),
                                "state_machine_room_side_gap_alignment",
                                {"room_side_gap_observation": active_room_side_gap_alignment},
                            )
                            item["room_viewpoint_suppressed_reason"] = "room_side_gap_alignment_pending"
                        elif latched_doorway_profile is not None and not latched_doorway_profile.get("partial_latch_exhausted"):
                            target, latch_authority = write_legacy_latch_or_corridor_target(
                                anchor,
                                latched_doorway_profile,
                                args.corridor_lookahead_m,
                                portal_bound_control_domain,
                                room_zone_reached,
                            )
                            item["legacy_latch_motion_authority"] = latch_authority
                            if latch_authority["legacy_motion_authorized"]:
                                item["room_viewpoint_suppressed_reason"] = "doorway_partial_latch_alignment_pending"
                            else:
                                item["room_viewpoint_suppressed_reason"] = "legacy_latch_diagnostic_only_portal_bound_domain"
                        else:
                            simple_search_active = bool(
                                args.enable_forced_room_entry_mvp
                                and args.enable_simple_room_entry
                                and room_zone_reached
                            )
                            search_lookahead_m = (
                                float(args.simple_room_entry_search_step_m)
                                if simple_search_active
                                else float(args.corridor_lookahead_m)
                            )
                            target = write_anchor_target(
                                anchor,
                                search_lookahead_m,
                                (
                                    "state_machine_simple_room_entry_gap_search"
                                    if simple_search_active
                                    else "state_machine_corridor_centerline_door_search"
                                ),
                                room_zone_active=room_zone_reached,
                            )
                            item["room_viewpoint_suppressed_reason"] = "doorway_not_confirmed"
                            if simple_search_active:
                                item["simple_room_entry_search_step_m"] = search_lookahead_m
                    item["target"] = target
                    item["target_diagnostics_before_runner"] = target_diagnostics(target, pose_tuple(read_odom()), anchor)
                    result = run_runner(args, state, args.runner_runtime_sec, args.runner_max_steps)
                    item.update(result)
                    runner = result["runner"]
                    item["follower_status_after_runner"] = read_follower_status(args)
                    if runner_is_stuck(args, runner, state=state):
                        next_state, reason = "STUCK_RECOVERY", "runner_stuck_during_follow_corridor"
                        last_normal_state = "FOLLOW_CORRIDOR"
                    else:
                        doorway_after = read_json(DOORWAY_PATH)
                        item["doorway_after"] = doorway_after
                        room_zone_reached_after = room_zone_reached
                        room_zone_reason_after = "ANCHOR_UNAVAILABLE"
                        if anchor is not None:
                            pose_after = pose_tuple(read_odom())
                            metrics_after = anchor_metrics(anchor, pose_after)
                            current_anchor_progress_after_m = metrics_after.get("anchor_progress_m")
                            effective_progress_after_m = effective_room_zone_progress_m(
                                anchor,
                                current_anchor_progress_after_m,
                            )
                            room_zone_reached_after = bool(
                                effective_progress_after_m is not None
                                and effective_progress_after_m >= args.room_zone_start_progress_m
                            )
                            room_zone_reason_after = (
                                "EFFECTIVE_ROOM_ZONE_PROGRESS_AT_OR_ABOVE_ROOM_ZONE_START"
                                if room_zone_reached_after else "EFFECTIVE_ROOM_ZONE_PROGRESS_BELOW_ROOM_ZONE_START"
                            )
                            item["corridor_anchor_metrics_after"] = metrics_after
                            item["target_diagnostics_after_runner"] = target_diagnostics(target, pose_after, anchor)
                            item["current_anchor_progress_m_after"] = current_anchor_progress_after_m
                            item["effective_room_zone_progress_m_after"] = effective_progress_after_m
                            item["room_zone_progress_offset_m_after"] = anchor.get("room_zone_progress_offset_m", 0.0)
                            item["room_zone_reached_after"] = room_zone_reached_after
                        room_zone_state_audit.publish(
                            room_zone_reached_after,
                            "FOLLOW_CORRIDOR",
                            room_zone_reason_after,
                            current_anchor_progress_m=(
                                metrics_after.get("anchor_progress_m") if anchor is not None else None
                            ),
                            effective_room_zone_progress_m=(
                                effective_room_zone_progress_m(anchor, metrics_after.get("anchor_progress_m"))
                                if anchor is not None else None
                            ),
                            room_zone_progress_offset_m=(
                                anchor.get("room_zone_progress_offset_m", 0.0) if anchor is not None else None
                            ),
                        )
                        doorway_after_status = (
                            doorway_opening_status(doorway_after)
                            if doorway_after.get("final_decision") == "DOORWAY_CANDIDATE_READY"
                            else {}
                        )
                        if doorway_after_status:
                            item["doorway_after_opening_status"] = doorway_after_status
                        if args.enable_forced_room_entry_mvp:
                            if args.enable_simple_room_entry:
                                forced_trigger_after = select_simple_room_entry_opening(
                                    doorway_after,
                                    room_zone_reached_after,
                                    (item.get("corridor_anchor_metrics_after") or {}).get("anchor_progress_m"),
                                    args,
                                    pending_forced_room_entry_opening,
                                )
                            else:
                                forced_trigger_after = select_forced_room_entry_opening(
                                    doorway_after,
                                    room_zone_reached_after,
                                    (item.get("corridor_anchor_metrics_after") or {}).get("anchor_progress_m"),
                                    args,
                                    pending_forced_room_entry_opening,
                                    latched_doorway_profile,
                                    forced_room_entry_opening_observation_cache,
                                )
                            pending_forced_room_entry_opening = forced_trigger_after.get("pending_forced_opening")
                            if forced_trigger_after.get("stop_observe_required"):
                                stop_result = (
                                    publish_stop_at_door(args)
                                    if args.execute
                                    else {"execute": False, "reason": "dry_run_no_stop_command"}
                                )
                                forced_trigger_after["stop_observe_result"] = stop_result
                                item["simple_room_entry_stop_observe_after"] = stop_result
                                if isinstance(pending_forced_room_entry_opening, dict):
                                    pending_forced_room_entry_opening["stop_observe_completed"] = True
                            last_forced_room_entry_trigger_status = forced_trigger_after
                            item["forced_room_entry_trigger_after"] = forced_trigger_after
                            if (
                                forced_trigger_after.get("trigger_ready")
                                and not forced_room_entry_done
                                and not args.room_side_turn_validation_v1
                            ):
                                forced_room_entry_result = execute_forced_room_entry_mvp(
                                    args,
                                    forced_trigger_after,
                                    side_gap_visual_audit,
                                    forced_entry_linear_x_clamped_from,
                                    follower,
                                    pose_after,
                                )
                                forced_room_entry_done = bool(forced_room_entry_result.get("forced_entry_done"))
                                item["forced_room_entry_result"] = forced_room_entry_result
                                raise ForcedRoomEntryMVPFinished(
                                    forced_room_entry_result,
                                    bool(args.forced_entry_stop_after_done),
                                )
                            elif forced_trigger_after.get("trigger_ready") and args.room_side_turn_validation_v1:
                                item["forced_room_entry_execution_suppressed_reason"] = (
                                    "room_side_turn_validation_v1_forbids_entry"
                                )
                        if anchor is not None and room_zone_reached_after:
                            side_gap_segment_audit_after = build_side_gap_segment_switch_audit_events(
                                doorway_after,
                                pose_after,
                                anchor,
                                iteration,
                                "after_corridor_motion",
                                room_zone_reached_after,
                                args,
                                side_gap_segment_previous_selected,
                            )
                            side_gap_segment_switch_audit_events.extend(side_gap_segment_audit_after)
                            item["side_gap_segment_switch_audit_after"] = side_gap_segment_audit_after
                        if anchor is not None:
                            door_landmark_observations_after = update_door_landmark_tracks(
                                door_tracks,
                                door_track_next_ids,
                                doorway_after,
                                {},
                                pose_after,
                                anchor,
                                iteration,
                                "after_corridor_motion",
                                args.door_landmark_corridor_half_width_m,
                            )
                            if door_landmark_observations_after:
                                annotate_door_cue_actionability(
                                    door_landmark_observations_after,
                                    room_zone_reached_after,
                                    args.room_zone_start_progress_m,
                                    metrics_after.get("anchor_progress_m") if isinstance(metrics_after, dict) else None,
                                )
                                side_gap_nav_after = build_side_gap_nav_debug_candidates(
                                    door_landmark_observations_after,
                                    side_gap_nav_cache,
                                    anchor,
                                    pose_after,
                                    args,
                                    room_zone_reached_after,
                                    metrics_after.get("anchor_progress_m") if isinstance(metrics_after, dict) else None,
                                )
                                side_gap_nav_debug_records.extend(side_gap_nav_after)
                                if side_gap_nav_after:
                                    item["side_gap_nav_debug_after"] = side_gap_nav_after
                                    if side_gap_visual_audit is not None:
                                        visual_events_after = [
                                            event
                                            for event in (side_gap_visual_audit.capture(candidate) for candidate in side_gap_nav_after)
                                            if event is not None
                                        ]
                                        if visual_events_after:
                                            item["side_gap_visual_audit_events_after"] = visual_events_after
                                            first_side_gap_visual_audit_event = (
                                                first_side_gap_visual_audit_event or visual_events_after[0]
                                            )
                                item["door_landmark_observations_after"] = door_landmark_observations_after
                        doorway_latch_alignment_after: Dict[str, Any] = {}
                        if room_zone_reached_after and anchor is not None:
                            latched_doorway_profile, doorway_latch_status_after = maybe_latch_doorway_profile(
                                args,
                                anchor,
                                pose_tuple(read_odom()),
                                doorway_after,
                                latched_doorway_profile,
                            )
                            item["doorway_latch_status_after"] = doorway_latch_status_after
                            if latched_doorway_profile is not None:
                                item["latched_doorway_profile"] = latched_doorway_profile
                                doorway_latch_alignment_after = doorway_latch_alignment(
                                    latched_doorway_profile,
                                    anchor,
                                    pose_tuple(read_odom()),
                                    args,
                                )
                                extend_partial_latch_target_if_needed(
                                    latched_doorway_profile,
                                    doorway_latch_alignment_after,
                                    args,
                                )
                                item["doorway_latch_alignment_after"] = doorway_latch_alignment_after
                        room_side_gap_transition_after = False
                        if (
                            args.enable_room_side_gap_trigger
                            and room_zone_reached_after
                            and anchor is not None
                        ):
                            pose_for_gap_after = pose_tuple(read_odom())
                            gap_candidate_after = room_side_gap_candidate(doorway_after, args)
                            room_side_gap_observation, gap_status_after = update_room_side_gap_observation(
                                args,
                                room_side_gap_observation,
                                gap_candidate_after,
                                anchor,
                                pose_for_gap_after,
                            )
                            item["room_side_gap_status_after"] = gap_status_after
                            current_progress_after = gap_status_after.get("anchor_progress_m")
                            if not finite_number(current_progress_after) and isinstance(metrics_after, dict):
                                current_progress_after = metrics_after.get("anchor_progress_m")
                            gap_landmark_observations_after = update_door_landmark_tracks(
                                door_tracks,
                                door_track_next_ids,
                                doorway_after,
                                gap_status_after,
                                pose_for_gap_after,
                                anchor,
                                iteration,
                                "room_side_gap_after",
                                args.door_landmark_corridor_half_width_m,
                            )
                            if gap_landmark_observations_after:
                                annotate_door_cue_actionability(
                                    gap_landmark_observations_after,
                                    room_zone_reached_after,
                                    args.room_zone_start_progress_m,
                                    float(current_progress_after)
                                    if finite_number(current_progress_after)
                                    else None,
                                )
                                side_gap_nav_after_gap = build_side_gap_nav_debug_candidates(
                                    gap_landmark_observations_after,
                                    side_gap_nav_cache,
                                    anchor,
                                    pose_for_gap_after,
                                    args,
                                    room_zone_reached_after,
                                    float(current_progress_after)
                                    if finite_number(current_progress_after)
                                    else None,
                                )
                                side_gap_nav_debug_records.extend(side_gap_nav_after_gap)
                                if side_gap_nav_after_gap:
                                    item.setdefault("side_gap_nav_debug_after", []).extend(side_gap_nav_after_gap)
                                item.setdefault("door_landmark_observations_after", []).extend(
                                    gap_landmark_observations_after
                                )
                            committed_room_side_gap, commit_status_after = update_committed_room_side_gap(
                                args,
                                committed_room_side_gap,
                                gap_status_after,
                                float(current_progress_after) if finite_number(current_progress_after) else None,
                            )
                            item["committed_room_side_gap_status_after"] = commit_status_after
                            committed_trigger_after = committed_room_side_gap_trigger_status(
                                args,
                                committed_room_side_gap,
                                float(current_progress_after) if finite_number(current_progress_after) else None,
                            )
                            item["committed_room_side_gap_trigger_after"] = committed_trigger_after
                            trigger_gap_after = (
                                committed_trigger_after if committed_trigger_after.get("trigger_ready") else gap_status_after
                            )
                            opening_decision_after = select_follow_corridor_opening_action(
                                doorway_control_enabled=bool(args.enable_doorway_control),
                                fully_bounded_doorway_ready=bool(
                                    doorway_after.get("final_decision") == "DOORWAY_CANDIDATE_READY"
                                    and doorway_after_status.get("opening_center_estimated")
                                ),
                                room_side_gap_enabled=bool(args.enable_room_side_gap_trigger),
                                room_side_gap_trigger_ready=bool(trigger_gap_after.get("trigger_ready")),
                                forced_room_entry_enabled=bool(args.enable_forced_room_entry_mvp),
                            )
                            item["follow_corridor_opening_decision_after"] = opening_decision_after
                            if opening_decision_after.get("action") == "ROOM_SIDE_TURN":
                                health_gate_after = room_side_turn_health.snapshot()
                                item["room_side_turn_health_gate_after"] = health_gate_after
                                item["stop_at_door_result"] = publish_stop_at_door(args)
                                room_side_gap_transition_after = True
                                if health_gate_after.get("ready"):
                                    pending_room_side_gap = trigger_gap_after
                                    latched_doorway_profile = None
                                    locked_doorway_target = None
                                    next_state, reason = "ROOM_SIDE_TURN", "room_side_gap_trigger_ready_after_corridor_motion"
                                else:
                                    next_state, reason = "FOLLOW_CORRIDOR", (
                                        "room_side_gap_health_gate_blocked_after:"
                                        + str(health_gate_after.get("primary_blocking_reason"))
                                    )
                        if room_side_gap_transition_after:
                            pass
                        elif (
                            args.enable_doorway_control
                            and doorway_after.get("final_decision") == "DOORWAY_CANDIDATE_READY"
                            and room_zone_reached_after
                        ):
                            if doorway_after_status.get("opening_center_estimated"):
                                locked_doorway_target = write_doorway_target(
                                    doorway_after,
                                    locked=True,
                                    commit_depth_m=0.0,
                                    args=args,
                                )
                                locked_doorway_target["doorway_verify_mode"] = "approach_threshold_before_commit"
                                item["locked_doorway_target"] = locked_doorway_target
                            else:
                                locked_doorway_target = None
                                item["doorway_after_suppressed_reason"] = (
                                    "doorway_opening_center_not_estimated:"
                                    + str(doorway_after_status.get("reason"))
                                )
                            if locked_doorway_target is not None and doorway_approach_target_available(locked_doorway_target):
                                next_state, reason = "DOORWAY_VERIFY", (
                                    "doorway_ready_needs_verify_after_corridor_motion:"
                                    + str(doorway_after_status.get("reason", "opening_center_estimated"))
                                )
                            elif doorway_latch_alignment_after.get("alignment_ready"):
                                next_state, reason = "DOORWAY_VERIFY", "doorway_partial_latch_alignment_ready_after_corridor_motion"
                            else:
                                if anchor is not None:
                                    item["restored_corridor_target_after_rejected_doorway"] = write_anchor_target(
                                        anchor,
                                        args.corridor_lookahead_m,
                                        "state_machine_corridor_centerline_after_rejected_doorway",
                                        room_zone_active=room_zone_reached_after,
                                    )
                                next_state, reason = "FOLLOW_CORRIDOR", (
                                    "continue_corridor_centerline_until_verified_doorway:"
                                    + str(doorway_after_status.get("reason", "grid_not_valid"))
                                )
                        else:
                            doorway_after_ready_diagnostic_only = (
                                doorway_after.get("final_decision") == "DOORWAY_CANDIDATE_READY"
                                and room_zone_reached_after
                                and not args.enable_doorway_control
                            )
                            if doorway_after_ready_diagnostic_only:
                                item["doorway_after_diagnostic_only"] = True
                                item["doorway_after_control_suppressed_reason"] = "doorway_control_disabled"
                            if doorway_after.get("final_decision") == "DOORWAY_CANDIDATE_READY" and not room_zone_reached_after:
                                item["doorway_after_suppressed_reason"] = "room_zone_not_reached"
                            elif doorway_after.get("final_decision") == "DOORWAY_CANDIDATE_READY" and not doorway_after_ready_diagnostic_only:
                                item["doorway_after_suppressed_reason"] = (
                                    "doorway_opening_center_not_estimated:"
                                    + str(doorway_after_status.get("reason"))
                                )
                            if doorway_latch_alignment_after.get("alignment_ready"):
                                next_state, reason = "DOORWAY_VERIFY", "doorway_partial_latch_alignment_ready_after_corridor_motion"
                            else:
                                next_state, reason = "FOLLOW_CORRIDOR", "continue_corridor_search"

            elif state == "DETECT_ROOM_DOOR":
                write_stage("inside_corridor", "navigation_state_machine.py")
                time.sleep(args.corridor_doorway_check_wait_sec)
                doorway = read_json(DOORWAY_PATH)
                item["doorway"] = doorway
                doorway_status = (
                    doorway_opening_status(doorway) if doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY" else {}
                )
                if doorway_status:
                    item["doorway_opening_status"] = doorway_status
                if doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY":
                    locked_doorway_target = write_doorway_target(
                        doorway,
                        locked=True,
                        commit_depth_m=args.enter_room_commit_depth_m,
                        args=args,
                    )
                    item["target"] = locked_doorway_target
                    if doorway_target_grid_validated(locked_doorway_target):
                        next_state, reason = "ENTER_ROOM", "doorway_target_written"
                    else:
                        locked_doorway_target = None
                        next_state, reason = "FOLLOW_CORRIDOR", "doorway_target_grid_validation_failed"
                elif doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY":
                    next_state, reason = "FOLLOW_CORRIDOR", (
                        "doorway_opening_center_not_estimated:"
                        + str(doorway_status.get("reason"))
                    )
                elif locked_doorway_target is not None:
                    write_json(TARGET_PATH, locked_doorway_target)
                    item["target"] = locked_doorway_target
                    item["doorway_lock_used"] = True
                    next_state, reason = "ENTER_ROOM", "locked_doorway_target_reused"
                else:
                    next_state, reason = "FOLLOW_CORRIDOR", f"doorway_not_ready:{doorway.get('final_decision')}"

            elif state == "DOORWAY_VERIFY" and args.room_side_turn_validation_v1:
                write_stage("doorway_verify_observation_only", "navigation_state_machine.py")
                locked_doorway_target = None
                item["doorway_verify_observation_only"] = True
                item["enter_room_forbidden_by_validation_v1"] = True
                item["entry_target_generated"] = False
                item["doorway_observation"] = read_json(DOORWAY_PATH)
                item["room_side_gap_observation_after_turn"] = room_side_gap_candidate(
                    item["doorway_observation"],
                    args,
                )
                item["room_side_turn_health_at_doorway_verify"] = room_side_turn_health.snapshot()
                item["safe_stop"] = publish_stop_at_door(args)
                next_state, reason = "DONE", "room_side_turn_validation_v1_complete_at_doorway_verify"

            elif state == "DOORWAY_VERIFY":
                write_stage("doorway_verify", "navigation_state_machine.py")
                doorway = read_json(DOORWAY_PATH)
                item["doorway_before"] = doorway
                doorway_status = (
                    doorway_opening_status(doorway) if doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY" else {}
                )
                if doorway_status:
                    item["doorway_before_opening_status"] = doorway_status
                if doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY":
                    approach_target = write_doorway_target(
                        doorway,
                        locked=True,
                        commit_depth_m=0.0,
                        args=args,
                    )
                    approach_target["doorway_verify_mode"] = "approach_threshold_before_commit"
                    locked_doorway_target = approach_target
                    item["target"] = approach_target
                    item["target_diagnostics_before_runner"] = target_diagnostics(
                        approach_target,
                        pose_tuple(read_odom()),
                        anchor,
                    )
                    result = run_runner(args, state, args.doorway_verify_runtime_sec, args.doorway_verify_max_steps)
                    item.update(result)
                    runner = result["runner"]
                    item["follower_status_after_runner"] = read_follower_status(args)
                    doorway_after = read_json(DOORWAY_PATH)
                    item["doorway_after"] = doorway_after
                    if doorway_after.get("final_decision") == "DOORWAY_CANDIDATE_READY":
                        doorway_after_status = doorway_opening_status(doorway_after)
                        item["doorway_after_opening_status"] = doorway_after_status
                        commit_target = write_doorway_target(
                            doorway_after,
                            locked=True,
                            commit_depth_m=args.enter_room_commit_depth_m,
                            args=args,
                        )
                        item["locked_doorway_target"] = commit_target
                        if doorway_after_status.get("opening_center_estimated") and doorway_target_grid_validated(commit_target):
                            locked_doorway_target = commit_target
                            next_state, reason = "ENTER_ROOM", "doorway_verified_grid_valid_enter_room_target"
                        elif runner_is_stuck(args, runner, state=state):
                            next_state, reason = "STUCK_RECOVERY", "runner_stuck_during_doorway_verify"
                            last_normal_state = "DOORWAY_VERIFY"
                        else:
                            locked_doorway_target = approach_target
                            next_state, reason = "DOORWAY_VERIFY", (
                                "doorway_verify_continue:"
                                + str(doorway_after_status.get("reason", "grid_not_valid"))
                            )
                    elif runner_is_stuck(args, runner, state=state):
                        next_state, reason = "STUCK_RECOVERY", "runner_stuck_during_doorway_verify"
                        last_normal_state = "DOORWAY_VERIFY"
                    else:
                        next_state, reason = "FOLLOW_CORRIDOR", f"doorway_lost_during_verify:{doorway_after.get('final_decision')}"
                elif latched_doorway_profile is not None and portal_bound_control_domain_entered:
                    item["latched_doorway_profile"] = latched_doorway_profile
                    item["legacy_latch_motion_authority"] = {
                        **legacy_latch_motion_authority(
                            latched_doorway_profile,
                            {"active": True},
                        ),
                        "reason": "PORTAL_BOUND_DOMAIN_DEAUTHORIZES_LEGACY_LATCH_IN_DOORWAY_VERIFY",
                    }
                    next_state, reason = "FOLLOW_CORRIDOR", "legacy_latch_diagnostic_only_portal_bound_domain"
                elif latched_doorway_profile is not None:
                    if anchor is None:
                        raise RuntimeError("corridor_anchor_unavailable")
                    alignment = doorway_latch_alignment(
                        latched_doorway_profile,
                        anchor,
                        pose_tuple(read_odom()),
                        args,
                    )
                    extend_partial_latch_target_if_needed(
                        latched_doorway_profile,
                        alignment,
                        args,
                    )
                    item["latched_doorway_profile"] = latched_doorway_profile
                    item["doorway_latch_alignment"] = alignment
                    target = write_anchor_progress_target(
                        anchor,
                        float(latched_doorway_profile["target_anchor_progress_m"]),
                        "state_machine_doorway_latch_alignment",
                        {"latched_doorway_profile": latched_doorway_profile},
                    )
                    item["target"] = target
                    item["target_diagnostics_before_runner"] = target_diagnostics(target, pose_tuple(read_odom()), anchor)
                    if latched_doorway_profile.get("partial_latch_exhausted"):
                        next_state, reason = "FOLLOW_CORRIDOR", "doorway_partial_latch_exhausted_continue_corridor_search"
                    elif alignment.get("alignment_ready"):
                        next_state, reason = "DOORWAY_VERIFY", "doorway_partial_latch_alignment_hold_wait_for_center"
                    else:
                        result = run_runner(args, state, args.doorway_verify_runtime_sec, args.doorway_verify_max_steps)
                        item.update(result)
                        runner = result["runner"]
                        item["follower_status_after_runner"] = read_follower_status(args)
                        if runner_is_stuck(args, runner, state=state):
                            next_state, reason = "STUCK_RECOVERY", "runner_stuck_during_doorway_verify"
                            last_normal_state = "DOORWAY_VERIFY"
                        else:
                            next_state, reason = "DOORWAY_VERIFY", "doorway_partial_latch_alignment_continue"
                elif locked_doorway_target is not None and not doorway_status:
                    write_json(TARGET_PATH, locked_doorway_target)
                    item["target"] = locked_doorway_target
                    item["doorway_lock_used"] = True
                    result = run_runner(args, state, args.doorway_verify_runtime_sec, args.doorway_verify_max_steps)
                    item.update(result)
                    next_state, reason = "DOORWAY_VERIFY", "doorway_verify_locked_target_reused"
                elif doorway.get("final_decision") == "DOORWAY_CANDIDATE_READY":
                    next_state, reason = "FOLLOW_CORRIDOR", (
                        "doorway_opening_center_not_estimated:"
                        + str(doorway_status.get("reason"))
                    )
                else:
                    next_state, reason = "FOLLOW_CORRIDOR", f"doorway_verify_not_ready:{doorway.get('final_decision')}"

            elif state == "ROOM_SIDE_TURN":
                write_stage("room_side_turn_validation", "navigation_state_machine.py", {"room_side_gap_turn": pending_room_side_gap})
                if not isinstance(pending_room_side_gap, dict) or pending_room_side_gap.get("door_side") not in {"left", "right"}:
                    next_state, reason = "FOLLOW_CORRIDOR", "room_side_gap_turn_unavailable"
                else:
                    item["room_side_gap_trigger"] = pending_room_side_gap
                    pre_turn_health = room_side_turn_health.snapshot()
                    pre_turn_grid = (pre_turn_health.get("streams") or {}).get("local_grid", {})
                    pre_turn_grid_count = int(pre_turn_grid.get("count") or 0)
                    item["room_side_turn_health_before_execute"] = pre_turn_health
                    turn_result = publish_room_side_turn(
                        args,
                        str(pending_room_side_gap["door_side"]),
                        room_side_turn_health,
                    )
                    item["room_side_gap_turn"] = turn_result
                    post_observation: Dict[str, Any] = {
                        "stop_reason": "turn_not_complete_no_post_observation",
                        "fresh_grid_count": 0,
                        "required_fresh_grid_count": int(args.room_side_turn_post_fresh_grid_count),
                        "grid_safe_for_navigation": False,
                        "fresh_side_observation": False,
                    }
                    if turn_result.get("target_reached") and turn_result.get("turn_yaw_sufficient"):
                        post_observation = wait_for_post_turn_observation(
                            args,
                            room_side_turn_health,
                            pre_turn_grid_count,
                        )
                    item["post_turn_observation"] = post_observation
                    transition = post_turn_transition(
                        turn_stop_reason=str(turn_result.get("stop_reason") or "unknown"),
                        minimum_yaw_sufficient=bool(turn_result.get("turn_yaw_sufficient")),
                        target_reached=bool(turn_result.get("target_reached")),
                        fresh_grid_count=int(post_observation.get("fresh_grid_count") or 0),
                        required_fresh_grids=int(post_observation.get("required_fresh_grid_count") or 1),
                        grid_safe_for_navigation=bool(post_observation.get("grid_safe_for_navigation")),
                        fresh_side_observation=bool(post_observation.get("fresh_side_observation")),
                    )
                    item["room_side_turn_transition_decision"] = transition
                    item["entry_target_generated"] = False
                    item["enter_room_forbidden_by_validation_v1"] = True
                    locked_doorway_target = None
                    pending_room_side_gap = None
                    room_side_gap_observation = None
                    committed_room_side_gap = None
                    next_state = str(transition["next_state"])
                    reason = str(transition["reason"])

            elif state == "ENTER_ROOM":
                write_stage("room_entry", "navigation_state_machine.py")
                if locked_doorway_target is not None:
                    write_json(TARGET_PATH, locked_doorway_target)
                    item["doorway_lock_used"] = True
                    item["target"] = locked_doorway_target
                target_before_runner = locked_doorway_target if locked_doorway_target is not None else read_json(TARGET_PATH)
                item["target_diagnostics_before_runner"] = target_diagnostics(
                    target_before_runner,
                    pose_tuple(read_odom()),
                    anchor,
                )
                item["room_entry_target_danger_before_runner"] = room_entry_target_danger(
                    item["target_diagnostics_before_runner"],
                    args,
                )
                if item["room_entry_target_danger_before_runner"].get("dangerous"):
                    locked_doorway_target = None
                    next_state, reason = "FOLLOW_CORRIDOR", (
                        "enter_room_target_geometry_dangerous_before_runner:"
                        + str(item["room_entry_target_danger_before_runner"].get("reason"))
                    )
                else:
                    result = run_runner(args, state, args.enter_room_runtime_sec, args.enter_room_max_steps)
                    item.update(result)
                    runner = result["runner"]
                    item["follower_status_after_runner"] = read_follower_status(args)
                    displacement = runner.get("observed_straight_line_displacement_m")
                    pose = pose_tuple(read_odom())
                    target = read_json(TARGET_PATH)
                    distance_to_room_entry = target_distance_from_pose(target, pose)
                    item["room_entry_target_after"] = target
                    item["distance_to_room_entry_target_m"] = distance_to_room_entry
                    item["target_diagnostics_after_runner"] = target_diagnostics(target, pose, anchor)
                    item["room_entry_target_danger_after_runner"] = room_entry_target_danger(
                        item["target_diagnostics_after_runner"],
                        args,
                    )
                    if item["room_entry_target_danger_after_runner"].get("dangerous"):
                        next_state, reason = "STUCK_RECOVERY", (
                            "enter_room_target_geometry_dangerous:"
                            + str(item["room_entry_target_danger_after_runner"].get("reason"))
                        )
                        last_normal_state = "DOORWAY_VERIFY"
                    elif runner_is_stuck(args, runner, state=state):
                        next_state, reason = "STUCK_RECOVERY", "runner_stuck_during_enter_room"
                        last_normal_state = "DOORWAY_VERIFY"
                    elif finite_number(distance_to_room_entry) and float(distance_to_room_entry) <= args.room_entry_target_tolerance_m:
                        next_state, reason = "CONFIRM_INSIDE_ROOM", "room_entry_target_reached"
                    elif count_state(trace, "ENTER_ROOM") + 1 >= args.room_entry_max_attempts:
                        next_state, reason = "FOLLOW_CORRIDOR", "room_entry_attempt_limit_reached"
                    elif finite_number(displacement) and float(displacement) >= args.enter_room_progress_threshold_m:
                        next_state, reason = "ENTER_ROOM", "entered_room_progress_observed_but_target_not_reached"
                    else:
                        next_state, reason = "FOLLOW_CORRIDOR", "enter_room_progress_insufficient"

            elif state == "CONFIRM_INSIDE_ROOM":
                write_stage("room_confirm", "navigation_state_machine.py")
                pose = pose_tuple(read_odom())
                target = read_json(TARGET_PATH)
                distance_to_room_entry = target_distance_from_pose(target, pose)
                item["room_entry_target"] = target
                item["distance_to_room_entry_target_m"] = distance_to_room_entry
                item["scan_inputs"] = room_scan_summary()
                room_side_gap_confirm = {}
                if target.get("subgoal_source") == "room_side_gap_entry" and anchor is not None:
                    metrics = anchor_metrics(anchor, pose)
                    room_side_gap_confirm = {
                        "anchor_lateral_error_m": metrics.get("anchor_lateral_error_m"),
                        "min_required_abs_lateral_offset_m": float(args.room_side_gap_inside_min_lateral_offset_m),
                        "inside_by_lateral_offset": bool(
                            finite_number(metrics.get("anchor_lateral_error_m"))
                            and abs(float(metrics["anchor_lateral_error_m"])) >= float(args.room_side_gap_inside_min_lateral_offset_m)
                        ),
                    }
                    item["room_side_gap_inside_confirm"] = room_side_gap_confirm
                if (
                    target.get("subgoal_source") == "room_side_gap_entry"
                    and room_side_gap_confirm
                    and not room_side_gap_confirm.get("inside_by_lateral_offset")
                ):
                    locked_doorway_target = None
                    next_state, reason = "FOLLOW_CORRIDOR", "room_side_gap_inside_lateral_offset_insufficient"
                elif finite_number(distance_to_room_entry) and float(distance_to_room_entry) <= args.room_entry_target_tolerance_m:
                    next_state, reason = "SCAN_ROOM_FOR_DANGER", "room_entry_confirmed_by_target_distance"
                else:
                    next_state, reason = "ENTER_ROOM", "room_entry_not_confirmed_continue"

            elif state == "SCAN_ROOM_FOR_DANGER":
                write_stage("room_search", "navigation_state_machine.py")
                time.sleep(args.room_scan_wait_sec)
                scan = room_scan_summary()
                item["scan_inputs"] = scan
                if scan.get("danger_source_visible"):
                    next_state, reason = "DONE", "danger_source_visible"
                    final_decision = "STATE_MACHINE_DANGER_SOURCE_VISIBLE"
                elif count_state(trace, "SCAN_ROOM_FOR_DANGER") + 1 >= args.room_scan_max_cycles:
                    next_state, reason = "DONE", "room_entry_confirmed_scan_no_danger_visible"
                    final_decision = "STATE_MACHINE_ROOM_ENTRY_CONFIRMED_SCAN_NO_DANGER_VISIBLE"
                else:
                    next_state, reason = "SCAN_ROOM_FOR_DANGER", "continue_room_scan"

            elif state == "NEXT_PORTAL_DISPATCH":
                write_stage("next_portal_dispatch", "navigation_state_machine.py")
                time.sleep(args.corridor_doorway_check_wait_sec)
                context = post_room_return_context if isinstance(post_room_return_context, dict) else {}
                returned_record = context.get("returned_portal_record") if isinstance(context.get("returned_portal_record"), dict) else None
                if anchor is None or returned_record is None:
                    next_state, reason = "FOLLOW_CORRIDOR", "next_portal_dispatch_context_unavailable_resume_deeper"
                    post_room_return_context = None
                    post_room_return_resume_marker = True
                else:
                    fresh_snapshot = portal_candidate_authority.snapshot()
                    candidate_accounting = {
                        "fresh_frame_sequence": fresh_snapshot.get("last_portal_frame_sequence"),
                        "fresh_frame_stamp": fresh_snapshot.get("last_portal_source_stamp"),
                        "return_reference_sequence": context.get("minimum_fresh_portal_frame_sequence"),
                        "returned_portal_progress_m": returned_record.get("corridor_progress_m"),
                        "total_effect_candidates_seen": 0,
                        "total_prepared_candidates": 0,
                        "prepared_candidates": [],
                        "preparation_rejections": [],
                        "selected_index": None,
                        "final_branch": None,
                    }
                    fresh_prepared = prepare_fresh_post_return_portal_candidates(
                        fresh_snapshot,
                        context.get("minimum_fresh_portal_frame_sequence"),
                        odom_cache,
                        pose_tuple(read_odom()),
                        anchor,
                        args.p_pre_upstream_tangent_m,
                        candidate_accounting,
                    )
                    dispatch = next_portal_dispatch_decision(
                        fresh_prepared,
                        visited_portals,
                        returned_record,
                        anchor,
                        candidate_accounting,
                    )
                    item["next_portal_dispatch"] = dispatch
                    if dispatch.get("state") == "NEXT_PORTAL_DISPATCH_AMBIGUOUS_CURRENT_STATION":
                        next_state, reason = "DONE", "next_portal_dispatch_ambiguous_current_station"
                        final_decision = "NEXT_PORTAL_DISPATCH_AMBIGUOUS_CURRENT_STATION"
                    elif dispatch.get("state") == "NEXT_PORTAL_DISPATCH_COMMIT":
                        prepared = dispatch["candidate"]
                        reset_state = reset_portal_entry_attempt_state()
                        portal_candidate_authority.reset_post_room_return_candidate()
                        portal_bound_candidate = portal_candidate_authority.commit_post_room_return_candidate(prepared["candidate"])
                        portal_g14_shadow_target = prepared["target"]
                        portal_g14_p_pre_admissibility = reset_state["portal_g14_p_pre_admissibility"]
                        portal_g14_p_pre_switch = reset_state["portal_g14_p_pre_switch"]
                        portal_g14_p_pre_runner = reset_state["portal_g14_p_pre_runner"]
                        room_search_v2_result = reset_state["room_search_v2_result"]
                        item["next_portal_dispatch_reset"] = list(reset_state)
                        item["next_portal_dispatch_committed_portal"] = portal_bound_candidate
                        post_room_return_context = None
                        post_room_return_resume_marker = True
                        next_state, reason = "FOLLOW_CORRIDOR", "next_portal_dispatch_unique_fresh_unvisited_opposite_portal"
                    else:
                        portal_candidate_authority.reset_post_room_return_candidate()
                        reset_state = reset_portal_entry_attempt_state()
                        portal_bound_candidate = reset_state["portal_bound_candidate"]
                        portal_g14_shadow_target = reset_state["portal_g14_shadow_target"]
                        portal_g14_p_pre_admissibility = reset_state["portal_g14_p_pre_admissibility"]
                        portal_g14_p_pre_switch = reset_state["portal_g14_p_pre_switch"]
                        portal_g14_p_pre_runner = reset_state["portal_g14_p_pre_runner"]
                        room_search_v2_result = reset_state["room_search_v2_result"]
                        post_room_return_context = None
                        post_room_return_resume_marker = True
                        next_state, reason = "FOLLOW_CORRIDOR", "next_portal_dispatch_no_fresh_opposite_resume_deeper"

            elif state == "STUCK_RECOVERY":
                recovery_count += 1
                write_stage("stuck_recovery", "navigation_state_machine.py")
                item["recovery"] = publish_zero_and_backoff(args)
                if recovery_count > args.max_recovery_count:
                    next_state, reason = "FAILED", "max_recovery_count_exceeded"
                    final_decision = "STATE_MACHINE_FAILED_STUCK_RECOVERY_LIMIT"
                else:
                    next_state, reason = last_normal_state, "recovery_backoff_complete"

            else:
                next_state, reason = "FAILED", f"unknown_state:{state}"
                final_decision = "STATE_MACHINE_FAILED_UNKNOWN_STATE"
        except RoomReturnNextPortalDispatch as room_return_dispatch:
            item["portal_g14_p_pre_terminal"] = room_return_dispatch.details
            next_state, reason = "NEXT_PORTAL_DISPATCH", "door_return_anchor_reached_next_portal_dispatch"
        except PortalG14PPreFinished as p_pre_finish:
            item["portal_g14_p_pre_terminal"] = p_pre_finish.details
            next_state, reason = "DONE", p_pre_finish.reason
            final_decision = p_pre_finish.final_decision
            debug_stop_reason = p_pre_finish.reason
            debug_stop_details = p_pre_finish.details
        except FastDebugPortalBoundCandidateStop as portal_stop:
            item["portal_bound_door_candidate"] = portal_stop.candidate
            is_g14_stop = g14_shadow_stop_eligible(portal_g14_shadow_target)
            next_state, reason = "DONE", (
                "stop_after_first_portal_g14_shadow_target" if is_g14_stop else "stop_after_first_portal_bound_candidate"
            )
            final_decision = (
                "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_PORTAL_G14_SHADOW_TARGET"
                if is_g14_stop else "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_PORTAL_BOUND_CANDIDATE"
            )
            debug_stop_reason = reason
            debug_stop_details = {
                "portal_bound_candidate": portal_stop.candidate,
                "portal_g14_shadow_target": portal_g14_shadow_target,
                "authoritative_control_source": DOORWAY_CANDIDATE_AUTHORITY,
                "actual_motion_authority_count": 0,
            }
        except ForcedRoomEntryMVPFinished as forced_entry_finish:
            forced_room_entry_result = forced_entry_finish.result
            forced_room_entry_done = bool(forced_room_entry_result.get("forced_entry_done"))
            item["forced_room_entry_result"] = forced_room_entry_result
            final_decision = str(forced_room_entry_result.get("final_decision"))
            debug_stop_reason = str(forced_room_entry_result.get("debug_stop_reason"))
            debug_stop_details = {
                "forced_room_entry_summary_path": str(FORCED_ENTRY_SUMMARY_PATH),
                "entry_side": forced_room_entry_result.get("entry_side"),
                "turn_completed": forced_room_entry_result.get("turn_completed"),
                "forward_completed": forced_room_entry_result.get("forward_completed"),
            }
            if forced_room_entry_done and not forced_entry_finish.stop_after_done:
                next_state, reason = "FOLLOW_CORRIDOR", "forced_room_entry_done_continue_requested"
            else:
                next_state, reason = "DONE", debug_stop_reason
        except FastDebugLocalEntryTargetStop as local_entry_stop:
            item["local_entry_target_debug_stop_candidate"] = local_entry_stop.candidate
            next_state, reason = "DONE", "stop_after_first_local_entry_target"
            final_decision = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_LOCAL_ENTRY_TARGET"
            debug_stop_reason = "stop_after_first_local_entry_target"
            debug_stop_details = {"first_valid_local_entry_target": local_entry_stop.candidate}
        except FastDebugSideGapVisualAuditStop as visual_stop:
            item["side_gap_visual_audit_stop_event"] = visual_stop.event
            next_state, reason = "DONE", "stop_after_first_side_gap_visual_audit"
            final_decision = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_SIDE_GAP_VISUAL_AUDIT"
            debug_stop_reason = "stop_after_first_side_gap_visual_audit"
            debug_stop_details = {"first_side_gap_visual_audit_event": visual_stop.event}
        except FastDebugSideGapSegmentSwitchAuditStop as switch_stop:
            item["side_gap_segment_switch_audit_stop_event"] = switch_stop.event
            next_state, reason = "DONE", "stop_after_first_side_gap_segment_switch_audit"
            final_decision = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_SIDE_GAP_SEGMENT_SWITCH_AUDIT"
            debug_stop_reason = "stop_after_first_side_gap_segment_switch_audit"
            debug_stop_details = {
                "side_gap_segment_switch_audit_count": len(side_gap_segment_switch_audit_events),
                "switch_event": switch_stop.event,
            }
        except FastDebugSideGapNavStop as side_gap_stop:
            item["side_gap_nav_debug_stop_candidate"] = side_gap_stop.candidate
            next_state, reason = "DONE", "stop_after_first_side_gap_nav_debug"
            final_decision = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_SIDE_GAP_NAV_DEBUG"
            debug_stop_reason = "stop_after_first_side_gap_nav_debug"
            debug_stop_details = {
                "side_gap_nav_debug_count": len(side_gap_nav_debug_records),
                "side_gap_nav_debug_confirmed_count": sum(
                    1 for candidate in side_gap_nav_debug_records if candidate.get("lightweight_confirmed")
                ),
                "first_side_gap_nav_debug": side_gap_stop.candidate,
            }
        except FastDebugDoorCueStop as cue_stop:
            item["door_cue_ahead_observations"] = cue_stop.observations
            next_state, reason = "DONE", cue_stop.stop_reason
            final_decision = (
                "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_ACTIONABLE_DOOR_CUE"
                if cue_stop.stop_reason == "stop_after_first_actionable_door_cue"
                else "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_DOOR_CUE"
            )
            debug_stop_reason = cue_stop.stop_reason
            debug_stop_details = {
                "raw_door_cue_ahead_count": len(cue_stop.observations),
                "actionable_door_cue_count": sum(
                    1 for observation in cue_stop.observations if observation.get("actionable_door_cue")
                ),
                "first_door_cue_ahead": cue_stop.observations[0] if cue_stop.observations else None,
                "first_actionable_door_cue": next(
                    (observation for observation in cue_stop.observations if observation.get("actionable_door_cue")),
                    None,
                ),
            }
            if (
                cue_stop.stop_reason == "stop_after_first_actionable_door_cue"
                and args.fast_debug
                and args.stop_after_first_actionable_door_cue
                and args.observe_after_actionable_door_cue
                and cue_stop.observations
            ):
                trigger_cue = cue_stop.observations[0]
                door_observe_debug = run_door_observe_debug(
                    args,
                    trigger_cue,
                    door_tracks,
                    door_track_next_ids,
                    anchor,
                    iteration,
                )
                door_observe_debug_count = 1
                door_observe_confirmed = bool(door_observe_debug.get("confirmed_door_after_observe"))
                first_door_observe_debug = door_observe_debug
                item["door_observe_debug"] = door_observe_debug
                write_json(
                    DOOR_OBSERVE_DEBUG_PATH,
                    {"latest_iteration": iteration, "updated_wall_time_sec": time.time(), **door_observe_debug},
                )
                debug_stop_details["door_observe_debug_count"] = door_observe_debug_count
                debug_stop_details["door_observe_confirmed"] = door_observe_confirmed
        except RoomLocalValidationEvidenceInsufficient as exc:
            item["online_evidence_status"] = "ONLINE_EVIDENCE_INSUFFICIENT"
            item["online_evidence_reason"] = str(exc)
            next_state, reason = "FAILED", "room_local_online_evidence_insufficient"
            final_decision = "ONLINE_EVIDENCE_INSUFFICIENT"
        except RoomLocalValidationAbort as exc:
            item["validation_abort_reason"] = str(exc)
            next_state, reason = "FAILED", "room_local_validation_abort"
            final_decision = "ROOM_LOCAL_ONLINE_VALIDATION_ABORT"
        except Exception as exc:
            item["exception"] = repr(exc)
            exception_type = type(exc).__name__
            exception_message = str(exc)
            exception_traceback = traceback.format_exc()
            item["exception_type"] = exception_type
            item["exception_message"] = exception_message
            item["exception_traceback"] = exception_traceback
            next_state, reason = "FAILED", "exception"
            final_decision = f"STATE_MACHINE_EXCEPTION:{type(exc).__name__}"

        # A defensive central invariant: no legacy or externally resumed path
        # may enter a doorway-control state in this identity-only phase.
        if next_state in {"ROOM_SIDE_TURN", "DOORWAY_VERIFY", "ENTER_ROOM"}:
            item["legacy_doorway_transition_blocked"] = {
                "attempted_next_state": next_state,
                "reason": reason,
                "disposition": legacy_doorway_control_disable_reason,
            }
            locked_doorway_target = None
            next_state, reason = "FOLLOW_CORRIDOR", "legacy_doorway_transition_deauthorized"

        if args.room_side_turn_validation_v1 and next_state == "ENTER_ROOM":
            locked_doorway_target = None
            item["enter_room_transition_blocked"] = True
            item["enter_room_transition_original_reason"] = reason
            item["entry_target_generated"] = False
            item["safe_stop_after_enter_room_guard"] = publish_stop_at_door(args)
            next_state, reason = "FAILED", "validation_v1_enter_room_transition_forbidden"
            final_decision = "STATE_MACHINE_VALIDATION_V1_ENTER_ROOM_GUARD"

        item["next_state"] = next_state
        item["transition_reason"] = reason
        landmark_payload: Dict[str, Any] = {}
        try:
            item_anchor_progress = trace_item_anchor_progress(item)
            if finite_number(item_anchor_progress):
                latest_door_anchor_progress = float(item_anchor_progress)
            landmark_payload = door_landmark_debug_payload(
                door_tracks,
                anchor,
                latest_door_anchor_progress,
            )
            landmark_summary = landmark_payload["summary"]
            item["door_landmark_debug_summary"] = landmark_summary
            write_json(
                DOOR_LANDMARK_DEBUG_PATH,
                {
                    "latest_iteration": iteration,
                    "updated_wall_time_sec": time.time(),
                    "coordinate_frame": "team_livox_odom",
                    "odom_topic": ODOM_TOPIC,
                    "diagnostic_only": True,
                    "controls_robot": False,
                    "controls_next_state": False,
                    "controls_committed_room_side_gap": False,
                    "door_landmark_debug_summary": landmark_summary,
                    "raw_door_cue_ahead_count": landmark_summary.get("raw_door_cue_ahead_count"),
                    "actionable_door_cue_count": landmark_summary.get("actionable_door_cue_count"),
                    "rejected_before_room_zone_count": landmark_summary.get("rejected_before_room_zone_count"),
                    "rejected_broad_bilateral_opening_count": landmark_summary.get(
                        "rejected_broad_bilateral_opening_count"
                    ),
                    "left_tracks": landmark_payload["left_tracks"],
                    "right_tracks": landmark_payload["right_tracks"],
                    "position_stable_left_tracks": landmark_payload["position_stable_left_tracks"],
                    "position_stable_right_tracks": landmark_payload["position_stable_right_tracks"],
                    "navigation_ready_left_tracks": landmark_payload["navigation_ready_left_tracks"],
                    "navigation_ready_right_tracks": landmark_payload["navigation_ready_right_tracks"],
                    "navigation_ready_tracks": landmark_payload["navigation_ready_tracks"],
                    "approach_candidate_valid_tracks": landmark_payload["approach_candidate_valid_tracks"],
                    "door_cue_ahead_observations": landmark_payload["door_cue_ahead_observations"],
                    "first_door_cue_ahead": landmark_payload["first_door_cue_ahead"],
                    "actionable_door_cue_observations": landmark_payload["actionable_door_cue_observations"],
                    "first_actionable_door_cue": landmark_payload["first_actionable_door_cue"],
                    "passed_or_expired_tracks": landmark_payload["passed_or_expired_tracks"],
                    "duplicate_suspect_tracks": landmark_payload["duplicate_suspect_tracks"],
                    "all_stable_door_landmarks": landmark_payload["all_stable_door_landmarks"],
                },
            )
        except Exception as debug_exc:
            item["door_landmark_debug_error"] = repr(debug_exc)

        landmark_summary = landmark_payload.get("summary") if isinstance(landmark_payload, dict) else {}
        # Legacy landmark tracks are diagnostic-only after 090 and therefore
        # cannot contribute a formal candidate count.
        approach_candidate_valid_count = 1 if portal_bound_candidate is not None else 0
        door_cue_ahead_count = int(landmark_summary.get("door_cue_ahead_count") or 0)
        raw_door_cue_ahead_count = int(landmark_summary.get("raw_door_cue_ahead_count") or 0)
        first_door_cue_ahead = landmark_summary.get("first_door_cue_ahead")
        actionable_door_cue_count = int(landmark_summary.get("actionable_door_cue_count") or 0)
        first_actionable_door_cue = landmark_summary.get("first_actionable_door_cue")
        rejected_before_room_zone_count = int(landmark_summary.get("rejected_before_room_zone_count") or 0)
        rejected_broad_bilateral_opening_count = int(
            landmark_summary.get("rejected_broad_bilateral_opening_count") or 0
        )
        side_gap_nav_debug_count = len(side_gap_nav_debug_records)
        side_gap_nav_debug_confirmed_count = sum(
            1 for candidate in side_gap_nav_debug_records if candidate.get("lightweight_confirmed")
        )
        first_side_gap_nav_debug = side_gap_nav_debug_records[0] if side_gap_nav_debug_records else None
        try:
            latest_side_gap = side_gap_nav_debug_records[-1] if side_gap_nav_debug_records else None
            write_json(
                SIDE_GAP_NAV_DEBUG_PATH,
                {
                    "latest_iteration": iteration,
                    "updated_wall_time_sec": time.time(),
                    "diagnostic_only": True,
                    "controls_robot": False,
                    "controls_next_state": False,
                    "writes_real_target": False,
                    "calls_runner": False,
                    "coordinate_frame": "team_livox_odom",
                    "room_zone_reached": latest_side_gap.get("room_zone_reached") if latest_side_gap else False,
                    "side_gap_nav_debug_count": side_gap_nav_debug_count,
                    "side_gap_nav_debug_confirmed_count": side_gap_nav_debug_confirmed_count,
                    "candidates": side_gap_nav_debug_records,
                    "first_confirmed_candidate": next(
                        (candidate for candidate in side_gap_nav_debug_records if candidate.get("lightweight_confirmed")),
                        None,
                    ),
                },
            )
        except Exception as side_gap_debug_exc:
            item["side_gap_nav_debug_error"] = repr(side_gap_debug_exc)
        segment_switch_summary = side_gap_segment_switch_summary(side_gap_segment_switch_audit_events)
        side_gap_segment_switch_audit_count = int(segment_switch_summary["audit_event_count"])
        left_selected_switch_count = int(segment_switch_summary["left_switch_count"])
        right_selected_switch_count = int(segment_switch_summary["right_switch_count"])
        max_left_selected_center_progress_jump_m = segment_switch_summary[
            "max_left_selected_center_progress_jump_m"
        ]
        first_large_left_switch = segment_switch_summary["first_large_left_switch"]
        try:
            write_json(
                SIDE_GAP_SEGMENT_SWITCH_AUDIT_PATH,
                {
                    "latest_iteration": iteration,
                    "updated_wall_time_sec": time.time(),
                    "diagnostic_only": True,
                    "controls_robot": False,
                    "controls_next_state": False,
                    "writes_real_target": False,
                    "calls_runner": False,
                    "coordinate_frame": "team_livox_odom",
                    "events": side_gap_segment_switch_audit_events,
                    "side_gap_segment_switch_summary": segment_switch_summary,
                },
            )
        except Exception as switch_audit_exc:
            item["side_gap_segment_switch_audit_error"] = repr(switch_audit_exc)
        side_gap_visual_audit_event_count = 0
        side_gap_visual_audit_broad_event_count = 0
        if side_gap_visual_audit is not None:
            try:
                visual_manifest = side_gap_visual_audit.manifest()
                side_gap_visual_audit_event_count = int(visual_manifest["event_count"])
                side_gap_visual_audit_broad_event_count = int(visual_manifest["broad_opening_event_count"])
                if first_side_gap_visual_audit_event is None and visual_manifest.get("events_summary"):
                    first_side_gap_visual_audit_event = visual_manifest["events_summary"][0]
                write_json(SIDE_GAP_VISUAL_AUDIT_MANIFEST_PATH, visual_manifest)
            except Exception as visual_manifest_exc:
                item["side_gap_visual_audit_error"] = repr(visual_manifest_exc)
        local_entry_target_debug_count = len(local_entry_debug_events)
        local_entry_target_valid_count = sum(
            int(event.get("entry_target_valid_count", 0)) for event in local_entry_debug_events
        )
        first_valid_local_entry_target = next(
            (event.get("first_valid_entry_target") for event in local_entry_debug_events if event.get("first_valid_entry_target")),
            None,
        )
        last_local_entry_event = local_entry_debug_events[-1] if local_entry_debug_events else latest_local_entry_debug
        last_local_entry_event = last_local_entry_event if isinstance(last_local_entry_event, dict) else {}
        last_left_candidate = last_local_entry_event.get("left_entry_candidate")
        last_right_candidate = last_local_entry_event.get("right_entry_candidate")
        last_left_candidate = last_left_candidate if isinstance(last_left_candidate, dict) else {}
        last_right_candidate = last_right_candidate if isinstance(last_right_candidate, dict) else {}
        last_event_summary = {
            "event_id": last_local_entry_event.get("event_id"),
            "room_zone_reached": last_local_entry_event.get("room_zone_reached"),
            "current_anchor_progress_m": last_local_entry_event.get("current_anchor_progress_m"),
            "left_reject_reason": last_left_candidate.get("reject_reason"),
            "right_reject_reason": last_right_candidate.get("reject_reason"),
            "left_sector_in_grid": last_left_candidate.get("sector_in_grid"),
            "right_sector_in_grid": last_right_candidate.get("sector_in_grid"),
            "left_sector_total_cell_count": last_left_candidate.get("sector_total_cell_count"),
            "right_sector_total_cell_count": last_right_candidate.get("sector_total_cell_count"),
            "left_sector_free_cell_count": last_left_candidate.get("sector_free_cell_count"),
            "right_sector_free_cell_count": last_right_candidate.get("sector_free_cell_count"),
        }
        if args.enable_local_free_space_entry_debug:
            try:
                write_json(
                    LOCAL_ENTRY_DEBUG_PATH,
                    {
                        "diagnostic_only": True,
                        "controls_robot": False,
                        "controls_next_state": False,
                        "writes_real_target": False,
                        "calls_runner": False,
                        "grid_available": last_local_entry_event.get("grid_available"),
                        "grid_source": last_local_entry_event.get("grid_source", GRID_TOPIC),
                        "grid_frame": last_local_entry_event.get("grid_frame"),
                        "grid_width": last_local_entry_event.get("grid_width"),
                        "grid_height": last_local_entry_event.get("grid_height"),
                        "grid_resolution_m": last_local_entry_event.get("grid_resolution_m"),
                        "grid_origin": last_local_entry_event.get("grid_origin"),
                        "grid_bounds_base_xy": last_local_entry_event.get("grid_bounds_base_xy"),
                        "room_zone_reached": last_local_entry_event.get("room_zone_reached"),
                        "current_anchor_progress_m": last_local_entry_event.get("current_anchor_progress_m"),
                        "event_count": local_entry_target_debug_count,
                        "latest": latest_local_entry_debug,
                        "events": local_entry_debug_events,
                        "entry_target_valid_count": local_entry_target_valid_count,
                        "first_valid_entry_target": first_valid_local_entry_target,
                        "local_entry_target_debug_count": local_entry_target_debug_count,
                        "local_entry_target_valid_count": local_entry_target_valid_count,
                        "first_valid_local_entry_target": first_valid_local_entry_target,
                        "last_event_summary": last_event_summary,
                        "reject_reasons": last_local_entry_event.get("reject_reasons", []),
                    },
                )
            except Exception as local_entry_write_exc:
                item["local_free_space_entry_debug_write_error"] = repr(local_entry_write_exc)
        approach_point_debug_count = 0
        approach_point_debug_pass_count = 0
        first_approach_point_debug = None
        item["approach_point_debug_summary"] = {
            "approach_point_debug_count": 0,
            "approach_point_debug_pass_count": 0,
            "first_approach_point_debug": None,
            "disabled_reason": legacy_doorway_control_disable_reason,
        }
        debug_stop_final: Optional[str] = None
        debug_stop_payload: Dict[str, Any] = {}
        runner = item.get("runner") if isinstance(item.get("runner"), dict) else None
        if debug_stop_reason in {"stop_after_first_portal_bound_candidate", "stop_after_first_portal_g14_shadow_target"}:
            debug_stop_final = (
                "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_PORTAL_G14_SHADOW_TARGET"
                if debug_stop_reason == "stop_after_first_portal_g14_shadow_target"
                else "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_PORTAL_BOUND_CANDIDATE"
            )
            debug_stop_payload = dict(debug_stop_details)
        elif debug_stop_reason == "stop_after_first_side_gap_visual_audit":
            debug_stop_final = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_SIDE_GAP_VISUAL_AUDIT"
            debug_stop_payload = dict(debug_stop_details)
        elif debug_stop_reason == "stop_after_first_side_gap_segment_switch_audit":
            debug_stop_final = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_SIDE_GAP_SEGMENT_SWITCH_AUDIT"
            debug_stop_payload = dict(debug_stop_details)
        elif debug_stop_reason == "stop_after_first_side_gap_nav_debug":
            debug_stop_final = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_SIDE_GAP_NAV_DEBUG"
            debug_stop_payload = dict(debug_stop_details)
        elif debug_stop_reason == "stop_after_first_local_entry_target":
            debug_stop_final = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_LOCAL_ENTRY_TARGET"
            debug_stop_payload = dict(debug_stop_details)
        elif debug_stop_reason in {"stop_after_first_door_cue", "stop_after_first_actionable_door_cue"}:
            debug_stop_final = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_DOOR_CUE"
            debug_stop_payload = dict(debug_stop_details)
            if debug_stop_reason == "stop_after_first_actionable_door_cue":
                debug_stop_final = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_ACTIONABLE_DOOR_CUE"
        elif args.stop_after_first_runner and runner is not None:
            debug_stop_final = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_RUNNER"
            debug_stop_reason = "stop_after_first_runner"
            debug_stop_payload = {
                "runner_final_decision": runner.get("runner_final_decision"),
                "timeout_trigger": runner.get("timeout_trigger"),
                "sim_elapsed": runner.get("run_sim_elapsed_sec"),
                "wall_elapsed": runner.get("run_wall_elapsed_sec"),
                "real_time_factor_estimate": runner.get("real_time_factor_estimate"),
            }
        elif args.stop_after_first_approach_candidate and portal_bound_candidate is not None:
            debug_stop_final = "STATE_MACHINE_DEBUG_STOP_AFTER_FIRST_APPROACH_CANDIDATE"
            debug_stop_reason = "stop_after_first_approach_candidate"
            debug_stop_payload = {
                "approach_candidate_valid_count": approach_candidate_valid_count,
                "candidate": portal_bound_candidate,
            }
        elif args.stop_after_door_landmark_debug:
            debug_stop_final = "STATE_MACHINE_DEBUG_STOP_AFTER_DOOR_LANDMARK"
            debug_stop_reason = "stop_after_door_landmark_debug"
            debug_stop_payload = {
                "approach_candidate_valid_count": approach_candidate_valid_count,
                "door_landmark_debug_written": bool(landmark_payload),
            }

        if debug_stop_final is not None:
            debug_stop_payload["safe_stop"] = publish_fast_debug_stop(args)
            item["debug_stop"] = debug_stop_payload
            next_state = "DONE"
            reason = str(debug_stop_reason)
            item["next_state"] = next_state
            item["transition_reason"] = reason
            final_decision = debug_stop_final
            debug_stop_details = debug_stop_payload
        if args.room_entry_observer_log:
            try:
                observer_path = Path(args.room_entry_observer_log)
                observer_path.parent.mkdir(parents=True, exist_ok=True)
                observer_record = {
                    "record_kind": "room_entry_iteration",
                    "wall_time_sec": time.time(),
                    "item": item,
                }
                with observer_path.open("a", encoding="utf-8") as observer_file:
                    observer_file.write(json.dumps(observer_record, sort_keys=True) + "\n")
            except Exception as observer_exc:
                item["room_entry_observer_log_error"] = repr(observer_exc)
        trace.append(item)
        state = next_state

    if args.enable_forced_room_entry_mvp and forced_room_entry_result is None:
        last_forced_status = last_forced_room_entry_trigger_status or {
            "reject_reason": "room_zone_not_reached",
            "room_zone_reached": False,
            "current_anchor_progress_m": None,
        }
        forced_room_entry_result = {
            "diagnostic_only": False,
            "controls_robot": True,
            "calls_runner": False,
            "writes_real_target": False,
            "forced_entry_enabled": True,
            "forced_entry_triggered": False,
            "forced_entry_done": False,
            "reject_reason": last_forced_status.get("reject_reason"),
            "room_zone_reached": last_forced_status.get("room_zone_reached"),
            "current_anchor_progress_m": last_forced_status.get("current_anchor_progress_m"),
            "pending_forced_opening": last_forced_status.get("pending_forced_opening"),
            "quality_rejected_candidates": last_forced_status.get("quality_rejected_candidates", []),
            "alignment_error_m": last_forced_status.get("alignment_error_m"),
            "forward_linear_x": float(args.forced_entry_linear_x),
            "forward_duration_sec": float(args.forced_entry_forward_duration_sec),
            "linear_x_clamped_from": forced_entry_linear_x_clamped_from,
            "linear_x_clamped_to": float(args.forced_entry_linear_x) if forced_entry_linear_x_clamped_from is not None else None,
            "final_decision": "STATE_MACHINE_FORCED_ROOM_ENTRY_NOT_TRIGGERED",
            "debug_stop_reason": None,
            "exception_type": None,
            "exception_message": None,
            "exception_traceback": None,
        }
        write_json(FORCED_ENTRY_SUMMARY_PATH, forced_room_entry_result)

    if state == "DONE" and final_decision == "STATE_MACHINE_NAVIGATION_INCOMPLETE":
        final_decision = "STATE_MACHINE_DONE"
    elif (
        state != "DONE"
        and not forced_room_entry_done
        and not final_decision.startswith("STATE_MACHINE_FAILED")
        and not final_decision.startswith("STATE_MACHINE_TARGET_PREP_FAILED")
        and "EXCEPTION" not in final_decision
    ):
        final_decision = "STATE_MACHINE_NAVIGATION_INCOMPLETE"

    summary = {
        "final_decision": final_decision,
        "final_state": state,
        "execute": bool(args.execute),
        "fast_debug_enabled": bool(args.fast_debug),
        "debug_run_label": args.debug_run_label,
        "debug_stop_reason": debug_stop_reason,
        "debug_stop_details": debug_stop_details,
        "doorway_candidate_authority": DOORWAY_CANDIDATE_AUTHORITY,
        "committed_portal_candidate": portal_bound_candidate,
        "portal_g14_shadow_target": portal_g14_shadow_target,
        "portal_g14_p_pre_admissibility": portal_g14_p_pre_admissibility,
        "portal_g14_p_pre_switch": portal_g14_p_pre_switch,
        "portal_g14_p_pre_runner": portal_g14_p_pre_runner,
        "room_search_v2": room_search_v2_result,
        "visited_portals": visited_portals,
        "target_authority_timeline": build_target_authority_timeline(trace),
        "exception_type": exception_type or (forced_room_entry_result.get("exception_type") if forced_room_entry_result else None),
        "exception_message": exception_message or (forced_room_entry_result.get("exception_message") if forced_room_entry_result else None),
        "exception_traceback": exception_traceback or (forced_room_entry_result.get("exception_traceback") if forced_room_entry_result else None),
        "debug_max_iterations": args.debug_max_iterations,
        "runner_runtime_sec_effective": args.runner_runtime_sec,
        "runner_wall_watchdog_sec_effective": args.runner_wall_watchdog_sec,
        "approach_candidate_valid_count": approach_candidate_valid_count,
        "approach_point_debug_count": approach_point_debug_count,
        "approach_point_debug_pass_count": approach_point_debug_pass_count,
        "first_approach_point_debug": first_approach_point_debug,
        "door_cue_ahead_count": door_cue_ahead_count,
        "raw_door_cue_ahead_count": raw_door_cue_ahead_count,
        "first_door_cue_ahead": first_door_cue_ahead,
        "actionable_door_cue_count": actionable_door_cue_count,
        "first_actionable_door_cue": first_actionable_door_cue,
        "rejected_before_room_zone_count": rejected_before_room_zone_count,
        "rejected_broad_bilateral_opening_count": rejected_broad_bilateral_opening_count,
        "door_observe_debug_count": door_observe_debug_count,
        "door_observe_confirmed": door_observe_confirmed,
        "first_door_observe_debug": first_door_observe_debug,
        "side_gap_nav_debug_count": side_gap_nav_debug_count,
        "side_gap_nav_debug_confirmed_count": side_gap_nav_debug_confirmed_count,
        "first_side_gap_nav_debug": first_side_gap_nav_debug,
        "side_gap_segment_switch_audit_count": side_gap_segment_switch_audit_count,
        "left_selected_switch_count": left_selected_switch_count,
        "right_selected_switch_count": right_selected_switch_count,
        "max_left_selected_center_progress_jump_m": max_left_selected_center_progress_jump_m,
        "first_large_left_switch": first_large_left_switch,
        "side_gap_visual_audit_enabled": bool(args.enable_side_gap_visual_coordinate_audit),
        "side_gap_visual_audit_event_count": side_gap_visual_audit_event_count,
        "side_gap_visual_audit_broad_event_count": side_gap_visual_audit_broad_event_count,
        "first_side_gap_visual_audit_event": first_side_gap_visual_audit_event,
        "side_gap_visual_audit_manifest_path": str(SIDE_GAP_VISUAL_AUDIT_MANIFEST_PATH),
        "local_free_space_entry_debug_enabled": bool(args.enable_local_free_space_entry_debug),
        "local_entry_target_debug_count": len(local_entry_debug_events),
        "local_entry_target_valid_count": sum(
            int(event.get("entry_target_valid_count", 0)) for event in local_entry_debug_events
        ),
        "first_valid_local_entry_target": next(
            (event.get("first_valid_entry_target") for event in local_entry_debug_events if event.get("first_valid_entry_target")),
            None,
        ),
        "local_free_space_entry_debug_path": str(LOCAL_ENTRY_DEBUG_PATH),
        "forced_room_entry_mvp_enabled": bool(args.enable_forced_room_entry_mvp),
        "simple_room_entry_enabled": bool(args.enable_simple_room_entry),
        "forced_room_entry_triggered": bool(
            forced_room_entry_result and forced_room_entry_result.get("forced_entry_triggered")
        ),
        "forced_room_entry_done": bool(
            forced_room_entry_result and forced_room_entry_result.get("forced_entry_done")
        ),
        "forced_room_entry_summary_path": str(FORCED_ENTRY_SUMMARY_PATH),
        "forced_room_entry_entry_side": forced_room_entry_result.get("entry_side") if forced_room_entry_result else None,
        "forced_room_entry_forward_linear_x": (
            forced_room_entry_result.get("forward_linear_x") if forced_room_entry_result else float(args.forced_entry_linear_x)
        ),
        "forced_room_entry_forward_duration_sec": (
            forced_room_entry_result.get("forward_duration_sec")
            if forced_room_entry_result
            else float(args.forced_entry_forward_duration_sec)
        ),
        "forced_room_entry_final_decision": (
            forced_room_entry_result.get("final_decision") if forced_room_entry_result else None
        ),
        "forced_room_entry_pending_opening": (
            forced_room_entry_result.get("pending_forced_opening") if forced_room_entry_result else None
        ),
        "forced_room_entry_quality_rejected_candidates": (
            forced_room_entry_result.get("quality_rejected_candidates", []) if forced_room_entry_result else []
        ),
        "forced_room_entry_front_clearance": (
            forced_room_entry_result.get("front_clearance") if forced_room_entry_result else None
        ),
        "forced_room_entry_side_free_space_check": (
            forced_room_entry_result.get("side_free_space_check") if forced_room_entry_result else None
        ),
        "forced_room_entry_side_free_space_pass": bool(
            forced_room_entry_result and forced_room_entry_result.get("side_free_space_pass")
        ),
        "forced_room_entry_forward_stalled": bool(
            forced_room_entry_result and forced_room_entry_result.get("forward_stalled")
        ),
        "forced_room_entry_actual_forward_progress_m": (
            forced_room_entry_result.get("forward_actual_progress_m") if forced_room_entry_result else None
        ),
        "forced_room_entry_signed_lateral_displacement_m": (
            forced_room_entry_result.get("signed_entry_lateral_displacement_m")
            if forced_room_entry_result
            else None
        ),
        "forced_room_entry_side_crossing_pass": bool(
            forced_room_entry_result and forced_room_entry_result.get("side_crossing_pass")
        ),
        "completed_iterations": len(trace),
        "run_id": os.environ.get("STATE_MACHINE_RUN_ID"),
        "run_archive_dir": os.environ.get("STATE_MACHINE_RUN_ARCHIVE_DIR"),
        "room_side_turn_validation_v1": bool(args.room_side_turn_validation_v1),
        "config": vars(args),
        "state_trace": trace,
        "commands": commands,
        "recovery_count": recovery_count,
        "forbidden_sources_used": [],
        "used_gazebo_truth": False,
        "called_move_base": False,
        "sent_navigation_goal": False,
        "safe_for_navigation": False,
        "planner_ready": False,
        "autonomous_l4_allowed": False,
        "git_add_or_commit": False,
    }
    motion_audit = build_corridor_motion_audit(trace, final_decision)
    runner_timeout_audit = build_runner_timeout_audit(trace)
    summary["corridor_motion_audit_path"] = str(CORRIDOR_MOTION_AUDIT_PATH)
    summary["runner_timeout_audit_path"] = str(RUNNER_TIMEOUT_AUDIT_PATH)
    write_json(CORRIDOR_MOTION_AUDIT_PATH, motion_audit)
    write_json(RUNNER_TIMEOUT_AUDIT_PATH, runner_timeout_audit)
    write_json(SUMMARY_PATH, summary)
    write_report(summary)
    print(json.dumps({"final_decision": final_decision, "summary": str(SUMMARY_PATH)}, ensure_ascii=False))
    return 0 if state == "DONE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
