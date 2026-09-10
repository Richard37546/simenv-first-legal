"""Shared gated-odom source-time binding contract for navigation consumers.

The cache is the existing bounded exact-or-bracketed Odom contract factored
out of the state machine so a separate runner process can use the same epoch
and interpolation semantics.  It has no planner, Grid, target, or command
authority.
"""
from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


def _normalize_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


class OdomCache:
    """Thread-safe source-time odom cache with reset-safe bounded binding."""

    def __init__(
        self,
        ros: Any,
        topic: str = "/team/livox/icp_odom_gated",
        message_type: Any = None,
        history_capacity: int = 512,
    ) -> None:
        self._ros = ros
        self._topic = str(topic)
        self._condition = threading.Condition()
        self._latest_msg: Optional[Any] = None
        self._latest_stamp_sec: Optional[float] = None
        self._sequence = 0
        self._epoch_generation = 0
        self._latest_frame_identity: Optional[Tuple[str, str]] = None
        self._history: deque = deque(maxlen=max(2, int(history_capacity)))
        self._subscriber = ros.Subscriber(self._topic, message_type, self._callback, queue_size=50)

    def _callback(self, msg: Any) -> None:
        stamp_sec = float(msg.header.stamp.to_sec())
        pose = msg.pose.pose
        with self._condition:
            frame_identity = (str(msg.header.frame_id), str(msg.child_frame_id))
            if (
                (self._latest_stamp_sec is not None and stamp_sec < self._latest_stamp_sec)
                or (self._latest_frame_identity is not None and frame_identity != self._latest_frame_identity)
            ):
                self._history.clear()
                self._epoch_generation += 1
            self._latest_msg = msg
            self._latest_stamp_sec = stamp_sec
            self._latest_frame_identity = frame_identity
            self._sequence += 1
            self._history.append({
                "stamp_sec": stamp_sec,
                "callback_sequence": self._sequence,
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "yaw": _yaw_from_quat(pose.orientation),
                "frame_id": frame_identity[0],
                "child_frame_id": frame_identity[1],
                "epoch_generation": self._epoch_generation,
            })
            self._condition.notify_all()

    def close(self) -> None:
        try:
            self._subscriber.unregister()
        except Exception:
            pass

    def snapshot(self) -> Tuple[Optional[Any], int, Optional[float]]:
        with self._condition:
            return self._latest_msg, self._sequence, self._latest_stamp_sec

    def current_sequence(self) -> int:
        with self._condition:
            return self._sequence

    def current_epoch_generation(self) -> int:
        with self._condition:
            return int(self._epoch_generation)

    def history_snapshot(self) -> List[Dict[str, Any]]:
        with self._condition:
            return [dict(sample) for sample in self._history]

    def median_positive_period_sec(self) -> Optional[float]:
        history = self.history_snapshot()
        periods = [
            float(history[index]["stamp_sec"]) - float(history[index - 1]["stamp_sec"])
            for index in range(1, len(history))
            if float(history[index]["stamp_sec"]) > float(history[index - 1]["stamp_sec"])
        ]
        return float(statistics.median(periods)) if periods else None

    def pose_at_source_stamp(self, source_stamp: float) -> Dict[str, Any]:
        """Return exact or bounded-bracket source-time pose, never latest."""
        stamp = float(source_stamp)
        history = self.history_snapshot()
        if not history:
            return {"binding_valid": False, "reason": "ODOM_HISTORY_UNAVAILABLE"}
        exact = next((sample for sample in history if float(sample["stamp_sec"]) == stamp), None)
        if exact is not None:
            # An exact source-time pose is self-identifying.  A second sample
            # is only required to establish the bounded interpolation period,
            # never to validate this exact observation.
            return {
                "binding_valid": True,
                "odom_binding_method": "EXACT",
                "odom_t0_stamp": stamp,
                "odom_t1_stamp": stamp,
                "interpolation_ratio": 0.0,
                "bracket_span_sec": 0.0,
                "source_pose_x_y_yaw": [float(exact["x"]), float(exact["y"]), float(exact["yaw"])],
                "binding_contract_limit_sec": None,
                "frame_id": exact["frame_id"],
                "child_frame_id": exact["child_frame_id"],
                "odom_callback_sequence_t0": exact["callback_sequence"],
                "odom_callback_sequence_t1": exact["callback_sequence"],
                "median_positive_period_sec": self.median_positive_period_sec(),
                "odom_epoch_generation": exact["epoch_generation"],
            }
        period = self.median_positive_period_sec()
        if period is None:
            return {"binding_valid": False, "reason": "ODOM_PERIOD_UNAVAILABLE_FOR_INTERPOLATION"}
        limit = 2.0 * period
        lower = [sample for sample in history if float(sample["stamp_sec"]) < stamp]
        upper = [sample for sample in history if float(sample["stamp_sec"]) > stamp]
        if not lower or not upper:
            return {
                "binding_valid": False,
                "reason": "ODOM_SOURCE_TIME_NOT_BRACKETED",
                "binding_contract_limit_sec": limit,
                "median_positive_period_sec": period,
            }
        t0, t1 = lower[-1], upper[0]
        span = float(t1["stamp_sec"]) - float(t0["stamp_sec"])
        if span <= 0 or span > limit:
            return {
                "binding_valid": False,
                "reason": "ODOM_BRACKET_SPAN_EXCEEDS_CONTRACT",
                "bracket_span_sec": span,
                "binding_contract_limit_sec": limit,
                "median_positive_period_sec": period,
            }
        if (
            t0["frame_id"] != t1["frame_id"]
            or t0["child_frame_id"] != t1["child_frame_id"]
            or t0["epoch_generation"] != t1["epoch_generation"]
        ):
            return {
                "binding_valid": False,
                "reason": "ODOM_BRACKET_EPOCH_OR_FRAME_CONTRACT_MISMATCH",
                "binding_contract_limit_sec": limit,
                "median_positive_period_sec": period,
            }
        ratio = (stamp - float(t0["stamp_sec"])) / span
        yaw_delta = _normalize_angle(float(t1["yaw"]) - float(t0["yaw"]))
        return {
            "binding_valid": True,
            "odom_binding_method": "BRACKET_INTERPOLATION",
            "odom_t0_stamp": float(t0["stamp_sec"]),
            "odom_t1_stamp": float(t1["stamp_sec"]),
            "interpolation_ratio": ratio,
            "bracket_span_sec": span,
            "source_pose_x_y_yaw": [
                float(t0["x"]) + ratio * (float(t1["x"]) - float(t0["x"])),
                float(t0["y"]) + ratio * (float(t1["y"]) - float(t0["y"])),
                _normalize_angle(float(t0["yaw"]) + ratio * yaw_delta),
            ],
            "binding_contract_limit_sec": limit,
            "frame_id": t0["frame_id"],
            "child_frame_id": t0["child_frame_id"],
            "odom_callback_sequence_t0": t0["callback_sequence"],
            "odom_callback_sequence_t1": t1["callback_sequence"],
            "median_positive_period_sec": period,
            "odom_epoch_generation": t0["epoch_generation"],
        }

    def get(
        self,
        timeout_sec: float = 5.0,
        *,
        after_sequence: Optional[int] = None,
    ) -> Tuple[Any, int]:
        """Return cached odom or wait by simulation-time freshness semantics."""
        freshness_sim_sec = max(0.0, float(timeout_sec))
        wait_start_sim_sec = float(self._ros.Time.now().to_sec())
        while True:
            with self._condition:
                msg = self._latest_msg
                sequence = self._sequence
                stamp_sec = self._latest_stamp_sec
                has_required_new_frame = after_sequence is None or sequence > after_sequence
                if msg is not None and has_required_new_frame:
                    return msg, sequence
            if self._ros.is_shutdown():
                raise self._ros.ROSInterruptException("shutdown while waiting for gated odom")
            sim_now_sec = float(self._ros.Time.now().to_sec())
            stale_reference_sec = stamp_sec if stamp_sec is not None else wait_start_sim_sec
            if sim_now_sec > stale_reference_sec + freshness_sim_sec:
                raise RuntimeError(
                    "gated_odom_stale_in_sim_time: "
                    f"latest_stamp={stamp_sec} sim_now={sim_now_sec:.6f} allowance={freshness_sim_sec:.6f}"
                )
            with self._condition:
                self._condition.wait(timeout=0.1)
