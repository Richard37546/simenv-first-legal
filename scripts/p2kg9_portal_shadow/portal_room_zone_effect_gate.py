#!/usr/bin/env python3
"""Passive P2KG12 room-zone effect gate for atomic Portal frame messages."""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional

import rospy
from std_msgs.msg import String

from p2kg11_portal_frame_contract import PORTAL_FRAME_CONTRACT, validate_portal_frame_v1

PORTAL_TOPIC = "/audit/p2kg11/portal_frame"
ZONE_TOPIC = "/audit/p2kg12/room_zone_state"
EFFECT_TOPIC = "/audit/p2kg12/portal_effect_gate"
PORTAL_CONTRACT = PORTAL_FRAME_CONTRACT
ZONE_CONTRACT = "p2kg12_room_zone_state_v1"
# v2 adds the already-validated Portal run identity.  The topic is deliberately
# unchanged, but producers now emit one unambiguous schema only.
EFFECT_CONTRACT = "p2kg12_portal_effect_gate_v2"
AUTHORITATIVE_SOURCE = "navigation_state_machine.FOLLOW_CORRIDOR.anchor_progress_m"
# The original v1 producer used the raw anchor projection.  After the
# CorridorAxis handoff, the producer correctly retains a handoff-local offset
# and reports the continuous (effective) projection instead.  Both spellings
# describe the same already-authoritative room-zone boolean; accepting neither
# new spelling would revoke a valid zone state and silently prevent Portal
# candidates from reaching the existing P_pre/P_through authority.
ACTIVE_REASONS = {
    "ANCHOR_PROGRESS_AT_OR_ABOVE_ROOM_ZONE_START",
    "EFFECTIVE_ROOM_ZONE_PROGRESS_AT_OR_ABOVE_ROOM_ZONE_START",
}
INACTIVE_REASONS = {
    "ANCHOR_PROGRESS_BELOW_ROOM_ZONE_START",
    "EFFECTIVE_ROOM_ZONE_PROGRESS_BELOW_ROOM_ZONE_START",
    "ANCHOR_UNAVAILABLE",
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class PortalRoomZoneEffectGate:
    """Keeps authoritative state history; never changes Portal data or tracks."""

    def __init__(self) -> None:
        self.zone_state: Optional[Dict[str, Any]] = None
        self.zone_state_invalid = False
        self.zone_error: Optional[Dict[str, Any]] = None
        self.last_zone_source_stamp: Optional[float] = None
        self.last_zone_transition_sequence: Optional[int] = None
        self.last_portal_source_stamp: Optional[float] = None

    def invalidate_zone(self, error: str, **context: Any) -> None:
        """Discard the old active state until a fresh authoritative state arrives."""
        self.zone_state = None
        self.zone_state_invalid = True
        self.zone_error = {"error": error, **context}

    def _validate_zone_state(self, state: Any) -> Optional[str]:
        if not isinstance(state, dict):
            return "payload"
        if state.get("contract_version") != ZONE_CONTRACT:
            return "contract_version"
        if state.get("authoritative_source") != AUTHORITATIVE_SOURCE:
            return "authoritative_source"
        if not _finite_number(state.get("source_stamp")):
            return "source_stamp"
        if not isinstance(state.get("room_zone_active"), bool):
            return "room_zone_active"
        sequence = state.get("transition_sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            return "transition_sequence"
        if state.get("transition_type") not in {"ENTER", "EXIT", "NONE"}:
            return "transition_type"
        if not isinstance(state.get("state_machine_stage"), str) or not state["state_machine_stage"]:
            return "state_machine_stage"
        reason = state.get("reason")
        if not isinstance(reason, str):
            return "reason"
        if state["room_zone_active"] and reason not in ACTIVE_REASONS:
            return "reason_active_mismatch"
        if not state["room_zone_active"] and reason not in INACTIVE_REASONS:
            return "reason_inactive_mismatch"
        stamp = float(state["source_stamp"])
        if self.last_zone_source_stamp is not None and stamp < self.last_zone_source_stamp:
            return "source_stamp_regression"
        if self.last_zone_transition_sequence is not None and sequence < self.last_zone_transition_sequence:
            return "transition_sequence_regression"
        if self.zone_state is not None:
            previous = self.zone_state
            if sequence == previous["transition_sequence"]:
                if state["transition_type"] != "NONE" or state["room_zone_active"] != previous["room_zone_active"]:
                    return "transition_sequence_state_mismatch"
            elif sequence != previous["transition_sequence"] + 1:
                return "transition_sequence_gap"
            elif state["transition_type"] == "ENTER" and not (not previous["room_zone_active"] and state["room_zone_active"]):
                return "enter_state_mismatch"
            elif state["transition_type"] == "EXIT" and not (previous["room_zone_active"] and not state["room_zone_active"]):
                return "exit_state_mismatch"
            elif state["transition_type"] == "NONE":
                return "transition_type_missing"
        return None

    def add_zone_state(self, state: Any) -> bool:
        error = self._validate_zone_state(state)
        if error is not None:
            self.invalidate_zone("ZONE_STATE_INVALID", violation=error)
            return False
        self.zone_state = dict(state)
        self.zone_state_invalid = False
        self.zone_error = None
        self.last_zone_source_stamp = float(state["source_stamp"])
        self.last_zone_transition_sequence = int(state["transition_sequence"])
        return True

    @staticmethod
    def _side(portal: Any, reason: str, eligible: bool) -> Dict[str, Any]:
        return {"portal": portal, "effect_eligible": eligible, "effect_reason": reason}

    def invalid_portal_result(self, payload: Any, error: str, **context: Any) -> Dict[str, Any]:
        sequence = payload.get("frame_sequence") if isinstance(payload, dict) else None
        stamp = payload.get("source_stamp") if isinstance(payload, dict) else None
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
        return {
            "contract_version": EFFECT_CONTRACT,
            "input_valid": False,
            "input_error": {"error": error, **context},
            "run_id": run_id,
            "portal_frame_sequence": sequence,
            "portal_source_stamp": stamp,
            "room_zone_active": False,
            "room_zone_source_stamp": None,
            "room_zone_transition_sequence": None,
            "left": self._side(None, "PORTAL_CONTRACT_INVALID", False),
            "right": self._side(None, "PORTAL_CONTRACT_INVALID", False),
        }

    def evaluate(self, frame: Any) -> Dict[str, Any]:
        errors = validate_portal_frame_v1(frame)
        if errors:
            # The formal v1 validator remains the gate.  Surface its missing
            # identity failure as the v2 fail-closed reason required by the
            # downstream candidate authority.
            if "run_id" in errors:
                return {
                    "contract_version": EFFECT_CONTRACT,
                    "input_valid": False,
                    "input_error": {"error": "PORTAL_RUN_ID_MISSING_OR_INVALID", "violations": errors},
                    "run_id": frame.get("run_id") if isinstance(frame, dict) else None,
                    "portal_frame_sequence": frame.get("frame_sequence") if isinstance(frame, dict) else None,
                    "portal_source_stamp": frame.get("source_stamp") if isinstance(frame, dict) else None,
                    "room_zone_active": False,
                    "room_zone_source_stamp": None,
                    "room_zone_transition_sequence": None,
                    "left": self._side(frame.get("left") if isinstance(frame, dict) else None, "PORTAL_RUN_ID_MISSING_OR_INVALID", False),
                    "right": self._side(frame.get("right") if isinstance(frame, dict) else None, "PORTAL_RUN_ID_MISSING_OR_INVALID", False),
                }
            return self.invalid_portal_result(frame, "PORTAL_CONTRACT_INVALID", violations=errors)
        run_id = frame.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            return {
                "contract_version": EFFECT_CONTRACT,
                "input_valid": False,
                "input_error": {"error": "PORTAL_RUN_ID_MISSING_OR_INVALID"},
                "run_id": run_id,
                "portal_frame_sequence": frame["frame_sequence"],
                "portal_source_stamp": frame["source_stamp"],
                "room_zone_active": False,
                "room_zone_source_stamp": None,
                "room_zone_transition_sequence": None,
                "left": self._side(frame["left"], "PORTAL_RUN_ID_MISSING_OR_INVALID", False),
                "right": self._side(frame["right"], "PORTAL_RUN_ID_MISSING_OR_INVALID", False),
            }
        sequence = frame["frame_sequence"]
        stamp = frame["source_stamp"]
        base = {
            "contract_version": EFFECT_CONTRACT,
            "input_valid": True,
            "run_id": run_id,
            "portal_frame_sequence": sequence,
            "portal_source_stamp": stamp,
            "room_zone_active": False,
            "room_zone_source_stamp": None,
            "room_zone_transition_sequence": None,
        }
        portal_stamp = float(stamp)
        if self.last_portal_source_stamp is not None and portal_stamp <= self.last_portal_source_stamp:
            return {**base, "left": self._side(frame["left"], "TIME_ORDER_INVALID", False), "right": self._side(frame["right"], "TIME_ORDER_INVALID", False)}
        self.last_portal_source_stamp = portal_stamp
        if self.zone_state_invalid:
            return {
                **base,
                "zone_error": self.zone_error,
                "left": self._side(frame["left"], "ZONE_STATE_INVALID", False),
                "right": self._side(frame["right"], "ZONE_STATE_INVALID", False),
            }
        zone = self.zone_state
        if zone is None or float(zone["source_stamp"]) > portal_stamp:
            return {**base, "left": self._side(frame["left"], "ZONE_STATE_UNAVAILABLE", False), "right": self._side(frame["right"], "ZONE_STATE_UNAVAILABLE", False)}
        base.update({
            "room_zone_active": zone["room_zone_active"],
            "room_zone_source_stamp": zone["source_stamp"],
            "room_zone_transition_sequence": zone["transition_sequence"],
        })
        sides = {}
        for side in ("left", "right"):
            portal = frame[side]
            if not zone["room_zone_active"]:
                reason, eligible = "OUTSIDE_ROOM_GENERATION_ZONE", False
            elif portal["observation_state"] != "confirmed":
                reason, eligible = "PORTAL_NOT_CONFIRMED", False
            elif not portal["candidate_available"]:
                reason, eligible = "CANDIDATE_NOT_AVAILABLE", False
            else:
                reason, eligible = "CURRENT_FRAME_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION", True
            sides[side] = self._side(portal, reason, eligible)
        return {**base, **sides}


class GateNode:
    def __init__(self) -> None:
        self.gate = PortalRoomZoneEffectGate()
        self.publisher = rospy.Publisher(EFFECT_TOPIC, String, queue_size=10, latch=False)
        self.zone_sub = rospy.Subscriber(ZONE_TOPIC, String, self._zone_callback, queue_size=20)
        self.portal_sub = rospy.Subscriber(PORTAL_TOPIC, String, self._portal_callback, queue_size=20)

    def _zone_callback(self, message: String) -> None:
        try:
            accepted = self.gate.add_zone_state(json.loads(message.data))
            if not accepted:
                rospy.logwarn("p2kg12 invalid room-zone state revoked cached state")
        except Exception as exc:
            self.gate.invalidate_zone(
                "ZONE_STATE_INVALID",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            rospy.logwarn("p2kg12 malformed room-zone state revoked cached state")

    def _portal_callback(self, message: String) -> None:
        try:
            output = self.gate.evaluate(json.loads(message.data))
        except Exception as exc:
            output = self.gate.invalid_portal_result(
                None,
                "PORTAL_CONTRACT_INVALID",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            rospy.logwarn("p2kg12 malformed Portal frame closed gate")
        self.publisher.publish(String(data=json.dumps(output, sort_keys=True)))


def main() -> None:
    rospy.init_node("p2kg12_portal_room_zone_effect_gate", anonymous=True)
    GateNode()
    rospy.spin()


if __name__ == "__main__":
    main()
