#!/usr/bin/env python3
"""Regression coverage for the room-zone producer/gate reason contract."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "portal_room_zone_effect_gate.py"
NAVIGATION_SOURCE = HERE.parents[0] / "local_subgoal_runner_mvp" / "navigation_state_machine.py"


def load_module():
    saved = {name: sys.modules.get(name) for name in ("rospy", "std_msgs", "std_msgs.msg")}
    inserted_path = str(HERE) not in sys.path
    if inserted_path:
        sys.path.insert(0, str(HERE))
    rospy = types.ModuleType("rospy")
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = object
    sys.modules.update({"rospy": rospy, "std_msgs": std_msgs, "std_msgs.msg": std_msgs_msg})
    try:
        spec = importlib.util.spec_from_file_location("portal_room_zone_effect_gate_under_test", SOURCE)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted_path:
            sys.path.remove(str(HERE))
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def zone_state(active, reason):
    return {
        "contract_version": "p2kg12_room_zone_state_v1",
        "authoritative_source": "navigation_state_machine.FOLLOW_CORRIDOR.anchor_progress_m",
        "source_stamp": 10.0,
        "room_zone_active": active,
        "transition_sequence": 0,
        "transition_type": "NONE",
        "state_machine_stage": "FOLLOW_CORRIDOR",
        "reason": reason,
    }


class PortalRoomZoneEffectGateTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_effective_progress_reasons_from_current_producer_are_accepted(self):
        gate = self.module.PortalRoomZoneEffectGate()
        self.assertTrue(gate.add_zone_state(zone_state(
            True, "EFFECTIVE_ROOM_ZONE_PROGRESS_AT_OR_ABOVE_ROOM_ZONE_START"
        )))
        self.assertFalse(gate.zone_state_invalid)

        gate = self.module.PortalRoomZoneEffectGate()
        self.assertTrue(gate.add_zone_state(zone_state(
            False, "EFFECTIVE_ROOM_ZONE_PROGRESS_BELOW_ROOM_ZONE_START"
        )))
        self.assertFalse(gate.zone_state_invalid)

    def test_legacy_anchor_reasons_remain_accepted(self):
        gate = self.module.PortalRoomZoneEffectGate()
        self.assertTrue(gate.add_zone_state(zone_state(
            True, "ANCHOR_PROGRESS_AT_OR_ABOVE_ROOM_ZONE_START"
        )))

    def test_producer_and_gate_reason_vocabularies_have_a_checked_overlap(self):
        navigation_source = NAVIGATION_SOURCE.read_text(encoding="utf-8")
        self.assertIn("EFFECTIVE_ROOM_ZONE_PROGRESS_AT_OR_ABOVE_ROOM_ZONE_START", navigation_source)
        self.assertIn("EFFECTIVE_ROOM_ZONE_PROGRESS_BELOW_ROOM_ZONE_START", navigation_source)
        self.assertIn("EFFECTIVE_ROOM_ZONE_PROGRESS_AT_OR_ABOVE_ROOM_ZONE_START", self.module.ACTIVE_REASONS)
        self.assertIn("EFFECTIVE_ROOM_ZONE_PROGRESS_BELOW_ROOM_ZONE_START", self.module.INACTIVE_REASONS)


if __name__ == "__main__":
    unittest.main()
