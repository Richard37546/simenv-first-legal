#!/usr/bin/env python3
"""Focused offline contracts for far constrained-arrival recovery wiring."""

import inspect
import unittest

from test_odom_cache import FakeRos, load_module


class RoomSearchTerminalRecoverabilityTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(FakeRos())

    def handoff(self, viability, pose, anchor=(0.0, 0.0, 0.0), tolerance=0.30):
        return self.module.room_search_v2_constrained_arrival_recovery_handoff(
            viability, pose, anchor, tolerance,
        )

    def test_near_door_and_exact_boundary_preserve_existing_room_return(self):
        for pose in ((0.0, 0.0, 0.0), (0.30, 0.0, 1.57)):
            result = self.handoff("CONSTRAINED_ARRIVAL", pose)
            self.assertEqual(result["action"], "PRESERVE_EXISTING_ROOM_RETURN")
            self.assertTrue(result["within_existing_door_anchor_tolerance"])
        self.assertAlmostEqual(self.handoff("CONSTRAINED_ARRIVAL", (0.30, 0.0, 0.0))["door_anchor_distance_m"], 0.30)

    def test_far_run0133_shape_attempts_existing_reposition_once(self):
        result = self.handoff("CONSTRAINED_ARRIVAL", (5.535986333502168, 0.0, 0.0))
        self.assertEqual(result["action"], "ATTEMPT_EXISTING_STRATEGIC_REPOSITION_ONCE")
        self.assertFalse(result["within_existing_door_anchor_tolerance"])
        self.assertAlmostEqual(result["door_anchor_distance_m"], 5.535986333502168)

    def test_non_constrained_arrival_does_not_enter_recovery(self):
        result = self.handoff("POST_ARRIVAL_VIABLE", (5.0, 0.0, 0.0))
        self.assertEqual(result["action"], "PRESERVE_EXISTING_ROOM_RETURN")
        self.assertEqual(result["reason"], "NOT_A_CONSTRAINED_ARRIVAL")

    def test_missing_proximity_inputs_fail_closed(self):
        result = self.handoff("CONSTRAINED_ARRIVAL", (), ())
        self.assertEqual(result["action"], "FAIL_CLOSED_RECOVERY_INPUT_UNAVAILABLE")
        source = inspect.getsource(self.module.execute_room_search_v2)
        self.assertIn('if recovery_handoff["action"] == "FAIL_CLOSED_RECOVERY_INPUT_UNAVAILABLE":', source)
        self.assertIn('"RECOVERY_INPUT_UNAVAILABLE"', source)

    def test_recovery_path_reuses_preflight_then_requires_fresh_viability_once(self):
        source = inspect.getsource(self.module.execute_room_search_v2)
        branch_start = source.index('if recovery_handoff["action"] == "ATTEMPT_EXISTING_STRATEGIC_REPOSITION_ONCE":')
        branch_end = source.index('candidate_audit.update({\n                    "decision_outcome": "CONSTRAINED_ARRIVAL",', branch_start)
        branch = source[branch_start:branch_end]
        self.assertEqual(branch.count("attempt_strategic_reposition(candidate, \"CONSTRAINED_ARRIVAL\")"), 1)
        self.assertIn("room_search_v2_postarrival_viability(\n                                args, recovered_context", branch)
        self.assertIn('recovered_viability.get("viability_result") == "POST_ARRIVAL_VIABLE"', branch)
        self.assertIn("return request_room_return_exit(completion_contract)", branch)
        self.assertIn("request_room_return=False", branch)
        self.assertNotIn("while", branch)

    def test_room_return_implementation_remains_outside_the_new_handoff(self):
        source = inspect.getsource(self.module.execute_room_search_v2)
        start = source.index("def execute_room_return_exit()")
        end = source.index("def request_room_return_exit", start)
        room_return = source[start:end]
        self.assertIn("search.door_return_target()", room_return)
        self.assertIn("search.choose_return_target", room_return)
        self.assertIn("search.choose_safe_return_transition", room_return)


if __name__ == "__main__":
    unittest.main(verbosity=2)
