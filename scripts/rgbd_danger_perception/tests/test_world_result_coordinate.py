#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest


MODULE = pathlib.Path(__file__).parents[1] / "world_result_coordinate.py"
SPEC = importlib.util.spec_from_file_location("world_result_coordinate", MODULE)
world = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = world
SPEC.loader.exec_module(world)


class WorldResultCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.spawn = world.EffectiveSpawn((0.0, -2.2, 0.6), 0.0, "EXPLICIT_AUDIT_WRAPPER_ENV")
        self.raw = world.Pose((1.0, 2.0, 0.5), (0.0, 0.0, 0.0, 1.0), 12.0)

    def test_transform_places_first_raw_at_spawn(self):
        transform = world.transform_from_spawn_and_first_raw(self.spawn, self.raw)
        self.assertEqual(world.apply_transform(transform, self.raw.position_xyz), [0.0, -2.2, 0.6])

    def test_no_result_before_raw_and_matching_continuity(self):
        authority = world.ResultCoordinateAuthority("run-1", self.spawn)
        authority.observe_gate_status({"continuity_id": "run-1:continuity-0001", "continuity_state": "CONTINUOUS", "localization_authority": "CONTINUOUS"})
        self.assertFalse(authority.ready())
        authority.observe_first_raw(self.raw)
        self.assertTrue(authority.ready())

    def test_wrong_run_continuity_fails_closed(self):
        authority = world.ResultCoordinateAuthority("run-1", self.spawn)
        authority.observe_first_raw(self.raw)
        authority.observe_gate_status({"continuity_id": "other:continuity-0001", "continuity_state": "CONTINUOUS", "localization_authority": "CONTINUOUS"})
        self.assertFalse(authority.ready())
        self.assertEqual(authority.provenance()["last_gate_continuity_id"], "other:continuity-0001")
        self.assertEqual(authority.provenance()["last_gate_rejection_reason"], "continuity_run_id_mismatch")

    def test_invalid_continuity_prevents_future_result(self):
        authority = world.ResultCoordinateAuthority("run-1", self.spawn)
        authority.observe_first_raw(self.raw)
        authority.observe_gate_status({"continuity_id": "run-1:continuity-0001", "continuity_state": "REBASE_REQUIRED", "localization_authority": "UNAVAILABLE"})
        self.assertFalse(authority.ready())
        self.assertIn("continuity_invalid", authority.provenance()["invalid_reason"])

    def test_result_schema_and_track_transform(self):
        authority = world.ResultCoordinateAuthority("run-1", self.spawn)
        authority.observe_first_raw(self.raw)
        authority.observe_gate_status({"continuity_id": "run-1:continuity-0001", "continuity_state": "CONTINUOUS", "localization_authority": "CONTINUOUS"})
        document = authority.result_document([{"position_xyz_m": [2.0, 2.0, 0.5]}], 3.5)
        self.assertEqual(document["detected_danger_sources"], [{"position": [1.0, -2.2, 0.6]}])
        self.assertEqual(document["coordinate_provenance"]["authority"], "RESULT_COORDINATE_ONLY")
        self.assertFalse(document["coordinate_provenance"]["gazebo_input_used"])

    def test_explicit_environment_required(self):
        with self.assertRaises(ValueError):
            world.explicit_spawn_from_environment({"ROBOT_X": "0"})
        spawn = world.explicit_spawn_from_environment({
            "ROBOT_X": "0", "ROBOT_Y": "1", "ROBOT_Z": "2", "ROBOT_YAW": "0.5",
            "STARTUP_ANCHOR_SPAWN_SOURCE": "EXPLICIT_AUDIT_WRAPPER_ENV",
        })
        self.assertEqual(spawn.position_xyz, (0.0, 1.0, 2.0))

    def test_pending_document_replaces_prior_run_semantics_with_current_run_provenance(self):
        authority = world.ResultCoordinateAuthority("run-current", self.spawn)
        document = authority.pending_result_document()
        self.assertEqual(document["exploration_time"], 0.0)
        self.assertEqual(document["detected_danger_sources"], [])
        self.assertEqual(document["coordinate_provenance"]["run_id"], "run-current")
        self.assertEqual(
            document["coordinate_provenance"]["result_state"],
            "PENDING_CURRENT_RUN_COORDINATE_AUTHORITY",
        )

    def test_raw_odom_same_domain_duration_is_finite_not_epoch_scale(self):
        authority = world.ResultCoordinateAuthority("run-1", self.spawn)
        authority.observe_first_raw(self.raw)
        authority.observe_first_raw(world.Pose((2.0, 2.0, 0.5), (0.0, 0.0, 0.0, 1.0), 112.119))
        self.assertAlmostEqual(authority.raw_elapsed_duration_sec(), 100.119)
        self.assertLess(authority.raw_elapsed_duration_sec(), 1e6)

    def test_raw_odom_duration_fails_closed_without_end_or_with_reverse_time(self):
        authority = world.ResultCoordinateAuthority("run-1", self.spawn)
        with self.assertRaises(RuntimeError):
            authority.raw_elapsed_duration_sec()
        authority.observe_first_raw(self.raw)
        authority.observe_first_raw(world.Pose((2.0, 2.0, 0.5), (0.0, 0.0, 0.0, 1.0), 11.0))
        with self.assertRaises(RuntimeError):
            authority.raw_elapsed_duration_sec()

    def test_duration_change_does_not_change_xyz_or_result_schema(self):
        authority = world.ResultCoordinateAuthority("run-1", self.spawn)
        authority.observe_first_raw(self.raw)
        authority.observe_first_raw(world.Pose((2.0, 2.0, 0.5), (0.0, 0.0, 0.0, 1.0), 13.5))
        authority.observe_gate_status({"continuity_id": "run-1:continuity-0001", "continuity_state": "CONTINUOUS", "localization_authority": "CONTINUOUS"})
        document = authority.result_document([{"position_xyz_m": [2.0, 2.0, 0.5]}], authority.raw_elapsed_duration_sec())
        self.assertEqual(document["detected_danger_sources"], [{"position": [1.0, -2.2, 0.6]}])
        self.assertEqual(set(document), {"exploration_time", "detected_danger_sources", "coordinate_provenance"})


if __name__ == "__main__":
    unittest.main()
