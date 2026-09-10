#!/usr/bin/env python3
"""Offline acceptance tests for the one-shot frozen-decision audit path."""

import copy
import json
import math
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "scripts/local_subgoal_runner_mvp"
sys.path.insert(0, str(MODULE_DIR))

import frozen_decision_audit as audit  # noqa: E402
import navigation_state_machine as navigation  # noqa: E402
from local_grid_contract import (  # noqa: E402
    GRID_CONTRACT_VERSION,
    GRID_STATUS_SCHEMA_VERSION,
    POINT_PLANNING_WITH_OBSTACLE_INFLATION,
    STATIC_PLANNING_FOOTPRINT_RADIUS_M,
    grid_content_hash,
)
from room_search_v1 import PortalAnchor, RoomSearchV2  # noqa: E402


class Stamp:
    def __init__(self, seconds):
        self.seconds = float(seconds)

    def to_sec(self):
        return self.seconds


def make_grid():
    resolution = 0.05
    width = 80
    height = 80
    origin_x = -2.0
    origin_y = -2.0
    data = [0] * (width * height)
    wall_x_index = int(math.floor((1.0 - origin_x) / resolution))
    for y_index in range(height):
        data[wall_x_index + width * y_index] = 100
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id="base", stamp=Stamp(50.0), seq=7),
        info=types.SimpleNamespace(
            resolution=resolution,
            width=width,
            height=height,
            origin=types.SimpleNamespace(
                position=types.SimpleNamespace(x=origin_x, y=origin_y, z=0.0),
                orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
        data=data,
    )


def make_status(grid):
    producer = "fixture-l3v"
    generation = 50
    stamp = 50.0
    status = {
        "contract_version": GRID_CONTRACT_VERSION,
        "schema_version": GRID_STATUS_SCHEMA_VERSION,
        "producer_instance_id": producer,
        "content_generation_id": generation,
        "grid_content_stamp": stamp,
        "tf_valid": True,
        "all_required_inputs_fresh": True,
        "frame_id": "base",
        "width": grid.info.width,
        "height": grid.info.height,
        "resolution": grid.info.resolution,
        "origin": {"x": grid.info.origin.position.x, "y": grid.info.origin.position.y},
        "input_time_monotonic": True,
        "diagnostic_only": False,
        "safe_for_navigation": True,
        "upstream_navigation_allowed": True,
        "rejection_reasons": [],
        "static_footprint_radius_m": STATIC_PLANNING_FOOTPRINT_RADIUS_M,
        "planning_collision_model": POINT_PLANNING_WITH_OBSTACLE_INFLATION,
        "local_traversability_status": "FREE_SUPPORTED",
        "input_freshness_window_sec": 2.0,
    }
    status["grid_content_hash"] = grid_content_hash(grid, producer, generation, stamp)
    return status


def make_search():
    portal = {
        "portal_width_m": 1.2,
        "frozen_geometry": {
            "portal_center_odom": [0.0, 0.0],
            "portal_normal_odom": [1.0, 0.0],
        },
    }
    return RoomSearchV2(PortalAnchor.from_frozen_target(portal, (0.0, 0.0, 0.0), 1.0))


def dry_runner_command():
    return [
        sys.executable,
        "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py",
        "--max-steps", "1",
        "--robot-radius-m", str(STATIC_PLANNING_FOOTPRINT_RADIUS_M),
        "--goal-tolerance-m", "0.30",
        "--enforce-min-forward-speed",
        "--min-linear-x", "0.30",
        "--disable-pointcloud-wall-heading",
        "--disable-imu-heading-hold",
        "--additional-clearance-margin-m", "0.0",
        "--local-control-mode", "ROOM_LOCAL",
    ]


class FrozenDecisionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="frozen-decision-audit-test-")
        cls.temp_path = Path(cls.temp.name)
        cls.grid = make_grid()
        cls.status = make_status(cls.grid)
        cls.context = {
            "matched_pair_found": True,
            "qualified": True,
            "qualification_errors": [],
            "grid_header_stamp_sec": 50.0,
            "grid_content_stamp": 50.0,
            "content_generation_id": 50,
            "grid_content_hash": cls.status["grid_content_hash"],
            "local_traversability_status": "FREE_SUPPORTED",
            "grid_msg": cls.grid,
            "status_payload": cls.status,
        }
        cls.off_result = cls.run_fixture(enabled=False, run_id="fixture-off")
        cls.on_result = cls.run_fixture(enabled=True, run_id=f"fixture-on-{os.getpid()}-{time.time_ns()}")
        cls.bundle = cls.on_result["capture"].wait(30.0)
        if cls.bundle is None:
            raise AssertionError(f"capture writer failed: {cls.on_result['capture'].error}")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def run_fixture(cls, *, enabled, run_id):
        search = make_search()
        search.observation.seen = {(0, 0), (0, 1)}
        before_seen = copy.deepcopy(search.observation.seen)
        generation_audit = {}

        def enrich(row):
            if float(row["base_xy"][0]) < 1.0:
                visible = [(0.5, -0.1), (1.0, -0.1), (1.5, -0.1)]
                occlusion = [(1.5, -0.1)]
            else:
                visible = [(-0.5, 1.0), (0.0, 1.0)]
                occlusion = []
            return {
                **row,
                "heading_change_rad": math.atan2(row["base_xy"][1], row["base_xy"][0]),
                "visible_room_points": visible,
                "occlusion_reveal_room_points": occlusion,
                "danger_reobserve_supported": False,
                "danger_reobserve_opportunities": [],
                "danger_reobserve_abs_bearing_rad": None,
            }

        candidates = search.candidates_from_planning_free_base(
            (0.0, 0.0, 0.0), [(0.5, -0.1), (1.4, 0.4)], enrich, audit=generation_audit,
        )
        ranked = search.cheap_rank_candidates(candidates)
        ranked_before = copy.deepcopy(ranked)
        preflight_calls = []

        def preflight(candidate):
            preflight_calls.append(candidate["_room_search_audit_candidate_id"])
            return {
                "legal": True,
                "path_length_m": 0.5,
                "runner": {
                    "runner_final_decision": "BLOCK_ASTAR_DWA_MAX_STEPS",
                    "last_dwa": {
                        "p_through_astar_path_exists": True,
                        "safe_moving_candidate_count": 22,
                        "sample_count": 33,
                    },
                },
            }

        admission = navigation.room_search_v2_admit_with_l3v_consistency(
            ranked, preflight, 5,
        )
        selected = admission["candidate"]
        capture = audit.FrozenDecisionCapture(ROOT, enabled, cls.temp_path / "bundles")
        capture_result = capture.capture_once(
            run_id=run_id,
            decision_id=1,
            ros_timestamp_sec=50.1,
            pose_odom_xy_yaw=(0.0, 0.0, 0.0),
            pose_room_xy=(0.0, 0.0),
            portal_context={
                "frame": "team_livox_odom",
                "center_xy": [0.0, 0.0],
                "inward_normal": [1.0, 0.0],
                "tangent": [0.0, 1.0],
                "width_m": 1.2,
                "door_return_anchor_xy_yaw": [0.0, 0.0, 0.0],
                "door_return_anchor_stamp_sec": 1.0,
            },
            planning_context=cls.context,
            seen_cells=search.observation.seen,
            ranked_candidates=ranked,
            selected_candidate=selected,
            parameters={"execute": False, "fixture": True},
            runner_command_template=dry_runner_command(),
            breadcrumbs=search.breadcrumbs,
            capture_trigger_reason="FIXTURE_FIRST_QUALIFIED_NORMAL_DECISION",
        )
        command_generation = [list(selected["target_xy_team_livox_odom"])]
        return {
            "raw_candidate_count": generation_audit["raw_candidate_count_before_sector_compression"],
            "ranked_ids": [item["_room_search_audit_candidate_id"] for item in ranked],
            "selected_target": list(selected["target_xy_team_livox_odom"]),
            "selected_preflight": {
                "legal": selected["legal"],
                "path_length_m": selected["path_length_m"],
            },
            "completion_control_state": {"mission_complete": False, "control_state": "ROOM_SEARCH"},
            "production_return_value": selected["_room_search_audit_candidate_id"],
            "preflight_calls": preflight_calls,
            "command_generation": command_generation,
            "seen_unchanged": search.observation.seen == before_seen,
            "ranked_unchanged": ranked == ranked_before,
            "capture": capture,
            "capture_result": capture_result,
        }

    def test_a_capture_is_default_off_and_has_no_side_effect(self):
        self.assertEqual(self.off_result["capture_result"]["status"], "CAPTURE_DISABLED")
        self.assertIsNone(self.off_result["capture"].bundle_path)

    def test_b_capture_off_on_is_behavior_neutral(self):
        keys = (
            "raw_candidate_count", "ranked_ids", "selected_target", "selected_preflight",
            "completion_control_state", "production_return_value", "preflight_calls", "command_generation",
        )
        self.assertEqual(
            {key: self.off_result[key] for key in keys},
            {key: self.on_result[key] for key in keys},
        )
        self.assertEqual(len(self.on_result["preflight_calls"]), 1)
        self.assertTrue(self.off_result["seen_unchanged"])
        self.assertTrue(self.on_result["seen_unchanged"])
        self.assertTrue(self.off_result["ranked_unchanged"])
        self.assertTrue(self.on_result["ranked_unchanged"])

    def test_c_snapshot_grid_seen_candidate_round_trip(self):
        validation = audit.validate_snapshot(self.bundle)
        self.assertEqual(validation["status"], "SNAPSHOT_VALID", validation["errors"])
        seen = json.loads((self.bundle / "seen.json").read_text(encoding="utf-8"))
        self.assertEqual(seen["cells"], [[0, 0], [0, 1]])
        self.assertEqual(seen["count"], 2)
        candidates = json.loads((self.bundle / "candidates.json").read_text(encoding="utf-8"))
        self.assertEqual(candidates["ranked_candidate_count"], 2)
        for candidate in candidates["ranked_candidates"]:
            opportunity = candidate["opportunity"]
            self.assertEqual(len(opportunity["predicted_visible_cell_ids"]), opportunity["predicted_visible_count"])
            self.assertEqual(len(opportunity["predicted_new_cell_ids"]), opportunity["predicted_new_count"])
            self.assertEqual(
                len(opportunity["predicted_occlusion_reveal_cell_ids"]),
                opportunity["predicted_occlusion_reveal_count"],
            )

    def test_d_source_and_parameter_identity_are_closed(self):
        source = json.loads((self.bundle / "source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(source["source_recovery_mode"], "EXACT_BUNDLE_SOURCE_COPIES")
        self.assertFalse(source["missing_source_files"])
        self.assertGreaterEqual(len(source["source_files"]), 8)
        parameters = json.loads((self.bundle / "parameters.json").read_text(encoding="utf-8"))
        self.assertFalse(parameters["execute_flag_present"])
        self.assertNotIn("--execute", parameters["production_dry_preflight_command_template"])

    def test_e_snapshot_invalid_reports_exact_mismatch(self):
        tampered = self.temp_path / "tampered"
        shutil.copytree(str(self.bundle), str(tampered))
        seen_path = tampered / "seen.json"
        seen = json.loads(seen_path.read_text(encoding="utf-8"))
        seen["cells"].append([99, 99])
        seen_path.write_text(json.dumps(seen, sort_keys=True) + "\n", encoding="utf-8")
        validation = audit.validate_snapshot(tampered)
        self.assertEqual(validation["status"], "SNAPSHOT_INVALID")
        self.assertIn("file_hash_mismatch:seen.json", validation["errors"])
        self.assertIn("seen_count_mismatch", validation["errors"])

    def test_f_offline_dry_preflight_checks_multiple_candidates_without_commands(self):
        replay = audit.replay_snapshot(self.bundle)
        self.assertEqual(replay["status"], "DRY_PREFLIGHT_COMPLETE", replay)
        self.assertEqual(len(replay["results"]), 2)
        self.assertFalse(replay["commands_published"])
        self.assertFalse(replay["selection_performed"])
        self.assertTrue(all(result["commands_published"] is False for result in replay["results"]))
        self.assertEqual(replay["results"][0]["legality"], "LEGAL_NOW")
        self.assertEqual(replay["results"][1]["legality"], "ILLEGAL")

    def test_g_source_identity_mismatch_stops_before_replay(self):
        real_hash = audit._sha256_path

        def mismatched_current_tool(path):
            if Path(path).resolve() == Path(audit.__file__).resolve():
                return "0" * 64
            return real_hash(Path(path))

        with mock.patch.object(audit, "_sha256_path", side_effect=mismatched_current_tool):
            replay = audit.replay_snapshot(self.bundle)
        self.assertEqual(replay["status"], "SOURCE_IDENTITY_MISMATCH")
        self.assertFalse(replay["results"])
        self.assertIn("frozen_replay_command", replay)

    def test_h_loading_frozen_runner_does_not_mutate_bundle_with_python_cache(self):
        module = audit._load_frozen_runner(self.bundle)
        self.assertTrue(hasattr(module, "BlockAStarDwaRunner"))
        self.assertEqual(audit.validate_snapshot(self.bundle)["status"], "SNAPSHOT_VALID")
        self.assertFalse(any(path.suffix == ".pyc" for path in (self.bundle / "sources").rglob("*")))

    def test_i_python_cache_is_derived_and_does_not_change_snapshot_provenance(self):
        cache = self.bundle / "sources/scripts/local_subgoal_runner_mvp/__pycache__/fixture.pyc"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"derived-python-cache")
        self.assertEqual(audit.validate_snapshot(self.bundle)["status"], "SNAPSHOT_VALID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
