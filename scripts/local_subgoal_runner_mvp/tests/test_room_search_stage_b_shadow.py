#!/usr/bin/env python3
"""Offline acceptance tests for behavior-neutral ROOM_SEARCH Stage B."""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "scripts/local_subgoal_runner_mvp"
sys.path.insert(0, str(MODULE_DIR))

import navigation_state_machine as navigation  # noqa: E402
import room_search_stage_a_contract as contract  # noqa: E402
import room_search_stage_b_shadow_capture as capture_module  # noqa: E402
import room_search_stage_b_shadow_sidecar as sidecar  # noqa: E402
from local_grid_contract import (  # noqa: E402
    GRID_CONTRACT_VERSION,
    GRID_STATUS_SCHEMA_VERSION,
    POINT_PLANNING_WITH_OBSTACLE_INFLATION,
    STATIC_PLANNING_FOOTPRINT_RADIUS_M,
    grid_content_hash,
)
from room_search_v1 import PortalAnchor, RoomSearchV2  # noqa: E402


RUN0139 = ROOT / (
    "debug/state_machine_navigation/run_archives/"
    "run_0139_20260822_224215_396672930_pid413263/frozen_decisions/"
    "run_0139_20260822_224215_396672930_pid413263/frozen_decision_0001"
)


class Stamp:
    def __init__(self, seconds):
        self.seconds = float(seconds)

    def to_sec(self):
        return self.seconds


def make_grid():
    resolution, width, height = 0.05, 80, 80
    origin_x, origin_y = -2.0, -2.0
    data = [0] * (width * height)
    wall_x = int(math.floor((1.0 - origin_x) / resolution))
    for y_index in range(height):
        data[wall_x + width * y_index] = 100
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id="base", stamp=Stamp(50.0), seq=7),
        info=types.SimpleNamespace(
            resolution=resolution, width=width, height=height,
            origin=types.SimpleNamespace(
                position=types.SimpleNamespace(x=origin_x, y=origin_y, z=0.0),
                orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
        data=data,
    )


def make_status(grid):
    producer, generation, stamp = "fixture-l3v", 50, 50.0
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


def make_search_and_ranked():
    anchor = PortalAnchor(
        center_xy=(0.0, 0.0), inward_normal=(1.0, 0.0), tangent=(0.0, 1.0), width_m=1.2,
        door_return_anchor_xy_yaw=(0.0, 0.0, 0.0), door_return_anchor_stamp_sec=1.0,
    )
    search = RoomSearchV2(anchor)
    search.observation.seen = {(0, 0), (0, 1)}

    def enrich(row):
        near = float(row["base_xy"][0]) < 1.0
        return {
            **row,
            "heading_change_rad": math.atan2(row["base_xy"][1], row["base_xy"][0]),
            "visible_room_points": (
                [(0.5, -0.1), (1.0, -0.1), (1.5, -0.1)] if near
                else [(-0.5, 1.0), (0.0, 1.0)]
            ),
            "occlusion_reveal_room_points": [(1.5, -0.1)] if near else [],
            "danger_reobserve_supported": False,
            "danger_reobserve_opportunities": [],
            "danger_reobserve_abs_bearing_rad": None,
        }

    candidates = search.candidates_from_planning_free_base(
        (0.0, 0.0, 0.0), [(0.5, -0.1), (1.4, 0.4)], enrich, audit={},
    )
    return search, search.cheap_rank_candidates(candidates)


def dry_runner_command():
    return [
        sys.executable, "scripts/local_subgoal_runner_mvp/block_astar_dwa_mature_runner.py",
        "--max-steps", "1", "--robot-radius-m", str(STATIC_PLANNING_FOOTPRINT_RADIUS_M),
        "--goal-tolerance-m", "0.30", "--enforce-min-forward-speed",
        "--min-linear-x", "0.30", "--disable-pointcloud-wall-heading",
        "--disable-imu-heading-hold", "--additional-clearance-margin-m", "0.0",
    ]


def portal_context():
    return {
        "frame": "team_livox_odom", "center_xy": [0.0, 0.0],
        "inward_normal": [1.0, 0.0], "tangent": [0.0, 1.0], "width_m": 1.2,
        "door_return_anchor_xy_yaw": [0.0, 0.0, 0.0], "door_return_anchor_stamp_sec": 1.0,
    }


def planning_context(grid, status):
    return {
        "matched_pair_found": True, "qualified": True, "qualification_errors": [],
        "grid_header_stamp_sec": 50.0, "grid_content_stamp": 50.0,
        "content_generation_id": 50, "grid_content_hash": status["grid_content_hash"],
        "local_traversability_status": "FREE_SUPPORTED",
        "grid_msg": grid, "status_payload": status,
    }


def capture_fixture(root: Path, run_id: str = "stage-b-fixture"):
    grid, status = make_grid(), None
    status = make_status(grid)
    search, ranked = make_search_and_ranked()
    manager = capture_module.RoomSearchStageBShadowCapture(ROOT, True, root)
    result = manager.try_capture_epoch(
        run_id=run_id, decision_id=1, ros_timestamp_sec=50.1,
        pose_odom_xy_yaw=(0.0, 0.0, 0.0), pose_room_xy=(0.0, 0.0),
        portal_context=portal_context(), planning_context=planning_context(grid, status),
        seen_cells=search.observation.seen, ranked_candidates=ranked,
        parameters={"execute": False, "fixture": True},
        runner_command_template=dry_runner_command(), breadcrumbs=search.breadcrumbs,
    )
    manager.record_production_admission(selected_candidate_id="fixture", selected_rank=1)
    manager.record_production_nbv(selected_candidate_id="fixture", selected_rank=1, nbv_value=1.0)
    manager.record_execution_handoff(selected_candidate_id="fixture", selected_rank=1, handoff="FIXTURE")
    return manager, result, search, ranked, grid


def rewrite_hash_manifest(bundle: Path):
    rows = []
    for path in sorted(path for path in bundle.rglob("*") if path.is_file()):
        relative = str(path.relative_to(bundle))
        if relative != "hashes.sha256":
            rows.append(f"{contract._sha256(path)}  {relative}")
    (bundle / "hashes.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def convert_run0139_to_stage_b(destination: Path) -> Path:
    shutil.copytree(RUN0139, destination)
    for name in (
        "manifest.json", "decision.json", "grid.json", "seen.json", "candidates.json",
        "parameters.json", "source_manifest.json",
    ):
        path = destination / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = capture_module.SCHEMA_VERSION
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest_path, decision_path = destination / "manifest.json", destination / "decision.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    epoch_id = "run0139-stage-b-offline-adapter"
    manifest.update({
        "bundle_type": "ROOM_SEARCH_STAGE_B_PRESELECTION_EPOCH", "epoch_id": epoch_id,
        "one_shot": True, "production_authority": False, "selection_authority": False,
        "command_authority": False, "completion_authority": False,
        "recoverability_authority": False, "fallback_authority": False,
    })
    decision.update({
        "epoch_id": epoch_id, "selected_candidate_id": None, "selected_rank": None,
        "production_authority": False, "preselection_snapshot": True,
    })
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    decision_path.write_text(json.dumps(decision, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    rewrite_hash_manifest(destination)
    return destination


class StageBShadowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="room-search-stage-b-test-")
        self.temp_path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_a_default_off_zero_work_and_flag_conflict(self):
        manager = capture_module.RoomSearchStageBShadowCapture(ROOT, False, self.temp_path)
        self.assertFalse(manager.enabled)
        self.assertFalse(manager._source_bytes)
        self.assertIsNone(manager._thread)
        self.assertFalse(self.temp_path.joinpath("unused").exists())
        grid = make_grid()
        search, ranked = make_search_and_ranked()
        result = manager.try_capture_epoch(
            run_id="off", decision_id=1, ros_timestamp_sec=1.0,
            pose_odom_xy_yaw=(0, 0, 0), pose_room_xy=(0, 0), portal_context=portal_context(),
            planning_context=planning_context(grid, make_status(grid)),
            seen_cells=search.observation.seen, ranked_candidates=ranked,
            parameters={}, runner_command_template=dry_runner_command(), breadcrumbs=[],
        )
        self.assertEqual(result["status"], "SHADOW_DISABLED")
        conflict = capture_module.RoomSearchStageBShadowCapture(ROOT, True, self.temp_path, conflict=True)
        self.assertFalse(conflict.enabled)
        self.assertTrue(conflict.conflict)
        with mock.patch.dict(os.environ, {
            capture_module.SHADOW_FLAG: "true",
            capture_module.FROZEN_CAPTURE_FLAG: "true",
        }, clear=False):
            from_env = capture_module.RoomSearchStageBShadowCapture.from_environment(ROOT)
        self.assertTrue(from_env.conflict)
        self.assertFalse(from_env.enabled)

    def test_b_hook_is_immutable_nonblocking_and_one_shot(self):
        manager, result, search, ranked, grid = capture_fixture(self.temp_path)
        self.assertEqual(result["status"], "SHADOW_CAPTURE_SUBMITTED")
        original_first_target = list(ranked[0]["target_xy_team_livox_odom"])
        ranked[0]["target_xy_team_livox_odom"][0] = 999.0
        grid.data[0] = 100
        search.observation.seen.add((99, 99))
        second = manager.try_capture_epoch(
            run_id="stage-b-fixture", decision_id=2, ros_timestamp_sec=51.0,
            pose_odom_xy_yaw=(0, 0, 0), pose_room_xy=(0, 0), portal_context=portal_context(),
            planning_context=planning_context(make_grid(), make_status(make_grid())),
            seen_cells=search.observation.seen, ranked_candidates=ranked,
            parameters={}, runner_command_template=dry_runner_command(), breadcrumbs=[],
        )
        manager.note_production_finish(final_decision="FIXTURE")
        bundle = manager.wait_for_writer(5.0)
        self.assertIsNotNone(bundle)
        self.assertEqual(second["reason"], "ONE_SHOT_ALREADY_SUBMITTED")
        self.assertEqual(len(list((self.temp_path / "stage-b-fixture").glob("*.ready"))), 1)
        candidates = json.loads((bundle / "candidates.json").read_text(encoding="utf-8"))
        seen = json.loads((bundle / "seen.json").read_text(encoding="utf-8"))
        grid_doc = json.loads((bundle / "grid.json").read_text(encoding="utf-8"))
        self.assertEqual(candidates["ranked_candidates"][0]["target_odom_xy"], original_first_target)
        self.assertNotIn([99, 99], seen["cells"])
        self.assertEqual(grid_doc["occupancy_grid"]["data"][0], 0)
        source = inspect.getsource(capture_module.RoomSearchStageBShadowCapture.try_capture_epoch)
        for forbidden in ("subprocess", "Publisher", "run_runner", "room_search_v2_preflight", "time.sleep"):
            self.assertNotIn(forbidden, source)

    def test_c_queue_capture_and_writer_failures_are_isolated(self):
        manager = capture_module.RoomSearchStageBShadowCapture(ROOT, True, self.temp_path)
        manager._epochs[1] = {"run_id": "run", "decision_id": 1, "epoch_id": "epoch"}
        for index in range(capture_module.EVENT_QUEUE_CAPACITY):
            manager._event_queue.put_nowait({"event": index})
        dropped = manager.record_production_admission(decision_id=1, selected_rank=1)
        self.assertEqual(dropped["status"], "SHADOW_INCOMPLETE")
        self.assertEqual(manager.event_drop_count, 1)

        snapshot_full = capture_module.RoomSearchStageBShadowCapture(ROOT, True, self.temp_path / "full")
        for _ in range(snapshot_full._job_queue_capacity):
            snapshot_full._job_queue.put_nowait({"occupied": True})
        grid_for_full = make_grid()
        search_for_full, ranked_for_full = make_search_and_ranked()
        full_result = snapshot_full.try_capture_epoch(
            run_id="full", decision_id=1, ros_timestamp_sec=1.0,
            pose_odom_xy_yaw=(0, 0, 0), pose_room_xy=(0, 0), portal_context=portal_context(),
            planning_context=planning_context(grid_for_full, make_status(grid_for_full)),
            seen_cells=search_for_full.observation.seen, ranked_candidates=ranked_for_full,
            parameters={}, runner_command_template=dry_runner_command(), breadcrumbs=[],
        )
        self.assertEqual((full_result["status"], full_result["reason"]), ("SHADOW_SKIPPED", "SNAPSHOT_JOB_QUEUE_FULL"))

        grid = make_grid()
        search, ranked = make_search_and_ranked()
        broken = capture_module.RoomSearchStageBShadowCapture(ROOT, True, self.temp_path / "broken")
        with mock.patch.object(capture_module, "_candidate_evidence", side_effect=RuntimeError("capture")):
            result = broken.try_capture_epoch(
                run_id="broken", decision_id=1, ros_timestamp_sec=1.0,
                pose_odom_xy_yaw=(0, 0, 0), pose_room_xy=(0, 0), portal_context=portal_context(),
                planning_context=planning_context(grid, make_status(grid)),
                seen_cells=search.observation.seen, ranked_candidates=ranked,
                parameters={}, runner_command_template=dry_runner_command(), breadcrumbs=[],
            )
        self.assertEqual(result["status"], "SHADOW_FAILED_ISOLATED")

        writer = capture_module.RoomSearchStageBShadowCapture(ROOT, True, self.temp_path / "writer")
        with mock.patch.object(writer, "_write_ready_bundle", side_effect=OSError("writer")):
            result = writer.try_capture_epoch(
                run_id="writer", decision_id=1, ros_timestamp_sec=1.0,
                pose_odom_xy_yaw=(0, 0, 0), pose_room_xy=(0, 0), portal_context=portal_context(),
                planning_context=planning_context(grid, make_status(grid)),
                seen_cells=search.observation.seen, ranked_candidates=ranked,
                parameters={}, runner_command_template=dry_runner_command(), breadcrumbs=[],
            )
            self.assertEqual(result["status"], "SHADOW_CAPTURE_SUBMITTED")
            writer._thread.join(2.0)
        self.assertIn("WRITER_EXCEPTION", writer.writer_error)

    def test_d_flag_on_off_production_trace_is_identical(self):
        def trace(enabled):
            grid = make_grid()
            search, ranked = make_search_and_ranked()
            manager = capture_module.RoomSearchStageBShadowCapture(ROOT, enabled, self.temp_path / str(enabled))
            manager.try_capture_epoch(
                run_id=f"trace-{enabled}", decision_id=1, ros_timestamp_sec=1.0,
                pose_odom_xy_yaw=(0, 0, 0), pose_room_xy=(0, 0), portal_context=portal_context(),
                planning_context=planning_context(grid, make_status(grid)),
                seen_cells=search.observation.seen, ranked_candidates=ranked,
                parameters={}, runner_command_template=dry_runner_command(), breadcrumbs=[],
            )
            calls = []
            selected, attempts = search.admit_ranked_candidates(
                ranked, lambda row: calls.append(row["_room_search_audit_candidate_id"]) or {
                    "legal": True, "path_length_m": 0.5,
                },
            )
            scored = search.score_candidates([selected])[0]
            marginal = search.evaluate_marginal_value(scored, 1)
            handoff = list(scored["target_xy_team_livox_odom"])
            manager.note_production_finish(final_decision="TRACE")
            manager.wait_for_writer(5.0)
            return {
                "ranked": [row["_room_search_audit_candidate_id"] for row in ranked],
                "preflight": calls, "attempts": attempts,
                "selected": scored["_room_search_audit_candidate_id"],
                "selected_target": list(scored["target_xy_team_livox_odom"]),
                "nbv": scored["nbv_value"], "completion": marginal["completion_reason"],
                "handoff": handoff, "return": "UNCHANGED_FIXTURE",
            }

        self.assertEqual(trace(False), trace(True))

    def test_e_sidecar_ready_only_timeout_exception_and_static_authority(self):
        malformed = self.temp_path / "not-ready"
        malformed.mkdir()
        result = sidecar.run_sidecar_once(malformed, result_path=self.temp_path / "bad.json")
        self.assertEqual(result["status"], "SHADOW_FAILED_ISOLATED")
        self.assertEqual(result["reason"], "READY_BUNDLE_REQUIRED")

        ready = self.temp_path / "timeout.ready"
        ready.mkdir()
        (ready / "manifest.json").write_text("{}", encoding="utf-8")
        (ready / "decision.json").write_text("{}", encoding="utf-8")
        (ready / "grid.json").write_text('{"planning_identity": {}}', encoding="utf-8")

        def sleeping(_bundle, _output):
            return [sys.executable, "-c", "import time; time.sleep(5)"]

        result = sidecar.run_sidecar_once(
            ready, result_path=self.temp_path / "timeout.json",
            evaluator_timeout_sec=0.05, evaluator_command_factory=sleeping,
        )
        self.assertEqual((result["status"], result["reason"]), ("SHADOW_FAILED_ISOLATED", "EVALUATOR_TIMEOUT"))
        self.assertTrue(result["evaluator_process_only_terminated"])
        incomplete = self.temp_path / "incomplete.ready"
        incomplete.mkdir()
        malformed_result = sidecar.run_sidecar_once(
            incomplete, result_path=self.temp_path / "malformed.json", production_event_wait_sec=0.0,
        )
        self.assertEqual(malformed_result["status"], "SHADOW_FAILED_ISOLATED")
        exception_result = sidecar.run_sidecar_once(
            ready, result_path=self.temp_path / "exception.json",
            evaluator_command_factory=lambda _bundle, _output: (_ for _ in ()).throw(RuntimeError("evaluator")),
        )
        self.assertEqual(exception_result["status"], "SHADOW_FAILED_ISOLATED")
        self.assertEqual(exception_result["reason"], "SIDECAR_EXCEPTION")
        source = Path(sidecar.__file__).read_text(encoding="utf-8")
        for forbidden in ("rospy.Publisher", "geometry_msgs", "Twist(", "write_absolute_target"):
            self.assertNotIn(forbidden, source)
        self.assertTrue(all(value is False for value in sidecar.AUTHORITY_FALSE.values()))

    def test_f_sidecar_complete_and_mission_stale_are_advisory(self):
        manager, _result, _search, _ranked, _grid = capture_fixture(self.temp_path / "capture", "sidecar")
        manager.note_production_finish(decision_id=1, final_decision="FIXTURE")
        bundle = manager.wait_for_writer(5.0)
        result = sidecar.run_sidecar_once(
            bundle, result_path=self.temp_path / "complete.json", evaluator_timeout_sec=20.0,
        )
        self.assertEqual(result["status"], "SHADOW_COMPLETE", result)
        self.assertFalse(result["execute"])
        self.assertTrue(all(result[key] is False for key in sidecar.AUTHORITY_FALSE))
        self.assertIn("evaluator_cpu_user_sec", result["timing"])
        self.assertIn("evaluator_peak_rss_bytes", result["timing"])
        self.assertGreaterEqual(result["timing"]["evaluator_cpu_user_sec"], 0.0)
        identities = iter([{"content_generation_id": 50}, {"content_generation_id": 51}])
        stale = sidecar.run_sidecar_once(
            bundle, result_path=self.temp_path / "stale.json", evaluator_timeout_sec=20.0,
            identity_provider=lambda: next(identities),
        )
        self.assertEqual(stale["status"], "SHADOW_STALE")
        self.assertEqual(stale.get("mission_decision_advanced"), False, stale)
        self.assertEqual(stale["staleness_state"], "GRID_STALE_ONLY")
        self.assertEqual(stale["grid_generation_delta"], 1)

    def test_g_run0139_stage_b_adapter_recovers_expected_evidence_without_winner(self):
        if not RUN0139.is_dir():
            self.skipTest("RUN0139 frozen decision fixture missing")
        bundle = convert_run0139_to_stage_b(self.temp_path / "run0139.ready")
        result = contract.run_stage_b_shadow_bundle(bundle)
        self.assertEqual(result["status"], "STAGE_B_SHADOW_COMPLETE", result)
        rows = result["legal_comparison_set"]["all_preflight_results"]
        legal = {row["rank"] for row in rows if row["legality"] == "LEGAL_NOW"}
        illegal = {row["rank"] for row in rows if row["legality"] == "ILLEGAL"}
        self.assertEqual(legal, {1, 2, 3, 5, 7, 8, 9, 10, 11})
        self.assertEqual(illegal, {4, 6})
        relations = {
            (row["a_candidate_id"], row["b_candidate_id"]): row["relation"]
            for row in result["opportunity_exact_set_relations"]
        }
        self.assertEqual(relations[("raw-0827", "raw-0896")], "PARTIAL_OVERLAP_WITH_UNIQUE_CELLS")
        self.assertEqual(relations[("raw-0827", "raw-0721")], "STRICT_SUPERSET")
        self.assertEqual(relations[("raw-0827", "raw-0548")], "DISJOINT")
        self.assertEqual([row["cheap_rank"] for row in result["nbv"]["candidate_evidence"]], [1, 2, 3, 5])
        self.assertFalse(result["selection_performed"])
        self.assertFalse(result["winner_selector_implemented"])
        self.assertIsNone(result["comparison_semantics"]["winner"])
        self.assertEqual(len(result["timing"]["candidate_preflights"]), 11)
        v1 = result["formal_mission_comparison_v1"]
        self.assertEqual(v1["comparison"]["selection_authority"], "SHADOW_ONLY")
        self.assertFalse(v1["command_authority"])
        self.assertEqual(len(v1["comparison"]["all_candidates"]), 11)

    def test_h_production_hook_is_one_way_and_core_sources_unchanged(self):
        navigation_source = (ROOT / "scripts/local_subgoal_runner_mvp/navigation_state_machine.py").read_text(encoding="utf-8")
        self.assertIn("stage_b_shadow_capture.try_capture_epoch", navigation_source)
        self.assertNotIn("stage_b_shadow_capture.ready_path", navigation_source)
        self.assertNotIn("stage_b_shadow_capture.writer_error", navigation_source)
        self.assertNotIn("run_stage_b_shadow_bundle", navigation_source)
        self.assertIn("candidate = admission[\"candidate\"]", navigation_source)
        self.assertIn("candidate = search.score_candidates([candidate])[0]", navigation_source)
        self.assertIn("attempt = execute_target(target_xy", navigation_source)
        shadow_capture_call = navigation_source.split("stage_b_shadow_capture.try_capture_epoch", 1)[1].split("def preflight_ranked", 1)[0]
        self.assertIn('local_control_mode="ROOM_LOCAL"', shadow_capture_call)


if __name__ == "__main__":
    unittest.main(verbosity=2)
