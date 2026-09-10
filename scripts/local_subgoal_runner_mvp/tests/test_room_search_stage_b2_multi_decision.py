#!/usr/bin/env python3
"""Offline Stage B2 contract tests; no ROS master or simulator is started."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))
sys.path.insert(0, str(THIS.parent))

import room_search_stage_b_shadow_capture as capture  # noqa: E402
import room_search_stage_b_shadow_sidecar as sidecar  # noqa: E402
from test_room_search_stage_b_shadow import (  # noqa: E402
    ROOT, dry_runner_command, make_grid, make_search_and_ranked, make_status,
    planning_context, portal_context,
)


class StageB2MultiDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="room-search-stage-b2-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _capture(self, manager: capture.RoomSearchStageBShadowCapture, decision_id: int) -> dict:
        grid = make_grid()
        search, ranked = make_search_and_ranked()
        return manager.try_capture_epoch(
            run_id="stage-b2-three", decision_id=decision_id, ros_timestamp_sec=float(decision_id),
            pose_odom_xy_yaw=(0.0, 0.0, 0.0), pose_room_xy=(0.0, 0.0),
            portal_context=portal_context(), planning_context=planning_context(grid, make_status(grid)),
            seen_cells=search.observation.seen, ranked_candidates=ranked, parameters={},
            runner_command_template=dry_runner_command(), breadcrumbs=[],
        )

    def test_three_decisions_keep_event_identity_separate(self) -> None:
        manager = capture.RoomSearchStageBShadowCapture(
            ROOT, True, self.root, mode=capture.MODE_MULTI_DECISION,
        )
        for decision_id in (1, 2, 3):
            self.assertEqual(self._capture(manager, decision_id)["status"], "SHADOW_CAPTURE_SUBMITTED")
            self.assertEqual(
                manager.record_production_admission(decision_id=decision_id, selected_rank=decision_id)["status"],
                "SHADOW_EVENT_SUBMITTED",
            )
        manager.note_production_finish(decision_id=3, final_decision="TEST_FINISH")
        first = manager.wait_for_writer(5.0)
        self.assertIsNotNone(first)
        self.assertEqual(set(manager.ready_paths), {1, 2, 3})
        for decision_id, ready in manager.ready_paths.items():
            rows = [json.loads(row) for row in (ready / "production_events.jsonl").read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all(row["decision_id"] == decision_id for row in rows))
            self.assertTrue(all(row["epoch_id"] == json.loads((ready / "manifest.json").read_text())["epoch_id"] for row in rows))
        self.assertEqual(
            manager.record_production_nbv(decision_id=99, selected_rank=1)["reason"],
            "NO_CAPTURE_FOR_THIS_DECISION",
        )

    def test_backlog_drop_is_nonblocking_and_bounded(self) -> None:
        manager = capture.RoomSearchStageBShadowCapture(
            ROOT, True, self.root, mode=capture.MODE_MULTI_DECISION, snapshot_job_capacity=1,
        )
        manager._job_queue.put_nowait({"occupied": True})
        result = self._capture(manager, 1)
        self.assertEqual(result["reason"], "SNAPSHOT_JOB_QUEUE_FULL")
        self.assertEqual(manager._job_queue.qsize(), 1)
        self.assertEqual(manager._capture_drop_count, 1)
        self.assertIsNone(manager._thread)

    def test_grid_and_mission_staleness_are_independent(self) -> None:
        current = sidecar._staleness({"content_generation_id": 10}, {"content_generation_id": 10}, {"mission_decision_advanced": False})
        grid = sidecar._staleness({"content_generation_id": 10}, {"content_generation_id": 11}, {"mission_decision_advanced": False})
        mission = sidecar._staleness({"content_generation_id": 10}, {"content_generation_id": 10}, {"mission_decision_advanced": True})
        both = sidecar._staleness({"content_generation_id": 10}, {"content_generation_id": 11}, {"mission_decision_advanced": True})
        self.assertEqual(current["staleness_state"], "CURRENT_SAME_DECISION")
        self.assertEqual(grid["staleness_state"], "GRID_STALE_ONLY")
        self.assertEqual(mission["staleness_state"], "MISSION_DECISION_ADVANCED")
        self.assertEqual(both["staleness_state"], "GRID_STALE_AND_MISSION_ADVANCED")
        self.assertEqual(grid["grid_generation_delta"], 1)

    def test_baseline_is_lifecycle_not_fixed_wait(self) -> None:
        waiting = sidecar._event_lifecycle([], 7)
        advanced = sidecar._event_lifecycle([{"decision_id": 7, "event": "MISSION_DECISION_ADVANCED"}], 7)
        finished = sidecar._event_lifecycle([{"decision_id": 7, "event": "PRODUCTION_FINISH"}], 7)
        self.assertTrue(all(value == "NOT_YET_OBSERVED" for value in waiting["required_event_lifecycle"].values()))
        self.assertTrue(all(value == "DECISION_ADVANCED_BEFORE_EVENT" for value in advanced["required_event_lifecycle"].values()))
        self.assertTrue(all(value == "RUN_FINISHED_BEFORE_EVENT" for value in finished["required_event_lifecycle"].values()))


if __name__ == "__main__":
    unittest.main()
