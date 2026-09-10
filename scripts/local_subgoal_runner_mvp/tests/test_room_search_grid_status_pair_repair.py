#!/usr/bin/env python3
"""Focused offline regressions for ROOM_SEARCH exact Grid/Status acquisition."""

import json
import types
import unittest
from unittest.mock import patch

from test_odom_cache import (
    FakeGrid,
    FakeRos,
    FakeStamp,
    GRID_CONTRACT_VERSION,
    GRID_STATUS_SCHEMA_VERSION,
    grid_content_hash,
    load_module,
)


class RoomSearchGridStatusPairRepairTests(unittest.TestCase):
    def setUp(self):
        self.ros = FakeRos()
        self.module = load_module(self.ros)
        self.args = types.SimpleNamespace(
            input_timeout_sec=0.0,
            runner_wall_watchdog_sec=60.0,
            robot_radius_m=0.2641935843278561,
            room_search_extra_clearance_margin_m=0.05,
        )
        self.pairs = self.module.FormalGridStatusPairSubscriber()

    def tearDown(self):
        self.pairs.close()

    @staticmethod
    def grid(stamp_sec, generation):
        grid = FakeGrid()
        grid.header.stamp = FakeStamp(stamp_sec)
        grid.header.seq = generation
        if generation > 1:
            grid.data[generation] = 100
        return grid

    @staticmethod
    def status(grid, generation, **overrides):
        stamp = grid.header.stamp.to_sec()
        producer = "room-search-pair-test"
        payload = {
            "contract_version": GRID_CONTRACT_VERSION,
            "schema_version": GRID_STATUS_SCHEMA_VERSION,
            "producer_instance_id": producer,
            "content_generation_id": generation,
            "grid_content_stamp": stamp,
            "grid_content_hash": grid_content_hash(grid, producer, generation, stamp),
            "frame_id": "base",
            "origin": {"x": -0.3, "y": -1.5},
            "resolution": 0.05,
            "width": 66,
            "height": 60,
            "tf_valid": True,
            "all_required_inputs_fresh": True,
            "input_time_monotonic": True,
            "diagnostic_only": False,
            "safe_for_navigation": True,
            "upstream_navigation_allowed": True,
            "rejection_reasons": [],
            "input_freshness_window_sec": 5.0,
        }
        payload.update(overrides)
        return payload

    def add_grid(self, grid):
        self.pairs._grid_cb(grid)

    def add_status(self, status):
        self.pairs._status_cb(types.SimpleNamespace(data=json.dumps(status)))

    def valid_context(self):
        grid = self.grid(1.0, 1)
        status = self.status(grid, 1)
        self.add_grid(grid)
        self.add_status(status)
        return self.module.room_search_v2_planning_context(self.args, self.pairs)

    def make_search(self):
        target = {
            "portal_width_m": 1.2,
            "frozen_geometry": {"portal_center_odom": [10.0, 20.0], "portal_normal_odom": [1.0, 0.0]},
        }
        return self.module.RoomSearchV2(self.module.PortalAnchor.from_frozen_target(target, (10.3, 20.0, 0.2), 12.0))

    def test_t1_mismatched_grid_status_never_constructs_context(self):
        self.add_grid(self.grid(1.0, 1))
        other = self.grid(2.0, 2)
        self.add_status(self.status(other, 2))
        context = self.module.room_search_v2_planning_context(self.args, self.pairs)
        self.assertFalse(context["matched_pair_found"])
        self.assertFalse(context["qualified"])
        self.assertEqual(context["failure_reason"], "ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE")
        self.assertFalse(context["candidate_generation_reached"])

    def test_t2_later_matching_status_completes_exact_pair(self):
        grid = self.grid(2.0, 2)
        self.add_grid(grid)
        self.add_status(self.status(self.grid(1.0, 1), 1))
        self.add_status(self.status(grid, 2))
        context = self.module.room_search_v2_planning_context(self.args, self.pairs)
        self.assertTrue(context["matched_pair_found"])
        self.assertTrue(context["qualified"])
        self.assertEqual(context["grid_content_stamp"], 2.0)
        self.assertEqual(context["content_generation_id"], 2)

    def test_t3_valid_pair_uses_existing_qualification_unchanged(self):
        context = self.valid_context()
        self.assertTrue(context["qualified"])
        self.assertEqual(context["qualification_errors"], [])
        self.assertTrue(context["candidate_generation_reached"])

    def test_t4_formally_matched_but_unsafe_pair_fails_closed(self):
        grid = self.grid(1.0, 1)
        self.add_grid(grid)
        self.add_status(self.status(grid, 1, safe_for_navigation=False))
        context = self.module.room_search_v2_planning_context(self.args, self.pairs)
        self.assertTrue(context["matched_pair_found"])
        self.assertFalse(context["qualified"])
        self.assertEqual(context["failure_reason"], "ROOM_SEARCH_GRID_STATUS_CONTEXT_UNQUALIFIED")
        self.assertIn("status_safe_for_navigation_false", context["qualification_errors"])

    def test_t5_no_pair_is_explicitly_distinct_from_no_safe_candidate(self):
        context = self.module.room_search_v2_planning_context(self.args, self.pairs)
        record = self.module.room_search_v2_context_record(context)
        self.assertEqual(record["failure_reason"], "ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE")
        self.assertNotIn("NO_SAFE_USEFUL_CANDIDATE", str(record))

    def test_t6_valid_pair_reaches_candidate_generation(self):
        context = self.valid_context()
        candidates = self.make_search().candidates_from_planning_free_base((10.3, 20.0, 0.0), context["free_base_points"])
        self.assertTrue(context["candidate_generation_reached"])
        self.assertGreater(len(candidates), 0)

    def test_t7_run0117_style_free_context_has_nonzero_candidates(self):
        context = self.valid_context()
        candidates = self.make_search().candidates_from_planning_free_base((16.43, 1.71, 1.78), context["free_base_points"])
        self.assertGreaterEqual(len(candidates), 1)
        # Raw angular representatives are value-ranked before the separate
        # five-target formal preflight cap.
        self.assertLessEqual(len(candidates), 12)

    def test_t8_candidate_generation_is_stable_for_same_valid_pair(self):
        context = self.valid_context()
        first = self.make_search().candidates_from_planning_free_base((10.3, 20.0, 0.0), context["free_base_points"])
        second = self.make_search().candidates_from_planning_free_base((10.3, 20.0, 0.0), context["free_base_points"])
        self.assertEqual(first, second)

    def test_t9_extra_clearance_is_unchanged(self):
        context = self.valid_context()
        self.assertEqual(self.args.room_search_extra_clearance_margin_m, 0.05)
        self.assertTrue(context["candidate_generation_reached"])

    def test_t10_nbv_ranking_is_unchanged(self):
        best = self.make_search().select_best([
            {"room_target_xy": [2.0, 0.0], "visible_room_points": [(0, 0)], "path_length_m": 0.5, "heading_change_rad": 0.5},
            {"room_target_xy": [2.0, 1.0], "visible_room_points": [(1, 0), (1, 1), (1, 2)], "path_length_m": 1.0, "heading_change_rad": 0.1},
        ])
        self.assertEqual(best["room_target_xy"], [2.0, 1.0])

    def test_t11_available_pair_returns_without_wait(self):
        grid = self.grid(1.0, 1)
        expected = (grid, self.status(grid, 1))
        self.pairs.matching_pair = lambda: expected
        self.assertIs(
            self.pairs.wait_for_matching_pair(1.0, wall_watchdog_sec=60.0),
            expected,
        )

    def test_t12_pair_arriving_during_sim_timeout_is_returned(self):
        grid = self.grid(1.0, 1)
        expected = (grid, self.status(grid, 1))
        attempts = []

        def delayed_pair():
            attempts.append(1)
            return expected if len(attempts) == 2 else None

        self.pairs.matching_pair = delayed_pair
        with patch.object(self.module.time, "sleep", return_value=None):
            self.assertIs(
                self.pairs.wait_for_matching_pair(1.0, wall_watchdog_sec=60.0),
                expected,
            )
        self.assertEqual(len(attempts), 2)

    def test_t13_no_pair_times_out_truthfully_without_attribute_or_deadline_error(self):
        self.pairs.matching_pair = lambda: None
        self.assertIsNone(self.pairs.wait_for_matching_pair(0.0, wall_watchdog_sec=60.0))

    def test_t14_low_rtf_wait_uses_sim_time_before_wall_watchdog(self):
        grid = self.grid(1.0, 1)
        expected = (grid, self.status(grid, 1))
        attempts = []

        def delayed_pair():
            attempts.append(1)
            return expected if len(attempts) == 3 else None

        self.pairs.matching_pair = delayed_pair
        # Simulation remains paused at 0.0 while wall time advances to 59 s;
        # the explicit 60 s watchdog must not preempt a still-pending sim wait.
        with patch.object(self.module.time, "monotonic", side_effect=[0.0, 30.0, 59.0]), patch.object(
            self.module.time, "sleep", return_value=None
        ):
            self.assertIs(
                self.pairs.wait_for_matching_pair(1.0, wall_watchdog_sec=60.0),
                expected,
            )

    def test_t15_valid_pair_starts_room_search_planning_context(self):
        context = self.valid_context()
        self.assertTrue(context["matched_pair_found"])
        self.assertTrue(context["qualified"])
        self.assertTrue(context["candidate_generation_reached"])

    def test_t16_terminal_after_cache_never_uses_a_preterminal_exact_pair(self):
        cache = self.module.ExactGridStatusPairCache()
        earlier = self.grid(1.0, 1)
        cache.add_grid(earlier, received_wall_sec=10.0)
        cache.add_status(self.status(earlier, 1), received_wall_sec=10.01)
        self.assertIsNone(cache.matching_pair_record(now_wall_sec=10.1, not_before_wall_sec=10.02))

    def test_t17_terminal_after_cache_recovers_only_the_later_exact_generation(self):
        cache = self.module.ExactGridStatusPairCache()
        earlier = self.grid(1.0, 1)
        later = self.grid(2.0, 2)
        cache.add_grid(earlier, received_wall_sec=10.0)
        cache.add_status(self.status(earlier, 1), received_wall_sec=10.01)
        cache.add_grid(later, received_wall_sec=11.0)
        cache.add_status(self.status(later, 2), received_wall_sec=11.01)
        record = cache.matching_pair_record(now_wall_sec=11.1, not_before_wall_sec=10.5)
        self.assertIsNotNone(record)
        self.assertEqual(record["grid"].header.stamp.to_sec(), 2.0)
        self.assertEqual(record["status"]["grid_content_stamp"], 2.0)
        self.assertGreaterEqual(record["grid_received_wall_sec"], 10.5)
        self.assertGreaterEqual(record["status_received_wall_sec"], 10.5)

    def test_t18_terminal_after_cache_rejects_mixed_generations(self):
        cache = self.module.ExactGridStatusPairCache()
        grid = self.grid(1.0, 1)
        later = self.grid(2.0, 2)
        cache.add_grid(grid, received_wall_sec=11.0)
        cache.add_status(self.status(later, 2), received_wall_sec=11.01)
        self.assertIsNone(cache.matching_pair_record(now_wall_sec=11.1, not_before_wall_sec=10.5))

    def test_t19_terminal_after_timeout_does_not_fallback_to_the_cached_preterminal_pair(self):
        earlier = self.grid(1.0, 1)
        self.add_grid(earlier)
        self.add_status(self.status(earlier, 1))
        self.pairs.receipt_watermark = lambda: 10.0
        self.pairs.wait_for_matching_pair_after = lambda *_args, **_kwargs: None
        context = self.module.room_search_v2_fresh_planning_context(self.args, self.pairs)
        self.assertFalse(context["matched_pair_found"])
        self.assertFalse(context["qualified"])
        self.assertEqual(context["failure_reason"], "ROOM_SEARCH_GRID_STATUS_PAIR_UNAVAILABLE")
        self.assertEqual(
            context["post_arrival_pair_acquisition"]["pair_acquisition_result"],
            "POST_TERMINAL_EXACT_PAIR_TIMEOUT",
        )

    def test_t20_terminal_after_exact_pair_is_recorded_as_provenance(self):
        grid = self.grid(2.0, 2)
        record = {
            "grid": grid,
            "status": self.status(grid, 2),
            "grid_received_wall_sec": 11.0,
            "status_received_wall_sec": 11.01,
        }
        self.pairs.receipt_watermark = lambda: 10.0
        self.pairs.wait_for_matching_pair_after = lambda *_args, **_kwargs: record
        context = self.module.room_search_v2_fresh_planning_context(self.args, self.pairs)
        self.assertTrue(context["matched_pair_found"])
        self.assertTrue(context["qualified"])
        provenance = context["post_arrival_pair_acquisition"]
        self.assertEqual(provenance["pair_acquisition_result"], "POST_TERMINAL_EXACT_PAIR_ACQUIRED")
        self.assertEqual(provenance["matched_grid_content_stamp"], 2.0)
        self.assertEqual(provenance["matched_status_grid_content_stamp"], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
