#!/usr/bin/env python3
"""Offline benchmark for C0's private same-epoch continuation evaluator."""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp/tests"))
sys.path.insert(0, str(ROOT / "scripts/local_subgoal_runner_mvp"))

from test_continuation_viability_c0_shadow import ContinuationViabilityC0ShadowTests


def percentiles(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"p50_ms": None, "p95_ms": None, "max_ms": None, "sample_count": 0}
    at = lambda fraction: ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]
    return {
        "p50_ms": at(0.50) / 1e6,
        "p95_ms": at(0.95) / 1e6,
        "max_ms": max(ordered) / 1e6,
        "sample_count": len(ordered),
    }


def main():
    fixture = ContinuationViabilityC0ShadowTests("test_a_actual_frozen_s1_replay_is_pure_and_reports_a_tristate")
    fixture.setUp()
    current = fixture.module.evaluate_frozen_room_local_candidate(fixture.epoch, fixture.candidate)
    productive = [row for row in current["motion_candidates"] if row.get("score_eligible")]
    if len(productive) < 6:
        raise RuntimeError("benchmark_fixture_has_fewer_than_six_productive_motions")
    iterations = 30
    benchmark = {
        "benchmark": "CONTINUATION_C0_PRIVATE_FROZEN_EPOCH",
        "iterations_per_cohort_size": iterations,
        "input": {
            "epoch_id": fixture.epoch.epoch_id,
            "command_slice_sec": fixture.args.command_slice_sec,
            "dwa_predict_time": fixture.args.dwa_predict_time,
            "control_authority": False,
        },
        "cohorts": {},
    }
    for count in (1, 3, 6):
        elapsed = []
        shared_preprocessing = []
        components = {
            "per_s1_footprint_preparation": [],
            "astar": [],
            "dwa_rollout_collision_phase2": [],
        }
        statuses = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            result = fixture.module.evaluate_frozen_room_local_continuation_cohort(
                fixture.epoch, fixture.candidate, productive[:count], cap=6,
            )
            elapsed.append(time.perf_counter_ns() - started)
            shared_preprocessing.append(result.get("shared_preprocessing_ns", 0))
            statuses.append(result["continuation_status"])
            for record in result["records"]:
                timing = record.get("timing_ns") or {}
                for name in components:
                    if name in timing:
                        components[name].append(timing[name])
        benchmark["cohorts"][str(count)] = {
            "cohort_total": percentiles(elapsed),
            "shared_preprocessing": percentiles(shared_preprocessing),
            "per_motion_component": {name: percentiles(values) for name, values in components.items()},
            "statuses": {status: statuses.count(status) for status in sorted(set(statuses))},
        }
    print(json.dumps(benchmark, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
