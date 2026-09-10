#!/usr/bin/env python3
"""Measure only the R1 pure evidence classifier; it has no ROS side effects."""

from __future__ import annotations

import json
import statistics
import time

from room_search_recoverability import (
    MAX_RECOVERABILITY_PREDECESSORS,
    RECOVERABLE,
    PredictedTerminalState,
    RecoverabilityCertificate,
    evaluate_predecessor_set,
)


EPOCH = {
    "grid_header_stamp_sec": 10.0,
    "grid_content_stamp": 10.0,
    "content_generation_id": "benchmark",
    "grid_content_hash": "benchmark",
}


def percentile(samples, fraction):
    values = sorted(samples)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
    return values[index]


def one_predecessor(index):
    return RecoverabilityCertificate(
        state_id="R%d" % index, pose_xy_yaw=(0.0, 0.0, 0.0), status=RECOVERABLE,
        predecessor_state_id=None, epoch_identity=dict(EPOCH), retreat_transition_type="ROOT",
        fresh=True, reason="BENCHMARK", evidence={"recency_index": index},
    )


def benchmark(count, iterations=2000):
    terminal = PredictedTerminalState(
        candidate_id="benchmark", decision_id=1, pose_xy_yaw=(1.0, 0.0, 0.0),
        completion_semantics="FULL_CANDIDATE_ACTION_COMPLETION", prediction_horizon_sec=1.0,
        source="BENCHMARK_FROZEN_INPUT", source_evidence={}, status="TERMINAL_STATE_AVAILABLE",
        reason="BENCHMARK", epoch_identity=dict(EPOCH),
    )
    predecessors = [one_predecessor(index) for index in range(count)]
    evidence = {
        predecessor.state_id: {
            "direct_translation": {"complete": True, "executable": index == 0},
            "orientation_then_translation": {"complete": True, "executable": False},
        }
        for index, predecessor in enumerate(predecessors)
    }
    elapsed_us = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        outcome = evaluate_predecessor_set(terminal, predecessors, evidence, max_predecessors=count)
        elapsed_us.append((time.perf_counter_ns() - started) / 1_000.0)
        if outcome.status != RECOVERABLE:
            raise RuntimeError("benchmark_expected_recoverable")
    return {
        "predecessor_count": count,
        "iterations": iterations,
        "p50_us": percentile(elapsed_us, 0.50),
        "p95_us": percentile(elapsed_us, 0.95),
        "max_us": max(elapsed_us),
        "terminal_state_preparation": "NOT_MEASURED_NO_FULL_ACTION_TERMINAL_SOURCE",
        "orientation_preflight": "NOT_EXECUTED_STATIC_FROZEN_EVIDENCE_ONLY",
        "translation_preflight": "NOT_EXECUTED_STATIC_FROZEN_EVIDENCE_ONLY",
        "grid_astar_phase2_collision": "NOT_EXECUTED_STATIC_FROZEN_EVIDENCE_ONLY",
    }


if __name__ == "__main__":
    payload = {
        "schema_version": "recoverability_r1_shadow_benchmark_v1",
        "authority_enabled": False,
        "hard_deadline_ms": None,
        "configured_predecessor_cap": MAX_RECOVERABILITY_PREDECESSORS,
        "scenarios": [benchmark(1), benchmark(2), benchmark(MAX_RECOVERABILITY_PREDECESSORS)],
        "interpretation": "Pure classification cost only; no hard authority deadline is justified until the same-epoch terminal/retreat evaluator exists.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
