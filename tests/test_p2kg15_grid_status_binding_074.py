#!/usr/bin/env python3
"""Unit coverage for the shared G15 grid/status content binding."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "local_subgoal_runner_mvp"))

from local_grid_contract import (  # noqa: E402
    GRID_CONTRACT_VERSION,
    GRID_STATUS_SCHEMA_VERSION,
    grid_content_hash,
    grid_metadata,
    validate_grid_status_content_binding,
)


class Stamp:
    def __init__(self, value):
        self.value = value

    def to_sec(self):
        return self.value


def make_grid(sequence=1, stamp=10.0, cells=None, resolution=0.5):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="base", seq=sequence, stamp=Stamp(stamp)),
        info=SimpleNamespace(
            resolution=resolution,
            width=4,
            height=4,
            origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=-1.0)),
        ),
        data=list(cells if cells is not None else [0] * 16),
    )


def make_status(grid, generation=41, producer="fixture", stamp=None):
    metadata = grid_metadata(grid)
    content_stamp = metadata["content_stamp"] if stamp is None else stamp
    status = {
        "contract_version": GRID_CONTRACT_VERSION,
        "schema_version": GRID_STATUS_SCHEMA_VERSION,
        "producer_instance_id": producer,
        "content_generation_id": generation,
        "grid_content_stamp": content_stamp,
        "tf_valid": True,
        "all_required_inputs_fresh": True,
    }
    status["grid_content_hash"] = grid_content_hash(grid, producer, generation, content_stamp)
    return status


class GridStatusBindingTests(unittest.TestCase):
    def assert_valid(self, grid, status):
        self.assertEqual(validate_grid_status_content_binding(grid, status), [])

    def assert_invalid(self, grid, status, reason):
        self.assertIn(reason, validate_grid_status_content_binding(grid, status))

    def test_matching_transport_sequence_and_generation_is_accepted(self):
        value = make_grid(sequence=41)
        self.assert_valid(value, make_status(value, generation=41))

    def test_different_transport_sequence_and_generation_is_accepted(self):
        value = make_grid(sequence=3)
        self.assert_valid(value, make_status(value, generation=41))

    def test_stamp_mismatch_is_rejected(self):
        value = make_grid()
        self.assert_invalid(value, make_status(value, stamp=11.0), "status_grid_content_stamp_mismatch")

    def test_hash_mismatch_is_rejected(self):
        value, status = make_grid(), make_status(make_grid())
        status["grid_content_hash"] = "wrong"
        self.assert_invalid(value, status, "status_grid_content_hash_mismatch")

    def test_changed_generation_with_stale_hash_is_rejected(self):
        value, status = make_grid(), make_status(make_grid(), generation=41)
        status["content_generation_id"] = 42
        self.assert_invalid(value, status, "status_grid_content_hash_mismatch")

    def test_changed_producer_with_stale_hash_is_rejected(self):
        value, status = make_grid(), make_status(make_grid(), producer="fixture")
        status["producer_instance_id"] = "other-producer"
        self.assert_invalid(value, status, "status_grid_content_hash_mismatch")

    def test_tf_invalid_is_rejected(self):
        value, status = make_grid(), make_status(make_grid())
        status["tf_valid"] = False
        self.assert_invalid(value, status, "status_tf_invalid")

    def test_stale_inputs_are_rejected(self):
        value, status = make_grid(), make_status(make_grid())
        status["all_required_inputs_fresh"] = False
        self.assert_invalid(value, status, "status_inputs_not_fresh")

    def test_heartbeat_republication_with_new_transport_sequence_is_accepted(self):
        source = make_grid(sequence=1)
        status = make_status(source, generation=41)
        received_heartbeat = make_grid(sequence=99)
        self.assert_valid(received_heartbeat, status)

    def test_ros_float32_resolution_round_trip_binds_exactly(self):
        source = make_grid(resolution=0.05)
        status = make_status(source)
        received = make_grid(resolution=0.05000000074505806)
        self.assert_valid(received, status)

    def test_changed_grid_content_with_old_hash_is_rejected(self):
        source = make_grid()
        status = make_status(source)
        received = make_grid(cells=[100] + [0] * 15)
        self.assert_invalid(received, status, "status_grid_content_hash_mismatch")


if __name__ == "__main__":
    unittest.main()
