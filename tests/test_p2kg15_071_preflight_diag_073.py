#!/usr/bin/env python3
"""Bounded exact-identity pairing tests for the G15 071 preflight."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "local_subgoal_runner_mvp"))
from local_grid_contract import GRID_CONTRACT_VERSION, GRID_STATUS_SCHEMA_VERSION, grid_content_hash, grid_metadata


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


MODULE = load_module("p2kg15_071_preflight_diag", ROOT / "audit_tools" / "p2kg15_071_preflight.py")


class Stamp:
    def __init__(self, value): self.value = value
    def to_sec(self): return self.value


def grid(generation=1, stamp=1.0):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="base", seq=generation, stamp=Stamp(stamp)),
        info=SimpleNamespace(resolution=0.5, width=4, height=4, origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=-1.0))),
        data=[0] * 16,
    )


def status(value, *, generation=41, stamp=None, hash_value=None, tf_valid=True, fresh=True):
    meta = grid_metadata(value)
    stamp = meta["content_stamp"] if stamp is None else stamp
    payload = {
        "contract_version": GRID_CONTRACT_VERSION, "schema_version": GRID_STATUS_SCHEMA_VERSION,
        "producer_instance_id": "fixture", "content_generation_id": generation, "grid_content_stamp": stamp,
        "tf_valid": tf_valid, "all_required_inputs_fresh": fresh,
    }
    payload["grid_content_hash"] = grid_content_hash(value, "fixture", generation, stamp) if hash_value is None else hash_value
    return SimpleNamespace(data=json.dumps(payload))


class ExactPairingTests(unittest.TestCase):
    def test_audit_control_surface_scan_has_no_self_false_positive(self):
        self.assertEqual(MODULE.control_surface_errors(), [])

    def pair(self, first, second):
        cache, value = MODULE.Cache(), grid()
        first(cache, value)
        second(cache, value)
        return cache

    def test_status_before_matching_grid_passes(self):
        cache = self.pair(lambda c, g: c.status_cb(status(g)), lambda c, g: c.grid_cb(g))
        self.assertTrue(cache.pair_ok)

    def test_grid_before_matching_status_passes(self):
        cache = self.pair(lambda c, g: c.grid_cb(g), lambda c, g: c.status_cb(status(g)))
        self.assertTrue(cache.pair_ok)

    def test_transport_sequence_difference_is_accepted(self):
        cache = self.pair(lambda c, g: c.grid_cb(g), lambda c, g: c.status_cb(status(g, generation=2)))
        self.assertTrue(cache.pair_ok)

    def test_stamp_mismatch_rejected(self):
        cache = self.pair(lambda c, g: c.grid_cb(g), lambda c, g: c.status_cb(status(g, stamp=2.0)))
        self.assertFalse(cache.pair_ok)

    def test_hash_mismatch_rejected(self):
        cache = self.pair(lambda c, g: c.grid_cb(g), lambda c, g: c.status_cb(status(g, hash_value="wrong")))
        self.assertFalse(cache.pair_ok)

    def test_tf_or_freshness_false_rejected(self):
        for kwargs in ({"tf_valid": False}, {"fresh": False}):
            cache = self.pair(lambda c, g: c.grid_cb(g), lambda c, g, value=kwargs: c.status_cb(status(g, **value)))
            self.assertFalse(cache.pair_ok)

    def test_caches_are_bounded_to_eight(self):
        cache = MODULE.Cache()
        for index in range(10):
            value = grid(index + 1, float(index + 1))
            cache.grid_cb(value)
            cache.status_cb(status(value))
        self.assertLessEqual(len(cache.grids), 8)
        self.assertLessEqual(len(cache.statuses), 8)


if __name__ == "__main__":
    unittest.main()
