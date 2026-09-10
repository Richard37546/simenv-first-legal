#!/usr/bin/env python3
"""Pure checks for G15 071 audit helpers; no ROS master or publisher."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "local_subgoal_runner_mvp"))
sys.path.insert(0, str(ROOT / "scripts" / "p2kg9_portal_shadow"))
from local_grid_contract import GRID_CONTRACT_VERSION, GRID_STATUS_SCHEMA_VERSION, flatten_index, grid_content_hash, grid_metadata
from p2kg11_portal_frame_contract import PORTAL_FRAME_SIDE_FIELDS


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


SHADOW = load_module("p2kg15_071_shadow", ROOT / "audit_tools" / "p2kg15_071_offline_shadow.py")


class Stamp:
    def __init__(self, value): self.value = value
    def to_sec(self): return self.value


def grid():
    return SimpleNamespace(header=SimpleNamespace(frame_id="base", seq=1, stamp=Stamp(1.0)), info=SimpleNamespace(resolution=0.5, width=6, height=6, origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=-1.5))), data=[0] * 36)


def status(value):
    meta = grid_metadata(value)
    out = {"contract_version": GRID_CONTRACT_VERSION, "schema_version": GRID_STATUS_SCHEMA_VERSION, "producer_instance_id": "fixture", "content_generation_id": 1, "grid_content_stamp": meta["content_stamp"], "frame_id": meta["frame_id"], "origin": {"x": meta["origin_x"], "y": meta["origin_y"]}, "resolution": meta["resolution"], "width": meta["width"], "height": meta["height"], "tf_valid": True, "input_time_monotonic": True, "all_required_inputs_fresh": True, "diagnostic_only": True, "safe_for_navigation": False, "upstream_navigation_allowed": False, "rejection_reasons": ["diagnostic_only"]}
    out["grid_content_hash"] = grid_content_hash(value, out["producer_instance_id"], out["content_generation_id"], out["grid_content_stamp"])
    return out


def portal_frame():
    def row(side):
        values = {key: None for key in PORTAL_FRAME_SIDE_FIELDS}
        values.update({"side": side, "source_stamp": 3.0, "frame_id": "base", "observation_state": "confirmed", "candidate_available": True, "portal_center_base": [1.0, 0.0], "portal_normal_base": [1.0, 0.0], "portal_width": 1.0, "left_boundary": [1.0, -0.5], "right_boundary": [1.0, 0.5], "track_id": side + "-1"})
        return values
    return {"contract_version": "p2kg11_portal_frame_v1", "run_id": "fixture", "frame_sequence": 1, "source_stamp": 3.0, "frame_id": "base", "left": row("left"), "right": row("right")}


class ShadowTests(unittest.TestCase):
    def test_grid_status_mismatch_rejected(self):
        value, pair = grid(), status(grid())
        pair["grid_content_hash"] = "tampered"
        self.assertIn("STATUS_GRID_CONTENT_HASH_MISMATCH", SHADOW.shadow_pair_errors(value, pair))

    def test_atomic_portal_contract_parses_before_side_validation(self):
        self.assertEqual(SHADOW.portal_side_errors(portal_frame(), "left"), [])

    def test_scan_stops_on_unknown(self):
        value = grid()
        value.data[flatten_index(3, 3, 6, 6)] = -1
        scan = SHADOW.scan_candidates(value, (1.0, 0.0), (1.0, 0.0), 1)
        self.assertEqual(scan["termination_reason"], "UNKNOWN")
        self.assertEqual(scan["cells"][-1]["classification"], "UNKNOWN")

    def test_opening_center_is_not_a_candidate_cell(self):
        value = grid()
        value.data[flatten_index(2, 3, 6, 6)] = 100
        scan = SHADOW.scan_candidates(value, (1.0, 0.0), (1.0, 0.0), 1)
        self.assertEqual(scan["cells"][0]["cell"], [3, 3])

    def test_portal_crossing_geometry(self):
        kind = SHADOW.segment_crossing_kind([(0.0, -1.0), (0.0, 1.0)], (-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
        self.assertEqual(kind, "PATH_CROSSES_PORTAL_INTERIOR")
        endpoint = SHADOW.segment_crossing_kind([(-1.0, -1.0), (-1.0, 1.0)], (-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
        self.assertEqual(endpoint, "PATH_TOUCHES_ENDPOINT_ONLY")

    def test_action_authorization_is_never_true(self):
        result = SHADOW.evaluate_side(object(), {"source_stamp": 1.0, "left": {}}, "left", [], [], [])
        self.assertFalse(result["action_authorized"])
        self.assertTrue(result["evaluation_only"])

    def test_audit_sources_have_no_control_publishers(self):
        paths = [ROOT / "audit_scripts" / "p2kg15_071_start_portal_shadow.sh", ROOT / "audit_scripts" / "p2kg15_071_record.sh", ROOT / "audit_tools" / "p2kg15_071_preflight.py", ROOT / "audit_tools" / "p2kg15_071_offline_shadow.py"]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("Publisher(\"/cmd_vel", combined)
        self.assertNotIn("Publisher('/cmd_vel", combined)
        self.assertNotIn("publish(Twist", combined)


if __name__ == "__main__":
    unittest.main()
