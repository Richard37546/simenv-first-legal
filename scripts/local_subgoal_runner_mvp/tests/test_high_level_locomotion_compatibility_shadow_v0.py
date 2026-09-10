#!/usr/bin/env python3
"""Offline invariants for the authority-free high-level compatibility shadow."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from high_level_locomotion_compatibility_shadow import (  # noqa: E402
    FLAG,
    HighLevelLocomotionCompatibilityShadow,
)
import block_astar_dwa_mature_runner as runner_module  # noqa: E402


class HighLevelLocomotionCompatibilityShadowV0Test(unittest.TestCase):
    def test_default_is_disabled_and_has_no_output(self):
        previous = os.environ.pop(FLAG, None)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["STATE_MACHINE_RUN_ARCHIVE_DIR"] = temporary
                shadow = HighLevelLocomotionCompatibilityShadow.from_environment(HERE.parents[1])
                self.assertFalse(shadow.enabled)
                self.assertEqual(list(Path(temporary).iterdir()), [])
        finally:
            if previous is not None:
                os.environ[FLAG] = previous
            os.environ.pop("STATE_MACHINE_RUN_ARCHIVE_DIR", None)

    def test_source_has_no_control_authority_and_reuses_existing_results(self):
        shadow_source = (HERE / "high_level_locomotion_compatibility_shadow.py").read_text(encoding="utf-8")
        navigation_source = (HERE / "navigation_state_machine.py").read_text(encoding="utf-8")
        runner_source = (HERE / "block_astar_dwa_mature_runner.py").read_text(encoding="utf-8")
        self.assertNotIn("rospy", shadow_source)
        self.assertNotIn("Publisher", shadow_source)
        self.assertNotIn("run_runner(", shadow_source)
        self.assertNotIn("block_astar(", shadow_source)
        self.assertIn("high_level_locomotion_shadow.record_preflight", navigation_source)
        self.assertIn("high_level_locomotion_shadow.record_outcome", navigation_source)
        self.assertIn("SAME_PRODUCTION_DWA_EVALUATION", runner_source)
        self.assertIn("SAME_PRODUCTION_ASTAR_PATH", runner_source)
        self.assertNotIn("high_level_locomotion_shadow.record_preflight(", runner_source)

    def test_shadow_records_only_descriptive_continuous_burden(self):
        source = (HERE / "high_level_locomotion_compatibility_shadow.py").read_text(encoding="utf-8")
        self.assertIn('"LARGE_INITIAL_HEADING_BURDEN": relative_bearing', source)
        self.assertNotIn("compatibility_boolean", source)

    def test_dwa_shadow_on_preserves_samples_scores_and_winner(self):
        def make_runner():
            args = runner_module.build_arg_parser().parse_args([
                "--max-linear-x", "0.60", "--max-angular-z", "0.35", "--min-linear-x", "0.30",
            ])
            runner = object.__new__(runner_module.BlockAStarDwaRunner)
            runner.args, runner.prev_cmd = args, (0.35, -0.1)
            runner.dynamic_window = lambda: (np.array([0.0, 0.30]), np.array([-0.1, 0.0, 0.1]))
            def collision(_grid, _blocked, v, _w, trace=None):
                if trace is not None:
                    trace.update({"endpoint_base_xy": [float(v), 0.0], "endpoint_base_yaw_rad": 0.0})
                return True, 0.4
            runner.collision_free_arc = collision
            return runner
        kwargs = dict(room_search_safe_moving_eligibility=True, target_in_front=True, astar_path_exists=True)
        previous = os.environ.pop(FLAG, None)
        try:
            off = make_runner().choose_dwa(None, np.zeros((1, 1), dtype=bool), (0.5, 0.1), (0.7, 0.2), 0.73, None, **kwargs)
            os.environ[FLAG] = "1"
            on = make_runner().choose_dwa(None, np.zeros((1, 1), dtype=bool), (0.5, 0.1), (0.7, 0.2), 0.73, None, **kwargs)
            self.assertEqual(off[:2], on[:2])
            on_without_shadow = dict(on[2]); on_without_shadow.pop("high_level_locomotion_shadow", None)
            self.assertEqual(off[2], on_without_shadow)
            self.assertEqual(on[2]["high_level_locomotion_shadow"]["evidence_source"], "SAME_PRODUCTION_DWA_EVALUATION")
        finally:
            if previous is None:
                os.environ.pop(FLAG, None)
            else:
                os.environ[FLAG] = previous


if __name__ == "__main__":
    unittest.main()
